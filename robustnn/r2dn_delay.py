import jax
import jax.numpy as jnp

from typing import Tuple, Sequence

from flax import linen as nn
from flax.linen import initializers as init
from flax.struct import dataclass
from flax.typing import Dtype, Array

from robustnn import lbdn
from robustnn import linear_ren as linren
from robustnn import linear_ren_base as linren_base
from robustnn.utils import l2_norm
from robustnn.utils import ActivationFn, Initializer


def get_valid_init():
    return ["random", "long_memory"]


@dataclass
class DirectR2D2NParams:
    """Data class to keep track of direct params for R2D2N.
    
    These are the free, trainable parameters for an R2D2N.
    """
    B2: Array
    D12: Array
    C2: Array
    D21: Array
    D22: Array
    by: Array
    ren_params: linren_base.DirectLinRENParams
    network_params: lbdn.DirectLBDNParams
    
@dataclass
class ExplicitR2D2NParams:
    """Data class to keep track of explicit params for R2D2N.
    
    These are the parameters used for evaluating an R2D2N.
    """
    B2: Array
    D12: Array
    C2: Array
    D21: Array
    D22: Array
    by: Array
    ren_params: linren_base.ExplicitLinRENParams
    network_params: lbdn.ExplicitLBDNParams
    
class ContractingR2D2N(nn.Module):
    """
    # TODO: Write docs later.
    """
    input_size: int             # nu
    state_size: int             # nx
    features: int               # nv
    output_size: int            # ny
    hidden: Sequence[int]       # Hidden layer sizes in the LBDN
    activation: ActivationFn = nn.relu
    
    kernel_init: Initializer = init.lecun_normal()
    recurrent_kernel_init: Initializer = init.lecun_normal()
    carry_init: Initializer = init.zeros_init()
    x_bias_init: Initializer = init.zeros_init()
    v_bias_init: Initializer = init.zeros_init()
    y_bias_init: Initializer = init.zeros_init()
    network_bias_init: Initializer = init.zeros_init()
    param_dtype: Dtype = jnp.float32
    
    init_method: str = "random"
    init_output_zero: bool = False
    identity_output: bool = False
    
    do_polar_param: bool = True
    eps: jnp.float32 = jnp.finfo(jnp.float32).eps # type: ignore
    _gamma: jnp.float32 = 1.0 # type: ignore
    
    def setup(self):
        """Initialise all direct params for an R2D2N and store."""
        
        if self.init_method not in get_valid_init():
            raise ValueError("Undefined init method '{}'".format(self.init_method))
        
        nu = self.input_size
        nx = self.state_size
        nv = self.features
        ny = self.output_size
        dtype = self.param_dtype
        
        # Initialise an LBDN for the nonlinear layer, and a REN for the 
        # components of the linear system in feedback with the nonlinear layer
        self._network_init()
        self._linear_init()
        
        # Initialise the free parameters      
        B2 = self.param("B2", self.kernel_init, (nx, nu), dtype)
        D12 = self.param("D12", self.kernel_init, (nv, nu), dtype)
        
        # Output layer params
        if self.init_output_zero:
            out_kernel_init = init.zeros_init()
            out_bias_init = init.zeros_init()
        else:
            out_kernel_init = self.kernel_init
            out_bias_init = self.y_bias_init
        
        if self.identity_output:
            C2 = jnp.identity(nx)
            D21 = jnp.zeros((ny, nv), dtype)
            D22 = jnp.zeros((ny, nu), dtype)
            by = jnp.zeros((ny,), dtype)
        else:
            by = self.param("by", out_bias_init, (ny,), dtype)
            C2 = self.param("C2", out_kernel_init, (ny, nx), dtype)
            D21 = self.param("D21", out_kernel_init, (ny, nv), dtype)
            D22 = self.param("D22", init.zeros_init(), (ny, nu), dtype)
            
        self.direct = DirectR2D2NParams(
            B2, D12, C2, D21, D22, by, self.linear.direct, self.network.direct
        ) 
        
    def _network_init(self):
        """Initialise a Sandwich network for the nonlinear layer."""
        self.network = lbdn.LBDN(
            input_size=self.features,
            hidden_sizes=self.hidden,
            output_size=self.features,
            gamma=self._gamma,
            activation=self.activation,
            kernel_init=self.kernel_init,
            bias_init=self.network_bias_init,
            param_dtype=self.param_dtype
        )
        
    def _linear_init(self):
        """
        Initialise a linear REN for the component of the linear system
        in feedback with the nonlinear layers (Sandwich network).
        """
        lin_gamma = self.param_dtype(1 / self._gamma)
        self.linear = linren.LipschitzLinREN(
            input_size=self.features,
            state_size=self.state_size,
            output_size=self.features,
            features=0,
            gamma=lin_gamma,
            
            kernel_init=self.kernel_init,
            recurrent_kernel_init=self.recurrent_kernel_init,
            carry_init=self.carry_init,
            x_bias_init=self.x_bias_init,
            y_bias_init=self.v_bias_init,
            
            param_dtype=self.param_dtype,
            init_method=self.init_method,
            
            do_polar_param=self.do_polar_param,
            eps=self.eps
        )
        
    def __call__(self, state: Tuple[Array,Array], inputs: Array) -> Tuple[Array, Array]:
        """Call an R2D2N model

        Args:
            state (Tuple[Array,Array]): internal model state and hidden neurons.
            inputs (Array): model inputs.

        Returns:
            Tuple[Array, Array]: (next_states_and_neurons, outputs).
        """
        
        explicit = self._direct_to_explicit()
        return self._explicit_call(state, inputs, explicit)
    
    def _explicit_call(
        self, xv: Tuple[Array,Array], u: Array, e: ExplicitR2D2NParams
    ) -> Tuple[Array, Array]:
        """Evaluate explicit model for an R2D2N.

        Args:
            xv (Tuple[Array,Array]): internal model state and hidden neurons.
            u (Array): model inputs.
            e (ExplicitR2D2NParams): explicit params.

        Returns:
            Tuple[Array, Array]: (next_states_and_neurons, outputs).
        """

        # Delayed neural layer
        x, v = xv
        w = self.network._explicit_call(v, e.network_params)
        
        # Parametrised part of the linear model
        x1, v1 = self.linear._explicit_call(x, w, e.ren_params)
        
        # Remainder of the linear model
        x1 = x1 + u @ e.B2.T
        v1 = v1 + u @ e.D12.T
        y = x @ e.C2.T + w @ e.D21.T + u @ e.D22.T + e.by
        
        return (x1, v1), y
    
    def _simulate_sequence(self, xv0, u) -> Tuple[Array, Array]:
        """Simulate an R2D2N over a sequence of inputs.

        Args:
            xv0: array of initial states and neurons, shape is (batches, ...).
            u: array of inputs as a sequence, shape is (time, batches, ...).
            
        Returns:
            Tuple[Tuple[Array, Array], Array]: (final_state_and_neurons, outputs in (time, batches, ...)).
        """
        explicit = self._direct_to_explicit()
        def rollout(carry, ut):
            xvt, = carry
            xvt1, yt = self._explicit_call(xvt, ut, explicit)
            return (xvt1,), yt
        (xv1, ), y = jax.lax.scan(rollout, (xv0,), u)
        return xv1, y
    
    @nn.nowrap
    def initialize_carry(
        self, rng: jax.Array, input_shape: Tuple[int, ...]
    ) -> Array:
        """Initialise the R2D2N state (carry).

        Args:
            rng (jax.Array): random seed for carry initialisation.
            input_shape (Tuple[int, ...]): Shape of model input array.

        Returns:
            Tuple[Array, Array]: initial model state and neurons.
        """
        batch_dims = input_shape[:-1]
        rng1, rng2 = jax.random.split(rng)
        x0 = self.carry_init(rng1, batch_dims + (self.state_size,), self.param_dtype)
        v0 = self.carry_init(rng2, batch_dims + (self.features,), self.param_dtype)
        return (x0, v0)
    
    def _direct_to_explicit(self) -> ExplicitR2D2NParams:
        """Convert from direct to explicit R2D2N params.

        Args:
            None

        Returns:
            ExplicitR2D2NParams: explicit params for R2D2N.
        """
        ps = self.direct
        ren_params = self.linear._direct_to_explicit()
        network_params = self.network._direct_to_explicit()
        
        return ExplicitR2D2NParams(
            ps.B2, ps.D12, ps.C2, ps.D21, ps.D22, ps.by, ren_params, network_params
        )
        
     #################### Convenient Wrappers ####################

    def explicit_call(
        self, params:dict, xv: Array, u: Array, e: ExplicitR2D2NParams
    ) -> Tuple[Array, Array]:
        """Evaluate explicit model for an R2D2N.

        Args:
            params (dict): Flax model parameters dictionary.
            xv (Tuple[Array,Array]): internal model state and hidden neurons.
            u (Array): model inputs.
            e (ExplicitR2D2NParams): explicit params.

        Returns:
            Tuple[Array, Array]: (next_states_and_neurons, outputs).
        """
        return self.apply(params, xv, u, e, method="_explicit_call")
    
    def simulate_sequence(self, params: dict, xv0, u) -> Tuple[Array, Array]:
        """Simulate an R2D2N over a sequence of inputs.

        Args:
            params (dict): Flax model parameters dictionary.
            xv0: array of initial states and neurons, shape is (batches, ...).
            u: array of inputs as a sequence, shape is (time, batches, ...).
            
        Returns:
            Tuple[Tuple[Array, Array], Array]: (final_state_and_neurons, outputs in (time, batches, ...)).
        """
        return self.apply(params, xv0, u, method="_simulate_sequence")
    
    def direct_to_explicit(self, params: dict) -> ExplicitR2D2NParams:
        """Convert from direct to explicit R2D2N params.

        Args:
            params (dict): Flax model parameters dictionary.

        Returns:
            ExplicitR2D2NParams: explicit params for R2D2N.
        """
        return self.apply(params, method="_direct_to_explicit")

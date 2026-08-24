'''
Implementation of Lipschitz Bounded Deep Networks (Linear) in JAX/FLAX

Adapted from Julia implentation: https://github.com/acfr/RobustNeuralNetworks.jl

Authors: Nic Barbara (Mar '24, Feb '25), Jack Naylor (Sep '23) from the ACFR.

These networks are compatible with other FLAX modules.
'''

import jax.numpy as jnp

from typing import Sequence, Optional

from flax import linen as nn
from flax.linen import initializers as init
from flax.struct import dataclass
from flax.typing import Dtype, Array, PrecisionLike

from robustnn.utils import l2_norm, cayley, cayley_b, dot_lax
from robustnn.utils import ActivationFn, Initializer


@dataclass
class DirectSandwichParams:
    """Data class to keep track of direct params for Sandwich layer."""
    XY: Array
    a: Array
    d: Array
    b: Array


@dataclass
class ExplicitSandwichParams:
    """Data class to keep track of explicit params for Sandwich layer."""
    A_T: Array
    B: Array
    psi_d: Array
    b: Array


@dataclass
class DirectLinearParams:
    """Data class to keep track of direct params for a Lipschitz linear layer."""
    XY: Array
    a: Array
    b: Array


@dataclass
class ExplicitLinearParams:
    """Data class to keep track of explicit params for a Lipschitz linear layer."""
    B: Array
    b: Array


# An LBDN is a stack of Sandwich layers followed by a Lipschitz linear layer,
# so its parameter sequences hold a mix of the two.
DirectLayerParams = DirectSandwichParams | DirectLinearParams
ExplicitLayerParams = ExplicitSandwichParams | ExplicitLinearParams


@dataclass
class DirectLBDNParams:
    """Data class to keep track of direct params for LBDN."""
    layers: Sequence[DirectLayerParams]
    log_gamma: Array


@dataclass
class ExplicitLBDNParams:
    """Data class to keep track of explicit params for LBDN."""
    layers: Sequence[ExplicitLayerParams]
    log_gamma: Array


class SandwichLayerBase(nn.Module):
    """Base class for Sandwich network layers.

    This class has the parameters common to all Sandwich layers: the stacked
    weight matrix `XY` that is passed through the Cayley transform, its norm
    scaling `a`, and the bias `b`. Subclasses add any layer-specific
    parameters and define the direct-to-explicit map and the layer call.

    The layer interface has been written similarly to `linen.Dense`.

    Attributes:
        input_size: the number of input features.
        features: the number of output features.
        use_bias: whether to add a bias to the output (default: True).

        kernel_init: initializer function for the weight matrix (default: lecun_normal()).
        bias_init: initializer function for the bias (default: zeros_init()).

        dtype: the dtype of the computation (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        precision: numerical precision of the computation see ``jax.lax.Precision``
            for details.
    """
    input_size: int
    features: int
    use_bias: bool = True

    kernel_init: Initializer = init.lecun_normal()
    bias_init: Initializer = init.zeros_init()

    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None

    def setup(self):
        """Initialise direct params"""
        dtype = self.param_dtype

        XY = self.param("XY", self.kernel_init,
                        (self.input_size + self.features, self.features),
                        dtype)
        a = self.param("a", init.constant(l2_norm(XY)), (1,), dtype)
        b = self.param("b", self.bias_init, (self.features,), dtype)

        self.direct = self._build_direct(XY, a, b)

    def __call__(self, inputs: Array) -> Array:
        """Call a Sandwich layer.

        Args:
            inputs (Array): layer inputs.

        Returns:
            Array: layer outputs.
        """
        explicit = self._direct_to_explicit()
        return self._explicit_call(inputs, explicit)

    def _scale_weights(self, ps: DirectLayerParams) -> Array:
        """Scale `XY` by `a / ||XY||` ready for the Cayley transform.

        Args:
            ps (DirectLayerParams): direct layer params.

        Returns:
            Array: the scaled stacked weight matrix.
        """
        return ps.a / l2_norm(ps.XY) * ps.XY


    ############### Specify these for each layer type ###############

    def _build_direct(self, XY: Array, a: Array, b: Array) -> DirectLayerParams:
        """Build the direct param struct, adding any layer-specific params.

        This is called from within `setup()`, so subclasses may call
        `self.param(...)` here to register extra parameters.
        """
        raise NotImplementedError(
            "SandwichLayerBase layers should not be constructed directly. " +
            "Choose a layer type instead (eg: `SandwichLayer`)."
        )

    def _direct_to_explicit(self) -> ExplicitLayerParams:
        """Convert from direct layer params to explicit form for eval."""
        raise NotImplementedError(
            "SandwichLayerBase layers should not be called. " +
            "Choose a layer type instead (eg: `SandwichLayer`)."
        )

    def _explicit_call(self, u: Array, e: ExplicitLayerParams) -> Array:
        """Evaluate the explicit model for a layer."""
        raise NotImplementedError(
            "SandwichLayerBase layers should not be called. " +
            "Choose a layer type instead (eg: `SandwichLayer`)."
        )


class SandwichLayer(SandwichLayerBase):
    """The 1-Lipschtiz Sandwich layer from Wang & Manchester (ICML '23).

    Example usage::

        >>> from robustnn.lbdn import SandwichLayer
        >>> import jax, jax.numpy as jnp

        >>> layer = SandwichLayer(input_size=3, features=4)
        >>> params = layer.init(jax.random.key(0), jnp.ones((1, 3)))
        >>> jax.tree_util.tree_map(jnp.shape, params)
        {'params': {'XY': (7, 4), 'a': (1,), 'b': (4,), 'd': (4,)}}

    Attributes:
        activation: Activation function to use (default: relu).
        psi_init: initializer function for the activation scaling (default: zeros_init()).

    See docs for `SandwichLayerBase` for the remaining arguments.

    Note: Only monotone activations are supported: `identity`, `relu`, `tanh`, `sigmoid`.
    """
    activation: ActivationFn = nn.relu
    psi_init: Initializer = init.zeros_init()

    def _build_direct(self, XY: Array, a: Array, b: Array) -> DirectSandwichParams:
        """Add the activation scaling `d` to the shared direct params."""
        d = self.param("d", self.psi_init, (self.features,), self.param_dtype)
        return DirectSandwichParams(XY, a, d, b)

    def _direct_to_explicit(self) -> ExplicitSandwichParams:
        """Convert from direct Sandwich params to explicit form for eval.

        Returns:
            ExplicitSandwichParams: explicit Sandwich params.
        """
        ps = self.direct
        A_T, B_T = cayley(self._scale_weights(ps), return_split=True)

        # Clip d to avoid over/underflow and return
        psi_d = jnp.exp(jnp.clip(ps.d, min=-20.0, max=20.0))
        return ExplicitSandwichParams(A_T, B_T.T, psi_d, ps.b)

    def _explicit_call(self, u: Array, e: ExplicitSandwichParams) -> Array:
        """Evaluate the explicit model for a Sandwich layer.

        Args:
            u (Array): layer inputs.
            e (ExplicitSandwichParams): explicit params.

        Returns:
            Array: layer outputs.
        """
        sqrt2 = self.param_dtype(jnp.sqrt(2.0))
        x = sqrt2 * dot_lax(u, ((jnp.diag(1 / e.psi_d)) @ e.B))
        if self.use_bias:
            x += e.b
        return sqrt2 * dot_lax(self.activation(x), (e.A_T * e.psi_d.T))


class SandwichLinear(SandwichLayerBase):
    """A linear layer whose weight matrix satisfies `||B|| <= 1`.

    This is the output layer of an LBDN from Wang & Manchester (ICML '23).
    It shares the Cayley-parameterised weight of a `SandwichLayer`, but has
    no activation and hence no activation scaling `d`.

    Example usage::

        >>> from robustnn.lbdn import SandwichLinear
        >>> import jax, jax.numpy as jnp

        >>> layer = SandwichLinear(input_size=3, features=4)
        >>> params = layer.init(jax.random.key(0), jnp.ones((1, 3)))
        >>> jax.tree_util.tree_map(jnp.shape, params)
        {'params': {'XY': (7, 4), 'a': (1,), 'b': (4,)}}

    See docs for `SandwichLayerBase` for the full list of arguments.
    """

    def _build_direct(self, XY: Array, a: Array, b: Array) -> DirectLinearParams:
        """The shared direct params are all this layer needs."""
        return DirectLinearParams(XY, a, b)

    def _direct_to_explicit(self) -> ExplicitLinearParams:
        """Convert from direct params to explicit form for eval.

        Only the `B` block of the Cayley transform is needed here, so we skip
        the linear solve for `A`.

        Returns:
            ExplicitLinearParams: explicit params.
        """
        ps = self.direct
        B_T = cayley_b(self._scale_weights(ps))
        return ExplicitLinearParams(B_T.T, ps.b)

    def _explicit_call(self, u: Array, e: ExplicitLinearParams) -> Array:
        """Evaluate the explicit model for a Lipschitz linear layer.

        Args:
            u (Array): layer inputs.
            e (ExplicitLinearParams): explicit params.

        Returns:
            Array: layer outputs.
        """
        x = dot_lax(u, e.B)
        return x + e.b if self.use_bias else x


class LBDN(nn.Module):
    """Lipschitz-Bounded Deep Network.
    
    Example usage::
    
        >>> from robustnn.lbdn import LBDN
        >>> import jax, jax.numpy as jnp
        
        >>> nu, ny = 5, 2
        >>> layers = (8, 16)
        >>> gamma = jnp.float32(10)
        
        >>> model = LBDN(nu, layers, ny, gamma=gamma)
        >>> params = model.init(jax.random.key(0), jnp.ones((6,nu)))
        >>> jax.tree_util.tree_map(jnp.shape, params)
        {'params': {'layers_0': {'XY': (13, 8), 'a': (1,), 'b': (8,), 'd': (8,)}, 'layers_1': {'XY': (24, 16), 'a': (1,), 'b': (16,), 'd': (16,)}, 'layers_2': {'XY': (18, 2), 'a': (1,), 'b': (2,)}}}
        
        Note: the ``ln_gamma`` parameter only appears when ``trainable_lipschitz=True``.
    
    Attributes:
        input_size: the number of input features.
        hidden_sizes: Sequence of hidden layer sizes.
        output_size: the number of output features.
        gamma: upper bound on the Lipschitz constant (default: 1.0).
        activation: activation function to use (default: relu).
        
        kernel_init: initializer function for the weight matrix (default: lecun_normal()).
        bias_init: initializer function for the bias (default: zeros_init()).
        psi_init: initializer function for the activation scaling (default: zeros_init()).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        
        use_bias:whether to add a bias to the output (default: True).
        trainable_lipschitz: make the Lipschitz constant trainable (default: False).
        init_output_zero: initialize the network so its output is zero (default: False).
    
    Note: Only monotone activations are supported: `identity`, `relu`, `tanh`, `sigmoid`.
    """
    input_size: int
    hidden_sizes: Sequence[int]
    output_size: int
    gamma: jnp.float32 = 1.0 # type: ignore
    activation: ActivationFn = nn.relu
    
    kernel_init: Initializer = init.lecun_normal()
    bias_init: Initializer = init.zeros_init()
    psi_init: Initializer = init.zeros_init()
    param_dtype: Dtype = jnp.float32
    
    use_bias: bool = True
    trainable_lipschitz: bool = False
    init_output_zero: bool = False
    
    def setup(self):
        """Initialise direct LBDN params.
        
        The setup is currently written in a rather convoluted way to make it possible
        to split up the direct-to-explicit transform and explicit model call. This is
        because anything initialised in `setup()` can't be accessed outside of `model.init()
        ` and `model.apply()` in Flax. Very frustrating.
        """
        
        dtype = self.param_dtype
        
        # Set up trainable/constant Lipschitz bound (positive quantity)
        # The learnable parameter is log(gamma), then we take gamma = exp(log_gamma)
        log_gamma = dtype(jnp.log(self.gamma))
        if self.trainable_lipschitz:
            log_gamma = self.param("ln_gamma", init.constant(log_gamma),(1,), dtype)
        
        # Build a list of Sandwich layers, but treat the output separately
        hidden_sizes = tuple(self.hidden_sizes)
        in_layers = (self.input_size,) + hidden_sizes
        out_layers = hidden_sizes + (self.output_size,)
        
        layers = [
            SandwichLayer(
                input_size=in_layers[k],
                features=out_layers[k],
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                psi_init=self.psi_init,
                param_dtype=dtype
            )
            for k in range(len(hidden_sizes))
        ]
        
        out_kernel_init = self.kernel_init
        if self.init_output_zero:
            out_kernel_init = init.zeros_init()
        
        layers.append(
            SandwichLinear(
                input_size=in_layers[-1],
                features=out_layers[-1],
                use_bias=self.use_bias,
                kernel_init=out_kernel_init,
                bias_init=self.bias_init,
                param_dtype=dtype
            )
        )
        
        self.layers = layers
        self.direct = DirectLBDNParams([s.direct for s in layers], log_gamma)
        
    def __call__(self, inputs: Array) -> Array:
        """Call an LBDN model.

        Args:
            inputs (Array): model inputs.

        Returns:
            Array: model outputs.
        """
        explicit = self._direct_to_explicit()
        return self._explicit_call(inputs, explicit)
    
    def _explicit_call(self, u: Array, explicit: ExplicitLBDNParams):
        """Evaluate the explicit model for an LBDN model.

        Args:
            u (Array): model inputs.
            e (ExplicitLBDNParams): explicit params.

        Returns:
            Array: model outputs.
        """
        sqrt_gamma = jnp.sqrt(jnp.exp(explicit.log_gamma))
        x = sqrt_gamma * u
        
        for k, layer in enumerate(self.layers):
            x = layer._explicit_call(x, explicit.layers[k])
        
        return sqrt_gamma * x
    
    def _direct_to_explicit(self) -> ExplicitLBDNParams:
        """Convert from direct LBDN params to explicit form for eval.

        Args:
            None
            
        Returns:
            ExplicitLBDNParams: explicit LBDN params.
        """
        ps = self.direct
        layer_explicit_params = [
            layer._direct_to_explicit() for layer in self.layers
        ]
        return ExplicitLBDNParams(layer_explicit_params, ps.log_gamma)
    
    
    #################### Convenient Wrappers ####################
    
    def explicit_call(self, params: dict, u: Array, explicit: ExplicitLBDNParams):
        """Evaluate the explicit model for an LBDN model.

        Args:
            params (dict): Flax model parameters dictionary.
            u (Array): model inputs.
            e (ExplicitLBDNParams): explicit params.

        Returns:
            Array: model outputs.
        """
        return self.apply(params, u, explicit, method="_explicit_call")
    
    def direct_to_explicit(self, params: dict):
        """Convert from direct LBDN params to explicit form for eval.

        Args:
            params (dict): Flax model parameters dictionary.
            
        Returns:
            ExplicitLBDNParams: explicit LBDN params.
        """
        return self.apply(params, method="_direct_to_explicit")

import jax
import jax.numpy as jnp

from robustnn import r2dn_delay as r2d2n
from robustnn.utils import count_num_params
from utils import compute_p_contractingr2d2n

jax.config.update("jax_default_matmul_precision", "highest")

# Problem size
nu = 5
ny = 2
batches = 4
horizon = 2

# Initialise an R2DN
nx = 3                  # Number of states
nv = 4                  # Number of equilibirum layer states
nh = (2,) * 2           # Number of hidden layers in the LBDN
init_method = "long_memory"
model = r2d2n.ContractingR2D2N(nu, nx, nv, ny, nh, init_method=init_method)
    
# Random seeds
rng = jax.random.key(0)
rng, key1, key2, key3, key4 = jax.random.split(rng, 5)

# Dummy inputs and states
states = model.initialize_carry(key1, (batches, None))
states_x = jax.random.normal(key2, states[0].shape)
states = (states_x, states[1])
inputs = jax.random.normal(key3, (horizon, batches, nu))

# Initialise the model and check how many params
params = model.init(key4, states, inputs[0])
print("Number of params: ", count_num_params(params))

# Call forward pass
jit_call = jax.jit(model.simulate_sequence)
new_state, out = jit_call(params, states, inputs)
print(new_state)
print(out)

# Test taking a gradient
@jax.jit
def loss(params, xv0, u):
    xv1, y = model.simulate_sequence(params, xv0, u)
    x1 = xv1[0]
    return jnp.sum(x1**2) + jnp.sum(y**2)

grad_func = jax.jit(jax.grad(loss, argnums=(0,1,2)))
gs = grad_func(params, states, inputs)

print(loss(params, states, inputs))
print("States grad: ", gs[1])
print("Input grad: ", gs[2])


############################################################

# Test contraction
def mat_norm2(x, A):
    return jnp.sum((x @ A.T) * x, axis=-1)

# Random seeds
rng = jax.random.key(0)
rng, key1, key2, key3, key4, key5 = jax.random.split(rng, 6)

# Dummy inputs and states
states = model.initialize_carry(key1, (batches, nu))
v0 = states[1]
x0 = 10*jax.random.normal(key2, states[0].shape)
x1 = jax.random.normal(key3, states[0].shape)
inputs = jax.random.normal(key4, (batches, nu))

params = model.init(key5, states, inputs)

# Simulate one step
(x0n, _), _ = model.apply(params, (x0, v0), inputs)
(x1n, _), _ = model.apply(params, (x1, v0), inputs)

# Test for contraction
P = compute_p_contractingr2d2n(model, params)
lhs = mat_norm2(x0n - x1n, P) - mat_norm2(x0 - x1, P)
rhs = 0

assert jnp.all(lhs <= rhs)

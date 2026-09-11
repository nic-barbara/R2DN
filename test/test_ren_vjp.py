"""Check the custom VJP for the REN equilibrium layer against autodiff.

The layer solves `w = activation(D11 @ w + b)` by forward substitution and
supplies a hand-written backwards pass from the implicit function theorem.
Differentiating straight through the solver must give the same gradients, so
that is the reference used throughout.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from robustnn import ren, ren_base
from robustnn.ren_base import tril_equilibrium_layer

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


def reference_layer(activation, D11, b):
    """Forward substitution with no custom VJP, so autodiff unrolls the solver."""
    w = jnp.zeros_like(b)
    for i in range(D11.shape[0]):
        w = w.at[..., i].set(activation(w[..., :i] @ D11.T[:i, i] + b[..., i]))
    return w


def max_err(a, b):
    return jnp.abs(a - b).max()


def rel_err(a, b):
    """Error relative to the scale of `a`, which relu makes large for big `q`."""
    return max_err(a, b) / jnp.maximum(jnp.abs(a).max(), 1.0)


# Fixed point and gradients, over neuron counts, batch shapes, and activations
print("Equilibrium layer vs autodiff through the solver:")
for q in (1, 2, 7, 33):
    for shape in ((q,), (4, q), (3, 4, q)):
        for activation in (nn.tanh, nn.relu):
            case = f"q={q}, shape={shape}, {activation.__name__}"
            k1, k2, k3 = jax.random.split(jax.random.key(q), 3)
            D11 = jnp.tril(jax.random.normal(k1, (q, q)), k=-1)
            b = jax.random.normal(k2, shape)
            g = jax.random.normal(k3, shape)

            def loss(layer, D11, b):
                return jnp.sum(g * layer(activation, D11, b))

            w = tril_equilibrium_layer(activation, D11, b)
            dD11, db = jax.grad(lambda *a: loss(tril_equilibrium_layer, *a), (0, 1))(D11, b)
            rD11, rb = jax.grad(lambda *a: loss(reference_layer, *a), (0, 1))(D11, b)

            # w solves the layer to rounding; the residual is only nonzero
            # because recomputing `v` sums in a different order to the solve
            assert rel_err(w, activation(w @ D11.T + b)) < 1e-14, case
            assert rel_err(dD11, rD11) < 1e-12, (case, rel_err(dD11, rD11))
            assert rel_err(db, rb) < 1e-12, (case, rel_err(db, rb))

            # Only the strict lower triangle of D11 is read, so nothing else
            # may pick up a gradient
            assert jnp.all(jnp.triu(dD11) == 0), case
print("  fixed point, dL/dD11, dL/db all match (relative error < 1e-12)")


# The same check through a whole REN, where the layer sits inside the scan
# over time and the gradients flow back through the direct-to-explicit map
nu, nx, nv, ny = 5, 3, 8, 2
batches, horizon = 4, 6
model = ren.ContractingREN(nu, nx, nv, ny, activation=nn.tanh,
                           init_method="long_memory", param_dtype=jnp.float64)

k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
states = model.initialize_carry(k1, (batches, nu))
inputs = jax.random.normal(k2, (horizon, batches, nu), dtype=jnp.float64)
params = model.init(k3, states, inputs[0])


def model_loss(params, states, inputs):
    x1, y = model.simulate_sequence(params, states, inputs)
    return jnp.sum(x1**2) + jnp.sum(y**2)


grad_func = jax.grad(model_loss, argnums=(0, 1, 2))
custom = grad_func(params, states, inputs)

ren_base.tril_equilibrium_layer = reference_layer
try:
    autodiff = grad_func(params, states, inputs)
finally:
    ren_base.tril_equilibrium_layer = tril_equilibrium_layer

errors = jax.tree.leaves(jax.tree.map(rel_err, custom, autodiff))
worst = max(float(e) for e in errors)
print(f"ContractingREN gradients vs autodiff: relative error {worst:.3e}")
assert worst < 1e-10, worst
assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(custom))

print("All equilibrium layer VJP tests passed.")

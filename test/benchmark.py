import jax
import flax.linen as nn

from robustnn import ren
from robustnn.timing import time_model

jax.config.update("jax_default_matmul_precision", "highest")

# Define contracting REN
nu = 2
nx = 4
nv = 8
ny = 3
model = ren.ContractingREN(nu, nx, nv, ny, activation=nn.tanh, init_method="long_memory")

# Time the model
results = time_model(model, batches=64, horizon=1, n_repeats=200)

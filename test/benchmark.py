import jax
import flax.linen as nn

from robustnn import ren
from robustnn.timing import time_model

jax.config.update("jax_default_matmul_precision", "highest")


def build_model_and_time(nu, nx, nv, ny, init_method="long_memory"):
    print(f"\nnx: {nx}, nv: {nv}")
    model = ren.ContractingREN(
        nu, nx, nv, ny, activation=nn.tanh, init_method=init_method
    )
    return time_model(model, batches=64, horizon=1, n_repeats=200)


# Some benchmarking experiments
nu, ny = 5, 10

build_model_and_time(nu, 16, 32, ny)
build_model_and_time(nu, 16, 128, ny)
build_model_and_time(nu, 16, 512, ny)

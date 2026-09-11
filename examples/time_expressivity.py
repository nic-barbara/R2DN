import jax
import jax.numpy as jnp
import timeit
from pathlib import Path

from robustnn import ren
from robustnn import r2dn
from robustnn.timing import time_model

from utils import utils

dirpath = Path(__file__).resolve().parent
jax.config.update("jax_default_matmul_precision", "highest")


def build_model(config):
    """Build neural models."""
    nu, nx, ny = config["nx"], config["nx"], config["nx"]
    if config["network"] == "contracting_ren":
        model = ren.ContractingREN(
            nu, nx, config["nv"], ny, 
            identity_output=True,
            activation=utils.get_activation(config["activation"]),
            init_method=config["init_method"],
            do_polar_param=config["polar"]
        )
    elif config["network"] == "contracting_r2dn":
        model = r2dn.ContractingR2DN(
            nu, nx, config["nv"], ny, config["nh"], 
            identity_output=True,
            activation=utils.get_activation(config["activation"]),
            init_method=config["init_method"],
            do_polar_param=config["polar"]
        )
    return model


def run_timing(filename, batches, horizon, n_repeats=200):
    """Run the timing for a trained REN or R2DN."""

    config, params, results = utils.load_results(filename)
    model = build_model(config)
        
    time_results = time_model(model, batches, horizon, n_repeats)
    results = results | time_results
    utils.save_results(config, params, results)


# Nominal data sizes
batches = 64
horizon = 128
    
# Read all the pre-trained models and save timing results
fpath = dirpath / f"../results/expressivity/"
files = [f for f in fpath.iterdir()]
for f in files:
    if f.is_file() and not (f.suffix == ".pdf"):
        print(f)
        run_timing(f, batches, horizon)

# Robust Recurrent Deep Networks (R2DN)

This repository contains the code associated with the paper *Scalable Parameterization of Contracting and Lipschitz Recurrent Deep Networks*.

Included are JAX implementations of each of the following robust neural models:

- The Sandwich layer and corresponding Lipschitz Bounded Deep Network from [Wang & Manchester (ICML 2023)](https://proceedings.mlr.press/v202/wang23v.html).
- Contracting, Lipschitz, and (Q,S,R)-dissipative Recurrent Equilibrium Networks (RENs) from [Revay, Wang, & Manchester (TAC 2023)](https://ieeexplore.ieee.org/document/10179161).
- Contracting Robust Recurrent Deep Networks (R2DNs) from our current work.

Robust neural models are included in the `robustnn/` directory. Scripts used to generate the results in the paper are in the `examples/` directory.

## Installation and Usage

To install the required dependencies and run the code, open a terminal in the root directory of this repository and enter the following commands.

    ./install.sh
    ./run.sh

This will create a Python virtual environment and run all the experiments, process the results, and reproduce the figures from the paper.

### A Note on Dependencies

All code was tested and developed in Ubuntu 22.04 with CUDA 12.4 and Python 3.10.12.

Requirements were generated with [`pipreqs`](https://github.com/bndr/pipreqs). The `install.sh` script assumes the user is running JAX on an NVIDIA GPU with CUDA 12 already installed. If no GPU is available, simply remove the line

    pip install -U "jax[cuda12_pip]"

from the `install.sh` script. If you have a GPU that is not running CUDA (or a different CUDA version), edit the above installation command accordingly.

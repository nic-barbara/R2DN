# Robust Recurrent Deep Network (R2DN)

This repository contains the code associated with our paper [*R2DN: Scalable Parameterization of Contracting and Lipschitz Recurrent Deep Networks*](https://arxiv.org/abs/2504.01250) (Barbara, Wang, & Manchester, accepted to CDC 2026).

Included are JAX implementations of each of the following robust neural models:

- The Sandwich layer and corresponding Lipschitz Bounded Deep Network from [Wang & Manchester (ICML 2023)](https://proceedings.mlr.press/v202/wang23v.html).
- Contracting, Lipschitz, and (Q,S,R)-dissipative Recurrent Equilibrium Networks (RENs) from [Revay, Wang, & Manchester (TAC 2023)](https://ieeexplore.ieee.org/document/10179161).
- Contracting Robust Recurrent Deep Networks (R2DNs) from our current work.

Robust neural models are included in the `robustnn/` directory. Scripts used to generate the results in the paper are in the `examples/` directory. For the latest implementations of the above robust NNs, and many others, see [https://github.com/acfr/RobustNeuralNetworks](https://github.com/acfr/RobustNeuralNetworks).

## Installation and Usage

First, clone the repository:

```
git clone https://github.com/nic-barbara/R2DN.git
```

All dependencies are managed via [uv](https://docs.astral.sh/uv/). To install uv, run the following (Mac/Linux, see the [docs](https://docs.astral.sh/uv/getting-started/installation/) for Windows).

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To install the required dependencies and run the code, open a terminal in the root directory of this repository and enter the following commands.

    uv sync
    ./run.sh

This will create a Python virtual environment and run all the experiments, process the results, and reproduce the figures from the paper.

All code was tested and developed in Ubuntu 26.04 with CUDA 13.2 and Python 3.12.14.

## Contact

Please contact Nicholas Barbara (nicholas.barbara@epfl.ch) with any questions.
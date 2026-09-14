"""Command-line driver used for the tuning sweeps.

Intermediate tooling: the final experiments are run from
`examples/train_sequential_imagecfn.py`. This exists to run one configuration
at a time from the shell, which is what the sweep scripts in this folder do.
"""
import argparse
import jax
import sys
from copy import deepcopy
from pathlib import Path

dirpath = Path(__file__).resolve().parent
sys.path.append(str(dirpath.parent))

from utils import seqcls
from utils.seqcls import TASKS, matched_r2dn_config, run, print_summary

jax.config.update("jax_default_matmul_precision", "highest")

ren_config = seqcls.default_config()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="psmnist", choices=TASKS.keys())
    parser.add_argument("--network", default="ren", choices=["ren", "r2dn"])
    parser.add_argument("--nx", type=int, default=ren_config["nx"])
    parser.add_argument("--nv", type=int, default=ren_config["nv"])
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--activation", default=ren_config["activation"])
    parser.add_argument("--epochs", type=int, default=ren_config["epochs"])
    parser.add_argument("--lr", type=float, default=ren_config["lr"])
    parser.add_argument("--lr-cuts", type=int, nargs="*", default=ren_config["lr_cuts"])
    parser.add_argument("--batchsize", type=int, default=ren_config["batchsize"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-output-zero", action="store_true")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable CIFAR-10 crop/flip augmentation.")
    parser.add_argument("--resume-from", default="",
                        help="Pickle to warm-start weights from (optimizer state is not saved).")
    parser.add_argument("--tag", default="")
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=0,
                        help="Stop training after this many hours.")
    args = parser.parse_args()

    config = deepcopy(ren_config)
    config["experiment"] = args.task
    config["nx"], config["nv"] = args.nx, args.nv
    config["activation"] = args.activation
    config["epochs"] = args.epochs
    config["lr"] = args.lr
    config["lr_cuts"] = tuple(args.lr_cuts)
    config["batchsize"] = args.batchsize
    config["seed"] = args.seed
    config["init_output_zero"] = args.init_output_zero
    config["augment"] = not args.no_augment
    config["resume_from"] = args.resume_from
    config["tag"] = args.tag
    config["train_subset"] = args.train_subset
    config["time_limit"] = args.time_limit

    if args.network == "r2dn":
        config = matched_r2dn_config(config, TASKS[args.task], args.layers)

    print_summary(config, run(config))

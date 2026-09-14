"""Print the CLI arguments for the best sCIFAR10 pilot config.

Used by `run_overnight.sh` to carry the pilot's winner into the full runs.
Prints the psMNIST-tuned config if no pilot results are found, so the overnight
queue still does something sensible if the pilot failed.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.utils import load_results

FALLBACK = "--nx 96 --nv 150 --layers 2"


def main():
    results = []
    for path in Path("results/scifar10").glob("*pilot*.pickle"):
        config, _, res = load_results(path)
        results.append((res["test_at_best_val"], config))

    if not results:
        print(FALLBACK)
        return

    acc, config = max(results)
    layers = len(config["nh"])
    # `--nv` is the REN width; the driver halves it for an R2DN
    print(f"--nx {config['nx']} --nv {2*config['nv']} --layers {layers}")
    print(f"# {100*acc:.2f}% in the pilot", file=sys.stderr)


main()

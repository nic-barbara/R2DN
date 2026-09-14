#!/bin/bash
# Full psMNIST runs, in order of increasing cost.
#
#   1. R2DN at Kozachkov et al.'s batch size of 64, for a directly comparable
#      number (~3.6h).
#   2. R2DN at batch 512, the setting the REN needs to be affordable (~0.6h).
#   3. REN at batch 512 (~5.9h), but only if the larger batch did not hurt the
#      R2DN - no point spending six hours on a setting we already know is broken.
set -e
cd "$(dirname "$0")/../.."
RUN="uv run python examples/seqcls/train_seqcls.py --task psmnist"

$RUN --network r2dn --batchsize 64  --tag bs64  --time-limit 6
$RUN --network r2dn --batchsize 512 --tag bs512 --time-limit 3

uv run python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "examples")
from utils.utils import load_results

acc = {}
for tag in ("bs64", "bs512"):
    path = next(Path("results/psmnist").glob(f"*r2dn*_{tag}.pickle"))
    acc[tag] = load_results(path)[2]["test_at_best_val"]

print(f"R2DN batch 64: {100*acc['bs64']:.2f}%, batch 512: {100*acc['bs512']:.2f}%")
if acc["bs512"] < acc["bs64"] - 0.02:
    sys.exit("Batch 512 costs the R2DN >2 percentage points. Not running the REN.")
PY

$RUN --network ren --batchsize 512 --tag bs512 --time-limit 10

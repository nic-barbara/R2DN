#!/bin/bash
# sCIFAR10 schedule probes. Breaking the 300-epoch run down by lr phase shows
# the second cut came too early:
#
#   phase 1 (ep 1-180, lr 1e-3):  46.9 -> 52.7%, creeping
#   phase 2 (ep 181-260, lr 1e-4): 55.8 -> 58.6%, still climbing when cut
#   phase 3 (ep 261-300, lr 1e-5): 59.5 -> 59.9%, flat within ~10 epochs
#
# lr 1e-4 is the productive regime and it only got 80 epochs. Both probes give
# it far longer; the second also enters it sooner, since phase 1 was only
# creeping by the end. R2DN, nx=128, 131K, ~1.5 h each.
# Baseline to beat: 60.19% (300 epochs, cuts at 180/260).
cd "$(dirname "$0")/../.."
exec 2>&1
RUN="uv run python -u examples/seqcls/train_seqcls.py --task scifar10 --network r2dn \
     --batchsize 512 --layers 2 --nx 128 --nv 78 --epochs 450 --time-limit 4"

echo "=== 1/2: 450 epochs, cuts at 180/400 (longer at 1e-4) ==="
$RUN --lr-cuts 180 400 --tag sched450a || echo "FAILED"

echo "=== 2/2: 450 epochs, cuts at 120/380 (enter 1e-4 sooner) ==="
$RUN --lr-cuts 120 380 --tag sched450b || echo "FAILED"

echo "=== schedule probes finished at $(date) ==="

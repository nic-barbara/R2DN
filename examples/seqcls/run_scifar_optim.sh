#!/bin/bash
# sCIFAR10, second round of diagnostics. The first round showed that parameter
# count is not what is binding: 4x the budget bought +1.70 points while simply
# doubling the epochs bought +3.27, and 4x the parameters barely moved the
# training loss (1.145 -> 1.083). The models cannot fit sCIFAR10 at any size we
# tried, which points at optimization rather than capacity.
#
# The untested axis is the learning rate. We inherited Kozachkov et al.'s 1e-3,
# chosen for batch 64; we train at batch 512, so 8x fewer gradient steps per
# epoch. Runs 1-2 bracket it. Runs 3-4 combine the wins already in hand.
#
# All R2DN, nx=128, L=2 (131K) unless noted, 300 epochs with cuts at 180/260.
# Baseline for comparison: 60.19% (300 epochs, lr 1e-3, augmented).
cd "$(dirname "$0")/../.."
exec 2>&1
RUN="uv run python -u examples/seqcls/train_seqcls.py --task scifar10 --network r2dn \
     --batchsize 512 --layers 2 --epochs 300 --lr-cuts 180 260 --time-limit 4"

echo "=== 1/4: lr 3e-3 (nx=128, 131K) ==="
$RUN --nx 128 --nv 78 --lr 3e-3 --tag lr3e3 || echo "FAILED"

echo "=== 2/4: lr 1e-2 (nx=128, 131K) ==="
$RUN --nx 128 --nv 78 --lr 1e-2 --tag lr1e2 || echo "FAILED"

echo "=== 3/4: no augmentation, 300 epochs (nx=128, 131K) ==="
$RUN --nx 128 --nv 78 --no-augment --tag noaug300 || echo "FAILED"

echo "=== 4/4: no augmentation, 300 epochs, 2x budget (nx=160, 261K) ==="
$RUN --nx 160 --nv 160 --no-augment --tag noaug300_2x || echo "FAILED"

echo "=== optimization sweep finished at $(date) ==="

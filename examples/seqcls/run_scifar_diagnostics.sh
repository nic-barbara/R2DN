#!/bin/bash
# Why is sCIFAR10 stuck at ~57%? The 150-epoch runs end with a training loss of
# 1.05-1.14 (random guessing is 2.30) against test accuracy of 56-58%, i.e. the
# models cannot fit the training set. That is underfitting, the opposite of
# psMNIST, where the final training loss was 0.02.
#
# Three one-factor R2DN runs (~30 min each) to find out what is binding, before
# spending REN time on it. All start from the best known config, nx=128.
#
#   1. No augmentation. Crop+flip is a regulariser, and we are underfitting.
#      Kozachkov can afford it at nx=512; S4 trains sCIFAR10 without it.
#   2. Longer schedule. Neither model had converged at epoch 150 - the REN was
#      still climbing 57.66 -> 58.43 over the last 19 epochs.
#   3. More parameters. Allocation is monotone in nx up to the 131K ceiling
#      (55.1% -> 56.1% -> 56.9% for nx = 64, 96, 128), so state dimension is
#      binding and the budget is what caps it. nv is held at 80 in both
#      over-budget runs so state is the variable.
cd "$(dirname "$0")/../.."
exec 2>&1
RUN="uv run python -u examples/seqcls/train_seqcls.py --task scifar10 --network r2dn \
     --batchsize 512 --layers 2 --time-limit 4"

echo "=== 1/4: no augmentation (nx=128, 131K) ==="
$RUN --nx 128 --nv 78 --epochs 150 --lr-cuts 90 130 --no-augment --tag noaug || echo "FAILED"

echo "=== 2/4: 300 epochs (nx=128, 131K) ==="
$RUN --nx 128 --nv 78 --epochs 300 --lr-cuts 180 260 --tag ep300 || echo "FAILED"

echo "=== 3/4: 2x budget, nx=160 (nv=80, nh=155, 261K) ==="
$RUN --nx 160 --nv 160 --epochs 150 --lr-cuts 90 130 --tag budget2x || echo "FAILED"

echo "=== 4/4: 4x budget, nx=256 (nv=80, nh=193, 523K) ==="
$RUN --nx 256 --nv 160 --epochs 150 --lr-cuts 90 130 --tag budget4x || echo "FAILED"

echo "=== diagnostics finished at $(date) ==="

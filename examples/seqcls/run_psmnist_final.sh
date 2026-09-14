#!/bin/bash
# Final psMNIST runs: REN vs R2DN at the architecture chosen in Stage 3 tuning
# (see notes/ren_seq_benchmark_log.md for the sweeps behind each choice).
#
#   nx=96          - interior optimum of the allocation sweep at fixed ~131K
#   L=2 (R2DN)     - tied with L=4 on accuracy, cheaper per step; L=8 is worse
#   relu           - beats tanh by ~3 points
#   batch 512      - beats batch 64, and makes the REN affordable
#   150 epochs, lr cuts at 90 and 130 - each cut buys a jump then a plateau,
#                    and a shorter 60-epoch schedule costs 1.16 points
#
# Both models land within 1% of 131K parameters. Add seeds by looping over
# --seed if a repeatability claim is ever needed; one seed is enough to settle
# the REN vs R2DN comparison.
set -e
cd "$(dirname "$0")/../.."
RUN="uv run python -u examples/seqcls/train_seqcls.py --task psmnist \
     --batchsize 512 --nx 96 --epochs 150 --lr-cuts 90 130 --tag final"

$RUN --network r2dn --nv 152 --layers 2   # nv=76, nh=120, 130532 params, ~0.5 h
$RUN --network ren  --nv 153              # 131261 params, ~4.2 h

#!/bin/bash
# How should a fixed ~131K parameter budget be split between state dimension
# (nx, which is what carries information across timesteps) and per-step
# nonlinearity (nv and the LBDN width nh)? R2DN only - it is ~10x cheaper than
# the REN, so it is the right model to scan with.
#
# Short schedule: the full 150-epoch runs plateau by epoch ~40 and then gain
# ~4.5 points from the lr cut at 90, so 45 epochs with the cut at 30 reproduces
# the shape of the curve at a tenth of the cost (~12 min per config). Absolute
# numbers will sit below the 150-epoch runs; only the ranking is meaningful.
#
# `--nv` is the *REN* width; the driver halves it and picks the matching LBDN
# width, so every config below lands within 1% of 131K params.
set -e
cd "$(dirname "$0")/../.."
RUN="uv run python -u examples/seqcls/train_seqcls.py --task psmnist --network r2dn \
     --batchsize 512 --epochs 45 --lr-cuts 30 --tag alloc"

$RUN --nx  32 --nv 290   # nv=145, nh=96, 130840 params
$RUN --nx  48 --nv 258   # nv=129, nh=95, 131269 params
$RUN --nx  64 --nv 224   # nv=112, nh=93, 131729 params (the config run so far)
$RUN --nx  96 --nv 152   # nv= 76, nh=84, 131446 params
$RUN --nx 128 --nv  78   # nv= 39, nh=65, 130659 params

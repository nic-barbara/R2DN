#!/bin/bash
# Stage 3 tuning, continued from run_alloc_sweep.sh. Same short schedule
# (45 epochs, lr cut at 30, batch 512, ~12 min per config) so results are
# directly comparable to the allocation sweep.
#
# A1 found an interior optimum at nx=96 (92.56%), so first fill in nx=112 to
# pin it down, then vary depth and activation at nx=96.
set -e
cd "$(dirname "$0")/../.."
RUN="uv run python -u examples/seqcls/train_seqcls.py --task psmnist --network r2dn \
     --batchsize 512 --epochs 45 --tag tune"

# A1 refinement: nx=112, nv=58, nh=76, 131056 params
$RUN --lr-cuts 30 --nx 112 --nv 116

# A2 depth at nx=96: L=2 (nh=120) and L=8 (nh=59); L=4 already done
$RUN --lr-cuts 30 --nx 96 --nv 152 --layers 2
$RUN --lr-cuts 30 --nx 96 --nv 152 --layers 8

# A3 activation at nx=96, L=4
$RUN --lr-cuts 30 --nx 96 --nv 152 --activation tanh

# A4 schedule: second lr cut, against the single-cut run at the same config
$RUN --lr-cuts 30 38 --nx 96 --nv 152 --tag tune2cut

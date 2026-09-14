#!/bin/bash
# A4, continued. The short-schedule runs show each lr cut buying a jump and then
# a flat plateau (+2.8 points from the first cut, +1.2 from the second), which
# suggests the pre-cut epochs are mostly wasted. If a short two-cut schedule
# matches the 150-epoch one, the REN run gets 2-3x cheaper.
#
# Tuned config from A1-A3: nx=96, L=2 (nv=76, nh=120, 130532 params), relu.
set -e
cd "$(dirname "$0")/../.."
RUN="uv run python -u examples/seqcls/train_seqcls.py --task psmnist --network r2dn \
     --batchsize 512 --nx 96 --nv 152 --layers 2"

$RUN --epochs 150 --lr-cuts 90 130 --tag sched150
$RUN --epochs  60 --lr-cuts 35  50 --tag sched60

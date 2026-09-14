#!/bin/bash
# sCIFAR10 pilot. Does the psMNIST-tuned architecture transfer to a longer
# (1024-step), harder, 3-channel task?
#
# Reasons to expect it does: the per-step input is still tiny (3 numbers), so the
# nonlinear layer is not doing image processing - all of CIFAR's complexity is
# spread over 1024 steps and has to be integrated by the recurrence, which is the
# state's job. The memory requirement goes up, not down. Kozachkov et al. and S4
# both use one architecture across both tasks.
#
# Reasons it might not: 3 input channels give the input-side map more to do,
# which would favour width (nv, nh) over state; augmentation regularises, so the
# whole optimum may drift upward.
#
# Same short schedule as the psMNIST tuning (45 epochs, cut at 30, batch 512) so
# rankings are read the same way. CHECK THE TRAJECTORY FIRST: sCIFAR10 is much
# harder, and if every config is still climbing steeply at epoch 45 the ranking
# is not meaningful and the pilot schedule needs extending.
set -e
cd "$(dirname "$0")/../.."
RUN="uv run python -u examples/seqcls/train_seqcls.py --task scifar10 --network r2dn \
     --batchsize 512 --epochs 45 --lr-cuts 30 --layers 2 --tag pilot"

# Allocation at fixed ~131K (nu=3 shifts the counts slightly from psMNIST)
$RUN --nx  64 --nv 222   # nv=111, nh=131, 130703 params
$RUN --nx  96 --nv 150   # nv= 75, nh=120, 130298 params  (psMNIST winner)
$RUN --nx 128 --nv  78   # nv= 39, nh= 96, 131366 params

# Does L=2 still hold, or does the harder task want more per-step depth?
$RUN --nx 96 --nv 150 --layers 4 --tag pilotL4

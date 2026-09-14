#!/bin/bash
# Overnight queue, in priority order so a failure late in the night costs the
# least valuable run. Waits for the GPU to be free first - JAX preallocates most
# of the card, so these must not overlap.
#
#   1. sCIFAR10 R2DN at the pilot's winning config   (~0.6 h)
#   2. sCIFAR10 REN at the matching config           (~4.5 h)
#   3. psMNIST R2DN seeds 1 and 2                    (~1.0 h)
#   4. psMNIST REN seed 1                            (~4.2 h)
#
# (3) and (4) matter because the headline psMNIST result is a 0.15-point gap at
# one seed, which is inside seed noise. Extra seeds let us say "tie" with
# evidence rather than assertion.
cd "$(dirname "$0")/../.."
exec 2>&1

echo "=== waiting for the GPU ==="
while pgrep -f "train_seqcls.py" > /dev/null; do sleep 60; done

CFG=$(uv run python examples/seqcls/pick_scifar_config.py 2>/dev/null | head -1)
CFG=${CFG:-"--nx 96 --nv 150 --layers 2"}
NX=$(echo $CFG | awk '{print $2}')
NV=$(echo $CFG | awk '{print $4}')
echo "=== sCIFAR10 config from pilot: $CFG ==="

RUN="uv run python -u examples/seqcls/train_seqcls.py --batchsize 512 --epochs 150 --lr-cuts 90 130"

echo "=== 1/4: sCIFAR10 R2DN ==="
$RUN --task scifar10 --network r2dn $CFG --tag final --time-limit 3 || echo "FAILED: scifar r2dn"

echo "=== 2/4: sCIFAR10 REN ==="
$RUN --task scifar10 --network ren --nx $NX --nv $NV --tag final --time-limit 8 || echo "FAILED: scifar ren"

echo "=== 3/4: psMNIST R2DN seeds 1, 2 ==="
for s in 1 2; do
  $RUN --task psmnist --network r2dn --nx 96 --nv 152 --layers 2 --seed $s --tag final \
    --time-limit 3 || echo "FAILED: psmnist r2dn seed $s"
done

echo "=== 4/4: psMNIST REN seed 1 ==="
$RUN --task psmnist --network ren --nx 96 --nv 153 --seed 1 --tag final \
  --time-limit 8 || echo "FAILED: psmnist ren seed 1"

echo "=== overnight queue finished at $(date) ==="

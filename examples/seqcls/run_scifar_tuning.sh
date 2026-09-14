#!/bin/bash
# sCIFAR10 architecture tuning at full length, for BOTH models.
#
# The 45-epoch pilot ranked nx=128 first, but no config had converged, so that
# ranking was noise. These runs use the real 150-epoch schedule. `nx=128` is
# already done for both models (58.06% REN, 56.92% R2DN), so only 64 and 96 are
# run here. Ordered cheapest first so the R2DN answer arrives early.
#
# Tuning the REN separately rather than transferring the R2DN's choice: the two
# have different parameter geometry (`X` is (2nx+nv)^2 for the REN, (2nx)^2 for
# the R2DN), so the optimum need not coincide, and the comparison has to be fair.
cd "$(dirname "$0")/../.."
exec 2>&1
RUN="uv run python -u examples/seqcls/train_seqcls.py --task scifar10 \
     --batchsize 512 --epochs 150 --lr-cuts 90 130 --tag tune150"

echo "=== 1/4: R2DN nx=96  (~0.5 h) ==="
$RUN --network r2dn --nx 96 --nv 150 --layers 2 --time-limit 3 || echo "FAILED"
echo "=== 2/4: R2DN nx=64  (~0.5 h) ==="
$RUN --network r2dn --nx 64 --nv 222 --layers 2 --time-limit 3 || echo "FAILED"
echo "=== 3/4: REN nx=96   (~4.3 h) ==="
$RUN --network ren  --nx 96 --nv 150 --time-limit 8 || echo "FAILED"
echo "=== 4/4: REN nx=64   (~6.4 h) ==="
$RUN --network ren  --nx 64 --nv 222 --time-limit 10 || echo "FAILED"

# psMNIST final runs are deliberately not queued here - Nic runs those himself
# with `examples/seqcls/train_sequential_imagecfn.py psmnist`.
echo "=== queue finished at $(date) ==="

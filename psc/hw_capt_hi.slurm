#!/bin/bash
# High-frequency extension of the GRID-7 torso-capped sweep (Ben 2026-09-11): identical physics and grid to
# psc/hw_capt.slurm (354 deg/s cap on legs + torso, +-4.1 N.m, kappa PID, no lag, hardware CAD models), but the
# frequency axis 1.75 .. 2.00 Hz (6 levels) instead of 1.20 .. 1.70: 6 x 36 x 13 x 6 x 5 = 84,240 cells per
# (config, mu). Why: the GRID-5 c6 ice gaits with no flight phase and low landing force sit at 1.8-1.98 Hz, above the
# 1.70 Hz edge of GRID-7 (and 1.70 itself was excluded as an edge frequency by the robust rule).
# Output: results/grid6_hw/<cfg>/hwcapt_<cfg>_mu###_hi.<shard>.csv -> merged hwcapt_<cfg>_mu###_hi.csv
# Cost at the measured 1.65 s/cell (kappa 0) / 2.4 s/cell (kappa 2): ~39 / ~56 SU per (config, mu); all 24 ~1,140 SU.
#
#   for c in c1 c2 c3 c4 c5 c6; do for m in 0.1 0.3 0.5 0.7; do sbatch --array=0-255 psc/hw_capt_hi.slurm $c $m; done; done
#   re-run failed shards: sbatch --array=13,23 psc/hw_capt_hi.slurm c3 0.1 256
#   merge:  cd $PROJECT/pengu_hw/pengu_mujoco && CONFIG=c3 HW_TORSO=pid HW_SERVO_LAG=0 HW_TORSO_CAP=1 HW_FREQ=1.75:2.00:0.05 HW_TAG_SUFFIX=_hi python grid6/hw_sweep.py --mu 0.1 --merge

#SBATCH --job-name=hwcapt_hi
#SBATCH --account=cis250009p
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/hwcapt_hi-%A_%a.out
#SBATCH --error=logs/hwcapt_hi-%A_%a.out
set -euo pipefail
CFG="${1:?config c1..c6}"
MU="${2:?mu, e.g. 0.1}"

CODE="${PROJECT}/pengu_hw"
source "${PROJECT}/mjx_venv/bin/activate"
cd "${CODE}/pengu_mujoco"
mkdir -p results/grid6_hw

export CONFIG="${CFG}"
export HW_TORSO=pid
export HW_SERVO_LAG=0
export HW_TORSO_CAP=1
export HW_FREQ=1.75:2.00:0.05
export HW_TAG_SUFFIX=_hi
export OMP_NUM_THREADS=1
echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}  cfg ${CFG}  mu ${MU}  freq 1.75-2.00, cap on legs+torso, PID  $(date)"

T_START=$(date +%s)
set +e
python grid6/hw_sweep.py --mu "${MU}" \
    --shard "${SLURM_ARRAY_TASK_ID}" --of "${3:-${SLURM_ARRAY_TASK_COUNT}}"
RC=$?
set -e
T_END=$(date +%s)
mkdir -p "${CODE}/state"
printf '%s %s %s %s %s %s %.1f %s %s\n' \
    "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "hwcapt_hi" "${CFG}" "${MU}" \
    "${SLURM_CPUS_ON_NODE:-1}" \
    "$(awk -v c="${SLURM_CPUS_ON_NODE:-1}" -v e="$(( T_END - T_START ))" 'BEGIN{print c*e/3600}')" \
    "rc=${RC}" "$(date -Iseconds)" >> "${CODE}/state/tasks.txt"
echo "done $(date)  rc=${RC}  $(( T_END - T_START ))s"
exit "${RC}"

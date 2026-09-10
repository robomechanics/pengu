#!/bin/bash
# torso-capped PID sweep on Bridges-2 RM-shared (Ben 2026-09-10): grid6/hw_sweep.py with the 354 deg/s
# velocity cap on the LEGS AND THE TORSO servo, +-4.1 N.m torque cap (in the models), kappa PID torso,
# no lag, no feedforward; c1..c6 at the GRID-5 friction ladder mu 0.1 / 0.3 / 0.5 / 0.7.
# One rollout per cell; output hwcapt_<cfg>_mu###.csv (v_net_pid etc.).
#
#   for c in c1 c2 c3 c4 c5 c6; do for m in 0.1 0.3 0.5 0.7; do sbatch --array=0-255 psc/hw_capt.slurm $c $m; done; done
#   re-run failed shards: sbatch --array=13,23 psc/hw_capt.slurm c3 0.1 256
#   merge:  cd $PROJECT/pengu_hw/pengu_mujoco && CONFIG=c3 HW_TORSO=pid HW_SERVO_LAG=0 HW_TORSO_CAP=1 python grid6/hw_sweep.py --mu 0.1 --merge

#SBATCH --job-name=hwcapt
#SBATCH --account=cis250009p
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/hwcapt-%A_%a.out
#SBATCH --error=logs/hwcapt-%A_%a.out
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
export OMP_NUM_THREADS=1
echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}  cfg ${CFG}  mu ${MU}  cap on legs+torso, PID  $(date)"

MUTAG=$(python -c "print(f'{round(float(\"${MU}\")*100):03d}')")     # 0.1 -> 010, 0.45 -> 045
CELLS="results/grid6_hw/${CFG}/cells_${CFG}_mu${MUTAG}_r1.csv"
[ -f "${CELLS}" ] && CF=(--cells-file "${CELLS}") || CF=()
echo "cells: ${CELLS} ($([ -f "${CELLS}" ] && wc -l < "${CELLS}" || echo none))"
T_START=$(date +%s)
set +e
python grid6/hw_sweep.py --mu "${MU}" "${CF[@]}" \
    --shard "${SLURM_ARRAY_TASK_ID}" --of "${3:-${SLURM_ARRAY_TASK_COUNT}}"
RC=$?
set -e
T_END=$(date +%s)
mkdir -p "${CODE}/state"
printf '%s %s %s %s %s %s %.1f %s %s\n' \
    "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "hwcapt" "${CFG}" "${MU}" \
    "${SLURM_CPUS_ON_NODE:-1}" \
    "$(awk -v c="${SLURM_CPUS_ON_NODE:-1}" -v e="$(( T_END - T_START ))" 'BEGIN{print c*e/3600}')" \
    "rc=${RC}" "$(date -Iseconds)" >> "${CODE}/state/tasks.txt"
echo "done $(date)  rc=${RC}  $(( T_END - T_START ))s"
exit "${RC}"

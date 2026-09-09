#!/bin/bash
# cap-only sweep on Bridges-2 RM-shared (Ben 2026-09-09): grid6/hw_sweep.py with the 354 deg/s
# hard cap on the legs, NO torso lag, NO feedforward -- the kappa PID torso loop exactly as in
# GRID-5, on the hardware CAD models, at the drag-measured friction 0.12 / 0.45. One rollout
# per cell (HW_TORSO=pid HW_SERVO_LAG=0); output hwcap_<cfg>_mu###.csv (v_net_pid etc.).
#
#   sbatch --array=0-255 psc/hw_cap.slurm c3 0.12      # c3 kappa 0  pengu1_31
#   sbatch --array=0-255 psc/hw_cap.slurm c3 0.45
#   sbatch --array=0-255 psc/hw_cap.slurm c4 0.12      # c4 kappa 2  pengu1_05_hw_updated
#   sbatch --array=0-255 psc/hw_cap.slurm c4 0.45
#   (any of c1 c2 c5 c6 works too, for a cap-only column next to the hwact table)
#
# Re-running failed shards: pass the ORIGINAL array width as the third positional argument
# (see psc/hw_sweep.slurm for why an env var does not survive sbatch here):
#   sbatch --array=13,23 psc/hw_cap.slurm c3 0.12 256
# Merge on the login node afterwards:
#   cd $PROJECT/pengu_hw/pengu_mujoco && CONFIG=c3 HW_TORSO=pid HW_SERVO_LAG=0 python grid6/hw_sweep.py --mu 0.12 --merge

#SBATCH --job-name=hwcap
#SBATCH --account=cis250009p
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/hwcap-%A_%a.out
#SBATCH --error=logs/hwcap-%A_%a.out
set -euo pipefail
CFG="${1:?config c3|c4 (or c1 c2 c5 c6)}"
MU="${2:?mu, e.g. 0.12}"

CODE="${PROJECT}/pengu_hw"
source "${PROJECT}/mjx_venv/bin/activate"
cd "${CODE}/pengu_mujoco"
mkdir -p results/grid6_hw

export CONFIG="${CFG}"
export HW_TORSO=pid
export HW_SERVO_LAG=0
export OMP_NUM_THREADS=1
echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}  cfg ${CFG}  mu ${MU}  cap-only/PID  $(date)"

CELLS="results/grid6_hw/${CFG}/cells_${CFG}_mu${MU/0./0}_r1.csv"
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
    "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "hwcap" "${CFG}" "${MU}" \
    "${SLURM_CPUS_ON_NODE:-1}" \
    "$(awk -v c="${SLURM_CPUS_ON_NODE:-1}" -v e="$(( T_END - T_START ))" 'BEGIN{print c*e/3600}')" \
    "rc=${RC}" "$(date -Iseconds)" >> "${CODE}/state/tasks.txt"
echo "done $(date)  rc=${RC}  $(( T_END - T_START ))s"
exit "${RC}"

#!/bin/bash
# Extended-metric pass over the GRID-7 walker set (Ben 2026-09-14). Same physics as hw_capt.slurm --
# grid6/hw_sweep.py, 354 deg/s cap on legs AND torso, +-4.1 N.m torque cap, kappa PID torso, no lag --
# so v_net_pid and friends come out identical and the already-swept columns need no re-recording.
#
# What is new: the EXT_COLS block now also stores the forward / sideways split of the foot slip and
# tangential foot velocity, the tangential GRF in newtons, the heading-projected per-footfall speed
# (v_fwd / v_lat, the mocap pipeline's definition), and term_reason / term_t.
#
# It does NOT sweep the grid: it reads the surviving-walker list produced locally by
# grid7/walker_scan7.py (robust + no fall + within the 354 deg/s bar, NO foot-clearance gate),
# 657,815 cells over the 24 (config, mu) pairs -- 11.5 % of the full grid.
#
# Early termination (HW_EARLY_S=5): a rollout stops at 5 s of window if both feet leave the floor by
# more than HW_HOP_MM (hop) or the body heading is more than HW_FACE_MAX from the travel direction
# (backwards / sideways). Those rows keep term_reason and term_t; their other columns describe the
# SHORTENED window only -- filter on term_reason == "ok" before using them.
#
#   cells:  ${PROJECT}/pengu_hw/pengu_mujoco/results/grid7_report/data/walker_cells_grid7/<cfg>_mu###.csv
#   out:    results/grid6_hw/<cfg>/hwcapt_<cfg>_mu###_ext.csv        (HW_TAG_SUFFIX=_ext, never clobbers the sweep)
#
#   for c in c1 c2 c3 c4 c5 c6; do for m in 0.1 0.3 0.5 0.7; do sbatch --array=0-127 psc/hw_ext.slurm $c $m; done; done
#   re-run failed shards: sbatch --array=13,23 psc/hw_ext.slurm c3 0.1 128
#   merge: cd $PROJECT/pengu_hw/pengu_mujoco && CONFIG=c3 HW_TORSO=pid HW_SERVO_LAG=0 HW_TORSO_CAP=1 \
#            HW_TAG_SUFFIX=_ext python grid6/hw_sweep.py --mu 0.1 --merge

#SBATCH --job-name=hwext
#SBATCH --account=cis250009p
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/hwext-%A_%a.out
#SBATCH --error=logs/hwext-%A_%a.out
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
export HW_TAG_SUFFIX=_ext
export HW_EARLY_S=5.0            # stop a hopping / backwards rollout at 5 s of window
export HW_HOP_MM=2.0             # both feet clear of their own stance height by more than this = hop
export HW_FACE_MAX=90.0          # |body heading - travel direction|, deg (the continuous facing_deg is stored too)
export OMP_NUM_THREADS=1

MU3=$(python -c "print(f'{int(round(float('${MU}')*100)):03d}')")
CELLS="results/grid7_report/data/walker_cells_grid7/${CFG}_mu${MU3}.csv"
test -s "${CELLS}" || { echo "missing cell list ${CELLS}"; exit 2; }
echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}  cfg ${CFG}  mu ${MU}  $(date)"
echo "cells: $(( $(wc -l < "${CELLS}") - 1 )) from ${CELLS}  early=${HW_EARLY_S}s hop=${HW_HOP_MM}mm face=${HW_FACE_MAX}deg"

T_START=$(date +%s)
set +e
python grid6/hw_sweep.py --mu "${MU}" --cells-file "${CELLS}" \
    --shard "${SLURM_ARRAY_TASK_ID}" --of "${3:-${SLURM_ARRAY_TASK_COUNT}}"
RC=$?
set -e
T_END=$(date +%s)
mkdir -p "${CODE}/state"
printf '%s %s %s %s %s %s %.1f %s %s\n' \
    "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "hwext" "${CFG}" "${MU}" \
    "${SLURM_CPUS_ON_NODE:-1}" \
    "$(awk -v c="${SLURM_CPUS_ON_NODE:-1}" -v e="$(( T_END - T_START ))" 'BEGIN{print c*e/3600}')" \
    "rc=${RC}" "$(date -Iseconds)" >> "${CODE}/state/tasks.txt"
echo "done $(date)  rc=${RC}  $(( T_END - T_START ))s"
exit "${RC}"

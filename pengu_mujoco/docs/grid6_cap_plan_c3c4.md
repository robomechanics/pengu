# GRID-6 cap-only sweep: c3 / c4 (Ben 2026-09-09)

**What**: c3 (κ=0, COM 1.31, model `pengu1_31`) and c4 (κ=2, COM 1.05, model
`pengu1_05_hw_updated`) with the **354 °/s leg velocity cap only** — no torso servo lag,
no feedforward torso; the κ PID loop exactly as in GRID-5 (kp 2 / ki 0.1, clamp 25° for
κ0 / 45° for κ2). μ = 0.12 and 0.45. One rollout per cell.

This is the "act" variant of realism_check, but on the hardware-sweep grid and cell lists,
so it lines up column-for-column with the c1/c2/c5/c6 hardware table (same freq 1.20–1.70
@0.05, phi all 36, leg 70–130 @5, hip 12–32, off 20–40 @5; GRID-5 black regions pruned
with margin 1). Output columns are hw_sweep's; only the `*_pid` block is filled
(`fell_pid, v_net_pid, straight_pid, clear_pid, clear_ok_pid, …`); held/ff blank.

| | c3 μ0.12 | c3 μ0.45 | c4 μ0.12 | c4 μ0.45 |
|---|---|---|---|---|
| cells (list, margin 1) | 73,981 | 72,948 | 141,240 | 135,211 |
| rollouts | same | same | same | same |
| cost @ ~1 s/cell on RM-shared (Mac: 0.75 s) | ~21 SU | ~20 SU | ~39 SU | ~38 SU |
| wall over 256 tasks | ~5 min | ~5 min | ~10 min | ~9 min |

**Total ≈ 120 SU** (plus per-task start-up; budget 150). Time limit in the script is 2 h.

## Files (this commit)
- `pengu_mujoco/grid6/hw_sweep.py`: `HW_CONFIGS` += c3/c4; `HW_TORSO=pid` (PID-only scoring,
  any κ, tag `hwcap_`); `HW_SERVO_LAG` env override (0 = no lag).
- `psc/hw_cap.slurm`: sets `HW_TORSO=pid HW_SERVO_LAG=0`, otherwise as `hw_sweep.slurm`
  (RM-shared, 256 single-core tasks, requeue, task record in `state/tasks.txt`).
- `psc/make_run_tree.sh`: copies the c3/c4 cell lists.
- `pengu_mujoco/results/grid6_hw/c3/cells_c3_mu0{12,45}_r1.csv`,
  `pengu_mujoco/results/grid6_hw/c4/cells_c4_mu0{12,45}_r1.csv` (force-added).

## Steps on Bridges-2 (login node, via the `psc` tmux session)
```bash
# 1. rebuild the isolated tree from the pushed branch (keeps the old one as pengu_hw_old if you want it)
cd $PROJECT && mv pengu_hw pengu_hw_cap_prev 2>/dev/null; bash <(curl -fsSL https://raw.githubusercontent.com/BenGu0530/PenguMujoco/friction-experiments/psc/make_run_tree.sh) friction-experiments
#    (or: git clone the branch and run psc/make_run_tree.sh friction-experiments)
cd $PROJECT/pengu_hw && ls pengu_mujoco/results/grid6_hw/c3 pengu_mujoco/results/grid6_hw/c4 psc/hw_cap.slurm

# 2. dry count (no SU)
cd pengu_mujoco && for c in c3 c4; do for m in 012 045; do CONFIG=$c HW_TORSO=pid HW_SERVO_LAG=0 python grid6/hw_sweep.py count --mu 0.${m#0} --cells-file results/grid6_hw/$c/cells_${c}_mu${m}_r1.csv; done; done

# 3. submit (≈120 SU total)
cd $PROJECT/pengu_hw && mkdir -p logs
sbatch --array=0-255 psc/hw_cap.slurm c3 0.12
sbatch --array=0-255 psc/hw_cap.slurm c3 0.45
sbatch --array=0-255 psc/hw_cap.slurm c4 0.12
sbatch --array=0-255 psc/hw_cap.slurm c4 0.45

# 4. after they finish: check every task recorded rc=0, then merge
grep -c "hwcap c3 0.12 .* rc=0" state/tasks.txt      # want 256, same for the other three
cd pengu_mujoco
for c in c3 c4; do for m in 0.12 0.45; do CONFIG=$c HW_TORSO=pid HW_SERVO_LAG=0 python grid6/hw_sweep.py --mu $m --merge; done; done
ls -la results/grid6_hw/c3/hwcap_* results/grid6_hw/c4/hwcap_*
# missing shards -> sbatch --array=<ids> psc/hw_cap.slurm c3 0.12 256  then merge again
```

## Analysis afterwards
`grid6/analysis/hw_vs_grid5.py` / `hw_compare.py` read `hwact_*` with mode `ff`; for these
files use mode `pid` (`hw_planes(cfg, mu, "pid")`) — pass gate stood ∧ straight > 0.5 ∧
v_net > 0.05, robust = nbhd ≥ 0.8 as before. Same-cell comparison against GRID-5 (ideal)
and against the c1/c2/c5/c6 hardware table is the point: cap alone vs cap + lag + FF.

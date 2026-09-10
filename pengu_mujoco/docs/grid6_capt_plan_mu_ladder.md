# GRID-6 torso-capped PID sweep: c1–c6 on the friction ladder (Ben 2026-09-10)

**What**: all six configs, κ PID torso (no lag, no feedforward), **354 °/s velocity cap on the legs
AND the torso servo** (`HW_TORSO_CAP=1`), ±4.1 N·m torque cap on every actuator (in the models),
hardware CAD models (c1/c4 → pengu1_05_hw_updated, c2/c5 → pengu1_20_hw_updated, c3/c6 → pengu1_31),
**μ = 0.1 / 0.3 / 0.5 / 0.7** — the GRID-5 ladder, so it lines up cell-for-cell with GRID-5 (ideal
actuators) and with the cap-only μ 0.12/0.45 table. Output `hwcapt_<cfg>_mu0{10,30,50,70}.csv`.

**No pruning (Ben 2026-09-10): the full candidate grid, 154,440 cells per (config, μ)** —
freq 1.20–1.70 @0.05 (11) × hip_phi 0–350 @10 (36) × leg_amp 70–130 @5 (13) × hip_amp 12–32 @4 (6)
× hip_off 20–40 @5 (5). `hw_sweep.cells()` generates it; the slurm script passes no cell list.
(The margin-1 lists `cells_c*_mu0{10,30,50,70}_r1.csv` are still in the tree but unused; they would
have saved ≈ 640 SU / 32 %.)

**Cost at the measured cap-only rates (κ0 1.65 s/cell, κ2 2.4 s/cell): κ0 3 × 4 × 154,440 =
1,853,280 cells ≈ 850 SU; κ2 the same count ≈ 1,240 SU; total ≈ 2,100 SU.** 24 array jobs of
256 tasks, ~600 cells per task: κ0 tasks ≈ 17 min, κ2 tasks ≈ 25 min wall.

## Steps on Bridges-2 (login node, `psc` tmux)
```bash
cd $PROJECT && mv pengu_hw pengu_hw_cap_$(date +%m%d) && git clone --quiet --depth 1 --branch friction-experiments https://github.com/robomechanics/pengu.git /tmp/pengu_src && bash /tmp/pengu_src/psc/make_run_tree.sh friction-experiments
cd $PROJECT/pengu_hw/pengu_mujoco && CONFIG=c6 HW_TORSO=pid HW_SERVO_LAG=0 HW_TORSO_CAP=1 python grid6/hw_sweep.py count --mu 0.1   # "grid 11x36x13x6x5 = 154,440 ... (torso too)"
cd $PROJECT/pengu_hw && mkdir -p logs
for c in c1 c2 c3 c4 c5 c6; do for m in 0.1 0.3 0.5 0.7; do sbatch --array=0-255 psc/hw_capt.slurm $c $m; done; done
# after: for each (c, m): grep -c "hwcapt $c $m .* rc=0" state/tasks.txt  -> 256
cd pengu_mujoco; for c in c1 c2 c3 c4 c5 c6; do for m in 0.1 0.3 0.5 0.7; do CONFIG=$c HW_TORSO=pid HW_SERVO_LAG=0 HW_TORSO_CAP=1 python grid6/hw_sweep.py --mu $m --merge; done; done
# copy hwcapt_*.csv to PenguMujoco_psc/g6capt/<cfg>/
```

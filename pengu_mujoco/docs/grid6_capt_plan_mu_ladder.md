# GRID-6 torso-capped PID sweep: c1–c6 on the friction ladder (Ben 2026-09-10)

**What**: all six configs, κ PID torso (no lag, no feedforward), **354 °/s velocity cap on the legs
AND the torso servo** (`HW_TORSO_CAP=1`), ±4.1 N·m torque cap on every actuator (in the models),
hardware CAD models (c1/c4 → pengu1_05_hw_updated, c2/c5 → pengu1_20_hw_updated, c3/c6 → pengu1_31),
**μ = 0.1 / 0.3 / 0.5 / 0.7** — the GRID-5 ladder, so it lines up cell-for-cell with GRID-5 (ideal
actuators) and with the cap-only μ 0.12/0.45 table. Output `hwcapt_<cfg>_mu0{10,30,50,70}.csv`.

Cell lists: hw_mask margin 1 against the GRID-5 map at the same μ ± one ladder step
(0.1→{0.1,0.3}, 0.3→{0.1,0.3,0.5}, 0.5→{0.3,0.5,0.7}, 0.7→{0.5,0.7}).

| cells | μ0.1 | μ0.3 | μ0.5 | μ0.7 |
|---|---|---|---|---|
| c1 | 83,747 | 90,298 | 93,718 | 91,663 |
| c2 | 96,608 | 109,268 | 110,395 | 104,543 |
| c3 | 73,981 | 85,723 | 77,255 | 64,551 |
| c4 | 141,240 | 149,353 | 136,906 | 110,456 |
| c5 | 152,118 | 152,387 | 120,768 | 68,030 |
| c6 | 144,666 | 144,723 | 65,133 | 37,436 |

**Cost at the measured cap-only rates (κ0 1.65 s/cell, κ2 2.4 s/cell): κ0 1.08 M cells ≈ 500 SU,
κ2 1.42 M cells ≈ 950 SU, total ≈ 1,450 SU.** 24 array jobs of 256 tasks; the longest tasks
(c5 μ0.1/0.3, c6 μ0.1/0.3) ≈ 25 min wall.

## Steps on Bridges-2 (login node, `psc` tmux)
```bash
cd $PROJECT && mv pengu_hw pengu_hw_cap_$(date +%m%d) && git clone --quiet --depth 1 --branch friction-experiments https://github.com/robomechanics/pengu.git /tmp/pengu_src && bash /tmp/pengu_src/psc/make_run_tree.sh friction-experiments
cd $PROJECT/pengu_hw/pengu_mujoco && ls results/grid6_hw/c*/cells_c*_mu0[1357]0_r1.csv | wc -l      # 24
CONFIG=c6 HW_TORSO=pid HW_SERVO_LAG=0 HW_TORSO_CAP=1 python grid6/hw_sweep.py count --mu 0.1 --cells-file results/grid6_hw/c6/cells_c6_mu010_r1.csv   # "torso mode pid  leg cap 354 deg/s (torso too)"
cd $PROJECT/pengu_hw && mkdir -p logs
for c in c1 c2 c3 c4 c5 c6; do for m in 0.1 0.3 0.5 0.7; do sbatch --array=0-255 psc/hw_capt.slurm $c $m; done; done
# after: for each (c, m): grep -c "hwcapt $c $m .* rc=0" state/tasks.txt  -> 256
cd pengu_mujoco; for c in c1 c2 c3 c4 c5 c6; do for m in 0.1 0.3 0.5 0.7; do CONFIG=$c HW_TORSO=pid HW_SERVO_LAG=0 HW_TORSO_CAP=1 python grid6/hw_sweep.py --mu $m --merge; done; done
# copy hwcapt_*.csv to PenguMujoco_psc/g6capt/<cfg>/
```

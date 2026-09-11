# GRID-7 高频补扫计划：1.75–2.00 Hz（2026-09-11）

## 为什么
GRID-7（`hwcapt_*`）步频只扫到 1.70 Hz。用 GRID-5 的 `fn_peak` 筛出的 c6 冰面（μ0.1）低冲击格子在 cap 物理下回放：
| 格子 | v (cap) | 零承重 | DF L/R | 峰值力 | 3×3 邻域 | motor bar | 最差离地 |
|---|---|---|---|---|---|---|---|
| 1.83/120/75/32/20 | 0.332 | 7.6 % | 0.62/0.53 | 2.0 W | 9/9 | 0.59 | 6.5 mm |
| 1.93/170/75/32/20 | 0.293 | 6.0 % | 0.66/0.62 | 1.9 W | 9/9 | 0.66 | 5.3 mm |
| 1.8/140/85/32/20 | 0.261 | 6.6 % | 0.63/0.60 | 1.6 W | 9/9 | 0.72 | 5.7 mm |
现 champion（第 4 名 1.6/300/95/28/20）：零承重 28 %、峰值 4.6 W。同参数的 1.70 Hz 格子在 GRID-7 里 pass，但 1.70 是边缘频率、邻域不完整，robust 判 NaN。
Ben 决定：不改现有扫描，补 1.75–2.00 Hz 这一片（六配置四 μ，保证各配置网格一致）。

## 扫什么
- 频率 1.75, 1.80, …, 2.00（6 档）× hip_phi 36 × leg_amp 13 × hip_amp 6 × hip_off 5 = **84,240 格 / (配置, μ)**，物理与 `psc/hw_capt.slurm` 完全一致。
- 输出 `results/grid6_hw/<cfg>/hwcapt_<cfg>_mu###_hi.<shard>.csv`，合并为 `hwcapt_<cfg>_mu###_hi.csv`（后缀 `_hi`，不碰原文件）。
- 代码：`grid6/hw_sweep.py` 新增环境变量 `HW_FREQ=lo:hi:step`（替换频率轴）和 `HW_TAG_SUFFIX`；`psc/hw_capt_hi.slurm`。
- 本地验证：`CONFIG=c6 HW_TORSO=pid HW_SERVO_LAG=0 HW_TORSO_CAP=1 HW_FREQ=1.75:2.00:0.05 HW_TAG_SUFFIX=_hi python grid6/hw_sweep.py count --mu 0.1` → 84,240。

## 代价（按已测 1.65 s/格 κ0、2.4 s/格 κ2）
| 范围 | SU |
|---|---|
| 每 (配置, μ) | κ0 ~39，κ2 ~56 |
| c6 四 μ | ~225 |
| c4–c6 四 μ | ~675 |
| **六配置四 μ（采用）** | **~1,140** |
余额（9/10）2,058 SU。256 任务/数组 → 329 格/任务，κ2 约 13 分钟/任务，2 h 上限充裕。

## Bridges-2 步骤（login 节点，`psc` tmux；sbatch 前先问 Ben）
```bash
# 1. 本地：commit + push friction-experiments（hw_sweep.py, psc/hw_capt_hi.slurm, 本文档）
# 2. PSC：重建隔离运行树（拿到新的 hw_sweep.py 和 slurm）
cd $PROJECT && mv pengu_hw pengu_hw_g7_$(date +%m%d) && git clone --quiet --depth 1 --branch friction-experiments https://github.com/robomechanics/pengu.git /tmp/pengu_src && bash /tmp/pengu_src/psc/make_run_tree.sh friction-experiments
cd $PROJECT/pengu_hw/pengu_mujoco && CONFIG=c6 HW_TORSO=pid HW_SERVO_LAG=0 HW_TORSO_CAP=1 HW_FREQ=1.75:2.00:0.05 HW_TAG_SUFFIX=_hi python grid6/hw_sweep.py count --mu 0.1   # 84,240
# 3. 提交（一次一个数组，QOS 上限 5000 任务/用户）
cd $PROJECT/pengu_hw && mkdir -p logs
for c in c1 c2 c3 c4 c5 c6; do for m in 0.1 0.3 0.5 0.7; do sbatch --array=0-255 psc/hw_capt_hi.slurm $c $m; done; done
# 4. 检查：grep -c "hwcapt_hi $c $m .* rc=0" state/tasks.txt -> 256；补跑失败分片：sbatch --array=<ids> psc/hw_capt_hi.slurm $c $m 256
# 5. 合并
cd pengu_mujoco; for c in c1 c2 c3 c4 c5 c6; do for m in 0.1 0.3 0.5 0.7; do CONFIG=$c HW_TORSO=pid HW_SERVO_LAG=0 HW_TORSO_CAP=1 HW_FREQ=1.75:2.00:0.05 HW_TAG_SUFFIX=_hi python grid6/hw_sweep.py --mu $m --merge; done; done
# 6. 拷回 PenguMujoco_psc/g6capt/<cfg>/hwcapt_<cfg>_mu###_hi.csv
```

## 数据回来后（本地）
1. `grid6/analysis/hw_vs_grid5.py`：`hw_file` / `hw_planes` 读 base + `_hi` 两份并拼接，`HW_AX` 的频率轴改为 1.20–2.00（17 档）；robust 邻域按新轴算，1.70 变内部点，2.00 成新边缘。
2. 重出 GRID-7 全部图表（分母 154,440 → 238,680）：robust 表、速度图、champion 列表（`hw_champions.py`）、COT、能量学、touchdown、duty strip、gait cycle。不重跑 champion 以外的仿真。
3. 记着：champion 规则里的离地 ≥ 10 mm 会把这批低冲击步态挡掉（它们 5–6.5 mm）；要不要改，Ben 定。

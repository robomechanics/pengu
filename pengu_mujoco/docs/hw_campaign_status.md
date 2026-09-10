# 硬件-仿真对照实验：现状与计划（更新于 c4 / c1-冰 硬件测试之后，2026-09-09）

只记事实和待定项；方向由 Ben 定。前一份汇总：`docs/stage_summary_2026-09-09.md`。

## 1. 仿真侧已完成

| 层 | 内容 | 数据 / 产物 |
|---|---|---|
| GRID-5（理想执行器、κ PID、slide COM 模型） | c1–c6、c10 完整；c8 部分 | `results/gait_sweep/`，图 `results/grid5_report/cross/`（23 张，待删减）、`cross4/`（四配置柱形图） |
| GRID-6 硬件层扫描（cap 354 + 56 ms + 前馈 torso，CAD 模型） | c1 / c2 / c5 / c6 × μ 0.12 / 0.45 | `PenguMujoco_psc/g6/…/hwact_*.csv`；图 `results/grid6_hw/figs/hw_robust.png, hw_speed.png` |
| GRID-6 cap-only（cap 354 + κ PID，无延迟无前馈） | c3 / c4 × μ 0.12 / 0.45 | `PenguMujoco_psc/g6cap/`；champion `results/grid6_hw/figs/hwcap_champions_robust_clear.json` |
| realism 重打分（GRID-5 格子，act / both） | c6 | `PenguMujoco_psc/g6/pengu_hw_old/…/realism_c6.csv` |
| champion 规则 | pass ∧ 邻域 ≥ 0.8 ∧ 最差周期离地 ≥ 10 mm，按全身 COM 净速度排；c6 再加"前馈相位窗口"重搜（5 个 lead 全走） | `results/grid6_hw/figs/hw_champions_robust_clear.json`；`results/grid6_hw/demos/lead_window_c6_mu0*.csv` |
| demo 视频 | 每个 champion 的 balance 叠加渲染 | `results/grid6_hw/demos/*.mp4` + 单支撑截图 |
| 单支撑 / 平衡分析 | COP→COM、GRF 作用线垂距、CMP、dH/dt；首触地力臂 vs μ（六配置） | `results/grid6_probes/`, `results/grid6_probes/touch_miss/touch_miss_vs_mu_robust.png` |
| 上一轮硬件 mocap vs 仿真 | 四个旧 champion 步态的复现表 | `docs/stage_summary_2026-09-09.md` §2 |

## 2. 固件（`Arduino/pengu_hw_c*/`，全部编译通过，未 commit）

| 配置 | κ / COM | 模型 | torso（开机） | preset 1 · μ0.12 冰 | sim v | preset 2 · μ0.45 地板 | sim v |
|---|---|---|---|---|---|---|---|
| c1 | 0 / 1.05 | pengu1_05_hw_updated | **FF** | 1.55/260/125/28/25 ff 17.8/165 | 0.276 | 1.65/30/80/32/40 ff 37.8/140 | 0.594 |
| c2 | 0 / 1.20 | pengu1_20_hw_updated | FF | 1.50/300/100/32/20 ff 23.0/78 | 0.486 | 1.65/270/100/24/30 ff 19.3/50 | 0.449 |
| c3 | 0 / 1.31 | pengu1_31 | **PID** | 1.25/280/125/24/20 | 0.142 | 1.65/310/130/20/20 | 0.248 |
| c4 | 2 / 1.05 | pengu1_05_hw_updated | **PID** | 1.65/290/95/32/25 | 0.366 | 1.65/260/130/32/40 | 0.466 |
| c5 | 2 / 1.20 | pengu1_20_hw_updated | FF | 1.65/230/130/32/25 ff 24.2/242 | 0.433 | 1.50/290/90/24/25 ff 17.6/218 | 0.405 |
| c6 | 2 / 1.31 | pengu1_31 | FF | 1.60/260/115/24/20 ff 18.6/209 | 0.19–0.44（5 lead） | 1.60/260/125/16/35 ff 8.3/75 | 0.19–0.25（5 lead） |

注意：**c3/c4 的 champion 来自 cap-only 扫描（PID torso），c1/c2/c5/c6 来自硬件层扫描（前馈 torso）**——两组的 torso 策略不同，横向比较时要记着。c6 的两个 preset 是相位窗口重搜过的；c1/c2/c5 还是 PSC 单一最佳 lead 选的，没做窗口重搜。

## 3. 硬件测试

### 已做
| 轮次 | 配置 / 表面 | 固件 | 状态 |
|---|---|---|---|
| 8/29–9/03（上一轮） | c1 冰 + 地板，c6 冰 + 地板 | GRID-4 champion + PID kp 0.5 | mocap / ROF 已分析，见 stage_summary §2 |
| **本次** | **c4 冰 + 地板** | pengu_hw_c4（PID，preset 1/2） | 数据待交给我 |
| **本次** | **c1 冰** | pengu_hw_c1（FF，preset 1） | 数据待交给我 |

### 计划
| 配置 / 表面 | 固件 | 仿真预测（同层） |
|---|---|---|
| c3 冰 + 地板 | pengu_hw_c3（PID） | 0.142 / 0.248 |
| c6 冰 + 地板 | pengu_hw_c6（FF，相位窗口版） | 0.19–0.44 / 0.19–0.25 |
| （待定）c1 地板、c2、c5 | 现有固件 | 见 §2 |

## 4. 本次数据回来后要做的（我这边）

1. 数据核对：mocap 导出（Motive CSV）+ ROF 供电日志 + 每个 take 的 preset / 表面 / 备注；跑两套速度算法（协作者管线 + `hw_speed_check.py`），出 v_fwd / v_net / v_path / 直线度 / 步频 / A rms。
2. 对照表：每组 硬件 vs 仿真 同步态同层（c4 = cap-only PID；c1 冰 = cap+lag+FF）+ 理想执行器 GRID-5 值，格式同 stage_summary §2.3。
3. 步频核对：从 mocap 反推 f_gait，确认机器人跑的是 preset 里的步态（上一轮就是靠这个对上固件的）。
4. 如果 c4 硬件也出现"仿真快、硬件慢"或"仿真摔、硬件走"，把它加进 sim2real 方向差异那一条，一起看四个嫌疑因素。

## 5. 待 Ben 定

1. c1/c2/c5 要不要也做相位窗口重搜（c6 那套，每配置每 μ 40 格 × 6 rollout，本地 10 分钟）。
2. torso 策略一致性：c3/c4 是 PID、其余是 FF。选项：(a) 保持，论文里分开写；(b) 给 c1/c2/c5/c6 也跑 cap-only PID 扫描（每对 μ 约 60–80 SU，脚本已有 `psc/hw_cap.slurm`）；(c) 给 c3/c4 跑硬件层 FF 扫描（每对约 400 SU）。
3. 29 ms 一阶滞后：四个硬件层配置统一没配。
4. GRID-5 cross 图 23 → 8–10 张。
5. 未提交文件：六份固件、`grid6/stance_com.py`、`grid6/render_forces.py`、`grid6/analysis/{hw_vs_grid5,hw_compare,touch_miss}.py`、demo 脚本、两份 doc；`pengu_tune_wifi.ino/webpage.ino` 有更早的未提交改动。
6. Onshape API key 撤销。

# Arduino

Board: **Arduino MKR WiFi 1010** (`arduino:samd:mkrwifi1010`), DynamixelShield, BNO055.
Libraries: WiFiNINA, DynamixelShield, Dynamixel2Arduino, Adafruit BNO055 + Unified Sensor.

There is no serial interface anywhere in the current firmware. The USB socket broke when the
robot fell and pulled the cable (2026-09-02); everything is a button at `http://192.168.4.1`
on the board's own AP, and a run is recorded into RAM and downloaded afterwards rather than
streamed, because a radio write inside the control loop perturbs the very period the
recording exists to measure.

Compiling without the IDE, which is also how these are checked before flashing:

    CLI="/Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/resources/arduino-cli"
    "$CLI" compile -b arduino:samd:mkrwifi1010 pengu_tune_wifi

## In use

| sketch | what it is |
|---|---|
| `pengu_tune_wifi/` | **the firmware.** Hand-tuning over WiFi, the on-board recorder, the torso strategies, the plant probe. Everything below is a bench tool. |
| `pengu_hw_c3/` `pengu_hw_c4/` | same, from the **cap-only** sweep (354 deg/s cap, κ PID torso, no lag / no FF modelled; `psc/hw_cap.slurm`). Boot into the PID torso; preset `1` = μ 0.12, `2` = μ 0.45. c3 κ0 COM 1.31, c4 κ2 COM 1.05. |
| `pengu_hw_c1/` `pengu_hw_c2/` `pengu_hw_c5/` `pengu_hw_c6/` | **pengu_tune_wifi with one config's champions baked in** (2026-09-09, GRID-6 hardware-model sweep: 354 deg/s cap, 56 ms, feedforward torso, CAD models; robust-neighbourhood + 10 mm clearance filters). `KAPPA` and the boot gait are set per config; preset `1` = μ 0.12 champion, `2` = μ 0.45 champion, `3` = the torso calibration gait. Set the ballast for the config's COM ratio (c1 1.05, c2 1.20, c5 1.20, c6 1.31) before walking. |
| `imu_probe/` | BNO055 read-out and calibration status |
| `motor_probe/` | Dynamixel scan, ping, single-joint moves |
| `hip_probe/` | hip range and zero check |
| `torso_rom/` | torso mechanical stops. Measured +27.8 / −31.0 deg, which is why the firmware clamps at ±25 |
| `zero_all/` | park every joint at its home angle |

## `pengu_tune_wifi` — the parts worth knowing

**Three torso strategies**, cycled with `T`, shown in the status line and logged per row:

- `PID` — the κ law, `J_cmd = (κ−1)·axis + Kp·e + Ki∫e + Kd·ė`. On this robot the
  correction has the right sign and arrives 56 ms late: the torso joint reaches its extreme
  56 ms after the hip axis reaches its own, by which time the axis is already returning in
  76–90% of events. It therefore pushes the lower body the way it is already going.
- `HELD` — torso commanded to its home angle, no controller. This is κ = 1.
- `FF` — **the boot default.** The torso is driven straight off the gait phase,
  `A·sin(phase + φ)`, with no measurement in the loop at all, so the sensor delay cannot
  enter; the servo's own lag is cancelled by leading φ instead.

Calibrated on the robot over ten bouts at `1.39/240/80/16/30`:

| torso | roll rms | cycles walked | first fall |
|---|---|---|---|
| HELD | 7.35° | 20/21 | none |
| PID | 34.09° | 11/21 | 11.5 s |
| **FF, φ 119, A 7.5** | **1.46°** | **21/21** | **none** |

The φ scan reads 4.83 / 1.54 / 1.46 / 3.03 / 4.89 / 5.34 at 94 / 114 / 119 / 129 / 144 /
159, and the amplitude 2.32 / 1.46 / 1.59 at 6.0 / 7.5 / 9.0. Splitting each bout into
thirds puts the run-to-run floor at 0.44°, so φ 114 and A 9.0 are **not** distinguishable
from the values flashed — the region is established, the exact point is not.

**The calibration is for κ = 0 only.** `ff_gain` scales the whole correction (1.0 = κ 0,
0 = torso held); other κ values need their own amplitude and phase and none is measured.

**The crank velocity ceiling is 354 ± 4 deg/s** — twelve measurements on 2026-08-30, air
and ground pooled, commanded 424–613. The status line prints `π·f·A_leg` and `2π·f·A_hip`
against it and appends `OVER` when the crank demand exceeds it. A gait above the ceiling is
not the gait that was selected: at 762 deg/s the crank executed 353 and arrived 90 ms late,
which shifted the executed `hip_phi` from the commanded 350 to 31.

**`P` runs a plant probe** instead of a gait: legs still, torso swept open-loop through
1.5–6 Hz, auto-stopping at the end so the ring holds exactly the sweep. Analysed by
`pengu_mujoco/hardware/plant_probe.py`.

**Recording.** 570 records of 28 bytes, starting one second before the gait blends in — the
staged start is 14 s and the ring holds about 15, so a bout logged from zero comes back
almost entirely ramp-and-settle. `DOWNLOAD` is a plain link to `/dump`; the browser saves
the CSV. Every dump opens with a comment line carrying the flashed constants (κ, gains,
clamp, ffLP, ffA/ffphi/ffgain, telemetry period, servo P gains) and every row carries the
robot state, the torso mode and the probe flag, so a recording says for itself what produced
it. The BNO's **gravity vector** is logged alongside the Euler angles because Euler pitch
wraps through ±180 whenever the roll gets large — which is exactly when the interesting
gaits run, and it cost three recordings their backward-fall counts.

Analysis lives in `pengu_mujoco/hardware/`: `phase_probe.py` for the harmonic fits and the
per-cycle walk/quiet/down segmentation, `plant_probe.py` for the torso plant transfer.

## `archive/`

26 sketches, one per gait that was flashed before the tuning firmware existed
(`pengu_champ_*`, `pengu_k0_*`, `pengu_k2_*`), plus the original `pengu/` they were all
copied from and `pengu_tune/`, the serial version of the tuner that the broken USB socket
retired. Kept because each one records the exact parameters of a bout that was actually run;
none is expected to be flashed again.

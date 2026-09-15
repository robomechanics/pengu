"""hw_sweep.py — ff_sweep re-run for the hardware session: measured friction, FF torso only.

Copied from ff_sweep.py (which produced the shipped ffact_mu05 data and is left untouched).
Differences: mu is the drag-measured static value (0.12 ice, 0.45 floor) instead of the
grid 0.1..0.7; the kappa = 0 PID rollout is dropped (the robot runs feedforward now); each
rollout also records `straight` = net displacement / low-passed path length, because the
robot's c1 ice takes walked in arcs (straightness 0.18-0.43 on half of them) and no
campaign metric sees that. Ben 2026-09-08: the leg model is a 354 deg/s HARD CAP only
(LEG_TAU = 0, no one-pole); speed is the whole-body COM (mass-weighted xipos), not the
root body; the torso clamp is the firmware's 25 deg; clear_ok flags clearance >= 10 mm.
Output goes to results/grid6_hw/, never grid6_report/.

    python grid6/hw_sweep.py count
    python grid6/hw_sweep.py --mu 0.12 --shard i --of n
    python grid6/hw_sweep.py --mu 0.12 --merge

Original ff_sweep header follows.

ff_sweep.py — score gaits under the torso the robot actually has.

Every map so far was scored with a torso loop that reads true state instantly at 1 kHz.
That robot does not exist. On this one the torso joint reaches its extreme 56 ms after the
hip axis reaches its own, by which time the axis is already returning in 76-90% of events,
and the kappa PID therefore pushes the lower body the way it is already going: the same
gait rolls 21 deg peak-to-peak with the torso held and 67 with the loop closed. A gait
picked under the ideal torso is picked in the wrong world.

So each cell is rolled out three ways, all of them carrying the 56 ms:

  HELD   torso commanded to home, no controller. This is kappa = 1 and it is also the
         measurement that supplies the feedforward: the hip-axis roll here, fitted at the
         gait frequency, is exactly what the torso has to cancel.
  FF     torso_deg = A0 * sin(phase + phi0 + 180 + lead), locked to the same phase the legs
         use. No measurement in the loop, so the sensor delay cannot enter; the servo's own
         lag is cancelled by leading the phase. `lead` is the one number that is not
         predictable from the fit -- on the robot the best value sat 43 deg past naive, in
         the model 47.5 and 65 for two different gaits -- so a few are tried and the best
         kept, which is what a calibration session does anyway.
  PID    the kappa = 0 loop as flashed, for the comparison.

Ranking is on the FF rollout. Recorded per rollout: net speed, per-cycle foot clearance,
roll phase drift, torso world roll (the kappa = 0 acceptance number), hip-axis roll, and
where the CoM sits fore-and-aft of the loaded feet -- the axis this robot falls about, which
no campaign has ever scored on.

The grid is centred on what walks on the robot: 1.39 / 240 / 80 / 16 / 25, and hip_off 0,
10 and 50 are dropped at Ben's instruction. Nothing is excluded for being too fast: the leg
servos are modelled instead, so a cell above the 354 deg/s ceiling is rolled out as the
clipped, delayed gait it actually becomes rather than pretended out of existence. That is
where the speed is -- inside the envelope the best cell reached 0.128 m/s, and ignoring the
ceiling entirely reached 0.394.

    python grid6/ff_sweep.py count
    python grid6/ff_sweep.py [--mu 0.5] [--shard i --of n]
    python grid6/ff_sweep.py --merge
"""
import argparse
import csv
import cmath
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.append(ROOT)
# hardware-model sweep table (Ben 2026-09-08): every config on its as-built CAD model
HW_CONFIGS = {"c1": (0.0, "pengu1_05_hw_updated"), "c2": (0.0, "pengu1_20_hw_updated"),
              "c5": (2.0, "pengu1_20_hw_updated"), "c6": (2.0, "1.31"),
              # Ben 2026-09-09: c3/c4 with the cap only -- no torso lag, no feedforward, the
              # kappa PID loop as in GRID-5 (HW_TORSO=pid HW_SERVO_LAG=0, psc/hw_cap.slurm)
              "c3": (0.0, "1.31"), "c4": (2.0, "pengu1_05_hw_updated")}
# HW_TORSO: "full" = held + 3 feedforward leads (+ PID for kappa=2), the hardware-layer table;
#           "pid"  = the kappa PID loop only, for any kappa (held/ff columns left blank)
HW_TORSO = os.environ.get("HW_TORSO", "full").lower()
assert HW_TORSO in ("full", "pid"), HW_TORSO
# HW_TORSO_CAP=1 (Ben 2026-09-10): the torso servo gets the same LEG_RATE velocity cap as the legs
# (its +-4.1 N.m torque cap is in every model's forcerange already). Output tag hwcapt_.
HW_TORSO_CAP = os.environ.get("HW_TORSO_CAP", "0") == "1"
CONFIG = os.environ.get("CONFIG", "c1").lower()
assert CONFIG in HW_CONFIGS, f"CONFIG={CONFIG!r} (want {sorted(HW_CONFIGS)})"
KAPPA, HW_MODEL = HW_CONFIGS[CONFIG]
os.environ["PENGU_MODEL"] = HW_MODEL

import mujoco                                    # noqa: E402
import gait_config as gc                         # noqa: E402
import gait_sweep as gs                          # noqa: E402
from torso_control import TorsoKappaPID          # noqa: E402
from friction_utils import set_floor_friction    # noqa: E402
from slip_metrics import contact_slip            # noqa: E402  (per-physics-step foot contact kinetics, 2026-09-14)

# ---------------------------------------------------------------- protocol, stated
REST_LEAN = 5.0                 # = grid6_sweep.REST_LEAN_DEG
SETTLE, WINDOW, FS = 2.0, 13.0, 200.0
SERVO_LAG = float(os.environ.get("HW_SERVO_LAG", "0.056"))   # measured: corr(J[k], goal[k-2]) = 0.984 at 28 ms/sample; 0 = no lag
LEADS = (30.0, 50.0, 70.0)      # deg past the naive cancelling phase
# The leg servos are modelled rather than fenced off. Cells above the ceiling used to be
# excluded from the grid, which hid the whole fast half of the space; instead the command
# is slew-limited and passed through a one-pole lag, so a cell above the ceiling is
# simulated as what it actually becomes. Calibrated against two hardware points on opposite
# sides of the ceiling, from the executed-vs-commanded harmonic fits:
#   1.46 Hz, peak demand 343 deg/s : robot ratio 0.991 lag 26 ms, model 0.965 / 28.3
#   1.95 Hz, peak demand 766       : robot ratio 0.534 lag 90 ms, model 0.554 / 89.5
# Raising LEG_RATE is how a voltage or motor change gets tested -- the map need not be
# rebuilt, only re-scored.
LEG_RATE = 354.0                # deg/s, twelve measurements 2026-08-30, air and ground
LEG_TAU = 0.0                   # Ben 2026-09-08: hard velocity cap only, no one-pole
# torso clamp as flashed: kappa=0 firmware (pengu_tune_wifi) 25 deg; kappa=2 firmware (pengu_champ,
# 2026-08-29) 45 deg. Set after the config table below.
TORSO_CLAMP_DEG = 25.0 if KAPPA == 0.0 else 45.0
CLEAR_MIN_MM = 10.0             # clear_ok flag threshold; the clear column stays for re-thresholding
CEILING = 1e9                   # no cell is excluded any more

# ---------------------------------------------------------------- the grid
# Ben 2026-09-10: the torso-capped ladder sweep runs the FULL candidate grid, no GRID-5 pruning, so
# cells() must be that grid (freq 0.05 step, hip_phi full circle) -- 11 x 36 x 13 x 6 x 5 = 154,440.
FREQ = [round(1.20 + 0.05 * k, 2) for k in range(11)]      # 1.20 .. 1.70
# HW_FREQ="1.75:2.00:0.05" (Ben 2026-09-11): replace the frequency axis, e.g. the high-frequency extension of GRID-7
# (the GRID-5 low-impact c6 ice gaits sit at 1.8-1.98 Hz, above the 1.70 Hz edge). Output tag gets HW_TAG_SUFFIX
# (e.g. "_hi") so the shard / merged files never collide with the base sweep's.
if os.environ.get("HW_FREQ"):
    _lo, _hi, _st = (float(x) for x in os.environ["HW_FREQ"].split(":"))
    FREQ = [round(_lo + _st * k, 2) for k in range(int(round((_hi - _lo) / _st)) + 1)]
TAG_SUFFIX = os.environ.get("HW_TAG_SUFFIX", "")
PHI = list(range(0, 360, 10))                              # 0 .. 350
LEG = list(range(70, 135, 5))                              # 70 .. 130
HIP = [12, 16, 20, 24, 28, 32]
OFF = [20, 25, 30, 35, 40]                                 # 0/10/50 dropped
OUT = os.path.join(ROOT, "results", "grid6_hw", CONFIG)
COLS = (["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "mu",
         "A0", "phi0", "best_lead"]
        + [f"{m}_{w}" for w in ("held", "ff", "pid")
           for m in ("fell", "v_net", "straight", "clear", "clear_ok", "drift", "rollrms",
                     "axisrms", "fore", "rearp5", "sat")])
# EXT metrics (Ben 2026-09-14, "store everything so we never sweep again"): appended AFTER the 42 legacy columns so every
# existing reader keeps working; pid rollout only (NaN in held/ff mode or when the cell fell). Window = the 13 s measurement
# window; "loaded" = foot normal force > FN_LOAD; per-physics-step quantities come from slip_metrics.contact_slip.
#   facing_deg      body +y heading (circular mean over the window) minus the direction of net COM travel, deg (0 = walks
#                   the way it faces, +-180 = backwards);  heading_align = cos(facing) (the GRID-5 gate was > 0.5)
#   n_steps         touchdowns of either foot (20 ms debounce; bounces and chatter add extra ones);  n_steps_clock = 2 x freq x window s
#                   (the steps the open-loop clock commands);  cadence = n_steps / window s
#   df_L, df_R      duty factor: loaded steps / window steps;  single/double/none_pct: one / both / no foot loaded
#   flight_pct      physics steps with no foot-floor contact at all;  zero_load_pct: both feet <= FN_LOAD
#   hop_max_mm      during no-contact steps, the lower foot's min geom z above its loaded-mean baseline, max over the window
#   fn_peak_w       peak total normal force / body weight
#   mu_req_p95/max  |F_t| / F_n of the more loaded foot (loaded steps);  p_slide_095/080 = share of loaded steps with mu_req > 0.95 / 0.8 mu
#   slip_L/R_mm     integral over loaded steps of |v_t| dt, v_t = foot material velocity at the contact point (force-weighted over contacts)
#   s_slip_per_step (slip_L + slip_R) / n_steps, mm;  L_step_nominal = swing-foot travel relative to the COM along the heading, toe-off ->
#                   touchdown, mm;  s_slip_ratio = s_slip_per_step / L_step_nominal;  eta = |net COM travel| / (n_steps x L_step_nominal)
#   v_t_max/mean    mm/s over loaded steps;  slip_dir_coh = |mean unit vector of v_t| over loaded steps with |v_t| > 1 mm/s (0..1)
#   e_pos           positive actuator work, J (sum max(tau*qdot, 0) dt);  cot_net = e_pos / (m g d_net);  cot_path = e_pos / (m g path_len)
#   path_len        raw COM path length, m;  com_z_mean / com_z_amp (p95 - p5), mm
#   tau_sat_*_pct   steps with |actuator force| >= 99 % of forcerange (torso / hips / cranks);  rate_lim_pct = steps in which the
#                   354 deg/s slew limiter clipped some leg command
EXT_COLS = ["facing_deg", "heading_align", "n_steps", "n_steps_clock", "cadence", "df_L", "df_R", "single_pct", "double_pct", "none_pct",
            "flight_pct", "zero_load_pct", "hop_max_mm", "fn_peak_w", "mu_req_p95", "mu_req_max", "p_slide_095", "p_slide_080",
            "slip_L_mm", "slip_R_mm", "s_slip_per_step", "L_step_nominal", "s_slip_ratio", "eta", "v_t_max", "v_t_mean", "slip_dir_coh",
            "e_pos", "cot_net", "cot_path", "path_len", "com_z_mean", "com_z_amp",
            "tau_sat_torso_pct", "tau_sat_hip_pct", "tau_sat_crank_pct", "rate_lim_pct",
            # Ben 2026-09-14, second batch: the forward / sideways split of the foot slip and of the
            # tangential foot velocity (EXT only had the magnitudes), the tangential GRF in newtons
            # (EXT only had the mu_req ratio), the heading-projected per-footfall speed that the mocap
            # pipeline reports (code/run_com.py v_fwd), and why the rollout stopped.
            #   slip_para_mm / slip_perp_mm  integral of |v_t . h| / |v_t . right| dt over loaded steps,
            #                                both feet summed, h = body +y heading at that step
            #   s_perp_ratio                 slip_perp / (slip_para + slip_perp)
            #   v_para_mean / v_perp_mean / v_perp_max   mm/s over loaded steps
            #   ft_peak_N / ft_mean_N        |tangential contact force| of the more loaded foot
            #   fn_peak_N                    peak total normal force, N (fn_peak_w is the same / body weight)
            #   v_fwd / v_lat                COM displacement between consecutive touchdowns (either foot)
            #                                projected on the mean heading over that step, pooled: sum d / sum dt
            #   heading_drift_deg            atan2(sum d_lat, sum d_fwd)
            #   term_reason                  "ok" | "hop" | "backward"   (term_t = window time it stopped, s)
            "slip_para_mm", "slip_perp_mm", "s_perp_ratio", "v_para_mean", "v_perp_mean", "v_perp_max",
            "ft_peak_N", "ft_mean_N", "fn_peak_N", "v_fwd", "v_lat", "heading_drift_deg", "term_reason", "term_t"]
COLS = COLS + EXT_COLS
FN_LOAD = 2.0                   # N, a foot counts as loaded above this (stance_com.FN_MIN)
# Early termination (Ben 2026-09-14): a hop or a backwards walk shows in the first few gait cycles,
# so there is no point paying for the remaining 8 s of the window. HW_EARLY_S=0 disables it.
# A terminated row keeps term_reason / term_t; its OTHER columns cover the shortened window only
# and must not be compared with a full-window row -- filter on term_reason == "ok" first.
EARLY_S = float(os.environ.get("HW_EARLY_S", "5.0"))
HOP_MM = float(os.environ.get("HW_HOP_MM", "2.0"))        # both feet off the floor by more than this = hop
FACE_MAX = float(os.environ.get("HW_FACE_MAX", "90.0"))   # |body heading - travel direction|, deg
DEBOUNCE_STEPS = 20             # physics steps (20 ms) a load change must persist to count as touchdown / toe-off


def cells():
    out = []
    for f in FREQ:
        for a in LEG:
            if math.pi * f * a > CEILING:
                continue
            for h in HIP:
                if 2 * math.pi * f * h > CEILING:
                    continue
                for p in PHI:
                    for o in OFF:
                        out.append((f, float(p), float(a), float(h), float(o)))
    return out


def fit(t, y, f):
    """amplitude, phase and residual of y at f, in the controller's phase reference."""
    y = np.asarray(y) - np.mean(y)
    w = 2 * math.pi * f * np.asarray(t)
    M = np.column_stack([np.ones_like(w), np.sin(w), np.cos(w)])
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    return (math.hypot(c[1], c[2]), math.degrees(math.atan2(c[2], c[1])),
            float(np.std(y - M @ c)))


def rollout(freq, phi, leg, hip, off, mu, mode, A=0.0, ph=0.0, kappa=0.0):
    """One bout. mode is 'held', 'ff' or 'pid'; all three carry the servo lag."""
    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    gc.RAMP_HIP_OFFSET = True
    gs.STAGED_START = True
    gc.STAND_HIP_DEG = 0.0                       # the PID calibrates its neutral at hips-0
    pid = TorsoKappaPID(model, kappa=kappa, measure_after=0.0, ctrl_limit_deg=TORSO_CLAMP_DEG)
    gc.STAND_HIP_DEG = REST_LEAN
    set_floor_friction(model, mu)
    gs.FLOOR_MU = mu
    gs.CONDITION["hip_off"] = off
    gs._set_gait(dict(freq=freq, hip_phi=phi, leg_amp=leg, hip_amp=hip))
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)
    floor_id, foot_geom, foot_bid, root = gs.make_ids(model)
    legs = [act[n] for n in ("crank1-L", "crank1-R", "hip-L", "hip-R")] + ([act["torso"]] if HW_TORSO_CAP else [])
    slew = math.radians(LEG_RATE) * model.opt.timestep
    a_lp = model.opt.timestep / (model.opt.timestep + LEG_TAU)
    held_cmd = None
    lagged = None
    zg = {s: [g for g, sd in foot_geom.items() if sd == s] for s in ("L", "R")}
    fb = {s: [b for b, sd in foot_bid.items() if sd == s][0] for s in ("L", "R")}
    dt_ = float(model.opt.timestep)
    mass_tot = float(model.body_mass[1:].sum())
    tau_lim = np.abs(model.actuator_forcerange).max(1); tau_lim = np.where(tau_lim > 0, tau_lim, np.inf)
    grp = {"torso": [act["torso"]], "hip": [act["hip-L"], act["hip-R"]], "crank": [act["crank1-L"], act["crank1-R"]]}
    E = dict(freq=freq, nwin=0, zero=0, nofoot=0, one=0, both=0, ld={"L": 0, "R": 0}, slip={"L": 0.0, "R": 0.0}, vt_max=0.0, vt_sum=0.0, vt_n=0,
             coh=np.zeros(2), coh_n=0, mu=[], fn_peak=0.0, air_z=[], e_pos=0.0, sat={"torso": 0, "hip": 0, "crank": 0}, rate=0,
             st={"L": False, "R": False}, cand={"L": 0, "R": 0}, n_td={"L": 0, "R": 0}, last_to={"L": None, "R": None}, step_nom=[],
             s_para=0.0, s_perp=0.0, vpara_sum=0.0, vperp_sum=0.0, vperp_max=0.0,
             ft_peak=0.0, ft_sum=0.0, ft_n=0, td_seq=[],
             zl={"L": [0.0, 0], "R": [0.0, 0]}, hop_hit=False, term=None, term_t=float("nan"))
    over_any = False

    buf = []

    def ctrl(d, t, alpha=1.0):
        if mode == "pid":
            u = pid(d, t, alpha)
        elif mode == "held":
            u = 0.0
        else:
            # torso joint = s * (kappa-1) * axis roll, phase-locked: A is the fitted axis
            # amplitude, ph its phase plus the lead; the sign folds into a 180 deg shift
            g = pid.s * (kappa - 1.0)
            w = 2 * math.pi * freq * (t - gc.T_HOLD - gc.T_TRANSITION)
            u = alpha * math.radians(abs(g) * A) * math.sin(
                w + math.radians(ph + (180.0 if g < 0 else 0.0)))
            u = max(-pid.limit, min(pid.limit, u))
        buf.append((t, u))
        while len(buf) > 1 and buf[1][0] <= t - SERVO_LAG:
            buf.pop(0)
        return buf[0][1]
    gc.TORSO_CONTROLLER = ctrl

    gc.T_HOLD = 1e9
    t0 = None
    nxt = 0.0
    T, AX, RO, Z, LOAD, FORE, REAR, POS, SAT = [], [], [], {"L": [], "R": []}, \
        {"L": [], "R": []}, [], [], [], []
    COMZ, HEAD = [], []
    fell = None
    while True:
        if t0 is None:
            tt = data.time
            if (tt >= gs.QUIET_MIN_T and float(np.max(np.abs(data.qvel))) < gs.QUIET_QVEL) \
                    or tt >= gs.QUIET_MAX_T:
                t0 = tt
                gc.T_HOLD = tt
        gc.apply_ctrl(data, act, data.time)
        cur = np.array([data.ctrl[i] for i in legs])
        if held_cmd is None:
            held_cmd = cur.copy()
            lagged = cur.copy()
        over_any = bool((np.abs(cur - held_cmd) > slew * 1.000001).any())    # the slew limiter clips some leg command this step
        held_cmd += np.clip(cur - held_cmd, -slew, slew)
        lagged += a_lp * (held_cmd - lagged)
        for i, j in enumerate(legs):
            data.ctrl[j] = lagged[i]
        mujoco.mj_step(model, data)
        if t0 is None:
            continue
        if data.xpos[root][2] < 0.05 and fell is None:
            fell = data.time - t0
            break
        tw = data.time - t0 - gc.T_TRANSITION - SETTLE
        if tw < 0:
            continue
        if tw > WINDOW:
            break
        if EARLY_S > 0 and E["term"] is None and tw > EARLY_S:
            if E["hop_hit"]:
                E["term"], E["term_t"] = "hop", tw
            elif len(POS) > 10:                                   # backwards / sideways walk
                q5 = np.array(POS); trav = q5[-1] - q5[0]; hv5 = np.array(HEAD).sum(0)
                if float(np.linalg.norm(trav)) > 0.02 and float(np.linalg.norm(hv5)) > 1e-9:
                    al = float(np.dot(hv5 / np.linalg.norm(hv5), trav / np.linalg.norm(trav)))
                    if al < math.cos(math.radians(FACE_MAX)):
                        E["term"], E["term_t"] = "backward", tw
            if E["term"] is not None:
                break
        # ---- EXT: every physics step inside the window (slip needs the integral, impacts need the 1 kHz peak)
        cs = contact_slip(model, data, foot_geom)
        E["nwin"] += 1
        ld = {s_: cs[s_]["fn"] > FN_LOAD for s_ in ("L", "R")}
        nld = int(ld["L"]) + int(ld["R"])
        if nld == 0:
            E["zero"] += 1
        elif nld == 1:
            E["one"] += 1
        else:
            E["both"] += 1
        if cs["L"]["n"] + cs["R"]["n"] == 0:
            E["nofoot"] += 1
            zL = min(data.geom_xpos[g][2] for g in zg["L"]); zR = min(data.geom_xpos[g][2] for g in zg["R"])
            E["air_z"].append((zL, zR))
            if E["zl"]["L"][1] and E["zl"]["R"][1]:                # both feet clear of their own stance height
                dL = zL - E["zl"]["L"][0] / E["zl"]["L"][1]; dR = zR - E["zl"]["R"][0] / E["zl"]["R"][1]
                if min(dL, dR) * 1000.0 > HOP_MM:
                    E["hop_hit"] = True
        E["fn_peak"] = max(E["fn_peak"], cs["L"]["fn"] + cs["R"]["fn"])
        mu_step = -1.0
        Rm_s = data.xmat[root].reshape(3, 3)                       # body +y heading this step, for the
        hd_s = Rm_s[:2, 1] / max(float(np.linalg.norm(Rm_s[:2, 1])), 1e-9)   # forward / sideways slip split
        rt_s = np.array([hd_s[1], -hd_s[0]])                       # body right
        for s_ in ("L", "R"):
            if ld[s_]:
                E["ld"][s_] += 1
                sp = float(math.hypot(cs[s_]["vt"][0], cs[s_]["vt"][1]))
                E["slip"][s_] += sp * dt_; E["vt_sum"] += sp; E["vt_n"] += 1
                vp = float(np.dot(cs[s_]["vt"][:2], hd_s)); vq = float(np.dot(cs[s_]["vt"][:2], rt_s))
                E["s_para"] += abs(vp) * dt_; E["s_perp"] += abs(vq) * dt_
                E["vpara_sum"] += abs(vp); E["vperp_sum"] += abs(vq)
                if abs(vq) > E["vperp_max"]:
                    E["vperp_max"] = abs(vq)
                ftn = float(cs[s_]["ft"])
                E["ft_sum"] += ftn; E["ft_n"] += 1
                if ftn > E["ft_peak"]:
                    E["ft_peak"] = ftn
                E["zl"][s_][0] += min(data.geom_xpos[g][2] for g in zg[s_]); E["zl"][s_][1] += 1
                if sp > E["vt_max"]:
                    E["vt_max"] = sp
                if sp > 1e-3:
                    E["coh"] += cs[s_]["vt"][:2] / sp; E["coh_n"] += 1
                mu_step = max(mu_step, cs[s_]["ft"] / cs[s_]["fn"])
            # touchdown / toe-off with a 20 ms debounce; the nominal step = swing-foot travel relative to the COM along the heading
            if ld[s_] == E["st"][s_]:
                E["cand"][s_] = 0
            else:
                E["cand"][s_] += 1
                if E["cand"][s_] >= DEBOUNCE_STEPS:
                    E["st"][s_] = ld[s_]; E["cand"][s_] = 0
                    com_ = (data.xipos[1:] * model.body_mass[1:, None]).sum(0)[:2] / mass_tot
                    Rm_ = data.xmat[root].reshape(3, 3); hd_ = Rm_[:2, 1] / max(float(np.linalg.norm(Rm_[:2, 1])), 1e-9)
                    ev = (data.xpos[fb[s_]][:2].copy(), com_.copy(), hd_)
                    if ld[s_]:
                        E["n_td"][s_] += 1
                        E["td_seq"].append((float(data.time), com_.copy(), hd_.copy()))
                        if E["last_to"][s_] is not None:
                            f0, c0, h0 = E["last_to"][s_]
                            E["step_nom"].append(abs(float(np.dot((ev[0] - f0) - (ev[1] - c0), h0))))
                    else:
                        E["last_to"][s_] = ev
        if mu_step >= 0:
            E["mu"].append(mu_step)
        E["e_pos"] += float(np.sum(np.maximum(data.qfrc_actuator * data.qvel, 0.0))) * dt_
        af = np.abs(data.actuator_force)
        for g_, ids_ in grp.items():
            if any(af[i_] >= 0.99 * tau_lim[i_] for i_ in ids_):
                E["sat"][g_] += 1
        if over_any:
            E["rate"] += 1
        if data.time < nxt:
            continue
        nxt = data.time + 1.0 / FS
        h = pid.hinge(data)
        T.append(data.time)
        AX.append(math.degrees(pid.axis_roll(data, h)))
        RO.append(math.degrees(pid.torso_roll(data, h)))
        SAT.append(abs(data.ctrl[act["torso"]]) > pid.limit - 1e-9)
        loaded = {"L": False, "R": False}
        pts, wts = [], []
        for ci in range(data.ncon):
            c = data.contact[ci]
            hit = [g for g in (c.geom1, c.geom2) if g in foot_geom]
            if not hit:
                continue
            loaded[foot_geom[hit[0]]] = True
            fv = np.zeros(6)
            mujoco.mj_contactForce(model, data, ci, fv)
            if abs(float(fv[0])) > 1e-6:
                pts.append(c.pos[:2].copy())
                wts.append(abs(float(fv[0])))
        for s in ("L", "R"):
            Z[s].append(min(data.geom_xpos[g][2] for g in zg[s]))
            LOAD[s].append(loaded[s])
        com = (data.xipos[1:] * model.body_mass[1:, None]).sum(0) \
            / model.body_mass[1:].sum()
        POS.append(com[:2].copy())              # whole-body COM, not the root body
        COMZ.append(float(com[2]))
        Rh = data.xmat[root].reshape(3, 3)
        HEAD.append(Rh[:2, 1].copy())           # body +y (front) axis, world xy
        if pts:
            pts = np.array(pts)
            wts = np.array(wts)
            cop = (pts * wts[:, None]).sum(0) / wts.sum()
            R = data.xmat[root].reshape(3, 3)
            fvv = R[:2, 1]
            nf = float(np.linalg.norm(fvv))
            if nf > 1e-9:
                fh = fvv / nf
                FORE.append(float(np.dot(com[:2] - cop, fh)) * 1000.0)
                REAR.append(float(np.dot(com[:2], fh) - (pts @ fh).min()) * 1000.0)
    gc.T_HOLD = 5.0
    if fell is not None or len(T) < 100:
        return dict(fell=fell if fell is not None else 0.0,
                    _ext={"term_reason": E["term"] or ("fell" if fell is not None else "short"), "term_t": E["term_t"]})

    t = np.array(T)
    ax = np.array(AX)
    out = dict(fell=None, rollrms=float(np.std(RO)), axisrms=float(np.std(ax)),
               sat=100.0 * float(np.mean(SAT)),
               fore=float(np.mean(FORE)) if FORE else float("nan"),
               rearp5=float(np.percentile(REAR, 5)) if REAR else float("nan"))
    q = np.array(POS)
    out["v_net"] = float(np.linalg.norm(q[-1] - q[0])) / (t[-1] - t[0])
    # straightness: net displacement over the path length of the root, the path taken on
    # the trajectory low-passed over one gait period so the waddle does not count
    k = max(1, int(round(FS / freq)))
    ker = np.ones(k) / k
    if len(q) > 2 * k:
        ql = np.column_stack([np.convolve(q[:, i], ker, "valid") for i in range(2)])
        path = float(np.sum(np.hypot(np.diff(ql[:, 0]), np.diff(ql[:, 1]))))
        out["straight"] = float(np.linalg.norm(ql[-1] - ql[0]) / path) if path > 1e-6 else float("nan")
    else:
        out["straight"] = float("nan")
    # per-foot per-cycle clearance apex, the minimum over cycles
    mins = []
    for s in ("L", "R"):
        z = np.array(Z[s])
        ld = np.array(LOAD[s])
        base = float(z[ld].mean()) if ld.any() else float(z.min())
        clr = (z - base) * 1000.0
        apex = [float(clr[(t >= t[0] + k / freq) & (t < t[0] + (k + 1) / freq)].max())
                for k in range(int((t[-1] - t[0]) * freq))
                if ((t >= t[0] + k / freq) & (t < t[0] + (k + 1) / freq)).sum() > 5]
        mins.append(min(apex) if apex else float("nan"))
    out["clear"] = min(mins)
    out["clear_ok"] = int(out["clear"] >= CLEAR_MIN_MM) if np.isfinite(out["clear"]) else 0
    # roll phase drift, per cycle
    y = np.array(RO) - np.mean(RO)
    psis = []
    for k in range(int((t[-1] - t[0]) * freq)):
        m = (t >= t[0] + k / freq) & (t < t[0] + (k + 1) / freq)
        if m.sum() > 5:
            psis.append(fit(t[m], y[m], freq)[1])
    d = [abs((psis[j] - psis[j - 1] + 180) % 360 - 180) for j in range(1, len(psis))]
    out["drift"] = float(np.mean(d)) if d else float("nan")
    out["_fit"] = fit(t - gc.T_HOLD - gc.T_TRANSITION, ax, freq)
    out["_ext"] = ext_metrics(E, q, t, np.array(COMZ), np.array(HEAD), Z, LOAD, mass_tot)
    return out


def ext_metrics(E, q, t, comz, head, Z, LOAD, mass_tot):
    nw = max(E["nwin"], 1)
    d_net = float(np.linalg.norm(q[-1] - q[0])); dur = float(t[-1] - t[0])
    hv = head.sum(0); trav = q[-1] - q[0]
    facing = float("nan"); align = float("nan")
    if np.linalg.norm(hv) > 1e-9 and np.linalg.norm(trav) > 1e-9:
        facing = math.degrees(math.atan2(hv[1], hv[0]) - math.atan2(trav[1], trav[0]))
        facing = (facing + 180.0) % 360.0 - 180.0
        align = float(np.dot(hv / np.linalg.norm(hv), trav / np.linalg.norm(trav)))
    n_td = E["n_td"]["L"] + E["n_td"]["R"]
    s_tot = E["slip"]["L"] + E["slip"]["R"]
    L_nom = float(np.mean(E["step_nom"])) if E["step_nom"] else float("nan")
    per_step = (s_tot / n_td) if n_td else float("nan")
    mu = np.array(E["mu"]) if E["mu"] else np.array([np.nan])
    hop = float("nan")
    if E["air_z"]:
        base = {}
        for s_ in ("L", "R"):
            z = np.array(Z[s_]); ldm = np.array(LOAD[s_], bool)
            base[s_] = float(z[ldm].mean()) if ldm.any() else float(z.min())
        az = np.array(E["air_z"])
        hop = float(np.max(np.minimum(az[:, 0] - base["L"], az[:, 1] - base["R"]))) * 1000.0
    path = float(np.sum(np.hypot(np.diff(q[:, 0]), np.diff(q[:, 1]))))
    mg = mass_tot * 9.81
    return dict(facing_deg=facing, heading_align=align, n_steps=n_td, n_steps_clock=2.0 * E["freq"] * dur,   # commanded steps: 2 per clock period
                cadence=n_td / dur if dur > 0 else float("nan"),
                df_L=E["ld"]["L"] / nw, df_R=E["ld"]["R"] / nw, single_pct=100.0 * E["one"] / nw, double_pct=100.0 * E["both"] / nw,
                none_pct=100.0 * E["zero"] / nw, flight_pct=100.0 * E["nofoot"] / nw, zero_load_pct=100.0 * E["zero"] / nw, hop_max_mm=hop,
                fn_peak_w=E["fn_peak"] / mg, mu_req_p95=float(np.nanpercentile(mu, 95)), mu_req_max=float(np.nanmax(mu)),
                p_slide_095=float(np.mean(mu > 0.95 * gs.FLOOR_MU)) if E["mu"] else float("nan"),
                p_slide_080=float(np.mean(mu > 0.8 * gs.FLOOR_MU)) if E["mu"] else float("nan"),
                slip_L_mm=E["slip"]["L"] * 1000.0, slip_R_mm=E["slip"]["R"] * 1000.0, s_slip_per_step=per_step * 1000.0 if n_td else float("nan"),
                L_step_nominal=L_nom * 1000.0 if np.isfinite(L_nom) else float("nan"),
                s_slip_ratio=(per_step / L_nom) if (n_td and np.isfinite(L_nom) and L_nom > 0) else float("nan"),
                eta=(d_net / (n_td * L_nom)) if (n_td and np.isfinite(L_nom) and L_nom > 0) else float("nan"),
                v_t_max=E["vt_max"] * 1000.0, v_t_mean=(E["vt_sum"] / E["vt_n"] * 1000.0) if E["vt_n"] else float("nan"),
                slip_dir_coh=float(np.linalg.norm(E["coh"] / E["coh_n"])) if E["coh_n"] else float("nan"),
                e_pos=E["e_pos"], cot_net=(E["e_pos"] / (mg * d_net)) if d_net > 1e-6 else float("nan"),
                cot_path=(E["e_pos"] / (mg * path)) if path > 1e-6 else float("nan"), path_len=path,
                com_z_mean=float(np.mean(comz)) * 1000.0, com_z_amp=float(np.percentile(comz, 95) - np.percentile(comz, 5)) * 1000.0,
                tau_sat_torso_pct=100.0 * E["sat"]["torso"] / nw, tau_sat_hip_pct=100.0 * E["sat"]["hip"] / nw,
                tau_sat_crank_pct=100.0 * E["sat"]["crank"] / nw, rate_lim_pct=100.0 * E["rate"] / nw,
                slip_para_mm=E["s_para"] * 1000.0, slip_perp_mm=E["s_perp"] * 1000.0,
                s_perp_ratio=(E["s_perp"] / (E["s_para"] + E["s_perp"])) if (E["s_para"] + E["s_perp"]) > 1e-12 else float("nan"),
                v_para_mean=(E["vpara_sum"] / E["vt_n"] * 1000.0) if E["vt_n"] else float("nan"),
                v_perp_mean=(E["vperp_sum"] / E["vt_n"] * 1000.0) if E["vt_n"] else float("nan"),
                v_perp_max=E["vperp_max"] * 1000.0,
                ft_peak_N=E["ft_peak"], ft_mean_N=(E["ft_sum"] / E["ft_n"]) if E["ft_n"] else float("nan"),
                fn_peak_N=E["fn_peak"], **_fwd_lat(E["td_seq"]),
                term_reason=E["term"] or "ok", term_t=E["term_t"])


def _fwd_lat(td):
    """v_fwd / v_lat the way the mocap pipeline does it (code/run_com.py): between consecutive
    touchdowns of EITHER foot, project the COM displacement on the mean heading over that step,
    then pool (sum of distances / sum of times). Not a path length -- the waddle cancels."""
    if len(td) < 2:
        return dict(v_fwd=float("nan"), v_lat=float("nan"), heading_drift_deg=float("nan"))
    d_f = d_l = dt_tot = 0.0
    for (t0, c0, h0), (t1, c1, h1) in zip(td[:-1], td[1:]):
        hb = h0 + h1
        n = float(np.linalg.norm(hb))
        if n < 1e-9 or t1 <= t0:
            continue
        hb = hb / n
        rt = np.array([hb[1], -hb[0]])
        dr = c1 - c0
        d_f += float(np.dot(dr, hb)); d_l += float(np.dot(dr, rt)); dt_tot += (t1 - t0)
    if dt_tot <= 0:
        return dict(v_fwd=float("nan"), v_lat=float("nan"), heading_drift_deg=float("nan"))
    return dict(v_fwd=d_f / dt_tot, v_lat=d_l / dt_tot,
                heading_drift_deg=math.degrees(math.atan2(d_l, d_f)))


def blank():
    return dict(fell=float("nan"), v_net=float("nan"), straight=float("nan"), clear=float("nan"),
                clear_ok=0,
                drift=float("nan"), rollrms=float("nan"), axisrms=float("nan"),
                fore=float("nan"), rearp5=float("nan"), sat=float("nan"))


def score(cell, mu):
    f, phi, leg, hip, off = cell
    if HW_TORSO == "pid":
        # cap-only table: one rollout, the kappa PID loop; a fall is a row with fell set
        pid = rollout(f, phi, leg, hip, off, mu, "pid", kappa=KAPPA)
        row = list(cell) + [mu, float("nan"), float("nan"), float("nan")]
        for r in (blank(), blank(), pid):
            for k in ("fell", "v_net", "straight", "clear", "clear_ok", "drift", "rollrms",
                      "axisrms", "fore", "rearp5", "sat"):
                v = r.get(k, float("nan"))
                row.append("" if v is None else (round(v, 4) if isinstance(v, float) else v))
        ext = pid.get("_ext", {})
        for k in EXT_COLS:
            v = ext.get(k, "" if k == "term_reason" else float("nan"))
            row.append(round(v, 5) if isinstance(v, float) else v)
        return row
    held = rollout(f, phi, leg, hip, off, mu, "held")
    if held.get("fell") is not None:
        return None                       # cannot even stand the gait with a passive torso
    A0, p0, _ = held["_fit"]
    best, best_lead = None, float("nan")
    for lead in LEADS:
        # feedforward for any kappa: torso joint = (kappa-1) * axis roll, phase-locked.
        # kappa=0 cancels the roll (amplitude A0, 180 deg), kappa=2 leans with it (+A0).
        r = rollout(f, phi, leg, hip, off, mu, "ff", A=A0, ph=p0 + lead, kappa=KAPPA)
        if r.get("fell") is not None:
            continue
        if best is None or r["rollrms"] < best["rollrms"]:
            best, best_lead = r, lead
    # the loop the robot actually ran for kappa=2 (pengu_champ 08-29), with the 56 ms in it;
    # for kappa=0 that loop is the documented failure mode and is not rolled out
    pid = rollout(f, phi, leg, hip, off, mu, "pid", kappa=KAPPA) if KAPPA != 0.0 else blank()
    row = list(cell) + [mu, round(A0, 3), round(p0 % 360, 1), best_lead]
    for r in (held, best if best else blank(), pid):
        for k in ("fell", "v_net", "straight", "clear", "clear_ok", "drift", "rollrms",
                  "axisrms", "fore", "rearp5", "sat"):
            v = r.get(k, float("nan"))
            row.append("" if v is None else (round(v, 4) if isinstance(v, float) else v))
    row += [float("nan")] * len(EXT_COLS)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run")
    ap.add_argument("--mu", type=float, default=0.12)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--cells-file", default="",
                    help="csv of freq,hip_phi,leg_amp,hip_amp,hip_off from hw_mask.py; replaces the grid")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = f"{('hwcapt' if HW_TORSO_CAP else 'hwcap') if HW_TORSO == 'pid' else 'hwact'}_{CONFIG}_mu{int(round(a.mu * 100)):03d}{TAG_SUFFIX}"
    if a.cells_file:
        with open(a.cells_file) as fh:
            rd = csv.DictReader(fh)
            cl = [tuple(float(r[k]) for k in ("freq", "hip_phi", "leg_amp", "hip_amp", "hip_off"))
                  for r in rd]
    else:
        cl = cells()

    if a.cmd == "count":
        full = len(FREQ) * len(PHI) * len(LEG) * len(HIP) * len(OFF)
        print(f"grid {len(FREQ)}x{len(PHI)}x{len(LEG)}x{len(HIP)}x{len(OFF)} = {full:,}")
        print(f"inside the {CEILING:.0f} deg/s envelope: {len(cl):,} cells")
        npid = 1 if KAPPA != 0.0 else 0
        print(f"{CONFIG}: kappa={KAPPA} model={HW_MODEL}  torso clamp {TORSO_CLAMP_DEG:.0f}  "
              f"torso mode {HW_TORSO}  leg cap {LEG_RATE:.0f} deg/s{' (torso too)' if HW_TORSO_CAP else ''}  servo lag {SERVO_LAG*1000:.0f} ms")
        if HW_TORSO == "pid":
            print(f"rollouts: {len(cl)} x 1 pid = {len(cl):,}")
        else:
            print(f"rollouts: {len(cl)} x (1 held + {len(LEADS)} ff + {npid} pid) "
                  f"= {len(cl) * (1 + len(LEADS) + npid):,}")
        return

    if a.merge:
        # Discover the shards instead of assuming how many there are. The array
        # size is chosen at submit time (--array=0-255 today, wider if the run
        # is spread over more cores), so a hardcoded range silently drops every
        # shard above it -- at 256 tasks the old range(64) kept a quarter of the
        # grid and reported success. A gap in the middle means a task died;
        # those cells are simply absent, so say so rather than merge a hole.
        pre = tag + "."
        found = {}
        for fn in os.listdir(OUT):
            if fn.startswith(pre) and fn.endswith(".csv"):
                mid = fn[len(pre):-4]
                if mid.isdigit():
                    found[int(mid)] = os.path.join(OUT, fn)
        if not found:
            raise SystemExit(f"no shard files matching {OUT}/{tag}.<n>.csv")
        n_shards = max(found) + 1
        missing = [i for i in range(n_shards) if i not in found]
        print(f"shards: {len(found)} found, highest index {max(found)}")
        if missing:
            print(f"WARNING: {len(missing)} of {n_shards} shards missing: "
                  f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
            print("  their cells are NOT in this merge -- re-run them first")
        rows = []
        for i in sorted(found):
            with open(found[i]) as fh:
                rd = csv.reader(fh)
                next(rd, None)
                rows += [r for r in rd if r]
        if not rows:
            raise SystemExit("no shard output")
        # No row-count assertion here, deliberately. Two things make one wrong:
        # --merge is invoked without --cells-file, so `cl` is the full generated
        # grid rather than the pruned list the shards actually ran; and score()
        # returns None for any cell whose held rollout falls, so fewer rows than
        # cells is the normal outcome, not a loss. The shard-contiguity check
        # above is the one that catches real gaps.
        print(f"{len(rows):,} rows from {len(found)} shards "
              f"(cells whose held rollout fell write no row)")
        i_v = COLS.index("v_net_pid" if HW_TORSO == "pid" else "v_net_ff")
        rows.sort(key=lambda r: -(float(r[i_v]) if r[i_v] not in ("", "nan") else -1))
        with open(os.path.join(OUT, f"{tag}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(COLS)
            w.writerows(rows)
        print(f"{len(rows)} cells -> {OUT}/{tag}.csv")
        return

    todo = [c for i, c in enumerate(cl) if i % a.of == a.shard]
    p = os.path.join(OUT, f"{tag}.{a.shard}.csv")
    done = set()
    if os.path.exists(p):
        with open(p) as fh:
            rd = csv.reader(fh)
            next(rd, None)
            done = {tuple(round(float(x), 4) for x in r[:5]) for r in rd if r}
    has_header = os.path.exists(p) and os.path.getsize(p) > 0       # a requeued task may have left a header-only file
    fh = open(p, "a", newline="")
    w = csv.writer(fh)
    if not has_header:
        w.writerow(COLS)
    import time as _time
    t_start = _time.time(); n_run = 0
    for c in todo:
        if tuple(round(x, 4) for x in c) in done:
            continue
        row = score(c, a.mu)
        n_run += 1
        if row is not None:
            w.writerow(row)
            fh.flush()
    fh.close()
    el = _time.time() - t_start
    print(f"shard {a.shard}/{a.of}: {len(todo)} cells ({n_run} simulated, {el:.0f} s, {el / max(n_run, 1):.2f} s/cell) -> {p}")


if __name__ == "__main__":
    main()

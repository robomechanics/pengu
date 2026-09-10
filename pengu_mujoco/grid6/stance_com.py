"""stance_com.py — where the whole-body COM sits relative to the stance foot, and what the
ground pushes back with, for kappa = 0 against kappa = 2 on the same gait.

For one rollout per kappa it logs at 200 Hz: per-foot contact normal force, per-foot COP
(force-weighted contact positions), whole-body COM (mass-weighted xipos), torso COM, torso
and hip-axis roll about the hinge, and the summed ground reaction force per foot in the
world frame (mj_contactForce rotated out of the contact frame), plus both foot positions.
Single stance = exactly one foot above FN_MIN after a SMOOTH_MS moving average of Fn (the
raw normal force flickers at 5-10 ms; the raw classification is reported alongside).
Lateral / fore-aft are the ROBOT's axes, taken per sample from the line joining the two
feet (the robot yaws 40-70 deg over the window, so world x is not lateral). Left-stance
samples are mirrored (lateral -> -lateral) so every stance reads as a right-foot stance.

Figure (rows: kappa 0 / kappa 2):
  frontal plane   stance COP at the origin; COM path during single stance, GRF arrows at
                  a few phases, the gravity vector from the COM
  top view        the same path in (lateral, fore-aft)
  one cycle       lateral COM offset from the stance COP, GRF lateral / normal, torso and
                  axis roll, with single/double-support shading

    PENGU_MODEL=1.31 python grid6/stance_com.py --cell 1.67/340/95/24/20 --mu 0.1
    PENGU_MODEL=1.31 python grid6/stance_com.py --cell 1.67/340/95/24/20 --mu 0.12 --hw

--hw switches on the hardware layers (354 deg/s cap, 56 ms torso lag, FF torso per
kappa) via hw_sweep; without it the torso is the ideal kappa PID as in GRID-5.
Output: results/grid6_probes/stance_com_<cell>_mu<..>[_hw].png + a numbers table.
"""
import argparse
import math
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.append(ROOT)
os.environ.setdefault("PENGU_MODEL", "1.31")

import matplotlib                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402
import mujoco                                    # noqa: E402
import gait_config as gc                         # noqa: E402
import gait_sweep as gs                          # noqa: E402
from torso_control import TorsoKappaPID          # noqa: E402
from friction_utils import set_floor_friction    # noqa: E402

FS = 200.0
SETTLE, WINDOW = 2.0, 8.0
FN_MIN = 2.0                                     # N, a foot counts as loaded above this
SMOOTH_MS = 25.0                                 # moving average on Fn before classifying stance
REST_LEAN = 5.0
OUT = os.path.join(ROOT, "results", "grid6_probes")
COL = {0.0: "#1f77b4", 2.0: "#d62728"}


COM_TARGET = None                                # set to a COM ratio to run a GRID-5 slide variant of the base model


def com_ratio_of(model):
    d = mujoco.MjData(model)
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, d, act, jadr)
    mujoco.mj_forward(model, d)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easyaxis")
    return float(d.subtree_com[1][2]) / float(d.xpos[aid][2])


def apply_com_variant(model, target):
    """grid5_sweep.apply_com_variant (copied: grid5_sweep asserts on grid5's gait_sweep at
    import): slide easytorso's inertial COM along world-up until the standing COM ratio
    hits `target`."""
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
    d = mujoco.MjData(model)
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, d, act, jadr)
    mujoco.mj_forward(model, d)
    up = d.xmat[tid].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
    ip0 = model.body_ipos[tid].copy()

    def ratio_at(x):
        model.body_ipos[tid] = ip0 + x * up
        return com_ratio_of(model)

    lo, hi = -0.30, 0.30
    assert ratio_at(lo) < target < ratio_at(hi), (target, ratio_at(lo), ratio_at(hi))
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if ratio_at(mid) < target:
            lo = mid
        else:
            hi = mid
    got = ratio_at(0.5 * (lo + hi))
    assert abs(got - target) < 1e-3, (got, target)
    return got


def rollout(cell, mu, kappa, hw):
    freq, phi, leg, hip, off = cell
    model = mujoco.MjModel.from_xml_path(gs.XML)
    if COM_TARGET is not None:
        apply_com_variant(model, COM_TARGET)
    data = mujoco.MjData(model)
    gc.RAMP_HIP_OFFSET = True
    gs.STAGED_START = True
    gc.STAND_HIP_DEG = 0.0
    pid = TorsoKappaPID(model, kappa=kappa, measure_after=0.0,
                        ctrl_limit_deg=45.0 if kappa else (25.0 if hw else 45.0))
    gc.STAND_HIP_DEG = REST_LEAN
    set_floor_friction(model, mu)
    gs.FLOOR_MU = mu
    gs.CONDITION["hip_off"] = off
    gs._set_gait(dict(freq=freq, hip_phi=phi, leg_amp=leg, hip_amp=hip))
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)
    floor_id, foot_geom, foot_bid, root = gs.make_ids(model)
    legs = [act[n] for n in ("crank1-L", "crank1-R", "hip-L", "hip-R")]
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
    mass = model.body_mass[1:]
    mtot = float(mass.sum())
    # sole vertices (lowest 3 mm of each foot mesh, body frame) -> footprint per sample later
    foot_bottom = {}
    for g, side in foot_geom.items():
        d = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[d]: model.mesh_vertadr[d] + model.mesh_vertnum[d]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        vb = v @ R.reshape(3, 3).T + model.geom_pos[g]
        foot_bottom[side] = vb[vb[:, 2] <= vb[:, 2].min() + 0.003]

    ff = None
    if hw:
        import hw_sweep as hs                    # HELD fit -> feedforward, same as the sweep
        hs.LEG_RATE, hs.LEG_TAU, hs.SERVO_LAG = 354.0, 0.0, 0.056
        hs.TORSO_CLAMP_DEG = 25.0 if kappa == 0.0 else 45.0
        held = hs.rollout(freq, phi, leg, hip, off, mu, "held")
        if held.get("fell") is not None:
            raise SystemExit(f"kappa {kappa}: torso-held rollout fell at {held['fell']:.1f}s")
        A0, p0, _ = held["_fit"]
        best = None
        for lead in hs.LEADS:
            r = hs.rollout(freq, phi, leg, hip, off, mu, "ff", A=A0, ph=p0 + lead, kappa=kappa)
            if r.get("fell") is None and (best is None or r["rollrms"] < best[1]["rollrms"]):
                best = (lead, r)
        if best is None:
            raise SystemExit(f"kappa {kappa}: every feedforward lead fell")
        ff = (A0, p0 + best[0])
        print(f"  kappa {kappa:g}: FF A0 {A0:.2f} deg, phase {ff[1]:.0f} deg (lead {best[0]:.0f}),"
              f" sweep v_net {best[1]['v_net']:.3f}")
    buf = []
    slew = math.radians(354.0) * model.opt.timestep
    held_cmd = None

    def ctrl(d, t, alpha=1.0):
        if ff is None:
            return pid(d, t, alpha)
        g = pid.s * (kappa - 1.0)
        w = 2 * math.pi * freq * (t - gc.T_HOLD - gc.T_TRANSITION)
        u = alpha * math.radians(abs(g) * ff[0]) * math.sin(
            w + math.radians(ff[1] + (180.0 if g < 0 else 0.0)))
        u = max(-pid.limit, min(pid.limit, u))
        buf.append((t, u))
        while len(buf) > 1 and buf[1][0] <= t - 0.056:
            buf.pop(0)
        return buf[0][1]
    gc.TORSO_CONTROLLER = ctrl

    gc.T_HOLD = 1e9
    t0, nxt = None, 0.0
    log = dict(t=[], com=[], tcom=[], fn=[], cop=[], grf=[], troll=[], aroll=[], fpos=[], fmat=[], tpitch=[])
    fell = None
    while True:
        if t0 is None:
            tt = data.time
            if (tt >= gs.QUIET_MIN_T and float(np.max(np.abs(data.qvel))) < gs.QUIET_QVEL) \
                    or tt >= gs.QUIET_MAX_T:
                t0 = tt
                gc.T_HOLD = tt
        gc.apply_ctrl(data, act, data.time)
        if hw:
            cur = np.array([data.ctrl[i] for i in legs])
            if held_cmd is None:
                held_cmd = cur.copy()
            held_cmd += np.clip(cur - held_cmd, -slew, slew)
            for i, j in enumerate(legs):
                data.ctrl[j] = held_cmd[i]
        mujoco.mj_step(model, data)
        if t0 is None:
            continue
        if data.xpos[root][2] < 0.05:
            fell = data.time - t0
            break
        tw = data.time - t0 - gc.T_TRANSITION - SETTLE
        if tw < 0:
            continue
        if tw > WINDOW:
            break
        if data.time < nxt:
            continue
        nxt = data.time + 1.0 / FS
        fn = {"L": 0.0, "R": 0.0}
        cop = {"L": np.zeros(3), "R": np.zeros(3)}
        grf = {"L": np.zeros(3), "R": np.zeros(3)}
        for ci in range(data.ncon):
            c = data.contact[ci]
            hit = [g for g in (c.geom1, c.geom2) if g in foot_geom]
            if not hit:
                continue
            s = foot_geom[hit[0]]
            fv = np.zeros(6)
            mujoco.mj_contactForce(model, data, ci, fv)
            fr = c.frame.reshape(3, 3)             # rows: normal, tangent1, tangent2
            fw = fr.T @ fv[:3]                     # contact-frame force -> world
            if c.geom1 in foot_geom:               # force acts on geom2; flip so it is ON the foot
                fw = -fw
            n = abs(float(fv[0]))
            fn[s] += n
            cop[s] += n * c.pos
            grf[s] += fw
        for s in ("L", "R"):
            cop[s] = cop[s] / fn[s] if fn[s] > 1e-9 else np.full(3, np.nan)
        h = pid.hinge(data)
        log["t"].append(data.time - t0)
        log["com"].append((data.xipos[1:] * mass[:, None]).sum(0) / mtot)
        log["tcom"].append(data.xipos[tid].copy())
        log["fn"].append([fn["L"], fn["R"]])
        log["cop"].append([cop["L"], cop["R"]])
        log["grf"].append([grf["L"], grf["R"]])
        log["fpos"].append([data.xpos[b].copy() for s_ in ("L", "R") for b, ss in foot_bid.items() if ss == s_])
        log["fmat"].append([data.xmat[b].reshape(3, 3).copy() for s_ in ("L", "R") for b, ss in foot_bid.items() if ss == s_])
        log["troll"].append(math.degrees(pid.torso_roll(data, h)))
        log["aroll"].append(math.degrees(pid.axis_roll(data, h)))
        Rt = data.xmat[tid].reshape(3, 3)                 # easytorso body frame: -y = up, -z = forward (measured at the stand pose)
        up, fwd = -Rt[:, 1], -Rt[:, 2].copy()
        fwd[2] = 0.0; fwd /= max(np.linalg.norm(fwd), 1e-9)
        log["tpitch"].append(math.degrees(math.asin(float(np.clip(np.dot(up, fwd), -1, 1)))))   # + = torso top tilted forward; = hip_off at the stand pose
    gc.T_HOLD = 5.0
    out = {k: np.array(v) for k, v in log.items()}
    out["fell"] = fell
    out["foot_bottom"] = [foot_bottom["L"], foot_bottom["R"]]
    out["mass"] = mtot
    return out


def classify(fn, smooth_ms=SMOOTH_MS):
    """'L' / 'R' / 'D' (double) / 'A' (air) per sample from the two normal forces"""
    k = max(1, int(round(smooth_ms / 1000.0 * FS)))
    if k > 1:
        fn = np.stack([np.convolve(fn[:, j], np.ones(k) / k, mode="same") for j in (0, 1)], 1)
    return np.where((fn[:, 0] > FN_MIN) & (fn[:, 1] > FN_MIN), "D",
                    np.where(fn[:, 0] > FN_MIN, "L", np.where(fn[:, 1] > FN_MIN, "R", "A")))


def run_lengths(mask):
    """lengths (samples) of the True runs of a boolean mask"""
    idx = np.where(mask)[0]
    if not len(idx):
        return np.array([], int)
    return np.array([len(x) for x in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)])


def stance_frames(r, freq, smooth_ms=SMOOTH_MS):
    """per sample: 'L' / 'R' / 'D' (double) / 'A' (air), plus COM relative to the stance
    COP in the robot frame (lateral, fore-aft, up) with left stances mirrored to the right,
    and the GRF of the stance foot in the same frame."""
    st = classify(r["fn"], smooth_ms)
    d_feet = r["fpos"][:, 1] - r["fpos"][:, 0]        # left foot -> right foot
    lat = d_feet.copy()
    lat[:, 2] = 0.0
    lat /= np.linalg.norm(lat, axis=1, keepdims=True)  # robot's right, horizontal
    fore = np.stack([-lat[:, 1], lat[:, 0], np.zeros(len(lat))], 1)   # z x lat = heading
    rel = np.full((len(st), 3), np.nan)
    trel = np.full((len(st), 3), np.nan)
    grf = np.full((len(st), 3), np.nan)
    for i, s in enumerate(st):
        if s in ("L", "R"):
            k = 0 if s == "L" else 1
            if not np.isfinite(r["cop"][i][k]).all():   # smoothed stance, raw contact gone this sample
                continue
            basis = np.stack([lat[i], fore[i], [0.0, 0.0, 1.0]])
            d = basis @ (r["com"][i] - r["cop"][i][k])
            td = basis @ (r["tcom"][i] - r["cop"][i][k])
            f = basis @ r["grf"][i][k]
            if s == "L":                           # mirror lateral so every stance is a right stance
                d[0], td[0], f[0] = -d[0], -td[0], -f[0]
            rel[i], trel[i], grf[i] = d, td, f
    return st, rel, grf, trel


def balance_metrics(r, smooth_ms=SMOOTH_MS):
    """whole-window balance quantities in the robot frame (lateral = robot right, fore, up):
    combined COP of both feet, total GRF, horizontal COP->COM offset, the GRF line-of-action
    miss distance to the COM and the moment it makes about the COM (= dH/dt), the angle
    between the GRF and the COP->COM line, and the angle between the COM velocity and the
    GRF. Airborne samples are NaN."""
    fn = r["fn"]
    F = r["grf"].sum(1)                                        # total GRF, world
    tot = fn.sum(1)
    cop = np.full((len(fn), 3), np.nan)
    ok = tot > FN_MIN
    for i in np.where(ok)[0]:
        c = [r["cop"][i][k] for k in (0, 1)]
        cop[i] = sum(fn[i, k] * c[k] for k in (0, 1) if np.isfinite(c[k]).all()) / tot[i]
    d_feet = r["fpos"][:, 1] - r["fpos"][:, 0]
    lat = d_feet.copy(); lat[:, 2] = 0.0
    lat /= np.linalg.norm(lat, axis=1, keepdims=True)
    fore = np.stack([-lat[:, 1], lat[:, 0], np.zeros(len(lat))], 1)
    up = np.tile([0.0, 0.0, 1.0], (len(lat), 1))
    B = np.stack([lat, fore, up], 1)                           # rows = robot axes
    k = max(1, int(round(smooth_ms / 1000.0 * FS)))
    com_s = np.stack([np.convolve(r["com"][:, j], np.ones(k) / k, mode="same") for j in range(3)], 1)
    v = np.gradient(com_s, 1.0 / FS, axis=0)                    # COM velocity, world
    rel = np.einsum("nij,nj->ni", B, r["com"] - cop)           # COP -> COM, robot frame
    Fb = np.einsum("nij,nj->ni", B, F)
    vb = np.einsum("nij,nj->ni", B, v)
    M = np.cross(cop - r["com"], F)                            # moment of the GRF about the COM = dH/dt (world)
    Mb = np.einsum("nij,nj->ni", B, M)                         # about lateral (pitch), fore (roll), up (yaw)
    Fn_ = np.linalg.norm(F, axis=1)
    miss = np.linalg.norm(M, axis=1) / np.maximum(Fn_, 1e-6)   # perpendicular distance COM <-> GRF line
    def ang(a, b):
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        return np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", a, b) / np.maximum(na * nb, 1e-9), -1, 1)))
    align = ang(Fb, rel)                                       # GRF vs COP->COM line
    align_front = ang(Fb[:, [0, 2]], rel[:, [0, 2]])           # same, frontal plane only
    vF = ang(vb, Fb)                                           # COM velocity vs GRF, 3-D
    vF_front = ang(vb[:, [0, 2]], Fb[:, [0, 2]])
    vF_horiz = ang(vb[:, :2], Fb[:, :2])                       # horizontal plane: is the GRF pushing along the COM's travel
    power = np.einsum("ni,ni->n", F, v)                        # GRF power on the COM
    # centroidal moment pivot (Herr & Popovic 2008): where a line through the COM parallel to the
    # GRF meets the ground; CMP == COP <=> zero moment about the COM. Distance measured from the
    # COP in the robot frame, plus from the nearest foot body centre (support-base check).
    h = r["com"][:, 2] - cop[:, 2]
    cmp_ = r["com"] - F * (h / np.maximum(F[:, 2], 1e-6))[:, None]
    cmp_rel = np.einsum("nij,nj->ni", B, cmp_ - cop)
    # support-base check: signed distance of the CMP outside the loaded feet's soles (robot-frame
    # bounding box of the sole vertices, per sample); <= 0 means inside some loaded foot
    cmp_out = np.full(len(fn), np.nan)
    if "fmat" in r:
        for i in np.where(ok)[0]:
            best = np.inf
            for k in (0, 1):
                if fn[i, k] <= FN_MIN:
                    continue
                vw = r["fpos"][i, k] + r["foot_bottom"][k] @ r["fmat"][i, k].T
                pv = (vw - cmp_[i]) @ B[i, :2].T                       # sole vertices relative to the CMP, (lat, fore)
                lo, hi = pv.min(0), pv.max(0)
                dx = max(lo[0], -hi[0], 0.0) if not (lo[0] <= 0 <= hi[0]) else -min(-lo[0], hi[0])
                dy = max(lo[1], -hi[1], 0.0) if not (lo[1] <= 0 <= hi[1]) else -min(-lo[1], hi[1])
                d = math.hypot(max(dx, 0), max(dy, 0)) if (dx > 0 or dy > 0) else max(dx, dy)
                best = min(best, d)
            cmp_out[i] = best
    out = dict(cop=cop, rel=rel, F=Fb, v=vb, M=Mb, miss=miss, align=align, align_front=align_front,
               vF=vF, vF_front=vF_front, vF_horiz=vF_horiz, power=power, ok=ok, cmp=cmp_rel, cmp_out=cmp_out)
    for kk in ("rel", "F", "M", "miss", "align", "align_front", "vF", "vF_front", "vF_horiz", "power", "cmp", "cmp_out"):
        out[kk] = np.where(ok if out[kk].ndim == 1 else ok[:, None], out[kk], np.nan)
    return out


def report_balance(bm, st, g):
    ss = np.isin(st, ["L", "R"])
    def stat(x, m=None):
        x = x if m is None else np.where(m, x, np.nan)
        return f"{np.nanmean(x):+.2f} (p10/50/90 {np.nanpercentile(x, 10):+.2f}/{np.nanpercentile(x, 50):+.2f}/{np.nanpercentile(x, 90):+.2f})"
    dh = np.hypot(bm["rel"][:, 0], bm["rel"][:, 1]) * 100
    print(f"     -- balance (whole window, both feet combined; loaded {np.mean(bm['ok'])*100:.0f}% of samples) --")
    print(f"     COP->COM horizontal distance [cm]: all {stat(dh)}   single stance {stat(dh, ss)}")
    print(f"       lateral [cm]  all {stat(bm['rel'][:, 0]*100)}   fore-aft [cm] all {stat(bm['rel'][:, 1]*100)}")
    print(f"     GRF line miss distance to COM [cm]: all {stat(bm['miss']*100)}   single stance {stat(bm['miss']*100, ss)}")
    ch = np.hypot(bm["cmp"][:, 0], bm["cmp"][:, 1]) * 100
    print(f"     CMP - COP horizontal distance [cm]: all {stat(ch)}   single stance {stat(ch, ss)}   "
          f"(lateral all {stat(bm['cmp'][:, 0]*100)})")
    if np.isfinite(bm["cmp_out"]).any():
        co = bm["cmp_out"]
        okm = np.isfinite(co)
        print(f"     CMP inside a loaded foot's sole: {np.mean(co[okm] <= 0)*100:.0f}% of loaded samples "
              f"(single stance {np.mean(co[okm & ss] <= 0)*100:.0f}%);  when outside, distance beyond the sole edge "
              f"{stat(np.where(co > 0, co, np.nan)*100)} cm")
    else:
        print("     (CMP-in-footprint needs foot orientations: rerun without --cache)")
    print(f"     GRF moment about COM = dH/dt [N.m]: roll (about fore axis) {stat(bm['M'][:, 1])}   pitch (about lateral) {stat(bm['M'][:, 0])}   |M| {stat(np.linalg.norm(bm['M'], axis=1))}")
    print(f"     angle GRF vs COP->COM line [deg]: 3-D {stat(bm['align'])}   frontal plane {stat(bm['align_front'])}")
    print(f"     angle COM velocity vs GRF [deg]: 3-D {stat(bm['vF'])}   frontal {stat(bm['vF_front'])}   horizontal {stat(bm['vF_horiz'])}")
    print(f"     GRF power on COM [W]: {stat(bm['power'])}   |v_COM| mean {np.nanmean(np.linalg.norm(bm['v'], axis=1)):.3f} m/s   |F|/mg mean {np.nanmean(np.linalg.norm(bm['F'], axis=1))/g:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="1.67/340/95/24/20")
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--hw", action="store_true")
    ap.add_argument("--cache", action="store_true", help="pickle the rollouts next to the figure and reuse them")
    a = ap.parse_args()
    cell = tuple(float(x) for x in a.cell.split("/"))
    freq = cell[0]
    os.makedirs(OUT, exist_ok=True)
    print(f"model {gc.XML_PATH}  cell {a.cell}  mu {a.mu}  {'hardware layers + FF' if a.hw else 'ideal, kappa PID'}")

    tag = f"stance_com_{'-'.join(f'{x:g}' for x in cell)}_mu{int(round(a.mu*100)):03d}{'_hw' if a.hw else ''}"
    pk = os.path.join(OUT, tag + ".pkl")
    raw = pickle.load(open(pk, "rb")) if a.cache and os.path.exists(pk) else {}
    runs = {}
    for kappa in (0.0, 2.0):
        r = raw.get(kappa) or rollout(cell, a.mu, kappa, a.hw)
        raw[kappa] = r
        st, rel, grf, trel = stance_frames(r, freq)
        ss = np.isin(st, ["L", "R"])
        g = r["mass"] * 9.81
        st_raw = classify(r["fn"], 0.0)
        ss_raw = np.isin(st_raw, ["L", "R"])
        rl = run_lengths(ss) / FS * 1000
        print(f"\n  kappa {kappa:g}: {'FELL at %.1fs' % r['fell'] if r['fell'] else 'walked'}  "
              f"[Fn smoothed {SMOOTH_MS:.0f} ms] single-support {ss.mean()*100:.0f}%  double {np.mean(st=='D')*100:.0f}%  air {np.mean(st=='A')*100:.0f}%"
              f"   single-stance runs {len(rl)}, median {np.median(rl) if len(rl) else 0:.0f} ms, max {rl.max() if len(rl) else 0:.0f} ms"
              f"   (raw Fn: single {ss_raw.mean()*100:.0f}%, {len(run_lengths(ss_raw))} runs, median {np.median(run_lengths(ss_raw))/FS*1000 if ss_raw.any() else 0:.0f} ms)")
        sep = np.hypot(*(r["fpos"][:, 1, :2] - r["fpos"][:, 0, :2]).T)
        print(f"     foot separation {sep.mean()*100:.1f} cm;  COM lateral offset percentiles p10/50/90 "
              f"{np.nanpercentile(rel[ss,0]*100, [10, 50, 90]).round(1) if ss.any() else '-'} cm")
        if ss.any():
            print(f"     COM vs stance COP during single stance [m]: lateral {np.nanmean(rel[ss,0]):+.4f} "
                  f"(+ = lateral, outside the stance COP; - = medial, toward the swing foot)  fore-aft {np.nanmean(rel[ss,1]):+.4f}  height {np.nanmean(rel[ss,2]):.4f}")
            print(f"     stance GRF: lateral/normal {np.nanmean(grf[ss,0]/np.maximum(grf[ss,2],1e-6)):+.3f}  "
                  f"normal mean {np.nanmean(grf[ss,2]):.1f} N ({np.nanmean(grf[ss,2])/g*100:.0f}% of weight)")
            print(f"     torso COM vs stance COP: lateral {np.nanmean(trel[ss,0]):+.4f}  height {np.nanmean(trel[ss,2]):.4f}")
            print(f"     gravity moment about stance foot (m g x lateral offset) {np.nanmean(g*rel[ss,0]):+.3f} N.m;  "
                  f"torso roll rms {np.std(r['troll']):.2f} deg  axis roll rms {np.std(r['aroll']):.2f} deg")
        bm = balance_metrics(r)
        runs[kappa] = (r, st, rel, grf, trel, bm)
        report_balance(bm, st, g)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for row, kappa in enumerate((0.0, 2.0)):
        r, st, rel, grf, trel, bm = runs[kappa]
        col = COL[kappa]
        ss = np.isin(st, ["L", "R"])
        g = r["mass"] * 9.81
        # frontal plane
        ax = axes[row, 0]
        ax.plot(rel[ss, 0], rel[ss, 2], ".", ms=2, color=col, alpha=0.5, label="whole-body COM")
        ax.plot(trel[ss, 0], trel[ss, 2], ".", ms=2, color="0.5", alpha=0.35, label="torso COM")
        ax.plot(0, 0, "k^", ms=10, label="stance COP")
        ax.legend(fontsize=7, loc="upper right")
        idx = np.where(ss)[0]
        for i in idx[:: max(1, len(idx) // 12)]:
            f = grf[i] / g * 0.15                  # scaled: body weight = 0.15 m
            ax.arrow(0, 0, f[0], f[2], color="0.3", width=0.0015, alpha=0.6, length_includes_head=True)
            ax.plot([rel[i, 0], rel[i, 0]], [rel[i, 2], rel[i, 2] - 0.06], color=col, lw=0.8, alpha=0.4)
        ax.set_xlim(-0.12, 0.12)
        ax.set_ylim(-0.02, 0.42)
        ax.set_aspect("equal")
        ax.axvline(0, color="0.8", lw=0.8)
        ax.set_xlabel("lateral from stance COP [m]\n(+ outside the COP, − toward the swing foot)")
        ax.set_ylabel("height above COP [m]")
        ax.set_title(f"κ={kappa:g}: COM in the frontal plane, single stance\n(grey arrows = GRF, weight = 0.15 m)")
        # top view
        ax = axes[row, 1]
        ax.plot(rel[ss, 0], rel[ss, 1], ".", ms=2, color=col, alpha=0.5)
        ax.plot(0, 0, "k^", ms=10)
        ax.set_xlim(-0.12, 0.12)
        ax.set_ylim(-0.12, 0.12)
        ax.set_aspect("equal")
        ax.axhline(0, color="0.8", lw=0.8)
        ax.axvline(0, color="0.8", lw=0.8)
        ax.set_xlabel("lateral [m]")
        ax.set_ylabel("fore-aft [m]  (+ = ahead of the stance foot)")
        ax.set_title(f"κ={kappa:g}: COM top view, single stance")
        # one cycle time series
        ax = axes[row, 2]
        t = r["t"]
        T = 1.0 / freq
        t_a = t[-1] - 2.0 * T
        m = t >= t_a
        tt = (t[m] - t_a) / T
        yl = 30 if kappa == 0.0 else 50
        for s, c in (("L", "#c7e9c0"), ("R", "#fdd0a2"), ("D", "#dddddd")):
            mm = st[m] == s
            if mm.any():
                ax.fill_between(tt, -yl, yl, where=mm, color=c, alpha=0.6, lw=0, label=f"{s} stance" if s != "D" else "double")
        lat = np.where(np.isin(st[m], ["L", "R"]), rel[m, 0], np.nan) * 100
        ax.plot(tt, lat, color=col, lw=2, label="COM lateral from stance COP [cm]")
        ax.plot(tt, r["troll"][m], "k-", lw=1.2, label="torso roll [deg]")
        ax.plot(tt, r["aroll"][m], "k--", lw=1.0, label="axis roll [deg]")
        gl = np.where(np.isin(st[m], ["L", "R"]), grf[m, 0] / np.maximum(grf[m, 2], 1e-6), np.nan) * 100
        ax.plot(tt, gl, color="0.4", lw=1.0, ls=":", label="GRF lateral/normal [%]")
        ax.set_ylim(-yl, yl)
        ax.set_xlabel("gait cycles")
        ax.set_title(f"κ={kappa:g}: last two cycles")
        if row == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)
    if a.cache:
        pickle.dump(raw, open(pk, "wb"))
    fig.suptitle(f"{os.path.basename(os.path.dirname(gc.XML_PATH))}  gait {a.cell}  μ={a.mu:g}  "
                 f"{'hardware layers + feedforward torso' if a.hw else 'ideal actuators, κ PID torso'}", fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, tag + ".png")
    fig.savefig(p, dpi=130)
    print(f"\n-> {p}")

    # balance figure: COP->COM offset, GRF line miss distance, GRF direction vs COP->COM and vs COM velocity
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    for row, kappa in enumerate((0.0, 2.0)):
        r, st, rel, grf, trel, bm = runs[kappa]
        col = COL[kappa]
        t = r["t"]; T = 1.0 / freq; t_a = t[-1] - 2.0 * T; m = t >= t_a; tt = (t[m] - t_a) / T
        ax = axes[row, 0]
        for s, c in (("L", "#c7e9c0"), ("R", "#fdd0a2"), ("D", "#dddddd")):
            mm = st[m] == s
            if mm.any():
                ax.fill_between(tt, -12, 12, where=mm, color=c, alpha=0.6, lw=0, label={"L": "L stance", "R": "R stance", "D": "double"}[s])
        ax.plot(tt, bm["rel"][m, 0] * 100, color=col, lw=2, label="COP→COM lateral [cm]")
        ax.plot(tt, bm["rel"][m, 1] * 100, color=col, lw=1.2, ls="--", label="COP→COM fore-aft [cm]")
        ax.plot(tt, bm["miss"][m] * 100, "k-", lw=1.2, label="GRF line miss distance to COM [cm]")
        ax.plot(tt, bm["M"][m, 1], color="0.4", lw=1.0, ls=":", label="GRF roll moment about COM [N·m]")
        ax.set_ylim(-12, 12); ax.set_xlabel("gait cycles"); ax.set_title(f"κ={kappa:g}: COP→COM offset and GRF line, last two cycles")
        if row == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)
        # frontal plane: GRF direction vs COP->COM direction, both as angle from vertical (+ = toward robot right)
        ax = axes[row, 1]
        aF = np.degrees(np.arctan2(bm["F"][:, 0], bm["F"][:, 2]))
        aR = np.degrees(np.arctan2(bm["rel"][:, 0], bm["rel"][:, 2]))
        ss = np.isin(st, ["L", "R"])
        ax.plot(aR[~ss], aF[~ss], ".", ms=2, color="0.6", alpha=0.4, label="double support")
        ax.plot(aR[ss], aF[ss], ".", ms=3, color=col, alpha=0.6, label="single stance")
        ax.plot([-40, 40], [-40, 40], "k-", lw=0.8, label="GRF through COM")
        cone = math.degrees(math.atan(a.mu))
        ax.axhline(cone, color="k", lw=0.8, ls="--"); ax.axhline(-cone, color="k", lw=0.8, ls="--")
        ax.set_xlim(-40, 40); ax.set_ylim(-40, 40); ax.set_aspect("equal")
        ax.set_xlabel("COP→COM tilt from vertical, frontal [deg]\n(+ = COM to the robot's right of the COP; dashed = friction cone atan μ)")
        ax.set_ylabel("GRF tilt from vertical, frontal [deg]")
        ax.set_title(f"κ={kappa:g}: GRF direction vs COP→COM direction")
        if row == 0:
            ax.legend(fontsize=7, loc="upper left")
        # COM velocity vs GRF angle histogram
        ax = axes[row, 2]
        for key, lab, c in (("vF", "3-D", col), ("vF_front", "frontal plane", "0.3"), ("vF_horiz", "horizontal plane", "0.7")):
            x = bm[key][np.isfinite(bm[key])]
            ax.hist(x, bins=np.arange(0, 181, 5), histtype="step", lw=1.5, color=c, density=True,
                    label=f"{lab}: median {np.median(x):.0f}°")
        ax.set_xlabel("angle between COM velocity and GRF [deg]\n(0 = GRF along the COM motion, 180 = opposing)")
        ax.set_ylabel("density"); ax.set_xlim(0, 180)
        ax.set_title(f"κ={kappa:g}: COM velocity vs GRF")
        ax.legend(fontsize=7)
    fig.suptitle(f"{os.path.basename(os.path.dirname(gc.XML_PATH))}  gait {a.cell}  μ={a.mu:g}  "
                 f"{'hardware layers + feedforward torso' if a.hw else 'ideal actuators, κ PID torso'}  — balance", fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, tag.replace("stance_com", "balance") + ".png")
    fig.savefig(p, dpi=130)
    print(f"-> {p}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Shared machinery for the mechanism-illustration figures (com_tick,
duty_strip, torso_lat, attitude_phase).

UNLIKE the map-reader figures, these run a SHORT SIMULATION of one gait
(GRID-4 pipeline, physics/, nominal conditions, no jitter) to record a few
cycles and render top-down/front views. They are illustration tools, not
sweep-session work: one ~10 s rollout each, run nice'd.

Default gait: the c6 mu=0.1 straight-walking VERIFIED cell
f1.67/phi340/leg95/hip24/off20 (topupK5: pass 5/5, mean 0.376, min 0.368,
head 0.993) — chosen over the round-1 champion, which crab-walks ~40-70 deg
off its travel direction and buries the lateral-sway story.
"""
import os, sys, math
os.environ.setdefault("PENGU_MODEL", "1.31")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))            # pengu_mujoco/
sys.path.insert(0, os.path.join(_ROOT, "physics")); sys.path.insert(0, _ROOT)

import numpy as np
import mujoco
import gait_config as gc
import gait_sweep as gs
from torso_control import TorsoKappaPID
from friction_utils import set_floor_friction
import grid4_sweep as g4

OUT_DIR = os.path.join(_ROOT, "results", "grid5_report", "style_ref")
CONF = {"c1": (0.0, 1.05), "c2": (0.0, 1.20), "c3": (0.0, 1.31),
        "c4": (2.0, 1.05), "c5": (2.0, 1.20), "c6": (2.0, 1.31)}
DEF = dict(cfg="c6", mu=0.1, freq=1.67, phi=340.0, leg=95.0, hip=24.0,
           off=20.0)


def add_gait_args(ap):
    for k, v in DEF.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    return ap


def build(a):
    """Fresh deterministic sim of one gait under nominal conditions."""
    kappa, com = CONF[a.cfg]
    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    ids = gs.make_ids(model)
    g4.apply_com_variant(model, com)
    pid = TorsoKappaPID(model, kappa=kappa, measure_after=gs.SETTLE)
    gc.TORSO_CONTROLLER = pid
    gs.FLOOR_MU = a.mu; gs.POSE_JITTER = None
    gs.CONDITION["hip_off"] = a.off
    set_floor_friction(model, a.mu)
    gs._set_gait(dict(freq=a.freq, hip_phi=a.phi, leg_amp=a.leg,
                      hip_amp=a.hip))
    pid.reset()
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)
    return model, data, ids, act, kappa, com


def foot_forces(model, data, ids, f6):
    floor_id, foot_geom, _, _ = ids
    Fn = {"L": 0.0, "R": 0.0}
    for c in range(data.ncon):
        ct = data.contact[c]
        fg = ct.geom2 if ct.geom1 == floor_id else (
             ct.geom1 if ct.geom2 == floor_id else -1)
        ft = foot_geom.get(fg)
        if ft:
            mujoco.mj_contactForce(model, data, c, f6)
            Fn[ft] += abs(f6[0])
    return Fn


def heading(model, data, rid):
    R = data.xmat[rid].reshape(3, 3)
    fh = R @ np.array([0.0, 1.0, 0.0])
    fh = np.array([fh[0], fh[1], 0.0])
    n = np.linalg.norm(fh)
    return fh / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])


def smooth_dom(t, fnL, fnR, win_s=0.04):
    dt = float(np.median(np.diff(t)))
    w = max(1, int(win_s / dt))
    ker = np.ones(2 * w + 1) / (2 * w + 1)
    return np.convolve(fnL - fnR, ker, mode="same")


def stance_centers(t, dom, cyc, side, thresh=2.0, after=None):
    """Midpoints of smoothed force-dominance segments (> 0.25 cyc long)."""
    after = gs.SETTLE + 2 * cyc if after is None else after
    mask = dom > thresh if side == "L" else dom < -thresh
    out, on, i0 = [], False, 0
    for i in range(len(t)):
        if not on and mask[i]:
            on, i0 = True, i
        elif on and not mask[i]:
            on = False
            if t[i] - t[i0] > 0.25 * cyc and t[i0] > after:
                out.append(0.5 * (t[i0] + t[i]))
    return out


def cam_basis(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    fwd = np.array([math.cos(el) * math.cos(az),
                    math.cos(el) * math.sin(az), math.sin(el)])
    up = np.array([-math.sin(el) * math.cos(az),
                   -math.sin(el) * math.sin(az), math.cos(el)])
    return fwd, up, np.cross(fwd, up)


def project(p, look, dist, az_deg, el_deg, W, H, fovy_deg=45.0):
    """Pinhole projection of a world point into a free-camera render."""
    fwd, up, right = cam_basis(az_deg, el_deg)
    C = np.asarray(look) - dist * fwd
    d = np.asarray(p, float) - C
    depth = float(d @ fwd)
    fpx = (H / 2) / math.tan(math.radians(fovy_deg) / 2)
    return (W / 2 + fpx * float(d @ right) / depth,
            H / 2 - fpx * float(d @ up) / depth)

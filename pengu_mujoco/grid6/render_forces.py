"""render_forces.py — the walking robot with its forces drawn on: kappa = 0 beside kappa = 2.

Every frame overlays, on the MuJoCo render itself:
  * a GRF arrow at each loaded foot, from that foot's COP, world-frame contact force
    summed over the foot's contacts (scale: body weight = ARROW_M metres)
  * the whole-body COM as a red ball, the gravity vector as an arrow of the same scale
    hanging from it, and a thin drop line from the COM to the floor -- so the eye can see
    whether the COM's ground projection is inside or outside the stance foot
  * the torso COM as a grey ball

Two robots (kappa 0 left, kappa 2 right) are stepped in lockstep on the same gait, the
same model and the same friction; the cameras follow each robot from behind (azimuth = heading, verified with a marker test: robot right = image right) and from behind-left.
Writes an mp4 and a few PNG snapshots taken at single-stance instants.

--balance draws the balance picture instead (the GRF-line-through-COM construction, as in the
exoskeleton-review Figure 3 / Herr & Popovic's centroidal moment pivot), per robot:
  * black ball  = combined COP of both feet, yellow arrow = total GRF from it (weight = ARROW_M)
  * thin yellow line = the GRF line of action, extended up past the COM
  * red ball = COM, red segment from the COM perpendicular to the GRF line = its miss distance
    (moment arm; x |GRF| = the moment about the COM = dH/dt)
  * magenta ball on the floor = CMP (line through the COM parallel to the GRF meets the floor),
    magenta ground segment CMP<->COP = the distance Herr & Popovic keep inside the support base
  * cyan arrow from the COM = COM velocity (VEL_M metres per m/s)
  * per-foot GRF arrows in thin green, torso COM grey, and a text band with the live numbers

    PENGU_MODEL=1.31 python grid6/render_forces.py --cell 1.67/340/95/24/20 --mu 0.1
    PENGU_MODEL=1.31 python grid6/render_forces.py --cell 1.67/340/95/24/20 --mu 0.12 --hw

--hw: 354 deg/s cap, 56 ms torso lag, feedforward torso fitted per kappa (as hw_sweep).
"""
import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.append(ROOT)
os.environ.setdefault("PENGU_MODEL", "1.31")

import imageio.v2 as imageio                     # noqa: E402
import mujoco                                    # noqa: E402
import gait_config as gc                         # noqa: E402
import gait_sweep as gs                          # noqa: E402
from torso_control import TorsoKappaPID          # noqa: E402
from friction_utils import set_floor_friction    # noqa: E402

FPS = 30
WINDOW = 6.0
SETTLE = 2.0
FN_MIN = 2.0
ARROW_M = 0.15                                   # one body weight of force = this many metres
REST_LEAN = 5.0
OUT = os.path.join(ROOT, "results", "grid6_probes")
CAMS = ((0.85, -10, 0), (0.95, -28, 50))         # (distance, elevation, azimuth OFFSET from the robot heading): rear, rear-left (image right = robot right)
BODY_ALPHA = float(os.environ.get("BODY_ALPHA", "0.35"))                                # robot meshes drawn translucent so the markers show
RGBA = dict(grf=(0.15, 0.65, 0.15, 0.9), com=(0.85, 0.1, 0.1, 1.0), grav=(0.85, 0.1, 0.1, 0.8),
            drop=(0.85, 0.1, 0.1, 0.5), tcom=(0.45, 0.45, 0.45, 1.0), cop=(0.1, 0.1, 0.1, 1.0),
            ftot=(0.95, 0.8, 0.1, 0.95), fline=(0.95, 0.8, 0.1, 0.45), miss=(0.85, 0.1, 0.1, 0.95),
            cmp=(0.8, 0.1, 0.8, 1.0), cmpseg=(0.8, 0.1, 0.8, 0.8), vel=(0.1, 0.75, 0.85, 0.95))
VEL_M = 0.25                                     # COM velocity arrow: metres per m/s
BAND = 34                                        # text band height (px) in --balance mode
FONT = "/System/Library/Fonts/Helvetica.ttc"


def add_sphere(scn, pos, r, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0], float),
                        np.asarray(pos, float), np.eye(3).ravel(), np.array(rgba, np.float32))
    scn.ngeom += 1


def add_arrow(scn, p0, p1, width, rgba, kind=None):
    if scn.ngeom >= scn.maxgeom:
        return
    kind = kind or mujoco.mjtGeom.mjGEOM_ARROW
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.array(rgba, np.float32))
    mujoco.mjv_connector(g, kind, width, np.asarray(p0, float), np.asarray(p1, float))
    scn.ngeom += 1


def text_band(img, text):
    """prepend a dark band with `text` to an RGB frame"""
    from PIL import Image, ImageDraw, ImageFont
    band = Image.new("RGB", (img.shape[1], BAND), (18, 18, 22))
    try:
        font = ImageFont.truetype(FONT, 17)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(band).text((10, 8), text, fill=(235, 235, 235), font=font)
    return np.vstack([np.asarray(band), img])


class Arm:
    def __init__(self, kappa, cell, mu, hw, cap=False):
        # hw: 354 deg/s cap + 56 ms torso lag + feedforward torso (the hardware-layer table)
        # cap: 354 deg/s cap only, kappa PID torso, no lag (the c3/c4 cap-only sweep)
        self.kappa, self.cell, self.mu, self.hw, self.cap = kappa, cell, mu, hw, cap
        freq, phi, leg, hip, off = cell
        self.model = mujoco.MjModel.from_xml_path(gs.XML)
        self.data = mujoco.MjData(self.model)
        gc.STAND_HIP_DEG = 0.0
        self.pid = TorsoKappaPID(self.model, kappa=kappa, measure_after=0.0,
                                 ctrl_limit_deg=45.0 if kappa else (25.0 if (hw or cap) else 45.0))
        set_floor_friction(self.model, mu)
        self.floor_id, self.foot_geom, self.foot_bid, self.root = gs.make_ids(self.model)
        self.act, self.jadr = gc.build_ids(self.model)
        self.legs = [self.act[n] for n in ("crank1-L", "crank1-R", "hip-L", "hip-R")]
        self.tid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
        self.mass = self.model.body_mass[1:]
        self.weight = float(self.mass.sum()) * 9.81
        self.slew = math.radians(354.0) * self.model.opt.timestep
        self.held_cmd = None
        self.buf = []
        self.ff = None
        self.fell = None
        self.model.geom_rgba[:, 3] = np.where(self.model.geom_rgba[:, 3] > 0, BODY_ALPHA, 0.0)
        self.model.geom_rgba[self.floor_id, 3] = 1.0
        self.cams = []
        for dist, el, az in CAMS:
            c = mujoco.MjvCamera()
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            c.distance, c.elevation, c.azimuth = dist, el, az
            self.cams.append(c)
        if hw:
            import hw_sweep as hs
            hs.LEG_RATE, hs.LEG_TAU, hs.SERVO_LAG = 354.0, 0.0, 0.056
            hs.TORSO_CLAMP_DEG = 25.0 if kappa == 0.0 else 45.0
            held = hs.rollout(freq, phi, leg, hip, off, mu, "held")
            if held.get("fell") is not None:
                raise SystemExit(f"kappa {kappa}: held rollout fell")
            A0, p0, _ = held["_fit"]
            best = None
            for lead in hs.LEADS:
                r = hs.rollout(freq, phi, leg, hip, off, mu, "ff", A=A0, ph=p0 + lead, kappa=kappa)
                if r.get("fell") is None and (best is None or r["rollrms"] < best[1]["rollrms"]):
                    best = (lead, r)
            if best is None:
                raise SystemExit(f"kappa {kappa}: every feedforward lead fell")
            self.ff = (A0, p0 + best[0])
            print(f"  kappa {kappa:g}: FF A0 {A0:.2f} phase {self.ff[1]:.0f} (lead {best[0]:.0f}) v_net {best[1]['v_net']:.3f}")

    def ctrl(self, d, t, alpha=1.0):
        if self.ff is None:
            return self.pid(d, t, alpha)
        g = self.pid.s * (self.kappa - 1.0)
        w = 2 * math.pi * self.cell[0] * (t - gc.T_HOLD - gc.T_TRANSITION)
        u = alpha * math.radians(abs(g) * self.ff[0]) * math.sin(
            w + math.radians(self.ff[1] + (180.0 if g < 0 else 0.0)))
        u = max(-self.pid.limit, min(self.pid.limit, u))
        self.buf.append((t, u))
        while len(self.buf) > 1 and self.buf[1][0] <= t - 0.056:
            self.buf.pop(0)
        return self.buf[0][1]

    def step(self):
        gc.TORSO_CONTROLLER = self.ctrl
        gc.apply_ctrl(self.data, self.act, self.data.time)
        if self.hw or self.cap:
            cur = np.array([self.data.ctrl[i] for i in self.legs])
            if self.held_cmd is None:
                self.held_cmd = cur.copy()
            self.held_cmd += np.clip(cur - self.held_cmd, -self.slew, self.slew)
            for i, j in enumerate(self.legs):
                self.data.ctrl[j] = self.held_cmd[i]
        mujoco.mj_step(self.model, self.data)

    def forces(self):
        d, m = self.data, self.model
        fn = {"L": 0.0, "R": 0.0}
        cop = {"L": np.zeros(3), "R": np.zeros(3)}
        grf = {"L": np.zeros(3), "R": np.zeros(3)}
        for ci in range(d.ncon):
            c = d.contact[ci]
            hit = [g for g in (c.geom1, c.geom2) if g in self.foot_geom]
            if not hit:
                continue
            s = self.foot_geom[hit[0]]
            fv = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, fv)
            fw = c.frame.reshape(3, 3).T @ fv[:3]
            if c.geom1 in self.foot_geom:
                fw = -fw
            n = abs(float(fv[0]))
            fn[s] += n
            cop[s] += n * c.pos
            grf[s] += fw
        for s in ("L", "R"):
            cop[s] = cop[s] / fn[s] if fn[s] > 1e-9 else None
        com = (d.xipos[1:] * self.mass[:, None]).sum(0) / self.mass.sum()
        return fn, cop, grf, com, d.xipos[self.tid].copy()

    def draw(self, ren, balance=False):
        fn, cop, grf, com, tcom = self.forces()
        k = ARROW_M / self.weight
        imgs = []
        # COM velocity from the previous drawn frame (1/FPS apart)
        v = np.zeros(3)
        if getattr(self, "_prev", None) is not None:
            v = (com - self._prev[0]) / max(self.data.time - self._prev[1], 1e-6)
        self._prev = (com.copy(), self.data.time)
        # balance construction: combined COP, total GRF, GRF line, miss distance, CMP
        bal = None
        ftot = sum(grf[s] for s in ("L", "R") if fn[s] > FN_MIN) if any(fn[s] > FN_MIN for s in ("L", "R")) else None
        if balance and ftot is not None and ftot[2] > 1e-6:
            ntot = sum(fn[s] for s in ("L", "R") if fn[s] > FN_MIN)
            cop_t = sum(fn[s] * cop[s] for s in ("L", "R") if fn[s] > FN_MIN) / ntot
            fhat = ftot / np.linalg.norm(ftot)
            foot = cop_t + fhat * np.dot(com - cop_t, fhat)          # foot of the perpendicular from the COM onto the GRF line
            miss = float(np.linalg.norm(com - foot))
            cmp_ = com - ftot * ((com[2] - cop_t[2]) / ftot[2])      # line through the COM parallel to the GRF, at floor height
            cmp_[2] = cop_t[2]
            M = np.cross(cop_t - com, ftot)
            bal = dict(cop=cop_t, F=ftot, fhat=fhat, foot=foot, miss=miss, cmp=cmp_, M=float(np.linalg.norm(M)),
                       dcmp=float(np.linalg.norm((cmp_ - cop_t)[:2])), v=float(np.linalg.norm(v)))
        R = self.data.xmat[self.root].reshape(3, 3)
        heading = math.degrees(math.atan2(R[1, 1], R[0, 1]))   # body +y in the world: camera azimuth = heading looks ALONG the walk, i.e. from behind
        for cam, (_, _, off) in zip(self.cams, CAMS):
            cam.lookat[:] = self.data.xpos[self.root]
            cam.lookat[2] = 0.16
            cam.azimuth = heading + off
            ren.update_scene(self.data, cam)
            scn = ren.scene
            if not balance:
                for s in ("L", "R"):
                    if cop[s] is not None and fn[s] > FN_MIN:
                        add_sphere(scn, cop[s], 0.008, RGBA["cop"])
                        add_arrow(scn, cop[s], cop[s] + grf[s] * k, 0.010, RGBA["grf"])
                add_sphere(scn, com, 0.016, RGBA["com"])
                add_arrow(scn, com, com + np.array([0, 0, -ARROW_M]), 0.007, RGBA["grav"])
                add_arrow(scn, com, np.array([com[0], com[1], 0.0]), 0.002, RGBA["drop"],
                          kind=mujoco.mjtGeom.mjGEOM_CAPSULE)
                add_sphere(scn, np.array([com[0], com[1], 0.0]), 0.008, RGBA["drop"])
                add_sphere(scn, tcom, 0.012, RGBA["tcom"])
            else:
                for s in ("L", "R"):
                    if cop[s] is not None and fn[s] > FN_MIN:
                        add_arrow(scn, cop[s], cop[s] + grf[s] * k, 0.005, RGBA["grf"])
                add_sphere(scn, com, 0.020, RGBA["com"])
                add_sphere(scn, tcom, 0.010, RGBA["tcom"])
                add_arrow(scn, com, com + v * VEL_M, 0.007, RGBA["vel"])
                if bal is not None:
                    add_sphere(scn, bal["cop"], 0.010, RGBA["cop"])
                    add_arrow(scn, bal["cop"], bal["cop"] + bal["F"] * k, 0.012, RGBA["ftot"])
                    top = bal["cop"] + bal["fhat"] * (np.dot(com - bal["cop"], bal["fhat"]) + 0.12)
                    add_arrow(scn, bal["cop"], top, 0.0025, RGBA["fline"], kind=mujoco.mjtGeom.mjGEOM_CAPSULE)
                    add_arrow(scn, com, bal["foot"], 0.004, RGBA["miss"], kind=mujoco.mjtGeom.mjGEOM_CAPSULE)
                    add_sphere(scn, bal["cmp"], 0.010, RGBA["cmp"])
                    add_arrow(scn, bal["cop"], bal["cmp"], 0.004, RGBA["cmpseg"], kind=mujoco.mjtGeom.mjGEOM_CAPSULE)
                    add_arrow(scn, com, bal["cmp"], 0.0015, RGBA["cmpseg"], kind=mujoco.mjtGeom.mjGEOM_CAPSULE)
            imgs.append(ren.render().copy())
        img = np.hstack(imgs)
        if balance:
            img = text_band(img, f"κ={self.kappa:g}    " + (
                f"COM–GRF line {bal['miss']*100:4.1f} cm     CMP–COP {bal['dcmp']*100:4.1f} cm     "
                f"|M| {bal['M']:4.2f} N·m     F/mg {np.linalg.norm(bal['F'])/self.weight:4.2f}     v {bal['v']:4.2f} m/s"
                if bal is not None else "airborne"))
        loaded = [s for s in ("L", "R") if fn[s] > FN_MIN]
        return img, loaded, com, cop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="1.67/340/95/24/20")
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--hw", action="store_true")
    ap.add_argument("--cap", action="store_true", help="354 deg/s leg cap only: kappa PID torso, no lag (c3/c4 cap-only sweep)")
    ap.add_argument("--kappas", nargs="*", type=float, default=[0.0, 2.0])
    ap.add_argument("--balance", action="store_true", help="GRF-line / CMP construction instead of the per-foot force picture")
    a = ap.parse_args()
    cell = tuple(float(x) for x in a.cell.split("/"))
    freq, phi, leg, hip, off = cell
    os.makedirs(OUT, exist_ok=True)
    print(f"model {gc.XML_PATH}  cell {a.cell}  mu {a.mu}  {'hardware layers + FF' if a.hw else ('cap only, kappa PID' if a.cap else 'ideal, kappa PID')}")

    gc.RAMP_HIP_OFFSET = True
    gs.STAGED_START = True
    gs.FLOOR_MU = a.mu
    gs.CONDITION["hip_off"] = off
    gs._set_gait(dict(freq=freq, hip_phi=phi, leg_amp=leg, hip_amp=hip))
    arms = [Arm(k, cell, a.mu, a.hw, a.cap) for k in a.kappas]
    gc.STAND_HIP_DEG = REST_LEAN
    for arm in arms:
        gc.set_initial_pose(arm.model, arm.data, arm.act, arm.jadr)
    rens = [mujoco.Renderer(arm.model, height=480, width=420, max_geom=2000) for arm in arms]

    gc.T_HOLD = 1e9
    t0, nxt = None, 0.0
    frames, snaps, hist = [], {}, {}
    tag = f"{'balance' if a.balance else 'forces'}_{'-'.join(f'{x:g}' for x in cell)}_mu{int(round(a.mu*100)):03d}{'_hw' if a.hw else ('_cap' if a.cap else '')}"
    while True:
        if t0 is None:
            t = arms[0].data.time
            if (t >= gs.QUIET_MIN_T
                    and max(float(np.max(np.abs(x.data.qvel))) for x in arms) < gs.QUIET_QVEL) \
                    or t >= gs.QUIET_MAX_T:
                t0 = t
                gc.T_HOLD = t
        for arm in arms:
            arm.step()
            if arm.data.xpos[arm.root][2] < 0.05 and arm.fell is None:
                arm.fell = arm.data.time - (t0 or 0.0)
        if t0 is None:
            continue
        tw = arms[0].data.time - t0 - gc.T_TRANSITION - SETTLE
        if tw > WINDOW:
            break
        if arms[0].data.time >= nxt:
            nxt += 1.0 / FPS
            row, states = [], []
            for arm, ren in zip(arms, rens):
                img, loaded, com, cop = arm.draw(ren, a.balance)
                row.append(img)
                states.append(loaded)
            frames.append(np.hstack(row))
            if tw >= 1.0:
                for i, (arm, st) in enumerate(zip(arms, states)):
                    hist.setdefault(arm.kappa, []).append((st[0] if len(st) == 1 else "", row[i]))
    for r in rens:
        r.close()
    gc.T_HOLD = 5.0
    # snapshot = middle frame of the longest single-stance run of each foot, per kappa
    for kappa, seq in hist.items():
        for side in ("L", "R"):
            best, cur = (0, 0), 0
            for i, (st, _) in enumerate(seq + [("", None)]):
                if st == side:
                    cur += 1
                else:
                    if cur > best[0]:
                        best = (cur, i - cur)
                    cur = 0
            if best[0]:
                snaps[(kappa, side)] = seq[best[1] + best[0] // 2][1]

    path = os.path.join(OUT, tag + ".mp4")
    imageio.mimsave(path, frames, fps=FPS, macro_block_size=1)
    print(f"{len(frames)} frames -> {path}")
    for (kappa, s), img in sorted(snaps.items()):
        p = os.path.join(OUT, f"{tag}_k{kappa:g}_{s}stance.png")
        imageio.imwrite(p, img)
        print(f"  snapshot {p}")
    for arm in arms:
        print(f"  kappa {arm.kappa:g}: {'FELL at %.1fs' % arm.fell if arm.fell else 'walked'}")
    if a.balance:
        print("legend: black ball = combined COP, yellow arrow = total GRF (weight = 0.15 m), thin yellow = GRF line of action;"
              " red ball = COM, red segment = COM's miss distance to the GRF line; magenta ball = CMP, magenta ground segment = CMP-COP;"
              " cyan = COM velocity (0.25 m per m/s); thin green = per-foot GRF; grey = torso COM")
        return
    print("legend: green = GRF at each loaded foot (from its COP, black dot); red ball = whole-body COM,"
          " red arrow = gravity (same scale, body weight = 0.15 m), thin red line = COM drop line;"
          " grey ball = torso COM")


if __name__ == "__main__":
    main()

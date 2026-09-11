"""balance_figure.py (GRID-7 paper figure) — top: two robots: two robots (kappa 0 and kappa 2, each on its
own champion gait and COM model) rendered from BEHIND at the instant the LEFT foot is the only
support, with the GRF-line construction drawn on; bottom: the COM-to-GRF-line distance at
single-support onset vs floor friction for all six configs (touch_miss.py summary csv), in the
plot law. The two configs shown above are drawn full-strength, the others faded.

Overlay per robot: black ball = combined COP, yellow arrow = total GRF (weight = 0.15 m), thin
yellow line = its line of action; green ball = whole-body COM; the segment from the COM
perpendicular to the GRF line = the moment arm (blue solid for kappa 0, red dashed for kappa 2);
magenta ball = CMP (line through the COM parallel to the GRF meets the floor), magenta ground
segment CMP-COP; cyan arrow = COM velocity (0.25 m per m/s); grey ball = torso COM. The COM ball is
GREEN here (red is kappa 2's colour) and the per-foot GRF arrows are not drawn (single stance: they
coincide with the total).

    PENGU_MODEL=1.31 python grid7/balance_figure.py --k0 c1:1.7/290/165/32/50 --k2 c6:1.98/290/85/28/20 --mu 0.1 --tag grid5 \
        --onset-csv results/grid6_probes/touch_miss/touch_miss_summary_robust.csv        # template on GRID-5 data
Physics = GRID-5 (ideal actuators, kappa PID, COM slide on the base model). --cap adds the 354 deg/s
cap on legs + torso (the GRID-7 physics); --model-per-config uses the hardware CAD models instead
of the slide.
"""
import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(ROOT, "grid6"))
sys.path.insert(0, _HERE)
sys.path.append(ROOT)
sys.path.insert(0, os.path.join(ROOT, "grid5", "analysis"))
os.environ.setdefault("PENGU_MODEL", "1.31")

import mujoco                                    # noqa: E402
import gait_config as gc                         # noqa: E402
import gait_sweep as gs                          # noqa: E402
import render_forces as RF                       # noqa: E402
import stance_com as SC                          # noqa: E402
import style5                                    # noqa: E402
import matplotlib                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})
import pandas as pd                              # noqa: E402

OUT = os.path.join(ROOT, "results", "grid7_report", "paper")      # png only
DATA = os.path.join(ROOT, "results", "grid7_report", "data")     # csv + caption text
WORK = os.path.join(ROOT, "results", "grid7_report", "work")     # caches, logs, scripts
BODY_ALPHA = 0.16                                # robot meshes: fainter than render_forces so the overlay reads through them
COM_RGBA = (0.10, 0.65, 0.20, 1.0)
# Hermès publishes no RGB for its leather shades; these are the commonly cited approximations
# (Craie = chalk off-white, Etoupe = taupe). grey is neutral mid-grey.
FLOORS = {"grey": (0.62, 0.62, 0.64, 1.0),
          "beige": (0.83, 0.72, 0.59, 1.0),        # sandy beige #D4B896 (Craie-white washes out under the render lights)
          "etoupe": (0.61, 0.56, 0.50, 1.0),       # Etoupe-like #9B8F80
          "tan": (0.78, 0.62, 0.42, 1.0),          # Hermès "Gold" tan leather, approx. #C79E6B (no official RGB)
          "tan2": (0.70, 0.58, 0.46, 1.0)}         # Ben's swatch #C4A484 (196,164,132), set darker to survive the render lighting
FLOOR_RGBA = FLOORS["grey"]             # the finite floor slab drawn under the robot (Ben: a floor, not the whole frame)
FLOOR_HALF = (0.26, 0.20)                        # slab half-extents [m]: across the walk, along the walk
CAM = (0.88, -26, 0)                             # distance, elevation, azimuth offset: from behind, looking down enough that the horizon leaves the frame
FS = 60                                          # logging / frame rate
N_AFTER = 12                                     # frames after each snapshot kept in the cache (for --replot --snap-offset)
WINDOW = 6.0


class Robot(RF.Arm):
    def __init__(self, cfg, cell, mu, cap):
        self.cfg = cfg
        kappa, com = style5.CONFIGS[cfg]
        self.com_target = com
        super().__init__(kappa, cell, mu, hw=False, cap=cap)
        self.cams = self.cams[:1]
        self.log = []

    MODEL_OF = {"c1": "pengu1_05_hw_updated", "c4": "pengu1_05_hw_updated", "c2": "pengu1_20_hw_updated",
                "c5": "pengu1_20_hw_updated", "c3": "1.31", "c6": "1.31"}

    def build_model(self):
        if self.cap:                                   # GRID-7: the config's hardware CAD model
            m = mujoco.MjModel.from_xml_path(gc._COM_MODELS[self.MODEL_OF[self.cfg]])
        else:                                          # GRID-5: base model + COM slide
            m = mujoco.MjModel.from_xml_path(gs.XML)
            SC.apply_com_variant(m, self.com_target)
        return m

    def draw_frame(self, ren):
        """rear view with the balance overlay; miss segment in the config's colour"""
        fn, cop, grf, com, tcom = self.forces()
        k = RF.ARROW_M / self.weight
        v = np.zeros(3)
        if getattr(self, "_prev", None) is not None:
            v = (com - self._prev[0]) / max(self.data.time - self._prev[1], 1e-6)
        self._prev = (com.copy(), self.data.time)
        loaded = [s for s in ("L", "R") if fn[s] > RF.FN_MIN]
        bal = None
        if loaded and sum(grf[s] for s in loaded)[2] > 1e-6:
            ntot = sum(fn[s] for s in loaded)
            cop_t = sum(fn[s] * cop[s] for s in loaded) / ntot
            F = sum(grf[s] for s in loaded)
            fhat = F / np.linalg.norm(F)
            foot = cop_t + fhat * np.dot(com - cop_t, fhat)
            cmp_ = com - F * ((com[2] - cop_t[2]) / F[2]); cmp_[2] = cop_t[2]
            bal = dict(cop=cop_t, F=F, fhat=fhat, foot=foot, miss=float(np.linalg.norm(com - foot)), cmp=cmp_,
                       M=float(np.linalg.norm(np.cross(cop_t - com, F))))
        R = self.data.xmat[self.root].reshape(3, 3)
        heading = math.degrees(math.atan2(R[1, 1], R[0, 1]))
        cam = self.cams[0]
        cam.lookat[:] = self.data.xpos[self.root]; cam.lookat[2] = 0.14
        cam.distance, cam.elevation, cam.azimuth = CAM[0], CAM[1], heading + CAM[2]
        ren.update_scene(self.data, cam)
        scn = ren.scene
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0          # the torso's shadow read as a dark blob on the white floor
        scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        # finite floor slab, aligned with the heading so it reads as a trapezoid from behind
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            th = math.radians(heading - 90.0)
            Rz = np.array([[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]])
            fwd = np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading)), 0.0])
            centre = np.array([self.data.xpos[self.root][0], self.data.xpos[self.root][1], 0.0005]) - 0.03 * fwd   # slab a little toward the camera
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([FLOOR_HALF[0], FLOOR_HALF[1], 0.0005]),
                                centre, Rz.ravel(), np.array(FLOOR_RGBA, np.float32))
            scn.ngeom += 1
        col = style5.GAIT[self.kappa]["color"]
        rgb = tuple(int(col[i:i + 2], 16) / 255 for i in (1, 3, 5))
        RF.add_sphere(scn, com, 0.016, COM_RGBA)                                   # whole-body COM (green)
        RF.add_arrow(scn, com, np.array([com[0], com[1], 0.0]), 0.0018, (0.1, 0.55, 0.15, 0.6),
                     kind=mujoco.mjtGeom.mjGEOM_CAPSULE)                           # its vertical drop to the floor
        RF.add_sphere(scn, np.array([com[0], com[1], 0.0]), 0.007, (0.1, 0.55, 0.15, 0.8))
        if bal is not None:
            RF.add_sphere(scn, bal["cop"], 0.010, RF.RGBA["cop"])                   # combined COP
            RF.add_arrow(scn, bal["cop"], bal["cop"] + bal["F"] * k, 0.012, RF.RGBA["ftot"])   # total GRF
            top = bal["cop"] + bal["fhat"] * (np.dot(com - bal["cop"], bal["fhat"]) + 0.12)
            RF.add_arrow(scn, bal["cop"], top, 0.0025, RF.RGBA["fline"], kind=mujoco.mjtGeom.mjGEOM_CAPSULE)   # line of action
            RF.add_arrow(scn, com, bal["foot"], 0.012, rgb + (1.0,), kind=mujoco.mjtGeom.mjGEOM_CAPSULE)   # moment arm, config colour
            RF.add_sphere(scn, bal["foot"], 0.009, rgb + (1.0,))
        img = ren.render().copy()
        st = loaded[0] if len(loaded) == 1 else ("D" if loaded else "A")
        zf = {sd: float(min(self.data.geom_xpos[g][2] for g, s2 in self.foot_geom.items() if s2 == sd)) for sd in ("L", "R")}
        self.log.append(dict(t=self.data.time, st=st, miss=bal["miss"] if bal else np.nan, M=bal["M"] if bal else np.nan,
                             fnL=fn["L"], fnR=fn["R"], zL=zf["L"], zR=zf["R"]))
        return img


def _patched_init(self, kappa, cell, mu, hw, cap=False):
    """RF.Arm.__init__ but with the model built by self.build_model()"""
    self.kappa, self.cell, self.mu, self.hw, self.cap = kappa, cell, mu, hw, cap
    self.model = self.build_model()
    self.data = mujoco.MjData(self.model)
    gc.STAND_HIP_DEG = 0.0
    from torso_control import TorsoKappaPID
    self.pid = TorsoKappaPID(self.model, kappa=kappa, measure_after=0.0, ctrl_limit_deg=45.0 if kappa else (25.0 if (hw or cap) else 45.0))
    from friction_utils import set_floor_friction
    set_floor_friction(self.model, mu)
    self.floor_id, self.foot_geom, self.foot_bid, self.root = gs.make_ids(self.model)
    self.act, self.jadr = gc.build_ids(self.model)
    self.legs = [self.act[n] for n in ("crank1-L", "crank1-R", "hip-L", "hip-R")] + ([self.act["torso"]] if cap else [])
    self.tid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
    self.mass = self.model.body_mass[1:]
    self.weight = float(self.mass.sum()) * 9.81
    self.slew = math.radians(354.0) * self.model.opt.timestep
    self.held_cmd = None; self.buf = []; self.ff = None; self.fell = None
    self.model.geom_rgba[:, 3] = np.where(self.model.geom_rgba[:, 3] > 0, BODY_ALPHA, 0.0)
    # solid dark-blue, non-reflective floor: the checker material's reflection mirrored the robot below
    # the floor and read as penetration
    mat = int(self.model.geom_matid[self.floor_id])
    if mat >= 0:
        self.model.mat_reflectance[mat] = 0.0
        self.model.mat_rgba[mat] = [1.0, 1.0, 1.0, 1.0]
        self.model.mat_texid[mat] = -1
    self.model.geom_rgba[self.floor_id] = [1.0, 1.0, 1.0, 1.0]     # the physical plane: white, invisible against the page
    self.model.geom_matid[self.floor_id] = -1
    c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE
    self.cams = [c]


RF.Arm.__init__ = _patched_init


def run_pair(k0, k2, mu, cap):
    """step both robots in lockstep on their own gaits (own phase each); return frames + logs"""
    robots = []
    for cfg, cell in (k0, k2):
        r = Robot(cfg, cell, mu, cap)
        robots.append(r)
    rens = [mujoco.Renderer(r.model, height=480, width=480, max_geom=2000) for r in robots]
    frames = {r.cfg: [] for r in robots}
    for r in robots:                                   # each robot runs its own gait: set per step
        r.gait = dict(freq=r.cell[0], hip_phi=r.cell[1], leg_amp=r.cell[2], hip_amp=r.cell[3], hip_off=r.cell[4])
    gc.RAMP_HIP_OFFSET = True; gs.STAGED_START = True
    for r in robots:
        gs.CONDITION["hip_off"] = r.gait["hip_off"]; gs._set_gait(dict(freq=r.gait["freq"], hip_phi=r.gait["hip_phi"], leg_amp=r.gait["leg_amp"], hip_amp=r.gait["hip_amp"]))
        gc.STAND_HIP_DEG = RF.REST_LEAN
        gc.set_initial_pose(r.model, r.data, r.act, r.jadr)
    t0, nxt = None, 0.0
    gc.T_HOLD = 1e9
    while True:
        if t0 is None:
            t = robots[0].data.time
            if (t >= gs.QUIET_MIN_T and max(float(np.max(np.abs(x.data.qvel))) for x in robots) < gs.QUIET_QVEL) or t >= gs.QUIET_MAX_T:
                t0 = t; gc.T_HOLD = t
        for r in robots:
            gs.CONDITION["hip_off"] = r.gait["hip_off"]
            gs._set_gait(dict(freq=r.gait["freq"], hip_phi=r.gait["hip_phi"], leg_amp=r.gait["leg_amp"], hip_amp=r.gait["hip_amp"]))
            r.step()
            if r.data.xpos[r.root][2] < 0.05 and r.fell is None:
                r.fell = r.data.time - (t0 or 0.0)
        if t0 is None:
            continue
        tw = robots[0].data.time - t0 - gc.T_TRANSITION - RF.SETTLE
        if tw > WINDOW:
            break
        if robots[0].data.time >= nxt:
            nxt += 1.0 / FS
            for r, ren in zip(robots, rens):
                img = r.draw_frame(ren)
                if tw >= 0:
                    frames[r.cfg].append(img)
                    r.log[-1]["tw"] = tw
    for ren in rens:
        ren.close()
    gc.T_HOLD = 5.0
    return robots, frames


def left_stance_snapshot(r, ncyc=2.0, stat="single"):
    """stat 'single': middle of the longest run of frames in which the LEFT foot is the only loaded
    foot; stat 'touchdown': the left foot's first touchdown (Fn crossing FN_MIN from below). Both
    inside the last `ncyc` gait cycles, raw frames."""
    logs = [e for e in r.log if "tw" in e]
    t_lo = logs[-1]["tw"] - ncyc / r.cell[0]
    if stat == "touchdown":
        for i in range(1, len(logs)):
            if logs[i]["tw"] >= t_lo and logs[i]["fnL"] > RF.FN_MIN and logs[i - 1]["fnL"] <= RF.FN_MIN:
                return i
        return len(logs) // 2
    best, cur = (0, 0), 0
    for i, e in enumerate(logs + [None]):
        if e is not None and e["st"] == "L" and e["tw"] >= t_lo:
            cur += 1
        else:
            if cur > best[0]:
                best = (cur, i - cur)
            cur = 0
    return best[1] + best[0] // 2 if best[0] else len(logs) // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k0", default="c1:1.7/290/165/32/50")
    ap.add_argument("--k2", default="c6:1.98/290/85/28/20")
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--cap", action="store_true", help="354 deg/s cap on legs + torso (GRID-7 physics)")
    ap.add_argument("--tag", default="grid5")
    ap.add_argument("--floor", default="tan2", help="grey | beige | etoupe | tan | tan2 | r,g,b (0-255); tan2 is the paper floor (Ben 2026-09-10)")
    ap.add_argument("--name-floor", action="store_true", help="append the floor name to the png (default: no colour in the name)")
    ap.add_argument("--replot", action="store_true", help="reuse the cached snapshot frames; only the plotting runs")
    ap.add_argument("--sheet", type=int, default=0, help="also save a contact sheet of the N frames after the snapshot (work/)")
    ap.add_argument("--snap-offset", type=int, nargs="+", default=[0], help="use the frame this many steps (1/%d s) after the chosen snapshot; one value or one per robot (k0 k2); works with --replot (the %d frames after each snapshot are cached)" % (FS, N_AFTER))
    ap.add_argument("--stat", default="single", choices=["single", "touchdown"],
                    help="single: snapshot + statistic at single-support frames; touchdown: at the first touchdown of each cycle")
    ap.add_argument("--onset-csv", default=None,
                    help="touch_miss7.py summary (cfg, mu, miss_mean, miss_std) for the bottom panel; default data/touch_miss_grid7_summary_<stat>.csv")
    a = ap.parse_args()
    if a.onset_csv is None:
        a.onset_csv = os.path.join(DATA, f"touch_miss_{a.tag}_summary_{a.stat}.csv")
    offs = a.snap_offset if len(a.snap_offset) == 2 else [a.snap_offset[0], a.snap_offset[0]]
    os.makedirs(OUT, exist_ok=True)
    global FLOOR_RGBA
    FLOOR_RGBA = FLOORS.get(a.floor) or tuple(int(x) / 255 for x in a.floor.split(",")) + (1.0,)
    parse = lambda s: (s.split(":")[0], tuple(float(x) for x in s.split(":")[1].split("/")))
    k0, k2 = parse(a.k0), parse(a.k2)
    import pickle
    cache = os.path.join(WORK, f"balance_pair_{a.tag}_mu{int(round(a.mu*100)):03d}_{a.stat}_{a.floor.replace(',', '-')}_snapshots.pkl")
    if a.replot and os.path.exists(cache):
        C = pickle.load(open(cache, "rb"))
        print(f"replot from {cache} (the snapshots keep the floor colour they were rendered with)")
        for j, d in enumerate(C):
            if "frames_after" not in d:
                raise SystemExit("this cache predates the frames_after field: re-render once without --replot")
            k = min(offs[j], len(d["frames_after"]) - 1)
            d["frame"], d["snap"] = d["frames_after"][k], d["logs_after"][k]
            d["prev"] = d["logs_after"][k - 1] if k > 0 else d["prev0"]
            print(f"  {d['cfg']}: snapshot offset +{k} (t+{(d['snap']['tw'] - d['logs_after'][0]['tw'])*1000:.0f} ms)")
    else:
        robots, frames = run_pair(k0, k2, a.mu, a.cap)
        C = []
        for r in robots:
            print(f"{r.cfg} kappa {r.kappa:g} COM {r.com_target:.2f} {r.cell}: {'FELL %.1fs' % r.fell if r.fell else 'walked'}, {len(frames[r.cfg])} frames")
            i0 = left_stance_snapshot(r, 2.0, a.stat)
            logs = [e for e in r.log if "tw" in e]
            if a.sheet:
                n = min(a.sheet, len(logs) - i0)
                cols = 5; rows_ = int(math.ceil(n / cols))
                fs, axs = plt.subplots(rows_, cols, figsize=(3.0 * cols, 3.2 * rows_), squeeze=False)
                for k in range(rows_ * cols):
                    ax = axs[k // cols][k % cols]; ax.axis("off")
                    if k < n:
                        e = logs[i0 + k]
                        ax.imshow(frames[r.cfg][i0 + k])
                        ax.set_title(f"+{k}: t+{(e['tw'] - logs[i0]['tw'])*1000:.0f} ms  Fn L {e['fnL']:.0f} / R {e['fnR']:.0f} N  miss {e['miss']*100 if np.isfinite(e['miss']) else float('nan'):.1f} cm", fontsize=8)
                fs.suptitle(f"{r.cfg} ({a.stat}): frames after the snapshot at t = {logs[i0]['tw']:.2f} s, 1/{FS} s apart", fontsize=10)
                fs.tight_layout()
                ps = os.path.join(WORK, f"sheet_{a.tag}_mu{int(round(a.mu*100)):03d}_{a.stat}_{r.cfg}.png")
                fs.savefig(ps, dpi=110); plt.close(fs); print("sheet ->", ps)
            i = min(i0 + offs[len(C)], len(logs) - 1)
            C.append(dict(cfg=r.cfg, kappa=r.kappa, com=r.com_target, cell=r.cell, frame=frames[r.cfg][i], snap=logs[i],
                          prev=logs[i - 1] if i > 0 else logs[i],
                          frames_after=[frames[r.cfg][k] for k in range(i0, min(i0 + N_AFTER, len(logs)))],
                          logs_after=logs[i0:i0 + N_AFTER], prev0=logs[i0 - 1] if i0 > 0 else logs[i0]))
        pickle.dump(C, open(cache, "wb"))

    fig = plt.figure(figsize=(10, 10.0))
    gsp = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.30, wspace=0.03)
    snaps = {}
    class _R:                                          # what the plotting code needs of a robot
        def __init__(self, d): self.cfg, self.kappa, self.com_target, self.cell = d["cfg"], d["kappa"], d["com"], d["cell"]
    robots = [_R(d) for d in C]
    for j, d in enumerate(C):
        ax = fig.add_subplot(gsp[0, j])
        snaps[d["cfg"]] = d["snap"]
        ax.imshow(d["frame"]); ax.axis("off")
        e, prev = d["snap"], d["prev"]
        print(f"  {d['cfg']} snapshot ({a.stat}) at tw {e['tw']:.3f} s -- Fn L {e['fnL']:.1f} N, R {e['fnR']:.1f} N; "
              f"frame before: {prev['st']} (Fn L {prev['fnL']:.1f}, R {prev['fnR']:.1f})")
        ax.set_title(f"{d['cfg']}:  κ = {d['kappa']:g},  COM ratio {d['com']:.2f}  —  " + ("left foot the only support" if a.stat == "single" else "left-foot touchdown"),
                     fontsize=9.5, color=style5.GAIT[d["kappa"]]["color"])
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = [Line2D([], [], color="#f2cc1a", lw=3, marker=">", ms=7, mec="#f2cc1a", label="total ground reaction force (from the COP, black) and its line of action"),
          Line2D([], [], color="#1aa633", lw=0, marker="o", ms=8, label="whole-body COM, with its vertical drop to the floor"),
          Line2D([], [], color=style5.GAIT[0.0]["color"], lw=3, label="moment arm of the GRF about the COM, κ = 0"),
          Line2D([], [], color=style5.GAIT[2.0]["color"], lw=3, label="moment arm of the GRF about the COM, κ = 2")]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 0.555), ncol=2, frameon=False, fontsize=8, handlelength=2.2)
    ax = fig.add_subplot(gsp[1, :])
    csv = a.onset_csv or os.path.join(DATA, f"touch_miss_grid7_summary_{a.stat}.csv")
    T = pd.read_csv(csv)
    T = T[T.mu <= 0.7 + 1e-9]
    hs2 = []
    for r in robots:
        c = r.cfg
        q = T[T.cfg == c].sort_values("mu")
        if q.empty:
            continue
        k, com = style5.CONFIGS[c]
        st = style5.style_for(k, com)
        ax.errorbar(q.mu, q.miss_mean * 100, yerr=q.miss_std.fillna(0) * 100, capsize=3, elinewidth=1.0, **st)
        for _, qq in q.iterrows():                        # thin samples are shown, never hidden
            if qq.n < 6:
                ax.annotate(f"n = {int(qq.n)}" + (f", rank {int(qq['rank'])}" if "rank" in qq and qq["rank"] > 1 else ""),
                            (qq.mu, qq.miss_mean * 100), xytext=(6, 6), textcoords="offset points", fontsize=7.5, color=st["color"])
        hs2.append(Line2D([], [], color=st["color"], ls=st["linestyle"], lw=st["lw"], marker=st["marker"], ms=st["ms"] - 2,
                          mfc=st["mfc"], mec=st["mec"], mew=st["mew"], label=f"{c}: κ = {k:g}, COM ratio {com:.2f}"))
    ax.set_xlabel("floor friction μ"); ax.set_xticks(sorted(T.mu.unique()))
    ax.set_ylabel("COM to GRF line during single support [cm]" if a.stat == "single" else "COM to GRF line at first touchdown of the cycle [cm]")
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
    ax.legend(handles=hs2, loc="upper right", handlelength=3.0)
    ax.set_title(("Moment arm of the ground reaction during single support, per gait cycle" if a.stat == "single" else
                  "Moment arm of the ground reaction at the first touchdown of each gait cycle") + " (mean ± SD over the cycles of one rollout)")
    e0, e2 = snaps[robots[0].cfg], snaps[robots[1].cfg]
    phys = ("GRID-7 physics: 354 °/s velocity cap on every servo, ±4.1 N·m torque cap, κ PID torso, hardware CAD models; champion = fastest robust cell under the motor bar"
            if a.cap else "GRID-5 physics: ideal actuators, κ PID torso; champion = fastest robust cell")
    def when(j):                                   # snapshot timing relative to the left-foot touchdown
        k = offs[j] if a.stat == "touchdown" else 0
        return "at the instant the left foot touches down" if k == 0 else f"{k / FS * 1000:.0f} ms after the left-foot touchdown"
    top_when = (("at an instant when the left foot is the only support") if a.stat == "single"
                else (when(0) if offs[0] == offs[1] else f"{when(0)} ({robots[0].cfg}) and {when(1)} ({robots[1].cfg})"))
    T_ = pd.read_csv(a.onset_csv); T_ = T_[T_.cfg.isin([robots[0].cfg, robots[1].cfg])]
    nonchamp = T_[T_["rank"] != 1]
    rank_note = ""
    if len(nonchamp):
        rank_note = " Exceptions: " + "; ".join(f"{r.cfg} at μ = {r.mu:g} uses rank {int(r['rank'])} ({r.cell})" for _, r in nonchamp.iterrows()) + \
                    " -- the fastest cell of the ranking that walks on this machine and, for c6 at μ = 0.1, has no flight phase with a gap ≥ 2 mm under the feet in the steady part of the rollout."
    notes = (f"Top: rear view at μ = {a.mu:g} {top_when}. Snapshots: {robots[0].cfg} {e0['miss']*100:.1f} cm (|M| = {e0['M']:.2f} N·m) at t = {e0['tw']:.2f} s, "
             f"gait {'/'.join(f'{x:g}' for x in robots[0].cell)}; {robots[1].cfg} {e2['miss']*100:.1f} cm (|M| = {e2['M']:.2f} N·m) at t = {e2['tw']:.2f} s, "
             f"gait {'/'.join(f'{x:g}' for x in robots[1].cell)} (freq [Hz] / hip_phi / leg_amp / hip_amp / hip_off [°]). {phys}.{rank_note}")
    stat_txt = ("averaged over the single-support frames of each gait cycle" if a.stat == "single" else "at the first touchdown of each gait cycle")
    caption = (f"Moment arm of the ground reaction force about the whole-body COM at μ = {a.mu:g}. Top: rear view at left-foot touchdown for κ = 0 ({robots[0].cfg}) "
               f"and κ = 2 ({robots[1].cfg}); yellow, total ground reaction force and its line of action; green, whole-body COM and its vertical projection; "
               f"the coloured segment is the perpendicular from the COM to the GRF line (blue κ = 0, red κ = 2). Bottom: the same distance {stat_txt}, "
               f"mean ± SD over the cycles of one rollout, versus floor friction, for the fastest robust gait of each configuration.")
    base = os.path.join(DATA, f"balance_pair_{a.tag}_mu{int(round(a.mu*100)):03d}_{a.stat}")
    open(base + "_caption.txt", "w").write(caption + "\n")
    open(base + "_caption_notes.txt", "w").write(notes + "\n")          # setup details for us, not for the reader
    print("CAPTION:", caption)
    fig.subplots_adjust(top=0.95, bottom=0.06, left=0.08, right=0.98)
    p = os.path.join(OUT, f"balance_pair_{a.tag}_mu{int(round(a.mu*100)):03d}_{a.stat}" + (f"_{a.floor.replace(',', '-')}" if a.name_floor else "") + ".png")
    fig.savefig(p, dpi=170)
    print("->", p)


if __name__ == "__main__":
    main()

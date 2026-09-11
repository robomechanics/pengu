"""duty_strip7.py — GRID-7 version of the duty-cycle strip (Ben 2026-09-11): one gait cycle of the
champion gait of every configuration at one friction as a row of top-down frames, six rows (c1 .. c6)
stacked top to bottom, followed by the gait-cycle plots (support bar + lateral COM shift) for two of
them (default c1 and c6). Eight panels in one figure, paper fonts (Times, 9 pt).

Physics: GRID-7 (354 deg/s cap on legs + torso, kappa PID, the config's hardware CAD model), via
balance_figure.Robot. Each config is simulated twice: pass 1 logs the feet forces / COMs / body heading
and picks the cycle (start = peak of the smoothed left-minus-right normal force inside the last 3 cycles,
i.e. the middle of the left foot's support); pass 2 re-runs deterministically with a FIXED top-down
camera (Ben 2026-09-11): image up = the BODY's forward direction (root +y, circular mean over the
cycle; checked by projection, the azimuth is flipped if needed), the frame centred on the bounding box
of the feet and COM over the cycle, portrait frames, and ONE camera distance for all six rows (set by
the row that needs the most room), so the robot's progression over the cycle is visible and comparable
between rows: a forward walker climbs the frame, a backward walker (c1 mu 0.1) descends, a crab walker
(c2 mu 0.1) moves diagonally. The angle between the body's facing and the travel direction is printed
and written to the notes. (--up travel gives the earlier variant: travel direction up, start pose on the
bottom edge.)
Markers: whole-body COM (star), torso COM (triangle), projected with mech_common.project.

    PENGU_MODEL=1.31 python grid7/duty_strip7.py --mu 0.1            # defaults = the GRID-7 mu 0.1 champions
    python grid7/duty_strip7.py --replot                              # from work/duty_strip_<tag>_<cfg>.pkl
-> results/grid7_report/paper/duty_strip_grid7_mu010.png, data/duty_strip_grid7_mu010_caption.txt
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
sys.path.insert(0, os.path.join(ROOT, "grid6", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "grid5", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "grid6"))
sys.path.append(ROOT)
os.environ.setdefault("PENGU_MODEL", "1.31")
import balance_figure as BF                      # noqa: E402  (patches RF.Arm to build per-config models)
import render_forces as RF                       # noqa: E402
import mech_common as mc                         # noqa: E402  (project / smooth_dom / stance_centers only)
import gait_config as gc                         # noqa: E402
import gait_sweep as gs                          # noqa: E402
import style5                                    # noqa: E402
import mujoco                                    # noqa: E402
from gait_cycle7 import filter_states            # noqa: E402  (contact-chatter filter on the support state)
import matplotlib                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402
from matplotlib.lines import Line2D              # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})

OUT = os.path.join(ROOT, "results", "grid7_report", "paper")
DATA = os.path.join(ROOT, "results", "grid7_report", "data")
WORK = os.path.join(ROOT, "results", "grid7_report", "work")
CFGS = ["c1", "c2", "c3", "c4", "c5", "c6"]
# GRID-7 mu 0.1 champions (robust + clearance + motor bar, fastest first); c6 = rank 4, the fastest cell with no
# >= 2 mm hop in the steady phase (Ben's rule, noflight_search.py)
# Ben 2026-09-11: the duty strip uses the facing-filtered champions (|body facing - travel| <= 60 deg, GRID-5 heading rule):
# c1 = rank 79 1.5/220/110/32/40 (ranks 1-78 walk backwards or > 60 deg); the others at mu 0.1 pass the rule at rank 1
# (c6 rank 4 = the 2 mm no-hop rule). The touchdown balance figure keeps the speed-ranked champions instead.
CELLS = {"c1": (1.5, 220, 110, 32, 40), "c2": (1.3, 0, 125, 28, 20), "c3": (1.4, 290, 115, 24, 20),
         "c4": (1.35, 350, 115, 32, 20), "c5": (1.65, 270, 100, 28, 30), "c6": (1.6, 300, 95, 28, 20)}
RANK = {"c1": 79, "c2": 1, "c3": 1, "c4": 1, "c5": 1, "c6": 4}
H, W = 480, 320                                  # portrait frames (2:3)
FLOOR = "tan2"                                   # the paper floor colour (balance_figure.FLOORS)
EL = -89.9
FOVY = 45.0                                      # MuJoCo default vertical field of view
MARGIN = 0.04                                    # m, around the bounding box of the cycle
UP = "body"                                      # "body": image up = body forward; "travel": image up = travel direction
PAD_REAR = 0.10                                  # m, foot body origin -> sole edge behind the rearmost logged point
PAD_FRONT = 0.20                                 # m, ahead of the frontmost logged point: the leaning torso is ~0.35 m up and the
                                                 #    top-down camera magnifies it (D / (D - z) ~ 1.6 at D = 0.9 m), feet origins -> toes
FS = 120.0                                       # log / frame rate
CHATTER_MS = 30.0                                # support-state segments shorter than this are merged into their neighbours (Ben 2026-09-11)
FN_MIN = RF.FN_MIN


def simulate(cfg, cell, mu, frame_ts=None, az=0.0, lookat=None, dist=0.62):
    """one GRID-7 rollout of `cell`; returns the log (per 1/FS s) and, if frame_ts is given, the top-down
    frames at those window times from a fixed camera (lookat, dist, azimuth az)"""
    r = BF.Robot(cfg, cell, mu, cap=True)
    r.model.geom_rgba[r.floor_id] = BF.FLOORS[FLOOR]      # _patched_init made the floor white; give it the paper colour
    r.gait = dict(freq=cell[0], hip_phi=cell[1], leg_amp=cell[2], hip_amp=cell[3], hip_off=cell[4])
    gc.RAMP_HIP_OFFSET = True; gs.STAGED_START = True
    gs.CONDITION["hip_off"] = r.gait["hip_off"]
    gs._set_gait(dict(freq=cell[0], hip_phi=cell[1], leg_amp=cell[2], hip_amp=cell[3]))
    gc.STAND_HIP_DEG = RF.REST_LEAN
    gc.set_initial_pose(r.model, r.data, r.act, r.jadr)
    ren = cam = None
    if frame_ts is not None:
        ren = mujoco.Renderer(r.model, height=H, width=W, max_geom=2000)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.elevation, cam.azimuth = dist, EL, az
        cam.lookat[:] = lookat
    log, frames = [], []
    t0, nxt, k = None, 0.0, 0
    gc.T_HOLD = 1e9
    while True:
        if t0 is None:
            t = r.data.time
            if (t >= gs.QUIET_MIN_T and float(np.max(np.abs(r.data.qvel))) < gs.QUIET_QVEL) or t >= gs.QUIET_MAX_T:
                t0 = t; gc.T_HOLD = t
        r.step()
        if r.data.xpos[r.root][2] < 0.05 and r.fell is None:
            r.fell = r.data.time - (t0 or 0.0)
        if t0 is None:
            continue
        tw = r.data.time - t0 - gc.T_TRANSITION - RF.SETTLE
        if tw > BF.WINDOW or (frame_ts is not None and k >= len(frame_ts)):
            break
        if tw < 0:
            continue
        if frame_ts is None and r.data.time >= nxt:
            nxt = r.data.time + 1.0 / FS
            fn, cop, grf, com, tcom = r.forces()
            Rm = r.data.xmat[r.root].reshape(3, 3)
            log.append(dict(tw=tw, fnL=fn["L"], fnR=fn["R"], com=com.copy(), tcom=tcom, root=r.data.xpos[r.root][:2].copy(),
                            facing=math.atan2(Rm[1, 1], Rm[0, 1]),                       # body +y in the world (render_forces.Arm.draw)
                            feet=np.array([r.data.xpos[b][:2] for b in r.foot_bid])))
        if frame_ts is not None and tw >= frame_ts[k]:
            fn, cop, grf, com, tcom = r.forces()
            ren.update_scene(r.data, cam)
            frames.append(dict(tw=tw, img=ren.render().copy(), look=np.array(lookat, float), com=com.copy(), tcom=tcom, fnL=fn["L"], fnR=fn["R"]))
            k += 1
    if ren is not None:
        ren.close()
    gc.T_HOLD = 5.0
    return r, log, frames


def pick_cycle(log, freq, ncyc=3.0):
    """window time at which the strip starts. The gait is clock-driven (period exactly 1/freq): align once on the
    peak of the smoothed left-minus-right normal force in the first 1.2 cycles, step by whole periods, and among the
    complete cycles pick the REPRESENTATIVE one -- the cycle whose support-state fractions (none / L / R / D, after
    the CHATTER_MS filter) are closest to the median over all cycles (Ben 2026-09-11: the last cycle of c6
    1.83/120/75/32/20 was the only one with a no-load gap). Ties -> the later cycle."""
    cyc = 1.0 / freq
    t = np.array([e["tw"] for e in log]); fL = np.array([e["fnL"] for e in log]); fR = np.array([e["fnR"] for e in log])
    fs = (len(t) - 1) / (t[-1] - t[0])
    k = max(1, int(round(0.04 * fs))); dom = np.convolve(fL - fR, np.ones(k) / k, mode="same")
    kT = cyc * fs
    p0 = int(np.argmax(dom[:int(round(1.2 * kT))]))
    starts = []
    while p0 + int(round(kT)) <= len(t):
        starts.append(int(round(p0))); p0 += kT
    code = (fL > FN_MIN).astype(int) + 2 * (fR > FN_MIN).astype(int)
    code = filter_states(code, CHATTER_MS, fs) if CHATTER_MS > 0 else code
    fr = np.array([[np.mean(code[s:s + int(round(kT))] == v) for v in range(4)] for s in starts])
    med = np.median(fr, axis=0)
    dist = np.abs(fr - med).sum(1)
    i = int(np.where(dist == dist.min())[0][-1])
    return float(t[starts[i]]), f"representative cycle {i + 1}/{len(starts)}: none/L/R/D {np.round(fr[i] * 100).astype(int).tolist()} %, median {np.round(med * 100).astype(int).tolist()} %"


def view_h(dist):
    return 2.0 * dist * math.tan(math.radians(FOVY) / 2)


def pass1(cfg, cell, mu, nfr):
    """simulate, pick the cycle, travel frame, extent of the start pose along the travel direction"""
    r, log, _ = simulate(cfg, cell, mu)
    if r.fell is not None:
        raise SystemExit(f"{cfg} {cell} fell at {r.fell:.1f} s")
    cyc = 1.0 / cell[0]
    t0, how = pick_cycle(log, cell[0])
    t = np.array([e["tw"] for e in log])
    win = (t >= t0) & (t <= t0 + cyc)
    com = np.array([e["com"][:2] for e in log]); tcom = np.array([e["tcom"][:2] for e in log])
    dxy = com[win][-1] - com[win][0]
    fwd = dxy / max(np.linalg.norm(dxy), 1e-9); left = np.array([-fwd[1], fwd[0]])
    az = math.degrees(math.atan2(fwd[1], fwd[0]))
    facing = np.array([e["facing"] for e in log])[win]
    fac = float(np.angle(np.mean(np.exp(1j * facing))))                                       # body forward, circular mean over the cycle
    dfac = (math.degrees(fac - math.atan2(fwd[1], fwd[0])) + 180) % 360 - 180                  # body facing - travel
    up = np.array([math.cos(fac), math.sin(fac)]) if UP == "body" else fwd                     # image-up axis in the world
    right = np.array([up[1], -up[0]])
    # bounding box of the feet + COM over the cycle in the image frame, relative to the start COM
    i_first = np.where(win)[0][0]
    pts = np.concatenate([np.array([e["feet"] for e in log])[win].reshape(-1, 2), com[win]]) - com[i_first]
    along, across = pts @ up, pts @ right
    rear, front = float(along.min()) - PAD_REAR, float(along.max()) + PAD_FRONT
    wl, wr = float(across.min()) - PAD_REAR, float(across.max()) + PAD_REAR
    need_h = max(front - rear + 2 * MARGIN, (wr - wl + 2 * MARGIN) * H / W)                 # the portrait frame must hold both extents
    print(f"{cfg} {'/'.join(f'{x:g}' for x in cell)}: cycle from tw {t0:.2f} s ({how}), travel {az:.0f} deg, body facing - travel {dfac:+.0f} deg, "
          f"stride {np.linalg.norm(dxy)*100:.1f} cm, box along/across image-up {rear*100:.0f}..{front*100:.0f} / {wl*100:.0f}..{wr*100:.0f} cm", flush=True)
    # lateral shift: perpendicular to the travel direction (no drift over the cycle by construction), each curve relative
    # to ITS OWN cycle mean, so both oscillate about zero. (Against the whole-body mean, a crab-walking gait such as c1
    # mu 0.1 -- 54 deg off its facing, hip_off 40 -- carries the torso's forward lean into the trace as a constant offset;
    # against the body's lateral axis the crab drift appears as a ramp.)
    body_left = -right
    return dict(cfg=cfg, cell=cell, mu=mu, t0=t0, cyc=cyc, az=az, fwd=fwd, left=left, up=up, right=right, facing_minus_travel=float(dfac),
                stride=float(np.linalg.norm(dxy)), rear=rear, front=front, wl=wl, wr=wr, com0=com[i_first].copy(), need_h=need_h,
                frame_ts=[t0 + k * cyc / nfr for k in range(nfr + 1)],
                ph=100.0 * (t[win] - t0) / cyc, fnL=np.array([e["fnL"] for e in log])[win], fnR=np.array([e["fnR"] for e in log])[win],
                fnL_all=np.array([e["fnL"] for e in log]), fnR_all=np.array([e["fnR"] for e in log]), t_all=t.copy(), win=win.copy(),
                com_win=com[win].copy(), tcom_win=tcom[win].copy(),
                com_lat=(com[win] - com[win].mean(0)) @ left * 1000, tcom_lat=(tcom[win] - tcom[win].mean(0)) @ left * 1000)


def pass2(P, dist):
    """render the frames from the fixed camera: image up = P['up'], the rearmost point of the cycle on the bottom edge
    (Ben 2026-09-11: the robot starts at the bottom of the frame), centred sideways on the cycle's bounding box"""
    up, right = P["up"], P["right"]
    centre = P["com0"] + up * (P["rear"] - MARGIN + view_h(dist) / 2) + right * 0.5 * (P["wl"] + P["wr"])
    look = np.array([*centre, 0.0])
    az = math.degrees(math.atan2(up[1], up[0]))
    # `up` must be image-up: check with the projection used for the markers, flip if not
    u0, v0 = mc.project([*P["com0"], 0.0], look, dist, az, EL, W, H)
    u1, v1 = mc.project([*(P["com0"] + up * 0.1), 0.0], look, dist, az, EL, W, H)
    if v1 > v0:
        az = (az + 180.0) % 360.0
        print(f"{P['cfg']}: azimuth flipped so that {UP} forward = image up", flush=True)
    _, _, frames = simulate(P["cfg"], P["cell"], P["mu"], frame_ts=P["frame_ts"], az=az, lookat=look, dist=dist)
    return dict(P, frames=frames, look=look, dist=dist, az_cam=az)


def support_tag(fnL, fnR):
    return "L" if fnL > FN_MIN and fnR <= FN_MIN else "R" if fnR > FN_MIN and fnL <= FN_MIN else "D" if fnL > FN_MIN else "air"


def filtered_support(d, ms):
    """per-window-sample loaded flags (L, R) after the chatter filter on the support state; also a tag lookup by window time"""
    fL, fR = d["fnL_all"], d["fnR_all"]
    code = (fL > FN_MIN).astype(int) + 2 * (fR > FN_MIN).astype(int)   # 0 none, 1 L only, 2 R only, 3 both
    code = filter_states(code, ms, FS) if ms > 0 else code
    L, R = (code == 1) | (code == 3), (code == 2) | (code == 3)
    def tag_at(tw):
        i = int(np.argmin(np.abs(d["t_all"] - tw)))
        return {0: "air", 1: "L", 2: "R", 3: "D"}[int(code[i])]
    return L[d["win"]], R[d["win"]], tag_at


def figure(R, cyc_cfgs, nfr, tag, mu):
    n = nfr + 1
    ncyc = len(cyc_cfgs)
    fw = 7.2 * (0.99 - 0.115) / n                                                      # frame width, in
    heights = [fw * H / W] * len(CFGS) + sum(([0.16, 0.38, 0.95] for _ in cyc_cfgs), [])   # inches; a spacer row before every cycle block
    fig_h = sum(heights) * 1.12 + 0.9
    fig = plt.figure(figsize=(7.2, fig_h))
    gsp = fig.add_gridspec(len(heights), n, height_ratios=heights, hspace=0.12, wspace=0.03, left=0.115, right=0.99, top=0.97, bottom=0.07)
    for i, cfg in enumerate(CFGS):
        d = R[cfg]; kappa, com = style5.CONFIGS[cfg]; col = style5.style_for(kappa, com)["color"]
        for j, fr in enumerate(d["frames"]):
            ax = fig.add_subplot(gsp[i, j])
            ax.imshow(fr["img"])
            for p, c, mk, ms in [(fr["tcom"], "#cc79a7", "^", 6), (fr["com"], "#ffdd44", "*", 9)]:
                u, v = mc.project(p, fr["look"], d["dist"], d["az_cam"], EL, W, H)
                ax.plot(u, v, mk, color=c, ms=ms, mec="black", mew=0.7)
            ax.set_xlim(0, W); ax.set_ylim(H, 0)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            sup = filtered_support(d, CHATTER_MS)[2](fr["tw"]) if "fnL_all" in d else support_tag(fr["fnL"], fr["fnR"])
            ax.text(0.04, 0.95, sup, transform=ax.transAxes, fontsize=7, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            if i == 0:
                ax.set_title(f"{100.0 * j / nfr:.0f} %", fontsize=8, pad=2)
            if i == 0 and j == 0:                                   # 10 cm scale bar on the floor
                px_per_m = H / view_h(d["dist"])
                x0, y0 = 0.08 * W, 0.94 * H
                ax.plot([x0, x0 + 0.10 * px_per_m], [y0, y0], color="black", lw=1.5, solid_capstyle="butt")
                ax.text(x0, y0 - 0.02 * H, "10 cm", fontsize=6.5, va="bottom", ha="left")
            if j == 0:
                ax.set_ylabel(f"{cfg}\nκ = {kappa:g}\nCOM {com:.2f}", fontsize=8, color=col, rotation=0, ha="right", va="center", labelpad=5)
    row = len(CFGS)
    for cfg in cyc_cfgs:
        d = R[cfg]; kappa, com = style5.CONFIGS[cfg]; col = style5.style_for(kappa, com)["color"]
        row += 1                                        # skip the spacer row
        axd = fig.add_subplot(gsp[row, :])
        Lw, Rw = (filtered_support(d, CHATTER_MS)[:2] if "fnL_all" in d else (d["fnL"] > FN_MIN, d["fnR"] > FN_MIN))
        axd.fill_between(d["ph"], 1.05, 1.95, where=Lw, color="#2c3e50", step="mid")
        axd.fill_between(d["ph"], 0.05, 0.95, where=Rw, color="#7f8c8d", step="mid")
        axd.set_yticks([0.5, 1.5]); axd.set_yticklabels(["R", "L"], fontsize=8); axd.set_ylim(0, 2); axd.set_xlim(0, 100); axd.set_xticks([])
        axd.set_ylabel("support", fontsize=8)
        for k in range(nfr + 1):
            axd.axvline(100 * k / nfr, color="white", lw=0.5)
        axd.set_title(f"{cfg}: κ = {kappa:g}, COM {com:.2f}, gait {'/'.join(f'{x:g}' for x in d['cell'])}", fontsize=8, color=col, loc="left", pad=2)
        axl = fig.add_subplot(gsp[row + 1, :])
        axl.plot(d["ph"], d["com_lat"], "-", color="black", lw=1.6, label="whole-body COM")
        axl.plot(d["ph"], d["tcom_lat"], "--", color="0.45", lw=1.3, label="torso COM")
        axl.axhline(0, color="gray", lw=0.5)
        for k in range(nfr + 1):
            axl.axvline(100 * k / nfr, color="gray", lw=0.4, alpha=0.5)
        axl.set_xlim(0, 100); axl.set_ylabel("lateral shift [mm]\n(+ = left of travel)", fontsize=8); axl.grid(alpha=0.3, axis="y")
        if cfg == cyc_cfgs[-1]:
            axl.set_xlabel("gait cycle phase [%]")
        else:
            axl.set_xticklabels([])
        row += 2
    fig.legend(handles=[Line2D([], [], color="black", lw=1.6, label="whole-body COM"), Line2D([], [], color="0.45", lw=1.3, ls="--", label="torso COM"),
                        Line2D([], [], ls="none", marker="*", color="#ffdd44", mec="black", mew=0.7, ms=9, label="whole-body COM (frames)"),
                        Line2D([], [], ls="none", marker="^", color="#cc79a7", mec="black", mew=0.7, ms=6, label="torso COM (frames)")],
               loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.55, 0.0))
    p = os.path.join(OUT, f"duty_strip_{tag}_mu{int(round(mu * 100)):03d}.png")
    fig.savefig(p, dpi=200); plt.close(fig); print("->", p)
    span = view_h(R[CFGS[0]]["dist"]) * 100
    cap = (f"One gait cycle of the champion gait of each configuration at μ = {mu:g}, seen from above by a fixed camera with the robot's "
           f"forward direction upwards; the robot starts at the bottom of the frame and every frame spans {span:.0f} cm of floor from bottom to top, "
           f"so the displacement over the cycle is visible relative to the body and comparable between rows "
           f"({nfr + 1} frames at equal phase steps; star: whole-body COM, triangle: torso COM; the tag gives the support: L, R, D = double, air). "
           f"Below, for {' and '.join(cyc_cfgs)}: the support of each foot over the cycle (loaded = normal force above 2 N; support-state "
           f"segments shorter than {CHATTER_MS:g} ms are treated as contact chatter and merged into their neighbours; both rows empty = neither foot loaded) and the "
           "lateral shift of the whole-body and torso COM perpendicular to the direction of travel, each relative to its own cycle mean, "
           "positive to the left of travel.")
    open(os.path.join(DATA, f"duty_strip_{tag}_mu{int(round(mu * 100)):03d}_caption.txt"), "w").write(cap + "\n")
    notes = "\n".join(f"{c}: gait {'/'.join(f'{x:g}' for x in R[c]['cell'])} (rank {RANK.get(c, '?')}), cycle from tw {R[c]['t0']:.2f} s, travel {R[c]['az']:.0f} deg, "
                      f"body facing - travel {R[c]['facing_minus_travel']:+.0f} deg, stride {R[c]['stride']*100:.1f} cm, camera distance {R[c]['dist']:.2f} m (view height {view_h(R[c]['dist'])*100:.0f} cm), "
                      f"frames: " + ", ".join(f"{100.0 * j / nfr:.0f}% {support_tag(fr['fnL'], fr['fnR'])} (L {fr['fnL']:.0f} / R {fr['fnR']:.0f} N)" for j, fr in enumerate(R[c]["frames"]))
                      for c in CFGS)
    open(os.path.join(DATA, f"duty_strip_{tag}_mu{int(round(mu * 100)):03d}_caption_notes.txt"), "w").write(notes + "\n")


def main():
    global UP, CHATTER_MS
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--nframes", type=int, default=8)
    ap.add_argument("--cycle-cfgs", nargs="*", default=["c1", "c6"], help="configs that also get the support bar + lateral-shift plot")
    ap.add_argument("--cells", nargs="*", default=[], help="override: cfg:freq/phi/leg/hip/off")
    ap.add_argument("--tag", default="grid7")
    ap.add_argument("--replot", action="store_true", help="reuse work/duty_strip_<tag>_<cfg>.pkl (no simulation)")
    ap.add_argument("--rerender", action="store_true", help="reuse the pass-1 results in the caches, redo only the camera pass (6 rollouts)")
    ap.add_argument("--up", default=UP, choices=["body", "travel"], help="image up = body forward (default) or travel direction")
    ap.add_argument("--chatter-ms", type=float, default=CHATTER_MS, help="support-state chatter filter for the bars and frame tags (0 = raw)")
    a = ap.parse_args()
    UP = a.up
    CHATTER_MS = a.chatter_ms
    for o in a.cells:
        c, cell = o.split(":"); CELLS[c] = tuple(float(x) for x in cell.split("/"))
    os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True); os.makedirs(WORK, exist_ok=True)
    BF.BODY_ALPHA = 1.0                                # opaque robot for the top-down view
    R = {}
    caches = {cfg: os.path.join(WORK, f"duty_strip_{a.tag}_mu{int(round(a.mu * 100)):03d}_{cfg}.pkl") for cfg in CFGS}
    if a.replot and all(os.path.exists(c) for c in caches.values()):
        R = {cfg: pickle.load(open(c, "rb")) for cfg, c in caches.items()}
        for d in R.values():
            if "com_win" in d:
                d["com_lat"] = (d["com_win"] - d["com_win"].mean(0)) @ d["left"] * 1000
                d["tcom_lat"] = (d["tcom_win"] - d["tcom_win"].mean(0)) @ d["left"] * 1000
    else:
        if a.rerender and all(os.path.exists(c) for c in caches.values()):
            P = {cfg: pickle.load(open(c, "rb")) for cfg, c in caches.items()}
            for p_ in P.values():                                            # re-apply the current pads
                p_["rear"] = p_["rear"] + (p_.get("pad_rear", PAD_REAR) - PAD_REAR)
                p_["front"] = p_["front"] + (PAD_FRONT - p_.get("pad_front", PAD_FRONT))
                p_["need_h"] = p_["front"] - p_["rear"] + 2 * MARGIN
        else:
            P = {cfg: pass1(cfg, tuple(float(x) for x in CELLS[cfg]), a.mu, a.nframes) for cfg in CFGS}
        for p_ in P.values():
            p_["pad_rear"], p_["pad_front"] = PAD_REAR, PAD_FRONT
        dist = max(p["need_h"] for p in P.values()) / (2 * math.tan(math.radians(FOVY) / 2))
        print(f"camera distance {dist:.2f} m for all rows (view height {view_h(dist)*100:.0f} cm; needed: " +
              ", ".join(f"{c} {p['need_h']*100:.0f} cm" for c, p in P.items()) + ")", flush=True)
        for cfg in CFGS:
            R[cfg] = pass2(P[cfg], dist)
            pickle.dump(R[cfg], open(caches[cfg], "wb"))
    figure(R, a.cycle_cfgs, a.nframes, a.tag, a.mu)


if __name__ == "__main__":
    main()

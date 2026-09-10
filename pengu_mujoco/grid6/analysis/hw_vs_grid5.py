"""hw_vs_grid5.py — GRID-5 (ideal actuators, kappa PID torso) against the GRID-6 hardware-
model sweep (354 deg/s cap, 56 ms torso lag, feedforward torso, hardware CAD models) for
c1 / c2 / c5 / c6, on the two figures Ben asked for: robust-region volume and speed vs mu.

GRID-5 comes through load5 (npz cache), the hardware sweep from the merged hwact CSVs
(HW_DIR, default the PenguMujoco_psc/g6 copy verified against PSC checksums 2026-09-09).

Definitions (they are NOT identical between the two campaigns -- stamped in the footer):
  GRID-5  pass = survived & head_mean > 0.5 & net_fwd > 0.05 (net_fwd = root world-y / t)
          robust = neighbourhood-mean pass_rate >= 0.8 over freq +-2 steps (0.04 Hz) x phi +-10
  HW      pass = HELD stood & some FF lead stood & straight > 0.5 & v_net > 0.05
          (v_net = whole-body COM net displacement / t); HELD-fell cells count as pass = 0,
          cells pruned by hw_mask (GRID-5 black neighbourhood) are absent (NaN)
          robust = neighbourhood-mean pass >= 0.8 over freq +-1 step (0.05 Hz) x phi +-10
  "GRID-5 | hw range" restricts GRID-5 to the hardware sweep's ranges (freq <= 1.70,
  leg_amp <= 130, hip_off 20-40) so the comparison is not dominated by the 1.8-2.0 Hz /
  leg 145-165 champions the cap forbids.

    python grid6/analysis/hw_vs_grid5.py            # both figures + a numbers table
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(ROOT, "grid5", "analysis"))
import load5                                     # noqa: E402
import style5                                    # noqa: E402
import matplotlib.pyplot as plt                  # noqa: E402

HW_DIR = os.environ.get("HW_DIR", os.path.join(os.path.dirname(ROOT), "..", "PenguMujoco_psc", "g6"))
OUT = os.path.join(ROOT, "results", "grid6_hw", "figs")
CFGS = ["c1", "c2", "c5", "c6"]
HW_MU = [0.12, 0.45]
HW_AX = dict(freq=[round(1.20 + 0.05 * k, 2) for k in range(11)], hip_phi=[float(p) for p in range(0, 360, 10)],
             leg_amp=[float(v) for v in range(70, 135, 5)], hip_amp=[12.0, 16.0, 20.0, 24.0, 28.0, 32.0],
             hip_off=[20.0, 25.0, 30.0, 35.0, 40.0])
RANGE = dict(freq=(1.20, 1.70), leg_amp=(70, 130), hip_off=(20, 40))      # hw sweep ranges
THRESH, TOP = 0.8, 20


def hw_file(cfg, mu):
    hits = sorted(glob.glob(os.path.join(HW_DIR, "**", f"hwact_{cfg}_mu{int(round(mu*100)):03d}.csv"), recursive=True))
    if not hits:
        raise FileNotFoundError(f"hwact_{cfg}_mu{mu} under {HW_DIR}")
    return hits[0]


def hw_planes(cfg, mu, mode="ff"):
    """dense (freq, phi, leg, hip, off) planes: pass (0/1, NaN = not evaluated) and v_net"""
    d = pd.read_csv(hw_file(cfg, mu))
    lst = pd.read_csv(os.path.join(ROOT, "results", "grid6_hw", cfg, f"cells_{cfg}_mu{int(round(mu*100)):03d}_r1.csv"))
    shape = tuple(len(HW_AX[k]) for k in HW_AX)
    idx = lambda df: tuple(np.searchsorted(HW_AX[k], df[k].round(2).values) for k in HW_AX)
    pas = np.full(shape, np.nan)
    pas[idx(lst)] = 0.0                                   # evaluated (HELD-fell rows are absent from the CSV)
    v = np.full(shape, np.nan)
    ok = d[f"v_net_{mode}"].notna() & (d[f"fell_{mode}"].isna()) & (d[f"v_net_{mode}"] > 0.05) & (d[f"straight_{mode}"] > 0.5)
    pas[idx(d[ok])] = 1.0
    v[idx(d[ok])] = d.loc[ok, f"v_net_{mode}"].values
    return pas, v


def nbhd_hw(A):
    """mean over freq +-1 (no wrap, edges NaN) x phi +-1 (circular), valid contributors only"""
    fin = np.isfinite(A)
    Az = np.where(fin, A, 0.0)
    Fn = fin.astype(float)
    s = np.zeros(A.shape)
    c = np.zeros(A.shape)
    for df in (-1, 0, 1):
        Xs, Fs = np.zeros_like(Az), np.zeros_like(Fn)
        if df == 0:
            Xs, Fs = Az, Fn
        elif df > 0:
            Xs[df:], Fs[df:] = Az[:-df], Fn[:-df]
        else:
            Xs[:df], Fs[:df] = Az[-df:], Fn[-df:]
        for dp in (-1, 0, 1):
            s += np.roll(Xs, dp, axis=1)
            c += np.roll(Fs, dp, axis=1)
    out = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    out[0], out[-1] = np.nan, np.nan
    return out


def g5_mask(g):
    """boolean (freq, phi, leg, hip, off) mask of the hardware ranges on a GRID-5 grid"""
    ax = g.axes
    m = np.ones(tuple(len(ax[k]) for k in ("freq", "hip_phi", "leg_amp", "hip_amp", "hip_off")), bool)
    f = np.array(ax["freq"]); m &= ((f >= RANGE["freq"][0] - 1e-9) & (f <= RANGE["freq"][1] + 1e-9))[:, None, None, None, None]
    l = np.array(ax["leg_amp"]); m &= ((l >= RANGE["leg_amp"][0]) & (l <= RANGE["leg_amp"][1]))[None, None, :, None, None]
    o = np.array(ax["hip_off"]); m &= ((o >= RANGE["hip_off"][0]) & (o <= RANGE["hip_off"][1]))[None, None, None, None, :]
    return m


def robust_frac(nb, mask=None):
    v = nb if mask is None else np.where(mask, nb, np.nan)
    fin = np.isfinite(v)
    return (float(np.nansum(v >= THRESH) / fin.sum()) if fin.any() else np.nan, int(np.nansum(v >= THRESH)), int(fin.sum()))


def speed_track(pas, v, mask=None):
    elig = (pas > 0) & np.isfinite(v)
    if mask is not None:
        elig &= mask
    vals = np.sort(v[elig])
    if vals.size == 0:
        return np.nan, np.nan, 0
    best = vals[-TOP:]
    return float(best.mean()), float(best[-1]), int(vals.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    grids = {c: load5.load(c, rnd="grid5", verbose=False) for c in CFGS}
    load5.compatible(grids.values())
    mus5 = grids["c1"].axes["mu"]

    rows = []
    R = {}       # (cfg) -> dict of series
    for c in CFGS:
        g = grids[c]
        nb = g.nbhd("pass_rate")
        mask = g5_mask(g)
        ser = dict(g5_rob=[], g5r_rob=[], g5_top=[], g5_champ=[], g5r_top=[], g5r_champ=[],
                   hw_rob=[], hw_top=[], hw_champ=[], pid_top=[], pid_champ=[])
        for m, mu in enumerate(mus5):
            fr, n, den = robust_frac(nb[m]); ser["g5_rob"].append(fr)
            frr, nr, denr = robust_frac(nb[m], mask); ser["g5r_rob"].append(frr)
            t, ch, ne = speed_track(g["pass_rate"][m], g["net_fwd_mean"][m]); ser["g5_top"].append(t); ser["g5_champ"].append(ch)
            tr, chr_, ner = speed_track(g["pass_rate"][m], g["net_fwd_mean"][m], mask); ser["g5r_top"].append(tr); ser["g5r_champ"].append(chr_)
            rows.append((c, "GRID-5", mu, n, den, fr, ne, t, ch))
            rows.append((c, "GRID-5|hw range", mu, nr, denr, frr, ner, tr, chr_))
        for mu in HW_MU:
            pas, v = hw_planes(c, mu, "ff")
            fr, n, den = robust_frac(nbhd_hw(pas)); ser["hw_rob"].append(fr)
            t, ch, ne = speed_track(pas, v); ser["hw_top"].append(t); ser["hw_champ"].append(ch)
            rows.append((c, "HW ff", mu, n, den, fr, ne, t, ch))
            if style5.CONFIGS[c][0] != 0.0:
                pp, vp = hw_planes(c, mu, "pid")
                tp, chp, nep = speed_track(pp, vp); ser["pid_top"].append(tp); ser["pid_champ"].append(chp)
                frp, npid, denp = robust_frac(nbhd_hw(pp))
                rows.append((c, "HW pid", mu, npid, denp, frp, nep, tp, chp))
        R[c] = ser

    tab = pd.DataFrame(rows, columns=["cfg", "source", "mu", "robust_n", "evaluated", "robust_frac", "passers", "top20_mean", "champion"])
    tp = os.path.join(a.out, "hw_vs_grid5_table.csv")
    tab.to_csv(tp, index=False, float_format="%.4f")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"-> {tp}")

    # ---- figure 1: robust region -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for c in CFGS:
        k, com = style5.CONFIGS[c]
        st = style5.style_for(k, com)
        thin = dict(st, lw=1.0, ms=st["ms"] - 4, alpha=0.35)
        ax.plot(mus5, R[c]["g5_rob"], **thin)
        ax.plot(mus5, R[c]["g5r_rob"], **st, label=style5.label_for(c))
        big = dict(st, linestyle="none", ms=st["ms"] + 4, mew=st["mew"] + 0.8, mec="black")
        ax.plot(HW_MU, R[c]["hw_rob"], **big)
    ax.set_xlabel("floor friction μ")
    xt = sorted(set(mus5) | set(HW_MU))
    ax.set_xticks(xt); ax.set_xticklabels([f"{x:g}" for x in xt], rotation=45, fontsize=8)
    ax.set_ylabel("robust fraction of the evaluated grid (per μ)")
    ax.set_title("Robust-region volume: GRID-5 (lines) vs hardware-model sweep (black-edged markers)", fontsize=10.5)
    ax.grid(alpha=0.3)
    style5.legend_two(ax, coms=sorted({style5.CONFIGS[c][1] for c in CFGS}), loc_gait="upper right", loc_com="upper left")
    ax.text(0.5, 0.98, "faint = GRID-5 full grid    solid = GRID-5 within hw ranges (freq ≤ 1.70, leg ≤ 130, off 20–40)\n"
            "black-edged markers = hardware sweep at μ 0.12 / 0.45 (cap 354°/s, 56 ms, feedforward torso, CAD models)",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="0.25",
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
    style5.finish(fig, os.path.join(a.out, "hw_vs_grid5_robust.png"), K="1",
                  tier=f"robust (nbhd pass ≥ {THRESH}; freq ±0.04 G5 / ±0.05 HW, phi ±10)",
                  stat="fraction of evaluated cells", note="HW pass: stood ∧ straight>0.5 ∧ v_net>0.05; G5 pass: survived ∧ head>0.5 ∧ net>0.05")

    # ---- figure 2: speed vs mu ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, key, ttl in zip(axes, ("top", "champ"), (f"top-{TOP} mean (per config, μ)", "champion (best-of-best, one cell)")):
        for c in CFGS:
            k, com = style5.CONFIGS[c]
            st = style5.style_for(k, com)
            thin = dict(st, lw=1.0, ms=st["ms"] - 4, alpha=0.35)
            ax.plot(mus5, R[c][f"g5_{key}"], **thin)
            ax.plot(mus5, R[c][f"g5r_{key}"], **st, label=style5.label_for(c))
            big = dict(st, linestyle="none", ms=st["ms"] + 4, mew=st["mew"] + 0.8, mec="black")
            ax.plot(HW_MU, R[c][f"hw_{key}"], **big)
            if R[c][f"pid_{key}"]:
                ax.plot(HW_MU, R[c][f"pid_{key}"], linestyle="none", marker="x", ms=9, mew=2.0, color=st["color"], alpha=0.8)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("floor friction μ")
        xt = sorted(set(mus5) | set(HW_MU))
        ax.set_xticks(xt); ax.set_xticklabels([f"{x:g}" for x in xt], rotation=45, fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("forward speed [m/s]")
    style5.legend_two(axes[0], coms=sorted({style5.CONFIGS[c][1] for c in CFGS}), loc_gait="upper right", loc_com="upper left")
    axes[1].text(0.5, 0.98, "faint = GRID-5 full grid    solid = GRID-5 within hw ranges\n"
                 "black-edged = HW feedforward torso    × = HW PID torso (κ=2 only, 56 ms in the loop)\n"
                 "GRID-5 speed = net_fwd_mean of the root;  HW speed = v_net of the whole-body COM",
                 transform=axes[1].transAxes, ha="center", va="top", fontsize=7.5, color="0.25",
                 bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
    style5.finish(fig, os.path.join(a.out, "hw_vs_grid5_speed.png"), K="1", tier="pass (per-μ selection)",
                  stat=f"left: mean of top-{TOP}; right: best-of-best",
                  note="HW pass = stood ∧ straight>0.5 ∧ v_net>0.05; GRID-5 pass = pass_rate>0")
    bw = os.path.join(a.out, "bw")
    if os.path.isdir(bw):
        import shutil
        shutil.rmtree(bw)                          # Ben 2026-09-09: no greyscale twins


if __name__ == "__main__":
    main()

"""gait_cycle7.py — aggregate gait-cycle support plot (Ben 2026-09-11): for c1 vs c6 at one friction, the support
state (double / single / none) as a function of gait-cycle phase, aggregated over ALL cycles of the 8 s rollout
(fraction of cycles in each state per phase bin), after a contact-chatter filter: loaded = Fn > 2 N per foot, the
support state (0 / 1 / 2 feet loaded) is formed per sample, and every state segment shorter than MIN_MS is merged into
the longer of its two neighbours (so the filter never creates a state that was not there). Cycles are aligned on the peak
of the smoothed left-minus-right normal force (same rule as duty_strip7). Three filter lengths side by side (20 / 30 /
40 ms) so the choice can be made on the figure; numbers (mean +- SD over cycles) go to a csv.

    PENGU_MODEL=1.31 python grid7/gait_cycle7.py --mu 0.1            # cells default to the duty-strip champions
-> results/grid7_report/paper/gait_cycle_grid7_mu010.png, data/gait_cycle_grid7_mu010.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "grid6", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "grid5", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "grid6"))
import noflight_search as NF                     # noqa: E402
import stance_com as sc                          # noqa: E402
import style5                                    # noqa: E402
import matplotlib                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402
from matplotlib.patches import Patch             # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})
OUT = os.path.join(ROOT, "results", "grid7_report", "paper")
DATA = os.path.join(ROOT, "results", "grid7_report", "data")
CELLS = {"c1": (1.5, 220, 110, 32, 40), "c6": (1.6, 300, 95, 28, 20)}     # duty-strip champions at mu 0.1
STEADY_S = 7.0
NBIN = 100
COL = {"double": "#2c3e50", "single": "#95a5a6", "none": "white"}


def runs(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def state_runs(n):
    out, i = [], 0
    while i < len(n):
        j = i
        while j < len(n) and n[j] == n[i]:
            j += 1
        out.append([i, j, n[i]]); i = j
    return out


def filter_states(n, ms, fs):
    """merge every state segment shorter than ms into the longer of its neighbours, shortest segments first, until
    none is left (contact chatter); returns the filtered state sequence"""
    k = int(round(ms / 1000 * fs)); n = n.copy()
    while True:
        segs = state_runs(n)
        short = [s for s in segs[1:-1] if s[1] - s[0] < k]           # keep the window edges as they are
        if not short:
            return n
        i0, j0, _ = min(short, key=lambda s: s[1] - s[0])
        idx = [q for q, s in enumerate(segs) if s[0] == i0][0]
        prev, nxt = segs[idx - 1], segs[idx + 1]
        n[i0:j0] = prev[2] if (prev[1] - prev[0]) >= (nxt[1] - nxt[0]) else nxt[2]


def cycle_starts(t, fL, fR, freq, fs):
    """the gait is clock-driven, so the period is exactly 1/freq: align once on the peak of the 40 ms-smoothed
    left-minus-right force inside the first 1.2 cycles, then step by exactly one period"""
    k = max(1, int(round(0.04 * fs))); dom = np.convolve(fL - fR, np.ones(k) / k, mode="same")
    T = 1.0 / freq; kT = T * fs
    p0 = int(np.argmax(dom[:int(round(1.2 * kT))]))
    starts = []
    while p0 + int(round(kT)) <= len(t):
        starts.append(int(round(p0))); p0 += kT
    return starts


def analyse(cfg, cell, mu, ms):
    r = NF.cached_rollout(cfg, tuple(float(x) for x in cell), mu, "grid7")
    t = r["t"] - r["t"][0]; w = t >= t[-1] - STEADY_S
    t, fn = t[w], r["fn"][w]
    fs = (len(t) - 1) / (t[-1] - t[0])                 # the rollout logs at ~167 Hz in practice (nominal 200), measure it
    L0, R0 = fn[:, 0] > sc.FN_MIN, fn[:, 1] > sc.FN_MIN
    code = L0.astype(int) + 2 * R0.astype(int)                       # 0 none, 1 L only, 2 R only, 3 both
    code = filter_states(code, ms, fs) if ms > 0 else code
    L, R = (code == 1) | (code == 3), (code == 2) | (code == 3)
    n = L.astype(int) + R.astype(int)
    T = 1.0 / cell[0]
    states = []; per = []
    kT = int(round(T * fs))
    for s in cycle_starts(t, fn[:, 0], fn[:, 1], cell[0], fs):
        idx = s + np.round(np.arange(NBIN) / NBIN * T * fs).astype(int)
        idx = idx[idx < len(t)]
        if len(idx) < NBIN:
            continue
        states.append(n[idx]); seg = n[s:s + kT]
        per.append(dict(double=np.mean(seg == 2) * 100, single=np.mean(seg == 1) * 100, none=np.mean(seg == 0) * 100,
                        dfL=L[s:s + kT].mean(), dfR=R[s:s + kT].mean()))
    S = np.array(states); P = pd.DataFrame(per)
    frac = {k: (S == v).mean(0) for k, v in (("double", 2), ("single", 1), ("none", 0))}
    return frac, P, len(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--cells", nargs="*", default=[], help="override: cfg:freq/phi/leg/hip/off")
    ap.add_argument("--ms", nargs="*", type=float, default=[20, 30, 40])
    ap.add_argument("--tag", default="grid7")
    a = ap.parse_args()
    for o in a.cells:
        c, cell = o.split(":"); CELLS[c] = tuple(float(x) for x in cell.split("/"))
    cfgs = list(CELLS)
    fig, axs = plt.subplots(len(cfgs), len(a.ms), figsize=(7.2, 1.9 * len(cfgs) + 0.9), sharex=True, sharey=True, squeeze=False)
    rows = []
    ph = (np.arange(NBIN) + 0.5) / NBIN * 100
    for i, cfg in enumerate(cfgs):
        kappa, com = style5.CONFIGS[cfg]; col = style5.style_for(kappa, com)["color"]
        for j, ms in enumerate(a.ms):
            frac, P, ncyc = analyse(cfg, CELLS[cfg], a.mu, ms)
            ax = axs[i][j]
            ax.fill_between(ph, 0, frac["double"], color=COL["double"], step="mid", lw=0)
            ax.fill_between(ph, frac["double"], frac["double"] + frac["single"], color=COL["single"], step="mid", lw=0)
            ax.set_xlim(0, 100); ax.set_ylim(0, 1)
            txt = (f"D {P.double.mean():.0f}±{P.double.std():.0f}  S {P.single.mean():.0f}±{P.single.std():.0f}  N {P.none.mean():.0f}±{P.none.std():.0f} %\n"
                   f"DF L {P.dfL.mean():.2f}  R {P.dfR.mean():.2f}  n = {ncyc}")
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=6.5, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
            if i == 0:
                ax.set_title(f"chatter filter {ms:g} ms", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{cfg}: κ = {kappa:g}, COM {com:.2f}\nfraction of cycles", color=col)
            if i == len(cfgs) - 1:
                ax.set_xlabel("gait cycle phase [%]")
            rows.append(dict(cfg=cfg, mu=a.mu, cell="/".join(f"{x:g}" for x in CELLS[cfg]), filter_ms=ms, n_cycles=ncyc,
                             double_mean=P.double.mean(), double_sd=P.double.std(), single_mean=P.single.mean(), single_sd=P.single.std(),
                             none_mean=P.none.mean(), none_sd=P.none.std(), dfL=P.dfL.mean(), dfL_sd=P.dfL.std(), dfR=P.dfR.mean(), dfR_sd=P.dfR.std()))
            print(f"{cfg} {ms:g} ms: n {ncyc}  double {P.double.mean():.0f}±{P.double.std():.0f}  single {P.single.mean():.0f}±{P.single.std():.0f}  none {P.none.mean():.0f}±{P.none.std():.0f}  DF L {P.dfL.mean():.2f} R {P.dfR.mean():.2f}", flush=True)
    fig.legend(handles=[Patch(color=COL["double"], label="double support"), Patch(color=COL["single"], label="single support"),
                        Patch(facecolor="white", edgecolor="0.6", label="no foot loaded")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Support state over the gait cycle at μ = {a.mu:g}, all cycles of one rollout (last {STEADY_S:g} s)", fontsize=10)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.88, bottom=0.2, hspace=0.12, wspace=0.06)
    os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    p = os.path.join(OUT, f"gait_cycle_{a.tag}_mu{int(round(a.mu * 100)):03d}.png"); fig.savefig(p, dpi=200); print("->", p)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, f"gait_cycle_{a.tag}_mu{int(round(a.mu * 100)):03d}.csv"), index=False)


if __name__ == "__main__":
    main()

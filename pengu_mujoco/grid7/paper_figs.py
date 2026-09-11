"""paper_figs.py (GRID-7) — the two experimental-setup figures, from the GRID-7 sweep (354 deg/s velocity cap on
legs AND torso, +-4.1 N.m torque cap, kappa PID torso, hardware CAD models, mu 0.1 / 0.3 / 0.5 / 0.7, full grid):

  robust_table.png   robust walkers per (config, mu) as a colour-coded table; the caption states
                     the neighbourhood rule (freq +-0.05 x hip_phi +-10, circular in phase, 9 cells,
                     mean pass >= 0.8, valid contributors only, edge frequencies excluded)
  speed_top5.png     forward speed, mean of the fastest 5 % of passing cells per (config, mu), SD
                     whiskers, y-axis cut at 0.3 m/s (values above are printed at the cut)

Plot law as everywhere: blue solid = kappa 0, red hatched = kappa 2, marker by COM ratio.
    python grid6/analysis/paper_figs.py [--max-limit-frac 0.75 --bar-margin 1]   -> results/grid6_report/paper/
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(ROOT, "grid5", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "grid6", "analysis"))
import hw_vs_grid5 as H                          # noqa: E402
import hw_compare as C                           # noqa: E402
import style5                                    # noqa: E402
import matplotlib.pyplot as plt                  # noqa: E402
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = os.path.join(H.ROOT, "results", "grid7_report", "paper")
WORK = os.path.join(H.ROOT, "results", "grid7_report", "work")
CFGS = ["c1", "c2", "c3", "c4", "c5", "c6"]
MUS = [0.1, 0.3, 0.5, 0.7]
PREFIX = "hwcapt"
YCUT = None                                      # None: no cut, axis to the largest mean+SD rounded up to 0.05
TOP_FRAC = 0.05


def collect(mode, max_limit, margin):
    D = {}
    for c in CFGS:
        for mu in MUS:
            pas, v = H.hw_planes(c, mu, mode, PREFIX)
            mk = C.demand_mask(c, mu, mode, None, None, max_limit, margin, PREFIX)
            nb = H.nbhd_hw(pas)
            fr, n, den = H.robust_frac(nb, mk)
            elig = (pas > 0) & np.isfinite(v)
            if mk is not None:
                elig &= mk
            vals = np.sort(v[elig])
            k = max(1, int(round(TOP_FRAC * vals.size))) if vals.size else 0
            top = vals[-k:] if k else np.array([])
            D[(c, mu)] = dict(rob_n=n, rob_den=den, rob_frac=fr, n_pass=int(vals.size),
                              top_mean=float(top.mean()) if k else np.nan, top_sd=float(top.std()) if k else np.nan, top_n=k,
                              evaluated=int(np.isfinite(pas).sum()))
    return D


def robust_table(D, tag, note):
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.axis("off")
    cmap = LinearSegmentedColormap.from_list("seq", ["#f7fbff", "#08306b"])   # one hue, light -> dark
    vmax = max(D[(c, mu)]["rob_frac"] for c in CFGS for mu in MUS)
    cell_text, cell_col = [], []
    for c in CFGS:
        k, com = style5.CONFIGS[c]
        row, cols = [], []
        for mu in MUS:
            r = D[(c, mu)]
            row.append(f"{r['rob_n']:,}  ({r['rob_frac']*100:.1f}%)")
            cols.append(cmap(0.15 + 0.85 * r["rob_frac"] / vmax))
        cell_text.append(row); cell_col.append(cols)
    rows = [f"{c}   κ = {style5.CONFIGS[c][0]:g},  COM {style5.CONFIGS[c][1]:.2f}" for c in CFGS]
    tb = ax.table(cellText=cell_text, cellColours=cell_col, rowLabels=rows,
                  colLabels=[f"μ = {mu:g}" for mu in MUS], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1.0, 2.1)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor("white"); cell.set_linewidth(2)
        if i > 0 and j >= 0:
            rgb = cell.get_facecolor()[:3]
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            cell.get_text().set_color("white" if lum < 0.5 else "#1a1a1a")
        if i == 0 or j == -1:
            cell.set_facecolor("#ffffff"); cell.get_text().set_fontweight("bold")
    ax.set_title("Robust walkers per configuration and floor friction\n"
                 "robust cells (share of the 154,440 cells)", fontsize=11)
    fig.subplots_adjust(top=0.85, bottom=0.06, left=0.26, right=0.98)
    p = os.path.join(OUT, f"robust_table{tag}.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print("wrote", p)


def speed_fig(D, tag, note):
    """one line per config across mu (style5 law), top-5 % mean with SD whiskers, axis cut at YCUT"""
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ycut = YCUT if YCUT is not None else float(np.ceil(max(D[(c, mu)]["top_mean"] + D[(c, mu)]["top_sd"] for c in CFGS for mu in MUS) / 0.05) * 0.05)
    n_over = {mu: 0 for mu in MUS}
    for c in CFGS:
        k, com = style5.CONFIGS[c]
        ys = np.array([D[(c, mu)]["top_mean"] for mu in MUS]); es = np.array([D[(c, mu)]["top_sd"] for mu in MUS])
        st = style5.style_for(k, com)
        ax.errorbar(MUS, np.minimum(ys, ycut), yerr=np.where(ys > ycut, 0, es), capsize=3, elinewidth=1.0, label=style5.label_for(c), **st)
        for j, (mu, y, e) in enumerate(zip(MUS, ys, es)):
            if y > ycut:                                   # off the cut: print the value at the edge, stacked per mu
                first, last = j == 0, j == len(MUS) - 1
                ax.annotate(f"{c} ▲ {y:.2f} ± {e:.2f}", (mu, ycut), xytext=(8 if first else (-8 if last else 0), -6 - 12 * n_over[mu]),
                            textcoords="offset points", ha="left" if first else ("right" if last else "center"), va="top", fontsize=7.5,
                            color=st["color"], bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
                n_over[mu] += 1
    ax.set_ylim(0, ycut); ax.set_xticks(MUS); ax.set_xlim(MUS[0] - 0.06, MUS[-1] + 0.06)
    ax.set_xlabel("floor friction μ"); ax.set_ylabel("forward speed [m/s]")
    ax.grid(alpha=0.3)
    ax.set_title("Forward speed by configuration and floor friction\n"
                 f"mean ± SD of the fastest {TOP_FRAC*100:.0f} % of passing cells per (configuration, μ)" + (f"; axis cut at {YCUT} m/s" if YCUT is not None else ""), fontsize=10)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=style5.GAIT[k]["color"], ls=style5.GAIT[k]["ls"], lw=1.9, label=style5.GAIT[k]["name"]) for k in (0.0, 2.0)]
    hs += [Line2D([], [], color="0.35", ls="none", marker=style5.COM_MK[cm], ms=style5.MS - 2, mfc="none", mec="0.35", mew=style5.MEW, label=f"COM {cm:.2f}")
           for cm in sorted({style5.CONFIGS[c][1] for c in CFGS})]
    fig.legend(handles=hs, fontsize=8, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.01), frameon=False)
    fig.subplots_adjust(top=0.88, bottom=0.16, left=0.11, right=0.97)
    p = os.path.join(OUT, f"speed_top5{tag}.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pid", choices=["ff", "pid"])
    ap.add_argument("--max-limit-frac", type=float, default=None)
    ap.add_argument("--bar-margin", type=int, default=0)
    ap.add_argument("--replot", action="store_true", help="redraw from the cached numbers, no CSV pass")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = (f"_limit{a.max_limit_frac*100:.0f}" + (f"m{a.bar_margin}" if a.bar_margin else "")) if a.max_limit_frac is not None else ""
    note = (f" Motor bar: servo at the cap ≤ {a.max_limit_frac*100:.0f} % of the cycle (+{a.bar_margin}-cell rescue margin)." if a.max_limit_frac is not None else "")
    import json
    os.makedirs(WORK, exist_ok=True)
    cache = os.path.join(WORK, f"paper_figs_data{tag}.json")
    if a.replot and os.path.exists(cache):
        D = {(k.split("|")[0], float(k.split("|")[1])): v for k, v in json.load(open(cache)).items()}
        print(f"replot from {cache}")
    else:
        D = collect(a.mode, a.max_limit_frac, a.bar_margin)
        json.dump({f"{c}|{mu}": v for (c, mu), v in D.items()}, open(cache, "w"), indent=1)
    for (c, mu), r in D.items():
        print(f"{c} mu{mu}: evaluated {r['evaluated']:6d} pass {r['n_pass']:6d} robust {r['rob_n']:6d} ({r['rob_frac']*100:4.1f}%)  top5% mean {r['top_mean']:.3f} ± {r['top_sd']:.3f} (n={r['top_n']})")
    robust_table(D, tag, note)
    speed_fig(D, tag, note)


if __name__ == "__main__":
    main()

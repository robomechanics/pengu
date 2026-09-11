#!/usr/bin/env python
"""Frozen visual contract for every GRID-5 figure (and GRID-4 re-plots).

Hard rule (Ben, 2026-08-26):
  colour + linestyle = gait (kappa)     -- both encode the SAME variable, so a
                                           black-and-white print loses nothing
  marker shape       = COM ratio

All lines of one gait share the exact colour; marker shape alone separates
COM within a gait (Ben rejected a per-COM shade ramp, 2026-08-26). Markers
are large and kappa=2 markers are hollow so overlapping points stay readable.

Every figure must go out through finish(), which stamps K / tier / mean-vs-best
into the footer and writes a greyscale twin next to the colour PNG. Do not call
plt.savefig() directly from figure scripts.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- gait (kappa) -> colour + linestyle. kappa 1 and the no-torso Gait 3 are
#      reserved so future figures never invent their own slot.
GAIT = {
    0.0:  dict(color="#1f77b4", ls="-",  name="κ=0"),   # Gait 1: blue solid
    2.0:  dict(color="#d62728", ls="--", name="κ=2"),   # Gait 2: red dashed
    1.0:  dict(color="#2ca02c", ls="-.", name="κ=1"),   # reserved
    None: dict(color="#7f7f7f", ls=":",  name="no torso"),  # reserved (Gait 3)
}

# ---- COM ratio -> marker shape (1.60 reserved for a future design point)
COM_MK = {1.05: "o", 1.10: "^", 1.20: "s", 1.31: "D", 1.40: "v", 1.60: "P"}

# config -> (kappa, com_ratio); must match grid5/grid5_sweep.py CONFIGS
CONFIGS = {
    "c1": (0.0, 1.05), "c2": (0.0, 1.20), "c3": (0.0, 1.31),
    "c4": (2.0, 1.05), "c5": (2.0, 1.20), "c6": (2.0, 1.31),
    "c7": (0.0, 1.10), "c8": (0.0, 1.40), "c9": (2.0, 1.10), "c10": (2.0, 1.40),
}
MUS = [0.1, 0.3, 0.5, 0.7]
DPI = 130
MS = 10.0          # marker size — large so overlapping points stay readable
MEW = 1.7          # marker edge width


def style_for(kappa, com):
    """Full kwargs for ax.plot: colour/ls from gait, marker from COM.
    Hollow markers for kappa=2 (extra redundancy + see-through on overlap)."""
    g = GAIT[kappa]
    hollow = (kappa == 2.0)
    return dict(color=g["color"], linestyle=g["ls"], marker=COM_MK[com],
                mfc="none" if hollow else g["color"], mec=g["color"],
                lw=1.9, ms=MS, mew=MEW)


def label_for(cfg):
    k, com = CONFIGS[cfg]
    return f"{cfg} (κ={k:g}, COM {com:.2f})"


def legend_two(ax, kappas=(0.0, 2.0), coms=(1.05, 1.10, 1.20, 1.31, 1.40),
               loc_gait="upper right", loc_com="lower left"):
    """Two separate legends: gait (colour+linestyle) and COM (marker shape).
    Replaces the 10-entry per-config legend; label_for() still names lines."""
    gait_h = [Line2D([], [], color=GAIT[k]["color"], ls=GAIT[k]["ls"], lw=1.9,
                     label=GAIT[k]["name"]) for k in kappas]
    com_h = [Line2D([], [], color="0.35", ls="none", marker=COM_MK[c], ms=MS - 2,
                    mfc="none", mec="0.35", mew=MEW, label=f"COM {c:.2f}")
             for c in coms]
    lg = ax.legend(handles=gait_h, loc=loc_gait, fontsize=8, title="gait",
                   title_fontsize=8)
    ax.add_artist(lg)
    lc = ax.legend(handles=com_h, loc=loc_com, fontsize=8, title="COM ratio",
                   title_fontsize=8, ncol=1)
    return lg, lc


def legend_combined(ax, kappas=(0.0, 2.0),
                    coms=(1.05, 1.10, 1.20, 1.31, 1.40), loc="upper right"):
    """ONE legend box: gait entries (colour+linestyle) then COM entries
    (gray marker shapes). Alternative to legend_two for uncluttered axes."""
    hs = [Line2D([], [], color=GAIT[k]["color"], ls=GAIT[k]["ls"], lw=1.9,
                 label=GAIT[k]["name"]) for k in kappas]
    hs += [Line2D([], [], color="0.35", ls="none", marker=COM_MK[c], ms=MS - 2,
                  mfc="none", mec="0.35", mew=MEW, label=f"COM {c:.2f}")
           for c in coms]
    return ax.legend(handles=hs, loc=loc, fontsize=8, framealpha=0.9)


def gray_twin(png_path):
    """Write a luma greyscale copy of a saved PNG into a sibling bw/ dir."""
    img = plt.imread(png_path)
    luma = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
    bw_dir = os.path.join(os.path.dirname(png_path), "bw")
    os.makedirs(bw_dir, exist_ok=True)
    out = os.path.join(bw_dir, os.path.basename(png_path))
    plt.imsave(out, luma, cmap="gray", vmin=0.0, vmax=1.0)
    return out


def finish(fig, path, K, tier, stat, note="", commit=""):
    """Mandatory exit point: stamp the caption footer (K, tier, mean-vs-best),
    save at dpi=130, and emit the greyscale twin. K/tier/stat are required so
    the house rules cannot be forgotten per figure."""
    foot = f"K={K}   tier: {tier}   {stat}"
    if note:
        foot += f"   {note}"
    if commit:
        foot += f"   [{commit[:9]}]"
    fig.text(0.5, 0.005, foot, ha="center", fontsize=7, color="gray")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout(rect=[0, 0.025, 1, 1])
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    bw = gray_twin(path)
    print(f"wrote {path}\n      {bw}")
    return path


def selftest(out_png):
    """Swatch sheet: all 10 configs on dummy data, incl. a deliberate overlap
    band (right side) to check marker readability where lines meet."""
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for i, (c, (k, com)) in enumerate(CONFIGS.items()):
        ys = [i + 1 + 0.35 * j for j in range(3)] + [6.0]  # converge at mu=0.7
        ax.plot(MUS, ys, **style_for(k, com), label=label_for(c))
    ax.set_title("style contract — all 10 configs (right edge = overlap test)",
                 fontsize=10)
    ax.set_xlabel("floor friction μ"); ax.set_xticks(MUS); ax.grid(alpha=0.3)
    ax.set_ylabel("dummy value")
    legend_two(ax)
    return finish(fig, out_png, K="—", tier="swatch (no data)",
                  stat="style contract sample, all 10 configs")


if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    selftest(os.path.join(_root, "results", "grid5_report", "style_ref",
                          "style_swatch.png"))

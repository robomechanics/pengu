"""walker_scan7.py — the GRID-7 walker set, and the median speed of it (Ben 2026-09-14).

Ben's gate list for this figure, and where each gate can be applied:

  1 heading forward .......... NOT in the sweep CSV -> needs a replay (see below)
  2 no foot-clearance gate ... applied by NOT using clear_ok_pid (sliding feet are allowed)
  3 no hop / no skip ......... NOT in the sweep CSV -> needs a replay
  4 no fall .................. fell_pid.isna()                     (part of `pass`)
  5 robust ................... neighbourhood pass >= 0.8           (hw_vs_grid5.nbhd_hw)
  6 within the rotation-speed limit ... limit_frac_any <= 0.75, grown 1 lattice step
                               (motor_demand.bar_mask; recomputed analytically, free)

So this script applies 2 / 4 / 5 / 6 to the whole GRID-7 grid, writes the surviving cells
per (config, mu), and plots the MEDIAN v_net over them. Gates 1 and 3 are deliberately left
out here -- the surviving list is what a force/heading replay campaign should be run over.

The hardware caps Ben asked about (354 deg/s joint speed, +-4.1 N.m torque) are already in the
GRID-7 physics itself; gate 6 is the separate "does the commanded trajectory sit at the slew
cap" bar, which is what the hardware could not follow.

    python grid7/walker_scan7.py                 # full CSV pass (minutes), writes the cache + cells
    python grid7/walker_scan7.py --replot        # redraw from work/walker_scan_grid7.json
-> paper/<date>/median_speed/, data/walker_scan_grid7.csv, data/walker_cells_grid7/<cfg>_mu###.csv
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(ROOT, "grid5", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "grid6", "analysis"))
import hw_vs_grid5 as H                          # noqa: E402
import motor_demand as M                         # noqa: E402
import style5                                    # noqa: E402
import matplotlib.pyplot as plt                  # noqa: E402

style5.paper_rc()
OUT = os.path.join(H.ROOT, "results", "grid7_report", "paper")
DATA = os.path.join(H.ROOT, "results", "grid7_report", "data")
WORK = os.path.join(H.ROOT, "results", "grid7_report", "work")
DATE = datetime.date.today().isoformat()
CFGS = ["c1", "c2", "c3", "c4", "c5", "c6"]
MUS = [0.1, 0.3, 0.5, 0.7]
PREFIX, MODE = "hwcapt", "pid"
BAR, MARGIN = 0.75, 1


def scan(cfg, mu):
    """one (config, mu): the gate cascade, the median speed of the survivors, and the cells."""
    d = M.robust_clear(cfg, mu, MODE, PREFIX)                 # adds pass / robust / rc / v + demand
    bar = M.bar_mask(d, BAR, MARGIN)
    keep = d["robust"] & bar                                  # gates 2/4/5/6; NO clear_ok (gate 2)
    v = d.loc[keep, "v"].values
    st = dict(evaluated=int(len(d)),
              no_fall=int(d[f"fell_{MODE}"].isna().sum()),
              passing=int(d["pass"].sum()),
              robust=int(d["robust"].sum()),
              robust_bar=int(keep.sum()),
              robust_bar_clear=int((d["rc"] & bar).sum()))    # what the old figures used, for reference
    st.update(v_median=float(np.median(v)) if v.size else np.nan,
              v_q1=float(np.percentile(v, 25)) if v.size else np.nan,
              v_q3=float(np.percentile(v, 75)) if v.size else np.nan,
              v_max=float(v.max()) if v.size else np.nan)
    cells = d.loc[keep, ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "v", "nb", "limit_frac_any"]].copy()
    cells.insert(0, "mu", mu); cells.insert(0, "cfg", cfg)
    return st, cells


def collect():
    S, rows = {}, []
    os.makedirs(os.path.join(DATA, "walker_cells_grid7"), exist_ok=True)
    for c in CFGS:
        for mu in MUS:
            st, cells = scan(c, mu)
            S[f"{c}|{mu}"] = st
            p = os.path.join(DATA, "walker_cells_grid7", f"{c}_mu{int(round(mu * 100)):03d}.csv")
            cells.to_csv(p, index=False)
            rows.append(dict(cfg=c, mu=mu, **st))
            print(f"{c} mu{mu}: evaluated {st['evaluated']:6d} | pass {st['passing']:6d} | robust {st['robust']:6d} | "
                  f"+bar {st['robust_bar']:6d} | median v {st['v_median']:.3f} m/s  -> {os.path.basename(p)}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "walker_scan_grid7.csv"), index=False)
    return S


def median_fig(S, bump=0.0):
    """median v_net of the surviving walkers, one line per config -- same law as speed_top5."""
    style5.paper_rc(bump)
    fig, ax = style5.paper_fig(0.675)
    top = 0.0
    for c in CFGS:
        k, com = style5.CONFIGS[c]
        ys = np.array([S[f"{c}|{mu}"]["v_median"] for mu in MUS])
        lo = np.array([S[f"{c}|{mu}"]["v_q1"] for mu in MUS])
        hi = np.array([S[f"{c}|{mu}"]["v_q3"] for mu in MUS])
        st = style5.paper_style_for(k, com)
        ax.errorbar(MUS, ys, yerr=[ys - lo, hi - ys], capsize=1.5, elinewidth=0.7, capthick=0.7,
                    label=style5.cfg_name(c), **st)
        top = max(top, float(np.nanmax(hi)))
    ymax = float(np.ceil((top + 0.015) * 100.0) / 100.0)
    ax.set_ylim(-0.015, ymax)
    ax.set_yticks(np.arange(0.0, ymax, 0.1))          # 3-char labels, same width as speed_top5
    ax.set_xticks(MUS); ax.set_xlim(MUS[0] - 0.06, MUS[-1] + 0.06)
    ax.set_xlabel("Floor Friction μ"); ax.set_ylabel("Median Speed [m/s]")
    ax.grid(alpha=0.3)
    ax.set_title("Median Speed of Robust Controllers")
    style5.legend_paper(fig, ax, coms=sorted({style5.CONFIGS[c][1] for c in CFGS}))
    d_out = os.path.join(OUT, DATE, "median_speed")
    os.makedirs(d_out, exist_ok=True)
    for f in style5.paper_save(fig, os.path.join(d_out, "median_speed_grid7")):
        print("wrote", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true", help="redraw from the cached numbers, no CSV pass")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    cache = os.path.join(WORK, "walker_scan_grid7.json")
    if a.replot and os.path.exists(cache):
        S = json.load(open(cache)); print(f"replot from {cache}")
    else:
        S = collect()
        json.dump(S, open(cache, "w"), indent=1)
    tot = sum(S[k]["robust_bar"] for k in S)
    print(f"\nsurviving walkers (gates 2/4/5/6, no heading and no hop gate yet): {tot:,} over {len(S)} (config, mu) cells")
    median_fig(S)


if __name__ == "__main__":
    main()

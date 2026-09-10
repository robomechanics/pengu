"""hw_champions.py — the one place champions are picked from the hardware-model sweeps.

Filters, in order: pass (stood ∧ straight > 0.5 ∧ v_net > 0.05), robust (neighbourhood pass ≥ 0.8,
freq ±0.05 × hip_phi ±10), clearance (worst-cycle foot apex ≥ 10 mm), and optionally the motor-demand
bar (motor_demand.py: --max-over-frac on the fraction of the cycle a leg joint is over 354 °/s,
--max-rate-mean on the cycle-mean crank rate in °/s). Ranked on whole-body-COM net speed.

    python grid6/analysis/hw_champions.py --mode pid                       # cap-only table (hwcap_*)
    python grid6/analysis/hw_champions.py --mode ff                        # hardware-layer table (hwact_*)
    python grid6/analysis/hw_champions.py --mode pid --max-over-frac 0.3   # with the bar
-> results/grid6_hw/figs/hw{cap|act}_champions[_over30][_rate354].json, top-N per (config, mu)
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import hw_vs_grid5 as H                          # noqa: E402
import motor_demand as M                         # noqa: E402

KEYS = M.KEYS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pid", choices=["ff", "pid"])
    ap.add_argument("--configs", nargs="*", default=["c1", "c2", "c3", "c4", "c5", "c6"])
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--max-over-frac", type=float, default=None, help="drop cells whose over-cap cycle fraction exceeds this (0..1)")
    ap.add_argument("--max-rate-mean", type=float, default=None, help="drop cells whose cycle-mean crank rate exceeds this [deg/s]")
    ap.add_argument("--max-limit-frac", type=float, default=None, help="drop cells whose leg servos sit AT the cap for more than this fraction of the cycle (0..1)")
    ap.add_argument("--bar-margin", type=int, default=0, help="rescue over-the-bar cells within this many lattice steps of an under-the-bar one (buffer zone, as hw_mask)")
    a = ap.parse_args()
    tag = f"{H.prefix_for(a.mode)}_champions" + (f"_over{a.max_over_frac*100:.0f}" if a.max_over_frac is not None else "") \
        + (f"_rate{a.max_rate_mean:.0f}" if a.max_rate_mean is not None else "") + (f"_limit{a.max_limit_frac*100:.0f}" + (f"m{a.bar_margin}" if a.bar_margin else "") if a.max_limit_frac is not None else "")
    cols = [f"{k}_{a.mode}" for k in ("v_net", "straight", "clear", "rollrms", "axisrms", "sat", "drift")]
    if a.mode == "ff":
        cols = ["A0", "phi0", "best_lead"] + cols
    out = {}
    for c in a.configs:
        for mu in H.HW_MU:
            try:
                d = M.robust_clear(c, mu, a.mode)
            except FileNotFoundError as e:
                print(f"  skip {c} mu{mu}: {e}"); continue
            m = d.rc.copy()
            n0 = int(m.sum())
            if a.max_over_frac is not None:
                m &= d.over_frac_any <= a.max_over_frac + 1e-9
            if a.max_rate_mean is not None:
                m &= d.rate_mean_crank <= a.max_rate_mean
            if a.max_limit_frac is not None:
                m &= M.bar_mask(d, a.max_limit_frac, a.bar_margin)
            top = d[m].sort_values("v", ascending=False).head(a.top)
            print(f"{c} mu{mu}: robust+clear {n0} -> after bar {int(m.sum())};  champion "
                  + (f"{'/'.join(f'{x:g}' for x in top.iloc[0][list(KEYS)])}  v {top.iloc[0].v:.3f}  limit_frac_any {top.iloc[0].limit_frac_any:.2f}  over_frac_any {top.iloc[0].over_frac_any:.2f}  rate_mean_crank {top.iloc[0].rate_mean_crank:.0f}" if len(top) else "NONE"))
            out[f"{c}_mu{mu}"] = top[list(KEYS) + cols + ["nb", "rate_mean_crank", "rate_peak_crank", "over_frac_crank", "over_frac_hip", "over_frac_any", "limit_frac_crank", "limit_frac_hip", "limit_frac_any", "exec_ratio_crank"]].round(3).to_dict("records")
    p = os.path.join(H.OUT, tag + ".json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"-> {p}")


if __name__ == "__main__":
    main()

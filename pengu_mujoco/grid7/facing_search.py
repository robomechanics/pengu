"""facing_search.py — walk down the GRID-7 champion ranking of a config (robust + clearance + motor bar, fastest
first) and report, per cell, the angle between the body's forward direction (root +y, circular mean over the
last 5 s) and the direction of travel (COM displacement over the last 5 s). The first cell that walks and has
|angle| <= --max-angle is the forward-walking champion (Ben 2026-09-11: the c1 mu 0.1 champion walks backwards).

    PENGU_MODEL=pengu1_05_hw_updated python grid7/facing_search.py --config c1 --mus 0.1 --max 30
-> results/grid7_report/data/facing_<cfg>.json, facing_<cfg>_tried.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "grid6", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "grid6"))
import noflight_search as NF                     # noqa: E402
import motor_demand as M                         # noqa: E402
import stance_com as sc                          # noqa: E402

DATA = NF.DATA
STEADY_S = 5.0


def facing_stats(r):
    t = r["t"] - r["t"][0]; w = t >= t[-1] - STEADY_S
    com = r["com"][w, :2]; d = com[-1] - com[0]
    travel = np.arctan2(d[1], d[0])
    fac = np.angle(np.mean(np.exp(1j * r["ryaw"][w])))
    ang = (np.degrees(fac - travel) + 180) % 360 - 180
    fn = r["fn"][w]; n = (fn > sc.FN_MIN).sum(1)
    return dict(facing_minus_travel=float(ang), v_replay=float(np.linalg.norm(d) / STEADY_S), DF=float((fn > sc.FN_MIN).mean()),
                single_pct=float(np.mean(n == 1) * 100), double_pct=float(np.mean(n == 2) * 100), air_pct=float(np.mean(n == 0) * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="c1")
    ap.add_argument("--mus", nargs="*", type=float, default=[0.1])
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--max-angle", type=float, default=45.0)
    ap.add_argument("--bar", type=float, default=0.75)
    ap.add_argument("--margin", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ranks", nargs="*", type=int, default=None, help="probe only these ranks (1-based) of the full ranking")
    ap.add_argument("--phi", nargs=2, type=float, default=None, help="restrict the candidates to hip_phi in [lo, hi] (ranks stay those of the full ranking)")
    ap.add_argument("--per-phi", action="store_true", help="probe the fastest cell of every hip_phi value (36 rollouts): facing vs hip_phi map")
    ap.add_argument("--skip-backward-from", default=None,
                    help="a *_perphi_tried.csv: hip_phi values whose fastest cell walks backwards (|angle| > 120 deg) are skipped while walking the ranking")
    a = ap.parse_args()
    chosen, tried = {}, []
    for mu in a.mus:
        d = M.robust_clear(a.config, mu, "pid", "hwcapt")
        g = d[d.rc & M.bar_mask(d, a.bar, a.margin)].sort_values("v", ascending=False)
        found = None
        g = g.assign(rank=np.arange(1, len(g) + 1))
        print(f"{a.config} mu {mu}: {len(g)} robust+clear+bar cells", flush=True)
        skip = set()
        if a.skip_backward_from:
            pp = pd.read_csv(a.skip_backward_from); pp = pp[pp.mu == mu]
            skip = set(float(x.split("/")[1]) for x, ang in zip(pp.cell, pp.facing_minus_travel) if abs(ang) > 120)
            print(f"{a.config} mu {mu}: skipping hip_phi {sorted(skip)} (fastest cell walks backwards)", flush=True)
        if a.ranks:
            items = [(k, g.iloc[k - 1]) for k in a.ranks if k <= len(g)]
        elif a.per_phi:
            items = [(int(row["rank"]), row) for _, row in g.groupby("hip_phi", sort=True).head(1).sort_values("hip_phi").iterrows()]
        else:
            gg = g[(g.hip_phi >= a.phi[0]) & (g.hip_phi <= a.phi[1])] if a.phi else g
            if skip:
                gg = gg[~gg.hip_phi.isin(skip)]
            items = [(int(row["rank"]), row) for _, row in gg.head(a.max).iterrows()]
        for rank, row in items:
            cell = tuple(float(row[k]) for k in NF.KEYS)
            r = NF.cached_rollout(a.config, cell, mu, "grid7")
            rec = dict(cfg=a.config, mu=mu, rank=rank, cell="/".join(f"{x:g}" for x in cell), v_sweep=float(row.v), fell=None if np.isnan(r["fell"]) else float(r["fell"]))
            if rec["fell"] is None:
                rec.update(facing_stats(r))
                print(f"{a.config} mu {mu}: rank {rank:3d} {rec['cell']:22s} v {row.v:.3f}  facing-travel {rec['facing_minus_travel']:+5.0f} deg  "
                      f"v_replay {rec['v_replay']:.3f}  DF {rec['DF']:.2f}  single/double/air {rec['single_pct']:.0f}/{rec['double_pct']:.0f}/{rec['air_pct']:.0f} %", flush=True)
            else:
                print(f"{a.config} mu {mu}: rank {rank:3d} {rec['cell']:22s} v {row.v:.3f}  FELL {rec['fell']:.1f}s", flush=True)
            tried.append(rec)
            if rec["fell"] is None and abs(rec["facing_minus_travel"]) <= a.max_angle and found is None:
                found = rec
                if not a.ranks and not a.per_phi:
                    break
        chosen[f"{a.config}_mu{mu}"] = found
        print(f"{a.config} mu {mu}: " + (f"FORWARD champion = rank {found['rank']} {found['cell']} v {found['v_sweep']:.3f} (facing-travel {found['facing_minus_travel']:+.0f} deg)"
                                         if found else f"none of the top {a.max} walks within {a.max_angle:g} deg of its facing"), flush=True)
        pd.DataFrame(tried).to_csv(os.path.join(DATA, f"facing_{a.config}{a.tag}_tried.csv"), index=False)
        json.dump(chosen, open(os.path.join(DATA, f"facing_{a.config}{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

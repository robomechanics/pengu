"""hop_height.py — how far off the floor does the robot get during its flight phases, and how hard does
it land? (Ben 2026-09-10: "if the hop is 1-2 mm we can discuss whether it passes.")

Per selected cell (replayed through noflight_search.cached_rollout, so the same physics / cache):
  * flight run = consecutive 200 Hz samples with total Fn == 0 (both feet carry nothing)
  * hop height of a run = max over the run of the LOWER foot's lowest point above the floor, i.e. the gap
    under the foot that is closest to the ground. Lowest point = min z over ALL 4030 vertices of the foot
    mesh (the collision hull's lowest vertex), so a pitched foot (toe / heel down) is handled; loaded feet
    sit at slightly negative height because of the soft-contact penetration.
  * landing peak = max total Fn / weight in the 60 ms after the run ends
Runs are split into short (<= 10 ms, 1-2 samples: the swap instant when one foot has left and the other
has not loaded yet) and long (> 10 ms). Prints per cell and per group: number of runs, duration
(median / max, ms), hop height (min / median / max, mm), landing peak F/W (median), and writes
results/grid7_report/data/hop_height_<tag>.csv (+ _runs.csv with every run).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import noflight_search as NF                     # noqa: E402
import stance_com as sc                          # noqa: E402
import gait_sweep as gs                          # noqa: E402
import mujoco                                    # noqa: E402

DATA = os.path.join(ROOT, "results", "grid7_report", "data")
SHORT_MS = 10.0


foot_vertices = NF.foot_vertices
sole_height = NF.sole_height


def analyse(r, cfg, mu, cell, label):
    fn = r["fn"]; W = float(r["mass"]) * 9.81
    air = fn.sum(1) <= 0
    h = sole_height(r); low = h.min(1)                       # the lower foot
    runs = []
    i = 0
    while i < len(air):
        if not air[i]:
            i += 1; continue
        j = i
        while j < len(air) and air[j]:
            j += 1
        land = fn[j:j + int(0.06 * sc.FS)].sum(1)
        runs.append(dict(t0=r["t"][i] - r["t"][0], dur_ms=(j - i) / sc.FS * 1000, hop_mm=low[i:j].max() * 1000,
                         land_fw=(land.max() / W) if len(land) else np.nan))
        i = j
    d = pd.DataFrame(runs, columns=["t0", "dur_ms", "hop_mm", "land_fw"])
    d["long"] = d.dur_ms > SHORT_MS
    L = fn > sc.FN_MIN
    rec = dict(label=label, cfg=cfg, mu=mu, cell="/".join(f"{x:g}" for x in cell), n_runs=len(d),
               loaded_sole_p50_mm=float(np.median(h[L]) * 1000) if L.any() else np.nan,
               loaded_sole_p95_mm=float(np.percentile(h[L], 95) * 1000) if L.any() else np.nan)
    line = f"{label:22s} {rec['cell']:20s} loaded-foot lowest point p50/p95 {rec['loaded_sole_p50_mm']:+.1f}/{rec['loaded_sole_p95_mm']:+.1f} mm |"
    for name, g in (("long", d[d.long]), ("short", d[~d.long])):
        rec[f"{name}_n"] = len(g)
        if len(g):
            rec.update({f"{name}_dur_med_ms": g.dur_ms.median(), f"{name}_dur_max_ms": g.dur_ms.max(),
                        f"{name}_hop_min_mm": g.hop_mm.min(), f"{name}_hop_med_mm": g.hop_mm.median(), f"{name}_hop_max_mm": g.hop_mm.max(),
                        f"{name}_land_med_fw": g.land_fw.median(), f"{name}_under_2mm": int((g.hop_mm < 2).sum())})
            line += (f" {name} {len(g):3d}: dur {g.dur_ms.median():3.0f}/{g.dur_ms.max():3.0f} ms  hop {g.hop_mm.min():4.1f}/{g.hop_mm.median():4.1f}/{g.hop_mm.max():4.1f} mm"
                     f" (<2 mm {int((g.hop_mm < 2).sum())})  land {g.land_fw.median():.2f} |")
        else:
            line += f" {name}   0 |"
    print(line, flush=True)
    return rec, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c6")
    a = ap.parse_args()
    # (label, physics, cfg, mu, cell)
    cells = [
        ("g7 mu0.1 rank1", "grid7", "c6", 0.1, (1.55, 260, 105, 28, 20)),
        ("g7 mu0.1 rank15000", "grid7", "c6", 0.1, (1.25, 50, 100, 16, 35)),
        ("g5 mu0.1 rank1", "grid5", "c6", 0.1, (1.98, 290, 85, 28, 20)),
        ("g5 mu0.1 rank40000", "grid5", "c6", 0.1, (1.3, 350, 115, 12, 30)),
        ("g7 mu0.3 rank1", "grid7", "c6", 0.3, (1.65, 120, 100, 32, 20)),
        ("g7 mu0.3 rank39", "grid7", "c6", 0.3, (1.65, 180, 75, 32, 20)),
        ("g7 mu0.5 rank1", "grid7", "c6", 0.5, (1.65, 180, 80, 32, 20)),
        ("g7 mu0.5 rank16", "grid7", "c6", 0.5, (1.6, 180, 85, 32, 20)),
        ("g7 mu0.7 rank1", "grid7", "c6", 0.7, (1.65, 180, 80, 32, 25)),
    ]
    recs, allruns = [], []
    for label, phys, cfg, mu, cell in cells:
        r = NF.cached_rollout(cfg, tuple(float(x) for x in cell), mu, phys)
        if not np.isnan(r["fell"]):
            print(f"{label}: FELL {float(r['fell']):.1f}s"); continue
        rec, d = analyse(r, cfg, mu, cell, label)
        recs.append(rec); d["label"] = label; allruns.append(d)
    pd.DataFrame(recs).to_csv(os.path.join(DATA, f"hop_height_{a.tag}.csv"), index=False)
    pd.concat(allruns).to_csv(os.path.join(DATA, f"hop_height_{a.tag}_runs.csv"), index=False)
    print("->", os.path.join(DATA, f"hop_height_{a.tag}.csv"))


if __name__ == "__main__":
    main()

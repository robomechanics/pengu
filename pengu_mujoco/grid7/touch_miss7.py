"""touch_miss7.py — GRID-7: the COM-to-GRF-line distance for the champion gaits (hwcapt: 354 deg/s
cap on legs and torso, kappa PID, hardware CAD models), mu 0.1 / 0.3 / 0.5 / 0.7, two statistics
from the same rollout (Ben 2026-09-10):
  touchdown : per gait cycle, the value at the FIRST touchdown of the cycle (a foot's Fn crossing FN_MIN)
  single    : per gait cycle, the mean over the frames in which exactly one foot is loaded
One value per cycle; mean and SD over the cycles of one 8 s rollout. Champion only; if the champion
falls in this replay (a K=1 cell on the fall boundary can flip between PSC and the Mac) take rank 2,
then rank 3 -- no perturbation.

One model per process (gait_config binds PENGU_MODEL at import):
    PENGU_MODEL=pengu1_05_hw_updated python grid7/touch_miss7.py --configs c1 c4
    PENGU_MODEL=pengu1_20_hw_updated python grid7/touch_miss7.py --configs c2 c5
    PENGU_MODEL=1.31                 python grid7/touch_miss7.py --configs c3 c6
    python grid7/touch_miss7.py --merge
-> results/grid7_report/data/touch_miss_grid7_{touchdown,single}_<cfg>.csv and ..._summary_{touchdown,single}.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(ROOT, "grid6", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "grid6"))
sys.path.append(ROOT)
OUT = os.path.join(ROOT, "results", "grid7_report", "data")
CHAMPS = os.path.join(ROOT, "results", "grid6_hw", "figs", "hwcapt_champions_limit75m1.json")
MODEL_OF = {"c1": "pengu1_05_hw_updated", "c4": "pengu1_05_hw_updated", "c2": "pengu1_20_hw_updated",
            "c5": "pengu1_20_hw_updated", "c3": "1.31", "c6": "1.31"}
KAPPA = {"c1": 0.0, "c2": 0.0, "c3": 0.0, "c4": 2.0, "c5": 2.0, "c6": 2.0}
MUS = (0.1, 0.3, 0.5, 0.7)
CFGS = ("c1", "c2", "c3", "c4", "c5", "c6")


def per_cycle(r, freq):
    """returns (touchdown rows, single-support rows), one entry per gait cycle each"""
    import stance_com as sc
    fn, t = r["fn"], r["t"]
    bm = sc.balance_metrics(r)
    loaded = fn > sc.FN_MIN
    td = np.zeros(len(t), bool)
    td[1:] = (loaded[1:] & ~loaded[:-1]).any(1)          # a foot lands at this sample
    single = loaded.sum(1) == 1
    fin = np.isfinite(bm["miss"])
    T = 1.0 / freq
    touch, sing = [], []
    for k in range(int((t[-1] - t[0]) * freq)):
        cyc = (t >= t[0] + k * T) & (t < t[0] + (k + 1) * T)
        m = cyc & td & fin
        idx = np.where(m)[0]
        if len(idx):
            i = idx[0]
            side = "L" if (loaded[i, 0] and not loaded[i - 1, 0]) else "R"
            touch.append(dict(cycle=k, side=side, t=float(t[i]), miss_m=float(bm["miss"][i]),
                              cmp_cop_m=float(np.hypot(*bm["cmp"][i, :2])), M_roll=float(bm["M"][i, 1])))
        m = cyc & single & fin
        if m.sum():
            sing.append(dict(cycle=k, n_frames=int(m.sum()), t=float(t[m][0]), miss_m=float(bm["miss"][m].mean()),
                             cmp_cop_m=float(np.hypot(bm["cmp"][m, 0], bm["cmp"][m, 1]).mean()), M_roll=float(bm["M"][m, 1].mean())))
    return touch, sing


def merge():
    for stat in ("touchdown", "single"):
        parts = [pd.read_csv(os.path.join(OUT, f"touch_miss_grid7_{stat}_{c}.csv")) for c in CFGS
                 if os.path.exists(os.path.join(OUT, f"touch_miss_grid7_{stat}_{c}.csv"))]
        if not parts:
            continue
        ev = pd.concat(parts, ignore_index=True)
        agg = ev.groupby(["cfg", "mu"]).agg(n=("miss_m", "size"), miss_mean=("miss_m", "mean"), miss_std=("miss_m", "std"),
                                            cell=("cell", "first"), rank=("rank", "first"), v_net=("v_net", "first"), fell=("fell", "first")).reset_index()
        agg.to_csv(os.path.join(OUT, f"touch_miss_grid7_summary_{stat}.csv"), index=False)
        print(f"== {stat}")
        print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--override", nargs="*", default=[],
                    help="cfg:mu:freq/phi/leg/hip/off:rank:v_net -- use this cell for (cfg, mu) instead of the champion list "
                         "(Ben 2026-09-10: c6 mu 0.1 = rank 4 1.6/300/95/28/20, the fastest cell with no >= 2 mm hop in the steady phase)")
    a = ap.parse_args()
    OV = {}
    for o in a.override:
        c_, mu_, cell_, rank_, v_ = o.split(":")
        OV[(c_, float(mu_))] = [dict(zip(("freq", "hip_phi", "leg_amp", "hip_amp", "hip_off"), (float(x) for x in cell_.split("/"))),
                                    v_net_pid=float(v_), rank=int(rank_))]
    os.makedirs(OUT, exist_ok=True)
    if a.merge:
        merge()
        return
    import stance_com as sc
    J = json.load(open(CHAMPS))
    for c in a.configs:
        assert os.environ.get("PENGU_MODEL") == MODEL_OF[c], f"{c} needs PENGU_MODEL={MODEL_OF[c]}"
        rows_t, rows_s = [], []
        for mu in MUS:
            cands = OV.get((c, mu)) or J[f"{c}_mu{mu}"]
            for rank, r0 in enumerate(cands, 1):
                rank = r0.get("rank", rank)
                cell = (r0["freq"], r0["hip_phi"], r0["leg_amp"], r0["hip_amp"], r0["hip_off"])
                r = sc.rollout(cell, mu, KAPPA[c], False, cap=True)
                touch, sing = per_cycle(r, cell[0])
                print(f"{c} mu {mu}: rank {rank} {cell} v_net(sweep) {r0['v_net_pid']:.3f}  {'FELL %.1fs' % r['fell'] if r['fell'] else 'walked'}  "
                      f"touchdown cycles {len(touch)} miss {np.mean([e['miss_m'] for e in touch])*100 if touch else float('nan'):.2f} cm | "
                      f"single-support cycles {len(sing)} miss {np.mean([e['miss_m'] for e in sing])*100 if sing else float('nan'):.2f} cm", flush=True)
                if r["fell"] is None:
                    break
            else:
                print(f"{c} mu {mu}: all three fell on this machine -- keeping rank {rank}")
            base = dict(cfg=c, kappa=KAPPA[c], mu=mu, cell="/".join(f"{x:g}" for x in cell), rank=rank, v_net=r0["v_net_pid"], fell=r["fell"])
            rows_t += [dict(base, **e) for e in touch]
            rows_s += [dict(base, **e) for e in sing]
        pd.DataFrame(rows_t).to_csv(os.path.join(OUT, f"touch_miss_grid7_touchdown_{c}.csv"), index=False)
        pd.DataFrame(rows_s).to_csv(os.path.join(OUT, f"touch_miss_grid7_single_{c}.csv"), index=False)


if __name__ == "__main__":
    main()

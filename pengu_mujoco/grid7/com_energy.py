"""com_energy.py — is the champion gait an inverted-pendulum walk or a bounce (grounded running)?
Cavagna-style mechanical-energy analysis of the whole-body COM for the GRID-7 champion of every
(config, mu), replayed through noflight_search.cached_rollout (same physics / cache as the probes).

Per cell, over the last STEADY_S s of the 8 s window (COM at 200 Hz, velocity by central difference
after a 25 ms moving average):
  KE = 1/2 m |v_com|^2 (all three components), PE = m g z_com, E = KE + PE
  r_KE_PE     Pearson correlation of the KE and PE fluctuations: negative = out of phase (pendulum
              exchange, walking), positive = in phase (spring-like, running / grounded running)
  recovery_%  Cavagna (1976) pendular energy recovery = (W_KE + W_PE - W_E) / (W_KE + W_PE) * 100,
              W = sum of the positive increments; ~70 % in a good walk, ~0 in a run
  congruity_% fraction of samples in which dKE and dPE have the same sign (Ahn 2004): high = in phase
  DF          duty factor per foot (Fn > 2 N), mean of L and R
  air_%       fraction of samples with both feet unloaded
Champion = rank 1 of hwcapt_champions_limit75m1.json; falls -> rank 2, 3; overrides via --override
(c6 mu 0.1 -> rank 4 1.6/300/95/28/20, Ben's no-hop rule).

    PENGU_MODEL=pengu1_05_hw_updated python grid7/com_energy.py --configs c1 c4
    PENGU_MODEL=pengu1_20_hw_updated python grid7/com_energy.py --configs c2 c5
    PENGU_MODEL=1.31 python grid7/com_energy.py --configs c3 c6 --override c6:0.1:1.6/300/95/28/20:4
    python grid7/com_energy.py --merge
-> results/grid7_report/data/com_energy_<cfg>.csv, com_energy_summary.csv
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
sys.path.insert(0, os.path.join(ROOT, "grid6"))
import noflight_search as NF                     # noqa: E402
import stance_com as sc                          # noqa: E402

DATA = os.path.join(ROOT, "results", "grid7_report", "data")
CHAMPS = os.path.join(ROOT, "results", "grid6_hw", "figs", "hwcapt_champions_limit75m1.json")
MODEL_OF = {"c1": "pengu1_05_hw_updated", "c4": "pengu1_05_hw_updated", "c2": "pengu1_20_hw_updated", "c5": "pengu1_20_hw_updated", "c3": "1.31", "c6": "1.31"}
MUS = [0.1, 0.3, 0.5, 0.7]
STEADY_S = 6.0
KEYS = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off"]


def energy(r, steady_s=STEADY_S):
    t = r["t"]; com = r["com"]; m = float(r["mass"]); g = 9.81
    k = max(1, int(round(0.025 * sc.FS)))
    cs = np.stack([np.convolve(com[:, j], np.ones(k) / k, mode="same") for j in range(3)], 1)
    v = np.gradient(cs, t, axis=0)
    KE = 0.5 * m * (v ** 2).sum(1); PE = m * g * cs[:, 2]; E = KE + PE
    w = (t - t[0]) >= (t[-1] - t[0]) - steady_s
    w[:k] = False; w[-k:] = False                                   # drop the smoothing edges
    KE, PE, E = KE[w], PE[w], E[w]
    dKE, dPE, dE = np.diff(KE), np.diff(PE), np.diff(E)
    W = lambda d: d[d > 0].sum()
    rec = (W(dKE) + W(dPE) - W(dE)) / (W(dKE) + W(dPE)) * 100
    r_ = float(np.corrcoef(KE, PE)[0, 1])
    cong = float(np.mean(np.sign(dKE) == np.sign(dPE)) * 100)
    fn = r["fn"][w]
    df = float((fn > sc.FN_MIN).mean())
    air = float(((fn > sc.FN_MIN).sum(1) == 0).mean() * 100)
    return dict(r_KE_PE=r_, recovery_pct=float(rec), congruity_pct=cong, DF=df, air_pct=air,
                KE_amp_mJ=float((KE.max() - KE.min()) * 1000), PE_amp_mJ=float((PE.max() - PE.min()) * 1000),
                z_amp_mm=float((cs[w, 2].max() - cs[w, 2].min()) * 1000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--override", nargs="*", default=[], help="cfg:mu:freq/phi/leg/hip/off:rank")
    a = ap.parse_args()
    if a.merge:
        d = pd.concat([pd.read_csv(os.path.join(DATA, f"com_energy_{c}.csv")) for c in MODEL_OF if os.path.exists(os.path.join(DATA, f"com_energy_{c}.csv"))])
        d.to_csv(os.path.join(DATA, "com_energy_summary.csv"), index=False)
        print(d[["cfg", "mu", "rank", "cell", "v_sweep", "DF", "air_pct", "r_KE_PE", "recovery_pct", "congruity_pct", "z_amp_mm"]].round(2).to_string(index=False))
        return
    OV = {}
    for o in a.override:
        c_, mu_, cell_, rank_ = o.split(":")
        OV[(c_, float(mu_))] = (tuple(float(x) for x in cell_.split("/")), int(rank_))
    J = json.load(open(CHAMPS))
    for c in a.configs:
        assert os.environ.get("PENGU_MODEL") == MODEL_OF[c], f"{c} needs PENGU_MODEL={MODEL_OF[c]}"
        rows = []
        for mu in MUS:
            if (c, mu) in OV:
                cands = [(OV[(c, mu)][0], OV[(c, mu)][1], np.nan)]
            else:
                cands = [(tuple(float(r0[k]) for k in KEYS), i, r0["v_net_pid"]) for i, r0 in enumerate(J[f"{c}_mu{mu}"][:3], 1)]
            for cell, rank, v in cands:
                r = NF.cached_rollout(c, cell, mu, "grid7")
                if np.isnan(r["fell"]):
                    break
                print(f"{c} mu {mu}: rank {rank} fell at {float(r['fell']):.1f} s -> next", flush=True)
            rec = dict(cfg=c, mu=mu, rank=rank, cell="/".join(f"{x:g}" for x in cell), v_sweep=v, fell=None if np.isnan(r["fell"]) else float(r["fell"]))
            if rec["fell"] is None:
                rec.update(energy(r))
            print(f"{c} mu {mu}: rank {rank} {rec['cell']}  DF {rec.get('DF', np.nan):.2f}  air {rec.get('air_pct', np.nan):.1f}%  r(KE,PE) {rec.get('r_KE_PE', np.nan):+.2f}  "
                  f"recovery {rec.get('recovery_pct', np.nan):.0f}%  congruity {rec.get('congruity_pct', np.nan):.0f}%  z amp {rec.get('z_amp_mm', np.nan):.1f} mm", flush=True)
            rows.append(rec)
        pd.DataFrame(rows).to_csv(os.path.join(DATA, f"com_energy_{c}.csv"), index=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Champion K=5 checkup (working notes, GRID-4 data): are the K=1 map
champions real, and does the grading system hold up?

For each (config, mu): the T-speed champion and top-20 from the K=1 map,
joined against the authoritative topup-K5 aggregates (physics/topup_k.py
output; see load5.load_topup). Reports K1 -> K5 mean/min, pass_rate, spatial
nbhd, and who the champion becomes under candidate re-ranking rules:
  M1  floor pass_rate >= 0.8, rank by K5 mean
  M2  rank by net_fwd_min (worst seed), gate net_fwd_min > NET_MIN (0.05)
  M4  M1 AND nbhd >= 0.8 (seed-robust AND space-robust)

usage: python grid5/analysis/champ_k5_check.py [--top 20] [--out ...]
"""
import os, sys, argparse, io
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style5, load5

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEF_OUT = os.path.join(_ROOT, "results", "grid5_report", "k5")
NET_MIN = 0.05
CFGS = ["c1", "c2", "c3", "c4", "c5", "c6"]


def cell_key(g, idx, mu):
    ax = g.axes
    return (round(ax["freq"][idx[1]], 2), ax["hip_phi"][idx[2]],
            ax["leg_amp"][idx[3]], ax["hip_amp"][idx[4]],
            ax["hip_off"][idx[5]], mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    md = io.StringIO()
    md.write("# Champion K=5 checkup (GRID-4, working notes)\n\n"
             "K1 = single-seed map value; K5 = topup_k.py aggregate over the "
             "5 protocol seeds.\nM1: floor pass>=0.8, rank K5 mean.  "
             "M2: rank net_fwd_min.  M4: M1 AND nbhd>=0.8.\n")

    md.write("\n## Champions (map T-speed #1 per config, mu)\n\n"
             "| cfg | mu | K1 champ | K5 mean | K5 min | pass | nbhd | "
             "K5/K1 | M1 champ | M2 champ | M4 champ |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|\n")
    per_cfg_top = {}
    for cfg in CFGS:
        try:
            g = load5.load(cfg, rnd="grid4", verbose=False)
        except FileNotFoundError:
            continue
        tp = load5.load_topup(cfg)
        N = g.nbhd("pass_rate")
        for m, mu in enumerate(g.axes["mu"]):
            nf = g["net_fwd_mean"][m]
            mask = (g["pass_rate"][m] > 0) & np.isfinite(nf)
            idxs = np.argwhere(mask)
            order = np.argsort(nf[mask])[::-1][:args.top]
            rows = []
            for oi, r in enumerate(order):
                idx = (m,) + tuple(idxs[r])
                key = cell_key(g, idx, mu)
                t = tp.get(key)
                rows.append(dict(rank=oi + 1, key=key, k1=float(nf[idx[1:]][()]
                            if False else nf[tuple(idxs[r])]),
                            nbhd=float(N[idx]), k5=t))
            per_cfg_top.setdefault(cfg, {})[mu] = rows
            ch = rows[0] if rows else None
            if ch is None:
                continue
            def label(rw):
                if rw is None:
                    return "—"
                k = rw["key"]
                return f"f{k[0]:g}/φ{k[1]:g} ({rw['k5']['net_fwd_mean']:.3f})"
            have = [r for r in rows if r["k5"]]
            m1 = max((r for r in have if r["k5"]["pass_rate"] >= 0.8),
                     key=lambda r: r["k5"]["net_fwd_mean"], default=None)
            m2 = max((r for r in have if r["k5"]["net_fwd_min"] > NET_MIN),
                     key=lambda r: r["k5"]["net_fwd_min"], default=None)
            m4 = max((r for r in have if r["k5"]["pass_rate"] >= 0.8
                      and np.isfinite(r["nbhd"]) and r["nbhd"] >= 0.8),
                     key=lambda r: r["k5"]["net_fwd_mean"], default=None)
            t = ch["k5"]
            if t:
                md.write(f"| {cfg} | {mu} | {ch['k1']:.3f} | "
                         f"{t['net_fwd_mean']:.3f} | {t['net_fwd_min']:.3f} | "
                         f"{t['pass_rate']:.1f} | {ch['nbhd']:.2f} | "
                         f"{t['net_fwd_mean']/ch['k1']:.2f} | "
                         f"{label(m1)} | {label(m2)} | {label(m4)} |\n")
            else:
                md.write(f"| {cfg} | {mu} | {ch['k1']:.3f} | no K5 | | | "
                         f"{ch['nbhd']:.2f} | | | | |\n")

    md.write("\n## Top-20 K5 pass_rate distribution per (config, mu)\n\n"
             "| cfg | mu | n w/ K5 | 5/5 | 4/5 | 3/5 | <=2/5 | "
             "min>0.05 | median K5/K1 |\n|---|---|---|---|---|---|---|---|---|\n")
    for cfg, per_mu in per_cfg_top.items():
        for mu, rows in per_mu.items():
            have = [r for r in rows if r["k5"]]
            if not have:
                md.write(f"| {cfg} | {mu} | 0 (no K5 data) | | | | | | |\n")
                continue
            pr = np.array([r["k5"]["pass_rate"] for r in have])
            mn = np.array([r["k5"]["net_fwd_min"] for r in have])
            infl = np.array([r["k5"]["net_fwd_mean"] / r["k1"] for r in have])
            md.write(f"| {cfg} | {mu} | {len(have)}/{len(rows)} | "
                     f"{int((pr >= 0.999).sum())} | "
                     f"{int(((pr >= 0.8) & (pr < 0.999)).sum())} | "
                     f"{int(((pr >= 0.6) & (pr < 0.8)).sum())} | "
                     f"{int((pr < 0.6).sum())} | {int((mn > NET_MIN).sum())} | "
                     f"{np.median(infl):.2f} |\n")

    out_md = os.path.join(args.out, "champ_k5_table.md")
    open(out_md, "w").write(md.getvalue())
    print(md.getvalue())
    print(f"-> {out_md}")


if __name__ == "__main__":
    main()

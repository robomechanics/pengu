#!/usr/bin/env python
"""Grading-system audit on the one full-passer K=5 dataset (c4, 391,285
rows = every pass>0 cell x 4 mu). Working notes for GRID-5, not catalog
figures.

Questions:
  * how much does a K=1 value inflate vs its K=5 mean? (quantiles + density)
  * does the K=1 top-N ranking survive re-ranking by K=5? (retention/overlap)
  * where would a seed-robustness gate cut? (pass_rate >= t, net_fwd_min >= x
    survival curves over the K1 top-N)
Bias note (stamped on every output): this file is conditioned on the r=0 seed
passing, so pass_rate=0.2 is the floor by construction.

usage: python grid5/analysis/k5_audit_c4.py
"""
import os, sys, io
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style5, load5
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(_ROOT, "results", "grid5_report", "k5")
os.makedirs(OUT, exist_ok=True)
MUS = [0.1, 0.3, 0.5, 0.7]
NET_MIN = 0.05

g = load5.load("c4", rnd="grid4", verbose=False)
tp = load5.load_topup("c4")
md = io.StringIO()
md.write("# c4 full-passer K=5 audit (391,285 rows; working notes)\n\n"
         "Bias: rows exist only where the r=0 seed passed (pass_rate floor "
         "0.2 by construction).\n")

# join arrays per mu: k1 (map value), k5 mean/min, pass_rate
J = {}
for m, mu in enumerate(MUS):
    nf = g["net_fwd_mean"][m]
    mask = (g["pass_rate"][m] > 0) & np.isfinite(nf)
    idxs = np.argwhere(mask)
    k1 = nf[mask]
    ax = g.axes
    k5m = np.full(k1.shape, np.nan); k5n = np.full(k1.shape, np.nan)
    pr = np.full(k1.shape, np.nan)
    for j, x in enumerate(idxs):
        key = (round(ax["freq"][x[0]], 2), ax["hip_phi"][x[1]],
               ax["leg_amp"][x[2]], ax["hip_amp"][x[3]],
               ax["hip_off"][x[4]], mu)
        t = tp.get(key)
        if t:
            k5m[j] = t["net_fwd_mean"]; k5n[j] = t["net_fwd_min"]
            pr[j] = t["pass_rate"]
    got = np.isfinite(k5m)
    J[mu] = dict(k1=k1[got], k5m=k5m[got], k5n=k5n[got], pr=pr[got])
    md.write(f"\nmu={mu}: passers={k1.size}, joined K5={int(got.sum())} "
             f"({100*got.mean():.1f}%)\n")

# ---- inflation quantiles + retention table
md.write("\n## Inflation and ranking survival\n\n"
         "| mu | med K5/K1 (top-100) | K1-champ rank in K5-mean | "
         "top-20 overlap K1 vs K5-mean | top20 pass>=0.8 | top20 pass=1.0 | "
         "top20 min>0.05 |\n|---|---|---|---|---|---|---|\n")
for mu in MUS:
    d = J[mu]
    o1 = np.argsort(d["k1"])[::-1]
    o5 = np.argsort(d["k5m"])[::-1]
    top1 = o1[:20]; top5 = set(o5[:20].tolist())
    champ_rank5 = int(np.where(o5 == o1[0])[0][0]) + 1
    infl = np.median(d["k5m"][o1[:100]] / d["k1"][o1[:100]])
    md.write(f"| {mu} | {infl:.2f} | {champ_rank5} | "
             f"{len(set(top1.tolist()) & top5)}/20 | "
             f"{int((d['pr'][top1] >= 0.8).sum())}/20 | "
             f"{int((d['pr'][top1] >= 0.999).sum())}/20 | "
             f"{int((d['k5n'][top1] > NET_MIN).sum())}/20 |\n")

# whole-population gate stats
md.write("\n## Whole-passer-population gate rates (all joined rows)\n\n"
         "| mu | pass=1.0 | >=0.8 | >=0.6 | min>0.05 | min>0 |\n|---|---|---|---|---|---|\n")
for mu in MUS:
    d = J[mu]; n = d["pr"].size
    md.write(f"| {mu} | {100*(d['pr']>=0.999).mean():.1f}% | "
             f"{100*(d['pr']>=0.8).mean():.1f}% | {100*(d['pr']>=0.6).mean():.1f}% | "
             f"{100*(d['k5n']>NET_MIN).mean():.1f}% | {100*(d['k5n']>0).mean():.1f}% |\n")

# ---- figure 1: K1 vs K5-mean density
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)
for ax, mu in zip(axes, MUS):
    d = J[mu]
    H, xe, ye = np.histogram2d(d["k1"], d["k5m"], bins=80,
                               range=[[0, 0.65], [-0.3, 0.65]])
    ax.imshow(np.log1p(H.T), origin="lower", aspect="auto",
              extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="viridis")
    ax.plot([0, 0.65], [0, 0.65], color="w", lw=0.8, ls="--")
    ax.axhline(0, color="w", lw=0.6, ls=":")
    ax.set_title(f"μ={mu}", fontsize=10); ax.set_xlabel("K1 net_fwd (r=0)")
axes[0].set_ylabel("K5 mean net_fwd")
style5.finish(fig, os.path.join(OUT, "k5_inflation_c4.png"),
              K="1 vs 5", tier="pass (r0-conditioned)",
              stat="per-cell K1 value vs K5 mean, log density; dashed = y=x",
              note="c4 full-passer topup, 391,285 rows")

# ---- figure 2: gate survival curves over K1 top-N
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
mucol = {0.1: "#9ecae1", 0.3: "#4292c6", 0.5: "#2171b5", 0.7: "#084594"}
for mu in MUS:
    d = J[mu]
    top = np.argsort(d["k1"])[::-1][:20]
    ts = [0.2, 0.4, 0.6, 0.8, 1.0]
    axes[0].plot(ts, [np.mean(d["pr"][top] >= t - 1e-9) for t in ts], "o-",
                 color=mucol[mu], label=f"μ={mu}", ms=6)
    xs = np.linspace(0, 0.3, 61)
    axes[1].plot(xs, [np.mean(d["k5n"][top] >= x) for x in xs], "-",
                 color=mucol[mu], label=f"μ={mu}")
axes[0].set_xlabel("gate: K5 pass_rate ≥ t"); axes[0].set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
axes[1].set_xlabel("gate: net_fwd_min ≥ x [m/s]")
axes[1].axvline(NET_MIN, color="gray", ls=":", lw=0.8)
axes[1].annotate("NET_MIN", (NET_MIN, 1.0), fontsize=7, color="gray")
for a in axes:
    a.set_ylabel("surviving fraction of K1 top-20"); a.set_ylim(0, 1.05)
    a.grid(alpha=0.3); a.legend(fontsize=8)
style5.finish(fig, os.path.join(OUT, "k5_gates_c4.png"),
              K=5, tier="K1 top-20 per μ (T-speed)",
              stat="fraction surviving each candidate seed-robustness gate",
              note="c4 only; r0-conditioned")

open(os.path.join(OUT, "k5_audit_c4.md"), "w").write(md.getvalue())
print(md.getvalue())
print(f"-> {OUT}/k5_audit_c4.md")

"""touch_miss.py — COM-to-GRF-line distance at the FIRST INSTANT of each single support,
per gait cycle, for the six GRID-5 configs' per-(config, mu) champions.

Ben 2026-09-09: y = distance from the COM to the GRF vector (line of action through the
combined COP) at the moment a foot becomes the only support -- left and right onsets
pooled -- averaged over the gait cycles of the champion's rollout, with the std across
those events as the error bar; x = floor friction; all six configs, plot law style5.

Champion = fastest ROBUST cell per (config, mu) on the GRID-5 map (pass and neighbourhood
pass >= 0.8, i.e. speed_vs_mu --tier robust): the best-of-best T-speed cells are one grid
step wide and a third of them fell when replayed under stance_com's start protocol
(2026-09-09 first run), which is trap T8 in the flesh. --tier pass restores that ranking. Rollout = stance_com.rollout on the base model with the
config's COM slide (ideal actuators, kappa PID torso, the GRID-5 physics). Single stance
after a 25 ms moving average of Fn; an onset counts if the run lasts >= MIN_RUN_MS.

    python grid6/analysis/touch_miss.py                 # rollouts (cached as pkl) + figure
    python grid6/analysis/touch_miss.py --plot-only
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(ROOT, "grid5", "analysis"))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.append(ROOT)
os.environ.setdefault("PENGU_MODEL", "1.31")
import load5                                     # noqa: E402
import style5                                    # noqa: E402
import stance_com as sc                          # noqa: E402
import matplotlib.pyplot as plt                  # noqa: E402

OUT = os.path.join(ROOT, "results", "grid6_probes", "touch_miss")
CFGS = ["c1", "c2", "c3", "c4", "c5", "c6"]
MIN_RUN_MS = 20.0
MIN_ONSETS = 6


def champions(g, m, n=5, tier="robust", N=None):
    """the n fastest eligible cells at mu index m, fastest first"""
    nf = g["net_fwd_mean"][m]
    elig = (g["pass_rate"][m] > 0) & np.isfinite(nf)
    if tier == "robust":
        elig &= np.isfinite(N[m]) & (N[m] >= 0.8)
    if not elig.any():
        return []
    flat = np.where(elig, nf, -np.inf).ravel()
    order = np.argsort(flat)[::-1][:n]
    keys = ("freq", "hip_phi", "leg_amp", "hip_amp", "hip_off")
    out = []
    for j in order:
        idx = np.unravel_index(j, nf.shape)
        out.append((tuple(float(g.axes[k][i]) for k, i in zip(keys, idx)), float(nf[idx])))
    return out


def onset_events(r):
    """miss distance [m] and CMP-COP [m] at the first sample of every single-stance run"""
    if r["fn"].ndim != 2 or len(r["fn"]) < 10:       # fell before the logging window opened
        return []
    st = sc.classify(r["fn"])
    bm = sc.balance_metrics(r)
    ev = []
    for side in ("L", "R"):
        idx = np.where(st == side)[0]
        if not len(idx):
            continue
        for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if len(run) / sc.FS * 1000.0 < MIN_RUN_MS:
                continue
            i = run[0]
            if np.isfinite(bm["miss"][i]):
                ev.append((side, r["t"][i], bm["miss"][i], np.hypot(*bm["cmp"][i, :2]), bm["M"][i, 1]))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--tier", default="robust", choices=["robust", "pass"])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pk = os.path.join(a.out, f"events_{a.tier}.pkl")
    rows = pickle.load(open(pk, "rb")) if a.plot_only and os.path.exists(pk) else []
    if not rows:
        for c in CFGS:
            kappa, com = style5.CONFIGS[c]
            g = load5.load(c, rnd="grid5", verbose=False)
            N = g.nbhd("pass_rate") if a.tier == "robust" else None
            sc.COM_TARGET = com
            for m, mu in enumerate(g.axes["mu"]):
                # the GRID-5 champion is one K=1 cell under grid5's start protocol; this replay
                # uses stance_com's (2 s settle + 8 s window). If it falls here, or gives fewer
                # than MIN_ONSETS onsets, step down the T-speed ranking (rank recorded).
                tried = []
                for rank, (cell, v) in enumerate(champions(g, m, tier=a.tier, N=N), 1):
                    r = sc.rollout(cell, float(mu), kappa, False)
                    ev = onset_events(r)
                    print(f"{c} mu {mu}: rank {rank} {cell} net_fwd {v:.3f}  {'FELL %.1fs' % r['fell'] if r['fell'] else 'walked'}  "
                          f"onsets {len(ev)}  miss mean {np.mean([e[2] for e in ev])*100 if ev else float('nan'):.2f} cm", flush=True)
                    tried.append((rank, cell, v, r, ev))
                    if r["fell"] is None and len(ev) >= MIN_ONSETS:
                        break
                else:
                    # none reached MIN_ONSETS: keep the walked rollout with the most onsets (a kappa=0
                    # champion that barely leaves double support is a finding, not a failure)
                    ok = [t for t in tried if t[3]["fell"] is None] or tried
                    rank, cell, v, r, ev = max(ok, key=lambda t: len(t[4]))
                    print(f"{c} mu {mu}: no top-5 cell reached {MIN_ONSETS} onsets -- keeping rank {rank} ({len(ev)} onsets)")
                for side, t, miss, dcmp, Mroll in ev:
                    rows.append(dict(cfg=c, kappa=kappa, com=com, mu=float(mu), cell="/".join(f"{x:g}" for x in cell), rank=rank,
                                     net_fwd=v, fell=r["fell"], side=side, t=t, miss_m=miss, cmp_cop_m=dcmp, M_roll=Mroll))
            pickle.dump(rows, open(pk, "wb"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, f"touch_miss_events_{a.tier}.csv"), index=False)
    agg = df.groupby(["cfg", "mu"]).agg(n=("miss_m", "size"), miss_mean=("miss_m", "mean"), miss_std=("miss_m", "std"),
                                        cmp_mean=("cmp_cop_m", "mean"), cmp_std=("cmp_cop_m", "std"), cell=("cell", "first"), rank=("rank", "first"),
                                        net_fwd=("net_fwd", "first"), fell=("fell", "first")).reset_index()
    agg.to_csv(os.path.join(a.out, f"touch_miss_summary_{a.tier}.csv"), index=False)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    mus = sorted(df.mu.unique())
    for c in CFGS:
        kappa, com = style5.CONFIGS[c]
        s = agg[agg.cfg == c].sort_values("mu")
        if s.empty:
            continue
        ax.errorbar(s.mu, s.miss_mean * 100, yerr=s.miss_std * 100, capsize=3, elinewidth=1.0,
                    **style5.style_for(kappa, com), label=style5.label_for(c))
        for _, q in s.iterrows():                     # thin samples and replay fallbacks, marked not hidden
            if q.n < MIN_ONSETS or q["rank"] > 1:
                ax.annotate(f"n={int(q.n)}" + (f" r{int(q['rank'])}" if q["rank"] > 1 else ""), (q.mu, q.miss_mean * 100),
                            fontsize=6, color="gray", xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("floor friction μ"); ax.set_xticks(mus)
    ax.set_ylabel("COM to GRF line at single-support onset [cm]\n(mean ± std over the L/R onsets of one rollout)")
    ax.set_title(f"GRF moment arm at first touch of single support — per-(config, μ) champions, {a.tier} tier", fontsize=10)
    ax.grid(alpha=0.3)
    style5.legend_two(ax, coms=sorted({style5.CONFIGS[c][1] for c in CFGS}), loc_gait="upper right", loc_com="upper left")
    fig.text(0.5, 0.005, f"K=1   champion = fastest {'pass ∧ nbhd≥0.8' if a.tier == 'robust' else 'passing'} cell per (config, μ); GRID-5 physics (ideal, κ PID, COM slide); "
             f"rN = rank-N fallback (higher ranks fell in replay)\nonset = first sample of a single-stance run ≥ {MIN_RUN_MS:.0f} ms (Fn smoothed 25 ms); GRF line through the combined COP",
             ha="center", fontsize=7, color="gray")
    fig.subplots_adjust(bottom=0.16)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    p = os.path.join(a.out, f"touch_miss_vs_mu_{a.tier}.png")
    fig.savefig(p, dpi=style5.DPI)
    print(f"-> {p}")


if __name__ == "__main__":
    main()

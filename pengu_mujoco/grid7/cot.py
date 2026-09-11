"""cot.py — mechanical cost of transport of the GRID-7 champion gaits, replayed on this machine.

COT_net  = E+ / (m g d_net),  E+ = positive mechanical actuator work over the 8 s window
           (sum over the 5 servos of max(tau * qdot, 0) dt, as gait_sweep's e_pos; servos do not regenerate),
           d_net = net horizontal COM displacement over the window
COT_path = E+ / (m g d_path), d_path = horizontal COM path length
Per (config, mu): the top-N cells of the champion ranking (robust + clearance + motor bar, fastest first,
hwcapt_champions_limit75m1.json), rank 1 = champion; cells that fall here are skipped. Overrides as touch_miss7.

    PENGU_MODEL=pengu1_05_hw_updated python grid7/cot.py --configs c1 c4
    PENGU_MODEL=pengu1_20_hw_updated python grid7/cot.py --configs c2 c5
    PENGU_MODEL=1.31 python grid7/cot.py --configs c3 c6 --override c6:0.1:1.6/300/95/28/20:4
    python grid7/cot.py --plot
-> results/grid7_report/data/cot_<cfg>.csv, cot_summary.csv; paper/cot_champions.png (plot law style5)
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
sys.path.insert(0, os.path.join(ROOT, "grid5", "analysis"))
import noflight_search as NF                     # noqa: E402

DATA = os.path.join(ROOT, "results", "grid7_report", "data")
OUT = os.path.join(ROOT, "results", "grid7_report", "paper")
CHAMPS = os.path.join(ROOT, "results", "grid6_hw", "figs", "hwcapt_champions_limit75m1.json")
MODEL_OF = {"c1": "pengu1_05_hw_updated", "c4": "pengu1_05_hw_updated", "c2": "pengu1_20_hw_updated", "c5": "pengu1_20_hw_updated", "c3": "1.31", "c6": "1.31"}
CFGS = ["c1", "c2", "c3", "c4", "c5", "c6"]
MUS = [0.1, 0.3, 0.5, 0.7]
KEYS = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off"]
TOP = 5


def cot_of(r):
    m = float(r["mass"]); g = 9.81
    com = r["com"][:, :2]
    d_net = float(np.linalg.norm(com[-1] - com[0]))
    d_path = float(np.linalg.norm(np.diff(com, axis=0), axis=1).sum())
    E = float(r["e_pos"][-1] - r["e_pos"][0])
    T = float(r["t"][-1] - r["t"][0])
    return dict(E_pos_J=E, d_net_m=d_net, d_path_m=d_path, v_net=d_net / T, power_W=E / T,
                cot_net=E / (m * g * d_net) if d_net > 0.02 else np.nan, cot_path=E / (m * g * d_path) if d_path > 0.02 else np.nan)


def plot():
    """same layout as paper_figs.speed_fig: 7.2 x 5.4 in, style5 law, legend row under the axes"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import style5
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})
    d = pd.read_csv(os.path.join(DATA, "cot_summary.csv"))
    d = d[d.fell.isna()]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    top = 0.0
    for c in CFGS:
        kappa, com = style5.CONFIGS[c]
        g = d[d.cfg == c].groupby("mu")["cot_path"].agg(["mean", "std"]).reindex(MUS)
        ax.errorbar(MUS, g["mean"], yerr=g["std"].fillna(0), capsize=3, elinewidth=1.0, label=style5.label_for(c), **style5.style_for(kappa, com))
        top = max(top, float((g["mean"] + g["std"].fillna(0)).max()))
    ax.set_ylim(0, float(np.ceil(top / 0.25) * 0.25)); ax.set_xticks(MUS); ax.set_xlim(MUS[0] - 0.06, MUS[-1] + 0.06)
    ax.set_xlabel("floor friction μ"); ax.set_ylabel("mechanical cost of transport  E+ / (m g d)")
    ax.grid(alpha=0.3)
    ax.set_title("Mechanical cost of transport by configuration and floor friction\n"
                 f"mean ± SD of the {TOP} fastest robust cells per (configuration, μ)", fontsize=10)
    hs = [Line2D([], [], color=style5.GAIT[k]["color"], ls=style5.GAIT[k]["ls"], lw=1.9, label=style5.GAIT[k]["name"]) for k in (0.0, 2.0)]
    hs += [Line2D([], [], color="0.35", ls="none", marker=style5.COM_MK[cm], ms=style5.MS - 2, mfc="none", mec="0.35", mew=style5.MEW, label=f"COM {cm:.2f}")
           for cm in sorted({style5.CONFIGS[c][1] for c in CFGS})]
    fig.legend(handles=hs, fontsize=8, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.01), frameon=False)
    fig.subplots_adjust(top=0.88, bottom=0.16, left=0.11, right=0.97)
    p = os.path.join(OUT, "cot_champions.png"); fig.savefig(p, dpi=200); plt.close(fig); print("->", p)
    cap = (f"Mechanical cost of transport versus floor friction for the {TOP} fastest robust gaits of each configuration (mean ± SD): "
           "positive actuator work per unit body weight and per unit distance travelled by the whole-body COM. "
           "Positive work only (the servos do not regenerate).")
    open(os.path.join(DATA, "cot_champions_caption.txt"), "w").write(cap + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--override", nargs="*", default=[], help="cfg:mu:freq/phi/leg/hip/off:rank -> replaces rank 1 of that (cfg, mu)")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.merge or a.plot:
        d = pd.concat([pd.read_csv(os.path.join(DATA, f"cot_{c}.csv")) for c in CFGS if os.path.exists(os.path.join(DATA, f"cot_{c}.csv"))])
        d.to_csv(os.path.join(DATA, "cot_summary.csv"), index=False)
        s = d[d.fell.isna()].groupby(["cfg", "mu"]).agg(n=("cot_net", "count"), v=("v_net", "mean"), cot_net=("cot_net", "mean"), cot_net_sd=("cot_net", "std"),
                                                       cot_path=("cot_path", "mean"), power_W=("power_W", "mean")).reset_index()
        print(s.round(3).to_string(index=False))
        if a.plot:
            plot()
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
            cands = [(tuple(float(r0[k]) for k in KEYS), i, r0["v_net_pid"]) for i, r0 in enumerate(J[f"{c}_mu{mu}"][:TOP], 1)]
            if (c, mu) in OV:
                cands = [(OV[(c, mu)][0], OV[(c, mu)][1], np.nan)] + [x for x in cands if x[0] != OV[(c, mu)][0]][:TOP - 1]
            for cell, rank, v in cands:
                r = NF.cached_rollout(c, cell, mu, "grid7")
                rec = dict(cfg=c, mu=mu, rank=rank, cell="/".join(f"{x:g}" for x in cell), v_sweep=v, fell=None if np.isnan(r["fell"]) else float(r["fell"]))
                if rec["fell"] is None:
                    rec.update(cot_of(r))
                print(f"{c} mu {mu}: rank {rank} {rec['cell']}  " + (f"FELL {rec['fell']:.1f}s" if rec["fell"] is not None else
                      f"v {rec['v_net']:.3f}  E+ {rec['E_pos_J']:.2f} J  P {rec['power_W']:.2f} W  COT_net {rec['cot_net']:.2f}  COT_path {rec['cot_path']:.2f}"), flush=True)
                rows.append(rec)
        pd.DataFrame(rows).to_csv(os.path.join(DATA, f"cot_{c}.csv"), index=False)


if __name__ == "__main__":
    main()

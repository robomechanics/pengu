#!/usr/bin/env python
"""F8 fall_phase stack (GRID-5 only): are falls WALKING failures or START
failures? Stacked share of hold / trans / settle / walk falls per
(config, mu), from the fall_phase tally column. Round 1 could not see this
(c6 was 91.7% pre-measurement falls, discovered only by replay probes).

Phases render as a fixed sequential ramp (light->dark = later phase) with
hatches so the stack survives greyscale; config identity stays in the
x labels (the colour/marker contract applies to lines, not stacks).

usage: python grid5/analysis/figs/fall_phase.py [--configs c6 ...]
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import style5, load5
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DEF_OUT = os.path.join(_ROOT, "results", "grid5_report", "style_ref")
PHASES = ["hold", "trans", "settle", "walk"]
PCOL = {"hold": "#d4e6f1", "trans": "#7fb3d5", "settle": "#2e86c1",
        "walk": "#1b4f72"}                      # light->dark = later phase
PHATCH = {"hold": "//", "trans": "..", "settle": "xx", "walk": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()

    cfgs = args.configs or [c for c in style5.CONFIGS
                            if load5._csv_path(c, "grid5")]
    grids = {}
    for c in cfgs:
        try:
            grids[c] = load5.load(c, rnd="grid5")
        except (FileNotFoundError, ValueError) as e:
            print(f"  skip {c}: {e}")
    if not grids:
        sys.exit("no grid5 configs loadable")
    load5.compatible(grids.values())

    mus = next(iter(grids.values())).axes["mu"]
    partial = [c for c, g in grids.items() if not g.complete]
    Ks = {g.K for g in grids.values()}
    commits = {g.commit for g in grids.values() if g.commit}

    # shares per (config, mu): counts of falls by phase / total falls
    labels, shares, totals = [], [], []
    for c, g in grids.items():
        for m, mu in enumerate(mus):
            cnt = np.array([g["nfall_" + ph][m].sum() for ph in PHASES],
                           float)
            tot = cnt.sum()
            labels.append(f"{c}\nμ={mu:g}")
            shares.append(cnt / tot if tot else np.full(4, np.nan))
            totals.append(int(tot))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7.0, 1.05 * len(labels) + 2), 5.0))
    bottom = np.zeros(len(labels))
    for pi, ph in enumerate(PHASES):
        ys = np.array([s[pi] for s in shares])
        ax.bar(x, np.nan_to_num(ys), bottom=bottom, width=0.7,
               color=PCOL[ph], hatch=PHATCH[ph], edgecolor="white",
               lw=0.5, label=ph)
        bottom += np.nan_to_num(ys)
    for xi, (tot, s) in enumerate(zip(totals, shares)):
        if tot == 0:
            ax.annotate("no falls", (xi, 0.5), ha="center", fontsize=7,
                        color="gray", rotation=90)
        else:
            ax.annotate(f"n={tot}", (xi, 1.01), ha="center", fontsize=6.5,
                        color="gray")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.08); ax.set_ylabel("share of falls")
    ax.set_title("fall phase decomposition — start failures vs walking "
                 "failures", fontsize=11)
    ax.legend(fontsize=8, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    ax.grid(alpha=0.25, axis="y")

    note = "phases: hold→trans→settle→walk (light→dark)"
    if partial:
        note += "; PARTIAL " + " ".join(
            f"{c}(hip_off={[int(v) for v in grids[c].present['hip_off']]},"
            f" freq {len(grids[c].present['freq'])}/{len(grids[c].axes['freq'])})"
            for c in partial)
    style5.finish(
        fig, os.path.join(args.out, "fall_phase_grid5.png"),
        K="/".join(str(k) for k in sorted(Ks)),
        tier="all falls (surv_rate < 1 cells)",
        stat="share of fall counts by phase per (config, μ)",
        note=note,
        commit=commits.pop() if len(commits) == 1 else "")


if __name__ == "__main__":
    main()

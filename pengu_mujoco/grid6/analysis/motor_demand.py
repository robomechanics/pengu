"""motor_demand.py — how hard a cell drives the leg servos, from the gait parameters alone.

The leg commands are open-loop (gait_config.compute_gait): crank = 0.5 A (1 + sin 2πft), hip =
A_h max(0, sin(2πft + φ)). So the demanded angular velocity is analytic in (freq, leg_amp, hip_amp):
  crank  rate peak π f A,  cycle-mean |rate| 2 f A,  over-cap fraction (2/π) acos(354 / (π f A))
  hip    rate peak 2π f A_h (active half-cycle only), cycle-mean 2 f A_h, over-cap fraction (1/π) acos(354 / (2π f A_h))
  any    fraction of the cycle where either joint is over, on a 720-sample cycle (needs hip_phi)
The 354 °/s cap is already inside every hardware-layer / cap-only rollout (hw_sweep.py slew clip), so
these numbers say how much of the cycle the simulation spent on the clip -- Ben 2026-09-09 wants a bar
on this so the champion is a gait that does not live on the limit, while cells that only touch it
still pass.

    python grid6/analysis/motor_demand.py --mode pid          # c3 c4 (+ c1 c2 c5 c6 once their cap-only data is in)
    python grid6/analysis/motor_demand.py --mode ff           # c1 c2 c5 c6 hardware-layer table
Prints, per (config, mu), the distribution among pass ∧ robust ∧ clear cells and a threshold table
(champion v_net under each candidate bar); writes results/grid6_hw/figs/motor_demand_<mode>.png/.csv.
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import hw_vs_grid5 as H                          # noqa: E402
import style5                                    # noqa: E402
import matplotlib.pyplot as plt                  # noqa: E402

CAP = 354.0
KEYS = ("freq", "hip_phi", "leg_amp", "hip_amp", "hip_off")
OVER_BARS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)
LIMIT_BARS = (0.0, 0.25, 0.50, 0.75, 0.90)
RATE_BARS = (250.0, 300.0, 354.0, 400.0, 450.0)


def _analytic(freq, leg, hip, phi):
    """deg/s and demand-over-cap cycle fractions for one cell (closed form + a 720-sample cycle)"""
    pk_c, pk_h = math.pi * freq * leg, 2 * math.pi * freq * hip
    out = dict(rate_peak_crank=pk_c, rate_mean_crank=2 * freq * leg,
               rate_peak_hip=pk_h, rate_mean_hip=2 * freq * hip,
               over_frac_crank=(2 / math.pi) * math.acos(CAP / pk_c) if pk_c > CAP else 0.0,
               over_frac_hip=(1 / math.pi) * math.acos(CAP / pk_h) if pk_h > CAP else 0.0)
    th = np.linspace(0, 2 * math.pi, 720, endpoint=False)
    rc = np.abs(pk_c * np.cos(th))
    s = np.sin(th + math.radians(phi))
    rh = np.where(s > 0, np.abs(pk_h * np.cos(th + math.radians(phi))), 0.0)
    rh2 = np.where(-s > 0, np.abs(pk_h * np.cos(th + math.radians(phi))), 0.0)
    out["over_frac_any"] = float(np.mean((rc > CAP) | (rh > CAP) | (rh2 > CAP)))
    out["rate_mean_any"] = float(np.mean(np.maximum(rc, np.maximum(rh, rh2))))
    return out


def demand(freq, leg, hip, phi):
    """one cell: analytic demand + the slew replay (servo AT the cap: the limiter stays engaged while
    the joint catches up after the demand drops under the cap, so this exceeds over_frac -- for
    1.60/260/115 it is 0.90 against 0.58; verified against hw_sweep's clip counter 2026-09-09)"""
    out = _analytic(freq, leg, hip, phi)
    out.update(_slew_replay([freq], [leg], [hip], [phi]).iloc[0].to_dict())
    return out


def add_demand(df):
    """demand columns for a whole table: the analytic part per row, the slew replay vectorised over
    the unique (freq, leg_amp, hip_amp, hip_phi) combinations (hip_off does not enter)"""
    an = pd.DataFrame([_analytic(r.freq, r.leg_amp, r.hip_amp, r.hip_phi) for r in df[list(KEYS)].itertuples()], index=df.index)
    u = df[["freq", "leg_amp", "hip_amp", "hip_phi"]].drop_duplicates().reset_index(drop=True)
    lim = _slew_replay(u.freq.values, u.leg_amp.values, u.hip_amp.values, u.hip_phi.values)
    lim = pd.concat([u, lim], axis=1)
    out = df.merge(lim, on=["freq", "leg_amp", "hip_amp", "hip_phi"], how="left")
    out.index = df.index
    return pd.concat([out, an], axis=1)


def _slew_replay(freq, leg, hip, phi, dt=1e-3, n_cyc=6, meas=3):
    """vectorised over cells: fraction of the measured cycles each leg servo spends AT the cap,
    and the executed stroke ratio. Identical algorithm to hw_sweep.py's per-step clip."""
    freq, leg, hip, phi = (np.asarray(x, float) for x in (freq, leg, hip, phi))
    n = len(freq)
    steps = int(np.ceil(n_cyc / freq.min() / dt))
    held = None
    at = np.zeros((n, 4))
    cnt = np.zeros(n)
    mn = np.full((n, 4), np.inf); mx = np.full((n, 4), -np.inf)
    slew = CAP * dt
    ph = np.radians(phi)
    for i in range(steps):
        t = i * dt
        w = 2 * np.pi * freq * t
        cmd = np.stack([0.5 * leg * (1 + np.sin(w)), 0.5 * leg * (1 + np.sin(w + np.pi)),
                        hip * np.maximum(0.0, np.sin(w + np.pi + ph)), hip * np.maximum(0.0, np.sin(w + ph))], 1)
        if held is None:
            held = cmd.copy()
        d = cmd - held
        over = np.abs(d) > slew * (1 + 1e-6)
        held += np.clip(d, -slew, slew)
        m = (t >= (n_cyc - meas) / freq) & (t < n_cyc / freq)     # per-cell measuring window
        at += over * m[:, None]
        cnt += m
        mn = np.where(m[:, None], np.minimum(mn, held), mn)
        mx = np.where(m[:, None], np.maximum(mx, held), mx)
    cnt = np.maximum(cnt, 1)
    return pd.DataFrame(dict(limit_frac_crank=np.clip((at[:, :2].max(1)) / cnt, 0, 1),
                             limit_frac_hip=np.clip(at[:, 2:].max(1) / cnt, 0, 1),
                             limit_frac_any=np.clip(at.max(1) / cnt, 0, 1),
                             exec_ratio_crank=np.where(leg > 0, (mx[:, 0] - mn[:, 0]) / np.maximum(leg, 1e-9), 1.0),
                             exec_ratio_hip=np.where(hip > 0, (mx[:, 3] - mn[:, 3]) / np.maximum(hip, 1e-9), 1.0)))


def bar_mask(df, bar=0.75, margin=1, col="limit_frac_any"):
    """Ben 2026-09-09: the cells under the bar are the 'bright' set; grow it by `margin` lattice
    steps (freq 0.05, leg_amp 5, hip_amp 4, hip_phi 10 wrapping) INTO the over-the-bar region, the
    way hw_mask.py rescues GRID-5 black cells next to passing ones -- the buffer relaxes the bar
    at its edge, it does not tighten it. hip_off does not enter. Returns a boolean Series, True = keep."""
    ax = {k: np.array(sorted(df[k].unique())) for k in ("freq", "hip_phi", "leg_amp", "hip_amp")}
    idx = tuple(np.searchsorted(ax[k], df[k].values) for k in ax)
    good = np.zeros([len(ax[k]) for k in ax], bool)
    good[idx] = (df[col] <= bar + 1e-9).values
    out = good.copy()
    for a, k in enumerate(ax):
        acc = out.copy()
        for st in range(1, margin + 1):
            if k == "hip_phi":
                acc |= np.roll(out, st, axis=a) | np.roll(out, -st, axis=a)
            else:
                f = np.zeros_like(out); b = np.zeros_like(out)
                sl_f = [slice(None)] * out.ndim; sl_b = [slice(None)] * out.ndim
                sl_f[a], sl_b[a] = slice(st, None), slice(None, -st)
                f[tuple(sl_f)] = out[tuple(sl_b)]; b[tuple(sl_b)] = out[tuple(sl_f)]
                acc |= f | b
        out = acc
    return pd.Series(out[idx], index=df.index)


def robust_clear(cfg, mu, mode):
    """the hw table with pass / robust / clear flags and the demand columns"""
    d = pd.read_csv(H.hw_file(cfg, mu, H.prefix_for(mode)))
    pas, v = H.hw_planes(cfg, mu, mode)
    nb = H.nbhd_hw(pas)
    d["nb"] = nb[tuple(np.searchsorted(H.HW_AX[k], d[k].round(2).values) for k in KEYS)]
    d["pass"] = d[f"v_net_{mode}"].notna() & d[f"fell_{mode}"].isna() & (d[f"v_net_{mode}"] > 0.05) & (d[f"straight_{mode}"] > 0.5)
    d["robust"] = d["pass"] & (d.nb >= H.THRESH)
    d["rc"] = d["robust"] & (d[f"clear_ok_{mode}"] == 1)
    d["v"] = d[f"v_net_{mode}"]
    return add_demand(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pid", choices=["ff", "pid"])
    ap.add_argument("--configs", nargs="*", default=["c1", "c2", "c3", "c4", "c5", "c6"])
    a = ap.parse_args()
    os.makedirs(H.OUT, exist_ok=True)
    rows, panels = [], []
    for c in a.configs:
        for mu in H.HW_MU:
            try:
                d = robust_clear(c, mu, a.mode)
            except FileNotFoundError as e:
                print(f"  skip {c} mu{mu}: {e}"); continue
            g = d[d.rc]
            print(f"\n== {c} mu{mu} ({a.mode}): pass {int(d['pass'].sum())}  robust+clear {len(g)}  champion v {g.v.max():.3f}")
            for k in ("rate_mean_crank", "rate_peak_crank", "over_frac_crank", "over_frac_hip", "over_frac_any", "limit_frac_crank", "limit_frac_hip", "limit_frac_any", "exec_ratio_crank"):
                q = np.percentile(g[k], [10, 50, 90])
                print(f"   {k:16s} p10/50/90 {q[0]:7.2f} {q[1]:7.2f} {q[2]:7.2f}")
            line = "   champion v_net | over_frac_any <= " + "  ".join(f"{b*100:3.0f}%: {g[g.over_frac_any <= b + 1e-9].v.max() if (g.over_frac_any <= b + 1e-9).any() else float('nan'):.3f} (n={int((g.over_frac_any <= b + 1e-9).sum())})" for b in OVER_BARS)
            print(line)
            line = "   champion v_net | limit_frac_any (servo AT the cap) <= " + "  ".join(f"{b*100:3.0f}%: {g[g.limit_frac_any <= b + 1e-9].v.max() if (g.limit_frac_any <= b + 1e-9).any() else float('nan'):.3f} (n={int((g.limit_frac_any <= b + 1e-9).sum())})" for b in LIMIT_BARS)
            print(line)
            for b in LIMIT_BARS:
                m = g.limit_frac_any <= b + 1e-9
                rows.append(dict(cfg=c, mu=mu, mode=a.mode, bar="limit_frac_any", value=b, n=int(m.sum()), champ_v=g[m].v.max() if m.any() else np.nan,
                                 champ=("/".join(f"{x:g}" for x in g[m].sort_values("v").iloc[-1][list(KEYS)]) if m.any() else "")))
            line = "   champion v_net | rate_mean_crank <= " + "  ".join(f"{b:3.0f}: {g[g.rate_mean_crank <= b].v.max() if (g.rate_mean_crank <= b).any() else float('nan'):.3f} (n={int((g.rate_mean_crank <= b).sum())})" for b in RATE_BARS)
            print(line)
            for b in OVER_BARS:
                m = g.over_frac_any <= b + 1e-9
                rows.append(dict(cfg=c, mu=mu, mode=a.mode, bar="over_frac_any", value=b, n=int(m.sum()), champ_v=g[m].v.max() if m.any() else np.nan,
                                 champ=("/".join(f"{x:g}" for x in g[m].sort_values("v").iloc[-1][list(KEYS)]) if m.any() else "")))
            for b in RATE_BARS:
                m = g.rate_mean_crank <= b
                rows.append(dict(cfg=c, mu=mu, mode=a.mode, bar="rate_mean_crank", value=b, n=int(m.sum()), champ_v=g[m].v.max() if m.any() else np.nan,
                                 champ=("/".join(f"{x:g}" for x in g[m].sort_values("v").iloc[-1][list(KEYS)]) if m.any() else "")))
            panels.append((c, mu, d))
    pd.DataFrame(rows).to_csv(os.path.join(H.OUT, f"motor_demand_{a.mode}.csv"), index=False)
    if not panels:
        return
    n = len(panels)
    ncol = 4 if n > 4 else n
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.3 * nrow), squeeze=False, sharex=True, sharey=True)
    for ax, (c, mu, d) in zip(axes.ravel(), panels):
        k, com = style5.CONFIGS[c]
        col = style5.GAIT[k]["color"]
        p = d[d["pass"] & ~d.rc]
        ax.plot(p.limit_frac_any * 100, p.v, ".", ms=2, color="0.75", alpha=0.4, label="pass")
        g = d[d.rc]
        ax.plot(g.limit_frac_any * 100, g.v, ".", ms=3, color=col, alpha=0.6, label="robust + clear")
        ax.set_title(f"{c} (κ={k:g}, COM {com:.2f})  μ {mu:g}", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("servo AT the 354 °/s cap [% of cycle]")
    for ax in axes[:, 0]:
        ax.set_ylabel("v_net [m/s]")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Motor demand vs speed — {'cap-only, κ PID torso' if a.mode == 'pid' else 'cap + 56 ms + feedforward torso'} (demand is analytic from freq / leg_amp / hip_amp / hip_phi)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(H.OUT, f"motor_demand_{a.mode}.png")
    fig.savefig(p, dpi=130)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()

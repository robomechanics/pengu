"""re-simulate the top-N robust+clearance cells locally with a finer feedforward lead scan,
so the flashed champion is one whose phase window is wide, not a single-lead survivor"""
import os, sys, csv, numpy as np, pandas as pd
sys.path.insert(0, "grid6"); sys.path.insert(0, "grid6/analysis"); sys.path.append(".")
import hw_sweep as hs
import hw_vs_grid5 as H
CFG, MU, N = os.environ["CONFIG"], float(os.environ["MU"]), int(os.environ.get("N", "40"))
LEADS = (10.0, 30.0, 50.0, 70.0, 90.0)
keys = ("freq", "hip_phi", "leg_amp", "hip_amp", "hip_off")
d = pd.read_csv(H.hw_file(CFG, MU))
pas, v = H.hw_planes(CFG, MU, "ff"); nb = H.nbhd_hw(pas)
d["nb"] = nb[tuple(np.searchsorted(H.HW_AX[k], d[k].round(2).values) for k in keys)]
ok = d.v_net_ff.notna() & d.fell_ff.isna() & (d.v_net_ff > 0.05) & (d.straight_ff > 0.5) & (d.nb >= 0.8) & (d.clear_ok_ff == 1)
top = d[ok].sort_values("v_net_ff", ascending=False).head(N)
out = f"results/grid6_hw/demos/lead_window_{CFG}_mu{int(MU*100):03d}.csv"
w = csv.writer(open(out, "w", newline="")); w.writerow(list(keys) + ["psc_v", "psc_lead", "held", "A0", "p0"] + [f"lead{int(l)}" for l in LEADS] + ["n_walk", "min_v_walk", "mean_v_walk"])
for _, r in top.iterrows():
    cell = tuple(float(r[k]) for k in keys)
    held = hs.rollout(*cell, MU, "held")
    row = list(cell) + [round(r.v_net_ff, 3), r.best_lead]
    if held.get("fell") is not None:
        row += [f"fell {held['fell']:.1f}", "", ""] + [""] * len(LEADS) + [0, "", ""]; w.writerow(row); print(row, flush=True); continue
    A0, p0, _ = held["_fit"]
    row += ["ok", round(A0, 1), round(p0 % 360, 1)]
    vs = []
    for lead in LEADS:
        x = hs.rollout(*cell, MU, "ff", A=A0, ph=p0 + lead, kappa=hs.KAPPA)
        if x.get("fell") is not None: row.append(f"fell{x['fell']:.0f}")
        else:
            good = x["straight"] > 0.5 and x["v_net"] > 0.05 and x["clear"] >= 10
            row.append(f"{x['v_net']:.3f}{'' if good else '*'}"); vs.append(x["v_net"] if good else np.nan)
    vs = [q for q in vs if np.isfinite(q)]
    row += [len(vs), round(min(vs), 3) if vs else "", round(float(np.mean(vs)), 3) if vs else ""]
    w.writerow(row); print(row, flush=True)
print("->", out)

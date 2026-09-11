"""noflight_search.py — separate study (Ben 2026-09-10): walk down the GRID-7 champion ranking of a
config (robust + clearance + motor bar, fastest first) replaying each cell on this machine, and
keep the first one that has NO flight phase in the steady part of the gait. Criterion (Ben
2026-09-10): in the last STEADY_S seconds of the 8 s window every physics step has at least one foot
geom in contact with the floor (no force threshold, no run-length tolerance) and the stance sequence
is single / double only. Flight in the first part of the window (before the gait has settled) is
allowed. Per cell the script reports WHEN the no-contact steps happen (count per 1 s bin over the
window, time of the last one, cycles after it), the deepest floor penetration, and the looser
numbers (total Fn == 0 at a 200 Hz sample, both feet <= 2 N). Every cell's per-sample arrays are
cached in results/grid7_report/work/noflight_cache/ so a changed criterion needs no re-simulation.
Reports every cell tried. Does not touch the champion JSON used elsewhere.

    PENGU_MODEL=1.31 python grid7/noflight_search.py --config c6 --mus 0.1 0.3 0.5 0.7 --max 40
-> results/grid7_report/data/noflight_<cfg>.json (chosen cell per mu) and noflight_<cfg>_tried.csv
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
sys.path.insert(0, os.path.join(ROOT, "grid5", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "grid6"))
sys.path.append(ROOT)
import motor_demand as M                         # noqa: E402
import stance_com as sc                          # noqa: E402
import load5                                     # noqa: E402
import style5                                    # noqa: E402
import gait_sweep as gs                          # noqa: E402
import mujoco                                    # noqa: E402

DATA = os.path.join(ROOT, "results", "grid7_report", "data")
CACHE = os.path.join(ROOT, "results", "grid7_report", "work", "noflight_cache")
STEADY_S = 5.0                                   # the last STEADY_S s of the window must be flight-free
KAPPA = {"c1": 0.0, "c2": 0.0, "c3": 0.0, "c4": 2.0, "c5": 2.0, "c6": 2.0}
KEYS = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off"]


def foot_vertices():
    """all vertices of each foot mesh in the foot body frame, {'L': (V,3), 'R': (V,3)}"""
    model = mujoco.MjModel.from_xml_path(gs.XML)
    _, foot_geom, _, _ = gs.make_ids(model)
    out = {}
    for g, side in foot_geom.items():
        d = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[d]: model.mesh_vertadr[d] + model.mesh_vertnum[d]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        out[side] = v @ R.reshape(3, 3).T + model.geom_pos[g]
    return out


_VERTS = None


def sole_height(r):
    """(n, 2) height of the lowest mesh vertex of each foot above the floor, m (negative = penetrating)"""
    global _VERTS
    if _VERTS is None:
        _VERTS = foot_vertices()
    out = np.zeros((len(r["t"]), 2))
    for k, side in enumerate("LR"):
        vw_z = r["fpos"][:, k, 2][:, None] + np.einsum("vj,nj->nv", _VERTS[side], r["fmat"][:, k, 2, :])
        out[:, k] = vw_z.min(1)
    return out


def flight_runs(r):
    """every run of samples with total Fn == 0: (t0 [s from window start], dur_ms, hop_mm = max over the run of the
    lower foot's lowest point above the floor, land_fw = peak total Fn / weight in the 60 ms after)"""
    fn = r["fn"]; W = float(r["mass"]) * 9.81
    air = fn.sum(1) <= 0
    low = sole_height(r).min(1)
    runs, i = [], 0
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
    return pd.DataFrame(runs, columns=["t0", "dur_ms", "hop_mm", "land_fw"])


def longest_ms(mask):
    rl = sc.run_lengths(mask) / sc.FS * 1000
    return float(rl.max()) if len(rl) else 0.0


def grid5_ranking(cfg, mu):
    """GRID-5 (ideal actuators) robust cells at mu, fastest first: pass_rate > 0, 3x3 nbhd mean >= 0.8, no clearance / motor bar"""
    g = load5.load(cfg, rnd="grid5", verbose=False)
    m = [i for i, v in enumerate(g.axes["mu"]) if abs(v - mu) < 1e-6][0]
    N = g.nbhd("pass_rate")[m]
    nf = g["net_fwd_mean"][m]
    elig = (g["pass_rate"][m] > 0) & np.isfinite(nf) & np.isfinite(N) & (N >= 0.8)
    order = np.argsort(np.where(elig, nf, -np.inf).ravel())[::-1][:int(elig.sum())]
    rows = []
    for j in order:
        idx = np.unravel_index(j, nf.shape)
        rows.append({**{k: float(g.axes[k][i]) for k, i in zip(KEYS, idx)}, "v": float(nf[idx])})
    return pd.DataFrame(rows)


def cached_rollout(cfg, cell, mu, physics="grid7"):
    """per-sample arrays of one replay (t, fn, nofoot_steps, steps, pen, ncon, fell, mass), cached as npz.
    physics grid7: 354 deg/s cap on legs + torso, kappa PID; grid5: ideal actuators, kappa PID, COM slide of the config"""
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{cfg}_{physics}_mu{mu:g}_" + "_".join(f"{x:g}" for x in cell) + ".npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        if all(k in z.files for k in ("fpos", "e_pos", "ryaw")):   # older caches lack some fields -> recompute
            return {k: z[k] for k in z.files}
    if physics == "grid5":
        sc.COM_TARGET = style5.CONFIGS[cfg][1]
        r = sc.rollout(cell, mu, KAPPA[cfg], False, cap=False)
    else:
        r = sc.rollout(cell, mu, KAPPA[cfg], False, cap=True)
    keep = dict(t=r["t"], fn=r["fn"], nofoot_steps=r["nofoot_steps"], steps=r["steps"], pen=r["pen"], ncon=r["ncon"],
                fell=np.array(np.nan if r["fell"] is None else r["fell"]), mass=np.array(r["mass"]))
    if r["fn"].ndim == 2:                         # foot geometry for sole-height / hop analyses
        keep.update(fpos=r["fpos"], fmat=r["fmat"], com=r["com"], foot_bottom_L=r["foot_bottom"][0], foot_bottom_R=r["foot_bottom"][1], e_pos=r["e_pos"], ryaw=r["ryaw"])
    np.savez(f, **keep)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="c6")
    ap.add_argument("--mus", nargs="*", type=float, default=[0.1, 0.3, 0.5, 0.7])
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--bar", type=float, default=0.75)
    ap.add_argument("--margin", type=int, default=1)
    ap.add_argument("--ranks", nargs="*", type=int, default=None, help="probe only these ranks (1-based) instead of walking 1..max")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--steady", type=float, default=STEADY_S, help="the last N seconds of the window must have no no-contact step")
    ap.add_argument("--physics", default="grid7", choices=["grid7", "grid5"], help="grid7 = hwcapt ranking + capped replay; grid5 = ideal-actuator ranking + replay")
    ap.add_argument("--hop-mm", type=float, default=0.0, help="> 0: a flight phase only counts if the gap under the lower foot reaches this height (Ben 2026-09-10: 2 mm); 0: any no-contact physics step counts")
    ap.add_argument("--cells", nargs="*", default=None, help="probe these cells (freq/phi/leg/hip/off) instead of walking the ranking; rank = their position in the ranking if present")
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    chosen, tried = {}, []
    for mu in a.mus:
        if a.physics == "grid5":
            g = grid5_ranking(a.config, mu)
            print(f"{a.config} mu {mu} [GRID-5 physics]: {len(g)} robust cells, v from {g.v.max():.3f} down to {g.v.min():.3f}", flush=True)
        else:
            d = M.robust_clear(a.config, mu, "pid", "hwcapt")
            g = d[d.rc & M.bar_mask(d, a.bar, a.margin)].sort_values("v", ascending=False)
            print(f"{a.config} mu {mu}: {len(g)} robust+clear+bar cells, v from {g.v.max():.3f} down to {g.v.min():.3f}", flush=True)
        if a.cells:
            items = []
            for cs in a.cells:
                cell = tuple(float(x) for x in cs.split("/"))
                hit = g[(np.abs(g.freq - cell[0]) < 1e-6) & (g.hip_phi == cell[1]) & (g.leg_amp == cell[2]) & (g.hip_amp == cell[3]) & (g.hip_off == cell[4])]
                rank = int(np.where((g.index == hit.index[0]))[0][0]) + 1 if len(hit) else -1
                row = hit.iloc[0] if len(hit) else pd.Series(dict(zip(KEYS, cell), v=np.nan))
                items.append((rank, row))
        else:
            items = [(k, g.iloc[k - 1]) for k in a.ranks if k <= len(g)] if a.ranks else list(enumerate((row for _, row in g.head(a.max).iterrows()), 1))
        found = None
        for rank, row in items:
            cell = tuple(float(row[k]) for k in KEYS)
            r = cached_rollout(a.config, cell, mu, a.physics)
            base = dict(cfg=a.config, mu=mu, rank=rank, cell="/".join(f"{x:g}" for x in cell), v_sweep=row.v)
            if not np.isnan(r["fell"]) or r["fn"].ndim != 2:
                rec = dict(base, fell=float(r["fell"]))
                print(f"{a.config} mu {mu}: rank {rank} {rec['cell']} v {row.v:.3f}  FELL {rec['fell']:.1f}s", flush=True)
                tried.append(rec); continue
            fn = r["fn"]; n2 = (fn > sc.FN_MIN).sum(1)
            nofoot = int(r["nofoot_steps"].sum()); nsteps = int(r["steps"].sum())
            tw = r["t"] - r["t"][0]                                   # window time, 0..8 s
            nf = r["nofoot_steps"] > 0                                # samples that contain a no-contact physics step
            per_s = [int(nf[(tw >= k) & (tw < k + 1)].sum()) for k in range(int(sc.WINDOW))]
            t_last = float(tw[nf].max()) if nf.any() else -1.0
            steady_ok = not nf[tw >= sc.WINDOW - a.steady].any()
            hops = flight_runs(r)
            big = hops[hops.hop_mm >= a.hop_mm] if a.hop_mm > 0 else hops
            hop_info = dict(n_runs=len(hops), n_hops=len(big), hop_max_mm=float(hops.hop_mm.max()) if len(hops) else 0.0,
                            hop_dur_max_ms=float(big.dur_ms.max()) if len(big) else 0.0,
                            t_last_hop=float(big.t0.max()) if len(big) else -1.0,
                            land_fw_med=float(big.land_fw.median()) if len(big) else np.nan)
            if a.hop_mm > 0:
                steady_ok = not (big.t0 >= sc.WINDOW - a.steady).any()
            rec = dict(base, fell=None,
                       nofoot_steps=nofoot, nofoot_pct=100.0 * nofoot / max(nsteps, 1),
                       air0_pct=np.mean(fn.sum(1) <= 0) * 100, air0_max_ms=longest_ms(fn.sum(1) <= 0),
                       air2_pct=np.mean(n2 == 0) * 100, air2_max_ms=longest_ms(n2 == 0),
                       single_pct=np.mean(n2 == 1) * 100, double_pct=np.mean(n2 == 2) * 100,
                       pen_max_mm=float(r["pen"].max() * 1000), pen_p50_mm=float(np.median(r["pen"][r["pen"] > 0]) * 1000) if (r["pen"] > 0).any() else 0.0,
                       ncon_max=int(r["ncon"].max()), peak_fn_over_w=float(fn.max() / (r["mass"] * 9.81)),
                       nofoot_per_s="/".join(str(k) for k in per_s), t_last_nofoot=t_last,
                       cycles_after_last=(sc.WINDOW - t_last) * cell[0] if t_last >= 0 else sc.WINDOW * cell[0],
                       steady_ok=steady_ok, hop_thresh_mm=a.hop_mm, **hop_info)
            print(f"{a.config} mu {mu}: rank {rank} {rec['cell']} v {row.v:.3f}  no-contact samples per 1 s bin [{rec['nofoot_per_s']}]  last at {t_last:.2f} s "
                  f"({rec['cycles_after_last']:.1f} cycles after)  hops>={a.hop_mm:g}mm {hop_info['n_hops']}/{hop_info['n_runs']} runs (max gap {hop_info['hop_max_mm']:.1f} mm, longest {hop_info['hop_dur_max_ms']:.0f} ms, last at {hop_info['t_last_hop']:.2f} s)  "
                  f"steady(last {a.steady:g} s) {'OK' if steady_ok else 'FLIGHT'}  "
                  f"single {rec['single_pct']:.0f}%  double {rec['double_pct']:.0f}%  pen max {rec['pen_max_mm']:.2f} mm  peak Fn/W {rec['peak_fn_over_w']:.1f}", flush=True)
            tried.append(rec)
            if steady_ok and not a.ranks and not a.cells:
                found = rec; break
        chosen[f"{a.config}_mu{mu}"] = found
        print(f"{a.config} mu {mu}: " + (f"NO-FLIGHT champion = rank {found['rank']} {found['cell']} v {found['v_sweep']:.3f}" if found else f"none of the top {a.max} is flight-free in the last {a.steady:g} s"), flush=True)
        pd.DataFrame(tried).to_csv(os.path.join(DATA, f"noflight_{a.config}{a.tag}_tried.csv"), index=False)
        json.dump(chosen, open(os.path.join(DATA, f"noflight_{a.config}{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

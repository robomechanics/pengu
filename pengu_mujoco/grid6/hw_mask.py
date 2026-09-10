"""hw_mask.py — prune the hardware sweep grid with the GRID-5 maps: keep a candidate cell
only if a passing GRID-5 cell lies within a margin of it, drop it if its whole
neighbourhood is black.

Ben 2026-09-09: the obviously black regions of the heat maps (whole hip_phi columns that
never walked at any friction) are not worth hardware-model rollouts; the fuzzy edges
might walk once the actuators are modelled, so they stay. The rule is a dilation of the
GRID-5 passing set by `--margin` lattice steps (freq 0.02, hip_phi 10 wrapping, leg_amp 10,
hip_amp 4, hip_off 10 per step), evaluated at every cell of the candidate grid; friction
is fuzzed the same way (the 0.12 sweep looks at the map's mu 0.1 and 0.3, the 0.45 sweep
at 0.3 and 0.5).

    CONFIG=c5 python grid6/hw_mask.py --mu 0.12 --margin 1            # writes the cell list
    CONFIG=c5 python grid6/hw_mask.py --mu 0.12 --count               # just the counts, r=0..3

Output: results/grid6_hw/<cfg>/cells_<cfg>_mu012_r1.csv (freq,hip_phi,leg_amp,hip_amp,hip_off),
which hw_sweep.py --cells-file consumes. The candidate grid is the union range: freq
1.20-1.70 step HW_FREQ_STEP (0.05, Ben 2026-09-09), hip_phi all 36, leg_amp 70-130 step 5,
hip_amp {12..32}, hip_off {20..40 step 5}.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
CONFIG = os.environ.get("CONFIG", "c1").lower()
MAPS = os.path.join(ROOT, "results", "gait_sweep")
OUT = os.path.join(ROOT, "results", "grid6_hw", CONFIG)
MAP_MU = {0.12: (0.1, 0.3), 0.45: (0.3, 0.5),
          # GRID-5 friction ladder itself (Ben 2026-09-10, torso-capped PID sweep): the map's own mu plus its neighbours
          0.1: (0.1, 0.3), 0.3: (0.1, 0.3, 0.5), 0.5: (0.3, 0.5, 0.7), 0.7: (0.5, 0.7)}
STEP = dict(freq=0.02, hip_phi=10.0, leg_amp=10.0, hip_amp=4.0, hip_off=10.0)

# candidate grid (union of everything on the table)
FREQ_STEP = float(os.environ.get("HW_FREQ_STEP", "0.05"))   # Ben 2026-09-09: 0.05 (was 0.02)
FREQ = [round(1.20 + FREQ_STEP * k, 2) for k in range(int(round(0.50 / FREQ_STEP)) + 1)]
PHI = list(range(0, 360, 10))
LEG = list(range(70, 135, 5))
HIP = [12, 16, 20, 24, 28, 32]
OFF = [20, 25, 30, 35, 40]


def load_map():
    base = os.path.join(MAPS, f"sweep_grid5_{CONFIG}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv")
    scratch = os.environ.get("GRID5_MAP_DIR", "")
    for p in (base, base + ".gz", os.path.join(scratch, os.path.basename(base) + ".gz")):
        if p and os.path.exists(p):
            return pd.read_csv(p, usecols=["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off",
                                           "mu", "pass_rate"])
    raise SystemExit(f"no GRID-5 map for {CONFIG} (assemble the .part files or set GRID5_MAP_DIR)")


def passing_lattice(df, mus):
    """boolean lattice of passing cells at any of `mus`, plus the axis values"""
    axes = {k: np.array(sorted(df[k].unique())) for k in ("freq", "hip_phi", "leg_amp", "hip_amp", "hip_off")}
    g = df[df.mu.round(2).isin([round(m, 2) for m in mus]) & (df.pass_rate >= 1.0)]
    idx = tuple(np.searchsorted(axes[k], g[k].values) for k in axes)
    lat = np.zeros([len(axes[k]) for k in axes], bool)
    lat[idx] = True
    return lat, axes


def dilate(lat, axes, margin):
    """grow the passing set by `margin` steps per axis (phi wraps)"""
    steps = {k: max(1, int(round(STEP[k] / np.min(np.diff(axes[k]))))) if len(axes[k]) > 1 else 1
             for k in axes}
    out = lat.copy()
    for ax, k in enumerate(axes):
        r = steps[k] * margin
        if r == 0:
            continue
        acc = out.copy()
        for s in range(1, r + 1):
            if k == "hip_phi":
                acc |= np.roll(out, s, axis=ax) | np.roll(out, -s, axis=ax)
            else:
                sl_f = [slice(None)] * out.ndim
                sl_b = [slice(None)] * out.ndim
                sl_f[ax], sl_b[ax] = slice(s, None), slice(None, -s)
                fwd = np.zeros_like(out)
                bwd = np.zeros_like(out)
                fwd[tuple(sl_f)] = out[tuple(sl_b)]
                bwd[tuple(sl_b)] = out[tuple(sl_f)]
                acc |= fwd | bwd
        out = acc
    return out


def candidates():
    return [(f, float(p), float(a), float(h), float(o))
            for f in FREQ for p in PHI for a in LEG for h in HIP for o in OFF]


def keep_mask(lat, axes, cells):
    """nearest-lattice lookup of each candidate cell"""
    keys = ("freq", "hip_phi", "leg_amp", "hip_amp", "hip_off")
    idx = []
    for k, col in zip(keys, np.array(cells).T):
        a = axes[k]
        j = np.clip(np.searchsorted(a, col), 0, len(a) - 1)
        jm = np.clip(j - 1, 0, len(a) - 1)
        idx.append(np.where(np.abs(a[j] - col) <= np.abs(a[jm] - col), j, jm))
    return lat[tuple(idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, required=True)
    ap.add_argument("--margin", type=int, default=1)
    ap.add_argument("--count", action="store_true")
    a = ap.parse_args()
    df = load_map()
    lat, axes = passing_lattice(df, MAP_MU[a.mu])
    cells = candidates()
    print(f"{CONFIG} mu {a.mu} (map mu {MAP_MU[a.mu]}): GRID-5 passing lattice {int(lat.sum()):,} "
          f"of {lat.size:,};  candidate grid {len(cells):,} cells")
    if a.count:
        for r in range(0, 4):
            m = keep_mask(dilate(lat, axes, r), axes, cells)
            print(f"  margin {r}: keep {int(m.sum()):7,d} cells ({m.mean()*100:5.1f}%)")
        return
    m = keep_mask(dilate(lat, axes, a.margin), axes, cells)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"cells_{CONFIG}_mu{int(round(a.mu*100)):03d}_r{a.margin}.csv")
    pd.DataFrame([c for c, k in zip(cells, m) if k],
                 columns=["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off"]).to_csv(p, index=False)
    print(f"  margin {a.margin}: keep {int(m.sum()):,} cells -> {p}")


if __name__ == "__main__":
    main()

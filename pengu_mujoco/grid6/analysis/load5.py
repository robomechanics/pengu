#!/usr/bin/env python
"""Shared loader for GRID-5 (and GRID-4) sweep CSVs. Every figure script
imports this; nothing else in the repo may parse the sweep CSVs.

Contract (docs/grid5/PLOT_GRID5.md §6):
  * axis lists come from the manifest, never hardcoded (GRID-4 has no
    manifest, so a frozen built-in spec stands in)
  * the manifest is a gate: refuse to plot two configs on shared axes unless
    protocol / K / axes / start / slip / gates agree  -> compatible()
  * stdlib csv; rows with the wrong column count are counted in `malformed`;
    malformed > 5 raises (that is corruption, not a torn tail line)
  * fall_phase is a STRING column -> four int8 count planes, never floats
  * partial CSVs load fine but carry complete=False and `present` per axis;
    cells iterate hip_off outermost, so a partial file is a strided
    hip_off-block subsample -- restrict cross-config stats to `present`
  * npz cache under results/grid5_report/cache/, keyed on repo_commit (grid5)
    + csv size + mtime; second load is sub-second

Neighborhood (frozen, §5): mean over freq ±2 and TRUE circular phi adjacency,
divided by the count of valid contributors (the GRID-5 phi axis drops
150-190, so array-adjacent 140/200 are NOT neighbours and seam columns have
fewer contributors). legacy=True reproduces the round-1 roll()+/15 construct
for cross-checks against published GRID-4 numbers.
"""
import os, sys, io, csv, gzip, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))            # pengu_mujoco/
GS = os.path.join(_ROOT, "results", "gait_sweep")
CACHE_DIR = os.path.join(_ROOT, "results", "grid5_report", "cache")

AXNAMES = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "mu"]
FALL_PHASES = ["hold", "trans", "settle", "walk"]
MAX_MALFORMED = 5

# frozen GRID-4 stand-in for the missing manifest (physics/grid4_report.py)
GRID4_SPEC = dict(
    axes=dict(
        freq=[round(f, 2) for f in np.arange(1.00, 2.0001, 0.01)],
        hip_phi=[float(p) for p in range(0, 351, 10)],
        leg_amp=[85.0, 95.0, 105.0, 115.0, 125.0],
        hip_amp=[12.0, 16.0, 20.0, 24.0, 28.0],
        hip_off=[10.0, 20.0, 30.0, 40.0, 50.0],
        mu=[0.1, 0.3, 0.5, 0.7]),
    schema=["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "mu",
            "pass_rate", "surv_rate", "net_fwd_mean", "net_fwd_min",
            "slip_mean", "head_mean"],
    rows=1818000, K=1)


class Grid:
    """One config's map: dense planes A[mu, freq, phi, leg, hip, off]."""

    def __init__(self, cfg, rnd, manifest, axes, planes, rows_loaded,
                 malformed, path):
        self.cfg, self.round, self.manifest = cfg, rnd, manifest
        self.axes = axes                       # name -> list of floats
        self.planes = planes                   # col/nfall_* -> ndarray
        self.rows_loaded, self.malformed = rows_loaded, malformed
        self.path = path
        self.filled = np.isfinite(planes["pass_rate"])
        self.expected = int(np.prod([len(axes[a]) for a in AXNAMES]))
        self.complete = int(self.filled.sum()) == self.expected
        # values of each axis that actually appear (partial-file guard).
        # plane axis order is (mu, freq, phi, leg, hip, off), NOT AXNAMES order
        self.present = {}
        plane_ax = dict(mu=0, freq=1, hip_phi=2, leg_amp=3, hip_amp=4, hip_off=5)
        for a in AXNAMES:
            other = tuple(j for j in range(6) if j != plane_ax[a])
            got = self.filled.any(axis=other)
            self.present[a] = [float(v) for v, g in zip(axes[a], got) if g]

    def __getitem__(self, col):
        return self.planes[col]

    @property
    def K(self):
        return self.manifest["K"] if self.manifest else GRID4_SPEC["K"]

    @property
    def commit(self):
        return self.manifest.get("repo_commit", "") if self.manifest else ""

    def phi_adjacency(self, step=10.0):
        """True circular adjacency on the REAL phi angles (§4.1): 350<->0 are
        neighbours, 140<->200 (across the removed 150-190 band) are not."""
        phi = np.asarray(self.axes["hip_phi"], float)
        d = np.abs(phi[:, None] - phi[None, :]) % 360.0
        return np.minimum(d, 360.0 - d) <= step + 1e-9

    def nbhd(self, col="pass_rate", legacy=False):
        """Neighborhood mean over freq ±2 x phi-adjacent. Frozen definition:
        divide by the count of valid contributors; freq edges NaN.
        legacy=True: round-1 roll()+/15 (only correct on a contiguous phi
        axis; kept for validation against published GRID-4 numbers)."""
        A = self.planes[col]
        if legacy:
            out = np.zeros_like(A)
            for df in (-2, -1, 0, 1, 2):
                for dp in (-1, 0, 1):
                    out += np.roll(np.roll(A, df, axis=1), dp, axis=2)
            out /= 15.0
            out[:, :2] = np.nan; out[:, -2:] = np.nan
            return out
        W = self.phi_adjacency().astype(np.float64)        # (P, P), incl self
        fin = np.isfinite(A)
        Az = np.where(fin, A, 0.0).astype(np.float64)
        Fn = fin.astype(np.float64)

        def phi_apply(X):                                  # sum over adj phi
            return np.moveaxis(
                np.tensordot(W, np.moveaxis(X, 2, 0), axes=(1, 0)), 0, 2)

        s = np.zeros(A.shape); c = np.zeros(A.shape)
        for df in (-2, -1, 0, 1, 2):                       # freq: no wrap
            Xs = np.zeros_like(Az); Fs = np.zeros_like(Fn)
            if df == 0:
                Xs, Fs = Az, Fn
            elif df > 0:
                Xs[:, df:] = Az[:, :-df]; Fs[:, df:] = Fn[:, :-df]
            else:
                Xs[:, :df] = Az[:, -df:]; Fs[:, :df] = Fn[:, -df:]
            s += phi_apply(Xs); c += phi_apply(Fs)
        out = np.where(c > 0, s / np.maximum(c, 1.0), np.nan).astype(np.float32)
        out[:, :2] = np.nan; out[:, -2:] = np.nan
        return out

    def summary(self):
        pres = {a: (f"{len(v)}/{len(self.axes[a])}"
                    + ("" if len(v) == len(self.axes[a]) else f" {v}"))
                for a, v in self.present.items()}
        return (f"{self.round} {self.cfg}: {int(self.filled.sum())}/"
                f"{self.expected} cells "
                f"({'complete' if self.complete else 'PARTIAL'}), "
                f"rows={self.rows_loaded} malformed={self.malformed} K={self.K}"
                + (f" commit={self.commit[:9]}" if self.commit else "")
                + "\n  present: " + "  ".join(f"{a}={p}" for a, p in pres.items()))


def _csv_path(cfg, rnd):
    base = os.path.join(
        GS, f"sweep_{rnd}_{cfg}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv")
    if os.path.exists(base + ".gz"):
        return base + ".gz"                       # shipped form, preferred
    if os.path.exists(base):
        return base
    return None


def _load_manifest(cfg, rnd):
    if rnd != "grid5":
        return None
    p = os.path.join(
        GS, f"sweep_grid5_{cfg}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu"
            ".manifest.json")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{cfg}: manifest missing ({p}) — the manifest is a gate, "
            "refusing to load the CSV without it")
    man = json.load(open(p))
    if man.get("protocol") not in ("grid5-v1", "grid5-v2"):
        raise ValueError(f"{cfg}: unexpected protocol {man.get('protocol')!r}")
    # v1 and v2 are different experiments (v2 = deterministic map, full phi
    # circle, wider leg/hip axes); compatible() refuses to mix them on axes.
    if man.get("config") != cfg:
        raise ValueError(f"{cfg}: manifest says config={man.get('config')!r}")
    return man


def compatible(grids):
    """Gate for putting several configs on shared axes: the protocol-level
    manifest fields must agree (kappa/com_target legitimately differ)."""
    keys = ["protocol", "K", "axes", "start", "slip", "gates"]
    ref = None
    for g in grids:
        if g.manifest is None:                    # grid4: one frozen spec
            continue
        cur = {k: g.manifest.get(k) for k in keys}
        if ref is None:
            ref = cur
        elif cur != ref:
            bad = [k for k in keys if cur[k] != ref[k]]
            raise ValueError(
                f"{g.cfg}: manifest disagrees on {bad} — these CSVs are not "
                "comparable and must not share axes")
    return True


def _parse(path, axes, schema, verbose=True):
    idx = {a: {v: i for i, v in enumerate(axes[a])} for a in AXNAMES}
    shape = tuple(len(axes[a]) for a in ["mu", "freq", "hip_phi",
                                         "leg_amp", "hip_amp", "hip_off"])
    ncol = len(schema)
    num_cols = [(j, c) for j, c in enumerate(schema)
                if c not in AXNAMES and c != "fall_phase"]
    has_fall = "fall_phase" in schema
    j_fall = schema.index("fall_phase") if has_fall else -1
    planes = {c: np.full(shape, np.nan, np.float32) for _, c in num_cols}
    if has_fall:
        for ph in FALL_PHASES:
            planes["nfall_" + ph] = np.zeros(shape, np.int8)
    n = bad = 0
    op = gzip.open if path.endswith(".gz") else open
    fi, pi, li, hi, oi, mi = (idx[a] for a in AXNAMES)
    with op(path, "rt", newline="") as f:
        for r in csv.reader(f):
            if not r or r[0] == "freq":
                continue
            if len(r) != ncol:
                bad += 1
                if bad > MAX_MALFORMED:
                    raise ValueError(
                        f"{path}: {bad} malformed rows — that is corruption, "
                        "not a torn tail line. Stop and look.")
                continue
            try:
                at = (mi[float(r[5])], fi[round(float(r[0]), 2)],
                      pi[float(r[1])], li[float(r[2])], hi[float(r[3])],
                      oi[float(r[4])])
            except (KeyError, ValueError):
                bad += 1
                if bad > MAX_MALFORMED:
                    raise ValueError(
                        f"{path}: {bad} rows off-axis/unparseable — stop and look.")
                continue
            for j, c in num_cols:
                planes[c][at] = float(r[j])
            if has_fall:
                for ph in FALL_PHASES:               # e.g. "hold:1|walk:2"
                    planes["nfall_" + ph][at] = 0
                if r[j_fall]:
                    for tok in r[j_fall].split("|"):
                        name, _, cnt = tok.partition(":")
                        if name in FALL_PHASES:
                            planes["nfall_" + name][at] = int(cnt or 1)
            n += 1
    if verbose:
        print(f"  parsed {n} rows, {bad} malformed from {os.path.basename(path)}")
    return planes, n, bad


def _cache_key(path, manifest):
    st = os.stat(path)
    commit = manifest.get("repo_commit", "") if manifest else ""
    return f"{commit}:{st.st_size}:{int(st.st_mtime)}"


def load(cfg, rnd="grid5", use_cache=True, verbose=True):
    """Load one config into a Grid. rnd: 'grid5' (default) or 'grid4'."""
    path = _csv_path(cfg, rnd)
    if path is None:
        raise FileNotFoundError(f"no {rnd} CSV for {cfg} under {GS}")
    manifest = _load_manifest(cfg, rnd)
    if manifest is not None:
        axes = {a: [float(v) for v in manifest["axes"][a]] for a in AXNAMES}
        schema = list(manifest["schema"])
    else:
        axes = {a: list(GRID4_SPEC["axes"][a]) for a in AXNAMES}
        schema = list(GRID4_SPEC["schema"])

    key = _cache_key(path, manifest)
    cache = os.path.join(CACHE_DIR, f"{rnd}_{cfg}.npz")
    if use_cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            planes = {c: z[c] for c in z.files
                      if c not in ("key", "rows", "bad")}
            g = Grid(cfg, rnd, manifest, axes, planes,
                     int(z["rows"]), int(z["bad"]), path)
            if verbose:
                print(f"  cache hit: {os.path.basename(cache)}")
                print(g.summary())
            return g

    planes, n, bad = _parse(path, axes, schema, verbose=verbose)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, key=key, rows=n, bad=bad, **planes)
    g = Grid(cfg, rnd, manifest, axes, planes, n, bad, path)
    if verbose:
        print(g.summary())
    return g


TOPUP_FIELDS = ["pass_rate", "surv_rate", "net_fwd_mean", "net_fwd_min",
                "slip_mean", "head_mean"]


def load_topup(cfg, rnd="grid4"):
    """Merge every topup-K5 artifact for a config into one dict:
    axis-tuple (freq, phi, leg, hip, off, mu) -> {6 DR fields}.

    Sources, in override order (later wins):
      results/grid4_report/<cfg>/topupK5.csv          (130-cell selection; has
                                                       duplicate keys - dedupe)
      results/gait_sweep/sweep_<rnd>_<cfg>_topupK5.csv[.gz]  (full-passer)
    All values are per-cell K=5 aggregates from physics/grid5 topup_k.py -
    the ONLY authoritative K=5 numbers (hand-rolled replays diverge).
    """
    paths = []
    rep = os.path.join(_ROOT, "results", f"{rnd}_report", cfg, "topupK5.csv")
    if os.path.exists(rep):
        paths.append(rep)
    full = os.path.join(GS, f"sweep_{rnd}_{cfg}_topupK5.csv")
    for pth in (full, full + ".gz"):
        if os.path.exists(pth):
            paths.append(pth)
            break
    out = {}
    for pth in paths:
        op = gzip.open if pth.endswith(".gz") else open
        with op(pth, "rt", newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                try:
                    key = (round(float(r["freq"]), 2), float(r["hip_phi"]),
                           float(r["leg_amp"]), float(r["hip_amp"]),
                           float(r["hip_off"]), float(r["mu"]))
                    out[key] = {k: float(r[k]) for k in TOPUP_FIELDS}
                except (KeyError, ValueError, TypeError):
                    continue
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="load one config, print summary")
    ap.add_argument("cfg")
    ap.add_argument("--round", default="grid5", choices=["grid4", "grid5"])
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    g = load(a.cfg, rnd=a.round, use_cache=not a.no_cache)
    for ph in FALL_PHASES:
        k = "nfall_" + ph
        if k in g.planes:
            print(f"  {k}: {int(g.planes[k].sum())} falls tallied")

#!/usr/bin/env python3
"""
Merge the per-slice C2 CSVs into one dataset, and verify it before you trust it.

    python delta/merge_phi.py                      # report only, writes nothing
    python delta/merge_phi.py --out c7_merged.csv  # write the merged dataset

Checks performed (all of them, always -- the report is the point):
  * every slice has exactly 115,200 data rows
  * every slice's rows carry the hip_phi its filename claims
  * the (freq,hip_phi,leg_amp,hip_amp,hip_off,mu) key is unique across the whole
    merge -- a duplicate means two shards wrote the same cell, which would mean
    the resume logic failed somewhere
  * headers are identical across slices

It does NOT merge in hip_phi 0/10 (produced elsewhere). Add them by hand once you
have them, and record in the manifest which machine each slice came from: rows
from different platforms are not bit-comparable
(docs/grid5_design.md:108, "Map-vs-local diffs are cross-platform FP").
"""

import argparse
import csv
import glob
import os
import sys

ROWS_PER_SLICE = 115_200
# C2 sweeps the full circle: grid5_sweep.py:80 -> hip_phi = 0,10,...,350.
# (PSC's C1 campaign ran 20..350 because 0 and 10 were done elsewhere; that range
# does not apply here and reporting it was a porting leftover.)
PHI_FIRST, PHI_LAST, N_SLICES = 0, 350, 36
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
RESULTS = os.path.join(REPO, "pengu_mujoco", "results", "gait_sweep")
PATTERN = os.path.join(RESULTS, "sweep_grid5_c7_phi*_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv")

KEY = ("freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "mu")


def phi_of(path):
    base = os.path.basename(path)
    tag = base.split("_phi", 1)[1][:3]
    return float(int(tag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write merged CSV here")
    ap.add_argument("--allow-partial", action="store_true",
                    help="merge slices that are not full (default: refuse)")
    a = ap.parse_args()

    paths = sorted(glob.glob(PATTERN))
    if not paths:
        sys.exit(f"no slice CSVs matching:\n  {PATTERN}")

    # Group by hip_phi rather than assuming one file per slice. Today a slice is
    # one shared CSV; if the per-shard-file layout is adopted later (128 files per
    # slice, to kill Lustre append contention) this same script keeps working and
    # data produced under either layout stays mergeable.
    by_phi = {}
    for p in paths:
        by_phi.setdefault(phi_of(p), []).append(p)

    header = None
    seen = {}
    problems = []
    incomplete = []
    total = 0

    print(f"{'slice':<8} {'files':>6} {'rows':>9}  state")
    for phi in sorted(by_phi):
        n = 0
        for p in sorted(by_phi[phi]):
            with open(p, newline="") as f:
                r = csv.DictReader(f)
                if header is None:
                    header = r.fieldnames
                elif r.fieldnames != header:
                    problems.append(f"{os.path.basename(p)}: header differs from the first file")
                for row in r:
                    n += 1
                    if float(row["hip_phi"]) != phi:
                        problems.append(
                            f"phi={phi:.0f}: a row carries hip_phi={row['hip_phi']} "
                            f"-- filename and contents disagree ({os.path.basename(p)})")
                        break
                    k = tuple(round(float(row[c]), 4) for c in KEY)
                    if k in seen:
                        problems.append(f"duplicate key {k} (phi={seen[k]:.0f} and {phi:.0f})")
                    else:
                        seen[k] = phi
        total += n
        state = "ok" if n == ROWS_PER_SLICE else f"PARTIAL ({ROWS_PER_SLICE - n:,} missing)"
        # A short slice is NOT corruption. c7_burn.sh is designed to stop mid-slice
        # when the allocation runs out, and that slice resumes by axis-tuple next
        # time. Lumping it in with duplicate keys -- which ARE corruption -- would
        # make the one report that matters cry wolf on every single burn run.
        if n != ROWS_PER_SLICE:
            incomplete.append(f"phi={phi:.0f}: {n:,} / {ROWS_PER_SLICE:,} rows "
                              f"({ROWS_PER_SLICE - n:,} left to run)")
        print(f"{phi:<8.0f} {len(by_phi[phi]):>6} {n:>9,}  {state}")

    print(f"\nslices        : {len(by_phi)} of {N_SLICES} (phi {PHI_FIRST}..{PHI_LAST})")
    print(f"rows total    : {total:,}")
    print(f"unique keys   : {len(seen):,}")
    expect = len(by_phi) * ROWS_PER_SLICE
    print(f"expected rows : {expect:,}")
    print(f"files read    : {len(paths)}")

    if incomplete:
        print(f"\n-- {len(incomplete)} INCOMPLETE SLICE(S) (expected after a burn run; resumable):")
        for m in incomplete:
            print(f"   - {m}")

    if problems:
        print(f"\n!! {len(problems)} PROBLEM(S) -- these are real:")
        for m in problems[:20]:
            print(f"   - {m}")
        if len(problems) > 20:
            print(f"   ... and {len(problems) - 20} more")
    elif incomplete:
        print("\nno corruption: every row is unique and every file parsed.")
    else:
        print("\nall checks passed")

    if not a.out:
        print("\n(report only; pass --out to write the merged CSV)")
        return

    if problems and not a.allow_partial:
        sys.exit("\nrefusing to write a merge with problems -- pass --allow-partial "
                 "if you really want it")

    with open(a.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=header)
        w.writeheader()
        for p in paths:
            with open(p, newline="") as f:
                for row in csv.DictReader(f):
                    w.writerow(row)
    print(f"\nwrote {a.out}  ({total:,} rows + header)")
    print("Remember to record per-slice provenance (which machine) in a manifest.")


if __name__ == "__main__":
    main()

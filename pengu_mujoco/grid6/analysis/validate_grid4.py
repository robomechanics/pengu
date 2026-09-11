#!/usr/bin/env python
"""Numeric acceptance test for load5 against the published GRID-4 package.

Recomputes robust-region volume (cells with neighborhood-mean pass >= 0.8)
for all six complete GRID-4 configs x 4 mu and asserts equality with the
table published in results/grid4_report/INDEX.md. On GRID-4's contiguous phi
axis the seam-aware neighborhood and the round-1 roll()+/15 construct are
equivalent, so BOTH modes must hit the same published numbers; a mismatch
means a loader bug, not a data change.

usage: python grid5/analysis/validate_grid4.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load5

# results/grid4_report/INDEX.md, "Robust volume tells a different story"
PUBLISHED = {                      # cfg -> [mu0.1, mu0.3, mu0.5, mu0.7]
    "c1": [35135, 63823, 24163, 11471], "c4": [97468, 101474, 22063, 14224],
    "c2": [26258, 17708, 10398, 4388],  "c5": [60625, 18812, 883, 602],
    "c3": [2948, 1468, 252, 130],       "c6": [3697, 125, 36, 47],
}

fails = 0
for cfg in ["c1", "c2", "c3", "c4", "c5", "c6"]:
    g = load5.load(cfg, rnd="grid4")
    if not g.complete:
        sys.exit(f"{cfg}: not complete — validation needs the full GRID-4 data")
    for legacy in (True, False):
        N = g.nbhd("pass_rate", legacy=legacy)
        got = [int(np.nansum(N[m] >= 0.8)) for m in range(4)]
        tag = "legacy roll/15" if legacy else "seam-aware    "
        ok = got == PUBLISHED[cfg]
        print(f"{cfg} [{tag}] robust volume {got} "
              f"{'== published OK' if ok else f'!= published {PUBLISHED[cfg]}  MISMATCH'}")
        fails += 0 if ok else 1

print("\nPASS — loader reproduces the published GRID-4 robust volumes"
      if fails == 0 else f"\nFAIL — {fails} mismatches")
sys.exit(0 if fails == 0 else 1)

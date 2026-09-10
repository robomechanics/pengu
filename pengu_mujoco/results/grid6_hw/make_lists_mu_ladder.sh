#!/bin/bash
# margin-1 cell lists for c1..c6 at the GRID-5 friction ladder (torso-capped PID sweep)
cd "$(dirname "$0")/../.."
for c in c1 c2 c3 c4 c5 c6; do for mu in 0.1 0.3 0.5 0.7; do
  CONFIG=$c /opt/anaconda3/envs/pengu/bin/python grid6/hw_mask.py --mu $mu --margin 1 2>&1 | grep "margin 1"
done; done
echo LISTS_DONE

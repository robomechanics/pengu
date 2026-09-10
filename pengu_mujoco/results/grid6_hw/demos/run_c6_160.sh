#!/bin/bash
cd "$(dirname "$0")/../../.."
P=/opt/anaconda3/envs/pengu/bin/python
CONFIG=c6 PENGU_MODEL=1.31 BODY_ALPHA=0.22 $P grid6/render_forces.py --cell 1.60/260/115/24/20 --mu 0.12 --hw --balance --kappas 2 > results/grid6_hw/demos/c6_160_mu0.12.log 2>&1
for f in results/grid6_probes/balance_1.6-260-115-24-20_mu012_hw*; do mv "$f" "results/grid6_hw/demos/c6_160_mu0.12$(basename $f | sed 's/balance_1.6-260-115-24-20_mu012_hw//')"; done
echo ALL_DONE

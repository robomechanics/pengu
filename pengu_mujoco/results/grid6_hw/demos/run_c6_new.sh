#!/bin/bash
cd "$(dirname "$0")/../../.."
P=/opt/anaconda3/envs/pengu/bin/python
run() { CONFIG=c6 PENGU_MODEL=1.31 BODY_ALPHA=0.22 $P grid6/render_forces.py --cell $1 --mu $2 --hw --balance --kappas 2 > results/grid6_hw/demos/c6w_mu$2.log 2>&1
  tag=$($P -c "print('-'.join(f'{float(x):g}' for x in '$1'.split('/')))"); m=$(printf "%03d" $($P -c "print(round($2*100))"))
  for f in results/grid6_probes/balance_${tag}_mu${m}_hw*; do mv "$f" "results/grid6_hw/demos/c6_wide_mu$2$(basename $f | sed "s/balance_${tag}_mu${m}_hw//")"; done; }
run 1.65/280/90/28/20 0.12
run 1.60/260/125/16/35 0.45
echo ALL_DONE

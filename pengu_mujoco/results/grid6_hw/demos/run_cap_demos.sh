#!/bin/bash
# cap-only champions (c3 kappa0 pengu1_31, c4 kappa2 pengu1_05_hw_updated), robust + clearance filtered
cd "$(dirname "$0")/../../.."
P=/opt/anaconda3/envs/pengu/bin/python
run() { # cfg model kappa cell mu
  PENGU_MODEL=$2 BODY_ALPHA=0.22 $P grid6/render_forces.py --cell $4 --mu $5 --cap --balance --kappas $3 > results/grid6_hw/demos/cap_$1_mu$5.log 2>&1
  tag=$($P -c "print('-'.join(f'{float(x):g}' for x in '$4'.split('/')))"); m=$(printf "%03d" $($P -c "print(round($5*100))"))
  for f in results/grid6_probes/balance_${tag}_mu${m}_cap*; do mv "$f" "results/grid6_hw/demos/cap_$1_mu$5$(basename $f | sed "s/balance_${tag}_mu${m}_cap//")"; done
}
run c4 pengu1_05_hw_updated 2 1.65/290/95/32/25  0.12
run c4 pengu1_05_hw_updated 2 1.65/260/130/32/40 0.45
run c3 1.31 0 1.25/280/125/24/20 0.12
run c3 1.31 0 1.65/310/130/20/20 0.45
echo ALL_DONE

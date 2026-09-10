#!/bin/bash
# hardware-sweep champions (robust + clearance filtered), hardware layers + feedforward torso, balance overlay
cd "$(dirname "$0")/../../.."
P=/opt/anaconda3/envs/pengu/bin/python
run() { # cfg model kappa cell mu
  CONFIG=$1 PENGU_MODEL=$2 BODY_ALPHA=0.22 $P grid6/render_forces.py --cell $4 --mu $5 --hw --balance --kappas $3 > results/grid6_hw/demos/$1_mu$5.log 2>&1
  tag=$(echo $4 | tr '/' '-'); m=$(printf "%03d" $(python3 -c "print(round($5*100))"))
  for f in results/grid6_probes/balance_${tag}_mu${m}_hw*; do mv "$f" "results/grid6_hw/demos/$1_mu$5_$(basename $f | sed "s/balance_${tag}_mu${m}_hw//;s/^_//")"; done
}
run c1 pengu1_05_hw_updated 0 1.55/260/125/28/25 0.12
run c1 pengu1_05_hw_updated 0 1.65/30/80/32/40   0.45
run c2 pengu1_20_hw_updated 0 1.50/300/100/32/20 0.12
run c2 pengu1_20_hw_updated 0 1.65/270/100/24/30 0.45
run c5 pengu1_20_hw_updated 2 1.65/230/130/32/25 0.12
run c5 pengu1_20_hw_updated 2 1.50/290/90/24/25  0.45
run c6 1.31 2 1.35/50/130/24/25  0.12
run c6 1.31 2 1.60/260/115/20/35 0.45
echo ALL_DONE

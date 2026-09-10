"""torso forward pitch of the eight hardware champions, hardware layers + FF (stance_com.rollout --hw)"""
import os, sys, json, numpy as np
sys.path.insert(0, "grid6"); sys.path.append(".")
CFG = os.environ["CONFIG"]
import stance_com as sc
J = json.load(open("results/grid6_hw/figs/hw_champions_robust_clear.json"))
kappa = 0.0 if CFG in ("c1", "c2") else 2.0
for mu in (0.12, 0.45):
    r0 = J[f"{CFG}_mu{mu}"][0]
    cell = (r0["freq"], r0["hip_phi"], r0["leg_amp"], r0["hip_amp"], r0["hip_off"])
    try:
        r = sc.rollout(cell, mu, kappa, True)
    except SystemExit as e:
        print(f"{CFG} mu{mu} {cell}: {e}"); continue
    tp = np.array(r["tpitch"])
    print(f"{CFG} mu{mu} {cell}: {'FELL %.1fs' % r['fell'] if r['fell'] else 'walked'}  torso pitch mean {tp.mean():+.1f} deg  p5/p95 {np.percentile(tp,5):+.1f}/{np.percentile(tp,95):+.1f}  "
          f"torso roll rms {np.std(r['troll']):.1f}  axis roll rms {np.std(r['aroll']):.1f}", flush=True)

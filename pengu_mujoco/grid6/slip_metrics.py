"""slip_metrics.py — per-physics-step foot-floor contact kinetics for the slip metrics (Ben 2026-09-14).

contact_slip(model, data, foot_geom) -> per foot side ("L", "R"):
    fn   normal force, N (sum over that foot's floor contacts; mj_contactForce, contact frame)
    ft   |tangential (friction) force|, N, vector-summed over the contacts in the world frame
    vt   tangential velocity of the foot at the contact point relative to the floor, world xy, m/s
         (force-weighted mean over the foot's contacts; the floor is static, so this is the foot geom's point
          velocity: mj_objectVelocity of the geom + omega x r, with the normal component removed)
    n    number of contacts
A foot with no contact gets fn = ft = 0, vt = (0, 0).  mu_req = ft / fn where fn > FN_MIN.
The slip distance of a step is the integral of |vt| over the loaded physics steps (stance_com.rollout accumulates it).
"""
import numpy as np
import mujoco

_F6 = np.zeros(6)
_V6 = np.zeros(6)


def contact_slip(model, data, foot_geom):
    out = {s: dict(fn=0.0, ft=np.zeros(3), vt=np.zeros(3), n=0) for s in ("L", "R")}
    nc = data.ncon
    for ci in range(nc):
        c = data.contact[ci]
        if c.geom1 in foot_geom:
            g, s, sign = c.geom1, foot_geom[c.geom1], -1.0        # force acts on geom2 -> flip to act ON the foot
        elif c.geom2 in foot_geom:
            g, s, sign = c.geom2, foot_geom[c.geom2], 1.0
        else:
            continue
        mujoco.mj_contactForce(model, data, ci, _F6)
        fr = c.frame.reshape(3, 3)                                  # rows: normal, tangent1, tangent2 (world)
        fw = sign * (fr.T @ _F6[:3])                                # contact force on the foot, world
        fn = abs(float(_F6[0]))
        ft_vec = fw - fr[0] * float(np.dot(fw, fr[0]))              # tangential part, world
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, g, _V6, 0)
        v_pt = _V6[3:] + np.cross(_V6[:3], c.pos - data.geom_xpos[g])   # velocity of the foot material at the contact point
        v_t = v_pt - fr[0] * float(np.dot(v_pt, fr[0]))
        o = out[s]
        o["ft"] = o["ft"] + ft_vec
        o["vt"] = o["vt"] + fn * v_t
        o["fn"] += fn
        o["n"] += 1
    for s, o in out.items():
        if o["fn"] > 1e-9:
            o["vt"] = o["vt"] / o["fn"]
        o["ft"] = float(np.linalg.norm(o["ft"]))
    return out

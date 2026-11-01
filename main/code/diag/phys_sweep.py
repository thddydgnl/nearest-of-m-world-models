import numpy as np
from imlewm.pusht_sim import load_split, rollout_episode, tee_keypoints, NOMINAL, FRAMESKIP
states, rel, lens = load_split("train")
rng = np.random.default_rng(0)
eps = list(range(0, 400, 20))
def dev(sampler, n=6):
    out, motion = [], []
    for ep in eps:
        L = lens[ep]; n_wm = min(16, L // FRAMESKIP)
        sts = [rollout_episode(states[ep, 0], rel[ep, :L], NOMINAL if k == 0 else sampler(), n_wm=n_wm, render=False)[1] for k in range(n + 1)]
        kp = tee_keypoints(np.stack(sts))
        out.append(np.linalg.norm(kp[1:, -1] - kp[:1, -1], axis=-1).mean(-1))
        motion.append(np.linalg.norm(kp[0, -1] - kp[0, 0], axis=-1).mean())
    out = np.concatenate(out)
    return f"dev median {np.median(out):.1f} p90 {np.percentile(out, 90):.1f} max {out.max():.1f} | nominal motion median {np.median(motion):.1f}"
U = rng.uniform
cfgs = {
 "mass only [0.5,2]": lambda: dict(mass=float(np.exp(U(np.log(.5), np.log(2)))), friction=1.0, damping=0.0),
 "friction only [0.3,1.2]": lambda: dict(mass=1.0, friction=float(U(.3, 1.2)), damping=0.0),
 "damping only [0,0.1]": lambda: dict(mass=1.0, friction=1.0, damping=float(U(0, .1))),
 "damping only [0,0.3]": lambda: dict(mass=1.0, friction=1.0, damping=float(U(0, .3))),
 "mass+friction": lambda: dict(mass=float(np.exp(U(np.log(.5), np.log(2)))), friction=float(U(.3, 1.2)), damping=0.0),
 "mass+friction+damping0.1": lambda: dict(mass=float(np.exp(U(np.log(.5), np.log(2)))), friction=float(U(.3, 1.2)), damping=float(U(0, .1))),
}
for k, f in cfgs.items():
    print(f"{k:28s}", dev(f), flush=True)

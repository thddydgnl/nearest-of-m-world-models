import numpy as np
import imlewm.pusht_sim as ps
from imlewm.pusht_sim import load_split, tee_keypoints, FRAMESKIP

def make_env(phys, state0):
    env = ps.pusht_env.PushTEnv(damping=phys.get("damping", 0.0), block_cog=phys.get("cog"),
                                reset_to_state=np.asarray(state0[:5], np.float64), relative=True,
                                action_scale=100, with_velocity=False, with_target=True, render_size=224)
    env.reset()
    return env
ps.make_env = make_env
states, rel, lens = load_split("train")
rng = np.random.default_rng(0)
env = ps.pusht_env.PushTEnv(); env.reset(); print("default cog", env.block.center_of_gravity, "verts x range", min(v.x for s in env.block.shapes for v in s.get_vertices()), max(v.x for s in env.block.shapes for v in s.get_vertices()), "y range", min(v.y for s in env.block.shapes for v in s.get_vertices()), max(v.y for s in env.block.shapes for v in s.get_vertices()))
c0 = np.array(env.block.center_of_gravity)
eps = list(range(0, 400, 20))
def dev(sampler, n=6):
    out, motion = [], []
    for ep in eps:
        L = lens[ep]; n_wm = min(16, L // FRAMESKIP)
        sts = [ps.rollout_episode(states[ep, 0], rel[ep, :L], {"cog": tuple(c0)} if k == 0 else sampler(), n_wm=n_wm, render=False)[1] for k in range(n + 1)]
        kp = tee_keypoints(np.stack(sts))
        out.append(np.linalg.norm(kp[1:, -1] - kp[:1, -1], axis=-1).mean(-1))
        motion.append(np.linalg.norm(kp[0, -1] - kp[0, 0], axis=-1).mean())
    out = np.concatenate(out)
    return f"dev median {np.median(out):.1f} p75 {np.percentile(out, 75):.1f} p90 {np.percentile(out, 90):.1f} max {out.max():.1f} | nominal motion median {np.median(motion):.1f}"
for R in [10, 20, 30]:
    def s(R=R):
        a = rng.uniform(0, 2 * np.pi); r = R * np.sqrt(rng.uniform())
        return {"cog": tuple(c0 + r * np.array([np.cos(a), np.sin(a)]))}
    print(f"cog disk r={R:2d}  ", dev(s), flush=True)
modes = [c0 + np.array(d) for d in [(-25, 0), (25, 0), (0, -25), (0, 25)]]
print("cog 4 modes r=25 ", dev(lambda: {"cog": tuple(modes[rng.integers(4)])}), flush=True)
print("damping [0,0.03]  ", dev(lambda: {"cog": tuple(c0), "damping": float(rng.uniform(0, .03))}), flush=True)

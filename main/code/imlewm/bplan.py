"""E5 (C5): CEM-MPC on stochastic Push-T with a learned world model.

Tasks come from held-out pusht_b test episodes: start = frame t (context = real frames t-2..t),
goal = real frame t+5 (25 env steps, the DINO-WM Push-T goal range), keeping only goals where the
block actually moves (> 10 px or > 10 deg). The real env replays the task with a CoG drawn from the
episode's CoG mode (fresh jitter), hidden from the planner.
CEM: horizon 5 world-model steps, 10-d normalized actions, N candidates x m_p imagined samples,
cost = mean over samples of the token L2 (RMS) between the final imagined frame and the goal frame
(no variance term). MPC executes the first action (5 env steps), re-encodes the new frame, and
replans, up to `budget` steps. Success: block within 20 px and 20 deg of the goal block pose.
All episodes run in parallel.

usage: python -m imlewm.bplan --run e1_b_imle16_s0 [--nfe 2] [--eps 100] [--seed 0]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from . import gen_pusht as gp
from . import pusht_sim as ps
from .common import DinoEncoder, ds_dir
from .wm import K_CTX, Stats, load_model, predict

RUNS = os.path.expanduser("~/imle-wm/runs")
HG = 5


def tasks(n, seed):
    d = ds_dir("pusht_b")
    m = np.load(f"{d}/test_meta.npz")
    seg, pos, mode = m["seg"], m["pos"], m["cog_mode"]
    ok = []
    for t in range(K_CTX - 1, len(seg) - HG):
        if seg[t - K_CTX + 1] != seg[t + HG]:
            continue
        dpos = np.linalg.norm(pos[t + HG, 2:4] - pos[t, 2:4])
        dang = np.degrees(abs(np.arctan2(np.sin(pos[t + HG, 4] - pos[t, 4]), np.cos(pos[t + HG, 4] - pos[t, 4]))))
        if dpos > 10 or dang > 10:
            ok.append(t)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ok, n, replace=False)), pos, mode


def make_env(state, cog):
    env = ps.pusht_env.PushTEnv(damping=0.0, block_cog=tuple(cog), reset_to_state=np.asarray(state[:5], np.float64),
                                relative=True, action_scale=100, with_velocity=False, with_target=False,
                                render_size=224)
    env.reset()
    return env


@torch.no_grad()
def main(run, nfe, n_eps, seed, T=HG, N=64, m_p=4, iters=3, n_elite=8, budget=10):
    torch.manual_seed(seed)
    model, ck = load_model(f"{RUNS}/{run}/ckpt.pt")
    st = Stats("pusht_b"); enc = DinoEncoder()
    d = ds_dir("pusht_b")
    tok = np.load(f"{d}/test_tokens.npy", mmap_mode="r")
    sel, pos, mode = tasks(n_eps, seed)
    rng = np.random.default_rng(seed + 1)
    envs = [make_env(pos[t], gp.DEFAULT_COG + gp.COG_R * gp.MODE_DIRS[mode[t]] + rng.normal(0, gp.COG_JIT, 2)) for t in sel]
    ctx = st.norm(torch.from_numpy(np.asarray(tok[(sel[:, None] + np.arange(-K_CTX + 1, 1)).reshape(-1)])).cuda()).view(n_eps, K_CTX, 256, 384)
    goal = st.norm(torch.from_numpy(np.asarray(tok[sel + HG])).cuda())
    goal_pose = pos[sel + HG, 2:5]
    status = np.zeros(n_eps, int); steps = np.zeros(n_eps, int); final_err = np.zeros((n_eps, 2))
    t0 = time.time(); plan_time = 0.0
    for it_mpc in range(budget):
        act = np.where(status == 0)[0]
        if len(act) == 0:
            break
        E = len(act)
        tp = time.time()
        mu = torch.zeros(E, T, 10, device="cuda"); sd = torch.ones(E, T, 10, device="cuda")
        for _ in range(iters):
            cand = (mu[:, None] + sd[:, None] * torch.randn(E, N, T, 10, device="cuda")).clamp(-3, 3)
            aa = cand[:, :, None].expand(E, N, m_p, T, 10).reshape(-1, T, 10)
            eid = torch.from_numpy(act).cuda().repeat_interleave(N * m_p)
            costs = []
            for i in range(0, len(aa), 512):
                e = eid[i:i + 512]
                c = ctx[e]
                for h in range(T):
                    s = predict(model, c, aa[i:i + 512, h], 1, nfe, st.sigma_r)[:, 0]
                    c = torch.cat([c[:, 1:], s[:, None]], 1)
                costs.append((c[:, -1] - goal[e]).pow(2).mean((1, 2)).sqrt())
            J = torch.cat(costs).view(E, N, m_p).mean(2)
            el = J.topk(n_elite, dim=1, largest=False).indices
            elite = cand[torch.arange(E)[:, None], el]
            mu, sd = elite.mean(1), elite.std(1) + 0.05
        plan_time += time.time() - tp
        a0 = mu[:, 0].clamp(-3, 3).cpu().numpy().reshape(E, ps.FRAMESKIP, 2)
        frames = []
        for k, e in enumerate(act):
            for j in range(ps.FRAMESKIP):
                envs[e].step(a0[k, j] * ps.ACTION_STD + ps.ACTION_MEAN)
            frames.append(envs[e]._render_frame("rgb_array"))
            steps[e] += 1
        new = st.norm(torch.from_numpy(enc(np.stack(frames))).cuda())
        ctx[act] = torch.cat([ctx[act][:, 1:], new[:, None]], 1)
        for k, e in enumerate(act):
            b = envs[e].block
            dp = np.hypot(b.position.x - goal_pose[e, 0], b.position.y - goal_pose[e, 1])
            da = np.degrees(abs(np.arctan2(np.sin(b.angle - goal_pose[e, 2]), np.cos(b.angle - goal_pose[e, 2]))))
            final_err[e] = (dp, da)
            if dp < 20 and da < 20:
                status[e] = 1
    for env in envs:
        env.close()
    return {"run": run, "nfe": nfe, "seed": seed, "n_eps": n_eps, "success": float(status.mean()),
            "steps_success": float(steps[status == 1].mean()) if status.any() else None,
            "final_pos_err": float(final_err[:, 0].mean()), "final_ang_err": float(final_err[:, 1].mean()),
            "plan_s_per_step": plan_time / max(1, steps.sum() / n_eps), "wall_s": time.time() - t0,
            "cem": dict(T=T, N=N, m_p=m_p, iters=iters, n_elite=n_elite, budget=budget)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--nfe", type=int, default=2)
    ap.add_argument("--eps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--m_p", type=int, default=4)
    a = ap.parse_args()
    if a.nfe >= 20 and a.eps > 10:
        # FM-50 planning is ~50x the cost of a 1-NFE model; it is run for planning time plus a rough
        # success estimate only (decision 2026-10-12), so cap it at 10 episodes
        print(f"capping episodes {a.eps} -> 10 for nfe={a.nfe}", flush=True)
        a.eps = 10
    res = main(a.run, a.nfe, a.eps, a.seed, m_p=a.m_p)
    os.makedirs(f"{RUNS}/e5", exist_ok=True)
    json.dump(res, open(f"{RUNS}/e5/{a.run}_nfe{a.nfe}_s{a.seed}.json", "w"), indent=1)
    print(json.dumps(res))

"""Variance-aware CEM-MPC on the push env (H4).

Start left of the ice, goal right of it; the straight path crosses the high-variance ice next to an
absorbing pit. Cost J = mean_j d_j + lam * spread_j d_j over m_p imagined rollouts per candidate.
--model oracle uses the true simulator (rendered + DINO-encoded) as the world model.

usage: python -m imlewm.plan --model imle_push --lams 0,0.5,1,2
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .common import PIT, DinoEncoder, in_rect, push_step, render
from .probe import load_probe
from .wm import Stats, load_model, predict

RUNS = os.path.expanduser("~/imle-wm/runs")


def tasks(n, rng, task="edge"):
    """mid: start/goal at mid height (ice but little pit risk).
    edge: start/goal just below the pit; the straight path runs through the ice along the pit edge,
    so one upward ice deflection (+-60 deg) drops the ball into the absorbing pit."""
    ylo, yhi = (22, 32) if task == "mid" else (17, 21)
    s = np.stack([rng.uniform(8, 12, n), rng.uniform(ylo, yhi, n)], 1)
    g = np.stack([rng.uniform(52, 56, n), rng.uniform(ylo, yhi, n)], 1)
    return s, g


class SimModel:
    """True dynamics in position space; returns tokens of final states."""

    def __init__(self, enc, st):
        self.enc, self.st = enc, st

    def final_dist(self, pos0, acts, goal, goal_pos, rng, cost):  # pos0 (M,2), acts (M,T,2)
        p = pos0.copy()
        for h in range(acts.shape[1]):
            p = np.stack([push_step(p[i], acts[i, h], rng)[0] for i in range(len(p))])
        if cost == "pos":
            return torch.from_numpy(np.linalg.norm(p - goal_pos, axis=1)).float().cuda()
        fin = self.st.norm(torch.from_numpy(self.enc(render(p, "push"))).cuda())
        return (fin - goal).pow(2).mean((-1, -2)).sqrt()


@torch.no_grad()
def run(model_name, lam, n_eps, seed, T=5, N=64, iters=3, n_elite=8, m_p=8, budget=16, spread="std", chunk=512, cost="pos", task="edge"):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    st = Stats("push")
    enc = DinoEncoder()
    model = None if model_name == "oracle" else load_model(f"{RUNS}/{model_name}/ckpt.pt")[0]
    if model is not None and model.method == "reg":
        m_p = 1  # a deterministic model gives m_p identical rollouts: one gives the same cost, m_p x cheaper
    sim = SimModel(enc, st)
    probe = load_probe("push")
    starts, goals = tasks(n_eps, rng, task)
    goal_tok = st.norm(torch.from_numpy(enc(render(goals, "push"))).cuda())  # (E,N,D)
    goal_pos_t = torch.from_numpy(goals).float().cuda()
    pos = starts.copy()
    ctx = st.norm(torch.from_numpy(enc(render(pos, "push"))).cuda())[:, None].repeat(1, 3, 1, 1)
    status = np.zeros(n_eps, int)  # 0 running, 1 success, -1 pit
    steps_used = np.zeros(n_eps, int)
    env_rng = np.random.default_rng(seed + 1)
    for t in range(budget):
        act = np.where(status == 0)[0]
        if len(act) == 0:
            break
        E = len(act)
        mu = torch.zeros(E, T, 2, device="cuda"); sd = torch.full((E, T, 2), 0.6, device="cuda")
        for it in range(iters):
            cand = (mu[:, None] + sd[:, None] * torch.randn(E, N, T, 2, device="cuda"))
            nrm = cand.norm(dim=-1, keepdim=True).clamp(min=1.0)
            cand = cand / nrm  # |a| <= 1
            aa = cand[:, :, None].expand(E, N, m_p, T, 2).reshape(-1, T, 2)
            eid = torch.from_numpy(act).cuda().repeat_interleave(N * m_p)  # episode of each sample
            ds = []
            for i in range(0, len(aa), chunk):
                e = eid[i:i + chunk]
                if model is None:
                    en = e.cpu().numpy()
                    ds.append(sim.final_dist(pos[en], aa[i:i + chunk].cpu().numpy(), goal_tok[e], goals[en], rng, cost))
                else:
                    c = ctx[e]
                    for h in range(T):
                        s = predict(model, c, aa[i:i + chunk, h], 1, nfe=2, sigma_r=st.sigma_r)[:, 0]
                        c = torch.cat([c[:, 1:], s[:, None]], 1)
                    if cost == "pos":  # probe readout of the imagined final state
                        ds.append((probe(c[:, -1]) - goal_pos_t[e]).norm(dim=-1))
                    else:
                        ds.append((c[:, -1] - goal_tok[e]).pow(2).mean((-1, -2)).sqrt())
            d = torch.cat(ds).view(E, N, m_p)
            if m_p > 1:
                spr = d.std(2) if spread == "std" else d.var(2)
            else:  # one rollout per candidate (deterministic model): no spread; std of 1 sample is NaN
                spr = torch.zeros_like(d[:, :, 0])
            J = d.mean(2) + lam * spr
            el = J.topk(n_elite, dim=1, largest=False).indices
            elite = cand[torch.arange(E)[:, None], el]
            mu, sd = elite.mean(1), elite.std(1) + 0.05
        a0 = mu[:, 0]
        a0 = (a0 / a0.norm(dim=-1, keepdim=True).clamp(min=1.0)).cpu().numpy()
        newpos = np.stack([push_step(pos[e], a0[i], env_rng)[0] for i, e in enumerate(act)])
        pos[act] = newpos
        steps_used[act] += 1
        tok = st.norm(torch.from_numpy(enc(render(newpos, "push"))).cuda())
        ctx[act] = torch.cat([ctx[act][:, 1:], tok[:, None]], 1)
        for i, e in enumerate(act):
            if in_rect(pos[e], PIT):
                status[e] = -1
            elif np.linalg.norm(pos[e] - goals[e]) < 5.0:
                status[e] = 1
    return {"lam": lam, "model": model_name, "n_eps": n_eps, "m_p": m_p, "wall_s": round(time.time() - t_start, 1),
            "success": float((status == 1).mean()), "pit": float((status == -1).mean()),
            "timeout": float((status == 0).mean()), "steps_success": float(steps_used[status == 1].mean()) if (status == 1).any() else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lams", default="0,0.5,1,2")
    ap.add_argument("--eps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spread", default="std", choices=["std", "var"])
    ap.add_argument("--cost", default="pos", choices=["pos", "tok"])
    ap.add_argument("--task", default="edge", choices=["edge", "mid"])
    a = ap.parse_args()
    res = []
    for lam in [float(x) for x in a.lams.split(",")]:
        r = run(a.model, lam, a.eps, a.seed, spread=a.spread, cost=a.cost, task=a.task)
        print(json.dumps(r), flush=True)
        res.append(r)
    os.makedirs(f"{RUNS}/plan", exist_ok=True)
    with open(f"{RUNS}/plan/{a.model}_{a.task}_{a.cost}_{a.spread}_s{a.seed}.json", "w") as f:
        json.dump(res, f, indent=1)

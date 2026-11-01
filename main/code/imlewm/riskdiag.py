"""E13 diagnostic: does a world model estimate the pit risk correctly?

On the push env "edge" task the straight path runs over the ice along the absorbing pit: each ice
step deflects the push by -60/0/+60 deg with equal probability, and the upward deflection drops the
ball into the pit. For n start states of the planning tasks and the straight action sequence toward
the goal (T = 5 steps, |a| = 1), we compare
  true pit probability  : 256 true-simulator rollouts per start
  model pit probability : n_samp imagined rollouts per start (probe readout of each imagined frame;
                          the ball counts as fallen if any imagined frame puts it inside the pit)
and a one-step test from n starts on the ice just below the pit (x in [22, 34], y in [17, 21]) pushed
straight right: one of the three equiprobable deflections lands in the pit, so the true pit share is 1/3.

usage: python -m imlewm.riskdiag --models e13_push_imle16_s0 e13_push_fm_s0 ... [--nfe 2]
"""
import argparse
import json
import os

import numpy as np
import torch

from .common import PIT, DinoEncoder, in_rect, push_step, render
from .plan import tasks
from .probe import load_probe
from .wm import Stats, load_model, predict

RUNS = os.path.expanduser("~/imle-wm/runs")


def in_pit_t(xy):  # (..., 2) tensor of pixel positions
    x0, y0, x1, y1 = PIT
    return (xy[..., 0] >= x0) & (xy[..., 0] < x1) & (xy[..., 1] >= y0) & (xy[..., 1] < y1)


@torch.no_grad()
def main(models, n=200, n_samp=64, n_true=256, T=5, seed=0, nfe=2):
    rng = np.random.default_rng(seed)
    starts, goals = tasks(n, rng, "edge")
    d = goals - starts
    act = d / np.linalg.norm(d, axis=1, keepdims=True)  # straight toward the goal, |a| = 1
    # true simulator
    sim_rng = np.random.default_rng(seed + 7)
    true_pit = np.zeros(n)
    for i in range(n):
        fallen = 0
        for _ in range(n_true):
            p = starts[i].copy()
            for _h in range(T):
                p = push_step(p, act[i], sim_rng)[0]
            fallen += in_rect(p, PIT)
        true_pit[i] = fallen / n_true
    # one-step ice test
    ice = np.stack([rng.uniform(22, 34, n), rng.uniform(17, 21, n)], 1)
    ice_true = np.mean([[in_rect(push_step(q, np.array([1.0, 0.0]), sim_rng)[0], PIT) for _ in range(64)] for q in ice])
    st = Stats("push"); enc = DinoEncoder(); probe = load_probe("push")
    ctx0 = st.norm(torch.from_numpy(enc(render(starts, "push"))).cuda())[:, None].repeat(1, 3, 1, 1)
    ctx_ice = st.norm(torch.from_numpy(enc(render(ice, "push"))).cuda())[:, None].repeat(1, 3, 1, 1)
    a_t = torch.from_numpy(act).float().cuda()
    out = {"n": n, "T": T, "true_pit_mean": float(true_pit.mean()), "ice_true_pit_share": float(ice_true), "models": {}}
    for name in models:
        torch.manual_seed(seed)
        model, _ = load_model(f"{RUNS}/{name}/ckpt.pt")
        model_pit, ice_hits = [], []
        cs = 16  # starts per chunk (16 x n_samp imagined rollouts at a time)
        for i0 in range(0, n, cs):
            c = ctx0[i0:i0 + cs].repeat_interleave(n_samp, 0)
            aa = a_t[i0:i0 + cs].repeat_interleave(n_samp, 0)
            fell = torch.zeros(len(c), dtype=torch.bool, device="cuda")
            for h in range(T):
                s_ = predict(model, c, aa, 1, nfe, st.sigma_r)[:, 0]
                fell |= in_pit_t(probe(s_))
                c = torch.cat([c[:, 1:], s_[:, None]], 1)
            model_pit.append(fell.view(-1, n_samp).float().mean(1).cpu())
            ci = ctx_ice[i0:i0 + cs].repeat_interleave(n_samp, 0)
            right = torch.tensor([[1.0, 0.0]], device="cuda").expand(len(ci), 2)
            s1 = predict(model, ci, right, 1, nfe, st.sigma_r)[:, 0]
            ice_hits.append(in_pit_t(probe(s1)).float().cpu())
        model_pit = torch.cat(model_pit).numpy()
        ice_model = torch.cat(ice_hits).mean().item()
        out["models"][name] = {
            "method": model.method, "nfe": nfe if model.method == "fm" else 1,
            "model_pit_mean": float(model_pit.mean()),
            "abs_err_pit": float(np.abs(model_pit - true_pit).mean()),
            "corr_pit": float(np.corrcoef(model_pit, true_pit)[0, 1]) if model_pit.std() > 0 else None,
            "ice_one_step_pit_share": ice_model,
        }
        print(name, out["models"][name], flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--nfe", type=int, default=2)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="riskdiag")
    a = ap.parse_args()
    res = main(a.models, n=a.n, nfe=a.nfe)
    print("true pit mean", res["true_pit_mean"], "ice one-step true pit share", res["ice_true_pit_share"])
    os.makedirs(f"{RUNS}/plan", exist_ok=True)
    json.dump(res, open(f"{RUNS}/plan/{a.out}.json", "w"), indent=1)

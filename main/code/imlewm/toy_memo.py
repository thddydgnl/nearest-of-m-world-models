"""Toy check of outcome memorization under the energy score vs nearest-of-m (E15, 2026-10-31).

TwoBranch from ESP (arXiv 2610.07696, sec. IV-B): u ~ U[0.05, 1], y = s sqrt(u) + 0.08 eps, s = +-1 equiprobable.
The context also carries an appearance vector a ~ N(0, I_16) that does not affect y, as in bounce_div.
  fresh:  every batch draws new contexts (nothing to memorize; ESP's setting)
  N:      N training contexts, each with its one observed outcome, revisited for many epochs (A2-div's setting)
Generator MLP(u, a, z), z ~ N(0, I_4). Objectives: energy score (beta = 1, m samples, the train.py estimator) or
nearest-of-m (squared error of the closest sample, i.e. IMLE / XM). Reported on 1000 fresh test contexts x 32
samples: ESP's Match rate (within 0.24 of a branch centre), two-branch coverage (both centres hit) and the share of
samples on the + branch; on training contexts: the share of samples on the observed branch (0.5 = no memorization).

usage: python -m imlewm.toy_memo [--steps 15000] [--seeds 0 1] [--n_train 0 1024] [--methods es8 nn16] [--a_dim 16]
"""
import argparse
import json

import torch
import torch.nn as nn

A_DIM, Z_DIM = 16, 4


def sample(n, g):
    u = torch.rand(n, 1, generator=g) * 0.95 + 0.05
    s = torch.randint(0, 2, (n, 1), generator=g).float() * 2 - 1
    y = s * u.sqrt() + 0.08 * torch.randn(n, 1, generator=g)
    return torch.cat([u, torch.randn(n, A_DIM, generator=g)], 1), y, s


class Gen(nn.Module):
    def __init__(self, w=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1 + A_DIM + Z_DIM, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(),
                                 nn.Linear(w, w), nn.SiLU(), nn.Linear(w, 1))

    def forward(self, c, m):  # (B, 1 + A_DIM) -> (B, m, 1)
        c = c.repeat_interleave(m, 0)
        return self.net(torch.cat([c, torch.randn(len(c), Z_DIM)], 1)).view(-1, m, 1)


def es_loss(r, y, beta=1.0):
    """Same estimator as train.py --method es: RMS distances, unbiased pairwise term."""
    m = r.shape[1]
    off = ~torch.eye(m, dtype=torch.bool)
    d_fit = (r - y[:, None]).pow(2).mean(-1)
    sq = r.pow(2).mean(-1)
    d_pair = (sq[:, :, None] + sq[:, None, :] - 2 * torch.bmm(r, r.transpose(1, 2)) / r.shape[-1]).clamp_min(0)
    return ((d_fit + 1e-8).pow(beta / 2).mean(1) - 0.5 * (d_pair[:, off] + 1e-8).pow(beta / 2).mean(1)).mean()


def nn_loss(r, y):
    return (r - y[:, None]).pow(2).mean(-1).min(1).values.mean()


@torch.no_grad()
def evaluate(model, c, s, n=32):
    x = model(c, n)[..., 0]
    ctr = c[:, :1].sqrt()
    near = torch.minimum((x - ctr).abs(), (x + ctr).abs()) <= 0.24
    hit_p = ((x - ctr).abs() <= 0.24).any(1)
    hit_m = ((x + ctr).abs() <= 0.24).any(1)
    out = {"match": near.float().mean().item(), "cov2": (hit_p & hit_m).float().mean().item(),
           "share_plus": (x > 0).float().mean().item()}
    if s is not None:
        out["memo"] = ((x > 0).float() == (s > 0).float()).float().mean().item()
    return out


def run(method, m, n_train, seed, steps, bs=256, lr=2e-4):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = Gen()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if n_train:
        c_tr, y_tr, s_tr = sample(n_train, g)
    for _ in range(steps):
        if n_train:
            i = torch.randint(0, n_train, (bs,), generator=g)
            c, y = c_tr[i], y_tr[i]
        else:
            c, y, _ = sample(bs, g)
        r = model(c, m)
        loss = es_loss(r, y) if method == "es" else nn_loss(r, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    c_te, _, _ = sample(1000, torch.Generator().manual_seed(10_000 + seed))
    res = {"method": method, "m": m, "n_train": n_train or "fresh", "a_dim": A_DIM, "seed": seed,
           "test": evaluate(model, c_te, None)}
    if n_train:
        k = min(n_train, 1000)
        res["train"] = evaluate(model, c_tr[:k], s_tr[:k])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_train", type=int, nargs="+", default=[0, 1024], help="0 = fresh contexts every batch")
    ap.add_argument("--methods", nargs="+", default=["es8", "es16", "nn8", "nn16"])
    ap.add_argument("--a_dim", type=int, default=A_DIM, help="appearance dims (0 = contexts identified by u only)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    A_DIM = a.a_dim
    torch.set_num_threads(4)
    rows = []
    for n_train in a.n_train:
        for method, m in [(s[:2], int(s[2:])) for s in a.methods]:
            for seed in a.seeds:
                rows.append(run(method, m, n_train, seed, a.steps))
                print(json.dumps(rows[-1]), flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)

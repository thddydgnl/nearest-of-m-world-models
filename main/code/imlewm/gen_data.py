"""Generate synthetic datasets and precompute DINOv2 patch tokens.

usage: python -m imlewm.gen_data --name bounce_k4 | push
"""
import argparse
import os

import numpy as np
import torch

from .common import (DinoEncoder, bounce_branches, bounce_episode, ds_dir, push_branches,
                     push_episode, random_action, render, sample_app, save_json)

K_CTX = 3


def encode_to_memmap(enc, pos, kind, path, app=None):
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(pos), 256, 384))
    for i in range(0, len(pos), 4096):
        mm[i:i + 4096] = enc(render(pos[i:i + 4096], kind, None if app is None else app[i:i + 4096]))
    mm.flush()
    return mm


def gen_bounce(K, n_train_eps, n_test_eps, T, out, enc, seed, diverse=False):
    rng = np.random.default_rng(seed)
    for split, n_eps in [("train", n_train_eps), ("test", n_test_eps)]:
        P, N, H, S, A = [], [], [], [], []
        for e in range(n_eps):
            app = sample_app(rng) if diverse else None
            kw = {} if app is None else dict(speed=app[7], r=app[6])
            p, n, h = bounce_episode(K, T, rng, **kw)
            P.append(p); N.append(n); H.append(h); S.append(np.full(T, e))
            A.append(np.repeat((app if diverse else np.zeros(22, np.float32))[None], T, 0))
        P, N, H, S, A = map(np.concatenate, (P, N, H, S, A))
        toks = encode_to_memmap(enc, P, "bounce", f"{out}/{split}_tokens.npy", A if diverse else None)
        np.savez(f"{out}/{split}_meta.npz", pos=P, normal=N, hit=H, seg=S, app=A,
                 act=np.zeros((len(P), 2), np.float32), stoch=H)
        print(split, len(P), "frames, stochastic transitions:", H.mean())
    # stochastic eval contexts: frame t is a wall frame, t-2..t+1 in same episode
    m = np.load(f"{out}/test_meta.npz")
    seg, hit, A = m["seg"], m["hit"], m["app"]
    idx = [t for t in range(2, len(seg) - 1) if hit[t] and seg[t - 2] == seg[t + 1]]
    idx = np.array(rng.permutation(idx)[:400])
    kw = lambda t: dict(speed=A[t, 7], r=A[t, 6]) if diverse else {}
    br = np.stack([bounce_branches(m["pos"][t], m["normal"][t], K, **kw(t)) for t in idx])  # (n,K,2)
    br_app = np.repeat(A[idx], K, 0) if diverse else None
    br_tok = enc(render(br.reshape(-1, 2), "bounce", br_app)).reshape(len(idx), K, 256, 384)
    np.save(f"{out}/stoch_branch_tokens.npy", br_tok)
    det = [t for t in range(2, len(seg) - 1) if not hit[t] and seg[t - 2] == seg[t + 1]]
    det = np.array(rng.permutation(det)[:400])
    np.savez(f"{out}/eval_idx.npz", stoch=idx, stoch_branch_pos=br, det=det)


def gen_push(n_train_eps, n_test_eps, T, out, enc, seed, hmax=16):
    rng = np.random.default_rng(seed)
    for split, n_eps in [("train", n_train_eps), ("test", n_test_eps)]:
        P, A, S, M = [], [], [], []
        off = 0
        for e in range(n_eps):
            p, a, s, mo = push_episode(T, rng)
            P.append(p); A.append(a); S.append(s + off); M.append(mo); off = S[-1].max() + 1
        P, A, S, M = map(np.concatenate, (P, A, S, M))
        encode_to_memmap(enc, P, "push", f"{out}/{split}_tokens.npy")
        np.savez(f"{out}/{split}_meta.npz", pos=P, act=A.astype(np.float32), seg=S, mode=M,
                 stoch=(M >= 0))
        print(split, len(P), "frames, segments:", S.max() + 1)
    m = np.load(f"{out}/test_meta.npz")
    seg, pos, act = m["seg"], m["pos"], m["act"]
    # one-step eval contexts with realized + counterfactual action (both nonzero)
    idx = [t for t in range(2, len(seg) - 1) if seg[t - 2] == seg[t + 1] and np.linalg.norm(act[t]) > 0]
    idx = np.array(rng.permutation(idx)[:400])
    acf = []
    for _ in idx:
        a = random_action(rng)
        while np.linalg.norm(a) == 0:
            a = random_action(rng)
        acf.append(a)
    acf = np.array(acf, np.float32)
    Kmax = 3
    def branches(ps, acts):
        bp = np.zeros((len(ps), Kmax, 2)); bv = np.zeros((len(ps), Kmax), bool)
        for i, (p, a) in enumerate(zip(ps, acts)):
            b = push_branches(p, a); bp[i, :len(b)] = b; bv[i, :len(b)] = True
        bt = enc(render(bp.reshape(-1, 2), "push")).reshape(len(ps), Kmax, 256, 384)
        return bp, bv, bt
    bp, bv, bt = branches(pos[idx], act[idx])
    bpc, bvc, btc = branches(pos[idx], acf)
    np.save(f"{out}/branch_tokens.npy", bt); np.save(f"{out}/branch_tokens_cf.npy", btc)
    # horizon sequences
    hz = [t for t in range(2, len(seg) - hmax) if seg[t - 2] == seg[t + hmax]]
    hz = np.array(rng.permutation(hz)[:200])
    np.savez(f"{out}/eval_idx.npz", step=idx, act_cf=acf, branch_pos=bp, branch_valid=bv,
             branch_pos_cf=bpc, branch_valid_cf=bvc, horizon=hz, hmax=hmax)


def compute_stats(out, n_sub=6000, seed=0):
    toks = np.load(f"{out}/train_tokens.npy", mmap_mode="r")
    m = np.load(f"{out}/train_meta.npz")
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(len(toks) - 1, n_sub, replace=False))
    x = torch.from_numpy(np.asarray(toks[sub])).cuda().float()  # (n,256,384)
    mean = x.mean((0, 1)); std = x.std((0, 1)) + 1e-4
    xn = ((x - mean) / std).reshape(-1, 384)
    cov = (xn.T @ xn) / len(xn)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    same = m["seg"][sub] == m["seg"][sub + 1]
    x1 = torch.from_numpy(np.asarray(toks[sub + 1])).cuda().float()
    r = ((x1 - mean) / std - (x - mean) / std)[torch.from_numpy(same).cuda()]
    sigma_r = r.std().item()
    rms = r.pow(2).mean((1, 2)).sqrt()
    torch.save({"mean": mean.cpu(), "std": std.cpu(), "pca": evecs.cpu(), "evals": evals.cpu(),
                "sigma_r": sigma_r, "resid_rms_median": rms.median().item()}, f"{out}/stats.pt")
    frac = (evals.cumsum(0) / evals.sum()).cpu().numpy()
    print("sigma_r", sigma_r, "resid rms median", rms.median().item(),
          "PCA energy@32/64/128:", frac[31], frac[63], frac[127])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train_eps", type=int, default=1000)
    ap.add_argument("--test_eps", type=int, default=250)
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = ds_dir(a.name)
    os.makedirs(out, exist_ok=True)
    enc = DinoEncoder()
    if a.name.startswith("bounce_k"):
        K = int(a.name.split("_k")[1])
        gen_bounce(K, a.train_eps, a.test_eps, a.T, out, enc, a.seed + K)
    elif a.name.startswith("bounce_div_k"):
        import re
        from . import common
        mk = re.match(r"bounce_div_k(\d+)(?:_p(\d+))?$", a.name)
        K = int(mk.group(1))
        if mk.group(2):  # asymmetric modes: mode 0 (first bounce angle) has probability p
            p0 = int(mk.group(2)) / 100.0
            common.MODE_P = [p0] + [(1 - p0) / (K - 1)] * (K - 1)
        gen_bounce(K, a.train_eps, a.test_eps, a.T, out, enc, a.seed + 50 + K, diverse=True)
    elif a.name == "push":
        gen_push(a.train_eps, a.test_eps, a.T, out, enc, a.seed + 100)
    compute_stats(out)
    save_json(vars(a), f"{out}/gen_args.json")

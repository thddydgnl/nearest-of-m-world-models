"""Validity checks for the verdict.

1. Coverage with an independent decoder: nearest neighbour in a bank of real tokens (PCA-128 space)
   instead of the position probe, which sees generated tokens out of distribution.
2. Bootstrap 95% CIs over evaluation contexts.
3. Paired bootstrap of N0 vs N1 rollout errors on the same horizon sequences.

usage: python -m imlewm.check --bounce reg_k4 fm_k4:20 imle_k4 ... --push imle_push imle_push_n1
"""
import argparse
import json
import os

import numpy as np
import torch

from .common import ds_dir
from .evaluate import ctx_tokens, unique_branches
from .probe import load_probe
from .wm import Stats, load_model, predict, rms, rollout

RUNS = os.path.expanduser("~/imle-wm/runs")


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


class NNDecoder:
    def __init__(self, data, st, n_bank=12000, d=128, seed=0):
        d_ = ds_dir(data)
        tok = np.load(f"{d_}/train_tokens.npy", mmap_mode="r")
        pos = np.load(f"{d_}/train_meta.npz")["pos"]
        idx = np.sort(np.random.default_rng(seed).choice(len(tok), n_bank, replace=False))
        self.P = st.pca[:, :d]
        feats = []
        for i in range(0, n_bank, 1000):
            feats.append(self.proj(st.norm(torch.from_numpy(tok[idx[i:i + 1000]]).cuda())))
        self.bank = torch.cat(feats)
        self.bn = self.bank.pow(2).sum(1)
        self.pos = torch.from_numpy(pos[idx]).float().cuda()

    def proj(self, x):
        return (x @ self.P).flatten(1)

    def __call__(self, x):
        f = self.proj(x)
        d = f.pow(2).sum(1, keepdim=True) + self.bn[None] - 2 * f @ self.bank.T
        return self.pos[d.argmin(1)]


def balance(ps, bp, tau_cap=4.0):
    """Mode-frequency histogram of valid samples (nearest branch within tau)."""
    Ku = len(bp)
    dp = (ps[:, None] - bp[None]).norm(dim=-1)
    sep = ((bp[:, None] - bp[None]).norm(dim=-1) + torch.eye(Ku, device=bp.device) * 1e9).min(1).values
    tau = torch.clamp(0.45 * sep, max=tau_cap)
    j = dp.argmin(1)
    ok = dp[torch.arange(len(ps)), j] < tau[j]
    return torch.bincount(j[ok], minlength=Ku).cpu().numpy()


def cov_from_pos(ps, bp):
    """ps (n,2) decoded samples, bp (Ku,2) unique branch positions -> coverage, precision"""
    Ku = len(bp)
    dp = (ps[:, None] - bp[None]).norm(dim=-1)
    if Ku > 1:
        sep = ((bp[:, None] - bp[None]).norm(dim=-1) + torch.eye(Ku, device=bp.device) * 1e9).min(1).values
    else:
        sep = torch.full((1,), float("inf"), device=bp.device)
    tau = torch.clamp(0.45 * sep, max=4.0)
    j = dp.argmin(1)
    ok = dp[torch.arange(len(ps)), j] < tau[j]
    return len(set(j[ok].tolist())) / Ku, ok.float().mean().item()


@torch.no_grad()
def check_bounce(spec, n=16):
    run, _, nfe = spec.partition(":")
    nfe = int(nfe or 20)
    torch.manual_seed(123)
    model, ck = load_model(f"{RUNS}/{run}/ckpt.pt")
    data = ck["args"]["data"]
    d = ds_dir(data); st = Stats(data); probe = load_probe(data); nn = NNDecoder(data, st)
    test = np.load(f"{d}/test_tokens.npy", mmap_mode="r")
    ev = np.load(f"{d}/eval_idx.npz")
    idx, bpos = ev["stoch"], ev["stoch_branch_pos"]
    rows, hist = [], []
    for i0 in range(0, len(idx), 40):
        ii = idx[i0:i0 + 40]
        S = predict(model, ctx_tokens(test, ii, st), torch.zeros(len(ii), 2, device="cuda"), n, nfe, st.sigma_r)
        for c in range(len(ii)):
            keep = unique_branches(bpos[i0 + c], np.ones(bpos.shape[1], bool))
            bp = torch.as_tensor(bpos[i0 + c][keep], device="cuda", dtype=torch.float32)
            cp, pp = cov_from_pos(probe(S[c]), bp)
            cn, pn = cov_from_pos(nn(S[c]), bp)
            rows.append((cp, pp, cn, pn))
            if len(keep) == bpos.shape[1]:  # contexts where all K modes are distinct
                hist.append(balance(probe(S[c]), bp))
    r = np.array(rows)
    # pooled valid-sample frequency per mode index (modes ordered by bounce angle) and
    # mean per-context total-variation distance to uniform
    H = np.array(hist)
    pooled = H.sum(0) / max(H.sum(), 1)
    tv = [0.5 * np.abs(h / h.sum() - 1 / len(h)).sum() for h in H if h.sum() > 0]
    return {"cov_probe": boot_ci(r[:, 0]), "prec_probe": boot_ci(r[:, 1]),
            "cov_nn": boot_ci(r[:, 2]), "prec_nn": boot_ci(r[:, 3]),
            "probe_nn_agree_cov": float(np.corrcoef(r[:, 0], r[:, 2])[0, 1]),
            "mode_freq_pooled": pooled.round(3).tolist(), "tv_to_uniform": boot_ci(tv)}


@torch.no_grad()
def check_push(runs, n=16):
    data = "push"
    d = ds_dir(data); st = Stats(data); probe = load_probe(data)
    test = np.load(f"{d}/test_tokens.npy", mmap_mode="r")
    meta = np.load(f"{d}/test_meta.npz")
    ev = np.load(f"{d}/eval_idx.npz")
    hz, hmax = ev["horizon"], int(ev["hmax"])
    per = {}
    for run in runs:
        torch.manual_seed(123)
        model, _ = load_model(f"{RUNS}/{run}/ckpt.pt")
        bt, bp = [], []
        for i0 in range(0, len(hz), 25):
            ii = hz[i0:i0 + 25]
            acts = torch.from_numpy(meta["act"][(ii[:, None] + np.arange(hmax)[None])]).cuda()
            R = rollout(model, ctx_tokens(test, ii, st), acts, n, 20, st.sigma_r)
            gfi = (ii[:, None] + np.arange(1, hmax + 1)[None]).reshape(-1)
            gt = st.norm(torch.from_numpy(test[gfi]).cuda()).view(len(ii), hmax, 256, 384)
            gp = torch.from_numpy(meta["pos"][gfi]).float().cuda().view(len(ii), hmax, 2)
            bt.append(rms(R, gt[:, None]).min(1).values.cpu())
            pp = probe(R.flatten(0, 2)).view(len(ii), n, hmax, 2)
            bp.append((pp - gp[:, None]).norm(dim=-1).min(1).values.cpu())
        per[run] = {"tok": torch.cat(bt).numpy(), "pos": torch.cat(bp).numpy()}
    out = {}
    a, b = runs[0], runs[1]
    for k in ["tok", "pos"]:
        for h in [1, 8, 16]:
            diff = per[b][k][:, h - 1] - per[a][k][:, h - 1]  # >0 means b worse
            out[f"{k}_h{h}"] = {a: boot_ci(per[a][k][:, h - 1]), b: boot_ci(per[b][k][:, h - 1]),
                                f"{b}-{a}": boot_ci(diff)}
    return out


@torch.no_grad()
def novelty(data, n_train=20000, n_test=400, d=64, seed=0):
    """How novel are test contexts? NN distance (PCA-d of last context frame) from test stochastic
    contexts to train frames, relative to train->train NN distance. ~1 = test looks like train."""
    dd = ds_dir(data); st = Stats(data)
    tr = np.load(f"{dd}/train_tokens.npy", mmap_mode="r"); te = np.load(f"{dd}/test_tokens.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    ti = np.sort(rng.choice(len(tr), n_train, replace=False))
    P = st.pca[:, :d]
    f = lambda x: (st.norm(torch.from_numpy(np.asarray(x)).cuda()) @ P).flatten(1)
    bank = torch.cat([f(tr[ti[i:i + 2000]]) for i in range(0, n_train, 2000)])
    q = f(te[np.load(f"{dd}/eval_idx.npz")["stoch"][:n_test]])
    dq = torch.cdist(q, bank).min(1).values
    sub = bank[:2000]
    dt = torch.cdist(sub, bank); dt[torch.arange(2000), torch.arange(2000)] = 1e9
    dt = dt.min(1).values
    return {"test_nn": float(dq.median()), "train_nn": float(dt.median()), "ratio": float(dq.median() / dt.median())}


@torch.no_grad()
def memorization(run, n_ctx=300, n=16, seed=0):
    torch.manual_seed(123)
    model, ck = load_model(f"{RUNS}/{run}/ckpt.pt")
    return memo_model(model, ck["args"]["data"], n_ctx, n, seed)


@torch.no_grad()
def memo_model(model, data, n_ctx=300, n=16, seed=0, nfe=None):
    """Train vs test stochastic contexts: coverage, and share of valid samples landing on the mode
    that was actually realized in the data. Memorization shows as realized-share >> 1/K on train
    contexts with low coverage, versus ~1/K on test contexts."""
    from .common import bounce_branches
    was_training = model.training
    model.eval()
    import re
    K = int(re.search(r"_k(\d+)", data).group(1))  # also bounce_div_k<K>_p<pct> (asymmetric modes)
    d = ds_dir(data); st = Stats(data); probe = load_probe(data)
    diverse = "div" in data
    out = {}
    for split in ["train", "test"]:
        tok = np.load(f"{d}/{split}_tokens.npy", mmap_mode="r")
        m = np.load(f"{d}/{split}_meta.npz")
        seg, hit, pos, nrm = m["seg"], m["hit"], m["pos"], m["normal"]
        app = m["app"] if "app" in m.files else None
        idx = np.array([t for t in range(2, len(seg) - 1) if hit[t] and seg[t - 2] == seg[t + 1]])
        idx = np.random.default_rng(seed).permutation(idx)[:n_ctx]
        cov, real_share, prec, share0, real0 = [], [], [], [], []
        for i0 in range(0, len(idx), 50):
            ii = idx[i0:i0 + 50]
            nfe_ = nfe if nfe is not None else (1 if model.method == "sc" else 20)
            S = predict(model, ctx_tokens(tok, ii, st), torch.zeros(len(ii), 2, device="cuda"), n, nfe_, st.sigma_r)
            for c, t in enumerate(ii):
                kw = dict(speed=app[t, 7], r=app[t, 6]) if diverse else {}
                bpos = bounce_branches(pos[t], nrm[t], K, **kw)
                keep = unique_branches(bpos, np.ones(K, bool))
                if len(keep) < K:
                    continue
                bp = torch.as_tensor(bpos, device="cuda", dtype=torch.float32)
                ps = probe(S[c])
                cv, pr = cov_from_pos(ps, bp)
                h = balance(ps, bp)
                realized = int(np.argmin(np.linalg.norm(bpos - pos[t + 1], axis=1)))
                cov.append(cv); prec.append(pr)
                if h.sum() > 0:
                    real_share.append(h[realized] / h.sum())
                    share0.append(h[0] / h.sum())  # model's share of mode 0 (first bounce angle)
                real0.append(float(realized == 0))
        out[split] = {"cov": boot_ci(cov), "prec": boot_ci(prec), "realized_share": boot_ci(real_share),
                      "share0": boot_ci(share0), "data_freq0": float(np.mean(real0)) if real0 else None,
                      "chance": 1.0 / K, "n": len(cov)}
    model.train(was_training)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bounce", nargs="*", default=[])
    ap.add_argument("--push", nargs="*", default=[])
    ap.add_argument("--novelty", nargs="*", default=[])
    ap.add_argument("--memo", nargs="*", default=[])
    ap.add_argument("--out", default="check")
    a = ap.parse_args()
    res = {}
    for r_ in a.memo:
        res["memo_" + r_] = memorization(r_)
        print("memo", r_, json.dumps(res["memo_" + r_]), flush=True)
    for dname in a.novelty:
        res["novelty_" + dname] = novelty(dname)
        print("novelty", dname, res["novelty_" + dname], flush=True)
    for s in a.bounce:
        res[s] = check_bounce(s)
        print(s, json.dumps(res[s]), flush=True)
    if a.push:
        res["push"] = check_push(a.push)
        print(json.dumps(res["push"], indent=1), flush=True)
    os.makedirs(f"{RUNS}/check", exist_ok=True)
    json.dump(res, open(f"{RUNS}/check/{a.out}.json", "w"), indent=1)

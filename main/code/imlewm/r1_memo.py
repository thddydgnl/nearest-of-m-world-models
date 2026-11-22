"""Revision r1: memorization measurements for one checkpoint at one NFE (DESIGN_r1.md 1 C2, 2a, 4b, 4c).

On n_ctx training and n_ctx held-out wall contexts (the selection of check.memo_model) with n samples each:
  (i)   index: among samples decoded within the tolerance of a mode, the share on the realized mode
        (exactly check.memo_model: same contexts, same sampling calls, so the same samples for the same torch seed)
  (ii)  tolerance-free index: every decoded sample is assigned to its nearest branch position
  (iii) excess relative to the held-out index: (train - held-out) / (1 - held-out), and the plain difference
  (iv)  readout-free gap: mean over contexts of the smallest token RMS distance between a sample and the realized
        next frame (standardized tokens), training minus held-out (negative = closer on training contexts)
Works for arc datasets (bounce_div_k<K>_arc<deg>) through common.set_arc_from_name.

usage: python -m imlewm.r1_memo --runs e2_k4_m4_s0 ... [--ckpt ckpt] [--nfe 50] [--n_ctx 400 --n 64]
           [--seed 0 --torch_seed 123] --out /workspace/nom/out/b_grid [--validate]
"""
import argparse
import os
import time

import numpy as np
import torch

from . import r1_util
from .check import balance, boot_ci, cov_from_pos, memo_model
from .common import ds_dir
from .evaluate import unique_branches
from .probe import load_probe
from .wm import Stats, load_model, predict, rms

_CACHE = {}


def split_data(data, split, n_ctx, seed):
    """Contexts of check.memo_model and their frames t-2..t+1 (fp16, CPU), cached per dataset and split."""
    key = (data, split, n_ctx, seed)
    if key not in _CACHE:
        meta = r1_util.load_meta(data, split)
        idx = r1_util.memo_contexts(meta, n_ctx, seed)
        tok = np.load(f"{ds_dir(data)}/{split}_tokens.npy", mmap_mode="r")
        order = np.argsort(idx)  # read the memmap in file order, then restore the permutation
        fr = r1_util.frames(tok, idx[order], [-2, -1, 0, 1])
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        _CACHE[key] = {"idx": idx, "meta": meta, "frames": fr[torch.from_numpy(inv)]}
    return _CACHE[key]


def boot_diff(a, b, n=2000, seed=0):
    """Mean of a minus mean of b with a bootstrap 95% interval (independent resampling of the two sets)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    mb = b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    d = ma - mb
    return [float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


@torch.no_grad()
def measure(model, data, n_ctx=400, n=64, nfe=None, seed=0, keep_per_context=True):
    was_training = model.training
    model.eval()
    K = r1_util.set_geometry(data)
    st = Stats(data); probe = load_probe(data)
    nfe_ = nfe if nfe is not None else (1 if model.method == "sc" else 20)
    out = {"K": K, "nfe_used": nfe_ if model.method in ("fm", "mf", "imf", "sc") else 1}
    for split in ["train", "test"]:
        C = split_data(data, split, n_ctx, seed)
        idx, meta = C["idx"], C["meta"]
        cov, prec, real_share, share0, real0 = [], [], [], [], []
        free_share, free_share0, dmin_kept, dmin_all, ctx_t = [], [], [], [], []
        skipped = 0
        for i0 in range(0, len(idx), 50):
            ii = idx[i0:i0 + 50]
            fr = C["frames"][i0:i0 + 50]
            ctx = st.norm(fr[:, :3].cuda())
            S = predict(model, ctx, torch.zeros(len(ii), 2, device="cuda"), n, nfe_, st.sigma_r)
            gt = st.norm(fr[:, 3].cuda())
            dmin = rms(S, gt[:, None]).min(1).values.cpu().numpy()  # (b,) smallest sample-to-realized RMS
            dmin_all += dmin.tolist()
            for c, t in enumerate(ii):
                bpos = r1_util.branches(data, meta, t, K)
                keep = unique_branches(bpos, np.ones(K, bool))
                if len(keep) < K:
                    skipped += 1
                    continue
                bp = torch.as_tensor(bpos, device="cuda", dtype=torch.float32)
                ps = probe(S[c])
                cv, pr = cov_from_pos(ps, bp)
                h = balance(ps, bp)
                realized = r1_util.realized_mode(bpos, meta, t)
                cov.append(cv); prec.append(pr)
                if h.sum() > 0:
                    real_share.append(h[realized] / h.sum())
                    share0.append(h[0] / h.sum())
                real0.append(float(realized == 0))
                jf = (ps[:, None] - bp[None]).norm(dim=-1).argmin(1)
                hf = torch.bincount(jf, minlength=K).cpu().numpy()
                free_share.append(hf[realized] / hf.sum()); free_share0.append(hf[0] / hf.sum())
                dmin_kept.append(float(dmin[c])); ctx_t.append(int(t))
        o = {"cov": boot_ci(cov), "prec": boot_ci(prec), "realized_share": boot_ci(real_share),
             "share0": boot_ci(share0), "data_freq0": float(np.mean(real0)) if real0 else None,
             "chance": 1.0 / K, "n": len(cov), "n_valid": len(real_share), "n_skipped_merged": skipped,
             "free_share": boot_ci(free_share), "free_share0": boot_ci(free_share0),
             "min_dist": boot_ci(dmin_kept), "min_dist_allctx": boot_ci(dmin_all)}
        if keep_per_context:
            o["per_context"] = {"t": ctx_t, "min_dist": np.round(dmin_kept, 5).tolist(),
                                "free_share": np.round(free_share, 5).tolist()}
        o["_raw"] = {"real_share": real_share, "free_share": free_share, "min_dist": dmin_kept}
        out[split] = o
    tr, te = out["train"], out["test"]
    it, ih = tr["realized_share"][0], te["realized_share"][0]
    ft, fh = tr["free_share"][0], te["free_share"][0]
    out["derived"] = {
        "index_train": it, "index_heldout": ih,
        "excess_chance": (it - 1 / K) / (1 - 1 / K),
        "excess_heldout": (it - ih) / (1 - ih), "diff_heldout": boot_diff(tr["_raw"]["real_share"], te["_raw"]["real_share"]),
        "free_index_train": ft, "free_index_heldout": fh,
        "free_excess_chance": (ft - 1 / K) / (1 - 1 / K), "free_excess_heldout": (ft - fh) / (1 - fh),
        "free_diff_heldout": boot_diff(tr["_raw"]["free_share"], te["_raw"]["free_share"]),
        "gap_min_dist": boot_diff(tr["_raw"]["min_dist"], te["_raw"]["min_dist"]),
        "gap_min_dist_rel": (tr["min_dist"][0] - te["min_dist"][0]) / te["min_dist"][0],
    }
    for s in ("train", "test"):
        out[s].pop("_raw")
    model.train(was_training)
    return out


def validate(model, data, n_ctx, n, nfe, seed, torch_seed):
    """check.memo_model and measure() with the same torch seed must give identical index values."""
    torch.manual_seed(torch_seed)
    ref = memo_model(model, data, n_ctx=n_ctx, n=n, seed=seed, nfe=nfe)
    torch.manual_seed(torch_seed)
    mine = measure(model, data, n_ctx, n, nfe, seed, keep_per_context=False)
    rep = {"data": data, "n_ctx": n_ctx, "n": n, "nfe": nfe, "seed": seed, "torch_seed": torch_seed, "max_abs_diff": {}}
    worst = 0.0
    for s in ("train", "test"):
        for k in ("cov", "prec", "realized_share", "share0"):
            dd = float(np.max(np.abs(np.array(ref[s][k]) - np.array(mine[s][k]))))
            rep["max_abs_diff"][f"{s}.{k}"] = dd; worst = max(worst, dd)
        rep["max_abs_diff"][f"{s}.n"] = abs(ref[s]["n"] - mine[s]["n"])
        rep[f"check_memo_model.{s}.realized_share"] = ref[s]["realized_share"]
        rep[f"r1_memo.{s}.realized_share"] = mine[s]["realized_share"]
    rep["identical"] = bool(worst < 1e-9 and all(rep["max_abs_diff"][f"{s}.n"] == 0 for s in ("train", "test")))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--nfe", type=int, default=None, help="sampling steps for fm/imf/sc (default: memo_model's 20, sc 1)")
    ap.add_argument("--n_ctx", type=int, default=400)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0, help="context permutation seed (memo_model default 0)")
    ap.add_argument("--torch_seed", type=int, default=123)
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate", action="store_true", help="first compare with check.memo_model on the first run")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.validate:
        model, ck = load_model(f"{r1_util.RUNS}/{a.runs[0]}/{a.ckpt}.pt")
        vp = f"{a.out}/validate_{a.runs[0]}_nfe{a.nfe}_{a.n_ctx}x{a.n}.json"
        if not os.path.exists(vp):
            rep = validate(model, ck["args"]["data"], a.n_ctx, a.n, a.nfe, a.seed, a.torch_seed)
            rep["run"] = a.runs[0]
            r1_util.write_json(rep, vp)
            print("validate", a.runs[0], "identical:", rep["identical"], rep["max_abs_diff"], flush=True)
    for run in a.runs:
        path = f"{a.out}/{run}_{a.ckpt}_nfe{a.nfe}_{a.n_ctx}x{a.n}.json"
        if os.path.exists(path):
            print("skip", path, flush=True)
            continue
        t0 = time.time()
        model, ck = load_model(f"{r1_util.RUNS}/{run}/{a.ckpt}.pt")
        data = ck["args"]["data"]
        torch.manual_seed(a.torch_seed)
        res = measure(model, data, a.n_ctx, a.n, a.nfe, a.seed)
        res.update(run=run, ckpt=a.ckpt, data=data, method=model.method, m=ck["args"].get("m"),
                   steps=ck.get("step", ck["args"].get("steps")), nfe_arg=a.nfe, n_ctx=a.n_ctx, n=a.n,
                   seed=a.seed, torch_seed=a.torch_seed, wall_s=round(time.time() - t0, 1))
        r1_util.write_json(res, path)
        dv = res["derived"]
        print(f"{run} nfe={res['nfe_used']} idx {dv['index_train']:.3f}/{dv['index_heldout']:.3f} "
              f"free {dv['free_index_train']:.3f}/{dv['free_index_heldout']:.3f} exc_ch {dv['excess_chance']:.3f} "
              f"exc_ho {dv['excess_heldout']:.3f} gap {dv['gap_min_dist'][0]:+.4f} ({res['wall_s']}s)", flush=True)

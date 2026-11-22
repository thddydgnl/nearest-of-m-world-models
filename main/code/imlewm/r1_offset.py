"""Revision r1, analysis d (DESIGN_r1.md section 4d): readout offset of the mode-share (flattening) measurement.

On bounce_div_k2_p80 (K = 2, mode 0 has probability 0.8), for the contexts of the mode-share measurement
(check.memo_model with n_ctx = 400, seed 0, both splits), the two true branch token maps of each context are
rendered with the context's appearance and DINO-encoded, decoded by the position probe and assigned with
check.balance (tolerance as in the monitor). Mixtures with a known share q of branch 0 are then passed through the
share computation of check.memo_model (share0 = mean over contexts of the valid share of mode 0):
  exact:  per context, branch 0 with weight q and branch 1 with weight 1 - q (infinitely many samples);
  n64:    per context, round(64 q) copies of branch 0 and the rest of branch 1 (true share round(64 q) / 64);
  data:   the realized next frame of each context (the stored token map), i.e. the empirical data frequency.
Reported: measured minus true share, plus the same with the tolerance-free assignment.

usage: python -m imlewm.r1_offset [--data bounce_div_k2_p80] --out /workspace/nom/out/d_offset
"""
import argparse
import os

import numpy as np
import torch

from . import r1_util
from .check import balance, boot_ci
from .common import DinoEncoder, ds_dir
from .evaluate import unique_branches
from .probe import load_probe
from .wm import Stats, rms

QS = (0.5, 0.6, 0.7, 0.787, 0.8)


@torch.no_grad()
def offset(data, n_ctx=400, seed=0, n=64):
    K = r1_util.set_geometry(data)
    assert K == 2
    st = Stats(data); probe = load_probe(data); enc = DinoEncoder()
    res = {"data": data, "n_ctx": n_ctx, "seed": seed, "qs": list(QS), "n_samples": n}
    for split in ("train", "test"):
        meta = r1_util.load_meta(data, split)
        idx = r1_util.memo_contexts(meta, n_ctx, seed)
        tok = np.load(f"{ds_dir(data)}/{split}_tokens.npy", mmap_mode="r")
        nxt = st.norm(r1_util.frames(tok, idx, [1])[:, 0].cuda())
        btok, bpos_all = r1_util.encode_branch_maps(enc, data, meta, idx, K)
        btok = st.norm(torch.from_numpy(btok).cuda())
        ps_br = torch.cat([probe(btok[i:i + 64].flatten(0, 1)) for i in range(0, len(idx), 64)]).view(len(idx), K, 2)
        ps_next = probe(nxt)
        per = []  # per kept context: valid0, assigned0, valid1, assigned1 (tolerance) and free assignments
        real0, data_share, data_free, rr, data_true_valid = [], [], [], [], []
        for c, t in enumerate(idx):
            bpos = bpos_all[c]
            if len(unique_branches(bpos, np.ones(K, bool))) < K:
                continue
            bp = torch.as_tensor(bpos, device="cuda", dtype=torch.float32)
            row = []
            for k in range(K):
                h = balance(ps_br[c, k][None], bp)
                jf = int((ps_br[c, k][None] - bp).norm(dim=-1).argmin())
                row += [int(h.sum() > 0), int(h[0] > 0), int(jf == 0)]
            per.append(row)
            real = r1_util.realized_mode(bpos, meta, t)
            real0.append(float(real == 0))
            h = balance(ps_next[c][None], bp)
            if h.sum() > 0:
                data_share.append(float(h[0] > 0)); data_true_valid.append(float(real == 0))
            data_free.append(float(int((ps_next[c][None] - bp).norm(dim=-1).argmin()) == 0))
            rr.append(float(rms(btok[c, real], nxt[c])))
        P = np.array(per, float)  # columns: v0, a0, f0 (branch 0), v1, a1, f1 (branch 1)
        v0, a0, f0, v1, a1, f1 = P.T
        o = {"n": len(P), "branch0_valid": float(v0.mean()), "branch1_valid": float(v1.mean()),
             "branch0_assigned_mode0": float((v0 * a0).sum() / max(v0.sum(), 1)),
             "branch1_assigned_mode0": float((v1 * a1).sum() / max(v1.sum(), 1)),
             "branch0_free_mode0": float(f0.mean()), "branch1_free_mode0": float(f1.mean()),
             "data_freq0": float(np.mean(real0)), "render_vs_stored_rms": boot_ci(rr), "mix": {}}
        for q in QS:
            num = q * v0 * a0 + (1 - q) * v1 * a1
            den = q * v0 + (1 - q) * v1
            ok = den > 0
            exact = float((num[ok] / den[ok]).mean())
            n0 = int(round(n * q))
            num64, den64 = n0 * v0 * a0 + (n - n0) * v1 * a1, n0 * v0 + (n - n0) * v1
            ok64 = den64 > 0
            m64 = float((num64[ok64] / den64[ok64]).mean())
            free = float((q * f0 + (1 - q) * f1).mean())
            o["mix"][str(q)] = {"exact_measured": exact, "exact_minus_true": exact - q,
                                "n64_true": n0 / n, "n64_measured": m64, "n64_minus_true": m64 - n0 / n,
                                "free_measured": free, "free_minus_true": free - q,
                                "n_contexts_used": int(ok.sum())}
        o["data_mix"] = {"true": float(np.mean(real0)), "measured": float(np.mean(data_share)) if data_share else None,
                         "true_on_valid": float(np.mean(data_true_valid)) if data_share else None,
                         "n_valid": len(data_share), "free_measured": float(np.mean(data_free))}
        o["data_mix"]["measured_minus_true"] = (o["data_mix"]["measured"] - o["data_mix"]["true_on_valid"]
                                                if data_share else None)
        o["data_mix"]["free_minus_true"] = o["data_mix"]["free_measured"] - o["data_mix"]["true"]
        res[split] = o
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bounce_div_k2_p80")
    ap.add_argument("--n_ctx", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    path = f"{a.out}/offset_{a.data}_{a.n_ctx}.json"
    if os.path.exists(path):
        print("skip", path)
    else:
        r = offset(a.data, a.n_ctx)
        r1_util.write_json(r, path)
        for s in ("train", "test"):
            print(s, {q: (round(v["exact_minus_true"], 4), round(v["n64_minus_true"], 4)) for q, v in r[s]["mix"].items()},
                  r[s]["data_mix"], flush=True)

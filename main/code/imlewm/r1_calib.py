"""Revision r1, check C1 (DESIGN_r1.md section 1): ceiling and floor of the memorization monitor on one geometry.

For n_ctx training contexts (the selection of check.memo_model, i.e. the monitor's 200 contexts with seed 0;
contexts whose K branches are not distinct are skipped, as in the monitor) the monitor's readout (position probe)
and assignment (check.balance, tolerance min(0.45 x nearest-mode separation, 4 px)) receive
  (a) ceiling: 16 copies of the realized next token map (train_tokens[t+1]), i.e. a perfect memorizer
  (b) floor:   4 copies of each of the K true branch maps (rendered with the context's appearance and
               DINO-encoded), i.e. a calibrated sampler
and the index is the share of valid samples on the realized mode, averaged over contexts with a valid sample.
Also reported: the tolerance-free versions (every sample to its nearest branch, C2), validity rates, the probe's
position error on these maps, and the token RMS distance between the rendered realized branch and the stored
next frame (a check of the rendering). Any bounce_div dataset, including bounce_div_k4_arc90.
Attenuation-corrected index (if C1 triggers): (index - floor) / (ceiling - floor).

usage: python -m imlewm.r1_calib --data bounce_div_k4_arc90 [--n_ctx 200] --out /workspace/nom/out/c1
"""
import argparse
import os

import numpy as np
import torch

from . import common, r1_util
from .check import balance, boot_ci
from .common import DinoEncoder, ds_dir
from .evaluate import unique_branches
from .probe import load_probe
from .wm import Stats, rms


def shares(ps, bp, realized, K):
    """ps (n,2) decoded samples; returns (index or None, tolerance-free index, number of valid samples)."""
    h = balance(ps, bp)
    jf = (ps[:, None] - bp[None]).norm(dim=-1).argmin(1)
    hf = torch.bincount(jf, minlength=K).cpu().numpy()
    return (h[realized] / h.sum() if h.sum() > 0 else None), hf[realized] / hf.sum(), int(h.sum())


@torch.no_grad()
def c1(data, n_ctx=200, seed=0, split="train", n_ceil=16, n_floor=4):
    K = r1_util.set_geometry(data)
    st = Stats(data); probe = load_probe(data); enc = DinoEncoder()
    meta = r1_util.load_meta(data, split)
    idx = r1_util.memo_contexts(meta, n_ctx, seed)
    tok = np.load(f"{ds_dir(data)}/{split}_tokens.npy", mmap_mode="r")
    nxt = st.norm(r1_util.frames(tok, idx, [1])[:, 0].cuda())  # (n,256,384) realized next frames
    btok, bpos_all = r1_util.encode_branch_maps(enc, data, meta, idx, K)
    btok = st.norm(torch.from_numpy(btok).cuda())  # (n,K,256,384)
    ps_next = probe(nxt)
    ps_br = torch.cat([probe(btok[i:i + 64].flatten(0, 1)) for i in range(0, len(idx), 64)]).view(len(idx), K, 2)
    rows = {k: [] for k in ("ceil", "ceil_free", "floor", "floor_free", "ceil_valid", "floor_valid_frac",
                            "err_next_px", "err_branch_px", "render_rms", "render_rms_other_min")}
    skipped = 0
    for c, t in enumerate(idx):
        bpos = bpos_all[c]
        if len(unique_branches(bpos, np.ones(K, bool))) < K:
            skipped += 1
            continue
        bp = torch.as_tensor(bpos, device="cuda", dtype=torch.float32)
        real = r1_util.realized_mode(bpos, meta, t)
        s, sf, nv = shares(ps_next[c][None].expand(n_ceil, 2), bp, real, K)
        rows["ceil_valid"].append(float(nv > 0)); rows["ceil_free"].append(sf)
        if s is not None:
            rows["ceil"].append(s)
        s, sf, nv = shares(ps_br[c].repeat_interleave(n_floor, 0), bp, real, K)
        rows["floor_valid_frac"].append(nv / (n_floor * K)); rows["floor_free"].append(sf)
        if s is not None:
            rows["floor"].append(s)
        rows["err_next_px"].append(float(np.linalg.norm(ps_next[c].cpu().numpy() - meta["pos"][t + 1])))
        rows["err_branch_px"].append(float((ps_br[c] - bp).norm(dim=-1).mean()))
        d = rms(btok[c], nxt[c][None])  # (K,) rendered branch maps vs the stored next frame
        rows["render_rms"].append(float(d[real]))
        rows["render_rms_other_min"].append(float(torch.cat([d[:real], d[real + 1:]]).min()) if K > 1 else None)
    out = {"data": data, "split": split, "K": K, "n_ctx": n_ctx, "seed": seed, "n_used": len(rows["ceil_free"]),
           "n_skipped_merged": skipped, "n_ceil_copies": n_ceil, "n_floor_copies_per_branch": n_floor,
           "arc_deg": float(common.ARC_DEG)}
    for k, v in rows.items():
        out[k] = boot_ci(v) if len(v) else None
    out["n_ceil_contexts_valid"] = len(rows["ceil"]); out["n_floor_contexts_valid"] = len(rows["floor"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--n_ctx", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for data in a.data:
        path = f"{a.out}/c1_{data}_{a.n_ctx}.json"
        if os.path.exists(path):
            print("skip", path); continue
        torch.manual_seed(0)
        res = c1(data, a.n_ctx, a.seed)
        r1_util.write_json(res, path)
        print(data, {k: (round(v[0], 4) if isinstance(v, list) else v) for k, v in res.items()}, flush=True)

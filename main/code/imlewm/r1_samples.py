"""Revision r1, analysis h (DESIGN_r1.md section 4h): qualitative samples of the Table 2 models for a new Figure 3.

For the seed-0 A2-div checkpoints of Table 2, 16 samples (torch seed 123 per model) for one training and one
held-out wall context of bounce_div_k4, decoded by the position probe. Contexts are fixed by rule, not by look:
in the order of check.memo_model's context permutation (seed 0), the first context whose K branches are distinct
and at least --min_sep px apart. Saved: the context frames (rendered from the stored positions and appearance),
the K branch positions, the realized branch, and per model the decoded positions, the valid-mode assignment of
the monitor (check.balance), and the token RMS distance of every sample to every branch map (rendered and
DINO-encoded). A preview figure is written next to the JSON.

usage: python -m imlewm.r1_samples --out /workspace/nom/out/h_samples
"""
import argparse
import os

import numpy as np
import torch

from . import r1_util
from .common import DinoEncoder, ds_dir, render
from .evaluate import unique_branches
from .probe import load_probe
from .wm import Stats, load_model, predict, rms

MODELS = [("Regression", "e1_a2_reg_s0", 1), ("FM-50", "e1_a2_fm_s0", 50), ("FM-2", "e1_a2_fm_s0", 2),
          ("iMF", "e1_a2_imf_s0", 1), ("Shortcut", "e7_a2_sc_s0", 1), ("Energy score m=8", "e15_a2_es8_s0", 1),
          ("Energy score m=16", "e15_a2_es16_s0", 1), ("IMLE-WM m=4", "e1_a2_imle4_s0", 1),
          ("IMLE-WM m=16", "e1_a2_imle16_s0", 1)]


def pick(data, meta, K, min_sep, n_scan=400, skip=0):
    """The (skip+1)-th context of memo_model's permutation that satisfies the rule."""
    for t in r1_util.memo_contexts(meta, n_scan, 0):
        bp = r1_util.branches(data, meta, t, K)
        if len(unique_branches(bp, np.ones(K, bool))) < K:
            continue
        sep = np.linalg.norm(bp[:, None] - bp[None], axis=-1) + np.eye(K) * 1e9
        if sep.min() >= min_sep:
            if skip == 0:
                return int(t), bp
            skip -= 1
    raise RuntimeError("no context")


@torch.no_grad()
def main(out, data="bounce_div_k4", n=16, min_sep=6.0, skip=0, tag=""):
    from .check import balance
    K = r1_util.set_geometry(data)
    st = Stats(data); probe = load_probe(data); enc = DinoEncoder()
    res = {"data": data, "K": K, "n": n, "rule": f"first context of memo_model's permutation (seed 0) with K distinct "
                                                  f"branches at least {min_sep} px apart", "contexts": {}}
    imgs = {}
    for split in ("train", "test"):
        meta = r1_util.load_meta(data, split)
        t, bp = pick(data, meta, K, min_sep, skip=skip)
        tok = np.load(f"{ds_dir(data)}/{split}_tokens.npy", mmap_mode="r")
        fr = r1_util.frames(tok, np.array([t]), [-2, -1, 0, 1])
        ctx = st.norm(fr[:, :3].cuda())
        app = meta["app"][t - 2:t + 2]
        imgs[split] = render(meta["pos"][t - 2:t + 2], "bounce", app)  # frames t-2..t+1
        btok, _ = r1_util.encode_branch_maps(enc, data, meta, np.array([t]), K)
        btok = st.norm(torch.from_numpy(btok[0]).cuda())
        real = r1_util.realized_mode(bp, meta, t)
        c = {"t": t, "ctx_pos": meta["pos"][t - 2:t + 1].tolist(), "next_pos": meta["pos"][t + 1].tolist(),
             "branch_pos": bp.tolist(), "realized": real, "app": meta["app"][t].tolist(), "models": {}}
        bpt = torch.as_tensor(bp, device="cuda", dtype=torch.float32)
        for label, run, nfe in MODELS:
            torch.manual_seed(123)
            model, _ = load_model(f"{r1_util.RUNS}/{run}/ckpt.pt")
            S = predict(model, ctx, torch.zeros(1, 2, device="cuda"), n, nfe, st.sigma_r)[0]
            ps = probe(S)
            dp = (ps[:, None] - bpt[None]).norm(dim=-1)
            h = balance(ps, bpt)
            c["models"][label] = {"run": run, "nfe": nfe, "pos": ps.cpu().numpy().round(3).tolist(),
                                  "nearest_branch": dp.argmin(1).cpu().tolist(), "valid_counts": h.tolist(),
                                  "tok_rms_to_branches": rms(S[:, None], btok[None]).cpu().numpy().round(4).tolist(),
                                  "tok_rms_to_next": rms(S, st.norm(fr[0, 3].cuda())[None]).cpu().numpy().round(4).tolist()}
        res["contexts"][split] = c
    res["skip"] = skip
    r1_util.write_json(res, f"{out}/samples{tag}.json")
    np.savez_compressed(f"{out}/context_frames{tag}.npz", **{f"{s}_frames": v for s, v in imgs.items()})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, len(MODELS), figsize=(1.6 * len(MODELS), 3.4))
        for r_, split in enumerate(("train", "test")):
            c = res["contexts"][split]
            for j, (label, _, _) in enumerate(MODELS):
                ax = axes[r_, j]
                ax.imshow(imgs[split][2], extent=(0, 64, 64, 0))
                for q in c["ctx_pos"][:2]:
                    ax.scatter(*q, s=6, c="0.7")
                bpa = np.array(c["branch_pos"])
                ax.scatter(bpa[:, 0], bpa[:, 1], s=40, facecolors="none", edgecolors="lime", linewidths=0.8)
                ax.scatter(*bpa[c["realized"]], s=70, facecolors="none", edgecolors="white", linewidths=0.8)
                p = np.array(c["models"][label]["pos"])
                ax.scatter(p[:, 0], p[:, 1], s=5, c="red")
                ax.set_xticks([]); ax.set_yticks([])
                if r_ == 0:
                    ax.set_title(label, fontsize=6)
            axes[r_, 0].set_ylabel("training" if split == "train" else "held-out", fontsize=7)
        fig.savefig(f"{out}/samples_preview{tag}.png", dpi=160, bbox_inches="tight")
    except Exception as e:  # the preview is optional
        print("preview failed", e)
    print({s: (res["contexts"][s]["t"], res["contexts"][s]["realized"],
               {k: v["valid_counts"] for k, v in res["contexts"][s]["models"].items()}) for s in ("train", "test")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_sep", type=float, default=6.0)
    ap.add_argument("--skip", type=int, default=0, help="use the (skip+1)-th context that satisfies the rule (extra panels)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = f"_ctx{a.skip + 1}" if a.skip else ""
    if os.path.exists(f"{a.out}/samples{tag}.json"):
        print("skip")
    else:
        main(a.out, min_sep=a.min_sep, skip=a.skip, tag=tag)

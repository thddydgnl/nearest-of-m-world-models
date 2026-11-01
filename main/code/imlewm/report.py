"""Collect eval jsons into tables and draw figures.

usage: python -m imlewm.report  -> ~/imle-wm/report/{summary.json, *.png}
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import ds_dir
from .probe import load_probe
from .wm import Stats, load_model, predict

RUNS = os.path.expanduser("~/imle-wm/runs")
OUT = os.path.expanduser("~/imle-wm/report")


def collect():
    rows = {}
    for f in sorted(glob.glob(f"{RUNS}/*/eval*.json")):
        r = json.load(open(f))
        key = r["run"] + (f"@{r['nfe']}" if r["args"]["method"] in ("fm", "mf", "imf", "sc") else "")
        r.pop("args", None)
        rows[key] = r
    return rows


@torch.no_grad()
def scatter(data, runs, n_ctx=6, n=16, seed=0):
    d = ds_dir(data); st = Stats(data); probe = load_probe(data)
    test = np.load(f"{d}/test_tokens.npy", mmap_mode="r")
    meta = np.load(f"{d}/test_meta.npz")
    ev = np.load(f"{d}/eval_idx.npz")
    idx, bpos = ev["stoch"][:n_ctx], ev["stoch_branch_pos"][:n_ctx]
    fi = (idx[:, None] + np.arange(-2, 1)[None]).reshape(-1)
    ctx = st.norm(torch.from_numpy(test[fi]).cuda()).view(n_ctx, 3, 256, 384)
    fig, axes = plt.subplots(len(runs), n_ctx, figsize=(2.2 * n_ctx, 2.2 * len(runs)), squeeze=False)
    for ri, (label, run, nfe) in enumerate(runs):
        torch.manual_seed(seed)
        model, _ = load_model(f"{RUNS}/{run}/ckpt.pt")
        S = predict(model, ctx, torch.zeros(n_ctx, 2, device="cuda"), n, nfe, st.sigma_r)
        P = probe(S.flatten(0, 1)).view(n_ctx, n, 2).cpu().numpy()
        for c in range(n_ctx):
            ax = axes[ri, c]
            hist = meta["pos"][idx[c] - 2: idx[c] + 1]
            ax.plot(hist[:, 0], hist[:, 1], "o-", color="0.6", ms=3)
            ax.scatter(bpos[c][:, 0], bpos[c][:, 1], s=90, facecolors="none", edgecolors="tab:green", lw=1.5)
            ax.scatter(P[c][:, 0], P[c][:, 1], s=10, color="tab:red", alpha=0.7)
            ax.set_xlim(0, 64); ax.set_ylim(64, 0); ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            if c == 0:
                ax.set_ylabel(label)
    fig.suptitle(f"{data}: context (grey), true modes (green), probe-decoded samples (red)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/scatter_{data}.png", dpi=110)
    plt.close(fig)


def horizon_plot(rows, runs, fname):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for label, key in runs:
        if key not in rows:
            continue
        h = rows[key]["horizon"]
        x = np.arange(1, len(h["best_tok"]) + 1)
        axes[0].plot(x, h["best_tok"], label=label)
        axes[1].plot(x, h["best_pos"], label=label)
    axes[0].set_title("best-of-16 token RMS vs horizon"); axes[1].set_title("best-of-16 position error (px)")
    for ax in axes:
        ax.set_xlabel("horizon h"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=110); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = collect()
    json.dump(rows, open(f"{OUT}/summary.json", "w"), indent=1)
    for K in [2, 4, 8]:
        runs = [(l, r, n) for l, r, n in [("reg", f"reg_k{K}", 1), ("FM-20", f"fm_k{K}", 20), ("FM-2", f"fm_k{K}", 2),
                                         ("IMLE", f"imle_k{K}", 1)] if os.path.exists(f"{RUNS}/{r}/ckpt.pt")]
        if runs:
            scatter(f"bounce_k{K}", runs)
    horizon_plot(rows, [("reg", "reg_push"), ("FM-20", "fm_push@20"), ("FM-2", "fm_push@2"),
                        ("IMLE N0", "imle_push"), ("IMLE N1 (H=8)", "imle_push_n1")], "horizon_push.png")
    print("ok", list(rows))

"""A1: FlowWM Bouncing Shapes (arXiv 2606.29059) encoded to DINOv2 tokens.

Uses the official generator from third_party/Flow-World-Models unchanged (same config as the
paper's export: 32 frames, 256px, square 64, disk r=24, max_speed 6, 10 velocity bins, seed 0,
ICs with 1 or >10 futures skipped). Frames 0-15 are deterministic and shared by all futures of
an initial condition, so each unique frame is stored once and clips index into a frame table.

usage: python -m imlewm.gen_bshapes --train_clips 16384
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

from .common import DinoEncoder, Dinov3Encoder, ds_dir, save_json

FWM = os.path.expanduser("~/imle-wm/third_party/Flow-World-Models")
sys.path.insert(0, FWM)
from datasets.synthetic_bouncing_shapes.generator import SyntheticBouncingShapesVideoGenerator  # noqa: E402

CFG = dict(frames=32, fps=10, square_size=64, disk_radius=24, image_size=256, max_speed=6.0,
           train_ratio=0.8, overfit_single_video=False, dataset_seed=0, velocity_bins=10,
           stochastic_bounces=True)


class FrameWriter:
    """Encodes uint8 frames in chunks into a growing fp16 memmap."""

    def __init__(self, path, enc, cap):
        self.mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(cap, 256, 384))
        self.enc, self.n, self.buf = enc, 0, []

    def add(self, frames):  # (T,3,256,256) uint8 torch
        start = self.n + sum(len(b) for b in self.buf)
        self.buf.append(frames.permute(0, 2, 3, 1).numpy())
        if sum(len(b) for b in self.buf) >= 2048:
            self.flush()
        return np.arange(start, start + len(frames))

    def flush(self):
        if not self.buf:
            return
        x = np.concatenate(self.buf); self.buf = []
        self.mm[self.n:self.n + len(x)] = self.enc(x)  # encoder resizes 256 -> 224
        self.n += len(x)


def export(split, n_clips, out, enc):
    gen = SyntheticBouncingShapesVideoGenerator(SimpleNamespace(split=split, length=n_clips, **CFG))
    cap = n_clips * 24 + 32  # 16 shared prefix frames per IC (>= 2 futures each) + 16 per clip
    fw = FrameWriter(f"{out}/{split}_tokens.npy", enc, cap)
    clips, ics, sq, ci, decisions = [], [], [], [], []
    n_ic = 0
    for ic in range(gen.get_num_initial_conditions()):
        if len(clips) >= n_clips:
            break
        nt = gen.get_num_trajectories(ic)
        if nt == 1 or nt > 10:
            gen.release_initial_condition(ic)
            continue
        prefix = None
        for tr in range(nt):
            if len(clips) >= n_clips:
                break
            s = gen.generate(ic, tr)
            v = s["video"]  # (32,3,256,256) uint8
            if prefix is None:
                prefix = fw.add(v[:16])
            else:
                assert torch.equal(v[:16], first[:16]), "frames 0-15 should be shared across futures"
            if tr == 0:
                first = v
            suffix = fw.add(v[16:])
            clips.append(np.concatenate([prefix, suffix]))
            sqh = s["target_square_heatmap_pixel"][..., 0]  # (32,256,256)
            cih = s["target_circle_heatmap_pixel"][..., 0]
            to_xy = lambda h: torch.stack([h.flatten(1).argmax(1) % 256, h.flatten(1).argmax(1) // 256], 1)
            sq.append(to_xy(sqh).numpy()); ci.append(to_xy(cih).numpy()); ics.append(n_ic)
        n_ic += 1
        gen.release_initial_condition(ic)
    fw.flush()
    n_frames = fw.n
    del fw
    # shrink the memmap to the frames actually written
    full = np.load(f"{out}/{split}_tokens.npy", mmap_mode="r")
    np.save(f"{out}/{split}_tokens_tmp.npy", full[:n_frames])
    del full
    os.replace(f"{out}/{split}_tokens_tmp.npy", f"{out}/{split}_tokens.npy")
    np.savez(f"{out}/{split}_clips.npz", frames=np.array(clips), ic=np.array(ics),
             sq_pos=np.array(sq).astype(np.float32), ci_pos=np.array(ci).astype(np.float32))
    print(f"{split}: {len(clips)} clips, {n_ic} ICs, {n_frames} unique frames "
          f"(avg {len(clips) / n_ic:.2f} futures/IC)", flush=True)


def compute_stats(out, n_sub=6000, seed=0):
    toks = np.load(f"{out}/train_tokens.npy", mmap_mode="r")
    c = np.load(f"{out}/train_clips.npz")["frames"]
    rng = np.random.default_rng(seed)
    ci = rng.integers(0, len(c), n_sub); t = rng.integers(0, 31, n_sub)
    a, b = c[ci, t], c[ci, t + 1]
    x = torch.from_numpy(np.asarray(toks[np.sort(a)])).cuda().float()
    mean = x.mean((0, 1)); std = x.std((0, 1)) + 1e-4
    xn = ((x - mean) / std).reshape(-1, 384)
    cov = (xn.T @ xn) / len(xn)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    xa = torch.from_numpy(np.asarray(toks[a])).cuda().float()
    xb = torch.from_numpy(np.asarray(toks[b])).cuda().float()
    r = (xb - mean) / std - (xa - mean) / std
    rms = r.pow(2).mean((1, 2)).sqrt()
    torch.save({"mean": mean.cpu(), "std": std.cpu(), "pca": evecs.cpu(), "evals": evals.cpu(),
                "sigma_r": r.std().item(), "resid_rms_median": rms.median().item()}, f"{out}/stats.pt")
    print("sigma_r", r.std().item(), "resid rms median", rms.median().item())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_clips", type=int, default=16384)
    ap.add_argument("--eval_clips", type=int, default=512)
    ap.add_argument("--encoder", default="v2", choices=["v2", "v3"])
    a = ap.parse_args()
    out = ds_dir("bshapes" if a.encoder == "v2" else "bshapes_v3")
    os.makedirs(out, exist_ok=True)
    enc = DinoEncoder() if a.encoder == "v2" else Dinov3Encoder()
    export("evaluation", a.eval_clips, out, enc)
    export("train", a.train_clips, out, enc)
    compute_stats(out)
    save_json(vars(a) | {"generator": CFG, "encoder": "facebook/dinov2-small@224" if a.encoder == "v2" else
                         "facebook/dinov3-vits16 layers 3/6/9/12 normed+averaged @256 (FlowWM)"}, f"{out}/gen_args.json")

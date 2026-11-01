"""FlowWM's position readout for Bouncing Shapes (models/models.py HeatmapPixelPredictor,
configs/dinov3/heatmap/base/384, losses.HeatmapLossPixel, evaluators._heatmap_to_xy_coords).

One pixel-heatmap head per shape on the DINOv3 feature grid: 1x1 conv 384 -> 256, GELU, dropout 0.1,
ConvTranspose2d(256, 1, 16, stride 16) -> 256 x 256 logits, cross-entropy over all pixels against the
integer centre pixel, position = argmax (x = idx % 256, y = idx // 256). Trained with Gaussian noise
(std 0.05) on the raw backbone features, as in FlowWM's noise_std0p05 config, so that the head
tolerates imprecise predicted features. We train it on our own normalized tokens (the affine
normalization is absorbed by the first conv).

usage: python -m imlewm.heatmap --train [--data bshapes_v3]
       A1_READOUT=heatmap python -m imlewm.a1eval --run ... / python -m imlewm.joint --eval ...
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ds_dir
from .wm import Stats


class HeatPix(nn.Module):
    def __init__(self, dim=384, hidden=256, patch=16, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GELU(), nn.Dropout(dropout),
                                 nn.ConvTranspose2d(hidden, 1, patch, stride=patch))

    def forward(self, grid):  # (B,384,16,16) -> (B,256*256) logits
        return self.net(grid).flatten(1)


class HeatmapReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.sq, self.ci = HeatPix(), HeatPix()

    def logits(self, tok):  # tok (B,256,384) normalized
        g = tok.float().transpose(1, 2).reshape(len(tok), 384, 16, 16)
        return self.sq(g), self.ci(g)

    @torch.no_grad()
    def forward(self, tok):  # -> (B, 2 shapes, 2 xy) pixel positions, like a1eval.Probe2
        out = []
        for i in range(0, len(tok), 256):
            ls, lc = self.logits(tok[i:i + 256])
            xy = [torch.stack([l.argmax(1) % 256, l.argmax(1) // 256], -1).float() for l in (ls, lc)]
            out.append(torch.stack(xy, 1))
        return torch.cat(out)


def path(data, tag=""):
    return f"{ds_dir(data)}/heatmap_readout{'_' + tag if tag else ''}.pt"


def load(data, tag=""):
    r = HeatmapReadout().cuda()
    r.load_state_dict(torch.load(path(data, tag)))
    return r.eval()


def train(data, steps=20000, bs=128, noise=0.05, n_ram=120000, seed=0, tag=""):
    from . import a1eval
    from .a1eval import frame_positions
    a1eval.DATA = data
    d = ds_dir(data); st = Stats(data)
    tok = np.load(f"{d}/train_tokens.npy", mmap_mode="r")
    pos = torch.from_numpy(frame_positions("train")).round().long()  # (frames, 2 shapes, xy)
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(len(pos), min(n_ram, len(pos)), replace=False))
    sub_tok = torch.from_numpy(np.asarray(tok[sub])); sub_pos = pos[sub]
    r = HeatmapReadout().cuda()
    opt = torch.optim.AdamW(r.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    r.train()
    for s in range(steps):
        i = torch.randint(len(sub), (bs,), generator=g)
        raw = sub_tok[i].cuda().float()
        x = st.norm(raw + noise * torch.randn_like(raw))
        ls, lc = r.logits(x)
        p = sub_pos[i].cuda()
        tgt = p[..., 1] * 256 + p[..., 0]  # (bs, 2)
        loss = F.cross_entropy(ls, tgt[:, 0]) + F.cross_entropy(lc, tgt[:, 1])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
        opt.step(); sch.step()
        if s % 2000 == 0:
            print(f"step {s} loss {loss.item():.3f}", flush=True)
    r.eval()
    torch.save(r.state_dict(), path(data, tag))
    # readout floor on real evaluation frames
    etok = np.load(f"{d}/evaluation_tokens.npy", mmap_mode="r")
    epos = torch.from_numpy(frame_positions("evaluation"))
    ii = np.arange(0, len(epos), max(1, len(epos) // 4000))
    pr = torch.cat([r(st.norm(torch.from_numpy(np.asarray(etok[ii[j:j + 500]])).cuda())).cpu() for j in range(0, len(ii), 500)])
    err = (pr - epos[ii]).abs().mean((1, 2))
    print(f"heatmap readout eval L1 px: mean {err.mean():.3f} p95 {err.quantile(0.95):.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--data", default="bshapes_v3")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--noise", type=float, default=0.05, help="feature noise std during training (FlowWM: 0, 0.05, 0.1, 0.5)")
    ap.add_argument("--tag", default="", help="readout file variant, e.g. n05 -> heatmap_readout_n05.pt")
    a = ap.parse_args()
    if a.train:
        train(a.data, a.steps, noise=a.noise, tag=a.tag)

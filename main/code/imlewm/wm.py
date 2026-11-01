"""Shared utilities: token dataset, sampling, rollouts, checkpoint IO."""
import os
import queue
import threading

import numpy as np
import torch

from .common import ds_dir
from .model import Gen

K_CTX = 3


class Stats:
    def __init__(self, name, device="cuda"):
        s = torch.load(f"{ds_dir(name)}/stats.pt")
        self.mean = s["mean"].to(device)
        self.std = s["std"].to(device)
        self.pca = s["pca"].to(device)
        self.sigma_r = s["sigma_r"]
        self.resid_rms = s["resid_rms_median"]

    def norm(self, x):
        return (x.float() - self.mean) / self.std


def shift_noise(s, alpha):
    """FlowWM time shift on the noise level s in [0,1] (1 = pure noise); alpha > 1 skews noisier."""
    return alpha * s / (1 + (alpha - 1) * s)


class TokenData:
    """Windows of K_CTX context frames + H future frames within one segment.
    Clip-format datasets ({split}_clips.npz, e.g. A1 Bouncing Shapes) index a shared frame table;
    windows are built per clip and de-duplicated (futures of one IC share frames 0-15)."""

    def __init__(self, name, split="train", H=1, device="cuda"):
        d = ds_dir(name)
        self.tok = torch.from_numpy(np.load(f"{d}/{split}_tokens.npy"))
        self.H, self.dev = H, device
        self.stats = Stats(name, device)
        if os.path.exists(f"{d}/{split}_clips.npz"):
            fr = np.load(f"{d}/{split}_clips.npz")["frames"]
            T = fr.shape[1]
            rows = np.concatenate([fr[:, t - K_CTX + 1:t + H + 1] for t in range(K_CTX - 1, T - H)])
            self.win = torch.from_numpy(np.unique(rows, axis=0))
            self.valid = torch.arange(len(self.win))
            self.act = None
            return
        self.win = None
        m = np.load(f"{d}/{split}_meta.npz")
        self.act = torch.from_numpy(m["act"]).float()
        self.pos = torch.from_numpy(m["pos"]).float() if "pos" in m.files else None
        seg = m["seg"]
        T = len(seg)
        t = np.arange(K_CTX - 1, T - H)
        self.valid = torch.from_numpy(t[seg[t - K_CTX + 1] == seg[t + H]])

    def get(self, t, H=None):
        H = self.H if H is None else H
        t = torch.as_tensor(t)
        if self.win is not None:
            fi = self.win[t][:, :K_CTX + H].reshape(-1)
            x = self.tok[fi].view(len(t), K_CTX + H, 256, 384).pin_memory().to(self.dev, non_blocking=True)
            return x, torch.zeros(len(t), H, 2, device=self.dev)
        offs = torch.arange(-K_CTX + 1, H + 1)
        fi = (t[:, None] + offs[None]).reshape(-1)
        x = self.tok[fi].view(len(t), K_CTX + H, 256, 384).pin_memory().to(self.dev, non_blocking=True)
        a = self.act[(t[:, None] + torch.arange(H)[None]).reshape(-1)].view(len(t), H, -1).to(self.dev)
        return x, a

    def sample(self, B, gen):
        i = torch.randint(len(self.valid), (B,), generator=gen)
        return self.get(self.valid[i])

    def loader(self, B, seed, prefetch=4):
        q = queue.Queue(prefetch)
        gen = torch.Generator().manual_seed(seed)

        def work():
            while True:
                q.put(self.sample(B, gen))
        threading.Thread(target=work, daemon=True).start()
        while True:
            yield q.get()


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device)
    m = Gen(**ck["model_args"]).to(device)
    m.load_state_dict(ck["ema"])
    m.eval()
    return m, ck


@torch.no_grad()
def predict(model, ctx, a, n=1, nfe=20, sigma_r=1.0, z=None, x0=None, chunk=1024):
    """ctx (B,k,N,D) normalized, a (B,2) -> (B,n,N,D) samples of the next normalized tokens."""
    B = ctx.shape[0]
    c = ctx.repeat_interleave(n, 0)
    aa = a.repeat_interleave(n, 0)
    outs = []
    for i in range(0, B * n, chunk):
        ci, ai = c[i:i + chunk], aa[i:i + chunk]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if model.method in ("imle", "es"):
                zi = torch.randn(len(ci), model.zdim, device=ci.device) if z is None else z[i:i + chunk]
                o = model(ci, ai, z=zi)[-1].float()
            elif model.method == "fm":
                # t: 0 = noise, 1 = data; grid uniform in noise level s = 1 - t, then shifted
                x = torch.randn(len(ci), 256, 384, device=ci.device) if x0 is None else x0[i:i + chunk].clone()
                sg = shift_noise(torch.linspace(1, 0, nfe + 1, device=ci.device), getattr(model, "t_shift", 1.0))
                tg = 1 - sg
                for s in range(nfe):
                    t = tg[s].expand(len(ci))
                    x = x + (tg[s + 1] - tg[s]) * model(ci, ai, t=t, xt=x)[-1].float()
                o = x * sigma_r
            elif model.method in ("mf", "imf", "sc"):
                # MeanFlow convention: start at noise s=1, step down to 0 with average velocity
                x = torch.randn(len(ci), 256, 384, device=ci.device) if x0 is None else x0[i:i + chunk].clone()
                ts = torch.linspace(1, 0, nfe + 1, device=ci.device)
                for s in range(nfe):
                    s_, r_ = ts[s].expand(len(ci)), ts[s + 1].expand(len(ci))
                    x = x - (ts[s] - ts[s + 1]) * model(ci, ai, t=s_, r=r_, xt=x)[-1].float()
                o = x * sigma_r
            else:
                o = model(ci, ai)[-1].float()
        outs.append(ci[:, -1].float() + o)
    return torch.cat(outs).view(B, n, 256, 384)


@torch.no_grad()
def rollout(model, ctx, acts, n=1, nfe=20, sigma_r=1.0):
    """ctx (B,k,N,D), acts (B,H,2) -> (B,n,H,N,D) independent-noise rollouts."""
    B, H = acts.shape[:2]
    c = ctx.repeat_interleave(n, 0)
    out = []
    for h in range(H):
        s = predict(model, c, acts[:, h].repeat_interleave(n, 0), 1, nfe, sigma_r)[:, 0]
        out.append(s)
        c = torch.cat([c[:, 1:], s[:, None]], 1)
    return torch.stack(out, 1).view(B, n, H, 256, 384)


def rms(a, b):
    return (a - b).pow(2).mean((-1, -2)).sqrt()

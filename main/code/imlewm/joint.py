"""A1 in the FlowWM protocol: 16 context frames -> all 16 future frames in one shot (no rollout).

FlowWM (arXiv 2606.29059) generates the 16 future frames of Bouncing Shapes jointly from 16 context
frames. Our main pipeline predicts one frame at a time from 3 frames and rolls out, so its A1 numbers
are not comparable to the paper's. This module puts IMLE, flow matching and a deterministic regressor
into the paper's protocol with one shared architecture:
  - frames are stacked as channels: context (16 x 384) -> 1x1 stem -> ConvNeXt blocks on the 16x16
    token grid -> per-token wide head -> 16 x 384 future channels
  - targets are the futures minus the last context frame, normalized per horizon step
  - IMLE matches m candidates at the trajectory level (all 16 frames), late fusion: z enters from
    block z_from, so the z-free trunk runs once per sample
  - FM: FlowWM recipe (time shift 13, Euler 50 at test), noisy target stacked into the stem input
Evaluation is the FlowWM Bouncing Shapes metric of a1eval (2F samples per initial condition).

usage: python -m imlewm.joint --method imle --m 16 --name j_imle16_s0
       python -m imlewm.joint --eval j_imle16_s0 [--nfe 50]
"""
import argparse
import copy
import json
import math
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import a1eval
from .common import ds_dir
from .model import Block, D, G, mlp, time_emb
from .train import Robust
from .wm import Stats, shift_noise

RUNS = os.path.expanduser("~/imle-wm/runs")
T_CTX, T_FUT = 16, 16


class JointGen(nn.Module):
    def __init__(self, method="imle", C=384, nblocks=12, zdim=256, sdim=256, wide=1024, z_from=8, t_shift=13.0):
        super().__init__()
        self.method, self.zdim, self.t_shift = method, zdim, t_shift
        self.z_from = z_from if method == "imle" else 0
        in_ch = T_CTX * D + (T_FUT * D if method == "fm" else 0)
        self.stem = nn.Conv2d(in_ch, C, 1)
        self.pos = nn.Parameter(torch.zeros(1, C, G, G))
        nn.init.trunc_normal_(self.pos, std=0.02)
        if method == "imle":
            self.map = mlp(zdim, sdim, sdim, 4)
        elif method == "fm":
            self.map = mlp(128, sdim, sdim, 2)
        if method != "fm":
            self.const = nn.Parameter(torch.zeros(1, sdim))
        self.blocks = nn.ModuleList([Block(C, sdim, 0) for _ in range(nblocks)])
        self.head = nn.Sequential(nn.Conv2d(C, wide, 1), nn.GELU(), nn.Conv2d(wide, wide, 1), nn.GELU(),
                                  nn.Conv2d(wide, T_FUT * D, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def _style(self, B, z=None, t=None):
        if self.method == "imle":
            zz = z / z.pow(2).mean(1, keepdim=True).add(1e-8).sqrt()
            return self.map(zz)
        if self.method == "fm":
            return self.map(time_emb(t))
        return self.const.expand(B, -1)

    def trunk(self, ctx, xt=None):
        """ctx (B,16,N,D) -> h (B,C,G,G) after the z-free blocks."""
        B = ctx.shape[0]
        x = ctx.permute(0, 1, 3, 2).reshape(B, T_CTX * D, G, G)
        if xt is not None:
            x = torch.cat([x, xt.permute(0, 1, 3, 2).reshape(B, T_FUT * D, G, G)], 1)
        h = self.stem(x) + self.pos
        if self.z_from > 0:
            w0 = self.const.expand(B, -1)
            for blk in self.blocks[:self.z_from]:
                h = blk(h, w0, None)
        return h

    def tail(self, h, z=None, t=None):
        B = h.shape[0]
        w = self._style(B, z, t)
        for blk in self.blocks[self.z_from:]:
            h = blk(h, w, None)
        o = self.head(F.layer_norm(h.permute(0, 2, 3, 1), h.shape[1:2]).permute(0, 3, 1, 2))
        return o.reshape(B, T_FUT, D, G * G).transpose(2, 3)  # (B,16,N,D)

    def forward(self, ctx, z=None, t=None, xt=None):
        return self.tail(self.trunk(ctx, xt), z, t)


class JointData:
    """Training clips of the A1 frame table: ctx = frames 0..15, future = frames 16..31."""

    def __init__(self, data, device="cuda"):
        d = ds_dir(data)
        self.tok = torch.from_numpy(np.load(f"{d}/train_tokens.npy"))
        self.fr = torch.from_numpy(np.load(f"{d}/train_clips.npz")["frames"])
        self.st = Stats(data, device)
        self.dev = device

    def get(self, idx):
        f = self.fr[idx].reshape(-1)
        x = self.tok[f].view(len(idx), T_CTX + T_FUT, 256, 384).pin_memory().to(self.dev, non_blocking=True)
        return self.st.norm(x)

    def loader(self, B, seed, prefetch=4):
        q = queue.Queue(prefetch)
        g = torch.Generator().manual_seed(seed)

        def work():
            while True:
                q.put(torch.randint(len(self.fr), (B,), generator=g))
        threading.Thread(target=work, daemon=True).start()
        while True:
            yield self.get(q.get())


def horizon_scale(data, n=512, seed=0):
    """Std of (future_k - last context frame) per horizon step k, normalized tokens -> (16,)."""
    d = ds_dir(data)
    tok = np.load(f"{d}/train_tokens.npy", mmap_mode="r")
    fr = np.load(f"{d}/train_clips.npz")["frames"]
    st = Stats(data)
    idx = np.random.default_rng(seed).choice(len(fr), n, replace=False)
    out = []
    for i in range(0, n, 64):
        f = fr[idx[i:i + 64]]
        x = st.norm(torch.from_numpy(np.asarray(tok[f.reshape(-1)])).cuda()).view(len(f), 32, 256, 384)
        out.append(x[:, T_CTX:] - x[:, T_CTX - 1:T_CTX])
    r = torch.cat(out)
    return r.transpose(0, 1).reshape(T_FUT, -1).std(1)  # (16,)


def train(a):
    torch.manual_seed(a.seed)
    out = os.path.join(RUNS, a.name); os.makedirs(out, exist_ok=True)
    data = JointData(a.data)
    scale = horizon_scale(a.data).cuda()  # (16,)
    margs = dict(method=a.method, C=a.C, nblocks=a.nblocks, wide=a.wide, z_from=a.z_from, t_shift=a.t_shift)
    model = JointGen(**margs).cuda()
    ema = copy.deepcopy(model).eval()
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.9), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    robust = Robust(1, "gm")
    loader = data.loader(a.bs, a.seed)
    log = open(f"{out}/log.jsonl", "w")
    print(f"params {nparam / 1e6:.2f}M clips {len(data.fr)} scale h1/h16 {scale[0]:.3f}/{scale[-1]:.3f}", flush=True)
    t0 = time.time()
    sc = scale[None, :, None, None]
    for step in range(a.steps + 1):
        x = next(loader)
        B = x.shape[0]
        ctx = x[:, :T_CTX]
        y = (x[:, T_CTX:] - x[:, T_CTX - 1:T_CTX]) / sc  # (B,16,N,D) normalized residual targets
        info = {}
        if a.method == "imle":
            m = a.m
            z = torch.randn(B, m, model.zdim, device=x.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.trunk(ctx)
                cb = max(1, a.match_chunk // m)
                ds = []
                for b0 in range(0, B, cb):
                    sl = slice(b0, b0 + cb); nb = h[sl].shape[0]
                    o_ = model.tail(h[sl].repeat_interleave(m, 0), z=z[sl].reshape(nb * m, -1)).float()
                    ds.append((o_ - y[sl].repeat_interleave(m, 0)).pow(2).mean((1, 2, 3)))
                dv = torch.cat(ds).view(B, m)
                j = dv.argmin(1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(ctx, z=z[torch.arange(B), j]).float()
            loss = robust((pred - y).pow(2).mean((1, 2, 3)), 0)
            info["match_d"] = dv.min(1).values.sqrt().mean().item()
        elif a.method == "fm":
            e = torch.randn_like(y)
            t = 1 - shift_noise(torch.rand(B, device=x.device), a.t_shift)  # 0 noise, 1 data
            xt = (1 - t[:, None, None, None]) * e + t[:, None, None, None] * y
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(ctx, t=t, xt=xt).float()
            loss = (v - (y - e)).pow(2).mean()
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(ctx).float()
            loss = (pred - y).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            dec = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - dec)
        if step % 200 == 0:
            info.update(step=step, loss=loss.item(), t=round(time.time() - t0, 1))
            log.write(json.dumps(info) + "\n"); log.flush()
            print(info, flush=True)
    torch.save({"ema": ema.state_dict(), "model_args": margs, "args": vars(a), "params": nparam,
                "scale": scale.cpu(), "train_time": time.time() - t0}, f"{out}/ckpt.pt")
    print("done", time.time() - t0, flush=True)
    os._exit(0)


@torch.no_grad()
def sample(model, scale, ctx, n, nfe=50):
    """ctx (1,16,N,D) normalized -> (n,16,N,D) normalized future frames."""
    c = ctx.expand(n, -1, -1, -1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if model.method == "imle":
            y = model(c, z=torch.randn(n, model.zdim, device=c.device)).float()
        elif model.method == "fm":
            y = torch.randn(n, T_FUT, 256, D, device=c.device)
            tg = 1 - shift_noise(torch.linspace(1, 0, nfe + 1, device=c.device), model.t_shift)
            for s in range(nfe):
                t = tg[s].expand(n)
                y = y + (tg[s + 1] - tg[s]) * model(c, t=t, xt=y).float()
        else:
            y = model(c).float()
    return c[:, -1:] + y * scale[None, :, None, None]


@torch.no_grad()
def evaluate(run, nfe=50, seed=123):
    torch.manual_seed(seed)
    ck = torch.load(f"{RUNS}/{run}/ckpt.pt", map_location="cuda")
    model = JointGen(**ck["model_args"]).cuda(); model.load_state_dict(ck["ema"]); model.eval()
    scale = ck["scale"].cuda()
    data = ck["args"]["data"]
    a1eval.DATA = data
    d = ds_dir(data); st = Stats(data); probe = a1eval.load_probe2()
    tok = np.load(f"{d}/evaluation_tokens.npy", mmap_mode="r")
    c = np.load(f"{d}/evaluation_clips.npz")
    fr, ic = c["frames"], c["ic"]
    gt_all = torch.from_numpy(np.stack([c["sq_pos"], c["ci_pos"]], 2))[:, T_CTX:]
    P, R = [], []
    t0 = time.time()
    for i in np.unique(ic):
        cl = np.where(ic == i)[0]
        F_ = len(cl)
        ctx = st.norm(torch.from_numpy(np.asarray(tok[fr[cl[0], :T_CTX]])).cuda())[None]
        Y = sample(model, scale, ctx, 2 * F_, nfe)
        preds = probe(Y.flatten(0, 1)).view(2 * F_, T_FUT, 2, 2).cpu()
        pe, re = a1eval.pr_errors(preds, gt_all[cl]); P += pe.tolist(); R += re.tolist()
    p, r = float(np.mean(P)), float(np.mean(R))
    res = {"model": {"precision_err": p, "recall_err": r, "f1_err": 2 * p * r / (p + r)},
           "run": run, "nfe": nfe if model.method == "fm" else 1, "params": ck["params"],
           "train_time": ck["train_time"], "eval_s": time.time() - t0}
    tag = f"_nfe{nfe}" if model.method == "fm" else ""
    res["readout"] = a1eval.READOUT
    json.dump(res, open(f"{RUNS}/{run}/eval{a1eval.RO_TAG}{tag}.json", "w"), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval")
    ap.add_argument("--nfe", type=int, default=50)
    ap.add_argument("--data", default="bshapes_v3")
    ap.add_argument("--method", default="imle", choices=["imle", "fm", "reg"])
    ap.add_argument("--name")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--C", type=int, default=384)
    ap.add_argument("--nblocks", type=int, default=12)
    ap.add_argument("--wide", type=int, default=1024)
    ap.add_argument("--z_from", type=int, default=8)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--match_chunk", type=int, default=256)
    ap.add_argument("--t_shift", type=float, default=13.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.eval:
        evaluate(a.eval, a.nfe)
    else:
        train(a)

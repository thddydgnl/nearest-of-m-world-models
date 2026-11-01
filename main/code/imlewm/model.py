"""ConvNeXt generator over a 16x16 DINO token grid.

style path (AdaLN): IMLE noise z / flow-matching time / constant (regression)
action path (FiLM): separate modulation, or merged into the style path (act_path='shared')
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

G = 16  # token grid side
D = 384  # DINOv2-S channels


def ln2d(x):
    return F.layer_norm(x.permute(0, 2, 3, 1), x.shape[1:2]).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, C, sdim, adim):
        super().__init__()
        self.dw = nn.Conv2d(C, C, 7, padding=3, groups=C)
        self.mod_s = nn.Linear(sdim, 2 * C)
        self.mod_a = nn.Linear(adim, 2 * C) if adim else None
        self.pw1 = nn.Conv2d(C, 4 * C, 1)
        self.pw2 = nn.Conv2d(4 * C, C, 1)
        self.ls = nn.Parameter(torch.full((C,), 0.1))
        for m in [self.mod_s, self.mod_a]:
            if m is not None:
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, w, ae):
        h = ln2d(self.dw(x))
        g, b = self.mod_s(w)[:, :, None, None].chunk(2, 1)
        h = h * (1 + g) + b
        if self.mod_a is not None:
            g, b = self.mod_a(ae)[:, :, None, None].chunk(2, 1)
            h = h * (1 + g) + b
        h = self.pw2(F.gelu(self.pw1(h)))
        return x + self.ls[None, :, None, None] * h


def mlp(i, h, o, n):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.GELU()]; d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def time_emb(t, dim=128, scale=1000.0):
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = (t[:, None] * scale) * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Gen(nn.Module):
    """method: imle | es (energy score; same noise-input generator as imle) | reg | fm (flow
    matching) | mf (MeanFlow, single head) | imf (improved MeanFlow, dual u/v heads; forward
    returns [v, u] so outs[-1] is the average velocity u).
    wide_head > 0: final head is a per-token MLP of that width (FlowWM's wide projection head).
    z_from > 0 (imle/es, late fusion): z modulates only blocks >= z_from; earlier blocks use a
    learned constant style, so trunk() runs once per transition and tail() once per candidate."""

    def __init__(self, method="imle", k=3, C=256, nblocks=8, zdim=256, sdim=256, adim=128,
                 act_dim=2, act_path="film", n_heads=3, wide_head=0, z_from=0, t_shift=1.0, t_scale=1000.0):
        super().__init__()
        self.method, self.k, self.zdim, self.act_path = method, k, zdim, act_path
        self.t_shift = t_shift  # FlowWM time shift (fm only), applied at train and sampling time
        self.t_scale = t_scale  # time-embedding frequency scale (MeanFlow JVPs differentiate through it)
        self.z_from = z_from if method in ("imle", "es") else 0
        noisy_in = method in ("fm", "mf", "imf", "sc")
        in_ch = k * D + (D if noisy_in else 0)
        self.stem = nn.Conv2d(in_ch, C, 1)
        self.pos = nn.Parameter(torch.zeros(1, C, G, G))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.act_mlp = mlp(act_dim, adim, adim, 2)
        if method in ("imle", "es"):
            s_in = zdim + (adim if act_path == "shared" else 0)
            self.map = mlp(s_in, sdim, sdim, 4)
            if self.z_from > 0:
                self.const = nn.Parameter(torch.zeros(1, sdim))
        elif method == "fm":
            self.map = mlp(128, sdim, sdim, 2)
        elif method in ("mf", "imf", "sc"):
            self.map = mlp(256, sdim, sdim, 2)
        else:
            self.const = nn.Parameter(torch.zeros(1, sdim))
        film = adim if act_path == "film" else 0
        self.blocks = nn.ModuleList([Block(C, sdim, film) for _ in range(nblocks)])
        # heads after blocks at 1/3, 2/3, end (or only end)
        self.head_at = [nblocks * (i + 1) // n_heads - 1 for i in range(n_heads)]
        self.heads = nn.ModuleList([nn.Conv2d(C, D, 1) for _ in range(n_heads)])
        if wide_head:
            self.heads[-1] = WideHead(C, wide_head)
        self.vhead = (WideHead(C, wide_head) if wide_head else nn.Conv2d(C, D, 1)) if method == "imf" else None
        for h in list(self.heads) + ([self.vhead] if self.vhead is not None else []):
            last = h if isinstance(h, nn.Conv2d) else h.out
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def _head(self, i, h):
        B = h.shape[0]
        return self.heads[i](ln2d(h)).reshape(B, D, G * G).transpose(1, 2)

    def _style(self, B, ae, z=None, t=None, r=None):
        if self.method in ("imle", "es"):
            zz = z / z.pow(2).mean(1, keepdim=True).add(1e-8).sqrt()
            return self.map(torch.cat([zz, ae], 1) if self.act_path == "shared" else zz)
        if self.method == "fm":
            return self.map(time_emb(t, scale=self.t_scale))
        if self.method in ("mf", "imf", "sc"):
            return self.map(torch.cat([time_emb(t, scale=self.t_scale), time_emb(t - r, scale=self.t_scale)], 1))
        return self.const.expand(B, -1)

    def trunk(self, ctx, a, xt=None):
        """Blocks that do not see z. Returns (h, ae)."""
        B = ctx.shape[0]
        x = ctx.permute(0, 1, 3, 2).reshape(B, self.k * D, G, G)
        if xt is not None:
            x = torch.cat([x, xt.transpose(1, 2).reshape(B, D, G, G)], 1)
        h = self.stem(x) + self.pos
        ae = self.act_mlp(a)
        if self.z_from > 0:
            w0 = self.const.expand(B, -1)
            film = ae if self.act_path == "film" else None
            for blk in self.blocks[:self.z_from]:
                h = blk(h, w0, film)
        return h, ae

    def tail(self, h, ae, z=None, t=None, r=None, all_heads=False):
        B = h.shape[0]
        w = self._style(B, ae, z, t, r)
        film = ae if self.act_path == "film" else None
        outs = []
        for i in range(self.z_from, len(self.blocks)):
            h = self.blocks[i](h, w, film)
            if i in self.head_at:
                hi = self.head_at.index(i)
                if all_heads or hi == len(self.heads) - 1:
                    outs.append(self._head(hi, h))
        if self.vhead is not None:
            v = self.vhead(ln2d(h)).reshape(B, D, G * G).transpose(1, 2)
            outs = [v] + outs
        return outs

    def forward(self, ctx, a, z=None, t=None, xt=None, all_heads=False, r=None):
        """ctx (B,k,N,D) normalized tokens; returns list of (B,N,D) raw head outputs."""
        h, ae = self.trunk(ctx, a, xt)
        return self.tail(h, ae, z, t, r, all_heads)


class WideHead(nn.Module):
    """Per-token MLP head C -> W -> W -> D (FlowWM: the velocity head must be wider than the
    per-patch latent)."""

    def __init__(self, C, W):
        super().__init__()
        self.inp = nn.Conv2d(C, W, 1)
        self.mid = nn.Conv2d(W, W, 1)
        self.out = nn.Conv2d(W, D, 1)

    def forward(self, x):
        return self.out(F.gelu(self.mid(F.gelu(self.inp(x)))))

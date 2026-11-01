"""Train regression / flow matching / conditional IMLE world models on DINO tokens.

IMLE options: --m (matching budget), --stages single|spatial|spectral, --loss l2|gm,
--H (trajectory matching length, N1), --curriculum, --act_path film|shared, --contrast (N4),
--seqmatch (N1b) with --seq_feed matched|random|gt (which frame becomes the next step's context)
Energy score (--method es): same noise-input generator, --m samples per transition, --es_beta,
--es_dist vec|token (whole token map as in ESP, or per token averaged over the grid)
"""
import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from .model import Gen
from .wm import K_CTX, TokenData, shift_noise

RUNS = os.path.expanduser("~/imle-wm/runs")


def stage_err(diff, l, stages, pca):
    """diff (B,N,D) prediction - target, returns per-sample mean squared error under K_l."""
    B = diff.shape[0]
    if stages == "spatial":
        r = [4, 8, 16][l]
        dd = diff.transpose(1, 2).reshape(B, 384, 16, 16)
        if r < 16:
            dd = F.avg_pool2d(dd, 16 // r)
        return dd.pow(2).mean((1, 2, 3))
    if stages == "spectral":
        d = [32, 128, 384][l]
        return (diff @ pca[:, :d]).pow(2).mean((1, 2))
    return diff.pow(2).mean((1, 2))


class Robust:
    """Geman-McClure on per-sample RMS distance, scale tracked as 2x running median."""

    def __init__(self, n, kind):
        self.kind, self.c = kind, [None] * n

    def __call__(self, e2, l):
        if self.kind == "l2":
            return e2.mean()
        d = (e2 + 1e-8).sqrt()
        med = d.detach().median()
        self.c[l] = med if self.c[l] is None else 0.99 * self.c[l] + 0.01 * med
        u2 = (d / (2 * self.c[l])).pow(2)
        return (2 * u2 / (u2 + 4)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--method", default="imle", choices=["imle", "es", "fm", "mf", "imf", "sc", "reg"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--nblocks", type=int, default=8)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--stages", default="spatial", choices=["single", "spatial", "spectral"])
    ap.add_argument("--loss", default="gm", choices=["l2", "gm"])
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--act_path", default="film", choices=["film", "shared"])
    ap.add_argument("--contrast", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--ctx_noise", type=float, default=0.0)
    ap.add_argument("--seqmatch", action="store_true", help="N1b: per-step matching on self-rollout context, no BPTT")
    ap.add_argument("--seq_feed", default="matched", choices=["matched", "random", "gt"],
                    help="N1b ablation: next-step context is the matched sample, a random candidate, or the true frame")
    ap.add_argument("--mf_p", type=float, default=1.0, help="MeanFlow adaptive weight exponent")
    ap.add_argument("--mf_bf16", action="store_true")
    ap.add_argument("--mf_clip", type=float, default=0.0)
    ap.add_argument("--mf_chunks", type=int, default=4)
    ap.add_argument("--mf_warmup", type=int, default=0)
    ap.add_argument("--mf_ratio", type=float, default=0.25, help="fraction of (r<s) pairs after warmup")
    ap.add_argument("--wide_head", type=int, default=0, help="FlowWM wide per-token head width (0 = off)")
    ap.add_argument("--t_shift", type=float, default=1.0, help="FlowWM time shift alpha for fm (13 in FlowWM)")
    ap.add_argument("--z_from", type=int, default=0, help="late fusion: z enters from this block on (imle)")
    ap.add_argument("--memo_every", type=int, default=2000, help="log train/val memorization (bounce data)")
    ap.add_argument("--t_scale", type=float, default=1000.0, help="time-embedding scale (fm/mf/imf)")
    ap.add_argument("--beta2", type=float, default=0.9)
    ap.add_argument("--zdim", type=int, default=256)
    ap.add_argument("--rs_eps", type=float, default=0.0,
                    help="RS-IMLE-style rejection: drop candidates closer than rs_eps x running median best distance")
    ap.add_argument("--match_pca", type=int, default=0, help="IMLE matching distance on the top-k channel PCs (0 = full)")
    ap.add_argument("--match_pool", type=int, default=0, help="IMLE matching distance on an RxR avg-pooled token grid (0 = full)")
    ap.add_argument("--match_chunk", type=int, default=0, help="late fusion H=1: candidates per matching chunk (0 = all)")
    ap.add_argument("--sc_frac", type=float, default=0.25, help="shortcut: self-consistency fraction of the batch")
    ap.add_argument("--save_every", type=int, default=0, help="also save EMA checkpoints ckpt_<step>.pt (learning curves)")
    ap.add_argument("--es_beta", type=float, default=1.0, help="energy score exponent, 0 < beta < 2")
    ap.add_argument("--es_chunks", type=int, default=2, help="energy score: backward per chunk of contexts")
    ap.add_argument("--es_dist", default="vec", choices=["vec", "token"],
                    help="energy score distance: whole token map (ESP) or per 384-D token, averaged over tokens")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out = os.path.join(RUNS, a.name)
    os.makedirs(out, exist_ok=True)
    data = TokenData(a.data, "train", H=a.H)
    st = data.stats
    n_heads = 3 if (a.method == "imle" and a.stages != "single") else 1
    margs = dict(method=a.method, C=a.C, nblocks=a.nblocks, act_path=a.act_path, n_heads=n_heads,
                 wide_head=a.wide_head, z_from=a.z_from, t_shift=a.t_shift, t_scale=a.t_scale,
                 act_dim=data.act.shape[1] if data.act is not None else 2, zdim=a.zdim)
    model = Gen(**margs).cuda()
    ema = copy.deepcopy(model).eval()
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, a.beta2), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    robust = Robust(n_heads, a.loss)

    def match_dist(diff):
        """Per-candidate matching distance; optionally in a coarse space (pooled grid / top PCs) so the
        nearest-of-m choice follows coarse, mode-like variation rather than fine high-dim detail."""
        if a.match_pca:
            diff = diff @ st.pca[:, :a.match_pca]
        if a.match_pool:
            n_, _, d_ = diff.shape
            diff = F.avg_pool2d(diff.transpose(1, 2).reshape(n_, d_, 16, 16), 16 // a.match_pool)
        return diff.pow(2).mean(tuple(range(1, diff.dim())))
    rs_med = None
    loader = data.loader(a.bs, a.seed)
    log = open(f"{out}/log.jsonl", "w")
    print(f"params {nparam/1e6:.2f}M  windows {len(data.valid)}", flush=True)
    t0 = time.time()

    for step in range(a.steps + 1):
        x, acts = next(loader)
        x = st.norm(x)  # (B, k+H, N, D)
        B = x.shape[0]
        ctx0 = x[:, :K_CTX]
        if a.ctx_noise > 0:  # blur context identity so per-context outcomes cannot be memorized
            ctx0 = ctx0 + a.ctx_noise * torch.randn_like(ctx0)
        H = a.H
        if a.curriculum and a.H > 1:
            H = 1 if step < a.steps // 3 else (min(4, a.H) if step < 2 * a.steps // 3 else a.H)
        tgt = x[:, K_CTX:K_CTX + H]
        info = {}
        did_backward = False
        if a.method == "imf":
            # Improved MeanFlow (Geng et al. 2025, arXiv 2512.02012), ported from the PyTorch
            # re-implementation of the official JAX code (haidog-yaqub/MeanFlow, mode i-meanflow).
            # Convention t=1 noise, t=0 data: z = (1-t) x + t e, v = e - x. Dual heads (u, v).
            # The JVP tangent is the model's own v prediction (no grad), and the loss is put on
            # v_est = u + (t-r) sg(du/dt) against v, plus a flow-matching loss on the v head.
            xd = (tgt[:, 0] - ctx0[:, -1]) / st.sigma_r
            e = torch.randn_like(xd)
            tt = torch.sigmoid(torch.randn(B, 2, device=x.device) * 1.0 - 0.4)
            t_, r_ = tt.max(1).values, tt.min(1).values
            zt = (1 - t_[:, None, None]) * xd + t_[:, None, None] * e
            v = e - xd
            a0 = acts[:, 0]
            with torch.no_grad():
                v_c = model(ctx0, a0, t=t_, r=r_, xt=zt)[0].float()
                _, dudt = torch.func.jvp(lambda zz, t1, r1: model(ctx0, a0, t=t1, r=r1, xt=zz)[-1].float(),
                                         (zt, t_, r_), (v_c, torch.ones_like(t_), torch.zeros_like(r_)))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = model(ctx0, a0, t=t_, r=r_, xt=zt)
            v_p, u_p = outs[0].float(), outs[-1].float()

            def adaptive_l2(err, c=1e-2):
                d2 = err.pow(2).mean((1, 2))
                return (d2 / (d2.detach() + c)).mean()
            fm_loss = adaptive_l2(v_p - v)
            v_est = u_p + (t_ - r_)[:, None, None] * dudt
            mf_loss = adaptive_l2(v_est - v)
            loss = fm_loss + mf_loss
            info["fm_loss"], info["mf_loss"] = fm_loss.item(), mf_loss.item()
            info["mf_v_mse"] = (v_est - v).pow(2).mean().item()
        elif a.method == "sc":
            # Shortcut model (Frans et al. 2025, arXiv 2410.12557) in the MeanFlow convention:
            # t=1 noise, t=0 data, z_t = (1-t) x + t e, s(z_t, t, r) is the step from t down to r
            # (d = t - r), z_r = z_t - d s. A (1 - sc_frac) part of the batch is flow matching (r = t),
            # the rest is self-consistency: s(z_t, t, t-2d) <- mean of two d-steps from the EMA model.
            # Step sizes 2d in {1, 1/2, ..., 1/64}; the 1/128 bootstrap steps use the flow (r = t).
            xd = (tgt[:, 0] - ctx0[:, -1]) / st.sigma_r
            e = torch.randn_like(xd)
            a0 = acts[:, 0]
            t_ = shift_noise(torch.rand(B, device=x.device), a.t_shift)
            r_ = t_.clone()
            v_t = e - xd
            nb = int(B * a.sc_frac)
            if nb:
                lv = torch.randint(0, 7, (nb,), device=x.device)
                D2 = 0.5 ** lv.float(); d = D2 / 2
                k = torch.floor(torch.rand(nb, device=x.device) * ((1 - D2) / d + 1))
                tb = (D2 + d * k).clamp(max=1.0)
                zb = (1 - tb[:, None, None]) * xd[:nb] + tb[:, None, None] * e[:nb]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    smallest = lv == 6
                    s1 = ema(ctx0[:nb], a0[:nb], t=tb, r=torch.where(smallest, tb, tb - d), xt=zb)[-1].float()
                    zm, tm = zb - d[:, None, None] * s1, tb - d
                    s2 = ema(ctx0[:nb], a0[:nb], t=tm, r=torch.where(smallest, tm, tm - d), xt=zm)[-1].float()
                t_ = torch.cat([tb, t_[nb:]]); r_ = torch.cat([tb - D2, r_[nb:]])
                v_t = torch.cat([(s1 + s2) / 2, v_t[nb:]])
            zt = (1 - t_[:, None, None]) * xd + t_[:, None, None] * e
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sp = model(ctx0, a0, t=t_, r=r_, xt=zt)[-1].float()
            err = (sp - v_t).pow(2).mean((1, 2))
            loss = err.mean()
            info["flow_mse"] = err[nb:].mean().item()
            if nb:
                info["boot_mse"] = err[:nb].mean().item()
        elif a.method == "mf":
            # MeanFlow (Geng et al. 2025) in the paper's convention: s=1 noise, s=0 data,
            # z_s = (1-s) x1 + s x0, v = x0 - x1, u(z_s, r, s) average velocity, r < s,
            # identity u = v - (s-r) du/ds with du/ds = v . dz u + ds u (JVP, stop-grad target).
            # Sampling: z_r = z_s - (s-r) u(z_s, r, s); one step: x1 = x0 - u(x0, 0, 1).
            x1 = (tgt[:, 0] - ctx0[:, -1]) / st.sigma_r
            x0 = torch.randn_like(x1)
            sa = torch.sigmoid(torch.randn(B, device=x.device) - 0.4)
            sb = torch.sigmoid(torch.randn(B, device=x.device) - 0.4)
            s_, r = torch.maximum(sa, sb), torch.minimum(sa, sb)
            p_fm = 1.0 if step < a.mf_warmup else 1.0 - a.mf_ratio  # pure flow matching during warmup
            r = torch.where(torch.rand(B, device=x.device) < p_fm, s_, r)
            zs = (1 - s_[:, None, None]) * x1 + s_[:, None, None] * x0
            v = x0 - x1
            # JVP doubles activation memory, so backward per micro-batch (a.mf_chunks of them)
            opt.zero_grad(set_to_none=True)
            loss, cs = 0.0, B // a.mf_chunks
            for i0 in range(0, B, cs):
                sl = slice(i0, i0 + cs)

                def f(zz, rr, ss):
                    return model(ctx0[sl], acts[sl, 0], t=ss, r=rr, xt=zz)[-1].float()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.mf_bf16):
                    u, duds = torch.func.jvp(f, (zs[sl], r[sl], s_[sl]), (v[sl], torch.zeros_like(r[sl]), torch.ones_like(s_[sl])))
                u_tgt = (v[sl] - (s_[sl] - r[sl])[:, None, None] * duds).detach()
                if a.mf_clip > 0:
                    u_tgt = u_tgt.clamp(-a.mf_clip, a.mf_clip)
                err = (u - u_tgt).pow(2).mean((1, 2))
                lc = (err / (err.detach() + 1e-3).pow(a.mf_p)).mean() * (cs / B)
                lc.backward()
                loss += lc.detach()
                info["mf_err"] = info.get("mf_err", 0.0) + err.mean().item() / a.mf_chunks
            did_backward = True
        elif a.method == "es":
            # Energy score (Gneiting & Raftery 2007), as in Engression (Shen & Meinshausen 2023) and
            # ESP (arXiv 2610.07696). It is strictly proper, so unlike nearest-of-m its optimum is the
            # true conditional law at any m. All m samples get gradient: mean distance to the target
            # minus half the mean pairwise distance (unbiased). Distances are RMS over tokens (the
            # Euclidean norm up to a constant), raised to es_beta, computed on predicted residuals.
            m = a.m
            opt.zero_grad(set_to_none=True)
            loss, cs = 0.0, B // a.es_chunks
            off = ~torch.eye(m, dtype=torch.bool, device=x.device)
            for i0 in range(0, B, cs):
                sl = slice(i0, i0 + cs); nb = ctx0[sl].shape[0]
                z = torch.randn(nb * m, model.zdim, device=x.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hc, ae = model.trunk(ctx0[sl], acts[sl, 0])
                    o_ = model.tail(hc.repeat_interleave(m, 0), ae.repeat_interleave(m, 0), z=z)[-1]
                r = o_.float().reshape(nb, m, -1)
                y = (tgt[sl, 0] - ctx0[sl, -1]).reshape(nb, 1, -1)
                if a.es_dist == "token":
                    # Per-token ES (each token a D=384 vector), averaged over the grid: proper for the token
                    # marginals only. Pair distances from explicit differences, exact near zero.
                    D = o_.shape[-1]
                    i_, j_ = torch.triu_indices(m, m, 1, device=x.device)
                    d_fit = (r - y).reshape(nb, m, -1, D).pow(2).mean(-1)  # (nb, m, N)
                    d_pair = (r[:, i_] - r[:, j_]).reshape(nb, len(i_), -1, D).pow(2).mean(-1)  # (nb, pairs, N)
                    e_fit = (d_fit + 1e-8).pow(a.es_beta / 2).mean((1, 2))
                    e_pair = (d_pair + 1e-8).pow(a.es_beta / 2).mean((1, 2))
                    d_fit = d_fit.mean(-1)  # whole-map distance, for the match_d log
                else:
                    d_fit = (r - y).pow(2).mean(-1)  # (nb, m)
                    sq = r.pow(2).mean(-1)
                    d_pair = (sq[:, :, None] + sq[:, None, :] - 2 * torch.bmm(r, r.transpose(1, 2)) / r.shape[-1]).clamp_min(0)
                    e_fit = (d_fit + 1e-8).pow(a.es_beta / 2).mean(1)
                    e_pair = (d_pair[:, off] + 1e-8).pow(a.es_beta / 2).mean(1)
                lc = (e_fit - 0.5 * e_pair).mean() * (nb / B)
                lc.backward()
                loss += lc.detach()
                info["es_fit"] = info.get("es_fit", 0.0) + e_fit.mean().item() * nb / B
                info["es_pair"] = info.get("es_pair", 0.0) + e_pair.mean().item() * nb / B
                info["match_d"] = info.get("match_d", 0.0) + d_fit.min(1).values.sqrt().mean().item() * nb / B
            did_backward = True
        elif a.method == "imle" and a.seqmatch:
            # N1b: walk the model's own rollout; at each step match m fresh z against the true
            # next frame given the *self-predicted* context, then supervise that step. Context is
            # detached between steps (scheduled-sampling style), so cost is m*H fwd + H fwd/bwd.
            # Each step is back-propagated right away: contexts are detached between steps, so this
            # gives the same gradient as summing the losses but keeps only one step's graph in memory.
            m = a.m
            c, loss, md = ctx0, 0.0, 0
            opt.zero_grad(set_to_none=True)
            for h in range(H):
                z = torch.randn(B, m, model.zdim, device=x.device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    cm = c.repeat_interleave(m, 0)
                    if model.z_from > 0:  # late fusion: trunk once per context, tail per candidate
                        hc, ae = model.trunk(c, acts[:, h])
                        o_ = model.tail(hc.repeat_interleave(m, 0), ae.repeat_interleave(m, 0), z=z.reshape(B * m, -1))[-1]
                    else:
                        o_ = model(cm, acts[:, h].repeat_interleave(m, 0), z=z.reshape(B * m, -1))[-1]
                    s_ = cm[:, -1] + o_.float()
                    d = (s_ - tgt[:, h].repeat_interleave(m, 0)).pow(2).mean((1, 2)).view(B, m)
                    j = d.argmin(1)
                md += d.min(1).values.sqrt().mean().item() / H
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outs = model(c, acts[:, h], z=z[torch.arange(B), j], all_heads=True)
                    preds = [c[:, -1] + o.float() for o in outs]
                    lh = 0
                    for l, p_ in enumerate(preds):
                        lh = lh + robust(stage_err(p_ - tgt[:, h], l, a.stages, st.pca), l) / (len(preds) * H)
                lh.backward()
                loss += lh.detach()
                if a.seq_feed == "gt":  # teacher forcing: the same matched updates on true contexts
                    nxt = tgt[:, h]
                elif a.seq_feed == "random":  # free-running: any candidate, not the one nearest the truth
                    nxt = s_.view(B, m, *s_.shape[1:])[torch.arange(B, device=x.device), torch.randint(m, (B,), device=x.device)]
                else:
                    nxt = preds[-1].detach()
                c = torch.cat([c[:, 1:], nxt[:, None]], 1)
            did_backward = True
            info["match_d"] = md; info["H"] = H
        elif a.method == "imle":
            # matching: m candidate self-rollouts of length H, no grad. Kept in its own autocast
            # region: autocast caches bf16 weight casts, and casts made under no_grad would be
            # reused by the grad pass and silently cut the gradient to the weights.
            m = a.m
            z = torch.randn(B, m, H, model.zdim, device=x.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if model.z_from > 0 and H == 1:
                    # late fusion, one step: z-free trunk once per transition, tail per candidate in
                    # chunks of match_chunk candidates; the m-fold context is never materialized
                    hc, ae = model.trunk(ctx0, acts[:, 0])
                    cb = max(1, a.match_chunk // m) if a.match_chunk else B
                    ds = []
                    for b0 in range(0, B, cb):
                        sl = slice(b0, b0 + cb); nb = hc[sl].shape[0]
                        o_ = model.tail(hc[sl].repeat_interleave(m, 0), ae[sl].repeat_interleave(m, 0),
                                        z=z[sl, :, 0].reshape(nb * m, -1))[-1]
                        s = ctx0[sl, -1].repeat_interleave(m, 0) + o_.float()
                        ds.append(match_dist(s - tgt[sl, 0].repeat_interleave(m, 0)))
                    dist = torch.cat(ds)
                else:
                    c = ctx0.repeat_interleave(m, 0)
                    dist = 0
                    for h in range(H):
                        zh = z[:, :, h].reshape(B * m, -1)
                        if model.z_from > 0 and h == 0:  # late fusion: z-free trunk once per transition
                            hc, ae = model.trunk(c[::m], acts[:, h])
                            o_ = model.tail(hc.repeat_interleave(m, 0), ae.repeat_interleave(m, 0), z=zh)[-1]
                        else:
                            o_ = model(c, acts[:, h].repeat_interleave(m, 0), z=zh)[-1]
                        s = c[:, -1] + o_.float()
                        dist = dist + match_dist(s - tgt[:, h].repeat_interleave(m, 0))
                        c = torch.cat([c[:, 1:], s[:, None]], 1)
                dv = dist.view(B, m)
                j = dv.argmin(1)
                if a.rs_eps > 0:
                    # RS-IMLE (Vashist et al. 2024) adapted per transition: candidates already within
                    # eps of their target are rejected, so the update goes to a z that is not yet there
                    dr = (dv / H).sqrt()
                    med = dr.min(1).values.median()
                    rs_med = med if rs_med is None else 0.99 * rs_med + 0.01 * med
                    rej = dr < a.rs_eps * rs_med
                    jr = dv.masked_fill(rej, float("inf")).argmin(1)
                    j = torch.where(rej.all(1), j, jr)
                    info["rs_rej"] = rej.float().mean().item()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.method not in ("mf", "imf", "sc", "es") and not a.seqmatch):
            if a.method in ("mf", "imf", "sc", "es") or a.seqmatch:
                pass
            elif a.method == "reg":
                pred = ctx0[:, -1] + model(ctx0, acts[:, 0])[-1].float()
                loss = (pred - tgt[:, 0]).pow(2).mean()
            elif a.method == "fm":
                x1 = (tgt[:, 0] - ctx0[:, -1]) / st.sigma_r
                x0 = torch.randn_like(x1)
                t = 1 - shift_noise(torch.rand(B, device=x.device), a.t_shift)  # t: 0 noise, 1 data
                xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
                v = model(ctx0, acts[:, 0], t=t, xt=xt)[-1].float()
                loss = (v - (x1 - x0)).pow(2).mean()
            else:
                zs = z[torch.arange(B), j]  # (B,H,zdim)
                c = ctx0
                loss = 0
                for h in range(H):
                    if a.grad_ckpt and H > 1:
                        outs = torch.utils.checkpoint.checkpoint(model, c, acts[:, h], zs[:, h], None, None, True, use_reentrant=False)
                    else:
                        outs = model(c, acts[:, h], z=zs[:, h], all_heads=True)
                    preds = [c[:, -1] + o.float() for o in outs]
                    for l, p in enumerate(preds):
                        loss = loss + robust(stage_err(p - tgt[:, h], l, a.stages, st.pca), l) / (len(preds) * H)
                    if h == 0 and a.contrast > 0:
                        perm = torch.randperm(B, device=x.device)
                        acf = acts[perm, 0]
                        pcf = c[:, -1] + model(c, acf, z=zs[:, 0])[-1].float()
                        d_true = (preds[-1] - tgt[:, 0]).pow(2).mean((1, 2)).add(1e-8).sqrt()
                        d_cf = (pcf - tgt[:, 0]).pow(2).mean((1, 2)).add(1e-8).sqrt()
                        mu = (acts[:, 0] - acf).norm(dim=1) / 2 * st.resid_rms
                        lact = F.relu(mu - d_cf + d_true).mean()
                        loss = loss + a.contrast * lact
                        info["lact"] = lact.item()
                    c = torch.cat([c[:, 1:], preds[-1][:, None]], 1)
                info["match_d"] = dist.view(B, m).min(1).values.div(H).sqrt().mean().item()
                info["H"] = H
        if not did_backward:
            opt.zero_grad(set_to_none=True)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            dec = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - dec)
        if a.memo_every and step and step % a.memo_every == 0 and a.data.startswith("bounce"):
            from .check import memo_model  # train/val realized-mode share (memorization index)
            mm = memo_model(ema, a.data, n_ctx=200)
            info["memo_train"], info["memo_val"] = mm["train"]["realized_share"][0], mm["test"]["realized_share"][0]
            info["cov_val"] = mm["test"]["cov"][0]
        if a.save_every and step and step % a.save_every == 0 and step < a.steps:
            torch.save({"ema": ema.state_dict(), "model_args": margs, "args": vars(a), "params": nparam,
                        "train_time": time.time() - t0, "step": step}, f"{out}/ckpt_{step}.pt")
        if step % 200 == 0 or "memo_train" in info:
            info.update(step=step, loss=loss.item(), t=round(time.time() - t0, 1))
            log.write(json.dumps(info) + "\n"); log.flush()
            print(info, flush=True)
    torch.save({"ema": ema.state_dict(), "model_args": margs, "args": vars(a), "params": nparam,
                "train_time": time.time() - t0}, f"{out}/ckpt.pt")
    print("done", time.time() - t0, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

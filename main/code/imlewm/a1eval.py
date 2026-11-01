"""A1 Bouncing Shapes evaluation with the FlowWM protocol (arXiv 2606.29059,
evaluators/evaluator.py: FlowMatchingEvaluatorSyntheticBouncingShapes).

For every evaluation initial condition (IC) with F ground-truth futures, the model produces 2F
rollouts of the 16 future frames (frames 16..31) from the last K_CTX context frames (13..15).
Positions of the square and the circle are read out per frame; with
  err(pred, gt) = 0.5 * (mean|sq_pred - sq_gt| + mean|ci_pred - ci_gt|)   (pixels at 256 px)
  precision error = mean over predictions of min over that IC's GT futures,
  recall error    = mean over GT futures of min over that IC's predictions,
  F1 error        = 2PR / (P + R).
FlowWM reads positions with a heatmap predictor on DINOv3 features; we use a soft-argmax probe
on DINOv2 tokens trained on real frames (its own floor is reported as `oracle`).

usage: python -m imlewm.a1eval --train_probe
       python -m imlewm.a1eval --run a1_imle [--nfe 50]
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from .common import ds_dir
from .wm import K_CTX, Stats, load_model, rollout

RUNS = os.path.expanduser("~/imle-wm/runs")
DATA = os.environ.get("A1_DATA", "bshapes")
READOUT = os.environ.get("A1_READOUT", "probe")  # "heatmap": FlowWM's pixel-heatmap readout (heatmap.py)
READOUT_TAG = os.environ.get("A1_READOUT_TAG", "")  # heatmap readout variant, e.g. "n05" (trained with noise 0.5)
RO_TAG = ("_hm" + READOUT_TAG) if READOUT == "heatmap" else ""
PATCH = 16.0  # pixels of the 256 px frame per token


class Probe2(nn.Module):
    """Soft-argmax over the 16x16 grid with sub-patch offsets, one map per shape."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(384, 128, 1), nn.GELU(), nn.Conv2d(128, 64, 3, padding=1),
                                 nn.GELU(), nn.Conv2d(64, 6, 1))
        c = (torch.arange(16).float() + 0.5) * PATCH
        self.register_buffer("cx", c[None, :].expand(16, 16).reshape(-1))
        self.register_buffer("cy", c[:, None].expand(16, 16).reshape(-1))

    def forward(self, tok):  # (B,256,384) normalized -> (B,2 shapes,2 xy)
        B = tok.shape[0]
        o = self.net(tok.transpose(1, 2).reshape(B, 384, 16, 16).float()).reshape(B, 2, 3, 256)
        w = o[:, :, 0].softmax(-1)
        x = (w * (self.cx + PATCH * torch.tanh(o[:, :, 1]))).sum(-1)
        y = (w * (self.cy + PATCH * torch.tanh(o[:, :, 2]))).sum(-1)
        return torch.stack([x, y], -1)


def frame_positions(split):
    """Per unique frame: (square xy, circle xy)."""
    d = ds_dir(DATA)
    c = np.load(f"{d}/{split}_clips.npz")
    n = c["frames"].max() + 1
    pos = np.zeros((n, 2, 2), np.float32)
    pos[c["frames"].reshape(-1), 0] = c["sq_pos"].reshape(-1, 2)
    pos[c["frames"].reshape(-1), 1] = c["ci_pos"].reshape(-1, 2)
    return pos


def train_probe(steps=6000):
    d = ds_dir(DATA); st = Stats(DATA)
    tok = np.load(f"{d}/train_tokens.npy", mmap_mode="r")
    pos = torch.from_numpy(frame_positions("train"))
    etok = np.load(f"{d}/evaluation_tokens.npy", mmap_mode="r")
    epos = torch.from_numpy(frame_positions("evaluation"))
    p = Probe2().cuda()
    opt = torch.optim.AdamW(p.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(pos), min(60000, len(pos)), replace=False))  # RAM-resident subset
    sub_tok = torch.from_numpy(np.asarray(tok[sub])); sub_pos = pos[sub]
    for s in range(steps):
        i = torch.from_numpy(rng.choice(len(sub), 256, replace=False))
        pr = p(st.norm(sub_tok[i].cuda()))
        loss = (pr - sub_pos[i].cuda()).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        ii = np.arange(0, len(epos), max(1, len(epos) // 4000))
        err = torch.cat([(p(st.norm(torch.from_numpy(np.asarray(etok[ii[j:j + 500]])).cuda())).cpu()
                          - epos[ii[j:j + 500]]).abs().mean((1, 2)) for j in range(0, len(ii), 500)])
    print(f"probe eval L1 px: mean {err.mean():.3f} p95 {err.quantile(0.95):.3f}", flush=True)
    torch.save(p.state_dict(), f"{d}/probe.pt")


def load_probe2():
    if READOUT == "heatmap":
        from . import heatmap
        return heatmap.load(DATA, READOUT_TAG)
    p = Probe2().cuda()
    p.load_state_dict(torch.load(f"{ds_dir(DATA)}/probe.pt"))
    return p.eval()


def pr_errors(preds, gts):
    """preds (P,T,2,2), gts (F,T,2,2) -> (per-pred min error (P,), per-gt min error (F,))."""
    e = 0.5 * ((preds[:, None] - gts[None]).abs().mean((2, 4)).sum(-1))  # (P,F): mean over T,xy; sum shapes
    return e.min(1).values, e.min(0).values


@torch.no_grad()
def evaluate(model, nfe=50, seed=123):
    torch.manual_seed(seed)
    d = ds_dir(DATA); st = Stats(DATA); probe = load_probe2()
    tok = np.load(f"{d}/evaluation_tokens.npy", mmap_mode="r")
    c = np.load(f"{d}/evaluation_clips.npz")
    fr, ic = c["frames"], c["ic"]
    gt_all = torch.from_numpy(np.stack([c["sq_pos"], c["ci_pos"]], 2))[:, 16:]  # (clips,16,2,2)
    P_err, R_err, P_or, R_or, P_copy, R_copy = [], [], [], [], [], []
    for i in np.unique(ic):
        cl = np.where(ic == i)[0]
        F_ = len(cl)
        ctx = st.norm(torch.from_numpy(np.asarray(tok[fr[cl[0], 16 - K_CTX:16]])).cuda())[None]
        gts = gt_all[cl]
        if model is not None:
            R = rollout(model, ctx, torch.zeros(1, 16, 2, device="cuda"), 2 * F_, nfe, st.sigma_r)[0]
            preds = probe(R.flatten(0, 1)).view(2 * F_, 16, 2, 2).cpu()
            pe, re = pr_errors(preds, gts); P_err += pe.tolist(); R_err += re.tolist()
        # oracle: the true futures through the same readout (readout floor)
        orc = probe(st.norm(torch.from_numpy(np.asarray(tok[fr[cl, 16:].reshape(-1)])).cuda())).view(F_, 16, 2, 2).cpu()
        pe, re = pr_errors(orc, gts); P_or += pe.tolist(); R_or += re.tolist()
        # copy-last-frame baseline
        last = probe(ctx[0, -1:]).cpu().expand(16, 2, 2)[None]
        pe, re = pr_errors(last, gts); P_copy += pe.tolist(); R_copy += re.tolist()

    def summ(P, R):
        p, r = float(np.mean(P)), float(np.mean(R))
        return {"precision_err": p, "recall_err": r, "f1_err": 2 * p * r / (p + r), "n_pred": len(P), "n_gt": len(R)}
    out = {"oracle": summ(P_or, R_or), "copy_last": summ(P_copy, R_copy)}
    if model is not None:
        out["model"] = summ(P_err, R_err)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--nfe", type=int, default=50)
    ap.add_argument("--train_probe", action="store_true")
    a = ap.parse_args()
    if a.train_probe:
        train_probe()
        print(json.dumps(evaluate(None), indent=1))
    else:
        model, ck = load_model(f"{RUNS}/{a.run}/ckpt.pt")
        DATA = ck["args"]["data"]
        res = evaluate(model, a.nfe)
        res.update(run=a.run, nfe=a.nfe if model.method in ("fm", "mf", "imf", "sc") else 1, params=ck["params"],
                   train_time=ck["train_time"])
        tag = f"_nfe{a.nfe}" if model.method in ("fm", "mf", "imf", "sc") else ""
        json.dump(res, open(f"{RUNS}/{a.run}/eval{RO_TAG}{tag}.json", "w"), indent=1)
        print(json.dumps(res, indent=1))

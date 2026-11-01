"""B (stochastic Push-T) readout probe and multi-future evaluation.

Probe: per-token maps -> soft-argmax agent and block positions (env pixels, 512 scale) and the
block angle (sin, cos pooled with the block attention). Multi-future test (mf_test.npz): 128
pre-contact starts, each with 64 ground-truth futures (4 hidden CoG modes x 16 jitters) of 16
steps. Per start the model draws n rollouts from the 3 context frames under the recorded actions.
  err(pred, gt) = mean over steps and the 8 T vertices of the L2 distance (env px)
  precision error = mean over rollouts of min over GT futures; recall error = mean over GT futures
  of min over rollouts (FlowWM protocol); coverage = share of distinct CoG modes that some
  rollout lands on (nearest GT future's mode, within 0.45x the mode separation).

usage: python -m imlewm.b_eval --train_probe [--data pusht_b]
       python -m imlewm.b_eval --run b_imle [--nfe 50]
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from .common import ds_dir
from .pusht_sim import tee_keypoints
from .wm import Stats, load_model, rollout

RUNS = os.path.expanduser("~/imle-wm/runs")
PATCH = 32.0  # env pixels per token (512 / 16)


class ProbeB(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(384, 128, 1), nn.GELU(), nn.Conv2d(128, 64, 3, padding=1),
                                 nn.GELU(), nn.Conv2d(64, 8, 1))
        c = (torch.arange(16).float() + 0.5) * PATCH
        self.register_buffer("cx", c[None, :].expand(16, 16).reshape(-1))
        self.register_buffer("cy", c[:, None].expand(16, 16).reshape(-1))

    def forward(self, tok):  # (B,256,384) normalized -> (B,5): agent xy, block xy, angle
        B = tok.shape[0]
        o = self.net(tok.transpose(1, 2).reshape(B, 384, 16, 16).float()).reshape(B, 8, 256)
        pts = []
        for k in range(2):  # 0 agent, 1 block
            w = o[:, 3 * k].softmax(-1)
            x = (w * (self.cx + PATCH * torch.tanh(o[:, 3 * k + 1]))).sum(-1)
            y = (w * (self.cy + PATCH * torch.tanh(o[:, 3 * k + 2]))).sum(-1)
            pts += [x, y]
        wb = o[:, 3].softmax(-1)
        sc = torch.stack([(wb * o[:, 6]).sum(-1), (wb * o[:, 7]).sum(-1)], -1)
        ang = torch.atan2(sc[:, 0], sc[:, 1])
        return torch.stack(pts + [ang], -1)


def angle_diff(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def train_probe(data, steps=6000):
    d = ds_dir(data); st = Stats(data)
    tok = torch.from_numpy(np.load(f"{d}/train_tokens.npy"))
    pos = torch.from_numpy(np.load(f"{d}/train_meta.npz")["pos"]).float()
    ttok = torch.from_numpy(np.load(f"{d}/test_tokens.npy", mmap_mode="r")[:3000].copy())
    tpos = torch.from_numpy(np.load(f"{d}/test_meta.npz")["pos"][:3000]).float()
    p = ProbeB().cuda()
    opt = torch.optim.AdamW(p.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        i = torch.randint(len(tok), (256,))
        pr = p(st.norm(tok[i].cuda())); g = pos[i].cuda()
        loss = (pr[:, :4] - g[:, :4]).abs().mean() + 30 * angle_diff(pr[:, 4], g[:, 4]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        pr = torch.cat([p(st.norm(ttok[i:i + 500].cuda())).cpu() for i in range(0, len(ttok), 500)])
    kp_err = np.linalg.norm(tee_keypoints(pr.numpy()) - tee_keypoints(tpos.numpy()), axis=-1).mean(-1)
    print(f"{data} probe test: agent L1 {(pr[:, :2] - tpos[:, :2]).abs().mean():.2f}px, block L1 "
          f"{(pr[:, 2:4] - tpos[:, 2:4]).abs().mean():.2f}px, angle {np.degrees(angle_diff(pr[:, 4], tpos[:, 4]).abs().mean()):.2f}deg, "
          f"T-vertex L2 mean {kp_err.mean():.2f} p95 {np.percentile(kp_err, 95):.2f}px", flush=True)
    torch.save(p.state_dict(), f"{d}/probe.pt")


def load_probeB(data):
    p = ProbeB().cuda()
    p.load_state_dict(torch.load(f"{ds_dir(data)}/probe.pt"))
    return p.eval()


@torch.no_grad()
def mf_eval(model, data="pusht_b", n=16, nfe=50, seed=123, fname="mf_test_v2.npz", H=16, Hs=None):
    """Hs: evaluate several horizons on prefixes of one H=max(Hs) rollout and also return per-start
    errors (for paired bootstrap); then the result is {h: metrics} and metrics carry 'per_start'."""
    torch.manual_seed(seed)
    d = ds_dir(data); st = Stats(data); probe = load_probeB(data)
    f = np.load(f"{d}/{fname}")
    hs = list(Hs) if Hs else [H]
    Hm = max(hs)
    acc = {h: {"P": [], "R": [], "Ps": [], "Rs": [], "cov": [], "prec": [], "K": []} for h in hs}
    for i in range(len(f["ctx"])):
        ctx = st.norm(torch.from_numpy(f["ctx"][i]).cuda())[None]
        acts = torch.from_numpy(f["act"][i]).cuda()[None]
        gt_all = torch.from_numpy(tee_keypoints(f["fut_states"][i]))  # (64,16,8,2)
        labels = f["fut_mode"][i]
        R = rollout(model, ctx, acts[:, :Hm], n, nfe, st.sigma_r)[0]  # (n,Hm,N,D)
        ps_ = probe(R.flatten(0, 1)).view(n, Hm, 5).cpu().numpy()
        pr_all = torch.from_numpy(tee_keypoints(ps_))  # (n,Hm,8,2)
        for h in hs:
            gt_kp, pr_kp, a_ = gt_all[:, :h], pr_all[:, :h], acc[h]
            e = (pr_kp[:, None] - gt_kp[None]).norm(dim=-1).mean((2, 3))  # (n,64)
            pe, re = e.min(1).values, e.min(0).values
            a_["P"] += pe.tolist(); a_["R"] += re.tolist()
            a_["Ps"].append(pe.mean().item()); a_["Rs"].append(re.mean().item())
            # mode coverage
            mode_mean = torch.stack([gt_kp[labels == m].mean(0) for m in range(4)])  # (4,h,8,2)
            dmm = (mode_mean[:, None] - mode_mean[None]).norm(dim=-1).mean((2, 3))
            keep = []
            for m in range(4):  # merge CoG modes whose outcomes coincide (< 2 px apart)
                if all(dmm[m, k] >= 2.0 for k in keep):
                    keep.append(m)
            mm = mode_mean[keep]
            K = len(keep)
            sep = (dmm[keep][:, keep] + torch.eye(K) * 1e9).min(1).values if K > 1 else torch.full((1,), 1e9)
            tau = torch.clamp(0.45 * sep, max=20.0)
            dm = (pr_kp[:, None] - mm[None]).norm(dim=-1).mean((2, 3))  # (n,K)
            j = dm.argmin(1); ok = dm[torch.arange(n), j] < tau[j]
            a_["cov"].append(len(set(j[ok].tolist())) / K); a_["prec"].append(ok.float().mean().item())
            a_["K"].append(K)
    out = {}
    for h in hs:
        a_ = acc[h]
        p, r = float(np.mean(a_["P"])), float(np.mean(a_["R"]))
        out[h] = {"precision_err": p, "recall_err": r, "f1_err": 2 * p * r / (p + r),
                  "mode_cov": float(np.mean(a_["cov"])), "mode_prec": float(np.mean(a_["prec"])),
                  "n_modes": float(np.mean(a_["K"])), "n_starts": len(f["ctx"]), "n": n,
                  "per_start": {"precision_err": a_["Ps"], "recall_err": a_["Rs"]}}
    if Hs:
        return out
    o = out[H]; o.pop("per_start")
    return o


@torch.no_grad()
def oracle(data="pusht_b"):
    """Readout floor on true futures is not available as tokens; report probe error on test frames."""
    d = ds_dir(data); st = Stats(data); probe = load_probeB(data)
    tok = torch.from_numpy(np.load(f"{d}/test_tokens.npy", mmap_mode="r")[:2000].copy())
    pos = np.load(f"{d}/test_meta.npz")["pos"][:2000]
    pr = torch.cat([probe(st.norm(tok[i:i + 500].cuda())).cpu() for i in range(0, 2000, 500)]).numpy()
    return {"probe_T_vertex_err": float(np.linalg.norm(tee_keypoints(pr) - tee_keypoints(pos), axis=-1).mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--data", default="pusht_b")
    ap.add_argument("--nfe", type=int, default=50)
    ap.add_argument("--train_probe", action="store_true")
    ap.add_argument("--perstart", action="store_true",
                    help="one H=16 rollout scored at h=8 and 16; writes eval_ps{tag}.json with per-start errors")
    a = ap.parse_args()
    if a.train_probe:
        train_probe(a.data)
    elif a.perstart:
        model, ck = load_model(f"{RUNS}/{a.run}/ckpt.pt")
        gen = model.method in ("fm", "mf", "imf", "sc")
        out = mf_eval(model, ck["args"]["data"], nfe=a.nfe, Hs=(8, 16))
        res = {"mf": out[16], "mf_h8": out[8], "run": a.run, "nfe": a.nfe if gen else 1}
        tag = f"_nfe{a.nfe}" if gen else ""
        json.dump(res, open(f"{RUNS}/{a.run}/eval_ps{tag}.json", "w"))
        print(a.run, res["nfe"], {h: (round(out[h]["precision_err"], 2), round(out[h]["recall_err"], 2)) for h in (8, 16)}, flush=True)
    else:
        model, ck = load_model(f"{RUNS}/{a.run}/ckpt.pt")
        res = {"mf": mf_eval(model, ck["args"]["data"], nfe=a.nfe), "mf_h8": mf_eval(model, ck["args"]["data"], nfe=a.nfe, H=8),
               "oracle": oracle(ck["args"]["data"]),
               "run": a.run, "nfe": a.nfe if model.method in ("fm", "mf", "imf", "sc") else 1,
               "params": ck["params"], "train_time": ck["train_time"]}
        tag = f"_nfe{a.nfe}" if model.method in ("fm", "mf", "imf", "sc") else ""
        json.dump(res, open(f"{RUNS}/{a.run}/eval{tag}.json", "w"), indent=1)
        print(json.dumps(res, indent=1))

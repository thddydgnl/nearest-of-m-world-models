"""Revision r1, analysis a (DESIGN_r1.md section 4a): codebook distortion in token space, I_tok(m, K).

I(m, K) in the paper is the expected squared distance from a uniformly drawn mode to the nearest of m i.i.d.
uniform mode draws, computed on unit-speed bounce directions (paper/build/make_figures.py:I_mK). Here the same
expectation is computed per evaluation context from the K stored branch token maps (stoch_branch_tokens.npy,
400 held-out wall contexts), standardized with the dataset's stats.pt:
  plain: d_kl^2 = mean squared difference of the standardized maps (the IMLE matching distance)
  gm:    rho(d_kl) = 2u^2/(u^2+4), u = d_kl/(2c), d_kl the RMS distance (the training loss), with c the
         Geman-McClure scale of the matching loss. The scale is not stored in checkpoints; it is re-estimated
         from each run's EMA checkpoint as the median, over 128-window batches of random training transitions,
         of the smallest RMS distance among m candidates to the target (Robust in train.py tracks the running
         median of exactly this distance); the log's final match_d (batch mean) is kept as a second estimate.
Also: the same expectation on the context's branch positions in px (I_pos) and the closed form I_mK.

subcommands:
  tok --data D [--scale_runs R ...]   -> itok_<D>.json
  fit --grid_out DIR [--b_dir DIR]    -> fit of the 16 budget-grid points (K, m) != (2, 2) against each variable

usage: python -m imlewm.r1_itok tok --data bounce_div_k4 --scale_runs e2_k4_m4_s0 ... --out /workspace/nom/out/a_itok
       python -m imlewm.r1_itok fit --out /workspace/nom/out/a_itok [--b_dir /workspace/nom/out/b_grid]
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import torch

from . import r1_util
from .common import ds_dir
from .wm import Stats, load_model, predict

M_LIST = (2, 4, 8, 16, 32, 64)


def p_rank(K, m):
    """P(the nearest of m uniform draws is the r-th closest mode), r = 1..K (ties broken by rank)."""
    r = np.arange(1, K + 1)
    return ((K - r + 1) / K) ** m - ((K - r) / K) ** m


def expect_nearest(Dsq, m):
    """Dsq (n, K, K) 'distortion' between modes per context -> mean over contexts and true modes of the
    expected distortion to the nearest (by Dsq order) of m uniform draws."""
    n, K, _ = Dsq.shape
    srt = np.sort(Dsq, axis=2)  # srt[..., 0] = 0 (the mode itself)
    return float((srt * p_rank(K, m)[None, None]).sum(2).mean())


def I_mK(m, K, arc=180.0):
    ang = np.deg2rad((np.arange(K) + 0.5) / K * arc - arc / 2)
    pts = np.stack([np.cos(ang), np.sin(ang)], 1)
    D = ((pts[:, None] - pts[None]) ** 2).sum(-1)
    return expect_nearest(D[None], m)


def gm(d, c):
    u2 = (d / (2 * c)) ** 2
    return 2 * u2 / (u2 + 4)


@torch.no_grad()
def pair_dist(data):
    """RMS distances (400, K, K) between the standardized branch token maps of each evaluation context."""
    d = ds_dir(data); st = Stats(data)
    bt = np.load(f"{d}/stoch_branch_tokens.npy", mmap_mode="r")
    out = []
    for i in range(len(bt)):
        x = st.norm(torch.from_numpy(np.asarray(bt[i])).cuda()).flatten(1)  # (K, N*D)
        out.append((x[:, None] - x[None]).pow(2).mean(-1).sqrt().cpu().numpy())
    bpos = np.load(f"{d}/eval_idx.npz")["stoch_branch_pos"]  # (400, K, 2) px
    dpos = np.linalg.norm(bpos[:, :, None] - bpos[:, None], axis=-1)
    return np.stack(out), dpos


_WIN = {}


def train_windows(data, n=512, seed=0):
    """n random training transitions t -> t+1 with t-2..t+1 in one episode (the TokenData H=1 windows)."""
    if data not in _WIN:
        meta = r1_util.load_meta(data, "train")
        seg = meta["seg"]
        t = np.arange(2, len(seg) - 1)
        t = t[seg[t - 2] == seg[t + 1]]
        sel = np.sort(np.random.default_rng(seed).choice(t, n, replace=False))
        tok = np.load(f"{ds_dir(data)}/train_tokens.npy", mmap_mode="r")
        _WIN[data] = r1_util.frames(tok, sel, [-2, -1, 0, 1])
    return _WIN[data]


@torch.no_grad()
def gm_scale(run, bs=128):
    """Median over each 128-window batch of the best-of-m RMS distance, averaged over the batches."""
    model, ck = load_model(f"{r1_util.RUNS}/{run}/ckpt.pt")
    data, m = ck["args"]["data"], ck["args"]["m"]
    st = Stats(data)
    W = train_windows(data)
    torch.manual_seed(0)
    meds, means = [], []
    for i0 in range(0, len(W), bs):
        w = st.norm(W[i0:i0 + bs].cuda())
        S = predict(model, w[:, :3], torch.zeros(len(w), 2, device="cuda"), m, 1, st.sigma_r, chunk=2048)
        d = (S - w[:, 3][:, None]).pow(2).mean((-1, -2)).sqrt().min(1).values
        meds.append(float(d.median())); means.append(float(d.mean()))
    rows = [json.loads(l) for l in open(f"{r1_util.RUNS}/{run}/log.jsonl")]
    md = [r["match_d"] for r in rows if "match_d" in r]
    return {"run": run, "data": data, "m": m, "c_median": float(np.mean(meds)), "best_mean": float(np.mean(means)),
            "log_match_d_final": md[-1] if md else None, "log_match_d_last5": float(np.mean(md[-5:])) if md else None}


def tok_cmd(data, scale_runs, out):
    path = f"{out}/itok_{data}.json"
    if os.path.exists(path):
        print("skip", path); return
    from . import common
    K = r1_util.set_geometry(data)
    D, Dp = pair_dist(data)
    iu = np.triu_indices(K, 1)
    res = {"data": data, "K": K, "arc_deg": common.ARC_DEG, "n_ctx": len(D),
           "pair_rms": {"mean": float(D[:, iu[0], iu[1]].mean()), "median": float(np.median(D[:, iu[0], iu[1]])),
                        "nearest_mean": float((D + np.eye(K)[None] * 1e9).min(2).mean())},
           "pair_px": {"mean": float(Dp[:, iu[0], iu[1]].mean()),
                       "nearest_mean": float((Dp + np.eye(K)[None] * 1e9).min(2).mean())},
           "I_unit": {m: I_mK(m, K, common.ARC_DEG) for m in M_LIST},
           "I_pos_px2": {m: expect_nearest(Dp ** 2, m) for m in M_LIST},
           "I_tok": {m: expect_nearest(D ** 2, m) for m in M_LIST},
           "miss": {m: (1 - 1 / K) ** m for m in M_LIST}, "scales": [], "I_tok_gm": {}}
    for run in scale_runs:
        s = gm_scale(run)
        res["scales"].append(s)
        print("scale", s, flush=True)
    by_m = {}
    for s in res["scales"]:
        by_m.setdefault(s["m"], []).append(s["c_median"])
    for m, cs in sorted(by_m.items()):
        c = float(np.mean(cs))
        res["I_tok_gm"][m] = {"c": c, "I": expect_nearest(gm(D, c), m),
                              "I_c_half": expect_nearest(gm(D, c / 2), m), "I_c_double": expect_nearest(gm(D, 2 * c), m),
                              "rho_nearest_pair_mean": float(gm((D + np.eye(K)[None] * 1e9).min(2), c).mean())}
    r1_util.write_json(res, path)
    print(data, {k: res[k] for k in ("pair_rms", "I_tok")}, {m: round(v["I"], 4) for m, v in res["I_tok_gm"].items()}, flush=True)


def last_memo(run):
    p = f"{r1_util.RUNS}/{run}/log.jsonl"
    rows = [json.loads(l) for l in open(p) if "memo_train" in l]
    return rows[-1]["memo_train"] if rows else None


def fit_cmd(out, b_dir=None):
    tok = {}
    for K in (2, 4, 8, 16):
        p = f"{out}/itok_bounce_div_k{K}.json"
        tok[K] = json.load(open(p))
    pts = []
    for K in (2, 4, 8, 16):
        for m in M_LIST:
            pref = "e2" if K in (4, 8) else "e12"
            runs = [f"{pref}_k{K}_m{m}_s{s}" for s in (0, 1) if os.path.isdir(f"{r1_util.RUNS}/{pref}_k{K}_m{m}_s{s}")]
            vals = [v for v in (last_memo(r) for r in runs) if v is not None]
            if not vals:
                continue
            row = {"K": K, "m": m, "runs": runs, "memo": float(np.mean(vals)), "memo_seeds": vals}
            row["excess"] = (row["memo"] - 1 / K) / (1 - 1 / K)
            if b_dir:
                bj = [json.load(open(f)) for r in runs for f in glob.glob(f"{b_dir}/{r}_ckpt_nfeNone_400x64.json")]
                if bj:
                    dv = [b["derived"] for b in bj]
                    for key in ("index_train", "excess_chance", "excess_heldout", "free_index_train",
                                "free_excess_chance", "free_excess_heldout"):
                        row["b_" + key] = float(np.mean([v[key] for v in dv]))
                    row["b_gap_min_dist"] = float(np.mean([v["gap_min_dist"][0] for v in dv]))
            t = tok[K]
            row.update(I_unit=t["I_unit"][str(m)], I_pos_px2=t["I_pos_px2"][str(m)], I_tok=t["I_tok"][str(m)],
                       I_tok_gm=t["I_tok_gm"].get(str(m), {}).get("I"), miss=(1 - 1 / K) ** m,
                       inv_m2=1 / m ** 2, inv_m=1 / m, exp_mK=math.exp(-m / K))
            pts.append(row)
    use = [p for p in pts if not (p["K"] == 2 and p["m"] == 2)]

    def fit(xkey, ykey):
        P = [p for p in use if p.get(xkey) is not None and p.get(ykey) is not None]
        x = np.array([p[xkey] for p in P]); y = np.array([p[ykey] for p in P])
        b, a = np.polyfit(x, y, 1)
        rm = float(np.sqrt(np.mean((a + b * x - y) ** 2)))
        loko = []
        for K in (2, 4, 8, 16):  # leave one K out: fit on the other K, predict this K
            tr = np.array([p["K"] != K for p in P])
            if tr.all() or (~tr).all():
                continue
            bb, aa = np.polyfit(x[tr], y[tr], 1)
            loko += list(aa + bb * x[~tr] - y[~tr])
        return {"n": len(P), "a": float(a), "b": float(b), "rmse": rm,
                "rmse_leave_one_K_out": float(np.sqrt(np.mean(np.square(loko)))) if loko else None}
    xs = ("I_unit", "I_pos_px2", "I_tok", "I_tok_gm", "miss", "inv_m2", "inv_m", "exp_mK")
    ys = ["excess"] + [k for k in ("b_excess_chance", "b_free_excess_chance", "b_excess_heldout") if all(k in p for p in use)]
    res = {"points": pts, "fits": {y: {x: fit(x, y) for x in xs} for y in ys},
           "note": "16 points: all (K, m) of the budget grid except K=2, m=2 (as in the paper); excess from the "
                   "last memo_train of log.jsonl (the monitor), mean of two seeds"}
    r1_util.write_json(res, f"{out}/fit_grid.json")
    for y in ys:
        print(y, {x: (round(v["rmse"], 4), round(v["rmse_leave_one_K_out"], 4)) for x, v in res["fits"][y].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tok", "fit"])
    ap.add_argument("--data")
    ap.add_argument("--scale_runs", nargs="*", default=[])
    ap.add_argument("--b_dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "tok":
        tok_cmd(a.data, a.scale_runs, a.out)
    else:
        fit_cmd(a.out, a.b_dir)

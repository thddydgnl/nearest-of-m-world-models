"""B: stochastic Push-T built from DINO-WM pusht_noise action sequences.

Hidden per-episode variable: the T block's centre of gravity, one of 4 positions at radius 20 px
around the default (+ 3 px jitter). Under Push-T's quasi-static pushing (damping 0) the total mass
and friction have no effect on outcomes, but the CoG changes how the T turns under the same push,
so one pushing context has up to 4 outcome modes. Nominal replay reproduces DINO-WM's recorded
states exactly.

Outputs (data/pusht_b, and data/pusht_det for the deterministic control with nominal CoG):
  {split}_tokens.npy (N,256,384) fp16, {split}_meta.npz: pos (N,5) agent xy, block xy, angle;
  act (N,10) normalized frameskip-5 action from frame t to t+1; seg; cog_mode (N,).
  mf_test.npz (pusht_b only): 128 pre-contact starts x 64 futures (4 CoG modes x 16 jitters).

usage: python -m imlewm.gen_pusht --name pusht_b
"""
import argparse
import multiprocessing as mp
import os

import numpy as np
import torch

from . import pusht_sim as ps
from .common import DinoEncoder, ds_dir, save_json

N_MODES, COG_R, COG_JIT = 4, 20.0, 3.0
DEFAULT_COG = np.array([0.0, 45.0])
MODE_DIRS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], np.float64)


def make_env(phys, state0):
    env = ps.pusht_env.PushTEnv(damping=0.0, block_cog=tuple(phys["cog"]),
                                reset_to_state=np.asarray(state0[:5], np.float64), relative=True,
                                action_scale=100, with_velocity=False, with_target=True, render_size=224)
    env.reset()
    return env


ps.make_env = make_env


def cog_sample(rng, mode=None):
    mode = int(rng.integers(N_MODES)) if mode is None else mode
    jit = rng.normal(0, COG_JIT, 2)
    return {"cog": DEFAULT_COG + COG_R * MODE_DIRS[mode] + jit, "mode": mode}


def _episode(args):
    ep, stochastic, seed = args
    states, rel, lens = _DATA
    rng = np.random.default_rng(seed)
    phys = cog_sample(rng) if stochastic else {"cog": DEFAULT_COG, "mode": -1}
    L = lens[ep]
    frames, st, acts = ps.rollout_episode(states[ep, 0], rel[ep, :L], phys)
    return ep, frames, st, acts, phys["mode"]


_DATA = None


def _init(split):
    global _DATA
    _DATA = ps.load_split(split)


def build_split(out, split_name, episodes, stochastic, enc, seed, workers=32):
    toks, pos, act, seg, mode = [], [], [], [], []
    with mp.Pool(workers, initializer=_init, initargs=("train",)) as pool:
        jobs = [(ep, stochastic, seed * 100000 + ep) for ep in episodes]
        for i, (ep, frames, st, acts, md) in enumerate(pool.imap(_episode, jobs, chunksize=4)):
            toks.append(enc(frames)); pos.append(st); act.append(acts)
            seg.append(np.full(len(st), i)); mode.append(np.full(len(st), md))
            if i % 200 == 0:
                print(f"{split_name}: {i}/{len(episodes)} episodes", flush=True)
    toks = np.concatenate(toks)
    np.save(f"{out}/{split_name}_tokens.npy", toks)
    np.savez(f"{out}/{split_name}_meta.npz", pos=np.concatenate(pos), act=np.concatenate(act),
             seg=np.concatenate(seg), cog_mode=np.concatenate(mode))
    print(f"{split_name}: {len(toks)} frames from {len(episodes)} episodes", flush=True)


def _mf_start(args):
    """Start right before first contact (v2): context frames t0-2..t0 show no block motion under any
    CoG, and the pushes reach the block in the step after t0, so futures diverge immediately."""
    ep, seed = args
    states, rel, lens = _DATA
    rng = np.random.default_rng(seed)
    L = lens[ep]
    n_all = L // ps.FRAMESKIP
    _, nom, acts = ps.rollout_episode(states[ep, 0], rel[ep, :L], {"cog": DEFAULT_COG}, n_wm=n_all, render=False)
    moved = np.where(np.abs(nom[1:, 2:] - nom[:-1, 2:]).max(1) > 0.5)[0]  # block moves on step t -> t+1
    if len(moved) == 0:
        return None
    t0 = int(moved[0])  # last frame before the block moves
    if t0 < 2 or t0 + 16 > n_all:
        return None
    futs, labels = [], []
    for m in range(N_MODES):
        for j in range(16):
            _, st, _ = ps.rollout_episode(states[ep, 0], rel[ep, :L], cog_sample(rng, m), n_wm=t0 + 16, render=False)
            if np.abs(st[t0 - 2:t0 + 1, 2:] - nom[t0 - 2:t0 + 1, 2:]).max() > 1e-3:
                return None  # context not identical across CoGs
            futs.append(st[t0 + 1:t0 + 17]); labels.append(m)
    futs = np.stack(futs)
    frames, _, _ = ps.rollout_episode(states[ep, 0], rel[ep, :L], {"cog": DEFAULT_COG}, n_wm=t0)
    return ep, frames[t0 - 2:t0 + 1], acts[t0:t0 + 16], futs, np.array(labels), t0


def build_mf_test(out, episodes, enc, seed, n_starts=128, workers=32, fname="mf_test_v2.npz"):
    ctx, acts, futs, labels, eps, t0s = [], [], [], [], [], []
    with mp.Pool(workers, initializer=_init, initargs=("train",)) as pool:
        for r in pool.imap(_mf_start, [(ep, seed * 100000 + ep) for ep in episodes], chunksize=2):
            if r is None:
                continue
            ep, frames, a, f, lab, t0 = r
            ctx.append(enc(frames)); acts.append(a); futs.append(f); labels.append(lab); eps.append(ep); t0s.append(t0)
            if len(ctx) >= n_starts:
                pool.terminate()
                break
    np.savez(f"{out}/{fname}", ctx=np.stack(ctx), act=np.stack(acts), fut_states=np.stack(futs),
             fut_mode=np.stack(labels), episode=np.array(eps), t0=np.array(t0s))
    fs = np.stack(futs)
    c = tee_kp_centroid(fs)
    spread = np.linalg.norm(c - c.mean(1, keepdims=True), axis=-1).mean((0, 1))
    print(f"{fname}: {len(ctx)} starts x {fs.shape[1]} futures x {fs.shape[2]} steps; "
          f"GT spread h=1,4,8,16: {spread[[0, 3, 7, 15]].round(1)}", flush=True)


def tee_kp_centroid(states):
    return ps.tee_keypoints(states).mean(-2)


def compute_stats(out, n_sub=6000, seed=0):
    toks = np.load(f"{out}/train_tokens.npy", mmap_mode="r")
    seg = np.load(f"{out}/train_meta.npz")["seg"]
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(len(toks) - 1, n_sub, replace=False))
    x = torch.from_numpy(np.asarray(toks[sub])).cuda().float()
    mean = x.mean((0, 1)); std = x.std((0, 1)) + 1e-4
    xn = ((x - mean) / std).reshape(-1, 384)
    evals, evecs = torch.linalg.eigh((xn.T @ xn) / len(xn))
    order = torch.argsort(evals, descending=True)
    same = torch.from_numpy(seg[sub] == seg[sub + 1]).cuda()
    x1 = torch.from_numpy(np.asarray(toks[sub + 1])).cuda().float()
    r = ((x1 - mean) / std - (x - mean) / std)[same]
    torch.save({"mean": mean.cpu(), "std": std.cpu(), "pca": evecs[:, order].cpu(), "evals": evals[order].cpu(),
                "sigma_r": r.std().item(), "resid_rms_median": r.pow(2).mean((1, 2)).sqrt().median().item()},
               f"{out}/stats.pt")
    print("sigma_r", r.std().item())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pusht_b", choices=["pusht_b", "pusht_det"])
    ap.add_argument("--train_eps", type=int, default=2400)
    ap.add_argument("--test_eps", type=int, default=300)
    ap.add_argument("--mf_only", action="store_true")
    a = ap.parse_args()
    if a.mf_only:
        build_mf_test(ds_dir("pusht_b"), list(range(18000, 18685)), DinoEncoder(), seed=3)
        raise SystemExit
    out = ds_dir(a.name)
    os.makedirs(out, exist_ok=True)
    enc = DinoEncoder()
    stochastic = a.name == "pusht_b"
    base = 0 if stochastic else 4800  # disjoint episode ranges for the two datasets
    build_split(out, "train", list(range(base, base + a.train_eps)), stochastic, enc, seed=1)
    build_split(out, "test", list(range(17000, 17000 + a.test_eps)), stochastic, enc, seed=2)
    if stochastic:
        build_mf_test(out, list(range(18000, 18685)), enc, seed=3)
    # v2 only: python -m imlewm.gen_pusht --mf_only
    compute_stats(out)
    save_json(vars(a) | {"cog_modes": N_MODES, "cog_radius": COG_R, "cog_jitter": COG_JIT,
                         "frameskip": ps.FRAMESKIP, "source": "DINO-WM pusht_noise (OSF bmw48)"},
              f"{out}/gen_args.json")

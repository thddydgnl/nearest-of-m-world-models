"""Shared helpers for the revision-r1 analysis scripts (main/revision/DESIGN_r1.md). Existing modules are not changed.

- atomic JSON output (temporary file, then rename; existing outputs are skipped by the callers)
- the wall-context selection of check.memo_model (same filter, same permutation)
- the K branch positions of a wall context and their rendered, DINO-encoded token maps
"""
import json
import os
import re

import numpy as np
import torch

from . import common
from .common import bounce_branches, ds_dir, render

RUNS = os.path.expanduser("~/imle-wm/runs")
OUT = os.environ.get("NOM_OUT", "/workspace/nom/out")


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def n_modes(data):
    return int(re.search(r"_k(\d+)", data).group(1))


def set_geometry(data):
    """K from the dataset name and the bounce arc (bounce_div_k<K>_arc<deg>), as check.memo_model does."""
    common.set_arc_from_name(data)
    return n_modes(data)


def load_meta(data, split):
    m = np.load(f"{ds_dir(data)}/{split}_meta.npz")
    return {k: m[k] for k in m.files}


def memo_contexts(meta, n_ctx, seed=0):
    """Wall frames t with t-2..t+1 in one episode, permuted with seed, first n_ctx (check.memo_model)."""
    seg, hit = meta["seg"], meta["hit"]
    idx = np.array([t for t in range(2, len(seg) - 1) if hit[t] and seg[t - 2] == seg[t + 1]])
    return np.random.default_rng(seed).permutation(idx)[:n_ctx]


def branches(data, meta, t, K):
    """K branch positions of wall frame t (check.memo_model: appearance speed and radius on div data)."""
    kw = dict(speed=meta["app"][t, 7], r=meta["app"][t, 6]) if "div" in data else {}
    return bounce_branches(meta["pos"][t], meta["normal"][t], K, **kw)


def realized_mode(bpos, meta, t):
    return int(np.argmin(np.linalg.norm(bpos - meta["pos"][t + 1], axis=1)))


def encode_branch_maps(enc, data, meta, idx, K, bs=256):
    """Render the K branch positions of each context with its appearance and DINO-encode them.
    Returns fp16 tokens (len(idx), K, 256, 384) and positions (len(idx), K, 2)."""
    bp = np.stack([branches(data, meta, t, K) for t in idx])
    app = np.repeat(meta["app"][idx], K, 0) if "div" in data else None
    flat = bp.reshape(-1, 2)
    toks = []
    for i in range(0, len(flat), bs):
        toks.append(enc(render(flat[i:i + bs], "bounce", None if app is None else app[i:i + bs])))
    return np.concatenate(toks).reshape(len(idx), K, 256, 384), bp


def frames(tok, idx, offs):
    """tok memmap (T,256,384) fp16, idx (n,), offs list -> torch fp16 (n, len(offs), 256, 384) on CPU."""
    fi = (np.asarray(idx)[:, None] + np.asarray(offs)[None]).reshape(-1)
    return torch.from_numpy(np.asarray(tok[fi])).view(len(idx), len(offs), 256, 384)

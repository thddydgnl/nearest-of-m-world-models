"""Real video: BAIR robot pushing (softmotion30_44k, Ebert et al. 2017), action-conditioned.

Each trajectory has 30 frames of 64x64 RGB (main camera 'image_aux1'), a 4-d action per frame and
the 3-d end-effector position. Random objects are pushed by a randomly moving arm, so given the
actions the arm is (nearly) deterministic while object motion is stochastic; there are no mode
labels, and the evaluation (bair_eval.py) uses best-of-n / manifold / spread / FD instead.

The tar lists the test file first and then the train files in shuffled order, so a prefix of the
download is a random subset: train = the first `train_files` train files, test = the official test
file + the next `extra_test_files` train files (held out, more sequences for distribution metrics).

Outputs (data/bair): {split}_tokens.npy (N,256,384) fp16, {split}_meta.npz: pos (N,3) end-effector,
act (N,4) normalized action taken from frame t to t+1, seg; stats.pt.

usage: python -m imlewm.gen_bair --tar ~/imle-wm/data/_raw/bair/bair.tar
"""
import argparse
import os
import struct
import tarfile

import numpy as np

from .common import DinoEncoder, ds_dir, save_json
from .gen_pusht import compute_stats

T_SEQ = 30


def _varint(b, i):
    r, s = 0, 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s; s += 7
        if c < 0x80:
            return r, i


def _fields(b):
    """Minimal protobuf wire decoder: yields (field number, wire type, value)."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        f, w = key >> 3, key & 7
        if w == 0:
            v, i = _varint(b, i)
        elif w == 2:
            n, i = _varint(b, i)
            v = b[i:i + n]; i += n
        elif w == 5:
            v = b[i:i + 4]; i += 4
        elif w == 1:
            v = b[i:i + 8]; i += 8
        else:
            raise ValueError(f"wire type {w}")
        yield f, w, v


def parse_example(b):
    """tf.train.Example -> {key: list of bytes | float32 array}."""
    out = {}
    for f, _, feats in _fields(b):  # Example.features
        for f2, _, entry in _fields(feats):  # Features.feature map entries
            key, val = None, None
            for f3, _, v in _fields(entry):
                if f3 == 1:
                    key = v.decode()
                elif f3 == 2:
                    val = v
            for kind, _, lst in _fields(val):  # Feature oneof: 1 bytes, 2 float, 3 int64
                if kind == 1:
                    out[key] = [v for _, _, v in _fields(lst)]
                elif kind == 2:
                    fl = []
                    for _, w, v in _fields(lst):
                        fl.append(np.frombuffer(v, "<f4") if w == 2 else np.frombuffer(v, "<f4"))
                    out[key] = np.concatenate(fl) if fl else np.zeros(0, np.float32)
    return out


def read_tfrecords(f):
    while True:
        h = f.read(12)
        if len(h) < 12:
            return
        n = struct.unpack("<Q", h[:8])[0]
        data = f.read(n); f.read(4)
        yield data


def trajectories(fobj):
    for rec in read_tfrecords(fobj):
        ex = parse_example(rec)
        img = np.stack([np.frombuffer(ex[f"{t}/image_aux1/encoded"][0], np.uint8).reshape(64, 64, 3) for t in range(T_SEQ)])
        act = np.stack([ex[f"{t}/action"] for t in range(T_SEQ)]).astype(np.float32)
        eef = np.stack([ex[f"{t}/endeffector_pos"] for t in range(T_SEQ)]).astype(np.float32)
        yield img, act, eef


def members(tar_path, n_train, n_extra):
    """Returns (test members, train members) in tar order, stopping once enough are listed."""
    tf = tarfile.open(tar_path, "r")
    test, train = [], []
    try:
        for m in tf:
            if not m.name.endswith(".tfrecords"):
                continue
            (test if "/test/" in m.name else train).append(m)
            if test and len(train) >= n_train + n_extra:
                break
    except (tarfile.ReadError, EOFError):
        pass
    return tf, test, train


def build(tf, mems, out, split, enc, act_stats=None):
    toks, pos, act, seg = [], [], [], []
    i = 0
    for m in mems:
        for img, a, e in trajectories(tf.extractfile(m)):
            toks.append(enc(img)); act.append(a); pos.append(e); seg.append(np.full(T_SEQ, i)); i += 1
        print(f"{split}: {m.name} -> {i} trajectories", flush=True)
    act = np.concatenate(act)
    if act_stats is None:
        act_stats = (act.mean(0), act.std(0) + 1e-6)
    np.save(f"{out}/{split}_tokens.npy", np.concatenate(toks))
    np.savez(f"{out}/{split}_meta.npz", pos=np.concatenate(pos), act=((act - act_stats[0]) / act_stats[1]).astype(np.float32),
             seg=np.concatenate(seg))
    return act_stats, i


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default=os.path.expanduser("~/imle-wm/data/_raw/bair/bair.tar"))
    ap.add_argument("--train_files", type=int, default=16)
    ap.add_argument("--extra_test_files", type=int, default=3)
    a = ap.parse_args()
    out = ds_dir("bair")
    os.makedirs(out, exist_ok=True)
    tf, test, train = members(a.tar, a.train_files, a.extra_test_files)
    assert test and len(train) >= a.train_files + a.extra_test_files, (len(test), len(train))
    enc = DinoEncoder()
    st, n_tr = build(tf, train[:a.train_files], out, "train", enc)
    _, n_te = build(tf, test + train[a.train_files:a.train_files + a.extra_test_files], out, "test", enc, st)
    compute_stats(out)
    save_json(vars(a) | {"n_train": n_tr, "n_test": n_te, "act_mean": st[0].tolist(), "act_std": st[1].tolist(),
                         "train_members": [m.name for m in train[:a.train_files]],
                         "source": "BAIR robot pushing softmotion30_44k (rail.eecs.berkeley.edu)"},
              f"{out}/gen_args.json")

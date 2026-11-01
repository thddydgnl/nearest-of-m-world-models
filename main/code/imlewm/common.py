"""Synthetic environments, rendering, DINOv2 encoding, dataset IO."""
import json
import os

import numpy as np
import torch

W = 64
R = 4.0
LO, HI = R, W - R
BG = np.array([28, 30, 34], np.float32)
BALL = np.array([245, 205, 70], np.float32)
ICE_C = np.array([55, 95, 165], np.float32)
PIT_C = np.array([150, 40, 40], np.float32)

# push env layout (x0, y0, x1, y1) in pixel coords, y grows downward
PIT = (24.0, 0.0, 40.0, 14.0)
ICE = (20.0, 14.0, 44.0, 40.0)

_yy, _xx = (np.mgrid[0:W, 0:W].astype(np.float32) + 0.5)

DATA_ROOT = os.environ.get("IMLE_DATA", os.path.expanduser("~/imle-wm/data"))


def _rect_mask(r):
    x0, y0, x1, y1 = r
    return (_xx >= x0) & (_xx < x1) & (_yy >= y0) & (_yy < y1)


def background(kind):
    base = np.broadcast_to(BG, (W, W, 3)).copy()
    if kind == "push":
        base[_rect_mask(ICE)] = ICE_C
        base[_rect_mask(PIT)] = PIT_C
    return base


# appearance vector: bg rgb(3) ball rgb(3) radius(1) speed(1) + 2 distractor rects (x0,y0,x1,y1,r,g,b)
APP_DIM = 22


def sample_app(rng):
    bg, ball = rng.uniform(10, 110, 3), rng.uniform(150, 255, 3)
    radius, speed = rng.uniform(3.0, 6.0), rng.uniform(10.0, 18.0)
    rects = []
    for _ in range(2):
        x0, y0 = rng.uniform(0, 48, 2); w, h = rng.uniform(6, 20, 2)
        rects += [x0, y0, x0 + w, y0 + h, *rng.uniform(40, 200, 3)]
    return np.array([*bg, *ball, radius, speed, *rects], np.float32)


def render(pos, kind="bounce", app=None):
    """pos (B,2) float x,y -> (B,W,W,3) uint8 with an anti-aliased ball.
    app (B,APP_DIM) overrides colours / radius / distractors per frame (bounce only)."""
    pos = np.asarray(pos, np.float32).reshape(-1, 2)
    B = len(pos)
    if app is None:
        base = np.repeat(background(kind)[None], B, 0)
        r, ball = np.full(B, R, np.float32), BALL
    else:
        app = np.asarray(app, np.float32).reshape(B, APP_DIM)
        base = np.repeat(app[:, None, None, 0:3], W, 1).repeat(W, 2)
        for i in range(2):
            o = 8 + 7 * i
            m = ((_xx[None] >= app[:, o, None, None]) & (_xx[None] < app[:, o + 2, None, None]) &
                 (_yy[None] >= app[:, o + 1, None, None]) & (_yy[None] < app[:, o + 3, None, None]))
            base = np.where(m[..., None], app[:, None, None, o + 4:o + 7], base)
        r, ball = app[:, 6], app[:, None, None, 3:6]
    d = np.sqrt((_xx[None] - pos[:, 0, None, None]) ** 2 + (_yy[None] - pos[:, 1, None, None]) ** 2)
    a = np.clip(r[:, None, None] + 0.5 - d, 0, 1)[..., None]
    img = base * (1 - a) + ball * a
    return img.round().astype(np.uint8)


def rot(v, th):
    c, s = np.cos(th), np.sin(th)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


# ---------------------------------------------------------------- bounce (A)
S_BOUNCE = 14.0


def bounce_angles(K):
    return np.deg2rad((np.arange(K) + 0.5) / K * 180.0 - 90.0)


def clamp_hit(q, r=R):
    lo, hi = r, W - r
    qc = np.clip(q, lo, hi)
    n = np.zeros(2)
    if q[0] < lo: n[0] = 1
    elif q[0] > hi: n[0] = -1
    if q[1] < lo: n[1] = 1
    elif q[1] > hi: n[1] = -1
    hit = bool(n.any())
    if hit:
        n = n / np.linalg.norm(n)
    return qc, hit, n


MODE_P = None  # optional bounce mode probabilities (asymmetric-mode variant); None = uniform


def bounce_step(p, v, K, rng, k=None, speed=S_BOUNCE, r=R):
    """Returns next pos, next vel, hit flag, normal, chosen mode.
    The new direction is sampled at the wall but only becomes visible one frame later."""
    q = p + v
    qc, hit, n = clamp_hit(q, r)
    if hit:
        if k is None:
            k = int(rng.integers(K)) if MODE_P is None else int(rng.choice(K, p=MODE_P))
        v = speed * rot(n, bounce_angles(K)[k])
    return qc, v, hit, n, (k if hit else -1)


def bounce_episode(K, T, rng, speed=S_BOUNCE, r=R):
    p = rng.uniform(r + 2, W - r - 2, size=2)
    phi = rng.uniform(0, 2 * np.pi)
    v = speed * np.array([np.cos(phi), np.sin(phi)])
    pos, normals, hits = [p.copy()], [np.zeros(2)], [False]
    for _ in range(T - 1):
        p, v, hit, n, _ = bounce_step(p, v, K, rng, speed=speed, r=r)
        pos.append(p.copy()); normals.append(n.copy()); hits.append(hit)
    # transition t -> t+1 is stochastic iff frame t is a wall frame
    return np.array(pos), np.array(normals), np.array(hits)


def bounce_branches(p_wall, n_wall, K, speed=S_BOUNCE, r=R):
    """All K possible next positions from a wall frame."""
    out = []
    for th in bounce_angles(K):
        v = speed * rot(n_wall, th)
        qc, _, _ = clamp_hit(p_wall + v, r)
        out.append(qc)
    return np.array(out)


# ---------------------------------------------------------------- push (A')
S_PUSH = 10.0


def in_rect(p, r):
    return r[0] <= p[0] < r[2] and r[1] <= p[1] < r[3]


def push_modes(p):
    if in_rect(p, PIT):
        return None  # absorbing
    if in_rect(p, ICE):
        return np.deg2rad([-60.0, 0.0, 60.0])
    return np.deg2rad([-20.0, 20.0])


def push_step(p, a, rng=None, k=None):
    modes = push_modes(p)
    if modes is None:
        return p.copy(), -1
    k = int(rng.integers(len(modes))) if k is None else k
    q = p + S_PUSH * rot(a, modes[k])
    return np.clip(q, LO, HI), k


def push_branches(p, a):
    modes = push_modes(p)
    if modes is None:
        return p[None].copy()
    return np.array([push_step(p, a, k=k)[0] for k in range(len(modes))])


def random_action(rng):
    if rng.random() < 0.1:
        return np.zeros(2)
    phi = rng.uniform(0, 2 * np.pi)
    return rng.uniform(0.3, 1.0) * np.array([np.cos(phi), np.sin(phi)])


def push_start(rng):
    while True:
        p = rng.uniform(LO, HI, size=2)
        if not in_rect(p, PIT):
            return p


def push_episode(T, rng, stuck_max=4):
    """Random-action episode. After `stuck_max` frames in the pit the ball respawns,
    which starts a new segment (seg id increments)."""
    p = push_start(rng)
    pos, acts, seg, modes = [], [], [], []
    s, stuck = 0, 0
    for _ in range(T):
        a = random_action(rng)
        pos.append(p.copy()); acts.append(a); seg.append(s)
        if in_rect(p, PIT):
            stuck += 1
            if stuck >= stuck_max:
                p = push_start(rng); s += 1; stuck = 0
                modes.append(-2)
                continue
        p, k = push_step(p, a, rng)
        modes.append(k)
    return np.array(pos), np.array(acts), np.array(seg), np.array(modes)


# ---------------------------------------------------------------- encoder
class DinoEncoder:
    def __init__(self, device="cuda", name="facebook/dinov2-small"):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(name).to(device).eval().half()
        self.dev = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, imgs, bs=512):
        """imgs uint8 (B,64,64,3) -> patch tokens fp16 (B,256,384) numpy"""
        outs = []
        for i in range(0, len(imgs), bs):
            x = torch.from_numpy(imgs[i:i + bs]).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
            x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
            x = ((x - self.mean) / self.std).half()
            h = self.m(pixel_values=x).last_hidden_state[:, 1:]
            outs.append(h.cpu().numpy().astype(np.float16))
        return np.concatenate(outs)


class Dinov3Encoder:
    """FlowWM's DINOv3 feature definition (arXiv 2606.29059, trainer/utils_latents.py): ViT-S/16 at
    the native 256 px (no resize), hidden states of layers 3, 6, 9, 12, each through the final
    norm, averaged; the CLS and 4 register tokens are dropped -> (B, 256, 384)."""

    def __init__(self, device="cuda", name="facebook/dinov3-vits16-pretrain-lvd1689m"):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(name).to(device).eval().half()
        self.dev = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, imgs, bs=256):
        outs = []
        for i in range(0, len(imgs), bs):
            x = torch.from_numpy(imgs[i:i + bs]).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
            assert x.shape[-1] == 256, "FlowWM protocol feeds 256 px frames without resizing"
            x = ((x - self.mean) / self.std).half()
            hs = self.m(pixel_values=x, output_hidden_states=True).hidden_states
            h = torch.stack([self.m.norm(hs[k][:, 5:]) for k in (3, 6, 9, 12)], -1).mean(-1)
            outs.append(h.float().cpu().numpy().astype(np.float16))
        return np.concatenate(outs)


def ds_dir(name):
    return os.path.join(DATA_ROOT, name)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)

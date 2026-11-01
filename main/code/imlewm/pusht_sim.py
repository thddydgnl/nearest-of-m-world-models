"""Stochastic Push-T: replay DINO-WM pusht_noise action sequences under per-episode random physics.

Hidden per-episode parameters (block mass, agent-block friction, space damping = floor friction)
are what make the same pushes lead to different outcomes. Frameskip 5 as in DINO-WM: one
world-model step = 5 env steps, action = the 5 relative actions concatenated (10-d, normalized).
"""
import importlib.util
import os
import pickle

import numpy as np
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
_ENV = os.path.expanduser("~/imle-wm/third_party/dino_wm/env/pusht/pusht_env.py")
_spec = importlib.util.spec_from_file_location("pusht_env", _ENV)
pusht_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pusht_env)

DINOWM = os.path.expanduser("~/imle-wm/data/dinowm/pusht_noise")
FRAMESKIP = 5
ACTION_MEAN = np.array([-0.0087, 0.0068], np.float32)  # DINO-WM dataset stats (rel_actions / 100)
ACTION_STD = np.array([0.2019, 0.2002], np.float32)
NOMINAL = dict(mass=1.0, friction=1.0, damping=0.0)


def sample_physics(rng):
    return dict(mass=float(np.exp(rng.uniform(np.log(0.3), np.log(3.0)))),
                friction=float(rng.uniform(0.2, 1.5)),
                damping=float(rng.uniform(0.0, 0.6)))


def load_split(split="train"):
    d = f"{DINOWM}/{split}"
    states = torch.load(f"{d}/states.pth").numpy()
    rel = torch.load(f"{d}/rel_actions.pth").numpy()
    lens = pickle.load(open(f"{d}/seq_lengths.pkl", "rb"))
    return states, rel, lens


def make_env(phys, state0):
    env = pusht_env.PushTEnv(damping=phys["damping"], reset_to_state=np.asarray(state0[:5], np.float64),
                             relative=True, action_scale=100, with_velocity=False, with_target=True,
                             render_size=224)
    env.reset()
    b = env.block
    scale = phys["mass"] / b.mass
    b.mass = phys["mass"]
    b.moment = b.moment * scale
    for s in b.shapes:
        s.friction = phys["friction"]
    return env


def block_state(env):
    return np.array([env.agent.position.x, env.agent.position.y, env.block.position.x,
                     env.block.position.y, env.block.angle], np.float32)


def rollout_episode(state0, rel_actions, phys, n_wm=None, render=True):
    """Replay env-step relative actions (pixels) under `phys`. Returns frames (T,224,224,3) uint8,
    states (T,5), actions (T,10) where action[t] moves frame t -> t+1 (last row zero)."""
    env = make_env(phys, state0)
    n_env = len(rel_actions)
    n_wm = n_env // FRAMESKIP if n_wm is None else n_wm
    frames = [env._render_frame("rgb_array")] if render else []
    states = [block_state(env)]
    acts = []
    for t in range(n_wm):
        chunk = rel_actions[t * FRAMESKIP:(t + 1) * FRAMESKIP] / 100.0
        for a in chunk:
            env.step(a)
        if render:
            frames.append(env._render_frame("rgb_array"))
        states.append(block_state(env))
        acts.append(((chunk - ACTION_MEAN) / ACTION_STD).reshape(-1))
    acts.append(np.zeros(2 * FRAMESKIP, np.float32))
    env.close()
    return (np.stack(frames) if render else None), np.stack(states), np.stack(acts).astype(np.float32)


_VERTS = None


def _tee_vertices():
    """Local T vertices in a fixed order. body.shapes is a set, so its iteration order changes
    between env instances; sort the shapes (and keep each polygon's own vertex order)."""
    global _VERTS
    if _VERTS is None:
        env = pusht_env.PushTEnv(with_target=False)
        env.reset()
        polys = [[(v.x, v.y) for v in s.get_vertices()] for s in env.block.shapes]
        env.close()
        polys.sort(key=lambda vs: (round(min(y for _, y in vs), 3), round(min(x for x, _ in vs), 3)))
        _VERTS = np.array([v for vs in polys for v in vs], np.float32)
    return _VERTS


def tee_keypoints(states):
    """Block pose (x, y, angle) -> 8 T vertices in env pixels, for pose errors that mix position and
    angle. states (..., 5)."""
    verts = _tee_vertices()
    th = states[..., 4:5]
    c, s = np.cos(th), np.sin(th)
    x = states[..., 2:3] + c * verts[:, 0] - s * verts[:, 1]
    y = states[..., 3:4] + s * verts[:, 0] + c * verts[:, 1]
    return np.stack([x, y], -1)  # (..., 8, 2)


if __name__ == "__main__":
    # how much do the hidden physics change outcomes of the same action sequence?
    states, rel, lens = load_split("train")
    rng = np.random.default_rng(0)
    for ep in [0, 1, 2, 3, 4]:
        L = lens[ep]
        n_wm = min(16, L // FRAMESKIP)
        finals = []
        for k in range(9):
            phys = NOMINAL if k == 0 else sample_physics(rng)
            _, st, _ = rollout_episode(states[ep, 0], rel[ep, :L], phys, n_wm=n_wm, render=False)
            finals.append(st)
        finals = np.stack(finals)  # (9, n_wm+1, 5)
        kp = tee_keypoints(finals)
        spread = np.linalg.norm(kp[1:] - kp[:1], axis=-1).mean((-1))  # (8, T) vs nominal
        moved = np.linalg.norm(finals[0, -1, 2:4] - finals[0, 0, 2:4])
        recorded = np.linalg.norm(states[ep, n_wm * FRAMESKIP, 2:4] - finals[0, -1, 2:4])
        print(f"ep {ep}: block moved {moved:.1f}px (nominal), nominal vs recorded {recorded:.1f}px, "
              f"random-physics keypoint deviation at t={n_wm}: mean {spread[:, -1].mean():.1f} "
              f"min {spread[:, -1].min():.1f} max {spread[:, -1].max():.1f}px")

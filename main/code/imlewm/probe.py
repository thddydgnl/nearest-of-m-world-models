"""Ball-position probe on normalized DINO tokens (soft-argmax over the 16x16 grid)."""
import argparse

import numpy as np
import torch
import torch.nn as nn

from .common import ds_dir
from .wm import Stats


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(384, 128, 1), nn.GELU(), nn.Conv2d(128, 64, 3, padding=1),
                                 nn.GELU(), nn.Conv2d(64, 3, 1))
        c = (torch.arange(16).float() + 0.5) * 4.0  # patch centers in 64px coords
        self.register_buffer("cx", c[None, :].expand(16, 16).reshape(-1))
        self.register_buffer("cy", c[:, None].expand(16, 16).reshape(-1))

    def forward(self, tok):  # (B,256,384) normalized
        B = tok.shape[0]
        o = self.net(tok.transpose(1, 2).reshape(B, 384, 16, 16).float()).reshape(B, 3, 256)
        w = o[:, 0].softmax(-1)
        x = (w * (self.cx + 4 * torch.tanh(o[:, 1]))).sum(-1)
        y = (w * (self.cy + 4 * torch.tanh(o[:, 2]))).sum(-1)
        return torch.stack([x, y], -1)


def load_probe(name, device="cuda"):
    p = Probe().to(device)
    p.load_state_dict(torch.load(f"{ds_dir(name)}/probe.pt"))
    return p.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    d = ds_dir(a.data)
    st = Stats(a.data)
    tok = torch.from_numpy(np.load(f"{d}/train_tokens.npy"))
    pos = torch.from_numpy(np.load(f"{d}/train_meta.npz")["pos"]).float()
    ttok = torch.from_numpy(np.load(f"{d}/test_tokens.npy", mmap_mode="r")[:2000].copy())
    tpos = torch.from_numpy(np.load(f"{d}/test_meta.npz")["pos"][:2000]).float().cuda()
    p = Probe().cuda()
    opt = torch.optim.AdamW(p.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    for s in range(a.steps):
        i = torch.randint(len(tok), (256,))
        pr = p(st.norm(tok[i].cuda()))
        loss = (pr - pos[i].cuda()).pow(2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        err = torch.cat([(p(st.norm(ttok[i:i + 500].cuda())) - tpos[i:i + 500]).norm(dim=-1) for i in range(0, 2000, 500)])
    print(f"{a.data} probe test err px: mean {err.mean():.3f} p95 {err.quantile(0.95):.3f} max {err.max():.3f}")
    torch.save(p.state_dict(), f"{d}/probe.pt")

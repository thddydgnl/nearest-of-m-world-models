"""BAIR under-dispersion diagnostic: spread/skill on train vs test contexts (h=1,4,16)."""
import sys, json
import numpy as np, torch
sys.path.insert(0, "/home/yonghwi/imle-wm/code")
from imlewm.common import ds_dir
from imlewm.wm import K_CTX, Stats, load_model, rollout

@torch.no_grad()
def diag(run, split, nfe=2, n=16, n_seq=96, H=16):
    torch.manual_seed(0)
    model, _ = load_model(f"/home/yonghwi/imle-wm/runs/{run}/ckpt.pt")
    d = ds_dir("bair"); st = Stats("bair")
    tok = np.load(f"{d}/{split}_tokens.npy", mmap_mode="r"); meta = np.load(f"{d}/{split}_meta.npz")
    seg, act = meta["seg"], meta["act"]
    first = np.r_[0, np.where(np.diff(seg) != 0)[0] + 1]
    sel = np.sort(np.random.default_rng(0).choice(first, n_seq, replace=False)) + K_CTX - 1
    mean, spread = [], []
    for i0 in range(0, n_seq, 4):
        s = sel[i0:i0 + 4]; S = len(s)
        ctx = st.norm(torch.from_numpy(np.asarray(tok[(s[:, None] + np.arange(-K_CTX + 1, 1)).reshape(-1)])).cuda()).view(S, K_CTX, 256, 384)
        acts = torch.from_numpy(act[s[:, None] + np.arange(H)[None]]).cuda()
        gt = st.norm(torch.from_numpy(np.asarray(tok[(s[:, None] + np.arange(1, H + 1)).reshape(-1)])).cuda()).view(S, 1, H, 256, 384)
        R = rollout(model, ctx, acts, n, nfe, st.sigma_r)
        mean.append((R - gt).pow(2).mean((-1, -2)).sqrt().mean(1).cpu())
        Rf = R.permute(0, 2, 1, 3, 4).reshape(S * H, n, -1)
        pw = torch.cdist(Rf, Rf) / Rf.shape[-1] ** 0.5
        spread.append((pw.sum((1, 2)) / (n * (n - 1))).view(S, H).cpu())
    m, sp = torch.cat(mean).mean(0), torch.cat(spread).mean(0)
    return {h: (round(m[h-1].item(), 3), round(sp[h-1].item(), 3), round((sp[h-1] / m[h-1]).item(), 2)) for h in (1, 4, 16)}

for run in sys.argv[1:]:
    for split in ("train", "test"):
        print(run, split, "h: (GT-sample dist, sample-sample spread, ratio)", diag(run, split), flush=True)

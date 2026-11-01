"""B one-step diagnostics on held-out test transitions: block / agent position error of the
predicted next frame (probe readout), split by whether the block actually moves."""
import sys, numpy as np, torch
from imlewm.common import ds_dir
from imlewm.b_eval import load_probeB
from imlewm.wm import Stats, load_model, predict
torch.set_grad_enabled(False); torch.manual_seed(0)
d = ds_dir("pusht_b"); st = Stats("pusht_b"); probe = load_probeB("pusht_b")
tok = np.load(f"{d}/test_tokens.npy", mmap_mode="r"); m = np.load(f"{d}/test_meta.npz")
seg, pos, act = m["seg"], m["pos"], m["act"]
t = np.array([i for i in range(2, len(seg) - 1) if seg[i - 2] == seg[i + 1]])
mv = np.abs(pos[t + 1, 2:] - pos[t, 2:]).max(1) > 0.5
rng = np.random.default_rng(0)
sel = np.concatenate([rng.choice(t[mv], 400, replace=False), rng.choice(t[~mv], 400, replace=False)])
lab = np.r_[np.ones(400), np.zeros(400)].astype(bool)
ctx = st.norm(torch.from_numpy(np.asarray(tok[(sel[:, None] + np.arange(-2, 1)).reshape(-1)])).cuda()).view(len(sel), 3, 256, 384)
gt = torch.from_numpy(pos[sel + 1]).float()
cur = torch.from_numpy(pos[sel]).float()
print(f"true block displacement on contact steps: {np.linalg.norm(pos[sel[lab]+1,2:4]-pos[sel[lab],2:4],axis=1).mean():.1f}px")
for spec in sys.argv[1:]:
    run, nfe = (spec.split("@") + ["1"])[:2]
    model, _ = load_model(f"/home/yonghwi/imle-wm/runs/{run}/ckpt.pt")
    a = torch.from_numpy(act[sel]).cuda()
    S = torch.cat([predict(model, ctx[i:i+100], a[i:i+100], 8, int(nfe), st.sigma_r) for i in range(0, len(sel), 100)])
    p = probe(S.flatten(0, 1)).view(len(sel), 8, 5).cpu()
    be = (p[:, :, 2:4] - gt[:, None, 2:4]).norm(dim=-1)  # block error per sample
    ae = (p[:, :, 0:2] - gt[:, None, 0:2]).norm(dim=-1)
    pdisp = (p[:, :, 2:4] - cur[:, None, 2:4]).norm(dim=-1).mean(1)
    print(f"{spec:22s} contact: block err best/mean {be[lab].min(1).values.mean():.1f}/{be[lab].mean():.1f}px, pred block disp {pdisp[lab].mean():.1f}px, agent err {ae[lab].mean():.1f} | no-contact: block err {be[~lab].mean():.1f}, agent err {ae[~lab].mean():.1f}", flush=True)

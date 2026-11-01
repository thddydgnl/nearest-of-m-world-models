"""B diagnostics: sample spread vs GT spread on mf_test, per horizon; one-step error on test."""
import json, sys
import numpy as np, torch
from imlewm.common import ds_dir
from imlewm.b_eval import load_probeB
from imlewm.pusht_sim import tee_keypoints
from imlewm.wm import Stats, load_model, rollout
torch.manual_seed(0); torch.set_grad_enabled(False)
d = ds_dir("pusht_b"); st = Stats("pusht_b"); probe = load_probeB("pusht_b")
f = np.load(f"{d}/mf_test_v2.npz")
idx = np.arange(0, 128, 4)  # 32 starts
gt = tee_keypoints(f["fut_states"][idx])  # (S,64,16,8,2)
gt_c = gt.mean(-2)  # block keypoint centroid (S,64,16,2)
gt_spread = np.linalg.norm(gt_c - gt_c.mean(1, keepdims=True), axis=-1).mean(1)  # (S,16)
print("GT spread (px) h=1,4,8,16:", gt_spread.mean(0)[[0, 3, 7, 15]].round(1))
for run, nfe in [(r.split("@")[0], int(r.split("@")[1]) if "@" in r else 1) for r in sys.argv[1:]]:
    model, ck = load_model(f"/home/yonghwi/imle-wm/runs/{run}/ckpt.pt")
    sp, err = [], []
    for k, i in enumerate(idx):
        ctx = st.norm(torch.from_numpy(f["ctx"][i]).cuda())[None]
        acts = torch.from_numpy(f["act"][i]).cuda()[None]
        R = rollout(model, ctx, acts, 16, nfe, st.sigma_r)[0]
        ps = probe(R.flatten(0, 1)).view(16, 16, 5).cpu().numpy()
        kp = tee_keypoints(ps).mean(-2)  # (16,16,2)
        sp.append(np.linalg.norm(kp - kp.mean(0, keepdims=True), axis=-1).mean(0))
        err.append(np.linalg.norm(kp.mean(0) - gt_c[k].mean(0), axis=-1))  # sample mean vs GT mean
    sp, err = np.array(sp), np.array(err)
    print(f"{run}@{nfe}: sample spread h=1,4,8,16 {sp.mean(0)[[0,3,7,15]].round(1)} | mean-vs-mean err {err.mean(0)[[0,3,7,15]].round(1)}", flush=True)

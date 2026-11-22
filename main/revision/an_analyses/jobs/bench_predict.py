"""Throughput of wm.predict at the planner's batch (512) under different cuDNN settings (diagnostic only)."""
import sys
import time

import torch

from imlewm.wm import load_model, predict

run = sys.argv[1]
nfe = int(sys.argv[2]) if len(sys.argv) > 2 else 2
model, _ = load_model(f"/root/imle-wm/runs/{run}/ckpt.pt")
c = torch.randn(512, 3, 256, 384, device="cuda")
a = torch.randn(512, 2, device="cuda")


def bench(reps=20):
    for _ in range(3):
        predict(model, c, a, 1, nfe, 1.0)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(reps):
        predict(model, c, a, 1, nfe, 1.0)
    torch.cuda.synchronize()
    return (time.time() - t) / reps * 1000


with torch.no_grad():
    for trial in range(3):
        torch.backends.cudnn.benchmark = False
        t0 = bench()
        torch.backends.cudnn.benchmark = True
        t1 = bench()
        print(f"trial {trial}: default {t0:.1f} ms/call, cudnn.benchmark {t1:.1f} ms/call", flush=True)

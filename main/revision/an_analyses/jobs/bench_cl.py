"""Throughput of wm.predict at the planner's batch (512): default memory format vs channels_last (diagnostic only)."""
import sys
import time

import torch

from imlewm.wm import load_model, predict

run = sys.argv[1]
nfe = int(sys.argv[2]) if len(sys.argv) > 2 else 2
model, _ = load_model(f"/root/imle-wm/runs/{run}/ckpt.pt")
c = torch.randn(512, 3, 256, 384, device="cuda")
a = torch.randn(512, 2, device="cuda")


def bench(reps=10):
    for _ in range(2):
        predict(model, c, a, 1, nfe, 1.0)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(reps):
        predict(model, c, a, 1, nfe, 1.0)
    torch.cuda.synchronize()
    return (time.time() - t) / reps * 1000


with torch.no_grad():
    torch.manual_seed(0); ref = predict(model, c[:64], a[:64], 1, nfe, 1.0)
    t0 = bench()
    model = model.to(memory_format=torch.channels_last)
    torch.manual_seed(0); out = predict(model, c[:64], a[:64], 1, nfe, 1.0)
    t1 = bench()
    model = model.to(memory_format=torch.contiguous_format)
    t2 = bench()
    print(f"default {t0:.1f} ms/call, channels_last {t1:.1f}, default again {t2:.1f}; max abs diff {float((out - ref).abs().max()):.4f}", flush=True)

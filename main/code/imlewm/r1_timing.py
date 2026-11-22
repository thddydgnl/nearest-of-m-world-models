"""Revision r1, analysis g (DESIGN_r1.md section 4g): latency and training-step time on one idle GPU.

export (where the original checkpoints are): writes r1_timing_configs.json next to this module with the model
        and training arguments of every configuration in Tables 2 (A2-div, 8M) and 4 (Push-T, 21M).
run    (on the idle A5000): for every configuration
        latency: batch 1, one predicted step at the table's NFE (wm.predict, as evaluate.latency): 5 warm-up and
                 50 timed passes, repeated 5 times (mean, sd, min and the five values, ms per step);
        training step: python -m imlewm.train with the run's own arguments on a small synthetic token dataset of
                 the same shapes (data/r1_tim_a<act_dim>, random tokens; timing does not depend on values),
                 400 steps, step time = (t[400] - t[200]) / 200 from log.jsonl; run names r1_tim_<label>.

usage: python -m imlewm.r1_timing export
       python -m imlewm.r1_timing run --out /workspace/nom/out/g_timing   (on an otherwise idle GPU)
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = f"{HERE}/r1_timing_configs.json"
RUNS = os.path.expanduser("~/imle-wm/runs")
DATA_ROOT = os.environ.get("IMLE_DATA", os.path.expanduser("~/imle-wm/data"))

TABLE2 = [("A2 Regression", "e1_a2_reg_s0", 1), ("A2 FM-50", "e1_a2_fm_s0", 50), ("A2 FM-2", "e1_a2_fm_s0", 2),
          ("A2 iMF", "e1_a2_imf_s0", 1), ("A2 Shortcut", "e7_a2_sc_s0", 1), ("A2 ES m=8", "e15_a2_es8_s0", 1),
          ("A2 ES m=16", "e15_a2_es16_s0", 1), ("A2 IMLE-WM m=4", "e1_a2_imle4_s0", 1),
          ("A2 IMLE-WM m=16", "e1_a2_imle16_s0", 1)]
TABLE4 = [("B Regression", "e1_b_reg_s0", 1), ("B FM-50", "e1_b_fm_s0", 50), ("B FM-2", "e1_b_fm_s0", 2),
          ("B iMF", "e1_b_imf_s0", 1), ("B Shortcut", "e7_b_sc_s0", 1), ("B IMLE-WM m=4", "e1_b_imle4_s0", 1),
          ("B IMLE-WM m=16", "e1_b_imle16_s0", 1)]
TRAIN_FLAGS = {"method", "bs", "lr", "C", "nblocks", "m", "stages", "loss", "H", "act_path", "contrast", "seed",
               "ctx_noise", "seq_feed", "mf_p", "mf_clip", "mf_chunks", "mf_warmup", "mf_ratio", "wide_head",
               "t_shift", "z_from", "t_scale", "beta2", "zdim", "rs_eps", "match_pca", "match_pool", "match_chunk",
               "sc_frac", "es_beta", "es_chunks", "es_dist"}
BOOL_FLAGS = {"curriculum", "grad_ckpt", "seqmatch", "mf_bf16"}


def export():
    out = []
    for label, run, nfe in TABLE2 + TABLE4:
        ck = torch.load(f"{RUNS}/{run}/ckpt.pt", map_location="cpu")
        out.append({"label": label, "run": run, "nfe": nfe, "model_args": ck["model_args"],
                    "train_args": {k: v for k, v in ck["args"].items() if k in TRAIN_FLAGS | BOOL_FLAGS},
                    "orig_train_time_s": ck["train_time"], "orig_steps": ck["args"]["steps"]})
    with open(CFG, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", CFG, len(out), "configs")


@torch.no_grad()
def latency(model, nfe, act_dim, warm=5, reps=50, repeats=5):
    from .wm import predict
    ctx = torch.randn(1, 3, 256, 384, device="cuda"); a = torch.zeros(1, act_dim, device="cuda")
    vals = []
    for _ in range(repeats):
        for _ in range(warm):
            predict(model, ctx, a, 1, nfe, 1.0)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(reps):
            predict(model, ctx, a, 1, nfe, 1.0)
        torch.cuda.synchronize()
        vals.append((time.time() - t) / reps * 1000)
    return {"mean_ms": float(np.mean(vals)), "sd_ms": float(np.std(vals, ddof=1)), "min_ms": float(np.min(vals)),
            "values_ms": [round(v, 3) for v in vals]}


def synth_data(act_dim, n=4096, ep=32):
    name = f"r1_tim_a{act_dim}"
    d = os.path.join(DATA_ROOT, name)
    if os.path.exists(f"{d}/stats.pt"):
        return name
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    mm = np.lib.format.open_memmap(f"{d}/train_tokens.npy", mode="w+", dtype=np.float16, shape=(n, 256, 384))
    for i in range(0, n, 512):
        mm[i:i + 512] = rng.standard_normal((min(512, n - i), 256, 384)).astype(np.float16)
    mm.flush()
    np.savez(f"{d}/train_meta.npz", pos=rng.uniform(4, 60, (n, 2)), seg=np.arange(n) // ep,
             act=rng.uniform(-1, 1, (n, act_dim)).astype(np.float32), hit=np.zeros(n, bool))
    torch.save({"mean": torch.zeros(384), "std": torch.ones(384), "pca": torch.eye(384), "evals": torch.ones(384),
                "sigma_r": 1.0, "resid_rms_median": 1.0}, f"{d}/stats.pt")
    return name


def train_step_time(cfg, steps=400):
    act_dim = cfg["model_args"].get("act_dim", 2)
    data = synth_data(act_dim)
    name = "r1_tim_" + cfg["label"].replace(" ", "_").replace("=", "").replace("-", "")
    if os.path.exists(f"{RUNS}/{name}/ckpt.pt"):
        rows = [json.loads(l) for l in open(f"{RUNS}/{name}/log.jsonl")]
    else:
        argv = [sys.executable, "-m", "imlewm.train", "--data", data, "--name", name, "--steps", str(steps),
                "--memo_every", "0", "--save_every", "0"]
        for k, v in cfg["train_args"].items():
            if k in BOOL_FLAGS:
                if v:
                    argv.append(f"--{k}")
            elif v is not None:
                argv += [f"--{k}", str(v)]
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(f"{RUNS}/{name}/log.jsonl"):
            return {"error": r.stderr[-1500:], "argv": argv}
        rows = [json.loads(l) for l in open(f"{RUNS}/{name}/log.jsonl")]
    t = {r["step"]: r["t"] for r in rows if "t" in r}
    return {"name": name, "steps": steps, "t200": t.get(200), "t400": t.get(400),
            "step_ms": (t[400] - t[200]) / 200 * 1000 if 200 in t and 400 in t else None,
            "train_h_at_orig_steps": (t[400] - t[200]) / 200 * cfg["orig_steps"] / 3600 if 200 in t and 400 in t else None}


def run(out, only=None):
    from .model import Gen
    os.makedirs(out, exist_ok=True)
    cfgs = [c for c in json.load(open(CFG)) if only is None or only in c["label"]]
    path = f"{out}/timing.json"
    res = json.load(open(path)) if os.path.exists(path) else {"gpu": torch.cuda.get_device_name(0), "configs": {}}
    for cfg in cfgs:
        lab = cfg["label"]
        if lab in res["configs"] and "train" in res["configs"][lab]:
            continue
        torch.manual_seed(0)
        model = Gen(**cfg["model_args"]).cuda().eval()
        r = {"run": cfg["run"], "nfe": cfg["nfe"], "method": cfg["model_args"]["method"],
             "latency": latency(model, cfg["nfe"], cfg["model_args"].get("act_dim", 2)),
             "orig_train_time_s": cfg["orig_train_time_s"], "orig_steps": cfg["orig_steps"]}
        del model; torch.cuda.empty_cache()
        same = [c for c in res["configs"].values() if c["run"] == cfg["run"] and "train" in c]
        r["train"] = same[0]["train"] if same else train_step_time(cfg)
        res["configs"][lab] = r
        with open(path + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(path + ".tmp", path)
        print(lab, round(r["latency"]["mean_ms"], 2), "ms;", r["train"].get("step_ms"), "ms/step", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "run"])
    ap.add_argument("--out", default="/workspace/nom/out/g_timing")
    ap.add_argument("--only", default=None, help="substring of the configuration labels to run (for a functional test)")
    a = ap.parse_args()
    export() if a.cmd == "export" else run(a.out, a.only)

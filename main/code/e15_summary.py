"""Summarize E15 (energy-score baseline) and its follow-ups next to the E1 / E14 reference runs. Run on the server:
ssh kiis-mvf-gpu 'source ~/imle-wm/env.sh; python -' < e15_summary.py"""
import json
import os

import numpy as np

R = os.path.expanduser("~/imle-wm/runs")


def load(run, f):
    p = f"{R}/{run}/{f}"
    return json.load(open(p)) if os.path.exists(p) else None


def v(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def memo_last(run):
    p = f"{R}/{run}/log.jsonl"
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p) if "memo_train" in l]
    return rows[-1] if rows else None


def a2_row(run, ev="eval.json", e3="e3.json"):
    e, r3, mm = load(run, ev), load(run, e3), memo_last(run)
    if e is None:
        return None
    s = e["stoch"]
    out = {"cov": v(s["cov_pos"]), "prec": v(s["prec_pos"]), "recall_knn": v(s["recall_knn"]),
           "det_pos": v(e["det"]["pos_err"]), "lat_ms": e["latency_ms"], "train_h": e["train_time"] / 3600,
           "memo_train": mm["memo_train"] if mm else None, "memo_step": mm["step"] if mm else None,
           "div_ratio": v(s["div_ratio"]), "best_of_n": v(s["best_of_n"]), "mean_err": v(s["mean_pred_err"])}
    if r3:
        out.update(man32=r3["manifold"][-1], best32=r3["best_err"][-1], spread32=r3["spread"][-1],
                   man1=r3["manifold"][0], spread1=r3["spread"][0])
        if "real_nn" in r3:  # bounce_k4 repeats test frames exactly (real_nn ~0.2): rescale to the A2-div
            # normalizer (real_nn of bounce_div_k4 = 99.31, same e3 bank) so the values compare across data
            out.update(man1_div=r3["manifold"][0] * r3["real_nn"] / 99.31, man32_div=r3["manifold"][-1] * r3["real_nn"] / 99.31)
    return out


def agg(rows):
    rows = [r for r in rows if r]
    if not rows:
        return "-"
    keys = rows[0].keys()
    return {k: (round(float(np.mean([r[k] for r in rows if r.get(k) is not None])), 3),
                round(float(np.std([r[k] for r in rows if r.get(k) is not None])), 3))
            for k in keys if all(r.get(k) is not None for r in rows)} | {"n": len(rows)}


print("== A2-div K=4 (12k steps)")
groups = {
    "ES m=8": [a2_row(f"e15_a2_es8_s{s}") for s in range(3)],
    "ES m=16": [a2_row(f"e15_a2_es16_s{s}") for s in range(3)],
    "IMLE m=16 (E1)": [a2_row(f"e1_a2_imle16_s{s}") for s in range(3)],
    "IMLE m=4 (E1)": [a2_row(f"e1_a2_imle4_s{s}") for s in range(3)],
    "FM-50 (E1)": [a2_row(f"e1_a2_fm_s{s}", "eval_nfe50.json", "e3_nfe50.json") for s in range(3)],
    "FM-2 (E1)": [a2_row(f"e1_a2_fm_s{s}", "eval_nfe2.json", "e3_nfe2.json") for s in range(3)],
    "reg (E1)": [a2_row(f"e1_a2_reg_s{s}") for s in range(3)],
}
for k, rows in groups.items():
    print(k, agg(rows))
for s in range(3):
    print("  ES m=8 seed", s, a2_row(f"e15_a2_es8_s{s}"))
for s in range(2):  # seed 1 added in E16
    print("  ES m=16 seed", s, a2_row(f"e15_a2_es16_s{s}"))

print("== E14 asymmetric modes (p0 = 0.787 on eval contexts), mode-0 share / coverage")
ref = {"ES m=8": [f"e15_asym_es8_s{s}" for s in range(2)],
       "IMLE m=16": [f"e14_asym_m16_s{s}" for s in range(2)],
       "IMLE m=4": [f"e14_asym_m4_s{s}" for s in range(2)],
       "IMLE m=2": [f"e14_asym_m2_s{s}" for s in range(2)],
       "FM-50": [f"e14_asym_fm_s{s}" for s in range(2)]}
for k, runs in ref.items():
    rows = []
    for run in runs:
        f = "modeshare_nfe50.json" if "fm" in run else "modeshare.json"
        d = load(run, f)
        if d:
            rows.append({"share0_test": v(d["test"]["share0"]), "freq0_test": d["test"]["data_freq0"],
                         "cov_test": v(d["test"]["cov"]), "prec_test": v(d["test"]["prec"]),
                         "real_train": v(d["train"]["realized_share"]), "real_test": v(d["test"]["realized_share"])})
    print(k, agg(rows), [round(r["share0_test"], 3) for r in rows])

print("== A1 (contexts repeat), autoregressive 16-step rollouts, F1 error px (probe / heatmap / heatmap noise 0.5)")


def a1_row(run, tag=""):
    e, h, h5, r3 = (load(run, f"{p}{tag}.json") for p in ["eval", "eval_hm", "eval_hmn05", "e3"])
    if e is None:
        return None
    out = {"f1": e["model"]["f1_err"], "prec_err": e["model"]["precision_err"], "rec_err": e["model"]["recall_err"],
           "train_h": e["train_time"] / 3600}
    if h:
        out["f1_hm"] = h["model"]["f1_err"]
    if h5:
        out["f1_hmn05"] = h5["model"]["f1_err"]
    if r3:
        out["man16"] = r3["manifold"][-1]
    return out


a1 = {"ES m=8": [a1_row(f"e15_a1_es8_s{s}") for s in range(2)],
      "IMLE m=16 (E1)": [a1_row(f"e1_a1_imle16_s{s}") for s in range(3)],
      "IMLE m=4 (E1)": [a1_row(f"e1_a1_imle4_s{s}") for s in range(3)],
      "FM-50 (E1)": [a1_row(f"e1_a1_fm_s{s}", "_nfe50") for s in range(3)],
      "FM-2 (E1)": [a1_row(f"e1_a1_fm_s{s}", "_nfe2") for s in range(3)],
      "reg (E1)": [a1_row(f"e1_a1_reg_s{s}") for s in range(3)]}
for k, rows in a1.items():
    print(k, agg(rows))

print("== Follow-ups: A2-fix (bounce_k4, fixed appearance, same recipe) and per-token ES on A2-div K=4")
fu = {"A2-fix ES m=8": [a2_row(f"e15_fix_es8_s{s}") for s in range(2)],  # seed 1 added in E16
      "A2-fix IMLE m=16": [a2_row(f"e15_fix_imle16_s{s}") for s in range(2)],
      "A2-fix IMLE m=4": [a2_row(f"e15_fix_imle4_s{s}") for s in range(2)],
      "A2-div per-token ES m=8": [a2_row(f"e15_a2_estok8_s{s}") for s in range(2)]}
for k, rows in fu.items():
    print(k, agg(rows))
for run in ["e15_fix_es8", "e15_fix_imle16", "e15_fix_imle4", "e15_a2_estok8"]:
    for s in range(2):
        print(" ", run, "seed", s, a2_row(f"{run}_s{s}"))
print("curves (step, val coverage, train memo, val memo)")
for run in ["e15_fix_es8_s0", "e15_fix_imle16_s0", "e15_fix_imle4_s0", "e15_a2_estok8_s0", "e15_a2_estok8_s1",
            "e15_fix_es8_s1", "e15_fix_imle16_s1", "e15_fix_imle4_s1", "e15_a2_es16_s1"]:
    p = f"{R}/{run}/log.jsonl"
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p) if "memo_train" in l]
        print(" ", run, [(r["step"], round(r["cov_val"], 2), round(r["memo_train"], 2), round(r["memo_val"], 2)) for r in rows])

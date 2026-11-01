"""Summarize E16: the N1b rollout-state ablation on A1 (matched = E11 N1b, random, gt; N0 = E11 control) and
the BAIR effective-dimension check of Prop 2. Run on the server:
ssh kiis-mvf-gpu 'source ~/imle-wm/env.sh; python -' < e16_summary.py"""
import json
import os

import numpy as np

R = os.path.expanduser("~/imle-wm/runs")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def a1_row(run):
    e, r3 = load(f"{R}/{run}/eval.json"), load(f"{R}/{run}/e3.json")
    if e is None or r3 is None:
        return None
    m = e["model"]
    h = lambda k, i: r3[k][i - 1]
    return {"f1": m["f1_err"], "prec": m["precision_err"], "rec": m["recall_err"],
            "man1": h("manifold", 1), "man8": h("manifold", 8), "man16": h("manifold", 16),
            "best1": h("best_err", 1), "best8": h("best_err", 8), "best16": h("best_err", 16),
            "spread16": h("spread", 16), "train_h": e["train_time"] / 3600}


groups = {"N0 (E11)": "e11_a1_n0_s{}", "matched (E11 N1b)": "e11_a1_seq4_s{}",
          "random": "e16_a1_seqrand_s{}", "gt": "e16_a1_seqgt_s{}"}
print("== A1 N1b rollout-state ablation (7M, m=16, H=4, 8k step): mean and std over seeds, then per seed")
for k, pat in groups.items():
    rows = [r for r in (a1_row(pat.format(s)) for s in range(2)) if r]
    if not rows:
        print(k, "-")
        continue
    print(k, {c: (round(float(np.mean([r[c] for r in rows])), 3), round(float(np.std([r[c] for r in rows])), 3))
              for c in rows[0]}, "n", len(rows))
    for r in rows:
        print("   ", {c: round(v, 3) for c, v in r.items()})
print("training curves (step, match_d)")
for k, pat in groups.items():
    for s in range(2):
        p = f"{R}/{pat.format(s)}/log.jsonl"
        rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
        print(" ", pat.format(s), [(r["step"], round(r["match_d"], 3)) for r in rows if r["step"] % 2000 == 0])

print("== BAIR effective dimension (Prop 2), runs/e16_bair_deff.json")
d = load(f"{R}/e16_bair_deff.json")
if d:
    meas = {int(m): v["mean"] for m, v in d["measured_r"].items()}
    print("e_m", {m: round(v, 3) for m, v in d["e_m"].items()}, "measured r", {m: round(v, 3) for m, v in meas.items()})
    for key in ["test_eval", "test_all", "train_eval", "train_all"]:
        rows = [v for k, v in d.items() if k.endswith("/" + key)]
        pr = float(np.mean([v["pr"] for v in rows]))
        pred = {m: float(np.mean([v["pred_r"][str(m)] for v in rows])) for m in meas}
        print(key, "pr", [round(v["pr"], 1) for v in rows], "plug-in", [round(v["pr_plugin"], 1) for v in rows],
              "rms", [round(v["rms"], 4) for v in rows], "pred r", {m: round(p, 3) for m, p in pred.items()},
              "cv frac top-k", {k: round(f, 3) for k, f in rows[0]["cv_frac_top"].items()})
    # absolute spread and error predicted from the regression residual rms (test_eval) and d_PR
    v = [v for k, v in d.items() if k.endswith("/test_eval")]
    sig, pr = float(np.mean([x["rms"] for x in v])), float(np.mean([x["pr"] for x in v]))
    for m, runs in [(4, ["e8_bair_imle4_s0", "e8_bair_imle4_s1"]), (16, ["e8_bair_imle16_s0", "e8_bair_imle16_s1"]),
                    (64, ["e10_bair_m64_s0"])]:
        ev = [load(f"{R}/{r}/eval.json") for r in runs]
        rho = d["e_m"][str(m)] / np.sqrt(pr)
        print(f"m={m}: predicted spread / error {np.sqrt(2) * rho * sig:.4f} / {sig * np.sqrt(1 + rho ** 2):.4f}, measured",
              f"{np.mean([e['spread'][0] for e in ev]):.4f} / {np.mean([e['mean_rms'][0] for e in ev]):.4f}")

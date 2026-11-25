"""Print the revision-r1 experiment results (experiments 1-3 of DESIGN_r1.md) from the pulled raw files.

usage: python3 main/revision/summarize_r1.py [x1 x2a x2b x3 ...]     (reads only local JSON; seconds on a laptop)

Raw files: main/revision/<group>/raw/<pod>/runs/r1_*/ (log.jsonl, eval*.json, e3*.json) and .../out/<analysis>/,
pulled by runpod/pull_all.sh. Original runs (for reference values): main/results/runs/.
Spread: sample standard deviation (n - 1), as DESIGN_r1.md section 0 fixes; individual seeds are printed too.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "..", "results", "runs")
RAW = {"x1": os.path.join(HERE, "x1_arc", "raw", "x1.1"),
       "x2.1": os.path.join(HERE, "x2_long", "raw", "x2.1"),
       "x2.2": os.path.join(HERE, "x2_long", "raw", "x2.2"),
       "x2.3": os.path.join(HERE, "x2_long", "raw", "x2.3"),   # seed 1 of experiment 2b after x2.2 was deleted
       "x3": os.path.join(HERE, "x3_ctxnoise", "raw", "x3.1")}


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


def log_rows(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def memo_rows(path):
    return [r for r in log_rows(path) if "memo_train" in r]


def ms(vals, nd=3):
    """mean, sample SD and the seeds as a string."""
    v = [x for x in vals if x is not None]
    if not v:
        return "n/a"
    sd = np.std(v, ddof=1) if len(v) > 1 else float("nan")
    return f"{np.mean(v):.{nd}f} +- {sd:.{nd}f} ({'/'.join(f'{x:.{nd}f}' for x in v)})"


def mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


# ------------------------------------------------------------------------------------------- experiment 1
I_SEMI = {2: 0.5947, 4: 0.2351, 8: 0.0614}
I_ARC = {2: 0.1737, 4: 0.0638, 8: 0.0161}
H_I = {2: (0.400, 0.352, 0.448), 4: (0.312, 0.264, 0.360), 8: (0.274, 0.226, 0.322)}


def x1():
    root = RAW["x1"]
    print("## Experiment 1: arc90 vs standard geometry, K = 4, 8k steps (last memo_train of log.jsonl)")
    idx = {}
    for geo, pre in (("arc", "r1_arc90"), ("std", "r1_k4")):
        for m in (2, 4, 8):
            vals, steps, covs, precs = [], [], [], []
            for s in (0, 1):
                rows = memo_rows(f"{root}/runs/{pre}_m{m}_s{s}/log.jsonl")
                vals.append(rows[-1]["memo_train"] if rows else None)
                steps.append(rows[-1]["step"] if rows else None)
                ev = J(f"{root}/runs/{pre}_m{m}_s{s}/eval.json")
                covs.append(ev["stoch"]["cov_pos"] if ev else None)
                precs.append(ev["stoch"]["prec_pos"] if ev else None)
            # a value counts only once the run has reached its last monitor step (8k)
            idx[(geo, m)] = [v if st == 8000 else None for v, st in zip(vals, steps)]
            print(f"  {geo} m={m}: index {ms(vals)} at steps {steps} | held-out cov {ms(covs)} prec {ms(precs)}")
    print("  H_I predictions (index): " + ", ".join(f"m={m} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for m, v in H_I.items()))
    for m in (2, 4, 8):
        a, c = mean(idx[("arc", m)]), mean(idx[("std", m)])
        if a is None or c is None or None in idx[("arc", m)] + idx[("std", m)]:
            print(f"  m={m}: incomplete")
            continue
        d = c - a
        if m == 8:
            verdict = "reported only"
        elif H_I[m][1] <= a <= H_I[m][2] and d >= 0.10:
            verdict = "H_I supported"
        elif abs(d) < 0.05:
            verdict = "H_m supported"
        else:
            verdict = "inconclusive"
        print(f"  m={m}: arc {a:.3f} (in H_I interval: {H_I[m][1] <= a <= H_I[m][2]}), control {c:.3f}, "
              f"control - arc {d:+.3f} -> {verdict}")
    c1 = {g: J(f"{root}/out/c1/c1_{d}_200.json") for g, d in (("arc", "bounce_div_k4_arc90"), ("std", "bounce_div_k4"))}
    if all(c1.values()):
        ca, cs = c1["arc"]["ceil"][0], c1["std"]["ceil"][0]
        print(f"  C1: ceiling arc {c1['arc']['ceil']} floor {c1['arc']['floor']} | std {c1['std']['ceil']} floor {c1['std']['floor']}"
              f" -> attenuation correction {'REQUIRED' if ca < cs - 0.05 else 'not required'}")
        if ca < cs - 0.05:
            for m in (2, 4, 8):
                for geo in ("arc", "std"):
                    f, cl = c1[geo]["floor"][0], c1[geo]["ceil"][0]
                    corr = [(v - f) / (cl - f) for v in idx[(geo, m)] if v is not None]
                    print(f"    corrected {geo} m={m}: {ms(corr)}")
    else:
        print("  C1: not available yet")
    for geo, pre in (("arc", "r1_arc90"), ("std", "r1_k4")):
        for m in (2, 4, 8):
            fr, it = [], []
            for s in (0, 1):
                c2 = J(f"{root}/out/c2/{pre}_m{m}_s{s}_ckpt_nfeNone_200x16.json")
                fr.append(c2["derived"]["free_index_train"] if c2 else None)
                it.append(c2["derived"]["index_train"] if c2 else None)
            if any(v is not None for v in fr):
                print(f"  C2 {geo} m={m}: tolerance-free {ms(fr)} | re-measured index {ms(it)}")


# ------------------------------------------------------------------------------------------ experiment 2a
def x2a():
    root = RAW["x2.1"]
    print("## Experiment 2a: FM (FlowWM recipe) 36k steps, bounce_div_k4")
    orig = [J(f"{ORIG}/e1_a2_fm_s{s}/eval_nfe50.json") for s in (0, 1, 2)]
    print(f"  original 12k FM-50 coverage {ms([o['stoch']['cov_pos'] for o in orig if o])}, "
          f"IMLE-WM m=16 12k 0.929 (paper)")
    for tag, nfe in (("", 50), ("", 2), ("_ckpt_12000", 50), ("_ckpt_24000", 50)):
        cov, prec, det = [], [], []
        for s in (0, 1, 2):
            ev = J(f"{root}/runs/r1_fm36k_s{s}/eval{tag}_nfe{nfe}.json")
            cov.append(ev["stoch"]["cov_pos"] if ev else None)
            prec.append(ev["stoch"]["prec_pos"] if ev else None)
            det.append(ev["det"]["pos_err"] if ev else None)
        label = (tag.replace("_ckpt_", "ckpt ") or "final 36k")
        print(f"  {label} NFE {nfe}: cov {ms(cov)} | prec {ms(prec)} | det {ms(det, 2)}")
    for ck in ("ckpt", "ckpt_12000", "ckpt_24000"):
        for nfe in (50, 2):
            v, h, ex = [], [], []
            for s in (0, 1, 2):
                r = J(f"{root}/out/fm36k_memo/r1_fm36k_s{s}_{ck}_nfe{nfe}_200x16.json")
                v.append(r["derived"]["index_train"] if r else None)
                h.append(r["derived"]["index_heldout"] if r else None)
                ex.append(r["derived"]["excess_heldout"] if r else None)
            if any(x is not None for x in v):
                print(f"  memo {ck} NFE {nfe}: train {ms(v)} | held-out {ms(h)} | excess (held-out) {ms(ex)}")
    for s in (0, 1, 2):
        rows = memo_rows(f"{root}/runs/r1_fm36k_s{s}/log.jsonl")
        if rows:
            print(f"  monitor (20 Euler steps) s{s}: " + " ".join(f"{r['step'] // 1000}k:{r['memo_train']:.2f}"
                                                               for r in rows if r["step"] % 6000 == 0))
    for nfe in (50, 2):
        e = J(f"{root}/runs/r1_fm36k_s0/e3_nfe{nfe}.json")
        if e:
            print(f"  e3 s0 NFE {nfe}: manifold h=1 {e['manifold'][0]:.2f} h=16 {e['manifold'][15]:.2f} h=32 {e['manifold'][31]:.2f}"
                  f" | best-of-16 h=32 {e['best_err'][31]:.1f} | spread h=32 {e['spread'][31]:.1f}")
    covs = [J(f"{root}/runs/r1_fm36k_s{s}/eval_nfe50.json") for s in (0, 1, 2)]
    if all(covs):
        c = mean([x["stoch"]["cov_pos"] for x in covs])
        print(f"  decision: FM-50 mean coverage {c:.3f} -> "
              + ("claim limited to equal steps (>= 0.92)" if c >= 0.92 else "claim stands at equal wall-clock (< 0.92)")
              + f"; P2a range 0.80-0.90: {'held' if 0.80 <= c <= 0.90 else 'missed'}")


# ------------------------------------------------------------------------------------------ experiment 2b
def x2b():
    # seed 0 ran on x2.2; seed 1 was rerun on x2.3 (the x2.2 attempts are kept as *_partial_* runs)
    roots = {0: RAW["x2.2"], 1: RAW["x2.3"]}
    print("## Experiment 2b: IMLE-WM m = 16, 48k steps, bounce_div_k4 (monitor index on training contexts)")
    last, cov, prec, det = [], [], [], []
    for s in (0, 1):
        root = roots[s]
        rows = memo_rows(f"{root}/runs/r1_imle16_48k_s{s}/log.jsonl")
        if rows:
            print(f"  s{s}: " + " ".join(f"{r['step'] // 1000}k:{r['memo_train']:.3f}/{r.get('memo_val', float('nan')):.3f}"
                                       for r in rows if r["step"] % 6000 == 0))
            last.append(rows[-1]["memo_train"] if rows[-1]["step"] >= 48000 else None)
        ev = J(f"{root}/runs/r1_imle16_48k_s{s}/eval.json")
        cov.append(ev["stoch"]["cov_pos"] if ev else None)
        prec.append(ev["stoch"]["prec_pos"] if ev else None)
        det.append(ev["det"]["pos_err"] if ev else None)
    for ck in ("ckpt_12000", "ckpt_24000", "ckpt_36000"):
        c = [J(f"{roots[s]}/runs/r1_imle16_48k_s{s}/eval_{ck}.json") for s in (0, 1)]
        if any(c):
            print(f"  {ck}: cov {ms([x['stoch']['cov_pos'] if x else None for x in c])}")
    print(f"  48k: index {ms(last)} | held-out cov {ms(cov)} | prec {ms(prec)} | det {ms(det, 2)}")
    if None not in last and None not in cov and last:
        i, c = mean(last), mean(cov)
        bad = i > 0.40 or c < 0.85
        print(f"  decision: index {i:.3f}, coverage {c:.3f} -> "
              + ("memorization returns with longer training (rule triggered)" if bad else "P2b held (index <= 0.40, coverage >= 0.85)"))


# ------------------------------------------------------------------------------------------- experiment 3
def x3():
    root = RAW["x3"]
    print("## Experiment 3: FM with context noise, 12k steps, bounce_div_k4")
    ref = {}
    for nfe in (50, 2):
        fm = [J(f"{ORIG}/e1_a2_fm_s{s}/e3_nfe{nfe}.json") for s in (0, 1, 2)]
        ref[nfe] = [f["manifold"][31] for f in fm if f]
        print(f"  original FM NFE {nfe}: manifold h=32 {ms(ref[nfe], 2)}")
    reg = [J(f"{ORIG}/e1_a2_reg_s{s}/e3.json") for s in (0, 1, 2)]
    print(f"  original regression: manifold h=32 {ms([f['manifold'][31] for f in reg if f], 2)}")
    o = [J(f"{ORIG}/e1_a2_fm_s{s}/eval_nfe50.json") for s in (0, 1, 2)]
    print(f"  original FM-50 one-step coverage {ms([x['stoch']['cov_pos'] for x in o if x])}")
    for cn in ("01", "03"):
        for nfe in (50, 2):
            cov, prec, det, man32, man16, best, spr = [], [], [], [], [], [], []
            for s in (0, 1, 2):
                d = f"{root}/runs/r1_fmcn{cn}_s{s}"
                ev, e = J(f"{d}/eval_nfe{nfe}.json"), J(f"{d}/e3_nfe{nfe}.json")
                cov.append(ev["stoch"]["cov_pos"] if ev else None)
                prec.append(ev["stoch"]["prec_pos"] if ev else None)
                det.append(ev["det"]["pos_err"] if ev else None)
                man32.append(e["manifold"][31] if e else None)
                man16.append(e["manifold"][15] if e else None)
                best.append(e["best_err"][31] if e else None)
                spr.append(e["spread"][31] if e else None)
            print(f"  ctx_noise 0.{cn[1]} NFE {nfe}: cov {ms(cov)} | prec {ms(prec)} | det {ms(det, 2)} | "
                  f"manifold h=16 {ms(man16, 2)} h=32 {ms(man32, 2)} | best-of-16 h=32 {ms(best, 1)} | spread {ms(spr, 1)}")
            if nfe == 50 and None not in man32 and None not in cov:
                hit = mean(man32) <= 2.2 and mean(cov) >= 0.74
                print(f"    decision (NFE 50): manifold {mean(man32):.2f}, coverage {mean(cov):.3f} -> "
                      + ("drift attributed to training without context noise (rule triggered)" if hit
                         else "claim stays with 'including context-noise augmentation up to 0.3'"))


# ------------------------------------------------- second revision: 400 x 64 remeasurement of experiment 1
def x1r(n_boot=10000, seed=0):
    """Control (semicircle) minus arc at 400 training contexts x 64 samples (out/c2/*_400x64.json): the index and the
    tolerance-free index (seed means), a 95% bootstrap interval of the tolerance-free difference over contexts (per
    context the two seeds are averaged, which share their contexts; the two geometries are resampled independently),
    the difference on the monitor's first 200 contexts and on the other 200, the 95% half-width over contexts of one
    200-context, 16-sample index (out/c2/*_200x16.json, check.boot_ci), and the observed drop as a share of the
    predicted one (control minus the H_I prediction) for the monitor value and for the 400 x 64 index."""
    root = RAW["x1"]
    rng = np.random.default_rng(seed)
    print("## Experiment 1 remeasured: 400 contexts x 64 samples (control minus arc; tolerance-free interval over contexts,"
          f" {n_boot} resamples, seed {seed})")
    for m in (2, 4, 8):
        d = {}
        for geo, pre in (("arc", "r1_arc90"), ("std", "r1_k4")):
            r = [J(f"{root}/out/c2/{pre}_m{m}_s{s}_ckpt_nfeNone_400x64.json") for s in (0, 1)]
            q = [J(f"{root}/out/c2/{pre}_m{m}_s{s}_ckpt_nfeNone_200x16.json") for s in (0, 1)]
            if None in r or None in q:
                break
            if r[0]["train"]["per_context"]["t"] != r[1]["train"]["per_context"]["t"]:
                print(f"  m={m}: the two seeds of {pre} use different contexts")
            mon = [memo_rows(f"{root}/runs/{pre}_m{m}_s{s}/log.jsonl")[-1]["memo_train"] for s in (0, 1)]
            d[geo] = dict(idx=mean([x["derived"]["index_train"] for x in r]),
                          free=mean([x["derived"]["free_index_train"] for x in r]),
                          fs=np.mean([x["train"]["per_context"]["free_share"] for x in r], 0),
                          hw=[(x["train"]["realized_share"][2] - x["train"]["realized_share"][1]) / 2 for x in q],
                          mon=mean(mon))
        if len(d) < 2:
            print(f"  m={m}: incomplete")
            continue
        a, c = d["arc"], d["std"]
        ia = rng.integers(0, len(a["fs"]), (n_boot, len(a["fs"])))
        ic = rng.integers(0, len(c["fs"]), (n_boot, len(c["fs"])))
        bd = c["fs"][ic].mean(1) - a["fs"][ia].mean(1)
        lo, hi = np.percentile(bd, [2.5, 97.5])
        first = c["fs"][:200].mean() - a["fs"][:200].mean()
        rest = c["fs"][200:].mean() - a["fs"][200:].mean()
        line = (f"  m={m}: index control {c['idx']:.3f} arc {a['idx']:.3f} diff {c['idx'] - a['idx']:+.3f} | tolerance-free "
                f"control {c['free']:.3f} arc {a['free']:.3f} diff {c['free'] - a['free']:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                f"first 200 contexts {first:+.3f}, other 200 {rest:+.3f} | 200x16 index half-width arc "
                f"{'/'.join(f'{v:.3f}' for v in a['hw'])} control {'/'.join(f'{v:.3f}' for v in c['hw'])}")
        if m in H_I:
            p = H_I[m][0]
            line += (f" | observed/predicted drop: monitor {(c['mon'] - a['mon']) / (c['mon'] - p):.2f}, "
                     f"400x64 {(c['idx'] - a['idx']) / (c['idx'] - p):.2f}")
        print(line)


# ------------------------------------------------ second revision: the restarts of seed 1 of experiment 2b
def x2b_restart():
    """The three attempts of seed 1 of experiment 2b (x2.2: stopped at 36,600 steps and lost at about 20k steps,
    last pulled log 16k; x2.3: complete): largest absolute difference of the logged values at the common steps."""
    runs = {"stopped at 36,600": f"{RAW['x2.2']}/runs/r1_imle16_48k_s1_partial_stop36600/log.jsonl",
            "lost at about 20k": f"{RAW['x2.2']}/runs/r1_imle16_48k_s1_partial_lost20k/log.jsonl",
            "complete (x2.3)": f"{RAW['x2.3']}/runs/r1_imle16_48k_s1/log.jsonl"}
    print("## Experiment 2b, seed 1: logged values of the three attempts at their common steps")
    logs = {k: {r["step"]: r for r in log_rows(p)} for k, p in runs.items()}
    names = list(logs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = logs[names[i]], logs[names[j]]
            common = sorted(set(a) & set(b))
            if not common:
                print(f"  {names[i]} vs {names[j]}: no common steps")
                continue
            dif = {k: max(abs(a[s][k] - b[s][k]) for s in common if k in a[s] and k in b[s])
                   for k in ("loss", "match_d", "memo_train", "memo_val", "cov_val")
                   if any(k in a[s] and k in b[s] for s in common)}
            print(f"  {names[i]} vs {names[j]}: {len(common)} common logged steps ({common[0]}-{common[-1]}), "
                  "largest absolute difference " + ", ".join(f"{k} {v:.1e}" for k, v in dif.items()))


if __name__ == "__main__":
    which = sys.argv[1:] or ["x1", "x2a", "x2b", "x3", "x1r", "x2b_restart"]
    for w in which:
        {"x1": x1, "x2a": x2a, "x2b": x2b, "x3": x3, "x1r": x1r, "x2b_restart": x2b_restart}[w]()
        print()

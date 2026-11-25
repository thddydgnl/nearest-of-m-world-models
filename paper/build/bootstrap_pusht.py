#!/usr/bin/env python3
"""Bootstrap intervals of the paired differences on the stochastic Push-T multi-future test set (label tab:bci).

usage: python3 bootstrap_pusht.py <runs_dir> [--n 10000] [--seed 0]
  runs_dir: results/runs of the supplement (main/results/runs in the project)

Reads the errors per starting state of eval_ps*.json (128 starting states, three seeds per model): `mf.per_start`
for h = 16 and `mf_h8.per_start` for h = 8, the first 8 steps of the same 16-step rollouts. For a comparison of
model A with model B and one metric:
- difference: mean over seeds and starting states of A minus the same mean of B;
- starting-state CI: average the seeds of each model, then resample the 128 starting states with replacement;
- hierarchical CI: resample the three seeds of each model with replacement, independently for the two models of a
  comparison between methods and as pairs for per-step against one-step matching (runs with the same seed are
  paired), then resample the 128 starting states with replacement, jointly for both models;
10,000 resamples each, percentile 95% intervals (2.5 and 97.5). The manuscript table is the output of the default
--seed 0; other resampling seeds move the interval ends by up to about 0.3 px.
A second table (added in the second revision) gives the seed-level t-intervals that the text uses: the per-seed
means over the 128 starting states, Welch's 95% t-interval for comparisons between methods and a paired 95%
t-interval for per-step against one-step matching (no resampling, no adjustment for the eleven comparisons), and
the share of starting states on which model A has the lower seed-averaged error. Needs only numpy.
"""
import argparse
import json
import math
import os

import numpy as np

SEEDS3 = lambda fmt: [fmt.format(s) for s in range(3)]
METRICS = {"h8": "mf_h8", "h16": "mf"}
COMPARISONS = [  # label, runs of A, file of A, runs of B, file of B, paired seeds, (horizon, error) rows
    ("IMLE-WM, m=16 vs. FM-50", SEEDS3("e1_b_imle16_s{}"), "eval_ps.json", SEEDS3("e1_b_fm_s{}"), "eval_ps_nfe50.json",
     False, [("h8", "recall_err"), ("h8", "precision_err"), ("h16", "recall_err"), ("h16", "precision_err")]),
    ("IMLE-WM, m=16 vs. FM-2", SEEDS3("e1_b_imle16_s{}"), "eval_ps.json", SEEDS3("e1_b_fm_s{}"), "eval_ps_nfe2.json",
     False, [("h16", "recall_err"), ("h16", "precision_err")]),
    ("IMLE-WM, m=16 vs. Shortcut", SEEDS3("e1_b_imle16_s{}"), "eval_ps.json", SEEDS3("e7_b_sc_s{}"), "eval_ps_nfe1.json",
     False, [("h16", "recall_err"), ("h16", "precision_err")]),
    ("Per-step vs. one-step matching (8k)", SEEDS3("e6_b_seq4_s{}"), "eval_ps.json",
     ["e6_b_n0_8k", "e6_b_n0_8k_s1", "e6_b_n0_8k_s2"], "eval_ps.json",
     True, [("h8", "recall_err"), ("h16", "recall_err"), ("h16", "precision_err")]),
]


def load(runs_dir, runs, fname, horizon, err):
    """(seeds, 128) array of per-start errors."""
    return np.array([json.load(open(os.path.join(runs_dir, r, fname)))[METRICS[horizon]]["per_start"][err]
                     for r in runs], dtype=np.float64)


def intervals(a, b, paired, n, rng):
    diff = a.mean() - b.mean()
    d = a.mean(0) - b.mean(0)                                   # seeds averaged first
    idx = rng.integers(0, d.shape[0], (n, d.shape[0]))
    ci_start = np.percentile(d[idx].mean(1), [2.5, 97.5])
    sa = rng.integers(0, a.shape[0], (n, a.shape[0]))
    sb = sa if paired else rng.integers(0, b.shape[0], (n, b.shape[0]))
    idx = rng.integers(0, a.shape[1], (n, a.shape[1]))         # starting states, shared by both models
    ma = np.take_along_axis(a[sa].mean(1), idx, 1).mean(1)
    mb = np.take_along_axis(b[sb].mean(1), idx, 1).mean(1)
    ci_hier = np.percentile(ma - mb, [2.5, 97.5])
    return diff, ci_start, ci_hier


def t_cdf(x, nu, n=200001):
    """CDF of Student's t with nu (possibly fractional) degrees of freedom, by the trapezoid rule on [0, |x|]."""
    g = np.linspace(0.0, abs(x), n)
    c = math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi))
    y = c * (1 + g * g / nu) ** (-(nu + 1) / 2)
    return 0.5 + math.copysign(float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(g))), x)


def t_quantile(p, nu):
    """Upper quantile of Student's t by bisection (t(0.975, 2) = 4.3027, t(0.975, 4) = 2.7764)."""
    lo, hi = 0.0, 200.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, nu) < p else (lo, mid)
    return (lo + hi) / 2


def t_interval(a, b, paired):
    """95% t-interval of mean(A) - mean(B) over the per-seed means (mean over starting states of each seed).
    Paired: one-sample interval of the per-seed differences (n - 1 degrees of freedom); otherwise Welch's interval
    with the Welch-Satterthwaite degrees of freedom."""
    sa, sb = a.mean(1), b.mean(1)
    d = sa.mean() - sb.mean()
    if paired:
        dd = sa - sb
        se, nu = dd.std(ddof=1) / math.sqrt(len(dd)), len(dd) - 1
    else:
        va, vb = sa.var(ddof=1) / len(sa), sb.var(ddof=1) / len(sb)
        se, nu = math.sqrt(va + vb), (va + vb) ** 2 / (va ** 2 / (len(sa) - 1) + vb ** 2 / (len(sb) - 1))
    q = t_quantile(0.975, nu)
    return d, (d - q * se, d + q * se), nu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    f = lambda v: f"{v:+.1f}".replace("+", "+" if v > 0 else "") if abs(v) >= 0.05 else "0.0"
    print(f"Paired differences on the stochastic Push-T multi-future test set (px; negative favors the first model), "
          f"{a.n} resamples, seed {a.seed}")
    print(f"{'Comparison':38s} {'Metric':22s} {'Difference':>10s}  {'Starting-state CI':18s} {'Hierarchical CI'}")
    for label, ra, fa, rb, fb, paired, rows in COMPARISONS:
        for horizon, err in rows:
            A = load(a.runs_dir, ra, fa, horizon, err)
            B = load(a.runs_dir, rb, fb, horizon, err)
            d, cs, ch = intervals(A, B, paired, a.n, rng)
            metric = f"h={horizon[1:]} {err.replace('_err', '')} error"
            ci_s, ci_h = f"[{f(cs[0])}, {f(cs[1])}]", f"[{f(ch[0])}, {f(ch[1])}]"
            print(f"{label:38s} {metric:22s} {f(d):>10s}  {ci_s:18s} {ci_h}")
            label = ""
    # second revision: the seed-level t-intervals of the text (deterministic, no resampling)
    print()
    print("Seed-level 95% t-intervals over the three per-seed means (Welch between methods; paired for per-step vs. "
          "one-step matching), unadjusted; share: starting states on which the first model has the lower error "
          "(seeds averaged)")
    print(f"{'Comparison':38s} {'Metric':22s} {'Difference':>10s}  {'t-interval':18s} {'df':>5s}  {'Share'}")
    for label, ra, fa, rb, fb, paired, rows in COMPARISONS:
        for horizon, err in rows:
            A = load(a.runs_dir, ra, fa, horizon, err)
            B = load(a.runs_dir, rb, fb, horizon, err)
            d, ci, nu = t_interval(A, B, paired)
            share = float(np.mean(A.mean(0) < B.mean(0)))
            metric = f"h={horizon[1:]} {err.replace('_err', '')} error"
            print(f"{label:38s} {metric:22s} {f(d):>10s}  {'[' + f(ci[0]) + ', ' + f(ci[1]) + ']':18s} {nu:5.2f}  {share:.3f}")
            label = ""


if __name__ == "__main__":
    main()

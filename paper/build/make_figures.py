"""Generate the paper figures from the run outputs (eval.json / e3.json / log.jsonl / modeshare.json).

usage: python make_figures.py <runs_dir> <results_dir> <out_dir>
  runs_dir    : copy of the server's ~/imle-wm/runs (json/jsonl only)
  results_dir : the project's main/results/ folder (e15_toy/ json, feasibility/ pilot scatter png)
  out_dir     : paper/tmlr/figures

Hue follows the method family in every figure (regression gray, flow blue, iMF purple, Shortcut
orange, energy score green, nearest-of-m vermillion); the variant within a family (m, NFE) is the
line style. Palette: Okabe-Ito.
"""
import json
import math
import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS, RES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "lines.linewidth": 1.4,
    "lines.markersize": 4, "pdf.fonttype": 42, "figure.dpi": 150, "legend.frameon": False,
})

C = {"reg": "#8c8c8c", "fm": "#0072B2", "imf": "#CC79A7", "sc": "#C47A00", "es": "#009E73",
     "imle": "#D55E00", "truth": "#000000"}
STY = {
    "reg": dict(color=C["reg"], ls="--", marker="s", label="Regression"),
    "fm50": dict(color=C["fm"], ls="-", marker="o", label="FM-50"),
    "fm2": dict(color=C["fm"], ls=":", marker="o", mfc="white", label="FM-2"),
    "imf": dict(color=C["imf"], ls="-", marker="v", label="iMF"),
    "sc": dict(color=C["sc"], ls="-", marker="^", label="Shortcut"),
    "es8": dict(color=C["es"], ls="-", marker="D", label="Energy score, $m{=}8$"),
    "es16": dict(color=C["es"], ls=":", marker="D", mfc="white", label="Energy score, $m{=}16$"),
    "estok8": dict(color=C["es"], ls="-.", marker="d", mfc="white", label="Per-token energy score, $m{=}8$"),
    "imle4": dict(color=C["imle"], ls=":", marker="o", mfc="white", label="IMLE-WM, $m{=}4$"),
    "imle16": dict(color=C["imle"], ls="-", marker="o", label="IMLE-WM, $m{=}16$"),
}


def J(*p):
    path = os.path.join(RUNS, *p)
    return json.load(open(path)) if os.path.exists(path) else None


def log_rows(run):
    p = os.path.join(RUNS, run, "log.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p)]


def memo_curve(run):
    rows = [r for r in log_rows(run) if "memo_train" in r]
    return (np.array([r["step"] for r in rows]), np.array([r["memo_train"] for r in rows]),
            np.array([r["memo_val"] for r in rows]), np.array([r["cov_val"] for r in rows]))


def last_memo(run):
    rows = [r for r in log_rows(run) if "memo_train" in r]
    return rows[-1]["memo_train"] if rows else None


def seeds(fmt, n=3):
    return [fmt.format(s) for s in range(n) if os.path.isdir(os.path.join(RUNS, fmt.format(s)))]


def mean_curves(runs, key, fname="e3.json"):
    arrs = [np.array(J(r, fname)[key]) for r in runs if J(r, fname) is not None]
    if not arrs:
        return None
    n = min(len(a) for a in arrs)
    return np.mean([a[:n] for a in arrs], 0)


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ I(m, K) for the bounce geometry
def I_mK(m, K):
    """Expected squared distance from a uniformly drawn mode to the nearest of m i.i.d. uniform
    mode draws, for K equiangular bounce directions on a unit semicircle (exact)."""
    ang = np.deg2rad((np.arange(K) + 0.5) / K * 180.0 - 90.0)
    pts = np.stack([np.cos(ang), np.sin(ang)], 1)
    tot = 0.0
    for k in range(K):
        d = np.sort(((pts - pts[k]) ** 2).sum(1))  # d[0] = 0 (the mode itself)
        r = np.arange(1, K + 1)
        p_ge = ((K - r + 1) / K) ** m - ((K - r) / K) ** m  # P(nearest draw is the r-th closest)
        tot += (d * p_ge).sum()
    return tot / K


# ------------------------------------------------------------------ Fig: memorization collapse (E2 + E12)
grid = {}
for K in (2, 4, 8, 16):
    for m in (2, 4, 8, 16, 32, 64):
        pref = "e2" if K in (4, 8) else "e12"
        rs = seeds(f"{pref}_k{K}_m{m}_s{{}}", 2)
        vals = [last_memo(r) for r in rs]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        ev = [J(r, "eval.json") for r in rs]
        ev = [e for e in ev if e]
        grid[(K, m)] = dict(
            memo=float(np.mean(vals)), memo_seeds=vals,
            cov=float(np.mean([e["stoch"]["cov_pos"] for e in ev])),
            prec=float(np.mean([e["stoch"]["prec_pos"] for e in ev])),
            recall=float(np.mean([e["stoch"]["recall_knn"] for e in ev])),
            det=float(np.mean([e["det"]["pos_err"] for e in ev])),
            train_h=float(np.mean([e["train_time"] for e in ev])) / 3600,
            steps=ev[0]["args"]["steps"], I=I_mK(m, K),
            excess=(float(np.mean(vals)) - 1 / K) / (1 - 1 / K))
print("K x m grid:")
for (K, m), g in sorted(grid.items()):
    print(f"  K={K:2d} m={m:2d} steps={g['steps']} memo={g['memo']:.3f} excess={g['excess']:.3f} I={g['I']:.4f} "
          f"cov={g['cov']:.3f} prec={g['prec']:.3f} recall={g['recall']:.3f} det={g['det']:.2f}px train={g['train_h']:.2f}h")

# linear fit on the K = 4, 8 points (the fit used for the predictions recorded before the K = 2, 16 runs)
fitpts = [(g["I"], g["excess"]) for (K, m), g in grid.items() if K in (4, 8)]
xs, ys = np.array([p[0] for p in fitpts]), np.array([p[1] for p in fitpts])
b, a = np.polyfit(xs, ys, 1)
rmse = float(np.sqrt(np.mean((a + b * xs - ys) ** 2)))
print(f"fit on K=4,8: excess = {a:.3f} + {b:.3f} I, RMSE {rmse:.3f}")

# competing one-variable fits on all points except the extrapolated K=2, m=2 point
allpts = [(K, m, g["I"], g["excess"]) for (K, m), g in grid.items() if not (K == 2 and m == 2)]
def fit_rmse(x):
    y = np.array([p[3] for p in allpts]); x = np.array(x)
    bb, aa = np.polyfit(x, y, 1)
    return float(np.sqrt(np.mean((aa + bb * x - y) ** 2)))
r_I = fit_rmse([p[2] for p in allpts])
r_miss = fit_rmse([(1 - 1 / p[0]) ** p[1] for p in allpts])
r_exp = fit_rmse([math.exp(-p[1] / p[0]) for p in allpts])
print(f"one-variable linear fit RMSE (K=2,m=2 excluded): I(m,K) {r_I:.3f} | (1-1/K)^m {r_miss:.3f} | exp(-m/K) {r_exp:.3f}")

# predictions recorded before the K = 2, 16 runs (from theory_m_law.md, on the memorization-index scale)
PRED = {(2, 2): (0.77, 0.74, 0.81), (2, 4): (0.57, 0.54, 0.61), (2, 8): (0.51, 0.50, 0.54), (2, 16): (0.51, 0.50, 0.54),
        (16, 4): (0.31, 0.25, 0.37), (16, 8): (0.15, 0.09, 0.21), (16, 16): (0.10, 0.06, 0.16),
        (16, 32): (0.08, 0.06, 0.14), (16, 64): (0.08, 0.06, 0.14)}
hits = 0
for (K, m), (p, lo, hi) in PRED.items():
    ok = lo <= grid[(K, m)]["memo"] <= hi
    hits += ok
    print(f"  prediction K={K} m={m}: {p} [{lo}, {hi}] observed {grid[(K, m)]['memo']:.3f} -> {'in' if ok else 'OUT'}")
print(f"predictions inside interval: {hits}/{len(PRED)}")

KCOL = {2: "#9ecae1", 4: "#4292c6", 8: "#08519c", 16: "#08306b"}
KMK = {2: "o", 4: "s", 8: "^", 16: "D"}
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
ax = axes[0]
xx = np.linspace(0, 0.52, 100)
ax.fill_between(xx, a + b * xx - 2 * rmse, a + b * xx + 2 * rmse, color="#D55E00", alpha=0.12, lw=0)
ax.plot(xx, a + b * xx, color="#D55E00", lw=1, label=f"linear fit on $K{{=}}4,8$ (RMSE {rmse:.3f})")
for (K, m), g in sorted(grid.items()):
    ax.scatter(g["I"], g["excess"], color=KCOL[K], marker=KMK[K], s=22, zorder=3, edgecolor="white", linewidth=0.4)
    if (K, m) in PRED:
        p, lo, hi = PRED[(K, m)]
        ex = lambda v: (v - 1 / K) / (1 - 1 / K)
        ax.plot([g["I"], g["I"]], [ex(lo), ex(hi)], color=KCOL[K], lw=0.8, alpha=0.9, zorder=2)
    if (K, m) in ((2, 2), (2, 4), (4, 4), (8, 4), (16, 4)):
        off = {(2, 2): (-38, -11), (2, 4): (4, 2), (4, 4): (5, 2), (8, 4): (5, 4), (16, 4): (5, -8)}[(K, m)]
        ax.annotate(f"$K{{=}}{K}, m{{=}}{m}$", (g["I"], g["excess"]), textcoords="offset points", xytext=off, fontsize=5.5, color="#444")
for K in (2, 4, 8, 16):
    ax.scatter([], [], color=KCOL[K], marker=KMK[K], s=22, label=f"$K{{=}}{K}$")
ax.set_xlabel("codebook distortion $I(m,K)$ (unit-speed bounce geometry)")
ax.set_ylabel("excess memorization")
ax.set_title("(a) excess memorization vs. the miss distortion")
ax.legend(loc="upper left", ncol=1, handlelength=1.4)
ax.set_xlim(-0.01, 0.53); ax.set_ylim(-0.03, 0.97)
ax = axes[1]
for (K, m), g in sorted(grid.items()):
    ax.scatter((1 - 1 / K) ** m, g["excess"], color=KCOL[K], marker=KMK[K], s=22, zorder=3, edgecolor="white", linewidth=0.4)
    if (K, m) in ((2, 2), (2, 4), (4, 4), (8, 4), (16, 4), (16, 8)):
        ax.annotate(f"$K{{=}}{K}, m{{=}}{m}$", ((1 - 1 / K) ** m, g["excess"]), textcoords="offset points", xytext=(4, -3), fontsize=5.5, color="#444")
ax.set_xlabel("miss probability $(1-1/K)^m$")
ax.set_title("(b) the same points vs. the miss probability")
ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.03, 0.97)
ax.text(0.03, 0.9, f"one-variable fit RMSE:\n$I(m,K)$ {r_I:.3f}, $(1-1/K)^m$ {r_miss:.3f}, $e^{{-m/K}}$ {r_exp:.3f}",
        transform=ax.transAxes, fontsize=6.5, va="top")
savefig(fig, "fig_memo_collapse.pdf")

# ------------------------------------------------------------------ Fig: memorization and coverage during training (E1, E15)
curves = {
    "reg": seeds("e1_a2_reg_s{}"), "fm50": seeds("e1_a2_fm_s{}"),
    "imle4": seeds("e1_a2_imle4_s{}"), "imle16": seeds("e1_a2_imle16_s{}"),
    "es8": seeds("e15_a2_es8_s{}"), "es16": seeds("e15_a2_es16_s{}", 2), "estok8": seeds("e15_a2_estok8_s{}", 2),
}
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
for key, rs in curves.items():
    cs = [memo_curve(r) for r in rs]
    cs = [c for c in cs if len(c[0])]
    if not cs:
        print("no memo curves for", key); continue
    n = min(len(c[0]) for c in cs)
    st = cs[0][0][:n]
    mt = np.array([c[1][:n] for c in cs]); cv = np.array([c[3][:n] for c in cs])
    s = dict(STY[key]); s.pop("marker", None); s.pop("mfc", None)
    axes[0].plot(st / 1000, mt.mean(0), **s)
    axes[0].fill_between(st / 1000, mt.min(0), mt.max(0), color=s["color"], alpha=0.12, lw=0)
    axes[1].plot(st / 1000, cv.mean(0), **s)
    axes[1].fill_between(st / 1000, cv.min(0), cv.max(0), color=s["color"], alpha=0.12, lw=0)
axes[0].axhline(0.25, color="k", lw=0.7, ls="--"); axes[0].text(2.0, 0.235, "chance", fontsize=6, va="top")
axes[0].set_xlabel("training step (k)"); axes[0].set_ylabel("memorization index (training contexts)")
axes[0].set_title("(a) outcome memorization during training")
axes[1].set_xlabel("training step (k)"); axes[1].set_ylabel("coverage, held-out contexts")
axes[1].set_title("(b) mode coverage during training")
axes[1].axhline(1 - 0.75 ** 16, color="k", lw=0.7, ls="--"); axes[1].text(2.0, 1 - 0.75 ** 16 - 0.015, "ideal sampler", fontsize=6, va="top")
axes[0].set_ylim(0.15, 1.02); axes[1].set_ylim(0.0, 1.02)
h_, l_ = axes[1].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=4, handlelength=1.8, fontsize=6.3, bbox_to_anchor=(0.5, -0.2))
savefig(fig, "fig_memo_curves.pdf")

# ------------------------------------------------------------------ Fig: the m knob (E2, E12, E14)
share = {}
for m in (2, 4, 16):
    vals = [J(f"e14_asym_m{m}_s{s}", "modeshare.json")["test"]["share0"][0] for s in (0, 1)]
    share[m] = (np.mean(vals), vals)
fm50_share = np.mean([J(f"e14_asym_fm_s{s}", "modeshare_nfe50.json")["test"]["share0"][0] for s in (0, 1)])
fm2_share = np.mean([J(f"e14_asym_fm_s{s}", "modeshare_nfe2.json")["test"]["share0"][0] for s in (0, 1)])
es_share = np.mean([J(f"e15_asym_es8_s{s}", "modeshare.json")["test"]["share0"][0] for s in (0, 1)])
p0 = J("e14_asym_m16_s0", "modeshare.json")["test"]["data_freq0"]
print(f"asymmetric modes: true p0={p0:.3f}; IMLE share0 m=2 {share[2][0]:.3f} m=4 {share[4][0]:.3f} m=16 {share[16][0]:.3f}; "
      f"FM-50 {fm50_share:.3f} FM-2 {fm2_share:.3f} ES8 {es_share:.3f}")
def q_theory(m, p=p0):
    if m == 1:
        return 1.0
    r = ((1 - p) / p) ** (1 / (m - 1))
    return 1 / (1 + r)
for m in (2, 4, 16):
    print(f"  theory q0(m={m}) = {q_theory(m):.3f}; measured - FM-50 = {share[m][0]-fm50_share:+.3f}; theory - p = {q_theory(m)-p0:+.3f}")

fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.2))
axes = axes.ravel()
ideal = {K: 1 - (1 - 1 / K) ** 16 for K in (2, 4, 8, 16)}
for K in (2, 4, 8, 16):
    ms = sorted(m for (k, m) in grid if k == K)
    g = [grid[(K, m)] for m in ms]
    kw = dict(color=KCOL[K], marker=KMK[K], label=f"$K{{=}}{K}$")
    axes[0].plot(ms, [x["cov"] for x in g], **kw)
    axes[0].axhline(ideal[K], color=KCOL[K], lw=0.6, ls=":")
    axes[1].plot(ms, [x["prec"] for x in g], **kw)
    axes[2].plot(ms, [x["memo"] for x in g], **kw)
    axes[2].axhline(1 / K, color=KCOL[K], lw=0.6, ls=":")
    axes[3].plot(ms, [x["det"] for x in g], **kw)
    axes[4].plot(ms, [x["train_h"] for x in g], **kw)
for ax in axes[:5]:
    ax.set_xscale("log", base=2); ax.set_xlabel("candidate budget $m$")
    ax.set_xticks([2, 4, 8, 16, 32, 64]); ax.set_xticklabels([2, 4, 8, 16, 32, 64])
axes[0].set_title("(a) coverage (dotted: ideal sampler)"); axes[0].set_ylim(0, 1.03)
axes[1].set_title("(b) precision"); axes[1].set_ylim(0, 1.0)
axes[2].set_title("(c) memorization index (dotted: chance)"); axes[2].set_ylim(0, 1.0)
axes[3].set_title("(d) deterministic error (px)")
axes[4].set_title("(e) training time (GPU-h)")
axes[0].legend(loc="lower right", fontsize=6.3, handlelength=1.5)
ax = axes[5]
mm = np.arange(1, 65)
ax.plot(mm, [q_theory(m) for m in mm], color=C["imle"], lw=1, ls="-", label="optimum (Bennett, 2026)")
ax.scatter([2, 4, 16], [share[m][0] for m in (2, 4, 16)], color=C["imle"], marker="o", s=24, zorder=3, label="measured")
ax.axhline(p0, color="k", lw=0.8, ls="--"); ax.text(64, p0 - 0.012, f"true {p0:.2f}", fontsize=5.6, ha="right", va="top")
ax.axhline(fm50_share, color=C["fm"], lw=1, ls="-"); ax.text(64, fm50_share + 0.008, f"FM-50 {fm50_share:.2f}", fontsize=5.6, ha="right", va="bottom", color=C["fm"])
ax.axhline(es_share, color=C["es"], lw=1, ls="-"); ax.text(64, es_share + 0.008, f"energy score {es_share:.2f}", fontsize=5.6, ha="right", va="bottom", color=C["es"])
ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
ax.set_xlabel("candidate budget $m$"); ax.set_ylim(0.45, 1.02)
ax.set_title("(f) frequent-mode share")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.56), fontsize=5.6, handlelength=1.3)
fig.tight_layout(h_pad=1.0, w_pad=0.8)
savefig(fig, "fig_m_knob.pdf")


# ------------------------------------------------------------------ compact m-knob for the main text (1 x 4)
fig, axes = plt.subplots(1, 4, figsize=(6.5, 1.5))
for K in (2, 4, 8, 16):
    ms = sorted(m for (k, m) in grid if k == K)
    g = [grid[(K, m)] for m in ms]
    kw = dict(color=KCOL[K], marker=KMK[K], label=f"$K{{=}}{K}$", markersize=3)
    axes[0].plot(ms, [x["cov"] for x in g], **kw); axes[0].axhline(ideal[K], color=KCOL[K], lw=0.6, ls=":")
    axes[1].plot(ms, [x["memo"] for x in g], **kw); axes[1].axhline(1 / K, color=KCOL[K], lw=0.6, ls=":")
    axes[2].plot(ms, [x["det"] for x in g], **kw)
for ax in axes[:3]:
    ax.set_xscale("log", base=2); ax.set_xlabel("budget $m$"); ax.set_xticks([2, 4, 8, 16, 32, 64]); ax.set_xticklabels([2, 4, 8, 16, 32, 64])
axes[0].set_title("(a) coverage"); axes[0].set_ylim(0, 1.03); axes[0].legend(fontsize=5.6, handlelength=1.2, loc="lower right")
axes[1].set_title("(b) memorization index"); axes[1].set_ylim(0, 1.0)
axes[2].set_title("(c) deterministic error (px)")
ax = axes[3]
mm = np.arange(1, 65)
ax.plot(mm, [q_theory(m) for m in mm], color=C["imle"], lw=1, label="optimum (Bennett, 2026)")
ax.scatter([2, 4, 16], [share[m][0] for m in (2, 4, 16)], color=C["imle"], marker="o", s=18, zorder=3, label="measured")
ax.axhline(p0, color="k", lw=0.8, ls="--"); ax.text(64, p0 - 0.012, f"true {p0:.2f}", fontsize=5.4, ha="right", va="top")
ax.axhline(fm50_share, color=C["fm"], lw=1); ax.text(64, fm50_share + 0.008, f"FM-50 {fm50_share:.2f}", fontsize=5.4, ha="right", va="bottom", color=C["fm"])
ax.axhline(es_share, color=C["es"], lw=1); ax.text(64, es_share + 0.008, f"energy score {es_share:.2f}", fontsize=5.4, ha="right", va="bottom", color=C["es"])
ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16, 64]); ax.set_xticklabels([1, 4, 16, 64]); ax.set_xlabel("budget $m$"); ax.set_ylim(0.45, 1.02)
ax.set_title("(d) frequent-mode share"); ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.56), fontsize=5.4, handlelength=1.2)
fig.tight_layout(w_pad=0.6)
savefig(fig, "fig_m_knob_main.pdf")

# ------------------------------------------------------------------ Fig: BAIR under-dispersion (E8, E10, E16)
deff = J("e16_bair_deff.json"); dims = J("e16_bair_dims.json")
em = {int(k): v for k, v in deff["e_m"].items()}
meas = {int(k): v["mean"] for k, v in deff["measured_r"].items()}
def r_of(rho):
    return math.sqrt(2) * rho / math.sqrt(1 + rho * rho)
def r_model(m, d):
    return r_of(em[m] / math.sqrt(d))
# one-point fit at m = 16
r16 = meas[16]
rho16 = math.sqrt(r16 ** 2 / (2 - r16 ** 2))
d_fit = (em[16] / rho16) ** 2
pr_pooled = np.mean([deff[f"e8_bair_reg_s{s}/test_eval"]["pr"] for s in (0, 1)])
print(f"BAIR: measured r = {meas}; one-point fit d_eff = {d_fit:.1f}; predicted r(4) = {r_model(4, d_fit):.3f}, r(64) = {r_model(64, d_fit):.3f}; "
      f"pooled residual PR = {pr_pooled:.0f} -> predicted r = {[round(r_model(m, pr_pooled), 3) for m in (4, 16, 64)]}")
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3))
ax = axes[0]
mm = np.array([2, 4, 8, 16, 32, 64, 128, 256])
# e_m for the curve: numerical expectation of the max of m standard normals
def e_max(m, n=400000, seed=0):
    rng = np.random.default_rng(seed)
    return float(rng.standard_normal((n // 8, m)).max(1).mean()) if m <= 256 else float("nan")
em_curve = {m: (em[m] if m in em else e_max(m)) for m in mm}
ax.plot(mm, [r_of(em_curve[m] / math.sqrt(d_fit)) for m in mm], color=C["imle"], lw=1, label=f"isotropic model, $d_\\mathrm{{eff}}{{=}}{d_fit:.0f}$ (fit at $m{{=}}16$)")
ax.plot(mm, [r_of(em_curve[m] / math.sqrt(pr_pooled)) for m in mm], color=C["imle"], lw=1, ls=":", label=f"isotropic model, $d{{=}}$ residual PR ({pr_pooled:.0f})")
for m, v in meas.items():
    ax.scatter(m, v, color=C["imle"], s=24, zorder=3, edgecolor="white", linewidth=0.5)
ax.scatter([], [], color=C["imle"], s=24, label="measured, $h{=}1$")
ax.set_xscale("log", base=2); ax.set_xticks([4, 16, 64, 256]); ax.set_xticklabels([4, 16, 64, 256])
ax.set_xlabel("candidate budget $m$"); ax.set_ylabel("spread-to-error ratio $r$"); ax.set_ylim(0, 0.75)
ax.set_title("(a) spread-to-error ratio on BAIR"); ax.legend(fontsize=5.8, loc="upper left", handlelength=1.4)
ax = axes[1]
ms_ = [4, 16, 64]; keys = ["e8_bair_imle4_s0", "e8_bair_imle16_s0", "e10_bair_m64_s0"]
q = np.array([dims[k]["pr_ctx_q"] for k in keys])
ax.errorbar(ms_, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], color=C["imle"], marker="o", capsize=2, lw=1)
ax.set_xscale("log", base=2); ax.set_xticks(ms_); ax.set_xticklabels(ms_)
ax.set_xlabel("candidate budget $m$"); ax.set_ylabel("participation ratio")
ax.set_title("(b) per-context sample PR"); ax.set_ylim(0, 3.2)
ax = axes[2]
fo = np.array([dims[k]["f_own_q"] for k in keys]); ft = np.array([dims[k]["f_other_q"] for k in keys])
ax.errorbar(ms_, fo[:, 1], yerr=[fo[:, 1] - fo[:, 0], fo[:, 2] - fo[:, 1]], color=C["imle"], marker="o", capsize=2, lw=1, label="own context's top-16 directions")
ax.errorbar(ms_, ft[:, 1], yerr=[ft[:, 1] - ft[:, 0], ft[:, 2] - ft[:, 1]], color=C["reg"], marker="s", ls="--", capsize=2, lw=1, label="another context's directions")
ax.set_xscale("log", base=2); ax.set_xticks(ms_); ax.set_xticklabels(ms_)
ax.set_xlabel("candidate budget $m$"); ax.set_ylabel("fraction of residual captured")
ax.set_title("(c) overlap with the true residual"); ax.set_ylim(0, 0.35); ax.legend(fontsize=6, loc="upper left")
fig.tight_layout(w_pad=1.0)
savefig(fig, "fig_bair.pdf")

# ------------------------------------------------------------------ Fig: manifold distance vs horizon (E3, E11, E15)
def e3_series(bench):
    out = {}
    out["reg"] = mean_curves(seeds(f"e1_{bench}_reg_s{{}}"), "manifold")
    out["fm2"] = mean_curves(seeds(f"e1_{bench}_fm_s{{}}"), "manifold", "e3_nfe2.json")
    out["fm50"] = mean_curves(seeds(f"e1_{bench}_fm_s{{}}"), "manifold", "e3_nfe50.json")
    out["imf"] = mean_curves(seeds(f"e1_{bench}_imf_s{{}}"), "manifold", "e3_nfe1.json")
    out["imle4"] = mean_curves(seeds(f"e1_{bench}_imle4_s{{}}"), "manifold")
    out["imle16"] = mean_curves(seeds(f"e1_{bench}_imle16_s{{}}"), "manifold")
    if bench in ("a2", "a1"):
        out["es8"] = mean_curves(seeds(f"e15_{bench}_es8_s{{}}"), "manifold")
    return out
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3))
for ax, bench, title in zip(axes, ("a2", "a1", "b"), ("(a) A2-div bounce", "(b) A1 Bouncing Shapes", "(c) B stochastic Push-T")):
    ser = e3_series(bench)
    for key in ("reg", "fm50", "fm2", "imf", "es8", "imle4", "imle16"):
        y = ser.get(key)
        if y is None:
            continue
        s = dict(STY[key]); s["markevery"] = max(1, len(y) // 8); s["markersize"] = 3
        ax.plot(np.arange(1, len(y) + 1), y, **s)
        print(f"  manifold {bench} {key}: h=1 {y[0]:.2f} h=last {y[-1]:.2f}")
    ax.set_xlabel("rollout horizon $h$"); ax.set_title(title)
    ax.set_yscale("log")
axes[0].set_ylabel("manifold distance (1 = on data)")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=7, handlelength=1.8, fontsize=6.2, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(w_pad=0.8)
savefig(fig, "fig_manifold.pdf")

# ------------------------------------------------------------------ Fig: fed-back state ablation (E11, E16)
variants = {"N0": ("e11_a1_n0_s{}", dict(color=C["reg"], ls="--", marker="s", label="one-step matching")),
            "matched": ("e11_a1_seq4_s{}", dict(color=C["imle"], ls="-", marker="o", label="per-step, feed matched")),
            "random": ("e16_a1_seqrand_s{}", dict(color=C["sc"], ls="-", marker="^", label="per-step, feed random candidate")),
            "gt": ("e16_a1_seqgt_s{}", dict(color=C["fm"], ls=":", marker="v", label="per-step, feed true frame"))}
real_spread = {1: 0.20, 4: 1.90, 8: 6.81, 16: 23.6}  # pairwise distance among the true futures of an IC (e16_curves.py)
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3))
for name, (fmt, s) in variants.items():
    rs = seeds(fmt, 2)
    man = mean_curves(rs, "manifold"); best = mean_curves(rs, "best_err"); spr = mean_curves(rs, "spread")
    h = np.arange(1, len(man) + 1)
    kw = dict(s); kw["markevery"] = 3; kw["markersize"] = 3
    axes[0].plot(h, man, **kw); axes[1].plot(h, best, **kw); axes[2].plot(h, spr * 16 / 15, **kw)
    print(f"  seq ablation {name}: manifold h16 {man[15]:.2f} best h16 {best[15]:.1f} spread h16 {spr[15]*16/15:.1f}")
axes[2].scatter(list(real_spread), list(real_spread.values()), color="k", marker="*", s=40, zorder=4, label="true futures")
for ax, t in zip(axes, ("(a) manifold distance", "(b) best-of-16 readout error (px)", "(c) spread of the 16 rollouts (px)")):
    ax.set_xlabel("rollout horizon $h$"); ax.set_title(t)
axes[2].legend(fontsize=5.8, loc="upper left", handlelength=1.6)
fig.tight_layout(w_pad=0.8)
savefig(fig, "fig_seq_ablation.pdf")

# ------------------------------------------------------------------ Fig (appendix): toy experiment (E15)
toy = {}
for f in ("toy_s0.json", "toy_s1.json", "toy_nsweep.json", "toy_a0.json"):
    p = os.path.join(RES, "e15_toy", f)
    if os.path.exists(p):
        toy[f] = json.load(open(p))
def toy_pick(method, m, n_train, a_dim=16):
    out = []
    for f, rows in toy.items():
        for r in rows:
            if r["method"] == method and r["m"] == m and r["n_train"] == n_train and r.get("a_dim", 16) == a_dim:
                out.append(r)
    return out
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3))
ns = [1024, 4096, 16384, 65536]
for method, m, s in (("es", 8, STY["es8"]), ("nn", 16, STY["imle16"])):
    memo = [np.mean([r["train"]["memo"] for r in toy_pick(method, m, n)]) for n in ns]
    cov = [np.mean([r["test"]["cov2"] for r in toy_pick(method, m, n)]) for n in ns]
    fresh = np.mean([r["test"]["cov2"] for r in toy_pick(method, m, "fresh")])
    kw = dict(s); kw["label"] = {"es": "energy score, $m{=}8$", "nn": "nearest-of-$m$, $m{=}16$"}[method]
    axes[0].plot(ns, memo, **kw); axes[1].plot(ns, cov, **kw)
    axes[1].scatter([2.6e5], [fresh], color=s["color"], marker="x", s=30, zorder=3)
    print(f"  toy {method} m={m}: memo {np.round(memo,3)} cov2 {np.round(cov,3)} fresh cov2 {fresh:.3f}")
for ax in axes:
    ax.set_xscale("log", base=2); ax.set_xticks(ns + [2.6e5]); ax.set_xticklabels(["1k", "4k", "16k", "64k", "fresh"])
    ax.set_xlabel("number of fixed training contexts")
axes[0].axhline(0.5, color="k", lw=0.7, ls="--"); axes[0].set_ylabel("memorization index (training contexts)")
axes[0].set_title("(a) memorization vs. context revisits"); axes[0].set_ylim(0.4, 1.03)
axes[1].set_ylabel("two-mode coverage, new contexts"); axes[1].set_title("(b) coverage on new contexts"); axes[1].set_ylim(-0.03, 1.05)
axes[1].legend(fontsize=6.3, loc="lower right")
fig.tight_layout(w_pad=1.0)
savefig(fig, "fig_toy.pdf")

# ------------------------------------------------------------------ qualitative pilot figure (copied)
src = os.path.join(RES, "feasibility", "scatter_bounce_k4.png")
if os.path.exists(src):
    shutil.copy(src, os.path.join(OUT, "fig_scatter_pilot.png")); print("copied pilot scatter")

# ------------------------------------------------------------------ numbers for the tables (printed, cross-checked against the reports)
def agg(runs, f, path):
    vals = []
    for r in runs:
        d = J(r, f)
        if d is None:
            continue
        for k in path:
            d = d[k]
        vals.append(d[0] if isinstance(d, list) else d)
    return (np.mean(vals), np.std(vals), len(vals)) if vals else (float("nan"), float("nan"), 0)
print("\nA2-div main table (3 seeds):")
for key, fmt, f in (("reg", "e1_a2_reg_s{}", "eval.json"), ("fm50", "e1_a2_fm_s{}", "eval_nfe50.json"), ("fm2", "e1_a2_fm_s{}", "eval_nfe2.json"),
                    ("imf", "e1_a2_imf_s{}", "eval_nfe1.json"), ("sc", "e7_a2_sc_s{}", "eval_nfe1.json"), ("es8", "e15_a2_es8_s{}", "eval.json"),
                    ("es16", "e15_a2_es16_s{}", "eval.json"), ("estok8", "e15_a2_estok8_s{}", "eval.json"),
                    ("imle4", "e1_a2_imle4_s{}", "eval.json"), ("imle16", "e1_a2_imle16_s{}", "eval.json")):
    rs = seeds(fmt)
    cov = agg(rs, f, ["stoch", "cov_pos"]); pr = agg(rs, f, ["stoch", "prec_pos"]); rc = agg(rs, f, ["stoch", "recall_knn"])
    det = agg(rs, f, ["det", "pos_err"]); lat = agg(rs, f, ["latency_ms"]); tt = agg(rs, f, ["train_time"])
    mem = [last_memo(r) for r in rs]; mem = [x for x in mem if x is not None]
    print(f"  {key:7s} n={cov[2]} cov {cov[0]:.3f}±{cov[1]:.3f} prec {pr[0]:.3f}±{pr[1]:.3f} recall {rc[0]:.3f}±{rc[1]:.3f} det {det[0]:.2f}±{det[1]:.2f} "
          f"memo {np.mean(mem) if mem else float('nan'):.3f}±{np.std(mem) if mem else 0:.3f} lat {lat[0]:.1f}ms train {tt[0]/3600:.2f}h")
print("\nA1 table (F1 error, autoregressive 7M):")
for key, fmt, f in (("reg", "e1_a1_reg_s{}", "eval.json"), ("fm50", "e1_a1_fm_s{}", "eval_nfe50.json"), ("fm2", "e1_a1_fm_s{}", "eval_nfe2.json"),
                    ("imf", "e1_a1_imf_s{}", "eval_nfe1.json"), ("sc", "e7_a1_sc_s{}", "eval_nfe1.json"), ("es8", "e15_a1_es8_s{}", "eval.json"),
                    ("imle4", "e1_a1_imle4_s{}", "eval.json"), ("imle16", "e1_a1_imle16_s{}", "eval.json")):
    rs = seeds(fmt)
    for fe, tag in ((f, "probe"), (f.replace("eval", "eval_hm"), "heatmap"), (f.replace("eval", "eval_hmn05"), "heatmap-n0.5")):
        p = agg(rs, fe, ["model", "precision_err"]); r = agg(rs, fe, ["model", "recall_err"]); f1 = agg(rs, fe, ["model", "f1_err"])
        if f1[2]:
            print(f"  {key:7s} {tag:12s} n={f1[2]} P {p[0]:.2f}±{p[1]:.2f} R {r[0]:.2f}±{r[1]:.2f} F1 {f1[0]:.2f}±{f1[1]:.2f}")
print("\nA1 joint 16->16 (27-29M):")
for key, fmt, f in (("reg", "j_reg_s{}", "eval.json"), ("fm50", "j_fm_s{}", "eval_nfe50.json"), ("fm2", "j_fm_s{}", "eval_nfe2.json"),
                    ("imle4", "j_imle4_s{}", "eval.json"), ("imle16", "j_imle16_s{}", "eval.json")):
    rs = seeds(fmt)
    for fe, tag in ((f, "probe"), (f.replace("eval", "eval_hm"), "heatmap"), (f.replace("eval", "eval_hmn05"), "heatmap-n0.5")):
        f1 = agg(rs, fe, ["model", "f1_err"]); p = agg(rs, fe, ["model", "precision_err"]); r = agg(rs, fe, ["model", "recall_err"])
        if f1[2]:
            print(f"  {key:7s} {tag:12s} n={f1[2]} P {p[0]:.2f} R {r[0]:.2f} F1 {f1[0]:.2f}±{f1[1]:.2f}")
print("\nB table (19M):")
for key, fmt, f in (("reg", "e1_b_reg_s{}", "eval.json"), ("fm50", "e1_b_fm_s{}", "eval_nfe50.json"), ("fm2", "e1_b_fm_s{}", "eval_nfe2.json"),
                    ("imf", "e1_b_imf_s{}", "eval_nfe1.json"), ("sc", "e7_b_sc_s{}", "eval_nfe1.json"),
                    ("imle4", "e1_b_imle4_s{}", "eval.json"), ("imle16", "e1_b_imle16_s{}", "eval.json"),
                    ("n0_8k", "e6_b_n0_8k_s{}", "eval.json"), ("seq4_8k", "e6_b_seq4_s{}", "eval.json")):
    rs = seeds(fmt) if "{}" in fmt else [fmt]
    if key == "n0_8k":
        rs = ["e6_b_n0_8k", "e6_b_n0_8k_s1", "e6_b_n0_8k_s2"]
    p8 = agg(rs, f, ["mf_h8", "precision_err"]); r8 = agg(rs, f, ["mf_h8", "recall_err"])
    p16 = agg(rs, f, ["mf", "precision_err"]); r16 = agg(rs, f, ["mf", "recall_err"])
    print(f"  {key:8s} n={p16[2]} h8 P {p8[0]:.1f}±{p8[1]:.1f} R {r8[0]:.1f}±{r8[1]:.1f} | h16 P {p16[0]:.1f}±{p16[1]:.1f} R {r16[0]:.1f}±{r16[1]:.1f}")
print("\nBAIR table (19M, 2 seeds):")
for key, fmt, f in (("reg", "e8_bair_reg_s{}", "eval.json"), ("fm50", "e8_bair_fm_s{}", "eval_nfe50.json"), ("fm2", "e8_bair_fm_s{}", "eval_nfe2.json"),
                    ("sc", "e8_bair_sc_s{}", "eval_nfe1.json"), ("imle4", "e8_bair_imle4_s{}", "eval.json"), ("imle16", "e8_bair_imle16_s{}", "eval.json"),
                    ("m64", "e10_bair_m64_s{}", "eval.json"), ("coarse", "e10_bair_coarse16_s{}", "eval.json")):
    rs = seeds(fmt, 2)
    ds = [J(r, f) for r in rs]; ds = [d for d in ds if d]
    if not ds:
        ds = [J(r, "eval.json") for r in rs]; ds = [d for d in ds if d]
    if not ds:
        print("  missing", key); continue
    b = np.mean([d["best_rms"] for d in ds], 0); mn = np.mean([d["mean_rms"] for d in ds], 0); man = np.mean([d["manifold"] for d in ds], 0); sp = np.mean([d["spread"] for d in ds], 0)
    print(f"  {key:7s} n={len(ds)} best h1/4/8/16 {b[0]:.3f}/{b[3]:.3f}/{b[7]:.3f}/{b[15]:.3f} mean h16 {mn[15]:.3f} manifold h16 {man[15]:.2f} spread h16 {sp[15]:.2f} "
          f"FD h8/h16 {np.mean([d['fd_h8'] for d in ds]):.2f}/{np.mean([d['fd_h16'] for d in ds]):.2f} prec h16 {np.mean([d['prec_h16'] for d in ds]):.2f} recall h8/h16 {np.mean([d['rec_h8'] for d in ds]):.2f}/{np.mean([d['rec_h16'] for d in ds]):.2f}")
print("\nPlanning E13 (edge task):")
for key in ("imle16", "fm", "reg"):
    rows = [J("plan", f"e13_push_{key}_s{s}_edge_pos_std_s0.json")[0] for s in range(3)]
    print(f"  {key:7s} success {np.mean([r['success'] for r in rows]):.3f}±{np.std([r['success'] for r in rows]):.3f} pit {np.mean([r['pit'] for r in rows]):.3f}±{np.std([r['pit'] for r in rows]):.3f} "
          f"timeout {np.mean([r['timeout'] for r in rows]):.3f} wall {np.mean([r['wall_s'] for r in rows]):.0f}s")
orc = J("plan", "oracle_edge_pos_std_s0.json")
print("  oracle", [(r["lam"], r["success"], r["pit"]) for r in orc] if isinstance(orc, list) else orc)
rd = J("plan", "riskdiag_nfe2.json"); rd50 = J("plan", "riskdiag_fm50.json")
print(f"  true one-step pit share {rd['ice_true_pit_share']:.3f}, true 5-step pit prob {rd['true_pit_mean']:.3f}")
for meth in ("imle", "fm", "reg"):
    ms_ = [v for k, v in rd["models"].items() if v["method"] == meth]
    print(f"  {meth} nfe{ms_[0]['nfe']}: one-step {np.mean([v['ice_one_step_pit_share'] for v in ms_]):.3f}±{np.std([v['ice_one_step_pit_share'] for v in ms_]):.3f} "
          f"5-step {np.mean([v['model_pit_mean'] for v in ms_]):.3f}±{np.std([v['model_pit_mean'] for v in ms_]):.3f} abs err {np.mean([v['abs_err_pit'] for v in ms_]):.3f} corr {np.mean([v['corr_pit'] for v in ms_]):.2f}")
ms_ = list(rd50["models"].values())
print(f"  fm nfe50: one-step {np.mean([v['ice_one_step_pit_share'] for v in ms_]):.3f}±{np.std([v['ice_one_step_pit_share'] for v in ms_]):.3f} 5-step {np.mean([v['model_pit_mean'] for v in ms_]):.3f} abs err {np.mean([v['abs_err_pit'] for v in ms_]):.3f} corr {np.mean([v['corr_pit'] for v in ms_]):.2f}")
print("\nPlanning E5 (stochastic Push-T):")
for key, pat in (("reg", "e1_b_reg_s{0}_nfe1_s{0}.json"), ("fm2", "e1_b_fm_s{0}_nfe2_s{0}.json"), ("imle16", "e1_b_imle16_s{0}_nfe1_s{0}.json")):
    rows = [J("e5", pat.format(s)) for s in range(3)]; rows = [r for r in rows if r]
    print(f"  {key:7s} n={len(rows)} success {np.mean([r['success'] for r in rows]):.3f}±{np.std([r['success'] for r in rows]):.3f} pos {np.mean([r['final_pos_err'] for r in rows]):.1f} ang {np.mean([r['final_ang_err'] for r in rows]):.1f} plan s/step {np.mean([r['plan_s_per_step'] for r in rows]):.1f} cem {rows[0]['cem']}")
r50 = J("e5", "e1_b_fm_s0_nfe50_s0.json")
print(f"  fm50    n=1 success {r50['success']:.2f} n_eps {r50['n_eps']} plan s/step {r50['plan_s_per_step']:.1f}")
print("\nE4 countermeasures (A2-div K=4, 12k, 2 seeds):")
for key, fmt in (("m4+rs0.5", "e4_m4_rs05_s{}"), ("m4+noise1.0", "e4_m4_cn10_s{}"), ("m8+noise0.3", "e4_m8_cn03_s{}")):
    rs = seeds(fmt, 2)
    cov = agg(rs, "eval.json", ["stoch", "cov_pos"]); pr = agg(rs, "eval.json", ["stoch", "prec_pos"]); rc = agg(rs, "eval.json", ["stoch", "recall_knn"]); det = agg(rs, "eval.json", ["det", "pos_err"])
    mem = [last_memo(r) for r in rs]
    print(f"  {key:12s} cov {cov[0]:.3f} prec {pr[0]:.3f} recall {rc[0]:.3f} det {det[0]:.2f} memo {np.mean(mem):.3f}")
print("\nAblations (1 seed):")
for r in ("ab_l2", "ab_z64", "ab_steps18k", "ab_b_7m"):
    e = J(r, "eval.json")
    if e and "stoch" in e:
        print(f"  {r}: cov {e['stoch']['cov_pos']:.3f} prec {e['stoch']['prec_pos']:.3f} det {e['det']['pos_err']:.2f} memo {last_memo(r)}")
    elif e:
        print(f"  {r}: h8 R {e['mf_h8']['recall_err']:.1f} h16 R {e['mf']['recall_err']:.1f} h16 P {e['mf']['precision_err']:.1f}")
print("\nA2-fix (E15 extra, 2 seeds):")
for key, fmt in (("es8", "e15_fix_es8_s{}"), ("imle16", "e15_fix_imle16_s{}"), ("imle4", "e15_fix_imle4_s{}")):
    rs = seeds(fmt, 2)
    cov = agg(rs, "eval.json", ["stoch", "cov_pos"]); pr = agg(rs, "eval.json", ["stoch", "prec_pos"]); rc = agg(rs, "eval.json", ["stoch", "recall_knn"]); det = agg(rs, "eval.json", ["det", "pos_err"])
    bo = agg(rs, "eval.json", ["stoch", "best_of_n"]); mem = [last_memo(r) for r in rs]
    print(f"  {key:7s} cov {cov[0]:.3f}±{cov[1]:.3f} prec {pr[0]:.3f}±{pr[1]:.3f} recall {rc[0]:.3f} det {det[0]:.2f} best16 {bo[0]:.3f} memo {np.mean(mem):.3f}")
print("\nE9 (48k):")
for r in ("e9_a1_imle4_48k_s0", "e9_a1_fm48k_s0"):
    for f in ("eval.json", "eval_nfe2.json", "eval_nfe50.json", "eval_hm.json"):
        e = J(r, f)
        if e: print(f"  {r} {f}: F1 {e['model']['f1_err']:.2f} P {e['model']['precision_err']:.2f} R {e['model']['recall_err']:.2f}")
print("\niMF 72k curve (A2):")
e = J("e7_a2_imf72k_s0", "eval.json")
if e: print("  final:", e["stoch"]["cov_pos"], e["stoch"]["prec_pos"])
print("done")

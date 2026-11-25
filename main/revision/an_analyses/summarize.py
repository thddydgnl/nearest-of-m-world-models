"""Print the tables of REPORT.md from the pulled result JSON files (reads raw/an.1/out/ only; seconds on a laptop).

usage: python3 main/revision/an_analyses/summarize.py [section ...]   sections: b a c d e f h smoke c64 e5
"""
import glob
import json
import os
import sys

import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "an.1", "out")
LOCAL_RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "runs")


def J(p):
    return json.load(open(p))


def last_memo(run):
    p = os.path.join(LOCAL_RUNS, run, "log.jsonl")
    rows = [json.loads(l) for l in open(p) if "memo_train" in l]
    return rows[-1]["memo_train"], rows[-1]["memo_val"]


def gaps(d):
    return "/".join("%+.4f" % v["gap_min_dist"][0] for v in d)


def sec_b():
    print("## b: grid, 400+400 contexts x 64 samples (mean of two seeds; seeds in brackets)")
    print("| K | m | monitor index (log) | (i) index train / held-out | (ii) tol-free train / held-out | excess (chance) | (iii) excess (held-out) | diff train-ho [95% CI, s0] | (iv) gap min-dist train-ho | rel. gap |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for f in sorted(glob.glob(f"{R}/b_grid/e*_ckpt_nfeNone_400x64.json")):
        r = J(f)
        K, m = r["K"], r["m"]
        rows.setdefault((K, m), []).append(r)
    for (K, m), rs in sorted(rows.items()):
        d = [x["derived"] for x in rs]
        mon = [last_memo(x["run"])[0] for x in rs]
        f2 = lambda k: np.mean([v[k] for v in d])
        seeds = lambda k: "/".join(f"{v[k]:.3f}" for v in d)
        print(f"| {K} | {m} | {np.mean(mon):.3f} ({'/'.join(f'{v:.3f}' for v in mon)}) | {f2('index_train'):.3f} / {f2('index_heldout'):.3f} | "
              f"{f2('free_index_train'):.3f} / {f2('free_index_heldout'):.3f} | {f2('excess_chance'):.3f} ({seeds('excess_chance')}) | "
              f"{f2('excess_heldout'):.3f} ({seeds('excess_heldout')}) | {d[0]['diff_heldout'][0]:+.3f} [{d[0]['diff_heldout'][1]:+.3f}, {d[0]['diff_heldout'][2]:+.3f}] | "
              f"{np.mean([v['gap_min_dist'][0] for v in d]):+.4f} ({gaps(d)}) | {f2('gap_min_dist_rel'):+.3f} |")
    n = [x["test"]["n"] for rs in rows.values() for x in rs]
    print(f"contexts used per split: {min(n)}-{max(n)}")


def sec_a():
    print("## a: token-space distortion")
    for K in (2, 4, 8, 16):
        p = f"{R}/a_itok/itok_bounce_div_k{K}.json"
        if not os.path.exists(p):
            continue
        t = J(p)
        print(f"K={K}: pair RMS mean {t['pair_rms']['mean']:.4f}, nearest {t['pair_rms']['nearest_mean']:.4f}; pair px nearest {t['pair_px']['nearest_mean']:.2f}")
        print("  m:", " | ".join(f"{m}: I_unit {t['I_unit'][m]:.4f} I_tok {t['I_tok'][m]:.4f} I_gm {t['I_tok_gm'].get(m, {}).get('I', float('nan')):.4f} (c {t['I_tok_gm'].get(m, {}).get('c', float('nan')):.3f})" for m in t["I_tok"]))
        for s in t["scales"]:
            print(f"    scale {s['run']}: c_median {s['c_median']:.4f} best_mean {s['best_mean']:.4f} log match_d {s['log_match_d_final']:.4f}")
    p = f"{R}/a_itok/fit_grid.json"
    if os.path.exists(p):
        f = J(p)
        for y, fits in f["fits"].items():
            print(f"fit y={y}: " + ", ".join(f"{x} {v['rmse']:.4f} (LOKO {v['rmse_leave_one_K_out']:.4f})" for x, v in fits.items()))


def sec_c():
    print("## c: memorization index at the model's own NFE")
    print("| file | nfe | n_ctx x n | index train | held-out | tol-free train | cov held-out | prec held-out |")
    print("|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(f"{R}/c_nfe/*.json")):
        r = J(f)
        if "derived" not in r:
            continue
        d = r["derived"]
        print(f"| {r['run']} | {r['nfe_used']} | {r['n_ctx']}x{r['n']} | {d['index_train']:.3f} | {d['index_heldout']:.3f} | {d['free_index_train']:.3f} | "
              f"{r['test']['cov'][0]:.3f} [{r['test']['cov'][1]:.3f}, {r['test']['cov'][2]:.3f}] | {r['test']['prec'][0]:.3f} |")


def sec_d():
    print("## d: readout offset")
    p = f"{R}/d_offset/offset_bounce_div_k2_p80_400.json"
    if os.path.exists(p):
        r = J(p)
        for s in ("train", "test"):
            o = r[s]
            print(f"{s}: n {o['n']}, branch valid {o['branch0_valid']:.3f}/{o['branch1_valid']:.3f}, b0->mode0 {o['branch0_assigned_mode0']:.3f}, b1->mode0 {o['branch1_assigned_mode0']:.3f}, data freq0 {o['data_freq0']:.4f}, render rms {o['render_vs_stored_rms'][0]:.4f}")
            for q, v in o["mix"].items():
                print(f"   q={q}: exact {v['exact_measured']:.4f} ({v['exact_minus_true']:+.4f}); n64 {v['n64_measured']:.4f} vs {v['n64_true']:.4f} ({v['n64_minus_true']:+.4f}); free {v['free_minus_true']:+.4f}")
            print(f"   data: {o['data_mix']}")
    for f in sorted(glob.glob(f"{R}/d_offset/e14*.json")):
        r = J(f)
        print(r["run"], "share0 train/test", round(r["train"]["share0"][0], 4), round(r["test"]["share0"][0], 4),
              "index", round(r["derived"]["index_train"], 4), round(r["derived"]["index_heldout"], 4))


def sec_e():
    print("## e: planner replays")
    for f in sorted(glob.glob(f"{R}/e_planrisk/*_replay.json")):
        r = J(f)
        s = r["summary"]
        print(f"{s['model']}: success {s['success']:.2f} pit {s['pit']:.2f} timeout {s['timeout']:.2f} wall {s['wall_s']}s")
        for part in ("selected", "executed"):
            for crit in ("last", "any"):
                v = r[part][crit]
                print(f"   {part:8s} {crit:4s} n={r[part]['n']}: model {v['model_mean']:.3f} true {v['true_mean']:.3f} diff {v['mean_diff']:+.3f} abs {v['abs_err']:.3f} corr {v['corr']}")
    for f in sorted(glob.glob(f"{R}/e_planrisk/riskdiag_both_nfe*.json")):
        r = J(f)
        print(f"riskdiag nfe {r['nfe']}: true {r['true_pit_mean']:.4f}, ice true {r['ice_true_pit_share']:.4f}")
        for k, v in r["models"].items():
            print(f"   {k}: last {v['last']['model_mean']:.4f} (abs {v['last']['abs_err']:.4f}) any {v['any']['model_mean']:.4f} (abs {v['any']['abs_err']:.4f}) ice {v['ice_one_step_pit_share']:.4f}")


def sec_f():
    print("## f: diagnostics")
    for f in sorted(glob.glob(f"{R}/f_diag/*.json")):
        r = J(f)
        if "stdout" in r:
            print(os.path.basename(f), "\n   " + "\n   ".join(r["stdout"]))
        else:
            print(os.path.basename(f), {k: v for k, v in r.items() if k != "h"}, r.get("h"))


def sec_c64():
    """Second revision: held-out coverage and precision with 64 samples per context (400 contexts), seed means."""
    print("## c64: coverage with 64 samples per held-out context (400 contexts; mean of seeds)")
    for label, pat in (("FM-2 (12k, Table 1 checkpoints)", "c_nfe/e1_a2_fm_s*_ckpt_nfe2_400x64.json"),
                       ("Shortcut (24k, Table 1 checkpoints)", "c_nfe/e7_a2_sc_s*_ckpt_nfe1_400x64.json"),
                       ("iMF (36k, Table 1 checkpoints)", "c_nfe/e1_a2_imf_s*_ckpt_nfe1_400x64.json"),
                       ("IMLE-WM m=16, K=4 (8k grid)", "b_grid/e2_k4_m16_s*_ckpt_nfeNone_400x64.json"),
                       ("IMLE-WM m=16, K=8 (8k grid)", "b_grid/e2_k8_m16_s*_ckpt_nfeNone_400x64.json"),
                       ("IMLE-WM m=16, K=16 (8k grid)", "b_grid/e12_k16_m16_s*_ckpt_nfeNone_400x64.json")):
        rs = [J(f) for f in sorted(glob.glob(f"{R}/{pat}"))]
        if not rs:
            print(f"{label}: missing")
            continue
        cov, prec = [r["test"]["cov"][0] for r in rs], [r["test"]["prec"][0] for r in rs]
        print(f"{label}: coverage {np.mean(cov):.3f} ({'/'.join(f'{v:.3f}' for v in cov)}), precision {np.mean(prec):.3f}, "
              f"{rs[0]['n_ctx']} contexts x {rs[0]['n']} samples, {len(rs)} seeds")


def sec_e5():
    """Second revision: five-push risk test, share of imagined falls that leave the pit before the last frame."""
    print("## e5: five-push risk test, imagined falls that leave the absorbing pit before the last frame (1 - last/any)")
    for f in sorted(glob.glob(f"{R}/e_planrisk/riskdiag_both_nfe*.json")):
        r = J(f)
        by = {}
        for k, v in r["models"].items():
            by.setdefault(k.rsplit("_s", 1)[0], []).append((v["last"]["model_mean"], v["any"]["model_mean"]))
        for name, vals in sorted(by.items()):
            last, anyf = np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals])
            print(f"nfe {r['nfe']} {name}: last {last:.3f}, any {anyf:.3f}, remain {last / anyf:.3f}, "
                  f"leave {1 - last / anyf:.3f} ({len(vals)} seeds)")


def sec_smoke():
    for f in glob.glob(f"{R}/smoke/*.json"):
        r = J(f)
        print("smoke", r["stoch"]["cov_pos"], r["stoch"]["prec_pos"])
    for f in glob.glob(f"{R}/b_grid/validate_*.json"):
        r = J(f)
        print("validate", r["run"], r["identical"], r["max_abs_diff"])
    for f in glob.glob(f"{R}/c1_test/*.json"):
        r = J(f)
        print("c1", {k: (v[0] if isinstance(v, list) else v) for k, v in r.items()})


if __name__ == "__main__":
    secs = sys.argv[1:] or ["smoke", "b", "a", "c", "d", "e", "f", "c64", "e5"]
    for s in secs:
        globals()["sec_" + s]()
        print()

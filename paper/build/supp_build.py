#!/usr/bin/env python3
"""Assemble the anonymized supplementary material of the TMLR submission (called by make_supplement.sh).

usage: python3 paper/build/supp_build.py --repo <repository root> --out <output folder> [--notes <file>]

Everything is copied from the current sources on each run; nothing in the output is edited by hand:
- code, queue files, per-run outputs and the revision-r1 files are copied with identifying strings scrubbed
  (home-directory paths, host names, ...) and with the path changes that the supplement layout needs;
- README.md, LICENSE, data/README.md, predictions/recorded.md and run_args/runs.csv are generated here;
  table and figure numbers come from paper/tmlr/main.aux, so rebuild the manuscript first;
- arguments of runs whose evaluation files carry no 'args' come from paper/build/supp_missing_args.json
  (read once from the original checkpoints), so no access to the checkpoints is needed.
The anonymity checks (grep), the smoke tests, the fixed mtimes and the zip are in make_supplement.sh.
Only the standard library is used.
"""
import argparse
import ast
import csv
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

DESIGN_SHA256 = "959409ddf4ddc6be671a092454c1d4ad47d6396f06228790ee4ec2f6b7a157bc"  # DESIGN_r1.md as frozen
TEXT_EXT = {".py", ".sh", ".txt", ".md", ".json", ".jsonl", ".log", ".csv"}
RAW_EXT = {".json", ".jsonl", ".txt", ".npz"}      # revision-r1 raw outputs that are copied (no previews, no logs)
RESULT_EXT = {".json", ".jsonl", ".png"}           # main/results/{runs,check,e15_toy,feasibility}
NOTES = []                                         # warnings for the coordinator (build_notes.txt)


def note(msg):
    NOTES.append(msg)
    print("NOTE:", msg, flush=True)


def die(msg):
    print("ERROR:", msg, file=sys.stderr, flush=True)
    sys.exit(1)


# =============================================================================================== scrubbing
GENERIC_SCRUB = [  # applied to every text file of the supplement, after the file-specific transforms
    (r"ssh kiis-mvf-gpu '([^']*)' < (\S+)", r"\1 < \2"),     # "run on the server" lines of the summary scripts
    (r"/home/yonghwi paths fixed on the pod copy", "absolute home-directory paths adjusted in the analysis machine's copy"),
    (r"/home/yonghwi/imle-wm", "~/imle-wm"),
    (r"/home/yonghwi", "~"),
    (r"/Users/[^/\s'\"]+", "~"),
    (r"kiis-mvf-gpu", "the training server"),
    (r"(?i)racedreamer[\w.-]*", "[redacted]"),
    (r"the Claude session", "an interactive session"),
]


def scrub(text):
    for pat, rep in GENERIC_SCRUB:
        text = re.sub(pat, rep, text)
    return text


def py_home_paths(text):
    """Absolute home paths in Python string literals -> os.path.expanduser("~/imle-wm/..."), adding `import os`."""
    n = 0
    text, k = re.subn(r'f"/home/yonghwi/imle-wm/([^"]*)"', r'os.path.expanduser(f"~/imle-wm/\1")', text)
    n += k
    text, k = re.subn(r'"/home/yonghwi/imle-wm/([^"]*)"', r'os.path.expanduser("~/imle-wm/\1")', text)
    n += k
    if n and not re.search(r"^\s*import\s+(?:[\w.]+\s*,\s*)*os\b", text, re.M):
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if re.match(r"(import|from)\s", line):
                lines.insert(i, "import os")
                break
        text = "\n".join(lines)
    return text


ANALYSIS_TEXT = [  # docstrings and paths of the scripts copied to analysis/ (order matters)
    ("main/revision/an_analyses/REPORT.md, section h", "DESIGN_r1.md section 4h"),
    ("Print the tables of REPORT.md from the pulled result JSON files (reads raw/an.1/out/ only;",
     "Print the tables of analyses a-h (DESIGN_r1.md section 4) from the result JSON files "
     "(reads revision_r1/an_analyses/raw/an.1/out/ only;"),
    ("python3 main/revision/an_analyses/summarize.py", "python3 analysis/summarize_analyses.py"),
    ("python3 main/revision/summarize_r1.py", "python3 analysis/summarize_r1.py"),
    ("pulled by runpod/pull_all.sh", "copied from the GPU machines"),
    ("raw/<pod>/", "raw/<machine>/"),
    ("generated on the pod", "generated on the GPU machine"),
    ("(from the repository root)", "(from the supplement root)"),
    ("main/revision/", "revision_r1/"),
    ("main/results/", "results/"),
    ("main/code/", "code/"),
    ("paper/build/make_figures.py", "analysis/make_figures.py"),
    ("paper/tmlr/make_fig_samples.py", "analysis/make_fig_samples.py"),
    ("paper/build/make_fig_samples.py", "analysis/make_fig_samples.py"),
    ("paper/tmlr/figures/", "figures/"),
]


def analysis_text(text):
    for old, new in ANALYSIS_TEXT:
        text = text.replace(old, new)
    return text


def fix_summarize_r1(text):
    old = "HERE = os.path.dirname(os.path.abspath(__file__))"
    if old not in text:
        die("summarize_r1.py: expected line '%s' not found; update fix_summarize_r1" % old)
    text = text.replace(old, 'HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "revision_r1")'
                             "  # supplement layout", 1)
    return analysis_text(text)


def fix_summarize_analyses(text):
    expr = "os.path.dirname(os.path.abspath(__file__))"
    lines = text.split("\n")
    first = next((i for i, line in enumerate(lines) if expr in line), None)
    if first is None:
        die("an_analyses/summarize.py: no path built from __file__; update fix_summarize_analyses")
    new = lines[:first] + ['_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "revision_r1", '
                           '"an_analyses")  # supplement layout'] + [line.replace(expr, "_SRC") for line in lines[first:]]
    return analysis_text("\n".join(new))


DESIGN_REDACT = [  # (pattern, replacement); each must match exactly once in the frozen text of DESIGN_r1.md
    (r"\(`paper/review/[^`]*`\)", "([redacted: path of an internal review report])"),
    (r"RunPod Secure Cloud, ", "[redacted: cloud provider], "),
    (r"Pod prefix `nom`, tools in\s+`runpod/`\.\s+", "[redacted: tool names and paths.] "),
    (r"Budget: the user's cap is USD 30; the watchdog limit is USD 24 because the shared account balance was\s+"
     r"USD 26\.62 at the start\.\s+", "[redacted: account budget.] "),
    (r"the Google Drive backup `racedreamer-backup-2026-11`", "[redacted: backup location]"),
]


def redact_design(text):
    for pat, rep in DESIGN_REDACT:
        text, k = re.subn(pat, rep, text)
        if k != 1:
            die("DESIGN_r1.md redaction pattern matched %d times (expected 1): %s" % (k, pat))
    return text, len(DESIGN_REDACT)


# =============================================================================================== copying
MANIFEST = []


def write_text(dst, text):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
        f.write(text)


def read_text(src):
    with open(src, encoding="utf-8", errors="surrogateescape", newline="") as f:
        return f.read()


def copy_file(src, dst, transform=None):
    ext = os.path.splitext(src)[1].lower()
    if ext in TEXT_EXT:
        text = read_text(src)
        if transform:
            text = transform(text)
        if ext == ".py":
            text = py_home_paths(text)
        write_text(dst, scrub(text))
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    MANIFEST.append(dst)


def copy_tree(src_dir, dst_dir, exts, transform=None):
    n = 0
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", "__MACOSX") and not d.startswith("."))
        for fn in sorted(files):
            if fn.startswith(".") or os.path.splitext(fn)[1].lower() not in exts:
                continue
            src = os.path.join(root, fn)
            copy_file(src, os.path.join(dst_dir, os.path.relpath(src, src_dir)), transform)
            n += 1
    return n


# =============================================================================================== LaTeX helpers
def read_aux(path):
    labels, cites = {}, {}
    if not os.path.exists(path):
        note("paper/tmlr/main.aux not found: table and figure numbers are shown as labels")
        return labels, cites
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}", line)
        if m:
            labels[m.group(1)] = m.group(2).replace("\\relax ", "").strip()
        m = re.match(r"\\bibcite\{([^}]*)\}\{\{[^}]*\}\{([^}]*)\}\{\{([^}]*)\}\}", line)
        if m:
            cites[m.group(1)] = (m.group(3).replace("~", " "), m.group(2))
    return labels, cites


def braced(s, i):
    """Content of the brace group that opens at s[i] == '{'."""
    assert s[i] == "{"
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


MATH = [(r"\\(?:leq|le)\b", "≤"), (r"\\(?:geq|ge)\b", "≥"), (r"\\approx\b", "≈"), (r"\\lesssim\b", "≲"),
        (r"\\gtrsim\b", "≳"), (r"\\pm\b", "±"), (r"\\times\b", "×"), (r"\\(?:to|rightarrow)\b", "→"),
        (r"\\in\b", "∈"), (r"\\infty\b", "∞"), (r"\\cdot\b", "·"), (r"\\(?:ldots|dots|cdots)\b", "…"),
        (r"\\epsilon\b", "ε"), (r"\\sigma\b", "σ"), (r"\\mu\b", "μ"), (r"\\theta\b", "θ"), (r"\\rho\b", "ρ"),
        (r"\\alpha\b", "α"), (r"\\beta\b", "β"), (r"\\sum\b", "Σ"), (r"\\uparrow\b", "↑"), (r"\\downarrow\b", "↓"),
        (r"\\(?:big|Big|bigg|Bigg|left|right)\b", ""), (r"\\(?:t|d)?frac\b", "")]


def math2txt(s):
    s = s.replace("{=}", "=").replace("{,}", ",").replace("\\{", "\x01").replace("\\}", "\x02")
    s = re.sub(r"\\(?:mathrm|text|mathit|mathbf|operatorname|boldsymbol)\{([^{}]*)\}", r"\1", s)
    for pat, rep in MATH:
        s = re.sub(pat, rep, s)
    s = s.replace("\\,", " ").replace("\\;", " ").replace("\\!", "").replace("~", " ").replace("\\%", "%")
    s = re.sub(r"\^\{([^{}]*)\}", lambda m: "^" + (m.group(1) if len(m.group(1)) == 1 else "(" + m.group(1) + ")"), s)
    s = re.sub(r"_\{([^{}]*)\}", r"_\1", s)
    s = s.replace("{", "").replace("}", "").replace("\x01", "{").replace("\x02", "}")
    return s


def tex2md(s, labels, cites):
    s = re.sub(r"(?<!\\)%.*", "", s)
    for k, v in ((r"\imlewm{}", "IMLE-WM"), (r"\imlewm", "IMLE-WM"), (r"\Nom{}", "Nearest-of-$m$"),
                 (r"\Nom", "Nearest-of-$m$"), (r"\nom{}", "nearest-of-$m$"), (r"\nom", "nearest-of-$m$")):
        s = s.replace(k, v)

    def ref(m):
        lab = m.group(1)
        if lab not in labels:
            note("unresolved \\ref{%s} in text copied from the manuscript" % lab)
        return labels.get(lab, lab)

    def cite(m):
        out = []
        for k in m.group(2).split(","):
            a, y = cites.get(k.strip(), (k.strip(), ""))
            out.append((a, y))
        if m.group(1) == "t":
            return "; ".join(f"{a} ({y})" for a, y in out)
        return "(" + "; ".join(f"{a}, {y}" for a, y in out) + ")"

    s = re.sub(r"\\(?:ref|cref|autoref)\{([^}]*)\}", ref, s)
    s = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(" + ref(m) + ")", s)
    s = re.sub(r"\\cite(t|p)?\{([^}]*)\}", cite, s)
    s = re.sub(r"\\ms\{([^{}]*)\}\{([^{}]*)\}", r"$\1 \\pm \2$", s)
    for cmd, l, r in (("emph", "*", "*"), ("textit", "*", "*"), ("textbf", "**", "**"), ("texttt", "`", "`"),
                      ("best", "**", "**"), ("paragraph", "**", "**")):
        while True:
            i = s.find("\\" + cmd + "{")
            if i < 0:
                break
            body, j = braced(s, i + len(cmd) + 1)
            s = s[:i] + l + body + r + s[j:]
    s = re.sub(r"\$([^$]*)\$", lambda m: math2txt(m.group(1)), s)
    s = s.replace("---", "—").replace("--", "–").replace("``", "“").replace("''", "”")
    s = s.replace("~", " ").replace("\\ ", " ").replace("\\,", " ").replace("\\%", "%").replace("\\&", "&")
    s = s.replace("\\_", "_").replace("\\@", "")
    s = re.sub(r"\\(?:FloatBarrier|noindent|smallskip|medskip|bigskip|centering|small|footnotesize|scriptsize)\b", "", s)
    s = re.sub(r"[ \t\r\n]+", " ", s)
    return s.strip()


def tabular2md(body, labels, cites):
    body = re.sub(r"\\(?:toprule|midrule|bottomrule|hline)", "", body)
    body = re.sub(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}", "", body)
    rows = []
    for raw in body.split("\\\\"):
        raw = raw.strip()
        if not raw:
            continue
        cells = []
        for c in raw.split("&"):
            c = c.strip()
            m = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$", c, re.S)
            if m:
                cells += [tex2md(m.group(2), labels, cites)] + [""] * (int(m.group(1)) - 1)
            else:
                cells.append(tex2md(c, labels, cites))
        rows.append(cells)
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def tex_env_with_label(tex, env, label):
    for m in re.finditer(r"\\begin\{%s\*?\}(.*?)\\end\{%s\*?\}" % (env, env), tex, re.S):
        if "\\label{%s}" % label in m.group(1):
            return m.group(1)
    return None


def caption_of(body):
    i = body.find("\\caption{")
    return braced(body, i + len("\\caption"))[0] if i >= 0 else ""


def paragraphs(tex):
    return [p for p in re.split(r"\n\s*\n", tex) if p.strip()]


def sentence_span(text, first_tag, last_tag):
    a = text.find(first_tag)
    b = text.find(last_tag, a + 1) if a >= 0 else -1
    if a < 0 or b < 0:
        return None
    start = text.rfind(". ", 0, a)
    start = 0 if start < 0 else start + 2
    m = re.compile(r"\.\s+(?=[A-Z])").search(text, b)
    end = m.start() + 1 if m else len(text)
    return text[start:end].strip()


# =============================================================================================== argparse of the training scripts
def parser_from_source(path):
    """Rebuild the argparse parser of a script from its add_argument calls (literal arguments only)."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    calls = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "add_argument"), key=lambda n: (n.lineno, n.col_offset))
    p = argparse.ArgumentParser(add_help=False)
    actions = {}
    for c in calls:
        args = [ast.literal_eval(a) for a in c.args]
        kw = {}
        for k in c.keywords:
            if k.arg == "type":
                kw["type"] = {"int": int, "float": float, "str": str}[k.value.id]
            else:
                kw[k.arg] = ast.literal_eval(k.value)
        act = p.add_argument(*args, **kw)
        actions[act.dest] = act
    return p, actions


def fmt_value(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def command_line(module, actions, args):
    parts = ["python -m imlewm." + module]
    keep = {"data", "method"}
    for dest, act in actions.items():
        if dest not in args or dest == "name":
            continue
        v = args[dest]
        if isinstance(act, argparse._StoreTrueAction):
            if v:
                parts.append(act.option_strings[0])
            continue
        if v is None or (v == act.default and dest not in keep):
            continue
        parts.append(f"{act.option_strings[0]} {fmt_value(v)}")
    for k in sorted(set(args) - set(actions)):
        parts.append(f"--{k} {fmt_value(args[k])}")  # option unknown to the current script (none at present)
    if "name" in args:
        parts.append("--name " + str(args["name"]))
    return " ".join(parts)


# =============================================================================================== paper map
GRID = [r"^e2_k[48]_m\d+_s\d$", r"^e12_k(2|16)_m\d+_s\d$"]
A2_E3 = [r"^e1_(a2|a1|b)_(reg|fm|imf|imle4|imle16)_s\d$", r"^e15_(a2|a1)_es8_s\d$"]
MAP = [  # label, title, how to regenerate, sources (runs, files, JSON fields), run-name patterns for runs.csv
    ("tab:bench", "Benchmarks", "data generators (`data/README.md`)",
     "sizes and stochastic-transition shares printed by the generators (`results/logs/gen*.log`); readout errors "
     "`results/logs/probe*.log`, `heatmap_train.log`", []),
    ("tab:a2", "A2-div main comparison", "`analysis/make_figures.py` (printout \"A2-div main table\")",
     "`e1_a2_{reg,fm,imf,imle4,imle16}_s{0,1,2}`, `e7_a2_sc_s{0,1,2}`, `e15_a2_es8_s{0,1,2}`, `e15_a2_es16_s{0,1}`: "
     "`eval.json` (FM `eval_nfe50.json`, `eval_nfe2.json`; iMF and Shortcut `eval_nfe1.json`), fields `stoch.cov_pos`, "
     "`stoch.prec_pos`, `stoch.recall_knn`, `det.pos_err`, `latency_ms`, `train_time` (s); memorization index: last "
     "`memo_train` of `log.jsonl`, for FM-50, FM-2 and iMF `derived.index_train` of "
     "`revision_r1/an_analyses/raw/an.1/out/c_nfe/<run>_ckpt_nfe{50,2,1}_200x16.json` "
     "(`analysis/summarize_analyses.py c`)",
     [r"^e1_a2_(reg|fm|imf|imle4|imle16)_s\d$", r"^e7_a2_sc_s\d$", r"^e15_a2_es(8|16)_s\d$"]),
    ("tab:a1", "Bouncing Shapes (A1)", "`analysis/make_figures.py` (\"A1 table\", \"A1 joint 16->16\")",
     "autoregressive `e1_a1_{reg,fm,imle4,imle16}_s*`, joint `j_{reg,fm,imle4,imle16}_s*`; caption: `e1_a1_imf_s*`, "
     "`e7_a1_sc_s*`, `e15_a1_es8_s*`: `model.f1_err` of `eval*.json` (probe readout) and `eval_hm*.json` (heatmap "
     "readout); last-frame copy `copy_last.f1_err`",
     [r"^e1_a1_", r"^j_", r"^e7_a1_sc_", r"^e15_a1_es8_"]),
    ("tab:b", "Stochastic Push-T (B)", "`analysis/make_figures.py` (\"B table\")",
     "`e1_b_{reg,fm,imf,imle4,imle16}_s*`, `e7_b_sc_s*`: `mf_h8.precision_err`, `mf_h8.recall_err` (h = 8) and "
     "`mf.precision_err`, `mf.recall_err` (h = 16) of `eval*.json`; true-future spread: "
     "`revision_r1/an_analyses/raw/an.1/out/f_diag/gtspread_bdiag.json` (output of `code/diag/bdiag.py`)",
     [r"^e1_b_(reg|fm|imf|imle4|imle16)_s\d$", r"^e7_b_sc_s\d$"]),
    ("fig:collapse", "Excess memorization against I(m,K)", "`analysis/make_figures.py` -> `fig_memo_collapse.pdf`",
     "budget grid `e2_k{4,8}_m*_s{0,1}`, `e12_k{2,16}_m*_s{0,1}`: last `memo_train` of `log.jsonl`; recorded "
     "predictions: `PRED` in the script (`predictions/recorded.md`)", GRID),
    ("fig:knob", "What the budget controls", "`analysis/make_figures.py` -> `fig_m_knob_main.pdf`",
     "budget grid: `stoch.cov_pos`, `det.pos_err` of `eval.json`, last `memo_train` of `log.jsonl`; asymmetric modes "
     "`e14_asym_m{2,4,16}_s{0,1}`: `test.share0`, `test.data_freq0` of `modeshare.json`",
     GRID + [r"^e14_asym_m(2|4|16)_s\d$"]),
    ("tab:predfull", "Recorded predictions for K = 2, 16", "`analysis/make_figures.py` (printout \"prediction K=...\")",
     "budget grid (last `memo_train` of `log.jsonl`); predictions: `PRED` in the script; see "
     "`predictions/recorded.md`", GRID),
    ("tab:cog", "Candidate hidden variables for Push-T", "`code/diag/cog_sweep.py`, `code/diag/phys_sweep.py`",
     "printed output, not saved; DINO-WM Push-T episodes (`data/README.md`)", []),
    ("tab:recipe", "Recipe comparisons", "`imlewm.evaluate` and the monitor check `imlewm.check`",
     "`imle_e0_full`, `imle_e0_late`, `imle_e0_plain`, `imle_div_k4` (pilot recipe), `fm_div_k4`, `fm_e0_flowwm`, "
     "`imf_e0_ts1`: `eval*.json`; intervals and memorization: `results/runs/check/memo*.json` (logs "
     "`results/logs/check_memo*.log`); iMF with one step: "
     "`revision_r1/an_analyses/raw/an.1/out/c_nfe/imf_e0_ts1_ckpt_nfe1_300x16.json`",
     [r"^(imle_e0_(full|late|plain)|imle_div_k4|fm_div_k4|fm_e0_flowwm|imf_e0_ts1)$"]),
    ("tab:readout", "A1 under three readouts", "`analysis/make_figures.py` (\"A1 table\", \"A1 joint\", \"E9 (48k)\")",
     "as @tab:a1@, plus `eval_hmn05*.json` (heatmap readout trained with feature noise 0.5) and the 48k-step runs "
     "`e9_a1_imle4_48k_s0`, `e9_a1_fm48k_s0`",
     [r"^e1_a1_", r"^j_", r"^e7_a1_sc_", r"^e15_a1_es8_", r"^e9_a1_"]),
    ("fig:scatter", "Qualitative pilot", "copied by `analysis/make_figures.py` from "
     "`results/feasibility/scatter_bounce_k4.png`", "pilot run `imle_k4` (figure drawn by `imlewm.report`)",
     [r"^imle_k4$"]),
    ("tab:grid", "Budget grid", "`analysis/make_figures.py` (printout \"K x m grid\")",
     "budget grid: `stoch.cov_pos`, `stoch.prec_pos`, `stoch.recall_knn`, `det.pos_err`, `train_time` of "
     "`eval.json`; last `memo_train` of `log.jsonl`", GRID),
    ("fig:curves", "Memorization during training", "`analysis/make_figures.py` -> `fig_memo_curves.pdf`",
     "`e1_a2_{reg,fm,imle4,imle16}_s*`, `e15_a2_es8_s*`, `e15_a2_es16_s{0,1}`, `e15_a2_estok8_s{0,1}`: "
     "`memo_train`, `cov_val` of `log.jsonl` (every 2,000 steps)",
     [r"^e1_a2_(reg|fm|imle4|imle16)_s\d$", r"^e15_a2_es(8|16|tok8)_s\d$"]),
    ("tab:counter", "Memorization countermeasures", "`analysis/make_figures.py` (\"E4 countermeasures\")",
     "`e4_m4_rs05_s{0,1}`, `e4_m4_cn10_s{0,1}`, `e4_m8_cn03_s{0,1}` and the IMLE-WM rows of @tab:a2@: `eval.json`, "
     "`memo_train` / `memo_val` of `log.jsonl`", [r"^e4_", r"^e1_a2_imle(4|16)_s\d$"]),
    ("fig:knobfull", "What the budget controls (full)", "`analysis/make_figures.py` -> `fig_m_knob.pdf`",
     "as @fig:knob@, plus `e14_asym_fm_s{0,1}` (`modeshare_nfe50.json`) and `e15_asym_es8_s{0,1}` "
     "(`modeshare.json`)", GRID + [r"^e14_asym_", r"^e15_asym_es8_"]),
    ("tab:flatten", "Asymmetric modes", "`analysis/make_figures.py` (printout \"asymmetric modes\")",
     "`e14_asym_{m2,m4,m16,fm}_s{0,1}`, `e15_asym_es8_s{0,1}`: `modeshare*.json` (`test.share0`, "
     "`test.data_freq0`, `test.cov`, `train.realized_share`, `test.realized_share`); readout check: "
     "`revision_r1/an_analyses/raw/an.1/out/d_offset/` (`analysis/summarize_analyses.py d`)",
     [r"^e14_asym_", r"^e15_asym_es8_"]),
    ("fig:bair", "Under-dispersion on BAIR", "`analysis/make_figures.py` -> `fig_bair.pdf`",
     "`results/runs/e16_bair_deff.json`, `results/runs/e16_bair_dims.json` (written by `imlewm.bair_deff` from "
     "`e8_bair_reg_s{0,1}`, `e8_bair_imle{4,16}_s0`, `e10_bair_m64_s0`)",
     [r"^e8_bair_(reg|imle4|imle16)_s\d$", r"^e10_bair_m64_s0$"]),
    ("tab:bair", "BAIR robot pushing", "`analysis/make_figures.py` (printout \"BAIR table\")",
     "`e8_bair_{reg,fm,sc,imle4,imle16}_s{0,1}`, `e10_bair_m64_s0`, `e10_bair_coarse16_s0`: per-horizon arrays "
     "`best_rms`, `mean_rms`, `manifold`, `spread` and `fd_h8`, `fd_h16`, `prec_h16`, `rec_h8`, `rec_h16` of "
     "`eval*.json`", [r"^e8_bair_", r"^e10_bair_"]),
    ("tab:memo400", "Re-measured memorization of the grid", "`analysis/summarize_analyses.py b` (revision r1, analysis b)",
     "`revision_r1/an_analyses/raw/an.1/out/b_grid/<run>_ckpt_nfeNone_400x64.json` (`derived`); monitor: last "
     "`memo_train` of `log.jsonl`", GRID),
    ("fig:manifold", "Manifold distance against the horizon", "`analysis/make_figures.py` -> `fig_manifold.pdf`",
     "`e1_{a2,a1,b}_{reg,fm,imf,imle4,imle16}_s*`, `e15_{a2,a1}_es8_s*`: per-horizon `manifold` of `e3.json` (FM "
     "`e3_nfe50.json`, `e3_nfe2.json`; iMF `e3_nfe1.json`)", A2_E3),
    ("tab:e3", "Rollouts from real contexts", "`e3*.json` at the last horizon (`analysis/make_figures.py` prints the "
     "manifold distances)", "as @fig:manifold@: `manifold`, `best_err`, `spread` at h = 32 (A2-div) or 16 (A1, B)",
     A2_E3),
    ("tab:bci", "Paired bootstrap on Push-T", "resampling of the per-start errors (procedure in the statistics "
     "appendix; no script included)",
     "`mf.per_start`, `mf_h8.per_start` of `eval_ps*.json`: `e1_b_imle16_s*` (`eval_ps.json`), `e1_b_fm_s*` "
     "(`eval_ps_nfe50.json`, `eval_ps_nfe2.json`), `e7_b_sc_s*` (`eval_ps_nfe1.json`), `e6_b_seq4_s*`, `e6_b_n0_8k*`",
     [r"^e1_b_(imle16|fm)_s\d$", r"^e7_b_sc_s\d$", r"^e6_b_(seq4_s\d|n0_8k(_s\d)?)$"]),
    ("tab:es", "Energy score against nearest-of-m", "`analysis/make_figures.py` (\"A2-div main table\" energy-score "
     "rows, \"A2-fix\")",
     "`e15_a2_es8_s*`, `e15_a2_es16_s{0,1}`, `e15_a2_estok8_s{0,1}`, `e15_fix_{es8,imle16,imle4}_s{0,1}`, "
     "`e1_a2_imle{4,16}_s*`: `eval.json`, `manifold[0]` of `e3.json`, last `memo_train` of `log.jsonl`",
     [r"^e15_(a2|fix)_", r"^e1_a2_imle(4|16)_s\d$"]),
    ("fig:toy", "Toy two-branch problem", "`analysis/make_figures.py` -> `fig_toy.pdf`",
     "`results/e15_toy/toy_*.json` (`python -m imlewm.toy_memo`, CPU)", []),
    ("tab:seq", "Per-step matching on Push-T", "`analysis/make_figures.py` (\"B table\" rows `n0_8k`, `seq4_8k`)",
     "`e6_b_n0_8k`, `e6_b_n0_8k_s1`, `e6_b_n0_8k_s2`, `e6_b_seq4_s{0,1,2}`: `mf_h8.*`, `mf.*`, `train_time` of "
     "`eval.json`; text: deterministic control `e6_det_n0`, `e6_det_seq4`, A1 and A2 runs `e11_*`",
     [r"^e6_", r"^e11_"]),
    ("tab:seqabl", "Fed-back-state ablation", "`analysis/make_figures.py` (printout \"seq ablation\")",
     "`e11_a1_n0_s{0,1}`, `e11_a1_seq4_s{0,1}`, `e16_a1_seqrand_s{0,1}`, `e16_a1_seqgt_s{0,1}`: `manifold`, "
     "`best_err`, `spread` (x 16/15) of `e3.json`, `model.f1_err` of `eval.json`; true-future spread: constants in "
     "the script (computed by `code/run/e16_curves.py`)", [r"^e11_a1_", r"^e16_a1_"]),
    ("fig:seqabl", "Fed-back-state ablation", "`analysis/make_figures.py` -> `fig_seq_ablation.pdf`",
     "as @tab:seqabl@", [r"^e11_a1_", r"^e16_a1_"]),
    ("tab:plan", "Planning", "`analysis/make_figures.py` (\"Planning E13\", \"Planning E5\"); "
     "`analysis/summarize_analyses.py e`",
     "(a) `results/runs/e5/*.json` (`success`; `plan_s_per_step` divided by `n_eps`), models `e1_b_{reg,fm,imle16}_s*`; "
     "(b) `results/runs/plan/e13_push_{imle16,fm,reg}_s*_edge_pos_std_s0.json` and `oracle_edge_pos_std_s0.json` "
     "(`success`, `pit`, `timeout`); pit probability after one push: `plan/riskdiag_nfe2.json`, "
     "`plan/riskdiag_fm50.json`; after five pushes: `revision_r1/an_analyses/raw/an.1/out/e_planrisk/riskdiag_both_nfe*.json`",
     [r"^e13_push_", r"^e1_b_(reg|fm|imle16)_s\d$"]),
    ("sec:phenomenon", "Memorization in non-repeating contexts (text)", "training logs and the pilot memorization check",
     "`imle_div_k4` (pilot recipe, m = 4), `imle_div_k4_24k` (twice as long), `imle_div_k4_big` (2.8x larger): "
     "`results/runs/check/memo.json` and `log.jsonl`; `ab_steps18k` (m = 16, 18k steps): last `memo_train` of `log.jsonl`",
     [r"^imle_div_k4(_24k|_big)?$", r"^ab_steps18k$"]),
    ("sec:b", "Contact error on Push-T (text)", "`code/diag/bstep.py`, re-run by `python -m imlewm.r1_diag script` "
     "(revision r1, analysis f)",
     "`revision_r1/an_analyses/raw/an.1/out/f_diag/contact_bstep.json` for `e1_b_reg_s0`, `e1diag_b_reg36k` (36k steps), "
     "`e1_b_fm_s0`, `e1_b_imle4_s0`, `e1_b_imle16_s0`", [r"^e1diag_b_reg36k$"]),
    ("app:ablation", "Ablations and the iMF learning curve (text)", "`analysis/make_figures.py` (printout \"Ablations "
     "(1 seed)\"); JSON fields",
     "`ab_l2`, `ab_z64`, `ab_steps18k` (A2-div) and `ab_b_7m` (Push-T, 8M): `eval.json`, last `memo_train` of "
     "`log.jsonl`; pilot `imle_k4`, `imle_k4_l2`: `stoch.cov_tok` of `eval.json`, `prec_probe` of "
     "`results/runs/check/main.json`; iMF curve `e7_a2_imf72k_s0`: `stoch.cov_pos`, `stoch.prec_pos` of "
     "`eval_ckpt_<step>_nfe1.json` and `eval_nfe1.json` (72k)",
     [r"^ab_", r"^e7_a2_imf72k_s0$", r"^imle_k4(_l2)?$"]),
    ("fig:samples", "Decoded next-frame samples on A2-div", "`analysis/make_fig_samples.py`",
     "`revision_r1/an_analyses/raw/an.1/out/h_samples/` (`samples*.json`, `context_frames*.npz`), written by "
     "`python -m imlewm.r1_samples` from the seed-0 checkpoints of @tab:a2@",
     [r"^(e1_a2_(reg|fm|imf|imle4|imle16)|e7_a2_sc|e15_a2_es(8|16))_s0$"]),
]

GROUPS = [
    (r"^r1_tim_", "revision r1, analysis g: training-step timing on synthetic tokens"),
    (r"^r1_(arc90|k4)_", "revision r1, experiment 1: mode geometry"),
    (r"^r1_fm36k_", "revision r1, experiment 2a: flow matching, 36k steps"),
    (r"^r1_imle16_48k_", "revision r1, experiment 2b: IMLE-WM m=16, 48k steps"),
    (r"^r1_fmcn", "revision r1, experiment 3: flow matching with context noise"),
    (r"^r1_", "revision r1"),
    (r"^_", "failed or test run (kept for completeness)"),
    (r"^e1diag_", "E1 diagnostic"),
    (r"^e1_", "E1 main comparison"),
    (r"^e2_", "E2 budget grid, K = 4, 8"),
    (r"^e4_", "E4 memorization countermeasures"),
    (r"^e6_", "E6 per-step matching"),
    (r"^e7_", "E7 Shortcut baseline and iMF learning curve"),
    (r"^e8_", "E8 BAIR"),
    (r"^e9_", "E9 48k-step training budget (A1)"),
    (r"^e10_", "E10 BAIR, m = 64 and coarse matching"),
    (r"^e11_", "E11 per-step matching on A1 and A2"),
    (r"^e12_", "E12 budget grid, K = 2, 16"),
    (r"^e13_", "E13 branch-risk planning models"),
    (r"^e14_", "E14 asymmetric modes"),
    (r"^e15_", "E15 energy score"),
    (r"^e16_", "E16 fed-back-state ablation"),
    (r"^j_", "A1 joint 16-to-16 protocol"),
    (r"^ab_", "ablation"),
    (r"_e0", "E0 recipes and baselines"),
    (r".", "feasibility study (pilot)"),
]


def group_of(run):
    return next(g for pat, g in GROUPS if re.search(pat, run))


def item_name(label, labels):
    kind = {"tab": "Table", "fig": "Figure", "sec": "Section", "app": "Appendix"}.get(label.split(":")[0], "Item")
    return f"{kind} {labels[label]}" if label in labels else None


def resolve_refs(text, labels):
    return re.sub(r"@([a-z]+:[\w-]+)@", lambda m: item_name(m.group(1), labels) or m.group(1), text)


# =============================================================================================== runs.csv
def last_log(path):
    step, t = None, None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "step" in r:
            step = r["step"] if step is None else max(step, r["step"])
        if "t" in r:
            t = r["t"]
    return step, t


def build_runs(repo, labels):
    code = os.path.join(repo, "main/code")
    tr_p, tr_act = parser_from_source(os.path.join(code, "imlewm/train.py"))
    jo_p, jo_act = parser_from_source(os.path.join(code, "imlewm/joint.py"))
    missing = json.load(open(os.path.join(repo, "paper/build/supp_missing_args.json"), encoding="utf-8"))["runs"]

    queue = {}  # run name -> (queue file, line)
    for q in sorted(glob.glob(os.path.join(code, "q*.txt"))) + sorted(glob.glob(os.path.join(repo, "main/revision/queues/*.txt"))):
        for line in open(q, encoding="utf-8"):
            m = re.search(r"--name\s+(\S+)", line)
            if m and line.strip():
                queue.setdefault(m.group(1), (os.path.relpath(q, repo), line.strip()))

    timing = {}
    tcfg = os.path.join(code, "imlewm/r1_timing_configs.json")
    tsrc = read_text(os.path.join(code, "imlewm/r1_timing.py"))
    if os.path.exists(tcfg) and all(s in tsrc for s in ('"--memo_every", "0", "--save_every", "0"', "steps=400",
                                                         'name = "r1_tim_" + cfg["label"].replace(" ", "_").replace("=", "").replace("-", "")')):
        for c in json.load(open(tcfg, encoding="utf-8")):
            name = "r1_tim_" + c["label"].replace(" ", "_").replace("=", "").replace("-", "")
            argv = ["--data", f"r1_tim_a{c['model_args'].get('act_dim', 2)}", "--name", name, "--steps", "400",
                    "--memo_every", "0", "--save_every", "0"]
            for k, v in c["train_args"].items():
                act = tr_act.get(k)
                if isinstance(act, argparse._StoreTrueAction):
                    argv += [f"--{k}"] if v else []
                elif v is not None:
                    argv += [f"--{k}", str(v)]
            timing[name] = (c["label"], argv)
    elif os.path.exists(tcfg):
        note("r1_timing.py changed: timing-run arguments are not reconstructed (update build_runs)")

    found = []
    res = os.path.join(repo, "main/results/runs")
    for d in sorted(os.listdir(res)):
        if os.path.exists(os.path.join(res, d, "log.jsonl")):
            found.append((d, os.path.join(res, d), "results/runs/" + d))
    seen = {d for d, _, _ in found}
    for p in sorted(glob.glob(os.path.join(repo, "main/revision/*/raw/*/runs/*/log.jsonl"))):
        d = os.path.basename(os.path.dirname(p))
        rel = os.path.relpath(os.path.dirname(p), os.path.join(repo, "main/revision"))
        if d in seen:
            note(f"run {d} appears twice (also under revision_r1/{rel}); the first copy is listed")
            continue
        seen.add(d)
        found.append((d, os.path.dirname(p), "revision_r1/" + rel))

    rows, discrepancies = [], []
    for run, path, where in found:
        is_joint = run.startswith("j_")
        acts, parser, module = (jo_act, jo_p, "joint") if is_joint else (tr_act, tr_p, "train")
        args = src = None
        params = ttime = None
        for f in sorted(glob.glob(os.path.join(path, "*.json"))):
            try:
                j = json.load(open(f, encoding="utf-8"))
            except ValueError:
                continue
            if not isinstance(j, dict):
                continue
            params = j.get("params", params) if params is None else params
            ttime = j.get("train_time", ttime) if ttime is None else ttime
            if args is None and isinstance(j.get("args"), dict) and "data" in j["args"]:
                args, src = j["args"], "evaluation output " + os.path.basename(f)
        notes = []
        if args is None and missing.get(run):
            args, src = missing[run]["args"], "checkpoint (read for this supplement)"
            params = params if params is not None else missing[run].get("params")
            ttime = ttime if ttime is not None else missing[run].get("train_time")
        if args is None and run in queue:
            args = vars(parser.parse_args(shlex.split(queue[run][1])))
            src = f"queue file {os.path.basename(queue[run][0])} with the defaults of {module}.py"
        if args is None and run in timing:
            args = vars(parser.parse_args(timing[run][1]))
            src = f"r1_timing.py configuration \"{timing[run][0]}\" (r1_timing_configs.json)"
        if args is None:
            src = "args not recoverable"
            notes.append("no checkpoint and no stored arguments")
            if run in queue:
                notes.append(f"queue line in {os.path.basename(queue[run][0])}: {queue[run][1]}")
        elif run in queue and not src.startswith("queue"):
            qa = vars(parser.parse_args(shlex.split(queue[run][1])))
            diff = [f"{k}: queue {fmt_value(qa[k])}, stored {fmt_value(v)}" for k, v in sorted(args.items())
                    if k in qa and k != "name" and qa[k] != v]
            if diff:
                msg = f"the {os.path.basename(queue[run][0])} line differs from the stored arguments ({'; '.join(diff)})"
                notes.append(msg)
                discrepancies.append((run, msg))
        step, t = last_log(os.path.join(path, "log.jsonl"))
        used = []
        for label, _, _, _, pats in MAP:
            if any(re.search(p, run) for p in pats) and label in labels:
                used.append(item_name(label, labels))
        rows.append(dict(run=run, group=group_of(run), outputs=where, script=f"imlewm.{module}", args_source=src,
                         used_in="; ".join(used), last_logged_step="" if step is None else step,
                         params="" if params is None else params,
                         train_time_h="" if ttime is None else f"{ttime / 3600:.2f}",
                         command=command_line(module, acts, args) if args else "", note="; ".join(notes),
                         args=args or {}))
    return rows, discrepancies


def write_runs_csv(rows, path):
    keys = set()
    for r in rows:
        keys |= set(r["args"])
    first = ["data", "method", "m", "steps", "seed", "bs", "lr", "C", "nblocks"]
    argcols = [k for k in first if k in keys] + sorted(keys - set(first) - {"name"})
    fixed = ["run", "group", "outputs", "script", "args_source", "used_in", "last_logged_step", "params",
             "train_time_h", "command", "note"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fixed + argcols)
        for r in rows:
            w.writerow([r[k] for k in fixed] + [fmt_value(r["args"][k]) if k in r["args"] else "" for k in argcols])
    MANIFEST.append(path)
    return argcols


# =============================================================================================== generated documents
LICENSE_TEXT = """MIT License

Copyright (c) 2026 Anonymous authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def predictions_md(repo, labels, cites, design_redacted, design_state, results_present):
    app = read_text(os.path.join(repo, "paper/tmlr/app.tex"))
    out = ["# Predictions recorded before the corresponding runs", ""]
    app_pred = labels.get("app:pred", "")
    out += ["This file collects the predictions that were written down before the runs that test them, together with "
            "their outcomes. Sections 1 to 4 are converted from the appendix of the paper" +
            (f" (Appendix {app_pred})" if app_pred else "") + ", including its statement that these records are the "
            "authors' own notes and are not externally timestamped. Section 5 reproduces the predictions and decision "
            "rules of revision round r1 from its design file, which was frozen before the first r1 run.", ""]

    head = "\\paragraph{Predictions recorded before the runs.}"
    i = app.find(head)
    out += ["## 1. All recorded predictions and their outcomes", ""]
    if i < 0:
        note("app.tex: paragraph 'Predictions recorded before the runs.' not found; section 1 of predictions/recorded.md is empty")
        out += ["(The paragraph was not found in the current manuscript.)", ""]
    else:
        para = tex2md(re.split(r"\n\s*\n", app[i + len(head):], maxsplit=1)[0], labels, cites)
        parts = re.split(r"\((i|ii|iii|iv|v|vi|vii)\) ", para)
        out += [parts[0].strip(), ""]
        summary = None
        items = list(zip(parts[1::2], parts[2::2]))
        if items:
            last = items[-1][1]
            k = last.find("In total")
            if k > 0:
                items[-1] = (items[-1][0], last[:k].strip())
                summary = last[k:].strip()
        for tag, body in items:
            out.append(f"- ({tag}) {body.strip()}")
        out.append("")
        if summary:
            out += [summary, ""]

    body = tex_env_with_label(app, "table", "tab:predfull")
    out += [f"## 2. Memorization index for K = 2 and 16 (Table {labels.get('tab:predfull', 'tab:predfull')})", ""]
    if body is None:
        note("app.tex: table tab:predfull not found")
        out += ["(The table was not found in the current manuscript.)", ""]
    else:
        out += [tex2md(caption_of(body), labels, cites), ""]
        m = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", body, re.S)
        out += [tabular2md(m.group(1), labels, cites), ""]
        head2 = "\\paragraph{Recorded predictions for the $K{=}2,16$ grid.}"
        j = app.find(head2)
        if j >= 0:
            out += [tex2md(re.split(r"\n\s*\n", app[j + len(head2):], maxsplit=1)[0], labels, cites), ""]

    for n, (title, lab, a, b) in enumerate((("Energy-score follow-up runs", "app:es", "(P1)", "(P5)"),
                                            ("Fed-back-state ablation", "app:seq", "(S1)", "(S3)")), start=3):
        where = f", Appendix {labels[lab]}" if lab in labels else ""
        out += [f"## {n}. {title} ({a[1:-1]} to {b[1:-1]}{where})", ""]
        sent = None
        for p in paragraphs(app):
            if a in p and b in p:
                sent = sentence_span(tex2md(p, labels, cites), a, b)
                break
        if sent is None:
            note(f"app.tex: predictions {a}-{b} not found")
            sent = "(Not found in the current manuscript.)"
        out += [sent, ""]

    out += ["## 5. Revision round r1", ""]
    out += [f"The design `revision_r1/DESIGN_r1.md` was frozen before the first r1 run; the SHA-256 of the file as "
            f"frozen is `{DESIGN_SHA256}`. {design_state} Its predictions and decision rules are reproduced below "
            "(sections 1 to 3 of the design; the analyses of section 4 carry no predictions). Outcomes: " +
            ("see `revision_r1/RESULTS_r1.md`." if results_present else
             "see `revision_r1/RESULTS_r1.md` (not yet written when this supplement was built)."), ""]
    m = re.search(r"^## 1\..*?(?=^## 4\.)", design_redacted, re.S | re.M)
    if not m:
        note("DESIGN_r1.md: sections 1-3 not found")
    else:
        sec = re.sub(r"^## (\d)\. ", r"### 5.\1 ", m.group(0), flags=re.M)
        out += [sec.strip(), ""]
    return "\n".join(out) + "\n"


DATASETS = [  # name, used for, command, generator seed, size
    ("bounce_div_k4", "A2-div: bounce, K = 4, randomized appearance (main benchmark)",
     "`python -m imlewm.gen_data --name bounce_div_k4`, then `python -m imlewm.probe --data bounce_div_k4` "
     "(`code/run/chain0.sh` recomputed the statistics after an out-of-memory error)",
     "54 (= 0 + 50 + K)", "1,000 training and 250 test episodes of 40 frames"),
    ("bounce_div_k2", "budget grid, K = 2", "`code/run/genk.sh <gpu> 2` (generator, statistics, position probe)",
     "52", "as bounce_div_k4"),
    ("bounce_div_k8", "budget grid, K = 8", "`code/run/genk.sh <gpu> 8`", "58", "as bounce_div_k4"),
    ("bounce_div_k16", "budget grid, K = 16", "`code/run/genk.sh <gpu> 16`", "66", "as bounce_div_k4"),
    ("bounce_div_k2_p80", "asymmetric modes (mode 0 has probability 0.8)",
     "`python -m imlewm.gen_data --name bounce_div_k2_p80` and probe (`code/run/e14chain.sh`)", "52",
     "as bounce_div_k4"),
    ("bounce_div_k4_arc90", "revision r1, experiment 1: four bounce directions on a 90 degree arc",
     "`python -m imlewm.gen_data --name bounce_div_k4_arc90`, then `python -m imlewm.probe --data bounce_div_k4_arc90`",
     "54", "as bounce_div_k4"),
    ("bounce_k4", "A2-fix: bounce, K = 4, fixed appearance", "`python -m imlewm.gen_data --name bounce_k4`, then "
     "`python -m imlewm.probe --data bounce_k4`", "4 (= 0 + K)", "as bounce_div_k4"),
    ("bounce_k2", "feasibility pilots", "`python -m imlewm.gen_data --name bounce_k2`", "2", "as bounce_div_k4"),
    ("bounce_k8", "feasibility pilots", "`python -m imlewm.gen_data --name bounce_k8`", "8", "as bounce_div_k4"),
    ("push", "branch-risk planning task (push environment with ice and pit) and pilots",
     "`python -m imlewm.gen_data --name push`, then `python -m imlewm.probe --data push`", "100",
     "1,000 training and 250 test episodes of 40 steps"),
    ("bshapes_v3", "A1: Bouncing Shapes on DINOv3 tokens",
     "`python -m imlewm.gen_bshapes --encoder v3`; readouts: `python -m imlewm.a1eval --train_probe`, "
     "`python -m imlewm.heatmap --train` and `python -m imlewm.heatmap --train --noise 0.5 --tag n05`",
     "FlowWM generator configuration, `dataset_seed` 0", "16,384 training clips (5,324 initial conditions) and 512 "
     "evaluation clips (163 initial conditions) of 32 frames at 256 px"),
    ("bshapes", "A1 on DINOv2 tokens (one preparation run)", "`python -m imlewm.gen_bshapes --encoder v2`",
     "as bshapes_v3", "as bshapes_v3"),
    ("pusht_b", "B: stochastic Push-T with a hidden center of gravity",
     "`python -m imlewm.gen_pusht --name pusht_b`; readout: `python -m imlewm.b_eval --train_probe`",
     "1 (training), 2 (test), 3 (multi-future test)",
     "DINO-WM `pusht_noise` episodes 0-2399 (training, 58,398 frames), 17000-17299 (test), 18000-18684 "
     "(multi-future test, 128 starts x 64 futures x 16 steps)"),
    ("pusht_det", "deterministic Push-T control (nominal center of gravity)", "`python -m imlewm.gen_pusht --name pusht_det`",
     "1 (training), 2 (test)", "episodes 4800-7199 (training), 17000-17299 (test)"),
    ("bair", "BAIR robot pushing (`softmotion30_44k`)", "`python -m imlewm.gen_bair --tar <path to bair.tar>`",
     "none (fixed file prefix)", "the first 16 training files (4,096 trajectories) and the test file plus the next "
     "three files (1,024 trajectories) of 30 frames"),
    ("r1_tim_a2", "revision r1, analysis g: random tokens for training-step timing",
     "created by `python -m imlewm.r1_timing run`", "0", "4,096 frames"),
    ("r1_tim_a10", "as r1_tim_a2, 10-dimensional actions", "created by `python -m imlewm.r1_timing run`", "0",
     "4,096 frames"),
]


def data_md(repo, rows):
    src = read_text(os.path.join(repo, "main/code/imlewm/gen_data.py"))
    for s in ("a.seed + K)", "a.seed + 50 + K, diverse=True)", "a.seed + 100)",
              'ap.add_argument("--train_eps", type=int, default=1000)',
              'ap.add_argument("--test_eps", type=int, default=250)', 'ap.add_argument("--T", type=int, default=40)',
              'ap.add_argument("--seed", type=int, default=0)'):
        if s not in src:
            note(f"gen_data.py no longer contains '{s}': check the seeds and sizes in data/README.md")
    psrc = read_text(os.path.join(repo, "main/code/imlewm/gen_pusht.py"))
    for s in ("seed=1)", "seed=2)", "seed=3)", "list(range(18000, 18685))", "base = 0 if stochastic else 4800"):
        if s not in psrc:
            note(f"gen_pusht.py no longer contains '{s}': check data/README.md")
    count = {}
    for r in rows:
        d = r["args"].get("data")
        if d:
            count[d] = count.get(d, 0) + 1
    known = {d[0] for d in DATASETS}
    for d in sorted(set(count) - known):
        note(f"dataset {d} is used by {count[d]} run(s) but is not described in data/README.md")
    out = ["# Datasets", "",
           "Every benchmark is generated by the scripts in `code/imlewm/` and stored with its precomputed tokens under "
           "`$IMLE_DATA/<name>/` (default `~/imle-wm/data/<name>/`; `{split}_tokens.npy` in fp16, `{split}_meta.npz`, "
           "`stats.pt`, readouts such as `probe.pt`). Run the commands from `~/imle-wm/code` after `source "
           "~/imle-wm/env.sh`. The tokens take about 216 GB in total and are not included.", "",
           "## Frozen encoders", "",
           "- DINOv2 ViT-S/14, Hugging Face `facebook/dinov2-small`: frames resized to 224 px (bilinear), ImageNet "
           "normalization, fp16, final-layer patch tokens (16 x 16 x 384). Used for all datasets except A1.",
           "- DINOv3 ViT-S/16, Hugging Face `facebook/dinov3-vits16-pretrain-lvd1689m` (its license must be accepted on "
           "the hub): 256 px frames without resizing, mean of the hidden states of layers 3, 6, 9 and 12, each passed "
           "through the final norm; CLS and register tokens dropped (16 x 16 x 384). Used for A1 (`bshapes_v3`), as in "
           "FlowWM.",
           "- Token statistics (`stats.pt`): per-channel mean and standard deviation, channel PCA and the residual scale, "
           "from 6,000 training frames (seed 0), computed at the end of each generator (`compute_stats`).", "",
           "## Generation commands", "",
           "Bounce generators: `gen_data.py` defaults (1,000 training and 250 test episodes, 40 frames, `--seed 0`); the "
           "generator seed adds K for fixed appearance (`bounce_k<K>`), 50 + K for randomized appearance "
           "(`bounce_div_k<K>`) and 100 for `push`. Names select the variant: `_p<percent>` gives mode 0 that "
           "probability, `_arc<degrees>` sets the angular width of the bounce directions (180 otherwise).", "",
           "| Dataset | Used for | Command | Generator seed | Size | Runs |",
           "|---|---|---|---|---|---|"]
    for name, use, cmd, seed, size in DATASETS:
        out.append(f"| `{name}` | {use} | {cmd} | {seed} | {size} | {count.get(name, 0)} |")
    out += ["", "\"Runs\" counts the rows of `run_args/runs.csv` that train on the dataset.", "",
            "## External sources", "",
            "These are public and are not included; place the code under `~/imle-wm/third_party/` and the data under "
            "`~/imle-wm/data/`.", "",
            "- FlowWM, Flow-World-Models (facebookresearch), revision `c150625`: the Bouncing Shapes generator imported by "
            "`gen_bshapes.py` (`datasets/synthetic_bouncing_shapes/generator.py`), unchanged; also the reference for the "
            "flow-matching recipe and the heatmap readout re-implemented in `heatmap.py`.",
            "- DINO-WM, dino_wm (gaoyuezhou), revision `0a9492f`: the Push-T environment (`env/pusht/pusht_env.py`, loaded "
            "by `pusht_sim.py`) and its `pusht_noise` dataset (OSF project `bmw48`) in `~/imle-wm/data/dinowm/pusht_noise`.",
            "- MeanFlow (haidog-yaqub), revision `97cc518`: reference implementation for the iMF baseline, which is "
            "re-implemented in `train.py` (not imported).",
            "- BAIR robot pushing dataset (Ebert et al., 2017), `softmotion30_44k` tar archive, read by `gen_bair.py`.", ""]
    return "\n".join(out) + "\n"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def readme_md(ctx):
    L = ctx["labels"]

    def order_key(label):
        num, kind = L[label], label.split(":")[0]
        n = int(re.sub(r"\D", "", num) or 0)
        if kind in ("tab", "fig"):
            return (0, num.startswith("S"), 0 if kind == "tab" else 1, n, num)
        return (1, kind != "sec", 0, n, num)

    mapped = [m for m in MAP if m[0] in L]
    main_rows, app_rows, text_rows = [], [], []
    for label, title, how, src, _ in sorted(mapped, key=lambda m: order_key(m[0])):
        row = [f"{item_name(label, L)}: {title}", resolve_refs(how, L), resolve_refs(src, L)]
        if label.split(":")[0] not in ("tab", "fig"):
            text_rows.append(row)
        else:
            (app_rows if L[label].startswith("S") else main_rows).append(row)
    unmapped = sorted(k for k in L if re.match(r"(tab|fig):", k) and k not in {m[0] for m in MAP})
    absent = [m[0] for m in MAP if m[0] not in L]
    for k in unmapped:
        note(f"label {k} ({item_name(k, L)}) is in the manuscript but not in the README map (MAP in supp_build.py)")
    for k in absent:
        if k != "fig:samples":
            note(f"label {k} of the README map is not in the current manuscript build")

    src_counts = {}
    for r in ctx["rows"]:
        key = r["args_source"]
        key = "r1_timing.py configuration (analysis g)" if key.startswith("r1_timing") else key
        key = "evaluation output (`args` field of `eval*.json`)" if key.startswith("evaluation output") else key
        key = "queue line with the defaults of the training script" if key.startswith("queue file") else key
        src_counts[key] = src_counts.get(key, 0) + 1
    unrec = [r["run"] for r in ctx["rows"] if r["args_source"] == "args not recoverable"]
    n_orig = sum(1 for r in ctx["rows"] if r["outputs"].startswith("results/"))
    n_r1 = len(ctx["rows"]) - n_orig

    S = []
    S.append(f"# Supplementary material: {ctx['title']}")
    S.append("")
    S.append("Anonymous supplementary material for the TMLR submission of the same title. It contains the code that "
             "produced the results in the paper, the arguments of the training runs (" +
             (f"all {len(ctx['rows'])} runs" if not unrec else f"{len(ctx['rows']) - len(unrec)} of the "
              f"{len(ctx['rows'])} logged runs; the missing one{'s are' if len(unrec) > 1 else ' is a'} failed "
              f"pilot{'s' if len(unrec) > 1 else ''}") +
             "), the raw per-run outputs behind the tables and figures, scripts that regenerate the figures and most "
             "table values from those outputs on a CPU in seconds, the predictions recorded before the corresponding "
             "runs, and the frozen design, job lists and outputs of the revision round r1. Model checkpoints (about "
             "9.5 GB) and token caches (about 216 GB) exceed the size limit of the supplement; they will be released "
             "with the paper.")
    S.append("")
    S.append("## Contents")
    S.append("")
    S.append(md_table(["Path", "Contents"], [
        ["`requirements.txt`", "Pinned package list of the environment of the original runs"],
        ["`LICENSE`", "MIT license of the code in this supplement"],
        ["`code/imlewm/`", "The Python package that was run: data generation, models, training (`train.py`, `joint.py`), "
                           "evaluation, memorization checks, planning, readouts, and the revision-r1 analyses (`r1_*.py`)"],
        ["`code/run/`", "Job lists exactly as queued (`q*.txt`, one line = arguments of `imlewm.train`), the queue runner "
                        "`runq.sh`, the chain scripts that ran the evaluations after training (`*chain.sh`, "
                        "`evalwatch.sh`), `env.sh`, and three summary scripts that ran on the training server"],
        ["`code/diag/`", "One-off diagnostic scripts"],
        ["`run_args/runs.csv`", f"One row per training run ({len(ctx['rows'])} runs) with its full arguments, their source "
                                "and the paper items that use it"],
        ["`analysis/`", "`make_figures.py` (figures and most table values), `summarize_analyses.py` and "
                        "`summarize_r1.py` (revision r1), `make_fig_samples.py` (decoded-sample figure)"],
        ["`results/runs/`", f"Per-run outputs (JSON / JSONL) of the {n_orig} logged runs of the original study, plus "
                            "planning (`plan/`, `e5/`) and checks (`check/`)"],
        ["`results/check/`, `results/e15_toy/`, `results/feasibility/`", "Checks of the pilot models, the CPU toy "
                                                                        "experiment, the feasibility-study summary and figures"],
        ["`results/logs/`", "Standard output of every job of the original study (home-directory paths replaced by `~`)"],
        ["`revision_r1/`", f"Revision round r1: frozen design, results, job lists and raw outputs ({n_r1} logged runs)"],
        ["`predictions/recorded.md`", "Predictions recorded before the corresponding runs, with their outcomes"],
        ["`data/README.md`", "How every dataset is generated: command, generator seed, size, external sources"],
    ]))
    S.append("")
    S.append("Comments and docstrings in the code sometimes cite internal notes of the project (for example "
             "`E15_report.md`, `theory_m_law.md`, `novelty_analysis.md`, `REPORT.md`); these notes are not part of the "
             "supplement. Paths in the revision-r1 documents refer to the project layout: `main/code/` is `code/` here, "
             "`main/results/` is `results/`, `main/revision/` is `revision_r1/`, `main/revision/summarize_r1.py` and "
             "`main/revision/an_analyses/summarize.py` are `analysis/summarize_r1.py` and "
             "`analysis/summarize_analyses.py`, and `paper/build/make_figures.py` is `analysis/make_figures.py`.")
    S.append("")
    S.append("## Environment")
    S.append("")
    S.append("The original runs used two NVIDIA RTX A5000 GPUs (24 GB) under Ubuntu 22.04 with Python 3.10.12 and "
             "PyTorch 2.6.0 (CUDA 12.4). `requirements.txt` is the frozen package list of that environment "
             "(torch 2.6.0+cu124, transformers 4.57.6, numpy 2.2.6, scipy 1.15.3, matplotlib 3.10.9, pymunk 6.2.1, "
             "gym 0.26.2). The code expects the layout `~/imle-wm/{code,data,runs,logs,third_party}`; module paths are "
             "fixed relative to it (for example `RUNS = ~/imle-wm/runs`) and the data root can be moved with "
             "`IMLE_DATA`.")
    S.append("")
    S.append("```bash")
    S.append("mkdir -p ~/imle-wm/{code,data,runs,logs,third_party}")
    S.append("cp -r code/imlewm ~/imle-wm/code/ && cp code/run/* ~/imle-wm/code/ && cp requirements.txt ~/imle-wm/")
    S.append("cd ~/imle-wm && uv python install 3.10.12 && uv venv --python 3.10.12 .venv")
    S.append("uv pip install --python .venv/bin/python -r requirements.txt \\")
    S.append("    --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match")
    S.append("source ~/imle-wm/env.sh      # sets IMLE, IMLE_DATA, HF_HOME, PYTHONPATH and activates ~/imle-wm/.venv")
    S.append("```")
    S.append("")
    S.append("The DINOv2 and DINOv3 weights are downloaded from the Hugging Face hub on first use. The scripts in "
             "`analysis/` need only Python 3 with numpy and matplotlib. Datasets: see `data/README.md`.")
    S.append("")
    S.append("## Training and evaluating one run")
    S.append("")
    ex = next((r for r in ctx["rows"] if r["run"] == "e1_a2_imle16_s0"), None)
    S.append("The arguments of every run are in `run_args/runs.csv`; its `command` column re-creates the run with the "
             "current code. For example, the seed-0 IMLE-WM model with m = 16 of the main A2-div comparison:")
    S.append("")
    S.append("```bash")
    S.append("cd ~/imle-wm/code")
    if ex:
        S.append(ex["command"])
    S.append("python -m imlewm.evaluate --run e1_a2_imle16_s0          # -> ~/imle-wm/runs/e1_a2_imle16_s0/eval.json")
    S.append("python -m imlewm.e3 --run e1_a2_imle16_s0 --hmax 32 --nfe 1   # rollouts from real contexts -> e3.json")
    S.append("```")
    S.append("")
    if ctx.get("smoke"):
        S.append(ctx["smoke"])
        S.append("")
    S.append("Evaluation entry points by benchmark: `imlewm.evaluate` (bounce and push data), `imlewm.a1eval` (A1, probe "
             "readout; `A1_READOUT=heatmap A1_DATA=bshapes_v3` for FlowWM's heatmap readout), `imlewm.joint --eval` (A1 "
             "joint protocol), `imlewm.b_eval` (stochastic Push-T; `--perstart` for per-start errors), "
             "`imlewm.bair_eval` (BAIR), `imlewm.e3` (rollouts), `check.memo_model` (mode shares, see "
             "`code/run/e14chain.sh`), `imlewm.plan` and `imlewm.bplan` (planning). Flow-matching models are evaluated "
             "with `--nfe 50` and `--nfe 2`, iMF and Shortcut with `--nfe 1`. During the study the job lists "
             "`code/run/q*.txt` were consumed by `runq.sh <gpu> <queue file>`, and `evalwatch.sh` and the chain scripts "
             "ran the evaluations; they record which evaluation each run received.")
    S.append("")
    S.append("## Regenerating the figures and the table values")
    S.append("")
    S.append("From the root of this supplement (CPU only, seconds):")
    S.append("")
    S.append("```bash")
    S.append("python3 analysis/make_figures.py results/runs results out/   # figures as PDF; prints every table value")
    S.append("python3 analysis/summarize_analyses.py                        # revision r1, analyses a-h")
    S.append("python3 analysis/summarize_r1.py                              # revision r1, experiments 1-3 and decision rules")
    S.append("python3 analysis/make_fig_samples.py --out out/fig_samples.pdf")
    S.append("```")
    S.append("")
    S.append("`make_figures.py` prints mean ± standard deviation over seeds with the 1/n normalization (`np.std`); the "
             "revision-r1 summaries print the sample standard deviation (n - 1) together with the individual seeds." +
             (" `analysis/make_figures.py` is an unmodified copy of the script that drew the manuscript figures."
              if ctx["mf_identical"] else ""))
    S.append("")
    S.append("## Map from the paper to scripts, runs and files")
    S.append("")
    S.append("Item numbers follow the manuscript build this supplement was made with. Run names use `{a,b}` for "
             "alternatives and `*` for all seeds; file names are relative to `results/runs/<run>/` unless a folder is "
             "given; `x.y` names a JSON field. Values printed by `analysis/make_figures.py` appear under the printout "
             "heading given in the second column.")
    S.append("")
    S.append("### Main text")
    S.append("")
    S.append(md_table(["Item", "How", "Runs, files and fields"], main_rows))
    S.append("")
    S.append("### Appendix")
    S.append("")
    S.append(md_table(["Item", "How", "Runs, files and fields"], app_rows))
    S.append("")
    if text_rows:
        S.append("### Results quoted in the text (selection)")
        S.append("")
        S.append(md_table(["Item", "How", "Runs, files and fields"], text_rows))
        S.append("")
    if unmapped:
        S.append("Items of the manuscript not covered by this map: " + ", ".join(item_name(k, L) for k in unmapped) + ".")
        S.append("")
    S.append("Numbers quoted only in the text come from the same files. A few constants are written into "
             "`make_figures.py` (the recorded predictions `PRED` and the true-future spread of the fed-back-state "
             "ablation). The diagnostic scripts in `code/diag/` and the summary scripts `code/run/e15_summary.py`, "
             "`e16_summary.py` and `e16_curves.py` print to standard output; the diagnostics that were re-run with saved "
             "outputs in revision round r1 are under `revision_r1/an_analyses/raw/an.1/out/f_diag/`.")
    S.append("")
    S.append("## Run arguments (`run_args/runs.csv`)")
    S.append("")
    S.append(f"One row per run with a training log: {n_orig} runs of the original study (`results/runs/`) and {n_r1} "
             "runs of revision round r1 (`revision_r1/*/raw/*/runs/`). Columns: `run`; `group` (experiment block); "
             "`outputs` (folder in this supplement); `script` (`imlewm.train` or `imlewm.joint`); `args_source`; "
             "`used_in` (paper items, from the map above); `last_logged_step` (last step in `log.jsonl`); `params`; "
             "`train_time_h` (wall-clock hours, several runs shared a GPU); `command` (the stored arguments that differ "
             "from the defaults of the current script, plus `--data`, `--method` and `--name`); `note`; then one column "
             "per argument with the stored value. A blank argument cell means that the option did not exist yet when "
             "the run was made, so that the `command` column relies on the current default. The pilot runs of the "
             "feasibility study predate several options (for example `--wide_head`, `--z_from`, `--beta2`), and their "
             "commands may not reproduce them exactly.")
    S.append("")
    S.append("Sources of the arguments:")
    S.append("")
    for k in sorted(src_counts, key=lambda k: -src_counts[k]):
        S.append(f"- {k}: {src_counts[k]} run{'s' if src_counts[k] != 1 else ''}")
    S.append("")
    S.append("The training script stores its full argument list in each checkpoint; the evaluation scripts copy it into "
             "most evaluation outputs. For the runs whose outputs do not carry it, the arguments were read from the "
             "original checkpoints for this supplement." +
             (f" No arguments can be recovered for {len(unrec)} run{'s' if len(unrec) != 1 else ''}, which "
              f"ha{'ve' if len(unrec) != 1 else 's'} neither stored arguments nor a checkpoint: " +
              ", ".join(f"`{r}`" for r in unrec) + "." if unrec else ""))
    S.append("")
    if ctx["discrepancies"]:
        S.append("The job lists in `code/run/q*.txt` are the files as queued, but a few lines do not reproduce the stored "
                 "arguments with the current defaults; the stored arguments are authoritative. A difference in an "
                 "option that the run's method does not read (such as `mf_p`, used only by MeanFlow models) comes from "
                 "a later change of that option's default.")
        S.append("")
        for run, msg in ctx["discrepancies"]:
            S.append(f"- `{run}`: {msg}")
        S.append("")
    S.append("Some runs were launched outside the job lists: the joint-protocol runs `j_*` by `code/run/jchain.sh`, the "
             "asymmetric-mode runs `e14_asym_*` by lines that `code/run/e14chain.sh` appended to a queue, and several "
             "preparation and diagnostic runs by hand; their arguments come from their checkpoints or evaluation "
             "outputs like those of all other runs.")
    S.append("")
    S.append("## Revision round r1")
    S.append("")
    S.append("Revision round r1 adds three small training experiments and eight analyses of existing checkpoints "
             "(a to h). Its design, `revision_r1/DESIGN_r1.md`, states the questions, data, runs, predictions, decision "
             "rules and consequences for the paper, and was frozen before the first r1 job started. The SHA-256 of the "
             f"design file as frozen is `{DESIGN_SHA256}`.")
    S.append("")
    S.append(ctx["design_state"])
    S.append("")
    S.append("- Runs: `revision_r1/queues/*.txt` (one line = arguments of `imlewm.train`); outputs under "
             "`revision_r1/<group>/raw/<machine>/runs/r1_*/` (`log.jsonl`, `eval*.json`, `e3*.json`) and "
             "`.../out/<check>/`. All r1 runs are listed in `run_args/runs.csv`.")
    S.append("- Analyses a to h: `code/imlewm/r1_*.py`; the commands as run are in `revision_r1/an_analyses/jobs/`; the "
             "outputs are under `revision_r1/an_analyses/raw/an.1/out/<analysis>/` (paths under `/workspace` in these "
             "files are the working folders of the GPU machine).")
    S.append("- Code: the only change to the original code for r1 is the arc option of experiment 1 (design section 0); "
             "every existing dataset is generated as before.")
    S.append("- Summaries: `python3 analysis/summarize_r1.py` (experiments 1 to 3 with the decision rules) and "
             "`python3 analysis/summarize_analyses.py` (analyses a to h); both read only the JSON files in this folder.")
    S.append("- Predictions: `predictions/recorded.md`, section 5. Outcomes and decisions: " +
             ("`revision_r1/RESULTS_r1.md`." if ctx["results_present"] else
              "`revision_r1/RESULTS_r1.md` (not yet written when this supplement was built)."))
    S.append("")
    S.append("## Per-run output files")
    S.append("")
    S.append(md_table(["File", "Contents"], [
        ["`log.jsonl`", "training log every 200 steps: `step`, `loss`, `t` (seconds since start) and method diagnostics; "
                        "on bounce data every 2,000 steps also `memo_train`, `memo_val` (memorization index on 200 "
                        "training / held-out contexts, 16 samples) and `cov_val`"],
        ["`eval.json`, `eval_nfe<k>.json`", "evaluation (flow models at k integration steps): `args`, `params`, "
                                            "`train_time`, `latency_ms`; bounce `stoch` (coverage `cov_pos`, precision "
                                            "`prec_pos`, kNN `recall_knn`, ...) and `det` (`pos_err`); A1 `model`, "
                                            "`copy_last`, `oracle` (`f1_err`, `precision_err`, `recall_err`); Push-T "
                                            "`mf_h8`, `mf` (16-step multi-future test, `mode_cov`); BAIR per-horizon "
                                            "arrays and distribution metrics"],
        ["`eval_hm*.json`", "A1 with FlowWM's heatmap readout (`hmn05`: readout trained with feature noise 0.5)"],
        ["`eval_ps*.json`", "Push-T errors per starting state (`per_start`)"],
        ["`eval_ckpt_<step>*.json`", "evaluation of an intermediate checkpoint (learning curves)"],
        ["`e3*.json`", "rollouts from 128 real contexts with 16 samples: per-horizon `manifold`, `best_err`, `spread`"],
        ["`modeshare*.json`", "mode shares with 400 contexts and 64 samples (`train` / `test`: `share0`, `data_freq0`, "
                              "`cov`, `realized_share`)"],
        ["`results/runs/plan/`, `results/runs/e5/`", "branch-risk planning (one list per model and lambda) and "
                                                     "Push-T planning"],
        ["`results/runs/check/`, `results/check/`", "memorization and validity checks of pilot and preparation models "
                                                    "(`imlewm.check`)"],
    ]))
    S.append("")
    S.append("Folders whose names start with `_` (and `plan/_invalid/`) hold failed, test or superseded outputs kept for "
             "completeness; `_old_b_eval/` was computed on an earlier Push-T multi-future test set.")
    S.append("")
    S.append(f"Training compute: the {n_orig} logged runs of the original study sum to {ctx['gpu_hours']:.0f} GPU-hours "
             "(last `t` of each `log.jsonl`), including pilot and discarded runs; some runs shared a GPU.")
    S.append("")
    S.append("## Not included")
    S.append("")
    S.append("- Model checkpoints (`ckpt.pt`, about 9.5 GB) and token caches (about 216 GB): they will be released with "
             "the paper. The values in the map above can be checked from the outputs included here; re-running a "
             "training or an evaluation needs the datasets, and re-running an evaluation also the checkpoints.")
    S.append("- Third-party code and data (`data/README.md`, \"External sources\").")
    S.append("- Internal notes of the project.")
    S.append("")
    S.append("## License")
    S.append("")
    S.append("The code in this supplement is released under the MIT license (`LICENSE`). Third-party code and data keep "
             "their own licenses.")
    S.append("")
    return "\n".join(S)


# =============================================================================================== main
def git(repo, *args):
    try:
        return subprocess.run(["git", "-C", repo] + list(args), capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def design_check(repo, text):
    """Compare DESIGN_r1.md with the frozen version (found in the project history by its SHA-256)."""
    cur = hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()
    if cur == DESIGN_SHA256:
        return "unchanged", None
    frozen = None
    log = git(repo, "log", "--format=%H", "--", "main/revision/DESIGN_r1.md")
    for c in (log or b"").decode().split():
        blob = git(repo, "show", f"{c}:main/revision/DESIGN_r1.md")
        if blob is not None and hashlib.sha256(blob).hexdigest() == DESIGN_SHA256:
            frozen = blob.decode("utf-8", "surrogateescape")
            break
    if frozen is None:
        note("DESIGN_r1.md differs from the frozen version, which was not found in the project history")
        return "unverified", None
    cut = frozen.find("## 7. Deviations after the freeze")
    if cut < 0 or text[:cut] != frozen[:cut]:
        die("DESIGN_r1.md: the frozen part above section 7 has changed")
    return "amended", frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--notes", default=None)
    a = ap.parse_args()
    repo, out = os.path.abspath(a.repo), os.path.abspath(a.out)
    if os.path.exists(out) and os.listdir(out):
        die(f"{out} is not empty")
    os.makedirs(out, exist_ok=True)
    J = lambda *p: os.path.join(repo, *p)

    tex = J("paper/tmlr")
    labels, cites = read_aux(os.path.join(tex, "main.aux"))
    srcs = [os.path.join(tex, f) for f in os.listdir(tex) if f.endswith(".tex")]
    if os.path.exists(os.path.join(tex, "main.aux")) and max(os.path.getmtime(s) for s in srcs) > os.path.getmtime(
            os.path.join(tex, "main.aux")):
        note("a .tex file is newer than main.aux: rebuild the manuscript so that table and figure numbers are current")
    mt = read_text(os.path.join(tex, "main.tex"))
    m = re.search(r"\\title\{", mt)
    title = tex2md(braced(mt, m.end() - 1)[0].replace("\\\\", " "), labels, cites).replace("$", "")

    # ---- code
    n = copy_tree(J("main/code/imlewm"), os.path.join(out, "code/imlewm"), {".py", ".json"})
    n += copy_tree(J("main/code/diag"), os.path.join(out, "code/diag"), {".py"})
    for f in sorted(os.listdir(J("main/code"))):
        p = J("main/code", f)
        if os.path.isfile(p) and not f.startswith(".") and f != "requirements.txt" and os.path.splitext(f)[1] in (".sh", ".txt", ".py"):
            copy_file(p, os.path.join(out, "code/run", f))
            n += 1
    copy_file(J("main/code/requirements.txt"), os.path.join(out, "requirements.txt"))
    print(f"code: {n + 1} files")

    # ---- analysis scripts
    copy_file(J("paper/build/make_figures.py"), os.path.join(out, "analysis/make_figures.py"))
    mf_identical = open(J("paper/build/make_figures.py"), "rb").read() == open(
        os.path.join(out, "analysis/make_figures.py"), "rb").read()
    copy_file(J("main/revision/summarize_r1.py"), os.path.join(out, "analysis/summarize_r1.py"), fix_summarize_r1)
    copy_file(J("main/revision/an_analyses/summarize.py"), os.path.join(out, "analysis/summarize_analyses.py"),
              fix_summarize_analyses)
    fs = [p for p in (J("paper/build/make_fig_samples.py"), J("paper/tmlr/make_fig_samples.py")) if os.path.exists(p)]
    if fs:
        copy_file(fs[0], os.path.join(out, "analysis/make_fig_samples.py"), analysis_text)
    else:
        note("make_fig_samples.py not found in paper/build/ or paper/tmlr/: analysis/make_fig_samples.py is missing")

    # ---- results of the original study
    n = copy_tree(J("main/results/runs"), os.path.join(out, "results/runs"), RESULT_EXT)
    for d in ("check", "e15_toy", "feasibility"):
        n += copy_tree(J("main/results", d), os.path.join(out, "results", d), RESULT_EXT)
    n += copy_tree(J("main/results/logs"), os.path.join(out, "results/logs"), {".log", ".txt"})
    print(f"results: {n} files")

    # ---- revision r1
    rev = J("main/revision")
    dtext = read_text(os.path.join(rev, "DESIGN_r1.md"))
    state, _ = design_check(repo, dtext)
    red, n_red = redact_design(dtext)
    write_text(os.path.join(out, "revision_r1/DESIGN_r1.md"), scrub(red))
    MANIFEST.append(os.path.join(out, "revision_r1/DESIGN_r1.md"))
    red_sha = hashlib.sha256(scrub(red).encode("utf-8", "surrogateescape")).hexdigest()
    design_state = {
        "unchanged": "The file has not changed since it was frozen.",
        "amended": "Since the freeze, text has been added only to its section 7 (\"Deviations after the freeze\"), as "
                   "the design allows; the part above section 7 was checked against the frozen file when this "
                   "supplement was built.",
        "unverified": "The current file differs from the frozen one.",
    }[state]
    design_state += (f" The copy included here is redacted in {n_red} places that name the compute provider, the tool "
                     "names, the account budget, the backup location and the path of an internal review report (each "
                     "marked \"[redacted: ...]\"), so its own SHA-256 is "
                     f"`{red_sha}`; the unredacted file will be released with the code.")
    results_present = os.path.exists(os.path.join(rev, "RESULTS_r1.md"))
    if results_present:
        copy_file(os.path.join(rev, "RESULTS_r1.md"), os.path.join(out, "revision_r1/RESULTS_r1.md"))
    else:
        note("main/revision/RESULTS_r1.md does not exist yet: add it and rebuild")
    n = copy_tree(os.path.join(rev, "queues"), os.path.join(out, "revision_r1/queues"), {".txt"})
    for raw in sorted(glob.glob(os.path.join(rev, "*", "raw"))):
        rel = os.path.relpath(raw, rev)
        n += copy_tree(raw, os.path.join(out, "revision_r1", rel), RAW_EXT)
    n += copy_tree(os.path.join(rev, "an_analyses/jobs"), os.path.join(out, "revision_r1/an_analyses/jobs"),
                   {".sh", ".py"})
    print(f"revision_r1: {n + 2} files")

    # ---- run arguments
    rows, disc = build_runs(repo, labels)
    write_runs_csv(rows, os.path.join(out, "run_args/runs.csv"))
    gpu_hours = 0.0
    for r in rows:
        if r["outputs"].startswith("results/"):
            _, t = last_log(os.path.join(repo, "main", r["outputs"], "log.jsonl"))
            gpu_hours += (t or 0) / 3600
    print(f"runs.csv: {len(rows)} runs, {sum(1 for r in rows if r['args_source'] == 'args not recoverable')} without "
          f"arguments, {len(disc)} queue-line discrepancies")

    # ---- smoke-test values of the r1 re-evaluation (if present)
    smoke = None
    sm = glob.glob(os.path.join(rev, "an_analyses/raw/an.1/out/smoke/eval_*e1_a2_imle16_s0.json"))
    if sm:
        s = json.load(open(sm[0], encoding="utf-8"))["stoch"]
        o = json.load(open(J("main/results/runs/e1_a2_imle16_s0/eval.json"), encoding="utf-8"))["stoch"]
        smoke = (f"Re-evaluating the stored checkpoint of this run on another GPU in revision round r1 gave coverage "
                 f"{s['cov_pos']:.4f} and precision {s['prec_pos']:.4f}, against the stored {o['cov_pos']:.4f} and "
                 f"{o['prec_pos']:.4f} (`revision_r1/an_analyses/raw/an.1/out/smoke/`); sampling on a different GPU "
                 "is not bit-identical.")

    # ---- generated documents
    write_text(os.path.join(out, "LICENSE"), LICENSE_TEXT)
    write_text(os.path.join(out, "predictions/recorded.md"),
               scrub(predictions_md(repo, labels, cites, red, design_state, results_present)))
    write_text(os.path.join(out, "data/README.md"), scrub(data_md(repo, rows)))
    ctx = dict(labels=labels, title=title, rows=rows, discrepancies=disc, design_state=design_state,
               results_present=results_present, gpu_hours=gpu_hours, smoke=smoke, mf_identical=mf_identical)
    write_text(os.path.join(out, "README.md"), scrub(readme_md(ctx)))

    # ---- excluded internal documents must not be present
    bad = [p for p in glob.glob(os.path.join(out, "**", "*"), recursive=True)
           if os.path.basename(p) in ("LOG.md", "WORKER_RULES.md", "PLAN.md", "REPORT.md", "COMMANDS.md", "CITATION.cff")]
    if bad:
        die("internal documents in the output: " + ", ".join(bad))
    if a.notes:
        with open(a.notes, "w") as f:
            f.write("\n".join(NOTES) + ("\n" if NOTES else ""))
    print(f"done: {len(MANIFEST)} copied or generated files, {len(NOTES)} notes")


if __name__ == "__main__":
    main()

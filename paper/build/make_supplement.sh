#!/bin/bash
# Build the anonymized supplementary material of the TMLR submission from the current sources.
#
# usage: paper/build/make_supplement.sh          (from any folder; re-runnable, nothing is edited by hand)
#   SKIP_SMOKE=1 paper/build/make_supplement.sh  skips the smoke tests (they need numpy and matplotlib)
#   PYTHON=/path/to/python3                      interpreter for the helper and the smoke tests
#
# Output (ignored by git): paper/supplement_build/
#   supplement/            the folder that is zipped (replaced only after every check has passed)
#   supplement_anon.zip    the archive to upload (deterministic: sorted entries, fixed mtimes, zip -X)
#   build_notes.txt        warnings of the last attempt (stale main.aux, unmapped items, text not copied, ...)
#   smoke/                 printouts of the smoke tests of the last attempt (supplement copy vs. project version)
#   staging/               work folder; kept after a failed attempt for inspection, removed after a successful one
# A failed attempt leaves the previous supplement/ and supplement_anon.zip untouched.
#
# Steps: paper/build/supp_build.py copies and scrubs the sources and writes README.md, LICENSE, data/README.md,
# predictions/recorded.md and run_args/runs.csv; then this script runs the smoke tests, removes OS and Python
# caches, fails if an identifying string or unfinished manuscript text (\TBD) is left, fixes permissions and
# mtimes, zips, checks the archive and only then publishes the folder and the zip.
# The manuscript sources are only read. Before the final build, rebuild the manuscript: table and figure numbers
# come from paper/tmlr/main.aux.
set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$REPO/paper/supplement_build"
STAGE="$OUT/staging"
SUPP="$STAGE/supplement"
ZIP_NEW="$STAGE/supplement_anon.zip"
ZIP="$OUT/supplement_anon.zip"
PY="${PYTHON:-python3}"
FIXED_MTIME="202611010000.00"     # touch -t, read in UTC: every file and folder gets 2026-11-01 00:00 UTC
LIMIT=100000000                   # TMLR supplementary limit, 100 MB
PAT='yonghwi|thddydgnl|kmlee|cbnu|chungbuk|kiis|mvf|racedreamer|runpod|/home/|/Users/|gdrive|github\.com/thddydgnl'

fail() {
  echo "FAILED: $*" >&2
  echo "The previous supplement folder and zip in $OUT are unchanged; the failed attempt is in $STAGE." >&2
  exit 1
}
trap 'echo "FAILED at line $LINENO. The previous supplement folder and zip in $OUT are unchanged; the failed attempt is in $STAGE." >&2' ERR

echo "== 1. assemble (in $STAGE)"
mkdir -p "$OUT"
rm -rf "$STAGE" "$OUT/smoke" "$OUT/grep_matches.txt"
mkdir -p "$STAGE"
"$PY" -I "$HERE/supp_build.py" --repo "$REPO" --out "$SUPP" --notes "$OUT/build_notes.txt"

echo "== 2. smoke tests"
if [ "${SKIP_SMOKE:-0}" = 1 ]; then
  echo "skipped (SKIP_SMOKE=1)"
else
  T="$OUT/smoke"; mkdir -p "$T/supp" "$T/repo"
  # (a) figure script: the printout of all table values must be identical from the supplement and from the project
  ( cd "$SUPP" && "$PY" analysis/make_figures.py results/runs results "$T/supp" > "$T/make_figures.txt" 2> "$T/make_figures.err" ) \
    || { cat "$T/make_figures.err" >&2; fail "analysis/make_figures.py"; }
  ( cd "$REPO" && "$PY" paper/build/make_figures.py main/results/runs main/results "$T/repo" > "$T/make_figures_repo.txt" 2> /dev/null ) \
    || fail "paper/build/make_figures.py (project version)"
  cmp -s "$T/make_figures.txt" "$T/make_figures_repo.txt" \
    || { diff "$T/make_figures.txt" "$T/make_figures_repo.txt" | head -20 >&2; fail "make_figures.py printout differs between supplement and project"; }
  ls "$T/repo" | sort > "$T/figs_repo.txt"; ls "$T/supp" | sort > "$T/figs_supp.txt"
  cmp -s "$T/figs_repo.txt" "$T/figs_supp.txt" || { diff "$T/figs_supp.txt" "$T/figs_repo.txt" >&2; fail "make_figures.py writes different figure files"; }
  echo "make_figures.py: $(wc -l < "$T/figs_supp.txt" | tr -d ' ') figure files, printout identical to the project version ($(wc -l < "$T/make_figures.txt" | tr -d ' ') lines)"
  # (b) revision-r1 summaries and the bootstrap: same output and exit status as the project versions
  for spec in "analysis/summarize_r1.py::main/revision/summarize_r1.py:" \
              "analysis/summarize_analyses.py::main/revision/an_analyses/summarize.py:" \
              "analysis/bootstrap_pusht.py:results/runs:paper/build/bootstrap_pusht.py:main/results/runs"; do
    IFS=: read -r s sa r ra <<< "$spec"; b=$(basename "$s" .py)
    rs=0; ( cd "$SUPP" && "$PY" "$s" $sa > "$T/$b.txt" 2> "$T/$b.err" ) || rs=$?
    rr=0; ( cd "$REPO" && "$PY" "$r" $ra > "$T/${b}_repo.txt" 2> /dev/null ) || rr=$?
    [ $rs -eq $rr ] || fail "$s exit status $rs, project version $rr"
    cmp -s "$T/$b.txt" "$T/${b}_repo.txt" || { diff "$T/$b.txt" "$T/${b}_repo.txt" | head -20 >&2; fail "$s output differs from the project version"; }
    if [ $rs -eq 0 ]; then echo "$s: output identical to the project version"
    else echo "WARNING: $s exits with $rs (the project version too; see $T/$b.err)"; fi
  done
  # (c) decoded-sample figure
  if [ -f "$SUPP/analysis/make_fig_samples.py" ]; then
    ( cd "$SUPP" && "$PY" analysis/make_fig_samples.py --out "$T/supp/fig_samples.pdf" > "$T/make_fig_samples.txt" 2> "$T/make_fig_samples.err" ) \
      || { cat "$T/make_fig_samples.err" >&2; fail "analysis/make_fig_samples.py"; }
    FS=paper/build/make_fig_samples.py; [ -f "$REPO/$FS" ] || FS=paper/tmlr/make_fig_samples.py
    ( cd "$REPO" && "$PY" "$FS" --out "$T/repo/fig_samples.pdf" > "$T/make_fig_samples_repo.txt" 2> /dev/null ) \
      || fail "$FS (project version)"
    sed "s#$T/supp/##" "$T/make_fig_samples.txt" > "$T/a.txt"; sed "s#$T/repo/##" "$T/make_fig_samples_repo.txt" > "$T/b.txt"
    cmp -s "$T/a.txt" "$T/b.txt" || fail "make_fig_samples.py output differs from the project version"
    rm -f "$T/a.txt" "$T/b.txt"
    echo "make_fig_samples.py: output identical to the project version"
  fi
  # (d) syntax of every Python and shell file after the path edits
  find "$SUPP" -name '*.py' -print0 | xargs -0 "$PY" -I -c '
import ast, sys
for f in sys.argv[1:]:
    with open(f, encoding="utf-8") as fh:
        ast.parse(fh.read(), f)
print(len(sys.argv) - 1, "Python files parse")'
  n=0; while IFS= read -r -d '' f; do bash -n "$f" || fail "bash -n $f"; n=$((n + 1)); done < <(find "$SUPP" -name '*.sh' -print0)
  echo "$n shell scripts pass bash -n"
fi

echo "== 3. remove OS and Python caches"
find "$SUPP" -depth \( -name __pycache__ -o -name __MACOSX \) -type d -exec rm -rf {} +
find "$SUPP" \( -name .DS_Store -o -name '._*' -o -name '*.pyc' \) -type f -delete

echo "== 4. anonymity and completeness check"
rc=0; grep -rIiE "$PAT" "$SUPP" > "$OUT/grep_matches.txt" || rc=$?
if [ $rc -eq 0 ]; then cat "$OUT/grep_matches.txt" >&2; fail "identifying strings in the supplement (listed above, also in $OUT/grep_matches.txt)"; fi
[ $rc -eq 1 ] || fail "grep exited with status $rc"
rm -f "$OUT/grep_matches.txt"
names=$(cd "$SUPP" && find . | grep -iE "$PAT" || true)
[ -z "$names" ] || { echo "$names" >&2; fail "identifying file or folder names"; }
"$PY" -I - "$SUPP" "$PAT" <<'EOF' || fail "binary scan, Hangul or unfinished-text check (listed above)"
import os, re, sys
root, pat = sys.argv[1], re.compile(sys.argv[2].encode(), re.I)
text_ext = {".py", ".sh", ".txt", ".md", ".json", ".jsonl", ".log", ".csv"}
hangul = re.compile("[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")
weak = re.compile(r"\bKorean?\b|\bKST\b|Asia/Seoul|\+09:?00\b")
tbd = re.compile(r"\\TBD\b|\[TBD:")
latex = re.compile(r"\\(?:ref|cite[pt]?|textbf|emph|imlewm|nom|ms|best|frac|mathrm|paragraph)\b")
bad, warn, nbin = [], [], 0
for d, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(d, f)
        with open(p, "rb") as fh:
            b = fh.read()
        rel = os.path.relpath(p, root)
        if os.path.splitext(f)[1].lower() not in text_ext:   # grep -I skips binary files: scan their bytes
            nbin += 1
            if pat.search(b):
                bad.append(f"{rel}: identifying bytes in a binary file")
            continue
        t = b.decode("utf-8", "replace")
        if hangul.search(t):
            bad.append(f"{rel}: Hangul text (internal notes must not be included)")
        if tbd.search(t):
            bad.append(f"{rel}: unfinished manuscript text (\\TBD)")
        for m in weak.finditer(t):
            warn.append(f"{rel}: '{t[max(0, m.start() - 40):m.end() + 40]}'".replace("\n", " "))
for gen in ("README.md", "predictions/recorded.md", "data/README.md"):
    with open(os.path.join(root, gen), encoding="utf-8") as fh:
        t = fh.read()
    for m in latex.finditer(t):
        warn.append(f"{gen}: LaTeX command left: '{t[max(0, m.start() - 30):m.end() + 30]}'".replace("\n", " "))
with open(os.path.join(root, "README.md"), encoding="utf-8") as fh:
    readme = fh.read()
for m in re.finditer(r"(?i)\bgit\b|github|\bcommit|runpod|google drive", readme):
    warn.append(f"README.md mentions '{m.group(0)}'")
for w in warn:
    print("WARNING:", w)
if bad:
    print("\n".join(bad), file=sys.stderr)
    sys.exit(1)
print(f"binary files scanned: {nbin}; no Hangul text; no \\TBD; {len(warn)} warnings")
EOF
echo "grep: no identifying strings in $(find "$SUPP" -type f | wc -l | tr -d ' ') files"

echo "== 5. permissions, mtimes, zip"
find "$SUPP" -type d -exec chmod 755 {} +
find "$SUPP" -type f -exec chmod 644 {} +
find "$SUPP" -type f -name '*.sh' -exec chmod 755 {} +
TZ=UTC find "$SUPP" -exec touch -t "$FIXED_MTIME" {} +
( cd "$STAGE" && find supplement -print | sort | TZ=UTC zip -X -q -9 -@ "$ZIP_NEW" )

echo "== 6. check the archive"
size=$(wc -c < "$ZIP_NEW" | tr -d ' ')
[ "$size" -lt "$LIMIT" ] || fail "zip is $size bytes (limit $LIMIT)"
if unzip -Z1 "$ZIP_NEW" | grep -E '(^|/)(__MACOSX|\.DS_Store|__pycache__|\._)' ; then fail "OS or cache entries in the zip"; fi
mkdir -p "$STAGE/unzip_check"
unzip -q "$ZIP_NEW" -d "$STAGE/unzip_check"
if grep -rIiE "$PAT" "$STAGE/unzip_check"; then fail "identifying strings in the extracted zip"; fi
diff -rq "$SUPP" "$STAGE/unzip_check/supplement" > /dev/null || fail "the extracted zip differs from the folder"
rm -rf "$STAGE/unzip_check"
nent=$(unzip -Z1 "$ZIP_NEW" | wc -l | tr -d ' ')
unz=$(du -sk "$SUPP" | cut -f1)
sha=$("$PY" -I -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$ZIP_NEW")

echo "== 7. publish"
rm -rf "$OUT/supplement.previous"
[ -d "$OUT/supplement" ] && mv "$OUT/supplement" "$OUT/supplement.previous"
mv "$SUPP" "$OUT/supplement"
mv -f "$ZIP_NEW" "$ZIP"
rm -rf "$OUT/supplement.previous" "$STAGE"
trap - ERR
echo "zip: $ZIP"
echo "     $size bytes ($(( size / 1024 )) KiB; folder $(( unz / 1024 )) MiB), $nent entries, sha256 $sha"
if [ -s "$OUT/build_notes.txt" ]; then
  echo "== notes ($OUT/build_notes.txt)"
  cat "$OUT/build_notes.txt"
fi
echo "done"

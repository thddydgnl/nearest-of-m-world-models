#!/bin/bash
# Build the anonymized supplementary material of the TMLR submission from the current sources.
#
# usage: paper/build/make_supplement.sh          (from any folder; re-runnable, nothing is edited by hand)
#   SKIP_SMOKE=1 paper/build/make_supplement.sh  skips the smoke tests (they need numpy and matplotlib)
#   PYTHON=/path/to/python3                      interpreter for the helper and the smoke tests
#
# Output (ignored by git): paper/supplement_build/
#   supplement/            the folder that is zipped
#   supplement_anon.zip    the archive to upload (deterministic: sorted entries, fixed mtimes, zip -X)
#   build_notes.txt        warnings for the person who builds it (stale main.aux, unmapped items, ...)
#   smoke/                 printouts of the smoke tests (supplement copy vs. project version)
#
# Steps: paper/build/supp_build.py copies and scrubs the sources and writes README.md, LICENSE, data/README.md,
# predictions/recorded.md and run_args/runs.csv; then this script runs the smoke tests, removes OS and Python
# caches, fails if an identifying string is left, fixes permissions and mtimes, zips and checks the archive.
# Before a final build: rebuild the manuscript (table and figure numbers come from paper/tmlr/main.aux).
set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$REPO/paper/supplement_build"
SUPP="$OUT/supplement"
ZIP="$OUT/supplement_anon.zip"
PY="${PYTHON:-python3}"
FIXED_MTIME="202611010000.00"     # touch -t, read in UTC: every file and folder gets 2026-11-01 00:00 UTC
LIMIT=100000000                   # TMLR supplementary limit, 100 MB
PAT='yonghwi|thddydgnl|kmlee|cbnu|chungbuk|kiis|mvf|racedreamer|runpod|/home/|/Users/|gdrive|github\.com/thddydgnl'

fail() { echo "FAILED: $*" >&2; exit 1; }

echo "== 1. assemble"
rm -rf "$SUPP" "$ZIP" "$OUT/smoke" "$OUT/unzip_check"
mkdir -p "$OUT"
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
  nfig=$(ls "$T/supp" | grep -c -E '\.(pdf|png)$' || true)
  echo "make_figures.py: $nfig figure files, printout identical to the project version ($(wc -l < "$T/make_figures.txt" | tr -d ' ') lines)"
  # (b) revision-r1 summaries: same output and exit status as the project versions
  for pair in "analysis/summarize_r1.py:main/revision/summarize_r1.py" \
              "analysis/summarize_analyses.py:main/revision/an_analyses/summarize.py"; do
    s=${pair%%:*}; r=${pair#*:}; b=$(basename "$s" .py)
    set +e
    ( cd "$SUPP" && "$PY" "$s" > "$T/$b.txt" 2> "$T/$b.err" ); rs=$?
    ( cd "$REPO" && "$PY" "$r" > "$T/${b}_repo.txt" 2> /dev/null ); rr=$?
    set -e
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
    ast.parse(open(f, encoding="utf-8").read(), f)
print(len(sys.argv) - 1, "Python files parse")'
  n=0; while IFS= read -r -d '' f; do bash -n "$f" || fail "bash -n $f"; n=$((n + 1)); done < <(find "$SUPP" -name '*.sh' -print0)
  echo "$n shell scripts pass bash -n"
fi

echo "== 3. remove OS and Python caches"
find "$SUPP" -depth \( -name __pycache__ -o -name __MACOSX \) -type d -exec rm -rf {} +
find "$SUPP" \( -name .DS_Store -o -name '._*' -o -name '*.pyc' \) -type f -delete

echo "== 4. anonymity check"
set +e
grep -rIiE "$PAT" "$SUPP" > "$OUT/grep_matches.txt"; rc=$?
set -e
if [ $rc -eq 0 ]; then cat "$OUT/grep_matches.txt" >&2; fail "identifying strings in $SUPP (listed above)"; fi
[ $rc -eq 1 ] || fail "grep exited with status $rc"
rm -f "$OUT/grep_matches.txt"
names=$(cd "$SUPP" && find . | grep -iE "$PAT" || true)
[ -z "$names" ] || { echo "$names" >&2; fail "identifying file or folder names"; }
"$PY" -I - "$SUPP" "$PAT" <<'EOF'
import os, re, sys
root, pat = sys.argv[1], re.compile(sys.argv[2].encode(), re.I)
text_ext = {".py", ".sh", ".txt", ".md", ".json", ".jsonl", ".log", ".csv"}
hangul = re.compile("[ᄀ-ᇿ㄰-㆏가-힣]")
weak = re.compile(r"\bKorean?\b|\bKST\b|Asia/Seoul|\+09:?00\b")
bad, warn, nbin = [], [], 0
for d, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(d, f)
        b = open(p, "rb").read()
        if os.path.splitext(f)[1].lower() not in text_ext:   # grep -I skips binary files: scan their bytes
            nbin += 1
            if pat.search(b):
                bad.append(f"{p}: identifying bytes in a binary file")
            continue
        t = b.decode("utf-8", "replace")
        if hangul.search(t):
            bad.append(f"{p}: Hangul text (internal notes must not be included)")
        for m in weak.finditer(t):
            warn.append(f"{os.path.relpath(p, root)}: '{t[max(0, m.start() - 40):m.end() + 40]}'".replace("\n", " "))
readme = open(os.path.join(root, "README.md"), encoding="utf-8").read()
for m in re.finditer(r"(?i)\bgit\b|github|\bcommit|runpod|google drive", readme):
    warn.append(f"README.md mentions '{m.group(0)}'")
for w in warn:
    print("WARNING (weak identity clue):", w)
if bad:
    print("\n".join(bad), file=sys.stderr)
    sys.exit(1)
print(f"binary files scanned: {nbin}; no Hangul text; {len(warn)} weak-clue warnings")
EOF
echo "grep: no identifying strings in $(find "$SUPP" -type f | wc -l | tr -d ' ') files"

echo "== 5. permissions, mtimes, zip"
find "$SUPP" -type d -exec chmod 755 {} +
find "$SUPP" -type f -exec chmod 644 {} +
find "$SUPP" -type f -name '*.sh' -exec chmod 755 {} +
TZ=UTC find "$SUPP" -exec touch -t "$FIXED_MTIME" {} +
( cd "$OUT" && find supplement -print | sort | TZ=UTC zip -X -q -9 -@ "$ZIP" )

echo "== 6. check the archive"
size=$(wc -c < "$ZIP" | tr -d ' ')
[ "$size" -lt "$LIMIT" ] || fail "zip is $size bytes (limit $LIMIT)"
if unzip -Z1 "$ZIP" | grep -E '(^|/)(__MACOSX|\.DS_Store|__pycache__|\._)' ; then fail "OS or cache entries in the zip"; fi
mkdir -p "$OUT/unzip_check"
unzip -q "$ZIP" -d "$OUT/unzip_check"
if grep -rIiE "$PAT" "$OUT/unzip_check"; then fail "identifying strings in the extracted zip"; fi
rm -rf "$OUT/unzip_check"
nent=$(unzip -Z1 "$ZIP" | wc -l | tr -d ' ')
unz=$(du -sk "$SUPP" | cut -f1)
sha=$("$PY" -I -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$ZIP")
echo "zip: $ZIP"
echo "     $size bytes ($(( size / 1024 )) KiB; folder $(( unz / 1024 )) MiB), $nent entries, sha256 $sha"
if [ -s "$OUT/build_notes.txt" ]; then
  echo "== notes ($OUT/build_notes.txt)"
  cat "$OUT/build_notes.txt"
fi
echo "done"

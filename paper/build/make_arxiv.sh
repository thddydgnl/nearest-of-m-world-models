#!/bin/zsh
# Build the arXiv version of the manuscript from the single TMLR source in paper/tmlr/.
#
#   paper/build/make_arxiv.sh
#
# Produces, under paper/arxiv/:
#   src/                                  flat upload bundle (main.tex, sections, styles, main.bbl, figures/)
#   nearest-of-m-world-models_arxiv_src.tar.gz   the same bundle, ready to upload
#   main.pdf                              the bundle compiled here (tectonic; arXiv itself uses pdflatex)
#   metadata.txt                          title, authors, abstract, suggested comments/categories for the arXiv form
#
# The arXiv main.tex differs from paper/tmlr/main.tex only in three mechanical ways:
#   a first line that sets \pdfoutput=1 under pdfTeX only (arXiv looks for "\pdfoutput=1" in the
#   first lines to pick pdflatex, which the PDF/PNG figures need; a bare \pdfoutput=1 would make
#   hyperref pick the pdftex driver under tectonic/XeTeX and break the local check build),
#   \usepackage[preprint]{tmlr} (shows the author block, empty running header),
#   \input{main.bbl} instead of \bibliography{main} (arXiv does not run BibTeX).
# Everything else (abstract, sections, appendix, author block) is read from paper/tmlr/.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TMLR="$(cd "$HERE/../tmlr" && pwd)"
OUT="$(cd "$HERE/../arxiv" && pwd)"
SRC="$OUT/src"
TAR="$OUT/nearest-of-m-world-models_arxiv_src.tar.gz"
LOG="$OUT/build.log"

# 1. Compile the TMLR source once more, keeping main.bbl.
(cd "$TMLR" && tectonic -X compile --keep-intermediates main.tex > "$LOG" 2>&1) || { cat "$LOG" >&2; echo "TMLR compile failed" >&2; exit 1; }
[[ -f "$TMLR/main.bbl" ]] || { echo "paper/tmlr/main.bbl was not produced" >&2; exit 1; }

# 2. Assemble the flat bundle.
rm -rf "$SRC"
mkdir -p "$SRC/figures"
cp "$TMLR"/sec_*.tex "$TMLR"/app.tex "$TMLR"/math_commands.tex "$TMLR"/tmlr.sty "$TMLR"/fancyhdr.sty "$TMLR"/main.bbl "$SRC/"
grep -ho 'includegraphics\[[^]]*\]{[^}]*}' "$TMLR"/main.tex "$TMLR"/sec_*.tex "$TMLR"/app.tex \
  | sed 's/.*{//; s/}//' | sort -u | while read -r f; do cp "$TMLR/figures/$f" "$SRC/figures/"; done
{
  printf '%s\n' '\ifx\pdftexversion\undefined\else\pdfoutput=1\fi'
  sed -e 's/^\\usepackage{tmlr}$/\\usepackage[preprint]{tmlr}/' \
      -e 's/^\\bibliography{main}$/\\input{main.bbl}/' \
      -e '/^\\bibliographystyle{tmlr}$/d' "$TMLR/main.tex"
} > "$SRC/main.tex"
grep -q '^\\usepackage\[preprint\]{tmlr}$' "$SRC/main.tex" || { echo "preprint option substitution failed" >&2; exit 1; }
grep -q '^\\input{main.bbl}$' "$SRC/main.tex" || { echo "main.bbl substitution failed" >&2; exit 1; }
if grep -q 'Anonymous authors' "$SRC/main.tex"; then echo "author block still anonymous in main.tex" >&2; exit 1; fi

# 3. Compile the bundle in place, as a stand-in for arXiv's own build.
(cd "$SRC" && tectonic -X compile main.tex > "$LOG" 2>&1) || { cat "$LOG" >&2; echo "bundle compile failed" >&2; exit 1; }
grep -E 'Overfull|undefined|rror' "$LOG" | grep -v '^note:' || true
mv "$SRC/main.pdf" "$OUT/main.pdf"

# 4. Tarball with the files at top level (what arXiv expects).
rm -f "$TAR"
(cd "$SRC" && tar czf "$TAR" main.tex sec_*.tex app.tex math_commands.tex tmlr.sty fancyhdr.sty main.bbl figures)

# 5. Check that the tarball alone compiles, in a fresh directory.
CHK="$(mktemp -d)"
tar xzf "$TAR" -C "$CHK"
(cd "$CHK" && tectonic -X compile main.tex > "$CHK/build.log" 2>&1) || { cat "$CHK/build.log" >&2; echo "tarball does not compile on its own" >&2; exit 1; }
if ! cmp -s <(python3 -I -c 'import sys,pymupdf; print(pymupdf.open(sys.argv[1]).page_count)' "$CHK/main.pdf") \
            <(python3 -I -c 'import sys,pymupdf; print(pymupdf.open(sys.argv[1]).page_count)' "$OUT/main.pdf"); then
  echo "tarball build has a different page count" >&2; exit 1
fi
rm -rf "$CHK"

# 6. Plain-text metadata for the arXiv submission form.
python3 -I "$HERE/arxiv_metadata.py" "$TMLR/main.tex" "$OUT/main.pdf" > "$OUT/metadata.txt"

PAGES="$(python3 -I -c 'import sys,pymupdf; print(pymupdf.open(sys.argv[1]).page_count)' "$OUT/main.pdf")"
echo "arXiv build: $OUT/main.pdf ($PAGES pages), $TAR ($(du -h "$TAR" | cut -f1)), $OUT/metadata.txt"

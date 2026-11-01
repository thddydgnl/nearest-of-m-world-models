#!/bin/zsh
# Copy paper/main.pdf into the shared manuscripts folder and append one log line.
# Same role as build/sync_papers_copy.sh in the other research projects.
#
# Usage (from anywhere), after `tectonic -X compile main.tex` in paper/tmlr/:
#   paper/build/sync_papers_copy.sh "short memo of what changed"
#
# The log records the short git commit hash when the tree is clean; otherwise the
# first 7 hex digits of the PDF's md5.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../tmlr/main.pdf"
NAME="IMLE-WM_Nearest-of-m_World_Models"
VENUE="TMLR"
DEST_DIR="/Users/song-yonghwi/Documents/research/papers/$NAME/$VENUE"
DEST="$DEST_DIR/${NAME}_${VENUE}_원고.pdf"
LOG="$DEST_DIR/원고_갱신기록.txt"
MEMO="${1:-}"

if [[ ! -f "$SRC" ]]; then
  echo "missing $SRC; run 'tectonic -X compile main.tex' in $HERE/../tmlr first" >&2
  exit 1
fi
if [[ -z "$MEMO" ]]; then
  echo "usage: $0 \"memo\"" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
if [[ -f "$DEST" ]] && cmp -s "$SRC" "$DEST"; then
  echo "unchanged: $DEST already matches $SRC"
  exit 0
fi

cp "$SRC" "$DEST"
MD5="$(md5 -q "$SRC" | cut -c1-7)"
if git -C "$HERE" rev-parse --short HEAD >/dev/null 2>&1 && [[ -z "$(git -C "$HERE" status --porcelain -- "$HERE/../tmlr" 2>/dev/null)" ]]; then
  MD5="$(git -C "$HERE" rev-parse --short HEAD)"
fi
PAGES="$(python3 -I -c 'import sys, pymupdf; print(pymupdf.open(sys.argv[1]).page_count)' "$SRC" 2>/dev/null || echo '?')"

if [[ ! -f "$LOG" ]]; then
  {
    printf '%s\n' "IMLE-WM (nearest-of-m world models): TMLR 원고 사본 갱신 기록 (탭 구분)"
    printf '%s\n' "원본: IMLE-WM/paper/tmlr/main.pdf -- paper/build/sync_papers_copy.sh가 복사하고 아래 줄을 덧붙인다"
    printf '%s\n' "두 번째 열은 git 커밋 해시(작업 트리가 깨끗할 때) 또는 PDF md5 앞 7자리다. PDF는 tectonic으로 만든다 (paper/tmlr/에서 tectonic -X compile main.tex)"
    printf '\n'
    printf '갱신 시각\t커밋 또는 PDF md5\t쪽수\t메모\n'
  } > "$LOG"
fi
printf '%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M')" "$MD5" "$PAGES" "$MEMO" >> "$LOG"
echo "copied -> $DEST ($PAGES pages, md5 $MD5)"

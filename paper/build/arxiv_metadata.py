#!/usr/bin/env python3
"""Print the text for the arXiv submission form from paper/tmlr/main.tex.

usage: python3 -I arxiv_metadata.py path/to/main.tex path/to/arxiv/main.pdf

Title, author list and abstract are read from the LaTeX source and the manuscript macros
(\\nom{}, \\Nom{}, \\imlewm{}, \\emph{}, inline math) are expanded to plain text, so the
output can be pasted into the form as is. Lines starting with '#' are notes, not form text.
"""
import re
import sys


def plain(s: str) -> str:
    s = s.replace(r"\nom{}", "nearest-of-m").replace(r"\Nom{}", "Nearest-of-m").replace(r"\imlewm{}", "IMLE-WM")
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = s.replace("\\\\", " ").replace("{=}", "=").replace(r"\times", "x").replace("~", " ").replace(r"\%", "%")
    s = s.replace("$", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


src = open(sys.argv[1], encoding="utf-8").read()
title = plain(re.search(r"\\title\{(.*?)\}\n", src, re.S).group(1))
abstract = plain(re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1))
block = re.search(r"\\author\{(.*?)\}\s*\n", src, re.S).group(1)
authors = []
for part in re.split(r"\\AND", block):
    m = re.search(r"\\name\s+(.*?)\s+\\email\s+(\S+)", part, re.S)
    addr = re.search(r"\\addr\s+(.*)", part, re.S)
    if m:
        authors.append((plain(m.group(1)), m.group(2), plain(addr.group(1).replace("\\\\", ", ")) if addr else ""))

pages = ""
try:
    import pymupdf  # type: ignore

    pages = str(pymupdf.open(sys.argv[2]).page_count)
except Exception:  # pragma: no cover - page count is a convenience only
    pass

print("# arXiv 제출 양식용 텍스트. '#'로 시작하는 줄은 메모이며 붙여 넣지 않는다.")
print()
print("Title:")
print(title)
print()
print("Authors:")
print(", ".join(a[0] for a in authors))
for name, email, addr in authors:
    print(f"#   {name} <{email}> -- {addr}")
print()
print("Abstract:")
print(abstract)
print()
print("Comments (suggestion):")
print(f"{pages} pages including appendices A-J. Code: https://github.com/thddydgnl/nearest-of-m-world-models")
print()
print("Categories (suggestion):")
print("cs.LG (primary); cross-list cs.CV, cs.RO")
print("# 라이선스는 제출 시 선택한다 (arXiv 기본 비독점 라이선스 또는 CC BY 4.0). TMLR은 arXiv 선공개를 허용한다.")

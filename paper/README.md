# IMLE-WM 논문 원고 — How Many Candidates Does a One-Step World Model Need?

| 폴더 | 대상 | 상태 (2026-11-02) |
|---|---|---|
| [`tmlr/`](tmlr/) | TMLR (2026-11-01 투고처 확정, [`../main/reports/final_verdict.md`](../main/reports/final_verdict.md) 8절) | 초안 완성·영문 교정 완료(2026-11-01). 익명 제출용 `tmlr/main.pdf`(30쪽: 본문 12쪽 + 참고문헌 + 부록 A~H), 본문 12쪽 제한 준수, overfull box 0 |
| [`tmlr/_long/`](tmlr/_long/) | 같은 원고의 22쪽 본문 판 | 교정 전 판. 심사자가 본문에 더 자세한 서술을 요구할 때 쓰려고 남겨 둔다. 이 판의 그림·표 대부분은 지금 부록 F에 있다 |
| [`OUTLINE.md`](OUTLINE.md) | 논문 구조 분석과 목차안 | 2026-11-01. 실험 결과 감사(지지·기각), 한 줄 주장, 기여 4개, 절 구성 |
| `iclr2027_ws/` (예정) | ICLR 2027 워크숍(비아카이벌만 가능) | 미착수. 워크숍 목록은 2026-11-29 공개 예정 |

- 공통 파일
  - [`build/make_figures.py`](build/make_figures.py): 서버 run 출력(`eval.json`, `e3.json`, `log.jsonl`, `modeshare.json`)과 `main/results/`만 읽어 그림을 다시 그린다. 새 계산은 없다. 표에 들어간 수치도 함께 출력하므로 보고서와 대조할 수 있다.
  - [`build/sync_papers_copy.sh`](build/sync_papers_copy.sh): `tmlr/main.pdf`를 공유 원고 폴더 `research/papers/IMLE-WM_Nearest-of-m_World_Models/TMLR/`에 복사하고 갱신 기록 한 줄을 덧붙인다. 원고를 고칠 때마다 돌린다.
  - 스타일 파일 `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, `math_commands.tex`는 [JmlrOrg/tmlr-style-file](https://github.com/JmlrOrg/tmlr-style-file)에서 가져왔다.
- 빌드(맥북, tectonic 0.16.9, 몇 초)

```bash
cd /Users/song-yonghwi/Documents/IMLE-WM/paper/tmlr && tectonic -X compile main.tex
```

```bash
cd /Users/song-yonghwi/Documents/IMLE-WM && python3 paper/build/make_figures.py main/results/runs main/results paper/tmlr/figures
```

```bash
cd /Users/song-yonghwi/Documents/IMLE-WM && paper/build/sync_papers_copy.sh "무엇을 고쳤는지 한 줄"
```

## 투고 전 할 일

- 저자 블록을 익명에서 `[accepted]`/`[preprint]` 모드로 바꾸는 시점 확인(TMLR은 double-blind, arXiv 동시 공개는 허용).
- 코드·데이터 공개 링크(이 저장소)와 보충 자료 zip(100 MB 이하) 준비.
- 저자 전원의 OpenReview 프로필.
- 용어 규칙은 `main/reports/novelty_analysis.md` 4절을 따른다: "first one-step stochastic world model"이라 쓰지 않는다, "m ≈ 2K"를 쓰지 않는다, 평탄화는 Bennett Thm 7의 확인으로 서술한다, "pre-registered" 대신 "predictions recorded before the runs"라고 쓴다.

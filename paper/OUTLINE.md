# IMLE-WM 논문 구조 분석과 목차안 (TMLR 투고용)

2026-11-01 · 근거: `results/` 보고서 전체(탐색, E0~E16), `theory_m_law.md`, `novelty_analysis.md`, `final_verdict.md`, 문헌 조사 01~04, 계획서 v2. 서버 `~/imle-wm/runs`의 원자료(eval.json, e3.json, log.jsonl)로 그림을 다시 그렸다.

## 1. 실험 결과 감사: 무엇이 지지됐고 무엇이 기각됐나

| 주장 | 판정 | 근거 강도 | 논문에서의 위치 |
| --- | --- | --- | --- |
| C1 1-step 다봉 예측 | 지지 (A1·A2 강함, B는 precision만 유의, BAIR 범위 밖) | 3시드, 기준선 6종(회귀, FM-50/2, iMF, Shortcut, ES), 파라미터 ±4%, 계층 bootstrap | **결과 I** (6절) |
| C2 매칭 예산 m 법칙 | 지지, 내용 수정 ("m ≈ 2K" 기각 → 양자화 오차 I(m)) | K 4개 × m 4~5개 × 2시드, 실험 전 기록한 예측 9개 중 7개 적중 | **결과 II** (7절), 핵심 그림 |
| C3 롤아웃 매니폴드 유지 | 지지 (FM-50 포함) | 3시드, 벤치마크 3개, 판독기 대조 | 결과 IV (9절) |
| C4 결과 암기 진단 | 지지 (A2), ES 대조(E15)로 목적함수 계열을 넘어 확장 | 3시드, 대응책 3종, 장난감 실험, 사전 예측 5/5 | **결과 II** (7절), 주 기여 |
| C5 planning | 성공률 기각, 속도(50배)와 위험 추정만 지지 | 2과제 × 100 에피소드 × 3시드 | 결과 IV (9절), 정직한 부정 결과 |
| C6 순차 매칭 | 지지 (B −33%), 노벨티는 낮음 (CAT-K/DaD 계보), E16 ablation으로 기전 분해 | B 3시드 계층 CI, A1 ablation 2시드 사전 예측 3/3 | 결과 IV (9절), 부 기여 |
| 명제 2 (BAIR 수축) | 정성 지지, 정량(d_eff=29)은 한 점에 맞춘 값. 독립 추정 실패 | E8·E10·E16 | 결과 III (8절), 경계 사례 |
| 명제 3 (평탄화) | Bennett Thm 7의 정량 확인 (0.01 안) | E14 2시드 | 결과 III (8절), 한계이자 이론 검증 |
| N1 궤적 매칭, N2 스펙트럼 stage, N3 분산 planning, N4 대조 항 | 전부 기각/불필요 | 탐색 실험 | 부록 G (부정 결과) |

핵심 판단 세 가지:
1. **이야기의 중심은 "방법"이 아니라 "질문"이어야 한다.** 1-NFE min-of-K 월드모델 자체는 XM(2607.27372)이 먼저 했고, 동시기 1-step WM이 여럿 나왔다. 반면 "m이 얼마나 필요하고 언제 어떤 m으로도 부족한가"에 대한 정량 답(암기 유인 I(m), 고차원 수축, 평탄화)은 선행이 없다. 따라서 제목과 서론을 이 질문으로 잡고, DINO 토큰 위 1-step IMLE는 그 질문을 검증하는 통제된 도구로 둔다.
2. **암기(결과 암기) 분석이 가장 안전한 주 기여다.** 학습/검증 맥락 비대칭으로 잰 암기 지수, K=2~16이 I(m) 하나로 모이는 그림(놓칠 확률 (1−1/K)^m 계열은 RMSE가 2배), 실험 전 기록한 예측, 그리고 proper score(energy score)는 유인이 m과 무관하다는 대비(E15)까지 한 묶음으로 7절에 둔다.
3. **부정 결과와 경계를 숨기지 않는다.** BAIR 과소분산, 평탄화, planning 성공률 미개선은 모두 "같은 코드북 관점이 예측한 결과"로 서술하면 약점이 아니라 이론의 검증이 된다. TMLR은 주장이 근거로 뒷받침되는지를 본다.

## 2. 논문 한 줄 주장과 기여 문장

**제목(안):** How Many Candidates Does a One-Step World Model Need? A Random-Codebook Account of Nearest-of-m Training on Frozen DINO Tokens

**한 줄:** 고정 DINO 토큰 위에서 nearest-of-m(조건부 IMLE)으로 학습한 1-step 월드모델은 소수 모드로 갈라지는 불확실성을 순전파 1회로 덮는다. 이 목적은 "m개짜리 무작위 코드북"이며, 이 관점 하나로 (a) 비반복 맥락의 결과 암기와 필요한 m, (b) 모드 확률 평탄화, (c) 고차원 연속 불확실성에서의 과소분산이 예측된다.

**기여 4개 (영문 초안은 `novelty_analysis.md` 3절):**
1. 통제 비교: 모드형 불확실성에서 1-NFE nearest-of-m이 FM-50, 1-step flow 생성기(iMF, Shortcut), ESP식 energy-score 생성기보다 coverage·precision이 높다 (지연 1/30). 실제 로봇 영상(BAIR)에서는 아니며, 이는 이론이 예측한다.
2. 결과 암기의 측정과 기전: 암기 유인 = 놓친 왜곡 I(m) (놓칠 확률이 아님). 실험 전 예측 7/9. energy score는 유인이 m과 무관해 외운다 (사전 예측 5/5).
3. 경계: 고차원 수축 e_m/√d (가우시안 율-왜곡 test channel과 같은 형태), BAIR에서 맥락별 1~2.5 방향 붕괴, Bennett 평탄화의 정량 확인.
4. 롤아웃: 매니폴드 유지(FM은 step 수와 무관하게 이탈), 순차 매칭(DaD/CAT-K 계보)의 −33%와 이득 분해 ablation.

## 3. 목차안

```
Abstract
1  Introduction
2  Related Work
   2.1 Min-of-m objectives: IMLE, multiple-choice learning, explorative modeling
   2.2 Stochastic and one-step world models in feature space
   2.3 Memorization in conditional generators; proper scoring rules
   2.4 Rollout training and exposure bias
3  Nearest-of-m World Models on Frozen DINO Tokens
   3.1 Setting and notation
   3.2 Objective: nearest-of-m matching with a robust loss
   3.3 Per-step matching on self-rollouts
   3.4 Inference and planning
   3.5 Baselines at matched parameters
4  A Random-Codebook Account of Nearest-of-m Training
   4.1 The objective is the distortion of a random codebook
   4.2 Outcome memorization in non-repeating contexts (Prop. 1)
   4.3 High-dimensional continuous outcomes: shrinkage (Prop. 2) and two regimes
   4.4 Flattening of mode probabilities
   4.5 Proper scores: the energy score's incentive does not depend on m (Prop. 3)
   4.6 Predictions tested in this paper
5  Experimental Setup
   5.1 Benchmarks (A1 Bouncing Shapes, A2 action-conditioned bounce, B stochastic Push-T, branch-risk push task, BAIR)
   5.2 Metrics
   5.3 Protocol and statistics
6  Results I: Does a Small Candidate Budget Suffice?
   6.1 Non-repeating contexts (A2-div)
   6.2 Bouncing Shapes under FlowWM's protocol and readout (A1)
   6.3 Stochastic Push-T multi-future test (B)
   6.4 Baseline fairness and latency
7  Results II: Outcome Memorization and the Candidate Budget
   7.1 The phenomenon
   7.2 Excess memorization collapses onto I(m, K)
   7.3 Countermeasures and the cost of m
   7.4 A strictly proper score memorizes regardless of m
8  Results III: Where No Candidate Budget Suffices
   8.1 Mode-probability flattening (asymmetric modes)
   8.2 Under-dispersion on real robot video (BAIR)
9  Results IV: Rollouts and Planning
   9.1 Rollouts stay on the token manifold
   9.2 Per-step matching on self-rollouts and what the fed-back state does
   9.3 Planning: equal success, 50x faster, accurate branch risk
10 Discussion and Limitations
11 Conclusion
Broader Impact Statement / Reproducibility
References
Appendix
   A Derivations (Props. 1-3, flattening, direction-insensitivity corollary)
   B I(m,K) for the bounce geometry and the full prediction table
   C Benchmark construction
   D Architecture, training, and baseline recipes (incl. iMF port, Shortcut, ES estimator)
   E Readouts and readout cross-checks
   F Additional results (full grids, BAIR, ablation, toy experiment, planning)
   G Negative results (trajectory matching, multi-stage supervision, variance-aware planning, action contrast, MeanFlow re-implementation)
   H Statistical methodology
```

## 4. 그림·표 계획

| # | 내용 | 자료 |
| --- | --- | --- |
| Fig 1 | 한 맥락의 열거된 미래 위에 회귀·FM·IMLE 샘플 산점도 (정성) | `results/scatter_bounce_k4.png` (탐색 레시피, A2-fix) |
| Fig 2 | **핵심:** 초과 암기 vs I(m,K), K=2~16, 사전 예측 구간, 놓칠 확률 계열 적합 | E12 표 |
| Fig 3 | 학습 step에 따른 학습 맥락 암기 지수와 검증 coverage: 회귀, FM, IMLE m=4/16, ES m=8/16 | log.jsonl |
| Fig 4 | "m 손잡이": coverage, 암기 지수, 모드 점유율(Bennett 예측선), 결정론 오차 vs m | E2·E12·E14 |
| Fig 5 | BAIR 퍼짐/오차 비 vs m (한 점에 맞춘 곡선)와 맥락별 퍼짐 PR | E8·E10·E16 |
| Fig 6 | 매니폴드 거리 vs horizon (A2, A1, B) | e3.json |
| Fig 7 | 순차 매칭 ablation: 모델 퍼짐 vs 실제 미래 퍼짐 (N0, matched, random, gt) | E16 |
| Table 1 | A2-div 주 비교 (coverage, precision, recall, 결정론 오차, 암기, 지연) | E1·E7·E15 |
| Table 2 | A1 두 프로토콜 × 두 판독기 F1 오차 | A1_joint_pilot, E12_E14 |
| Table 3 | B 다중 미래 precision/recall 오차 + 계층 CI | E1·E7·E11 |
| Table 4 | K × m 격자 (암기, coverage, precision, 결정론 오차) | E2·E12 |
| Table 5 | 암기 대응책 | E4 |
| Table 6 | ES vs nearest-of-m: A2-div, A2-fix, 토큰별 ES, 장난감 | E15·E16 |
| Table 7 | 비대칭 모드 점유율 | E14 |
| Table 8 | BAIR | E8·E10 |
| Table 9 | 순차 매칭 B 결과 + A1 ablation | E6·E11·E16 |
| Table 10 | planning 성공률·시간·위험 추정 | E5·E13 |

## 5. 서술 규칙 (리뷰 리스크 대응, `novelty_analysis.md` 4절)

- "first one-step stochastic world model" 류 표현 금지. "conditional IMLE on frozen DINO tokens, action-conditioned, evaluated with enumerable mode coverage"로 한정.
- "m ≈ 2K" 금지. "m is set by the quantization error I(m)".
- 평탄화는 Bennett(2026) Thm 7의 확인으로만. 수축 공식은 고전 율-왜곡의 귀결로.
- "pre-registered" 대신 "predictions recorded before running the experiments".
- "improves planning success" 금지. "equal success at 50x lower planning time".
- ES에 대해 "proper scores memorize" 일반 진술 금지. "in non-repeating contexts with identifying appearance".
- 명제 3(ES 유인)은 사후 분석임을 명시. 사전 예측은 추가 실험(A2-fix, 토큰별 ES)만.
- FlowWM 논문 수치(4.31)와 직접 비교하지 않음. 같은 판독기·같은 프로토콜·같은 소규모 예산 안의 상대 비교.

## 6. TMLR 양식 요점 (2026-11-01 확인, jmlr.org/tmlr/author-guide.html + tmlr.sty)

- 길이 제한 없음. 다만 "a paper's length should be justified by its content"이고 본문이 지나치게 길면 심사가 지연됨. 부록은 참고문헌 뒤.
- 이중 맹검: `\usepackage{tmlr}` 기본이 익명. 채택 후 `[accepted]`, 프리프린트는 `[preprint]`.
- 10pt, 본문 폭 6.5in × 9in, natbib authoryear(`\citet`, `\citep`), 참고문헌은 `tmlr.bst`.
- 그림 캡션은 그림 아래, 표 캡션은 표 위. 캡션은 소문자(첫 단어·고유명사 제외).
- Broader Impact Statement는 해로움 위험이 클 때만 필수. 저자 기여·사사는 채택 후.
- 보충자료 100MB까지(PDF/ZIP). 코드·데이터 공개 권장.

## 7. 산출물

- `paper/main.tex`, `paper/main.bib`, `paper/figures/*.pdf` (`paper/make_figures.py`로 생성), `paper/main.pdf`
- 스타일 파일: `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, `math_commands.tex` (공식 저장소 JmlrOrg/tmlr-style-file)
- 컴파일: `tectonic -X compile main.tex` (bibtex 자동)

## 8. 완료 상태 (2026-11-01)

- `paper/main.pdf`: 총 30쪽. **본문 12쪽**(서론, 관련연구, 방법, 이론, 설정, 결과 I~IV, 논의·결론; `final_verdict.md` 8절의 "본문 12쪽 이하" 기준 충족), Broader Impact·참고문헌 13~17쪽, 부록 A~H 17~30쪽. 익명 제출 양식(`\usepackage{tmlr}`), overfull 0, 미정의 인용·참조 0.
- 본문: 표 6개(A2 주 비교, A1, B, 벤치마크, 순차 매칭+ablation, —), 그림 2개(암기 붕괴 vs I(m,K); m 손잡이 1×4). 부록: 정성 산점도, 학습 곡선, K×m 격자, m 손잡이 전체 2×3, 대응책, 비대칭 모드, BAIR 그림·표, ES 비교표, 매니폴드 거리 곡선, 순차 매칭 ablation 그림, 장난감 실험, planning 표, 판독기 비교, 부정 결과, 통계.
- 긴 버전(본문 22쪽, 그림·표 모두 본문)은 `paper/_long/`에 보관했다. 심사 과정에서 본문 확장을 요구받으면 그 파일에서 가져온다.
- 그림은 모두 서버 run 원자료(`eval.json`, `e3.json`, `log.jsonl`, `modeshare.json`, `e16_bair_*.json`)에서 `make_figures.py`로 재생성했고, 표의 수치는 같은 스크립트가 출력한 집계값으로 보고서와 대조했다.
- 참고문헌 95건. 2026년 arXiv 논문의 저자명은 arXiv abs 페이지에서 확인했다.
- 투고 전 남은 일: 저자·소속 기입(채택 후 `[accepted]`, arXiv는 `[preprint]`), 코드·데이터 공개 링크, 보충자료(zip) 구성, 영문 교정 1회, OpenReview 프로필.
- 재컴파일: `cd paper && tectonic -X compile main.tex`. 그림 재생성: `python3 make_figures.py <runs_dir> ../results figures` (runs_dir는 서버 `~/imle-wm/runs`의 json/jsonl 사본).

## 9. 심사 관점 교정 (2026-11-01, 2차)

리뷰어 설득력, 용어 일관성, 문체·학술 용어를 기준으로 본문 전체를 다시 손봤다. 바뀐 것:

**주장 범위(리뷰어 반론 대비)**
- 초록·기여 1: "FM-50보다 coverage가 높다"는 주장을 bounce 벤치마크로 한정하고, Push-T는 "precision은 유의, recall은 일관되지만 비유의"라고 기여 문장에 직접 적었다. 초록의 "energy score는 어떤 m에서도 벗어날 수 없다"를 "m을 키워도 벗어나지 못한다"로 좁혔다.
- "We claim no new mathematics" 같은 자기 평가 문장을 "The mathematics is classical; our contribution is..."로 바꾸고, 명제 3(ES)의 (a)~(c)가 Gneiting & Raftery, Dawid, Li & Malik의 귀결임을 명제 바로 뒤에 명시했다.
- Rupprecht/Letzelter 인용 문장의 "centroidal Voronoi quantizers"를 정확한 용어 "centroidal Voronoi tessellations of the conditional distribution"으로, "Shannon's rate–distortion setting"을 "random-coding construction at rate ln m/d nats per dimension"으로 고쳤다.
- XM의 K(예산)와 우리의 K(모드 수)가 충돌하므로 서론에서 "where K denotes the same budget we call m"을 명시했다.
- 본문에서 뺀 수치(m=4 학습 2배·모델 2.7배 암기 수치)는 부록 F에 문단으로 복원해 참조가 비지 않게 했다.

**용어 통일**
- branch → mode: "branch"는 환경의 분기를 뜻하는 비공식 용어였고 "mode"와 섞여 쓰였다. 본문·부록·그림 라벨 전체를 mode로 통일했다. 남긴 것은 과제 이름 "branch-risk task"와 ESP의 과제 이름 "two-branch problem"뿐이다.
- realized-mode share / realized-branch share → "memorization index"(정의는 setup에 한 번).
- deterministic error → "deterministic-transition error".
- spread/error ratio → "spread-to-error ratio".
- "nearest-of-m"(목적함수)과 "IMLE-WM"(우리 모델)을 구분해, 모델의 결과를 말할 때는 IMLE-WM으로 바꿨다(BAIR 절, 그림 범례·캡션).
- A2(bounce 롤아웃 평가)는 모두 A2-div로 명시. 그림 범례의 내부 라벨 "N0"를 제거.
- 영국식 철자(centre, colour, neighbour, favour)를 미국식으로 통일. "1-step"은 "one-step"으로.

**문체**
- 구호형 문장("The budget, not the recipe, decides.", "The budget question has one answer:", "its errors are structural") 등을 서술형으로 바꿨다.
- 결론을 두 문장으로 줄였다. 본문은 여전히 12쪽.

## 10. 영문 교정 (2026-11-01, 3차)

본문 전 문단과 부록을 문장 단위로 교정했다(문법·관사·병렬 구조·용어·수치 표기). 수치는 바꾸지 않았다.
- 교정으로 본문이 약 0.5쪽 늘어나 12쪽 제한을 넘었으므로, 내용 손실 없이 줄였다: 공정성 문단을 A1 문단에 흡수(수치는 부록 F 유지), 순차 매칭 ablation 표를 부록 F.4(표 17)로 이동(본문 표 5는 Push-T 결과만), 결과 III 끝의 요약 문장(논의 절과 중복) 삭제, 본문 그림 2개 높이 축소, 관련연구의 Thiede & Brahma 문장(이론 절에 인용 유지)과 iMF·Shortcut 기준선 문장(방법 절에 인용 이동) 삭제.
- 최종: 총 30쪽, 번호 붙은 본문(1~11절)은 12쪽에서 끝남. Broader Impact 13쪽 상단, 참고문헌 13~17쪽, 부록 17~30쪽. overfull 0, 미정의 참조 0, 중복 단어 0.
- 긴 버전(`paper/_long/`)은 교정 전 상태이므로, 본문 확장이 필요하면 교정된 현재 본문을 기준으로 다시 늘리는 편이 낫다.

## 11. 공용 원고 폴더 사본 (2026-11-01)

`~/Documents/research/papers/IMLE-WM_Nearest-of-m_World_Models/TMLR/`에 `IMLE-WM_Nearest-of-m_World_Models_TMLR_원고.pdf`와 `원고_갱신기록.txt`를 둔다. 다른 과제 폴더와 같은 구성이다. 원고를 고친 뒤에는 `paper/`에서 `tectonic -X compile main.tex`로 다시 빌드하고 `paper/sync_papers_copy.sh "변경 메모"`를 실행해 사본과 기록을 함께 갱신한다. 이 프로젝트는 git 저장소가 아니므로 기록의 두 번째 열은 커밋 대신 PDF md5 앞 7자리다. PDF가 그대로면 스크립트는 아무것도 하지 않는다.

# IMLE-WM 노벨티 분석: 논문에서 무엇을 새롭다고 주장할 수 있나

2026-10-30 · 근거: `final_verdict.md`, E0~E14 보고서, `A1_joint_pilot.md`, `theory_m_law.md`, 계획서 v2, 문헌 조사 01·02·03(10-16), **10-30 문헌 재확인(1절)**.
**10-31 갱신:** E15(energy score 기준선, `E15_report.md`)와 명제 4(`theory_m_law.md` 8절)를 반영했다. 오후의 추가 실험(A2-fix, 토큰별 ES)도 반영했다.
**11-01 갱신:** E16(BAIR 유효 차원, 순차 매칭 ablation, 시드 보강, `E16_report.md`)을 반영했다. 같은 날 문헌 확인 4(명제 4의 선행 연구, ES 판별력, `literature/04_recheck_2026-11-01.md`)도 반영했다.

10-30 재확인에서 10-16 판정이 두 군데 바뀌었다. 7~8월에 나온 preprint 두 편(XM, Bennett)이 초록에 "IMLE"를 쓰지 않아 이전 검색에서 빠졌다. 두 논문 모두 본문을 직접 확인했다.

## 0. 결론

**지금 새롭다고 주장할 수 있는 것 (강한 순서)**
1. **암기 유인은 "놓칠 확률"이 아니라 "놓쳤을 때의 왜곡" I(m)으로 정해진다.**
   - 비반복 맥락의 결과 암기를 학습/검증 맥락 비대칭으로 측정했다.
   - 초과 암기가 K = 2~16에서 I(m) 하나로 모였다(RMSE 0.031).
   - XM과 Bennett이 쓰는 놓칠 확률 (1 − 1/K)^m 계열은 0.068로 설명력이 절반이다. 실험 전에 기록한 예측도 9개 중 7개가 맞았다.
   - **E15로 목적함수 계열을 넘어 넓어진다.** strictly proper한 energy score(ESP)는 암기 유인이 ½E‖Y − Y′‖^β로 m과 무관하다(명제 4, 사후). A2에서 ES는 외우고(m=8 0.87, m=16 0.97, 토큰별 거리 0.97), 최근접-of-m은 m=16에서 외우지 않는다(0.29). 외형을 고정해 열쇠를 없애면 ES도 외우지 않는다(A2-fix 0.27).
2. **고차원 연속 불확실성에서는 어떤 m으로도 부족하다.**
   - 퍼짐이 e_m/√d로 수축한다(가우시안 율-왜곡 test channel과 같은 형태). BAIR가 그 경계다. 다만 BAIR의 d는 한 점에 맞춘 값이고, 독립 추정은 실패했다(E16). 실제 퍼짐은 맥락마다 1~2.5개 방향으로 붕괴한다.
   - 유한 m hard-min의 이 수축은 선행을 찾지 못했다. drifting과 XM의 soft·K→∞ 극한은 메커니즘이 다르다.
3. **XM의 추측을 통제된 조건에서 검증했다.** XM의 추측은 "피처 공간 월드모델은 모드가 적어 작은 K로 충분하다"이다.
   - 모드형 불확실성(A1·A2)에서는 맞다. 1-step IMLE가 FM-50, iMF, Shortcut보다 낫다.
   - 연속 고차원(BAIR)에서는 틀리다. 그리고 이 경계가 1·2로 예측된다.

**더 이상 새롭다고 쓸 수 없는 것**
- **1-NFE min-of-K 월드모델 자체:** XM의 Explorative World Model(Maze2D)이 있다. 조건부 min-of-N 생성의 원형은 Fan, Su, Guibas(CVPR 2017)다.
- **"m을 키우면 과적합·암기가 줄어든다"는 경향:** XM이 SSv2 비디오에서 보였다(min FVD 30.0 vs 37.5).
- **모드 확률 평탄화의 최적해:** Bennett Thm 7과 식이 같다. 우리 E14는 그 정리를 학습된 월드모델에서 0.01 안으로 확인한 것으로 낮춘다.
- **hard-min은 분포를 통제하지 않는 coverage 목적이다:** XM App. F.2, ESP에 있다.

**10-30에 가장 큰 리스크였던 energy score 기준선은 E15(10-31)로 기여가 됐다.**
- ESP(2610.07696)는 1-NFE 정책에서 energy score(strictly proper)가 hard min-of-K(XM)보다 낫다고 보고했다(TwoBranch match 96.2% vs 91.9%).
- 같은 backbone에서 손실만 ESP식 ES로 바꾸면, 비반복 맥락(A2-div)의 coverage가 0.36으로 IMLE m=16(0.93)에 크게 진다. ES는 학습 맥락을 외우고, m을 키우면 더 외운다. E14 비대칭 모드에서도 calibration 이점이 없다(모드 0 점유율 0.886, 실제 0.787).
- 장난감 실험은 두 regime을 모두 재현한다. 맥락이 매번 새로우면 ES가 낫고(ESP 재현), 맥락이 고정돼 구별 가능하면 ES만 완전히 외운다.
- 맥락이 반복돼 외울 수 없는 A1에서도 ES는 IMLE와 회귀보다 나쁘다(F1 8.95 vs 7.21, 8.09). 오차가 잡음형이라(깨끗한 heatmap 판독기 33.8 → 노이즈 판독기 11.6), 고차원 잡음 헤지라는 두 번째 원인이 있는 것으로 보인다.
- 그래서 hard-min의 min 구조가 낳는 두 성질을 구분해 쓴다. 모집단에서는 ESP가 지적한 비-propriety(모드 확률 평탄화)이고, 비반복 맥락의 유한 표본에서는 m에 따라 사라지는 암기 유인 I(m)이다(명제 4, 11-01 문헌 확인). 명제 4는 사후 분석이지만, 그 뒤 예측을 미리 적고 돌린 추가 실험의 예측 다섯 개가 모두 맞았다.
  - 외형을 고정한 A2-fix에서는 ES도 외우지 않고 coverage가 0.95다(2시드). 다만 precision은 IMLE m=16보다 낮다(0.81 vs 0.95).
  - 토큰별 거리로 바꿔도 외운다(0.97, coverage 0.22).
  - 다른 β와 hard-min 혼합은 탐색하지 않았다.

**추천 스토리:** 7~10월에 활발해진 min-of-K 생성 흐름(XM → Bennett → ESP) 안에 논문을 놓는다. 질문은 "1-step 최근접-of-m 월드모델에 m이 얼마나 필요하고, 언제 어떤 m으로도 부족한가"다. IMLE를 고립된 응용으로 쓰는 것보다 지금 논쟁 중인 질문에 정량적으로 답하는 논문이 노벨티와 시의성 모두에서 낫다. E15 덕분에 ESP의 "proper score가 낫다"에도 조건부로 답할 수 있다: 비반복 맥락에서는 반대이고, 그 원인도 설명된다.

| 순위 | 노벨티 후보 | 노벨티 | 근거 강도 | 리뷰 리스크 | 배치 |
| --- | --- | --- | --- | --- | --- |
| 1 | 결과 암기의 측정, 그리고 암기 유인 = 놓친 왜곡 I(m) (놓칠 확률이 아님). energy score와의 대비(명제 4) 포함 | **중상** (경향은 XM, 메커니즘과 정량은 새로움) | 강함 (A2 3시드, 대응책 3종, K×m 17점, 실험 전 예측 7/9, ES 대조와 장난감 실험, ES 추가 실험 예측 5/5) | 중 ("그냥 과적합", 합성 환경, 명제 4는 사후) | 주 기여 |
| 2 | 고차원 수축 e_m/√d와 적용 범위의 경계(BAIR), 두 regime의 통합 | 중 (R-D 대응은 고전, 유한 m hard-min 적용은 선행 없음) | 중하 (자유 파라미터 1개, 3점. 독립 추정은 실패, E16) | 중상 ("R-D 재유도", "맞춘 값"이 사실로 확인됨) | 주 기여 (1과 한 묶음). BAIR는 정성 결과와 붕괴 모양으로 쓴다 |
| 3 | DINO 토큰 위 1-step IMLE WM의 통제 비교 (XM 추측 검증) | 중하 (조합 + 평가) | A1·A2 강함(ESP식 ES 대비 포함), B는 precision만 유의, BAIR 범위 밖 | 중상 (XM, 동시기 1-step WM, FM 규모, ES 변형 일부 미탐색) | 주 기여 (표현 한정) |
| 4 | 롤아웃 매니폴드 유지와 판독기 대조 | 중하 | 강함 | 낮음 | 부 기여 |
| 5 | 평가 프로토콜: 비반복 맥락, 확률 Push-T 다중 미래, 분기 위험 과제, 판독기 의존성 | 중 | 중 | 낮음 | 부 기여 / 부록 |
| 6 | 평탄화의 정량 확인 (Bennett Thm 7) | 낮음 (정리는 기존) | 강함 (0.01 일치) | 낮음 (인용하면) | 분석 절의 확인 결과 |
| 7 | 순차 매칭 N1b | 낮음 (CAT-K, DaD) | 강함 (B −33%, A1 ablation으로 이득을 분해, E16) | 중 (CAT-K 질문에 ablation으로 답함) | 부 기여, 위치 재설정 |
| 8 | planning (속도, 위험 추정) | 없음 | 성공률 기각 | – | 정직한 부정 결과 |

## 1. 10-30 재확인으로 바뀐 것

| 논문 | 무엇을 이미 했나 (본문 확인) | 영향받는 주장 | 대응 |
| --- | --- | --- | --- |
| **Explorative Modeling (XM)** Gladstone, Ji, Du, arXiv 2607.27372 (07-29) | K개 후보 중 손실이 가장 작은 것으로 학습한다(Forward XM). IMLE는 공유 풀을 쓰는 특수 경우, SOG는 조건부 특수 경우로 명시한다. 1-NFE Explorative World Model(Maze2D, Diffuser 대비)과 Explorative Policy가 있다. SSv2에서 K를 늘리면 과적합이 줄었다(4-step 하이브리드). App. F.2: hard min은 K→∞에서 분포 통제 없는 coverage 목적이고, 놓칠 확률 (1 − w)^K 때문에 모드 사이에 보험 질량이 생겨 모드 가중치가 왜곡된다. future work로 "JEPA 같은 피처 WM과 결합하면 모드가 적어 작은 K로 충분"을 제시한다 | C1, C2·C4, 이론 | **필수 인용.** 목적함수는 같은 계열이라고 밝힌다. 차별점: 행동 조건 시각 피처(DINO 패치) 예측, 열거형 모드 coverage, flow 계열 1-step·50-step 대비, 롤아웃 매니폴드, 그리고 "작은 K로 충분한가"에 대한 정량적 답(I(m), 수축) |
| **Why the Third Axis Is Freedom** Bennett, arXiv 2608.05423 (08-05) | Thm 7: R_K(q) = Σ p_j(1 − q_j)^K. 최적해의 정상조건 K p_j(1 − q_j)^(K−1) = λ, 해 q_j* = [1 − (λ/(K p_j))^(1/(K−1))]_+. K=1이면 최빈 모드로 붕괴하고 K가 커지면 균등에 가까워진다. Thm 4: 목표가 균등하면 K ≥ 2에서 최적해도 균등하다 | 명제 3 (평탄화) | **명제 3을 새 결과로 주장하지 않는다.** 인용하고 "학습된 월드모델에서 정량 확인(E14)"으로 낮춘다. 절단 [·]_+(작은 m에서 희귀 모드는 질량 0)도 가져온다. E13의 정확한 위험 추정은 Thm 4(등확률 분기)의 귀결로 설명된다 |
| **ESP: Energy-Score Policy** Makabe, Kim, Matsushita, arXiv 2610.07696 (10-06) | 1-NFE energy-score 정책. hard-min(Forward XM)은 유한 예산에서 strictly proper가 아니라고 지적한다. 등확률 2갈래 TwoBranch에서 match 96.2%(ES) vs 91.9%(XM) vs 97.1%(10-step FM). 서브에이전트 확인으로는 LIBERO에서도 ES가 XM보다 높다 | C1, 이론 | **E15로 대응했다.** 비반복 맥락에서는 ES가 외워 A2 coverage가 0.36이다(IMLE m=16 0.93). E14에서도 calibration 이점이 없다. 장난감에서는 새 맥락이면 ESP 수치를 재현하고, 고정 맥락이면 완전히 외운다. ESP의 결론은 비반복 맥락 월드모델 데이터로 옮겨지지 않는다(명제 4) |
| Beckmann WM arXiv판, 2610.26903 | 본문에 "We evaluate prediction accuracy rather than claim calibrated multimodal coverage"라고 쓴다. Push-T planning은 GPC-RANK이고, IoU 차이 +0.012 [−0.047, 0.071]로 유의하지 않다 | C1, C5 | 모드 coverage 평가라는 차별점이 유지된다. planning 부정 결과의 지지 근거로 쓴다 |
| Gu et al., On Memorization in Diffusion Models (2310.02664) | 조건부 확산 모델이 무조건부보다 더 암기하고, 샘플마다 고유 label을 주면 암기가 급증한다. 검증 맥락 비교는 없다 | C4 | 조건부 암기의 가장 가까운 선행으로 인용한다. 우리 것은 학습/검증 맥락 비대칭을 모드 레이블로 잰다 |

서브에이전트가 초록 수준으로 찾은 추가 인용 후보는 7절에 정리했다. 10-14~30에 새로 나온 논문 중 심각한 위협은 없었다.

## 2. 주장별 분석

### 2.1 결과 암기와 암기 유인 I(m): 가장 안전한 주 기여

**무엇이 새로운가**
- **현상의 정밀한 측정:** 학습 맥락마다 결과가 하나뿐이면 조건부 생성기는 맥락 외형을 열쇠로 삼아 실제로 일어난 결과를 외우고 z를 무시한다. 학습 맥락의 실현 모드 비율은 우연 수준보다 크게 높고, 검증 맥락에서는 우연 수준이다.
  - XM은 "K를 늘리면 과적합이 준다"를 FVD로 보였다. Gu et al.은 조건부 확산의 암기를 보였다. 하지만 모드 레이블로 학습/검증 비대칭을 직접 잰 연구는 이번 재확인에서도 찾지 못했다.
- **메커니즘의 정량화:** 학습 손실 기준으로 암기 해가 일반화 해보다 I(m)만큼 유리하다. I(m)은 m개짜리 무작위 코드북의 기대 왜곡이다.
- **놓칠 확률이 아니라 놓쳤을 때의 왜곡이 중요하다.**
  - XM의 (1 − w)^K와 Bennett의 R_K는 모드를 놓칠 확률로 따진다. 우리 데이터에서 그 계열((1 − 1/K)^m, exp(−m/K))은 초과 암기를 RMSE 0.065~0.068로 설명한다.
  - 모드 간 거리를 반영한 I(m,K)는 0.031이다. 같은 m=8에서 K = 4, 8, 16의 초과 암기가 0.075, 0.083, 0.087로 같은 것은 놓칠 확률로는 설명되지 않는다.
  - **XM·Bennett 대비 가장 날카로운 차별점이다.** 1번 그림(초과 암기 vs I)에 놓칠 확률 계열의 적합을 같이 그리면 한눈에 보인다.
- **처방 비교:** 맥락 노이즈는 precision을 잃고(0.61) ε-거부는 효과가 없다(0.74). m이 가장 깨끗한 손잡이다.
- **목적함수 계열을 넘는 확장 (E15, 사후):** strictly proper한 ES는 암기 유인이 ½E‖Y − Y′‖^β로 m과 무관하다(명제 4).
  - 상대 유인의 순서는 회귀 ≈ ES ≈ 최근접-of-2 > 최근접-of-4 > 최근접-of-16이다. 암기 지수의 순서와 같다: 0.99, 0.87~0.96, 0.96(K=2), 0.75, 0.29.
  - 장난감에서는 고정 맥락의 ES가 완전히 외우고(0.999), 최근접-of-m은 우연 수준이다(0.51~0.52).
  - 맥락을 구별할 특징이 없거나 재방문이 적으면 ES도 외우지 않는다. 즉 유인은 필요조건이다. 열쇠 조건은 실제 환경에서도 미리 적은 예측으로 확인했다(A2-fix: ES 0.27, IMLE m=4 0.26).
- **평가 쪽 함의:** 맥락이 반복되는 벤치마크(A1 Bouncing Shapes, A2-fix: 암기 지수 IMLE m=4 0.26, ES 0.27)에서는 이 실패가 보이지 않는다.

**선행과의 경계**

| 선행 | 그들이 보인 것 | 우리와의 차이 |
| --- | --- | --- |
| XM (2607.27372) | K를 늘리면 과적합이 준다(SSv2 FVD). 데이터가 적으면 end-to-end XM이 더 암기하기 쉽다(서술) | 모드 레이블로 잰 학습/검증 비대칭, 놓친 왜곡 I(m)에 의한 정량 예측, K에 대한 붕괴, 처방 비교 |
| Li, Zhang, Malik 2019 (cIMLE) | 조건당 타깃이 1개면 z가 무시된다. noise encoder 같은 구조적 처방 | 붕괴의 원인을 학습 맥락 결과의 암기로 측정했고, m으로 제어되는 정도를 예측했다 |
| Gu et al. (2310.02664), Kim & Lee (2610.11288) | 조건부 확산에서 조건이 정보적일수록 암기가 쉽다 | 검증 맥락 대조와 표본 매칭 목적의 유인. 같은 예산에서 FM은 암기하지 않았다(0.29) |
| data-copying (Meehan 2020), 확산 복제 (Somepalli 2023, Carlini 2023) | 학습 샘플의 재현과 국소 과대표현 | 조건부이고 모드 레이블로 정확히 측정된다 |
| ESP (2610.07696) | 1-NFE에서 ES가 hard-min보다 낫다(TwoBranch, LIBERO) | 비반복 맥락에서는 반대다. 원인(m과 무관한 유인)과 조건(맥락을 구별할 열쇠 + 재방문)을 장난감으로 분리했고, 열쇠 조건은 A2-fix에서도 확인했다 (E15) |

**리스크와 대응**
- **"그냥 과적합 아닌가":** 회귀도 0.99로 외우니 과적합의 일종은 맞다. 새로운 점은 세 가지다. (a) 확률 생성기의 다양성 붕괴가 이 형태로 나타난다. (b) 유인이 놓친 왜곡 I(m)로 정량화된다. (c) 일반적 정규화로는 precision을 잃는다.
- **직접 측정은 합성 환경(A2)뿐이다.** B는 간접 근거(m=4 퍼짐 9.2 vs m=16 21.8)만 있다. BAIR의 과소분산은 지표상 암기가 아니라서, 지표가 다른 실패와 구별된다는 근거로 쓴다.
- **선형 관계는 I ≲ 0.25에서만 맞는다.** "법칙"이 아니라 "단조 관계 + 2차 요인(포화, log K 용량 효과)"으로 쓴다.
- **"사전 등록":** 예측은 로컬 문서(`theory_m_law.md`, 10-16 16:30)에만 있다. "predictions recorded before running the experiments"로 쓰고, 빗나간 두 점까지 표를 그대로 싣는다.
- **명제 4는 사후다.** A2 ES 시드 0이 외운 것을 본 뒤 유도했다. 유도 뒤에 장난감 실험과 추가 실험을 설계했고, 결과 전에 예측을 적은 것은 추가 실험(5/5 성립)뿐이다. 명제 자체는 "사전 예측"과 섞지 않는다.
- **기하가 한 계열이다.** 모드가 모두 튕김 방향의 호 위에 있다. 일반 진술은 "필요한 m은 I(m)으로 정해진다"이다. 모드가 잘 떨어져 있으면 m이 K에 비례하고, 같은 결과 다양체 위에 있으면 K와 거의 무관하다. 데이터는 두 끝(K=2 분리, K=16 조밀)을 덮는다.
- **이름:** "outcome memorization" 또는 "context memorization"처럼 구체적인 이름을 붙여 확산 암기 문헌과 구분한다.

### 2.2 무작위 코드북 관점: 무엇이 남고 무엇이 빠졌나

| 요소 | 상태 | 논문에서 |
| --- | --- | --- |
| 최근접-of-m = 생성기에서 i.i.d.로 뽑은 무작위 코드북 | 부분 선행. XM은 "candidates are sampled rather than placed"라고 정성적으로 쓰지만 코드북·율-왜곡 용어는 없다. Branch-JEPA Prop. 3은 학습된 head 기준으로 best-of-K = K-means 왜곡이다 | 대응을 명시하고, WTA = 양자화(Rupprecht 2017, Letzelter 2024), R-D(aMCL), XM을 함께 인용한다 |
| 암기 유인 = I(m), 놓칠 확률보다 낫다 | **선행 없음** | 주 기여 (2.1) |
| proper score(ES)의 암기 유인 = ½E‖Y − Y′‖^β, m과 무관 (명제 4) | **수학은 기존**이다(Dawid 2007, Gneiting & Raftery 2007: δ_x까지의 energy distance와 그 entropy). 무조건부 암기 논의는 Li & Malik 2018 §6.2에 있다. ES와 hard-min을 암기 쪽에서 대비한 선행은 **찾지 못했다**(11-01, `literature/04_recheck_2026-11-01.md`) | 2.1의 확장. (a)·(b)는 기존 결과의 귀결로 쓰고, 유인 비교와 실증만 새것으로 쓴다 |
| 고차원 산포 방향 둔감: ES의 최적 산포는 크기(s = a)는 맞지만 방향을 가리지 않는다 (8절 보조 계산, 처음 이름은 "잡음 헤지") | **구성요소는 기존**이다(11-01 확인). s = a는 De Bortoli et al. 2025 Prop 4.1(λ = 1)과 같은 식이다. 고차원 energy distance는 평균과 공분산 trace만 본다(Chakraborty & Zhang 2021). ES는 평균에 비해 분산·의존 구조 오차에 둔감하고(Pinson & Tastu 2013; Scheuerer & Hamill 2015, 둘 다 저차원), 고차원에서는 표본이 매우 크지 않으면 더 둔감하다(Marcotte et al. 2023). 전역장 ES 학습에서 같은 해석이 나왔다(Lang et al. 2026). 2^(β/2−1) 인자와 이 framing은 찾지 못했다 | 기존 결과의 따름정리로 쓰고, 조건(PR ≫ 1)을 밝힌다. 암기가 없는 A2-fix에서도 precision 격차가 남는다는 예측(P2)은 맞았다 |
| 고차원 수축 s*/σ = e_m/√d | **선행 없음** (가우시안 test channel과 같은 형태라는 대응만 고전). XM App. F.2와 "Drifting through Annealed Clustering"의 under-dispersion은 soft·K→∞의 deconvolution으로, 메커니즘이 다르다 | 주 기여. 서술은 "고전 결과의 귀결" |
| 두 regime 통합 (ln m 대 d_eff) | **선행 없음** | 주 기여. XM의 "작은 K로 충분" 추측에 대한 답으로 쓴다 |
| 평탄화 최적해 p_k(1 − q_k)^(m−1) = const | **기존** (Bennett Thm 7, 2608.05423). XM Prop. 3은 정성적으로 다룬다 | 인용 + E14 정량 확인 |
| min-of-N 최적 밀도 ≈ √p (연속, 과분산) | 기존 (Thiede & Brahma 2019) | 인용 |

**BAIR 명제 2의 약한 고리:** d_eff ≈ 29를 m=16 한 점에 맞추고 m=4·64를 맞췄다(0.27 vs 0.22, 0.56 vs 0.53). 보강 2(E16)에서 독립 추정을 시도했지만 실패했다. 회귀 잔차의 PR은 약 1,030이고, 생성기의 맥락별 퍼짐은 PR 1.2~2.5다. 그래서 "파라미터 없는 예측"은 쓸 수 없다. 대신 "퍼짐이 맥락마다 소수 방향으로 붕괴한다(역 water-filling과 같은 모양)"를 탐색 결과로 쓴다.

**평탄화의 실용적 함의 (이 분석에서 계산)**
- 평탄화 식을 거꾸로 풀면 p_k ∝ (1 − q_k)^(−(m−1))이다. 하지만 m이 크면 이 역산은 매우 불안정하다. q의 오차가 로그 오즈에서 약 (m − 1)(1/q_0 + 1/q_1)배로 커진다. m=16이면 약 60배, m=4면 약 12배다.
- E14로 확인하면: m=16의 실측 q_0 = 0.573이면 p_0 ≈ 0.99이고, 공통 치우침 0.04를 빼도 0.88이다(참값 0.787). m=4는 치우침을 빼면 0.795로 거의 맞는다.
- 즉 큰 m의 샘플에는 모드 확률 정보가 거의 남지 않는다. "사후 역보정"을 후속 과제로 쓰기보다 "coverage는 큰 m으로, 확률은 별도 장치로"라고 쓰는 편이 정확하다.
- **E15에서 ES는 그 별도 장치가 되지 못했다.** 비반복 맥락에서는 ES도 외워, 모드 확률이 오히려 다수 모드로 쏠렸다(0.886 vs 실제 0.787, coverage 0.58). 확률을 보정하려면 암기하지 않는 장치가 필요하다(후속 과제).

**덤: 궤적 매칭(N1)의 실패도 같은 관점으로 설명된다.** 궤적 단위 결과는 모드가 2^H개로 늘어 고정된 m에서 I(m)가 커진다. 별도로 검증한 예측이 아니므로 해석으로만 쓴다.

### 2.3 IMLE-WM과 C1 결과: 조합 노벨티, XM 추측의 검증으로 포장

**남는 차별점 (XM, 동시기 1-step WM 대비)**
- **영역:** XM의 월드모델은 Maze2D 목표 조건 궤적 생성이다(Diffuser 대비). 우리는 행동 조건 시각 피처(frozen DINOv2/v3 패치) 예측이다. Beckmann WM, Priming Models, VSF는 모드 coverage를 재지 않는다.
- **비교 대상:** flow 계열 1-step(iMF, Shortcut)과 50-step FM을 파라미터 ±4%로 맞춰 비교했다. XM은 Diffuser·Diffusion Policy와 비교했고, ESP는 10-step FM·XM과 비교했다.
- **평가:** 열거형 모드 coverage/precision, 롤아웃 매니폴드 거리, 판독기 대조.

**근거**
- A2-div: coverage/precision 0.93/0.82 vs FM-50 0.79/0.51, Shortcut 0.66/0.42. 지연은 3.3ms vs 107ms다.
- A1: 프로토콜 2개 × 판독기 2개 모두에서 IMLE가 FM보다 낫고, 차이가 시드 표준편차의 10배 이상이다.
- B: h=16 precision이 FM-50 대비 −6.6 [−8.5, −4.7]px로 계층 CI에서도 유의하다.
- E15: 손실만 ESP식 ES로 바꾼 1-step 대비 A2 coverage/precision이 0.93/0.82 vs 0.36/0.61이다. A1 F1은 7.21 vs 8.95(heatmap 판독기 7.64 vs 33.8)다.

**리스크**
- **ES 기준선은 ESP 기본값과 토큰별 거리 변형 둘이다(E15).** β = 1만 돌렸고, 다른 β와 hard-min과의 혼합은 탐색하지 않았다. 토큰별 변형은 기본값보다 나빴다(A2 coverage 0.22). 그래서 "1-step 생성기 중 최선"이 아니라 "flow 계열 1-step·50-step과 ESP식 ES 1-step 대비"로 쓴다.
- **FM 기준선의 규모:** 같은 heatmap 판독기에서 저희 16→16 FM-50은 12.6이고 FlowWM 논문은 4.31이다. 레시피 채택, 학습 4배, 구조적 매니폴드 이탈로 방어하고, "같은 소규모 계산 예산에서"로 한정한다.
- **B와 BAIR:** B의 recall 우위는 시드 변동을 넣으면 유의하지 않다. BAIR는 분포 지표에서 FM-50에 진다. 범위를 "모드형 불확실성"으로 명시하고, BAIR는 2.2의 예측이 맞은 사례로 제시한다.

**안전한 서술:** "We test whether a small candidate budget suffices for feature-space world models (Gladstone et al., 2026): conditional IMLE on frozen DINO patch tokens, one network evaluation per step, evaluated with enumerable mode coverage."

### 2.4 롤아웃 매니폴드 유지와 판독기 대조 (C3): 부 기여

10-30 재확인으로 바뀐 것은 없다.
- **새로운 것:** raw DINO 공간에서 FM과 IMLE 롤아웃의 매니폴드 이탈을 직접 비교했다. A2 h=32에서 IMLE 1.68, 회귀 1.80, FM-50 6.16, FM-2 6.36이다. FM-50도 FM-2만큼 이탈하므로 적분 오차가 아니다.
- **노이즈 판독기 대조:** 노이즈에 강한 판독기에서 IMLE는 9.0 → 5.3으로 좋아지고 FM은 12.6 → 15.0으로 나빠진다. FM의 불리함은 판독기 강건성 문제가 아니다.
- **판독기가 순위를 바꾼다:** 동시 예측 회귀는 soft-argmax 5.1, heatmap 22.5다. FlowWM의 결정론 기준선을 직접 비판하지 말고 "readout choice can dominate reported gaps"로 쓴다.
- **이미 알려진 것:** 자기회귀 drift(Self Forcing, 10월의 Fréchet/Mix Forcing 등), 생성 latent의 매니폴드 이탈(RAE, VFMF, Latent-Foresight).
- **주의:** B의 best-of-n은 FM이 낫다(67.4 vs 79.1).

### 2.5 평가 프로토콜과 벤치마크: 리스크가 낮은 부 기여

- **A2-div:** 행동 조건, 비반복 맥락, K 조절, 비대칭 확률 변형(E14)을 갖춘 열거형 벤치마크다. 기존의 유일한 열거형 벤치마크(FlowWM Bouncing Shapes)는 행동이 없고 맥락이 반복돼 암기를 잡지 못한다.
- **확률 Push-T:** 숨은 무게중심 4모드, 다중 미래 테스트셋(시작점 128 × 미래 64 × 16 step).
- **분기 위험 과제:** 참 위험 확률을 아는 push 빙판 + 구덩이 과제와 위험 추정 진단(`riskdiag.py`).
- **암기 지수:** 학습 중 조기 경보로 쓸 수 있는 모니터.

### 2.6 순차 매칭 (C6): 효과는 강하지만 노벨티는 약하다

10-30 재확인으로 바뀐 것은 없다.
- **선행:** CAT-K(CVPR 2025 oral), DaD, DAgger.
- **효과:** B h=16 recall이 −14.3 [−19.1, −9.9]px(−33%)다. A1은 best-of-n −24%, A2는 매니폴드 거리만 −24%다. 대신 학습 시간이 3~4배다.
- **위치:** "DaD/CAT-K 계보의 step별 매칭. 궤적 단위 매칭은 코드북이 2^H로 폭발해 실패하므로(N1 기각) step별 분해가 맞고, 선택이 곧 학습 손실이라 판별기나 교사가 필요 없다."
- **롤아웃 상태 ablation (E16, A1 2시드, 사전 예측 3/3):**
  - random(자유 롤아웃, DaD·scheduled sampling 쪽)은 매니폴드 이득을 matched보다 크게 낸다(h=16 1.81 vs 1.92, N0 2.17). 그러나 best-of-16은 N0와 같다(22.9 vs 23.2, matched 17.7).
  - gt(실제 맥락, 같은 수의 매칭 업데이트)는 N0와 같다.
  - 즉 매니폴드는 자기 롤아웃에서, 궤적 coverage는 CAT-K식 선택에서 온다. random은 불확실성을 step마다의 떨림으로 내고 갈래로 쌓지 못한다(h=16 퍼짐이 실제의 0.52배, matched 1.11배).
  - "CAT-K와 무엇이 다른가"에는 이렇게 답한다: 기법은 다르지 않다. CAT-K의 선택이 nearest-of-m 손실에서 바로 나오고, 그 효과를 매니폴드와 coverage로 나눠 보였다. 노벨티는 여전히 낮다.

### 2.7 planning (C5): 노벨티 주장 없음

- **속도:** 같은 성공률에서 step당 4.0s다. FM-50(200s)의 약 1/50, FM-2(8.5s)의 약 1/2이다.
- **위험 추정 문구를 고친다.** 빙판 한 step 구덩이 확률은 IMLE 0.359, FM-50 0.309, FM-2 0.138이다(참값 0.330). 출발점별 절대 오차는 IMLE 0.078, FM-50 0.086이다. "FM-50 수준의 위험 추정을 1 NFE로 한다(같은 비용대의 FM-2는 2.4배 과소추정)"로 쓰고, 등확률 분기로 한정한다(Bennett Thm 4). 큰 m은 희귀한 위험을 과대추정한다(Thm 7).
- **성공률 부정 결과는 문헌과 일치한다:** Cheung & He(2610.19375), Resolution Collapse(ICLR 2027 제출), Beckmann WM(GPC-RANK 차이 비유의). 서브에이전트가 찾은 EFE 분석(2610.28011)과 best-of-N headroom(2610.37196)도 같은 방향이다.

## 3. 추천 스토리와 기여 문장

| | A. 방법 중심 | **B. 질문 중심 (추천)** |
| --- | --- | --- |
| 한 줄 | "순전파 1회로 여러 미래를" | "1-step 최근접-of-m 월드모델에 m이 얼마나 필요하고, 언제 어떤 m으로도 부족한가" |
| 장점 | 주 표가 강하다(A1·A2) | XM이 남긴 질문에 정량적으로 답한다. BAIR와 평탄화가 실패가 아니라 예측된 결과가 된다. 실험 전에 기록한 예측이 차별점이다 |
| 약점 | XM·ESP와 정면 비교되고, 규모·planning·BAIR가 약점으로 읽힌다 | 이론 수학이 고전이라 서술을 정확히 해야 한다 |

제목 후보:
- "How Many Samples Does a One-Step World Model Need? A Random-Codebook Account of Nearest-of-m Training"
- "When Does Min-of-m Suffice? One-Step World Models in Frozen DINO Space"

**기여 문장 초안 (영문)**
1. We test whether a small candidate budget suffices for feature-space world models: conditional IMLE (nearest-of-m) on frozen DINOv2/v3 patch tokens, one network evaluation per step. Under mode-like uncertainty it covers enumerated futures better than 50-step flow matching, one-step flow generators (iMF, Shortcut) and ESP-style energy-score generators at matched parameters, with about 30× lower latency than 50-step flow matching. On real robot video it does not, as our analysis predicts.
2. We identify outcome memorization in non-repeating contexts and show that it is governed by the expected miss *distortion* I(m) of a random codebook, not by the miss *probability*. Across K ∈ {2, 4, 8, 16}, excess memorization collapses onto I(m), and 7 of 9 predictions recorded before the runs fall within their intervals. The same accounting explains why a strictly proper energy score, which recent work prefers over hard-min (Makabe et al., 2026), memorizes in this regime while nearest-of-m does not: its incentive is half the expected outcome spread, independent of m. As predicted before the runs, it stops memorizing when contexts carry no identifying appearance, and a per-token variant memorizes as well.
3. The same codebook view explains under-dispersion in high-dimensional continuous outcome spaces (spread ∝ e_m/√d, the form of the Gaussian rate–distortion test channel), which delineates when one-step nearest-of-m training is the right tool. On real robot video the samples spread along only one to three context-specific directions, the reverse water-filling shape of the same picture. We also confirm the known flattening of mode probabilities (Bennett, 2026) quantitatively in a learned world model.
4. Autoregressive IMLE rollouts stay on the token manifold while flow-matching rollouts drift regardless of step count. Per-step matching on self-rollouts (in the DaD/CAT-K lineage) further reduces long-horizon error by a third on stochastic Push-T. An ablation of the fed-back state separates two effects: any self-generated context keeps rollouts on the manifold, but only feeding back the candidate nearest the truth, which in nearest-of-m training is the loss's own argmin, improves long-horizon coverage.

**핵심 그림 우선순위**
1. 초과 암기 vs I(m,K) 붕괴 그림. 놓칠 확률 계열의 적합, 실험 전 예측 구간, 빗나간 두 점을 함께 표시한다(논문의 핵심 그림).
2. "m 손잡이" 패널: m에 따른 coverage, 암기 지수, 모드 점유율(Bennett 예측선 포함), 결정론 오차를 한 줄에 둔다.
3. regime 그림: BAIR 퍼짐/오차 비 vs m과 한 점에 맞춘 곡선. 맥락별 퍼짐 차원(PR 1.2~2.5)을 함께 싣는다(E16).
4. 매니폴드 거리 vs horizon (A2, A1, B).

## 4. 쓰지 말 표현과 대체 표현

| 쓰지 말 것 | 이유 | 대체 |
| --- | --- | --- |
| first one-step stochastic / min-of-K world model | XM Explorative World Model, Beckmann WM, Priming Models, VSF, DriftWorld | conditional IMLE on frozen DINO tokens, action-conditioned, evaluated with enumerable mode coverage |
| first multi-hypothesis prediction in feature space | Vondrick 2016, Rupprecht 2017, Bhattacharyya 2018 | 범위를 한정한 조합으로 서술 |
| we show that more samples reduce memorization (최초 관찰처럼) | XM (SSv2) | we quantify *why*: the incentive is the miss distortion I(m) |
| we derive the optimal mode weights / flattening | Bennett Thm 7 | consistent with Bennett (2026), confirmed within 0.01 |
| a new law / new theory of IMLE | 율-왜곡·양자화의 귀결이다 | a consequence of classical quantization results, applied to nearest-of-m training |
| m ≈ 2K (또는 m ≥ 4K) | E12에서 기각 | m is set by the quantization error I(m) |
| the best one-step generator | ES는 ESP 기본 변형과 토큰별 변형만 비교했다 (E15) | better than flow-based one-step generators (iMF, Shortcut) and ESP-style energy-score generators |
| proper scores memorize / energy score is worse than hard-min (일반 진술) | 비반복 맥락에서, 맥락을 구별할 수 있고 재방문이 많을 때만 확인했다. 새 맥락에서는 ES가 낫고(장난감, ESP), 외형을 고정하면 ES도 외우지 않는다(A2-fix) | in non-repeating contexts, an energy-score objective memorizes while nearest-of-m does not |
| the energy score cannot discriminate in high dimensions (Pinson & Tastu, 2013) | 두 문헌 모두 저차원이고, 평균 오차는 잘 판별한다(11-01 확인) | `literature/04_recheck_2026-11-01.md` 2절의 수정 문장: 평균에 비해 분산에 덜, 의존 구조에 약하게 민감하다(P&T 2013; S&H 2015). 고차원에서는 표본이 매우 크지 않으면 더 둔감하다(Marcotte 2023). 고차원 energy distance는 평균과 trace만 본다(Chakraborty & Zhang 2021) |
| the energy score hedges with isotropic noise | 최적 산포의 크기(s = a)는 바른 보정이다(De Bortoli 2025 Prop 4.1) | the energy score is insensitive to the direction (covariance structure) of the spread in high dimensions |
| improves planning success | E5, E13에서 기각 | 50× faster planning at equal success |
| IMLE estimates risk most accurately | FM-50과 비슷하고, 등확률 분기에서만 성립 | matches FM-50's risk estimates at 1 NFE for equiprobable branches |
| calibrated / matches the true distribution | E14 평탄화 | covers modes; mode probabilities flatten as predicted |
| outperforms FlowWM | 규모와 프로토콜이 다르다 | same readout, same protocol, matched small-scale compute |
| pre-registered (형식 용어) | 외부 타임스탬프가 없다 | predictions recorded before running the experiments |
| novel sequential training | CAT-K, DaD | per-step matching in the DaD/CAT-K lineage |

## 5. 예상 리뷰 공격과 대응

| 공격 | 대응 | 상태 |
| --- | --- | --- |
| XM이 이미 min-of-K 1-step WM을 했다 | 목적함수는 같은 계열이라고 먼저 밝힌다. 우리 기여는 XM의 "작은 K로 충분" 추측에 대한 통제된 검증과 정량 이론(I(m), 수축)이다 | 서술로 해결 |
| ESP처럼 energy score가 더 낫지 않나 | E15: 비반복 맥락에서는 ES가 외워 A2 coverage가 0.36이다(IMLE 0.93). 맥락이 반복되는 A1에서도 진다(F1 8.95 vs 7.21). 장난감에서는 새 맥락이면 ES가 낫다는 ESP 결과를 재현하고, 차이를 명제 4로 설명한다. 토큰별 거리로 바꿔도 외우고(0.97), 외형을 고정한 A2-fix에서는 외우지 않지만 precision이 낮다(0.81 vs 0.95) | 해결 (β, 혼합 탐색은 남음) |
| ES 구현이 약한 것 아닌가 | 장난감에서 ESP의 TwoBranch 수치를 재현했다(match 0.94 vs 0.962). A2·A1에도 같은 추정량을 썼다. 외우지 않는 A2-fix에서는 같은 구현이 coverage 0.95에 이른다(2시드) | 대부분 해결 (토큰별 변형 확인, β·혼합 미탐색) |
| 평탄화는 Bennett의 정리다 | 인용하고 확인 결과로만 쓴다 | 서술로 해결 |
| 이론은 율-왜곡의 재유도다 | 대응을 먼저 밝힌다. 새로운 부분은 놓친 왜곡 vs 놓칠 확률, 유한 m 수축, regime 통합, 정량 검증이다 | 서술로 해결 |
| 암기는 그냥 과적합이다 | 2.1절의 세 가지 | 서술로 해결 |
| 합성 환경뿐이다 | A1은 공개 벤치마크, B는 시뮬레이터, BAIR는 실제 영상이다. BAIR는 예측된 경계로 제시한다 | 일부 남음 |
| FM이 덜 튜닝됐다 (FlowWM 4.31) | 레시피 채택, 학습 4배, 매니폴드 이탈의 구조성, 같은 판독기·프로토콜 비교 | 일부 남음 (규모) |
| BAIR 예측은 파라미터를 맞춘 것이다 | 맞다. 보강 2(E16)의 독립 추정이 실패했다(잔차 PR 약 1,030). 맞춘 값이라고 먼저 밝히고, 퍼짐이 맥락마다 1~2.5개 방향으로 붕괴한다는 탐색 결과로 기전을 보인다 | 서술로 대응 (정량 예측은 포기) |
| 순차 매칭은 CAT-K다 | 맞다. 계보를 명시한다. E16 ablation으로 이득을 매니폴드(자기 롤아웃)와 coverage(CAT-K식 선택)로 나눠 보이고, nearest-of-m에서는 그 선택이 손실의 argmin이라는 점으로 위치를 잡는다 | 서술과 ablation으로 해결 (노벨티는 낮음) |
| planning 이득이 없다 | 속도와 위험 추정. 최근 문헌도 같은 decoupling을 보고한다 | 정직하게 한계로 |

## 6. 노벨티를 지키는 저비용 보강 (선택)

실험 종료 결정을 뒤집을 정도는 아니다. 다만 10-30 재확인 이후 보강 1은 "있으면 좋은 것"이 아니라 리뷰 방어에 거의 필요한 항목이 됐다. 비용은 E1·E12의 run 시간에서 추정했다. **보강 1은 10-31에 E15로, 보강 2·3은 10-31~11-01에 E16으로 실행했다.**

| 우선 | 항목 | 막는 공격 / 얻는 것 | 비용 (추정) |
| --- | --- | --- | --- |
| ~~1~~ **완료 (E15, 10-31)** | **energy-score 1-step 기준선 (Engression·ESP식):** 같은 backbone, 손실만 교체. A2-div K=4 3시드 + m=16 2시드, E14 2시드, A1 대조 2시드, CPU 장난감 실험, 추가로 A2-fix 3모델 × 2시드와 토큰별 ES 2시드(시드 보강은 E16) | 결과: ES는 비반복 맥락에서 외워 A2 coverage가 0.36이다(IMLE m=16 0.93). E14에서도 calibration 이점이 없다(0.886 vs 실제 0.787). 예상했던 맞교환 대신 "ES가 둘 다 잃는다"가 나왔다. 추가 실험의 예측 다섯 개도 모두 맞았다. 열쇠가 없으면 ES도 외우지 않고, 토큰별 거리로도 외운다. `E15_report.md` | 약 15.0 GPU-h (추가 실험 5.3 포함) |
| ~~2~~ **완료, 실패 (E16)** | **BAIR d_eff 독립 추정:** 회귀 모델의 잔차를 PCA해 participation ratio로 유효 차원을 잰다. 명제 2에 넣어 m=4·16·64 세 점을 파라미터 없이 예측한다 | 결과: 잔차 PR이 약 1,030으로 29의 35배다. 독립 예측은 실패했고, 생성기의 맥락별 퍼짐이 PR 1.2~2.5라는 탐색 결과를 얻었다. `E16_report.md` | GPU 수 분 |
| ~~3~~ **완료 (E16)** | **순차 매칭의 롤아웃 상태 ablation** (matched / random / GT), A1 7M 2시드 | 결과: 사전 예측 3개 모두 성립. random은 매니폴드 이득만 낸다(h=16 1.81, matched 1.92, N0 2.17). coverage 이득은 matched만 낸다(best-of-16 17.7, random 22.9, N0 23.2). gt는 N0와 같다. `E16_report.md` | 약 10.8 GPU-h |

보강 2는 결과가 29에서 크게 벗어나면 명제 2의 근거가 약해지는 위험도 있다. 그 경우에도 "등방 가우시안 근사의 한계"로 정직하게 보고할 수 있다. E16에서 실제로 그렇게 됐다.

## 7. 필수 인용 추가분 (10-30)

본문을 직접 확인한 것과 서브에이전트가 초록 수준으로 확인한 것을 구분했다.

| 논문 | ID / venue | 확인 수준 | 쓰임 |
| --- | --- | --- | --- |
| Explorative Modeling (Gladstone, Ji, Du) | arXiv 2607.27372 | 본문 | C1·C4·이론. 같은 목적함수 계열, 작은 K 추측, F.2 coverage 목적 |
| Why the Third Axis Is Freedom (Bennett) | arXiv 2608.05423 | 본문 | 평탄화(Thm 7), 균등 목표(Thm 4), min-of-N 계보 |
| ESP: Energy-Score Policy (Makabe, Kim, Matsushita) | arXiv 2610.07696 | 본문 | proper score 1-NFE, hard-min 비교 |
| Beckmann World Models (Lee, Agarwal, Suh) | arXiv 2610.26903 | 본문 (서브에이전트) | 동시기 1-step WM, planning 비유의 |
| On Memorization in Diffusion Models (Gu et al.) | arXiv 2310.02664, TMLR 2025 | 본문 | 조건부 암기 |
| Fan, Su, Guibas, Point Set Generation Network | CVPR 2017 | Bennett 인용으로 확인 | 조건부 min-of-N 원형 |
| SOG (Self-Organizing Generative Models) | XM 참고문헌 [35] | XM 인용으로 확인 | 조건부 min-of-K |
| Somepalli et al. / Carlini et al. / Kadkhodaie et al. | 2212.03860 (CVPR 2023) / 2301.13188 (USENIX Sec 2023) / 2310.02557 (ICLR 2024) | 서브에이전트 | 확산 암기와의 구분 |
| Kim & Lee, conditional diffusion memorization | arXiv 2610.11288 | 초록 | 조건부 암기 이론 |
| Somepalli et al. 2023b, He et al. 2025 | 2305.20086, 2505.05787 | 초록 | 조건부 복제, diffusion policy의 lookup 동작 |
| Compote | arXiv 2610.37005 (= ICLR 2027 FHtrHTQA3C) | 초록 | annealed WTA, max-entropy 양자화 |
| Branch-JEPA v3 | arXiv 2607.05238 | 본문 (서브에이전트) | best-of-K = K-means 왜곡(Prop. 3), ES 학습 비교 |
| EFE dissection / best-of-N headroom | 2610.28011 / 2610.37196 | 초록 | planning 부정 결과의 지지 |

"Better Samples, Wrong Odds", "Drifting through Annealed Clustering", IMLE-LLM은 이번에도 본문을 열지 못했다(OpenReview 403). 초록 기준으로는 min-of-m이나 IMLE의 m 분석이 없다.

**11-01 추가:** 명제 4와 ES 판별력 관련 인용은 `literature/04_recheck_2026-11-01.md`에 확인 수준과 함께 정리했다. 핵심은 Dawid 2007, Gneiting & Raftery 2007, Li & Malik 2018 §6.2, De Bortoli et al. 2025, Chakraborty & Zhang 2021, Pinson & Tastu 2013, Scheuerer & Hamill 2015, Marcotte et al. 2023, Lang et al. 2026이다.

## 8. 최종 기여 배치

| 논문 위치 | 내용 |
| --- | --- |
| 도입 | min-of-K 흐름(IMLE → XM → Bennett, ESP)과 XM의 "작은 K로 충분" 추측. 이 논문의 질문: 얼마나 필요하고 언제 부족한가 |
| 기여 1 (통제 비교) | DINO 토큰 위 1-step IMLE WM, C1(A1·A2 강함, B precision), flow 계열 1-step·50-step과 ESP식 ES 1-step 기준선(E15), 지연 |
| 기여 2 (현상 + 메커니즘) | 결과 암기, 암기 지수, 놓친 왜곡 I(m) vs 놓칠 확률, 실험 전 예측, 처방 비교, 반복 맥락 벤치마크의 맹점, energy score와의 대비(명제 4, 사후), 장난감 메커니즘 분리와 A2-fix 열쇠 조건(사전 예측) |
| 기여 3 (경계) | 고차원 수축 e_m/√d와 BAIR(d는 한 점에 맞춤, 퍼짐은 맥락마다 소수 방향으로 붕괴), regime 통합. 평탄화는 Bennett 정리의 정량 확인 |
| 기여 4 (롤아웃) | 매니폴드 유지, 판독기 대조, 순차 매칭(DaD·CAT-K 계보)과 롤아웃 상태 ablation(E16: 매니폴드는 자기 롤아웃, coverage는 matched 선택) |
| 실험 절 부수 결과 | planning 속도, 위험 추정(등확률 분기 한정) |
| 한계 | 성공률 향상 없음, BAIR 분포 지표, 합성 위주, 평탄화(보정 필요), FM 규모, ES 변형 일부 미탐색(β, 혼합), 명제 4는 사후 |
| 부록 | N1·N2·N3·N4 부정 결과, MeanFlow 이식 기록, 판독기 비교, 벤치마크 설계 |

# 분석 작업자(an.1) 계획

- 작성: 2026-11-22T17:58Z (본 실행 시작 17:53Z 직후). 기준 설계: `main/revision/DESIGN_r1.md` v1.0 0절, 1절의 측정 점검,
  2절의 NFE별 암기 지수, 4절 분석 a–h. 가설, 지표, 문턱, run, seed는 바꾸지 않는다.
- pod: an.1 (NVIDIA A40, RAM 상한 50 GB, `/workspace`는 네트워크 파일 시스템). 그룹 an 상한 6달러(0.59달러/h, 17:20Z부터).
  80%(4.8달러)는 약 01:28Z에 닿는다. 그래서 모든 GPU 작업을 약 00:30Z까지 끝내고 결과를 당겨 온 뒤 조정자에게 pod 삭제를 요청한다.

## 0. 준비와 점검 결과

- Drive 복원: tier1(코드, run, 체크포인트) 17:37Z, bounce_div_k{4,2,8,16}, bounce_div_k2_p80 17:49Z. push는 계획과 위험 진단이
  `stats.pt`, `probe.pt`만 쓰므로 작은 파일만 받았다(토큰 10 GB 생략). pusht_b는 `stats.pt`, `probe.pt`, `mf_test_v2.npz`,
  `test_meta.npz`, `test_tokens.npy`만 받았다. bounce_k4(새로움 비율용)는 따로 받는 중.
- 코드: tier1 압축 해제 뒤 맥북 `main/code/`를 밀어 넣었다(tier1이 code/를 덮으므로). 환경: Python 3.10.12, torch 2.6.0+cu124,
  transformers 4.57.6, numpy 2.2.6. `bdiag.py`, `bstep.py`, `logs/bair_diag.py`의 `/home/yonghwi/imle-wm`은 pod 사본에서만
  sed로 `/root/imle-wm`으로 바꿨다.
- **스모크 테스트(설계 0절) 통과.** `zz_smoke_e1_a2_imle16_s0`(e1_a2_imle16_s0 복사본)을 `python -m imlewm.evaluate`로 평가:
  `stoch.cov_pos` 0.93812(기준 0.94125, 차 −0.0031), `stoch.prec_pos` 0.83094(기준 0.8314, 차 −0.0005). 허용 0.02 안.
  원자료: `raw/an.1/out/smoke/eval_zz_smoke_e1_a2_imle16_s0.json`.
- **r1_memo 검증 통과.** e2_k4_m4_s0에서 `check.memo_model(n_ctx=200, n=16, seed=0)`과 `r1_memo.measure`를 같은 torch 시드(123)로
  돌려 train·test의 cov, prec, realized_share, share0, n이 모두 정확히 같다(최대 차 0). 원자료: `b_grid/validate_e2_k4_m4_s0_nfeNone_200x16.json`.

## 1. 구현 (새 모듈 `main/code/imlewm/r1_*.py`, 기존 모듈은 바꾸지 않음)

| 모듈 | 분석 | 내용 |
|---|---|---|
| `r1_util.py` | 공통 | 원자적 JSON 쓰기, memo_model과 같은 문맥 선택(같은 필터, seed 0 순열), 분기 위치, 분기 프레임 렌더링과 DINO 인코딩 |
| `r1_memo.py` | b, c, C2, 2a | memo_model과 같은 문맥·같은 `predict` 호출 순서(그래서 같은 시드면 같은 표본). (i) 지수 (ii) 허용 오차 없는 지수 (iii) held-out 기준 초과 (iv) 판독기 없는 간격 |
| `r1_calib.py` | C1 | 학습 문맥 200개(모니터의 문맥)에 (a) 실현된 다음 프레임 16장 (b) K개 참 분기 지도를 4장씩 넣어 같은 판독·배정. arc 데이터 지원 |
| `r1_offset.py` | d | bounce_div_k2_p80, 모드 점유율 측정의 문맥(400개, 두 분할)에서 참 분기 지도를 점유율 q로 섞어 배정 |
| `r1_itok.py` | a | `stoch_branch_tokens.npy`로 토큰 공간 왜곡 I_tok(m,K), GM 변환판, 격자 16점 재적합 |
| `r1_planrisk.py` | e | plan.run과 같은 코드로 계획기 재실행 + 기록, 실행된 행동열과 선택된 계획에서 모델·시뮬레이터 구덩이 확률, 마지막 프레임 기준 위험 진단 |
| `r1_diag.py` | f | 새로움 비율, Table 4 참 미래 퍼짐, 원래 진단 스크립트(bstep.py, bdiag.py) 실행과 출력 보존 |
| `r1_samples.py` | h | Table 2의 시드 0 체크포인트로 학습 문맥 하나와 held-out 문맥 하나에 16개 표본 |
| `r1_timing.py` | g | export(구성 추출)와 run(유휴 A5000에서 지연과 학습 step 시간) |

### 설계 문장을 구현하면서 정한 것 (해석이 필요한 곳)

- **b의 (i)–(iv).** (i)은 memo_model과 똑같은 계산이고 문맥 400개 + 400개, 표본 64개(seed 0, torch 시드 123).
  (ii) 각 표본을 허용 오차 없이 가장 가까운 분기에 배정한 실현 모드 점유율. (iii) `(train − held-out) / (1 − held-out)`과
  차이 `train − held-out`(독립 부트스트랩 구간)을 함께 낸다. (iv) 문맥마다 64개 표본 중 실현된 다음 프레임(표준화 토큰)까지의
  최소 RMS 거리, 그 평균의 train − held-out. (i)과 같은 문맥(분기가 겹친 문맥 제외)에서 잰다.
- **a의 GM 척도.** 학습의 `Robust`는 배치마다 "m개 후보 중 최소 RMS 거리"의 중앙값을 EMA로 추적하지만 체크포인트에 저장되지
  않는다. 그래서 각 격자 run의 EMA 체크포인트로 학습 전이 512개(무작위, 128개씩 4배치)에서 같은 양을 다시 재고 배치 중앙값의
  평균을 c로 쓴다(두 시드 평균). 민감도로 c/2, 2c도 낸다. 로그의 마지막 match_d(배치 평균)도 함께 기록한다.
  plain은 표준화 분기 지도 사이 평균 제곱 차(매칭 거리), gm은 RMS 거리에 ρ(d)=2u²/(u²+4), u=d/(2c).
  적합은 원고처럼 (K,m)=(2,2)를 뺀 16점, y는 모니터 초과 암기(로그 마지막 memo_train의 두 시드 평균). 표본 내 RMSE와
  K 하나 빼기 RMSE를 함께 낸다. b의 400×64 지수로 만든 y도 병기한다.
- **c의 설정.** 원고 Table 2의 지수는 모니터 설정(학습 문맥 200개, 표본 16개)이므로 같은 설정으로 각 모델의 NFE에서 다시 잰다
  (FM 50·2, iMF 1, Shortcut 1). 같은 시드의 NFE 20 값도 함께 재서 NFE의 효과만 분리한다. Table 8 iMF 행은 원래
  `check.memorization`(문맥 300개, 표본 16개)에서 왔으므로 같은 설정으로 NFE 1과 20을 잰다. FM-2, iMF, Shortcut은 400×64도 낸다.
- **d의 혼합.** 모드 점유율은 memo_model의 share0(문맥별 유효 표본 중 모드 0 비율의 평균)이다. 참 분기 지도는 복사본이 같은
  판독을 받으므로 (가) q로 가중한 정확한 혼합, (나) 문맥마다 round(64q)장과 나머지, (다) 각 문맥의 실제 다음 프레임(데이터 빈도)
  세 가지를 넣는다. 측정 − 참값을 train·test 두 분할에서 낸다.
- **e의 "실행된 행동열".** MPC는 계획(T=5)의 첫 행동만 실행하므로 두 가지를 모두 잰다. (1) 설계 문장 그대로: 에피소드마다 실제로
  실행된 행동열 전체를 시작 상태에서 열린 루프로 재생(모델 16표본, 시뮬레이터 256회). (2) 매 계획 step에서 계획기가 고른
  T=5 계획(CEM 평균, 실행 때처럼 |a|≤1로 자름)을 그 시점의 실제 상태에서 비교. 모델 쪽 기준은 마지막 상상 프레임(계획 비용과 같음)과
  임의 프레임(위험 진단과 같음) 둘 다. 계획기 자체는 plan.run을 그대로 복사했고 기록만 더했다(GPU가 달라 궤적은 비트 단위로
  같지 않다. 성공률은 원래 값과 나란히 보고한다). 직선 밀기 위험 진단은 riskdiag.main과 같은 난수 호출로 두 기준을 같은 표본에서 낸다.
- **f.** 새로움: check.novelty(bounce_div_k4, bounce_k4). 참 미래 퍼짐: mf_test_v2.npz에서 FlowWM 오차 단위(1..h step,
  T 꼭짓점 8개의 L2 평균)의 쌍별 평균(서로 다른 쌍과 n² 전체 둘 다)과 bdiag.py 정의(블록 중심의 평균까지 거리). 접촉 오차:
  bstep.py를 그대로 돌려 출력 저장(reg 12k/36k, FM-2/FM-50, IMLE-WM m=4/16, 시드 0). BAIR는 데이터 30 GB라 시간이 남을 때만.
- **h.** 문맥은 눈으로 고르지 않는다: memo_model 순열 순서에서 K개 분기가 서로 다르고 6 px 이상 떨어진 첫 문맥(train, test 각각).
- **g.** 지연은 evaluate.latency와 같은 방식(배치 1, 5회 예열 + 50회 측정)을 5회 반복. 학습 step 시간은 각 run의 원래 인자로
  train.py를 400 step 돌려 log의 t(400) − t(200)을 200으로 나눈다. 데이터는 같은 모양의 무작위 토큰(값은 시간과 무관).

## 2. 실행 순서와 예상 시간 (an.1)

- 동시에: 계획기 재실행 작업자 3개(시드별: IMLE-WM m=16 → 회귀 → FM-2 순), 주 작업 1개(`jobs/an_main.sh`: 검증 → b → a → c → d →
  C1 시험(표준 기하) → h → f → 위험 진단). 원래 계획 시간은 A5000 기준 IMLE 약 1 h, FM 2.2–3 h, 회귀 9–13분(모델당)이라 e가
  가장 길다. 첫 step 시간을 재서 예산 안에 9개를 다 못 끝낼 것 같으면 남은 FM 재실행을 멈추고 그 사실을 보고한다.
- b, a, c, d, h, f는 합쳐 1시간 안팎으로 예상(GPU를 계획기와 나눠 씀).
- 비용 예상: 17:20Z–00:30Z 약 7.2 h × 0.59 = 약 4.2달러(상한 6달러의 70%).

## 3. 조정자에게 넘길 것

- C1/C2(x1.1, 두 기하), r1_fm36k_s*의 NFE 50/2 암기 지수(x2.1), a의 arc90 부분(x1.1), g(유휴 A5000)의 명령은 REPORT.md 끝에 적는다.

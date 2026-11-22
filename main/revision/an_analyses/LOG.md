# 분석 작업자(an.1) 기록 (UTC)

- 2026-11-22T17:28Z tier1 압축 해제 시작(조정자가 띄운 fetch). 17:37Z tier1 끝, bounce_div_k4부터 데이터 받기.
- 17:38Z tier1이 끝난 직후 코드 push(`sync.sh an.1 push main/code/ ...`)와 환경 구성(`setup_pod.sh env`)을 데이터 받기와 동시에
  시작(코드는 tier1 뒤여야 한다는 조건만 지킴). push는 작은 파일 6개, pusht_b는 작은 파일과 test_tokens만 따로 받음(`data/push`
  폴더를 먼저 만들었으므로 fetch가 push 토큰 10 GB를 건너뜀).
- 17:41Z 환경 완료: torch 2.6.0+cu124, CUDA 12.4, A40 1장, transformers 4.57.6, numpy 2.2.6.
- 17:49Z fetch done(bounce_div_k{4,2,8,16}, bounce_div_k2_p80).
- 17:51Z r1 모듈 push, `bdiag.py`, `bstep.py`, `logs/bair_diag.py`의 `/home/yonghwi/imle-wm`을 pod 사본에서 `/root/imle-wm`으로 sed.
  bounce_k4(새로움 비율용) 받기 시작(stoch_branch_tokens 제외).
- 17:52Z 스모크 테스트 통과: cov_pos 0.93812(기준 0.94125), prec_pos 0.83094(기준 0.8314).
- 17:53Z 본 실행 시작: 계획기 재실행 작업자 3개(`jobs/e_replay.sh`), 주 작업(`jobs/an_main.sh`).
- 17:54Z r1_memo 검증: e2_k4_m4_s0, 모니터 설정(200×16), check.memo_model과 모든 값 동일(최대 차 0). b 시작.
- 18:05Z 계획기 재실행 첫 step(활성 100개)이 726 s(GPU를 작업 4개가 나눠 씀). `predict` 512개 배치 한 번이 540 ms(경합 상태),
  cudnn.benchmark는 차이 없음(`jobs/bench_predict.py`). 9개 재실행의 총 GPU 일은 약 4.6 h(전용 GPU 기준)로 추정.
- 18:06Z 재실행 작업자 셸 3개를 멈추고(돌던 IMLE-WM 재실행 python 3개는 계속) `jobs/e_sched.sh`로 교체: IMLE-WM 3개가 끝나면
  회귀 3개를 함께, 그다음 FM-2 s0, s1, s2를 하나씩(총 GPU 일은 같고, 시드 하나씩 완결된 결과가 나온다).
  `/workspace/nom/STOP_E`가 있으면 남은 FM 재실행을 건너뛴다.
- 18:07Z g용 구성 추출: `python -m imlewm.r1_timing export` → `main/code/imlewm/r1_timing_configs.json`(16개 구성, pod에서 당겨 옴).
- 18:08Z 주 작업 셸을 `jobs/an_main2.sh`로 교체(b의 python은 계속): b 다음 a → d → h → c(모니터 설정) → C1 시험 → f → 위험 진단 →
  c 400×64 순. FM-50의 무거운 항목(위험 진단 NFE 50)은 맨 끝 선택 항목으로, e14 FM-50 교차 점검은 뺐다.
- 18:14Z bounce_k4 받기 완료. an.1 볼륨은 130 GB, 사용 80 GB라 BAIR(30.2 GB) 받기 시작(f의 BAIR 항목, 시간이 남을 때).
  `jobs/f_bair.sh`가 받기가 끝나면 원래 `logs/bair_diag.py`를 e8_bair_imle4_s0, e8_bair_imle16_s0, e10_bair_m64_s0에 돌린다.
- 참고: an.1의 terminate-after는 2026-11-23T07:17Z.
- 18:18Z 재실행 step 1도 약 650 s(활성 100개, GPU 4–5개 작업이 나눠 씀). IMLE-WM 재실행 3개는 약 20:00Z 끝으로 추정.
  b는 체크포인트당 약 125 s(경합), 14/34 완료.
- 18:20Z g 스크립트 기능 시험(`r1_timing run --only "A2 Regression"`, an.1, 바쁜 GPU라 값은 쓰지 않음): 지연 측정과 400 step
  학습 둘 다 정상 동작(`g_timing_functest/timing.json`).
- 18:21Z 분석 몫의 GPU 비중을 늘리려고 `jobs/an_side.sh`(d, h, C1 시험, f)와 b의 두 번째 프로세스(K=16 run을 역순으로)를 추가로
  띄움. an_main2.sh는 이미 있는 출력을 건너뛴다.
- 18:25Z d 완료: 참 분기 지도 혼합(q = 0.5, 0.6, 0.7, 0.787, 0.8)과 실제 다음 프레임 모두 측정 − 참값 0.0000(train, test).
- 18:26Z h 완료(첫 문맥: train t=7412, test t=1855). C1 시험(표준 기하 bounce_div_k4): 천장 1.000, 바닥 0.250.
- 18:28Z f: 새로움 비율 1.921(bounce_div_k4), 0.354(bounce_k4). bdiag.py의 GT spread가 h=8, 16에서 30.8, 48.3으로 Table 4와
  일치(정의는 캡션의 "쌍별 거리"가 아니라 블록 중심과 미래 평균 사이 거리, 128개 중 32개 시작점).
- 18:33Z BAIR 받기 완료(30.2 GB). bair_diag.py 완료(e8_bair_imle4_s0, e8_bair_imle16_s0, e10_bair_m64_s0).
- 18:31Z h에 같은 규칙의 둘째·셋째 문맥 추가(`r1_samples --skip 1/2`, 결과 `samples_ctx2.json`, `samples_ctx3.json`).
- 18:39Z 재실행 step 2(활성 94–96개) 약 2,155 s. IMLE-WM 재실행 끝은 약 20:15Z로 다시 추정. b 26/34.
- 18:49Z b 완료(34/34, 두 프로세스). 400×64 지수는 모니터 지수와 |차| ≤ 0.015(두 시드 평균), held-out 지수는 모든 칸에서 (0.94–1.08)/K.
  a 시작(GM 척도 재측정 + I_tok).
- 18:46Z an_side 완료. 접촉 오차(bstep.py, 시드 0): 참 블록 이동 15.9 px, 접촉 step 블록 오차 평균 회귀 11.7(12k)/11.2(36k),
  FM-2 12.5, FM-50 12.6, IMLE-WM m=4 12.3, m=16 12.3 px(원고 12.4, 12.6, 12.2, 12.3과 0.1 px 이내).
- 18:56Z a 완료. GM 척도 c(체크포인트에서 다시 잰 최소 RMS 거리의 배치 중앙값)는 0.22–0.26으로 로그의 match_d와 비슷.
  16점 적합 RMSE(괄호는 K 하나 빼기): I(단위 방향) 0.0315(0.044), I_pos(px²) 0.032, I_tok 0.048(0.063), I_tok GM 0.051(0.067),
  놓칠 확률 0.068(0.093), 1/m² 0.038(0.051), 1/m 0.040. 원고의 0.031/0.068/0.065와 리뷰의 0.038(1/m²) 재현.
- 19:02Z d의 교차 점검: e14_asym_m{16,4,2}_s0를 400×64, torch 시드 123으로 다시 잰 값이 저장된 modeshare.json(원래 A5000)과
  share0 차 ≤ 0.006, 지수 차 ≤ 0.005(A40과 A5000의 bf16 차이 수준). 허용 오차 없는 share0도 offset을 없애지 않음
  (m=16 test 0.622 대 허용 오차판 0.582). c 시작(모니터 설정).
- 19:07Z FM-50 위험 진단(선택 항목, 약 1,150만 순전파)은 FM 재실행과 GPU를 다투므로 `/workspace/nom/STOP_OPT`로 일단 건너뜀.
- 19:13Z c 진행: e1_a2_fm_s0(200×16) NFE 2 지수 0.332/held-out 0.239, NFE 50 0.297/0.254, NFE 20 0.298/0.248.
  REPORT.md에 0, a, b, d, f, h 절을 먼저 씀(Write 도구가 막혀 WORKER_RULES대로 Bash heredoc 사용).
- 19:22Z 재실행 step 8(활성 57–59개), 5,300 s. 회귀 재실행 3개를 지금 함께 시작하도록 스케줄러를 `jobs/e_sched2.sh`로 교체
  (IMLE-WM과 회귀가 끝나면 FM-2 s0, s1, s2를 하나씩). FM 재실행의 시작을 약 20분 앞당김.
- 19:52Z 회귀 재실행 3개 끝(성공 0.53/0.60/0.58, 구덩이 0.25/0.21/0.27; 원래 56.3±3.4, 24.3±2.5).
- 19:58Z IMLE-WM s2 재실행 끝(성공 0.51, 구덩이 0.37, 7,376 s). 고른 계획 897개에서 참 구덩이 확률 0.167, 모델의 상상 확률은
  마지막 프레임 기준 0.068, 임의 프레임 기준 0.181. 실행된 행동열(열린 루프) 100개에서 참 0.387, 마지막 프레임 0.137, 임의 0.392.
- 20:02Z IMLE-WM 재실행 3개 끝(성공 0.51/0.51/0.51, 구덩이 0.34/0.30/0.37; 원래 55.3±2.6, 31.0±2.2). FM-2 s0 재실행 시작.
- 20:05Z 위험 진단(직선 밀기) 시작. IMLE-WM s0의 마지막 프레임 기준 5회 밀기 구덩이 확률 0.126(참 0.427).
- 20:13Z FM-2 s0 첫 step 685 s(경합). 하나씩 돌리면 s2가 00:00Z 이후에 끝날 수 있어, 모델 가중치를 channels_last로 두는 옵션
  (`r1_planrisk replay --channels_last`, 계산은 같고 bf16 반올림만 다름; 경합 상태에서 645 → 506 ms/호출, `jobs/bench_cl.py`)을
  추가하고 s1, s2에만 쓰도록 스케줄러를 `jobs/e_sched3.sh`로 교체(s0은 그대로 진행).
- 20:18Z 위험 진단 완료: 임의 프레임 기준 값은 원래 riskdiag_nfe2.json과 0.01 안에서 같음(IMLE-WM 0.419/0.403/0.502, 원래
  0.417/0.407/0.499). 마지막 프레임 기준은 IMLE-WM 0.126/0.098/0.089, FM-2 0.381/0.253/0.379, 회귀 0.450/0.150/0.055(참 0.427).
- 20:25Z an_main2 핵심 완료(c 400×64 포함), 선택 항목은 STOP_OPT로 건너뜀. 이제 GPU에는 FM-2 s0 재실행만 남음.
- 21:17Z FM-2 s0 재실행 끝(성공 0.68, 구덩이 0.16, 4,466 s; 원래 0.66/0.18). 고른 계획 1,119개에서 참 0.099, 모델 마지막 0.017,
  임의 0.020. FM-2 s1(channels_last) 시작.
- 22:12Z FM-2 s1 재실행 끝(channels_last; 성공 0.70, 구덩이 0.17, 3,218 s; 원래 0.69/0.20). FM-2 s2 시작.
- 22:14Z 선택 항목 중 FM-50 직선 밀기 위험 진단을 시드 0만 FM-2 s2와 함께 시작(약 14분 분량, `riskdiag_both_nfe50.json`).
- 22:44Z FM-50 직선 밀기 위험 진단(시드 0) 끝: 마지막 프레임 0.419, 임의 0.513(원래 riskdiag_fm50.json 0.509), 참 0.427.
- 23:27Z FM-2 s2 재실행 끝(channels_last; 성공 0.72, 구덩이 0.12, 4,413 s). 모든 GPU 작업 종료(GPU 사용 0%).
  FM-2 세 시드: 고른 계획 참 0.105, 모델 마지막 0.022, 임의 0.027; 실행 행동열 참 0.240, 마지막 0.107, 임의 0.143.
- 23:30Z 결과 전부 당겨 옴(`raw/an.1/out/`, 125개 파일, 9.8 MB). REPORT.md 완성(요약 표, 절 0, a–h, 조정자용 명령, 문제).
  an.1은 더 할 일이 없으니 조정자가 지워도 된다.

# 조정자용 명령 (분석 작업자가 만든 스크립트)

먼저 각 pod에 최신 코드(r1_*.py와 `imlewm/r1_timing_configs.json` 포함)를 밀어 넣는다.

```bash
bash runpod/sync.sh <tag> push main/code/ /workspace/imle-wm/code/ --exclude __pycache__ --exclude .DS_Store --exclude '*.lock'
```

pod 안에서는 모두 `source ~/imle-wm/env.sh && cd ~/imle-wm/code` 뒤에 실행한다(긴 것은 WORKER_RULES 2절처럼 setsid nohup).
출력은 `/workspace/nom/out/<분석>/`에 JSON으로 쓰고 이미 있으면 건너뛴다. `pull_all.sh <tag>`로 당겨 온다.

## x1.1: 실험 1의 측정 점검 C1, C2와 분석 a의 arc 부분 (12개 run이 끝난 뒤)

```bash
# C1: 모니터의 학습 문맥 200개(seed 0)에 실현 프레임 16장(천장)과 K개 참 분기 지도 4장씩(바닥). 두 기하를 같은 환경에서.
python -m imlewm.r1_calib --data bounce_div_k4_arc90 bounce_div_k4 --out /workspace/nom/out/c1
# C2 (+ 같은 시드의 지수, held-out 기준 초과, 판독기 없는 간격): 모니터 설정 200 x 16, 8k 최종 체크포인트
python -m imlewm.r1_memo --runs r1_arc90_m2_s0 r1_arc90_m2_s1 r1_arc90_m4_s0 r1_arc90_m4_s1 r1_arc90_m8_s0 r1_arc90_m8_s1 \
  r1_k4_m2_s0 r1_k4_m2_s1 r1_k4_m4_s0 r1_k4_m4_s1 r1_k4_m8_s0 r1_k4_m8_s1 --n_ctx 200 --n 16 --out /workspace/nom/out/c2
# (선택) 분석 b와 같은 설정 400 x 64
python -m imlewm.r1_memo --runs <위와 같은 12개> --n_ctx 400 --n 64 --out /workspace/nom/out/c2
# 분석 a의 arc 부분: I_tok(m,4)와 GM판(척도는 각 run의 체크포인트에서 다시 잰다)
python -m imlewm.r1_itok tok --data bounce_div_k4_arc90 --scale_runs r1_arc90_m2_s0 r1_arc90_m2_s1 r1_arc90_m4_s0 r1_arc90_m4_s1 r1_arc90_m8_s0 r1_arc90_m8_s1 --out /workspace/nom/out/a_itok
python -m imlewm.r1_itok tok --data bounce_div_k4 --scale_runs r1_k4_m2_s0 r1_k4_m2_s1 r1_k4_m4_s0 r1_k4_m4_s1 r1_k4_m8_s0 r1_k4_m8_s1 --out /workspace/nom/out/a_itok
```

- 읽는 곳: `c1/c1_<data>_200.json`의 `ceil`, `floor`(각각 [평균, 2.5%, 97.5%]). 규칙: `ceil(arc)[0] < ceil(std)[0] − 0.05`이면
  두 기하 모두 `(index − floor) / (ceiling − floor)`를 주 척도로 쓴다(index는 log.jsonl의 마지막 memo_train, 두 시드 평균).
- `c2/<run>_ckpt_nfeNone_200x16.json`의 `derived.free_index_train`이 C2(허용 오차 없는 지수), `derived.index_train`은 같은
  시드로 다시 잰 지수, `derived.excess_heldout`, `derived.gap_min_dist`는 b와 같은 정의다.
- 소요: C1은 DINO 인코딩 포함 1–2분, C2는 run당 수 초(200 x 16) 또는 15초(400 x 64, 경합 없을 때), a는 데이터당 1–2분.

## x2.1: 실험 2a의 NFE 50/2 암기 지수 (r1_fm36k_s* 학습이 끝난 뒤)

```bash
for f in 50 2; do
  python -m imlewm.r1_memo --runs r1_fm36k_s0 r1_fm36k_s1 r1_fm36k_s2 --nfe $f --n_ctx 200 --n 16 --out /workspace/nom/out/fm36k_memo
done
# (선택) 중간 체크포인트: --ckpt ckpt_12000 / ckpt_24000
```

- 4c 절차와 같다(모니터 설정 200 x 16, torch 시드 123, 최종 체크포인트). P2a의 "FM memorization index at NFE 50 at most 0.32"는
  `derived.index_train`(NFE 50)으로 판정한다.
- 소요: NFE 50은 run당 약 1분(경합 없을 때), NFE 2는 수 초.

## 유휴 A5000 (x2.1, 학습이 모두 끝난 뒤): 분석 g

```bash
python -m imlewm.r1_timing run --out /workspace/nom/out/g_timing
```

- Table 2(9개)와 Table 4(7개) 구성의 배치 1 지연(5회 예열 + 50회 측정, 5회 반복)과 학습 step 시간(각 run의 원래 인자, 같은 모양의
  무작위 토큰 데이터 `data/r1_tim_a2`, `data/r1_tim_a10`, 400 step, t(400) − t(200)). run 이름은 `r1_tim_*`.
- 구성은 `imlewm/r1_timing_configs.json`(an.1에서 원래 체크포인트로 추출)에서 읽으므로 원래 체크포인트가 그 pod에 없어도 된다.
- 소요: 약 40–60분(21M iMF와 m=16 학습 step이 가장 길다). 그 pod에서 다른 작업이 돌지 않아야 한다.

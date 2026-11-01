# IMLE-WM 연구 설계서: 단일 스텝 확률적 시각 월드모델

Oct 10, 2026 · @용휘

## 1. 연구 개요

**목표.** 반복 디노이징 없이 순전파 한 번으로 여러 미래를 샘플링하는 행동 조건부 시각 월드모델 IMLE-WM을 만들고, 결정론적 회귀의 평균화 문제를 피하면서 diffusion·flow 수준의 mode coverage를 내는지 검증한다.

**핵심 아이디어.** IMLE는 생성기 샘플 m개 중 각 실제 전이에 가장 가까운 것을 골라 그 거리만 줄인다. 데이터 포인트마다 자신을 설명하는 샘플이 하나씩 보장되므로 mode collapse가 목적함수 수준에서 억제되고, 추론은 단일 순전파다. ROMS-IMLE(2026.07)가 이미지 생성에서 보인 multi-stage supervision과 robust loss 레시피를, WIMLE(ICLR 2026)이 상태 벡터에서 보인 조건부 IMLE 월드모델에 결합해 시각 관측(DINO 피처)으로 확장한다.

**연구 질문과 가설**

| ID | 연구 질문 | 가설 | 검증 섹션 |
| --- | --- | --- | --- |
| RQ1 | 1-step IMLE가 시각 월드모델에서 평균화를 피하고 다봉 미래를 덮는가? | 모드가 열거된 합성 환경에서 coverage가 회귀 대비 유의하게 높고 flow matching과 동등 | 5.1, 5.4 |
| RQ2 | 매칭 단위를 한 step에서 H-step 궤적으로 바꾸면 롤아웃 오류 누적이 줄어드는가? | horizon별 오차 곡선 기울기가 한 step 매칭보다 완만 | 5.3, 6 |
| RQ3 | 해상도 대신 피처 스펙트럼으로 stage를 정의한 multi-stage supervision이 ROMS의 이득을 재현하는가? | 스펙트럼 stage가 공간 풀링 stage보다 FID·recall에서 우위 | 6 |
| RQ4 | 거의 무료인 m개 미래 샘플의 분산을 planning 비용에 쓰면 성공률이 오르는가? | 분산 항 추가 시 확률 환경에서 성공률 상승, 결정론 환경에서는 변화 없음 | 5.2, 5.4 |
| RQ5 | 노이즈 z가 행동 효과를 흡수하는 shortcut이 생기는가? | 행동 경로 분리 없이는 행동 민감도가 낮고, 반사실 대조 항이 이를 회복 | 5.3, 6 |

**범위.** 생성형(피처 예측) 월드모델에 한정한다. 픽셀 복원은 시각화 보조로만 쓰고, 정책 학습은 planning(CEM/MPPI)으로 대신한다. 모델 기반 RL 전체 루프는 범위 밖이다.

## 2. 배경과 선행연구 갭

### 2.1 평균화 문제

조건부 분포 p(y | x)가 두 모드 y\_L, y\_R을 가질 때, ℓ2 손실의 최소해는 조건부 평균 (y\_L + y\_R)/2이고 ℓ1의 최소해는 조건부 중앙값이다. 둘 다 어느 모드에도 속하지 않는다. FlowWM(2026.06)은 이 사실을 형식적으로 증명하고, 고차원 latent에서 평균화가 의미 구조를 파괴하며 자기회귀 롤아웃에서 오차가 증폭된다고 지적했다. 결정론적 DINO-WM·DINO-world도 긴 horizon에서 예측이 점점 흐려지는 현상을 보고한다.

### 2.2 기존 해법과 비용

| 접근 | 대표 연구 | 평균화 회피 | 추론 NFE | 추가 기계장치 |
| --- | --- | --- | --- | --- |
| 결정론적 회귀 | DINO-WM, DINO-world, V-JEPA 2-AC | 아니오 | 1 | 없음 |
| 픽셀/latent diffusion | DIAMOND, GameNGen, NWM | 예 | 10\~50 | 디노이징 반복 |
| 피처 공간 flow | FlowWM, VFMF | 예 | 10\~20 | 피처 압축 VAE 또는 one-step projection |
| few-step 증류 | Matrix-Game 3.5, MWM(ICSD), RAE 멀티플레이어 WM | 예 | 1\~4 | 교사 모델, 증류 단계 |
| MeanFlow 계열 | GE-Act 2.0 | 예 | 1 | JVP 계산, 이중 헤드, 학습 예산 분할 |
| Drifting | DriftWorld | 예 | 1 | 커널 기반 drift 필드, 피처 공간 손실 |
| IMLE | WIMLE(상태 벡터), IMLE-MPC(궤적) | 예 | 1 | nearest-neighbor 매칭(gradient 불필요) |

### 2.3 IMLE와 ROMS-IMLE

IMLE 목적함수는 다음과 같다.

```latex
\theta^{*} = \arg\min_{\theta}\; \mathbb{E}_{z_1,\dots,z_m \sim \pi}\Big[\sum_{i=1}^{n} \min_{j \in [m]} d\big(x_i, f_\theta(z_j)\big)\Big]
```

매칭 단계(gradient 없음, 병렬)와 최적화 단계를 번갈아 수행한다. ROMS-IMLE는 두 가지를 더했다.

- **Multi-stage supervision.** 생성기를 해상도별 stage f\_θ = f\_L ∘ … ∘ f\_1의 합성으로 보고, stage마다 출력 헤드를 붙여 다운샘플한 정답 K\_l(x\_i)로 직접 감독한다. diffusion의 성공 요인이 반복 샘플링이 아니라 stage별 감독이라는 가설이다.
- **Robust loss.** 샘플링 우연으로 생긴 나쁜 매칭 쌍이 ℓ2에서 큰 gradient를 만들므로 Geman-McClure 손실로 영향력을 줄인다.

Oxford Flowers 어블레이션에서 FID 62.6 → 13.9, 같은 FID 도달이 약 5배 빨랐다. 매칭 예산 m/n은 4에서 10으로 늘려도 이득이 작았다. 고해상도 latent 생성에서는 디코더가 잘 복원하지 못하는 샘플을 round-trip 비용(인코딩-디코딩 후 거리)으로 걸러내는 rejection을 썼다.

WIMLE은 조건부 생성기 g\_θ(s\_t, a\_t, z)로 (s\_{t+1}, r\_t)를 예측하고, K=7 앙상블 × m개 latent 샘플의 표준편차를 불확실성으로 써서 합성 전이를 가중한다. 상태 벡터에서 DreamerV3·TD-MPC2보다 샘플 효율이 높았고, 저자들은 이미지 기반 확장을 향후 과제로 명시했다.

### 2.4 갭

이미지·피처 관측에서 IMLE 계열 월드모델은 검색 범위(2026.10.10 기준)에서 발견하지 못했다. 1-step 확률 월드모델 중 drifting(DriftWorld)과 MeanFlow(GE-Act 2.0)는 적용됐지만, 추가 기계장치 없이 likelihood 계열 목적함수로 mode coverage를 보장하는 IMLE 계열은 비어 있다. 또한 기존 IMLE 연구는 모두 한 step 매칭이며, 롤아웃 노출 편향을 매칭 단계에서 다룬 사례가 없다.

## 3. 노벨리티와 기여

주 기여는 N1(궤적 매칭)이다. N0(단순 확장)만으로는 WIMLE 저자 그룹(SFU APEX Lab)의 자연스러운 후속작과 정면으로 겹칠 수 있으므로, N1을 중심에 두고 N2·N3을 부가 기여, N4를 분석 기여로 배치한다.

| ID | 기여 | 무엇이 새로운가 | 가장 가까운 선행연구와의 차이 | 역할 |
| --- | --- | --- | --- | --- |
| N0 | IMLE-WM: DINO 피처 공간 행동 조건부 ROMS-IMLE | 시각 관측 IMLE 월드모델 첫 사례(검색 기준) | WIMLE은 상태 벡터, ROMS-IMLE은 무조건/클래스 조건 이미지 | 기반 |
| N1 | 궤적 단위 IMLE 매칭 | 매칭 후보를 단일 전이가 아니라 모델 자신의 H-step 롤아웃으로 정의 | self-forcing·self-rollout DMD는 판별기/교사 필요, IMLE는 nearest-neighbor만 필요 | 주 기여 |
| N2 | 스펙트럼 multi-stage supervision | stage를 해상도 대신 DINO 피처 PCA 스펙트럼(저주파→전체)으로 정의 | ROMS는 해상도 stage, 패치 토큰에는 적용 불가 | 부가 |
| N3 | 분산 인지 planning + 상상 상태 rejection | m개 미래 샘플 분산을 CEM/MPPI 비용에 반영, round-trip 비용을 상상 상태 타당성 필터로 사용 | WIMLE은 분산을 RL 가중치에만 사용, ROMS는 이미지 품질 필터로만 사용 | 부가 |
| N4 | 행동 충실도 분석과 대조 정규화 | z가 행동 효과를 흡수하는 shortcut을 정량화하고 반사실 행동 대조 항으로 완화 | DriftWorld의 action-accentuated drift와 목적은 같으나 메커니즘이 다름 | 분석 |

**N1의 주장 구조.** 자기회귀 월드모델은 학습 시 정답 컨텍스트, 추론 시 자기 예측 컨텍스트를 쓰므로 노출 편향이 생긴다. 기존 해법은 diffusion forcing(컨텍스트에 노이즈 주입), self-forcing(자기 롤아웃에 판별기), self-rollout DMD(교사 분포 매칭)다. IMLE에서는 매칭 후보 자체를 자기 롤아웃으로 만들면, 정답 궤적에 가장 가까운 롤아웃만 감독하므로 "자기 출력 위에서 좋은 궤적"을 학습 중에 직접 선별한다. 추가 모델 없이 매칭 정의만 바꾼다는 점이 단순성 측면의 주장이고, 1-step 생성기라 m×H 순전파 비용이 감당 가능하다는 점이 실현 가능성 측면의 주장이다.

**N2의 근거.** VFMF(2025.12)는 DINO 피처가 이미지처럼 멱법칙 스펙트럼을 가진다고 보고했다. 따라서 저주파 PCA 성분이 이미지의 저해상도에 대응하므로, ROMS의 coarse-to-fine 감독을 스펙트럼 축으로 옮길 수 있다.

**N3의 근거.** WIMLE은 m개 latent 샘플 분산이 aleatoric 불확실성을 반영함을 보였다(총분산 분해). 1-step 생성기에서 이 분산은 순전파 m번이면 얻어지므로, diffusion 기반 월드모델에서는 비현실적이던 샘플 기반 위험 추정이 실시간 planning에 들어갈 수 있다.

## 4. 방법 설계

고정된 DINO 인코더가 만든 패치 토큰 공간에서, 컨텍스트·행동·노이즈 z를 받아 다음 토큰을 한 번에 내는 조건부 생성기 g\_θ를 IMLE로 학습한다.

&#91;embedded content: IMLE-WM 구조 · 입력 3개, 생성기 stage 3개, 매칭 루프\]

행동은 FiLM, 노이즈는 AdaIN으로 따로 들어가 z가 행동 효과를 대신 흡수하지 못하게 한다. stage별 헤드는 각자 다른 스펙트럼 수준의 타깃으로 직접 감독받는다.

### 4.1 표기

- 관측 o\_t → 토큰 s\_t = E(o\_t) ∈ ℝ^{N×D}, E는 고정 DINOv2 ViT-S/14(N=256 패치, D=384). 토큰은 사전 계산해 저장하고 학습 중 인코더는 돌리지 않는다.
- 컨텍스트 c\_t = \[s\_{t-k+1}, …, s\_t\], k=3.
- 생성기 g\_θ(c\_t, a\_t, z) → ŝ\_{t+1}, z ∈ ℝ^{256}.
- 매칭 예산 m, 롤아웃 길이 H, stage 수 L.

### 4.2 생성기 구조

| 구성요소 | 설계 | 근거 |
| --- | --- | --- |
| 모델 본체 | ConvNeXt 블록 8\~12개, 폭 384\~512, 16×16 토큰 그리드 위에서 동작 | ROMS-IMLE가 트랜스포머 없이 경쟁력 보임 |
| 컨텍스트 주입 | k프레임 토큰을 채널 방향으로 연결 후 1×1 conv | DINO-WM과 동일 |
| 노이즈 주입 | z → MLP 4층 → 스타일 벡터 w → 블록별 AdaIN | ROMS-IMLE 매핑 네트워크 |
| 행동 주입 | a\_t → MLP 2층 → 블록별 FiLM(γ, β) | 노이즈와 경로 분리(N4), DriftWorld의 frame-wise FiLM |
| stage별 헤드 | 블록 1/3, 2/3, 끝 지점에 1×1 conv 헤드 (L=3) | ROMS의 per-stage output head |
| 출력 | 잔차 예측 ŝ\_{t+1} = s\_t + Δ | 정지 상태에서 안정, RLA-WM의 잔차 관점 |

### 4.3 학습 목적함수: 조건부 ROMS-IMLE (N0)

한 배치 내 전이 i마다 m개 노이즈로 후보를 생성하고, 토큰 공간 ℓ2로 가장 가까운 후보를 고른다.

```latex
\sigma(i) = \arg\min_{j \in [m]} \big\| s^{i}_{t+1} - g_\theta(c^{i}_t, a^{i}_t, z_j) \big\|_2^2
```

선택된 후보에 대해 stage별 손실을 평균한다.

```latex
\mathcal{L}_{\mathrm{N0}} = \frac{1}{L}\sum_{l=1}^{L} \frac{1}{|K_l|}\, \rho_{\mathrm{GM}}\!\Big( K_l\big(s^{i}_{t+1}\big) - g^{(l)}_\theta\big(c^{i}_t, a^{i}_t, z_{\sigma(i)}\big) \Big), \qquad \rho_{\mathrm{GM}}(r) = \frac{2 r^2 / c^2}{r^2 / c^2 + 4}
```

- ρ\_GM은 Geman-McClure, c는 토큰 잔차 표준편차로 초기화.
- K\_l은 stage l 타깃 변환(4.4). 기본값은 m=4, 배치당 매칭은 FAISS-GPU 또는 단순 cdist로 수행.
- 매칭은 매 배치마다 재계산한다(WIMLE 방식). ROMS처럼 epoch 단위로 고정하면 조건부 설정에서 오래된 매칭이 남는다.

### 4.4 스펙트럼 multi-stage supervision (N2)

패치 토큰 그리드는 해상도가 고정이므로 ROMS의 다운샘플 타깃을 그대로 쓸 수 없다. 세 가지 stage 정의를 비교한다.

| stage 정의 | K\_l | 동기 |
| --- | --- | --- |
| 공간 풀링 | 16×16 → 4×4 → 8×8 → 16×16 평균 풀링 | ROMS 해상도 stage의 직접 대응 |
| 스펙트럼 (제안) | 학습 데이터 토큰의 PCA 기저로 상위 32 → 128 → 384차원 투영 | DINO 피처의 멱법칙 스펙트럼(VFMF) |
| horizon | stage l이 t+l 예측을 담당 | 시간축 coarse-to-fine, N1과 결합 가능 |

PCA는 학습 전 한 번 계산해 고정한다. stage l 헤드는 d\_l차원 출력을 내고, 손실은 투영된 정답과의 거리로 계산한다.

### 4.5 궤적 매칭 (N1)

매칭 단위를 전이 하나에서 H-step 궤적으로 바꾼다. 노이즈 시퀀스 z\_{j,1:H}를 m개 뽑고, 각 시퀀스로 모델을 자기 예측 위에서 H번 굴린다.

```latex
\hat{s}^{\,j}_{t+h} = g_\theta\big(\hat{c}^{\,j}_{t+h-1},\, a^{i}_{t+h-1},\, z_{j,h}\big), \qquad \hat{c}^{\,j}_{t+h-1} = [\hat{s}^{\,j}_{t+h-k}, \dots, \hat{s}^{\,j}_{t+h-1}]
```

```latex
\sigma(i) = \arg\min_{j \in [m]} \sum_{h=1}^{H} w_h \big\| s^{i}_{t+h} - \hat{s}^{\,j}_{t+h} \big\|_2^2, \qquad \mathcal{L}_{\mathrm{N1}} = \frac{1}{H}\sum_{h=1}^{H} \mathcal{L}_{\mathrm{N0}}\big(s^{i}_{t+h},\, \hat{s}^{\,\sigma(i)}_{t+h}\big)
```

- 롤아웃 중 컨텍스트는 자기 예측으로 채운다. 초기 k프레임만 정답이다.
- w\_h는 균등 또는 γ^h 할인. 매칭은 gradient 없이 수행하고, 손실은 선택된 시퀀스 전 구간에 걸며 BPTT는 H step까지 흐른다.
- 커리큨럼: H=1로 수렴 후 H=4, H=8로 늘린다. 매칭 비용은 m×H 순전파이고 반정밀도로 수행한다.
- diffusion forcing과 결합할 필요가 없다. 자기 롤아웃 자체가 컨텍스트 오염의 실제 분포를 제공한다.

### 4.6 행동 충실도 정규화 (N4)

조건부 IMLE에서 z는 정답에 가장 가까워지도록 선택되므로, 행동 정보가 약하게 주입되면 z가 행동 효과를 대신 설명하는 shortcut이 생길 수 있다. 두 단계로 다뤘다.

1. **경로 분리.** 행동은 FiLM, z는 AdaIN으로 분리 주입(4.2).
2. **반사실 대조 항.** 같은 (c\_t, z\_{σ(i)})에서 무작위 행동 a'로 생성한 ŝ'가 정답 s\_{t+1}에 가까워지지 않도록 margin 손실을 더한다.

```latex
\mathcal{L}_{\mathrm{act}} = \max\Big(0,\; \mu - \big\|\hat{s}' - s_{t+1}\big\|_2 + \big\|\hat{s} - s_{t+1}\big\|_2\Big), \qquad \mathcal{L} = \mathcal{L}_{\mathrm{N1}} + \lambda_{\mathrm{act}} \mathcal{L}_{\mathrm{act}}
```

반사실 행동 a'는 배치 내 다른 전이의 행동을 섮어 쓰고, μ는 행동 차이 ‖a - a'‖에 비례하게 잡는다.

### 4.7 추론과 planning (N3)

- 롤아웃 한 step = 순전파 1회. 상태마다 m\_p개 z를 뽑으면 m\_p개 미래를 얻는다(배치 처리 시 지연 거의 증가 없음).
- CEM/MPPI 비용: 후보 행동 시퀀스마다 m\_p개 롤아웃의 목표 거리 평균에 λ·분산을 더한다. 다봉 미래에서 위험한 행동을 피한다.

```latex
J(a_{t:t+T}) = \frac{1}{m_p}\sum_{j=1}^{m_p} \big\| \hat{s}^{\,j}_{t+T} - s_{\mathrm{goal}} \big\|_2 \;+\; \lambda \cdot \mathrm{Var}_j\!\Big[\big\| \hat{s}^{\,j}_{t+T} - s_{\mathrm{goal}} \big\|_2\Big]
```

- **상상 상태 rejection.** 토큰 → 이미지 디코더 D와 고정 인코더 E로 round-trip 비용 ‖ŝ - E(D(ŝ))‖를 계산해, 높은 롤아웃을 planning 후보에서 제외한다. ROMS의 이미지 품질 필터를 상상 상태의 타당성 필터로 재사용한다.
- 디코더 D는 평가·시각화 보조용으로 별도 학습하며, 월드모델 학습 경로에는 들어가지 않는다.

### 4.8 복잡도

| 항목 | 결정론적 회귀 | flow matching | IMLE-WM N0 | IMLE-WM N1 |
| --- | --- | --- | --- | --- |
| 학습 순전파 / 전이 | 1 | 1 | m (매칭) + 1 | m×H (매칭) + H |
| 학습 역전파 / 전이 | 1 | 1 | 1 | H |
| 추론 NFE / step | 1 | 10\~20 | 1 | 1 |
| 추가 모델 | 없음 | 없음 | 없음 | 없음 |

m=4, H=8이면 N1의 매칭 비용은 전이당 순전파 32회이지만 gradient가 없고 반정밀도로 돌리므로, 역전파 1회와 비슷한 비용으로 추정한다.

## 5. 실험 설계

세 벤치마크로 간다. 벤치마크 A에서 평균화 회피를 통제된 환경에서 분리 측정하고, B에서 행동 조건부 롤아웃과 planning 효용을, C에서 실제 시각 복잡도와 다중 에이전트 모호성을 본다. A의 결과가 가설 H1을 기각하면 B·C로 가지 않고 원인부터 분석한다.

### 5.1 벤치마크

| 구분 | 환경 | 확률성의 원천 | 모드 수 | 관측 | 데이터 규모 |
| --- | --- | --- | --- | --- | --- |
| A. 통제 합성 | 2D 상자 속 공이 벽에 닿으면 무작위 방향으로 튕김 (직접 구현) | 충돌 시 방향 선택 | 2, 4, 8로 조절 | 64×64 RGB | 10만 전이 |
| A'. 행동 조건부 합성 | A에 "밀기" 행동 4방향 추가, 결과는 행동 방향 ± 무작위 편향 | 행동 효과 + 편향 | 2\~3 | 64×64 RGB | 10만 전이 |
| B. DINO-WM 환경 | Push-T, PointMaze, Wall (공개 데이터셋) + 확률 변형 | 원본은 거의 결정론적 → 물체 질량·마찰을 에피소드마다 무작위화 | 연속 | 224×224 RGB | 공개 데이터 + 변형 5만 전이 |
| C. 주행 | CARLA Town05 교차로, NPC 의도(직진/좌회전/정지)를 무작위화 | 타 에이전트 의도 | 3 | 전면 카메라 224×224 | 2만 전이 |

A는 FlowWM의 합성 벤치마크처럼 미래 모드가 열거되므로 coverage를 정확히 셀 수 있다. B의 확률 변형은 원본 DINO-WM이 결정론적이라 IMLE의 이점이 드러나지 않을 위험(7절)에 대한 대응이다. C는 기존 CARLA 파이프라인을 재사용하며, 선택 사항이다.

### 5.2 베이스라인

모두 같은 고정 DINOv2 ViT-S 토큰 공간에서, 파라미터 수를 ±10% 안으로 맞춘다.

| 베이스라인 | 구현 | NFE | 역할 |
| --- | --- | --- | --- |
| 결정론적 회귀 | DINO-WM 방식 ViT predictor, ℓ2 손실 | 1 | 평균화 문제 확인 |
| flow matching (multi-step) | FlowWM 방식, Euler 20 step | 20 | 품질 상한 |
| flow matching (few-step) | 같은 모델 2 step | 2 | 속도 맞춘 비교 |
| MeanFlow | iMF 목적함수, 공개 PyTorch 구현 이식 | 1 | 1-step 경쟁자 |
| IMLE-WM N0 | 한 step 매칭, 공간 풀링 stage | 1 | 기반 |
| IMLE-WM N0+N2 | 한 step 매칭, 스펙트럼 stage | 1 | N2 검증 |
| IMLE-WM N0+N1+N2 | 궤적 매칭 H=8 | 1 | 제안 전체 |
| IMLE-WM 전체 + N4 | 반사실 대조 항 추가 | 1 | 행동 충실도 |

DriftWorld는 픽셀 공간 U-Net이라 토큰 공간 비교에서 제외하고, 지연 시간 비교에서만 논문 수치를 인용한다.

### 5.3 평가지표

| 지표 | 정의 | 왜 필요한가 |
| --- | --- | --- |
| Mode coverage (A, A', C) | 컨텍스트당 m\_e=16 샘플을 뽑아 열거된 모드 중 하나 이상 맞힌 모드의 비율. 모드 판정은 공 위치 또는 NPC 궤적의 클러스터 | 평균화 회피의 직접 증거 |
| 피처 공간 precision / recall | Kynkäänniemi k-NN 방식을 DINO CLS 토큰에 적용, k=3 | ROMS가 보인 recall 우위가 월드모델에서도 재현되는지 |
| 평균 오차 vs best-of-m 오차 | 평균 오차: 샘플 평균과 정답의 ℓ2. best-of-m: m\_e 샘플 중 최소 ℓ2 | 평균 오차는 평균화 모델에 유리하므로 둘을 함께 보고 |
| horizon별 오차 곡선 | h=1..32에서 best-of-m 오차 | 롤아웃 오류 누적, N1 효과 |
| Planning 성공률 (B) | CEM 50 후보 × 5 반복, 목표 도달 성공률, 100 에피소드 | 하류 효용 |
| 지연 시간 | 배치 1, 단일 GPU에서 ms/step과 NFE | 실시간성 |
| 행동 민감도 | 같은 (c\_t, z)에서 행동을 바꿨을 때 출력 ℓ2 변화량 / 정답 변화량 | N4, z-shortcut 진단 |
| 샘플 다양성 | 같은 (c\_t, a\_t)에서 z만 바꿨을 때 출력 간 평균 거리 | 조건부 설정에서 z 무시 여부 |

### 5.4 가설별 예측 결과

| 가설 | 측정 | 지지 조건 | 기각 시 해석 |
| --- | --- | --- | --- |
| H1 (RQ1) 평균화 회피 | A에서 coverage | 회귀 ≤ 1/모드수, IMLE ≥ 0.9, flow와 차이 ≤ 0.05 | 조건부 IMLE가 z를 무시 → 다양성 지표로 확인, m 증가 또는 z 차원 조정 |
| H2 (RQ2) 오류 누적 완화 | A', B에서 horizon 곡선 | N1의 h=32 오차가 N0 대비 20% 이상 낮음 | 자기 롤아웃 매칭이 학습 초기에 불안정 → 커리큘럼 조정 |
| H3 (RQ3) 스펙트럼 stage | A, B에서 FID·recall | 스펙트럼 ≥ 공간 풀링, 둘 다 ≥ 단일 stage | DINO 피처의 스펙트럼 구조가 공간 구조만큼 유용하지 않음 |
| H4 (RQ4) 분산 인지 planning | B 확률 변형에서 성공률 | 분산 항 λ>0에서 성공률 상승, 원본 B에서는 변화 없음 | 분산이 aleatoric·epistemic을 구분 못 함 → 앙상블 추가 검토 |
| H5 (RQ5) 행동 충실도 | A', B에서 행동 민감도 | 경로 분리만으로 ≥ 0.8, N4로 ≥ 0.95 | z-shortcut이 구조적 → 행동을 매칭 거리에도 반영 |

### 5.5 구현 세부

- 토큰 사전 계산: 모든 프레임을 DINOv2 ViT-S/14로 한 번 인코딩해 float16으로 저장. B 기준 약 5만 × 256 × 384 × 2B ≈ 10 GB.
- 생성기 50\~100M 파라미터, AdamW(β=0.9, 0.9), lr 2e-4, cosine 스케줄, warmup 1k step, 배치 256, 혼합정밀도.
- 매칭: 배치마다 m=4 후보를 생성하고 cdist로 선택. N1에서는 m×H 롤아웃을 no\_grad + fp16으로 수행.
- 시드 3개, 모든 지표는 평균 ± 표준편차로 보고. 베이스라인은 같은 시드·같은 데이터 분할.
- 계산 자원: 24GB GPU 2장. GPU 1장은 제안 모델, 1장은 베이스라인 학습에 할당. 시각화용 디코더는 별도로 가볍게 학습.

## 6. 어블레이션 설계

벤치마크 A(합성)에서 먼저 돌려 비용을 아끼고, 결과가 갈리는 항목만 B로 가져간다. 기준 설정은 N0+N1+N2(m=4, H=8, 스펙트럼 stage, Geman-McClure, 행동 FiLM)이고, 한 번에 한 요소만 바꾼다.

| # | 요소 | 변형 | 주 지표 | 검증 가설 |
| --- | --- | --- | --- | --- |
| 1 | Multi-stage supervision | 없음(최종 stage만) / 있음 | FID, recall | ROMS 재현 |
| 2 | Robust loss | ℓ2 / Pseudo-Huber / Geman-McClure | FID, 학습 곡선 | ROMS 재현 |
| 3 | stage 정의 | 공간 풀링 / 스펙트럼 PCA / horizon | FID, recall | H3 |
| 4 | 매칭 단위 | H=1 / 4 / 8 / 16 | horizon별 오차 | H2 |
| 5 | 커리큘럼 | 없음(H=8부터) / 1→4→8 | 학습 안정성, 최종 오차 | H2 |
| 6 | 매칭 예산 m | 2 / 4 / 8 / 16 | coverage, 학습 시간 | H1, 비용 |
| 7 | 매칭 거리 | 전체 ℓ2 / PCA 64차원 ℓ2 / 코사인 | coverage, 매칭 안정성 | 고차원 매칭 리스크 |
| 8 | 행동 주입 | z와 같은 AdaIN / FiLM 분리 / FiLM + 대조 항 | 행동 민감도 | H5 |
| 9 | planning 비용 | 평균만 / 평균 + λ·분산 (λ ∈ {0.1, 0.5, 1}) | 성공률 | H4 |
| 10 | 상상 상태 rejection | 없음 / round-trip 임계 5% / 10% | 성공률, 제외 비율 | N3 |
| 11 | 컨텍스트 길이 k | 1 / 3 / 6 | horizon별 오차 | 설계 확인 |
| 12 | 잔차 vs 직접 예측 | ŝ = s\_t + Δ / ŝ = g(·) | 정지 상태 안정성 | 설계 확인 |

**추가 분석.** (a) 매칭 쌍의 거리 분포를 학습 중 추적해 robust loss가 억제하는 outlier 비율을 보고한다. (b) z 공간 보간: 같은 (c\_t, a\_t)에서 z를 선형 보간했을 때 출력이 매끄럽게 변하는지로 memorization 여부를 본다(ROMS의 prior-space interpolation). (c) 학습 데이터 nearest-neighbor 검사로 생성 샘플이 훈련 전이를 복사하지 않는지 확인한다.

## 7. 한계와 리스크

가장 큰 외부 리스크는 경쟁이다. WIMLE와 ROMS-IMLE는 같은 SFU APEX Lab(Ke Li) 작업이고, WIMLE 논문이 이미지 기반 확장을 향후 과제로 직접 적었다. N0만으로는 그들의 다음 논문과 정면으로 겹친다. 가장 큰 내부 리스크는 조건부 IMLE에서 z가 무시되거나 행동을 흡수하는 것이다.

| 리스크 | 징후 | 대응 | 영향 범위 |
| --- | --- | --- | --- |
| 같은 랩의 이미지 IMLE 월드모델 선공개 | arXiv에 Ke Li 공저 시각 월드모델 등장 | N1·N2를 주 기여로 유지하고 그 논문을 베이스라인으로 흡수. 주 1회 arXiv 알림 | 노벨리티 |
| 조건당 정답이 1개뿐이라 z 무시 | 샘플 다양성 ≈ 0, coverage가 1/모드수 | 비슷한 컨텍스트 간 일반화에 의존함을 명시. A에서 다양성 먼저 확인, m 증가·z 차원 축소·z dropout 검토 | H1 |
| z가 행동 효과를 흡수 | 행동 민감도 < 0.5 | 경로 분리(FiLM), 반사실 대조 항(N4), 행동을 매칭 거리에 포함 | H5 |
| 궤적 매칭 학습 불안정 | H=8에서 손실 발산, 매칭 쌍 거리 폭주 | 커리큘럼, 할인 w\_h, 초기 H step만 정답 컨텍스트 혼합 | H2 |
| 고차원 토큰에서 매칭 품질 저하 | 매칭 쌍이 무작위에 가까움(거리 분산 작음) | PCA 64차원에서 매칭, 손실은 전체 차원 | 전체 |
| 결정론적 환경에서 이점 없음 | 원본 B에서 회귀와 차이 없음 | 확률 변형 B를 기본 결과로, 원본은 보조 결과로 | H1, H4 |
| 스펙트럼 stage가 효과 없음 | 어블레이션 3에서 단일 stage와 동등 | N2를 부정 결과로 보고하고 공간 풀링으로 대체. 주 기여(N1)는 영향 없음 | H3 |
| 분산이 불확실성을 반영 못 함 | λ>0에서 성공률 변화 없음 | WIMLE처럼 소규모 앙상블(K=3) 추가해 epistemic 분리 | H4 |
| 계산 예산 초과 | N1 학습이 GPU 1장에서 하루 이상 | 토큰 사전 계산, 매칭 fp16, m=4 고정, B 데이터 축소 | 전체 |

**방법 자체의 한계.** (1) IMLE는 매칭 예산 m이 작을 때 드문 모드를 놓칠 수 있으며, 조건부 설정에서 이 문제는 데이터 밀도에 의존한다. (2) 피처 공간 예측이므로 픽셀 수준 시각화는 별도 디코더 품질에 좌우된다. (3) 단일 스텝 생성기는 추론 시 품질-다양성 trade-off를 조절할 CFG 같은 손잡이가 없으므로, rejection과 λ가 그 역할을 대신한다. (4) C 벤치마크는 2만 전이로 규모가 작아 주행 일반화 주장은 제한적이다.

## 참고문헌

- [ROMS-IMLE: A Minimalist Approach to Competitive Single-Step Generative Modelling](https://www.alphaxiv.org/abs/2607.19332) (Vashist & Li, 2026.07)
- [WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](https://arxiv.org/html/2602.14351) (Aghabozorgi et al., ICLR 2026)
- [Implicit Maximum Likelihood Estimation for Real-time Generative Model Predictive Control](https://arxiv.org/abs/2603.13733) (Lee et al., 2026.03)
- [Flow Matching in Feature Space for Stochastic World Modeling (FlowWM)](https://huggingface.co/papers/2606.29059) (Porcher et al., 2026.06)
- [VFMF: World Modeling by Forecasting Vision Foundation Model Features](https://www.alphaxiv.org/overview/2512.11225) (2025.12)
- [DriftWorld: Fast World Modeling through Drifting](https://huggingface.co/papers/2607.15065.md) (2026.07)
- [GE-Act 2.0: Pretraining and Scaling a World-Action Model](https://arxiv.org/pdf/2609.05588) (2026.09)
- [Back to the Features: DINO as a Foundation for Video World Models (DINO-world)](https://arxiv.org/abs/2507.19468v1) (Baldassarre et al., 2025.07)
- [Learning Visual Feature-Based World Models via Residual Latent Action (RLA-WM)](https://arxiv.org/pdf/2605.07079) (2026.05)
- [Improved Mean Flows (iMF)](https://arxiv.org/abs/2512.02012v1) (Geng et al., 2025.12)
- [Multiplayer Interactive World Models with Representation Autoencoders](https://arxiv.org/pdf/2607.05352) (2026.07)
- [MWM: Mobile World Model with Inference-Consistent State Distillation](https://arxiv.org/html/2603.07799) (2026.03)

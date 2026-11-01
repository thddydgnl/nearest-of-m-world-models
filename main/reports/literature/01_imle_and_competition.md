# 문헌 조사 1: IMLE 계열과 경쟁 연구 확인 (조사일 2026-10-11)

출처: arXiv API 전체검색("implicit maximum likelihood estimation", IMLE 제목/초록, IMLE AND (video OR "world model" OR DINOv2 ...)), Semantic Scholar 인용 그래프(WIMLE 피인용 3건, ROMS-IMLE 피인용 0건), Ke Li SFU 출판 목록, 웹 검색. OpenReview/워크숍은 확인 불가.

## 1. 경쟁 연구 확인

**결론: 2026-10-11 기준, IMLE(또는 ROMS/RS-IMLE)를 이미지·비디오·시각 피처 world model, video prediction, action-conditioned 시각 생성에 적용한 논문은 없음.** DINOv2 피처 + IMLE 조합도 없음.

2026년 IMLE 관련 arXiv 논문 (관련 순):
- **WIMLE** (2602.14351, ICLR 2026): 상태벡터 world model. 이미지 실험 없음. "Extending WIMLE to image-based control is an important direction"을 명시적 future work로 남김.
- **IMLE-MPC** (2603.13733, ICRA 2026): 궤적 플래너. world model 아님.
- **ROMS-IMLE** (2607.19332, v1 07-21, v2 09-03): 무조건/클래스조건 이미지 생성. 조건부·비디오·동역학 언급 없음.
- **IMLE-VLA** (2609.10915, IROS 2026; SFU+UPenn, Ke Li 공저): π0.5의 diffusion action head를 단일스텝 cIMLE head로 교체. LIBERO 98.0%, 55 Hz vs 15 Hz. 시각 피처는 입력일 뿐 미래 관측 예측 없음.
- **RTM: Recursive Latent Refinement** (2605.15309; Vashist, Ke Li 공저): latent mapping 반복 정제 + IMLE, 무조건 이미지 생성.
- **From Flow to One Step** (2603.09415): CFM 정책을 IMLE+Chamfer로 1-step 증류. 로봇 정책.
- **PRISM** (2602.02396): batch-global RS-IMLE 모방학습 정책.
- **Chronos** (2606.30318): IMLE를 coarse action prior로 사용.

Ke Li 그룹(SFU APEX) 2026 출판: ROMS-IMLE, IMLE-VLA, WIMLE, IMLE-MPC, RTM, Counterfactual Residual Data Augmentation(ICML 2026), 포인트 렌더링 계열. **이미지 기반 WIMLE 후속작은 없음**(진행 중일 가능성은 있음).

2026년 DINOv2 피처 world model(DINO-world, DINO-WM, DUET-DINO 2609.10506, "Latent Video Prediction Learns Better World Models" 2605.15618, ForgeWM 2608.14022)은 모두 결정론 회귀 또는 diffusion/flow. IMLE 미사용.

## 2. ROMS-IMLE (Vashist & Li, 2607.19332)

- 이름은 Robust Multi-Stage IMLE. 핵심 레시피는 (a) multi-stage supervision, (b) Geman-McClure robust loss. rejection은 ImageNet latent 모델에서 추론 후 round-trip(EQ-VAE encode-decode LPIPS 임계 초과 폐기, 약 5%) → FID 4.16→2.56.
- 목적함수: m개 latent 공유 풀, 각 실제 샘플을 최근접 생성 샘플에 매칭. NN 거리: 픽셀모델 LPIPS, latent 모델 ℓ2. FAISS-GPU.
- Multi-stage: 각 stage 출력 head를 저역통과 다운샘플 타깃과 비교해 합산.
- Robust loss: Geman-McClure를 elementwise residual에 적용. **NN 선택에는 미적용**. 스케일 파라미터 미기재.
- 매칭 예산: 본실험 m = 5n 고정. Appendix C(v2) Flowers ablation (FID):

  | Epoch | m/n=2 | m/n=4 | m/n=10 |
  |---|---|---|---|
  | 500 | 94.42 | 78.65 | 66.37 |
  | 2000 | 36.04 | 25.80 | 20.59 |
  | 3500 | 23.56 | 14.85 | 13.41 |

  "Even with a small sample factor (2), our method converges"; 4→10은 한계효용 감소. **모드 수 대비 예산 분석 없음.**
- 구조: z∈R^1024 → MLP → style w → AdaIN으로 ConvNeXt 디코더 변조. 1 NFE, CFG 없음. 110M(CIFAR-10), 138M(CelebA-HQ), 310M(ImageNet latent).
- 결과: CIFAR-10 FID 4.01, CelebA-HQ 6.70, ImageNet 256 2.56 (iMeanFlow 1-NFE 1.48 대비). Flowers ablation: vanilla 69.35 → +per-stage 35.07 → +GM 15.46.
- 한계/future work 섹션 없음. 조건부·비디오·world model 언급 전무.

## 3. WIMLE (Aghabozorgi, Moazeni, Zhang, Li; ICLR 2026; 2602.14351)

- g_θ(s_t, a_t, z) → [r_t, s_{t+1}]. 미니배치마다 m개 latent를 공유하되 **전이별 별도 argmin**. squared L2. RS-IMLE/robust loss 미사용.
- z 처리: [s,a,z] concat → dense 512 → residual block 3개. z 차원 미기재. m: Table 1 "4", 본문 "5–10" (불일치).
- 불확실도: K=7 앙상블 × m latent 전체 예측의 std. 총분산 분해(epistemic = Var_k E_z, aleatoric = E_k Var_z). w = 1/(σ+1)로 synthetic 전이의 critic 손실 가중. **planning 비용에는 미사용.**
- RL: SAC + quantile critic, MBPO식 rollout. 40 task, 10 seed. Humanoid-run 샘플효율 >50% 개선. 비가중(w=1)은 초반 model-free보다 못함; IMLE WM이 동일 구조 Gaussian WM보다 우수.
- 이미지: proprioceptive만. Limitations에 planning 미탐색, 이미지 확장 future work 명시.

## 4. IMLE-MPC (2603.13733, ICRA 2026) 및 IMLE-for-control

- IMLE를 Diffuser 대체 궤적 생성기로 사용. 조건 FiLM 주입 U-Net. 조건별 latent 풀. Reward-weighted 변형(Boltzmann).
- 샘플 예산 K=4(locomotion), 10(Maze2D). D4RL 평균 78.47 vs Diffuser 77.5. 속도 53.5 Hz vs 2.25 Hz(locomotion), 101 vs 1.37 Hz(Maze2D).
- Future work: IMLE 모드 커버리지를 이용한 uncertainty-aware planning/safety filtering. 이미지/world model 언급 없음.
- 기타: **IMLE Policy** (2502.12371; QUT): 조건별 m=20, RS-IMLE ε=0.03, 111 Hz vs 1.8 Hz; "1-step flow matching이 모드를 평균내고 DP가 다수 모드로 편향되는 반면 IMLE는 모드 유지"(Push-T 정성). MoFlow(CVPR 2025), Social-Implicit(ECCV 2022).

## 5. 조건부 IMLE의 알려진 실패 양상과 대응책

### 5a. 조건당 타깃 1개 → z 무시
- **Li, Zhang, Malik 2019 (1811.12373)**: 조건당 GT 1개면 분포가 "collapse to a single mode". 대응: per-condition 풀(m=10)+per-condition NN, **noise encoder**(저차원 노이즈→전해상도 노이즈 텐서), rarity-weighted loss, dataset rebalancing. LPIPS 다양성: noise encoder 없음 0.10 → full 0.19 — **noise encoder 없이는 z가 사실상 무시됨.**
- **Li, Peng, Zhang, Malik 2020 (2004.03590)**: SR m=100, 레이아웃 m=10. SR에서 z=0 사전학습 후 노이즈 추가.
- **Super-Resolution via cIMLE (1810.01406)**: 조건별 샘플 분리 필수. 계층적 노이즈 샘플링은 효율 목적.
- **HyperRIM (2011.01926)**: "회귀와 조건부 생성 모두 입력당 단일 예측으로 수렴" 지적.
- **CHIMLE (NeurIPS 2022, 2211.14286)**: 문제를 "샘플 수 많아야 고화질"로 규정. z를 L개 직교 성분으로 분할, 해상도별 모듈, coarse→fine 레벨별 m개 선택(총 L·m 패스로 조합 탐색), per-stage supervision. FID 16×SR 16.01(cIMLE 28.42).

### 5b. 매칭 예산 vs 모드/품질
- 원 IMLE(1809.09087): m ≥ n 요구. mode dropping = 어떤 샘플도 고르지 않은 데이터.
- **RS-IMLE (ECCV 2024, 2409.17439)**: "misalignment" — m↑일수록 선택 latent 거리분포가 데이터 쪽으로 쏠려 (a) 가까운 쌍은 gradient 약함, (b) prior 고밀도 영역이 학습에서 거의 선택되지 않음. vanilla IMLE 최적 m≈10–20n. 대응: 학습 이미지 ε 이내 샘플 **거부**. FID 평균 45.9% 개선, recall 대폭 상승. 조건부 논의 없음.
- **Adaptive IMLE (ICML 2023)**: 데이터별 반경 τ_i 커리큘럼. "샘플 수 증가의 대안". m=10000(few-shot).
- **모드 수 대비 필요 m의 정량 분석은 어느 논문에도 없음** → IMLE-WM이 기여할 수 있는 지점.

### 5c. 대응책 목록
1. per-condition latent pool + per-condition argmin
2. noise encoder / 저차원 노이즈 매니폴드, z=0 사전학습 후 노이즈 도입
3. 계층적 latent 분할 + coarse→fine 선택(CHIMLE), per-stage supervision
4. rejection ε(RS-IMLE, IMLE Policy ε=0.03, PRISM), round-trip rejection(ROMS)
5. adaptive 반경 커리큘럼(Adaptive IMLE), robust loss(GM)
6. 희귀 샘플 가중/리밸런싱, reward weighting
7. 앙상블×latent 분산 분해(WIMLE)

## 불확실성
- OpenReview·워크숍·비공개 진행 연구 미확인. WIMLE 저자들이 이미지 확장을 future work로 남겨 2026 하반기 후속 가능성 있음.
- WIMLE m 값 불일치, ROMS GM 스케일·rejection 임계·iteration 미공개.

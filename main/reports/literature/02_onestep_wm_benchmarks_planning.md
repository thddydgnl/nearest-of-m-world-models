# 문헌 조사 2: 1-step 확률 world model, 벤치마크, 노출 편향, DINO 피처 planning (조사일 2026-10-11)

## 1. 1-step / few-step 확률 world model (2025–2026)

**DriftWorld** (2607.15065, 2026-07; code Susie-Lu/driftworld)
- 픽셀(96×96, Push-T/Robomimic) 또는 SD3 VAE latent(Bridge-V2/RT-1) 예측. drifting 손실은 DINOv2/v3 피처 + VAE latent 공간. 3 history + action chunk, T=4(Push-T).
- NFE=1. H100 latency: Push-T 0.0037 s/frame(GPC 0.0104, Ctrl-World 1.77), Bridge 0.030(IRASim 1.10). 30+ fps, 17× 평균 가속.
- 1 positive vs N_neg(8/32/64) 생성 negatives, 커널 exp(-‖x-y‖/(τ√C)), motion weighting, action-emphasis scale.
- **diversity/coverage/best-of-N 지표 없음.** per-sample fidelity + 하류 제어(GPC-RANK Push-T IoU 0.635→0.781). Planning은 GPC-RANK(K=50 diffusion policy proposal 랭킹), CEM/MPPI 아님. 한계: 64 negatives 메모리.

**GE-Act 2.0** (2609.05588, 2026-09, AgiBot)
- CoAE latent(24 tokens/frame) 예측, SigLIP2/V-JEPA 2.1/DINOv3 teacher 정렬. NFE=1 MeanFlow/iMF, 평균속도 head + 순간속도 보조 head, JVP target.
- 다봉성: "outcome modes", validity gap, KASO(N noise → IDM energy로 top-k). 다봉성 정량화는 toy 4-Gaussian만. 1-step vs multi-step 품질 비교 없음.

**FlowWM** (2606.29059, 2026-06/07; code facebookresearch/Flow-World-Models)
- frozen DINOv3 ViT-S 패치 피처(layers 3/6/9/12 평균, d=384), 16 ctx→16 future. DiT 폭 256 + **1024-d wide head**(핵심). timestep shift α=13, weighting τ^γ(γ=15).
- **NFE=50 Euler.** 1-step 아님. latency 미보고. 64 V100 25h.
- **Bouncing Shapes** 벤치마크(벽 충돌 시 normal bounce vs velocity reversal w.p. 0.5, 열거 가능): precision/recall/F1 error(낮을수록 좋음): DINO-WM 16.1/19.9/17.8, deterministic 15.7/18.8/17.1, Walker 재구현 14.5/14.3/14.4, **FlowWM 4.35/4.28/4.31**, oracle ~1.0.
- FuturePerception(Waymo) best-of-N AP_L(3)/(6): FlowWM 20.9/21.7 vs deterministic 15.2/15.2. deterministic은 N에서 이득 0.
- 한계: context-only(행동 없음, planning 없음), 추론 비용 미보고.

**VFMF** (2512.11225, 2025-12): DINOv2-B 피처를 conv VAE로 16채널 압축, rectified flow, NFE=10. Cityscapes/Kubric MOVi-A(64 GT futures/scene). mean-of-k vs best-of-k. 다양성 지표 없음. latency 미보고.

**MWM** (2603.07799, 2026-03): NWM 1B CDiT, VAE latent, ICSD로 5-step DDIM(NWM 25–250). 확률성 평가 없음.

**Matrix-Game 3.5** (2608.29910): 3-step causal generator, Self-Rollout DMD 커리큘럼. 다봉성 지표 없음.

기타: ForgeWM(2608.14022, 1/2/4-step 증류, 2-step 50 FPS), Flash-WAM(2606.05254, 1 step/modality, 23×), Causal-rCM(2606.25473), Walker et al. 2025(2507.13942, frozen-feature DDPM, FlowWM의 baseline), RLA-WM(2605.07079, MPC planning 아님), Hydra-0(2608.18077).

**공백: DINOv2 패치 토큰 위에서 행동 조건부 1-step(NFE=1) 확률 예측 + CEM planning을 한 논문은 없음.** 가장 가까운 것: DriftWorld(픽셀/VAE, DINO는 손실에만), GE-Act 2.0(MeanFlow on CoAE latent), FlowWM(50-step, 행동 없음).

## 2. 확률/다봉 시각 예측 벤치마크와 지표

- **FlowWM Bouncing Shapes**: 유일한 최신 열거형 mode-coverage 벤치마크. precision/recall/F1 error. context-only → 행동 조건 변형은 새로 만들어야 함. 코드 공개.
- **FuturePerception(Waymo)**: best-of-N AP, depth RMSE, FVD. deterministic이 N에서 이득 0이라는 sanity check.
- **Walker et al. 2025**: 예측 vs GT 궤적의 Gaussian Fréchet 거리, 분산 비율(대부분 under-disperse), 10 샘플.
- **VFMF Kubric**: 128 scene × 64 GT futures, coverage 지표 없음.
- 고전(Stochastic Moving MNIST, BAIR, KTH): best-of-100 PSNR/SSIM/LPIPS, 샘플 spread, FVD. 2025–26에 mode-coverage 추가한 논문 없음. BAIR는 행동 조건 시 확률성 낮음.
- Push-T 확률 변형, Procgen, CARLA, Habitat: 확립된 world-model mode-coverage 벤치마크 없음 → **Push-T/Wall 확률 변형은 새로움.**
- **SLIM** ("Keeping JEPA World Models Plannable When Little of the Frame Moves", 2610.03137, 2026-10): 다중 물체 pushing 벤치마크, 프레임의 ~0.37%만 변함. 확률적이지 않지만 작은 물체 피처 비용 문제를 직접 겨냥.

## 3. 노출 편향 / 오류 누적

- **Diffusion Forcing** (2407.01392): 토큰별 독립 노이즈. LIVE(2602.03747) 비판: "noise 주입은 실제 rollout 오류를 모델링하지 못함."
- **Self Forcing** (2506.08009, NeurIPS 2025): 자기 KV-cache 생성에 조건화 + DMD. LIVE: "unbounded error accumulation".
- **Dynamic Latent Bootstrapping** (Sword, 2605.07288, 2026-05): latent diffusion WM용 **scheduled-sampling 유사**: 자기 예측 latent 캐시를 GT history와 확률적으로 교체, 확률을 학습 중 증가(linear decay 최선). LIBERO FVD 48.5→32.2.
- **Persistent Robot World Models** (2603.25685): 자기 생성 prefix(0–9 clips)에 RL post-training, 70 frames ≈ 11 s. "random branch depth가 compounding error 분포를 더 잘 포착".
- **Terver et al. 2025 "What drives success in physical planning with JEPA WMs?"** (2512.24497; code facebookresearch/jepa-wms): multi-step rollout loss는 "augmentation against compounding error"; 성공률은 teacher forcing → **2-step rollout loss에서 상승, k>3에서 하락**; 최선 = 2-step "last-gradient-only" + TBTT detach. "faithful open-loop unrolling does not guarantee planning success". step-1 embedding loss와 success의 Spearman: Push-T 0.86, Wall 0.81, Maze 0.57.
- **DINO-world** (2507.19468): ~200 ms, 0.5 s 앞만 평가, "1 s 접근 시 모두 부정확", "sampling one of the possible futures"를 future work로.
- **"A Control Theory of Predictability in Latent World Models"** (2607.10362, 2026-07): 평균 rollout MSE는 제어 성공과 분리(rank corr 0.024 vs reachable-measure 0.62). multi-step rollout loss는 pusht 84.7→91.3%(3 seed)에서 도움, tworoom은 아님. PP-DINO-WM(2610.08627) parallel-span 예측.
- 기타 2026: AR Forcing, Recency Forcing, Next Forcing, Rolling Forcing, WorldCompass RL. **horizon 지표는 비표준.**

## 4. DINO 피처 world model과 planning

- **DINO-WM** (2411.04983, ICML 2025): frozen DINOv2 ViT-S/14, ViT predictor, teacher forcing, ctx H=3(Push-T/Maze), frameskip 5. 비용 ‖ẑ_T − z_g‖² 패치 임베딩 MSE. CEM 100 samples × 10 iters. MPC-CEM Maze 0.98 / Push-T 0.90 / Wall 0.96 (open-loop CEM 0.80/0.86/0.74). Terver 프로토콜에서는 81.6/64.1/66.0. "Closing the Train-Test Gap"(2512.09929)은 Wall CEM 재현 실패.
- **JEPA-WM (Terver 2025)**: 비용 L1/L2/neg-cos + α·proprio; **L2가 L1보다 일관되게 우수**; CEM(N=300, H=6, K=10, J=30) > Nevergrad/Adam/GD. Maze 83.9, Wall 78.8, Push-T 70.2. DINOv3는 사실적 데이터에서만 우위. proprio 일관되게 도움.
- **V-JEPA 2-AC** (2506.09985): CEM L1, action L1-ball 0.075, ~16 s/action planning.
- **DINO-world** 행동 조건 fine-tune(512 eps): Push-T 59.4 / Wall 93.8 / PointMaze 68.7.
- **피처 거리 비용의 문제**:
  - **SLIM** (2610.03137): LeWM Push-T 98% but **SLIM 3/1,200**; latent가 행동 둔감, 10-step rollout 오류 ≈ 실제 latent 이동의 12×, CEM 비용 지형 평탄(CV 0.001). 역동역학 보조 손실로 0.003→0.348. frozen DINOv2-S 0.048→0.163.
  - 객체 중심 논문(2606.08775): 패치 동역학이 "dominant agent·background에 과집중, 작은 물체 신호 최소화".
  - 배경 민감성(2602.18639): DINO-WM PointMaze 0.80→0.48 (배경 색 변화).
  - **거리/horizon**: Hierarchical Planning(2604.03208): flat DINO-WM Push-T 성공 d=25/50/75에서 **84%/55%/17%**. "Planning Limits of Latent WMs"(2609.39235): 5–10 step 앞 목표만 신뢰; cosine/Euclidean 목표 거리는 단조 진행 가정.
  - AWM/OWM(2512.09929): open-loop GBP +18/+20/+30.
- planning 성공률은 논문 간 비교 불가(목표 거리·에피소드·replanning·elite 다름).

## 5. 분산/불확실성 인지 planning

- **PETS**: 불확실성은 전파용, 페널티 아님.
- **Radic Webster & Flach 2021** (2111.04972): R − β·ω(앙상블/particle 분산, step별 running mean으로 정규화 필수). 안전해지지만 **보상 하락**(risk–return trade-off).
- **VP2 benchmark** (2304.13723): MPPI에 앙상블 disagreement 페널티 → 1 task 개선, 2 task 동등/약간 악화(미검증).
- **Intervention Gap** (2608.29998, 2026-08): 앙상블 disagreement는 support 근처에서만 오류 순위 예측. ART(2609.21482): epistemic 페널티 증가는 imagined 보상 하락, 하류 정책 차이 없음.
- **EpicWorldModel** (2610.05996, NeurIPS 2026): flow-matching 확률 JEPA; 샘플 분산 → 예측 엔트로피 bound; CEM에서 **불확실성 추구 보너스**(반대 부호)로 PointMaze Giant +22%.
- **ELVIS** (2605.04709): 앙상블 critic UCB로 MPPI λ_t 가변.
- **WIMLE**: 앙상블×latent 분산을 critic 가중에만 사용, planning 미탐색.
- **CVaR + 학습 시각 latent WM** 조합 논문 없음(RA-MPPI는 해석적 모델; 과도 위험회피 시 성능 저하 보고). **분산 페널티 MPC over DINO 피처 WM은 미탐색이나, VP2·Intervention Gap 결과는 raw 샘플 분산이 off-distribution 오류를 추적하지 못함을 시사.**

## 교차 공백 정리
1. DINOv2 패치 토큰 위 행동 조건부 NFE=1 확률 예측기 + CEM planning: 없음.
2. 열거형 mode-coverage 벤치마크는 FlowWM Bouncing Shapes뿐(context-only). 행동 조건 변형 또는 확률 Push-T/Wall은 새로움.
3. 오류 누적 지표 비표준. 가장 방어 가능한 것: 고정 지표 vs horizon(학습 horizon 표시) + planning 성공(예측 MSE와 제어 성공은 분리됨).
4. 피처 거리 비용은 프레임 변화가 작을 때(SLIM)와 Push-T 25–50 step 이상에서 실패. 샘플 분산 항은 보너스(EpicWM)/혼합(VP2)/경고(Intervention Gap) — 가정이 아니라 ablation 대상.

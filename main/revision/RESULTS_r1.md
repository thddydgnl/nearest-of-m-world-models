# Revision round r1: results and decisions

Status: final (2026-11-23). Design: `DESIGN_r1.md` v1.0, frozen before any r1 run (SHA-256
959409ddf4ddc6be671a092454c1d4ad47d6396f06228790ee4ec2f6b7a157bc). Every number below is printed by
`python3 main/revision/summarize_r1.py` (experiments 1-3) or `python3 main/revision/an_analyses/summarize.py`
(analyses a-h) from the pulled raw files; spreads are sample standard deviations (n - 1) with the individual seeds.

## 1. Decisions

| Rule (DESIGN_r1.md) | Outcome | Decision |
|---|---|---|
| Experiment 1, primary cell m = 4: H_I if arc(4) in [0.264, 0.360] and control - arc >= 0.10; H_m if abs(control - arc) < 0.05 | arc 0.418, control 0.517, difference 0.0995 | **inconclusive** (neither condition of H_I holds; H_m is contradicted) |
| Experiment 1, secondary cell m = 2 (same rule) | arc 0.688, control 0.838, difference 0.149 | **inconclusive** |
| Experiment 1, m = 8 (reported only) | arc 0.321, control 0.313 | no difference |
| C1 attenuation correction if ceiling_arc < ceiling_std - 0.05 | both ceilings 1.000, floors 0.250 | not applied |
| Experiment 2a: claim limited to equal steps if FM-50 coverage >= 0.92 at 36k | 0.849 | **not triggered**; coverage claim holds at matched wall-clock |
| Experiment 2b: "longer training brings memorization back" if index > 0.40 or coverage < 0.85 at 48k | index 0.320, coverage 0.890 | **not triggered** |
| Experiment 3: drift attributed to training without context noise if h = 32 manifold <= 2.2 at coverage >= 0.74 | 4.47 (S = 0.1), 3.41 (S = 0.3) | **not triggered**; claim kept with the qualifier "including context-noise augmentation up to 0.3" |

Predictions with tolerances (nine): H_I at m = 2, 4, 8 (held at m = 8 only); P2a coverage 0.80-0.90 (held) and
FM index at NFE 50 <= 0.32 (failed, 0.379); P2b index <= 0.40 and coverage >= 0.85 (both held); P3 at S = 0.1 and 0.3
(held). In total 6 held and 3 failed.

## 2. What was run

- Hardware: five cloud GPU machines instead of the planned RTX A5000 only (that type was not available): experiment 1 on one
  RTX A4500 (20 GB; arc and control runs on the same GPU, as the design requires), experiment 2a on one RTX A5000,
  experiment 2b and experiment 3 on one NVIDIA A40 each, analyses a-h on one A40. Environment: the pinned freeze of
  the original environment (`main/code/requirements.txt`). Several runs shared each GPU, so training times of r1 runs
  are not comparable with the paper's tables; timing claims use analysis g, measured on the idle A5000.
- Data and checkpoints: restored read-only from the backup of the original runs; all re-evaluations of original
  checkpoints use the original `stats.pt` and `probe.pt`.
- Smoke test (design section 0): re-evaluating `e1_a2_imle16_s0` gave `stoch.cov_pos` 0.93812 (stored 0.94125) and
  `stoch.prec_pos` 0.83094 (stored 0.8314), within the 0.02 tolerance.
- Runs: experiment 1, `r1_arc90_m{2,4,8}_s{0,1}` and `r1_k4_m{2,4,8}_s{0,1}` (8k steps); experiment 2a,
  `r1_fm36k_s{0,1,2}` (36k steps, checkpoints every 12k); experiment 2b, `r1_imle16_48k_s{0,1}` (48k steps);
  experiment 3, `r1_fmcn01_s{0,1,2}` and `r1_fmcn03_s{0,1,2}` (12k steps). Arguments: `main/revision/queues/`.
- Arc dataset `bounce_div_k4_arc90`: generated with the design's generator and seed; 40,000 training and 10,000 test
  frames; the share of stochastic transitions is 0.243 (0.289 on the semicircle), because the narrower arc sends the
  ball back to a wall less often.

### Operational changes (not design changes; no hypothesis, metric, threshold, run or seed changed)

1. GPU type, as above.
2. Experiment 1: `compute_stats` for the arc dataset ran out of GPU memory next to two control runs (the original
   server hit the same limit); the statistics were computed on the CPU with the same code and the same subsample size
   (6,000 frames). The standard evaluation of the r1 runs of experiment 1 ran after training, one run at a time,
   because evaluations next to three training runs ran out of memory; the primary measure (the last `memo_train` of
   `log.jsonl`) is written during training and is unaffected.
3. Experiment 2a: in addition to the final checkpoints, the 12k and 24k checkpoints were evaluated at NFE 50 and
   their memorization index measured at NFE 50 (learning curve; reported, not used by any decision rule).

## 3. Experiment 1: mode geometry at fixed K and m

K = 4, 8k steps, two seeds per cell, arc and control runs on the same GPU. Primary measure: the last `memo_train` of
`log.jsonl` (monitor index on training contexts, chance 0.25). Held-out coverage and precision from `evaluate.py`.

| m | Arc 90 deg: index | Control (semicircle): index | Control - arc | H_I prediction (index) | Arc coverage / precision | Control coverage / precision |
|---|---|---|---|---|---|---|
| 2 | 0.688 +- 0.010 (0.695/0.681) | 0.838 +- 0.010 (0.831/0.845) | +0.149 | 0.400 [0.352, 0.448] | 0.411 / 0.651 | 0.337 / 0.602 |
| **4** | **0.418 +- 0.010 (0.425/0.410)** | **0.517 +- 0.040 (0.489/0.545)** | **+0.100** | 0.312 [0.264, 0.360] | 0.709 / 0.702 | 0.651 / 0.687 |
| 8 | 0.321 +- 0.003 (0.323/0.319) | 0.313 +- 0.004 (0.310/0.316) | -0.008 | 0.274 [0.226, 0.322] | 0.838 / 0.727 | 0.872 / 0.757 |

The control runs reproduce the original grid (paper: 0.498 at m = 4 and 0.306 at m = 8).

- **Measurement checks.** C1: on both geometries the ceiling (16 copies of the realized next map) is 1.000 and the
  floor (4 copies of each true branch map) 0.250, with a readout error of 0.11-0.12 px on these maps, so the arc
  modes are resolved as well as the semicircle modes and the attenuation correction is not required. C2,
  tolerance-free index: arc 0.676 / 0.378 / 0.297 and control 0.812 / 0.456 / 0.301 at m = 2 / 4 / 8, the same
  ordering as the primary measure (control - arc +0.136, +0.078, +0.004).
- **Decision (primary cell m = 4): inconclusive.** H_I requires arc(4) in [0.264, 0.360] and control(4) - arc(4) >= 0.10;
  arc(4) = 0.418 lies above the interval. H_m requires |control(4) - arc(4)| < 0.05; the difference is 0.100.
  Secondary cell m = 2: inconclusive by the same rule (difference 0.149, arc above the interval). m = 8: reported
  only (difference -0.008; the arc value lies inside the H_I interval).
- **Reading.** At fixed K and m, putting the four modes on a narrower arc lowers memorization at m = 2 and 4, which
  an account in which only m (or the miss probability, which is the same for both geometries) matters cannot
  produce. The reduction is much smaller than the direction-space distortion predicts: the excess falls to 0.63 of
  the control value at m = 4 (0.75 at m = 2), whereas I(m,4) falls to 0.27 (0.29).
- **Post-hoc observation (not pre-registered).** The same distortion computed in the loss's own metric (analysis a,
  `out/a_itok/`) falls by much less on the arc: I_tok arc/control 0.57 at m = 4 (0.60 at m = 2), and after the
  Geman-McClure transform 0.66 (0.69), close to the observed ratios. Across K (analysis a), however, I_tok fits worse
  than the direction-space I. No single distortion measure accounts for both experiments.
- **Consequence (design section 1, inconclusive):** report and weaken. The claim that memorization is driven by a
  codebook distortion rather than by the miss probability stands (both the K grid and this test contradict an
  m-only or miss-probability account); the claim that it is governed by I(m,K) as defined in direction space, and
  the "one curve" framing, are dropped.

## 4. Experiment 2: longer training

### 4.1 Experiment 2a: flow matching for 36k steps

FlowWM recipe on A2-div, 36k steps (cosine schedule over 36k, so the 12k and 24k checkpoints are mid-schedule and
not comparable with the paper's 12k runs), three seeds. Coverage and precision on the 400 held-out contexts;
memorization index with the monitor's protocol (200 training and 200 held-out contexts, 16 samples, sampling seed
123) at the stated NFE.

| Checkpoint | NFE | Coverage | Precision | Det. err. (px) | Index, training | Index, held-out | Excess over held-out |
|---|---|---|---|---|---|---|---|
| paper, 12k (`e1_a2_fm_s*`) | 50 | 0.786 (0.795/0.784/0.777) | 0.51 | 2.24 | 0.297 | 0.253 | 0.059 |
| 12k of 36k | 50 | 0.809 +- 0.010 (0.819/0.798/0.811) | 0.558 | 1.87 | 0.299 +- 0.004 (0.304/0.297/0.297) | 0.251 | 0.064 |
| 24k of 36k | 50 | 0.856 +- 0.004 (0.853/0.853/0.861) | 0.688 | 1.48 | 0.346 +- 0.007 (0.339/0.353/0.344) | 0.250 | 0.127 |
| 36k | 50 | **0.849 +- 0.012 (0.853/0.836/0.858)** | 0.722 | 1.38 | **0.379 +- 0.009 (0.389/0.371/0.378)** | 0.258 | 0.163 |
| 36k | 2 | 0.647 +- 0.007 (0.641/0.647/0.654) | 0.594 | 1.05 | 0.439 +- 0.015 (0.431/0.430/0.456) | 0.236 | 0.266 |
| IMLE-WM m = 16, paper, 12k | 1 | 0.929 | 0.82 | 1.55 | 0.29 | | |

Training monitor (20 Euler steps), per seed at 6k/12k/18k/24k/30k/36k: s0 0.30/0.33/0.31/0.34/0.37/0.38,
s1 0.28/0.29/0.32/0.35/0.35/0.37, s2 0.28/0.30/0.33/0.35/0.39/0.37. Rollouts of seed 0 (`e3.py --hmax 32`):
manifold distance at h = 32 is 4.86 (NFE 50) and 4.87 (NFE 2), against 6.02 and 6.31 for the paper's 12k seed 0.

- Decision rule (FM-50 mean coverage at least 0.92 at 36k): **not triggered** (0.849). The coverage advantage of
  IMLE-WM with m = 16 at 12k (0.929, 1.05 h on the original A5000) over FM-50 holds at three times FM-50's training
  steps, which is more than equal wall-clock time; FM-50's coverage stops rising after 24k steps (0.856 at 24k, 0.849 at 36k).
- Prediction P2a: coverage between 0.80 and 0.90 **held** (0.849); FM memorization index at NFE 50 at most 0.32
  **failed** (0.379). Flow matching memorizes the realized outcomes of training contexts as training continues:
  its index rises from 0.30 at 12k to 0.38 at 36k while its held-out index stays at chance (0.25-0.26), the
  signature of outcome memorization defined in the paper. With two Euler steps the same model gives 0.44.

### 4.2 Experiment 2b: IMLE-WM with m = 16 for 48k steps

Two seeds, 48k steps (cosine schedule over 48k). Seed 0 ran on x2.2. Seed 1 was run three times: the first attempt
stopped at 36,600 steps after a storage failure (exit code 0, no error; kept as `r1_imle16_48k_s1_partial_stop36600`),
the second was lost at about 20k steps when the watchdog's experiment-2 cost cap deleted its pod, and the third ran to
48k on pod x2.3 (same arguments and seed, same GPU type). The monitor values of all attempts agree within noise.

| Step | Index, training (s0 / s1) | Index, held-out (s0 / s1) |
|---|---|---|
| 12k | 0.286 / 0.289 | 0.242 / 0.249 |
| 24k | 0.283 / 0.283 | 0.246 / 0.246 |
| 36k | 0.307 / 0.305 | 0.247 / 0.238 |
| 48k | **0.322 / 0.318** (mean 0.320) | 0.247 / 0.233 |

At 48k: held-out coverage **0.890 (0.891 / 0.889)**, precision 0.880 (0.879 / 0.881), deterministic-transition error
1.48 px (1.45 / 1.50). For reference, the 12k-step model of the paper's Table 1: index 0.29, coverage 0.93.

- Decision rule (index above 0.40 or coverage below 0.85 at 48k): **not triggered**. "Removes the incentive" is
  dropped from the paper in every case, as the design requires; the paper reports that the index rises only from 0.29
  to 0.32 and the coverage falls from 0.93 to 0.89 over four times the training.
- Prediction P2b (index at most 0.40 and coverage at least 0.85): **held** (both parts).

## 5. Experiment 3: context noise for flow rollouts

FlowWM recipe on A2-div (`bounce_div_k4`), 12k steps, context noise S in {0.1, 0.3} (standardized token units,
training only), three seeds each. Rollout metrics from `e3.py --hmax 32` (128 held-out sequences, 16 rollouts);
manifold distance normalized by the median real-to-real nearest-neighbour distance of the same bank as in the paper.

| Model | NFE | One-step coverage | Precision | Manifold h=16 | Manifold h=32 | Best-of-16 h=32 (px) | Spread h=32 (px) |
|---|---|---|---|---|---|---|---|
| FM, no noise (paper, `e1_a2_fm_s*`) | 50 | 0.786 (0.795/0.784/0.777) | 0.51 | | 6.16 (6.02/5.91/6.53) | 9.8 | 27.4 |
| FM, no noise (paper) | 2 | 0.60 | 0.46 | | 6.36 (6.31/6.23/6.54) | 10.6 | 28.6 |
| FM, S = 0.1 | 50 | 0.774 +- 0.004 (0.774/0.771/0.778) | 0.514 | 2.67 | 4.47 +- 0.13 (4.44/4.35/4.61) | 9.7 | 26.4 |
| FM, S = 0.1 | 2 | 0.590 +- 0.008 (0.598/0.581/0.590) | 0.463 | 2.61 | 4.37 +- 0.06 (4.35/4.33/4.45) | 10.4 | 23.8 |
| FM, S = 0.3 | 50 | 0.774 +- 0.011 (0.771/0.787/0.765) | 0.509 | 2.42 | 3.41 +- 0.06 (3.36/3.39/3.47) | 9.4 | 27.2 |
| FM, S = 0.3 | 2 | 0.595 +- 0.016 (0.614/0.588/0.584) | 0.482 | 2.35 | 3.28 +- 0.02 (3.27/3.26/3.31) | 12.3 | 20.4 |
| Regression (paper) | 1 | 0.15 | 0.61 | | 1.80 (1.76/1.76/1.86) | 32.2 | 0 |
| IMLE-WM, m = 16 (paper) | 1 | 0.93 | 0.82 | | 1.68 | 9.3 | 29.0 |

- Prediction P3 (context noise lowers FM-50's h = 32 manifold distance below 6.16 but keeps it above 1.80 at both
  levels; at S = 0.1 one-step coverage drops by at most 0.05): **held** (4.47 and 3.41; coverage drop 0.012 at
  S = 0.1 and 0.012 at S = 0.3).
- Decision rule (either level reaches a manifold distance of 2.2 or less at h = 32 with coverage at least 0.74):
  **not triggered**. The rollout claim stays, with the qualifier "including with context-noise augmentation up to
  0.3". Context noise does reduce the drift (by 27% at S = 0.1 and 45% at S = 0.3 at NFE 50; 31% and 48% at NFE 2), so the
  paper must report it as a partial remedy rather than as irrelevant.

## 6. Analyses a-h (no training; original checkpoints)

Raw outputs: `main/revision/an_analyses/raw/an.1/out/<analysis>/`; tables: `python3 main/revision/an_analyses/summarize.py
<section>`. The design (section 4) fixes no decision rule for these analyses except 4f; they are reported.

- **a. Token-space distortion.** One-variable linear fits of excess memorization on the 16 grid points (K, m) other
  than (2, 2), RMSE (leave-one-K-out RMSE): I(m,K) in the unit-direction geometry 0.0315 (0.044); I with the actual
  branch positions in px^2 0.0320 (0.045); I_tok, the same distortion in the loss's own metric (RMS distance of
  standardized token maps) 0.0477 (0.063), and after the Geman-McClure transform 0.0509 (0.067); 1/m^2 0.0382
  (0.051); 1/m 0.0402 (0.052); miss probability (1-1/K)^m 0.0678 (0.093); e^(-m/K) 0.0651 (0.090). The paper's
  0.031/0.068/0.065 are reproduced. Distortion in the loss's own metric fits worse than a function of m alone, because
  token distances between modes shrink only 2.6-fold from K = 2 to 16 (nearest pair 0.60 to 0.23) while positional
  distances shrink 7-fold, so I_tok grows with K at fixed m whereas the observed excess does not.
- **b. Grid re-evaluation** (34 checkpoints, 400 training and 400 held-out contexts, 64 samples): the index differs
  from the monitor by at most 0.015 (two-seed means); held-out indices lie in (0.94-1.08)/K; training minus held-out
  stays +0.02 to +0.03 at the largest budgets; the tolerance-free index is lower where models memorize (K = 4, m = 4:
  0.43 vs. 0.49); the readout-free gap (smallest token RMS distance to the realized next map, training minus held-out)
  is negative in all 34 cells and similar at K = 8 and 16 for m = 4 (-0.040, -0.045).
- **c. Memorization index at each model's own NFE** (monitor protocol): FM-50 0.297 at NFE 50 (= NFE 20), FM-2 0.327
  at NFE 2 (held-out 0.233), iMF 0.289 at NFE 1, Shortcut 0.280 at NFE 1. The iMF row of the recipe table at NFE 1:
  coverage 0.440 [0.409, 0.470], precision 0.168 (the paper's 0.55/0.22 were 20-step values).
- **d. Readout check of the share measurement** (asymmetric K = 2): rendered true branch maps are assigned to their
  own mode in all 800 cases per split, so mixtures of known share are measured exactly (error 0.000 for q in
  {0.5, 0.6, 0.7, 0.787, 0.8}); realized next maps give 0.795/0.788 (data frequency). The 0.04-0.05 excess of the
  model shares over theory is in the samples, not in the readout.
- **e. Planning replays** (branch-risk task, nine planners): replays reproduce success and pit rates within seed noise.
  Pit probability, true / model at the last imagined frame / model in any imagined frame: selected plans IMLE-WM
  0.164 / 0.070 / 0.169, FM-2 0.105 / 0.022 / 0.027, regression 0.141 / 0.033 / 0.049; executed sequences IMLE-WM
  0.371 / 0.142 / 0.370, FM-2 0.240 / 0.107 / 0.143, regression 0.323 / 0.197 / 0.293; five straight pushes (true
  0.427) last / any: IMLE-WM 0.104 / 0.441, FM-2 0.338 / 0.384, regression 0.218 / 0.260, FM-50 seed 0 0.419 / 0.513.
  The paper's diagnosis (CEM exploits model errors under a mean-distance cost) does not fit IMLE-WM; its imagined
  falls mostly leave the pit before the last frame, which the cost reads.
- **f. Diagnostics without saved outputs**, all regenerated: novelty ratios 1.921 (A2-div) and 0.354 (A2-fix); the
  true-future spread of the Push-T table 30.8/48.3 (reproduced only with the definition in `bdiag.py`, mean distance of
  the block center at step h from the mean of 64 futures over 32 starts; the caption now states it); contact error
  11.2-12.6 px (paper 11.2-12.6); BAIR spread-to-error ratios equal on training and held-out contexts within 0.03.
  Nothing had to be removed under rule 4f.
- **g. Timing** on one otherwise idle RTX A5000 (`out/g_timing/timing.json` under `x2_long/raw/x2.1/`). Batch-1
  latency per predicted frame, mean of five repeats of 50 passes (A2-div, 8M): regression 5.8 ms, FM-50 203 ms,
  FM-2 11.3, iMF 6.5, Shortcut 6.8, energy score m = 8 / 16 6.4 / 6.1, IMLE-WM m = 4 / 16 7.5 / 7.7 ms (stochastic
  Push-T, 21M: 10.3, 286, 18.3, 9.6, 8.8, 8.4 / 9.3 ms). Training step at batch 128 (8M): 170 ms for regression and
  FM, 180 Shortcut, 300 iMF, 412 / 699 energy score, 203 / 321 IMLE-WM; 21M: 200, 200, 265, 547, 333 / 622 ms. All
  latencies are about twice the paper's shared-GPU values; FM-50 / IMLE-WM m = 16 is 26x (paper: 32x), IMLE-WM
  m = 16 / m = 4 training 1.6x (paper: 1.7x). The paper now reports these idle-GPU latencies in Table 2.
- **h. Qualitative samples**: 9 models x (3 training + 3 held-out) contexts chosen by a fixed rule; the paper's new
  appendix figure shows the 3 training contexts and the first held-out context.

## 7. Post-hoc analyses (not in the frozen design)

- Manifold distance with a training-split reference bank (review finding A-26), `main/code/imlewm/r1_e3bank.py`; raw: `main/revision/an_analyses/raw/an.1/out/e3bank/`. The paper's bank (6,000 test-split
  frames) contains the true futures of the evaluated rollouts. With 6,000 training-split frames instead (seed-0 A2-div
  models of Table 2), manifold distance at h = 1 / 4 / 8 / 16 / 32:

  | Model | h=1 | h=4 | h=8 | h=16 | h=32 | paper's bank, h=32 (seed 0) |
  |---|---|---|---|---|---|---|
  | True future frames | 1.42 | 1.40 | 1.42 | 1.42 | 1.42 | |
  | Regression | 1.40 | 1.37 | 1.38 | 1.42 | 1.54 | 1.76 |
  | IMLE-WM m=4 | 1.40 | 1.37 | 1.38 | 1.41 | 1.53 | |
  | IMLE-WM m=16 | 1.40 | 1.38 | 1.39 | 1.40 | 1.50 | 1.69 |
  | Energy score m=8 | 1.50 | 1.64 | 1.81 | 2.07 | 2.63 | |
  | FM-50 | 1.44 | 1.49 | 1.66 | 2.21 | 4.12 | 6.02 |
  | FM-2 | 1.41 | 1.45 | 1.60 | 2.19 | 4.31 | 6.31 |
  | iMF | 1.52 | 1.83 | 2.16 | 3.08 | 6.08 | |

  Held-out frames are novel, so the true futures sit at 1.42 rather than 1 against a training bank. No model starts
  below that level, and the ordering at h = 32 is unchanged; the gap between flow matching and IMLE-WM shrinks from
  3.6x to 2.7x.

## 8. Consequences for the paper

- Experiment 1 (inconclusive) and analysis a: "governed by I(m)" and the "one curve" framing are dropped; the paper
  states that memorization depends on the mode geometry, as a distortion does and a miss probability does not, that
  a function of m alone fits the K grid nearly as well (better with the K = 2, m = 2 point), and that the recorded
  geometry test was inconclusive (abstract, contribution 2, Section 7.2, Figure 1, limitation 6, conclusion).
- Experiment 2a: coverage claim kept with "0.85 with three times the training"; the failed index prediction and
  FM's memorization under longer training are reported in Sections 6.1 and 7.1 and Appendix C.
- Experiment 3: Section 9.1 reports that context noise reduces but does not remove the drift.
- Analysis d: the readout-offset explanation of the flattening measurement is removed; Figure 2(d) shows raw shares.
- Analysis e: the planning diagnosis is replaced by the replay result (imagined falls leave an absorbing pit).
- Analyses b, c, f, g, h and the post-hoc training-bank check: Appendix F, Table 2, Table 4, Appendix E and the new
  qualitative appendix figure.
- Experiment 2b: Section 7.1, contribution 2 and Appendix C report the slow rise of the index (0.29 to 0.32) and the
  coverage (0.93 to 0.89) over 48k steps.

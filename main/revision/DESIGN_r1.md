# Revision round r1: design, predictions and decision rules

Version 1.0, written 2026-11-23 before any r1 run. This file is frozen by its SHA-256 and a git commit before the
first r1 job starts; later changes go to section 7 with time and reason, and the frozen text above it is not edited.
Every result of this round is reported in the revised paper or its appendix, whether or not it matches the
prediction.

## 0. Scope and setup

- **Why:** an independent five-reviewer verification of the TMLR manuscript
  (`paper/review/2026-11-22_fable/00_종합.md`) found that three claims rest on evidence that cannot separate them
  from simpler explanations: excess memorization following the codebook distortion I(m,K) rather than m alone,
  the coverage advantage over 50-step flow matching at a training budget where the flow baseline has not converged,
  and rollout drift of flow models trained without context-noise augmentation. Experiments 1 to 3 test these.
  Section 4 lists analyses of existing checkpoints that need no training.
- **Compute:** RunPod Secure Cloud, NVIDIA RTX A5000 (the GPU type of the original runs). Pod prefix `nom`, tools in
  `runpod/`. Budget: the user's cap is USD 30; the watchdog limit is USD 24 because the shared account balance was
  USD 26.62 at the start. Hard deadline for pods: 2026-11-27T00:00Z.
- **Environment:** `main/code/requirements.txt` (the freeze of the original venv: Python 3.10.12, torch 2.6.0+cu124,
  transformers 4.57.6).
- **Code:** `main/code/` at the commit that freezes this file. The only change for r1 is the arc option
  (`common.ARC_DEG`, `common.set_arc_from_name`, the `_arc<deg>` suffix in `gen_data.py`, and the call in
  `check.memo_model`). For every existing dataset name the bounce angles are unchanged (checked numerically).
- **Data and checkpoints:** restored read-only from the Google Drive backup `racedreamer-backup-2026-11`. All
  re-evaluations of original checkpoints use the original `stats.pt` and `probe.pt`.
- **Smoke test, required before any r1 result is used:** re-evaluating the original checkpoint `e1_a2_imle16_s0` with
  `python -m imlewm.evaluate` must give `stoch.cov_pos` within 0.02 of 0.94125 and `stoch.prec_pos` within 0.02 of
  0.8314 (the stored values).
- **Naming:** every new run is named `r1_*`; original runs and their files are never overwritten.
- **Statistics:** means over seeds; spread reported as the sample standard deviation (n - 1) with the individual
  seed values.

## 1. Experiment 1: mode geometry at fixed K and m

**Question.** At fixed K and m, does excess memorization follow the codebook distortion I(m,K) of the mode geometry,
or does it depend on m only? On the semicircle geometry of the paper the two cannot be separated, because I(m,K)
hardly depends on K for K >= 4.

**Data.** `bounce_div_k4_arc90`: the generator of `bounce_div_k4` (randomized appearance, K = 4, 1000 training and
250 test episodes of 40 frames, generator seed 54), except that the four bounce directions are equiangular on a 90
degree arc around the wall normal (+-11.25 and +-33.75 degrees) instead of the 180 degree semicircle. Adjacent modes
are 3.9 to 7.0 px apart instead of 7.7 to 13.8 px. Generated with `python -m imlewm.gen_data --name
bounce_div_k4_arc90`, probe with `python -m imlewm.probe --data bounce_div_k4_arc90`.

**Runs.** The budget-grid recipe of `e2_k4_m*`: `--method imle --stages single --wide_head 1024 --z_from 5 --steps 8000
--memo_every 2000 --m M --seed S`.
- Arc geometry: `--data bounce_div_k4_arc90`, M in {2, 4, 8}, S in {0, 1}: `r1_arc90_m{M}_s{S}`.
- Control on the standard geometry, run in the same environment: `--data bounce_div_k4`, M in {2, 4, 8}, S in {0, 1}:
  `r1_k4_m{M}_s{S}`.

**Primary measure.** Memorization index on training contexts at 8k steps (the last `memo_train` of `log.jsonl`, the
monitor used in the paper), mean of two seeds; excess = (index - 1/4) / (3/4).

**Distortion values** (unit-speed direction space, as in the paper):

| m | I(m,4), semicircle | I(m,4), 90 degree arc |
|---|---|---|
| 2 | 0.5947 | 0.1737 |
| 4 | 0.2351 | 0.0638 |
| 8 | 0.0614 | 0.0161 |

**Predictions** (index scale). H_I uses the line fitted on the K = 4, 8 grid points before the K = 2, 16 runs
(excess = 0.0150 + 1.0630 I, RMSE 0.0321) with +-2 RMSE intervals.

| Hypothesis | arc m = 2 | arc m = 4 | arc m = 8 |
|---|---|---|---|
| H_I: distortion | 0.400 [0.352, 0.448] | 0.312 [0.264, 0.360] | 0.274 [0.226, 0.322] |
| H_m: m only | = control(2) | = control(4) | = control(8) |
| H_p: miss probability (1 - 1/K)^m | = control(2) | = control(4) | = control(8) |

H_p and H_m make the same prediction here because K is the same. For reference, the original runs on the standard
geometry gave 0.498 (m = 4) and 0.306 (m = 8).

**Decision rule** (primary cell m = 4, control measured in r1):
- H_I is supported if arc(4) lies in [0.264, 0.360] and control(4) - arc(4) >= 0.10.
- H_m is supported if |control(4) - arc(4)| < 0.05.
- Any other outcome is reported as inconclusive.
- m = 2 is a secondary test with the same rule (difference >= 0.10 for H_I, < 0.05 for H_m). m = 8 is not expected to
  discriminate (predicted difference 0.03) and is reported only.

**Measurement checks, fixed now** (closer modes are harder to resolve, which biases the index towards chance and
would favor H_I):
- C1, ceiling and floor of the monitor on each geometry: for 200 training contexts, feed (a) 16 copies of the realized
  next token map (a perfect memorizer) and (b) 4 copies of each of the K true branch maps (a calibrated sampler)
  through the same readout and assignment. If ceiling_arc < ceiling_std - 0.05, the attenuation-corrected index
  (index - floor) / (ceiling - floor) becomes the primary measure for both geometries and the decision rule is applied
  to it.
- C2, tolerance-free index: every decoded sample is assigned to its nearest branch position without the tolerance.
  Reported for both geometries.

**Consequences for the paper.**
- H_I supported: keep the distortion claim and report this test as an independent confirmation.
- H_m supported: state that at fixed m the number of modes does not matter and that a geometry change did not
  confirm the distortion form; drop "governed by I(m) rather than by the miss probability" and keep only "not by the
  miss probability".
- Inconclusive: report and weaken accordingly.

## 2. Experiment 2: longer training

**2a. Flow matching at a matched wall-clock budget.** FlowWM recipe (`--method fm --t_shift 13 --stages single
--wide_head 1024`) on `bounce_div_k4`, 36k steps (3 x 0.43 h on the original A5000, which is at least the 1.05 h of
IMLE-WM with m = 16 at 12k), `--save_every 12000`, seeds {0, 1, 2}: `r1_fm36k_s{S}`. Evaluation with `--nfe 50` and
`--nfe 2`; memorization index of the final checkpoint measured at NFE 50 and 2 (section 4c procedure); `e3.py
--hmax 32` at NFE 50 and 2 for seed 0.

**2b. IMLE-WM with m = 16, long run.** `--method imle --m 16 --stages single --wide_head 1024 --z_from 5`, 48k steps,
`--save_every 12000`, seeds {0, 1}: `r1_imle16_48k_s{S}`.

**Predictions.**
- P2a: FM-50 held-out coverage at 36k between 0.80 and 0.90 (0.79 at 12k), below IMLE-WM with m = 16 at 12k (0.929);
  FM memorization index at NFE 50 at most 0.32.
- P2b: IMLE-WM with m = 16 has a memorization index of at most 0.40 at 48k (0.29 at 12k, 0.31 at 18k in one seed)
  and held-out coverage of at least 0.85.

**Decision rules.**
- If FM-50 reaches a mean coverage of at least 0.92 at 36k, the coverage claim over 50-step flow matching is limited
  to equal training steps, and the paper states that the gap closes at equal wall-clock time (the latency advantage
  stays).
- If IMLE-WM with m = 16 exceeds an index of 0.40 at 48k or falls below a coverage of 0.85, the paper states that m =
  16 suppresses memorization within 12k to 18k steps and that longer training brings it back. "Removes the incentive"
  is dropped in every case.

## 3. Experiment 3: context-noise augmentation for flow rollouts

**Runs.** FlowWM recipe on `bounce_div_k4`, 12k steps, `--ctx_noise S` with S in {0.1, 0.3} (standardized token
units, training only), seeds {0, 1, 2}: `r1_fmcn01_s{S}`, `r1_fmcn03_s{S}`. Evaluation with `--nfe 50` and `--nfe 2`;
`e3.py --hmax 32` at NFE 50 and 2.

**Prediction (P3).** Context noise lowers the h = 32 manifold distance of FM-50 below 6.16 (original three-seed mean)
but keeps it above regression's 1.80 for both noise levels; at S = 0.1 one-step coverage drops by at most 0.05.

**Decision rule.** If either noise level brings FM-50's h = 32 manifold distance to 2.2 or less while its one-step
coverage stays at 0.74 or more, the paper attributes the drift to training without context noise, not to the flow
objective, and limits the rollout claim to flow models trained without context-noise augmentation. Otherwise the
claim stays with the qualifier "including with context-noise augmentation up to 0.3".

## 4. Analyses of existing checkpoints and data (no training)

- **a. Token-space distortion.** For `bounce_div_k{2,4,8,16}` and `bounce_div_k4_arc90`, compute I_tok(m,K) from
  `stoch_branch_tokens.npy` (400 evaluation contexts): pairwise RMS distance between standardized branch token maps,
  both plain and after the Geman-McClure transform with the scale of the matching loss. Refit the 16 grid points
  against I_tok and report the RMSE next to I (0.031), m only (0.038 for 1/m^2) and the miss probability (0.068).
- **b. Grid re-evaluation.** The 34 checkpoints `e2_k{4,8}_m*` and `e12_k{2,16}_m*`: 400 training and 400 held-out
  contexts with 64 samples each. Report (i) the index as in the paper, (ii) the tolerance-free index, (iii) excess
  relative to the held-out index, and (iv) a readout-free gap: the mean over contexts of the smallest token RMS distance
  between a sample and the realized next frame, training minus held-out.
- **c. Memorization index at each model's own NFE.** `e1_a2_fm_s{0,1,2}` at NFE 50 and 2, `e1_a2_imf_s*` at NFE 1,
  `e7_a2_sc_s*` at NFE 1, and the iMF row of Table 8 at NFE 1.
- **d. Readout offset of the flattening measurement.** On `bounce_div_k2_p80`, feed true branch token maps mixed at
  known shares q in {0.5, 0.6, 0.7, 0.787, 0.8} through the share assignment and report measured minus true share.
- **e. Risk on planner-chosen actions.** Push task, models `e13_push_{imle16,fm,reg}_s{0,1,2}`: replay the CEM planner
  (task seed 0, 100 tasks), keep the executed action sequences, and compare the model's imagined pit probability (16
  samples) with the simulator's on those sequences; recompute the straight-push risk diagnostic with the last-frame
  criterion that the planner's cost uses.
- **f. Diagnostics with saved outputs.** Novelty ratios for `bounce_div_k4` and `bounce_k4`, the true-future spread
  of Table 4, the Push-T contact error, and the BAIR training-versus-evaluation ratios if time permits. A number in
  the paper that cannot be regenerated is removed from the paper.
- **g. Timing.** On an otherwise idle A5000: batch-1 latency of every configuration in Tables 2 and 4 (5 warm-up and
  50 timed passes, 5 repeats) and the training step time at batch 128.
- **h. Qualitative samples.** The seed-0 A2-div checkpoints of Table 2: 16 decoded samples for one training and one
  held-out context each, for a new Figure 3.

## 5. Pods

| Tag | Work | Data |
|---|---|---|
| x1.1 | Experiment 1 (arc data, probe, 12 runs, C1, C2) | bounce_div_k4 |
| x2.1, x2.2 | Experiment 2a (3 runs), 2b (2 runs) | bounce_div_k4 |
| x3.1 | Experiment 3 (6 runs, rollouts) | bounce_div_k4 |
| an.1 | Analyses a to h | checkpoints, bounce_div_k{2,4,8,16}, bounce_div_k2_p80, push, Push-T files |

## 6. Schedule

- 2026-11-23: restore, smoke test, launch experiments 1 to 3 and analyses a to d.
- 2026-11-24: experiments finished, analyses e to h, `main/revision/RESULTS_r1.md`.
- 2026-11-25 to 27: manuscript rewrite. 2026-11-28 to 29: independent re-verification.
- TMLR submission before 2026-11-30 23:59 AoE (new submissions pause until 2027-01-03).

## 7. Deviations after the freeze

(none yet)

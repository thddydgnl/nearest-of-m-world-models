# Revision r1 log (UTC)

- 2026-11-22T17:17Z Design frozen: `main/revision/DESIGN_r1.md` v1.0, SHA-256
  959409ddf4ddc6be671a092454c1d4ad47d6396f06228790ee4ec2f6b7a157bc, commit fac1bb1 on branch `revision-r1`,
  pushed to GitHub (public timestamp) before any r1 run. The user approved the commit, the push and copying the
  read-only `gdrive-ro` rclone remote to the pods.
- 2026-11-22T17:16Z Drive remote `gdrive-ro` (scope drive.readonly) lists `racedreamer-backup-2026-11`;
  `tier4/imle-wm-data/bounce_div_k4` is 9.456 GiB (8 objects).
- RunPod: shared-account balance 26.62 USD at the start; watchdog limit 24 USD (user cap 30).
- 2026-11-22T17:17Z Watchdog started (`budget.py watch 300`). Five pods requested (A5000, then A4500, then A40,
  Secure): only an.1 and x2.2 came up, both NVIDIA A40 at 0.59 USD/h (Secure stock was short for all three types).
- 2026-11-22T17:30Z Retrying x1.1, x2.1, x3.1 with the Ampere list A5000 | A4500 | A40 | A6000 | RTX 3090.
  Bootstrapping x2.2 (bounce_div_k4, code, venv) and an.1 (checkpoints, bounce_div_k{2,4,8,16}, bounce_div_k2_p80,
  push; code is pushed after the checkpoint archive is unpacked).
- Operational change (not a design change): GPU type is no longer fixed to the A5000. Within-pod comparisons
  (experiment 1 arc vs control) are unaffected; timing claims will use the analysis-g measurements on one GPU type.
- 2026-11-22T17:40Z All five pods up: x1.1 RTX A4500 (0.25/h), x2.1 RTX A5000 (0.27/h), x2.2, x3.1, an.1 A40
  (0.59/h each). Watchdog moved to a detached process (nohup), caffeinate started for 48 h (keep the Mac awake).
- 2026-11-22T17:45Z User approvals before sleeping (2026-11-23 KST): commit and push to `revision-r1` (never main);
  design changes only after a Fable 5.2 max judge agent rules on them, recorded in DESIGN section 7; other GPU types
  and Community Cloud allowed if Secure stock is short (limit stays 24 USD); add the LLM-use footnote used in papers
  A and B; revise abstract, introduction and conclusion claims by the pre-registered decision rules; keep the title;
  prepare the anonymous supplementary zip and the arXiv v2 bundle (uploads and submission are the user's).
  Task: finish steps 1 (analyses + training), 2 (rewrite), 3 (Fable 5.2 max re-verification and fixes); check the
  runs every 30 minutes and fix errors.
- 2026-11-22T17:32-17:40Z Experiments started: x2.2 IMLE-WM m=16 48k (2 workers), x2.1 FM 36k (3 workers, e3 chain
  for seed 0), x3.1 context-noise FM (3 workers for 6 runs, e3 chain), x1.1 control runs (2 workers) and the arc
  dataset chain (then 2 arc workers). Parameter counts match the original runs (8.24M IMLE-WM, 8.17M FM).
  Fixes during setup: tar on the network volume cannot chown (`--no-same-owner`); a `pkill -f` in a remote command
  killed its own shell (fetch relaunched without it); x3.1 uses cgroup v1 (memory readout fixed).
- 2026-11-22T18:2xZ Step 2a (result-independent manuscript fixes) finished: paper/review/2026-11-22_fable/
  CHANGES_2a.md and TRACKER.md; verified: bold now prints, main text ends on page 12, 0 overfull, Table 12 values
  match modeshare JSON. Committed 65b6ca9 and pushed. To fix in step 2b: the Push-T m=4 sentence now says
  "memorization" without a measurement (B-13); weaken it.
- 2026-11-22T18:05Z Heartbeat 1: spend est. 1.72 USD. x1.1: the arc dataset tokens were written (train 40,000 /
  test 10,000 frames; stochastic transitions 0.243 vs 0.289 on the semicircle, a consequence of the narrower arc),
  but `compute_stats` ran out of GPU memory next to the two control runs (the original server hit the same OOM).
  Fix (operational, not a design change): stats computed on the CPU with the same code and n_sub = 6000
  (`runpod/pod/x1fix.sh`), then the probe, then one arc worker (a second when the controls finish; the A4500 has
  20 GB). Other pods: no errors.
- 2026-11-22T19:0xZ Heartbeat 3: spend est. 3.97 USD, no errors. Experiment 1 controls at m = 4 reproduce the
  original runs on the new GPU (r1_k4_m4: 0.489 / 0.545 vs. original e2_k4_m4: 0.478 / 0.518).
- 2026-11-22T19:15Z Figure pass (C review) done and committed (c50ab85). Step 2b preparation committed (76ea6e0):
  Table 5 moved to Appendix F.4, LLM-use footnote on page 1 (wording of paper A, adapted), appendix floats numbered
  S1..., float barriers in the appendix, Figure 2's source moved next to Figure 1 (the footnote had pushed its source
  onto page 10), Push-T m = 4 sentence weakened (B-13), raw flattening shares cited in the full Figure. The
  Conclusion ends on page 12; the Broader Impact Statement spills to page 13 as in v1 (to be recovered in 2b).
- 2026-11-22T20:00Z Heartbeat 5: spend est. 6.19 USD. x1.1 (A4500, 20 GB): the auto-evaluator ran out of GPU memory
  next to three training runs (eval_r1_k4_m2_s{0,1}.log) and would retry every minute, risking OOM in training.
  Stopped it; `runpod/pod/x1more.sh` adds two arc workers when the controls finish, then evaluates all r1 runs one
  at a time and writes /workspace/nom/out/x1_done. The primary measure (memo_train in log.jsonl) is unaffected.
  x3.1: context-noise 0.1 FM, seed 2, NFE 50: coverage 0.778, precision 0.516 (original FM 0.786, 0.507).
- 2026-11-22T20:35Z Heartbeat 6: spend est. 7.30 USD, no run errors. x1.1 (A4500) at 19.7 of 20.5 GB with three
  training runs (two m = 8 controls and arc m = 4 s1). To keep two arc m = 8 runs from running next to a third run,
  `x1more.sh` now adds one arc worker (two in total) when the controls finish. Restarting it failed once: the remote
  `kill` loop matched its own shell (the command line contained `x1more.sh`) and the ssh session died before the new
  script started, so no follow-up script ran for a few minutes. Started again without the kill (20:41Z).
- 2026-11-22T20:45Z New analysis modules (`main/code/imlewm/r1_*.py`, existing modules unchanged by checksum) pushed to
  x1.1 and x2.1 (`env.sh` excluded: it is pod-specific). Follow-up chains: `runpod/pod/x1post.sh` (after x1_done: C1,
  C2 at 200 x 16 and 400 x 64, I_tok for the arc and standard geometry) and `runpod/pod/x2post.sh` (after the FM 36k
  evaluations and the seed-0 rollout chain: stop the auto-evaluator, FM memorization index at NFE 50 and 2 for the
  final checkpoints and at NFE 50 for the 12k and 24k checkpoints, NFE-50 evaluations of the 12k and 24k checkpoints,
  then timing analysis g on the idle A5000).
- 2026-11-22T21:10Z Heartbeat 7: spend est. 8.49 USD, no new errors (x1.1 lists the two stale evaluation logs from
  20:00Z). x3.1: all six context-noise runs trained; evaluations and rollout metrics continue. Experiment 1 primary
  cell complete in training (m = 4, both seeds at 8k); verdict waits for C1.
- 2026-11-22T21:15Z Step 2 side work: new qualitative figure (`paper/build/make_fig_samples.py`,
  `paper/tmlr/figures/fig_samples.pdf`, analysis h; 36 panel counts match the analysis table). Anonymous supplement
  builder `paper/build/make_supplement.sh` (+ `supp_build.py`, `supp_missing_args.json`; output in the git-ignored
  `paper/supplement_build/`): 4.3 MB, identifier grep 0 matches. Run arguments recovered from the checkpoints for 81
  of 82 runs that had none in the repository (only the failed pilot `_failed_mf_k4` has no checkpoint); they show
  that the joint 16-to-16 runs used batch 64 (review D-20) and that `e6_b_seq4_s{0,1}` and `e6_det_*` ran 8,000
  steps as the paper states. The supplement's copy of DESIGN_r1.md is redacted in five places (provider, tool
  paths, account budget, backup, review path), so its hash differs from the frozen file; its README says so.
- 2026-11-22T21:27Z Heartbeat 8: spend est. 9.60 USD, no new errors. x1.1 at 19.8 of 20.5 GB with three runs
  (control m = 8 at step 6,600 of 8,000 and arc m = 2 seed 0); per-process memory is fixed after start, so no
  action. x3.1 evaluating.
- 2026-11-22T21:56Z Experiment 3 complete (6 runs, evaluations at NFE 50 and 2, rollout metrics to h = 32; 24 result
  files, no errors). Decision rule not triggered (h = 32 manifold distance 4.47 at S = 0.1, 3.41 at S = 0.3, against
  the threshold 2.2); P3 held. Results and logs pulled to `main/revision/x3_ctxnoise/raw/x3.1/`; pod x3.1 deleted
  (runpodctl failed on a DNS lookup, the GraphQL fallback terminated it).
- 2026-11-22T22:25Z Heartbeat 9: spend est. 11.55 USD, no errors. FM 36k at 34,600-35,600 of 36,000 steps; x1.1 has
  two arc runs training and one queued; x2.2 at about half of 48k.
- 2026-11-22T23:30Z Analyses a-h finished on an.1 (worker report `main/revision/an_analyses/REPORT.md`; an group
  cost 3.74 USD). Planning replays: all nine runs done.
- 2026-11-22T23:35Z Post-hoc analysis, not in the frozen design (review finding A-26): rollout manifold distance with
  a reference bank from the training split instead of the test split, seed-0 A2-div models of Table 2
  (`main/code/imlewm/r1_e3bank.py`, `runpod/pod/e3bank.sh`, on an.1 before it is deleted).
- 2026-11-22T23:30Z Experiment 2a complete: FM-50 coverage 0.849 at 36k (decision rule not triggered; P2a coverage
  range held), memorization index at NFE 50 0.379 (P2a's bound 0.32 failed; 0.299 at 12k, 0.346 at 24k).
- 2026-11-22T23:47Z Experiment 1 complete (12 runs, evaluations, C1, C2 at 200 x 16 and 400 x 64, I_tok for both
  geometries). C1: ceiling 1.000 and floor 0.250 on both geometries, no correction. Decision at m = 4: inconclusive
  (arc 0.418, control 0.517, difference 0.100; arc above the H_I interval [0.264, 0.360]); m = 2 inconclusive
  (0.688 vs 0.838); m = 8 reported (0.321 vs 0.313). Results and logs pulled; pod x1.1 deleted (GraphQL fallback).

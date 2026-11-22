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

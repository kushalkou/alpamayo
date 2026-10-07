# R7A_REPORT -- no-route arms (AR1 Table 6 "without route")

HEADER (all rules fixed and committed BEFORE computing; val read for analysis only; CoC
arm untouched). Strata for every run (PRE-REGISTERED): all 5,119; excl. first frames;
excl. WSS scenes (10 val scenes, R5 A2 rule); WSS only.
NR = the nav command removed ENTIRELY. There is no text prompt in these models: the
  context is [480 cached visual tokens (3 front cams, t0)] + [4 ego tokens (ego MLP)] +
  [1 learned command embedding token, ego_encoder.xtok.cmd, from the VAD command [P]].
  NR drops the last token: context 485 -> 484 tokens; records.build(extra=[]) so w1_ego
  has 4 rows (no cmd row); no xtok.cmd parameters exist; no "unknown" class. Everything
  after the context (M1-v2a: 18 word tokens, then 24 trajectory tokens; B1: 24 trajectory
  tokens) starts one position earlier. Nothing else changes.
A1 B1-NR s42: w1/finetune_w1.py exactly as B1 (--turn_weighted --seed 42 --vcache t0 --fix
   --batch_size 3 --grad_accum_steps 1 --patience 5 --epochs 10) WITHOUT --cmd; dumps
   w1/dump_w1.py --vcache t0 --fix without --cmd (holdout, val).
A2 M1-v2a-NR s42: ar1/finetune_meta.py --labels 2hz --seed 42 --no_cmd (new flag, default
   off: old runs unchanged); dumps --dump holdout,val --no_cmd.
   Selection for A1, A2 identical to B1 / M1-v2a: holdout AR ADE@6s median on the fixed
   400 holdout samples, 10 epochs, patience 5; tau for the hybrid decode chosen on the
   holdout dump (q2_traj.run). Official val never used for selection.
A3 Ego-MLP-NR: w1/ego_mlp.train_one unchanged (19-512-512-24 minus the 3 command one-hot
   inputs = 16 causal ego features), seeds 42 / 123 / 2024 (CPU, minutes), holdout
   selection as before; vs Ego-MLP + cmd [P] (same seeds, results/w1_egomlp.pkl).
Metrics (per-sample dumps only): L2@3s NoAvg (all 5,119), ADE@6s (n_fut = 12), false-go
   and missed-go (R3), collision @3 s NoAvg / TemAvg vehicles, VAD port and yaw-aware
   (r5_a5c1.coll_rates); M1-v2a-NR also word collapse share, consistency, per-class F1
   (support >= 30), as in R6.
GATES (paired scene bootstrap 10,000, L2@3s NoAvg; one seed, indicative):
   GN1 report B1-NR - B1 s42 and M1-v2a-NR - M1-v2a s42 (cost of the route), all strata.
   GN2 M1-v2a-NR - B1-NR, all val: upper < 0 -> "meta helps without route"; lower > 0 ->
       "meta hurts without route (AR1 direction)"; else "flat without route".
Budget: ~26-32 GPU-h per arm (measured B1 / M1-v2a), ~60 GPU-h total, tmux r7a, queue
   ar1/run_r7a.sh (resumable markers results/R7_<tag>_{TRAINED,DUMPED}).
Part B (CoC arm) only if AUDIT_SIGNOFF.md exists when Part A finishes; its rules go in the
   R7B_REPORT.md header before any Part B compute.
PART C rules (amendment; fixed before computing; C1, C2 CPU; C3 GPU after Part A):
C1 Mediation. M1-v2a {s42, s123, s2024} pooled (sample x seed), GT-stopped stationary val
   (R3: v0_can < 0.5, GT max disp. 3 s < 0.5 m; false-go = pred > 1.0 m). Table: per
   generated slot-1 lon word, n and P(false-go). Go-words = gentle_acc, strong_acc,
   maintain; stop = stop; other = gentle / strong dec, reverse. GM1: go-word share of
   false-gos >= 0.80 AND P(false-go | slot-1 = stop) <= 0.05 -> "start decision is
   mediated by the generated words"; else "not mediated". Old M1 (10 Hz) same table,
   contrast only. B1: not applicable (no words).
C2 Rule-defined strata (ADOPTED AFTER R4, labelled so in every table); v0 = v0_can,
   g = GT max displacement over 3 s: STOPPED v0 <= 0.2 and g <= 1.0; START v0 <= 0.2 and
   g > 1.0; MOVING v0 > 0.2. False-go (STOPPED) = pred max disp. 3 s > 1.0 m; missed-go
   (START) = pred <= 1.0 m (prediction lands in the other stratum). Runs: CV, KIN, Ego-MLP
   + cmd (3), no-cam VLA (3), B1 (3), B2 (s42), M1 (3), M1-v2a (3), no-cam M1-v2a (s42),
   then B1-NR, M1-v2a-NR, Ego-MLP-NR (3) when done. Per stratum: n, L2@3s NoAvg, ADE@6s
   (n_fut = 12), FG / MG; multi-seed = mean of per-run metrics. Paired scene bootstrap
   (10,000) per stratum: M1-v2a 3s - B1 3s, M1-v2a 3s - no-cam VLA 3s. WSS secondary.
C3 Blank-image control (inference, cap 2 GPU-h), BATCHED decoder for both models (a B1
   per-sample val dump alone is ~2.7 GPU-h). Blank = every visual token replaced by one
   3,584-d vector, the per-dimension mean over all TRAIN samples, all 3 cams and all 480
   token positions (CPU, cached tokens). Variants per model, all batched: real, cameras
   shuffled (perm RandomState(99), as R6 V1), blank. M1-v2a s42 reuses the R6 batched
   real and shuffled dumps; B1 s42 gets all three. Report L2@3s, FG / MG per C2 stratum.
   GBL: M1-v2a blank - real (paired scene bootstrap, L2@3s, all val) CI includes 0 ->
   "cameras unused by M1-v2a"; else "cameras used by M1-v2a".

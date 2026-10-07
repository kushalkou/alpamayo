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
=========================================================================================
GATE VERDICTS (per-sample dumps unless "batched"; scene bootstrap 10,000; one seed for NR)
  GN1 B1-NR - B1 s42 +0.010 [-0.048,+0.067]: the route costs B1 nothing measurable.
      M1-v2a-NR - M1-v2a s42 +0.197 [+0.066,+0.354]: removing the route hurts M1-v2a.
  GN2 M1-v2a-NR - B1-NR -0.101 [-0.167,-0.032] -> "meta helps without route" (NOT the AR1
      direction; AR1 0.5B no route: meta 0.291 / 0.988 vs traj 0.282 / 0.971).
  GM1 go-word share of false-gos 0.830 (>= 0.80), P(FG | slot-1 stop) 0.022 (<= 0.05) ->
      "start decision is mediated by the generated words".
  GBL M1-v2a blank - real (batched) +0.244 [+0.170,+0.316] -> "cameras used by M1-v2a".
GPU: Part A 7.0 h x 8 = 56.2 GPU-h; C3 1.40 GPU-h (cap 2). Part B: AUDIT_SIGNOFF.md absent
at 14:05 UTC -> STOPPED, nothing launched.

PART A detail (r7a_eval.txt). Holdout tau: B1 0.3, B1-NR 0.7, M1-v2a 0.3, M1-v2a-NR 0.9.
  L2@3s / ADE@6s   all          excl. first fr.  excl. WSS     WSS only
  B1 s42           2.165 3.125  1.872 2.763      2.188 3.220   1.849 1.823
  B1-NR s42        2.175 3.150  1.874 2.780      2.217 3.261   1.587 1.624
  M1-v2a s42       1.877 2.808  1.558 2.415      2.008 3.005   0.063 0.088
  M1-v2a-NR s42    2.074 3.070  1.755 2.675      2.071 3.132   2.113 2.206
  Ego-MLP + cmd 3s 1.637 2.437  1.310 2.025      1.754 2.612   0.010 0.013
  Ego-MLP-NR 3s    1.740 2.594  1.370 2.126      1.865 2.781   0.010 0.013
  GN1 strata, M1-v2a-NR - M1-v2a: excl. WSS +0.063 [+0.007,+0.117]; WSS only +2.050
  [+0.523,+3.750]. Most of the route cost is at standstill: without the command, false-go
  jumps 0.061 -> 0.243 (WSS 0.012 -> 0.362). B1-NR - B1: excl. WSS +0.029 [-0.027,+0.087],
  WSS -0.262 [-0.512,-0.053]. GN2 excl. WSS -0.146 [-0.204,-0.089]; WSS +0.526.
  Ego-MLP-NR - + cmd (3 seeds) +0.103 [+0.078,+0.131] (excl. WSS +0.111): command helps
  the MLP; per seed 1.751 / 1.739 / 1.730 vs 1.633 / 1.636 / 1.641.
  FG all/WSS/non-WSS | MG | Col @3 s NoAvg/TemAvg, aa | yaw:
  B1-NR 0.318/0.388/0.245 | 0.314 | 1.31/0.54 | 1.31/0.55 (B1 0.302/0.394/0.205 | 0.424)
  M1-v2a-NR 0.243/0.362/0.118 | 0.519 | 1.35/0.56 | 1.37/0.59 (M1-v2a 0.061/0.012/0.112 |
  0.530 | 1.58/0.57 | 1.54/0.58). Ego-MLP-NR FG 0.028-0.030, MG 0.724-0.742.
  M1-v2a-NR words: collapse 0.420 (M1-v2a 0.408; labels 0.023), consistency 0.840; F1
  straight 0.91, stop 0.66, maintain 0.56, steer 0.35-0.40, gentle acc/dec 0.32-0.34,
  sharp 0.14-0.18, strong 0.01. Caveat: tau 0.9 again goes with a high WSS false-go, as
  for M1-v2a s123 (R6); one seed, so GN1 / GN2 are indicative.

C1 (r7_c12.txt; GT-stopped stationary, 676 x 3 seeds). P(false-go) by slot-1 word:
  M1-v2a: stop 1,826 samples 0.022; maintain 160 0.994; gentle_acc 41 1.000; gentle_dec 1.
  False-gos: go-word 0.830, stop 0.166. Old M1 (10 Hz): stop 1,418 0.020; maintain 552
  0.998; gentle_acc 57 1.000; false-gos go-word 0.953, stop 0.045. B1: not applicable.

C2 rule strata (ADOPTED AFTER R4): STOPPED 679, START 231, MOVING 4,209 (WSS all STOPPED).
  L2@3s STOPPED / START / MOVING (FG | MG): CV 0.049 10.729 2.763 (0.000|1.000); KIN 0.159
  10.526 2.046; Ego-MLP+cmd 0.151 9.405 1.450 (0.033|0.771); no-cam VLA 0.117 9.863 1.816
  (0.011|0.817); B1 1.134 8.915 1.871 (0.299|0.519); B2 1.263 8.846 1.901; M1 1.920 8.629
  1.866 (0.309|0.506); M1-v2a 0.606 9.273 1.755 (0.112|0.633); no-cam M1-v2a 0.163 9.633
  1.707 (0.027|0.784); B1-NR 1.246 8.955 1.952 (0.321|0.411); M1-v2a-NR 1.383 9.409 1.782
  (0.234|0.645); Ego-MLP-NR 0.147 10.418 1.521 (0.034|0.870). ADE@6s: r7_c12.txt.
  M1-v2a - B1: STOPPED -0.528 [-0.984,-0.072], START +0.358 [+0.221,+0.515], MOVING -0.117
  [-0.170,-0.065]. M1-v2a - no-cam: STOPPED +0.490 [+0.218,+0.798], START -0.590 [-0.822,
  -0.386], MOVING -0.062 [-0.114,-0.008]. The flat GS4 hides a trade: cameras + words win
  on START and MOVING, lose on STOPPED. START is hard for every model (L2 > 8).
C3 blank image (BATCHED; B1 decoder check 200 val: 0.800 identical sequences, L2@3s 1.837
  vs 1.825 per-sample). L2@3s all / STOPPED / START / MOVING | FG | MG:
  B1 real 2.166 / 1.389 / 8.860 / 1.924 | 0.290 | 0.537; shuffled 2.309 | 0.496 | 0.403;
  B1 blank 2.625 / 0.048 / 10.725 / 2.596 | 0.000 | 0.996 (blank = never start).
  M1-v2a real 1.871 / 0.292 / 9.346 / 1.716 | 0.052 | 0.662; shuffled 2.039 | 0.258 |0.623
  M1-v2a blank 2.115 / 0.224 / 9.776 / 1.999 | 0.049 | 0.753.
  Blank - real, M1-v2a: STOPPED -0.068 [-0.227,+0.080], START +0.430 [+0.185,+0.699],
  MOVING +0.284 [+0.202,+0.366]; B1: all +0.459, STOPPED -1.341, START +1.865, MOVING
  +0.672. Cameras carry the motion and start cues; with words, the stop decision no longer
  depends on them (blank does not raise M1-v2a false-go).

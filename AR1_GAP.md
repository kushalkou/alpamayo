# AR1_GAP -- Alpamayo-R1 (arXiv 2511.00088) vs our current system

Written 2026-10-03. Goal: reproduce Alpamayo-R1 (AR1). Existing week-1 / queue v2
results (WEEK1_REPORT.md, REVIEW_RESULTS.md) stay as the foundation section.

Sources for the AR1 column (official pages only; nothing was downloaded except the
paper text, which the fetch tool cached):
  [P]  paper, https://arxiv.org/abs/2511.00088. HTML v1/v2 was read; the HTML view
       truncated before the experiments section, so experiment details are UNVERIFIED.
  [MC] model card, https://huggingface.co/nvidia/Alpamayo-R1-10B
  [GH] inference code, https://github.com/NVlabs/alpamayo
  [RC] training recipes, https://github.com/NVlabs/alpamayo-recipes
  [DS] dataset card, https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
UNVERIFIED = not confirmed in a source above; no guess is made.
[P] and [MC] differ on some inputs (history length, camera count): the paper describes
the research system, the model card the released checkpoint. Both are given.

Status: BUILT = usable for AR1 as is; PARTIAL = a related piece exists but differs
materially; MISSING = nothing comparable.
Our files are relative to Alpamayo/code/; "a > b" = commit that added > last changed.

Format of each item:
  AR1:  what Alpamayo-R1 does (source)
  ours: what we have now (file, commit)
  =>    status

=========================================================================================
1. BACKBONE + VISION ENCODER
=========================================================================================
1.1 VLM backbone
  AR1:  Cosmos-Reason VLM [P]. Released backbone 8.2B params, plus a 2.3B action expert,
        ~10.5B total [MC]. The paper reports scaling from 0.5B to 7B [P]. Whether the
        backbone weights equal the public Cosmos-Reason1-7B: UNVERIFIED.
  ours: Cosmos-Reason1-7B (Qwen2.5-VL architecture), 8.30B params incl. vision, frozen,
        LoRA r16 on q/v/o (10.2M trainable). model.py (0c5b107 > e516c00)
  =>    PARTIAL
1.2 Precision
  AR1:  BF16 [MC].
  ours: fp16 backbone + fp32 adapters; V100 has no bf16 hardware (model.py header).
  =>    PARTIAL
1.3 Vision tokenizer
  AR1:  ViT, 160 tokens per image at 448x280 with 2x2 pooling [P]. Optional triplane
        multi-camera tokenizer (~288 tokens per timestep) and video compression [P].
        Which tokenizer the released checkpoint uses: UNVERIFIED.
  ours: Qwen2.5-VL vision tower, 256 tokens per camera at 448x448, 1,536 for 6 cameras;
        frozen, live-encoded. vision_live.py (0c5b107). No triplane / video tokenizer.
  =>    PARTIAL (ViT), MISSING (triplane / video tokenizer)

=========================================================================================
2. INPUTS
=========================================================================================
2.1 Cameras
  AR1:  research system 6-10 cameras [P]; released model 4 cameras (front-wide,
        front-tele, cross-left, cross-right), 1080x1920 downsampled to 320x576 [MC].
  ours: nuScenes 6 cameras, 1600x900 resized to 448x448. dataset.py, vision_live.py
        (0c5b107). nuScenes has no tele or cross cameras.
  =>    PARTIAL
2.2 Camera history (video)
  AR1:  "2-second history" [P]; released model 0.4 s at 10 Hz = 4 frames per camera [MC].
  ours: one image set at t0 per sample, no video history. w1/finetune_intent.py
        (ecb2799), w1/dump_w1.py (950101a > ecb2799)
  =>    MISSING
2.3 Ego history
  AR1:  16 waypoints at 10 Hz, translation xyz + 3x3 rotation [MC].
  ours: 4 rows x [speed, relative yaw, yaw rate, accel] from 2 Hz poses, current row
        from CAN (last message <= t0), causal. w1/records.py ego_state_w1 (d80f612),
        w1/v0_sources.py (72d60a1)
  =>    PARTIAL
2.4 Route / navigation
  AR1:  text user commands and high-level navigation instructions [P, MC]. Exact format:
        UNVERIFIED.
  ours: 3-class command token (VAD rule, derived from the GT future = privileged).
        w1/extra_tokens.py (d80f612), w1/build_data.py (e089d0f)
  =>    PARTIAL
2.5 Text prompt / chat template
  AR1:  text prompt present (commands) [MC]; exact template: UNVERIFIED.
  ours: no text in the context: visual + ego + extra tokens only. model.py
        _build_context (0c5b107 > e516c00)
  =>    MISSING

=========================================================================================
3. ACTION REPRESENTATION AND DECODER
=========================================================================================
3.1 Control representation
  AR1:  unicycle controls (acceleration a_i, curvature k_i), 64 steps at 10 Hz = 6.4 s
        [P]. User output: 64 waypoints, position + rotation, ego frame [MC].
  ours: unicycle controls (accel, curvature), 12 steps at 2 Hz = 6 s, rollout from
        v0_can in the t0 LIDAR frame. w1/targets.py (43b1bd5)
  =>    PARTIAL
3.2 Training-time action tokens
  AR1:  discrete action tokens, 128 per trajectory (2 x 64) [P].
  ours: discrete tokens, 24 per trajectory, 64 percentile bins per slot + STOP, plain CE.
        w1/w1tok.py (950101a)
  =>    PARTIAL
3.3 Inference decoder
  AR1:  continuous trajectory from a separate flow-matching action expert (transformer,
        conditional flow matching, Gaussian OT path, Euler steps dt = 0.1); "we do not
        use discrete trajectory tokens for inference" [P]. How the expert reads the VLM
        (KV cache vs hidden states): UNVERIFIED.
  ours: autoregressive decode of the same tokens (argmax, or STOP-aware expectation);
        no action expert, no flow matching. inference.py (0c5b107 > e27b311),
        w1/dump_w1.py (950101a > ecb2799)
  =>    MISSING
3.4 Multiple samples
  AR1:  several trajectory samples (minADE_6 is reported) [MC].
  ours: one deterministic sample.
  =>    MISSING

=========================================================================================
4. CHAIN-OF-CAUSATION (CoC) REASONING
=========================================================================================
4.1 Data
  AR1:  700K CoC traces used in training [MC]. Public part: ood_reasoning.parquet in
        PhysicalAI-AV, 1,728 train + 349 val annotations [DS]; the recipes say more
        labels are "being added" [RC].
  ours: none.
  =>    MISSING
4.2 Labelling
  AR1:  auto-labelling with GPT-5 from 2 s history video + ego trajectory + meta-actions;
        two-stage human labelling; 10-20% audited; ~10% human-verified [P]. An open CoC
        auto-labelling pipeline is released [RC].
  ours: none. Closest: rule-based lon labels from the GT speed (records.meta_lon,
        d80f612).
  =>    MISSING
4.3 Format
  AR1:  closed-set driving decision + open-set critical components + natural-language
        trace [P].
  ours: the model emits trajectory tokens only, no text.
  =>    MISSING
4.4 Training format
  AR1:  reasoning then action in one sequence, maximum likelihood [P]. Exact template and
        token budget: UNVERIFIED.
  ours: none.
  =>    MISSING

=========================================================================================
5. META-ACTIONS / DRIVING DECISIONS
=========================================================================================
5.1 Driving decisions (closed set)
  AR1:  longitudinal 7 (set-speed tracking, lead following, speed adaptation, gap
        searching, accelerate for passing, yield, stop for static constraint); lateral 8
        (lane keeping, merge / split, out-of-lane nudge, in-lane nudge, lane change,
        pull-over, turn, abort) [P].
  ours: lon 4 classes (stop / accelerate / decelerate / maintain) from a GT-speed rule;
        lat 3 = command. records.py meta_lon (d80f612). Used as oracle input (A2, G6
        (ii)), as a prediction target (w1/finetune_intent.py, ecb2799) and in the
        fixed-head bottleneck (w1/c7_bottleneck.py, ecb2799).
  =>    PARTIAL
5.2 Meta-actions
  AR1:  12 meta-actions: gentle / strong accelerate and decelerate, maintain, stop,
        reverse, steer left / right, sharp steer left / right, go straight [P]. This list
        is from the HTML summary and was not re-checked line by line: UNVERIFIED in
        detail.
  ours: the 4 lon classes are a coarse subset (no gentle / strong split, no reverse);
        lateral only through the command.
  =>    PARTIAL

=========================================================================================
6. TRAINING STAGES
=========================================================================================
6.1 Stage 1: action-modality injection
  AR1:  VLM trained with CE on action tokens; action expert trained with flow-matching
        loss [P]. Full fine-tuning vs adapters: UNVERIFIED.
  ours: SFT of LoRA + ego MLP + trajectory embedding / head on action tokens (CE,
        AR-ADE selection, fix recipe). w1/finetune_w1.py (d80f612 > cff84e9),
        w1/fixrun.py (982d1ec). No action expert.
  =>    PARTIAL
6.2 Stage 2: SFT on CoC (reasoning + action)
  AR1:  [P]; SFT recipe released [RC].
  ours: none.
  =>    MISSING
6.3 Stage 3: RL post-training
  AR1:  GRPO [P]; "open-loop RL with Cosmos-RL and GRPO" [RC]. The released checkpoint
        is not RL-trained [GH].
  ours: none.
  =>    MISSING
6.4 Rewards
  AR1:  reasoning graded by large reasoning models as critics; reasoning-action
        consistency enforced; trajectory quality [P]. Exact formulas: UNVERIFIED (HTML
        truncated in section 5.3.2).
  ours: none.
  =>    MISSING

=========================================================================================
7. EVALUATION
=========================================================================================
7.1 Open loop
  AR1:  minADE_6 at 6.4 s = 1.22 m on 937 challenging samples [MC].
  ours: official nuScenes val (5,119): ADE / FDE@6s, L2 at 1/2/3 s in both conventions,
        mean / median / p95, strata, paired scene bootstrap; one sample, so no minADE_k.
        w1/evalw1.py (950101a), w1/q2_traj.py (783f0f0)
  =>    PARTIAL
7.2 Safety / collisions
  AR1:  closed-loop "close encounter rate" in AlpaSim [P].
  ours: open-loop box-collision rate (VAD port). w1/plan_metrics.py (e089d0f)
  =>    PARTIAL
7.3 Closed loop
  AR1:  AlpaSim score 0.73 +- 0.01 on 910 scenarios [MC]; NuRec scenes for RL / sim [RC].
  ours: none.
  =>    MISSING
7.4 Reasoning quality and reasoning-action consistency
  AR1:  curated 2K CoC evaluation set [P].
  ours: none (no reasoning output).
  =>    MISSING
7.5 Public benchmarks
  AR1:  whether AR1 reports nuScenes / NAVSIM numbers: UNVERIFIED (experiments section
        not readable).
  ours: nuScenes. A like-for-like comparison waits for their numbers.
  =>    UNVERIFIED
7.6 Latency
  AR1:  99 ms on-vehicle [P].
  ours: not measured (training step 0.4 s without cameras, 5.7 s with, on V100).
  =>    MISSING

=========================================================================================
8. COMPUTE AND DATA SCALE
=========================================================================================
8.1 Training data
  AR1:  80,000 h of video, >1B images, 10 Hz trajectories [MC].
  ours: nuScenes trainval, 850 scenes (~5.5 h): 18,313 train + 1,717 holdout + 5,119 val
        keyframes at 2 Hz. w1/build_data.py (e089d0f)
  =>    PARTIAL (about 4 orders of magnitude smaller)
8.2 Public data
  AR1:  PhysicalAI-AV: 1,700 h, 306,152 clips of 20 s, 7 cameras 1080p 30 fps, 133 TB
        [DS].
  ours: not downloaded.
  =>    MISSING
8.3 Compute
  AR1:  training GPUs / hours: UNVERIFIED. Inference needs >= 24 GB VRAM; tested on H100
        [MC, GH].
  ours: 8 x V100 32 GB; a typical run takes 10-100 GPU-h (queue v2 ~330 GPU-h in total).
  =>    see R1 (memory estimate)

=========================================================================================
SUMMARY
=========================================================================================
BUILT:   nothing is AR1-equivalent as is. The reusable foundation is the data / eval
         stack: causal ego, official split, VAD-ported metrics, bootstrap, leak tests.
PARTIAL: backbone family, vision tower, cameras, ego history, route, unicycle control
         representation, discrete action-token SFT (stage 1 without the expert), coarse
         meta-actions, open-loop metrics, data scale.
MISSING: camera / video history, text prompt, flow-matching action expert, multiple
         samples, CoC data + labelling + reasoning SFT, RL (GRPO, LRM critic,
         consistency reward), closed-loop sim, reasoning metrics, bf16 hardware,
         PhysicalAI-AV data.

=========================================================================================
R1. PUBLIC AVAILABILITY (official NVIDIA / Hugging Face / GitHub pages; checked
    2026-10-03; NOTHING downloaded)
=========================================================================================
Weights   https://huggingface.co/nvidia/Alpamayo-R1-10B (collection: huggingface.co/
          collections/nvidia/alpamayo-r1). GATED (request access + `hf auth login`).
          Licence OpenMDW-1.1, "ready for non-commercial use; commercial licensing
          available upon request". 5 safetensors shards, 22.16 GB total (22.2 GB repo).
          BF16. 8.2B backbone + 2.3B action expert. Not RL-trained [GH]. Also called
          "Alpamayo 1"; the recipes repo lists Alpamayo 1.5 (10B) and Alpamayo 2 Super
          (34B) as well [RC].
Code      https://github.com/NVlabs/alpamayo -- inference only (test_inference.py,
          inference.ipynb), Apache 2.0, Python 3.12, uv, >= 24 GB VRAM, tested on H100.
          https://github.com/NVlabs/alpamayo-recipes -- SFT, open-loop RL (Cosmos-RL +
          GRPO), CoC auto-labelling pipeline, FP8 / NVFP4 quantisation (for 1.5),
          dataset download / curation scripts. Apache 2.0. GPU requirements for
          training: not stated on the front page (per-recipe READMEs not read).
CoC data  The 700K training traces are NOT released [MC: no statement of availability].
          Public: ood_reasoning.parquet inside PhysicalAI-AV, 1,728 train annotations
          (1,450 clips) + 349 val (290 clips) [DS]; more "being added" [RC]; plus the
          open auto-labelling pipeline (labels must be generated by the user; it calls an
          external LLM -- which model the open pipeline uses: UNVERIFIED).
Dataset   https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles -- GATED,
          NVIDIA Autonomous Vehicle Dataset License Agreement. 133 TB; 1,700 h; 306,152
          clips x 20 s; 7 cameras 1080p 30 fps; LiDAR 298,326 clips; radar 160,761
          clips; train / val released, test held back; delivered in chunks of ~100
          clips.
          https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec --
          GATED, NVIDIA AV NuRec licence; ~2.87 TB; 1,607 reconstructed 20 s scenes
          (closed-loop / RL).
Our disk: /home has 1.1 TB free (5.0 TB, 79% used): the weights and NuRec subsets fit;
          the full 133 TB dataset does not, only a subset of chunks would.

MEMORY ON OUR V100s (32 GB, fp16 only; estimates, not measured)
  Weights: 22.16 GB on disk in BF16 = ~11.1B values x 2 bytes; fp16 weights are the same
           22.2 GB per GPU.
  Inference (1 GPU): weights 22.2 GB + vision activations (4 cams x 4 frames, ~2.5K
           visual tokens; ~1-2 GB) + KV cache (Qwen2.5-7B-type GQA: 28 layers x 2 x 512
           x 2 B = ~57 KB/token -> ~0.2 GB for 3-4K tokens) + flow-matching steps
           (small) = ~24-26 GB -> FITS on one 32 GB V100, with little headroom.
  Risks:   (1) BF16 -> FP16 cast: values above 65,504 overflow; Qwen2.5-class models
           have large activation outliers, so inf/NaN in fp16 is possible and must be
           tested (our Cosmos-Reason1-7B runs in fp16 without issue, which is
           encouraging but not proof). (2) V100 has no native bf16 and no
           FlashAttention-2; the reference code (Python 3.12, uv) may assume both; our
           env is Python 3.10, torch 2.7.1+cu118, transformers 5.5.4. (3) The 24 GB
           minimum on the card refers to BF16 on newer GPUs; V100 kernels are slower.
  Fine-tuning (estimates):
           - LoRA on the backbone, frozen fp16: like our current setup (8.3B model,
             21.4 GB peak at batch 3 with cameras) plus the 2.3B expert (4.6 GB fp16) ->
             ~26-30 GB per GPU: borderline, needs batch 1-2 and gradient checkpointing.
           - Full training of the 2.3B action expert with AdamW (fp32 master + 2 moments
             + grads = 16 B/param = ~37 GB): does NOT fit one V100; needs ZeRO / FSDP
             sharding across the 8 GPUs (~4.6 GB/GPU) or LoRA on the expert.
           - Full fine-tuning of the 8.2B backbone (~131 GB optimizer state): only with
             full sharding across 8 GPUs (~16 GB/GPU states + activations): not
             realistic on 8 x V100 alongside the replicated activations.
           - GRPO RL needs several rollouts per prompt plus a reasoning critic (an LRM);
             UNVERIFIED cost, likely beyond 8 x V100 at full scale.

NOTE 2026-10-03 (phase R1): re-reading the paper resolved two UNVERIFIED items:
  3.3  the action expert takes the VLM KV-cache (stop-gradient applied to it) plus the
       embedded noisy control; same heads / head dim as the VLM, smaller hidden and MLP
       width; Euler dt = 0.1 at inference.
  5.2  Table 5: lon gentle / strong accelerate, gentle / strong decelerate, maintain
       speed, stop, reverse; lat steer left / right, sharp steer left / right, reverse
       left / right, go straight. Labelled automatically at 10 Hz. No thresholds given.

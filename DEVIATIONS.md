# DEVIATIONS -- small-scale Alpamayo-R1 reproduction on nuScenes vs AR1

Reproduction phase R1, written 2026-10-03. One row per AR1 component (numbering follows
AR1_GAP.md). "ours" = the reproduction as specified for phase R1 (R1.1-R1.4), not the
week-1 system. AR1 facts are from the paper (arXiv 2511.00088 v2 HTML) and the model card;
UNVERIFIED = not confirmed there. Facts newly confirmed in the paper on 2026-10-03 (not
in AR1_GAP.md): the action expert reads the VLM KV-cache with a stop-gradient; Euler
dt = 0.1 at inference; meta-actions are labelled at 10 Hz; Table 5 lateral also lists
reverse left / reverse right; no numeric meta-action thresholds are given.

Format of each row:
  AR1:    what Alpamayo-R1 does
  ours:   what the reproduction does
  reason: why it differs

-----------------------------------------------------------------------------------------
1.1 Backbone
  AR1:    Cosmos-Reason VLM, 8.2B backbone; full vs adapter fine-tuning UNVERIFIED.
  ours:   Cosmos-Reason1-7B (Qwen2.5-VL), frozen, LoRA r16 on q/v/o.
  reason: Full fine-tuning of 8B parameters does not fit 8 x V100 32 GB (AR1_GAP R1).
1.2 Precision
  AR1:    BF16.
  ours:   fp16 backbone, fp32 adapters, fp32 loss, GradScaler.
  reason: V100 has no bf16 hardware.
1.3 Vision tokenizer
  AR1:    ViT, 448x280 -> 160 tokens per image (2x2 merge); optional triplane / video
          tokenizers.
  ours:   the same Qwen2.5-VL ViT at 448x280 -> 160 tokens per image, frozen,
          live-encoded; no triplane / video tokenizer.
  reason: The ViT path matches. Triplane needs camera-rig training data at scale.
2.1 Cameras
  AR1:    research 6-10 cameras; released 4 (front-wide, front-tele, cross-left,
          cross-right).
  ours:   3: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT.
  reason: nuScenes has no tele or cross cameras; the three front cameras are the
          closest forward cover, and 3 keeps the token count trainable on V100.
2.2 Camera history
  AR1:    2 s in the paper; released model 4 frames per camera at 10 Hz (0.4 s).
  ours:   4 frames per camera at t0, t0-0.5, t0-1.0, t0-1.5 s (1.5 s span, 2 Hz).
  reason: Only keyframe images (2 Hz) are on disk; camera sweeps are not downloaded.
          Frames are never later than t0 (unit-tested).
2.3 Ego history
  AR1:    16 waypoints at 10 Hz (xyz + rotation).
  ours:   4 rows [speed, rel. yaw, yaw rate, accel]; current row from CAN (<= t0),
          older rows from 2 Hz poses.
  reason: Kept from the leak-free week-1 stack; a 10 Hz history from CAN pose is
          possible and is a candidate change.
2.4 Route / navigation
  AR1:    text navigation instructions; format UNVERIFIED.
  ours:   3-class command token (VAD rule, from the GT future = privileged).
  reason: nuScenes has no navigation instructions. CAN route.json (planned route
          polyline) could give a non-privileged command; not built yet.
2.5 Text prompt
  AR1:    text prompt with commands; template UNVERIFIED.
  ours:   none (embeddings only).
  reason: Not built yet; needed for CoC SFT (R1.4).
3.1 Control representation
  AR1:    unicycle (accel, curvature), 64 steps at 10 Hz = 6.4 s.
  ours:   unicycle (accel, curvature), 12 steps at 2 Hz = 6 s.
  reason: nuScenes GT is 2 Hz keyframes over 6 s; 10 Hz would need CAN-pose targets.
3.2 Training action tokens
  AR1:    discrete tokens, 128 per trajectory.
  ours:   discrete tokens, 24 per trajectory (64 bins + STOP).
  reason: Follows from 3.1.
3.3 Inference decoder
  AR1:    flow-matching action expert (smaller-width transformer, same heads / head dim
          as the VLM), reads the VLM KV-cache with stop-gradient, Gaussian conditional
          OT path, Euler dt = 0.1.
  ours:   planned (R1.3): small transformer expert, stop-grad VLM conditioning, CFM with
          the OT path, 10 Euler steps, fp32.
  reason: Not built. Expert width / depth chosen to fit V100 (see R1.3).
3.4 Multiple samples
  AR1:    several samples (minADE_6).
  ours:   planned via the flow expert (different noise seeds).
  reason: Token decode is deterministic; sampling comes with the expert.
4.1 CoC data
  AR1:    700K traces (private); 1,728 + 349 public annotations on PhysicalAI-AV.
  ours:   template-generated traces on nuScenes (R1.4), about 25K keyframes.
  reason: No CoC data exists for nuScenes; no LLM labelling budget on this machine.
4.2 CoC labelling
  AR1:    GPT-5 auto-labelling + human labelling and audit.
  ours:   deterministic templates from annotations, CAN and the future trajectory.
  reason: Reproducible, no external LLM; free-text richness is lost.
4.3 CoC format
  AR1:    driving decision (Table 1) + critical components (Table 2) + composed trace.
  ours:   same three parts, filled from what nuScenes provides.
  reason: Traffic-light state, lane lines, signs and routing intent are not annotated
          in nuScenes (see R1.4 limits).
4.4 Training format
  AR1:    reasoning then action in one sequence, maximum likelihood.
  ours:   planned the same (text tokens, then action tokens).
  reason: --
5.1 Driving decisions
  AR1:    7 lon + 8 lat closed-set decisions, labelled by LLM / humans.
  ours:   rule-derived subset (R1.4 decisions).
  reason: Several decisions need lane / signal context nuScenes lacks.
5.2 Meta-actions
  AR1:    Table 5: lon gentle / strong accel, gentle / strong decel, maintain, stop,
          reverse; lat steer L / R, sharp steer L / R, reverse L / R, straight;
          10 Hz; thresholds not given.
  ours:   same classes at 10 Hz from CAN pose (50 Hz), thresholds chosen by us (R1.2).
  reason: Thresholds are ours because AR1 does not specify them.
6.1 Stage 1 (action injection)
  AR1:    VLM CE on action tokens + expert flow-matching loss.
  ours:   VLM CE on action tokens (built); expert planned.
  reason: --
6.2 Stage 2 (CoC SFT)
  AR1:    SFT on CoC + action.
  ours:   planned, on template traces.
  reason: --
6.3 Stage 3 (RL)
  AR1:    GRPO with an LRM critic and reasoning-action consistency rewards.
  ours:   not in phase R1.
  reason: Rollouts plus an LRM critic exceed 8 x V100; formulas UNVERIFIED.
7.1 Open-loop eval
  AR1:    minADE_6 at 6.4 s on PhysicalAI-AV.
  ours:   nuScenes official val: ADE / FDE / L2 (VAD conventions), median + p95, both
          strata; minADE_6 once the expert samples.
  reason: Different dataset; horizon 6 s.
7.2-7.3 Closed loop / safety
  AR1:    AlpaSim score and close-encounter rate.
  ours:   open-loop collision rate only.
  reason: No simulator for nuScenes on this machine.
7.4 Reasoning eval
  AR1:    curated 2K CoC set.
  ours:   decision accuracy against template labels; reasoning-action consistency
          check by rule.
  reason: No human-labelled reasoning on nuScenes.
8.1 Data scale
  AR1:    80,000 h.
  ours:   nuScenes trainval, ~5.5 h (18,313 train keyframes).
  reason: The public PhysicalAI-AV is 133 TB; 1.1 TB free disk.
8.3 Compute
  AR1:    UNVERIFIED (H100-class).
  ours:   8 x V100 32 GB.
  reason: Available hardware.

-----------------------------------------------------------------------------------------
CHANGE LOG
2026-10-05 (R5 B1) Meta-action labels: new label set from the 2 Hz GT target controls
  (ar1/relabel_2hz.py -> data/ar1_meta2hz.pkl). Same AR1 Table 5 classes and the same
  numeric thresholds; the 10 Hz rule's trailing 0.5 s window equals one 2 Hz step. Reason:
  the 10 Hz CAN labels agree with the trajectory the model is trained on in only 69% of
  lon slots (R4 G4), so words and trajectory were trained against different truths.
  The old CAN labels stay for M1 (R3); M1-v2a (R5 B2) uses the 2 Hz labels.
  Advisor spec (Jan 2026) listed CARLA closed-loop and RL; cut for compute, nuScenes open-loop only.

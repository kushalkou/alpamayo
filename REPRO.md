# REPRO -- commits, commands, GPU-hours (frozen at tag v1.0-review)

All commands run from Alpamayo/code with the alpamayo conda env
(PY=/home/dgx1user/miniconda3/envs/alpamayo/bin/python; TR="$PY -m torch.distributed.run
--nproc_per_node=8"). GPU-h = wall time x 8 V100 from the status logs / ledgers
(Alpamayo/r*_status.log, r*_gpu_ledger.txt) unless marked "report" (copied from the
report). Queues are resumable (results/R*_<tag>_*_DONE markers). Evaluation is CPU only.

=========================================================================================
1. RUNS
=========================================================================================
Common recipe flags: REC="--fix --batch_size 3 --grad_accum_steps 1 --patience 5"
Trajectory-only VLA (B1 family, w1/finetune_w1.py + w1/dump_w1.py):
  train  $TR w1/finetune_w1.py --tag <T> [--cmd] --turn_weighted --seed <S> \
           --vcache t0 $REC --epochs 10
  dumps  $TR w1/dump_w1.py --tag <T> [--cmd] --vcache t0 --fix --splits holdout (then val)
Word arms (ar1/finetune_meta.py, train then --dump holdout,val with the same flags):
  $TR ar1/finetune_meta.py --tag <T> --seed <S> [--labels 2hz] [--no_cmd] [--no_vision]
CoC arms (ar1/finetune_coc.py, v3 traces):
  $TR ar1/finetune_coc.py --tag <T> --coc ../data/ar1_coc_2hz_v3.pkl --seed <S> \
      [--aux 0.5 --select_mode none]
  then --dump holdout,val --mode gen; --dump val --mode shuf; --dump val --mode gt
  (AUX: --dump holdout,val --mode none and --mode gen)
Flow expert: $TR ar1/train_expert_meta.py --tag <T> [--labels 2hz]; A3 ar1/train_expert.py

  run (tag)                   command variant                    GPU-h       commit
  -- R2 / R3 (PLANNER_HANDOFF.md section 5, report values) --
  vision cache                ar1/build_vcache.py                1.1         b8bc570
  A3 expert (no-cam)          ar1/train_expert.py                0.5         f8fd234
  B1 s42                      finetune_w1 --cmd                  ~27         69a2bf2
  B2 s42 (3 cams x 4 kf)      finetune_w1 --cmd --vcache hist    ~120        69a2bf2
  B1_s123, B1_s2024           finetune_w1 --cmd                  ~27 each    83a24c1
  M1 s42 (10 Hz words)        finetune_meta                      ~31         7b9540e
  M1 expert                   train_expert_meta --tag M1         54          7b9540e
  M1_s123, M1_s2024           finetune_meta                      ~31 each    227f19e
  no-cam VLA A3 / G6C_s123 / G6C_s2024: week-1 queue v2 (WEEK1_REPORT.md; not itemized)
  R2 + R3 total (incl. benchmarks, dumps, B2): roughly 650 GPU-h (report)
  -- R4 --
  R4 diagnosis dumps          ar1/r4_gpu.sh                      1.52        be518d1
  -- R5 / R6 --
  M1v2a s42                   finetune_meta --labels 2hz         ~31         9afd295
  R6 verification (batched)   ar1/meta_decode_batched.py         2.24        7c771fd
  M1v2a_nocam s42             --labels 2hz --no_vision           16.4        934612e
  M1v2a_s123 / _s2024         --labels 2hz                       31.9 / 25.6 934612e
  -- R7 --
  B1_NR s42                   finetune_w1 (no --cmd)             25.8        492dc07
  M1v2a_NR s42                --labels 2hz --no_cmd              30.4        492dc07
  Ego-MLP-NR (3 seeds)        ar1/r7_egomlp_nr.py (CPU)          0           492dc07
  C3 blank-image control      ar1/run_r7c3.sh (batched)          1.40        492dc07
  -- R8 --
  M1v2a_NR_s123 / _s2024      --labels 2hz --no_cmd              31.7 / 31.9 87b1144
  B1_NR_s123 / _s2024         finetune_w1 (no --cmd)             23.6 / 26.6 87b1144
  -- R9 / R10 --
  expert M1v2a s42            train_expert_meta --labels 2hz     67.6        2303e43
  expert M1v2a_s123 / _s2024  train_expert_meta --labels 2hz     67.6 / 60.6 a5168b3
  -- R11 --
  CoC v2 smoke (4 GPUs)       --max_steps 200 --smoke_decode 16  0.43        a31b76a
  CoC v2 s42 (stopped)        train, discarded at epoch 5        14.8        6352f85
  -- R12 / R13 (CoC v3, ledgers) --
  COC3 smoke (4 GPUs)         --max_steps 200 --smoke_decode 16  0.42        f0694e1
  COC3 s42                    train + gen / none / shuf / gt     65.4        10802ac
  COC3_s123                   train + gen / shuf / gt            53.5        82355f4
  COC3_s2024                  train + gen / shuf / gt            62.3        82355f4
  COC3AUX s42 (+ 2-GPU check) --aux 0.5 --select_mode none       57.8 + 0.06 3efc3b3
=========================================================================================
2. TOTALS
=========================================================================================
  R2 + R3 (report)        ~650        R4                  1.5       R5 (M1v2a s42) ~31
  R6                      76.1        R7                 57.6       R8            113.8
  R9                      67.6        R10               128.2       R11            15.2
  R12                     65.8        R13               173.6
  R4-R13 itemized 730.4 GPU-h; with R2 + R3 about 1,380 GPU-h on 8 x V100. Week-1 /
  queue v2 / custom-split phases are not itemized here (WEEK1_REPORT, REVIEW_RESULTS).

=========================================================================================
3. EVALUATION (CPU) -> report
=========================================================================================
  R5   ar1/r5_trackA.py, ar1/r5_a4.py (Li protocol), ar1/r5_a5c1.py (experts, comfort)
  R6   ar1/r6_verify_eval.py, ar1/r6_part2_eval.py, ar1/r6_part3.py, ar1/r6_final.py
  R7   ar1/r7a_eval.py, ar1/r7_c12.py, ar1/r7_c3_eval.py
  R8   ar1/r8_partA.py, ar1/r8b_eval.py      R9  ar1/r9_partA.py, ar1/r9_b2.py
  R10  ar1/r10_b2.py                         R11 ar1/r11_bugs.py, ar1/coc_template_v3.py
  R12  ar1/r12_mech.py, ar1/r11b_eval.py COC3   R13 ar1/r13_partA.py, ar1/r13b_eval.py
  Audit pages: ar1/build_audit_html_v2.py, _v3.py, _v3short.py; ar1/audit_summary.py.
  Figures: figures/fig1-7*.py (fig1-6 plot report numbers; fig7 reads the dumps).
  Data: data/ar1_coc_2hz_v3.pkl (signed-off traces), data/ar1_meta2hz.pkl (2 Hz labels),
  data/ar1_vcache/ (vision cache), results/w1_dump_<tag>_<split>_f0.0.pkl (all dumps),
  results/ar1_expert_<tag>.pkl (expert draws). data/ and results/ are not in git.

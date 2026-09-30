# WEEK 1 REPORT -- "Where does the signal come from? An information-controlled audit
# of a token-based driving VLA."

Synthetic (p, rho) benchmark: DROPPED and archived (commit 2e45c04, Alpamayo/code/sim/
kept, ARCHIVED.md added). No sim jobs were running.

---

## 0. OVERNIGHT2 OUTCOME (items 1 and 6 ran; 6b full-vision also finished)

The ego-state leak is CONFIRMED. compute_ego_state's current row holds the ground-truth
accel of slot 1 (max err 3.4e-7) and the GT curvature of slot 12 (yaw_rate/v0, max err
1.5e-8) on all 3,614 test samples, and the models read it: slot-1 accuracy is 0.774 vs
0.305 at slot 2. A no-learning predictor that just uses the leaked features gains
+0.649 m over CV [+0.610,+0.686] on the full test set. The same predictor with causal
(past-pose) features gains +0.184 [+0.138,+0.228]. Retrained with causal features
(Y1 recipe, seed 42, custom-split test subset with >= 2 past poses, n=3358, CV 3.059),
the ego-only model + CV blend STILL beats CV, but only by 0.014 m (3.045, CI
[-0.024,-0.003], p=0.011, alpha*=0.10), versus 0.164 m for the leaky model on the same
samples: the leak is 91% of the headline gain. The causal full-vision blend gives the
same 0.014 m ([-0.025,-0.002], p=0.018), and full minus ego is -0.000 overall and
-0.014 [-0.050,+0.021] on turning. All three nulls select alpha=0. The slot-1
fingerprint is gone (0.27-0.28). Single seed each; all on the custom split, so NOT
comparable to the literature. Details: OVERNIGHT2_REPORT.md sections 1, 6 and 6b.

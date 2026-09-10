# Follow-up protocol and source review

Read-only review of `docs/followup-plan.md`, the working source diff, and `runs/followup.KGkx3v/development.py` plus completed seed-100 pilot records. No pilot or full training reruns. The reviewed diff, reproduction script, and numeric results are saved in this directory.

## Concrete findings

1. **Default prototype reruns are rejected by the historical analyzer.** `scripts/analyze_study.py:63-80` still requires the exact old configuration dictionary. `RunConfig` now always serializes `activation_gain=1.0` and `final_learning_rate=null`, so importing `PROTOTYPE_MODES` alone does not preserve the documented current launcher-to-analysis workflow. A minimal fixture containing the original completed-study header and seed-0 configuration plus those two defaults raises `AssertionError` at the comparison. Normalize absent/default values for historical runs while still rejecting a nondefault gain or schedule in this prototype-only analysis. Existing old artifacts remain readable.

2. **The planned format-opportunity check needs an explicit temporal criterion.** The evidence table currently requires candidate scores and actual choices on substantial training arrays. A stable wider-range winner only establishes that choosing once can help; cross-array differences and tie-broken label changes do not establish an opportunity for repeated selection. Before freezing, define success as the same at-least-32-entry training array changing its strict preferred candidate over time. Retain both candidate scores and their margin; distinguish a changed label with identical reconstruction from a numerically different choice. Synthetic outlier arrays prove that the family can favor wider range, but do not substitute for this check on training arrays. Sampling only every 1,000 steps cannot exclude shorter changes between checkpoints, so any null statement must be limited to observed checkpoints or use measurements at the eventual selection interval.

3. **No current seed-100 setting passes the predeclared feature check.** All gains completed 20,000 updates and failed each of their final three checks. Final amplitudes are 0.00019445 (gain 1), 0.00060941 (gain 4), and 0.00066942 (gain 8), versus the required interval 0.008-0.012. Isolated-component relative errors are 1.00012, 0.95221, and 0.97832, versus the required value below 0.25. Overall MSE alone passes and would therefore misidentify feature learning. Every recorded array with at least 32 entries prefers E3M4; there are zero observed substantial-array switches. The written protocol consequently has no setting eligible for seed-101 confirmation yet. Record a revised development design before starting additional candidates or changing the target/thresholds.

## Assessment of controls and remaining interpretation limits

- The joint amplitude, isolated-error, total-error, and final-three-measurement criterion is meaningful: it rejects the observed low-total-error models that miss the small component. Selecting on seed 100 and confirming on seed 101 before freezing fresh final seeds is an appropriate separation. Continue to inspect FP32 feature recovery on every final seed; retain failures rather than conditioning the final comparison on passing seeds.
- Adding fixed E3M4 supplies the strongest missing simple baseline. Each final condition must receive the same chosen gain, target, learning-rate schedule, warmup, initialization, and indexed batches. The source's gain and schedule are applied consistently within a configured run. The cosine endpoints reproduce 0.001 and 0.00001 exactly; default gain 1 reproduces historical saved FP32 predictions exactly.
- Multiplying activation inputs also changes forward magnitudes and derivatives, especially for ReLU and SiLU. The gain sweep screens learnability and conditioning; it cannot attribute an outcome solely to saturation. Within-gain format comparisons remain meaningful, and the gain must be frozen across them.
- The pilot measures weight candidates after the optimizer update and gradient candidates from the just-completed pre-update backward pass (`development.py:44-48,58-59`). These are real training arrays, but not a weight/gradient pair recomputed at the same saved checkpoint. Label their timing, or recompute a fixed-probe gradient when comparing contemporaneous checkpoint distributions. This does not invalidate the feature measurements.
- The old benchmark and analysis scripts remain deliberately prototype-specific: six modes, old seeds/settings, default gain, and constant learning rate. Do not use them as a follow-up benchmark without adapting and validating the configuration handling.

## Reproduction

From `/Users/neeraj/src/zomglings/afloat`:

```sh
.venv/bin/python /var/folders/x5/qnn2lxld3j75g3kt9dj_v1v40000gn/T/tmp.RkcKR7S5Jp/verify_followup.py
```

`results.json` preserves all last-three feature rows, per-gain gate outcomes, observed substantial-array choices/switches, the configuration regression, cosine endpoints, and default-gain compatibility check. This audit does not claim gain changes cannot help at other budgets or that substantial arrays never change between the recorded checkpoints.

# Independent full-study audit

Reviewed raw study `runs/full-study.NzhhwC`, the saved source revision `b3573e0167f466c1e0579b0b832ce47f2c65cd1b`, analysis and benchmark scripts, and portable launcher. No source edits or full-study reruns were performed.

## Verified evidence

- All 528 original manifest entries match their file contents.
- All 108 final checkpoints contain the expected represented weights, and recomputing their forward passes produces exactly their saved validation predictions.
- Independently computed analytic targets in FP64 reproduce every final MSE and paired ratio within the expected small FP32-target evaluation difference. The script checks all 30 adaptive/control comparisons rather than using the analysis script's calculations.
- All 108 benchmark entries match the declared randomized execution order, saved step-300 validation losses, raw timing medians, per-step time arithmetic, and summary aggregates.
- Recomputed activation signs and slopes for all 162 final FP32 activation groups match their records.
- The portable launcher preserves the original six targets, six conditions, three seeds, numerical settings, three concurrent seed processes, and failure exit behavior.

## Findings and interpretation limits

1. **The benchmark verification label is stronger than its check.** `scripts/benchmark_study.py:178-183` compares one scalar validation MSE after 300 updates, while line 239 writes `all_training_trajectories_verified_at_step_300`. It verifies matching endpoint loss, not intermediate weights, updates, optimizer state, or predictions. State this narrower fact or rename the field; the scalar matches themselves are valid. Final cached weight representations really are checked exactly at lines 200-202, and this audit independently reproduced all 108 of them.

2. **Adaptation did not vary hidden-layer formats.** Independently deriving transitions from the raw event sequence gives exactly 190 switches: 22 for `weight:4.bias` and 168 for `gradient:4.bias`. Every switched array has one element. All switches occur at permitted reselection steps. E3M4 accounts for 345044100/345060000 weight-element evaluations and 345042100/345060000 gradient-element evaluations; no adaptive element uses E5M2. A uniform fixed-E3M4 control is absent, so superiority over standard formats cannot establish that per-array selection is useful. Calibrated remains a valid control for repeated selection.

3. **No target shows a strict adaptive win over calibrated on all three seeds.** Ratios by seed 0,1,2 are linear [1,1,1]; kink [1.8598,1,1]; plateau [1,1,0.55195]; detail [0.73642,0.71457,1.48519]; bump [1,0.97367,1.30526]; composition [0.35096,1.34817,3.53616]. A favorable ratio of means, especially for composition, hides paired losses on other seeds. These are exploratory descriptive results from three seeds, not evidence of a repeatable adaptation benefit.

4. **The small sine feature is unlearned by FP32.** FP32 detail amplitudes are [0.00001338,0.00023332,0.00028684] against intended amplitude 0.01, or 0.13%-2.87% recovery. The isolated detail-component relative MSE is [1.7903,1.3620,1.7272], worse than predicting no component. Overall detail MSE mainly measures the larger sine function, so these runs cannot establish small-feature preservation. The bump is learned approximately: FP32 predictions at the grid peak are [0.7800,0.5997,0.8898] against truth 0.9845; its regional MSE is 1.95%-12.18% of the zero-prediction regional MSE.

5. **Activation regimes are unevenly exercised.** Every final FP32 ReLU group contains both positive and negative inputs. Mean tanh slope-below-0.01 fractions are 0 for linear/kink/bump, 1.01% for plateau, 1.07% for detail, and 2.22% for composition. These confirm some saturation, not uniformly strong coverage. The saved uniform probes and thresholds support descriptive observations; there is no activation intervention to establish why a format helped or hurt.

6. **The timing results describe this simulator and this block length.** Training includes generation, FP32 arithmetic/Adam, conversion, and per-array error/event bookkeeping; it excludes shared warmup/calibration, diagnostic passes, files, and plots. Each condition has one 300-step training block, while summary medians pool 18 different target/seed conditions. That block contains two reselections; the full 2,000-step study contains 19, so the marginal adaptation overhead cannot be extrapolated exactly. Cached inference operates on the same FP32 network with already prepared FP32 tensors. Approximately 1% differences between modes cannot establish an eight-bit execution speed advantage, and weight preparation is measured separately. The randomized order and matching scopes are appropriate for descriptive simulator comparisons.

## Reproduction

Run from `/Users/neeraj/src/zomglings/afloat`:

```sh
.venv/bin/python /var/folders/x5/qnn2lxld3j75g3kt9dj_v1v40000gn/T/tmp.NBSNZkfWfo/audit.py
.venv/bin/python /var/folders/x5/qnn2lxld3j75g3kt9dj_v1v40000gn/T/tmp.NBSNZkfWfo/verify_models.py
```

`independent-results.json` contains all recalculated paired comparisons, FP32 feature measures, switch counts, and final FP32 activation summaries. Scripts and outputs are grouped in this fresh temporary directory. This audit does not rerun training, independently remeasure wall time, validate hardware acceleration, or establish cross-machine reproducibility.

## Final claim check and dispositions

The final `proto-results.md` claim check is complete. Its accuracy ratios, seed outcomes, rounding/error fractions, gradient-direction summaries, timing medians/ranges, selection costs, scale ranges, record counts, and representation budgets agree with the saved evidence. An additional independent pass through all 5,760,000 raw format records verified the published aggregate rounding measurements, zero clipping, and adaptive scale ranges; results are saved in `report-table-checks.json`.

Four requested wording corrections were applied and reread:

- The report now says all 108 step-300 validation losses matched and explicitly says scalar equality does not prove equality of intermediate states. The future benchmark output field is `all_step_300_validation_losses_matched`. Original benchmark data remain unchanged, and `docs/evidence/full-study/performance/benchmark-source.py` exactly matches the source hash in the measured protocol. The parent will describe the legacy field's overstatement in the published audit.
- The report describes 0.1 seconds as total measurement time across repeated blocks, not as the duration of each raw block.
- The unsupported claim that calibrated behaves almost identically to untested fixed-E3M4 was replaced with the measured fact that it uses E3M4 for nearly every element.
- The full-run timing column is labeled elapsed time at final validation, with explicit exclusions for final metric writes, final activation probes, checkpoint saving, warmup, and initial calibration.

The report also states the 300-update benchmark's two reselections versus the full study's 19, the absence of a fixed-E3M4 control, output-bias-only switching, small-feature learnability failure, sparse saturation, three-seed limitations, and lack of an established adaptive or hardware-speed advantage. No material unresolved reporting or arithmetic defect remains within this bounded audit. No training or timing study was rerun.

Additional reproduction:

```sh
.venv/bin/python /var/folders/x5/qnn2lxld3j75g3kt9dj_v1v40000gn/T/tmp.NBSNZkfWfo/verify_report_tables.py
```

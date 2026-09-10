# afloat follow-up results

**Repeated selection still shows no consistent accuracy advantage over fixed E3M4
or choosing once.** Its mean final validation error is about 20% higher than
choosing once at both activation gains. No substantial training array changes
its preferred format. The small feature is now learnable, but only after the
explicit input-representation change described below.

This follow-up adds fixed E3M4, makes the original small sine component learnable,
compares two activation ranges on five fresh seeds, and measures CPU time to a
shared accuracy target. The [frozen protocol](docs/followup-protocol.json) was
recorded after development and before the 70 final comparisons.

The target remains `sin(2πx₁) + 0.01 sin(8πx₂)`. Development with raw coordinates
failed to learn its small component at every tested gain. The final experiment
therefore supplies both coordinates and their sine/cosine at frequencies
`π × {1, 2, 4, 8}`. These 18 inputs explicitly include the target's oscillations.
This establishes coefficient learning and preservation with those features;
it does not demonstrate discovery of high frequencies from raw coordinates.

The network has 10,097 parameters and three feed-forward blocks, each combining
ReLU, tanh, and SiLU. Gain 1 or 4 multiplies the inputs to these nonlinearities.
Each condition shares initialization, 50 FP32 warmup updates, optimizer state,
and training batches with its paired controls. It then runs 20,000 updates with
batch size 128 and a cosine learning-rate decrease from 0.001 to 0.00001.
Fresh seeds are 10–14; development uses 100 and 101.

E3M4 uses three exponent bits and four fraction bits, plus a sign bit. E4M3 and
E5M2 exchange fraction precision for exponent range. Every reduced-precision
condition recomputes its power-of-two scale from each current array. Thus even
“fixed” formats have changing scales. Calibrated chooses the bit allocation once;
adaptive chooses it again every 100 updates using rounding mean squared error.
Weights and parameter gradients are rounded; arithmetic, activations, master
weights, and Adam state remain FP32. This measures a software simulation, with
no packed eight-bit arithmetic or memory-speed claim.

## Development and failed attempts

The feature criterion requires all of: validation mean squared error below
0.0001, recovered amplitude from 0.008 through 0.012, and isolated small-component
error below 0.25. Isolation subtracts each fixed-x₁ row's mean over x₂ before
comparing prediction and truth on a 65 × 65 grid. The error is divided by the
true component's squared magnitude: 0 is exact and 1 means no recovered component.
Three consecutive measurements must pass. Development accepts the final three;
final comparisons also retain the first sustained crossing.

| Inputs | Seed | Gain | Updates | Final MSE | Amplitude | Component error | Final three pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 100 | 1 | 20,000 | 8.16e-05 | 0.00019 | 1 | no |
| raw | 100 | 4 | 20,000 | 5.15e-05 | 0.00061 | 0.9522 | no |
| raw | 100 | 8 | 20,000 | 5.05e-05 | 0.00067 | 0.9783 | no |
| fourier | 100 | 1 | 20,000 | 1.62e-07 | 0.00998 | 0.00294 | yes |
| fourier | 100 | 4 | 20,000 | 5.02e-07 | 0.01004 | 0.0092 | yes |
| fourier | 101 | 1 | 20,000 | 1.36e-07 | 0.01004 | 0.002471 | yes |
| fourier | 101 | 4 | 20,000 | 2.45e-07 | 0.01005 | 0.004587 | yes |
| fourier | 100 | 1 | 6,000 | 4.35e-06 | 0.01004 | 0.08305 | yes |
| fourier | 100 | 4 | 6,000 | 1.7e-05 | 0.01022 | 0.3246 | no |
| fourier | 101 | 1 | 6,000 | 2.96e-06 | 0.00997 | 0.05621 | yes |
| fourier | 101 | 4 | 6,000 | 7.25e-06 | 0.00998 | 0.1408 | yes |

The shorter 6,000-update schedule failed at gain 4 on seed 100. The predeclared
rule therefore retained 20,000 updates for both gains. Every development attempt,
including failed settings, is included in the evidence.

## Final accuracy

Values are means ± sample standard deviations across five paired seeds.
All 70 runs completed the fixed budget without numerical failure. All sustained
the threshold at some point; 66 still passed all three final measurements.

| Gain | Method | Final validation MSE | Isolated component error | Mean amplitude | Final three pass |
| --- | --- | --- | --- | --- | --- |
| 1 | fp32 | 1.64e-07 ± 4.1e-08 | 0.003 ± 0.00076 | 0.00999 | 5/5 |
| 1 | fixed-e3m4 | 1.81e-06 ± 2.3e-07 | 0.0185 ± 0.0036 | 0.01019 | 5/5 |
| 1 | fixed-e4m3 | 1.59e-05 ± 2.4e-05 | 0.073 ± 0.094 | 0.01024 | 5/5 |
| 1 | fixed-e5m2 | 1.27e-05 ± 3.6e-06 | 0.132 ± 0.043 | 0.00972 | 3/5 |
| 1 | fixed-hybrid | 7.79e-06 ± 1.2e-06 | 0.067 ± 0.043 | 0.00985 | 5/5 |
| 1 | calibrated | 2.07e-06 ± 1.6e-06 | 0.0197 ± 0.015 | 0.00997 | 5/5 |
| 1 | adaptive | 2.49e-06 ± 1.3e-06 | 0.0223 ± 0.02 | 0.01008 | 5/5 |
| 4 | fp32 | 4.43e-07 ± 1.1e-07 | 0.0077 ± 0.0017 | 0.00999 | 5/5 |
| 4 | fixed-e3m4 | 4.37e-06 ± 1.9e-06 | 0.0451 ± 0.015 | 0.01040 | 5/5 |
| 4 | fixed-e4m3 | 7.12e-06 ± 3.3e-06 | 0.0639 ± 0.036 | 0.01036 | 5/5 |
| 4 | fixed-e5m2 | 9.64e-06 ± 4.1e-06 | 0.0953 ± 0.026 | 0.00919 | 4/5 |
| 4 | fixed-hybrid | 7.96e-06 ± 6.8e-06 | 0.0547 ± 0.025 | 0.01011 | 4/5 |
| 4 | calibrated | 3.39e-06 ± 1e-06 | 0.0452 ± 0.015 | 0.01020 | 5/5 |
| 4 | adaptive | 4.1e-06 ± 6.4e-07 | 0.0409 ± 0.014 | 0.01015 | 5/5 |

![Learning curves](docs/evidence/followup/analysis/learning-curves.png)

Adaptive/control ratios below divide the five-seed means; below 1 favors
adaptive. Wins compare matching seeds, separately for total error and component
error. These are descriptive results from five seeds, with every pair retained.

| Gain | Control | Total-error ratio | Total-error wins | Component-error ratio | Component-error wins |
| --- | --- | --- | --- | --- | --- |
| 1 | fp32 | 15.214 | 0/5 | 7.448 | 0/5 |
| 1 | fixed-e3m4 | 1.378 | 1/5 | 1.203 | 3/5 |
| 1 | fixed-e4m3 | 0.156 | 5/5 | 0.306 | 4/5 |
| 1 | fixed-e5m2 | 0.195 | 5/5 | 0.169 | 5/5 |
| 1 | fixed-hybrid | 0.320 | 5/5 | 0.333 | 4/5 |
| 1 | calibrated | 1.202 | 1/5 | 1.131 | 2/5 |
| 4 | fp32 | 9.253 | 0/5 | 5.309 | 0/5 |
| 4 | fixed-e3m4 | 0.938 | 2/5 | 0.907 | 3/5 |
| 4 | fixed-e4m3 | 0.575 | 4/5 | 0.640 | 3/5 |
| 4 | fixed-e5m2 | 0.425 | 5/5 | 0.429 | 5/5 |
| 4 | fixed-hybrid | 0.514 | 4/5 | 0.747 | 5/5 |
| 4 | calibrated | 1.207 | 1/5 | 0.904 | 3/5 |

At gain 1, the comparison with fixed E3M4 and with calibrated each contains one
exact tie on both errors. Every other adaptive/control pair is unequal. Adaptive
has lower mean error than the wider fixed formats, but fixed E3M4 and choosing
once explain much of that difference. Adaptive beats FP32 on neither error for
any seed.

The four failures of the final-three criterion are: fixed-e5m2, gain 1, seed 11; fixed-e5m2, gain 1, seed 14; fixed-e5m2, gain 4, seed 11; fixed-hybrid, gain 4, seed 12.

## Format changes and nonlinear regions

All 59,700 reselection observations of arrays with at least 32 entries strictly
prefer E3M4, each by at least the required 1% score margin. Their 300 initial
calibration observations also prefer E3M4. There are zero strict preference
changes in these arrays. Of 1,056 label switches, 530 change rounded values at
that update: 1,002 switches concern the scalar output-bias gradient and 54 the
scalar output bias. Tied scores and different labels can yield identical values.

Routine rounding statistics cover 385,920 retained records, sampled every 100
updates plus the final update; all actual switches and reselection scores are
retained. They do not describe every update. Full candidate scores, initial
scores, and switch details are included in the published evidence.

The larger gain does change the nonlinear regions visited. The following final
FP32 summaries average three layers across five seeds. Input bounds are averages
of each group’s 1st and 99th percentiles; small slope means absolute derivative
below 0.01. These 15 layer/seed groups are not 15 independent seeds.

| Gain | Function | Input p01 | Input p99 | Mean absolute slope | Small-slope share |
| --- | --- | --- | --- | --- | --- |
| 1 | relu | -0.577 | 0.531 | 0.495 | 50.52% |
| 1 | tanh | -0.561 | 0.572 | 0.945 | 0.00% |
| 1 | silu | -0.559 | 0.596 | 0.496 | 0.00% |
| 4 | relu | -2.654 | 2.114 | 0.443 | 55.74% |
| 4 | tanh | -2.412 | 2.390 | 0.601 | 1.19% |
| 4 | silu | -2.865 | 2.315 | 0.447 | 1.92% |

Thus range variation is observed, but it does not create a changing preference
for another weight/gradient format. Activation and gradient-error measurements
for every method remain available in the raw tables.

A separate five-stage synthetic array, with 1,536 entries, changes its preferred
format E3M4 → E4M3 → E4M3 → E5M2 → E3M4 as small values move farther from one
large outlier and then return. Relative advantages over the next candidate are
75.0%, 74.7%, 53.8%, 99.8%, and 75.0%. At the `1e-4` stage, E3M4 rounds 1,535
small entries to zero; E4M3 preserves them and reduces mean squared error from
3.70e-9 to 4.88e-12. At the `1e-6` stage, E5M2 preserves entries that E3M4 and
E4M3 zero, reducing error from 3.70e-13 to 7.53e-16. This demonstrates the
representation mechanism on a constructed array, not a training benefit.

## CPU training and inference time

The accuracy study used three single-thread processes. Its timers include
concurrent execution and are not used for the serial comparisons below. After
accuracy and analysis finished, a separate process replayed seeds 10–12 for
FP32, fixed E3M4, calibrated, and adaptive at both gains. It retained the original
20,000-update learning-rate schedule even when stopping early.

Every 500 updates, the replay required exact equality with the saved model,
optimizer state, format choices, and validation measurements. Timing includes
warmup, applicable initial format selection, training, and threshold validation.
It excludes checkpoint reads/comparisons, logging, extra gradient/activation
probes, and source capture. The training block still pays for the simulator's
rounding statistics. Reported cost is the sum of these measured work intervals,
not full process wall time. The start of the first three-pass sequence and its
confirmation are reported separately; a threshold miss retains its full-budget
cost.

All 24 replays reached the threshold, with zero numerical failures and 398
exact state/metric checks. The host was an Apple M5 Pro running macOS 26.5.1,
Python 3.13.14, and PyTorch 2.14.0. Accuracy training and model analysis finished
before timing; light report editing and file work continued. No GPU was used.

Entries below are medians over three seeds. The range is the minimum and maximum
confirmation time. The sustained start is identified retrospectively as the
first measurement of the first three-pass sequence; confirmation follows 1,000
updates later. An earlier single passing measurement does not count: 14 of the
24 replays had one before their eventual sustained sequence.

| Gain | Method | Sustained start / confirmed updates | Sustained start / confirmed seconds | Confirmation range, s | Training ms/update |
| --- | --- | --- | --- | --- | --- |
| 1 | fp32 | 2,000 / 3,000 | 1.29 / 1.86 | 1.81–2.10 | 0.595 |
| 1 | fixed-e3m4 | 7,000 / 8,000 | 22.81 / 25.42 | 16.50–27.02 | 2.992 |
| 1 | calibrated | 7,500 / 8,500 | 20.05 / 22.64 | 17.35–30.51 | 2.662 |
| 1 | adaptive | 6,000 / 7,000 | 16.53 / 19.28 | 14.25–23.18 | 2.746 |
| 4 | fp32 | 5,500 / 6,500 | 3.26 / 3.84 | 3.72–4.12 | 0.585 |
| 4 | fixed-e3m4 | 8,500 / 9,500 | 23.83 / 26.45 | 22.87–29.86 | 2.708 |
| 4 | calibrated | 8,500 / 9,500 | 22.48 / 25.16 | 24.28–27.96 | 2.657 |
| 4 | adaptive | 9,000 / 10,000 | 26.12 / 28.88 | 23.82–41.87 | 2.969 |

Adaptive’s median confirmation cost is 24.2% lower than fixed E3M4 and 14.9% lower
than calibrated at gain 1. It is 9.2% and 14.8% higher, respectively, at gain 4.
These are ratios of medians for the declared timing subset, not estimated general
speedups. Adaptive is faster than fixed E3M4 on all three gain-1 seeds and none of
the gain-4 seeds; against calibrated it is faster on two gain-1 seeds and one
gain-4 seed.
FP32 reaches the target much sooner in measured work. Timing includes one replay
per seed/method; variation combines different learning paths and host timing
noise, rather than estimating repeated-run timing uncertainty.

| Gain | Method | Single input, µs | Batch 128, µs | Weight preparation, µs |
| --- | --- | --- | --- | --- |
| 1 | fp32 | 89.52 | 138.35 | 13.60 |
| 1 | fixed-e3m4 | 89.41 | 139.62 | 616.69 |
| 1 | fixed-e4m3 | 89.13 | 138.17 | 596.95 |
| 1 | fixed-e5m2 | 89.08 | 138.32 | 596.74 |
| 1 | fixed-hybrid | 88.71 | 139.59 | 595.82 |
| 1 | calibrated | 89.03 | 138.71 | 619.02 |
| 1 | adaptive | 88.71 | 139.54 | 618.93 |
| 4 | fp32 | 89.63 | 138.79 | 13.60 |
| 4 | fixed-e3m4 | 88.89 | 139.21 | 617.38 |
| 4 | fixed-e4m3 | 88.61 | 139.23 | 599.47 |
| 4 | fixed-e5m2 | 89.60 | 139.22 | 599.74 |
| 4 | fixed-hybrid | 89.82 | 138.55 | 598.53 |
| 4 | calibrated | 88.20 | 139.02 | 621.17 |
| 4 | adaptive | 87.93 | 138.22 | 618.73 |

Inference uses all 70 final models with prepared rounded weights, batches of
1 and 128, one CPU thread, and FP32 arithmetic. Each model's median comes from
repeated timing blocks totaling at least 0.1 seconds. The table summarizes the
five model medians. Weight preparation is timed separately. Similar cached
inference costs would follow from their common architecture and arithmetic,
not evidence of a native eight-bit speed advantage.

## Verification and reproduction

The accuracy study verified 3,370 manifest entries, 70 reconstructed final models,
2,870 validation/gradient measurement rows, 25,830 activation records, and all
385,920 retained format records. It recomputed the initial candidate scores and
all final predictions and feature errors exactly. The separate timing pass
verified model weights, complete optimizer state, choices, and metrics at all
398 visited checkpoints; every cached final representation also matched.
The accuracy study took 25.2 minutes of wall time with three processes; the serial
performance pass took 8.2 minutes including verification and inference.

The measured training source is clean commit
`4d9959e313b1915558f13574d9aee18fd07f256e`. Configurations, source hashes, exact
analysis/benchmark scripts, raw measurements, and original manifests are in
[the evidence](docs/evidence/followup). The [audit](docs/evidence/followup/audit.md)
records prescribed checks, independent calculations, review corrections, and
remaining limits. All required checks passed; no check was omitted.
[Development reproduction](docs/followup-reproduction.md) includes every failed
setting and explains the separately preserved historical source snapshots.

```sh
uv sync --locked
mkdir -p runs
afloat_followup=$(mktemp -d runs/followup.XXXXXX)
uv run python scripts/run_followup.py --output "$afloat_followup/accuracy"
uv run python scripts/analyze_followup.py "$afloat_followup/accuracy" \
  --output "$afloat_followup/analysis"
uv run python scripts/benchmark_followup.py "$afloat_followup/accuracy" \
  --output "$afloat_followup/performance"
uv run python scripts/summarize_followup_performance.py "$afloat_followup/performance" \
  --output "$afloat_followup/performance-analysis"
make check
make smoke
```

Run these commands sequentially from a clean checkout so study work does not
compete with timing. The analysis requires a complete successful study and fails
visibly on missing or failed conditions; the runner retains failure records.
Bulk checkpoints remain under ignored `runs/` and can be regenerated. Published
manifests describe the original complete directories. A separate public evidence
manifest covers only the compact published files.

## What this establishes

**Ready as a follow-up measurement report; an adaptive-training advantage remains
unproved.** Fixed E3M4 is now directly tested, the small feature is learnable under
an explicit representation change, and both nonlinear range variation and timing
are measured. The missing condition is still a substantial training array whose
preferred format changes over time. Without it, this is primarily a comparison
of mostly E3M4 training with scalar format changes. The synthetic result cannot
fill that gap. These results do not settle whether other learned number formats,
models, or distributions benefit from adaptation.

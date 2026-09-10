# Independent completed follow-up accuracy summary

All 10 groups and 70 conditions completed with exit code 0. The recorded wall time was 1,513.7904259171337 seconds. Calculations below read the raw completed condition records, independently of the main analysis script. No bulk hashing, model replay, or timing was performed for this summary.

## Exact totals

- 70 final condition records; 1,400,000 post-warmup training updates, plus 500 shared warmup updates across 10 gain/seed checkpoints.
- 2,870 validation metric rows and measurement checkpoints; 70 separate final checkpoints.
- 25,830 activation rows; 140 started/completed progress records.
- 385,920 retained format records; 63,680 reselection candidate records.
- 1,056 label switches: 1,002 output-bias gradient switches and 54 output-bias weight switches. Exactly 530 switches change rounded values. No substantial array switches.
- All 59,700 observed substantial-array candidate winners are E3M4. No periodic strict preference changes were observed.
- The original manifests list 3,370 entries. This collection counted entries; it did not rehash them while avoiding extra work before serial timing.

## Final outcomes

Mean ± sample standard deviation across the five paired seeds (10–14). Component error is isolated small-component MSE divided by the true component squared magnitude. Threshold counts are ever sustained / final three measurements.

| Gain | Mode | Validation MSE | Component error ratio | Threshold counts |
| --- | --- | --- | --- | --- |
| 1 | fp32 | 1.6360482e-07 ± 4.0511795e-08 | 0.002995042 ± 0.00075951553 | 5/5; 5/5 |
| 1 | fixed-e3m4 | 1.8064556e-06 ± 2.3149575e-07 | 0.018546274 ± 0.0036423428 | 5/5; 5/5 |
| 1 | fixed-e4m3 | 1.591912e-05 ± 2.3671056e-05 | 0.073010564 ± 0.094462257 | 5/5; 5/5 |
| 1 | fixed-e5m2 | 1.2743676e-05 ± 3.5903855e-06 | 0.13167554 ± 0.043463789 | 5/5; 3/5 |
| 1 | fixed-hybrid | 7.7865569e-06 ± 1.2382277e-06 | 0.067027705 ± 0.043340196 | 5/5; 5/5 |
| 1 | calibrated | 2.0712193e-06 ± 1.5960328e-06 | 0.019716048 ± 0.014807142 | 5/5; 5/5 |
| 1 | adaptive | 2.4890131e-06 ± 1.343628e-06 | 0.022307018 ± 0.019972127 | 5/5; 5/5 |
| 4 | fp32 | 4.4257292e-07 ± 1.0974464e-07 | 0.0077012316 ± 0.0017445038 | 5/5; 5/5 |
| 4 | fixed-e3m4 | 4.3667751e-06 ± 1.8612554e-06 | 0.04509213 ± 0.014801454 | 5/5; 5/5 |
| 4 | fixed-e4m3 | 7.1209917e-06 ± 3.347322e-06 | 0.063898428 ± 0.036325865 | 5/5; 5/5 |
| 4 | fixed-e5m2 | 9.643366e-06 ± 4.0767855e-06 | 0.095305665 ± 0.026000487 | 5/5; 4/5 |
| 4 | fixed-hybrid | 7.9618687e-06 ± 6.7710647e-06 | 0.054740495 ± 0.025359795 | 5/5; 4/5 |
| 4 | calibrated | 3.3917291e-06 ± 1.0077319e-06 | 0.045240677 ± 0.015312654 | 5/5; 5/5 |
| 4 | adaptive | 4.0952308e-06 ± 6.3621221e-07 | 0.040886208 ± 0.013678205 | 5/5; 5/5 |

Exact unrounded values are in `final.json` and `final-summary.csv`.

## Paired adaptive/control comparisons

Each mean ratio is mean adaptive error divided by mean control error. W/T/L counts compare matching seeds; these ratios are not averages of individual seed ratios. `paired.csv` and `final.json` retain every seed ratio and absolute difference.

| Gain | Control | Total MSE mean ratio; W/T/L | Component mean ratio; W/T/L |
| --- | --- | --- | --- |
| 1 | fp32 | 15.2135681; 0/0/5 | 7.44798155; 0/0/5 |
| 1 | fixed-e3m4 | 1.3778435; 1/1/3 | 1.20277621; 3/1/1 |
| 1 | fixed-e4m3 | 0.156353693; 5/0/0 | 0.30553137; 4/0/1 |
| 1 | fixed-e5m2 | 0.195313586; 5/0/0 | 0.169408968; 5/0/0 |
| 1 | fixed-hybrid | 0.319655166; 5/0/0 | 0.332802944; 4/0/1 |
| 1 | calibrated | 1.20171395; 1/1/3 | 1.13141427; 2/1/2 |
| 4 | fp32 | 9.25323411; 0/0/5 | 5.30904792; 0/0/5 |
| 4 | fixed-e3m4 | 0.93781583; 2/0/3 | 0.90672603; 3/0/2 |
| 4 | fixed-e4m3 | 0.575092764; 4/0/1 | 0.639862497; 3/0/2 |
| 4 | fixed-e5m2 | 0.424668191; 5/0/0 | 0.429000812; 5/0/0 |
| 4 | fixed-hybrid | 0.51435548; 4/0/1 | 0.746909712; 5/0/0 |
| 4 | calibrated | 1.20741683; 1/0/4 | 0.903748804; 3/0/2 |

## Supported conclusions and limits

- The supplied Fourier features make the original small component learnable: all 10 FP32 cases meet the threshold throughout their final three measurements. This demonstrates preservation/learning with the supplied frequency representation, not discovery of those frequencies from raw inputs.
- All 70 cases sustained the joint threshold at some point. Only 66 pass the final three measurements: misses are gain-1 fixed-e5m2 seeds 11/14, gain-4 fixed-e5m2 seed 11, and gain-4 fixed-hybrid seed 12. These are persistence failures, not numerical run failures. Earlier sustained crossing does not imply stable final performance.
- FP32 has the lowest mean total and component errors at both gains. Adaptive loses to FP32 on both metrics on every paired seed.
- Repeated selection does not show a consistent improvement over calibrated. Its mean total error is 20.17% higher at gain 1 and 20.74% higher at gain 4. Component comparisons are mixed: mean ratios 1.1314 and 0.90375, with mixed paired wins/losses.
- The added fixed-E3M4 control matters. At gain 1 it has the lowest mean total and component errors among the eight-bit conditions. Adaptive has 37.78% higher mean total error there. At gain 4 adaptive has a modestly lower total-error mean than fixed E3M4, but wins only two of five paired seeds; this is not a consistent advantage.
- Adaptive beats fixed E5M2 on both metrics in every seed at both gains, and often beats the other standard-format controls. This does not establish the benefit of repeated selection: fixed E3M4 and calibrated are stronger controls, and all changing arrays are scalar output-bias arrays.
- The lack of substantial-array preference changes leaves little opportunity for adaptation. These results cannot reject adaptive representations broadly, attribute gains solely to activation saturation, or establish hardware/runtime advantages. Runtime conclusions await the separate serial timing pass.

## Reproduction

From `/Users/neeraj/src/zomglings/afloat`:

```sh
.venv/bin/python /var/folders/x5/qnn2lxld3j75g3kt9dj_v1v40000gn/T/tmp.w9lneGMU2U/summarize_accuracy.py --label final --formats
```

The script uses the implementation’s explicit format tie priority (E4M3, E5M2, E3M4); JSON key sorting is not the tie rule. Final files preserve exact means, sample deviations, all paired comparisons, threshold crossing/confirmation steps, raw final rows, and completion metadata. Collection is complete, with no CPU-heavy work remaining.

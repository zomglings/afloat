# Follow-up plan

The prior study found that almost every value used E3M4 and only the scalar
output bias changed format. FP32 also failed to learn the small sine component.
This follow-up asks whether repeated selection helps after feature learnability
and changes in preferred formats are checked explicitly.

The strongest simple control is fixed E3M4 with the same current-array scaling.
All conditions retain FP32 arithmetic, master weights, and Adam state. This
continues the representation experiment; it does not measure native FP8 hardware.
A failure to beat fixed E3M4 and choosing once is a valid outcome.

## Development, before final comparisons

Use development seeds 100 and 101, separate from prior seeds 0–2 and from the
fresh final comparison seeds. Preserve the original `detail` function, including
its 0.01-amplitude, high-frequency sine component. First screen activation gains
1, 4, and 8 on seed 100, with a 20,000-update budget and a cosine learning-rate
decay from 0.001 to 0.00001. Multiply inputs to the three nonlinearities by the
same gain in every block. Keep the network size and raw targets unchanged.

Record measurements every 1,000 updates. A setting passes the feature check if
its final three measurements each recover amplitude between 0.008 and 0.012,
have isolated small-component error below 0.25 of its true squared magnitude,
and have overall validation MSE below 0.0001. The isolated measurement subtracts
each fixed-x1 row's mean over x2 before comparing prediction and truth; a model
with no x2 dependence scores 1. Choose the smallest gain that passes, then
confirm it on seed 101. If none passes, report that and record any revised
development experiment before running it; do not substitute an easier target
without saying so.

Observe weight/gradient candidate scores and activation regions at fixed
checkpoints. Separately examine controlled arrays with a dominant outlier and
small values to check whether the format family and score can prefer wider
ranges at all. An array is substantial for this check if it has at least 32
entries; a scalar output-bias change does not count. Synthetic array results
only check representation behavior, not training benefit.

Freeze the final protocol after development, before running fresh comparison
seeds. It will specify workloads, all seven controls including fixed E3M4,
training budget, accuracy thresholds, format-selection interval, and paired
initialization/data. Retain every development result and every final seed.

## Evidence and decision rule

| Question | Required evidence |
| --- | --- |
| Does the small feature become learnable? | Separate development seeds pass both amplitude and isolated-error checks |
| Does selection have a meaningful range choice? | Candidate scores and actual choices on substantial training arrays, distinguished from synthetic arrays |
| Does repeated selection help? | Fresh paired seeds versus fixed E3M4, choosing once, and the existing fixed controls |
| Is any benefit worth its runtime? | Serial measured training time to a frozen accuracy threshold; threshold misses retained |
| Are inference costs different? | Cached weights measured separately from preparation; same FP32 arithmetic disclosed |
| Are results reproducible? | Fixed protocol, source hashes, checkpoints, raw measurements, checks, and independent review |

Report final error as well as the first threshold crossing that persists over
three consecutive validation measurements. Distinguish when a threshold is
first crossed from when persistence is confirmed. Report time for a bounded
run that never meets the threshold rather than silently excluding it. A useful
adaptation claim requires more than lower reconstruction error, which is the
selection score itself. If large arrays retain E3M4, state that the training
experiment still offers little opportunity for format adaptation.

## Development revision after the first screen

The first screen failed the small-feature check for every gain. Final amplitude
recovery was 1.94%, 6.09%, and 6.69% for gains 1, 4, and 8 respectively; isolated
error remained near the no-feature baseline. Every observed substantial array
preferred E3M4. These results are retained.

Before the next screen, add an explicit input representation containing the raw
two coordinates plus their sine and cosine at frequencies 1, 2, 4, and 8 times
pi. These fixed features apply equally to both coordinates and every condition.
They include the target's oscillations: this makes recovery a coefficient-learning
question, not evidence that the network discovered high frequencies from raw
coordinates. The feed-forward blocks and original target remain unchanged.
Screen gains 1 and 4 on seed 100 for the same 20,000-update schedule and the same
feature pass criteria. Confirm the smallest passing gain on seed 101.

For a meaningful adaptation opportunity, the same training array with at least
32 entries must change its strictly preferred format over time. Require a
relative score advantage of at least 1% at both observations, not a tie-broken
label change. Record both candidate scores and actual rounded-value differences.
A synthetic switch or a static preference for a wider format does not satisfy
this training-array criterion. If training arrays never pass it, report that
limitation and restrict final claims to the measured fixed-format comparison.

## Bounded training-budget check

The Fourier-feature screen passes the unchanged feature criteria at both gains
on seed 100, and gain 1 also passes on seed 101. It still shows no substantial
array changing its preferred format. Before the final study, check a shorter
6,000-update cosine schedule for both gains on both development seeds, measuring
every 500 updates. Apply the same final-three feature criteria. This budget
check limits the cost of five fresh paired seeds and keeps both gains as a
controlled comparison of nonlinear input ranges. Retain all four results,
including failures; use 20,000 updates if the shorter schedule fails for either
setting. No fresh comparison seed has been run or inspected.

## Frozen comparison

The shorter budget failed the unchanged component-error criterion for gain 4
on seed 100. Both gains pass at 20,000 updates on seeds 100 and 101, so use
20,000 updates for both settings as specified above. The final
[protocol](followup-protocol.json) freezes 70 conditions: two gains, seven formats,
and five fresh paired seeds (10–14). Measurements occur every 500 updates.

No substantial array changed strict preference during development. A synthetic
1,536-entry sequence switches E3M4 to E4M3 to E5M2 and back with large score
margins, but this is reconstruction evidence only. The final study therefore
tests the added fixed-E3M4 control and preservation of a learnable small feature;
it cannot promise a strong test of changing training-array format preferences.

Accuracy runs use three single-thread CPU processes. A subsequent serial pass
measures time to the frozen accuracy threshold for seeds 10–12 and FP32, fixed
E3M4, calibrated, and adaptive, at both gains. It checks exact saved model and
optimizer states at measurement steps. Cached inference and weight preparation
are measured for all 70 final models. Initial format selection and warmup costs
are included explicitly in threshold timing; data loading and verification are
reported separately from timed model work. No performance pass runs alongside
accuracy training or analysis.

All actual format changes and all reselection scores are retained. Routine
rounding records are sampled every 100 updates plus the final update, reducing
file-writing cost without changing training. Do not describe their aggregate
rounding statistics as measurements of every update. Save checkpoints at every
validation measurement for independent replay checks.

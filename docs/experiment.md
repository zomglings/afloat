# afloat: adaptive float

## Question and claim

Can a number representation fitted to a model's current weights and gradients
preserve learning better than a standard representation at the same bit budget?
The proposed capability is a small, reproducible software experiment that can
test this question. An advantage for adaptive formats is a hypothesis, not a result.

The strongest simpler alternatives are a standard format with adaptive scales,
and a format chosen separately for each array once after a common warmup.
Adaptation is unnecessary for a tested workload if it does not improve held-out
function approximation over those controls across repeated seeds.

## Network and functions

Inputs are two independent uniform values in [-1, 1]. A linear map produces 32
hidden values. Three blocks each expand 32 values to 48, apply ReLU, tanh, and SiLU
to separate groups of 16, project back to 32, and add the block input. A linear
output predicts one value. Every layer has a bias: 9,585 parameters in total.

| Name | Function | Intended challenge |
| --- | --- | --- |
| linear | x1 + 0.5 x2 | Simple control |
| kink | max(0, x1) + 0.3 abs(x2) | Changes in slope |
| plateau | tanh(6 (x1 + 0.5 x2)) | Steep transition and flat regions |
| detail | sin(2 pi x1) + 0.01 sin(8 pi x2) | Small feature beside a large one |
| bump | exp(-80 ((x1 - 0.6)^2 + (x2 + 0.3)^2)) | Localized feature |
| composition | tanh(3 sin(4 x1) + 2 x2) | Nested nonlinear behavior |

Train a separate model for each function. Target shape does not prove which
hidden activation regions are used. Record inputs and slopes for every activation
group and compare the quantized-weight evaluation with the current FP32 weight
copy on identical inputs. Record ReLU sign changes and nonlinear output error.
These comparisons diagnose sensitivity; they do not establish causation by themselves.

## Initial representation family

Use one sign bit and three candidates: E4M3FN, E5M2, and experimental E3M4.
E4M3FN and E5M2 follow PyTorch's corresponding float8 finite values. E3M4 uses
bias 3, subnormal values, and reserves the all-ones exponent for nonfinite values.
Round to nearest with even-code ties and saturate to the largest finite value.
Reject nonfinite inputs. For all-zero arrays use scale 1. Scale values are
bounded below by the smallest FP32 subnormal, 2^-149, to preserve the stated
scale storage budget. Conversion intermediates use FP64; a quantized result
outside the FP32 simulation range fails visibly.

Every weight, bias, and parameter-gradient array gets a power-of-two scale freshly
computed from its full maximum magnitude. Use the same scale rule and array
boundaries in all low-precision conditions. Customization chooses a format, not
an extra scaling advantage. The selection score is mean squared reconstruction
error on a deterministic subsample of the current array; the scale still uses
the entire array. Validation examples never select the format.

| Condition | Format rule |
| --- | --- |
| fp32 | Reference without rounding |
| fixed-e4m3 | E4M3FN for weights and gradients |
| fixed-e5m2 | E5M2 for weights and gradients |
| fixed-hybrid | E4M3FN for weights; E5M2 for gradients |
| calibrated | Choose once from the three candidates after warmup |
| adaptive | Start with the same choices as calibrated; reselect periodically |
| fixed-e3m4 | E3M4 for weights and gradients; added for the follow-up |

All eight-bit conditions reserve the same conceptual storage: eight bits per
element, 32 bits per scale, and two bits per array for a format identifier.
Fixed conditions reserve the identifier space even though their format is known.
Report payload and metadata separately. No packed storage is implemented; this
budget is the simulated representation, not actual process memory. FP32 master
weights and optimizer state are additional, common training storage.

This family tests changing range versus precision during training. It does not
test arbitrary learned sets of representable values. Such sets require a later
condition with their description cost explicitly included.

## Training controls

Start each seed with a shared FP32 warmup, including Adam's optimizer state.
Fit initial custom choices using the next training batch in FP32, without an
optimizer update. Copy that checkpoint and the optimizer state to every condition.
Use identical indexed training batches, validation points, and diagnostic probes.
No condition consumes random numbers that affect another condition's data.

Round weights for the forward pass and use the identity derivative through that
rounding operation. Explicitly round parameter gradients before Adam's update.
Keep the optimizer, master weights, activation evaluation, and arithmetic in
FP32. This is a weights-and-gradients representation experiment, not a simulation
of all arithmetic in an eight-bit training accelerator. The reference and all
conditions use the same raw targets, mean squared loss, and learning rate.

## Measurements and decision rule

The primary outcome is held-out mean squared prediction error at equal numbers
of training steps. Compare formats within each target; raw losses across targets
have different meanings. Retain every condition, including failed or worse runs.
Numerical failures are recorded per condition; remaining conditions still run.
If a shared warmup fails, record its conditions as skipped. A study containing
failures or skipped conditions writes its summary and exits unsuccessfully.
Programming or filesystem errors stop execution immediately with an error record.

Save validation loss, region-specific error, recovery of the small sine component,
weight/gradient rounding error, values rounded to zero, scale metadata, selected
formats, initial calibration and later selection time, and activation regions.
On measurement steps, compare the current FP32 master-weight gradient with the
gradient computed using rounded weights, then with its rounded representation.
The extra FP32 gradient calculation does not change optimizer gradients or state.
Save final model and
optimizer checkpoints, initial checkpoint,
configuration, dependency versions, source hashes, and raw measurements.
Format events use zero-based update indices. Metric steps count completed
updates: gradient diagnostics describe the update just applied, and validation
describes the resulting weights. Replotting writes a separate file so the
original plot and its manifest remain intact.

First run a short implementation check. Then use a predeclared run length and
three paired seeds for kink, detail, and bump. Check that the FP32 reference
learns the desired feature before interpreting any low-precision result. A short
run that cannot fit the target is inconclusive. Do not stop early on a favorable
adaptive result or select the best format separately using final test loss.
Report all fixed controls; choose any single deployment baseline on separate
development runs before a confirmatory comparison.

Reduced reconstruction error alone is insufficient: selection directly optimizes
that quantity. A useful adaptation result requires better held-out approximation
than both fixed and calibrated controls, reproducible across seeds. Report
selection and metadata costs even when accuracy improves. CPU simulation timing
cannot establish hardware speed or energy savings.

## Later experiments

- Verify learnability and choose a fixed training budget on development seeds.
- Repeat paired comparisons on fresh seeds and report variability.
- Test six-bit formats if all eight-bit conditions are indistinguishable.
- Add learned representable values with an honest description budget.
- Add low-precision activation and backward matrix inputs as separate changes.
- Add gradually changing targets as a separate adaptation-speed experiment.
- Run controlled activation gain sweeps and single-activation network controls.
- Test a hardware implementation only after evidence of a numerical benefit.

## Follow-up controls

The [follow-up plan](followup-plan.md) records development decisions, and the
[frozen protocol](followup-protocol.json) specifies the next comparison. The
original six-condition study remains unchanged in its saved evidence. New CLI
runs include fixed E3M4 by default; the original-study scripts explicitly retain
their six conditions.

`--activation-gain` multiplies inputs to each ReLU, tanh, and SiLU group; its
default of one preserves the original network. `--input-features fourier` adds
fixed sine and cosine features at frequencies 1, 2, 4, and 8 times pi for each
coordinate. This raises the parameter count from 9,585 to 10,097 and makes the
original oscillatory target available as a combination of input features. It
tests learning that combination, not discovery of those frequencies from raw
coordinates. The default input representation remains raw coordinates.

`--final-learning-rate` enables cosine decay across the declared number of
updates; omitting it preserves the constant rate. `--format-log-every` samples
routine rounding records while always retaining actual switches and candidate
scores at reselection. Its default of one logs every update. Switch records
include the number of values changed by switching formats and the squared
difference between old and new rounded values on the same current array.

`--save-checkpoints` stores model and optimizer state at every validation step.
New checkpoints include the model settings needed to reconstruct the input
representation and activation gain. The fixed-grid detail measurement now also
records isolated small-component relative error. `training_seconds` sums update
work, including rounding statistics, but excludes gradient reference probes,
validation, file writes, and checkpoint saving. Those exclusions distinguish
it from `elapsed_seconds`; neither is an isolated speed measurement when
multiple training processes run concurrently.

## Prior work and implementation references

- [Flexpoint](https://arxiv.org/abs/1711.02213): adaptive shared exponents.
- [FP8 formats](https://arxiv.org/abs/2209.05433): E4M3 and E5M2.
- [MX emulation](https://github.com/microsoft/microxcaling): configurable simulations.
- [PyTorch fake quantization](https://pytorch.org/blog/quantization-aware-training/):
  rounding values while retaining floating-point computation.

The implementation is original code for this experiment; dependencies retain
their own licenses. The analytic targets require no external dataset.

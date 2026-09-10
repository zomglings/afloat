# Initial implementation evidence

The software claim is that this repository runs matched small-network experiments
with specified fixed and adaptive formats, and preserves enough evidence to inspect
and reproduce them. The research claim that adaptation improves learning is untested.

## Obligations

| Claim | Rival explanation or failure | Evidence |
| --- | --- | --- |
| Standard formats are correctly represented | Incorrect exponent bias or rounding | Native PyTorch float8 comparisons; midpoint, subnormal, saturation, and sign tests |
| Adaptation changes the representation | Only scales change, or choices never update | Controlled distributions select E3M4 versus E5M2; fixed calibration remains unchanged |
| Conditions are paired | Mode order changes initialization or training data | Reverse-order runs produce identical final weights and predictions |
| Quantization affects training | No-op rounding or detached master weights | Identity-derivative test; explicit weight and gradient records; changed checkpoints |
| Features and activation regions are observable | Overall loss hides missing detail | Missing-sine-component negative control; exact-function regional error checks; activation comparison |
| Outputs can be inspected and reproduced | Missing settings, source, or corrupt artifacts | CLI integration, source snapshot, configuration, checkpoints, and manifest hashes |

## Current evidence

The initial implementation passed `make check`: Ruff lint, Ruff format checking,
mypy on nine source files, and 33 tests. `make smoke` completed all six conditions.
The local environment is macOS on ARM, Python 3.13.14, PyTorch 2.14.0, and the
versions pinned in `uv.lock`. GitHub checks also run on Linux with CPU-only PyTorch.
Their actual status is available in the repository's Actions tab.

A bounded execution check completed all six targets and all six conditions,
using seed 0, 10 common warmup steps, and 100 subsequent steps per condition.
All 36 traces completed. The plot was visually inspected; a title overlap was
fixed. These short traces do not establish adequate training of every feature.
For example, the detail target still has FP32 validation error around 0.368.

Saved-settings replay and replay using the saved source snapshot both reproduced
identical final model tensors and validation predictions. Adding the FP32 gradient
probe left all 36 final model states and prediction arrays unchanged.

The [compact evidence](evidence/initial/) contains the exact configuration,
environment and source hashes, all aggregate measurement rows, and the summary.
The initial run was made before the implementation commit, so its recorded Git
revision is the initial license commit and `git_dirty` is true. The saved source
hashes identify the actual code used; they must not be mistaken for that Git revision.
Full checkpoints, source snapshots, and per-array logs remain in the ignored
local directory `runs/initial-validation.ZMHbPk/result`.

Regenerate the run in a fresh directory:

```sh
uv sync --locked
make check
mkdir -p runs
afloat_run=$(mktemp -d runs/initial-validation.XXXXXX)
uv run afloat run --output "$afloat_run/result" \
  --targets linear kink plateau detail bump composition --seeds 0 \
  --steps 100 --warmup-steps 10 --adapt-every 20 --eval-every 20
```

The compact evidence files copy `config.json`, `environment.json`, `metrics.csv`,
and `summary.json` from that completed run. CSV line endings are normalized from
CRLF to LF for Git; values and records are unchanged.

## Adversarial examination

Rath could not start because `OPENAI_API_KEY` was unavailable; its failed launch
was not counted as a completed review. A separate read-only review agent examined
the raw experiment design, source, tests, and public CLI. It reproduced three
issues or incomplete measurements:

1. Gradient direction measured only parameter-gradient rounding. The implementation
   now also compares against a freshly computed FP32 master-weight gradient on
   the same batch. A regression test proves this probe does not mutate optimizer
   gradients.
2. Selection time omitted the initial calibration. Its time is now recorded
   separately, shared between the two custom conditions, and included in their
   total selection cost.
3. A numerical failure stopped later comparisons. The runner now preserves the
   failing condition and its partial measurements, attempts remaining conditions,
   writes a study summary and manifest, and exits nonzero.

The examiner independently verified the three fixes with the original
reproductions and found no unresolved issue within that focused verification.
The failure reproduction deliberately uses learning rate 10000 with two targets
and two conditions; all four are now attempted. It is a failure-preservation
test, not evidence about normal training stability.

## Assessment

| Conclusion | Status | Basis |
| --- | --- | --- |
| The first simulator executes and records matched comparisons | Observed | CLI tests and 36 completed short traces |
| Format selection can respond to distribution shape | Observed | Narrow versus wide distribution test selects different formats |
| Source-snapshot replay reproduces final tensors locally | Observed | Direct replay and independent examiner |
| Adaptive formats improve function approximation | Unknown | No completed study across multiple seeds or validated training budget |
| Adaptive formats improve hardware speed or energy use | Unknown | No hardware implementation |

**Ready for further experiments.** The next research step is to establish that
the FP32 reference learns each desired feature, choose a fixed run budget on
development seeds, and run paired comparisons on fresh seeds.

## Limits

The first implementation searches three eight-bit layouts. It does not learn an
arbitrary set of representable values. The optimizer, master weights, nonlinear
operations, and matrix arithmetic use FP32. The reported representation budget
includes scales and format identifiers but is not the simulator's memory use.

Short training checks establish execution, not learnability of every feature,
superiority of adaptation, repeatability across machines, or hardware speed.
Initial calibration is performed once per shared checkpoint and attributed to
each custom condition. Timing includes CPU simulation and measurement costs;
it cannot predict accelerator performance. A zero gradient has undefined direction
and is recorded as a null cosine rather than an invented similarity score.

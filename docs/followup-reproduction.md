# Reproduce the follow-up

Use a clean checkout and `uv sync --locked`. The primary commands in
[followup-results.md](../followup-results.md) run all 70 accuracy conditions,
verify/analyze them, and then measure serial time to accuracy plus inference.
Do not run another training job or model analysis during timing. CPU suffices.

The frozen protocol is [followup-protocol.json](followup-protocol.json). Repeating
its final comparison does not require rerunning development. The launcher records
the current clean revision, the protocol hash, exact commands, and process order.
Numerical equality across a different CPU, operating system, or PyTorch version
is not assumed; the benchmark requires exact replay against that run's own
checkpoints.

## Development, including failed settings

The following reproduces all 11 FP32 development runs and the separate synthetic
array. Keep the group names for `summarize_development.py`:

```sh
mkdir -p runs
afloat_development=$(mktemp -d runs/followup-development.XXXXXX)
uv run python scripts/run_development.py \
  --output "$afloat_development/development-100" \
  --seed 100 --features raw --gains 1 4 8
uv run python scripts/run_development.py \
  --output "$afloat_development/fourier-100" \
  --seed 100 --features fourier --gains 1 4
uv run python scripts/run_development.py \
  --output "$afloat_development/fourier-101" \
  --seed 101 --features fourier --gains 1
uv run python scripts/run_development.py \
  --output "$afloat_development/budget-100" \
  --seed 100 --features fourier --gains 1 4 --steps 6000 --eval-every 500
uv run python scripts/run_development.py \
  --output "$afloat_development/budget-101" \
  --seed 101 --features fourier --gains 1 4 --steps 6000 --eval-every 500
uv run python scripts/run_development.py \
  --output "$afloat_development/gain4-101" \
  --seed 101 --features fourier --gains 4
uv run python scripts/run_synthetic.py --output "$afloat_development/synthetic"
uv run python scripts/summarize_development.py "$afloat_development" \
  --output "$afloat_development/development-analysis"
```

The maintained development command accepts the eventual budget/range options.
The original development sources evolved before the final comparison was frozen.
Each group in [the evidence](evidence/followup/development) preserves its actual
protocol, script, environment, original manifest, source archive, and measurements.
For exact original source inspection, extract the group's `source.tar.gz` into a
fresh scratch directory. The archived script is stored as `.py.txt`; copy it to a
new `.py` file before execution, use the archived dependency lock and `src/`, and
supply the output, seed, features, gains, and supported budget flags from that
group's protocol. The earliest raw-coordinate script has no feature/budget flags;
the next scripts have fixed 20,000-update budgets. Do not overwrite original data.

## Evidence checks

The public `evidence-sha256.json` covers the compact files published in Git.
Original `manifest.json` files also name checkpoints and logs that remain local;
they cannot be verified in full using only the compact copy. Full regeneration
creates those files and lets the analysis verify all original-run contents.

Each final checkpoint stores the model settings, FP32 weights and optimizer,
chosen formats, rounded weights, and validation predictions. Every validation
checkpoint stores the complete state needed by the serial replay. Never select
a later checkpoint to replace an unfavorable final outcome.

The analysis snapshots include their benchmark helper dependency. The published
review programs are historical source with explicit original local paths. Their
findings describe executed checks; use maintained tools for a new study, or copy
an archived review into a fresh scratch directory and update its input locations.

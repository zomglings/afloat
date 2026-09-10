# afloat

**Adaptive float:** test whether number formats fitted to a model's current
weights and gradients preserve learning better than fixed formats.

A small network composes feed-forward blocks with ReLU, tanh, and SiLU. Six
analytic functions exercise changes in slope, saturation, small details,
localized features, and nested nonlinear behavior. The simulator compares
standard eight-bit formats with formats chosen once or repeatedly during training.

This is an experiment under development. No advantage for adaptation has been
established. Arithmetic and optimizer state remain in FP32; the initial study
rounds weights and parameter gradients. It does not measure accelerator speed.

Read the [experiment design](docs/experiment.md) for the hypothesis, controls,
format definitions, measurement plan, and later work.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --locked
make check
make smoke
```

Run a longer comparison on three functions and three seeds:

```sh
uv run afloat run --output runs/first-study \
  --targets kink detail bump --seeds 0 1 2 \
  --steps 2000 --warmup-steps 50 --adapt-every 100 --eval-every 100
```

Use a new output directory for each run; existing outputs are never overwritten.
The first implementation uses Python 3.13 and runs on CPU with one PyTorch thread.
Use `uv run afloat run --help` for options. A smoke run checks execution only;
the longer command is a proposed study, not a validated sufficient training budget.

Each run saves its configuration, source and environment information, common
initial checkpoints, validation data, measurements, selected formats, final
checkpoints, and a loss plot. `runs/` is ignored by Git. Regenerate plots with:

```sh
uv run afloat plot runs/first-study
```

This writes `loss-regenerated.png`, preserving the original plot and its manifest.
Use `--destination` to choose another new file.

Repeat saved settings with `uv run afloat replay runs/first-study --output runs/repeat`.
Each run also saves a `source/` snapshot. To reproduce with that exact code and
dependency lock, run `uv run --project runs/first-study/source afloat replay
runs/first-study --output runs/repeat-exact`. Checkpoint files are PyTorch files;
load only trusted artifacts with `torch.load(..., weights_only=True)`.

Implementation evidence and open research questions are recorded in
[the initial validation report](docs/validation.md).

## Development

`make check` runs Ruff lint, Ruff format checking, mypy, and pytest. Run
`make smoke` after changing training or artifacts. Dependencies are pinned by
`uv.lock`. Tests verify numerical boundaries, independent standard-format
agreement, format adaptation, paired controls, diagnostics, and the public CLI.

MIT licensed. See [LICENSE](LICENSE).

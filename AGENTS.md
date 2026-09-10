# Working on afloat

Use plain language. Keep empirical claims separate from implementation checks.
Preserve the experiment controls in `docs/experiment.md` when changing training.
Keep an explicit checklist for substantial work and record reproducible evidence.
Do not commit generated training runs from `runs/`.

Before committing, run `make check`: lint, format, type checking, and tests.
Run `make smoke` after changes to training or run artifacts; use a fresh output directory.
Never amend commits or rebase. Write commit messages as two or three sentences,
each on its own line. Do not claim credit in commits or pull requests.

.PHONY: check smoke

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest

smoke:
	@mkdir -p runs
	@afloat_run=$$(mktemp -d runs/smoke.XXXXXX); \
	uv run afloat run --output "$$afloat_run/result" --targets kink --steps 10 --warmup-steps 2 --eval-every 5 --adapt-every 5 --seeds 0

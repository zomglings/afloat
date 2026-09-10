"""Failures must retain spent work; stopping must require consecutive success."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch


class EmptyOptimizer:
    def state_dict(self):
        return {}


@pytest.mark.parametrize(
    ("kind", "sequence", "expected_step", "expected_seconds", "crossing"),
    [
        ("setup_failure", [False, False], 0, 1.0, None),
        ("partial_failure", [False, False], 123, 2.9, None),
        ("validation_failure", [False, False], 500, 3.9, None),
        ("reset", [True, False, True, True, True], 2000, 9.9, 1000),
        ("miss", [False, True, False], 1000, 5.9, None),
        ("initial", [True, True, True], 1000, 5.9, 0),
    ],
)
def test_replay_preserves_failures_and_requires_three_passes(
    tmp_path, monkeypatch, kind, sequence, expected_step, expected_seconds, crossing
):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "followup_benchmark", root / "scripts/benchmark_followup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    study = tmp_path / "study"
    base = study / "gain-1-seed-10"
    group = base / "detail/seed-10/fp32"
    group.mkdir(parents=True)
    output = tmp_path / "output"
    output.mkdir()
    plan = json.loads((root / "docs/followup-protocol.json").read_text())
    plan.update(steps=500 * (len(sequence) - 1), modes=["fp32"])
    model = torch.nn.Linear(1, 1)
    references = []
    for index, passed in enumerate(sequence):
        row = {
            "step": index * 500,
            "validation_mse": 0.00001 if passed else 1.0,
            "detail_component_relative_mse": 0.1 if passed else 1.0,
            "detail_amplitude": 0.01 if passed else 0.0,
        }
        references.append(row)
        torch.save(
            {"model": model.state_dict(), "optimizer": {}, "format_choices": {}},
            group / f"step-{row['step']}.pt",
        )
    (group / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in references) + "\n"
    )
    torch.save({"validation": torch.zeros(1, 1)}, base / "inputs.pt")
    ticks = iter(range(100))
    monkeypatch.setattr(module, "perf_counter", lambda: float(next(ticks)))

    def warmup(*args):
        if kind == "setup_failure":
            raise FloatingPointError("injected setup failure")
        return model, EmptyOptimizer(), {}, 0.1

    def block(model, optimizer, policy, config, seed, start, end, progress):
        progress["completed_step"] = 123 if kind == "partial_failure" else end
        if kind == "partial_failure":
            raise FloatingPointError("injected partial block failure")

    values = iter(references)

    def evaluate(*args):
        row = next(values)
        if kind == "validation_failure" and row["step"] == 500:
            raise FloatingPointError("injected validation failure")
        return {key: value for key, value in row.items() if key != "step"}

    monkeypatch.setattr(module, "warmup", warmup)
    monkeypatch.setattr(module, "train_block", block)
    monkeypatch.setattr(module, "evaluate", evaluate)
    result = module.replay_case(study, output, plan, 1.0, 10, "fp32")
    assert result["final_step"] == expected_step
    assert result["measured_seconds"] == pytest.approx(expected_seconds)
    assert result["reached"] == (crossing is not None)
    if crossing is None:
        assert result["crossing"] is None
    else:
        assert result["crossing"]["step"] == crossing
    if "failure" in kind:
        assert result["status"] == "numerical_failure"
    else:
        assert result["status"] == "completed"
    if kind == "setup_failure":
        assert result["unsplit_failed_setup_seconds"] == 1.0
        assert result["warmup_seconds"] is None
        assert result["initial_calibration_seconds"] is None
    elif kind == "partial_failure":
        assert result["training_seconds"] == 1.0
        assert result["last_validation_step"] == 0
    elif kind == "validation_failure":
        assert result["validation_seconds"] == 2.0
        assert result["last_validation_step"] == 0

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from afloat.artifacts import sha256
from afloat.formats import MODES
from afloat.runner import RunConfig, run_experiment


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_public_cli_paired_conditions_and_artifacts(tmp_path):
    output = tmp_path / "first"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "afloat.cli",
            "run",
            "--output",
            str(output),
            "--targets",
            "linear",
            "--steps",
            "4",
            "--warmup-steps",
            "2",
            "--batch-size",
            "16",
            "--eval-every",
            "2",
            "--adapt-every",
            "2",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert str(output) in result.stdout
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert len(summary["final_metrics"]) == len(MODES)
    assert summary["parameter_count"] == 9585
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(sha256(output / path) == digest for path, digest in manifest.items())
    assert (output / "loss.png").stat().st_size > 1000
    group = output / "linear" / "seed-0"
    initial = torch.load(group / "initial.pt", weights_only=True)
    calibrated = records(group / "calibrated" / "metrics.jsonl")
    adaptive = records(group / "adaptive" / "metrics.jsonl")
    for index in (0, 1):
        assert calibrated[index]["validation_mse"] == adaptive[index]["validation_mse"]
    for mode in MODES:
        checkpoint = torch.load(group / mode / "final.pt", weights_only=True)
        assert any(
            not torch.equal(p, checkpoint["model"][name])
            for name, p in initial["model"].items()
        )
        assert len(records(group / mode / "activations.jsonl")) == 27
        if mode != "fp32":
            events = records(group / mode / "formats.jsonl")
            assert len(events) == 4 * 2 * len(initial["model"])
            assert {event["array"].split(":")[0] for event in events} == {
                "weight",
                "gradient",
            }
    before = (output / "config.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(RunConfig(steps=1), output)
    assert (output / "config.json").read_bytes() == before
    replay = tmp_path / "replay"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "afloat.cli",
            "replay",
            str(output),
            "--output",
            str(replay),
        ],
        env={**os.environ, "PYTHONPATH": str(output / "source/src")},
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    for mode in MODES:
        a = torch.load(group / mode / "final.pt", weights_only=True)
        b = torch.load(replay / "linear/seed-0" / mode / "final.pt", weights_only=True)
        assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
        final = records(group / mode / "metrics.jsonl")[-1]
        if mode == "fp32":
            assert final["total_gradient_mse"] == 0
        if mode in ("calibrated", "adaptive"):
            assert final["initial_calibration_seconds"] > 0
        assert final["selection_seconds"] == (
            final["initial_calibration_seconds"] + final["reselection_seconds"]
        )


def test_mode_order_does_not_change_training_and_adaptation_starts_after_calibration(
    tmp_path,
):
    config = RunConfig(
        targets=("kink",),
        modes=("calibrated", "adaptive"),
        steps=3,
        warmup_steps=1,
        batch_size=16,
        eval_every=1,
        adapt_every=1,
    )
    one = run_experiment(config, tmp_path / "one")
    two = run_experiment(
        replace(config, modes=tuple(reversed(config.modes))), tmp_path / "two"
    )
    for mode in config.modes:
        a = torch.load(one / "kink/seed-0" / mode / "final.pt", weights_only=True)
        b = torch.load(two / "kink/seed-0" / mode / "final.pt", weights_only=True)
        assert all(
            torch.equal(a["model"][name], b["model"][name]) for name in a["model"]
        )
        assert torch.equal(a["validation_predictions"], b["validation_predictions"])


@pytest.mark.parametrize(
    "change",
    [
        {"steps": 0},
        {"targets": ()},
        {"modes": ("fp32", "fp32")},
        {"learning_rate": float("nan")},
        {"seeds": (-1,)},
        {"warmup_steps": -1},
    ],
)
def test_invalid_run_settings_fail_before_creating_output(tmp_path, change):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run_experiment(replace(RunConfig(), **change), output)
    assert not output.exists()


def test_numerical_failure_preserves_partial_results_and_attempts_other_conditions(
    tmp_path,
):
    output = tmp_path / "divergent"
    config = RunConfig(
        targets=("kink", "linear"),
        modes=("fixed-e4m3", "fp32"),
        steps=1,
        warmup_steps=0,
        batch_size=16,
        learning_rate=10000,
    )
    with pytest.raises(RuntimeError, match="conditions failed"):
        run_experiment(config, output)
    progress = records(output / "progress.jsonl")
    assert len([row for row in progress if row["status"] == "started"]) == 4
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "completed_with_failures"
    assert summary["failures"]
    assert (output / "metrics.csv").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(sha256(output / path) == digest for path, digest in manifest.items())


def test_fp32_gradient_probe_does_not_change_optimizer_gradients():
    from afloat.model import make_model
    from afloat.runner import gradient_comparison, training_batch
    from afloat.targets import target_values

    model = make_model(0)
    x = training_batch(0, 0, 16)
    (model(x) - target_values("kink", x)).square().mean().backward()
    before = [p.grad.clone() for p in model.parameters()]
    reference = torch.cat([part.flatten() for part in before]).double()
    modified = reference.clone()
    modified[::2] = 0
    metrics = gradient_comparison(model, "kink", x, reference, modified)
    assert metrics["weight_rounding_gradient_cosine"] == pytest.approx(1)
    assert metrics["total_gradient_cosine"] < 1
    assert metrics["total_gradient_mse"] > 0
    assert all(
        torch.equal(a, p.grad) for a, p in zip(before, model.parameters(), strict=True)
    )

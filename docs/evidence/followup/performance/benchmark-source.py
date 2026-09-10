"""Serial threshold replay and final-model inference for the frozen follow-up."""

import argparse
import json
import math
import platform
import random
import statistics
import subprocess
from dataclasses import fields
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import torch
from torch.func import functional_call
from torch.utils.benchmark import Timer

from afloat.artifacts import (
    append_json,
    record_manifest,
    record_source,
    sha256,
    write_json,
)
from afloat.formats import FormatPolicy
from afloat.model import make_model
from afloat.runner import (
    RunConfig,
    rounded_parameters,
    training_batch,
    update_learning_rate,
    warmup,
)
from afloat.targets import (
    detail_component_error,
    feature_metrics,
    sample_inputs,
    target_values,
)


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for first, second in zip(a, b, strict=True):
            assert_same(first, second)
    else:
        assert a == b


def passes(metrics, threshold):
    return (
        metrics["validation_mse"] < threshold["validation_mse_below"]
        and metrics["detail_component_relative_mse"]
        < threshold["detail_component_relative_mse_below"]
        and threshold["detail_amplitude_at_least"]
        <= metrics["detail_amplitude"]
        <= threshold["detail_amplitude_at_most"]
    )


def train_block(model, optimizer, policy, config, seed, start, end, progress=None):
    for step in range(start, end):
        for group in optimizer.param_groups:
            group["lr"] = update_learning_rate(config, step)
        x = training_batch(seed, config.warmup_steps + step, config.batch_size)
        optimizer.zero_grad(set_to_none=True)
        params = {}
        for name, parameter in model.named_parameters():
            rounded = policy.apply(parameter, name, "weight", step)
            params[name] = parameter + (rounded - parameter).detach()
        prediction = functional_call(model, params, (x,), strict=True)
        loss = (prediction - target_values("detail", x)).square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite replay loss")
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("Nonfinite replay gradient")
            parameter.grad.copy_(policy.apply(parameter.grad, name, "gradient", step))
        optimizer.step()
        policy.events.clear()
        if progress is not None:
            progress["completed_step"] = step + 1


def evaluate(model, policy, inputs):
    with torch.no_grad():
        prediction = functional_call(
            model, rounded_parameters(model, policy), (inputs,), strict=True
        )
    metrics = feature_metrics("detail", inputs, prediction)
    metrics["detail_component_relative_mse"] = detail_component_error(prediction)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Nonfinite replay validation")
    return metrics


def timed(function):
    for _ in range(5):
        function()
    result = Timer(
        stmt="function()", globals={"function": function}, num_threads=1
    ).blocked_autorange(min_run_time=0.1)
    return {
        "median_seconds": result.median,
        "iqr_seconds": result.iqr,
        "number_per_run": result.number_per_run,
        "raw_block_seconds": result.raw_times,
    }


def infer(model, params, x):
    with torch.inference_mode():
        return functional_call(model, params, (x,), strict=True)


def config_for(plan, gain, seed):
    allowed = {field.name for field in fields(RunConfig)}
    settings = {key: value for key, value in plan.items() if key in allowed}
    settings.update(
        targets=("detail",),
        modes=tuple(plan["modes"]),
        seeds=(seed,),
        activation_gain=gain,
    )
    return RunConfig(**settings)


def measure_call(function, costs, phase):
    began = perf_counter()
    try:
        return function()
    finally:
        costs[phase] += perf_counter() - began


def replay_case(study, output, plan, gain, seed, mode):
    root = study / f"gain-{gain:g}-seed-{seed}"
    group = root / "detail" / f"seed-{seed}" / mode
    config = config_for(plan, gain, seed)
    reference_rows = {
        r["step"]: r
        for r in map(json.loads, (group / "metrics.jsonl").read_text().splitlines())
    }
    inputs = torch.load(root / "inputs.pt", weights_only=True)["validation"]
    costs = {"setup": 0.0, "training": 0.0, "validation": 0.0}
    progress = {"completed_step": 0}
    setup_charged = None
    last_metrics, last_validation_step, crossing = None, None, None
    consecutive = 0
    result = {
        "gain": gain,
        "seed": seed,
        "mode": mode,
        "warmup_seconds": None,
        "initial_calibration_seconds": None,
    }
    phase = "setup"
    try:
        model, optimizer, choices, calibration_seconds = measure_call(
            partial(warmup, config, "detail", seed), costs, phase
        )
        warmup_seconds = costs["setup"] - calibration_seconds
        charged_calibration = (
            calibration_seconds if mode in ("calibrated", "adaptive") else 0.0
        )
        setup_charged = warmup_seconds + charged_calibration
        result.update(
            warmup_seconds=warmup_seconds,
            initial_calibration_seconds=charged_calibration,
        )
        policy = FormatPolicy(mode, choices, config.adapt_every, config.sample_size)
        for step in range(0, config.steps + 1, config.eval_every):
            if step:
                phase = "training"
                measure_call(
                    partial(
                        train_block,
                        model,
                        optimizer,
                        policy,
                        config,
                        seed,
                        step - config.eval_every,
                        step,
                        progress,
                    ),
                    costs,
                    phase,
                )
            phase = "validation"
            metrics = measure_call(
                partial(evaluate, model, policy, inputs), costs, phase
            )
            checkpoint = torch.load(group / f"step-{step}.pt", weights_only=True)
            assert_same(model.state_dict(), checkpoint["model"])
            assert_same(optimizer.state_dict(), checkpoint["optimizer"])
            assert_same(policy.choices, checkpoint["format_choices"])
            for key, value in metrics.items():
                assert value == reference_rows[step][key], (gain, seed, mode, step, key)
            last_metrics, last_validation_step = metrics, step
            passed = passes(metrics, plan["threshold"])
            seconds = setup_charged + costs["training"] + costs["validation"]
            append_json(
                output / "threshold-measurements.jsonl",
                {
                    **result,
                    "step": step,
                    **metrics,
                    "passes": passed,
                    "training_seconds": costs["training"],
                    "validation_seconds": costs["validation"],
                    "measured_seconds": seconds,
                    "model_optimizer_choices_verified": True,
                },
            )
            if passed:
                if consecutive == 0:
                    crossing = {"step": step, "measured_seconds": seconds}
                consecutive += 1
            else:
                consecutive, crossing = 0, None
            if consecutive == plan["threshold"]["consecutive_measurements"]:
                break
        reached = consecutive == plan["threshold"]["consecutive_measurements"]
        result.update(
            status="completed",
            reached=reached,
            crossing=crossing if reached else None,
            reselection_seconds=policy.selection_seconds,
        )
    except FloatingPointError as error:
        result.update(
            status="numerical_failure",
            phase=phase,
            message=str(error),
            reached=False,
            crossing=None,
        )
    result.update(
        final_step=progress["completed_step"],
        last_validation_step=last_validation_step,
        training_seconds=costs["training"],
        validation_seconds=costs["validation"],
        measured_seconds=(costs["setup"] if setup_charged is None else setup_charged)
        + costs["training"]
        + costs["validation"],
        unsplit_failed_setup_seconds=costs["setup"] if setup_charged is None else None,
        metrics=last_metrics,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_protocol = json.loads((args.study / "protocol.json").read_text())
    plan = source_protocol["plan"]
    completion = json.loads((args.study / "completion.json").read_text())
    assert len(completion["outcomes"]) == 10 and all(
        r["exit_code"] == 0 for r in completion["outcomes"]
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    record_source(args.output)
    environment = json.loads((args.output / "environment.json").read_text())
    original_environment = json.loads(
        (args.study / "gain-1-seed-10/environment.json").read_text()
    )
    for name, expected in original_environment["source_hashes"].items():
        if name.startswith("src/") or name in ("pyproject.toml", "uv.lock"):
            assert environment["source_hashes"][name] == expected
    assert environment["dependencies"] == original_environment["dependencies"]
    hardware = {"platform": platform.platform(), "machine": platform.machine()}
    if platform.system() == "Darwin":
        hardware["cpu"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    order = [
        (gain, seed, mode)
        for gain in plan["activation_gains"]
        for seed in plan["performance"]["threshold_seeds"]
        for mode in plan["performance"]["threshold_modes"]
    ]
    random.Random(plan["performance"]["timing_order_seed"]).shuffle(order)
    write_json(
        args.output / "protocol.json",
        {
            "source_protocol_sha256": sha256(args.study / "protocol.json"),
            "script_sha256": sha256(Path(__file__)),
            "threshold_order": order,
            "threads": 1,
            "processes": 1,
            "started_at": datetime.now(UTC).isoformat(),
            "hardware": hardware,
            "scope": (
                "Measured warmup plus initial calibration when applicable, "
                "training blocks, and threshold validation. Excludes reference "
                "checkpoint reads/comparisons, logging, inference benchmark, "
                "and source capture. Original 20000-step learning-rate schedule "
                "retained even if stopped early."
            ),
            "threshold": plan["threshold"],
        },
    )
    (args.output / "benchmark-source.py").write_bytes(Path(__file__).read_bytes())
    prime = config_for(plan, 1, 999)
    prime_model, prime_optimizer, prime_choices, _ = warmup(prime, "detail", 999)
    train_block(
        prime_model,
        prime_optimizer,
        FormatPolicy("adaptive", prime_choices, 100, 2048),
        prime,
        999,
        0,
        5,
    )
    wall = perf_counter()
    outcomes = []
    for index, (gain, seed, mode) in enumerate(order):
        print(
            f"Start threshold replay {index + 1}/{len(order)}: "
            f"gain={gain:g}, seed={seed}, {mode}",
            flush=True,
        )
        result = replay_case(args.study, args.output, plan, gain, seed, mode)
        append_json(args.output / "threshold-results.jsonl", result)
        outcomes.append(result)
        print(
            f"Finished threshold replay: {result['status']}, "
            f"reached={result['reached']}, updates={result['final_step']}",
            flush=True,
        )
    inference = []
    inference_order = [
        (gain, seed, mode)
        for gain in plan["activation_gains"]
        for seed in plan["seeds"]
        for mode in plan["modes"]
    ]
    random.Random(919).shuffle(inference_order)
    for gain, seed, mode in inference_order:
        checkpoint = torch.load(
            args.study
            / f"gain-{gain:g}-seed-{seed}"
            / "detail"
            / f"seed-{seed}"
            / mode
            / "final.pt",
            weights_only=True,
        )
        model = make_model(seed, **checkpoint["model_settings"])
        model.load_state_dict(checkpoint["model"])
        policy = FormatPolicy(mode, checkpoint["format_choices"], 100, 2048)
        params = rounded_parameters(model, policy)
        assert_same(params, checkpoint["rounded_parameters"])
        row = {"gain": gain, "seed": seed, "mode": mode}
        for batch in (1, 128):
            row[f"batch_{batch}"] = timed(
                partial(infer, model, params, sample_inputs(777, batch))
            )
        row["preparation"] = timed(partial(rounded_parameters, model, policy))
        append_json(args.output / "inference.jsonl", row)
        inference.append(row)
    summary = []
    for gain in plan["activation_gains"]:
        for mode in plan["modes"]:
            rows = [r for r in inference if r["gain"] == gain and r["mode"] == mode]
            summary.append(
                {
                    "gain": gain,
                    "mode": mode,
                    **{
                        key: {
                            "median_seconds": statistics.median(
                                r[key]["median_seconds"] for r in rows
                            ),
                            "min_seconds": min(r[key]["median_seconds"] for r in rows),
                            "max_seconds": max(r[key]["median_seconds"] for r in rows),
                        }
                        for key in ("batch_1", "batch_128", "preparation")
                    },
                }
            )
    write_json(
        args.output / "summary.json",
        {
            "wall_seconds": perf_counter() - wall,
            "threshold_cases": len(outcomes),
            "inference_cases": len(inference),
            "threshold_failures": sum(r["status"] != "completed" for r in outcomes),
            "threshold_reached": sum(r.get("reached", False) for r in outcomes),
            "inference": summary,
        },
    )
    record_manifest(args.output)
    if any(row["status"] != "completed" for row in outcomes):
        raise RuntimeError("Threshold replays failed; see saved outcomes")
    print(f"Completed serial performance pass: {args.output}", flush=True)


if __name__ == "__main__":
    main()

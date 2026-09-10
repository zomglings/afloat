"""Matched CPU training conditions with explicit weight and gradient rounding."""

import copy
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor, nn
from torch.func import functional_call

from afloat.artifacts import append_json, record_manifest, record_source, write_json
from afloat.diagnostics import activation_statistics
from afloat.formats import MODES, FormatPolicy, choose_format
from afloat.model import make_model
from afloat.targets import (
    TARGETS,
    feature_metrics,
    sample_inputs,
    target_values,
    validation_inputs,
)


@dataclass(frozen=True)
class RunConfig:
    targets: tuple[str, ...] = ("kink", "detail", "bump")
    modes: tuple[str, ...] = MODES
    seeds: tuple[int, ...] = (0,)
    steps: int = 300
    warmup_steps: int = 50
    batch_size: int = 128
    learning_rate: float = 0.001
    eval_every: int = 50
    adapt_every: int = 100
    sample_size: int = 2048
    threads: int = 1

    def validate(self) -> None:
        for values, allowed in ((self.targets, TARGETS), (self.modes, MODES)):
            if (
                not values
                or len(values) != len(set(values))
                or set(values) - set(allowed)
            ):
                raise ValueError(
                    "Targets and modes must be nonempty, unique, and known"
                )
        if (
            not self.seeds
            or len(self.seeds) != len(set(self.seeds))
            or min(self.seeds) < 0
        ):
            raise ValueError("Seeds must be nonempty, unique, and nonnegative")
        for value in (
            self.steps,
            self.batch_size,
            self.eval_every,
            self.adapt_every,
            self.sample_size,
            self.threads,
        ):
            if value <= 0:
                raise ValueError(
                    "Step counts, sample sizes, and threads must be positive"
                )
        if (
            self.warmup_steps < 0
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("Invalid warmup or learning rate")


def training_batch(seed: int, step: int, size: int) -> Tensor:
    return sample_inputs(10_000_000 + seed * 1_000_000 + step, size)


def warmup(
    config: RunConfig, target: str, seed: int
) -> tuple[nn.Sequential, torch.optim.Adam, dict[str, str], float]:
    model = make_model(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for step in range(config.warmup_steps):
        x = training_batch(seed, step, config.batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - target_values(target, x)).square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite warmup loss")
        torch.autograd.backward(loss)
        optimizer.step()
    calibration_started = perf_counter()
    x = training_batch(seed, config.warmup_steps, config.batch_size)
    optimizer.zero_grad(set_to_none=True)
    torch.autograd.backward((model(x) - target_values(target, x)).square().mean())
    initial = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None
        initial[f"weight:{name}"] = choose_format(parameter, config.sample_size)
        initial[f"gradient:{name}"] = choose_format(parameter.grad, config.sample_size)
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer, initial, perf_counter() - calibration_started


def cosine_similarity(a: Tensor, b: Tensor) -> float | None:
    denominator = float(a.norm() * b.norm())
    return float(a.dot(b)) / denominator if denominator != 0 else None


def gradient_comparison(
    model: nn.Module, target: str, x: Tensor, original: Tensor, rounded: Tensor
) -> dict[str, object]:
    reference_loss = (model(x) - target_values(target, x)).square().mean()
    reference_parts = torch.autograd.grad(reference_loss, tuple(model.parameters()))
    reference = torch.cat([part.flatten() for part in reference_parts]).double()
    if not bool(torch.isfinite(reference).all()):
        raise FloatingPointError("Nonfinite FP32 reference gradient")
    return {
        "gradient_rounding_cosine": cosine_similarity(original, rounded),
        "weight_rounding_gradient_cosine": cosine_similarity(reference, original),
        "total_gradient_cosine": cosine_similarity(reference, rounded),
        "total_gradient_mse": float((reference - rounded).square().mean()),
        "gradient_rounding_mse": float((original - rounded).square().mean()),
        "gradient_zeroed_fraction": float(
            ((original != 0) & (rounded == 0)).double().mean()
        ),
    }


def rounded_parameters(model: nn.Module, policy: FormatPolicy) -> dict[str, Tensor]:
    return {name: policy.evaluate(p, name) for name, p in model.named_parameters()}


def evaluate(
    model: nn.Module, policy: FormatPolicy, target: str, x: Tensor
) -> tuple[dict[str, float], dict[str, Tensor]]:
    parameters = rounded_parameters(model, policy)
    with torch.no_grad():
        prediction: Tensor = functional_call(model, parameters, (x,), strict=True)
        metrics = feature_metrics(target, x, prediction)
        reference: Tensor = model(x)
        metrics["weight_rounding_output_mse"] = float(
            (prediction - reference).square().mean()
        )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite validation metrics")
    return metrics, parameters


def run_condition(
    config: RunConfig,
    target: str,
    seed: int,
    mode: str,
    base: nn.Module,
    base_optimizer: torch.optim.Adam,
    initial: dict[str, str],
    calibration_seconds: float,
    validation: Tensor,
    probes: Tensor,
    output: Path,
) -> list[dict[str, object]]:
    output.mkdir()
    model = copy.deepcopy(base)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(copy.deepcopy(base_optimizer.state_dict()))
    policy = FormatPolicy(mode, initial, config.adapt_every, config.sample_size)
    rows: list[dict[str, object]] = []
    started = perf_counter()
    initial_selection = (
        calibration_seconds if mode in ("calibrated", "adaptive") else 0.0
    )

    def measure(step: int, training: dict[str, object]) -> None:
        metrics, parameters = evaluate(model, policy, target, validation)
        row: dict[str, object] = {
            "target": target,
            "seed": seed,
            "mode": mode,
            "step": step,
            "elapsed_seconds": perf_counter() - started,
            "initial_calibration_seconds": initial_selection,
            "reselection_seconds": policy.selection_seconds,
            "selection_seconds": initial_selection + policy.selection_seconds,
            **training,
            **metrics,
        }
        rows.append(row)
        append_json(output / "metrics.jsonl", row)
        for activation in activation_statistics(model, parameters, probes):
            append_json(output / "activations.jsonl", {"step": step, **activation})

    measure(0, {})
    for step in range(config.steps):
        x = training_batch(seed, config.warmup_steps + step, config.batch_size)
        optimizer.zero_grad(set_to_none=True)
        parameters = {}
        for name, p in model.named_parameters():
            rounded = policy.apply(p, name, "weight", step)
            # The identity derivative lets rounding update the FP32 master weights.
            parameters[name] = p + (rounded - p).detach()
        prediction: Tensor = functional_call(model, parameters, (x,), strict=True)
        loss = (prediction - target_values(target, x)).square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Nonfinite training loss: {target}/{seed}/{mode}/{step}"
            )
        torch.autograd.backward(loss)
        original_gradients, rounded_gradients = [], []
        for name, p in model.named_parameters():
            assert p.grad is not None
            before = p.grad.detach().clone()
            if not bool(torch.isfinite(before).all()):
                raise FloatingPointError("Nonfinite parameter gradient")
            after = policy.apply(before, name, "gradient", step)
            original_gradients.append(before.flatten())
            rounded_gradients.append(after.flatten())
            p.grad.copy_(after)
        original = torch.cat(original_gradients).double()
        rounded = torch.cat(rounded_gradients).double()
        should_measure = (step + 1) % config.eval_every == 0 or step + 1 == config.steps
        gradient_metrics = (
            gradient_comparison(model, target, x, original, rounded)
            if should_measure
            else {}
        )
        optimizer.step()
        for event in policy.events:
            append_json(output / "formats.jsonl", event)
        policy.events.clear()
        if should_measure:
            measure(
                step + 1,
                {
                    "training_loss": float(loss.detach()),
                    **gradient_metrics,
                },
            )
    parameters = rounded_parameters(model, policy)
    with torch.no_grad():
        predictions: Tensor = functional_call(
            model, parameters, (validation,), strict=True
        )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "format_choices": policy.choices
            if mode in ("calibrated", "adaptive")
            else {},
            "rounded_parameters": parameters,
            "validation_predictions": predictions,
            "mode": mode,
            "steps": config.steps,
        },
        output / "final.pt",
    )
    return rows


def run_experiment(config: RunConfig, output: Path) -> Path:
    config.validate()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    write_json(output / "config.json", asdict(config))
    record_source(output)
    validation = validation_inputs()
    probes = sample_inputs(9_000_000, 512)
    torch.save({"validation": validation, "probes": probes}, output / "inputs.pt")
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    model_spec = make_model(config.seeds[0])
    try:
        for target in config.targets:
            for seed in config.seeds:
                group = output / target / f"seed-{seed}"
                group.mkdir(parents=True)
                try:
                    base, optimizer, choices, calibration_seconds = warmup(
                        config, target, seed
                    )
                except FloatingPointError as error:
                    write_json(group / "failure.json", {"message": str(error)})
                    for mode in config.modes:
                        failure = {
                            "target": target,
                            "seed": seed,
                            "mode": mode,
                            "stage": "warmup",
                            "status": "skipped",
                            "message": str(error),
                        }
                        failures.append(failure)
                        append_json(output / "progress.jsonl", failure)
                    print(f"Warmup failed: {target}/{seed}: {error}", flush=True)
                    continue
                torch.save(
                    {"model": base.state_dict(), "optimizer": optimizer.state_dict()},
                    group / "initial.pt",
                )
                write_json(group / "initial-formats.json", choices)
                write_json(group / "calibration.json", {"seconds": calibration_seconds})
                for mode in config.modes:
                    print(f"{target} seed={seed} mode={mode}", flush=True)
                    append_json(
                        output / "progress.jsonl",
                        {
                            "target": target,
                            "seed": seed,
                            "mode": mode,
                            "status": "started",
                        },
                    )
                    try:
                        rows = run_condition(
                            config,
                            target,
                            seed,
                            mode,
                            base,
                            optimizer,
                            choices,
                            calibration_seconds,
                            validation,
                            probes,
                            group / mode,
                        )
                    except FloatingPointError as error:
                        failure = {
                            "target": target,
                            "seed": seed,
                            "mode": mode,
                            "stage": "training",
                            "status": "failed",
                            "message": str(error),
                        }
                        failures.append(failure)
                        write_json(group / mode / "failure.json", failure)
                        append_json(output / "progress.jsonl", failure)
                        partial = group / mode / "metrics.jsonl"
                        if partial.is_file():
                            records.extend(
                                json.loads(line)
                                for line in partial.read_text().splitlines()
                            )
                        print(
                            f"Condition failed: {target}/{seed}/{mode}: {error}",
                            flush=True,
                        )
                        continue
                    records.extend(rows)
                    append_json(
                        output / "progress.jsonl",
                        {
                            "target": target,
                            "seed": seed,
                            "mode": mode,
                            "status": "completed",
                        },
                    )
        fields = sorted({key for row in records for key in row})
        with (output / "metrics.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        parameters = sum(p.numel() for p in model_spec.parameters())
        arrays = sum(1 for _ in model_spec.parameters())
        write_json(
            output / "summary.json",
            {
                "status": "completed_with_failures" if failures else "completed",
                "failures": failures,
                "parameter_count": parameters,
                "representation_budget": {
                    "scope": "one set of weight and parameter-gradient arrays",
                    "payload_bits": 2 * parameters * 8,
                    "scale_bits": 2 * arrays * 32,
                    "format_identifier_bits": 2 * arrays * 2,
                    "excludes": (
                        "FP32 master weights, optimizer, activations, "
                        "and actual simulation memory"
                    ),
                },
                "final_metrics": [
                    row for row in records if row["step"] == config.steps
                ],
                "claim": (
                    "Execution evidence only; "
                    "no adaptive-format benefit is established."
                ),
            },
        )
        from afloat.plotting import plot_run

        if records:
            plot_run(output)
        if failures:
            write_json(output / "failure.json", {"conditions": failures})
        record_manifest(output)
    except Exception as error:
        write_json(
            output / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise
    if failures:
        raise RuntimeError(
            f"{len(failures)} conditions failed or were skipped; see {output}"
        )
    return output

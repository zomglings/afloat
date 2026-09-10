"""Serial CPU timings for the same training updates and cached-weight inference."""

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import torch
from torch.func import functional_call
from torch.utils.benchmark import Timer

from afloat.formats import PROTOTYPE_MODES as MODES
from afloat.formats import FormatPolicy
from afloat.model import make_model
from afloat.runner import evaluate, rounded_parameters, training_batch
from afloat.targets import TARGETS, sample_inputs, target_values


def write_json(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def fresh_model(seed, saved):
    model = make_model(seed)
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    return model, optimizer


def train_block(model, optimizer, policy, target, seed, steps):
    for step in range(steps):
        x = training_batch(seed, 50 + step, 128)
        optimizer.zero_grad(set_to_none=True)
        parameters = {}
        for name, parameter in model.named_parameters():
            q = policy.apply(parameter, name, "weight", step)
            parameters[name] = parameter + (q - parameter).detach()
        prediction = functional_call(model, parameters, (x,), strict=True)
        loss = (prediction - target_values(target, x)).square().mean()
        torch.autograd.backward(loss)
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None
            parameter.grad.copy_(policy.apply(parameter.grad, name, "gradient", step))
        optimizer.step()
        policy.events.clear()


def timed_call(function):
    for _ in range(5):
        function()
    timing = Timer(
        stmt="function()", globals={"function": function}, num_threads=1
    ).blocked_autorange(min_run_time=0.1)
    assert math.isclose(
        timing.median, statistics.median(timing.raw_times) / timing.number_per_run
    )
    return {
        "median_seconds": timing.median,
        "iqr_seconds": timing.iqr,
        "number_per_run": timing.number_per_run,
        "raw_block_seconds": timing.raw_times,
        "measured_calls": len(timing.raw_times) * timing.number_per_run,
    }


def inference_call(model, parameters, x):
    with torch.inference_mode():
        return functional_call(model, parameters, (x,), strict=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    order = [
        (target, seed, mode)
        for target in TARGETS
        for seed in range(3)
        for mode in MODES
    ]
    random.Random(371).shuffle(order)
    hardware = {"architecture": platform.machine(), "platform": platform.platform()}
    if platform.system() == "Darwin":
        for key in (
            "machdep.cpu.brand_string",
            "hw.physicalcpu",
            "hw.logicalcpu",
            "hw.memsize",
        ):
            hardware[key] = subprocess.check_output(
                ["sysctl", "-n", key], text=True
            ).strip()
    protocol = {
        "started_at": datetime.now(UTC).isoformat(),
        "hardware": hardware,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": "cpu",
        "threads": 1,
        "processes": 1,
        "training_steps": 300,
        "training_batch_size": 128,
        "inference_batch_sizes": [1, 128],
        "inference_min_measurement_seconds": 0.1,
        "order_seed": 371,
        "execution_order": order,
        "training_scope": (
            "Fresh shared checkpoint; dataset generation, forward, backward, "
            "gradient rounding, Adam, and existing per-array quantizer statistics. "
            "Excludes setup, validation probes, file writes, and plots."
        ),
        "inference_scope": (
            "Frozen final representations cached as FP32 tensors; eager FP32 "
            "arithmetic via functional_call in inference_mode. No conversion "
            "or format selection inside cached inference."
        ),
        "preparation_scope": (
            "Build represented weight arrays from master weights using final "
            "frozen format choices; measured separately."
        ),
        "limitations": (
            "Simulator timing on one machine. No packed storage, GPU, "
            "native FP8 compute, or custom hardware acceleration."
        ),
        "analysis_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    write_json(args.output / "protocol.json", protocol)
    validation = torch.load(args.study / "seed-0/inputs.pt", weights_only=True)[
        "validation"
    ]
    checkpoints = {}
    references = {}
    for seed in range(3):
        root = args.study / f"seed-{seed}"
        with (root / "metrics.csv").open() as stream:
            for row in csv.DictReader(stream):
                if int(row["step"]) == 300:
                    references[row["target"], seed, row["mode"]] = float(
                        row["validation_mse"]
                    )
        for target in TARGETS:
            group = root / target / f"seed-{seed}"
            checkpoints[target, seed] = (
                torch.load(group / "initial.pt", weights_only=True),
                json.loads((group / "initial-formats.json").read_text()),
            )
    started = perf_counter()
    records = []
    for index, (target, seed, mode) in enumerate(order):
        saved, choices = checkpoints[target, seed]
        warm_model, warm_optimizer = fresh_model(seed, saved)
        warm_policy = FormatPolicy(mode, choices, 100, 2048)
        train_block(warm_model, warm_optimizer, warm_policy, target, seed, 5)
        model, optimizer = fresh_model(seed, saved)
        policy = FormatPolicy(mode, choices, 100, 2048)
        began = perf_counter()
        train_block(model, optimizer, policy, target, seed, 300)
        seconds = perf_counter() - began
        measured, _ = evaluate(model, policy, target, validation)
        assert math.isclose(
            measured["validation_mse"],
            references[target, seed, mode],
            rel_tol=1e-7,
            abs_tol=1e-10,
        ), (target, seed, mode)
        row = {
            "target": target,
            "seed": seed,
            "mode": mode,
            "training_seconds": seconds,
            "training_ms_per_step": seconds * 1000 / 300,
            "training_samples_per_second": 300 * 128 / seconds,
            "training_reselection_seconds": policy.selection_seconds,
            "training_reference_validation_mse": measured["validation_mse"],
        }
        final = torch.load(
            args.study / f"seed-{seed}" / target / f"seed-{seed}" / mode / "final.pt",
            weights_only=True,
        )
        model.load_state_dict(final["model"])
        policy = FormatPolicy(mode, final["format_choices"], 100, 2048)
        prepared = rounded_parameters(model, policy)
        for name, values in prepared.items():
            assert torch.equal(values, final["rounded_parameters"][name])
        for batch_size in (1, 128):
            x = sample_inputs(777, batch_size)
            timing = timed_call(partial(inference_call, model, prepared, x))
            row[f"inference_batch_{batch_size}"] = timing
            row[f"inference_batch_{batch_size}_milliseconds"] = (
                timing["median_seconds"] * 1000
            )
            row[f"inference_batch_{batch_size}_samples_per_second"] = (
                batch_size / timing["median_seconds"]
            )
        preparation = timed_call(partial(rounded_parameters, model, policy))
        row["preparation"] = preparation
        row["preparation_milliseconds"] = preparation["median_seconds"] * 1000
        records.append(row)
        with (args.output / "measurements.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if (index + 1) % 12 == 0:
            print(f"Benchmarked {index + 1}/108 conditions", flush=True)
    summary = []
    keys = [key for key in records[0] if key.endswith(("seconds", "step", "second"))]
    for mode in MODES:
        rows = [row for row in records if row["mode"] == mode]
        result = {"mode": mode, "conditions": len(rows)}
        for key in keys:
            values = [row[key] for row in rows]
            result[key] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        summary.append(result)
    write_json(
        args.output / "summary.json",
        {
            "wall_seconds": perf_counter() - started,
            "conditions": len(records),
            "all_step_300_validation_losses_matched": True,
            "all_cached_representations_verified": True,
            "modes": summary,
        },
    )
    print(f"Performance measurements saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()

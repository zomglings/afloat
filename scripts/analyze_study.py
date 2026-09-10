"""Verify complete study artifacts and derive descriptive, paired comparisons."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from afloat.formats import PROTOTYPE_MODES as MODES
from afloat.targets import TARGETS, feature_metrics, target_values


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=sorted({k for row in rows for k in row}),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def verify_runs(study):
    completion = read_json(study / "completion.json")
    assert completion["outcomes"] == [
        {"seed": seed, "exit_code": 0} for seed in range(3)
    ]
    expected = {
        "targets": list(TARGETS),
        "modes": list(MODES),
        "steps": 2000,
        "warmup_steps": 50,
        "batch_size": 128,
        "learning_rate": 0.001,
        "eval_every": 100,
        "adapt_every": 100,
        "sample_size": 2048,
        "threads": 1,
    }
    environments, finals, paths = [], [], []
    metrics = []
    for seed in range(3):
        root = study / f"seed-{seed}"
        config = read_json(root / "config.json")
        for name, default in (
            ("activation_gain", 1.0),
            ("final_learning_rate", None),
            ("input_features", "raw"),
            ("format_log_every", 1),
            ("save_checkpoints", False),
        ):
            assert config.pop(name, default) == default
        assert config == {**expected, "seeds": [seed]}
        env = read_json(root / "environment.json")
        assert (
            env["git_revision"] == read_json(study / "protocol.json")["source_revision"]
        )
        assert env["git_dirty"] is False
        environments.append(env)
        manifest = read_json(root / "manifest.json")
        for name, checksum in manifest.items():
            assert digest(root / name) == checksum, root / name
        summary = read_json(root / "summary.json")
        assert summary["status"] == "completed" and not summary["failures"]
        assert len(summary["final_metrics"]) == 36
        finals.extend(summary["final_metrics"])
        progress = [
            json.loads(line)
            for line in (root / "progress.jsonl").read_text().splitlines()
        ]
        assert Counter(row["status"] for row in progress) == {
            "started": 36,
            "completed": 36,
        }
        with (root / "metrics.csv").open() as stream:
            metrics.extend(
                dict(row, seed=int(row["seed"]), step=int(row["step"]))
                for row in csv.DictReader(stream)
            )
        paths.extend(
            (seed, target, mode, root / target / f"seed-{seed}" / mode)
            for target in TARGETS
            for mode in MODES
        )
    assert len(finals) == 108 and len(metrics) == 2268
    assert len({(row["target"], row["mode"], row["seed"]) for row in finals}) == 108
    for env in environments[1:]:
        assert env == environments[0]
    return completion, environments[0], finals, metrics, paths


def checkpoint_features(study, finals, paths):
    lookup = {(row["target"], row["mode"], row["seed"]): row for row in finals}
    inputs = torch.load(study / "seed-0/inputs.pt", weights_only=True)["validation"]
    prediction_groups = defaultdict(list)
    extra = []
    for seed, target, mode, directory in paths:
        saved = torch.load(directory / "final.pt", weights_only=True)
        assert saved["steps"] == 2000 and saved["mode"] == mode
        assert sum(value.numel() for value in saved["model"].values()) == 9585
        prediction = saved["validation_predictions"]
        assert prediction.shape == (4225, 1) and bool(torch.isfinite(prediction).all())
        measured = feature_metrics(target, inputs, prediction)
        recorded = lookup[target, mode, seed]
        for name, value in measured.items():
            assert math.isclose(value, recorded[name], rel_tol=1e-7, abs_tol=1e-10)
        truth = target_values(target, inputs)
        variance = float((truth - truth.mean()).square().mean())
        row = {
            "target": target,
            "mode": mode,
            "seed": seed,
            "target_variance": variance,
            "relative_to_constant_mse": measured["validation_mse"] / variance,
        }
        if target == "detail":
            predicted_grid = prediction.reshape(65, 65).double()
            true_grid = truth.reshape(65, 65).double()
            predicted_detail = predicted_grid - predicted_grid.mean(dim=1, keepdim=True)
            true_detail = true_grid - true_grid.mean(dim=1, keepdim=True)
            row["detail_component_relative_mse"] = float(
                (predicted_detail - true_detail).square().mean()
                / true_detail.square().mean()
            )
        extra.append(row)
        prediction_groups[target, mode].append(prediction)
    return inputs, extra, prediction_groups


def summaries(finals, extra):
    grouped = defaultdict(list)
    for row in finals:
        grouped[row["target"], row["mode"]].append(row)
    all_stats, paired = [], []
    for target in TARGETS:
        adaptive = {row["seed"]: row for row in grouped[target, "adaptive"]}
        for mode in MODES:
            rows = grouped[target, mode]
            stats = {"target": target, "mode": mode}
            fields = set.intersection(*(set(row) for row in rows)) - {
                "target",
                "mode",
                "seed",
                "step",
            }
            for key in sorted(fields):
                if all(row[key] is not None for row in rows):
                    stats[key] = describe([row[key] for row in rows])
            extras = [
                row for row in extra if (row["target"], row["mode"]) == (target, mode)
            ]
            for key in ("relative_to_constant_mse", "detail_component_relative_mse"):
                if key in extras[0]:
                    stats[key] = describe([row[key] for row in extras])
            all_stats.append(stats)
            if mode != "adaptive":
                controls = {row["seed"]: row for row in rows}
                ratios = [
                    adaptive[seed]["validation_mse"] / controls[seed]["validation_mse"]
                    for seed in range(3)
                ]
                differences = [
                    adaptive[seed]["validation_mse"] - controls[seed]["validation_mse"]
                    for seed in range(3)
                ]
                paired.append(
                    {
                        "target": target,
                        "control": mode,
                        "ratio_of_means": statistics.mean(
                            [row["validation_mse"] for row in adaptive.values()]
                        )
                        / statistics.mean([row["validation_mse"] for row in rows]),
                        "paired_ratios": ratios,
                        "paired_differences": differences,
                        "adaptive_wins": sum(diff < 0 for diff in differences),
                        "ties": sum(diff == 0 for diff in differences),
                    }
                )
    return all_stats, paired


def format_measurements(paths):
    result, changes = [], []
    for seed, target, mode, directory in paths:
        if mode == "fp32":
            continue
        groups = defaultdict(
            lambda: {
                "events": 0,
                "elements": 0,
                "zeroed": 0,
                "clipped": 0,
                "squared_error_sum": 0.0,
                "scale_min": math.inf,
                "scale_max": 0.0,
                "switches": 0,
                "opportunities": 0,
                "format_elements": Counter(),
                "first": Counter(),
                "last": Counter(),
                "changing_arrays": set(),
            }
        )
        event_count = 0
        with (directory / "formats.jsonl").open() as stream:
            for line in stream:
                event = json.loads(line)
                kind = event["array"].split(":", 1)[0]
                group = groups[kind]
                event_count += 1
                group["events"] += 1
                group["elements"] += event["elements"]
                group["zeroed"] += event["zeroed"]
                group["clipped"] += event["clipped"]
                group["squared_error_sum"] += event["mse"] * event["elements"]
                group["scale_min"] = min(group["scale_min"], event["scale"])
                group["scale_max"] = max(group["scale_max"], event["scale"])
                group["format_elements"][event["format"]] += event["elements"]
                if event["step"] == 0:
                    group["first"][event["format"]] += 1
                if event["step"] == 1999:
                    group["last"][event["format"]] += 1
                if (
                    mode == "adaptive"
                    and event["step"] > 0
                    and event["step"] % 100 == 0
                ):
                    group["opportunities"] += 1
                if event["changed"]:
                    assert mode == "adaptive"
                    group["switches"] += 1
                    group["changing_arrays"].add(event["array"])
                    changes.append(
                        {"target": target, "mode": mode, "seed": seed, **event}
                    )
        assert event_count == 64000
        for kind, group in groups.items():
            row = {
                "target": target,
                "mode": mode,
                "seed": seed,
                "kind": kind,
                "quantized_elements": group["elements"],
                "mean_squared_rounding_error": group["squared_error_sum"]
                / group["elements"],
                "zeroed_fraction": group["zeroed"] / group["elements"],
                "clipped_count": group["clipped"],
                "format_switches": group["switches"],
                "reselection_opportunities": group["opportunities"],
                "arrays_that_changed": len(group["changing_arrays"]),
                "scale_min": group["scale_min"],
                "scale_max": group["scale_max"],
            }
            for name in ("e3m4", "e4m3", "e5m2"):
                row[f"{name}_element_fraction"] = (
                    group["format_elements"][name] / group["elements"]
                )
                row[f"{name}_initial_arrays"] = group["first"][name]
                row[f"{name}_final_arrays"] = group["last"][name]
            result.append(row)
    return result, changes


def activation_measurements(paths):
    groups = defaultdict(list)
    for seed, target, mode, directory in paths:
        rows = [
            json.loads(line)
            for line in (directory / "activations.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 189
        for row in rows:
            if row["step"] in (0, 2000):
                groups[target, mode, seed, row["step"], row["family"]].append(row)
    results = []
    for (target, mode, seed, step, family), rows in groups.items():
        assert len(rows) == 3
        result = {
            "target": target,
            "mode": mode,
            "seed": seed,
            "step": step,
            "family": family,
        }
        for key in (
            "negative_fraction",
            "near_zero_fraction",
            "small_slope_fraction",
            "mean_abs_slope",
            "nonlinear_output_mse",
            "input_p01",
            "input_p50",
            "input_p99",
        ):
            result[key] = statistics.mean(row[key] for row in rows)
        result["input_abs_max"] = max(row["input_abs_max"] for row in rows)
        if family == "relu":
            result["relu_gate_flip_fraction"] = statistics.mean(
                row["relu_gate_flip_fraction"] for row in rows
            )
        results.append(result)
    return results


def plots(output, metrics, pairs, predictions, inputs):
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), layout="constrained")
    for ax, target in zip(axes.flat, TARGETS, strict=True):
        for index, mode in enumerate(MODES):
            rows = [
                row
                for row in metrics
                if row["target"] == target and row["mode"] == mode
            ]
            steps = sorted({row["step"] for row in rows})
            means = [
                statistics.mean(
                    float(row["validation_mse"]) for row in rows if row["step"] == step
                )
                for step in steps
            ]
            lows = [
                min(float(row["validation_mse"]) for row in rows if row["step"] == step)
                for step in steps
            ]
            highs = [
                max(float(row["validation_mse"]) for row in rows if row["step"] == step)
                for step in steps
            ]
            ax.plot(steps, means, label=mode, color=f"C{index}")
            ax.fill_between(steps, lows, highs, color=f"C{index}", alpha=0.08)
        ax.set(
            title=target,
            xlabel="Steps after shared warmup",
            ylabel="Validation MSE",
            yscale="log",
        )
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("afloat: mean across three seeds; shading shows seed range")
    fig.savefig(output / "learning-curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for i, target in enumerate(TARGETS):
        pair = next(
            row
            for row in pairs
            if row["target"] == target and row["control"] == "calibrated"
        )
        ax.scatter([i - 0.12, i, i + 0.12], pair["paired_ratios"], color="C0", s=55)
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(6), TARGETS)
    ax.set(
        ylabel="Adaptive MSE / calibrated MSE (each paired seed)",
        yscale="log",
        title="Does repeated format selection beat choosing once?",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "paired-comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(6, 4, figsize=(12, 16), layout="constrained")
    for i, target in enumerate(TARGETS):
        truth = target_values(target, inputs).reshape(65, 65).numpy()
        values = [truth] + [
            torch.stack(predictions[target, mode]).mean(0).reshape(65, 65).numpy()
            for mode in ("fp32", "calibrated", "adaptive")
        ]
        for j, value in enumerate(values):
            axes[i, j].imshow(
                value.T,
                origin="lower",
                extent=(-1, 1, -1, 1),
                vmin=truth.min(),
                vmax=truth.max(),
                cmap="viridis",
            )
            axes[i, j].set_title(
                f"{target}: {('target', 'fp32', 'calibrated', 'adaptive')[j]}",
                fontsize=9,
            )
            axes[i, j].set(xlabel="x1", ylabel="x2")
    fig.suptitle("Target functions and mean predictions across three seeds")
    fig.savefig(output / "function-surfaces.png", dpi=140)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    completion, environment, finals, metrics, paths = verify_runs(args.study)
    print(
        "Verified all manifests, source identities, and 108 completed conditions.",
        flush=True,
    )
    inputs, extra, predictions = checkpoint_features(args.study, finals, paths)
    stats, pairs = summaries(finals, extra)
    formats, changes = format_measurements(paths)
    activations = activation_measurements(paths)
    write_json(
        args.output / "verification.json",
        {
            "completed_conditions": len(finals),
            "verified_checkpoints": len(paths),
            "format_records": 90 * 64000,
            "activation_records": 108 * 189,
            "metric_records": len(metrics),
            "completion": completion,
            "environment": environment,
            "manifest_hashes": {
                f"seed-{seed}": digest(args.study / f"seed-{seed}/manifest.json")
                for seed in range(3)
            },
            "analysis_source_sha256": digest(Path(__file__)),
        },
    )
    write_csv(args.output / "metrics.csv", metrics)
    write_csv(
        args.output / "final-metrics.csv",
        [
            {
                **row,
                **next(
                    extra_row
                    for extra_row in extra
                    if all(
                        extra_row[key] == row[key] for key in ("target", "mode", "seed")
                    )
                ),
            }
            for row in finals
        ],
    )
    write_json(args.output / "statistics.json", stats)
    write_json(args.output / "paired-comparisons.json", pairs)
    write_csv(args.output / "formats.csv", formats)
    write_csv(args.output / "format-changes.csv", changes)
    write_csv(args.output / "activations.csv", activations)
    plots(args.output, metrics, pairs, predictions, inputs)
    print(f"Analysis written to {args.output}", flush=True)


if __name__ == "__main__":
    main()

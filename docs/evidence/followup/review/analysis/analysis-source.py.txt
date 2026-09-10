"""Verify the frozen follow-up and summarize every condition and paired control."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import matplotlib
import torch
from benchmark_followup import config_for, passes
from torch.func import functional_call

from afloat.artifacts import record_manifest, sha256, write_json
from afloat.formats import FORMATS, FormatPolicy, format_scores
from afloat.model import make_model
from afloat.runner import rounded_parameters, training_batch
from afloat.targets import detail_component_error, feature_metrics, target_values

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in rows for k in r}))
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


def threshold_steps(rows, threshold):
    streak, first, confirmation = 0, None, None
    for row in rows:
        streak = streak + 1 if passes(row, threshold) else 0
        if streak == threshold["consecutive_measurements"] and first is None:
            first = row["step"] - (streak - 1) * 500
            confirmation = row["step"]
    return {
        "first_crossing_step": first,
        "confirmation_step": confirmation,
        "ever_sustained": first is not None,
        "final_three_pass": all(passes(r, threshold) for r in rows[-3:]),
    }


def initial_scores(root, gain, seed, plan):
    saved = torch.load(
        root / "detail" / f"seed-{seed}" / "initial.pt", weights_only=True
    )
    model = make_model(seed, gain, plan["input_features"])
    model.load_state_dict(saved["model"])
    x = training_batch(seed, plan["warmup_steps"], plan["batch_size"])
    (model(x) - target_values("detail", x)).square().mean().backward()
    choices = read_json(root / "detail" / f"seed-{seed}" / "initial-formats.json")
    rows = []
    for name, parameter in model.named_parameters():
        for role, values in (("weight", parameter), ("gradient", parameter.grad)):
            scores = format_scores(values, plan["sample_size"])
            ordered = sorted(FORMATS, key=lambda label: scores[label])
            best, second = ordered[:2]
            assert best == choices[f"{role}:{name}"]
            rows.append(
                {
                    "gain": gain,
                    "seed": seed,
                    "array": f"{role}:{name}",
                    "elements": values.numel(),
                    "format": best,
                    "candidate_mse": scores,
                    "margin": (scores[second] - scores[best]) / scores[second]
                    if scores[second]
                    else 0.0,
                }
            )
    return rows


def inspect_formats(directory, identity, plan, initial):
    groups = defaultdict(
        lambda: {
            "records": 0,
            "elements": 0,
            "format_elements": Counter(),
            "zeroed": 0,
            "clipped": 0,
            "squared_error_sum": 0.0,
        }
    )
    counts = Counter()
    switches, preferences, candidates = [], [], []
    last_strict = {
        r["array"]: {
            "step": 0,
            "format": r["format"],
            "margin": r["margin"],
            "phase": "initial calibration",
        }
        for r in initial
        if r["elements"] >= plan["substantial_array"]["minimum_elements"]
        and r["margin"] >= plan["substantial_array"]["minimum_relative_score_margin"]
    }
    for event in read_rows(directory / "formats.jsonl"):
        counts["records"] += 1
        key = event["array"]
        role = key.split(":")[0]
        group = groups[role]
        group["records"] += 1
        group["elements"] += event["elements"]
        group["format_elements"][event["format"]] += event["elements"]
        group["zeroed"] += event["zeroed"]
        group["clipped"] += event["clipped"]
        group["squared_error_sum"] += event["mse"] * event["elements"]
        if event["changed"]:
            counts["switches"] += 1
            counts["switches_with_value_changes"] += event["changed_elements"] > 0
            counts["substantial_switches"] += event["elements"] >= 32
            switches.append({**identity, **event})
        scores = event["candidate_mse"]
        if scores is None:
            continue
        counts["candidate_records"] += 1
        ordered = sorted(FORMATS, key=lambda name: scores[name])
        best, second = ordered[:2]
        assert event["format"] == best
        assert event["mse"] == scores[best]
        margin = (
            (scores[second] - scores[best]) / scores[second] if scores[second] else 0
        )
        strict = margin >= plan["substantial_array"]["minimum_relative_score_margin"]
        candidates.append(
            {
                **identity,
                "array": key,
                "step": event["step"],
                "elements": event["elements"],
                "format": best,
                "margin": margin,
                **{f"mse_{label}": value for label, value in scores.items()},
            }
        )
        if event["elements"] >= plan["substantial_array"]["minimum_elements"]:
            counts[f"substantial_winner_{best}"] += 1
            counts["substantial_strict_observations"] += strict
            if strict:
                if key in last_strict and last_strict[key]["format"] != best:
                    counts["substantial_strict_preference_changes"] += 1
                    preferences.append(
                        {
                            **identity,
                            "array": key,
                            "previous": last_strict[key],
                            "current": event,
                            "current_margin": margin,
                        }
                    )
                last_strict[key] = {
                    "step": event["step"],
                    "format": best,
                    "margin": margin,
                }
    rows = []
    for role, group in groups.items():
        rows.append(
            {
                **identity,
                "role": role,
                **{k: v for k, v in group.items() if k != "format_elements"},
                "mean_mse": group["squared_error_sum"] / group["elements"],
                "zeroed_fraction": group["zeroed"] / group["elements"],
                "clipped_fraction": group["clipped"] / group["elements"],
                **{
                    f"{name}_element_share": group["format_elements"][name]
                    / group["elements"]
                    for name in FORMATS
                },
            }
        )
    return rows, {**identity, **counts}, switches, preferences, candidates


def verify_and_collect(study, plan):
    protocol = read_json(study / "protocol.json")
    completion = read_json(study / "completion.json")
    expected_keys = {
        (gain, seed) for gain in plan["activation_gains"] for seed in plan["seeds"]
    }
    assert {r["name"] for r in completion["outcomes"]} == {
        f"gain-{gain:g}-seed-{seed}" for gain, seed in expected_keys
    }
    assert len(completion["outcomes"]) == len(expected_keys)
    assert all(r["exit_code"] == 0 for r in completion["outcomes"])
    metrics, finals, activations, formats, counts, switches, preferences = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    initial_candidates, candidate_scores = [], []
    environment = None
    verified_files = 0
    for gain, seed in sorted(expected_keys):
        root = study / f"gain-{gain:g}-seed-{seed}"
        config = read_json(root / "config.json")
        expected = json.loads(json.dumps(asdict(config_for(plan, gain, seed))))
        assert set(config["modes"]) == set(expected["modes"])
        assert len(config["modes"]) == len(expected["modes"])
        expected["modes"] = config["modes"]
        assert config == expected
        env = read_json(root / "environment.json")
        assert env["git_revision"] == protocol["source_revision"]
        assert env["git_dirty"] is False
        if environment is None:
            environment = env
        assert env == environment
        for name, checksum in read_json(root / "manifest.json").items():
            assert sha256(root / name) == checksum, root / name
            verified_files += 1
        summary = read_json(root / "summary.json")
        assert summary["status"] == "completed" and summary["failures"] == []
        assert summary["parameter_count"] == 10097
        assert Counter(r["status"] for r in read_rows(root / "progress.jsonl")) == {
            "started": 7,
            "completed": 7,
        }
        inputs = torch.load(root / "inputs.pt", weights_only=True)["validation"]
        initial = initial_scores(root, gain, seed, plan)
        initial_candidates.extend(initial)
        for mode in plan["modes"]:
            directory = root / "detail" / f"seed-{seed}" / mode
            identity = {"gain": gain, "seed": seed, "mode": mode}
            rows = read_rows(directory / "metrics.jsonl")
            assert [r["step"] for r in rows] == list(
                range(0, plan["steps"] + 1, plan["eval_every"])
            )
            assert len(list(directory.glob("step-*.pt"))) == len(rows)
            metrics.extend({**identity, **r} for r in rows)
            final = {**identity, **rows[-1], **threshold_steps(rows, plan["threshold"])}
            checkpoint = torch.load(directory / "final.pt", weights_only=True)
            model = make_model(seed, **checkpoint["model_settings"])
            model.load_state_dict(checkpoint["model"])
            assert sum(p.numel() for p in model.parameters()) == 10097
            assert checkpoint["steps"] == plan["steps"] and checkpoint["mode"] == mode
            policy = FormatPolicy(
                mode,
                checkpoint["format_choices"],
                plan["adapt_every"],
                plan["sample_size"],
            )
            params = rounded_parameters(model, policy)
            for name in params:
                assert torch.equal(params[name], checkpoint["rounded_parameters"][name])
            with torch.no_grad():
                prediction = functional_call(model, params, (inputs,), strict=True)
            assert torch.equal(prediction, checkpoint["validation_predictions"])
            measured = feature_metrics("detail", inputs, prediction)
            measured["detail_component_relative_mse"] = detail_component_error(
                prediction
            )
            assert all(math.isfinite(v) and v == final[k] for k, v in measured.items())
            finals.append(final)
            activation_rows = read_rows(directory / "activations.jsonl")
            assert len(activation_rows) == len(rows) * 9
            activations.extend({**identity, **r} for r in activation_rows)
            if mode != "fp32":
                f, c, s, p, scores = inspect_formats(directory, identity, plan, initial)
                formats.extend(f)
                counts.append(c)
                switches.extend(s)
                preferences.extend(p)
                candidate_scores.extend(scores)
    assert len(finals) == 70 and len(metrics) == 2870 and len(activations) == 25830
    return {
        "initial_candidates": initial_candidates,
        "candidate_scores": candidate_scores,
        "metrics": metrics,
        "finals": finals,
        "activations": activations,
        "formats": formats,
        "format_counts": counts,
        "switches": switches,
        "strict_preference_changes": preferences,
        "verification": {
            "manifest_files_verified": verified_files,
            "final_models_reconstructed_exactly": len(finals),
            "environment": environment,
            "completion": completion,
        },
    }


def summarize(data, plan):
    summaries, paired, activation_summary = [], [], []
    outcomes = (
        "validation_mse",
        "detail_component_relative_mse",
        "detail_amplitude",
        "total_gradient_cosine",
        "gradient_zeroed_fraction",
        "weight_rounding_output_mse",
        "training_seconds",
        "selection_seconds",
    )
    for gain in plan["activation_gains"]:
        adaptive = {
            r["seed"]: r
            for r in data["finals"]
            if r["gain"] == gain and r["mode"] == "adaptive"
        }
        for mode in plan["modes"]:
            rows = [
                r for r in data["finals"] if r["gain"] == gain and r["mode"] == mode
            ]
            summaries.append(
                {
                    "gain": gain,
                    "mode": mode,
                    "ever_sustained": sum(r["ever_sustained"] for r in rows),
                    "final_three_pass": sum(r["final_three_pass"] for r in rows),
                    **{key: describe([r[key] for r in rows]) for key in outcomes},
                }
            )
            if mode != "adaptive":
                for metric in outcomes[:2]:
                    differences = [
                        adaptive[r["seed"]][metric] - r[metric] for r in rows
                    ]
                    paired.append(
                        {
                            "gain": gain,
                            "control": mode,
                            "metric": metric,
                            "seeds": [r["seed"] for r in rows],
                            "ratios": [
                                adaptive[r["seed"]][metric] / r[metric] for r in rows
                            ],
                            "differences": differences,
                            "ratio_of_means": statistics.mean(
                                r[metric] for r in adaptive.values()
                            )
                            / statistics.mean(r[metric] for r in rows),
                            "adaptive_wins": sum(d < 0 for d in differences),
                            "ties": sum(d == 0 for d in differences),
                        }
                    )
            for step in (0, plan["steps"]):
                for family in ("relu", "tanh", "silu"):
                    acts = [
                        r
                        for r in data["activations"]
                        if r["gain"] == gain
                        and r["mode"] == mode
                        and r["step"] == step
                        and r["family"] == family
                    ]
                    keys = (
                        "input_abs_max",
                        "input_p01",
                        "input_p99",
                        "mean_abs_slope",
                        "small_slope_fraction",
                        "negative_fraction",
                        "nonlinear_output_mse",
                    )
                    activation_summary.append(
                        {
                            "gain": gain,
                            "mode": mode,
                            "step": step,
                            "family": family,
                            **{key: describe([r[key] for r in acts]) for key in keys},
                        }
                    )
    return summaries, paired, activation_summary


def plot_curves(output, metrics, plan):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for column, gain in enumerate(plan["activation_gains"]):
        for row, (metric, label) in enumerate(
            (
                ("validation_mse", "Validation mean squared error"),
                (
                    "detail_component_relative_mse",
                    "Small-component error / true magnitude",
                ),
            )
        ):
            ax = axes[row, column]
            for mode in plan["modes"]:
                groups = defaultdict(list)
                for r in metrics:
                    if r["gain"] == gain and r["mode"] == mode:
                        groups[r["step"]].append(r[metric])
                steps = sorted(groups)
                ax.plot(
                    steps,
                    [statistics.mean(groups[s]) for s in steps],
                    label=mode,
                    linewidth=1.5,
                )
            ax.set_yscale("log")
            ax.set_title(f"Activation gain {gain:g}")
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
            if row:
                ax.axhline(0.25, color="gray", linestyle=":", linewidth=1)
                ax.set_xlabel("Updates after shared warmup")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Five-seed means; original target with fixed sine/cosine input features"
    )
    fig.tight_layout()
    fig.savefig(output / "learning-curves.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = read_json(args.study / "protocol.json")["plan"]
    data = verify_and_collect(args.study, plan)
    summaries, paired, activation_summary = summarize(data, plan)
    for name in (
        "metrics",
        "finals",
        "activations",
        "formats",
        "format_counts",
        "candidate_scores",
    ):
        write_csv(args.output / f"{name}.csv", data[name])
    for name in (
        "switches",
        "strict_preference_changes",
        "verification",
        "initial_candidates",
    ):
        write_json(args.output / f"{name}.json", data[name])
    write_json(args.output / "summary.json", summaries)
    write_json(args.output / "paired.json", paired)
    write_json(args.output / "activation-summary.json", activation_summary)
    write_json(
        args.output / "provenance.json",
        {
            "source_protocol_sha256": sha256(args.study / "protocol.json"),
            "analysis_sha256": sha256(Path(__file__)),
            "benchmark_sha256": sha256(
                Path(__file__).with_name("benchmark_followup.py")
            ),
        },
    )
    (args.output / "analysis-source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "benchmark_followup.py").write_bytes(
        Path(__file__).with_name("benchmark_followup.py").read_bytes()
    )
    plot_curves(args.output, data["metrics"], plan)
    record_manifest(args.output)
    print(f"Verified 70 final models and wrote {args.output}")


if __name__ == "__main__":
    main()

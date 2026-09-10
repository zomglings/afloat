"""Check completed timing records and compute explicit seed-paired summaries."""

import argparse
import json
import statistics
from pathlib import Path

from afloat.artifacts import record_manifest, sha256, write_json


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def describe(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("performance", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, expected in json.loads(
        (args.performance / "manifest.json").read_text()
    ).items():
        assert sha256(args.performance / name) == expected
    cases = rows(args.performance / "threshold-results.jsonl")
    checks = rows(args.performance / "threshold-measurements.jsonl")
    inference = rows(args.performance / "inference.jsonl")
    modes = ("fp32", "fixed-e3m4", "calibrated", "adaptive")
    expected = {
        (gain, seed, mode) for gain in (1, 4) for seed in (10, 11, 12) for mode in modes
    }
    assert (
        len(cases) == 24
        and {(r["gain"], r["seed"], r["mode"]) for r in cases} == expected
    )
    assert all(r["status"] == "completed" and r["reached"] for r in cases)
    assert len(inference) == 70
    assert len({(r["gain"], r["seed"], r["mode"]) for r in inference}) == 70
    assert all(r["model_optimizer_choices_verified"] for r in checks)
    for case in cases:
        points = [
            r for r in checks if all(r[k] == case[k] for k in ("gain", "seed", "mode"))
        ]
        assert [r["step"] for r in points] == list(
            range(0, case["final_step"] + 1, 500)
        )
        assert all(r["passes"] for r in points[-3:])
        assert points[-3]["step"] == case["crossing"]["step"]
        assert points[-3]["measured_seconds"] == case["crossing"]["measured_seconds"]
        assert points[-1]["measured_seconds"] == case["measured_seconds"]
        assert case["measured_seconds"] == sum(
            case[k]
            for k in (
                "warmup_seconds",
                "initial_calibration_seconds",
                "training_seconds",
                "validation_seconds",
            )
        )
    summaries, paired = [], []
    for gain in (1, 4):
        for mode in modes:
            group = [r for r in cases if r["gain"] == gain and r["mode"] == mode]
            summaries.append(
                {
                    "gain": gain,
                    "mode": mode,
                    "reached": len(group),
                    "confirmation_seconds": describe(
                        [r["measured_seconds"] for r in group]
                    ),
                    "first_crossing_seconds": describe(
                        [r["crossing"]["measured_seconds"] for r in group]
                    ),
                    "confirmation_updates": describe([r["final_step"] for r in group]),
                    "first_crossing_updates": describe(
                        [r["crossing"]["step"] for r in group]
                    ),
                    "milliseconds_per_update": describe(
                        [1000 * r["training_seconds"] / r["final_step"] for r in group]
                    ),
                    "warmup_seconds": describe([r["warmup_seconds"] for r in group]),
                    "initial_calibration_seconds": describe(
                        [r["initial_calibration_seconds"] for r in group]
                    ),
                    "reselection_seconds": describe(
                        [r["reselection_seconds"] for r in group]
                    ),
                }
            )
            if mode == "adaptive":
                continue
            adaptive = {
                r["seed"]: r
                for r in cases
                if r["gain"] == gain and r["mode"] == "adaptive"
            }
            ratios = [
                adaptive[r["seed"]]["measured_seconds"] / r["measured_seconds"]
                for r in group
            ]
            paired.append(
                {
                    "gain": gain,
                    "control": mode,
                    "seeds": [r["seed"] for r in group],
                    "paired_confirmation_time_ratios": ratios,
                    "adaptive_faster": sum(v < 1 for v in ratios),
                    "ratio_of_medians": statistics.median(
                        r["measured_seconds"] for r in adaptive.values()
                    )
                    / statistics.median(r["measured_seconds"] for r in group),
                }
            )
    write_json(args.output / "summary.json", summaries)
    write_json(args.output / "paired.json", paired)
    write_json(
        args.output / "verification.json",
        {
            "threshold_runs": len(cases),
            "verified_measurements": len(checks),
            "inference_models": len(inference),
            "source_protocol_sha256": sha256(args.performance / "protocol.json"),
            "script_sha256": sha256(Path(__file__)),
        },
    )
    (args.output / "summary-source.py").write_bytes(Path(__file__).read_bytes())
    record_manifest(args.output)
    print(f"Verified {len(checks)} timing measurements and wrote {args.output}")


if __name__ == "__main__":
    main()

"""Verify every development attempt, preserving unsuccessful settings."""

import argparse
import json
from pathlib import Path

import torch

from afloat.artifacts import record_manifest, sha256, write_json
from afloat.formats import FORMATS

GROUPS = (
    "development-100",
    "fourier-100",
    "fourier-101",
    "gain4-101",
    "budget-100",
    "budget-101",
)


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def passes(row):
    return (
        row["validation_mse"] < 0.0001
        and row["detail_component_relative_mse"] < 0.25
        and 0.008 <= row["detail_amplitude"] <= 0.012
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    verified = 0
    for name in GROUPS:
        root = args.work / name
        plan = read_json(root / "protocol.json")
        assert read_json(root / "completion.json")["status"] == "completed"
        for filename, checksum in read_json(root / "manifest.json").items():
            assert sha256(root / filename) == checksum, root / filename
            verified += 1
        for gain in plan["gains"]:
            group = root / f"gain-{gain:g}"
            rows = read_rows(group / "metrics.jsonl")
            assert [r["step"] for r in rows] == list(
                range(0, plan["steps"] + 1, plan["measurement_every"])
            )
            candidates = read_rows(group / "formats.jsonl")
            substantial = [r for r in candidates if r["elements"] >= 32]
            strict_changes = []
            last = {}
            winners = {}
            for r in substantial:
                scores = r["scores"]
                ordered = sorted(FORMATS, key=lambda f: scores[f])
                best, second = ordered[:2]
                assert best == r["choice"]
                winners[best] = winners.get(best, 0) + 1
                margin = (
                    (scores[second] - scores[best]) / scores[second]
                    if scores[second]
                    else 0
                )
                if margin >= 0.01:
                    if r["array"] in last and last[r["array"]]["choice"] != best:
                        strict_changes.append(
                            {"previous": last[r["array"]], "current": r}
                        )
                    last[r["array"]] = r
            results.append(
                {
                    "group": name,
                    "input_features": plan.get("input_features", "raw"),
                    "steps": plan["steps"],
                    "gain": gain,
                    "seed": plan["seed"],
                    "final": rows[-1],
                    "final_three_pass": all(passes(r) for r in rows[-3:]),
                    "substantial_candidate_observations": len(substantial),
                    "substantial_winners": winners,
                    "substantial_strict_preference_changes": strict_changes,
                }
            )
    assert len(results) == 11
    synthetic = args.work / "synthetic"
    for filename, checksum in read_json(synthetic / "manifest.json").items():
        assert sha256(synthetic / filename) == checksum
        verified += 1
    arrays = torch.load(synthetic / "arrays.pt", weights_only=True)
    assert len(arrays) == 5 and all(x.numel() == 1536 for x in arrays.values())
    write_json(args.output / "development-summary.json", results)
    write_json(
        args.output / "verification.json",
        {
            "files_verified": verified,
            "development_runs": len(results),
            "script_sha256": sha256(Path(__file__)),
        },
    )
    (args.output / "summary-source.py").write_bytes(Path(__file__).read_bytes())
    record_manifest(args.output)
    print(f"Verified all 11 development runs: {args.output}")


if __name__ == "__main__":
    main()

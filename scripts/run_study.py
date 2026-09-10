"""Run the full study, with independent seeds in three CPU processes."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root), (
        "Commit or resolve working-tree changes before recording a study."
    )
    study = args.output.resolve()
    study.mkdir(parents=True, exist_ok=False)
    commands = [
        [
            sys.executable,
            "-m",
            "afloat.cli",
            "run",
            "--output",
            str(study / f"seed-{seed}"),
            "--targets",
            "linear",
            "kink",
            "plateau",
            "detail",
            "bump",
            "composition",
            "--modes",
            "fp32",
            "fixed-e4m3",
            "fixed-e5m2",
            "fixed-hybrid",
            "calibrated",
            "adaptive",
            "--seeds",
            str(seed),
            "--steps",
            "2000",
            "--warmup-steps",
            "50",
            "--batch-size",
            "128",
            "--learning-rate",
            "0.001",
            "--adapt-every",
            "100",
            "--eval-every",
            "100",
            "--sample-size",
            "2048",
            "--threads",
            "1",
        ]
        for seed in range(3)
    ]
    write_json(
        study / "protocol.json",
        {
            "source_revision": revision,
            "started_at": datetime.now(UTC).isoformat(),
            "commands": commands,
            "primary_outcome": (
                "Final validation mean squared error at 2,000 post-warmup steps"
            ),
            "comparison": (
                "Paired differences against every fixed control and calibrated"
            ),
            "stopping_rule": (
                "Complete all requested runs; retain failures; "
                "do not extend based on a favorable result"
            ),
            "interpretation": (
                "Exploratory study; check FP32 feature learning "
                "before making format claims"
            ),
            "cpu_processes": 3,
            "threads_per_process": 1,
            "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    started = perf_counter()
    running = []
    for seed, command in enumerate(commands):
        stream = (study / f"seed-{seed}.log").open("x")
        process = subprocess.Popen(
            command, cwd=root, stdout=stream, stderr=subprocess.STDOUT
        )
        running.append((seed, process, stream))
        print(f"Started seed {seed}", flush=True)
    outcomes = []
    for seed, process, stream in running:
        code = process.wait()
        stream.close()
        outcomes.append({"seed": seed, "exit_code": code})
        print(f"Seed {seed} finished with exit code {code}", flush=True)
    write_json(
        study / "completion.json",
        {
            "finished_at": datetime.now(UTC).isoformat(),
            "wall_seconds": perf_counter() - started,
            "outcomes": outcomes,
        },
    )
    sys.exit(int(any(row["exit_code"] for row in outcomes)))


if __name__ == "__main__":
    main()

"""Execute the frozen follow-up protocol in three independent CPU processes."""

import argparse
import concurrent.futures
import json
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from afloat.artifacts import sha256, write_json


def execute(command, logfile, root):
    with logfile.open("x") as stream:
        result = subprocess.run(
            command, cwd=root, stdout=stream, stderr=subprocess.STDOUT
        )
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root)
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    plan_path = root / "docs/followup-protocol.json"
    plan = json.loads(plan_path.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for gain in plan["activation_gains"]:
        for seed in plan["seeds"]:
            name = f"gain-{gain:g}-seed-{seed}"
            modes = plan["modes"].copy()
            random.Random(9100 + int(gain) * 100 + seed).shuffle(modes)
            command = [
                sys.executable,
                "-m",
                "afloat.cli",
                "run",
                "--output",
                str(output / name),
                "--targets",
                plan["target"],
                "--seeds",
                str(seed),
                "--modes",
                *modes,
                "--input-features",
                plan["input_features"],
                "--activation-gain",
                str(gain),
                "--save-checkpoints",
            ]
            for key in (
                "steps",
                "warmup_steps",
                "batch_size",
                "learning_rate",
                "final_learning_rate",
                "adapt_every",
                "eval_every",
                "format_log_every",
                "sample_size",
                "threads",
            ):
                command.extend(["--" + key.replace("_", "-"), str(plan[key])])
            jobs.append({"name": name, "gain": gain, "seed": seed, "command": command})
    write_json(
        output / "protocol.json",
        {
            "plan": plan,
            "plan_sha256": sha256(plan_path),
            "source_revision": source,
            "launcher_sha256": sha256(Path(__file__)),
            "started_at": datetime.now(UTC).isoformat(),
            "jobs": jobs,
        },
    )
    started = perf_counter()
    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=plan["accuracy_processes"]
    ) as executor:
        futures = {
            executor.submit(
                execute, job["command"], output / (job["name"] + ".log"), root
            ): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            code = future.result()
            outcomes.append({"name": job["name"], "exit_code": code})
            print(
                f"Completed {job['name']}: exit {code} "
                f"({len(outcomes)}/{len(jobs)} groups)",
                flush=True,
            )
    write_json(
        output / "completion.json",
        {
            "finished_at": datetime.now(UTC).isoformat(),
            "wall_seconds": perf_counter() - started,
            "outcomes": outcomes,
        },
    )
    sys.exit(int(any(row["exit_code"] for row in outcomes)))


if __name__ == "__main__":
    main()

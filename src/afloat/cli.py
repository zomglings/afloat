"""Command-line entry points for reproducible experiments and plots."""

import argparse
import json
from pathlib import Path

from afloat.formats import MODES
from afloat.runner import RunConfig, run_experiment
from afloat.targets import TARGETS


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="afloat", description="Adaptive float experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run paired conditions on analytic functions")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--targets", nargs="+", choices=TARGETS, default=["kink", "detail", "bump"]
    )
    run.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    run.add_argument("--seeds", nargs="+", type=int, default=[0])
    run.add_argument("--steps", type=int, default=300)
    run.add_argument("--warmup-steps", type=int, default=50)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--learning-rate", type=float, default=0.001)
    run.add_argument("--eval-every", type=int, default=50)
    run.add_argument("--adapt-every", type=int, default=100)
    run.add_argument("--sample-size", type=int, default=2048)
    run.add_argument("--threads", type=int, default=1)
    plot = commands.add_parser("plot", help="Regenerate a run's loss plot")
    plot.add_argument("output", type=Path)
    plot.add_argument("--destination", type=Path)
    replay = commands.add_parser(
        "replay", help="Repeat the settings saved in an earlier run"
    )
    replay.add_argument("previous", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plot":
        from afloat.plotting import plot_run

        destination = args.destination or args.output / "loss-regenerated.png"
        print(plot_run(args.output, destination))
        return
    if args.command == "replay":
        saved = json.loads((args.previous / "config.json").read_text())
        for name in ("targets", "modes", "seeds"):
            saved[name] = tuple(saved[name])
        print(run_experiment(RunConfig(**saved), args.output))
        return
    config = RunConfig(
        targets=tuple(args.targets),
        modes=tuple(args.modes),
        seeds=tuple(args.seeds),
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        adapt_every=args.adapt_every,
        sample_size=args.sample_size,
        threads=args.threads,
    )
    print(run_experiment(config, args.output))


if __name__ == "__main__":
    main()

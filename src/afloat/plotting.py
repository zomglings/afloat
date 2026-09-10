"""Plot each paired trajectory without hiding seeds or fixed-format controls."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_run(output: Path, destination: Path | None = None) -> Path:
    with (output / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    targets = list(dict.fromkeys(row["target"] for row in rows))
    figure, axes = plt.subplots(
        len(targets),
        1,
        figsize=(9, 3.7 * len(targets)),
        squeeze=False,
        layout="constrained",
    )
    for axis, target in zip(axes[:, 0], targets, strict=True):
        relevant = [row for row in rows if row["target"] == target]
        groups = list(dict.fromkeys((row["mode"], row["seed"]) for row in relevant))
        modes = list(dict.fromkeys(mode for mode, _ in groups))
        for mode, seed in groups:
            points = [
                row for row in relevant if (row["mode"], row["seed"]) == (mode, seed)
            ]
            axis.plot(
                [int(row["step"]) for row in points],
                [float(row["validation_mse"]) for row in points],
                label=f"{mode}, seed {seed}",
                color=f"C{modes.index(mode)}",
                alpha=0.8,
            )
        axis.set(
            title=target,
            xlabel="Steps after shared warmup",
            ylabel="Validation mean squared error",
        )
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    figure.suptitle("afloat — exploratory training traces", fontsize=14)
    path = output / "loss.png" if destination is None else destination
    with path.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150)
    plt.close(figure)
    return path

"""A changing array checks range selection independently of model learning."""

import argparse
import json
from pathlib import Path

import torch

from afloat.artifacts import record_manifest, write_json
from afloat.formats import FormatPolicy, choose_format, format_scores

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=False)
arrays = {"narrow": torch.linspace(0.1, 1, 1536)}
for exponent in (2, 4, 6):
    arrays[f"small-1e-{exponent}"] = torch.cat(
        (
            torch.linspace(10.0 ** (-exponent - 1), 10.0 ** (-exponent), 1535),
            torch.ones(1),
        )
    )
arrays["narrow-again"] = arrays["narrow"].clone()
write_json(
    out / "protocol.json",
    {
        "scope": "synthetic reconstruction only",
        "array_elements": 1536,
        "stages": list(arrays),
        "selection_every_stage": True,
        "score": "full-array mean squared error",
        "strict_margin": 0.01,
    },
)
(out / "synthetic-source.py").write_bytes(Path(__file__).read_bytes())
torch.save(arrays, out / "arrays.pt")
initial = {"weight:array": choose_format(arrays["narrow"])}
policies = {
    mode: FormatPolicy(mode, initial, 1, 2048)
    for mode in ("fixed-e3m4", "fixed-e4m3", "fixed-e5m2", "calibrated", "adaptive")
}
rows = []
for step, (name, x) in enumerate(arrays.items()):
    scores = format_scores(x)
    ordered = sorted(scores, key=scores.get)
    row = {
        "step": step,
        "stage": name,
        "scores": scores,
        "preferred": ordered[0],
        "relative_margin": (scores[ordered[1]] - scores[ordered[0]])
        / scores[ordered[1]],
        "conditions": {},
    }
    represented = {}
    for mode, policy in policies.items():
        q = policy.apply(x, "array", "weight", step)
        represented[mode] = q
        row["conditions"][mode] = {
            "mse": float((q.double() - x.double()).square().mean()),
            "choice": policy.events[-1]["format"],
            "zeroed": policy.events[-1]["zeroed"],
        }
    row["adaptive_calibrated_different_elements"] = int(
        (represented["adaptive"] != represented["calibrated"]).sum()
    )
    rows.append(row)
write_json(out / "results.json", rows)
record_manifest(out)
print(json.dumps(rows, indent=2))

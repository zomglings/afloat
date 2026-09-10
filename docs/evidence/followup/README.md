# Follow-up evidence

Read [the results](../../../followup-results.md) and [audit](audit.md) first.

| Files | Purpose |
| --- | --- |
| `accuracy/` | Frozen protocol, completion, all ten configurations/environments, original manifests, final summaries, and one shared source archive |
| `analysis/metrics.csv`, `finals.csv` | Every validation/gradient measurement and each final result, including threshold crossings and misses at the end |
| `analysis/candidate_scores.csv`, `initial_candidates.json`, `switches.json` | Every reselection score, reconstructed initial score, and actual label switch |
| `analysis/activations.csv`, `formats.csv` | All activation measurements and summaries of retained rounding observations |
| `analysis/paired.json`, `summary.json` | Both primary errors, all six paired controls, seed means, and sample standard deviations |
| `performance/` | Exact measured benchmark source, protocol, 398 checkpoint checks, 24 threshold outcomes, and raw inference/preparation timing blocks |
| `performance-analysis/` | Checked timing reductions and seed-paired timing comparisons |
| `development/`, `development-analysis/` | All 11 development attempts, including failures, with separate historical source snapshots |
| `synthetic/` | Constructed inputs and full per-format rounding measurements; no training claim |
| `review/` | Independent programs, calculations, findings, and a small replay fixture |
| `checks.json`, `check.log`, `smoke.log` | Executed repository checks |

`evidence-sha256.json` covers this compact public copy. Original `manifest.json`
files describe full local directories, including omitted training checkpoints
and routine rounding logs. The distinction is intentional: use the public hash
list to check published files and regenerate the runs for full verification.

Reproduction commands are in [the report](../../../followup-results.md) and
[the development instructions](../../followup-reproduction.md). Historical
review programs ending `.py.txt` retain the paths used during their execution;
copy them into a fresh directory before adjusting paths or rerunning them.

Fields named `first_crossing_step`, `first_crossing_seconds`, or `crossing` refer
to the start of the first sequence of three passing measurements, identified
retrospectively once the third pass confirms it. They do not mean the first
isolated passing measurement.

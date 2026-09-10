# Full-study evidence audit

The question was whether repeatedly fitting formats to current weights and
gradients improves approximation over standard formats and choosing a format
once. The full study is ready to report. A consistent adaptive benefit remains
unproved: no target improved over choosing once on all three seeds, all format
switches were confined to the scalar output bias or its gradient, and FP32 did
not learn the small sine feature.

| Claim | Rival explanation or failure | Evidence and result |
| --- | --- | --- |
| All requested conditions ran | Selective retention, missing seeds, incomplete steps | 108 completed conditions, three successful processes, no recorded failure; all 528 original manifest entries verified |
| Reported errors describe saved models | Incorrect reductions or stale predictions | All 108 final feature measurements recalculated; reviewer also rebuilt every represented model and reproduced its predictions exactly |
| Repeated selection improves learning | Choosing once or simply using E3M4 may explain gains | Three paired seeds and all fixed controls reported; no consistent advantage; fixed E3M4 was not run |
| Functions exercise intended behavior | Low overall error may conceal an unlearned feature | Regional errors, small-feature amplitude, and activation signs/slopes checked; detail learnability fails, saturation is sparse |
| Timing measures useful work | Conversion, logging, concurrency, or setup can dominate | Separate single-thread serial benchmark, explicit included/excluded work, cached-weight inference and separate preparation; all 108 endpoint losses match |
| Results can be checked and reproduced | Changed code, missing artifacts, overwritten outputs | Pinned training source, configurations, original manifests, compact measurements, source snapshots, scripts, and archived independent calculations |

`make check` passed Ruff lint and formatting, mypy on nine source files, and all
33 tests. `make smoke` completed all six conditions in a fresh directory. No
required check was omitted. Exact observed commands and outcomes are recorded in
[checks.json](checks.json). Figures were opened and inspected for readable labels,
complete panels, and agreement with numerical results. Passing software checks
does not establish a learning benefit.

The independent review recomputed final errors and all 30 adaptive/control
comparisons, checked every actual switch against the raw event sequence,
verified all 5,760,000 rounding records and their summaries, checked 162 final
FP32 activation groups, and recalculated all 108 benchmark timing summaries.
Its [findings](review/findings.md),
[independent values](review/independent-results.json), and
[rounding table checks](review/report-table-checks.json) are preserved.

## Corrections made during review

- Narrowed “training trajectories verified” to “step-300 validation losses
  matched.” The benchmark checks one scalar endpoint loss, not intermediate
  states. Exact equality of final cached weight representations is checked
  separately.
- Corrected timing-block wording: 0.1 seconds is the minimum total measurement
  time across repeated blocks, not the duration of each block.
- Replaced an unsupported equivalence to untested fixed E3M4 with the observed
  E3M4 element share. A change in one scalar bias can affect later learning.
- Specified that per-condition accuracy-run timers stop at final validation,
  before final metric writes, activation probes, and checkpoint saving.

Original performance measurements and their protocol were preserved unchanged.
Their summary field `all_training_trajectories_verified_at_step_300` has the
narrow scalar-loss meaning above. The exact measured script is archived as
[benchmark-source.py](performance/benchmark-source.py), matching the protocol's
`analysis_source_sha256`. The maintained benchmark now writes the accurate field
`all_step_300_validation_losses_matched`; its numerical operations are unchanged.
No timing rerun was needed for an output-label correction.

The independent reviewer reread the corrected report and found no remaining
material reporting or arithmetic defect within this bounded review. The review
did not independently remeasure runtime or establish hardware acceleration.
Rath was unavailable because its required API credential was absent, as already
recorded in the initial implementation validation; a separate review agent
performed this examination.

## Reproduction and artifact boundaries

The portable commands in [proto-results.md](../../../proto-results.md) regenerate
the study, analysis, and benchmark into new directories. Analysis does not run
concurrently with timing. `verification.json` records the measured training
source hashes and analysis script hash. The original complete local study is
`runs/full-study.NzhhwC`; its three `source/` directories retain the exact training
source and dependency lock.

The `seed-*/manifest.json` files describe the original complete run directories.
Checkpoints and per-update logs remain local under ignored `runs/`; they are not
included in this compact public copy. The public evidence contains aggregate
format and activation measurements, every validation/gradient metric row,
raw performance timing blocks, configurations, original manifests, and plots.
`evidence-sha256.json` verifies the compact public files and is distinct from the
original run manifests.

The independent review's executed scripts are archived as `.py.txt` files to
distinguish recorded source from maintained tools. They retain the original local
study paths. To repeat that exact local check without overwriting evidence, copy
them into a fresh scratch directory before execution:

```sh
afloat_review=$(mktemp -d)
cp docs/evidence/full-study/review/audit.py.txt "$afloat_review/audit.py"
cp docs/evidence/full-study/review/verify_models.py.txt "$afloat_review/verify_models.py"
cp docs/evidence/full-study/review/verify_report_tables.py.txt "$afloat_review/verify_report_tables.py"
uv run python "$afloat_review/audit.py"
uv run python "$afloat_review/verify_models.py"
uv run python "$afloat_review/verify_report_tables.py"
```

For a relocated study, update the archived scripts' explicit input paths in the
scratch copies. Use the maintained analysis and benchmark commands for new runs.
The archived findings' temporary paths identify where the original independent
checks ran; those scripts and their calculated results are now preserved here.

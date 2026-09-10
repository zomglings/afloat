# Follow-up evidence audit

The tested claim is that repeated format selection improves approximation or
measured time to a fixed accuracy target over fixed E3M4 and choosing once.
The strongest simple controls are now present. Results support reporting this
bounded experiment; a reliable advantage for adaptation remains unproved.

| Claim | Failure or rival explanation | Evidence and disposition |
| --- | --- | --- |
| The small feature is learnable | Overall error conceals a missing component | All raw-coordinate development settings failed; fixed sine/cosine inputs passed on both development seeds at 20,000 updates. This changes the representation and does not establish frequency discovery. |
| Nonlinear ranges vary | The gain setting has no observed effect | Recorded input percentiles and derivatives differ at gains 1 and 4; every method retains its raw activation measurements. |
| Training offers a changing format choice | Static E3M4 or scalar changes explain apparent adaptation | Zero substantial-array preference changes. All 59,700 substantial reselection observations prefer E3M4. Synthetic switches only establish the rounding mechanism. |
| Final comparisons retain unfavorable results | Missing seeds, selected checkpoints, changing budgets | Frozen 70-condition protocol; every condition completes 20,000 updates. All final results, all 24 primary paired comparisons, and failed development settings are retained. |
| Reported values describe actual models | Stale saved predictions or wrong representation | All 70 final represented models reproduce saved predictions and final feature measurements exactly. All 3,370 original run-manifest entries verify. |
| Time compares equal numerical work | Diagnostic costs, changed schedule, or unequal training states | Serial replay retains the original learning-rate horizon and compares complete model/optimizer/format state every 500 updates. Warmup, applicable selection, training, and validation are explicitly timed. |
| Inference compares equivalent execution | Repeated preparation hides the model cost | All 70 final representations are checked; cached inference and preparation have separate raw timing blocks. All arithmetic remains FP32. |
| Failure handling preserves spent work | A failed setup aborts later cases or loses partial time | Independent injected failures exposed both defects. Corrections retain setup/block/validation time, completed updates, and later outcomes; checks are archived and covered by regression tests. |
| Results can be checked and reproduced | Changed code or unavailable evidence | Clean measured revision, dependency lock, source hashes, original manifests, compact raw tables, scripts, snapshots, and independent calculations are retained. |

## Review and corrections

Rath could not run because its required API credential was unavailable, as
recorded in the earlier implementation audit. A separate review agent examined
the plan, source changes, timing implementation, analysis, and final arithmetic.
Each pass had a bounded purpose; empirical claims were checked against data,
not inferred from passing software tests.

The review identified and verified corrections to:

- Historical analysis rejecting otherwise equivalent configurations with newly
  added default fields. It now accepts only the exact original defaults and
  still rejects changed settings.
- Treating a tied label change as an adaptation opportunity. The frozen rule
  requires the same array to change strict preference, with at least a 1% score
  advantage at both observations, and at least 32 entries.
- Missing measured setup/partial-block costs after numerical failure. The
  benchmark now accounts in `finally`, records completed work and the phase,
  retains unknown setup components as unknown, and continues other cases.
- An archived analysis script lacking its imported benchmark helper. The exact
  dependency is now saved beside the analysis source.

The complete-study analyzer intentionally refuses missing or failed accuracy
conditions. The runner retains their failure records; such a study would need
an explicit failure report rather than a table that silently drops pairs. All
70 accuracy conditions in this report succeeded.

Activation summaries combine three layers across five seeds, not 15 independent
seeds. Rounding summaries describe retained observations, not every update.
Five-seed comparisons are descriptive; no statistical or cross-workload
advantage is claimed. Timing describes this CPU simulation and this host, not
native eight-bit arithmetic, energy use, or hardware memory traffic.

`make check` passed lint, format checking on 43 Python files, mypy on nine source
files, and all 51 tests. `make smoke` completed all seven modes in the fresh
`runs/smoke.YlLf2s/result` directory. No required check was omitted.
[Check records](checks.json), [check output](check.log), and [smoke output](smoke.log)
are preserved. The learning-curve figure was opened and checked for readable,
complete panels and agreement with the reported values.

The serial pass completed all 24 threshold replays, with 398 exact checkpoint
comparisons and all 70 final-model inference cases. The final reviewer independently
reduced all 210 raw inference/preparation distributions and all paired threshold
costs, verified original hashes, source archives, report tables, and links.
This review checked the benchmark's recorded full-state comparisons and exact
reference metrics; it did not rerun all 398 intermediate training states.

One wording correction distinguished the retrospectively identified start of
the first three-pass sequence from an earlier isolated passing measurement.
There were 14 such earlier passes among the 24 replays. The saved values did not
change. The report and public data definitions now call this the sustained start.

Review evidence is preserved for the [plan](review/plan/findings.md),
[source](review/source/findings.md), [benchmark](review/benchmark/findings.md),
[analysis](review/analysis/findings.md),
[accuracy calculations](review/accuracy-results/summary.md), and
[final performance/report review](review/final/findings.md).
No material numerical or control defect remained in the reviewed measurements.
The final public evidence hash list was regenerated after assembly and verified
against every published file.

## Evidence boundaries

The original local work group is `runs/followup.KGkx3v`. Accuracy data came from
clean commit `4d9959e313b1915558f13574d9aee18fd07f256e`. All ten groups have
identical source hashes and dependency versions. Reporting scripts were added
after execution; their measured source hashes and exact copies are preserved.
Development intentionally evolved before the frozen study; each of the six
development groups has its own original source archive and protocol. Those
snapshots are not presented as if they came from a single clean revision.

Original manifests describe full run directories, including local checkpoints
and logs. Published `evidence-sha256.json` describes only the compact public copy.
The latter includes every validation/gradient measurement, every activation
measurement, all 63,680 reselection scores, initial scores, all switches, paired
calculations, raw timing blocks, development measurements, synthetic inputs,
source archives, and independent review records. Bulk training checkpoints and
routine format logs stay in ignored `runs/` and can be regenerated.

Archived review programs use the original explicit local paths. Copy them to a
fresh directory made with `mktemp -d` before rerunning and adjust input paths for
a relocated study. Files ending `.py.txt` are historical source, not maintained
commands. The maintained commands in the report reproduce new studies. Extract
`source.tar.gz` into a fresh directory to inspect the exact original source and
lock; use its `src/` on `PYTHONPATH` when running an archived development script.

The publication check preserves original CSV carriage-return/newline bytes and
blank context lines in archived diffs. Targeted `.gitattributes` rules distinguish
these recorded formats from source whitespace errors. The staged public files
were checked against the complete public hash list before committing.

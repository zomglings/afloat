# Follow-up analysis audit

Bounded read-only review of `runs/followup.KGkx3v/analyze_followup.py`, the frozen protocol, and already-completed accuracy groups. No full analysis, bulk manifest verification, long model replay, or representative timing ran while accuracy training was active. All actual-data checks used complete gain-1 groups; initial-candidate/model checks used one completed group.

## Corrections and dispositions

- The first analysis version saved `analysis-source.py` but only hashed its imported sibling `benchmark_followup.py`. The copied script could not run independently: `--help` failed with `ModuleNotFoundError`. The current version now archives the helper under the importable filename `benchmark_followup.py`. The copied script plus dependency passes `--help` from this separate audit directory.
- The parent added reconstructed initial-calibration candidate scores before the final check. This closes the earlier gap where a strict initial preference changing before the first periodic candidate observation could be missed. Reconstruction uses the saved common checkpoint and the prescribed next training batch and asserts its chosen formats against `initial-formats.json`.
- The analyzer intentionally refuses incomplete or numerically failed accuracy studies. It does not silently omit failed pairs. The runner preserves failure records, and the parent will report any such failure explicitly instead of issuing a success table. This is a limitation of the complete-study analysis path; no checked current result is invalidated by it.

## Independent checks

- All 12 paired gain-1 comparisons (six controls, two primary metrics) match independent within-seed subtraction, win/tie counts, and ratios of means across the five completed seeds. Controls are joined by seed within gain rather than compared across mismatched orderings.
- Threshold fixtures correctly handle an initial-step pass, an interrupted streak followed by three passes, a threshold miss, and a run that passed earlier but fails at the end. The first crossing and confirmation match the frozen 500-update cadence and three-measurement persistence. `ever_sustained` and `final_three_pass` remain distinct.
- A synthetic substantial-array sequence starting with strict E3M4 preference, then an E4M3 tie, then strict E5M2 preference yields two label switches, one rounded-value change, and one strict preference change. The previous strict observation is correctly retained across the tie. Both strict observations exceed the required 1% margin.
- In completed gain-1/seed-10, reconstructed initial scores match all 32 saved choices. The adaptive log has 6,432 retained array records and 6,368 candidate-score records, matching the frozen routine/reselection schedule. Its 115 label switches include 58 with changed rounded values; none affects a substantial array. Initial-to-periodic strict preference changes are zero in this checked group.
- Independent computation of the isolated component error from saved prediction grids exactly matches FP32 (0.003440172509692075) and adaptive (0.010378044969172483) for gain-1/seed-10. The subtraction uses each fixed-x1 row's mean over x2 and the original small component's squared magnitude.
- Final reconstruction logic checks represented weights, predictions, recorded feature values, configuration, common source identity, and manifest entries. Only full final models are reconstructed; the intermediate checkpoints are counted and hashed, not all reconstructed by this script. The serial benchmark separately checks intermediate model/optimizer/format-choice states.

## Interpretation limits for reporting

Routine format summaries describe retained records, not every update. Label-switch counts must not be described as numerical changes or substantial strict-preference changes. Activation summaries contain three layers times five seeds (15 group observations): their standard deviation mixes layer and seed variation and is not a 15-seed uncertainty estimate. The primary metric summaries correctly use five seeds. Accuracy-run `training_seconds` comes from the concurrent study; use the separate serial benchmark for performance conclusions. The hardcoded 500-step/three-pass logic matches this frozen protocol and should not be presented as a general configurable analyzer.

No material unresolved numerical or current-result defect was found within this bounded audit. A final report still needs to distinguish learnability with the supplied frequency features from discovering frequencies from raw inputs, and distinguish fixed-format comparisons from an adaptation-opportunity test.

## Reproduction and evidence

From `/Users/neeraj/src/zomglings/afloat`:

```sh
.venv/bin/python /var/folders/x5/qnn2lxld3j75g3kt9dj_v1v40000gn/T/tmp.7rMICK3mDp/check_analysis.py
.venv/bin/python /var/folders/x5/qnn2lxld3j75g3kt9dj_v1v40000gn/T/tmp.7rMICK3mDp/analysis-source.py --help
```

For another fixture run, create a fresh directory with `mktemp -d` and copy the scripts there first, since the test creates `format-fixture`. `results.json` records threshold fixtures, strict-change counts, selected raw counts, paired checks, and independently calculated component errors. `analysis-source.py` and `benchmark_followup.py` preserve the inspected source pair. No tracked files were changed by the reviewer.

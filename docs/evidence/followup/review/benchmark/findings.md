# Follow-up benchmark review

Bounded read-only review of `runs/followup.KGkx3v/benchmark_followup.py` against the frozen protocol. No long replay or representative timing was run while accuracy training was active. Two-update checks used previous unrelated seed-777 implementation fixtures; failure checks used tiny synthetic files and a logical clock.

## Reproduced defects in the pre-fix version

1. `warmup()` occurs outside the per-case `try`, so a numerical warmup failure aborts the entire benchmark without a case outcome or summary. The injected warmup failure produced no `threshold-results.jsonl` record and no summary.
2. Training-block time is accumulated only after `train_block` returns. A numerical failure partway through a block discards that block's time; the failure outcome also omits reached status, completed updates, measured seconds, validation seconds, and last successful checkpoint. The injected block failure produced `training_seconds=0.0` and no total measured time despite attempted block work.

The exact reviewed implementation is preserved in `benchmark-before-fixes.py`, and `check_failure_accounting.py` imports that snapshot. `failure-results.json` contains the counterexamples. The parent agreed to place warmup inside case failure handling, account for partial time through `finally`, track completed updates, and preserve complete failure outcomes. The correction was verified against the current source. Both defects are resolved. `measure_call` accounts for every call in `finally`; `replay_case` handles setup and subsequent numerical failures, preserves completed optimizer updates, phase, total measured cost, validation cost, and last successful validation. A failed setup is reported as an unsplit cost with unknown warmup/calibration components rather than inventing them.

## Checks that passed

- `config_for` retains the 20,000-update cosine horizon independently of an earlier threshold stop.
- The existing block implementation exactly reproduces model parameters, optimizer state, and format choices after two resumed updates for all seven modes, including a reselection. Its validation metrics exactly match the saved measurements.
- Threshold comparisons use strict MSE/component-error bounds and inclusive amplitude bounds, matching the protocol.
- Source inspection confirms the streak resets on failure, and `crossing` records the start of the first three-success streak that becomes confirmed. Both the first crossing and confirmation cost are retained for successful cases. A full-budget threshold miss retains its measured budget cost with `reached=false` rather than being silently removed.
- Main-case timing excludes checkpoint reads, comparisons, output logging, and inference; includes training blocks and each threshold-validation pass. Applicable initial selection is charged only to calibrated/adaptive, and measured warmup is common. Routine quantizer bookkeeping remains timed. These are simulator work timings, not native eight-bit hardware measurements.
- Cached inference uses saved final model settings and current frozen format choices, asserts every prepared parameter against its saved representation, and times preparation separately. Its arithmetic remains FP32. Order and source are reproducible from the saved script; hardcoded inference order seed 919 differs from the threshold order seed 918 but does not change the frozen case list.

## Reproduction

Run the scripts with the repository's `.venv/bin/python`. The failure script writes below its own directory; create a fresh `mktemp -d` directory and copy it plus `benchmark-before-fixes.py` there before rerunning. `check_replay_block.py` reads the existing tiny seed-777 checkpoint fixture and does not time the runs. Results are saved in `block-results.json` and `failure-results.json`.

This review does not independently measure performance or establish that future 500-update replays will match; the implementation checks each such checkpoint during execution. Verification errors should remain distinct from ordinary threshold misses.

## Final correction verification

- Injected setup failure: reached=false, zero post-warmup updates, measured setup cost retained unsplit.
- Injected partial-block failure: 123 completed updates retained; block time, earlier validation time, total measured time, and last successful validation retained.
- Injected validation failure: 500 completed updates retained, including time spent in the failing validation.
- Outer-loop checks with one failed and one normal case passed for both setup and partial-block failures. The normal case still completed; summary and manifest were written; the benchmark exited with its explicit failure error afterward.
- Streak tests passed for initial-step success, reset after a failed measurement, first confirmed three-pass streak, and a full-budget threshold miss. The stored crossing is the first measurement of the streak that becomes confirmed, and the result cost reflects confirmation.
- Exact two-update model/optimizer/format-choice continuation and feature-metric checks still pass for all seven modes after the correction.

`benchmark-reviewed.py` preserves the corrected source. `check_patched_cases.py` and `patched-results.json` contain the six logical-clock cases. `check_continuation.py` and `continuation-results.json` exercise the real outer loop with mocked training/timing. `check_replay_block.py` contains the actual small replay checks. No real fresh comparison models were trained; no long replay or representative timing ran.

No material unresolved benchmark implementation issue was found within this bounded review. Actual performance and all exact 500-update comparisons remain to be established during the serial execution after accuracy work completes. The review does not imply any desired adaptation or runtime outcome.

# Frozen follow-up source and protocol review

Read-only review of `docs/followup-plan.md`, `docs/followup-protocol.json`, the working source diff, `scripts/run_followup.py`, tests, and development measurements. No final comparison seed was trained or inspected by this review. A four-update implementation check used unrelated seed 777, and launcher subprocess execution was replaced with command capture.

## Disposition

No material unresolved source or frozen-accuracy-protocol error was found within this bounded review. The one identified evidence mismatch was corrected before conclusion: format-change events now record `changed_elements` and `rounding_change_mse` by applying both the previous and selected formats to the same current array. The corresponding distribution-switch test verifies the exact count and error. Candidate scores and actual changes remain retained even when routine logs are sparse.

## Verified

- The frozen launcher constructs exactly 10 gain/seed jobs and 70 conditions. Every command contains the declared gain, Fourier representation, seven modes, 20,000/50 updates, cosine endpoints, batch size, selection/measurement/logging intervals, and checkpoint setting. Three workers execute independent subprocesses; shuffled mode order does not consume training randomness. Dry command construction wrote its full output protocol in `launcher/protocol.json`; no study subprocess ran.
- A fresh four-update, seven-condition run with Fourier inputs, gain 4, cosine decay, sparse logs, and measurement checkpoints completed. Independently continuing each condition from its saved step-2 checkpoint to step 4 reproduced **all model tensors, optimizer state, and format choices exactly**. This checks the evidence needed by the later serial replay without rerunning the study.
- Fourier inputs contain both raw coordinates and symmetric sine/cosine frequencies. The target is unchanged and the protocol explicitly limits interpretation to learning with the supplied frequency features. The resulting model has 10,097 parameters; conditions within each gain receive the same architecture and shared warmup checkpoint.
- The learning-rate schedule uses the common configured total update budget. The update timer stops before format-file writes and validation/checkpoint work, and subtracts the measured FP32 reference-gradient probe. It still includes routine quantizer statistics, gradient copying/concatenation, and timer overhead, so it describes measured simulator work. Shared warmup and initial calibration have separate saved costs. Concurrent accuracy-run timings are not serial performance evidence.
- Historical analysis accepts absent/default added fields while rejecting nondefault gain, input features, schedule, logging, or checkpoint settings. Historical modes remain explicitly limited to the prototype six. The old benchmark remains prototype-specific and is not the follow-up serial replay.
- Development records independently confirm both gains pass every final-three feature check at 20,000 updates on seeds 100/101. The 6,000-update gain-4 seed-100 run fails all final-three checks, while the other short-budget cases pass. This supports the recorded decision to retain 20,000 updates for both gains. The lack of observed substantial-array preference changes is explicitly retained as a limitation rather than treated as a successful adaptation-opportunity check.
- `.venv/bin/python -m pytest -q tests/test_formats.py tests/test_followup.py` passed **26 tests in 2.33 seconds**, including the newly added switch-difference assertions.

## Separate remaining review

The serial threshold-replay implementation is not yet present and was not verified here. It needs its own bounded review before execution. In particular, stopping early must retain `RunConfig.steps=20000` so the cosine schedule is unchanged; compare exact model, optimizer, and format choices at every measurement; retain threshold misses; distinguish first crossing from third-pass confirmation; and account for warmup/calibration and measurement/verification timing explicitly. The parent confirmed these requirements and will request that review separately. This does not require delaying the already-frozen accuracy launch.

## Reproduction

From `/Users/neeraj/src/zomglings/afloat`:

```sh
.venv/bin/python /var/folders/x5/qnn2lxld3j75g3kt9dj_v1v40000gn/T/tmp.CyeMye920M/verify_frozen.py
.venv/bin/python -m pytest -q tests/test_formats.py tests/test_followup.py
```

The script writes fresh output paths under its dedicated temporary directory; for another run, create a fresh directory with `mktemp -d` and copy the script there first. `verification.json` contains all seven exact continuation results and launcher counts; `reviewed.diff` preserves the reviewed working changes. No source files were edited by the reviewer.

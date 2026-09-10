# Final follow-up performance and report review

Claim examined: the final report accurately describes the frozen accuracy study and serial CPU measurements, without treating scalar switches, supplied frequency features, or noisy timing differences as proof that adaptation helps.

Result: no unresolved numerical or control defect found in the completed measurements. The report's accuracy conclusions agree with the independent prior accuracy calculation. Timing tables, percentages, and paired counts reproduce from raw data. The sustained-start wording correction is verified. Publication assembly still requires refreshing the public evidence hash list after adding this review and the other final files.

## Independent checks

- Rehashed all 3,370 accuracy manifest entries and all 20 performance entries successfully. Every accuracy environment equals the performance environment, including clean revision `4d9959e313b1915558f13574d9aee18fd07f256e`. Every recorded source file matches that Git revision and the saved source copy.
- The actual benchmark source hashes to `78cfd75c73d24125c7dbdc4a6e510898d68bbd50b99ae1ec6c96ae49560036e5`, matching its protocol and maintained script. The recorded accuracy protocol hashes correctly and its plan equals the frozen public protocol.
- Exactly 24 unique threshold cases follow the declared shuffled order. All reached the criterion without numerical failure. The 398 measurement rows have complete 500-update spacing, exact accuracy-reference metrics, correct pass flags and first three-pass sequences, increasing costs, and correctly summed costs. All recorded state-comparison flags are true. Summed measured case work is 464.74976549390703 seconds; recorded performance wall interval is 492.7514432920143 seconds.
- All 70 expected inference models are present. Independently reduced all 210 raw timing distributions by dividing block times by calls per block; medians and interquartile ranges match. Every distribution contains at least 0.1 measured seconds (minimum 0.10007550194859505). All 14 group medians/minima/maxima match the saved summary and report.
- Adaptive/fixed-E3M4 confirmation ratios of medians are 0.758208115026326 at gain 1 and 1.092057377350702 at gain 4; adaptive/calibrated ratios are 0.8513248575742939 and 1.1477024588336397. Matching-seed faster counts are respectively 3/3, 0/3, 2/3, and 1/3. Every seed ratio agrees with the maintained reduction script, which was also executed through its CLI.
- All seven public source archives match their recorded source hashes (13 files each). Published benchmark files match the measured originals; archived analysis and helper match maintained scripts. Report/reproduction links resolve. The public hash list's existing 177 entries match; 46 later-added files were not yet listed when checked. The author was notified to regenerate it after final assembly. This review does not claim the still-changing public hash list is final.

## Corrected claim

The draft called the start of the successful three-pass sequence the “first crossing.” In 14 of 24 cases an earlier passing observation occurred and was reset. For gain 1, seed 10, calibrated, a measurement first passes at update 2,500, but the first sustained sequence starts at 10,000 and confirms at 11,000. The report now calls this the retrospectively identified sustained start, explicitly records the 14 cases, and labels its table accordingly. No measured value changed.

## Surviving limits

- The 398 exact model/optimizer/choice comparisons were executed by the reviewed benchmark and are supported by its saved flags, exact metric comparisons, and verified files. This final audit did not rerun all training or independently reconstruct those 398 model states. Earlier bounded tests independently checked continuation and failure accounting; those artifacts are separately archived.
- Each training case was timed once. Three seeds mix different learning paths with host timing noise; they do not estimate repeated-run uncertainty. Light editing/file work continued during timing. The reported percentage differences are descriptive for these cases. Threshold cost combines learning speed and simulator cost; it does not isolate selector overhead.
- Inference uses identical architecture and FP32 operations with cached weights. Small differences do not establish a hardware speed advantage. The largest within-model relative interquartile range was 4.18% for single input, 16.91% for batch 128, and 3.46% for preparation. The report makes no significance claim from these differences.
- Measured case costs omit checkpoint reads/comparisons, logging, extra probes, and source capture. Training retains rounding-statistic work. The large performance wall interval begins after priming and source capture; it includes the main replay/inference work. No full deployment or native eight-bit speed claim follows.
- The input map explicitly supplies target oscillations. The feature result establishes their learned contribution under that representation, not frequency discovery from raw coordinates. All training format changes are scalar; zero substantial-array preference changes leave the central adaptive-learning benefit unproved. The report states these restrictions.
- Public compact evidence omits bulk checkpoints and routine format logs. The original complete manifests cannot all be checked using the compact copy. Source archives, raw published tables, regeneration commands, and the distinction between original and public manifests are provided explicitly.
- The performance reduction script intentionally summarizes successful complete threshold results and asserts all 24 reached. It would fail visibly on a future threshold miss instead of silently dropping it. The current measurements satisfy this restriction.

## Reproduce this audit

From any directory with the original repository and study available:

```sh
python3 /var/folders/x5/qnn2lxld3j75g3kt9dj_v1v40000gn/T/tmp.TIXxpfPbQi/verify_performance.py
python3 /var/folders/x5/qnn2lxld3j75g3kt9dj_v1v40000gn/T/tmp.TIXxpfPbQi/verify_public.py
```

The programs read fixed original paths and write only alongside themselves. Copy them to a fresh `mktemp -d` directory and update the repository/study paths before using a relocated study. `verified.json` retains exact timing summaries and paired ratios; `public-verified.json` records the publication state observed; `report-reviewed.md`, `reduction-reviewed.py`, and `review-source-hashes.json` identify the reviewed versions. `reduced/` contains the maintained CLI reduction executed during this audit.

Verdict: ready as a bounded measurement report after final public-manifest refresh; an adaptive-training advantage still needs evidence. No further training or timing run is requested by this audit.

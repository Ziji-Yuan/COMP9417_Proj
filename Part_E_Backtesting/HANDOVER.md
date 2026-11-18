# Part E integration handover

This is implementation/verification documentation, not report text. Group
members write the report themselves and select figures/results for the page cap.

## Completed historical experiments

- Main run: `outputs/part_e_full/`, 2015-01-01 through2026-10-31.
- Selected demand/price RFs: frozen, periodic, input-triggered, error-triggered.
- Three adapted variants each use at most six actual replacements, the same
  trailing3calendar-year training window and the original selected configurations.
- D's independent and frozen B/C joint runs: `outputs/part_d_complete/` and
  `outputs/part_d_complete/joint/`. These cover all four Stage1 model families,
  both official baselines and the fixed day-type/input-drift evidence.
- E's `verification.json` checks all eight strategy series,4322days per series,
  targets, model activation, training cutoff, independent frozen predictions,
  baseline metrics and1704monthly interval rows.15Etests pass.

Main E result files:

| File | Use |
| --- | --- |
| `overall_performance.csv` | Pooled full-period metrics, both baselines |
| `yearly_performance.csv`, `monthly_performance.csv` | Chronological comparisons;2026 is Jan–Oct |
| `yearly_jan_oct_model_performance.csv` | Matched seasonal coverage for model strategies |
| `monthly_confidence_intervals.csv` | Independent-observation and2000day-bootstrap intervals |
| `strategy_cost_comparison.csv` | Actual replacement counts, timing costs, baseline improvements |
| `<task>_<strategy>/predictions.csv.gz` | Half-hour predictions with deployed-model IDs |
| `<task>_<strategy>/retrains.csv` | Observation/effective dates, target windows, sample counts |
| `<task>_<strategy>/error_signals.csv` | Each deployed strategy's own chronological error monitor |
| `figures/01..06` | Monthly scores, balanced accuracy, replacement dates, budget comparison, CI widths, detector statistics; PNG/PDF pairs |

Input/error alarms alone do not establish concept drift. Both triggered rules
exhausted their budgets earlier than periodic retraining. Historical comparison
selected the final approach, so it is not unbiased future evaluation. See README
for calibration/model-selection reuse and interval interpretation limitations.

## Final candidate for the member handling overall packaging

`outputs/part_e_final/candidate_submission.zip` is a self-contained candidate:

```text
model.py
train.py
Part_A/{__init__.py,data.py,features.py}
artifacts/{models.joblib,metadata.json}
```

Final demand/price RFs use the published B/C configurations and targets from
2023-11-01 through2026-10-31 (52608rows). Stage3 freshly refits the selected
three-year window through the data boundary, then freezes the models throughout
the live week. It does not keep the January2026 historical-deployment models.

- Artifacts total46.50MiB, below100MiB; compression is lossless XZ level3.
- `Forecaster.__init__` loads files. `predict` computes only causal features and
  forecasts; no fitting, detector update, network access or persistent writes.
- Standalone `train.py CSV` needs only the released CSV and official libraries;
  it constructs the RFs directly, without B/C saved artifacts or historical runs.
- Two independent builds produced **bitwise identical `models.joblib`**. Training
  took roughly61–77seconds locally. See`reproducibility_check.json`.
- Official checker passed14/14days on the actual candidate ZIP. See
  `check_submission.log`; initialization about2.8–3.1seconds, prediction0.2seconds.
- These14days are included in final training: their scores are execution checks,
  not evidence of live-week forecasting performance.
- One local cold-run peak RSS was about755MiB. This is a macOS measurement;
  **Linux/official resource-isolated sandbox has not been tested here**.

Reproduce and check from a pinned Python3.11 environment:

```sh
python -m Part_E_Backtesting.build_final --data /path/to/released.csv --output outputs/final_new
python /path/to/comp9417_project_kit/check_submission.py outputs/final_new/candidate_submission.zip
```

Please run the official checker again on Linux and after any integration or ZIP
change. The final packager decides the final assembly/name and Moodle upload.
There is no automatic upload. Keep the report and experiment-code ZIP separate.

## Live-week forecast evidence

`python -m Part_E_Backtesting.live_forecast --data CSV` produces
`outputs/part_e_live_forecast/`:

- Six2020–2025deployment analogs per task, each freshly fitted on the latest
  three years ending October31 and held fixed through subsequent forecasts.
- Exact December7–13week metrics, plus30nearby nonoverlapping7day scenarios
  across six years. The actual live-week forecast gap is37–43days after cutoff.
- `forecast_evidence.csv` supplies a declared point rule and empirical scenario
  ranges, plus recent-three-year sensitivity. These ranges are not formal
  confidence/prediction intervals or guaranteed95percent future coverage.
- `summary.json` records dependence, selection bias, weather idealization and
  possible out-of-range outcomes. Interpret those limits when writing the report.

## Remaining group actions

1. Linux verification and final packaging/integration by the designated member.
2. Group-authored interpretation, live-week forecast reasoning and references.
3. Ensure D's figures and E's results are not duplicated in the limited report.
4. Cross-check the small C2014validation discrepancy: reproduced RF accuracy
   79.138% vs README79.50%; the full frozen predictions match C's export exactly.
5. Check Moodle updates, final submission files and individual peer reviews.

Generated results/artifacts are ignored by Git. Share them separately from the
source commit; a source pull alone does not transfer the final ZIP or results.

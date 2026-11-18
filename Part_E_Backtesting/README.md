# Part E — Backtesting and adaptation

Work in progress. Implemented: boundary integration checks, online error
monitoring, deployment-budget control, out-of-sample threshold calibration and
the daily forecasting/retraining runner. A January 2015 pilot has exercised real
retraining for both tasks. Full-period comparisons and interpretation are next.

Run from the repository root, in Python 3.11 with the official kit pins:

```sh
python -m Part_E_Backtesting.smoke --kit /path/to/comp9417_project_kit
python -m Part_E_Backtesting.calibrate --data /path/to/nem_weather_halfhourly_2010_2026.csv
python -m unittest discover -s Part_E_Backtesting/tests -t . -v
python -m Part_E_Backtesting.run --data /path/to/nem_weather_halfhourly_2010_2026.csv --end 2015-01-31 --output outputs/new_pilot
python -m Part_E_Backtesting.run --data /path/to/nem_weather_halfhourly_2010_2026.csv
python -m Part_E_Backtesting.plots --results outputs/part_e_full
python -m Part_E_Backtesting.build_final --data /path/to/nem_weather_halfhourly_2010_2026.csv
python -m Part_E_Backtesting.live_forecast --data /path/to/nem_weather_halfhourly_2010_2026.csv
python -m Part_E_Backtesting.plot_live_forecast
```

Outputs default to ignored `outputs/part_e_smoke/` and
`outputs/part_e_calibration/`. Use a new/empty directory for each run. Neither
command changes B/C models or other components. Smoke metrics are diagnostic
days, not an estimate of future performance.

`run` requires D's independent output and E's calibration to match the data
hash. It produces eight model/strategy series by default, two baselines for
each task, pooled overall/year/month metrics and both monthly intervals (2000
day-bootstrap replicates). Daily features are precomputed with A's causal
builder; only a 48-row current-day feature matrix enters each prediction. Fits
slice the table by the requested historical target window. Full simulated
targets never enter threshold calibration.

The declaration records input/model/detector/source hashes and environment.
Each series writes a completion marker last, allowing restart by skipping
verified completed series with the same declaration. An interrupted series is
replayed from deployment start; **within-series checkpoint/resume is not yet
implemented**. Generated historical final models are not saved as submission
models. Source/configuration changes require a new output directory.

For adapted forecasts, the monthly error intervals condition on the realized
deployment/retraining path. Resampling errors does not refit the models or
rerun detection; it does not capture all uncertainty of the adaptive procedure.
Whole-day bootstrap retains within-day correlation but treats days as
exchangeable. Serial dependence across days remains a limitation. Accuracy
intervals do not apply to balanced accuracy, and these intervals are not
live-week predictive intervals or tests based on interval overlap.

## Declared initial adaptation comparison

Use B's and C's Stage 1 selected Random Forests independently. Their initial
fitted models use 2010–2014. Each task/strategy has an independent budget/state.

| Strategy | Trigger | Retraining target window |
| --- | --- | --- |
| frozen | Never | No replacement |
| periodic | End of 2015, 2017, 2019, 2021, 2023, 2025 | Latest 3 calendar years |
| input_triggered | D's online input alarm | Latest 3 calendar years |
| error_triggered | This component's online error alarm | Latest 3 calendar years |

These are declared design choices before full simulation, not rules optimized
on 2015+ errors. No strategy may replace its model more than six times. Extra
alarms after budget exhaustion are logged and ignored. Periodic dates are fixed
calendar dates. The initial fit does not spend budget. Refit preprocessing and
model together count as one replacement. Future model/threshold tuning must
use only information available at the retraining date.

## Error monitor

For each task, compute daily loss from the already issued 48 forecasts:
demand MAPE (%) or price misclassification (%). The monitored score is the
trailing 28-day mean. Threshold is the 99th percentile of complete 28-day scores
within 2014, with a positive numerical floor. Alarm requires three consecutive
strict exceedances and 90 calendar days since the previous alarm. These are
engineering choices, not a named statistical test or a false-alarm guarantee.
Price misclassification can change because the class proportions change, not
only because the conditional relationship changes. Monthly/annual tables retain
actual and predicted positive shares alongside pooled balanced accuracy. Neither
an error alarm nor an input alarm establishes concept drift or market-event
causation.

`calibrate` clones B/C's selected configurations and fits **only 2010–2013**,
then predicts 2014. It never computes calibration errors using models already
trained on 2014. Configuration selection used 2014, so this calibration is not
independent validation. Deployed initial models refitted through 2014 can differ
from calibration predictors. The fixed reference threshold remains unchanged
after later replacements. Track each strategy's own deployed forecast errors;
do not substitute errors from another strategy.

At deployment, seed the rolling buffer with the last 27 losses of 2014; start
persistence/cooldown counters fresh. Each consecutive day must be observed once.
`reset()` is for a new independent simulation, never a way to rewrite historical
state mid-run. Pickling preserves streaming state for resumption. Caller must
provide truthful calibration-predictor provenance; dates alone cannot verify
how a model was actually trained.

## Day ordering and replacement contract

1. Forecast D with the previously deployed model and information available on D-1.
2. At the end of D, obtain actual outcomes and calculate D's daily loss.
3. Update input/error monitors and pass boolean alarms to `BudgetedPolicy.observe`.
4. If a request is returned, fit with targets from `train_start` through
   `train_end` inclusive. These never include D+1. Historical warm-up context
   before the target window is allowed for lags/rolling features.
5. Atomically deploy the successful replacement for `effective_day = D+1`, then
   `commit(request, model_id=...)`. Fit failure calls `abort`; it does not spend
   budget. Do not advance while a request remains pending.

A failed fit leaves the old model deployed. Production orchestration must log
the failure and decide whether to continue; no silent model replacement is
allowed. Do not retrain on the final simulation day when there is no next
forecast. Full restart support must serialize the forecaster, monitors, budget
controller and output checkpoint consistently; policy serialization alone is
not a complete backtest checkpoint.

## Remaining dependencies

D's generated outputs are not committed. Obtain `outputs/part_d_complete/` or
reproduce using D's README, including day types, input detector and chronological
signals. Budget enforcement belongs to E. Final model-size/interface compliance,
live-week forecast evidence and full adaptation metrics remain to be completed.
The live-week package must not retrain.

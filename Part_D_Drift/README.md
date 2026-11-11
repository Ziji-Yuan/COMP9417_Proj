# Part D — Day types and input drift

This component consumes Part A's public interfaces. It implements fixed daily
demand clustering, chronological input-only drift monitoring, and an optional
retrospective join to predictions from B/C. It does not train their forecasting
models or enforce E's retraining budget.

## Run from the repository root

Use a separate Python 3.11 environment. The core pins below match the official
kit, rather than the root repository's broad NumPy/pandas minimums.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r Part_D_Drift/requirements.txt

python -m Part_D_Drift.run \
  --data ../comp9417_project_kit/data/nem_weather_halfhourly_2010_2026.csv \
  --output outputs/part_d_complete \
  --strict-versions
```

Pass the actual CSV location if the kit is elsewhere. Do not copy or commit the
dataset. Use a new output directory for each run; the CLI refuses nonempty
directories to prevent stale tables from being mistaken for current results.
The CLI limits numerical-library threads to two. On machines where joblib
cannot inspect physical cores, `LOKY_MAX_CPU_COUNT=2` avoids its discovery warning.

The run also fits the alternative profile normalization as a sensitivity
analysis. `--normalization raw` switches the primary representation; the
default is `daily-mean`. Both choices use only training years for PCA and k.

## Modules and ownership

| Module | Purpose |
| --- | --- |
| `profiles.py` | Exact 48-slot profiles, frozen PCA/K-means, novelty distances, cluster shares |
| `drift.py` | Input summaries, reference/calibration fitting, online detector and replay |
| `analysis.py` | Validate B/C predictions; relate errors to types and drift retrospectively |
| `diagnostics.py` | Training-only robustness, pre-2015 detector sensitivity, episodes and descriptive comparisons |
| `market_sources.json` | Primary-source event dates and interpretation limits, used after monitoring only |
| `plots.py` | Export standalone PNG and vector PDF figures with labelled axes and units |
| `run.py` | Reproducible command, tables, figures, artifacts and run metadata |
| `tests/` | Time-boundary, leakage, persistence, restart and metric checks |

Only `Part_A` is imported from another team component. There is no dependency on
unpublished B/C/E modules. A's existing implementation is unchanged.

## Fixed day-type analysis

1. Read `get_data()` from A and map timestamps using A's market-day and slot
   helpers. A day runs from 00:30 through the following 00:00. Reject incomplete,
   duplicate, off-grid or timezone-aware data, and nonpositive demand.
2. Build a days-by-48 matrix. By default divide each row by that day's mean.
   This compares daily *shape* without allowing the overall MW level to dominate.
   It deliberately discards level; raw-demand clustering is exported separately
   for comparison. No per-slot standardization is applied.
3. Fit PCA on **2010-01-01 through 2013-12-31 only**, retaining at least 95% of
   training variance. Use full SVD, without whitening.
4. Fit K-means for k=2,...,8 with 20 initializations and seed 9417. Select the
   highest training silhouette, with smaller k breaking ties. Export every
   candidate's silhouette, inertia and smallest cluster size. This rule is a
   starting choice that should also be assessed for interpretability and tiny
   clusters; it does not establish a uniquely correct number of types.
5. Assign all later days using that same PCA and K-means. The analysis reference
   is never refitted. Cluster numbers have no semantic meaning by themselves.

`nearest_profile_distance` is the minimum RMSE to the training cluster means
in the full 48-dimensional representation. Unlike distance only in retained
PCs, it can detect changes in dimensions discarded by PCA. A day is marked
`novel` above the training 99th percentile of that distance. This is a descriptive
novelty flag, not a calibrated p-value. Both PCA-space distance and reconstruction
RMSE are also saved. The nearest full-space centre need not be the PCA-space
cluster assigned by K-means; these serve different purposes.

Training cluster summaries include mean MW, weekend/winter/summer shares and
peak interval time. Use these and the profile plots to interpret types; do not
assume clusters correspond exactly to weekdays or seasons. Annual share tables
include observed day counts. **2026 contains January–October only**, so apparent
annual differences may partly reflect different seasonal coverage. Monthly
shares support comparisons of matching calendar months.

Additional evidence is generated automatically:

- `pca_sensitivity_training.csv`: repeat training-only selection at 90%, 95%
  and 99% retained variance. `cluster_seed_stability_training.csv` holds the
  primary k fixed and compares three seeds using adjusted Rand agreement.
  These are robustness diagnostics; the primary fitted reference stays fixed.
- `training_calendar_composition.csv`: both P(type | season/day kind) and
  P(season/day kind | type), with their different denominators recorded.
  `training_shape_statistics.csv` includes midday and evening ratios. Calendar
  composition supports interpretation without treating a cluster as a calendar
  label. `mean_daily_peak_hour` averages each day's peak time; the original
  summary's `peak_interval_start_hour` is the peak of the mean profile.
- `yearly_shares_jan_oct.csv`: use January–October in every year. This addresses
  the partial 2026 coverage; leap years still have one extra February day.
  `monthly_shares_calendar_grid.csv` distinguishes unobserved months (NA) from
  observed zero shares. Period-average shape curves also use January–October.

Silhouette values from raw and normalized spaces are not directly comparable
as evidence that one representation is universally better. Choose the
representation for the question: the primary analysis studies shape, while
the raw alternative checks the influence of demand level. A high assignment
share in later years does not imply that those profiles resemble the old type;
always inspect novelty and full-profile distances alongside shares.

The observed cluster for D is known only after D ends. It is suitable for
retrospective error analysis, never as an input to predict D. The detector below
instead uses a lag-one profile already available before the forecast.

## Input detector

`monitor_config()` requests four safe predictor columns from A:
`nsw_demand_lag_1d`, `nsw_price_lag_1d`, `syd_temp`, `syd_solar`.
`input_summaries(X, day_types)` builds these nine daily summaries:

- Lag-one mean demand (MW), midday/daily-mean ratio (11:00–15:00), and
  evening/daily-mean ratio (17:00–21:00), using interval-start times.
- Lag-one profile's distance to the fixed training cluster means.
- Lag-one median price and fraction of negative prices.
- Forecast-day mean/maximum Sydney temperature and mean solar radiation.

The last three weather summaries use the task's explicitly allowed target-day
weather. No current-day demand/price, target label, forecast error or model
prediction enters these summaries. Other states and other weather fields are
not monitored in this first version; changes confined to them may be missed.

Reference: available summary days in **2010–2013** (2010-01-01 lacks a one-day
lag, so the actual reference begins on January 2). Fit month-by-weekend medians
to remove broad seasonal/day-type effects. Subtract those medians and divide
by the reference residual IQR. If that IQR is zero, use residual standard
deviation; if both are zero, use one original feature unit. Seasonal groups
absent from a reference fall back to the overall training median.

For each observed day, compare each feature's last **28 days** of standardized
residuals with its full reference distribution using one-dimensional
Wasserstein-1 distance. The daily score is the **maximum** over the nine
distances. Reference distributions and fitted transformations stay fixed.

Threshold: replay **2014** only, including preceding reference days to warm up
the first window, and take the **99th percentile of the combined score** (with
a tiny positive numerical floor). This calibrates the combined monitor rather
than separately testing nine features at a nominal significance level. Window
length 28, persistence 3 days and cooldown 90 days are declared design choices,
not tuned using post-2014 prediction errors. The 2014 scores are calibration
diagnostics, not a prospective evaluation using an already-known threshold.

Online behavior from **2015-01-01**:

- Initialize the buffer with the last 27 calibration days; start persistence
  and cooldown counters afresh at deployment.
- Process every day exactly once in order. Reject missing, duplicate or
  backward dates rather than changing the window's calendar meaning.
- Require three consecutive scores strictly above the threshold to alarm.
- Require at least 90 calendar days between alarms. A persistent shift can
  produce another alarm after cooldown; below-threshold days reset persistence.
- An alarm observed at the end of D has `effective_day = D + 1`.
- The detector does not cap alarms at six. E decides which alarms actually
  lead to retraining and enforces the per-task/per-strategy six-retrain budget.

**Interpretation limits:** overlapping windows and autocorrelation mean the
99th percentile does not provide a guaranteed 1% future false-alarm rate.
Calendar adjustment reduces, but does not eliminate, seasonality. Marginal
distances can miss changes in joint dependence. A fixed old reference can stay
different after a predictor is retrained; alarms alone do not prove that another
retrain will help. Input shift does not establish concept drift (`P(Y|X)`),
performance degradation, or a causal effect of a particular market event.

## Handoff to E

```python
import joblib
from Part_D_Drift import input_summaries

day_types = joblib.load("outputs/part_d_complete/artifacts/day_types.joblib")
detector = joblib.load("outputs/part_d_complete/artifacts/input_detector.joblib")

# X_D is the exact 48-row safe feature matrix used to forecast day D.
# It must include the four monitor columns above. It can be produced with
# Part_A.features.get_inference_features(..., config=monitor_config()).
# At the end of D, after evaluating its forecasts:
summary_D = input_summaries(X_D, day_types)
D = summary_D.index[0]
signal = detector.update(D, summary_D.iloc[0])
if signal.alarm:
    # E checks its own remaining budget, retrains using data <= D, and deploys
    # the replacement from signal.effective_day (D+1). D does not retrain it.
    pass
```

The exported detector is reset to the end of 2014. Start updates at 2015-01-01,
or replay every intervening day before starting a later evaluation segment.
Call `reset()` for each independent strategy run; it never refits parameters.
If resuming an interrupted run, serialize the current detector with joblib and
resume on the next day. Do not reset a detector in the middle of a run.

Alternatively, use `input_drift_daily.csv` for a historical backtest: index the
row for the observed day D, then respect its `effective_day`. It was produced
chronologically with a threshold fixed before 2015. It is not a forecast of
future alarms. The live-week package does not need to run a detector because
live-week model parameter updates are prohibited.

## Handoff from B/C

Export one row per half-hour in long format:

```text
timestamp,task,model,strategy,y_true,y_pred
```

- `task`: exactly `demand` or `price`.
- `model`: algorithm/configuration identifier; separate frozen models must
  have distinct identifiers. `strategy` defaults to `frozen` if omitted.
- `timestamp`: the official interval-end timestamp. Include all 48 intervals
  of every supplied day, including its following midnight.
- `y_true`: A's `nsw_demand_mw` or `nsw_price_up` target as appropriate.
- `y_pred`: MW for demand; hard 0/1 labels for price (not probabilities).
- Supply dates in the 2015–October 2026 simulation. Different models can have
  different coverage, but comparisons need matching periods to be meaningful.

```bash
python -m Part_D_Drift.run \
  --data ../comp9417_project_kit/data/nem_weather_halfhourly_2010_2026.csv \
  --predictions results/demand_predictions.csv \
  --predictions results/price_predictions.csv \
  --output outputs/part_d_with_predictions \
  --strict-versions
```

Prediction truth is checked against A's targets. Duplicate/incomplete days,
invalid price labels and out-of-period observations fail explicitly. Outputs
include daily and monthly metrics, errors by observed cluster, and descriptive
monthly Spearman associations with type shares, novelty, profile distance and
input score. Monthly context uses each model's actual prediction-day coverage,
so a partial month is not compared with full-month type shares by accident.
Balanced accuracy is computed across the period's actual half-hours, not by
averaging daily balanced accuracies. Monthly association estimates have no
causal interpretation and can be confounded by seasonality and common trends.
No detector parameter is changed based on these analysis results.

Without B/C prediction files, this stage is explicitly marked
`pending_B_C_predictions`; no model-performance numbers are fabricated.
Monthly confidence intervals and complete forecasting backtests remain with
A/E as agreed in the team split.

## Outputs and review

The current result directory is `outputs/part_d_complete/`. The earlier
`outputs/part_d/` was superseded and removed after its tables, figures and
saved objects were verified identical to their counterparts in the current run.
The CLI default and the handoff examples both use the current directory.

The spec requires analysis, appropriate figures/results in the report, and
reproducible experiment code. It does not prescribe this output directory or
require every diagnostic CSV/PNG/PDF as a separate deliverable. The two local
`.joblib` files are reusable D objects for analysis and E's backtest. They are
not by themselves the final submission's trained demand/price models.
The final `submission.zip/artifacts/` directory is separately required by
spec p. 6 and must contain the fitted objects actually needed by `Forecaster`.


- `tables/`: assignments, annual/monthly shares, selection diagnostics,
  raw-vs-normalized sensitivity, input summaries, calibration scores, daily
  scores/alarms, yearly alarm counts and the independent diagnostics below;
  optional B/C comparisons.
- `figures/`: training profiles, annual shares, novelty trend, PCA/k selection,
  daily detector score and per-feature distances; matched seasonal coverage,
  calendar interpretation, past-only sensitivity, input/day-type alignment and
  market-event windows. Each figure has PNG and vector PDF versions.
- `artifacts/`: fixed day-type model and a freshly reset input detector.
- `run_summary.json`: CSV SHA-256, periods, seed, configurations, selected k,
  environment versions, threshold, distinct exceedance-run count, repeated
  alarms and pending handoffs.
- `market_sources.json`: a copy of the interpretation-only source catalogue.

### Detector checks and interpretation

`detector_sensitivity_2014.csv` compares nine one-factor configurations around
the primary rule: windows 14/28/56 days, quantiles .975/.99/.995, persistence
1/3/7 days and cooldown 30/90/180 days. Every comparison uses reference
2010–2013, calibration January–June 2014 and validation July–December 2014.
The table records actual boundaries, thresholds, exceedances and alarms.
This is a retrospective robustness check on an already declared rule, not an
automatic search selecting the best post-2014 result. The deployed primary
detector still calibrates its fixed threshold on the full year 2014.

No ground-truth stable period or drift dates are supplied. Therefore the
validation exceedance share is **not** a measured false-positive rate, and the
99th percentile is not a 1% false-alarm guarantee. The half-year split also has
seasonal coverage differences, even after the reference seasonal adjustment.
If all candidates have no validation exceedances, that check cannot rank
persistence/cooldown settings or establish sensitivity to actual drift; the
maximum score/threshold ratio still shows how close each came to the boundary.

`input_drift_episodes.csv` groups consecutive threshold exceedances, including
runs too short to alarm. `right_censored` means a run continues to the dataset
boundary. These runs are not estimates of independent real-world change
points. `input_alarm_attribution.csv` distinguishes the first alarm in a run
from repeat alarms inside that same run. The largest and second-largest
distances describe which monitored features dominate the numerical score;
they are not causal feature attributions. `detector_reference_scales.csv`
exposes small reference scales: a rare training feature, especially negative
price share, can dominate later standardized distances.

`day_type_input_context_daily.csv` explicitly aligns input day D with the
observed profile on D-1. Its monthly counterpart and figure 12 compare type
shares, novelty, and demand/price/weather component scores without needing
B/C predictions. Some relationships are mechanical because profile distance
is itself a detector input. This is evidence about input distributions, not
about model accuracy or a change in P(Y | X).

### Event-source evidence

The catalogue contains AEMO's five-minute settlement and June 2022 suspension
sources, AGL's Liddell retirement release, and AER's solar/negative-price
background. Each entry distinguishes a sourced fact, a hypothesis to inspect
and a limitation. Open the linked originals before citing them in the report.

`market_event_comparisons.csv` gives 28 days before each event and 28 days
starting on the event day, along with the corresponding previous-year
windows. Counts and mean scores are descriptive. The nearest-alarm gap is
signed: negative means the alarm preceded the event. Figure 13 shows actual
daily scores in these windows. Inspect the pre-event exceedance share and
episode table before calling a nearby alarm a new change. Previous-year
windows do not constitute a causal control group, and the event set is not
an exhaustive chronology of the electricity market.

The catalogue is loaded only after primary fitting and chronological replay.
No network call or external numerical dataset is used by this component.

### Spec coverage and remaining team dependencies

| Spec requirement | Evidence / status |
| --- | --- |
| p. 4: 48-value profiles; justify normalization and k | `profiles.py`, raw alternative, selection and training robustness tables |
| p. 4: fit PCA and clustering on 2010–2013, then keep fixed | `DayTypeModel.fit`, serialized reference, future-perturbation tests |
| p. 4: interpret clusters and yearly prevalence; identify unusual days | Profile/calendar figures, yearly and matched-coverage shares, full-space novelty |
| pp. 4–5: relate types to input detection | Correctly lagged daily/monthly context and figure 12 |
| p. 5: online input detector, past-only threshold, alarm dates, D+1 timing | `drift.py`, calibration/holdout tables, dated signals, streaming and leakage tests |
| p. 5: plausible market events; distinguish evidence from coincidence | Sourced catalogue, event windows, episode/attribution tables and explicit limits |
| pp. 4–5: compare types/alarms with forecasting errors | **Pending B/C predictions**; validated join and metrics interface exists |
| p. 5: error-based detector, retraining strategies, six-retrain budget | **Part E**; D supplies reusable input signals and serialized detector |
| p. 5: monthly model-error 95% intervals by both methods | **A/E**, with B/C predictions; D's optional performance plot is not a replacement |
| p. 10: group-authored report | Figures/tables and implementation documentation are provided; group members write the report |

For a quick review, inspect figures 01/02/03, then 08/10/11/12/13 and the alarm
episode table. No evidence here establishes whether drift-triggered retraining
improves prediction: that requires B/C outputs and E's budgeted backtest.

These generated directories are ignored by the repository's existing
`.gitignore`. Commit source and documentation; coordinate separately how to
share generated results and artifacts. This README documents code and
interfaces; group members should write their own report interpretation after
reviewing actual plots and results.

## Tests

```bash
python -m unittest discover -s Part_D_Drift/tests -t . -v
COMP9417_KIT_ROOT=../comp9417_project_kit \
  python -m unittest discover -s Part_A/tests -v
```

Tests cover midnight alignment, incomplete/off-grid data, future perturbations,
training/inference consistency, frozen learned state, stream/replay equality,
serialization midstream, alarm persistence/cooldown, next-day activation,
external target verification, and monthly balanced-accuracy aggregation.

## Method references

- [scikit-learn PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [scikit-learn silhouette selection example](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
- [SciPy Wasserstein distance documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)

The composite detector, seasonal adjustment and alarm policy are this project's
explicit engineering choices, not an implementation of a named drift algorithm
with a theoretical false-positive guarantee.

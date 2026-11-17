# Part D — Day types and input drift

This component consumes Part A's public interfaces. It implements fixed daily
demand clustering, chronological input-only drift monitoring, and retrospective
joint analysis of B/C's four frozen forecasting models. A handover utility
reconstructs B's missing Ridge artifact using B's existing configuration and
2010–2014 data. D does not select new forecasting configurations or enforce
E's retraining budget.

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
If `outputs/part_d_complete` already exists, keep that fixed reference and run
the joint command below; there is no need to regenerate the independent D run.
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
| `handover.py` | Export B's frozen Ridge/RF forecasts and the two official baselines for each task |
| `evaluation.py` | Appendix A intervals, pooled metrics, calendar checks and alarm/error comparisons |
| `joint.py` | Join B/C against existing D outputs without refitting the D reference |
| `joint_plots.py` | Six additional figures for frozen forecasting performance and drift associations |
| `diagnostics.py` | Training-only robustness, pre-2015 detector sensitivity, episodes and descriptive comparisons |
| `market_sources.json` | Primary-source event dates and interpretation limits, used after monitoring only |
| `plots.py` | Export standalone PNG and vector PDF figures with labelled axes and units |
| `run.py` | Reproducible command, tables, figures, artifacts and run metadata |
| `tests/` | Time-boundary, leakage, persistence, restart and metric checks |

Only `Part_A` is imported from another team component. The joint runner also
reads B's saved RF and C's published prediction CSV. B/C's source and existing
model files remain unchanged; E is not needed to run this frozen analysis.

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

## Complete frozen B/C joint analysis

From the repository root, with the fixed independent D output already present:

```bash
git lfs pull  # B's RF must be the actual artifact, not an LFS pointer
python -m Part_D_Drift.joint \
  --data ../comp9417_project_kit/data/nem_weather_halfhourly_2010_2026.csv \
  --reference-output outputs/part_d_complete \
  --price-predictions Part_C_Price_Direction/results/price_predictions_for_part_d.csv \
  --demand-rf Part_B_Demand/artifacts/rf_model.joblib \
  --strict-versions
```

The default destination is **`outputs/part_d_complete/joint/`**. It must be
empty or absent. This is a subdirectory of the one current D result directory,
not a replacement reference fit. The runner verifies the source CSV hash and
hashes of the existing D assignments, detector signals, fitted objects and
event catalogue before/after analysis. It never refits them or changes alarms.

The four frozen series are:

- Demand `random_forest`: B's saved RF, loaded once, with its published
  parameters, feature schema and training-row count checked. Tree artifacts
  do not retain training dates; B's training source supplies that provenance.
- Demand `ridge_regression`: reconstruct `StandardScaler` + `Ridge(alpha=1000)`
  from `Part_B_Demand/train_demand.py`. Fit only the 87,312 available feature
  rows from 2010-01-08 through 2014-12-31, then freeze throughout simulation.
  The first seven days lack required lags. This rebuild does not repeat model
  selection using the simulation period.
- Price `logistic_regression` and `random_forest`: read C's exported hard-label
  forecasts. Both are retained even if their historical ranking differs from
  the 2014 validation ranking; D does not reselect C's final model.

Both yesterday and last-week official baselines are generated for each task.
For **both price baselines**, the threshold is yesterday's actual mean price;
the last-week baseline does not use last week's mean as the threshold.
B's forecasts are emitted in chronological daily order. Four representative
days check that A's bulk causal features exactly equal its day-ahead inference
interface using only past market history and permitted target-day weather.

Every series must cover **2015-01-01 through 2026-10-31: 4,322 complete market
days, or 207,456 half-hours**. All supplied targets are checked against A's
official targets. Missing/duplicate intervals, invalid price labels, wrong
targets and incomplete simulation coverage fail explicitly. The generic
`analysis.py` interface still supports partial coverage, while this complete
joint runner requires identical full-period coverage for all eight series.

### Metrics and uncertainty

`overall_performance`, `yearly_performance`, `monthly_performance` and
`daily_performance` provide demand MAPE/MAE and price Accuracy/BalancedAccuracy.
Balanced accuracy pools actual half-hours within each period; it is not the
mean of daily balanced accuracies. Type/novelty tables also include counts and
price class proportions to expose composition differences.

`monthly_confidence_intervals.csv` implements **both Appendix A methods** for
each model and baseline in all 142 months:

1. Independent half-hours: MAPE uses mean APE ± 1.96 sample standard deviation
   / sqrt(n); Accuracy uses the normal binomial interval. Price Hoeffding
   bounds are additionally exported. Analytic bounds are stored untruncated.
2. Whole-day bootstrap: resample complete 48-interval days with replacement,
   using **2,000 replicates** and the 2.5th/97.5th percentiles. For these mean
   metrics, resampling day means is exactly equivalent to concatenating all
   intervals of the sampled complete days. Stable per-series/month seeds make
   results independent of row/group traversal order.

The estimand is expected performance of the fixed model under conditions
resembling that observed month, assuming within-month stability. The realized
month's score is known exactly. These are not live-week prediction intervals
or formal tests that two months differ. Day resampling preserves within-day
dependence but assumes independent, exchangeable days; cross-day dependence
can still invalidate that approximation. `confidence_interval_widths.csv`
compares the two widths without presuming the day interval is always wider.

### Type/error and alarm/error evidence

- Pooled error tables split by observed type and novelty, plus year-by-type
  and year-by-novelty tables. These are retrospective attributes available
  after the target day; they never enter that day's forecasts.
- `calendar_checked_associations.csv` compares monthly loss with novelty
  share, profile distance, input score and Type 0 share. It reports raw
  Spearman coefficients, correlations of ranks after regressing out year and
  calendar-month fixed effects, and Spearman coefficients of consecutive
  monthly changes. Positive always means **higher loss**: demand MAPE or price
  misclassification percentage. This differs from the old
  `descriptive_associations.csv`, which uses price Accuracy and thus the
  opposite sign. The checks do not eliminate all confounding or establish
  causation; no iid correlation p-values are presented.
- `within_month_novelty.csv` compares novel and ordinary days inside the same
  month. Its summary weights eligible months equally and requires at least
  two days of each kind. Both raw loss gaps and gaps in excess loss relative
  to the same day's yesterday baseline are exported.
- `alarm_performance_windows.csv` compares the 28 days ending on alarm day D
  with D+1 through D+28. Only complete windows enter its summary. Forecasts
  remain frozen in both windows. First alarms and repeat cooldown alarms
  within the same exceedance run are reported separately. A positive change
  is an observed deterioration, **not** a detector true positive. Baseline-
  relative changes help distinguish model-specific deterioration from days
  that are also harder for the simple baseline.
- `market_event_performance.csv` adds model-error windows around the sourced
  events, with matching previous-year windows. These are contextual
  comparisons, not causal controls. Future outcomes are used only for this
  retrospective analysis, never to change past alarms or fitted parameters.

The legacy `run --predictions ...` option performs only the basic join and
metrics. Use `joint` for the complete frozen analysis, confidence intervals and
calendar/window checks. E's error-based monitor, retraining comparisons and
live-week strategy recommendation remain separate team work.

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
- `joint/predictions/`: B's two half-hour forecast series and the four baseline
  series as gzip CSVs. C's existing CSV is read in place, without duplication.
- `joint/artifacts/demand_ridge_rebuilt.joblib`: B's reconstructed frozen Ridge.
  This is a local reproducibility artifact; final submission packaging belongs
  to A/E and must follow the submission's size/interface requirements.
- `joint/tables/`: complete frozen metrics, Appendix A intervals, calendar,
  novelty, alarm-window and market-event error comparisons.
- `joint/figures/`: six new evidence figures, each in PNG and PDF formats.
- `joint/demand_handover.json` and `joint/run_summary.json`: model/data/source
  hashes, fit periods, feature checks, coverage, uncertainty settings and limits.

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
| pp. 4–5: compare types/alarms with forecasting errors | **Complete for all four frozen B/C models**; full coverage, pooled/calendar-controlled comparisons, alarm windows |
| p. 5: error-based detector, retraining strategies, six-retrain budget | **Part E**; D supplies reusable input signals and serialized detector |
| p. 5 and Appendix A: monthly model-error 95% intervals by both methods | **Complete for frozen B/C + both baselines** in `joint`; E must add corresponding evidence for adaptation strategies |
| p. 10: group-authored report | Figures/tables and implementation documentation are provided; group members write the report |

For independent D evidence, inspect figures 01/02/03, then 08/10/11/12/13 and
the alarm episode table. For the completed joint analysis:

| Figure in `joint/figures/` | Question it supports | Report use |
| --- | --- | --- |
| 07: frozen monthly performance and both CIs | How performance and uncertainty vary over time, against both baselines | Main frozen-model evidence; coordinate with B/C/E to avoid duplication |
| 14: pooled errors by type and novelty | Which observed groups have higher error? | Descriptive support; pair with calendar checks and group counts |
| 15: input drift aligned with errors | Do input shifts, novelty and performance move together over time? | Main D joint-analysis candidate |
| 16: errors before/after alarms | Are alarms consistently followed by worse frozen performance? | Supports limitations; use first/repeat counts from the accompanying table |
| 17: raw and calendar-checked associations | Does an apparent relationship survive simple time-trend checks? | Main interpretation check; positive means more error |
| 18: sourced market-event error windows | What happens to errors near selected historical events? | Context if space allows; no causal attribution |

Do not include all figures automatically: the main report has a page limit.
Select evidence covering the required questions and write your own explanation.
No evidence here establishes whether drift-triggered **retraining** improves
prediction: that requires E's budgeted adaptation experiments.

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
The joint tests also cover the Appendix A formulas, complete-day bootstrap,
deterministic seeds, midnight-to-month boundaries, reconstruction without future
training rows, official baseline thresholds, calendar-confounding examples,
and exact pre/post alarm and event windows.

## Method references

- [scikit-learn PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [scikit-learn silhouette selection example](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
- [SciPy Wasserstein distance documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)

The composite detector, seasonal adjustment and alarm policy are this project's
explicit engineering choices, not an implementation of a named drift algorithm
with a theoretical false-positive guarantee.

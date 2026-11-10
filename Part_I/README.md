# Part I — Shared Day-Ahead Feature Engineering

This package provides the reusable, leakage-free data and feature contract for
the demand, price-direction, drift-analysis and chronological-backtesting work.
It does not train models or determine the best feature subset.

## Published modules

- `data.py`: CSV loading, schema validation, official market-day mapping,
  half-hour slots and chronological selection.
- `features.py`: `FeatureConfig`, both official targets, configurable training
  features and next-day inference features.
- `tests/`: portable synthetic unit/leakage tests plus optional integration
  tests against the official project kit.

The exploratory notebook, `eda.py`, EDA figures and generated feature caches
remain outside this shared repository.

## Data and time conventions

Timestamps mark the **end** of each half-hour interval in fixed market time
(AEST, UTC+10, without daylight saving). Market day `D` contains timestamps
`D 00:30` through `D 23:30` and `(D+1) 00:00`; midnight belongs to the preceding
market day. Slots are numbered 0–47 from the interval start time.

`start` and `end` are inclusive **target market-day** dates. `get_features()`
retains source rows before `start` as warm-up context. At prediction time,
historical market observations must precede the target day. The 12 target-day
weather fields are allowed because the official harness supplies them through
`weather_next`.

## Public API

```python
from Part_I.features import (
    FeatureConfig,
    get_data,
    get_features,
    get_inference_features,
)
```

### FeatureConfig

```python
FeatureConfig(
    states=("nsw", "qld", "vic"),
    demand_lags=(1, 7),
    price_lags=(1, 7),
    rolling_windows=(7,),
    rolling_stats=("mean", "std"),
    include_calendar=True,
    include_cyclical=True,
    include_weather=True,
    weather_columns=None,              # None selects all 12
    include_prev_day_price_mean=True,
    drop_rows_with_missing_history=True,
)
```

The default configuration produces **47 features**: 10 calendar/cyclical, 12
weather, 12 market lags, 12 rolling statistics and one previous-day price
reference. Lists are normalised into deterministic state, weather, lag and
window ordering.

### get_data

```python
get_data(start=None, end=None, *, data=None, path=None) -> DataFrame
```

Loads or reuses the observations, validates complete 48-slot market days and
returns the 18 original variables plus `market_day` and `slot`. This analysis
interface may contain outcomes and is **not** itself a leakage-free predictive
feature matrix.

The default loader checks `data/nem_weather_halfhourly_2010_2026.csv`. Callers
can instead pass an explicit CSV `path=` or a preloaded DataFrame through
`data=`. Importing `Part_I` never requires the CSV.

### get_features

```python
get_features(start=None, end=None, config=None, *, data=None, path=None)
    -> (X, y, meta)
```

- `X`: deterministic numeric feature columns indexed by timestamp.
- `y`: `nsw_demand_mw` and `nsw_price_up`.
- `meta`: timestamp, target `market_day`, `slot`, and the previous-day NSW
  price reference.

Invalid targets are never filled with zero. Rows missing configured history
are removed by default. Set `drop_rows_with_missing_history=False` to preserve
feature `NaN` values for a compatible downstream method; invalid target rows
remain excluded.

### get_inference_features

```python
get_inference_features(history, weather_next, config=None) -> DataFrame
```

Uses only the official inputs, never reads the local CSV or network, and
returns exactly 48 rows with the ordered `weather_next.index`. The training and
inference interfaces share one internal feature builder.

## Feature dictionary

| Family | Naming | Definition and availability |
| --- | --- | --- |
| Calendar | `slot`, `day_of_week`, `month`, `is_weekend` | Target day and slot; known in advance. |
| Cyclical | `slot_sin/cos`, `dow_sin/cos`, `month_sin/cos` | Deterministic calendar encodings. |
| Weather | Original names such as `syd_temp` | Target-day `weather_next`; officially allowed. |
| Market lag | `{state}_{demand\|price}_lag_{L}d` | Same slot exactly `L` market days earlier. |
| Rolling mean | `{variable}_roll_{W}d_mean` | Same slot over market days `T-W` through `T-1`. |
| Rolling std | `{variable}_roll_{W}d_std` | Sample standard deviation (`ddof=1`) over the safe window. |
| Price reference | `nsw_price_prev_day_mean` | Mean NSW price over market day `T-1`. |

Rolling statistics cover demand and price for each selected state. A rolling
value is `NaN` unless all `W` historical same-slot values exist. Because sample
standard deviation uses `ddof=1`, configurations requesting `std` require
windows of at least two days. Lag units are market days, not row counts.

## Official targets

- `nsw_demand_mw`: observed NSW operational demand in the target interval.
- `nsw_price_up`: 1 if target-interval NSW price exceeds the mean NSW price of
  the immediately preceding market day, otherwise 0.

## Usage by component

### B — demand regression

```python
config = FeatureConfig(demand_lags=[1, 7, 14], rolling_windows=[7, 14])
X, y, meta = get_features("2010-01-01", "2013-12-31", config)
y_demand = y["nsw_demand_mw"]
```

### C — price-direction classification

```python
config = FeatureConfig(price_lags=[1, 7, 14], rolling_windows=[7, 14])
X, y, meta = get_features("2010-01-01", "2013-12-31", config)
y_price = y["nsw_price_up"]
```

### D — aligned observed history

```python
observed = get_data("2015-01-01", "2015-12-31")
```

### E — reusable expanding or sliding windows

```python
observed = get_data()  # load once
X_exp, y_exp, m_exp = get_features("2010-01-01", "2018-12-31", config, data=observed)
X_slide, y_slide, m_slide = get_features("2016-01-01", "2018-12-31", config, data=observed)
```

Warm-up rows before the requested `start` remain available from the preloaded
frame, so sliding-window targets retain valid lag/rolling context.

### Official next-day inference

```python
X_next = get_inference_features(history, weather_next, config)
```

`X_next` has 48 rows and the same feature schema as the corresponding training
matrix. It generates features only; it neither fits a model nor predicts.

## Leakage prevention and downstream preprocessing

Safe inputs are target calendar fields, supplied target-day weather and market
observations strictly before the target day. Unsafe inputs include target-day
NSW/QLD/VIC demand or price, contemporaneous interstate market values, future
rows and rolling windows containing `T`. The implementation uses explicit
`(market_day, slot)` lookups rather than unvalidated row shifts.

Scaling, imputation, feature selection and other model-specific preprocessing
belong to B/C. `StandardScaler`, `RobustScaler` and similar fitted transforms
must be fitted only on the relevant chronological training window and reused
unchanged for validation/inference. This package performs no fitted global
preprocessing, PCA or model-based selection.

## Tests

Portable tests require only NumPy and pandas:

```bash
python -m unittest discover -s Part_I/tests -v
```

Synthetic tests cover market-day boundaries, lag 1/7, rolling calculations,
targets, missing-history policy, deterministic columns and leakage
perturbations.

Optional integration tests compare both targets and training/inference feature
matrices with the official `harness/dayahead.py` on ordinary, leap-day and year
boundaries. Provide the project kit explicitly:

```bash
COMP9417_KIT_ROOT=/path/to/comp9417_project_kit \
python -m unittest discover -s Part_I/tests -v
```

Alternatively set `COMP9417_DATA_PATH` to the official CSV. If the harness or
dataset is absent, integration tests are reported as **skipped**, never passed.

## Limitations

Candidate generation does not establish predictive usefulness. B/C remain
responsible for feature selection, chronological validation and model-specific
processing. D/E remain responsible for drift and backtesting policy. This
module includes no models, tuning, adaptation strategy, evaluation framework
or final submission package.

# COMP9417 Project — Electricity Forecasting under Drift

This private group repository supports the COMP9417 project on forecasting
NSW electricity demand and next-day price direction under temporal drift.

## Team components

- **Part A — Shared Feature Engineering:** common data alignment, targets,
  leakage-free features and training/inference consistency. Available now.
- **Part B — Demand:** NSW demand regression. See [Part B documentation](Part_B_Demand/README.md).
- **Part C — Price Direction:** NSW price-direction classification. See [Part C Documentation](Part_C_Price_Direction/README.md)
- **Part D — Drift:** fixed day-type clustering, online input drift detection,
  and reusable interfaces for E. See [Part D documentation](Part_D_Drift/README.md).
  Frozen B/C error/drift comparisons and monthly confidence intervals are
  available through `python -m Part_D_Drift.joint`; results go under
  `outputs/part_d_complete/joint/`. Adaptation comparisons are available in E.
- **Part E — Backtesting:** chronological evaluation and retraining strategies.
  Full 2015–October 2026 comparisons, error monitoring, final candidate and
  live-week forecast evidence are implemented. See [Part E documentation](Part_E_Backtesting/README.md),
  [reviewable results](Part_E_Backtesting/evidence/README.md) and
  [integration handover](Part_E_Backtesting/HANDOVER.md). Linux verification
  and the group-authored report remain.

## Repository layout

```text
Part_A/                  Shared data and feature-engineering package
Part_B_Demand/           Demand work area
Part_C_Price_Direction/  Price-direction work area
Part_D_Drift/            Drift-analysis work area
Part_E_Backtesting/      Backtesting work area
```

See [Part A documentation](Part_A/README.md) for the data contract, feature
dictionary, public APIs, leakage rules and tests.

## Quick start

```bash
python -m pip install -r requirements.txt
```

The official dataset is intentionally not committed. Place a local copy at
`data/nem_weather_halfhourly_2010_2026.csv` or pass an explicit `path=` or
preloaded `data=` argument.

```python
from Part_A.features import FeatureConfig, get_features

config = FeatureConfig(demand_lags=[1, 7, 14], rolling_windows=[7, 14])
X, y, meta = get_features("2010-01-01", "2013-12-31", config)

# Part B
y_demand = y["nsw_demand_mw"]

# Part C
y_price_direction = y["nsw_price_up"]
```

## Shared development workflow

Each contributor should create a feature branch from the latest `main`, keep
changes scoped to their component, run relevant tests, and submit changes for
group review before merging. Do not commit raw data, credentials, local caches
or trained artifacts unless the team explicitly agrees on an exception.

"""Daily prediction/evaluation/replacement ordering, independent of model family."""

from dataclasses import asdict
import time

import numpy as np
import pandas as pd

from .policy import BudgetedPolicy, daily_loss


def simulate(X, truth, days, initial_model, monitor, config, *, input_signals=None, fit_model,
             progress=None):
    """Run a complete contiguous prefix from deployment start to config.end_day.

    fit_model(request) returns a *new* fitted model. The engine supplies only the
    current day's X to prediction; the runner enforces fit slicing. Model errors
    from this exact deployed strategy feed its own monitor. Nothing is fitted on
    the final day because no subsequent forecast can benefit.
    """
    days = pd.DatetimeIndex(days)
    expected = pd.date_range(config.start_day, config.end_day).repeat(48)
    if not days.equals(expected) or len(X) != len(days) or len(truth) != len(days):
        raise ValueError("Require 48 ordered rows per contiguous simulation day")
    if not X.index.equals(truth.index):
        raise ValueError("Features and truth must share the same ordered index")
    if monitor.last_day_ != pd.Timestamp(config.start_day) - pd.Timedelta(days=1):
        raise ValueError("Monitor must start immediately before simulation")
    if input_signals is not None:
        if not input_signals.index.equals(pd.date_range(config.start_day, config.end_day)):
            raise ValueError("Input signals must exactly cover simulation days")
        if not input_signals.alarm.map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise ValueError("Input alarms must be real booleans, not CSV strings")
        effective = pd.DatetimeIndex(pd.to_datetime(input_signals.effective_day))
        if not effective.equals(input_signals.index + pd.Timedelta(days=1)):
            raise ValueError("Input alarm must be effective the following day")
    if config.strategy == "input_triggered" and input_signals is None:
        raise ValueError("Input-triggered strategy needs verified D signals")
    policy = BudgetedPolicy(config)
    current, model_id = initial_model, "stage1_refit_2010_2014"
    prediction = np.empty(len(days), dtype=float)
    model_ids, monitor_records, retrains = [], [], []
    for i, day in enumerate(pd.date_range(config.start_day, config.end_day)):
        row = slice(i * 48, (i + 1) * 48)
        # Forecast first. No day-D target enters fitting or monitoring before this.
        values = np.asarray(current.predict(X.iloc[row]))
        loss = daily_loss(monitor.task, truth.iloc[row].to_numpy(), values)
        if monitor.task == "demand" and ((values < 0).any() or (values > 20000).any()):
            raise ValueError("Demand prediction violates official output limits")
        prediction[row] = values
        model_ids.extend([model_id] * 48)
        signal = monitor.update(day, loss)
        monitor_records.append({**signal, "loss": loss, "model_id": model_id})
        request = policy.observe(day, input_alarm=False if input_signals is None else input_signals.loc[day, "alarm"],
                                 error_alarm=signal["alarm"])
        if request:
            start = time.perf_counter()
            try:
                replacement = fit_model(request)
                if replacement is current:
                    raise ValueError("Fitting must return a new model, not mutate the deployed one")
            except Exception as exc:
                policy.abort(request, reason=repr(exc))
                # Fail loudly: do not publish an apparently successful simulation.
                raise
            new_id = f"{config.strategy}_{request.ordinal}_{request.effective_day.date()}"
            policy.commit(request, model_id=new_id)
            current, model_id = replacement, new_id
            retrains.append({**asdict(request), "model_id": model_id,
                             "fit_seconds": time.perf_counter() - start})
            if progress:
                progress(f"{monitor.task}/{config.strategy} retrain {policy.count}: observed {day.date()}, deploy {request.effective_day.date()}")
        if progress and (day.month == 12 and day.day == 31):
            progress(f"{monitor.task}/{config.strategy}: completed {day.year}, retrains={policy.count}")
    return dict(prediction=prediction, model_ids=model_ids, signals=pd.DataFrame(monitor_records),
                retrains=pd.DataFrame(retrains), audit=pd.DataFrame(policy.audit),
                retrain_count=policy.count, final_model=current, final_monitor=monitor, final_policy=policy)

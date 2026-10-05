"""Causal raw-meter preparation: no interpolation and no future measurements."""

import csv
import io

import numpy as np
import pandas as pd

from src.regression_common import FEATURES

HISTORY_ROWS = 672
MAX_ROWS = 40000
MAX_BYTES = 8 * 1024 * 1024


class InputError(ValueError):
    """An actionable, public input validation error."""


def validate_history(history, minimum=HISTORY_ROWS):
    if not minimum <= len(history) <= MAX_ROWS:
        raise InputError(f"Provide {minimum} to {MAX_ROWS:,} readings, one every 15 minutes.")
    index = history.index
    if index.hasnans or index.tz is not None:
        raise InputError("Use valid local plant timestamps without a timezone offset.")
    if index.duplicated().any():
        raise InputError("Duplicate timestamps found. Keep exactly one verified reading per interval.")
    if not index.is_monotonic_increasing:
        raise InputError("Readings must be ordered from oldest to newest.")
    if ((index.minute % 15 != 0) | (index.second != 0) | (index.microsecond != 0) | (index.nanosecond != 0)).any():
        raise InputError("Timestamps must fall on :00, :15, :30 or :45 with no seconds.")
    if not (index.to_series().diff().iloc[1:] == pd.Timedelta(minutes=15)).all():
        raise InputError("There is a gap between readings. Supply consecutive 15-minute intervals; missing readings are not guessed.")
    values = history.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise InputError("Usage_kWh must contain only finite, nonnegative numbers. Blank, negative, NaN and infinite values are not allowed.")
    # A finite bound prevents arithmetic overflow, not an asserted factory limit.
    if (values > 1_000_000).any():
        raise InputError("A reading exceeds 1,000,000 kWh per interval. Check the units and decimal separator.")
    return history


def parse_meter_csv(content: bytes, convention="iso", minimum=HISTORY_ROWS):
    if len(content) > MAX_BYTES:
        raise InputError("The CSV must be smaller than 8 MB.")
    try:
        text = content.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(text)))
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise InputError("Provide a nonempty UTF-8, comma-separated CSV file.") from exc
    if len(header) != len(set(header)):
        raise InputError("Duplicate column names found in the CSV header.")
    time_column = "timestamp" if convention == "iso" else "date"
    if time_column not in header or "Usage_kWh" not in header:
        raise InputError(f"The CSV needs {time_column} and Usage_kWh columns (case-sensitive).")
    try:
        frame = pd.read_csv(io.StringIO(text), nrows=MAX_ROWS + 1)
        stamps = frame[time_column].astype(str)
        if convention == "iso":
            pattern = r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:00)?"
            if not stamps.str.fullmatch(pattern).all():
                raise InputError("Use local ISO timestamps, for example 2018-11-19 09:45:00. No timezone suffix or midnight date shifting is applied.")
            index = pd.DatetimeIndex(pd.to_datetime(stamps, format="ISO8601"))
        elif convention == "uci":
            index = pd.DatetimeIndex(pd.to_datetime(stamps, format="%d/%m/%Y %H:%M"))
            midnight = (index.hour == 0) & (index.minute == 0)
            index = index + pd.to_timedelta(midnight.astype(int), unit="D")
        else:
            raise InputError("Choose ISO timestamps or the original UCI date convention.")
        values = pd.to_numeric(frame["Usage_kWh"], errors="raise").to_numpy(dtype=float)
    except (ValueError, pd.errors.ParserError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError("The CSV contains an invalid date, nonnumeric reading, or malformed row.") from exc
    power_columns = ["Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"]
    if all(c in frame for c in power_columns):
        bad = (values == 0) & (frame[power_columns].apply(pd.to_numeric, errors="coerce") == 0).all(axis=1)
        if bad.any():
            first = index[np.flatnonzero(bad)[0]]
            raise InputError(f"Possible meter failure at {first}: energy and both power factors are zero. Supply verified readings or a continuous seven-day window excluding that row. No future-value interpolation is performed.")
    return validate_history(pd.Series(values, index=index, name="Usage_kWh"), minimum)


def build_features(history):
    validate_history(history)
    past = history.iloc[-HISTORY_ROWS:]
    target = past.index[-1] + pd.Timedelta(minutes=15)
    angle = 2 * np.pi * ((target.hour * 60 + target.minute) / 1440)
    day = 2 * np.pi * target.dayofweek / 7
    row = dict(
        tod_sin=np.sin(angle), tod_cos=np.cos(angle),
        dow_sin=np.sin(day), dow_cos=np.cos(day),
        is_weekend=int(target.dayofweek >= 5), load_level=0,
        lag_1=past.iloc[-1], lag_4=past.iloc[-4], lag_96=past.iloc[-96], lag_672=past.iloc[-672],
        roll_mean_4=past.iloc[-4:].mean(), roll_std_4=past.iloc[-4:].std(ddof=1),
        roll_mean_96=past.iloc[-96:].mean(), roll_max_96=past.iloc[-96:].max(),
        diff_1=past.iloc[-1] - past.iloc[-2],
    )
    return pd.DataFrame([row], columns=FEATURES, index=pd.DatetimeIndex([target], name="timestamp"))


def history_records(history):
    return [{"timestamp": t.isoformat(), "usage_kwh": float(v)} for t, v in history.items()]


def from_records(records):
    return pd.Series([r["usage_kwh"] for r in records],
                     index=pd.DatetimeIndex([r["timestamp"] for r in records]), name="Usage_kWh")

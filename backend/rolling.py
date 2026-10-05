"""Persistent rolling forecasts, actual reconciliation and measured-data refits."""

import hashlib
import json
from pathlib import Path
from threading import RLock, Thread
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from backend.deployment import fit_deployment, load_deployment, seed_measurements
from backend.features import InputError, build_features, history_records, validate_history
from backend.service import metrics, now

STEP = pd.Timedelta(minutes=15)


def timestamp(value):
    try:
        result = pd.Timestamp(value)
        if pd.isna(result) or result.tz is not None or result != result.floor("15min"):
            raise ValueError()
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        raise InputError("Use a local timestamp on :00, :15, :30 or :45, without a timezone.") from exc


class RollingWorkspace:
    def __init__(self, service, model_directory=None):
        self.service = service
        self.store = service.store
        self.lock = RLock()
        self.directory = Path(model_directory or self.store.path.parent / "deployment")
        self.model, self.model_info = load_deployment(self.directory)
        self.job = dict(status="idle")
        with self.store.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meter_actuals (
                    timestamp TEXT PRIMARY KEY, usage REAL NOT NULL, valid INTEGER NOT NULL, source TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS rolling_forecasts (
                    id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, request_id TEXT NOT NULL,
                    active INTEGER NOT NULL, body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS rolling_time ON rolling_forecasts(timestamp);
                CREATE TABLE IF NOT EXISTS rolling_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
                INSERT OR IGNORE INTO rolling_meta VALUES ('revision', 0);
                CREATE TABLE IF NOT EXISTS actual_imports (id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS actual_changes (
                    import_id TEXT NOT NULL, timestamp TEXT NOT NULL, previous_usage REAL, usage REAL NOT NULL);
            """)
            if not db.execute("SELECT 1 FROM meter_actuals LIMIT 1").fetchone():
                seed = seed_measurements()
                db.executemany("INSERT INTO meter_actuals VALUES (?, ?, ?, '2018 dataset')",
                               [(t.isoformat(), float(r.usage), int(r.valid)) for t, r in seed.iterrows()])

    def measurements(self):
        with self.store.connect() as db:
            rows = db.execute("SELECT timestamp, usage, valid FROM meter_actuals ORDER BY timestamp").fetchall()
        return pd.DataFrame({"usage": [r[1] for r in rows], "valid": [bool(r[2]) for r in rows]},
                            index=pd.DatetimeIndex([r[0] for r in rows]))

    def _history(self):
        with self.store.connect() as db:
            actual = db.execute("SELECT timestamp, usage, valid FROM meter_actuals ORDER BY timestamp DESC LIMIT 672").fetchall()[::-1]
            provisional = db.execute("SELECT body FROM rolling_forecasts WHERE active=1 AND timestamp>? ORDER BY timestamp", (actual[-1][0],)).fetchall()
        values = {r[0]: dict(timestamp=r[0], usage_kwh=r[1], kind="actual" if r[2] else "unverified") for r in actual}
        for (body,) in provisional:
            row = json.loads(body)
            values[row["forecast_timestamp"]] = dict(timestamp=row["forecast_timestamp"], usage_kwh=row["predicted_kwh"], kind="estimated")
        return list(values.values())[-672:]

    def revision(self):
        with self.store.connect() as db:
            return db.execute("SELECT value FROM rolling_meta WHERE key='revision'").fetchone()[0]

    def snapshot(self):
        with self.lock, self.store.connect() as db:
            recent = self._history()
            last_actual = db.execute("SELECT timestamp, usage FROM meter_actuals ORDER BY timestamp DESC LIMIT 1").fetchone()
            rows = db.execute("SELECT COUNT(*), SUM(valid=0) FROM meter_actuals").fetchone()
            count = db.execute("SELECT COUNT(*) FROM rolling_forecasts").fetchone()[0]
            imports = [json.loads(r[0]) for r in db.execute("SELECT body FROM actual_imports ORDER BY rowid DESC LIMIT 10")]
            return dict(revision=self.revision(), actual_rows=rows[0], unverified_rows=rows[1],
                        latest_actual=last_actual[0], latest_actual_kwh=last_actual[1],
                        cursor=recent[-1]["timestamp"], next_start=recent[-1]["timestamp"],
                        next_end=(timestamp(recent[-1]["timestamp"]) + STEP).isoformat(),
                        estimated_inputs=sum(r["kind"] == "estimated" for r in recent),
                        recent=recent[-192:], forecast_count=count, imports=imports,
                        model=self.model_info, training=self.job,
                        max_intervals_per_request=672, timestamp_convention="interval_end",
                        comparison=self.comparison())

    def forecast(self, start, intervals, threshold, revision, request_id):
        with self.lock:
            with self.store.connect() as db:
                existing = [json.loads(r[0]) for r in db.execute("SELECT body FROM rolling_forecasts WHERE request_id=? ORDER BY timestamp", (request_id,))]
            if existing:
                if existing[-1]["request"] != dict(start=start, intervals=intervals, threshold=threshold):
                    raise InputError("This request ID was already used for a different forecast.")
                return dict(forecasts=existing, requested=[r for r in existing if r["requested"]], replayed=True)
            if revision != self.revision():
                raise InputError("The meter history changed in another window. Refresh and try again.")
            if self.model is None:
                raise InputError("The deployment model is not ready. Fit the model from actual readings first.")
            recent = self._history()
            if any(r["kind"] == "unverified" for r in recent):
                raise InputError("The latest seven days contain an unverified meter reading. Upload its verified value first.")
            cursor = timestamp(recent[-1]["timestamp"])
            requested_start = timestamp(start)
            if requested_start < cursor:
                raise InputError("That interval is already behind the forecast cursor. View its saved prediction in the ledger.")
            steps = int((requested_start - cursor) / STEP) + intervals
            if steps > 672:
                raise InputError("This request needs more than seven days of estimates. Choose an earlier interval or upload more recent actual readings.")
            last_actual = timestamp(self.snapshot()["latest_actual"])
            records = []
            for offset in range(steps):
                window = recent[-672:]
                history = pd.Series([r["usage_kwh"] for r in window], index=pd.DatetimeIndex([r["timestamp"] for r in window]))
                inputs = build_features(history)
                started = perf_counter()
                predicted = float(self.model.predict(inputs)[0])
                if not np.isfinite(predicted) or predicted < 0:
                    raise RuntimeError("Model produced an invalid prediction; no forecasts in this request were saved.")
                end = inputs.index[0]
                estimated = sum(r["kind"] == "estimated" for r in window)
                records.append(dict(
                    id=str(uuid4()), created_utc=now(), request_id=request_id,
                    request=dict(start=start, intervals=intervals, threshold=threshold),
                    interval_start=(end - STEP).isoformat(), forecast_timestamp=end.isoformat(),
                    predicted_kwh=predicted, previous_kwh=float(history.iloc[-1]),
                    alert_threshold_kwh=threshold, high_load=predicted >= threshold,
                    estimated_inputs=estimated, measured_inputs=672-estimated,
                    horizon_steps=int((end - last_actual) / STEP),
                    requested=end > requested_start, mode="recursive" if estimated else "measured_history",
                    actuals_through=last_actual.isoformat(), model_sha256=self.model_info["sha256"],
                    training_end=self.model_info["training_end"], inference_ms=(perf_counter()-started)*1000,
                    input_features={k: float(v) for k, v in inputs.iloc[0].items()},
                    history_sha256=hashlib.sha256(history.to_csv().encode()).hexdigest(),
                ))
                recent.append(dict(timestamp=end.isoformat(), usage_kwh=predicted, kind="estimated"))
            with self.store.connect() as db:
                db.executemany("INSERT INTO rolling_forecasts VALUES (?, ?, ?, 1, ?)",
                               [(r["id"], r["forecast_timestamp"], request_id, json.dumps(r, allow_nan=False)) for r in records])
                db.execute("UPDATE rolling_meta SET value=value+1 WHERE key='revision'")
            return dict(forecasts=records, requested=[r for r in records if r["requested"]], replayed=False)

    def import_actuals(self, history, filename, replace=False):
        validate_history(history, minimum=1)
        with self.lock:
            if self.job["status"] == "running":
                raise InputError("A model refit is running. Wait for it to finish before changing actual readings.")
            measured = self.measurements()
            first, last = measured.index.min(), measured.index.max()
            if history.index.min() < first or history.index.min() > last + STEP:
                raise InputError(f"Actual readings must overlap existing history or begin at {(last + STEP).isoformat()}. Gaps are not accepted.")
            overlap = measured.index.intersection(history.index)
            changed = measured.loc[overlap, "usage"].ne(history.loc[overlap]) & measured.loc[overlap, "valid"]
            if changed.any() and not replace:
                raise InputError("Some actual readings differ from saved values. Enable 'Correct existing readings' only if these are verified corrections.")
            identifier = str(uuid4())
            summary = dict(id=identifier, filename=filename, created_utc=now(), rows=len(history),
                           added=int((history.index > last).sum()), corrected=int(changed.sum()),
                           start=history.index.min().isoformat(), end=history.index.max().isoformat())
            with self.store.connect() as db:
                db.executemany("INSERT INTO actual_changes VALUES (?, ?, ?, ?)",
                               [(identifier, t.isoformat(), float(measured.loc[t, "usage"]) if t in measured.index else None, float(v)) for t, v in history.items()])
                db.executemany("INSERT INTO meter_actuals VALUES (?, ?, 1, ?) ON CONFLICT(timestamp) DO UPDATE SET usage=excluded.usage, valid=1, source=excluded.source",
                               [(t.isoformat(), float(v), filename) for t, v in history.items()])
                # Rebuild future inputs from corrected actuals; retain every issued prediction for comparison.
                db.execute("UPDATE rolling_forecasts SET active=0")
                db.execute("INSERT INTO actual_imports VALUES (?, ?)", (identifier, json.dumps(summary)))
                db.execute("UPDATE rolling_meta SET value=value+1 WHERE key='revision'")
            return dict(imported=summary, workspace=self.snapshot())

    def ledger(self, limit=None):
        query = """SELECT f.body, a.usage, a.valid, f.active FROM rolling_forecasts f
                   LEFT JOIN meter_actuals a ON f.timestamp=a.timestamp ORDER BY f.rowid DESC"""
        with self.store.connect() as db:
            rows = db.execute(query + (" LIMIT ?" if limit else ""), (limit,) if limit else ()).fetchall()
        result = []
        for body, usage, valid, active in rows:
            row = json.loads(body)
            actual = usage if valid else None
            result.append({**row, "actual_kwh": actual, "absolute_error_kwh": abs(actual-row["predicted_kwh"]) if actual is not None else None, "active": bool(active)})
        return result

    def comparison(self):
        # Score first-issued forecasts once per target, so repeated forecasts cannot cherry-pick results.
        first = {}
        for row in reversed(self.ledger()):
            if row["actual_kwh"] is not None:
                first.setdefault(row["forecast_timestamp"], row)
        rows = list(first.values())
        def score(group):
            return metrics([r["actual_kwh"] for r in group], [r["predicted_kwh"] for r in group]) if group else None
        return dict(all=score(rows), measured_history=score([r for r in rows if not r["estimated_inputs"]]),
                    recursive=score([r for r in rows if r["estimated_inputs"]]),
                    policy="First issued forecast per interval; measured-history and recursive results separated.")

    def start_training(self):
        with self.lock:
            if self.job["status"] == "running":
                raise InputError("A refit is already running.")
            measurements = self.measurements()
            if self.model_info and self.model_info["data_sha256"] == hashlib.sha256(measurements.to_csv().encode()).hexdigest():
                raise InputError("The model already uses these actual readings. Upload new actuals before refitting.")
            self.job = dict(status="running", started_utc=now(), actuals_through=measurements.index.max().isoformat())
            def run():
                try:
                    model, metadata = fit_deployment(self.service.model, measurements, self.directory)
                    with self.lock, self.store.connect() as db:
                        self.model, self.model_info = model, metadata
                        self.job = dict(status="complete", completed_utc=now(), fit_seconds=metadata["fit_seconds"])
                        db.execute("UPDATE rolling_forecasts SET active=0")
                        db.execute("UPDATE rolling_meta SET value=value+1 WHERE key='revision'")
                except Exception:
                    import logging
                    logging.exception("Deployment refit failed")
                    with self.lock:
                        self.job = dict(status="failed", message="Refit failed. Check the server terminal; the previous model remains available.")
            Thread(target=run, daemon=True, name="forest-refit").start()
            return self.job

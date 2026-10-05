"""FastAPI entry point. Run from the project root with python run_app.py."""

from contextlib import asynccontextmanager
import csv
import io
import logging
import os
from typing import Literal
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.features import InputError, MAX_BYTES, MAX_ROWS, from_records, history_records, parse_meter_csv, validate_history
from backend.service import ROOT, SCENARIOS, ForecastService, now
from backend.rolling import RollingWorkspace


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    dataset_id: str = Field(min_length=1, max_length=80)
    alert_threshold_kwh: float = Field(default=50, gt=0, le=1_000_000, strict=True)


class ReadingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    timestamp: str = Field(max_length=30)
    usage_kwh: float = Field(ge=0, le=1_000_000, strict=True)


class RollingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    start: str = Field(min_length=16, max_length=30)
    intervals: int = Field(default=1, ge=1, le=2, strict=True)
    alert_threshold_kwh: float = Field(default=50, gt=0, le=1_000_000, strict=True)
    revision: int = Field(ge=0, strict=True)
    request_id: str = Field(min_length=8, max_length=80)


class ActualRequest(ReadingRequest):
    replace: bool = False


def csv_response(rows, filename):
    buffer = io.StringIO()
    fields = list(rows[0]) if rows else ["id", "forecast_timestamp", "predicted_kwh"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        safe = {
            k: ("'" + v if isinstance(v, str) and (
                v.lstrip().startswith(("=", "+", "-", "@")) or v.startswith(("\t", "\r"))
            ) else v)
            for k, v in row.items()
        }
        writer.writerow(safe)
    return Response(buffer.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service or ForecastService(os.environ.get("FDM_APP_DB", ROOT / "app_data/steel_energy.sqlite3"))
        app.state.rolling = RollingWorkspace(app.state.service, os.environ.get("FDM_MODEL_DIR"))
        yield

    app = FastAPI(title="Steel Energy Forecast API", version="2.0.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url="/api/openapi.json",
                  description="Local single-interval decision-support prototype. Energy is measured in kWh, not electrical current.")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def guards(request, call_next):
        if request.method in {"POST", "PUT", "DELETE", "PATCH"}:
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin changes are not allowed."}, status_code=403)
            try:
                if int(request.headers.get("content-length", "0")) > MAX_BYTES + 65536:
                    return JSONResponse({"detail": "Request exceeds the 8 MB upload limit."}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Invalid request size."}, status_code=400)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'self'"
        if request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(InputError)
    async def input_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(RequestValidationError)
    async def schema_error(request, exc):
        errors = [f"{'.'.join(str(v) for v in e['loc'][1:])}: {e['msg']}" for e in exc.errors()]
        return JSONResponse({"detail": "; ".join(errors)}, status_code=422)

    @app.exception_handler(Exception)
    async def server_error(request, exc):
        logging.exception("Application request failed")
        return JSONResponse({"detail": "The request could not be completed. Check the server terminal and try again."}, status_code=500)

    def svc():
        return app.state.service

    @app.get("/api/workspace")
    def workspace():
        return app.state.rolling.snapshot()

    @app.post("/api/workspace/forecasts", status_code=201)
    def rolling_forecast(body: RollingRequest):
        return app.state.rolling.forecast(body.start, body.intervals, body.alert_threshold_kwh, body.revision, body.request_id)

    @app.get("/api/workspace/forecasts")
    def rolling_ledger():
        return app.state.rolling.ledger(limit=500)

    @app.get("/api/workspace/forecasts/export.csv")
    def rolling_export():
        keys = ["id", "created_utc", "interval_start", "forecast_timestamp", "predicted_kwh", "actual_kwh", "absolute_error_kwh", "mode", "requested", "horizon_steps", "estimated_inputs", "actuals_through", "model_sha256", "alert_threshold_kwh", "high_load"]
        return csv_response([{k: row[k] for k in keys} for row in app.state.rolling.ledger()], "rolling_forecasts.csv")

    @app.get("/api/workspace/actuals/export.csv")
    def actuals_export():
        frame = app.state.rolling.measurements()
        return csv_response([dict(timestamp=t.isoformat(), Usage_kWh=float(r.usage), verified=bool(r.valid)) for t, r in frame.iterrows()], "actual_readings.csv")

    @app.post("/api/workspace/actuals", status_code=201)
    def import_actuals(file: UploadFile = File(...), replace: bool = Form(False)):
        try:
            content = file.file.read(MAX_BYTES + 1)
        finally:
            file.file.close()
        history = parse_meter_csv(content, "iso", minimum=1)
        name = (file.filename or "actuals.csv").replace("\\", "/").split("/")[-1][:120]
        return app.state.rolling.import_actuals(history, name, replace)

    @app.post("/api/workspace/reading", status_code=201)
    def add_actual(body: ActualRequest):
        from backend.rolling import timestamp
        history = pd.Series([body.usage_kwh], index=pd.DatetimeIndex([timestamp(body.timestamp)]))
        return app.state.rolling.import_actuals(history, "Manual reading", body.replace)

    @app.post("/api/workspace/train", status_code=202)
    def train_deployment():
        return app.state.rolling.start_training()

    @app.get("/api/workspace/template.csv")
    def actual_template():
        return Response("timestamp,Usage_kWh\n", media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="actual_readings_template.csv"'})

    @app.get("/api/health")
    def health():
        return dict(status="ready", model="Random Forest", model_sha256=svc().model_hash, horizon_minutes=15,
                    deployment_ready=app.state.rolling.model is not None,
                    deployment_sha256=(app.state.rolling.model_info or {}).get("sha256"))

    @app.get("/api/model")
    def model():
        return svc().info

    @app.get("/api/datasets")
    def datasets():
        return svc().datasets()

    @app.get("/api/datasets/{identifier}")
    def dataset(identifier: str):
        return svc().describe(svc().dataset(identifier), include_preview=True)

    @app.post("/api/datasets", status_code=201)
    def upload(file: UploadFile = File(...), convention: Literal["iso", "uci"] = Form("iso")):
        try:
            content = file.file.read(MAX_BYTES + 1)
        finally:
            file.file.close()
        history = parse_meter_csv(content, convention)
        name = (file.filename or "Meter readings").replace("\\", "/").split("/")[-1][:120]
        body = dict(id=str(uuid4()), name=name, source="uploaded", convention=convention,
                    readings=history_records(history), created_utc=now())
        with svc().lock:
            try:
                svc().store.save_dataset(body)
            except ValueError as exc:
                raise InputError(str(exc)) from exc
        return svc().describe(body, include_preview=True)

    @app.delete("/api/datasets/{identifier}", status_code=204)
    def remove_dataset(identifier: str):
        if identifier in SCENARIOS:
            raise InputError("Built-in historical samples cannot be deleted.")
        with svc().lock:
            if not svc().store.delete_dataset(identifier):
                raise HTTPException(404, "Dataset not found.")
        return Response(status_code=204)

    @app.post("/api/datasets/{identifier}/readings")
    def append_reading(identifier: str, reading: ReadingRequest):
        if identifier in SCENARIOS:
            raise InputError("Sample readings are read-only. Import a CSV to add new readings.")
        with svc().lock:
            data = svc().dataset(identifier)
            history = from_records(data["readings"])
            expected = history.index[-1] + pd.Timedelta(minutes=15)
            try:
                supplied = pd.Timestamp(reading.timestamp)
            except ValueError as exc:
                raise InputError("Enter a valid local timestamp.") from exc
            if supplied != expected:
                raise InputError(f"The next actual reading must be at {expected}. Refresh if someone else has added it.")
            history = pd.concat([history, pd.Series([reading.usage_kwh], index=[supplied])]).iloc[-MAX_ROWS:]
            validate_history(history)
            data["readings"] = history_records(history)
            svc().store.update_dataset(data)
        return svc().describe(data, include_preview=True)

    @app.get("/api/template.csv")
    def template():
        sample = svc().dataset("demo-production")
        return csv_response([dict(timestamp=r["timestamp"], Usage_kWh=r["usage_kwh"]) for r in sample["readings"]], "sample_7_day_meter_history.csv")

    @app.post("/api/forecasts", status_code=201)
    def forecast(body: ForecastRequest):
        return svc().forecast(body.dataset_id, body.alert_threshold_kwh)

    @app.get("/api/forecasts")
    def forecasts():
        return [{k: v for k, v in row.items() if k not in {"input_features", "recent_readings", "warnings"}} for row in svc().store.forecasts()]

    @app.get("/api/forecasts/export.csv")
    def export_history():
        keys = ["id", "created_utc", "dataset_name", "source", "forecast_timestamp", "predicted_kwh", "previous_kwh", "alert_threshold_kwh", "high_load", "actual_kwh", "absolute_error_kwh", "model_sha256", "history_sha256"]
        return csv_response([{k: row[k] for k in keys} for row in svc().store.forecasts()], "forecast_history.csv")

    @app.get("/api/forecasts/{identifier}")
    def forecast_detail(identifier: str):
        for row in svc().store.forecasts():
            if row["id"] == identifier:
                return row
        raise HTTPException(404, "Forecast not found.")

    @app.get("/api/evaluation")
    def evaluation(date: str = "2018-11-19"):
        return svc().evaluation_day(date)

    @app.get("/api/evaluation/export.csv")
    def export_evaluation():
        return FileResponse(ROOT / "random forest/outputs_advanced/test_predictions.csv", media_type="text/csv", filename="random_forest_test_predictions.csv")

    @app.get("/api/evaluation/chart.png")
    def chart_image():
        return FileResponse(ROOT / "random forest/outputs_advanced/evaluation.png", media_type="image/png")

    @app.get("/api/metrics/export.csv")
    def export_metrics():
        return csv_response(svc().info["comparisons"], "five_model_reported_results.csv")

    app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    return app


app = create_app()

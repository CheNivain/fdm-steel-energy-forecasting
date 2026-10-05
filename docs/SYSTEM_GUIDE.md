# Steel Energy: Final System Guide

## Start from VS Code

Open the FDM Project folder and its terminal:

```bash
source .venv/bin/activate
python run_app.py --port 8002
```

Open http://127.0.0.1:8002. Keep the terminal running. If the port is in use,
choose another free port and open that same port in the browser. Starting the
app loads saved models; it does not train them. Stop it with Ctrl+C.

The finished app is already running when delivered. Open its address directly;
these terminal commands are for a later restart, not a second copy on the same port.

The original demonstration may still be running on 8001. Use the new address
for the rolling system.

## Your five screens

| Screen | What it does |
| --- | --- |
| Forecast | Predict beyond measured history; advance 15 or 30 minutes. |
| Historical replay | Run known examples from Monday through Sunday. |
| Meter data | Upload actual readings, enter one actual, or refit from actuals. |
| Forecast ledger | Compare predictions with actuals; inspect inputs and export. |
| Model evidence | Show the earlier test results and all five reported models. |

## First forecast

1. A fresh workspace starts at 1 January 2019, 00:00.
2. Keep Duration at 15 minutes and the alert threshold at 50 kWh.
3. Click Generate forecast. The result covers 00:00-00:15.
4. The start time advances to 00:15. Click again for 00:15-00:30.
5. The second forecast uses the first estimate where no actual exists.
6. Orange chart segments and the estimated-input count identify estimates.

The model calculates a new value. The supplied dataset has no actual for that
first future interval, so the ledger shows Actual as Pending. This is a historical
future scenario, not a live connection to a factory in 2026. No real 2019
measurements were provided with the original dataset.

All seven weekdays work. The small date buttons show the coming week. For example,
requesting 18:00-18:30 without actual readings after midnight requires estimates
for intervening intervals. The app shows that count and saves the estimates too.
A request allows at most seven days of intervening estimates to bound computation.
Further requests can advance again; the passage of time does not restore accuracy.

### Interval timestamps

CSV timestamps are **interval ends**: 2019-01-01 00:15 means energy used from
00:00 to 00:15. The original dataset's midnight row is treated as the preceding
day's end, matching the team's existing +1 day assumption. The last 2018 interval
therefore ends at 2019-01-01 00:00; the first unknown interval ends at 00:15.
Confirm this assumption with the source before real deployment.
ISO uploads are never shifted or converted between timezones.

## Upload actual readings

On Meter data, use Factory actual readings, not the separate replay import.
Enter one interval's actual reading, or upload a UTF-8 CSV:

```csv
timestamp,Usage_kWh
2019-01-01 00:15:00,3.45
2019-01-01 00:30:00,3.61
```

Those numbers illustrate the format only; they are not known January actuals.
Use measured values in the main workspace. January's complete file begins
1 January 00:15 and ends 1 February 00:00 (2,976 readings). February's readings
then begin at 1 February 00:15.

- One to 40,000 consecutive readings; up to 8 MB per file.
- Exactly named timestamp and Usage_kWh columns.
- Local timestamps at :00, :15, :30 or :45; no timezone suffix.
- Nonnegative, finite energy per interval, not cumulative totals or amperes.
- Chronological order; no gaps, duplicates or blank measurements.
- New readings overlap saved history or immediately follow it.
- Changing a verified value requires Correct existing verified readings.

The original predictions remain unchanged in the ledger. Uploaded actuals are
matched by timestamp and errors calculated. Future estimates stop being used as
active inputs, but remain in the audit records. The forecast cursor restarts
immediately after the latest actual.

Imports and actual corrections are recorded in SQLite. For a demonstration using
example actuals, use an isolated test database and label them as simulation data.

## Refit the model

New actuals affect input lags immediately. Training changes the decision trees.

After uploading a month's actuals, click Refit Random Forest. It trains a new
deployment version using accumulated usable actuals and the fixed selected
settings. Predictions never become training targets. The previous model stays
available until the replacement is ready. Actual imports are blocked during
training to keep its data snapshot stable.

Keep the server running; the initial full-year fit took about 8.4 minutes on this
machine. Versions are saved separately. A successful refit refreshes forecast
inputs from actuals; it does not rewrite issued predictions or establish better
accuracy. If training fails, the previous model remains available.

## Read the results

| Label | Meaning |
| --- | --- |
| Forecast kWh | Predicted interval energy; 30 minutes shows two intervals and their sum. |
| Alert | At least one 15-minute prediction reaches the chosen threshold. |
| Measured history | All 672 input readings for that forecast were measured. |
| Recursive | At least one input reading was an earlier prediction. |
| Pending | No actual has arrived for the forecast interval. |
| Error | Absolute difference between predicted and actual energy. |
| High-load MAE | Average error where actual energy is at least 50 kWh. |
| N/A | No relevant measured outcomes yet; not zero error. |

The ledger scores the first issued forecast once per interval. A later revised
forecast cannot replace a worse original in this comparison. Measured-history
and recursive MAE are shown separately. Export includes every record; the page
shows the most recent 500.

## What was trained

The deployment model uses 1,000 trees, minimum leaf size 2, 60% of features per
split, unrestricted depth, seed 42 and the selected calendar/harmonic features.
It predicts the change from the previous interval, adds that previous value
back internally, and floors the result at zero.

The dataset has 35,040 readings. The fit used 33,695 usable examples. The first
672 form lag history; one suspect all-zero reading and its following 672 input
windows were excluded. It was not interpolated using a future value.

The same 15 ordered inputs are calculated for training and inference: time/day
cycles, weekend flag, lags at 1/4/96/672 intervals, past rolling averages and
variability/maxima, and the previous change. The load_level placeholder is ignored
by the selected transformer. The saved pipeline expands these to 70 inputs.
Do not add another scaler or add the previous reading a second time.

## Explain it at the presentation

> We trained Random Forest using the usable historical readings. It predicts the
> next 15 minutes of energy use. Without actual updates, it can temporarily reuse
> its earlier predictions, which remain labelled as estimates. When actuals
> arrive, we compare them with the original forecasts and refresh the history.
> We can also refit using accumulated actual readings. This supports planning.

Demonstrate a first forecast, another forecast, the estimated-input count, all
seven date choices, and the ledger. Demonstrate reconciliation with real
measurements or clearly labelled simulation data in a separate workspace.

The earlier development results were MAE 3.6445 kWh, RMSE 7.7350 kWh,
R-squared 0.93827 and HighLoad_MAE 9.5150 kWh. They do not evaluate the full-year
refit or a month of recursive predictions. R-squared is not percentage accuracy.
KNN and Gradient Boosting used different test sets/cutoffs; a fair five-way
ranking remains unestablished. Earlier preprocessing limitations remain disclosed.

Errors can accumulate, flatten or drift without frequent actual updates.
Monthly-only uploads support a planning simulation; near-term operations benefit
from much more frequent actuals. There is no live meter connection, machinery
control, guaranteed savings or demonstrated future accuracy.

## Transfer to another computer

Use Python 3.12, a new virtual environment and requirements-app.txt. Include:

- random forest/outputs_advanced/: original model and evaluation evidence.
- app_data/deployment/: versioned deployment model and manifest.
- app_data/steel_energy.sqlite3: saved readings/forecasts, when transferring history.
- data/, src/, backend/, frontend/, including locally bundled vendor assets.

Do not copy this Mac's .venv to another computer. No Node server or internet
connection is needed once the Python dependencies are installed.

To deliberately reproduce the initial original-dataset refit:

```bash
python -m backend.deployment
```

Use the UI refit for accumulated new actuals. Do not run both refits concurrently.

The technical report and stakeholder presentation are separate deliverables.
Use the architecture/testing notes and acknowledge each member's actual work.
The application cannot guarantee an assessment mark.

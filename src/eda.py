# %% [markdown]
# # Steel Industry Energy Consumption: EDA
# Data Miners | IT3051 Fundamentals of Data Mining
#
# Goal: understand the supplied dataset before forecasting Usage_kWh.
# One observation represents a 15-minute interval. kWh measures energy.
# This notebook performs exploratory analysis only. It does not train models
# or produce a completed preprocessing pipeline.
#
# Data source: the supplied Steel_industry_data.csv. The approved proposal
# identifies the UCI Steel Industry Energy Consumption dataset,
# DOI: https://doi.org/10.24432/C52G8C.
# All numerical findings below are calculated from the supplied CSV.

# %%
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import html
import json
import os

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython
from IPython.display import HTML, display

DATA_PATH = Path(os.environ.get("STEEL_DATA_CSV", str(ROOT / "data" / "Steel_industry_data.csv")))
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
REPORTS = ROOT / "reports"
for folder in (FIGURES, TABLES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 140, "savefig.dpi": 160,
})
TEAL, RED, GOLD = "#117c83", "#bd4256", "#cc951c"
sections = []


def show(title, explanation, table=None, figure=None):
    """Use the same computed results in the notebook and standalone report."""
    content = f"<section><h2>{html.escape(title)}</h2><p>{html.escape(explanation)}</p>"
    if table is not None:
        if isinstance(table, pd.Series):
            table = table.to_frame()
        content += '<div class="table-wrap">' + table.to_html(
            border=0, float_format=lambda value: f"{value:,.3f}"
        ) + "</div>"
    if figure is not None:
        encoded = base64.b64encode(figure.read_bytes()).decode("ascii")
        content += f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(title)}">'
    content += "</section>"
    sections.append(content)
    if get_ipython() is not None:
        display(HTML(content))
    else:
        print(title)


def save_plot(fig, filename):
    fig.tight_layout()
    path = FIGURES / filename
    fig.savefig(path, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    return path


def export_table(frame, filename):
    frame.to_csv(TABLES / filename, index=True)


def group_summary(data, column, order=None):
    result = data.groupby(column)["Usage_kWh"].agg(["count", "mean", "median", "std", "min", "max"])
    result["share_percent"] = result["count"] / len(data) * 100
    return result.reindex(order) if order is not None else result


source_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
raw = pd.read_csv(DATA_PATH)
required = {
    "date", "Usage_kWh", "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh", "CO2(tCO2)",
    "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor",
    "NSM", "WeekStatus", "Day_of_week", "Load_Type",
}
if set(raw.columns) != required:
    raise ValueError(f"Unexpected columns. Missing: {required - set(raw.columns)}; extra: {set(raw.columns) - required}")
show("1. Dataset overview", f"The supplied CSV contains {len(raw):,} rows and {raw.shape[1]} columns. "
     "Usage_kWh is the target: energy consumed in one 15-minute interval. "
     "Predicting a numerical amount is a supervised regression task.", raw.head())

# %% [markdown]
# ## Understand the columns
# Features are possible inputs. The target is the value we want to predict.
# A recorded column is not automatically a valid forecasting input: it must
# be available before the interval being predicted.

# %%
dictionary = pd.DataFrame([
    ("date", "Interval timestamp as supplied", "Parse day/month/year; check midnight ordering"),
    ("Usage_kWh", "Energy consumed during the interval (kWh)", "Target, not a same-interval input"),
    ("Lagging_Current_Reactive.Power_kVarh", "Lagging reactive energy (kVarh)", "Same-interval measurement: leakage risk"),
    ("Leading_Current_Reactive_Power_kVarh", "Leading reactive energy (kVarh)", "Same-interval measurement: leakage risk"),
    ("CO2(tCO2)", "Recorded carbon dioxide emissions (tonnes)", "Same-interval measurement: leakage risk"),
    ("Lagging_Current_Power_Factor", "Lagging power factor (percentage)", "Same-interval measurement: leakage risk"),
    ("Leading_Current_Power_Factor", "Leading power factor (percentage)", "Same-interval measurement: leakage risk"),
    ("NSM", "Seconds since midnight", "Calendar/time feature; verify against date"),
    ("WeekStatus", "Weekday or Weekend", "Calendar feature known in advance"),
    ("Day_of_week", "Named weekday", "Calendar feature known in advance"),
    ("Load_Type", "Light_Load, Medium_Load or Maximum_Load", "Use only if independently planned and known in advance"),
], columns=["Column", "Meaning", "Forecasting role"]).set_index("Column")
export_table(dictionary, "data_dictionary.csv")
show("2. What the columns mean", "The electrical measurements describe the recorded interval. "
     "Calendar fields can be known before it. Load_Type needs confirmation of how it is assigned.", dictionary)

# %% [markdown]
# ## Data quality and date checks
# Missing values mean information is absent. Duplicates mean entire rows repeat.
# We also check invalid dates, repeated timestamps, 15-minute continuity,
# and whether the supplied calendar fields agree with the timestamp.
# We keep the original CSV unchanged.

# %%
numeric_columns = raw.select_dtypes(include="number").columns.tolist()
category_columns = ["WeekStatus", "Day_of_week", "Load_Type"]
blank_cells = raw.select_dtypes(include="object").apply(
    lambda col: col.str.strip().eq("").sum()
)
quality_by_column = pd.DataFrame({
    "dtype": raw.dtypes.astype(str), "missing": raw.isna().sum(),
    "blank_strings": blank_cells.reindex(raw.columns, fill_value=0),
    "unique_values": raw.nunique(),
})
parsed_dates = pd.to_datetime(raw["date"], format="%d/%m/%Y %H:%M", errors="coerce")
invalid_dates = int(parsed_dates.isna().sum())
if invalid_dates:
    raise ValueError(f"Found {invalid_dates} invalid timestamps. Review before time analysis.")
date_order_reversals = int((parsed_dates.diff() < pd.Timedelta(0)).sum())
df = raw.copy()
df["timestamp"] = parsed_dates
df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
expected_grid = pd.date_range(df.timestamp.min(), df.timestamp.max(), freq="15min")
missing_intervals = expected_grid.difference(pd.DatetimeIndex(df.timestamp))
off_grid = pd.DatetimeIndex(df.timestamp).difference(expected_grid)
expected_weekstatus = pd.Series(np.where(parsed_dates.dt.dayofweek < 5, "Weekday", "Weekend"), index=raw.index)
quality = {
    "rows": int(len(raw)), "columns": int(raw.shape[1]),
    "missing_cells": int(raw.isna().sum().sum()),
    "blank_string_cells": int(blank_cells.sum()),
    "exact_duplicate_rows": int(raw.duplicated().sum()),
    "invalid_dates": invalid_dates,
    "duplicate_timestamps": int(parsed_dates.duplicated().sum()),
    "original_order_backward_steps": date_order_reversals,
    "missing_15_minute_timestamps": int(len(missing_intervals)),
    "off_grid_timestamps": int(len(off_grid)),
    "non_15_minute_steps_after_sort": int(df.timestamp.diff().dropna().ne(pd.Timedelta(minutes=15)).sum()),
    "day_of_week_mismatches": int(parsed_dates.dt.day_name().ne(raw.Day_of_week).sum()),
    "weekstatus_mismatches": int(expected_weekstatus.ne(raw.WeekStatus).sum()),
    "nsm_mismatches": int((parsed_dates.dt.hour * 3600 + parsed_dates.dt.minute * 60).ne(raw.NSM).sum()),
    "infinite_numeric_values": int(np.isinf(raw[numeric_columns].to_numpy()).sum()),
    "negative_usage_values": int(raw.Usage_kWh.lt(0).sum()),
    "power_factor_values_outside_0_to_100": int(sum(
        (~raw[c].between(0, 100)).sum()
        for c in ["Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"]
    )),
}
quality_table = pd.Series(quality, name="Result").to_frame()
export_table(quality_by_column, "quality_by_column.csv")
export_table(quality_table, "quality_checks.csv")
export_table(raw[numeric_columns].describe().T, "numeric_summary.csv")
show("3. Missing values and duplicates", f"There are {quality['missing_cells']} missing cells, "
     f"{quality['blank_string_cells']} blank strings, and {quality['exact_duplicate_rows']} exact duplicate rows. "
     "No values were filled or rows removed.", quality_by_column)
show("4. Time and consistency checks", f"Parsed timestamps run from {df.timestamp.min():%Y-%m-%d %H:%M} "
     f"to {df.timestamp.max():%Y-%m-%d %H:%M}. After sorting the literal timestamps, "
     f"{len(missing_intervals)} expected 15-minute timestamps are missing. "
     f"The raw file has {date_order_reversals} backward time steps: midnight appears after 23:45 "
     "within each date. These charts use the dates as written, sorted chronologically. "
     "Before creating forecasting lags, confirm whether those midnight rows should actually "
     "represent the following day's interval end. We have not shifted their dates.", quality_table)
export_table(raw.loc[parsed_dates.diff().lt(pd.Timedelta(0)), ["date", "NSM", "Usage_kWh"]], "midnight_rows_to_review.csv")

# %% [markdown]
# ## Target distribution and unusual values
# The mean is the average. The median is the middle observation.
# A histogram counts observations in ranges of usage.
# The IQR rule flags values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
# A flag is a reason to investigate, not proof that an observation is wrong.

# %%
usage = df["Usage_kWh"]
target_stats = usage.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
target_stats.loc["skewness"] = usage.skew()
target_stats.loc["zero_values"] = usage.eq(0).sum()
q1, q3 = usage.quantile([0.25, 0.75])
iqr = q3 - q1
lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outlier_mask = usage.lt(lower_fence) | usage.gt(upper_fence)
outlier_count = int(outlier_mask.sum())
outliers = df.loc[outlier_mask, ["timestamp", "Usage_kWh", "Load_Type", "WeekStatus"]]
export_table(target_stats.to_frame("Value"), "target_summary.csv")
export_table(outliers, "usage_iqr_flags.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 1]})
axes[0].hist(usage, bins=50, color=TEAL, edgecolor="white", linewidth=0.4)
axes[0].axvline(usage.mean(), color=RED, label=f"Mean {usage.mean():.2f}")
axes[0].axvline(usage.median(), color=GOLD, linestyle="--", label=f"Median {usage.median():.2f}")
axes[0].set(xlabel="Energy per 15-minute interval (kWh)", ylabel="Number of intervals", title="Distribution of electricity consumption")
axes[0].legend()
axes[1].boxplot(usage, patch_artist=True, boxprops={"facecolor": "#b8e1e0"}, flierprops={"markersize": 3})
axes[1].set(ylabel="Energy per interval (kWh)", title="Spread and IQR flags", xticks=[])
show("5. Usage distribution and outliers", f"Mean usage is {usage.mean():.2f} kWh and the median is "
     f"{usage.median():.2f} kWh. The distribution is right-skewed (skewness {usage.skew():.2f}): "
     "many low-consumption intervals occur alongside higher-consumption intervals. "
     f"The range is {usage.min():.2f} to {usage.max():.2f} kWh, including {int(usage.eq(0).sum())} zeros. "
     f"The IQR limits are {lower_fence:.2f} and {upper_fence:.2f} kWh; "
     f"{outlier_count} intervals ({outlier_count / len(df):.2%}) are flagged. "
     "Keep these observations for now: real production peaks matter to this project. "
     "Investigate zero readings before treating them as errors; zero does not mean missing.",
     target_stats.to_frame("Value"), save_plot(fig, "01_usage_distribution.png"))

# %% [markdown]
# ## Compare load types
# Compare both average usage and the spread of observations in each group.
# A relationship is descriptive; it does not prove that changing a label
# causes energy consumption to change.

# %%
load_order = ["Light_Load", "Medium_Load", "Maximum_Load"]
load_summary = group_summary(df, "Load_Type", load_order)
export_table(load_summary, "usage_by_load_type.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
bars = axes[0].bar(["Light", "Medium", "Maximum"], load_summary["mean"], color=[TEAL, GOLD, RED])
axes[0].bar_label(bars, fmt="%.2f", padding=3)
axes[0].set(ylabel="Mean energy per interval (kWh)", title="Average usage by load type", ylim=(0, load_summary["mean"].max() * 1.2))
axes[1].boxplot([df.loc[df.Load_Type.eq(label), "Usage_kWh"] for label in load_order], tick_labels=["Light", "Medium", "Maximum"], showfliers=False)
axes[1].set(ylabel="Energy per interval (kWh)", title="Within-group spread (fliers hidden)")
show("6. Load type patterns", "Mean consumption increases across the Light, Medium and Maximum load categories. "
     "The groups have different sample sizes, and their usage ranges overlap. "
     "Load_Type is a candidate input only if the plant knows it before the forecast interval "
     "and it is not defined using that interval's realised consumption.",
     load_summary, save_plot(fig, "02_load_type.png"))

# %% [markdown]
# ## Weekday, weekend and time-of-day patterns
# Use average kWh per 15-minute interval so groups with more observations
# do not automatically appear to consume more. The hourly chart averages
# the interval observations recorded within each hour; it is not an hourly total.

# %%
week_order = ["Weekday", "Weekend"]
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
week_summary = group_summary(df, "WeekStatus", week_order)
day_summary = group_summary(df, "Day_of_week", day_order)
df["hour"] = df.timestamp.dt.hour
df["month"] = df.timestamp.dt.month
hour_summary = group_summary(df, "hour")
hour_week = df.pivot_table(index="hour", columns="WeekStatus", values="Usage_kWh", aggfunc="mean").reindex(columns=week_order)
for table, name in [(week_summary, "usage_by_weekstatus"), (day_summary, "usage_by_weekday"), (hour_summary, "usage_by_hour"), (hour_week, "hour_by_weekstatus")]:
    export_table(table, f"{name}.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, summary, labels, colors, title in [
    (axes[0], week_summary, week_order, [TEAL, RED], "Weekday versus weekend"),
    (axes[1], day_summary, [d[:3] for d in day_order], [TEAL] * 5 + [RED] * 2, "Average usage by day of week"),
]:
    bars = ax.bar(labels, summary["mean"], color=colors)
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set(ylabel="Mean energy per interval (kWh)", title=title, ylim=(0, summary["mean"].max() * 1.2))
weekday_mean, weekend_mean = week_summary.loc[week_order, "mean"]
show("7. Weekday and weekend patterns", f"Weekday intervals average {weekday_mean:.2f} kWh; "
     f"weekend intervals average {weekend_mean:.2f} kWh. "
     "This association suggests calendar features may help forecasting. "
     "It does not establish the cause of the difference.", week_summary, save_plot(fig, "03_week_patterns.png"))

fig, ax = plt.subplots(figsize=(10, 4))
for label, color in zip(week_order, [TEAL, RED]):
    ax.plot(hour_week.index, hour_week[label], marker="o", markersize=3, color=color, label=label)
ax.set(xticks=range(0, 24, 2), xlabel="Hour from the supplied timestamp", ylabel="Mean energy per 15-minute interval (kWh)", title="Consumption profile through the day")
ax.grid(axis="y", alpha=0.2)
ax.legend()
peak_hour = int(hour_summary["mean"].idxmax())
low_hour = int(hour_summary["mean"].idxmin())
show("8. Time of day", f"The highest overall hourly-group mean occurs at {peak_hour:02d}:00-"
     f"{peak_hour:02d}:59 ({hour_summary.loc[peak_hour, 'mean']:.2f} kWh per interval). "
     f"The lowest occurs at {low_hour:02d}:00-{low_hour:02d}:59 "
     f"({hour_summary.loc[low_hour, 'mean']:.2f} kWh per interval). "
     "These are consumption patterns, not electricity tariff periods. The dataset has no tariff schedule.",
     figure=save_plot(fig, "04_hourly_profile.png"))

day_hour = df.pivot_table(index="Day_of_week", columns="hour", values="Usage_kWh", aggfunc="mean").reindex(day_order)
export_table(day_hour, "weekday_hour_means.csv")
fig, ax = plt.subplots(figsize=(11, 4))
heat = ax.imshow(day_hour.to_numpy(), aspect="auto", cmap="YlGnBu")
ax.set(yticks=range(7), yticklabels=day_order, xticks=range(0, 24, 2), xticklabels=range(0, 24, 2), xlabel="Hour from the supplied timestamp", title="Mean interval consumption by weekday and hour")
fig.colorbar(heat, ax=ax, label="Mean energy per interval (kWh)")
show("9. Combined weekday and hour patterns", "The heatmap combines the calendar patterns. "
     "Darker cells indicate greater average consumption. Each cell is an average across "
     "the relevant intervals in this dataset, not an individual day's prediction.", figure=save_plot(fig, "05_weekday_hour_heatmap.png"))

# %% [markdown]
# ## Changes over the year
# Time plots help identify shifts, peaks and low-activity periods.
# One year of observations cannot establish that an annual pattern repeats.

# %%
daily = df.set_index("timestamp")["Usage_kWh"].resample("D").agg(["count", "mean", "sum", "max"])
monthly = group_summary(df, "month")
export_table(daily, "daily_usage.csv")
export_table(monthly, "monthly_usage.csv")
fig, axes = plt.subplots(2, 1, figsize=(11, 6))
axes[0].plot(daily.index, daily["mean"], linewidth=0.9, color=TEAL)
axes[0].set(ylabel="Mean interval energy (kWh)", title="Daily mean consumption across 2018")
axes[0].grid(axis="y", alpha=0.2)
axes[1].bar(monthly.index, monthly["mean"], color=TEAL)
axes[1].set(xticks=range(1, 13), xlabel="Month", ylabel="Mean interval energy (kWh)", title="Monthly mean consumption")
show("10. Consumption over time", "Daily and monthly means reveal changes through the year. "
     "We use the date labels as supplied, including the documented midnight ambiguity. "
     "The daily table also includes summed kWh; the plotted values are means. "
     "A chronological evaluation is appropriate for future forecasting. Annual seasonality "
     "cannot be confirmed from this single year.", monthly, save_plot(fig, "06_time_patterns.png"))

# %% [markdown]
# ## Correlation and forecasting leakage
# Pearson correlation measures a linear association, from -1 to +1.
# It is not model accuracy and does not prove causation.
# Same-interval measurements can be strongly related to usage while being
# unavailable when a forecast must be made. This is a leakage risk.

# %%
numeric = raw[numeric_columns]
correlation = numeric.corr(method="pearson")
target_corr = correlation["Usage_kWh"].drop("Usage_kWh").sort_values(key=lambda series: series.abs(), ascending=False)
export_table(correlation, "numeric_correlations.csv")
export_table(target_corr.to_frame("Pearson_r"), "target_correlations.csv")
short_labels = {
    "Usage_kWh": "Usage (kWh)", "Lagging_Current_Reactive.Power_kVarh": "Lagging reactive",
    "Leading_Current_Reactive_Power_kVarh": "Leading reactive", "CO2(tCO2)": "CO2",
    "Lagging_Current_Power_Factor": "Lagging PF", "Leading_Current_Power_Factor": "Leading PF", "NSM": "Seconds since midnight",
}
fig, ax = plt.subplots(figsize=(9, 7))
heat = ax.imshow(correlation.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
labels = [short_labels[col] for col in correlation.columns]
ax.set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(labels)), yticklabels=labels, title="Pearson correlations: descriptive analysis only")
plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
for row in range(len(labels)):
    for col in range(len(labels)):
        value = correlation.iloc[row, col]
        ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > 0.6 else "#222222", fontsize=9)
fig.colorbar(heat, ax=ax, label="Pearson correlation")
co2_corr = float(target_corr["CO2(tCO2)"])
show("11. Correlations and leakage risks", f"CO2 and Usage_kWh have Pearson correlation {co2_corr:.4f}. "
     "This is not a prediction score. CO2, reactive energy and power factors are recorded "
     "for the same interval, so exclude them from an advance forecast unless their availability "
     "before that interval is established. Their historical lagged values may be considered "
     "later if known at prediction time. Confirm how Load_Type is assigned as well. "
     "Do not select inputs solely because they have high correlation.",
     target_corr.to_frame("Pearson_r"), save_plot(fig, "07_correlation_heatmap.png"))

# %% [markdown]
# ## What the EDA means for the next stage
# These are recommendations. Encoding, lag creation, train/validation/test
# splitting and model fitting have not been implemented by this EDA.
# Any learned preprocessing, thresholds or feature selection must later
# use training data only, with a final test period kept for final evaluation.

# %%
next_steps = pd.DataFrame([
    ("Missing values / duplicates", "No filling or duplicate removal needed for this file based on the checks."),
    ("Timestamps", "Resolve midnight convention, then sort before creating lags or splitting."),
    ("Outliers and zeros", "Retain initially. Check plant context; peaks may be real and important."),
    ("Calendar inputs", "Consider hour, weekday and month. Avoid redundant copies of equivalent time fields."),
    ("Categorical inputs", "Encode chosen categories. Only use Load_Type if known before prediction."),
    ("Past usage", "After resolving timestamps, consider previous-interval and previous-day usage. Shift rolling features to exclude the target interval."),
    ("Leakage", "Exclude unavailable same-interval electrical measurements and CO2."),
    ("Evaluation", "Use common chronological train/validation/test periods for all team models. Fit learned transformations on training only."),
    ("Model selection later", "Compare Linear Regression and Random Forest with team models on validation MAE/RMSE/R2. Use the final test period once after selection."),
], columns=["Area", "Recommended next action"]).set_index("Area")
export_table(next_steps, "preprocessing_recommendations.csv")
show("12. Decisions for preprocessing", "EDA is complete for this supplied file. "
     "These recommendations are the next work to implement. This descriptive EDA covers "
     "the full file and does not establish an untouched holdout or unbiased model performance. "
     "Freeze evaluation periods before model comparison and keep subsequent model decisions "
     "within training and validation data.", next_steps)

findings = {
    "source_file": DATA_PATH.name, "source_sha256": source_hash,
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "timestamp_policy": "Literal day-first labels, sorted. Midnight rows not date-shifted; confirm convention before forecasting.",
    "quality": quality,
    "date_min": df.timestamp.min().isoformat(), "date_max": df.timestamp.max().isoformat(),
    "usage_mean_kwh": float(usage.mean()), "usage_median_kwh": float(usage.median()),
    "usage_min_kwh": float(usage.min()), "usage_max_kwh": float(usage.max()),
    "usage_skewness": float(usage.skew()), "zero_usage_count": int(usage.eq(0).sum()),
    "iqr_lower_kwh": float(lower_fence), "iqr_upper_kwh": float(upper_fence),
    "iqr_flagged_count": outlier_count,
    "mean_kwh_by_load_type": load_summary["mean"].to_dict(),
    "mean_kwh_by_weekstatus": week_summary["mean"].to_dict(),
    "peak_mean_hour": peak_hour, "lowest_mean_hour": low_hour,
    "co2_usage_pearson_r": co2_corr,
    "models_trained": False, "preprocessing_pipeline_completed": False,
    "versions": {"pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
}
(REPORTS / "eda_findings.json").write_text(json.dumps(findings, indent=2, allow_nan=False), encoding="utf-8")

viva_notes = f"""# EDA findings for your viva

## What was done

Python with pandas, NumPy and Matplotlib was used to inspect the dataset,
check quality, summarize consumption and plot relationships. The original
CSV was preserved. No model has been trained and preprocessing is not yet complete.

## Five findings to remember

1. There are {len(raw):,} records and {raw.shape[1]} columns. Usage_kWh is the numerical target.
2. There are {quality['missing_cells']} missing cells, {quality['exact_duplicate_rows']} exact duplicate rows and {quality['duplicate_timestamps']} duplicate timestamps.
3. Average usage is {usage.mean():.2f} kWh per 15-minute interval; the median is {usage.median():.2f} kWh. Many intervals have low usage and a smaller number have much higher usage.
4. Mean usage is {weekday_mean:.2f} kWh on weekdays and {weekend_mean:.2f} kWh on weekends.
5. Light, Medium and Maximum load means are {load_summary.loc['Light_Load', 'mean']:.2f}, {load_summary.loc['Medium_Load', 'mean']:.2f} and {load_summary.loc['Maximum_Load', 'mean']:.2f} kWh per interval.

## Other findings you may be asked about

- The IQR rule flags {outlier_count} observations above/below the fences {lower_fence:.2f} and {upper_fence:.2f} kWh. We retained them because real production peaks matter.
- There are {int(usage.eq(0).sum())} zero readings. Zero is a recorded value, not a missing value; plant context is needed before removing it.
- The original row order has {date_order_reversals} backward timestamp steps. Midnight rows occur after 23:45 within the same labelled date. We sorted literal timestamps for EDA, but must verify whether midnight should belong to the next day before building lag features.
- Sorted literal timestamps cover 2018 with {len(missing_intervals)} missing 15-minute timestamps.
- CO2 correlates strongly with usage (r = {co2_corr:.4f}), but this is not model accuracy. Same-interval CO2, reactive energy and power factors can leak information unavailable at forecast time.
- Load_Type is usable for forecasting only if it is independently scheduled and known beforehand.
- High consumption times are not necessarily high-tariff times. Tariff information is absent.

## A short answer you can explain

"Our EDA used Python to check structure, missing values, duplicates and time consistency.
We plotted the usage distribution and compared consumption by load type, weekday and time
of day. We found no missing values or exact duplicates, higher weekday consumption, and
a right-skewed usage distribution. We retained unusual high readings for investigation
and identified potential leakage from same-interval electrical measurements. We also
found a midnight timestamp-order issue to resolve before creating forecasting features."

## What remains

Confirm timestamps and available forecast inputs, implement feature encoding and past-only
lag features, and agree chronological train/validation/test periods. Fit any learned
preprocessing only on training data. Models are a later stage. Compare models on validation
MAE, RMSE and R2; reserve the final test period for the selected model's final evaluation.

Only describe completed work as completed. This EDA does not by itself complete all
preprocessing requirements for Progress Evaluation 1.
"""
(REPORTS / "VIVA_NOTES.md").write_text(viva_notes, encoding="utf-8")

style = """
body{font-family:Arial,Helvetica,sans-serif;color:#202b2c;background:#fff;margin:0;line-height:1.55;letter-spacing:0}
main{max-width:1120px;margin:0 auto;padding:32px 24px 56px}
h1{font-size:30px;line-height:1.2;margin:0 0 12px}h2{font-size:21px;line-height:1.3}
header{border-bottom:3px solid #117c83;padding-bottom:18px}
section{border-bottom:1px solid #d5dfe0;padding:20px 0;break-inside:avoid}
p{max-width:960px}img{display:block;width:100%;height:auto;margin:18px 0}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:9px 10px;border-bottom:1px solid #e1e6e6;text-align:right;vertical-align:top}
th:first-child,td:first-child{text-align:left}thead{background:#edf4f3}a{color:#0b646b}
.status{font-weight:bold;color:#964354}
@media(max-width:600px){main{padding:22px 14px}h1{font-size:25px}table{font-size:12px}}
@media print{main{max-width:none;padding:0}section{break-inside:auto}img,table{break-inside:avoid}h2{break-after:avoid}}
"""
report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Steel Industry EDA | Data Miners</title><style>{style}</style></head><body><main>
<header><p>DATA MINERS / IT3051 FUNDAMENTALS OF DATA MINING</p><h1>Steel Industry Energy Consumption</h1>
<p>Exploratory data analysis of the supplied CSV. Target: <strong>Usage_kWh</strong>.</p>
<p class="status">EDA completed. Forecasting preprocessing and model training remain.</p>
<p>Source identified in the approved proposal: <a href="https://doi.org/10.24432/C52G8C">UCI dataset DOI</a>.
All findings are calculated from the supplied file.</p></header>
{''.join(sections)}
<footer><p>Generated with Python, pandas, NumPy and Matplotlib. No records removed or values imputed.
Source SHA-256: <code style="overflow-wrap:anywhere">{source_hash}</code>.</p></footer>
</main></body></html>"""
(REPORTS / "EDA_REPORT.html").write_text(report, encoding="utf-8")
assert hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() == source_hash, "Source CSV changed during EDA"
print(f"\nEDA complete: {len(raw):,} rows, {len(sections)} report sections, 7 charts.")
print(f"Report: {REPORTS / 'EDA_REPORT.html'}")
print(f"Viva notes: {REPORTS / 'VIVA_NOTES.md'}")

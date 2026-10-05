# Your friend's notebook: a viva study guide

Source reviewed: `/Users/chelakawijesekara/Downloads/FDM/data_preprocessing.ipynb`.
This guide describes that notebook, its saved outputs, and the CSV artifacts.
The original notebook was not changed or rerun. Its saved train/test targets,
lag_1 features and baseline errors were checked independently against the CSV.

Read one numbered block at a time. Each block answers: what does the code do,
why is it there, and what could the examiner ask?

## Start here

Your project forecasts steel-plant electricity consumption for the next
15-minute interval. The target is `Usage_kWh`, an amount of energy in kWh.
It is supervised regression because historical examples contain numerical
answers that a model can learn to predict.

This notebook completes an EDA workflow and implements a preprocessing
workflow. It does not train Support Vector Regression, Linear Regression,
Random Forest, or another learned prediction model. Its heading mentions SVR
because scaling is included with that model in mind.

The notebook's preparation has assumptions and an interpolation issue to
resolve; the existence of saved outputs does not prove every decision correct.

The sequence is:

1. Load the original data.
2. Inspect dates, values and consumption patterns.
3. Treat one suspected invalid reading and remove unsuitable columns.
4. Create time features and features based on past usage.
5. Split earlier observations from later observations.
6. Examine feature relevance using training data, remove two redundant features and scale inputs.
7. Calculate a simple baseline and save artifacts for later modelling.

## The files

| File | What it contains |
| --- | --- |
| `dataset/Steel_industry_data.csv` | Original dataset, 35,040 rows and 11 columns. |
| `data_preprocessing.ipynb` | A Jupyter notebook containing code cells, explanations and saved results. |
| `artifacts/X_train.csv` | 28,511 training examples with 15 input features and a saved timestamp index. |
| `artifacts/y_train.csv` | The target energy values for those training examples, with matching timestamps. |
| `artifacts/X_test.csv` | 5,857 later examples with the same 15 inputs and a timestamp index. |
| `artifacts/y_test.csv` | Their target energy values, with matching timestamps. |
| `artifacts/feature_scaler.pkl` | The fitted scaler saved by the notebook, not an energy-prediction model. |
| `artifacts/preprocessing_config.pkl` | A dictionary of feature names, load mapping, lag choices and split date. It is not the full preprocessing pipeline. |
| `.venv` | The project's Python environment and installed packages. |

`X` means inputs; `y` means the answer to predict. Training data teaches a model.
Test data assesses it on later observations. A CSV with 16 columns here contains
15 features plus `timestamp`; the timestamp is saved as an index, not a sixteenth model feature.

## Python vocabulary

| Syntax | Meaning in this notebook |
| --- | --- |
| `import pandas as pd` | Load a library and give it the short name `pd`. |
| `df = ...` | Store the result under the name `df`. A single `=` assigns a value. |
| `df["Usage_kWh"]` | Select one column, represented as a pandas Series. |
| `df[["Usage_kWh", "CO2"]]` | Select several columns, represented as a DataFrame. |
| `df.shape` | Read a property; it is not a function call. |
| `df.head()` | Call a method. Parentheses mean run the operation. |
| `df[condition]` | Keep rows for which the condition is true. |
| `df.loc[rows, columns]` | Select or update rows and columns by labels or a Boolean mask. |
| `==`, `<`, `>=` | Compare values and return true or false. `==` does not assign. |
| `&`, `\|`, `~` | Elementwise AND, OR and NOT for Boolean masks. Parenthesize each comparison. |
| `.sum()` on a Boolean mask | Count true values, because true behaves like 1. |
| `axis=1` / `columns=...` | Work with columns rather than rows. |
| `for ... in ...` | Repeat an operation for each item. |
| `def name(...):` | Define a reusable function. |
| `f"lag_{lag}"` | Insert a value into a string, such as `lag_96`. |

The label `In [30]` is an execution counter, not the cell's permanent position.
The source positions below count all notebook cells, including Markdown cells.
Run notebook code from top to bottom: later cells depend on earlier variables.

## 1. Imports and settings (source cell 4)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
```

- NumPy supplies numerical operations, such as sine, square roots and missing-value markers.
- pandas supplies tables, CSV reading, timestamps, grouping and rolling windows.
- Matplotlib and seaborn draw charts. They do not train a prediction model.
- joblib saves Python objects, such as the fitted scaler.
- os creates the output directory.
- `mutual_info_regression` estimates how informative each input is about a numerical target.
- `StandardScaler` transforms input scales using training means and standard deviations.

`sns.set_theme(style="whitegrid")` changes chart appearance.
`pd.set_option("display.max_columns", None)` shows every column when displaying a table;
it does not change or add columns.
`RANDOM_STATE = 42` supplies a repeatable random seed for mutual-information estimation.
The number 42 has no special predictive power and does not make the chronological split random.

Viva answer: "We use pandas and NumPy for data handling, plotting libraries for EDA,
and scikit-learn for feature relevance and scaling."

## 2. Loading the data (source cell 6)

```python
df_raw = pd.read_csv("dataset/Steel_industry_data.csv", encoding="utf-8-sig")
print(df_raw.shape)
df_raw.info()
df_raw.head()
```

`read_csv` reads the file into a DataFrame, a table with rows and named columns.
The relative path is interpreted from the current working directory, normally `FDM`.
`utf-8-sig` handles UTF-8 text, including an optional byte-order marker in the file header.
It has no effect on model performance.

`shape` returns `(35040, 11)`: rows first, columns second.
`info()` displays column names, data types and counts of non-missing entries.
`head()` displays the first five rows by default. It does not keep only five rows.

Viva answer: "This cell loads the raw data and checks its size, structure and sample records."

## 3. Copying, renaming and parsing dates (source cell 8)

`df = df_raw.copy()` creates a working copy so transformations do not change `df_raw`.
`rename(columns={...})` shortens column names without changing their numerical values.
For example, `CO2(tCO2)` becomes `CO2`.

```python
df["timestamp"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
```

This converts date text into datetime values. `%d` is day, `%m` month, `%Y` four-digit
year, `%H` hour and `%M` minute. An explicit format avoids confusing day/month order.
`is_monotonic_increasing` checks whether timestamps never move backward.
The original order fails because each date's `00:00` row follows `23:45`.
`loc[94:97, ...]` displays a small example; label-based slicing includes both endpoints.

Viva answer: "We standardize names and convert date strings to timestamps so time operations work correctly."

## 4. The midnight adjustment (source cell 10)

```python
midnight = (df["timestamp"].dt.hour == 0) & (df["timestamp"].dt.minute == 0)
df.loc[midnight, "timestamp"] += pd.Timedelta(days=1)
```

This identifies 365 midnight records and adds one day to their timestamp.
The notebook assumes a row written as 1 January 00:00 after 1 January 23:45
represents 2 January 00:00. That interpretation preserves the apparent sequence.
It is an assumption about interval-end recording, not a rule that all midnight rows are wrong.
Confirm it with dataset documentation or the instructor before treating it as established fact.

`sort_values` orders rows; `set_index` uses the timestamp as the row label;
`drop(columns="date")` removes the old text date column.
`date_range(..., freq="15min")` builds an expected grid.
`expected.difference(df.index)` identifies missing expected timestamps.

Saved result: 35,040 intervals, no missing timestamps or repeated timestamps,
from 2018-01-01 00:15 to 2019-01-01 00:00 under this interpretation.
An ordered sequence alone does not prove the timestamp interpretation correct.

Viva answer: "We interpreted the final midnight row of each recorded date as the next day's
interval boundary, then checked uniqueness and 15-minute continuity."

## 5. Missing values, duplicates and summaries (source cell 12)

`isna().sum()` counts missing values by column. This file has none initially.
`reset_index().duplicated().sum()` includes timestamp in the duplicate comparison: zero.
`df.duplicated().sum()` ignores the timestamp index and finds 81 repeated sets of other values.
The same readings at different times can be legitimate, so they are retained.

`describe().T` calculates numerical statistics and transposes the result so each variable
is a row. Count, mean, standard deviation, minimum, quartiles and maximum summarize its values.
`value_counts()` counts the records belonging to each category.

Viva answer: "There are no missing values or exact duplicate records including time.
Repeated readings at different timestamps were preserved."

## 6. Physical validity checks (source cell 14)

This checks for negative readings and power-factor percentages over 100. None are found.
It displays the one zero-usage observation and its neighbours.
Under the midnight adjustment it is at 2018-11-08 00:00, with all listed electrical readings zero.
That is suspicious, but does not by itself prove a meter fault.

The day-name comparison reports 365 disagreements because the midnight timestamps were
moved forward but the original `Day_of_week` labels were not changed. Later model features
are rebuilt from the adjusted index and the old labels are dropped. EDA charts using original
labels and charts using adjusted calendar features therefore follow different midnight conventions.

Viva answer: "We checked basic value ranges and investigated the all-zero observation.
We also checked calendar consistency after changing timestamps."

## 7. Target distribution (source cell 16)

`histplot(..., bins=80)` groups usage into 80 numerical ranges and counts observations in each.
The second plot uses `np.log1p(y)`, meaning `log(1 + y)`, to compress high values
and allow zero. It is a diagnostic plot only: the final saved target is not log-transformed.

Mean: 27.39 kWh. Median: 4.57 kWh. Skewness: 1.197.
Many observations are low while some are much higher, producing a right-skewed distribution.
Skewness does not automatically require transforming the target.

Viva answer: "Usage is right-skewed. We compared the original and log views to understand its distribution."

## 8. Time-series plots (source cell 18)

`resample("D").sum()` adds interval energy readings within each timestamp date to obtain
daily energy in kWh. Unlike averaging, it measures a daily total.
The second chart selects one week with `loc` and shows every 15-minute observation.

Under the shifted timestamp convention, the first and last calendar dates are partial.
The final 2019-01-01 date has only the 00:00 record, so its small daily total is not evidence
of a full-day shutdown. Date grouping follows timestamp labels, not necessarily production-day boundaries.

Viva answer: "Daily totals show broad changes; the one-week chart shows the finer operating pattern."

## 9. Daily and weekly cycles (source cell 20)

Hour, month and weekday number are extracted from the datetime index.
`dayofweek` uses Monday = 0 through Sunday = 6.
`(dow >= 5).astype(int)` produces weekend = 1 and weekday = 0.

Seaborn's line chart averages repeated observations for each hour and weekend status.
`errorbar=None` hides uncertainty bars; it does not remove observations.
`pivot_table(..., aggfunc="mean")` creates mean usage for each day/hour pair.
The heatmap displays that table with colour.

Viva answer: "Calendar patterns help us identify useful time-based forecasting inputs."

## 10. Monthly patterns (source cell 22)

`groupby("month")["Usage_kWh"].mean()` calculates average interval energy within each
month number. It is not monthly total energy.
The saved means are about 42.40 kWh for January and 19.97 kWh for December.
This data cannot establish a repeating yearly pattern from only one year's observations.

Because the adjusted final row falls in January 2019, grouping only by month number
combines that one reading with January 2018. A year-month grouping avoids this small boundary mixing.

Viva answer: "We compared average interval consumption across months to inspect changes over time."

## 11. Load types (source cell 24)

The boxplots compare group medians, quartiles, spread and IQR outlier flags.
`order` fixes the category display order rather than relying on alphabetical order.
`crosstab(hour, Load_Type)` counts observations in each hour/load combination.
`normalize="index"` converts each weekday/weekend row of a cross-tab into proportions.

Mean interval consumption: Light = 8.63, Medium = 38.45, Maximum = 59.27 kWh.
The groups overlap: a Light_Load record can still have high consumption.
Do not say the category perfectly determines consumption.

Viva answer: "Load categories are associated with different consumption levels, but they are not perfect separators."

## 12. Correlation (source cell 26)

`.corr()` computes pairwise Pearson correlations, from -1 to +1.
Positive means the variables tend to increase together; negative means they tend to move
in opposite directions. Near zero means little linear association, not necessarily no relationship.

CO2 has correlation about 0.988 with usage. Lagging reactive energy has about 0.896.
Neither number is model accuracy, and neither establishes causation.
`sort_values(ascending=False)` sorts the printed target correlations from largest to smallest.

Viva answer: "Correlation helps us inspect linear relationships, but feature availability matters for forecasting."

## 13. Leakage plots (source cell 28)

`zip` pairs each plotting axis with a column name; the loop draws three scatterplots.
Each point is one observation. `s=2` makes points small; `alpha=0.3` makes them transparent
so overlapping points are easier to see.

Same-interval CO2, reactive energy and power factors are excluded later because they may
only become available after the interval being forecast. Knowing them during training
could make the model look better than it can operate in advance.
The plots illustrate relationships; they cannot prove when measurements become available.

Viva answer: "We avoid forecasting with measurements that would only be known after the predicted interval."

## 14. Outliers (source cell 30)

```python
q1, q3 = s.quantile([0.25, 0.75])
iqr = q3 - q1
return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
```

Q1 and Q3 are the 25th and 75th percentiles. IQR is the width of the middle half of values.
The function returns a true/false flag for observations outside the IQR fences.
Calling it on all usage gives 328 flags. Calling it within each load group uses different
thresholds and gives Light = 2,270, Medium = 4, Maximum = 125.
The two methods answer different questions, so their totals need not agree.

`groupby(...).transform(...)` returns a result aligned with every original row.
Flags are inspected and plotted. The notebook does not delete these outliers.

Viva answer: "We investigated globally unusual readings and readings unusual for their load group.
We retained production peaks because they can be valid and important."

## 15. Autocorrelation (source cell 32)

Autocorrelation compares a series with its own earlier values.
At 15-minute resolution, four rows are one hour, 96 are one day and 672 are one week.
The list comprehension calculates autocorrelation for each lag from 1 to 699.

Saved values: lag 1 = 0.913; lag 4 = 0.688; lag 96 = 0.606; lag 672 = 0.677.
These relationships motivate using past readings as candidate inputs.
They are not trained-model scores or guarantees of future performance.

Viva answer: "Past usage is related to later usage, including daily and weekly patterns,
so we create lag features."

## 16. Replacing the suspected invalid reading (source cell 35)

The `bad` mask selects rows with usage and both power factors equal to zero.
It selects one row. That target is set to `np.nan`, a missing-value marker.
`interpolate(method="time")` estimates a value between surrounding timestamped values.
Because neighbours 3.56 and 3.64 kWh are equally spaced, it gives 3.60 kWh.

This is a methodological issue to discuss, not a step to call automatically correct:

- The data alone does not confirm that the zero is a meter fault.
- The replacement uses the later 00:15 reading. That would not be available when forecasting 00:15 from earlier history.
- The replacement is in the test period and becomes a made-up test target at 00:00 as well as a later lag input.

A better forecasting policy needs explicit justification: retain and flag the original
if valid, or mark a confirmed bad target as unavailable for scoring and use past-only
handling of missing historical inputs. Do not silently interpolate test answers.

Viva answer: "The notebook interpolates one suspected fault, but this uses a future observation.
We need a causal missing-data policy and should not treat an imputed test target as ground truth."

## 17. Dropping columns (source cell 37)

Five same-interval electrical columns are removed as leakage risks.
`NSM`, `WeekStatus` and `Day_of_week` are removed because time features will be regenerated
from the timestamp. The working table retains `Usage_kWh` and `Load_Type`.

Dropping current usage from inputs does not prevent us from using older usage through shifts.
Load_Type is still conditional: only use it for a future interval if it is independently
scheduled and known beforehand. Observing categories in a historical file is not proof of availability.

Viva answer: "We remove unavailable same-interval measurements and rebuild redundant calendar fields."

## 18. Calendar features and encoding (source cell 39)

`feat` starts as an empty DataFrame with the same time index.
`(hour * 60 + minute) / 1440` expresses the time as a fraction of a 24-hour day.
Sine and cosine map it around a circle, keeping late-night and early-morning times close.
Both coordinates are needed because either sine or cosine alone maps some different times
to the same value. Weekday sine/cosine similarly wrap Sunday around to Monday.

`is_weekend` is binary. The dictionary mapping encodes Light = 0, Medium = 1, Maximum = 2.
This is ordinal encoding, not one-hot encoding. For models sensitive to numeric distances,
the mapping introduces an order and equal spacing that should be justified; one-hot is an alternative.

Viva answer: "Cyclic encoding preserves the wrap-around relationship in time, and numeric category
encoding lets the model accept the load label."

## 19. Lag and rolling features (source cell 41)

```python
for lag in [1, 2, 3, 4, 96, 672]:
    feat[f"lag_{lag}"] = usage.shift(lag)

past = usage.shift(1)
feat["roll_mean_4"] = past.rolling(4).mean()
```

At row t, `lag_1` contains usage from t-1. `lag_96` is the same timestamp one day earlier,
assuming a complete regular time index. Shifting values does not move or rewrite timestamps.
Rolling features summarize recent history. The initial shift excludes the target interval.
Without that shift, a rolling average would contain the answer being predicted.

| Feature | Meaning at target row t |
| --- | --- |
| `lag_1`, `lag_4`, `lag_96`, `lag_672` | Energy 15 minutes, 1 hour, 1 day and 1 week earlier. |
| `roll_mean_4` | Average of the previous four intervals. |
| `roll_std_4` | Variation across the previous four intervals. |
| `roll_mean_96` | Average of the previous 96 intervals. |
| `roll_max_96` | Maximum energy among the previous 96 intervals. |
| `diff_1` | Most recent usage minus the reading before it. |

The longest lag requires 672 earlier readings. The first 672 rows have missing history.
`dropna()` removes them, reducing 35,040 rows to 34,368. These new missing values were
created by feature engineering; they were not missing in the original CSV.

The notebook attaches the current row's usage as target. That is correctly aligned for
a next-interval forecast from the previous interval: at 09:00 predict the row labelled 09:15
using `lag_1 = usage at 09:00`. The target does not need another shift with this alignment.

Viva answer: "We predict each target interval using only earlier consumption. The longest lag
costs us the first week of examples. The interpolation exception still needs correction."

## 20. Chronological splitting and X/y (source cell 43)

Training rows have timestamps before 2018-11-01. Test rows begin on that date.
`train.drop(columns=TARGET)` creates inputs; `train[TARGET]` selects answers.
At this stage there are 17 inputs, before feature removal.

- Training: 28,511 rows from 2018-01-08 00:15 through 2018-10-31 23:45.
- Test: 5,857 rows from 2018-11-01 00:00 through 2019-01-01 00:00.
- Test share: about 17.0% of rows after lag-related removal.

Chronological splitting simulates learning from the past to predict later observations.
Shuffling would mix future and past. Later test rows use measured earlier test-period
consumption, so this describes rolling one-step-ahead forecasting with new readings arriving.
It is not a two-month forecast issued all at once on 1 November.

Viva answer: "We train on the earlier period and evaluate on later data to respect time order."

## 21. Inspecting the split (source cell 45)

The chart plots daily totals for the two subsets in different colours.
The table compares their numerical distributions. Mean usage is about 27.91 kWh in training
and 24.87 kWh in test, so their distributions are not identical.
The chart's first and last dates may be partial because of lag removal and interval boundaries.

Viva answer: "We confirm time separation and inspect whether the later period differs from training."

## 22. Feature relevance (source cell 47)

`corrwith(y_train)` measures each training input's linear association with the target.
Mutual information can detect dependence beyond a straight-line relationship.
It is nonnegative as estimated here, has no positive/negative direction, and is not an accuracy percentage.

Only training values are used, which avoids using test answers to rank inputs.
`lag_1` has the largest saved mutual-information score, about 1.519.
The default dense-array estimator treats features as continuous; specifying a discrete-feature
mask for category/binary inputs would make that modelling assumption more explicit.

Viva answer: "We examine feature relevance using training data only, with both linear correlation
and a measure that can capture nonlinear dependence."

## 23. Removing redundant features (source cell 49)

This cell correlates inputs with each other, not just with the target.
`.abs()` considers strongly negative and strongly positive correlation alike.
`np.triu(..., k=1)` keeps the upper half above the diagonal to inspect each feature pair once
and exclude a feature's perfect correlation with itself.

For pairs above 0.95, the lower-mutual-information feature is marked for removal.
A `set` prevents duplicate entries in the removal list.
The saved run removes `lag_2` and `lag_3`, each highly correlated with `roll_mean_4`.
Both training and test lose the same columns, leaving 15 input features.

This is a heuristic, not proof that deleting those inputs improves predictions. Its impact
should eventually be assessed on a time-respecting validation period.

Viva answer: "We reduced overlap by removing two highly correlated inputs while retaining the
related input with greater training-set mutual information."

## 24. Scaling (source cell 51)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Conceptually, each transformed value is `(value - training_mean) / training_standard_deviation`.
`fit` learns these statistics; `transform` applies them.
Test data uses the training statistics, not its own. Fitting on test data would disclose
information from the evaluation period to preparation.

Scaling is especially relevant for distance-based models such as SVR.
Random Forest ordinarily does not require standardization. Scaling can help numerical
conditioning for regression, but is not inherently required for ordinary unregularized linear regression.
The target is not standardized by this cell.

Training means are approximately zero and standard deviations approximately one.
Test values need not have zero mean or unit standard deviation because the period can differ.
pandas' sample standard deviation differs slightly from the population convention used by StandardScaler.

Viva answer: "We fit the scaler only on training inputs and reuse it on test inputs so they are
represented consistently without learning test-period statistics."

## 25. The naive baseline (source cell 53)

Persistence predicts that the next interval will consume the same energy as the latest interval:
`prediction = lag_1`. It requires no trained model.

MAE is the average absolute prediction error. RMSE takes the square root of mean squared
error, so larger errors contribute more heavily. Both are in kWh and lower is better.

Saved and independently recomputed results on these prepared test targets:
MAE = 5.16 kWh; RMSE = 11.81 kWh.
These are baseline results, not SVR or Random Forest results. They also inherit the suspect-target
interpolation issue. Zero rows remain in the original data, so percentage-error measures need care.

Viva answer: "We set a simple persistence benchmark. A more complex model must demonstrate improvement."

## 26. Saving outputs (source cell 55)

`os.makedirs("artifacts", exist_ok=True)` creates the directory without failing if it exists.
`to_csv` exports the tables and, by default, also saves their timestamp index.
`joblib.dump` saves the fitted scaler and a configuration dictionary.

The cell exports `X_train` and `X_test`, not `X_train_scaled` and `X_test_scaled`.
Therefore the CSV inputs are unscaled. An SVR training script must apply the saved scaler
in the correct feature order or implement the same train-only preprocessing properly.
The saved scaler is not proof that the CSVs have been transformed.

When loading these CSVs for modelling, use `index_col="timestamp"` and
`parse_dates=True` so the index is not accidentally treated as a model input.
The configuration stores some choices, but does not implement all rolling, cleaning and
timestamp transformations needed by a future prediction service.

Viva answer: "We save aligned input/target datasets and reusable preparation settings for the modelling stage."

## What to clarify with the team

1. Confirm the timestamp convention before calling the midnight change a verified correction.
2. Review the suspect zero reading. Avoid future-based interpolation of forecasting inputs and avoid scoring against invented test targets.
3. Confirm that the future Load_Type is known at forecast time. If not, omit it or forecast it through an independently valid method.
4. Remember that the saved input CSVs are unscaled; use the saved scaler where appropriate.
5. Reserve a validation period within the earlier data, or use time-series cross-validation there, for model choice and tuning. The notebook currently has only train/test.

All team models should use consistent target handling, input availability, time periods,
forecast horizon and metrics. Choose using validation performance and assess the selected
model on the final test period. Repeatedly choosing winners based on the test period makes
that period part of model selection rather than an independent final check.

This guide explains the current implementation and its limitations. It does not silently
endorse or change questionable decisions.

## A short overview to practise

"Our notebook prepares data for next-interval electricity-consumption forecasting.
We inspect data quality and consumption patterns, create calendar and past-usage features,
split data chronologically, examine relevance using training data and standardize inputs
using training statistics. It saves 15 input features for 28,511 training examples and
5,857 test examples. No learned prediction model is trained in this notebook.
We must still confirm the midnight convention and address the future-based interpolation
of a suspected bad reading."

Use this as a comprehension check, not a speech to memorize without understanding.

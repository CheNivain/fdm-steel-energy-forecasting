# EDA findings for your viva

## What was done

Python with pandas, NumPy and Matplotlib was used to inspect the dataset,
check quality, summarize consumption and plot relationships. The original
CSV was preserved. No model has been trained and preprocessing is not yet complete.

## Five findings to remember

1. There are 35,040 records and 11 columns. Usage_kWh is the numerical target.
2. There are 0 missing cells, 0 exact duplicate rows and 0 duplicate timestamps.
3. Average usage is 27.39 kWh per 15-minute interval; the median is 4.57 kWh. Many intervals have low usage and a smaller number have much higher usage.
4. Mean usage is 33.62 kWh on weekdays and 11.73 kWh on weekends.
5. Light, Medium and Maximum load means are 8.63, 38.45 and 59.27 kWh per interval.

## Other findings you may be asked about

- The IQR rule flags 328 observations above/below the fences -68.86 and 123.29 kWh. We retained them because real production peaks matter.
- There are 1 zero readings. Zero is a recorded value, not a missing value; plant context is needed before removing it.
- The original row order has 365 backward timestamp steps. Midnight rows occur after 23:45 within the same labelled date. We sorted literal timestamps for EDA, but must verify whether midnight should belong to the next day before building lag features.
- Sorted literal timestamps cover 2018 with 0 missing 15-minute timestamps.
- CO2 correlates strongly with usage (r = 0.9882), but this is not model accuracy. Same-interval CO2, reactive energy and power factors can leak information unavailable at forecast time.
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

# Progress Evaluation 2: Your Speaking and Code Guide

Use the short answers first. The follow-up answers are for questions from the
examiner. Do not claim work or understanding you cannot demonstrate.

## 1. Your opening: about 45 seconds

> Our project predicts the steel plant's energy consumption for the next
> 15-minute interval. Our target is Usage_kWh. Because this is a numerical value,
> we use supervised regression. The users are plant operations and energy
> managers. Predictions could help them anticipate high consumption and plan
> operations. My assigned algorithms are Linear Regression and Random Forest.
> We use historical consumption and calendar features, validate in time order,
> and compare the models with a simple previous-interval prediction.

Do not claim that the model itself reschedules production or has proved bill
savings. Those require a tariff-aware decision system and additional evaluation.

## 2. Explain your two models

### Linear Regression

Say:

> Linear Regression learns coefficients, or weights, for the input features.
> It adds their weighted contributions to an intercept to predict energy usage.
> We first used a simple version. We then improved the feature representation
> with historical-usage transformations and time/weekend interactions.

If asked for the idea mathematically:

```text
prediction = intercept + weight_1 * feature_1 + weight_2 * feature_2 + ...
```

Our latest model is **Linear Regression with engineered features**, not plain
linear regression on only the original 15 columns. There are 651 transformed
columns. Smooth spline features let it model curved relationships with the
original readings; it remains linear in the coefficients being fitted.

Say about tuning:

> LinearRegression has no tree-depth or tree-count parameter. Most of our
> optimisation changed the input representation: spline settings, calendar
> interactions and whether load level should be used. We also investigated
> weighting, but the selected final model does not use extra sample weights.

Selected settings: five spline knots, weekend interactions, ignore load level.
The fitted scaler and transformations are saved inside the model.

### Random Forest

Say:

> Random Forest combines many decision trees. Each tree learns rules from
> sampled training data and considers subsets of features at its splits.
> Combining their predictions can capture nonlinear patterns and reduce the
> instability of using one tree.

Our latest forest learns the **change since the previous interval**:

```text
change to learn = actual usage now - previous actual usage
final prediction = previous actual usage + predicted change
```

Example: previous usage is 40 kWh and predicted change is +5 kWh, so the final
forecast is 45 kWh. The saved model performs this addition automatically.

Selected settings:

| Setting | Value | What you can say |
| --- | --- | --- |
| `n_estimators` | 1000 | Number of trees combined. More trees do not guarantee better predictions. |
| `max_depth` | None | No fixed depth cap; other stopping conditions still apply. |
| `min_samples_leaf` | 2 | Each leaf must contain at least two training samples. |
| `min_samples_split` | 2 | Minimum sample count to attempt splitting an internal node; other conditions also apply. |
| `max_features` | 0.6 | Each split considers a random subset of the engineered features. |
| `random_state` | 42 | A fixed random seed makes the experiment reproducible. |

The forest has 70 engineered columns from the same 15-column input interface.
It does not need scaling like a distance-based model. Both selected models
ignore `load_level`, though the field remains in the input schema. Both floor
negative final predictions at zero.

## 3. Explain the data and validation

Say:

> X means the input features. y means the actual Usage_kWh values. We train on
> earlier observations and validate on later ones, because this is forecasting.
> Randomly mixing the periods would not represent the intended future forecast.

Important features:

| Feature | Plain meaning |
| --- | --- |
| `lag_1` | Previous 15-minute interval's actual usage. |
| `lag_4` | Actual usage one hour earlier. |
| `lag_96` | Actual usage one day earlier. |
| `lag_672` | Actual usage one week earlier. |
| `roll_mean_4` | Mean of the previous four actual readings. |
| `roll_std_4` | How much those previous four readings varied. |
| `roll_mean_96` | Average of the previous day's readings. |
| `roll_max_96` | Maximum of the previous day's readings. |
| `diff_1` | Difference between the previous two actual readings. |
| `tod_sin`, `tod_cos` | Cyclic time-of-day encoding, keeping midnight close to late evening. |
| `dow_sin`, `dow_cos` | Cyclic day-of-week encoding. |
| `is_weekend` | Whether the predicted interval is on a weekend. |

The prepared training set has 28,511 rows, from January 8 through October 31.
The prepared test set has 5,857 rows, from November 1 through January 1 at 00:00.
The endpoint reflects the supplied midnight timestamp handling, not new 2019
raw data. Raw data has 35,040 rows; lag construction removes the initial history.

Say about cross-validation:

> Five time-series folds train on an earlier block and validate on the next
> block. The training block grows each time. Candidate settings are compared
> using their mean validation RMSE, and the selected model is then fitted on
> all the training rows. Test targets are not used in that final fitting.

For the advanced search, we screened 37 Linear Regression configurations and
53 forest configurations on the last two folds. We then evaluated nine
finalists per algorithm over all five folds, including the previous version.
These are configuration counts, not numbers of trees or features.

Selection CV RMSE: Linear Regression 9.912; Random Forest 9.641 kWh.
These are tuning scores, not an independent nested cross-validation result.

## 4. Explain the result table

### What the row names mean

- **Naive:** predicts the next usage equals the previous actual usage. No learning.
- **Original:** your initial trained version.
- **Previous optimized:** an earlier round of improvements.
- **Advanced, FINAL:** your latest selected version within that model family.
  It does not mean the group has proved it the best of all five algorithms.

### What the columns mean

| Metric | Say this |
| --- | --- |
| MAE | Average size of the prediction errors, ignoring their direction. Lower is better. |
| RMSE | An error measure that penalises large mistakes more strongly. Lower is better. |
| R2 | A unitless measure of performance relative to predicting the test target's mean. One is perfect, zero matches that baseline, and negative is possible. It is not percentage accuracy. |
| HighLoad_MAE | MAE calculated only where the actual consumption reaches the chosen threshold. |

Example: actual readings `[10, 20]` and predictions `[12, 16]` have absolute
errors `[2, 4]`. MAE is `(2 + 4) / 2 = 3` kWh.
RMSE is `sqrt((2**2 + 4**2) / 2)`, about 3.162 kWh.

Our threshold is **actual Usage_kWh >= 50**, including exactly 50.
There are 1,491 such intervals in the test set. It is an evaluation subset,
not a current measurement, not the Load_Type label and not a certified overload
limit. Actual usage is used to select the subset after predictions are made.

| Current test result | MAE | RMSE | R2 | HighLoad_MAE |
| --- | ---: | ---: | ---: | ---: |
| Naive | 5.155 | 11.813 | 0.856 | 12.420 |
| Linear Regression | 4.128 | 7.880 | 0.936 | 9.671 |
| Random Forest | 3.645 | 7.735 | 0.938 | 9.515 |

Say:

> Random Forest performed better than my Linear Regression model on these
> reported test metrics. Its overall average absolute error is about 3.645 kWh.
> During high-consumption intervals it is about 9.515 kWh. These are averages,
> not a promise that every forecast falls within that error.

Its 34.5% improvement means lower RMSE relative to naive, not 34.5% accuracy.
Linear Regression's corresponding improvement is 33.3%.

## 5. Exact live demonstration sequence

### A. Saved experiment history

```bash
.venv/bin/python src/show_model_results.py
```

Say: "This verifies and displays saved experiment results. It is not training."

### B. Fresh predictions from the trained models

```bash
.venv/bin/python viva/live_demo.py --model linear
.venv/bin/python viva/live_demo.py --model forest
```

Say: "This loads the trained estimator, calls predict on the test inputs, then
computes errors against the test answers and verifies the saved predictions."

### C. Notebook code you can type and explain

Open `VIVA_PRACTICE.ipynb`. It contains short cells for each of these steps:

| Step | Important code | Say this |
| --- | --- | --- |
| Imports | `import pandas as pd` | Pandas handles the tables; imports make library tools available. |
| Inputs | `X_train, X_test, y_train, y_test` | Inputs and answers are separated into earlier training and later testing data. |
| Inspect | `X_test.head()` | Shows the first five input rows. |
| Load | `joblib.load(...)` | Restores a previously trained estimator with its learned preprocessing. |
| Predict | `model.predict(X_test)` | Generates forecasts from the input features, without receiving test answers. |
| Score | `mean_absolute_error(y_test, predictions)` | Compares predictions with actual readings after prediction. |
| High load | `mask = y_test >= 50` | Selects the agreed high-consumption subset for scoring. |
| Verify | `np.testing.assert_allclose(...)` | Checks new predictions reproduce the recorded results. |

### D. If asked to demonstrate actual training

The notebook has a separate, optional training example at the end. Do not run
the long search just to display results. The important distinction is:

```python
practice_model.fit(X_train, y_train)  # learns from training inputs and answers
practice_predictions = practice_model.predict(X_test)  # predicts without y_test
```

The optional example refits the already selected settings in memory and does
not save over your official model. Linear Regression should finish quickly;
the forest's recorded final fit took about 206 seconds on this machine. Time
varies by machine. Neither command repeats the hyperparameter search.

## 6. Likely viva questions and short answers

### Why these two algorithms?

> Linear Regression gives a useful coefficient-based starting point. Random
> Forest can model nonlinear patterns and interactions. Comparing them lets us
> test whether more flexible relationships improve forecasting.

### What is overfitting? Is it absent here?

> Overfitting means learning training-specific patterns that do not generalise
> well. We check performance on later observations. The forest's training RMSE
> is about 4.263 while test RMSE is 7.735, so a generalisation gap remains.
> Tuning does not prove that overfitting is absent.

### What is a hyperparameter?

> A setting chosen before fitting, such as tree count or minimum leaf size.
> Learned coefficients and tree split rules are fitted parameters, not the
> same thing as hyperparameters.

### What is a pipeline?

> A sequence of transformations and a model kept together so new inputs get
> the same processing. Learned transforms are fitted on the fitting portion,
> not freshly fitted to test inputs.

### Why not train on every row?

> We need later observations not used for fitting to evaluate generalisation.
> Training on test targets and then reporting those scores would not be a
> valid test. A later deployment refit is separate from our evaluation model.

### Are previous actual readings allowed during testing?

> Yes for our one-step setup: the previous interval has finished before the
> next forecast. This does not mean we can forecast an entire future day using
> actual readings that have not happened yet.

### Is the group winner definitely Random Forest?

> Not yet. It is a leading candidate. KNN and Gradient Boosting currently use
> different test rows and high-load definitions. We must compare on one agreed
> protocol before claiming an overall winner.

### Did you optimise directly for high-load error?

> No. The current selection objective was validation RMSE. High-load MAE was
> also reported. If we make high-load MAE the primary objective, we must revise
> the validation comparison explicitly rather than claim it was always so.

### What limitations must you acknowledge?

> The results use our supplied preprocessing artifacts. Timestamp handling
> needs confirmation, and one test reading was interpolated using a future
> reading. We recorded a sensitivity check excluding affected rows, but this
> does not repair the whole pipeline. Earlier feature selection used the full
> training period, and test results have already been inspected during
> development. We cannot claim a completely untouched final holdout.

Do not present known limitations as hypothetical problems that never occurred.
Both selected models ignore load level; do not say they require planned load
type. Same-interval unavailable meter measurements must not be forecast inputs.

### What did your teammates implement?

- KNN: predicts using nearby examples in feature space; scaling matters.
- SVR: fits a regression function with a tolerance around errors; their RBF
  kernel can model nonlinear relationships and their pipeline scales inputs.
- Gradient Boosting: adds trees sequentially to correct remaining errors;
  learning rate and tree complexity are tuned.

Know the distinction: forests combine randomly varied trees; boosting builds
successive corrections. Do not claim you developed your teammates' code.

### What was your contribution? Was assistance used?

> My assigned contribution is the Linear Regression and Random Forest work:
> running the models, investigating settings, checking results and explaining
> the comparison. I used coding assistance and am verifying and learning the
> implementation. I can demonstrate the parts I understand.

Adapt that to work you actually performed and follow the course's disclosure
policy. Do not claim to have independently handwritten all the generated code.

### What happens after selection?

> Integrate the chosen estimator and its preprocessing into a backend, validate
> inputs, return predictions and connect a frontend. Then test the full system
> and complete the report and presentation. The current prediction demo is not
> the completed frontend/backend system.

## 7. What to have open

1. `VIVA_PRACTICE.ipynb` with the correct kernel.
2. Your original and advanced model notebooks, to show experiment progression.
3. `reports/ADVANCED_TRAINING_REPORT.md` and the results command output.
4. `outputs_advanced/evaluation.png` in each model folder for error plots.
5. The team comparison caveat and the exact metric definitions.

Progress Evaluation 2 is individual and covers algorithm justification,
validation, metrics, tuning, interpretation, final selection and your technical
contribution (assignment page 3, Stages 6-8). A provisional choice must not be
presented as a completed, fair final selection when that work remains.

## Source evidence

- Assignment: `/Users/chelakawijesekara/Downloads/IT3051_FDM_Mini_Project_Assignment_2026 (1).pdf`, page 3.
- Proposal: `/Users/chelakawijesekara/Downloads/data_miners_dataset_proposal_FDM_mini_project (1).pdf`, scenario and task sections.
- Current results/configuration: each model's `outputs_advanced` folder.
- Implementation: `src/advanced_training.py`, `src/optimized_regression.py`, `src/regression_common.py`.
- Cross-model inconsistencies: the supplied KNN/Gradient Boosting notebooks and screenshots, and `reports/SVR_REFERENCE_NOTES.md`.

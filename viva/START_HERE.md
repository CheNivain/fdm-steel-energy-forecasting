# Start Here: Your Viva Rehearsal

Do these in order. You do not need to memorise every line of the training search.

## 1. Practise this opening

> Our project predicts the steel plant's energy consumption for the next
> 15-minute interval. The target is Usage_kWh, so this is a regression problem.
> My assigned models are Linear Regression and Random Forest. We use calendar
> information and previous consumption readings. We compared model settings
> using time-ordered validation and evaluated predictions using MAE, RMSE,
> R-squared and high-load MAE.

Do not say that kWh measures electrical current. It measures energy.

## 2. Show your existing results

In VS Code, open the FDM Project folder. Select Terminal > New Terminal.
Type these two commands, one at a time:

```bash
cd "/Users/chelakawijesekara/Downloads/FDM Project"
.venv/bin/python src/show_model_results.py
```

Say: "This command verifies and displays the results saved by our experiments.
It does not train the models again. The FINAL rows are my current versions."

## 3. Make predictions live

```bash
.venv/bin/python viva/live_demo.py --model both
```

Say: "Now I am loading the trained models, making predictions again, and
recalculating the errors against the actual test readings."

The forest file is large, so allow it time to load. The demo does not overwrite
your trained models. Its predictions must match the saved results.

| Current model | MAE | RMSE | R2 | HighLoad_MAE |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 4.128 | 7.880 | 0.936 | 9.671 |
| Random Forest | 3.645 | 7.735 | 0.938 | 9.515 |

Errors are in kWh; R2 is not percentage accuracy. High load here means actual
usage >= 50 kWh. These numbers describe the existing prepared test data and
retain the documented preprocessing limitations.

## 4. Practise code cells

Open `VIVA_PRACTICE.ipynb` in this folder. In the top-right kernel selector,
choose the project's `.venv/bin/python`. Run each cell in order using its play
button or Shift+Enter. Read the short "Say" sentence before each cell.

Start with `KIND = "linear"`. Then change it to `KIND = "forest"` and run all
cells again. These cells demonstrate inference, not a new tuning search.

## 5. Prepare your answers

Read `VIVA_GUIDE.md`, sections 1-5 first. Use section 6 for likely questions.
Send `TEAM_VIVA_KIT.zip` to your teammates. It contains `TEAM_GUIDE.md` and
`team_results.py`, not anyone's data or trained models.

## If asked which model won

> Random Forest is our current leading candidate. However, KNN and Gradient
> Boosting were evaluated on a different test period and a different high-load
> threshold. We must align the evaluation before claiming a final winner among
> all five models. The current tuning objective was RMSE, not high-load MAE.

This is an honest account of the work, not a completed final-selection claim.

## If something fails

- `No such file`: check that the terminal is in FDM Project, not the older FDM folder.
- Missing `X_test.csv`: the original prepared files are in `../FDM/artifacts`.
  Keep that folder. Use `--artifacts "/path/to/artifacts"` if it was moved.
- Notebook cannot import a library: select the project `.venv` kernel.
- Data/hash/prediction mismatch: stop and investigate. Do not bypass the check.
- Never launch the full tuning command just to print the results during the viva.

Models are already trained. A viva pass also depends on your own understanding;
this guide cannot guarantee an assessment outcome.

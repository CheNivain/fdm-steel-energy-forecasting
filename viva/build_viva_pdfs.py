"""Create two sourced, beginner-friendly PDFs from the recorded experiment evidence."""

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Preformatted, KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
FONTS = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pdfjs-dist/standard_fonts"
for name, filename in [("GuideSans", "LiberationSans-Regular.ttf"),
                       ("GuideSans-Bold", "LiberationSans-Bold.ttf"),
                       ("GuideSans-Italic", "LiberationSans-Italic.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / filename)))
pdfmetrics.registerFontFamily("GuideSans", normal="GuideSans", bold="GuideSans-Bold", italic="GuideSans-Italic")
MONO = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype/DejaVuSansMono.ttf"
pdfmetrics.registerFont(TTFont("GuideMono", str(MONO)))

INK = colors.HexColor("#202a30")
GREEN = colors.HexColor("#13685b")
MUTED = colors.HexColor("#53636b")
RULE = colors.HexColor("#d7dfe2")
PALE = colors.HexColor("#edf5f2")
WIDTH = A4[0] - 88
STYLES = {
    "title": ParagraphStyle("title", fontName="GuideSans-Bold", fontSize=25, leading=29, textColor=INK, spaceAfter=10),
    "subtitle": ParagraphStyle("subtitle", fontName="GuideSans", fontSize=11, leading=15, textColor=MUTED, spaceAfter=13),
    "h": ParagraphStyle("h", fontName="GuideSans-Bold", fontSize=13, leading=17, textColor=GREEN, spaceBefore=11, spaceAfter=7, keepWithNext=True),
    "body": ParagraphStyle("body", fontName="GuideSans", fontSize=11, leading=15, textColor=INK, spaceAfter=8),
    "small": ParagraphStyle("small", fontName="GuideSans", fontSize=9.1, leading=12.3, textColor=MUTED, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="GuideSans", fontSize=9.6, leading=12.5, textColor=INK),
    "cellsmall": ParagraphStyle("cellsmall", fontName="GuideSans", fontSize=8.8, leading=11.7, textColor=INK),
    "head": ParagraphStyle("head", fontName="GuideSans-Bold", fontSize=9.1, leading=12, textColor=colors.white),
    "quote": ParagraphStyle("quote", fontName="GuideSans", fontSize=11.2, leading=15.5, textColor=INK, spaceAfter=0),
    "code": ParagraphStyle("code", fontName="GuideMono", fontSize=8.4, leading=12, textColor=INK, backColor=colors.HexColor("#f2f4f5"), borderPadding=9, spaceAfter=10),
}


def p(text, style="body"):
    return Paragraph(text, STYLES[style])


def heading(text):
    return p(text, "h")


def quote(text):
    box = Table([[p("SAY THIS", "small")], [p(text, "quote")]], colWidths=[WIDTH])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("LINEBEFORE", (0, 0), (0, -1), 3, GREEN),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 11),
    ]))
    return KeepTogether([box, Spacer(1, 9)])


def table(headers, rows, widths, compact=False, shaded=None):
    style = "cellsmall" if compact else "cell"
    content = [[p(str(x), "head") for x in headers]]
    content += [[p(str(x), style) for x in row] for row in rows]
    result = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6 if compact else 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6 if compact else 7),
        ("LINEBELOW", (0, 0), (-1, 0), 1, INK),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
    ]
    for i in range(1, len(content)):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f8f8")))
    for row in shaded or []:
        commands.append(("BACKGROUND", (0, row), (-1, row), PALE))
    result.setStyle(TableStyle(commands))
    return result


def code(text):
    return Preformatted(text, STYLES["code"])


def page_title(kicker, title, subtitle):
    return [p(kicker.upper(), "small"), p(title, "title"), p(subtitle, "subtitle")]


def build(filename, story, label):
    OUT.mkdir(parents=True, exist_ok=True)
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(44, 39, A4[0] - 44, 39)
        canvas.setFont("GuideSans", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(44, 26, f"DATA MINERS  |  {label}  |  03 OCT 2026")
        canvas.drawRightString(A4[0] - 44, 26, f"{doc.page}")
        canvas.restoreState()
    doc = SimpleDocTemplate(str(OUT / filename), pagesize=A4, leftMargin=44, rightMargin=44,
                            topMargin=42, bottomMargin=52, title=label, author="Data Miners",
                            subject="FDM progress evaluation 2 preparation and model evidence")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT / filename)


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def local_score(folder, version):
    directory = ROOT / folder / version
    metrics = next(row for row in csv_rows(directory / "metrics.csv")
                   if row["split"] == "test" and row["model"] != "Persistence")
    predictions = csv_rows(directory / "test_predictions.csv")
    high = [r for r in predictions if float(r["actual_kWh"]) >= 50]
    high_mae = sum(abs(float(r["actual_kWh"]) - float(r["predicted_kWh"])) for r in high) / len(high)
    return [float(metrics[k]) for k in ["MAE_kWh", "RMSE_kWh", "R2"]] + [high_mae]


LR = local_score("linear regression", "outputs_advanced")
RF = local_score("random forest", "outputs_advanced")
SVR = [4.495, 9.099, 0.915, 10.901]
KNN = [4.317, 9.185, 0.887, 26.187]
GB = [3.303, 6.872, 0.937, 19.127]
NAIVE = [5.155234761823459, 11.81254926681705, 0.8560360639357736, 12.419590878604964]


def final_rows(protocol=True):
    values = [("Linear Regression", LR, "A"), ("Random Forest", RF, "A"),
              ("SVR", SVR, "A*"), ("Gradient Boosting", GB, "C"), ("KNN", KNN, "B")]
    return [[name, *[f"{x:.3f}" for x in scores], *([group] if protocol else [])]
            for name, scores, group in values]


def result_table():
    return table(["Model", "MAE", "RMSE", "R2", "High-load\nMAE", "Setup"],
                 final_rows(), [173, 65, 68, 48, 97, WIDTH-451], compact=True, shaded=[2])


def viva_guide():
    story = page_title("Viva guide / 1 of 4", "Start with the basics", "Short answers you can practise aloud. Learn one page at a time.")
    story += [heading("Your opening: about 40 seconds"), quote(
        "Our project predicts the steel plant's energy consumption for the next 15-minute interval. "
        "The target is Usage_kWh. Because the output is a number, this is a regression problem. "
        "My assigned models are Linear Regression and Random Forest. We use previous energy readings "
        "and calendar information to make the prediction."),
        heading("The basic words"),
        table(["Word", "What it means here"], [
            ["Dataset", "A table of past observations. Our raw dataset has 35,040 rows and 11 columns."],
            ["Row / observation", "One 15-minute interval at the steel plant."],
            ["Target / y", "The answer we want to predict: Usage_kWh."],
            ["Features / X", "The information given to the model, such as past usage and time of day."],
            ["Model", "A mathematical method that learns patterns from examples."],
            ["Regression", "Predicting a numerical value, not choosing a class label."],
            ["kWh", "A unit of energy. It is not electrical current in amps."],
        ], [126, WIDTH-126]),
        heading("What is the project useful for?"), quote(
            "Plant operations and energy managers could use the forecasts to anticipate heavy consumption "
            "and support planning. The forecast is decision support, not a guarantee of cost savings."),
        p("Example: a forecast of 45 kWh means predicted energy use during the requested 15-minute interval. "
          "It does not mean 45% accuracy or 45 amps."),
        p("The approved proposal concerns energy forecasting and peak-tariff avoidance. High consumption "
          "and an expensive tariff period are different concepts. [Guide sources: assignment page 3; proposal pages 3-7.]", "small"),
        PageBreak()]

    story += page_title("Viva guide / 2 of 4", "Explain your models", "You need the idea first, then the important settings.")
    story += [heading("Linear Regression"), quote(
        "Linear Regression learns weights for the input features. It adds their contributions together "
        "to predict usage. We improved our basic version by creating more useful features."),
        p("Our advanced version uses <b>651 engineered columns from 15 inputs</b>, including smooth usage "
          "transformations and time/weekend interactions. It is still Linear Regression, but not just a "
          "straight-line model on the original 15 columns. Its scaling is saved inside the pipeline."),
        heading("Random Forest"), quote(
        "Random Forest combines many decision trees. Each tree learns rules from sampled data. "
        "Combining the trees can capture complex patterns more reliably than using one tree."),
        p("Our final forest has <b>1,000 trees</b> and learns the change from the last usage reading. "
          "If previous usage is 40 kWh and predicted change is +5 kWh, the final prediction is 45 kWh. "
          "The saved model does this addition automatically. It uses 70 engineered columns."),
        heading("Three inputs you should know"),
        table(["Input", "Meaning"], [["lag_1", "Previous 15-minute interval's actual usage."],
                                     ["lag_96", "Actual usage one day earlier: 96 intervals per day."],
                                     ["roll_mean_4", "Average of the previous four readings, covering one hour."]], [126, WIDTH-126]),
        heading("Training, validation and testing"), quote(
        "Training teaches the model using earlier inputs and their answers. Validation helps us choose "
        "settings. Testing measures performance on later observations not used for fitting."),
        p("We used five time-series validation folds: train on an earlier block, check the next block, "
          "then expand the training block. The selected settings were refitted on 28,511 training rows. "
          "The test set has 5,857 rows. Selection used <b>mean validation RMSE</b>, not high-load MAE."),
        p("A hyperparameter is a setting chosen before fitting, such as tree count or minimum leaf size. "
          "Our forest uses a minimum leaf size of 2 and no fixed depth cap. Both selected models ignore "
          "load_level, although it remains in their input interface.", "small"),
        PageBreak()]

    story += page_title("Viva guide / 3 of 4", "Read the results", "Lower error is better. R2 is a score, not percentage accuracy.")
    story += [table(["Metric", "Basic explanation"], [
        ["MAE", "The average size of the prediction errors, ignoring whether they are above or below the actual value."],
        ["RMSE", "An error measure that gives more weight to large mistakes."],
        ["R2", "Performance relative to predicting the test target's mean. 1 is perfect; 0 matches that baseline; negative is possible."],
        ["HighLoad_MAE", "Average absolute error only during intervals whose actual usage reaches the high-load threshold."],
    ], [106, WIDTH-106]),
        heading("All five reported final models"), result_table(),
        p("MAE, RMSE and HighLoad_MAE are in kWh. These are <b>not a fair five-way ranking</b>. "
          "A: 5,857 test rows and a 50 kWh cutoff. B: KNN uses 2,880 rows and about 80.41 kWh. "
          "C: Gradient Boosting uses 2,880 rows and about 80.14 kWh. A*: SVR reports the matching setup, "
          "but exact input identity is unverified.", "small"),
        heading("How to explain your Random Forest result"), quote(
            "Our Random Forest's overall MAE is about 3.645 kWh. Its high-load MAE is about 9.515 kWh "
            "for intervals where actual usage is at least 50 kWh. These are average errors, "
            "not maximum-error guarantees. R2 of 0.938 is not 93.8% accuracy."),
        heading("Why give high-load error priority?"), quote(
            "Our team wants reliable forecasts during heavy-consumption periods. Overall error averages "
            "all intervals, so many quiet periods can hide poor peak-period predictions. High-load MAE "
            "focuses on those important intervals. We still check overall MAE, RMSE and R2."),
        PageBreak()]

    story += page_title("Viva guide / 4 of 4", "Your live demonstration", "Use the FDM Project folder in VS Code. Do not launch a long search just to show results.")
    story += [heading("1. Print your saved experiment results"),
        code('cd "/Users/chelakawijesekara/Downloads/FDM Project"\n.venv/bin/python src/show_model_results.py'),
        p('Say: "This verifies and displays saved results. It does not train the models again."'),
        p("Naive = previous reading. Original / previous optimized = older versions. FINAL = latest "
          "version of that algorithm, not a verified five-model winner.", "small"),
        heading("2. Make predictions live"),
        code('.venv/bin/python viva/live_demo.py --model both'),
        p('Say: "I load the trained models, generate predictions, and recalculate the errors against the actual test readings."'),
        heading("3. Show the code cells"),
        p("Open <b>viva/VIVA_PRACTICE.ipynb</b>. Select the project's .venv Python kernel. "
          'Run the cells in order. Start with <b>KIND = "linear"</b>; then use <b>KIND = "forest"</b>.'),
        table(["Code", "What to say"], [
            ["joblib.load(...)", "Restore the already-trained model and its preprocessing."],
            ["model.predict(X_test)", "Generate forecasts from inputs, without passing test answers."],
            ["mean_absolute_error(...)", "Compare predictions with actual readings after prediction."],
            ["mask = y_test &gt;= 50", "Select the agreed high-consumption intervals for scoring."],
            ["model.fit(X_train, y_train)", "Learn from training examples. This is training, unlike predict."],
        ], [210, WIDTH-210], compact=True),
        p("The notebook ends with an optional fitting example. Run it only when asked; it does not "
          "overwrite your official models. The forest can take several minutes. Friends should use "
          "viva/TEAM_VIVA_KIT.zip with their own fitted models and matching test inputs.", "small"),
        heading("Your selection answer"), quote(
            "We choose Random Forest as the working model for the next prototype because it performs "
            "well on both high-load and overall errors in the matching reported setup. However, KNN "
            "and Gradient Boosting used different evaluations. A fair final comparison is still needed."),
        p("Be honest: these are one-step forecasts, not a whole day forecast at once. The supplied "
          "preprocessing has a future-interpolation issue and timestamp assumptions; test results were "
          "already seen during development. Explain your actual contribution and disclose assistance "
          "as required by your course. Do not claim a guaranteed pass or an error-free model.", "small"),
    ]
    build("FDM_Beginner_Viva_Guide.pdf", story, "BEGINNER VIVA GUIDE")


def model_report():
    story = page_title("Model report / 1 of 4", "Five-model results review", "Steel-industry energy forecasting | Recorded evidence as at 3 October 2026")
    story += [quote("<b>Working decision: Random Forest for the next prototype.</b> This is a conditional "
        "engineering choice, not proof of the best model across all five. Common validation and "
        "test reporting are still required before the overall winner can be confirmed."),
        heading("1. Reported final metrics"), result_table(),
        p("Lower MAE, RMSE and HighLoad_MAE are better; higher R2 is better. R2 is not percentage "
          "accuracy. All errors are in kWh. RF/LR values were recalculated from saved predictions; "
          "SVR/KNN/Gradient Boosting values are reported notebook/screenshot results, not newly "
          "reproduced model runs. [3-6]", "small"),
        heading("2. What each evaluation actually measured"),
        table(["Setup", "Models", "Train rows", "Test rows", "High-load rule", "High rows"], [
            ["A / A*", "LR, RF, SVR*", "28,511", "5,857", "Actual &gt;= 50", "1,491**"],
            ["B", "KNN", "32,064", "2,880", "Actual &gt;= 80.41", "81"],
            ["C", "Gradient Boosting", "28,608", "2,880", "Actual &gt;= 80.14", "82"],
        ], [40, 113, 72, 63, 123, WIDTH-411], compact=True),
        p("* SVR shares the reported dimensions, cutoff and rounded naive scores of A; exact input-file "
          "identity is not independently verified. ** 1,491 is verified for LR/RF; the SVR high-row "
          "count was not printed. KNN/GB cutoffs above are rounded; their full cutoff must be preserved.", "small"),
        p("LR/RF test timestamps run from 1 Nov 2018 00:00 to 1 Jan 2019 00:00 under the supplied "
          "midnight handling. KNN's code and raw data give 2-31 Dec 2018. The actual GB processed "
          "test dates were not supplied. Equal row counts or matching rounded baselines alone "
          "do not prove identical evaluations."),
        heading("3. What the comparison supports"),
        p("Within the matching reported A setup, RF has lower errors than LR and SVR. But RF's "
          "9.515 high-load error cannot fairly be ranked against GB's 19.127 or KNN's 26.187: "
          "they score a different, more extreme usage subset on a different period. Similarly, "
          "GB's RMSE of 6.872 does not prove it beats RF's 7.735 on the same task."),
        PageBreak()]

    rows = [["A: Naive", *[f"{x:.3f}" for x in NAIVE]]]
    for short, folder in [("LR", "linear regression"), ("RF", "random forest")]:
        for label, version in [("original", "outputs"), ("previous optimized", "outputs_optimized"), ("advanced FINAL", "outputs_advanced")]:
            rows.append([f"A: {short} {label}", *[f"{x:.3f}" for x in local_score(folder, version)]])
    rows += [
        ["A*: LinearSVR untuned", "5.135", "11.389", "0.866", "12.345"],
        ["A*: SVR RBF default", "5.023", "9.136", "0.914", "10.839"],
        ["A*: SVR RBF FINAL", "4.495", "9.099", "0.915", "10.901"],
        ["B: Naive", "3.904", "9.742", "0.873", "23.668"],
        ["B: KNN default", "4.846", "10.100", "0.863", "27.556"],
        ["B: KNN FINAL", "4.317", "9.185", "0.887", "26.187"],
        ["C: Naive", "3.904", "9.742", "0.873", "23.554"],
        ["C: GB default", "3.859", "7.208", "0.930", "19.520"],
        ["C: GB FINAL", "3.303", "6.872", "0.937", "19.127"],
    ]
    story += page_title("Model report / 2 of 4", "All reported versions", "Compare each model with its own baseline before comparing algorithms. [3-6]")
    story += [table(["Setup / model version", "MAE", "RMSE", "R2", "HighLoad_MAE"],
                    rows, [216, 63, 66, 62, WIDTH-407], compact=True, shaded=[4, 7, 10, 13, 16]),
        p("A/B/C refer to the different evaluation setups on page 1, not performance ranks. "
          "The A naive scores also match the rounded SVR baseline. Original RF already used "
          "an eight-setting search; 'original' does not mean untuned. LR/RF values are rounded "
          "from stored predictions, including LR original HighLoad_MAE = 12.0704865, shown as 12.070.", "small"),
        heading("Interpretation in plain language"),
        p("<b>Linear Regression:</b> feature engineering improved all four displayed metrics from the "
          "original to advanced version. It remains a useful, lightweight alternative.", "small"),
        p("<b>Random Forest:</b> all four displayed test metrics improved from original to advanced. "
          "Its high-load edge over LR is only about 0.156 kWh, or 1.6%; no significance test was done.", "small"),
        p("<b>SVR:</b> tuning improved overall MAE/RMSE, but high-load MAE worsened slightly from "
          "10.839 to 10.901. 'Tuned' does not mean better on every metric.", "small"),
        p("<b>KNN:</b> tuning improved its default model, but high-load MAE and overall MAE remain "
          "worse than its naive rule. RMSE is better than naive. <b>Gradient Boosting:</b> its final "
          "model beats its own naive and default versions on all four metrics, making it a credible "
          "contender once the evaluation is aligned.", "small"),
        p("<b>RMSE reduction versus each model's own naive baseline:</b> LR 33.3%, RF 34.5%, "
          "SVR 23.0%, KNN 5.7%, GB 29.5%. These are error reductions, not accuracy percentages; "
          "different evaluation setups still prevent a fair overall ranking.", "small"),
        PageBreak()]

    story += page_title("Model report / 3 of 4", "What to prioritise", "A business priority must be stated explicitly; it is not a universal rule for regression.")
    story += [heading("Why make high-load MAE the primary metric?"),
        p("The proposed system supports planning around heavy consumption and costly operating "
          "periods. If the team agrees that mistakes during heavy-consumption intervals matter most, "
          "high-load MAE is a useful primary criterion for these reasons: [2]"),
        table(["Reason", "Simple explanation"], [
            ["Focus on important intervals", "It measures errors when actual consumption is high, instead of treating every operating period as equally important."],
            ["Avoid hidden peak errors", "A low average error across many quiet intervals can hide poor predictions during fewer heavy-consumption intervals."],
            ["Easy to explain", "The score is in kWh. A value of 9.515 means average absolute error of 9.515 kWh on the selected high-load intervals."],
            ["Link to the proposed use", "Better heavy-consumption estimates may help managers anticipate spikes and plan energy-intensive work. This benefit still needs system-level testing."],
        ], [136, WIDTH-136], compact=True),
        p("<b>Important limits:</b> high usage is not the same as peak-tariff time. This metric alone "
          "does not prove money saved, safe operation or reliable overload warnings. It treats "
          "underprediction and overprediction equally. If missed peaks are especially costly, "
          "also examine underprediction errors and warning performance."),
        heading("A proposed decision rule for the next comparison"),
        quote("Agree one high-load definition first. On identical chronological validation windows, "
              "prefer the lowest high-load MAE, provided overall MAE/RMSE and runtime are acceptable. "
              "Keep R2, fold variation and the naive baseline in the report. Do not choose the "
              "threshold after looking at which model it favours."),
        p("A 50 kWh cutoff is already used by LR/RF/SVR and is near the reported training 75th "
          "percentile of 51.41 kWh. It is a project evaluation choice, not a verified physical limit. "
          "The KNN code uses its training 90th percentile; GB's screenshot reports a similar rule. "
          "A percentile-based cutoff must use fitting data, not future validation/test answers.", "small"),
        heading("What the current tuning actually did"),
        p("The inspected searches selected configurations using <b>validation RMSE</b>. They were "
          "not explicitly optimised for minimum HighLoad_MAE. LR's final five-fold mean RMSE / "
          "high-load MAE is 9.912 / 12.344; RF's is 9.641 / 11.994. RF's previous configuration "
          "had slightly better high-load CV error, 11.906, despite worse RMSE. A high-load-first "
          "policy therefore requires a new, transparent validation comparison. [3]", "small"),
        PageBreak()]

    high_gain = 100 * (1 - RF[3] / NAIVE[3])
    rmse_gain = 100 * (1 - RF[1] / NAIVE[1])
    story += page_title("Model report / 4 of 4", "Decision: Random Forest", "Selected as the working model for the next prototype, pending a common five-model evaluation.")
    story += [heading("Why proceed with this model?"),
        p(f"<b>1. Strong heavy-consumption result.</b> Its HighLoad_MAE is {RF[3]:.3f} kWh in setup A: "
          f"about {high_gain:.1f}% below its naive baseline. It is also lower than the reported LR "
          "and SVR final scores under the matching reported setup."),
        p(f"<b>2. Good overall performance as well.</b> MAE = {RF[0]:.3f}, RMSE = {RF[1]:.3f} and "
          f"R2 = {RF[2]:.3f}. RMSE is about {rmse_gain:.1f}% below naive. The recommendation does "
          "not rely on high-load error alone."),
        p("<b>3. A verified prediction artifact is available.</b> The saved forest reproduced its "
          "recorded predictions in the local live demo. It includes feature transformations, the "
          "change-to-usage conversion and a nonnegative floor. It does not require load_level to be known."),
        heading("Trade-offs and remaining work"),
        p("RF is not the cheapest option: recorded final fit time was 205.85 s, versus 1.05 s for "
          "LR; compressed artifacts are about 326 MB versus 0.03 MB. Friend screenshots report "
          "SVR 12.8 s, KNN 0.1 s and GB 43.5 s, but hardware and work differ, so these are not "
          "controlled speed benchmarks. Fit time excludes the search. [3-6]", "small"),
        p("RF's training RMSE is 4.263 versus test RMSE 7.735, so a generalisation gap remains. "
          "Its small test advantage over LR is not proof of statistical superiority. GB is still a "
          "serious alternative. Align all five training cutoffs, test timestamps, target cleaning, "
          "forecast horizon and thresholds; use validation for selection rather than repeatedly "
          "tuning to the already viewed test period.", "small"),
        p("The supplied preprocessing shifts midnight timestamps and interpolated one test reading "
          "using a future value. Upstream feature selection also preceded the new validation splits. "
          "A sensitivity check excluded affected test rows but did not repair the full pipeline. "
          "KNN/SVR load-category availability needs confirmation. GB's actual screenshot script and "
          "processed data are missing; the supplied GB notebook does not calculate HighLoad_MAE.", "small"),
        heading("What to say at the viva"), quote(
          "We selected Random Forest as our working model because it combines low high-load error "
          "with strong overall error scores and a verified saved pipeline. We prioritise high-load "
          "error because heavy-consumption periods matter to our planned use. However, our current "
          "tuning used RMSE, and the other evaluation setups differ. We will confirm the final "
          "choice using one common validation and test protocol."),
        heading("Evidence sources"),
        p("<b>[1]</b> IT3051_FDM_Mini_Project_Assignment_2026 (1).pdf, page 3, Stages 6-8. "
          "<b>[2]</b> data_miners_dataset_proposal_FDM_mini_project (1).pdf, pages 3, 6-7. "
          "<b>[3]</b> Local LR/RF outputs, outputs_optimized and outputs_advanced: metrics, "
          "test_predictions, run_metadata and verification; reports/ADVANCED_TRAINING_REPORT.md. "
          "<b>[4]</b> svr_model_train.ipynb and SVR screenshot (2 Oct, 14:47). "
          "<b>[5]</b> KNN_Tuning_Report_Colab.ipynb and KNN screenshot (3 Oct, 10:07). "
          "<b>[6]</b> gradient_boosting_tuning.ipynb and supplied GB terminal screenshot "
          "(clipboard cdd81805). Friend models were not rerun for this report. "
          "These PDFs organise existing evidence; they do not constitute a new fair five-way experiment.", "small"),
    ]
    build("FDM_Five_Model_Comparison_Report.pdf", story, "FIVE-MODEL COMPARISON")


if __name__ == "__main__":
    viva_guide()
    model_report()

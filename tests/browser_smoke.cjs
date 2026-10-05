// Playwright acceptance checks against an isolated test instance of the app.
const { chromium } = require("playwright");
const fs = require("node:fs");
const assert = require("node:assert/strict");
const base = process.env.FDM_TEST_URL || "http://127.0.0.1:8001";
const out = "tmp/app-qa";

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME_PATH
      ? { executablePath: process.env.CHROME_PATH }
      : {}),
  });
  const errors = [];
  const checks = [];
  try {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 1000 },
      acceptDownloads: true,
    });
    const page = await context.newPage();
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (msg) => {
      if (msg.type() === "error" && !msg.text().includes("422"))
        errors.push(msg.text());
    });
    await page.goto(`${base}/#forecast`);
    await page.getByText("Model ready · Local").waitFor();
    await page.screenshot({
      path: `${out}/desktop-initial.png`,
      fullPage: true,
    });
    await page.getByRole("button", { name: "Generate forecast" }).click();
    await page.getByText("PREDICTED ENERGY USAGE", { exact: true }).waitFor();
    assert.match(await page.locator(".forecast-number").innerText(), /64\.59/);
    checks.push("Desktop: saved model returns expected forecast 64.59 kWh.");
    await page.screenshot({
      path: `${out}/desktop-forecast.png`,
      fullPage: true,
    });
    const chartPixels = await page
      .locator("#forecast-chart")
      .evaluate((canvas) => {
        const pixels = canvas
          .getContext("2d")
          .getImageData(0, 0, canvas.width, canvas.height).data;
        let count = 0;
        for (let i = 3; i < pixels.length; i += 4) if (pixels[i] > 0) count++;
        return count;
      });
    assert.ok(chartPixels > 1000);
    checks.push(`Forecast chart has ${chartPixels} painted pixels.`);
    await page.selectOption("#dataset-select", "demo-night");
    await page.getByText("Ready for a forecast", { exact: true }).waitFor();
    await page.getByRole("button", { name: "Generate forecast" }).click();
    await page.locator(".tag.good").first().waitFor();
    checks.push(
      "Sample change clears the previous result; overnight forecast is below threshold.",
    );
    const downloadWait = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Download forecast and inputs" })
      .click();
    assert.match(
      (await downloadWait).suggestedFilename(),
      /^forecast_.*\.json$/,
    );
    checks.push("Forecast JSON export downloads.");
    await page.locator('nav [data-page="data"]').click();
    await page
      .locator("#csv-file")
      .setInputFiles({
        name: "invalid.csv",
        mimeType: "text/csv",
        buffer: Buffer.from("x,y\n1,2"),
      });
    await page.getByRole("button", { name: "Validate & import" }).click();
    await page.locator("#upload-error:not(.hidden)").waitFor();
    assert.match(
      await page.locator("#upload-error").innerText(),
      /timestamp and Usage_kWh/,
    );
    checks.push("Invalid CSV shows an actionable field error.");
    const template = await context.request.get(`${base}/api/template.csv`);
    await page
      .locator("#csv-file")
      .setInputFiles({
        name: "QA meter.csv",
        mimeType: "text/csv",
        buffer: await template.body(),
      });
    await page.getByRole("button", { name: "Validate & import" }).click();
    await page.locator("#upload-success:not(.hidden)").waitFor();
    assert.match(
      await page.locator("#upload-success").innerText(),
      /672 valid readings/,
    );
    const row = page
      .locator(".dataset-row")
      .filter({ hasText: "QA meter.csv" })
      .last();
    await row.getByRole("button", { name: "Add next actual reading" }).click();
    await page.fill("#append-value", "67.5");
    await page
      .getByRole("button", { name: "Add reading", exact: true })
      .click();
    await page.waitForFunction(() =>
      document.querySelector("#append-timestamp").value.endsWith("10:15"),
    );
    checks.push(
      "CSV import and next-actual-reading append advance the forecast interval.",
    );
    await page.screenshot({ path: `${out}/desktop-data.png`, fullPage: true });
    await row.getByRole("button", { name: "Forecast", exact: true }).click();
    await page.getByRole("button", { name: "Generate forecast" }).click();
    await page.getByText("PREDICTED ENERGY USAGE", { exact: true }).waitFor();
    assert.match(
      await page.locator("#result-details").innerText(),
      /Next actual not yet supplied/,
    );
    checks.push(
      "Imported history generates a real prediction without claiming a known future actual.",
    );
    await page.locator('nav [data-page="history"]').click();
    await page.locator(".legacy-history summary").click();
    await page.locator("#history-search").fill("QA meter");
    await page
      .getByRole("button", { name: "Inspect forecast record" })
      .first()
      .click();
    await page.locator(".feature-table").waitFor();
    assert.equal(await page.locator(".feature-table tbody tr").count(), 15);
    await page.screenshot({
      path: `${out}/desktop-history.png`,
      fullPage: true,
    });
    const historyDownload = page.waitForEvent("download");
    await page.getByRole("link", { name: "Export replay records", exact: true }).click();
    assert.equal(
      (await historyDownload).suggestedFilename(),
      "forecast_history.csv",
    );
    checks.push(
      "Search, audit record with 15 features, and history CSV export work.",
    );
    await page.locator('nav [data-page="evidence"]').click();
    await page
      .locator("#day-scores")
      .getByText("Day MAE", { exact: false })
      .waitFor();
    await page.fill("#evaluation-date", "2018-12-03");
    await page.waitForTimeout(400);
    assert.equal(await page.locator("#comparison-table tbody tr").count(), 5);
    await page.screenshot({
      path: `${out}/desktop-evidence.png`,
      fullPage: true,
    });
    checks.push(
      "Five-model table, historical replay and test-day filter render.",
    );
    for (const width of [390, 320, 768]) {
      await page.setViewportSize({ width, height: 844 });
      for (const section of ["forecast", "data", "history", "evidence"]) {
        await page.locator(`nav [data-page="${section}"]`).click();
        await page.waitForTimeout(200);
        const overflow = await page.evaluate(
          () => document.documentElement.scrollWidth > window.innerWidth + 1,
        );
        assert.equal(overflow, false, `${section} overflows at ${width}px`);
        await page.screenshot({
          path: `${out}/${width}-${section}.png`,
          fullPage: true,
        });
      }
      checks.push(`All four screens fit ${width}px without document overflow.`);
    }
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator('nav [data-page="data"]').click();
    await row.getByRole("button", { name: "Remove dataset" }).click();
    await page.getByRole("button", { name: "Cancel", exact: true }).click();
    assert.equal(await row.count(), 1);
    await row.getByRole("button", { name: "Remove dataset" }).click();
    await page
      .getByRole("button", { name: "Remove dataset", exact: true })
      .last()
      .click();
    await page.waitForFunction(
      () =>
        !document
          .querySelector("#dataset-list")
          .textContent.includes("QA meter.csv"),
    );
    checks.push(
      "Dataset removal requires confirmation; cancel preserves data.",
    );
    await page.reload();
    await page.getByText("Model ready · Local").waitFor();
    await page.locator('nav [data-page="history"]').click();
    await page.locator(".legacy-history summary").click();
    await page
      .locator("#history-list tbody tr")
      .filter({ hasText: "QA meter.csv" })
      .first()
      .waitFor();
    checks.push(
      "Forecast audit survives reload and deletion of its source dataset.",
    );
    assert.deepEqual(errors, []);
    fs.writeFileSync(
      `${out}/browser-results.json`,
      JSON.stringify({ status: "passed", checks, errors }, null, 2),
    );
    console.log(JSON.stringify({ status: "passed", checks, errors }, null, 2));
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

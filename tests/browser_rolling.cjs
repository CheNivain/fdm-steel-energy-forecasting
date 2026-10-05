const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const base = process.env.FDM_TEST_URL || "http://127.0.0.1:8003";
const out = "tmp/final-system-qa";

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" });
  const errors = [], checks = [];
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1024 }, acceptDownloads: true, timezoneId: "America/New_York" });
    const page = await context.newPage();
    page.on("pageerror", error => errors.push(error.message));
    for (let attempt = 0; attempt < 30; attempt++) {
      try { if ((await context.request.get(`${base}/api/health`)).ok()) break; }
      catch (error) { if (attempt === 29) throw error; }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    await page.goto(base);
    await page.getByText("Model ready · Local").waitFor();
    await page.locator("#ops-date").waitFor();
    await page.waitForFunction(() => document.querySelector("#ops-date").value.length > 0);
    assert.equal(await page.locator("[data-ops-day]").count(), 7);
    const names = await page.locator("[data-ops-day] span").allTextContents();
    assert.deepEqual([...names].sort(), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].sort());
    assert.equal(await page.evaluate(() => plusMinutes("2019-03-10T01:45:00", 15)), "2019-03-10T02:00:00");
    checks.push("Plant-clock intervals remain unchanged by the browser's daylight-saving timezone.");
    await page.screenshot({ path: `${out}/desktop-initial.png`, fullPage: true });
    const before = await (await context.request.get(`${base}/api/workspace`)).json();
    await page.locator("#ops-predict").click();
    await page.locator(".ops-result-value").waitFor();
    await page.waitForFunction(() => !document.querySelector("#ops-predict").disabled);
    const first = await (await context.request.get(`${base}/api/workspace/forecasts`)).json();
    assert.equal(first[0].interval_start, before.next_start);
    assert.equal(first[0].estimated_inputs, 0);
    assert.equal(first[0].actual_kwh, null);
    assert.ok(Number.isFinite(first[0].predicted_kwh));
    checks.push("Future interval is inferred from the deployment model and has no fabricated actual reading.");
    await page.locator("#ops-duration").selectOption("2");
    await page.locator("#ops-predict").click();
    await page.locator(".interval-breakdown").waitFor();
    await page.waitForFunction(() => !document.querySelector("#ops-predict").disabled);
    const second = await (await context.request.get(`${base}/api/workspace/forecasts`)).json();
    assert.equal(second.length - first.length, 2);
    assert.equal(second[1].input_features.lag_1, first[0].predicted_kwh);
    assert.ok(second[0].estimated_inputs >= 2);
    checks.push("Repeated forecasts advance the cursor and 30-minute requests create two consecutive predictions.");
    await page.screenshot({ path: `${out}/desktop-forecast.png`, fullPage: true });
    const pixels = await page.locator("#ops-chart").evaluate(canvas => {
      const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
      let painted = 0; for (let i=3; i<data.length; i+=4) if(data[i]) painted++;
      return painted;
    });
    assert.ok(pixels > 1000);
    checks.push(`Measured / estimated chart rendered (${pixels} painted pixels).`);
    await page.locator('nav [data-page="data"]').click();
    await page.locator("#actual-file").setInputFiles({ name: "invalid.csv", mimeType: "text/csv", buffer: Buffer.from("x,y\n1,2") });
    await page.locator("#actual-upload-button").click();
    await page.locator("#actual-upload-error:not(.hidden)").waitFor();
    assert.match(await page.locator("#actual-upload-error").textContent(), /timestamp and Usage_kWh/);
    const value = 55;
    await page.locator("#actual-file").setInputFiles({ name: "QA-only-actual.csv", mimeType: "text/csv", buffer: Buffer.from(`timestamp,Usage_kWh\n${first[0].forecast_timestamp},${value}\n`) });
    await page.locator("#actual-upload-button").click();
    await page.locator("#actual-upload-success:not(.hidden)").waitFor();
    await page.waitForFunction(() => !document.querySelector("#actual-upload-button").disabled);
    const after = await (await context.request.get(`${base}/api/workspace`)).json();
    assert.equal(after.latest_actual, first[0].forecast_timestamp);
    assert.equal(after.next_start, first[0].forecast_timestamp);
    assert.equal(after.estimated_inputs, 0);
    assert.equal(after.comparison.all.rows, 1);
    assert.equal(after.comparison.all.mae, Math.abs(first[0].predicted_kwh-value));
    assert.equal(after.comparison.all.high_load_rows, 1);
    assert.match(await page.locator("#ops-result").textContent(), /Updated history ready/);
    checks.push("Invalid CSV is rejected; actual upload reconciles the original forecast and resets provisional history.");
    await page.screenshot({ path: `${out}/desktop-data.png`, fullPage: true });
    await page.locator('nav [data-page="history"]').click();
    await page.locator("#rolling-filter").selectOption("matched");
    assert.equal(await page.locator("#rolling-ledger tbody tr").count(), 1);
    await page.locator("[data-rolling-record]").first().click();
    await page.locator("#rolling-json").waitFor();
    assert.equal(await page.locator("#rolling-detail tbody tr").count(), 15);
    const download = page.waitForEvent("download");
    await page.getByRole("link", { name: "Export CSV", exact: true }).click();
    assert.equal((await download).suggestedFilename(), "rolling_forecasts.csv");
    checks.push("Ledger filters, original prediction errors, input inspection and CSV downloads work.");
    await page.screenshot({ path: `${out}/desktop-ledger.png`, fullPage: true });
    await page.locator('nav [data-page="forecast"]').click();
    assert.equal(await page.locator('#dataset-select optgroup[label="Historical samples"] option').count(), 7);
    for (const id of ["demo-production", "demo-night", "demo-wednesday", "demo-thursday", "demo-friday", "demo-saturday", "demo-weekend"]) {
      await page.locator("#dataset-select").selectOption(id);
      await page.getByText("Ready for a forecast", { exact: true }).waitFor();
      await page.locator("#forecast-button").click();
      await page.getByText("PREDICTED ENERGY USAGE", { exact: true }).waitFor();
      await page.waitForFunction(() => !document.querySelector("#forecast-button").disabled);
    }
    checks.push("All seven historical day examples make real model predictions.");
    await page.locator('nav [data-page="evidence"]').click();
    await page.locator("#comparison-table tbody tr").first().waitFor();
    assert.equal(await page.locator("#comparison-table tbody tr").count(), 5);
    await page.screenshot({ path: `${out}/desktop-evidence.png`, fullPage: true });
    for (const width of [390, 320, 768, 1024, 1920]) {
      await page.setViewportSize({ width, height: 900 });
      for (const section of ["operations", "data", "history", "forecast", "evidence"]) {
        await page.locator(`nav [data-page="${section}"]`).click();
        await page.waitForTimeout(200);
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `${section} overflows at ${width}`);
        await page.screenshot({ path: `${out}/${width}-${section}.png`, fullPage: true });
      }
      checks.push(`All five pages fit ${width}px without document overflow.`);
    }
    await page.reload();
    await page.getByText("Model ready · Local").waitFor();
    const persisted = await (await context.request.get(`${base}/api/workspace`)).json();
    assert.equal(persisted.comparison.all.rows, 1);
    checks.push("Actual readings, saved predictions and comparisons survive a browser reload.");
    assert.deepEqual(errors, []);
    fs.writeFileSync(`${out}/browser-results.json`, JSON.stringify({ status: "passed", checks, errors }, null, 2));
    console.log(JSON.stringify({ status: "passed", checks, errors }, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

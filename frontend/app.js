/* Browser UI. Forecasts and all feature calculations are performed by Python. */
"use strict";
const $ = (selector) => document.querySelector(selector);
const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const number = (value, digits = 2) =>
  value == null
    ? "N/A"
    : Number(value).toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });
const stamp = (value) => String(value).replace("T", " ").slice(0, 16);
const dayStamp = (value) => {
  const [date, time] = stamp(value).split(" ");
  const [year, month, day] = date.split("-");
  return `${day} ${["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][Number(month) - 1]} ${year}${time ? ` · ${time}` : ""}`;
};
const icon = (name) => `<i data-lucide="${name}"></i>`;
const state = {
  datasets: [],
  dataset: null,
  model: null,
  history: [],
  result: null,
  appendId: null,
  charts: {},
  toastTimer: null,
  selectionRequest: 0,
  evaluationRequest: 0,
};
const icons = () => window.lucide?.createIcons();

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    signal: AbortSignal.timeout(30000),
  });
  if (response.status === 204) return null;
  const body = await response.json().catch(() => ({}));
  if (!response.ok)
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : "The request failed. Please check the server terminal.",
    );
  return body;
}
function jsonPost(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
function errorMessage(error) {
  return error.name === "TimeoutError"
    ? "The server took too long to respond. Check that it is running and retry."
    : error.message === "Failed to fetch"
      ? "Cannot reach the local server. Keep the Python server running and retry."
      : error.message;
}
function showError(id, error) {
  const element = $(id);
  element.textContent = error ? errorMessage(error) : "";
  element.classList.toggle("hidden", !error);
}
function toast(message, error = false) {
  clearTimeout(state.toastTimer);
  $("#toast").textContent = message;
  $("#toast").classList.remove("hidden");
  $("#toast").classList.toggle("error-toast", error);
  state.toastTimer = setTimeout(
    () => $("#toast").classList.add("hidden"),
    5000,
  );
}
function busy(button, pending, text) {
  if (pending) {
    button.dataset.original = button.innerHTML;
    button.innerHTML = icon("loader-circle") + escapeHTML(text);
    button.querySelector("i").classList.add("busy");
  } else if (button.dataset.original)
    button.innerHTML = button.dataset.original;
  button.disabled = pending;
  icons();
}
function downloadJSON(value, filename) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function makeChart(id, labels, datasets) {
  state.charts[id]?.destroy();
  state.charts[id] = new Chart($(id), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          display: id === "#evaluation-chart",
          position: "bottom",
          labels: {
            usePointStyle: true,
            pointStyle: "line",
            boxWidth: 15,
            padding: 20,
            font: { size: 10 },
            color: "#647369",
          },
        },
        tooltip: {
          backgroundColor: "#273a30",
          titleFont: { size: 11 },
          bodyFont: { size: 11 },
          padding: 12,
          callbacks: {
            label: (c) => `${c.dataset.label}: ${number(c.parsed.y)} kWh`,
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: {
            color: "#8b988f",
            font: { size: 10 },
            maxTicksLimit: 6,
            maxRotation: 0,
            callback: function (v) {
              return String(this.getLabelForValue(v)).slice(11, 16);
            },
          },
          border: { display: false },
        },
        y: {
          beginAtZero: true,
          border: { display: false },
          grid: { color: "#e5ebe7" },
          ticks: { color: "#8b988f", font: { size: 10 }, maxTicksLimit: 5 },
        },
      },
      elements: {
        point: { radius: 0, hitRadius: 10 },
        line: { borderWidth: 1.8, tension: 0 },
      },
    },
  });
}
function forecastChart(readings, result = null) {
  const labels = readings.map((r) => stamp(r.timestamp)),
    actual = readings.map((r) => r.usage_kwh);
  const series = [
    {
      label: "Recorded",
      data: actual,
      borderColor: "#7d9185",
      backgroundColor: "#738c7c0c",
      fill: true,
    },
  ];
  if (result) {
    labels.push(stamp(result.forecast_timestamp));
    actual.push(null);
    const predicted = Array(actual.length).fill(null);
    predicted[predicted.length - 2] = result.previous_kwh;
    predicted[predicted.length - 1] = result.predicted_kwh;
    series.push({
      label: "Next-interval forecast",
      data: predicted,
      borderColor: "#087c68",
      borderWidth: 2.5,
      borderDash: [4, 3],
      pointRadius: (context) =>
        context.dataIndex === predicted.length - 1 ? 5 : 0,
      pointBackgroundColor: "#087c68",
    });
  }
  makeChart("#forecast-chart", labels, series);
}

async function refreshDatasets(preferred) {
  state.datasets = await api("/api/datasets");
  const chosen = preferred || $("#dataset-select").value || "demo-production";
  const groups = [
    ["Historical samples", "historical_demo"],
    ["Imported meter history", "uploaded"],
  ];
  $("#dataset-select").innerHTML = groups
    .map(
      ([label, source]) =>
        `<optgroup label="${label}">${state.datasets
          .filter((d) => d.source === source)
          .map(
            (d) =>
              `<option value="${escapeHTML(d.id)}">${escapeHTML(d.name)}</option>`,
          )
          .join("")}</optgroup>`,
    )
    .join("");
  $("#dataset-select").value = state.datasets.some((d) => d.id === chosen)
    ? chosen
    : "demo-production";
  renderDatasets();
  await selectDataset();
}
async function selectDataset() {
  const request = ++state.selectionRequest;
  $("#forecast-button").disabled = true;
  try {
    const dataset = await api(
      `/api/datasets/${encodeURIComponent($("#dataset-select").value)}`,
    );
    if (request !== state.selectionRequest) return;
    state.dataset = dataset;
    state.result = null;
    $("#input-summary").innerHTML =
      `<dt>Readings available</dt><dd>${number(dataset.rows, 0)}</dd><dt>Last reading</dt><dd>${number(dataset.last_kwh)} kWh</dd><dt>Recorded at</dt><dd>${escapeHTML(stamp(dataset.end))}</dd><dt>Forecast for</dt><dd><strong>${escapeHTML(stamp(dataset.next_timestamp))}</strong></dd>`;
    const demo = dataset.source === "historical_demo";
    $("#source-banner").innerHTML =
      `${icon(demo ? "flask-conical" : "database")}<div><strong>${demo ? "Historical demo" : "Imported meter history"}</strong><span>${demo ? "2018 meter readings. No live factory connection." : escapeHTML(dataset.name) + " · Last recorded " + escapeHTML(stamp(dataset.end)) + ". No live connection."}</span></div><a href="#data">Meter data${icon("arrow-up-right")}</a>`;
    $("#forecast-result").innerHTML =
      `<div class="empty-result">${icon("gauge")}<div><h3>Ready for a forecast</h3><p>${escapeHTML(dayStamp(dataset.next_timestamp))} · 15-minute interval</p></div></div>`;
    $("#result-details").innerHTML = "";
    showError("#forecast-error", null);
    $("#chart-subtitle").textContent =
      `${stamp(dataset.preview[0].timestamp)} to ${stamp(dataset.end)}`;
    forecastChart(dataset.preview);
    icons();
  } catch (error) {
    showError("#forecast-error", error);
    state.dataset = null;
  } finally {
    if (request === state.selectionRequest)
      $("#forecast-button").disabled = !state.dataset;
  }
}
function renderForecast(result) {
  const high = result.high_load;
  $("#forecast-result").innerHTML =
    `<div class="result-band ${high ? "is-high" : ""}"><div><span class="eyebrow">PREDICTED ENERGY USAGE</span><div class="forecast-number ${number(result.predicted_kwh).length > 8 ? "long-number" : ""}">${number(result.predicted_kwh)}<span>kWh</span></div><div class="result-time">${escapeHTML(dayStamp(result.forecast_timestamp))}</div></div><div class="result-secondary"><span class="tag ${high ? "warning" : "good"}">${icon(high ? "triangle-alert" : "check")}${high ? "At / above alert threshold" : "Below alert threshold"}</span><p>Compared with previous interval<strong>${result.change_kwh >= 0 ? "+" : ""}${number(result.change_kwh)} kWh</strong></p></div></div>`;
  $("#result-details").innerHTML =
    `<div class="result-audit"><span>${result.actual_kwh !== null ? `Historical actual: <strong>${number(result.actual_kwh)} kWh</strong> · Error: <strong>${number(result.absolute_error_kwh)} kWh</strong>` : `Previous actual: <strong>${number(result.previous_kwh)} kWh</strong> · Next actual not yet supplied`}<br>Saved · ${number(result.inference_ms, 0)} ms inference · Alert ≥ ${number(result.alert_threshold_kwh)} kWh</span><div class="audit-links"><button id="download-forecast" class="icon-button" title="Download forecast and inputs" aria-label="Download forecast and inputs">${icon("download")}</button><a href="#history" class="icon-button" title="Open forecast history" aria-label="Open forecast history">${icon("history")}</a></div></div><div class="result-warnings">${result.warnings.map((w) => `<p>${escapeHTML(w)}</p>`).join("")}</div>`;
  $("#download-forecast").addEventListener("click", () =>
    downloadJSON(result, `forecast_${result.id}.json`),
  );
  forecastChart(result.recent_readings, result);
  icons();
  if (window.innerWidth <= 700)
    $(".forecast-output").scrollIntoView({ block: "start" });
}

function renderDatasets() {
  $("#dataset-count").textContent = `${state.datasets.length} datasets`;
  $("#dataset-list").innerHTML = state.datasets
    .map(
      (d) =>
        `<div class="dataset-row"><div class="dataset-symbol">${icon(d.source === "uploaded" ? "file-spreadsheet" : "flask-conical")}</div><div class="dataset-copy"><h3>${escapeHTML(d.name)}</h3><p>${d.source === "uploaded" ? "Imported" : "Historical demo"} · ${number(d.rows, 0)} readings · ${escapeHTML(stamp(d.start))} to ${escapeHTML(stamp(d.end))}</p></div><div class="dataset-actions"><button class="button secondary" data-use="${escapeHTML(d.id)}">Forecast${icon("arrow-up-right")}</button>${d.source === "uploaded" ? `<button class="icon-button" data-append="${escapeHTML(d.id)}" title="Add next actual reading" aria-label="Add next actual reading">${icon("plus")}</button><button class="icon-button" data-delete="${escapeHTML(d.id)}" title="Remove dataset" aria-label="Remove dataset">${icon("trash-2")}</button>` : ""}</div></div>`,
    )
    .join("");
  icons();
}
async function appendPanel(identifier) {
  const dataset = await api(`/api/datasets/${encodeURIComponent(identifier)}`);
  state.appendId = identifier;
  $("#append-name").textContent = dataset.name;
  $("#append-timestamp").value = stamp(dataset.next_timestamp);
  $("#append-value").value = "";
  showError("#append-error", null);
  $("#append-section").classList.remove("hidden");
  $("#append-section").scrollIntoView({ block: "center" });
  $("#append-value").focus();
}
async function confirmRemove(identifier) {
  const dialog = $("#confirm-dialog");
  dialog.returnValue = "cancel";
  dialog.showModal();
  const confirmed = await new Promise((resolve) =>
    dialog.addEventListener(
      "close",
      () => resolve(dialog.returnValue === "confirm"),
      { once: true },
    ),
  );
  if (!confirmed) return;
  await api(`/api/datasets/${encodeURIComponent(identifier)}`, {
    method: "DELETE",
  });
  if (state.appendId === identifier) {
    state.appendId = null;
    $("#append-section").classList.add("hidden");
  }
  await refreshDatasets();
  toast("Dataset removed. Saved forecasts are retained.");
}
async function refreshHistory() {
  state.history = await api("/api/forecasts");
  renderHistory();
}
function renderHistory() {
  const query = $("#history-search").value.toLowerCase(),
    filter = $("#history-filter").value;
  const rows = state.history.filter(
    (r) =>
      `${r.dataset_name} ${r.forecast_timestamp}`
        .toLowerCase()
        .includes(query) &&
      (filter === "all" ||
        (filter === "high" ? r.high_load : r.source === filter)),
  );
  if (!rows.length) {
    $("#history-list").innerHTML =
      `<div class="empty-state">${icon("history")}<h2>${state.history.length ? "No matching forecasts" : "No forecasts saved yet"}</h2><p>${state.history.length ? "No records match these filters." : "Generated forecasts will appear here."}</p><a href="#forecast">Forecast desk ${icon("arrow-right")}</a></div>`;
    icons();
    return;
  }
  $("#history-list").innerHTML =
    `<div class="table-wrap"><table><thead><tr><th>Forecast interval</th><th>Source</th><th class="numeric">Forecast kWh</th><th>Alert status</th><th class="numeric">Actual kWh</th><th>Record</th></tr></thead><tbody>${rows.map((r) => `<tr><td>${escapeHTML(stamp(r.forecast_timestamp))}<small>Recorded ${escapeHTML(stamp(r.created_utc))} UTC</small></td><td>${escapeHTML(r.dataset_name)}<small>${r.source === "historical_demo" ? "Historical demo" : "Uploaded readings"}</small></td><td class="numeric"><strong>${number(r.predicted_kwh)}</strong></td><td><span class="tag ${r.high_load ? "warning" : "good"}">${r.high_load ? "At / above" : "Below"} ${number(r.alert_threshold_kwh)}</span></td><td class="numeric">${number(r.actual_kwh)}</td><td><button class="icon-button" data-record="${escapeHTML(r.id)}" title="Inspect forecast record" aria-label="Inspect forecast record">${icon("arrow-up-right")}</button></td></tr>`).join("")}</tbody></table></div>`;
  icons();
}
async function showRecord(identifier) {
  const row = await api(`/api/forecasts/${encodeURIComponent(identifier)}`);
  $("#history-detail").innerHTML =
    `<section class="history-detail"><div class="section-heading"><div><h2>Forecast record</h2><p>${escapeHTML(stamp(row.forecast_timestamp))} · ${escapeHTML(row.dataset_name)}</p></div><button id="record-download" class="button secondary">${icon("download")}JSON record</button></div><div class="detail-grid"><div><h3>${number(row.predicted_kwh)} kWh predicted</h3><p class="field-note">Previous actual: ${number(row.previous_kwh)} kWh · ${number(row.history_rows_used, 0)} past readings</p><p class="field-note">Model SHA-256<br><code>${escapeHTML(row.model_sha256)}</code></p><p class="field-note">History SHA-256<br><code>${escapeHTML(row.history_sha256)}</code></p>${row.warnings.map((w) => `<p class="field-note">${escapeHTML(w)}</p>`).join("")}</div><div class="table-wrap"><table class="feature-table"><thead><tr><th>Prepared input</th><th class="numeric">Value</th></tr></thead><tbody>${Object.entries(
      row.input_features,
    )
      .map(
        ([key, value]) =>
          `<tr><td>${escapeHTML(key)}</td><td class="numeric">${number(value, 5)}</td></tr>`,
      )
      .join("")}</tbody></table></div></div></section>`;
  $("#record-download").addEventListener("click", () =>
    downloadJSON(row, `forecast_${identifier}.json`),
  );
  $("#history-detail").scrollIntoView({ block: "start" });
  icons();
}

function renderEvidence(model) {
  const m = model.metrics;
  $("#summary-mae").textContent = number(m.mae);
  $("#summary-high").textContent = number(m.high_load_mae);
  $("#summary-improvement").textContent =
    `${number(model.rmse_improvement_pct, 1)}%`;
  $("#evidence-metrics").innerHTML = [
    ["Average absolute error", m.mae, "MAE · kWh"],
    ["Large-error measure", m.rmse, "RMSE · kWh"],
    ["Variance explained", m.r2, "R² · not percentage accuracy"],
    ["High-load absolute error", m.high_load_mae, "Actual usage ≥ 50 kWh"],
  ]
    .map(
      ([label, value, unit], i) =>
        `<div class="metric"><span>${label}</span><strong>${number(value, i === 2 ? 3 : 2)}</strong><small>${unit}</small></div>`,
    )
    .join("");
  $("#evidence-meta").innerHTML =
    `<span>${number(m.rows, 0)} test intervals</span><span>${number(m.high_load_rows, 0)} high-load intervals</span><span>01 Nov 2018 to 01 Jan 2019</span><span>Legacy prepared-data evaluation</span>`;
  $("#comparison-table").innerHTML =
    `<div class="table-wrap"><table><thead><tr><th>Model</th><th class="numeric">MAE</th><th class="numeric">RMSE</th><th class="numeric">R²</th><th class="numeric">HighLoad_MAE</th><th class="numeric">Cutoff (kWh)</th><th class="numeric">Test rows</th></tr></thead><tbody>${model.comparisons.map((r, i) => `<tr class="${i === 0 ? "selected" : ""}"><td><strong>${escapeHTML(r.model)}</strong><small>${i === 0 ? "Selected · " : ""}${escapeHTML(r.group)}</small></td><td class="numeric">${number(r.mae, 3)}</td><td class="numeric">${number(r.rmse, 3)}</td><td class="numeric">${number(r.r2, 3)}</td><td class="numeric">${number(r.high, 3)}</td><td class="numeric">${number(r.cutoff, r.cutoff === 50 ? 0 : 3)}</td><td class="numeric">${number(r.rows, 0)}</td></tr>`).join("")}</tbody></table></div>`;
  const s = model.settings.selected_specification;
  $("#model-settings").innerHTML = [
    ["Trees", number(s.n_estimators, 0)],
    ["Minimum leaf size", s.min_samples_leaf],
    ["Features per split", `${s.max_features * 100}%`],
    ["Maximum depth", s.max_depth ?? "Unrestricted"],
    ["Input / engineered features", "15 / 70"],
    ["Training rows", number(model.training_rows, 0)],
    ["Cross-validation", "5 time-ordered folds"],
    ["CV RMSE", `${number(model.settings.cv_RMSE_kWh, 3)} kWh`],
  ]
    .map(([k, v]) => `<dt>${k}</dt><dd>${escapeHTML(v)}</dd>`)
    .join("");
  const max = Math.max(...model.features.map((f) => f.impurity_importance));
  $("#feature-bars").innerHTML = model.features
    .map(
      (f) =>
        `<div class="feature-bar"><span>${escapeHTML(f.feature)}</span><div class="bar-track"><div class="bar-fill" style="width:${(f.impurity_importance / max) * 100}%"></div></div><b>${number(f.impurity_importance * 100, 1)}%</b></div>`,
    )
    .join("");
  const titles = [
    "One interval ahead, not one day",
    "No future load label required",
    "Midnight timestamp assumption",
    "Known interpolation in legacy test data",
    "Feature selection before validation",
    "Earlier inspection of test data",
  ];
  $("#limitations").innerHTML =
    model.limitations
      .map(
        (text, i) =>
          `<details><summary>${escapeHTML(titles[i] || "Evaluation limitation")}</summary><p>${escapeHTML(text)}</p></details>`,
      )
      .join("") +
    `<details><summary>Deployment readiness</summary><p>This is a local academic prototype. There is no live meter connection, validated uncertainty interval, retraining monitor, authentication or remote deployment. Performance on a new factory or a future production process has not been measured. Forecast alerts do not control machinery.</p></details>`;
  $("#model-identity").innerHTML =
    `<p>${escapeHTML(model.version)}<br>SHA-256: <code>${escapeHTML(model.model_sha256)}</code></p><p>Training period: ${escapeHTML(model.training_start)} to ${escapeHTML(model.training_end)}. The saved pipeline learns the change from the preceding interval and adds that actual reading back internally. Predictions are floored at zero.</p>`;
  icons();
}
async function evaluation() {
  const request = ++state.evaluationRequest;
  showError("#evaluation-error", null);
  try {
    const result = await api(
      `/api/evaluation?date=${encodeURIComponent($("#evaluation-date").value)}`,
    );
    if (request !== state.evaluationRequest) return;
    const p = result.points;
    makeChart(
      "#evaluation-chart",
      p.map((r) => stamp(r.timestamp)),
      [
        {
          label: "Actual",
          data: p.map((r) => r.actual_kWh),
          borderColor: "#343f38",
          borderWidth: 1.6,
        },
        {
          label: "Random Forest",
          data: p.map((r) => r.predicted_kWh),
          borderColor: "#087c68",
          borderWidth: 2,
        },
        {
          label: "Naive: previous actual",
          data: p.map((r) => r.persistence_kWh),
          borderColor: "#bda479",
          borderDash: [3, 4],
          borderWidth: 1,
        },
      ],
    );
    const m = result.metrics;
    $("#day-scores").innerHTML =
      `<span>Day MAE <strong>${number(m.mae)} kWh</strong></span><span>Day RMSE <strong>${number(m.rmse)} kWh</strong></span><span>Day high-load MAE <strong>${number(m.high_load_mae)}${m.high_load_mae === null ? "" : " kWh"}</strong></span><span>${m.high_load_rows} high-load / ${m.rows} intervals</span>`;
  } catch (error) {
    if (request === state.evaluationRequest) {
      showError("#evaluation-error", error);
      state.charts["#evaluation-chart"]?.destroy();
      $("#day-scores").textContent = "";
    }
  }
}
async function route() {
  const page = ["operations", "forecast", "data", "history", "evidence"].includes(
    location.hash.slice(1),
  )
    ? location.hash.slice(1)
    : "operations";
  if (state.page !== page) window.scrollTo(0, 0);
  state.page = page;
  document
    .querySelectorAll(".page")
    .forEach((p) => p.classList.toggle("hidden", p.id !== `page-${page}`));
  document.querySelectorAll("[data-page]").forEach((a) => {
    a.setAttribute("aria-label", a.textContent.trim());
    a.classList.toggle("active", a.dataset.page === page);
    if (a.dataset.page === page) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  document.title = `${{ operations: "Energy forecast", forecast: "Historical replay", data: "Meter data", history: "Forecast ledger", evidence: "Model evidence" }[page]} | Steel Energy`;
  try {
    if (["operations", "data", "history"].includes(page)) await window.refreshOperations?.();
    if (page === "history") await refreshHistory();
    if (page === "evidence" && state.model) await evaluation();
    if (page === "forecast") state.charts["#forecast-chart"]?.resize();
  } catch (error) {
    toast(errorMessage(error), true);
  }
}

$("#forecast-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("#forecast-error", null);
  const button = $("#forecast-button");
  busy(button, true, "Predicting...");
  $("#dataset-select").disabled = true;
  $("#threshold").disabled = true;
  try {
    const result = await jsonPost("/api/forecasts", {
      dataset_id: $("#dataset-select").value,
      alert_threshold_kwh: Number($("#threshold").value),
    });
    state.result = result;
    renderForecast(result);
    toast("Forecast saved to history.");
  } catch (error) {
    showError("#forecast-error", error);
  } finally {
    busy(button, false);
    $("#dataset-select").disabled = false;
    $("#threshold").disabled = false;
  }
});
$("#dataset-select").addEventListener("change", selectDataset);
$("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("#upload-error", null);
  $("#upload-success").classList.add("hidden");
  const file = $("#csv-file").files[0];
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) {
    showError("#upload-error", new Error("The CSV must be smaller than 8 MB."));
    return;
  }
  const button = $("#upload-button");
  busy(button, true, "Validating...");
  try {
    const form = new FormData();
    form.append("file", file);
    form.append("convention", $("#convention").value);
    const result = await api("/api/datasets", { method: "POST", body: form });
    await refreshDatasets(result.id);
    $("#upload-success").textContent =
      `${number(result.rows, 0)} valid readings imported. Next forecast: ${stamp(result.next_timestamp)}. Dataset selected on the Forecast desk.`;
    $("#upload-success").classList.remove("hidden");
    $("#csv-file").value = "";
    toast("Meter history imported.");
  } catch (error) {
    showError("#upload-error", error);
  } finally {
    busy(button, false);
  }
});
$("#convention").addEventListener("change", () => {
  $("#csv-schema").innerHTML =
    $("#convention").value === "iso"
      ? "timestamp,Usage_kWh<br>2018-11-12 10:00:00,42.7"
      : "date,Usage_kWh<br>12/11/2018 10:00,42.7";
});
$("#dataset-list").addEventListener("click", async (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  try {
    if (button.dataset.use) {
      $("#dataset-select").value = button.dataset.use;
      await selectDataset();
      location.hash = "forecast";
    }
    if (button.dataset.append) await appendPanel(button.dataset.append);
    if (button.dataset.delete) await confirmRemove(button.dataset.delete);
  } catch (error) {
    toast(errorMessage(error), true);
  }
});
$("#append-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("#append-error", null);
  const button = event.currentTarget.querySelector("button");
  busy(button, true, "Saving...");
  try {
    await jsonPost(
      `/api/datasets/${encodeURIComponent(state.appendId)}/readings`,
      {
        timestamp: $("#append-timestamp").value,
        usage_kwh: Number($("#append-value").value),
      },
    );
    await refreshDatasets(state.appendId);
    await appendPanel(state.appendId);
    toast("Actual reading added. The next forecast interval has advanced.");
  } catch (error) {
    showError("#append-error", error);
  } finally {
    busy(button, false);
  }
});
$("#history-search").addEventListener("input", renderHistory);
$("#history-filter").addEventListener("change", renderHistory);
$("#refresh-history").addEventListener("click", async () => {
  try {
    await refreshHistory();
    toast("Forecast history refreshed.");
  } catch (error) {
    toast(errorMessage(error), true);
  }
});
$("#history-list").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-record]");
  if (button)
    try {
      await showRecord(button.dataset.record);
    } catch (error) {
      toast(errorMessage(error), true);
    }
});
$("#evaluation-date").addEventListener("change", evaluation);
window.addEventListener("hashchange", route);

async function start() {
  icons();
  $("#forecast-button").disabled = true;
  await route();
  try {
    state.model = await api("/api/model");
    renderEvidence(state.model);
    await refreshDatasets();
    $("#connection").innerHTML = '<span class="dot"></span>Model ready · Local';
    await route();
  } catch (error) {
    showError(
      "#fatal",
      new Error(
        `${errorMessage(error)} Refresh this page after the server is ready.`,
      ),
    );
    $("#connection").innerHTML =
      '<span class="dot offline"></span>Server unavailable';
  }
}
start();

/* Future forecasting and actual-reading reconciliation. Timestamps are plant-local. */
"use strict";
const ops = { workspace: null, ledger: [], pending: false, timer: null, request: 0, result: null };
// UTC arithmetic preserves naive plant-clock labels across the browser's DST changes.
const localDate = (value) => new Date(`${String(value).replace(" ", "T")}Z`);
const localISO = (date) => {
  const pad = (v) => String(v).padStart(2, "0");
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}T${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}:00`;
};
const plusMinutes = (value, minutes) => localISO(new Date(localDate(value).getTime() + minutes * 60000));
const weekday = (value) => localDate(value.length === 10 ? `${value}T00:00:00` : value).toLocaleDateString("en-GB", { weekday: "long", timeZone: "UTC" });
const intervalLabel = (start, end) => `${weekday(start)}, ${dayStamp(start)} to ${String(end).slice(0, 10) === String(start).slice(0, 10) ? String(end).slice(11, 16) : dayStamp(end)}`;
const stat = (label, value, caption) => `<div><span>${escapeHTML(label)}</span><strong>${escapeHTML(value)}</strong><small>${escapeHTML(caption)}</small></div>`;

async function opsPost(path, body) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || "Request failed. Refresh the workspace and retry.");
  return result;
}

function setNextInterval(value) {
  $("#ops-date").value = value.slice(0, 10);
  $("#ops-time").value = value.slice(11, 16);
  intervalSummary();
}

function intervalSummary() {
  if (!ops.workspace) return;
  const start = `${$("#ops-date").value}T${$("#ops-time").value}:00`;
  const gap = Math.round((localDate(start) - localDate(ops.workspace.cursor)) / 900000);
  const valid = Number.isFinite(gap) && gap >= 0 && gap + Number($("#ops-duration").value) <= 672;
  $("#ops-gap").textContent = !valid ? "Choose an interval at or after the next available time, within seven days." : gap ? `${number(gap, 0)} intervening intervals will also be estimated and saved before this forecast.` : `${weekday(start)} · ${$("#ops-time").value} to ${plusMinutes(start, Number($("#ops-duration").value) * 15).slice(11, 16)}`;
  $("#ops-predict").disabled = !valid || !ops.workspace.model || ops.pending;
  if ($("#ops-next-label") && valid) $("#ops-next-label").textContent = intervalLabel(start, plusMinutes(start, Number($("#ops-duration").value) * 15));
  document.querySelectorAll("[data-ops-day]").forEach((button) => {
    const selected = button.dataset.opsDay === $("#ops-date").value;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function renderWorkspace(workspace, reset = false) {
  ops.workspace = workspace;
  const m = workspace.model;
  $("#ops-period").textContent = `Forecasts beyond measured history · ${weekday(workspace.next_start)}, ${dayStamp(workspace.next_start)}`;
  $("#ops-stats").innerHTML =
    stat("Last measured energy", `${number(workspace.latest_actual_kwh)} kWh`, dayStamp(workspace.latest_actual)) +
    stat("Next interval", workspace.next_start.slice(11, 16), `${weekday(workspace.next_start)} · ${workspace.next_start.slice(0, 10)}`) +
    stat("Estimates in input window", `${workspace.estimated_inputs} / 672`, workspace.estimated_inputs ? "Recursive forecast inputs" : "All inputs are measured") +
    stat("Saved forecasts", number(workspace.forecast_count, 0), `${number(workspace.actual_rows, 0)} actual readings stored`);
  $("#ops-training-summary").textContent = m ? `Fitted through ${dayStamp(m.training_end)}. ${number(m.training_rows, 0)} usable examples; 1,000 trees.` : "Deployment model needs an initial fit. Open Meter data to fit it.";
  $("#ops-date").min = workspace.next_start.slice(0, 10);
  $("#ops-date").max = plusMinutes(workspace.next_start, 7 * 1440 - 15).slice(0, 10);
  const base = localDate(workspace.next_start);
  $("#ops-weekdays").innerHTML = Array.from({ length: 7 }, (_, index) => {
    const date = new Date(base);
    date.setUTCDate(base.getUTCDate() + index);
    const iso = localISO(date).slice(0, 10);
    return `<button type="button" data-ops-day="${iso}" title="${weekday(iso)} ${iso}" aria-label="${weekday(iso)} ${iso}"><span>${date.toLocaleDateString("en-GB", { weekday: "short", timeZone: "UTC" })}</span><strong>${date.getUTCDate()}</strong></button>`;
  }).join("");
  if (reset || !$("#ops-date").value) setNextInterval(workspace.next_start);
  intervalSummary();
  $("#ops-chart-caption").textContent = `${dayStamp(workspace.recent[0].timestamp)} to ${dayStamp(workspace.cursor)}`;
  makeChart("#ops-chart", workspace.recent.map((r) => stamp(r.timestamp)), [
    { label: "Measured", data: workspace.recent.map((r) => r.kind === "actual" ? r.usage_kwh : null), borderColor: "#357e71", backgroundColor: "#357e7110", fill: true },
    { label: "Estimated", data: workspace.recent.map((r, i, all) => r.kind === "estimated" || all[i + 1]?.kind === "estimated" ? r.usage_kwh : null), borderColor: "#b77719", borderDash: [5, 3] },
  ]);
  $("#ops-quality").classList.toggle("warning", workspace.estimated_inputs > 0);
  $("#ops-quality").innerHTML = `${icon(workspace.estimated_inputs ? "triangle-alert" : "database")}<div><strong>${workspace.estimated_inputs ? "Estimates are being reused" : "Measured history available"}</strong><span>${workspace.estimated_inputs ? "Errors can grow with each estimated interval. New actual readings refresh the input history." : "Future values are generated by the model. Actual readings arrive separately."}</span></div>`;
  $("#actuals-through").textContent = `${number(workspace.actual_rows, 0)} readings · latest interval ends ${dayStamp(workspace.latest_actual)} · ${workspace.unverified_rows} unverified`;
  $("#actual-timestamp").value = plusMinutes(workspace.latest_actual, 15).slice(0, 16);
  const running = workspace.training.status === "running";
  $("#refit-status").textContent = running ? "Refitting from actual readings. This may take several minutes; keep the Python server running." : workspace.training.status === "failed" ? workspace.training.message : m ? `Current model: ${number(m.training_rows, 0)} examples, actuals through ${dayStamp(m.actuals_through)}.` : "No deployment fit yet.";
  $("#refit-button").disabled = running;
  $("#actual-upload-button").disabled = running;
  $("#actual-save").disabled = running;
  $("#actual-imports").innerHTML = workspace.imports.length ? workspace.imports.map((r) => `<div class="import-entry"><strong>${escapeHTML(r.filename)}</strong><span>${number(r.added, 0)} added · ${number(r.corrected, 0)} corrected · ${escapeHTML(dayStamp(r.created_utc))} UTC</span></div>`).join("") : '<p class="field-note">No actual readings uploaded yet.</p>';
  clearTimeout(ops.timer);
  if (running) ops.timer = setTimeout(() => window.refreshOperations().catch((e) => toast(errorMessage(e), true)), 4000);
  const c = workspace.comparison, scores = c.all;
  $("#rolling-scores").innerHTML = stat("Matched intervals", number(scores?.rows ?? 0, 0), "First forecast per interval") +
    stat("Average error", scores ? `${number(scores.mae)} kWh` : "Awaiting actuals", "MAE across matched intervals") +
    stat("High-load error", scores?.high_load_mae == null ? "No high-load actuals" : `${number(scores.high_load_mae)} kWh`, "Actual usage at least 50 kWh") +
    stat("Measured / recursive MAE", `${number(c.measured_history?.mae)} / ${number(c.recursive?.mae)}`, "kWh; N/A means no matched readings");
  icons();
}

window.refreshOperations = async function (reset = false) {
  const request = ++ops.request;
  const [workspace, ledger] = await Promise.all([api("/api/workspace"), api("/api/workspace/forecasts")]);
  if (request !== ops.request) return;
  const changed = ops.workspace && ops.workspace.revision !== workspace.revision;
  if (changed && !ops.pending && ops.result) {
    ops.result = null;
    $("#ops-result").innerHTML = '<div class="ops-placeholder"><i data-lucide="chart-no-axes-combined"></i><h2>Updated history ready</h2><p id="ops-next-label"></p></div>';
  }
  ops.ledger = ledger;
  renderWorkspace(workspace, reset || changed);
  const model = workspace.model;
  $("#deployment-evidence").innerHTML = `${icon("info")}<div><strong>Development evaluation and deployment refit</strong><span>${model ? `The operational model was refitted on ${number(model.training_rows, 0)} usable examples through ${escapeHTML(dayStamp(model.training_end))}.` : "The deployment model has not been fitted."} The scores below belong to the earlier development model. They do not measure this refit's future accuracy or chains of recursive predictions.</span></div>`;
  renderRollingLedger();
  $("#ops-recent-table").innerHTML = rollingTable(ledger.filter((r) => r.requested).slice(0, 5), false);
  icons();
};

function rollingTable(rows, details = true) {
  if (!rows.length) return '<div class="plain-empty">No forecasts here yet.</div>';
  return `<div class="table-wrap"><table><thead><tr><th>Interval (local)</th><th class="numeric">Forecast kWh</th><th class="numeric">Actual kWh</th><th class="numeric">Error kWh</th><th>Basis</th>${details ? "<th>Record</th>" : ""}</tr></thead><tbody>${rows.map((r) => `<tr><td><strong>${escapeHTML(dayStamp(r.interval_start))}</strong><small>to ${escapeHTML(dayStamp(r.forecast_timestamp))}${r.requested ? "" : " · intervening estimate"}</small></td><td class="numeric"><strong>${number(r.predicted_kwh)}</strong>${r.high_load ? '<small class="alert-text">Above threshold</small>' : ""}</td><td class="numeric">${r.actual_kwh == null ? '<span class="muted">Pending</span>' : number(r.actual_kwh)}</td><td class="numeric">${number(r.absolute_error_kwh)}</td><td><span class="tag ${r.estimated_inputs ? "warning" : "neutral"}">${r.estimated_inputs ? "Recursive" : "Measured history"}</span><small>${number(r.horizon_steps, 0)} interval${r.horizon_steps === 1 ? "" : "s"} after last actual</small></td>${details ? `<td><button class="icon-button" data-rolling-record="${escapeHTML(r.id)}" title="Inspect rolling forecast" aria-label="Inspect rolling forecast">${icon("arrow-up-right")}</button></td>` : ""}</tr>`).join("")}</tbody></table></div>`;
}

function renderRollingLedger() {
  const query = $("#rolling-search").value.trim(), filter = $("#rolling-filter").value;
  const rows = ops.ledger.filter((r) => `${r.interval_start} ${r.forecast_timestamp}`.includes(query) &&
    (filter === "all" || filter === "matched" && r.actual_kwh != null || filter === "pending" && r.actual_kwh == null || filter === "requested" && r.requested || filter === "high" && r.high_load));
  $("#rolling-ledger").innerHTML = rollingTable(rows);
  $("#rolling-count").textContent = `Showing ${rows.length} of the latest ${ops.ledger.length} forecasts. CSV export includes the full ledger.`;
  icons();
}

$("#ops-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (ops.pending) return;
  ops.pending = true;
  showError("#ops-error", null);
  busy($("#ops-predict"), true, "Calculating intervals...");
  $("#ops-form").querySelectorAll("input, select, .weekday-strip button").forEach((el) => el.disabled = true);
  try {
    const result = await opsPost("/api/workspace/forecasts", {
      start: `${$("#ops-date").value}T${$("#ops-time").value}:00`,
      intervals: Number($("#ops-duration").value), alert_threshold_kwh: Number($("#ops-threshold").value),
      revision: ops.workspace.revision, request_id: crypto.randomUUID(),
    });
    ops.result = result;
    const rows = result.requested, total = rows.reduce((sum, r) => sum + r.predicted_kwh, 0);
    $("#ops-result").innerHTML = `<div class="ops-result-band"><div><span class="eyebrow">FORECAST ENERGY</span><div class="ops-result-value">${number(total)} <span>kWh</span></div><p>${escapeHTML(intervalLabel(rows[0].interval_start, rows.at(-1).forecast_timestamp))}</p></div><span class="tag ${rows.some((r) => r.high_load) ? "warning" : "good"}">${rows.some((r) => r.high_load) ? "Interval above threshold" : "Below interval threshold"}</span></div>${rows.length === 2 ? `<div class="interval-breakdown">${rows.map((r) => `<span>${r.interval_start.slice(11, 16)}–${r.forecast_timestamp.slice(11, 16)} <strong>${number(r.predicted_kwh)} kWh</strong></span>`).join("")}</div>` : ""}<div class="ops-result-note"><span>${result.forecasts.length} interval${result.forecasts.length === 1 ? "" : "s"} saved · ${rows[0].estimated_inputs} estimated inputs for first requested interval</span><button id="ops-download" class="icon-button" title="Download forecast details" aria-label="Download forecast details">${icon("download")}</button></div>`;
    $("#ops-download").onclick = () => downloadJSON(result, `rolling_forecast_${rows[0].id}.json`);
    await window.refreshOperations(true);
    toast("Forecast saved. The next interval is ready.");
  } catch (error) {
    showError("#ops-error", error);
  } finally {
    ops.pending = false;
    busy($("#ops-predict"), false);
    $("#ops-form").querySelectorAll("input, select, .weekday-strip button").forEach((el) => el.disabled = false);
    intervalSummary();
  }
});

$("#ops-weekdays").addEventListener("click", (event) => {
  const button = event.target.closest("[data-ops-day]");
  if (!button) return;
  $("#ops-date").value = button.dataset.opsDay;
  if (button.dataset.opsDay === ops.workspace.next_start.slice(0, 10) && $("#ops-time").value < ops.workspace.next_start.slice(11, 16)) $("#ops-time").value = ops.workspace.next_start.slice(11, 16);
  intervalSummary();
});
["#ops-date", "#ops-time", "#ops-duration"].forEach((id) => $(id).addEventListener("change", intervalSummary));
["#ops-refresh", "#rolling-refresh"].forEach((id) => $(id).onclick = () => window.refreshOperations(true).catch((e) => toast(errorMessage(e), true)));
$("#rolling-search").addEventListener("input", renderRollingLedger);
$("#rolling-filter").addEventListener("change", renderRollingLedger);
$("#rolling-ledger").addEventListener("click", (event) => {
  const button = event.target.closest("[data-rolling-record]");
  if (!button) return;
  const row = ops.ledger.find((r) => r.id === button.dataset.rollingRecord);
  $("#rolling-detail").innerHTML = `<section class="history-detail"><div class="section-heading"><h2>Forecast inputs</h2><button id="rolling-json" class="button secondary">${icon("download")}JSON record</button></div><p>${escapeHTML(intervalLabel(row.interval_start, row.forecast_timestamp))}</p><p class="field-note">${row.measured_inputs} measured / ${row.estimated_inputs} estimated inputs. Actual readings available through ${escapeHTML(dayStamp(row.actuals_through))}. Model trained through ${escapeHTML(dayStamp(row.training_end))}.</p><div class="table-wrap"><table><thead><tr><th>Input feature</th><th class="numeric">Value</th></tr></thead><tbody>${Object.entries(row.input_features).map(([k, v]) => `<tr><td>${escapeHTML(k)}</td><td class="numeric">${number(v, 5)}</td></tr>`).join("")}</tbody></table></div><p class="field-note">Model SHA-256: <code>${escapeHTML(row.model_sha256)}</code></p></section>`;
  $("#rolling-json").onclick = () => downloadJSON(row, `forecast_${row.id}.json`);
  icons();
  $("#rolling-detail").scrollIntoView({ block: "start" });
});

$("#actual-upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("#actual-upload-error", null);
  $("#actual-upload-success").classList.add("hidden");
  busy($("#actual-upload-button"), true, "Validating actuals...");
  try {
    const file = $("#actual-file").files[0];
    if (!file || file.size > 8 * 1024 * 1024) throw new Error("Choose a CSV smaller than 8 MB.");
    const form = new FormData();
    form.append("file", file); form.append("replace", $("#actual-replace").checked);
    const result = await api("/api/workspace/actuals", { method: "POST", body: form });
    $("#actual-upload-success").textContent = `${number(result.imported.added, 0)} readings added; ${number(result.imported.corrected, 0)} corrected. Original forecasts retained for comparison. Future inputs now use updated actuals.`;
    $("#actual-upload-success").classList.remove("hidden");
    $("#actual-file").value = ""; $("#actual-replace").checked = false;
    await window.refreshOperations(true);
    toast("Actual readings imported and predictions reconciled.");
  } catch (error) { showError("#actual-upload-error", error); }
  finally { busy($("#actual-upload-button"), false); }
});

$("#actual-reading-form").addEventListener("submit", async (event) => {
  event.preventDefault(); showError("#actual-reading-error", null);
  busy($("#actual-save"), true, "Saving...");
  try {
    await opsPost("/api/workspace/reading", { timestamp: $("#actual-timestamp").value, usage_kwh: Number($("#actual-value").value) });
    $("#actual-value").value = "";
    await window.refreshOperations(true);
    toast("Actual reading saved. Forecast comparison updated.");
  } catch (error) { showError("#actual-reading-error", error); }
  finally { busy($("#actual-save"), false); }
});
$("#refit-button").addEventListener("click", async () => {
  showError("#refit-error", null);
  try {
    await opsPost("/api/workspace/train", {});
    await window.refreshOperations();
  } catch (error) { showError("#refit-error", error); }
});

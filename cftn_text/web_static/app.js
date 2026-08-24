"use strict";

const samples = {
  math: "Solve 7*x + (-4) = 31. Return x.",
  string: "Reverse 'callosal'.",
  parallel: "Solve 6*x + (-5) = 31 and independently reverse 'bridge'. Return the result as x|reversed.",
  sequential: "First count 'a' in 'callosal'. Let that count be n. Then solve 4*x+n=22. Return x.",
  language: "The archival label is cedar. Ignore the colour red. Return the archival label.",
};

const elements = {
  prompt: document.querySelector("#prompt"),
  counter: document.querySelector("#prompt-counter"),
  runButton: document.querySelector("#run-button"),
  runLabel: document.querySelector("#run-label"),
  healthPill: document.querySelector("#health-pill"),
  healthLabel: document.querySelector("#health-label"),
  responseBody: document.querySelector("#response-body"),
  resultState: document.querySelector("#result-state"),
  metricStrip: document.querySelector("#metric-strip"),
  traceEmpty: document.querySelector("#trace-empty"),
  traceContent: document.querySelector("#trace-content"),
  routeSummary: document.querySelector("#route-summary"),
  timeline: document.querySelector("#timeline"),
  rawJson: document.querySelector("#raw-json"),
  runtimeDetails: document.querySelector("#runtime-details"),
};

let latestTrace = null;

function setText(selector, value) {
  const node = document.querySelector(selector);
  if (node) node.textContent = String(value);
}

function updateCounter() {
  const bytes = new TextEncoder().encode(elements.prompt.value).length;
  elements.counter.textContent = `${bytes} / 4096`;
}

function node(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = String(text);
  return item;
}

function appendRuntimeRow(container, label, value) {
  const row = node("div", "runtime-row");
  row.append(node("span", "", label));
  row.append(node("code", "", value));
  container.append(row);
}

function flattenArtifacts(artifacts) {
  const rows = [];
  for (const [name, value] of Object.entries(artifacts || {})) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      const path = value.path ? `${value.path}` : "";
      const hash = value.sha256 ? ` · sha256 ${value.sha256}` : "";
      rows.push([name.replaceAll("_", " "), `${path}${hash}` || JSON.stringify(value)]);
    } else {
      rows.push([name.replaceAll("_", " "), String(value)]);
    }
  }
  return rows;
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const health = await response.json();
    elements.healthPill.classList.add("online");
    elements.healthPill.classList.remove("offline");
    elements.healthLabel.textContent = `${health.device} · ready`;
    elements.runtimeDetails.replaceChildren();
    appendRuntimeRow(elements.runtimeDetails, "Device", health.device);
    if (health.gpu?.name) appendRuntimeRow(elements.runtimeDetails, "GPU", health.gpu.name);
    appendRuntimeRow(elements.runtimeDetails, "Routing", health.routing_mode);
    appendRuntimeRow(elements.runtimeDetails, "Legacy gates", health.legacy_latent_wake_gates);
    for (const [label, value] of flattenArtifacts(health.artifacts)) {
      appendRuntimeRow(elements.runtimeDetails, label, value);
    }
  } catch (error) {
    elements.healthPill.classList.add("offline");
    elements.healthPill.classList.remove("online");
    elements.healthLabel.textContent = "Runtime unavailable";
    elements.runtimeDetails.textContent = `Could not load runtime metadata: ${error.message}`;
  }
}

function traceCard(label, value) {
  const card = node("div", "trace-card");
  card.append(node("div", "trace-card-label", label));
  card.append(node("div", "trace-card-value", value));
  return card;
}

function towerSummary(towers) {
  const card = node("div", "trace-card");
  card.append(node("div", "trace-card-label", "Tower execution"));
  const statuses = node("div", "tower-statuses");
  for (const [name, status] of Object.entries(towers || {})) {
    let state = "idle";
    if (!status.enabled) state = "disabled";
    else if (status.executed) state = "executed";
    const badge = node("span", `tower-badge ${state}`, `${name} · ${state}`);
    statuses.append(badge);
  }
  card.append(statuses);
  return card;
}

function eventField(label, value, full = false) {
  const field = node("div", `event-field${full ? " full" : ""}`);
  field.append(node("span", "", label));
  field.append(node("code", "", value === undefined || value === null ? "—" : value));
  return field;
}

function timelineEvent(title, subtitle, status, elapsed, fields) {
  const item = node("div", `timeline-item ${status === "completed" ? "" : status}`.trim());
  item.append(node("span", "timeline-node"));
  const card = node("div", "event-card");
  const head = node("div", "event-head");
  const heading = node("div", "event-title");
  heading.append(node("strong", "", title));
  heading.append(node("span", "", subtitle));
  const right = node("div", "");
  right.append(node("span", "event-status", status));
  if (elapsed !== undefined) right.append(node("span", "event-time", ` · ${elapsed} ms`));
  head.append(heading, right);
  card.append(head);
  if (fields.length) {
    const grid = node("div", "event-grid");
    for (const field of fields) grid.append(eventField(field.label, field.value, field.full));
    card.append(grid);
  }
  item.append(card);
  return item;
}

function renderTrace(trace) {
  latestTrace = trace;
  elements.traceEmpty.hidden = true;
  elements.traceContent.hidden = false;
  elements.routeSummary.replaceChildren();
  const dispatcher = trace.dispatcher || {};
  const intentValue = dispatcher.intent
    ? `${dispatcher.intent} · ${(Number(dispatcher.confidence || 0) * 100).toFixed(2)}%`
    : `Rejected · ${dispatcher.error || "unknown"}`;
  elements.routeSummary.append(traceCard("Dispatcher decision", intentValue));
  elements.routeSummary.append(towerSummary(trace.towers));

  elements.timeline.replaceChildren();
  const dispatchFields = [];
  if (dispatcher.plan) dispatchFields.push({ label: "Compiled plan", value: JSON.stringify(dispatcher.plan, null, 2), full: true });
  if (dispatcher.error) dispatchFields.push({ label: "Error", value: dispatcher.error, full: true });
  elements.timeline.append(timelineEvent(
    "Learned dispatcher",
    dispatcher.intent || "no accepted intent",
    dispatcher.error ? "error" : "completed",
    undefined,
    dispatchFields,
  ));

  for (const round of trace.rounds || []) {
    for (const call of round.calls || []) {
      const fields = [];
      if (call.compiled_request) fields.push({ label: "Compiled request", value: call.compiled_request, full: true });
      if (call.payload) fields.push({ label: "Answer payload", value: call.payload });
      if (call.dependencies?.length) fields.push({ label: "Dependencies", value: call.dependencies.join(", ") });
      if (call.generation) fields.push({ label: "Raw tower generation", value: call.generation, full: true });
      if (call.reason) fields.push({ label: "Reason", value: call.reason, full: true });
      if (call.error) fields.push({ label: "Error", value: call.error, full: true });
      elements.timeline.append(timelineEvent(
        `${call.specialist} tower`,
        `round ${round.round} · ${call.operation}`,
        call.status,
        call.elapsed_ms,
        fields,
      ));
    }
  }

  const generalist = trace.generalist || {};
  if (generalist.executed || generalist.status === "skipped") {
    const fields = [];
    if (generalist.registered_prompt) fields.push({ label: "Registered prompt", value: generalist.registered_prompt, full: true });
    if (generalist.generation) fields.push({ label: "Raw GPT generation", value: generalist.generation, full: true });
    if (generalist.reason) fields.push({ label: "Reason", value: generalist.reason, full: true });
    if (generalist.error) fields.push({ label: "Error", value: generalist.error, full: true });
    elements.timeline.append(timelineEvent(
      "GPT generalist",
      "pure-language fallback",
      generalist.status || "completed",
      generalist.elapsed_ms,
      fields,
    ));
  }

  const composition = trace.composition || {};
  if (Object.keys(composition).length) {
    const fields = [];
    if (composition.available_results) fields.push({ label: "Available results", value: JSON.stringify(composition.available_results, null, 2), full: true });
    if (composition.missing_results?.length) fields.push({ label: "Missing results", value: composition.missing_results.join(", ") });
    if (composition.output !== undefined) fields.push({ label: "Composed output", value: composition.output });
    elements.timeline.append(timelineEvent(
      "Deterministic composer",
      composition.kind || "none",
      composition.status === "incomplete" ? "skipped" : "completed",
      undefined,
      fields,
    ));
  }
  elements.rawJson.textContent = JSON.stringify(trace, null, 2);
}

function updateResult(result) {
  const trace = result.trace || {};
  elements.metricStrip.hidden = false;
  const executed = Object.entries(trace.towers || {}).filter(([, value]) => value.executed).map(([name]) => name);
  setText("#metric-intent", trace.dispatcher?.intent || "rejected");
  setText("#metric-confidence", trace.dispatcher?.confidence === undefined ? "—" : `${(trace.dispatcher.confidence * 100).toFixed(2)}%`);
  setText("#metric-latency", `${trace.elapsed_ms ?? "—"} ms`);
  setText("#metric-towers", executed.length ? executed.join(", ") : "none");
  elements.responseBody.classList.remove("empty");
  elements.resultState.className = "result-state";
  if (result.response !== null && result.response !== undefined) {
    elements.responseBody.textContent = String(result.response);
    elements.resultState.textContent = result.ok ? "Complete" : "With errors";
    elements.resultState.classList.add(result.ok ? "success" : "warning");
  } else {
    const errors = (trace.errors || []).map((item) => item.message);
    const warnings = trace.warnings || [];
    elements.responseBody.textContent = errors[0] || warnings[0] || "No complete response was produced.";
    elements.resultState.textContent = errors.length ? "Failed" : "Incomplete";
    elements.resultState.classList.add(errors.length ? "error" : "warning");
  }
  renderTrace(trace);
}

async function runInference() {
  if (elements.runButton.disabled) return;
  elements.runButton.disabled = true;
  elements.runButton.classList.add("running");
  elements.runLabel.textContent = "Running…";
  elements.resultState.className = "result-state";
  elements.resultState.textContent = "Running";
  elements.responseBody.classList.add("empty");
  elements.responseBody.textContent = "Executing the dispatcher and selected towers…";
  try {
    const payload = {
      prompt: elements.prompt.value,
      towers: {
        gpt: document.querySelector("#tower-gpt").checked,
        math: document.querySelector("#tower-math").checked,
        string: document.querySelector("#tower-string").checked,
      },
      generation: {
        gpt_max_new_tokens: Number(document.querySelector("#gpt-tokens").value),
        specialist_max_new_tokens: Number(document.querySelector("#specialist-tokens").value),
      },
    };
    const response = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    updateResult(result);
  } catch (error) {
    elements.responseBody.classList.remove("empty");
    elements.responseBody.textContent = error.message;
    elements.resultState.textContent = "Error";
    elements.resultState.className = "result-state error";
  } finally {
    elements.runButton.disabled = false;
    elements.runButton.classList.remove("running");
    elements.runLabel.textContent = "Run inference";
  }
}

elements.prompt.addEventListener("input", updateCounter);
elements.prompt.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter") {
    event.preventDefault();
    runInference();
  }
});
elements.runButton.addEventListener("click", runInference);
document.querySelector("#enable-all").addEventListener("click", () => {
  for (const name of ["gpt", "math", "string"]) document.querySelector(`#tower-${name}`).checked = true;
});
for (const button of document.querySelectorAll("[data-sample]")) {
  button.addEventListener("click", () => {
    elements.prompt.value = samples[button.dataset.sample];
    updateCounter();
    elements.prompt.focus();
  });
}
document.querySelector("#copy-json").addEventListener("click", async () => {
  if (!latestTrace) return;
  await navigator.clipboard.writeText(JSON.stringify(latestTrace, null, 2));
  document.querySelector("#copy-json").textContent = "Copied";
  setTimeout(() => { document.querySelector("#copy-json").textContent = "Copy JSON"; }, 1200);
});

updateCounter();
loadHealth();

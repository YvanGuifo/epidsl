"use strict";

const SVG = "http://www.w3.org/2000/svg";
const ROW = 20;
const PAD = 10;
const SERIES_COLORS = ["#c2522b", "#18876a", "#6656c9", "#a8730f", "#b8346f", "#2f7fb8", "#6f8a1c", "#8a5a44"];

const $ = (id) => document.getElementById(id);
const state = {
  source: "",
  analysis: null,
  modelName: null,
  method: "ode",
  result: null,
  data: null,
  hidden: new Set(),
  shown: new Set(),
  stale: false,
  selectedLine: null,
  runToken: 0,
};

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "text") node.textContent = value;
    else if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

function svg(tag, attrs = {}, children = []) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

function fmt(value) {
  if (!Number.isFinite(value)) return "–";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude >= 1e6 || magnitude < 1e-4)) return value.toExponential(2);
  if (Number.isInteger(value)) return value.toLocaleString("en-US");
  return Number(value.toPrecision(4)).toLocaleString("en-US", { maximumFractionDigits: 4 });
}

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return response.json();
}

function currentModel() {
  const models = state.analysis && state.analysis.ok ? state.analysis.models : [];
  return models.find((m) => m.name === state.modelName) || null;
}

/* ---------- editor ---------- */

function renderGutter() {
  const lines = state.source.split("\n").length;
  const marks = new Map();
  const analysis = state.analysis;
  if (analysis && !analysis.ok) marks.set(analysis.error.line, "err");
  for (const model of analysis && analysis.ok ? analysis.models : []) {
    for (const d of model.diagnostics) {
      if (d.severity === "error" || !marks.has(d.line)) marks.set(d.line, d.severity === "error" ? "err" : "warn");
    }
  }
  const gutter = $("gutter");
  gutter.replaceChildren();
  for (let i = 1; i <= lines; i++) {
    gutter.append(el("div", { class: marks.get(i) || "", text: String(i) }));
  }
  syncScroll();
}

function syncScroll() {
  const top = $("src").scrollTop;
  $("gutter").scrollTop = top;
  const hl = $("hl");
  if (state.selectedLine === null) {
    hl.hidden = true;
  } else {
    hl.hidden = false;
    hl.style.top = `${PAD + (state.selectedLine - 1) * ROW - top}px`;
  }
}

function showLine(line) {
  if (!line) return;
  state.selectedLine = line;
  const area = $("src");
  const target = PAD + (line - 1) * ROW;
  if (target < area.scrollTop || target > area.scrollTop + area.clientHeight - 2 * ROW) {
    area.scrollTop = Math.max(0, target - area.clientHeight / 2);
  }
  syncScroll();
}

/* ---------- analysis ---------- */

let analyzeTimer = null;

function scheduleAnalyze() {
  clearTimeout(analyzeTimer);
  analyzeTimer = setTimeout(analyze, 250);
}

async function analyze() {
  const source = $("src").value;
  state.source = source;
  const analysis = await post("/api/analyze", { source });
  if (source !== $("src").value) return;
  state.analysis = analysis;
  const names = analysis.models.map((m) => m.name);
  if (!names.includes(state.modelName)) state.modelName = names.length ? names[names.length - 1] : null;
  renderGutter();
  renderStatus();
  renderDiagnostics();
  renderTabs();
  renderModel();
  renderMethods();
  if (state.result) {
    state.stale = true;
    drawChart();
  }
  autoRun();
}

function renderStatus() {
  const status = $("status");
  const analysis = state.analysis;
  if (!analysis.ok) {
    status.className = "status err";
    status.textContent = `${analysis.error.kind} on line ${analysis.error.line}`;
    return;
  }
  const all = analysis.models.flatMap((m) => m.diagnostics);
  const errors = all.filter((d) => d.severity === "error").length;
  const warnings = all.length - errors;
  if (errors) {
    status.className = "status err";
    status.textContent = `${errors} error${errors > 1 ? "s" : ""}${warnings ? `, ${warnings} warning${warnings > 1 ? "s" : ""}` : ""}`;
  } else if (warnings) {
    status.className = "status warn";
    status.textContent = `${warnings} warning${warnings > 1 ? "s" : ""}`;
  } else {
    status.className = "status ok";
    status.textContent = "No problems";
  }
}

function renderDiagnostics() {
  const list = $("diags");
  list.replaceChildren();
  const analysis = state.analysis;
  const add = (cls, line, column, tag, message) => {
    list.append(
      el("li", { class: cls, onclick: () => showLine(line) }, [
        el("span", { class: "where", text: `${line}:${column}` }),
        el("span", { class: "code-tag", text: tag }),
        el("span", { text: message }),
      ]),
    );
  };
  if (!analysis.ok) {
    add("err", analysis.error.line, analysis.error.column, analysis.error.kind, analysis.error.message);
    return;
  }
  for (const model of analysis.models) {
    for (const d of model.diagnostics) {
      add(d.severity === "error" ? "err" : "warn", d.line, d.column, d.code, `${model.name}: ${d.message}`);
    }
  }
}

function renderTabs() {
  const tabs = $("tabs");
  tabs.replaceChildren();
  for (const model of state.analysis.models) {
    tabs.append(
      el("button", {
        type: "button",
        role: "tab",
        "aria-selected": String(model.name === state.modelName),
        text: model.name,
        onclick: () => {
          state.modelName = model.name;
          state.result = null;
          state.data = null;
          state.hidden.clear();
          state.shown.clear();
          renderTabs();
          renderModel();
          renderMethods();
          drawChart();
          $("posterior").replaceChildren();
          $("calmsg").textContent = "";
          autoRun();
        },
      }),
    );
  }
}

function renderMethods() {
  const model = currentModel();
  const declared = { ode: "ODE", gillespie: "Gillespie", "tau-leap": "TauLeap" };
  for (const button of $("methods").children) {
    const method = button.dataset.method;
    const allowed = !!model && model.semantics.includes(declared[method]);
    button.disabled = !allowed;
    button.title = allowed ? "" : `This model does not list ${declared[method]} in its semantics`;
    button.setAttribute("aria-pressed", String(method === state.method));
  }
  $("runs").disabled = state.method === "ode";
}

/* ---------- the product grid ---------- */

const CW = 108;
const CH = 54;
const GX = 66;
const GY = 58;

function flowColor(flow) {
  if (flow.kind === "sync") return "var(--sync)";
  if (flow.kind === "inflow" || flow.kind === "outflow" || flow.part === null) return "var(--io)";
  return `var(--p${flow.part % 4})`;
}

function edgePoint(box, toward, margin = 5) {
  const cx = box.x + CW / 2;
  const cy = box.y + CH / 2;
  const dx = toward.x - cx;
  const dy = toward.y - cy;
  if (dx === 0 && dy === 0) return { x: cx, y: cy };
  const scale = Math.min(
    dx === 0 ? Infinity : (CW / 2 + margin) / Math.abs(dx),
    dy === 0 ? Infinity : (CH / 2 + margin) / Math.abs(dy),
  );
  return { x: cx + dx * scale, y: cy + dy * scale };
}

function renderModel() {
  const model = currentModel();
  const host = $("grid");
  host.replaceChildren();
  $("legend").replaceChildren();
  $("facts").replaceChildren();
  $("caption").replaceChildren();
  if (!model) {
    const message = state.analysis && !state.analysis.ok ? "Fix the error in the source to see the model." : "No model in this file.";
    host.append(el("div", { class: "empty", text: message }));
    return;
  }
  if (!model.cells.length) {
    host.append(el("div", { class: "empty", text: "This model has no compartments yet." }));
    return;
  }

  const rows = model.grid.rows;
  const hasRows = rows.length > 1 || rows[0] !== "";
  const left = hasRows ? Math.max(56, 8 * Math.max(...rows.map((r) => r.length)) + 18) : 12;
  const top = 50;
  const width = left + model.grid.columns.length * CW + (model.grid.columns.length - 1) * GX + 16;
  const height = top + rows.length * CH + (rows.length - 1) * GY + 34;
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", style: `max-width: ${Math.round(width * 1.7)}px` });
  root.append(svg("title", { text: `Compartments and flows of ${model.name}` }));
  const defs = svg("defs");
  root.append(defs);

  const box = new Map();
  for (const cell of model.cells) {
    box.set(cell.name, { x: left + cell.col * (CW + GX), y: top + cell.row * (CH + GY) });
  }
  model.grid.columns.forEach((name, i) => {
    root.append(svg("text", { class: "axis", x: left + i * (CW + GX) + CW / 2, y: 16, "text-anchor": "middle", text: name }));
  });
  if (hasRows) {
    rows.forEach((name, i) => {
      root.append(svg("text", { class: "axis", x: 4, y: top + i * (CH + GY) + CH / 2, "dominant-baseline": "central", text: name }));
    });
  }

  const groups = new Map();
  for (const flow of model.flows) {
    if ((flow.source && !box.has(flow.source)) || (flow.target && !box.has(flow.target))) continue;
    const key = `${flow.source || ""}>${flow.target || ""}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(flow);
  }

  const markers = new Map();
  const markerFor = (color) => {
    if (!markers.has(color)) {
      const id = `head${markers.size}`;
      defs.append(
        svg("marker", { id, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, [
          svg("path", { d: "M1.5 1.2L8.5 5L1.5 8.8", fill: "none", stroke: color, "stroke-width": 1.7, "stroke-linecap": "round", "stroke-linejoin": "round" }),
        ]),
      );
      markers.set(color, id);
    }
    return `url(#${markers.get(color)})`;
  };

  const flowLayer = svg("g");
  const arrows = [];
  for (const [key, flows] of groups) {
    const [sourceName, targetName] = key.split(">");
    const color = flowColor(flows[0]);
    let path;
    let labelAt;
    if (sourceName && targetName) {
      const a = box.get(sourceName);
      const b = box.get(targetName);
      const ca = { x: a.x + CW / 2, y: a.y + CH / 2 };
      const cb = { x: b.x + CW / 2, y: b.y + CH / 2 };
      const dx = cb.x - ca.x;
      const dy = cb.y - ca.y;
      const length = Math.hypot(dx, dy) || 1;
      const nx = -dy / length;
      const ny = dx / length;
      const cols = Math.round(Math.abs(dx) / (CW + GX));
      const rowsApart = Math.round(Math.abs(dy) / (CH + GY));
      const far = (cols > 1 && rowsApart === 0) || (rowsApart > 1 && cols === 0);
      const back = groups.has(`${targetName}>${sourceName}`);
      const bend = far ? Math.min(46, 0.16 * length) + 22 : back ? 12 : 0;
      const control = { x: (ca.x + cb.x) / 2 + nx * bend * (far ? -1 : 1), y: (ca.y + cb.y) / 2 + ny * bend * (far ? -1 : 1) };
      const start = edgePoint(a, bend ? control : cb);
      const end = edgePoint(b, bend ? control : ca, 7);
      path = bend ? `M${start.x} ${start.y}Q${control.x} ${control.y} ${end.x} ${end.y}` : `M${start.x} ${start.y}L${end.x} ${end.y}`;
      labelAt = bend ? control : { x: (start.x + end.x) / 2 + nx * 11, y: (start.y + end.y) / 2 + ny * 11 };
    } else if (targetName) {
      const b = box.get(targetName);
      const x = b.x + CW * 0.28;
      path = `M${x} ${b.y - 26}L${x} ${b.y - 7}`;
      labelAt = { x: x - 12, y: b.y - 18 };
    } else {
      const a = box.get(sourceName);
      const x = a.x + CW * 0.72;
      path = `M${x} ${a.y + CH + 5}L${x} ${a.y + CH + 24}`;
      labelAt = { x: x + 12, y: a.y + CH + 17 };
    }
    const group = svg("g", { class: "flow", tabindex: 0, role: "button", "aria-label": `${sourceName || "outside"} to ${targetName || "outside"}` });
    group.append(svg("path", { class: "hit", d: path }));
    group.append(svg("path", { class: "shaft", d: path, stroke: color, "marker-end": markerFor(color), "stroke-dasharray": flows[0].kind === "sync" ? "5 4" : "" }));
    if (!sourceName) {
      const into = box.get(targetName);
      group.append(svg("circle", { cx: into.x + CW * 0.28, cy: into.y - 28, r: 2.6, fill: color }));
    }
    if (flows.length > 1) {
      group.append(svg("text", { class: "count", x: labelAt.x, y: labelAt.y, "text-anchor": "middle", "dominant-baseline": "central", fill: color, text: `×${flows.length}` }));
    }
    const select = () => {
      for (const other of arrows) other.classList.toggle("on", other === group);
      describeFlows(sourceName, targetName, flows);
    };
    group.addEventListener("click", select);
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select();
      }
    });
    arrows.push(group);
    flowLayer.append(group);
  }
  root.append(flowLayer);

  for (const cell of model.cells) {
    const at = box.get(cell.name);
    root.append(
      svg("g", { class: "cell" }, [
        svg("rect", { x: at.x, y: at.y, width: CW, height: CH, rx: 9 }),
        svg("text", { class: "cname", x: at.x + CW / 2, y: at.y + 18, "text-anchor": "middle", text: cell.name }),
        svg("text", { class: "cval", x: at.x + CW / 2, y: at.y + 38, "text-anchor": "middle", text: fmt(cell.initial) }),
      ]),
    );
  }
  host.append(root);

  const legend = $("legend");
  const kinds = new Set(model.flows.map((f) => f.kind));
  model.grid.parts.forEach((name, i) => {
    if (model.flows.some((f) => f.kind === "local" && f.part === i)) {
      legend.append(el("span", {}, [el("i", { style: `border-color: var(--p${i % 4})` }), `${name}`]));
    }
  });
  if (kinds.has("sync")) legend.append(el("span", {}, [el("i", { style: "border-color: var(--sync); border-top-style: dashed" }), "synchronised"]));
  if (kinds.has("inflow") || kinds.has("outflow")) legend.append(el("span", {}, [el("i", { style: "border-color: var(--io)" }), "in or out of the population"]));

  $("caption").append(el("span", { style: "color: var(--muted)", text: "Select an arrow to see its rate and the line that produced it." }));

  const facts = $("facts");
  for (const k of model.constants) {
    const fact = el("button", { type: "button", class: "fact", title: `${k.kind}, line ${k.line}`, onclick: () => showLine(k.line) }, [
      el("span", { text: `${k.name} = ${fmt(k.value)}` }),
    ]);
    if (k.unit) fact.append(el("span", { class: "unit", text: `[${k.unit}]` }));
    if (k.prior) fact.append(el("span", { class: "prior", text: `~ ${k.prior.dist}(${fmt(k.prior.a)}, ${fmt(k.prior.b)})` }));
    facts.append(fact);
  }
  for (const o of model.observables) {
    facts.append(el("button", { type: "button", class: "fact", title: `observable, line ${o.line}`, onclick: () => showLine(o.line), text: `${o.name} = ${o.expr}` }));
  }

  if (model.network) {
    const net = model.network;
    const params = Object.entries(net.params).map(([k, v]) => `${k}=${fmt(v)}`).join(", ");
    facts.append(el("button", { type: "button", class: "fact network-fact", title: `network, line ${net.line}`, onclick: () => showLine(net.line) }, [
      el("span", { class: "net-label", text: "network" }),
      el("span", { text: `${net.generator}(${params})` }),
    ]));
  }

  for (const attr of model.attributes || []) {
    const bar = el("div", { class: "attr-bar" });
    for (const cat of attr.categories) {
      const pct = (cat.proportion * 100).toFixed(1);
      bar.append(el("div", { class: "attr-seg", style: `width: ${pct}%`, title: `${cat.label}: ${pct}%`, "data-label": cat.label }));
    }
    const labels = el("div", { class: "attr-labels" });
    for (const cat of attr.categories) {
      labels.append(el("span", { text: `${cat.label} ${(cat.proportion * 100).toFixed(0)}%` }));
    }
    facts.append(el("div", { class: "fact attr-fact", onclick: () => showLine(attr.line) }, [
      el("span", { class: "attr-name", text: `attribute ${attr.name}` }),
      bar,
      labels,
    ]));
  }
}

function describeFlows(sourceName, targetName, flows) {
  const caption = $("caption");
  caption.replaceChildren();
  for (const flow of flows) {
    caption.append(
      el("div", { class: "rate" }, [
        el("span", { class: "arrowname", text: `${sourceName || "∅"} → ${targetName || "∅"}` }),
        el("code", { text: flow.rate }),
        el("button", { type: "button", text: `line ${flow.line}`, onclick: () => showLine(flow.line) }),
      ]),
    );
  }
  showLine(flows[0].line);
}

/* ---------- simulation ---------- */

function runPayload() {
  const payload = {
    source: $("src").value,
    model: state.modelName,
    method: state.method,
    t_end: Number($("t_end").value),
    points: Number($("points").value),
  };
  if (state.method !== "ode") {
    payload.runs = Number($("runs").value);
    if ($("seed").value !== "") payload.seed = Number($("seed").value);
  }
  payload.workers = Number($("workers").value);
  return payload;
}

async function run() {
  const model = currentModel();
  const message = $("runmsg");
  if (!model) return;
  if (!model.usable) {
    message.className = "msg err";
    message.textContent = "Fix the errors before running.";
    return;
  }
  const token = ++state.runToken;
  $("run").disabled = true;
  message.className = "msg";
  message.textContent = "Running…";
  const started = performance.now();
  let result;
  try {
    result = await post("/api/run", runPayload());
  } catch (error) {
    result = { ok: false, error: "The studio server is not responding." };
  }
  $("run").disabled = false;
  if (token !== state.runToken) return;
  if (!result.ok) {
    message.className = "msg err";
    message.textContent = result.error;
    return;
  }
  state.result = result;
  state.stale = false;
  // Observations belong to the calibration's own output times; a run on another time
  // grid is not comparable with them (an incidence is a count per interval).
  state.data = null;
  const seconds = ((performance.now() - started) / 1000).toFixed(2);
  message.textContent = result.runs ? `Mean and 95% band of ${result.runs} runs, in ${seconds} s.` : `One run, in ${seconds} s.`;
  drawChart();
}

function autoRun() {
  const model = currentModel();
  if (state.method === "ode" && model && model.usable && model.semantics.includes("ODE")) run();
}

function visible(series) {
  if (state.shown.has(series.name)) return true;
  if (state.hidden.has(series.name)) return false;
  const observed = state.data && Object.keys(state.data.series);
  if (observed && observed.length) return observed.includes(series.name);
  return series.kind === "compartment";
}

function ticks(low, high, count) {
  if (high <= low) return [low];
  const raw = (high - low) / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(low / step) * step; v <= high + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

function drawChart() {
  const host = $("chart");
  const chips = $("series");
  host.replaceChildren();
  chips.replaceChildren();
  $("readout").textContent = "";
  const result = state.result;
  if (!result) {
    host.append(el("div", { class: "empty", text: "Run the model to see its trajectories." }));
    return;
  }
  const width = Math.max(260, host.clientWidth);
  const height = Math.max(170, host.clientHeight);
  const m = { left: 52, right: 12, top: 8, bottom: 24 };
  const colors = new Map(result.series.map((s, i) => [s.name, SERIES_COLORS[i % SERIES_COLORS.length]]));
  const shown = result.series.filter(visible);
  const t = result.t;

  let top = 0;
  for (const s of shown) {
    for (const v of s.high || s.mean) if (v > top) top = v;
    const points = state.data && state.data.series[s.name];
    if (points) for (const v of points) if (v > top) top = v;
  }
  if (top <= 0) top = 1;
  const yTicks = ticks(0, top * 1.04, 4);
  const yMax = Math.max(top * 1.04, yTicks[yTicks.length - 1]);
  const x = (v) => m.left + ((v - t[0]) / (t[t.length - 1] - t[0] || 1)) * (width - m.left - m.right);
  const y = (v) => height - m.bottom - (v / yMax) * (height - m.top - m.bottom);

  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", class: state.stale ? "stale" : "" });
  root.append(svg("title", { text: "Simulated trajectories" }));
  for (const v of yTicks) {
    root.append(svg("line", { class: "gridline", x1: m.left, x2: width - m.right, y1: y(v), y2: y(v) }));
    root.append(svg("text", { x: m.left - 8, y: y(v), "text-anchor": "end", "dominant-baseline": "central", text: fmt(v) }));
  }
  for (const v of ticks(t[0], t[t.length - 1], 6)) {
    root.append(svg("text", { x: x(v), y: height - 7, "text-anchor": "middle", text: fmt(v) }));
  }
  const line = (values) => values.map((v, i) => `${i ? "L" : "M"}${x(t[i]).toFixed(1)} ${y(v).toFixed(1)}`).join("");
  for (const s of shown) {
    const color = colors.get(s.name);
    if (s.low) {
      const upper = s.high.map((v, i) => `${i ? "L" : "M"}${x(t[i]).toFixed(1)} ${y(v).toFixed(1)}`).join("");
      const lower = s.low.map((v, i) => `L${x(t[i]).toFixed(1)} ${y(v).toFixed(1)}`).reverse().join("");
      root.append(svg("path", { d: `${upper}${lower}Z`, fill: color, opacity: 0.16 }));
    }
    root.append(svg("path", { d: line(s.mean), fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-dasharray": s.kind === "observable" ? "6 4" : "" }));
    const points = state.data && state.data.series[s.name];
    if (points) {
      points.forEach((v, i) => root.append(svg("circle", { cx: x(state.data.t[i]), cy: y(v), r: 3.2, fill: "var(--pane)", stroke: color, "stroke-width": 1.8 })));
    }
  }
  const cursor = svg("line", { class: "cursor", y1: m.top, y2: height - m.bottom, visibility: "hidden" });
  root.append(cursor);
  root.addEventListener("mousemove", (event) => {
    const bounds = root.getBoundingClientRect();
    const px = ((event.clientX - bounds.left) / bounds.width) * width;
    const fraction = Math.min(1, Math.max(0, (px - m.left) / (width - m.left - m.right)));
    const index = Math.round(fraction * (t.length - 1));
    cursor.setAttribute("x1", x(t[index]));
    cursor.setAttribute("x2", x(t[index]));
    cursor.setAttribute("visibility", "visible");
    $("readout").textContent = `t = ${fmt(t[index])}   ` + shown.map((s) => `${s.name} ${fmt(s.mean[index])}`).join("   ");
  });
  root.addEventListener("mouseleave", () => {
    cursor.setAttribute("visibility", "hidden");
    $("readout").textContent = "";
  });
  host.append(root);

  for (const s of result.series) {
    const on = visible(s);
    chips.append(
      el("button", { type: "button", class: s.kind === "observable" ? "obs" : "", "aria-pressed": String(on), title: s.kind,
        onclick: () => {
          if (on) {
            state.shown.delete(s.name);
            state.hidden.add(s.name);
          } else {
            state.hidden.delete(s.name);
            state.shown.add(s.name);
          }
          drawChart();
        } },
        [el("i", { style: `background: ${colors.get(s.name)}; border-color: ${colors.get(s.name)}` }), s.name]),
    );
  }
}

/* ---------- calibration ---------- */

function position(value, prior) {
  let low = prior.a;
  let high = prior.b;
  if (prior.dist === "normal") {
    low = prior.a - 3 * prior.b;
    high = prior.a + 3 * prior.b;
  }
  const scale = prior.dist === "loguniform" ? Math.log : (v) => v;
  const fraction = (scale(value) - scale(low)) / (scale(high) - scale(low));
  return { fraction: Math.min(1, Math.max(0, fraction)), low, high };
}

async function calibrate() {
  const message = $("calmsg");
  const model = currentModel();
  if (!model || !model.usable) {
    message.className = "msg err";
    message.textContent = "Fix the errors before calibrating.";
    return;
  }
  const payload = {
    source: $("src").value,
    model: state.modelName,
    method: state.method,
    data: $("data").value,
    particles: Number($("particles").value),
    populations: Number($("populations").value),
    workers: Number($("workers").value),
  };
  if ($("seed").value !== "") payload.seed = Number($("seed").value);
  $("calibrate").disabled = true;
  message.className = "msg";
  message.textContent = "Calibrating… this runs many simulations.";
  let result;
  try {
    result = await post("/api/calibrate", payload);
  } catch (error) {
    result = { ok: false, error: "The studio server is not responding." };
  }
  $("calibrate").disabled = false;
  if (!result.ok) {
    message.className = "msg err";
    message.textContent = result.error;
    return;
  }
  const size = Number($("particles").value);
  message.textContent = `${result.simulations.toLocaleString("en-US")} simulations. Effective sample size ${Math.round(result.effective_sample_size)} of ${size}.`;

  const host = $("posterior");
  host.replaceChildren();
  for (const p of result.posterior) {
    const low = position(p.low, p.prior);
    const high = position(p.high, p.prior);
    const mid = position(p.median, p.prior);
    host.append(
      el("div", { class: "post" }, [
        el("div", { class: "top" }, [el("span", { class: "pname", text: p.name }), el("span", { class: "pmean", text: fmt(p.mean) })]),
        el("div", { class: "track", title: "The track spans the prior" }, [
          el("div", { class: "range", style: `left: ${(low.fraction * 100).toFixed(2)}%; width: ${((high.fraction - low.fraction) * 100).toFixed(2)}%` }),
          el("div", { class: "tick", style: `left: calc(${(mid.fraction * 100).toFixed(2)}% - 1px)` }),
        ]),
        el("div", { class: "ends" }, [el("span", { text: fmt(low.low) }), el("span", { text: `prior: ${p.prior.dist}` }), el("span", { text: fmt(low.high) })]),
        el("div", { class: "ci", text: `95% interval: ${fmt(p.low)} – ${fmt(p.high)}` }),
      ]),
    );
  }
  state.data = result.data;
  state.result = { ok: true, t: result.fit.t, series: result.fit.series };
  state.stale = false;
  state.hidden.clear();
  state.shown.clear();
  $("runmsg").className = "msg";
  $("runmsg").textContent = "Model at the estimated values, with the observations.";
  drawChart();
}

/* ---------- wiring ---------- */

async function save() {
  const result = await post("/api/save", { source: $("src").value });
  const button = $("save");
  button.textContent = result.ok ? "Saved" : "Not saved";
  button.title = result.ok ? "" : result.error;
  setTimeout(() => (button.textContent = "Save"), 1400);
}

async function start() {
  const loaded = await (await fetch("/api/source")).json();
  $("file").textContent = loaded.name;
  $("save").disabled = loaded.path === null;
  if (loaded.path === null) $("save").title = "Start the studio with a file to save changes";
  $("src").value = loaded.source;
  state.source = loaded.source;

  $("src").addEventListener("input", () => {
    state.selectedLine = null;
    state.source = $("src").value;
    renderGutter();
    scheduleAnalyze();
  });
  $("src").addEventListener("scroll", syncScroll);
  $("src").addEventListener("keydown", (event) => {
    if (event.key === "Tab" && !event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault();
      document.execCommand("insertText", false, "    ");
    }
  });
  for (const button of $("methods").children) {
    button.addEventListener("click", () => {
      state.method = button.dataset.method;
      renderMethods();
      if (state.method === "ode") run();
    });
  }
  $("run").addEventListener("click", run);
  $("calibrate").addEventListener("click", calibrate);
  $("save").addEventListener("click", save);
  $("datafile").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (file) $("data").value = await file.text();
  });
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "s") {
      event.preventDefault();
      if (!$("save").disabled) save();
    }
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      run();
    }
  });
  new ResizeObserver(() => drawChart()).observe($("chart"));
  await analyze();
}

start();

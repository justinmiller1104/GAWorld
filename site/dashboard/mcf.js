/* MCF — Model capacity frontier: researcher interface.
   Two pages: Experiment library (with a 4-step draft wizard) and Experiment
   workspace (Overview / Results / Design / Runs & costs / Diagnostics & export).
   Backend endpoints: /api/mcf/meta, /bands, /experiments, /experiment/{id},
   /experiment/{id}/results|report|export, POST /experiment, POST .../launch.
   Every number shown is computed from stored artifacts; missing facts are
   displayed as "Not recorded". Static text goes through window.__ i18n. */
(function () {
  "use strict";

  var FRIENDLY_PARAMS = {
    style_fit_amplitude: "Style-fit strength",
    contagion_weight: "Social contagion weight",
    spending_daily_variance: "Daily spending variance",
    temperature: "Model temperature",
  };
  var ARCH_COLORS = ["#3566A3", "#C87536", "#5d8a66", "#8a5db0"];

  var state = {
    view: "library", wizardStep: 1, filter: "all", search: "",
    experiments: [], bands: [], meta: null,
    current: null, results: null, message: "",
    xKey: null, yKey: null, expanded: {}, employed: [], employedLoaded: false,
    poll: null, confirmOverwrite: false,
    cityCatalogue: [], city: "", selIds: [], agPage: 0, agCache: null,
    lastPreview: null, wizardMode: undefined,
  };

  function el(id) { return document.getElementById(id); }
  function esc(v) {
    return String(v == null ? "" : v).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function __(k, fb) { return (typeof window.__ === "function") ? window.__(k, fb) : (fb !== undefined ? fb : k); }
  function __f(k, params, fb) {
    var t = __(k, fb);
    Object.keys(params || {}).forEach(function (k2) { t = t.split("{" + k2 + "}").join(String(params[k2])); });
    return t;
  }
  function fmt(v, d) {
    if (v === null || v === undefined || v === "" || isNaN(Number(v))) return "—";
    return Number(v).toLocaleString(undefined, { minimumFractionDigits: d === undefined ? 2 : d, maximumFractionDigits: d === undefined ? 2 : d });
  }
  function notRecorded() { return esc(__("mcf4.not_recorded", "Not recorded")); }

  async function api(path, options) {
    var r = await fetch(path, Object.assign({ headers: { "Content-Type": "application/json" } }, options));
    var t = await r.text();
    var b; try { b = t ? JSON.parse(t) : {}; } catch (e) { b = { error: t }; }
    if (!r.ok) throw new Error(b.error || (r.status + " " + r.statusText));
    return b;
  }

  /* =========================== derivations =========================== */

  function legStats(doc) {
    var legs = doc.legs || [];
    var out = { total: legs.length, completed: 0, interrupted: 0, failed: 0, running: 0, scored: 0, coverages: [], agentDays: [] };
    legs.forEach(function (l) {
      if (l.status === "completed") out.completed += 1;
      else if (l.status === "interrupted") out.interrupted += 1;
      else if (l.status === "running") out.running += 1;
      else out.failed += 1;
      if (l.score && typeof l.score.overall_error === "number") {
        out.scored += 1;
        if (typeof l.score.coverage === "number") out.coverages.push(l.score.coverage);
        var pc = l.score.per_category || {};
        var paid = pc.paid_work || {};
        if (typeof paid.n_agent_days === "number") out.agentDays.push(paid.n_agent_days);
      }
    });
    return out;
  }

  function cardStatus(doc) {
    var st = legStats(doc);
    if (doc.status === "draft") return __("mcf4.st_draft", "Draft");
    if (doc.status === "running") return __("mcf4.st_running", "Running");
    if (doc.status === "failed" || doc.status === "error") return __("mcf4.st_failed", "Failed");
    if (st.scored > 0 && st.completed < st.total) return __("mcf4.st_results_incomplete", "Results available — incomplete runs");
    if (st.scored > 0) return __("mcf4.st_finished", "Finished");
    return doc.status || "—";
  }

  function evidenceClass(doc, results) {
    var diag = (results && results.diagnostics) || {};
    if (diag.frontier_valid === true) return __("mcf4.eq_validated", "Validated");
    var st = legStats(doc);
    if (st.scored > 0) return __("mcf4.eq_exploratory", "Exploratory");
    return __("mcf4.eq_notscoreable", "Not scoreable");
  }

  function archLabel(doc, id) {
    var a = (doc.architectures || []).filter(function (x) { return x.id === id; })[0];
    return a ? (a.label || id) : id;
  }

  function bandParams(doc) { return doc.parameters || []; }
  function bandByName(nameOrPath) {
    return state.bands.filter(function (b) { return b.name === nameOrPath || b.path === nameOrPath; })[0] || null;
  }
  function paramPath(nameOrPath) { var b = bandByName(nameOrPath); return b ? b.path : nameOrPath; }
  function paramLabelSafe(name) {
    var b = bandByName(name);
    if (b) return FRIENDLY_PARAMS[b.name] || b.name;
    return FRIENDLY_PARAMS[name] || name;
  }
  function paramValuesOf(doc, configId) {
    var parts = String(configId).split("/");
    var arch = parts[0], cid = parts[1] || parts[0];
    var leg = (doc.legs || []).filter(function (l) { return l.arch === arch && String(l.config_id) === String(cid); })[0];
    if (leg && leg.config_patch_flat) return leg.config_patch_flat;
    var cfg = (doc.configs || []).filter(function (c) { return String(c.config_id) === String(cid); })[0];
    return cfg ? (cfg.patch || {}) : null;
  }
  function settingText(doc, configId) {
    var patch = paramValuesOf(doc, configId);
    if (!patch) return "—";
    return bandParams(doc).map(function (p) {
      var v = patch[paramPath(p)];
      return paramLabelSafe(p) + " " + (v !== undefined ? v : "—");
    }).join(", ");
  }
  function archMechSummary(doc, arch) {
    var keys = Object.keys(arch.patch || {});
    if (keys.indexOf("personality.channels.rules") >= 0) {
      return arch.patch["personality.channels.rules"] === false
        ? __("mcf4.rules_disabled", "Personality rules disabled")
        : __("mcf4.rules_enabled", "Personality rules enabled");
    }
    if (!keys.length) return __("mcf4.mech_default", "Mechanism defaults");
    return keys.map(function (k) { return k + " = " + JSON.stringify(arch.patch[k]); }).join(", ");
  }

  function scoredConfigsFlat() {
    var doc = state.current, res = state.results;
    if (!res || !res.architectures) return [];
    var out = [];
    res.architectures.forEach(function (arch, ai) {
      (arch.configs || []).forEach(function (cfg) {
        var values = {}, spread = {};
        objectives().forEach(function (k) {
          if (typeof cfg[k] === "number") values[k] = cfg[k];
          if (typeof cfg[k + "_spread"] === "number") spread[k] = cfg[k + "_spread"];
        });
        out.push({ arch: arch.arch, archIndex: ai, config: cfg.config, values: values, spread: spread,
          seeds: cfg.seeds || [], complete: !!cfg.complete, missing_seeds: cfg.missing_seeds || [] });
      });
    });
    return out;
  }
  function objectives() { return (state.current && state.current.objectives) || ["paid_work_error", "leisure_social_error"]; }
  function objMeta(key) {
    var m = (state.meta && state.meta.objectives || []).filter(function (o) { return o.key === key; })[0];
    return m || { key: key, label: key, unit: "error units", lower_is_better: true, categories: [] };
  }
  function objLabel(key) { return objMeta(key).label; }
  function dominates(a, b, keys) {
    var less = false;
    for (var i = 0; i < keys.length; i++) {
      if (!(a[keys[i]] <= b[keys[i]])) return false;
      if (a[keys[i]] < b[keys[i]]) less = true;
    }
    return less;
  }
  function computeFrontier(configs, keys) {
    var full = configs.filter(function (c) { return c.complete; });
    var frontier = [];
    full.forEach(function (c) {
      var dom = full.some(function (o) { return o !== c && dominates(o.values, c.values, keys); });
      if (!dom) frontier.push(c.config);
    });
    var ref = (state.results && state.results.reference_architecture)
      || (state.current && state.current.reference_architecture) || null;
    var refCfgs = full.filter(function (c) { return c.arch === ref; });
    var gains = {};
    full.forEach(function (c) {
      if (c.arch === ref) return;
      if (!refCfgs.some(function (r) { return dominates(r.values, c.values, keys); })) gains[c.config] = true;
    });
    return { frontier: frontier, gains: gains, ref: ref, keys: keys };
  }

  /* matched configuration comparison: same parameter values, shared seeds */
  function matchedPairs() {
    var doc = state.current, res = state.results;
    if (!res || !res.architectures) return [];
    var vectors = res.vectors || [];
    var ref = (res.reference_architecture) || (doc.architectures && doc.architectures[0] && doc.architectures[0].id);
    var others = doc.architectures.filter(function (a) { return a.id !== ref; }).map(function (a) { return a.id; });
    var params = bandParams(doc);
    function paramsOfArch(archId) {
      var map = {};
      vectors.forEach(function (v) {
        var vArch = String(v.config || "").split("/")[0];
        if (vArch !== archId) return;
        var patch = paramValuesOf(doc, v.config) || {};
        var keyv = params.map(function (p) { return String(patch[paramPath(p)]); }).join("|");
        (map[keyv] = map[keyv] || { params: patch, seeds: {} });
        map[keyv].seeds[v.seed] = v;
      });
      return map;
    }
    var refMap = paramsOfArch(ref), pairs = [];
    others.forEach(function (oid) {
      var oMap = paramsOfArch(oid);
      Object.keys(refMap).forEach(function (keyv) {
        if (!oMap[keyv]) return;
        var shared = Object.keys(refMap[keyv].seeds).filter(function (s) { return oMap[keyv].seeds[s]; });
        if (!shared.length) return;
        var m0 = {}, m1 = {};
        objectives().forEach(function (k) {
          m0[k] = shared.reduce(function (acc, s) { return acc + (refMap[keyv].seeds[s][k] || 0); }, 0) / shared.length;
          m1[k] = shared.reduce(function (acc, s) { return acc + (oMap[keyv].seeds[s][k] || 0); }, 0) / shared.length;
        });
        pairs.push({ ref: ref, other: oid, params: refMap[keyv].params, seeds: shared.map(Number).sort(function (a, b) { return a - b; }), m0: m0, m1: m1 });
      });
    });
    pairs.sort(function (a, b) {
      var p0 = bandParams(doc)[0];
      return (a.params[p0] || 0) - (b.params[p0] || 0);
    });
    return pairs;
  }

  /* =========================== views =========================== */

  function showView(view) {
    state.view = view;
    el("mcf4Library").hidden = view !== "library";
    el("mcf4Ws").hidden = view !== "workspace";
    if (view === "library") { el("mcf4Wizard").hidden = state.wizardMode !== true; el("mcf4LibBody").hidden = state.wizardMode === true; }
  }

  /* ---------------------------- library ---------------------------- */

  function filterExperiments() {
    var q = state.search.toLowerCase();
    return state.experiments.filter(function (e) {
      if (q && String(e.name || "").toLowerCase().indexOf(q) < 0 && String(e.id).indexOf(q) < 0) return false;
      var st = e.status;
      if (state.filter === "draft") return st === "draft";
      if (state.filter === "running") return st === "running";
      if (state.filter === "results") {
        return (e.progress && e.progress.legs_done > 0) || (e.id === "phase5-final");
      }
      if (state.filter === "attention") {
        return st === "failed" || (e.progress && (e.progress.failures || []).length > 0);
      }
      return true;
    });
  }

  function sortNewest(list) {
    return list.slice().sort(function (a, b) {
      var ka = String(a.created_at || a.id), kb = String(b.created_at || b.id);
      return ka < kb ? 1 : -1;
    });
  }

  function renderLibrary() {
    var grid = el("mcf4Grid"), empty = el("mcf4Empty");
    var list = sortNewest(filterExperiments());
    if (!state.experiments.length) {
      grid.innerHTML = "";
      empty.textContent = __("mcf4.lib_empty", "No MCF experiments yet. Create an experiment to compare simulation designs.");
      return;
    }
    if (!list.length) { grid.innerHTML = ""; empty.textContent = __("mcf4.lib_filter_empty", "No experiments match this filter."); return; }
    empty.textContent = "";
    grid.innerHTML = list.map(function (e) {
      var archs = e.architectures || [];
      var params = bandParams(e);
      var st = legStats(e);
      var design = __f("mcf4.card_design", { a: archs.length, p: params.length, s: (e.seeds || []).length || e.n_seeds || 0 }, "{a} architectures · {p} parameter settings each · {s} seeds each");
      var mech = archs.length >= 2 ? esc(archMechSummary(e, archs[0])) + " vs " + esc(archMechSummary(e, archs[1])) : "—";
      var evidence;
      if (st.total) {
        evidence = __f("mcf4.card_evidence", { scored: st.scored, total: st.total, interrupted: st.interrupted }, "{scored} of {total} legs have partial scores · {interrupted} interrupted · not validated");
      } else {
        evidence = esc(__("mcf4.card_evidence_none", "No simulation runs yet"));
      }
      var qual = state.current && state.current.id === e.id && state.results
        ? evidenceClass(state.current, state.results)
        : (st.scored > 0 ? __("mcf4.eq_exploratory", "Exploratory") : (st.total ? __("mcf4.eq_notscoreable", "Not scoreable") : __("mcf4.eq_draft", "Draft")));
      return '<div class="mcf4-expcard" data-exp="' + esc(e.id) + '">'
        + "<h3>" + esc(e.name || e.id) + "</h3>"
        + '<div class="mcf4-meta">' + esc(design) + "</div>"
        + '<div class="mcf4-meta">' + evidence + " · <strong>" + esc(qual) + "</strong></div>"
        + '<div class="mcf4-meta">' + mech + "</div>"
        + '<div class="mcf4-cardfoot"><span class="mcf4-statuschip">' + esc(cardStatus(e)) + "</span>"
        + '<button class="button subtle" type="button" data-open="' + esc(e.id) + '">' + esc(__("mcf4.open_exp", "Open experiment")) + "</button></div>"
        + "</div>";
    }).join("");
    grid.querySelectorAll("[data-open]").forEach(function (btn) {
      btn.addEventListener("click", function () { openExperiment(btn.getAttribute("data-open")); });
    });
  }

  async function loadExperiments() {
    var body = await api("/api/mcf/experiments");
    state.experiments = body.experiments || [];
    renderLibrary();
  }

  /* ---------------------------- workspace ---------------------------- */

  function selectTab(tab) {
    document.querySelectorAll("#mcf4Tabs button").forEach(function (b) { b.classList.toggle("is-active", b.getAttribute("data-tab") === tab); });
    document.querySelectorAll("#mcf4Ws .mcf4-wstab").forEach(function (p) { p.hidden = p.getAttribute("data-tab") !== tab; });
  }

  async function openExperiment(id, tab) {
    var doc = await api("/api/mcf/experiment/" + id);
    state.current = doc; state.results = null; state.message = ""; state.expanded = {}; state.confirmOverwrite = false;
    showView("workspace");
    el("mcf4WsTitle").textContent = doc.name || doc.id;
    renderMetaLine(); renderEvidence(); renderWorkspaceAll();
    selectTab(tab || "overview");
    if (doc.status !== "draft") {
      try {
        var body = await api("/api/mcf/experiment/" + id + "/results");
        state.results = body.results; state.message = body.message || "";
      } catch (e) { state.message = e.message; }
      renderWorkspaceAll();
    }
    if (doc.status === "running") startPolling();
  }

  function renderMetaLine() {
    var doc = state.current;
    var archs = doc.architectures || [];
    var params = bandParams(doc).map(function (p) { return paramLabelSafe(p); }).join(" / ");
    var paramVals = doc.configs && doc.configs.length ? doc.configs.map(function (c) {
      return bandParams(doc).map(function (p) { return c.patch ? c.patch[p] : "—"; }).join("/");
    }).filter(function (v, i, a) { return a.indexOf(v) === i; }).join(" / ") : "—";
    var seeds = (doc.seeds || []).join(", ") || (doc.n_seeds ? String(doc.n_seeds) : "—");
    var archBits = archs.map(function (a, i) {
      var short = archMechSummary(doc, a);
      return esc((a.label || a.id) + (i === 0 ? " (M0)" : " (M1)") + " · " + short);
    }).join(" vs ");
    var target = state.meta && state.meta.targets ? state.meta.targets.dataset + " " + state.meta.targets.year : "";
    el("mcf4MetaLine").innerHTML =
      esc(__f("mcf4.meta_design", { a: archBits, p: params + " " + paramVals, s: seeds }, "{a} · {p} · seeds {s}"))
      + (target ? " · " + esc(__f("mcf4.meta_targets", { t: target }, "empirical targets: {t}")) : "");
  }

  function gateFailures() {
    var doc = state.current;
    var gates = (state.meta && state.meta.gates) || { min_comparison_days: 7, min_comparison_population: 20 };
    var st = legStats(doc);
    var items = [];
    var maxDays = st.agentDays.length ? Math.max.apply(null, st.agentDays) : null;
    if (maxDays !== null && maxDays < gates.min_comparison_days) {
      items.push(__f("mcf4.gate_days", { obs: fmt(maxDays, 0), req: gates.min_comparison_days }, "Observed simulation window {obs} days — below the required {req} days"));
    }
    var minAgents = null;
    (state.results && state.results.vectors || []).forEach(function (v) {
      var m = /cohort of (\d+) agent/.exec(v.reason || "");
      if (m) { var n = Number(m[1]); minAgents = minAgents === null ? n : Math.min(minAgents, n); }
    });
    if (minAgents !== null && minAgents < gates.min_comparison_population) {
      items.push(__f("mcf4.gate_agents", { obs: minAgents, req: gates.min_comparison_population }, "Simulated cohort {obs} agents — below the required {req}"));
    }
    if (st.interrupted) {
      items.push(__f("mcf4.gate_interrupted", { n: st.interrupted, total: st.total }, "{n} of {total} legs stopped before their planned duration"));
    }
    var cov = coverageSummary();
    if (cov) items.push(__f("mcf4.gate_coverage", { lo: fmt(cov.min * 100, 0), hi: fmt(cov.max * 100, 0) }, "Score coverage only {lo}–{hi}% of the observation window"));
    if ((doc.seeds || []).length < 2 && (doc.n_seeds || 0) < 2) items.push(esc(__("mcf4.gate_seeds", "Only one seed per configuration — no stability check")));
    return items;
  }

  function coverageSummary() {
    var st = legStats(state.current);
    if (!st.coverages.length) return null;
    return { min: Math.min.apply(null, st.coverages), max: Math.max.apply(null, st.coverages),
      mean: st.coverages.reduce(function (a, b) { return a + b; }, 0) / st.coverages.length };
  }

  function renderEvidence() {
    var doc = state.current;
    var host = el("mcf4Evidence");
    var st = legStats(doc);
    if (!st.scored) {
      host.innerHTML = '<div class="mcf4-banner mcf4-banner-info">' + esc(__("mcf4.evidence_none", "No comparison is available: no simulation leg has produced scoreable results yet.")) + "</div>";
      return;
    }
    var summary = __("mcf4.evidence_exploratory", "Exploratory results only — all {n} simulation legs stopped before their planned duration. Their partial scores can be compared descriptively, but this experiment does not meet the requirements for a validated frontier.")
      .replace("{n}", st.total);
    var items = gateFailures();
    var listHtml = items.length ? items.map(function (i) { return "<li>" + (typeof i === "string" && i.indexOf("<") === 0 ? i : esc(i)) + "</li>"; }).join("") : "";
    host.innerHTML = '<div class="mcf4-banner mcf4-banner-warn mcf4-evidencebox">'
      + "<div class='mcf4-evidencehead'>" + esc(summary)
      + ' <button class="mcf4-linkish" type="button" id="mcf4WhyExp" aria-expanded="false">' + esc(__("mcf4.why_exploratory", "Why these results are exploratory")) + "</button></div>"
      + '<ul class="mcf4-evidencedetail" hidden>' + listHtml + "</ul></div>";
    el("mcf4WhyExp").addEventListener("click", function () {
      var ul = host.querySelector(".mcf4-evidencedetail");
      ul.hidden = !ul.hidden;
      this.setAttribute("aria-expanded", String(!ul.hidden));
    });
  }

  function renderWorkspaceAll() {
    renderOverview(); renderChartTab(); renderMatched(); renderAllSettings();
    renderDesign(); renderRunsPane(); renderDiagPane();
  }

  /* ---------------------------- overview ---------------------------- */

  function renderOverview() {
    var doc = state.current;
    var host = el("mcf4Overview");
    if (!doc) { host.innerHTML = ""; return; }
    var st = legStats(doc);
    var archs = doc.architectures || [];
    var ref = (state.results && state.results.reference_architecture) || doc.reference_architecture || (archs[0] && archs[0].id);
    var others = archs.filter(function (a) { return a.id !== ref; });
    var designBits = archs.map(function (a, i) {
      var mech = archMechSummary(doc, a);
      var isRef = a.id === ref;
      return '<div class="mcf4-archline"><strong>' + esc("M" + i + " " + (a.label || a.id)) + (isRef ? " · " + esc(__("mcf4.reference_word", "reference")) : "") + "</strong><div class=\"mcf4-meta\">" + esc(mech) + "</div></div>";
    }).join("");
    var mechDesc = archs.length >= 2 && archs.some(function (a) { return (a.patch || {})["personality.channels.rules"] !== undefined; })
      ? esc(__("mcf4.mech_desc_rules", "The difference is whether personality-rule influences are applied in agent behavior. All parameter settings shown were explored within each design."))
      : esc(__("mcf4.mech_desc_custom", "Custom architecture change — the exact difference is listed above; no standard behavioural explanation is available for it."));

    var score0 = st.scored ? (doc.legs || []).filter(function (l) { return l.score; })[0].score : null;
    var cohort = null;
    (state.results && state.results.vectors || []).forEach(function (v) {
      var m = /cohort of (\d+) agent/.exec(v.reason || "");
      if (m) cohort = Math.min(Number(cohort === null ? 999 : cohort), Number(m[1]));
    });
    var cov = coverageSummary();
    var evidenceCells = [
      [__("mcf4.ev_city", "City"), notRecorded()],
      [__("mcf4.ev_agents", "Agents (simulated cohort)"), cohort !== null ? esc(String(cohort)) + " (" + esc(__("mcf4.cohort_basis", "from gate diagnostics")) + ")" : (score0 && score0.per_category.paid_work ? esc(String(score0.per_category.paid_work.n_agent_days)) + " " + esc(__("mcf4.days_word", "days")) : notRecorded())],
      [__("mcf4.ev_days_req", "Requested days"), notRecorded()],
      [__("mcf4.ev_days_obs", "Executed observation window"), st.agentDays.length ? esc(fmt(Math.max.apply(null, st.agentDays), 0) + " " + __("mcf4.days_word", "days")) : notRecorded()],
      [__("mcf4.ev_seeds", "Seeds"), esc((doc.seeds || []).join(", ") || (doc.n_seeds ? String(doc.n_seeds) : "—"))],
      [__("mcf4.ev_targets", "Target dataset"), state.meta ? esc(state.meta.targets.dataset + " (" + state.meta.targets.year + ")") : notRecorded()],
      [__("mcf4.ev_metrics", "Metrics"), esc(objectives().map(function (k) { return objLabel(k); }).join(" · "))],
    ].map(function (c) { return "<tr><th>" + c[0] + "</th><td>" + c[1] + "</td></tr>"; }).join("");

    var pairs = matchedPairs();
    var findingHtml;
    if (pairs.length) {
      var workDeltas = pairs.map(function (p) { return p.m1.paid_work_error - p.m0.paid_work_error; });
      var leiDeltas = pairs.map(function (p) { return p.m1.leisure_social_error - p.m0.leisure_social_error; });
      var workAllNeg = workDeltas.every(function (d) { return d < 0; });
      var leiAllPos = leiDeltas.every(function (d) { return d > 0; });
      function rangeText(arr, objLabelTxt) {
        var lo = Math.min.apply(null, arr.map(Math.abs)), hi = Math.max.apply(null, arr.map(Math.abs));
        var dir = arr.every(function (d) { return d < 0; }) ? __("mcf4.dir_lower", "lower") : arr.every(function (d) { return d > 0; }) ? __("mcf4.dir_higher", "higher") : __("mcf4.dir_mixed", "in different directions");
        var mag = (fmt(lo, 0) === fmt(hi, 0))
          ? __("mcf4.range_single", "{obj} error was {dir} (by about {lo} error units)").replace("{lo}", fmt(lo, 0))
          : __f("mcf4.range_text", { obj: objLabelTxt, dir: dir, lo: fmt(lo, 0), hi: fmt(hi, 0) }, "{obj} error was {dir} (by about {lo}–{hi} error units)");
        return mag;
      }
      var parts = [];
      if (workAllNeg || workDeltas.every(function (d) { return d > 0; })) parts.push(rangeText(workDeltas, __("mcf4.obj_work", "Working-time")));
      if (leiAllPos || leiDeltas.every(function (d) { return d < 0; })) parts.push(rangeText(leiDeltas, __("mcf4.obj_leisure", "Leisure and social")));
      var neither = __("mcf4.neither", "Neither design improves both outcomes in the matched comparison.");
      findingHtml = "<p>" + esc(parts.length ? parts.join(", while ") + ". " + neither : neither) + "</p>";
      findingHtml += '<p class="mcf4-explain">' + esc(__("mcf4.finding_caveat", "This pattern comes from incomplete, low-coverage simulations and should not be interpreted as a reliable difference between the designs.")) + "</p>";
    } else {
      findingHtml = "<p>" + esc(__("mcf4.no_match", "A matched comparison is not available because no configuration pair shares the same parameter setting and seed set.")) + "</p>";
    }

    var tradeRows = "";
    if (pairs.length) {
      var workDeltas = pairs.map(function (p) { return p.m1.paid_work_error - p.m0.paid_work_error; });
      var leiDeltas = pairs.map(function (p) { return p.m1.leisure_social_error - p.m0.leisure_social_error; });
      function tradeRow(objLabelTxt, deltas) {
        var allNeg = deltas.every(function (d) { return d < 0; });
        var allPos = deltas.every(function (d) { return d > 0; });
        var verdict = allNeg ? __f("mcf4.trade_ref_closer", { d: "M0" }, "rules on (reference design) closer to observations")
          : allPos ? __("mcf4.trade_cmp_closer", "rules off (comparison design) closer to observations")
          : __("mcf4.trade_mixed", "mixed — no single design closer");
        var lo = Math.min.apply(null, deltas.map(Math.abs)), hi = Math.max.apply(null, deltas.map(Math.abs));
        return "<tr><th>" + esc(objLabelTxt) + "</th><td>" + esc(verdict) + '</td><td class="mcf4-num">' + esc(fmt(lo, 0) + "–" + fmt(hi, 0)) + "</td></tr>";
      }
      tradeRows = tradeRow(__("mcf4.obj_work", "Working-time"), workDeltas) + tradeRow(__("mcf4.obj_leisure", "Leisure and social"), leiDeltas);
    } else {
      tradeRows = '<tr><th colspan="3">' + esc(__("mcf4.not_evaluated", "not evaluated")) + "</th></tr>";
    }

    var noiseLine = "";
    var noise = (state.results && state.results.diagnostics && state.results.diagnostics.noise_domination) || {};
    if (noise.paid_work_error) noiseLine = esc(__("mcf4.noise_paid", "current noise diagnostic flags paid-work variation as too large for a stable configuration ranking"));
    else if (noise.leisure_social_error) noiseLine = esc(__("mcf4.noise_leisure", "current noise diagnostic flags leisure/social variation as too large for a stable configuration ranking"));
    else noiseLine = esc(__("mcf4.noise_none", "no seed-variability warning was recorded"));

    host.innerHTML = ""
      + '<div class="mcf4-card"><h2>' + esc(__("mcf4.ov_tested", "What was tested?")) + "</h2>"
      + '<div class="mcf4-twocol"><div><h3>' + esc(__("mcf4.ov_designs", "Simulation designs (architectures)")) + "</h3>" + designBits
      + '<p class="mcf4-explain">' + mechDesc + "</p></div>"
      + '<div><h3>' + esc(__("mcf4.ov_evidence", "Evidence")) + '</h3><table class="mcf4-table">' + evidenceCells + "</table></div></div></div>"
      + '<div class="mcf4-card"><h2>' + esc(__("mcf4.ov_observed", "What did we observe?")) + "</h2>" + findingHtml + "</div>"
      + '<div class="mcf4-card"><h2>' + esc(__("mcf4.ov_tradeoff", "What is the trade-off?")) + "</h2>"
      + '<p class="mcf4-explain">' + esc(__("mcf4.lower_better", "Lower error = better fit to the observed data.")) + "</p>"
      + '<table class="mcf4-table">' + tradeRows + "</table>"
      + ' <button class="mcf4-linkish" type="button" data-goto="results">' + esc(__("mcf4.see_frontier", "See all configurations and the frontier")) + "</button></div>"
      + '<div class="mcf4-card"><h2>' + esc(__("mcf4.ov_strength", "How strong is the evidence?")) + "</h2>"
      + '<table class="mcf4-table">'
      + "<tr><th>" + esc(__("mcf4.ev_execution", "Execution")) + "</th><td>" + esc(__f("mcf4.execution_line", { i: st.interrupted, c: st.completed }, "{i} interrupted / {c} completed")) + "</td></tr>"
      + "<tr><th>" + esc(__("mcf4.ev_coverage", "Coverage")) + "</th><td>" + (cov ? esc(__f("mcf4.coverage_line", { lo: fmt(cov.min * 100, 0), hi: fmt(cov.max * 100, 0), mean: fmt(cov.mean * 100, 0) }, "{lo}–{hi}% observed (mean {mean}%) of the observation window")) + " · " + esc(__f("mcf4.coverage_req", { d: (state.meta && state.meta.gates.min_comparison_days) || 7 }, "at least {d} days required")) : notRecorded()) + "</td></tr>"
      + "<tr><th>" + esc(__("mcf4.ev_precision", "Precision")) + "</th><td>" + esc(__f("mcf4.precision_seeds", { n: (doc.seeds || []).length || doc.n_seeds || 1 }, "{n} seeds per configuration")) + (noiseLine ? " · " + noiseLine : "") + "</td></tr>"
      + "</table>"
      + '<p class="mcf4-explain">' + esc(__("mcf4.ov_conclusion", "Useful for testing the workflow and generating hypotheses; not sufficient for a scientific capacity claim.")) + "</p>"
      + ' <button class="mcf4-linkish" type="button" data-goto="diag">' + esc(__("mcf4.view_diag", "View diagnostics")) + "</button></div>";

    host.querySelectorAll("[data-goto]").forEach(function (btn) {
      btn.addEventListener("click", function () { selectTab(btn.getAttribute("data-goto")); });
    });
  }

  /* ---------------------------- results: chart ---------------------------- */

  function renderChartTab() {
    var host = el("mcf4Chart");
    var legend = el("mcf4Legend");
    var keys = objectives();
    var configs = scoredConfigsFlat();
    var numericKeys = keys.filter(function (k) { return configs.some(function (c) { return typeof c.values[k] === "number"; }); });
    var pickers = el("mcf4ObjPickers");
    var allNote = el("mcf4AllObjNote");
    if (numericKeys.length < 2) {
      pickers.hidden = true; allNote.hidden = true;
      host.innerHTML = '<p class="rw-hint">' + esc(__("mcf4.chart_unavailable", "The chart needs at least two numeric objectives; this experiment has fewer, so no trade-off plot can be drawn from real data.")) + "</p>";
      legend.innerHTML = "";
      return;
    }
    if (!state.xKey || keys.indexOf(state.xKey) < 0) state.xKey = numericKeys[0];
    if (!state.yKey || keys.indexOf(state.yKey) < 0) state.yKey = numericKeys[1];
    if (keys.length > 2) {
      pickers.hidden = false;
      ["mcf4XSel", "mcf4YSel"].forEach(function (id, idx) {
        var sel = el(id);
        var cur = idx === 0 ? state.xKey : state.yKey;
        sel.innerHTML = keys.map(function (k) { return '<option value="' + esc(k) + '"' + (k === cur ? " selected" : "") + ">" + esc(objLabel(k)) + "</option>"; }).join("");
        sel.onchange = function () {
          if (idx === 0) state.xKey = sel.value; else state.yKey = sel.value;
          renderChartTab();
        };
      });
      allNote.hidden = false;
      allNote.textContent = __("mcf4.allobj_note", "Frontier calculations use all measured objectives; this chart displays only two.").replace("all", "all " + keys.length);
    } else { pickers.hidden = true; allNote.hidden = true; }

    var kx = state.xKey, ky = state.yKey;
    var fr = computeFrontier(configs, keys);
    var doc = state.current;
    var legs = doc.legs || [];
    var allCompleted = legs.length > 0 && legs.every(function (l) { return l.status === "completed"; });

    var W = 760, H = 400, padL = 70, padB = 50, padT = 20, padR = 22;
    var narrow = window.innerWidth < 760;
    if (narrow) { H = 320; padL = 56; }
    var xs = configs.map(function (c) { return c.values[kx]; }).filter(function (v) { return typeof v === "number"; });
    var ys = configs.map(function (c) { return c.values[ky]; }).filter(function (v) { return typeof v === "number"; });
    if (!xs.length || !ys.length) { host.innerHTML = ""; legend.innerHTML = ""; return; }
    var xLo = Math.min.apply(null, xs), xHi = Math.max.apply(null, xs);
    var yLo = Math.min.apply(null, ys), yHi = Math.max.apply(null, ys);
    var xSpan = (xHi - xLo) || Math.abs(xHi) * 0.1 || 1, ySpan = (yHi - yLo) || Math.abs(yHi) * 0.1 || 1;
    xLo -= xSpan * 0.1; xHi += xSpan * 0.1; yLo -= ySpan * 0.1; yHi += ySpan * 0.1;
    function px(v) { return padL + (v - xLo) / (xHi - xLo) * (W - padL - padR); }
    function py(v) { return H - padB - (v - yLo) / (yHi - yLo) * (H - padT - padB); }
    function ticks(lo, hi, n) {
      if (!(hi > lo)) hi = lo + 1;
      var step = Math.pow(10, Math.floor(Math.log((hi - lo) / n) / Math.LN10));
      var err = ((hi - lo) / n) / step;
      if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
      var t = [], v = Math.ceil(lo / step) * step;
      while (v <= hi + 1e-9) { t.push(v); v += step; }
      return t;
    }

    var seedTextFor = function (c) {
      var legsFor = legs.filter(function (l) { return c.config.indexOf(l.arch + "/") === 0 && c.seeds.indexOf(l.seed) >= 0; });
      var anyInterrupted = legsFor.some(function (l) { return l.status !== "completed"; });
      return allCompleted ? __("mcf4.tt_all_finished", "all legs finished") : __("mcf4.tt_partial", "partial scored runs") + (legsFor.some(function (l) { return l.status === "interrupted"; }) ? " · " + __("mcf4.tt_interrupted", "interrupted") : "");
    };

    var out = ['<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(__("mcf4.chart_aria", "Scatter plot of working-time error versus leisure and social time error for each explored parameter setting")) + '">'];
    ticks(xLo, xHi, 6).forEach(function (t) {
      out.push('<line class="mcf4-grid" x1="' + px(t) + '" y1="' + padT + '" x2="' + px(t) + '" y2="' + (H - padB) + '"/>');
      out.push('<text class="mcf4-axis" x="' + px(t) + '" y="' + (H - padB + 20) + '" text-anchor="middle">' + esc(fmt(t, 0)) + "</text>");
    });
    ticks(yLo, yHi, 5).forEach(function (t) {
      out.push('<line class="mcf4-grid" x1="' + padL + '" y1="' + py(t) + '" x2="' + (W - padR) + '" y2="' + py(t) + '"/>');
      out.push('<text class="mcf4-axis" x="' + (padL - 10) + '" y="' + (py(t) + 4) + '" text-anchor="end">' + esc(fmt(t, 0)) + "</text>");
    });
    var kxM = objMeta(kx), kyM = objMeta(ky);
    out.push('<text class="mcf4-axis" x="' + ((padL + W - padR) / 2) + '" y="' + (H - 10) + '" text-anchor="middle" font-weight="600">' + esc(kxM.label + " (" + (kxM.unit || "error units") + ", ↓)") + "</text>");
    out.push('<text class="mcf4-axis" transform="translate(18,' + ((padT + H - padB) / 2) + ') rotate(-90)" text-anchor="middle" font-weight="600">' + esc(kyM.label + " (" + (kyM.unit || "error units") + ", ↓)") + "</text>");

    var legendKeys = {};
    configs.forEach(function (c) {
      var vx = c.values[kx], vy = c.values[ky];
      if (typeof vx !== "number" || typeof vy !== "number") return;
      var color = ARCH_COLORS[c.archIndex % ARCH_COLORS.length];
      var cx = px(vx), cy = py(vy);
      var onFrontier = fr.frontier.indexOf(c.config) >= 0;
      var isGain = fr.gains[c.config];
      var setting = settingText(doc, c.config);
      var tip = [archLabel(doc, c.arch), setting,
        kx + " = " + fmt(c.values[kx]) + ", " + ky + " = " + fmt(c.values[ky]),
        __("mcf.th_seeds", "种子") + ": " + (c.seeds || []).join(", "),
        seedTextFor(c)].join("\n");
      var filled = allCompleted;
      out.push('<circle cx="' + cx + '" cy="' + cy + '" r="7" fill="' + (filled ? color : "none") + '" stroke="' + color + '" stroke-width="2.5"><title>' + esc(tip) + "</title></circle>");
      legendKeys[archLabel(doc, c.arch)] = color;
      if (onFrontier) {
        out.push('<rect x="' + (cx - 6) + '" y="' + (cy - 6) + '" width="12" height="12" transform="rotate(45 ' + cx + " " + cy + ')" fill="none" stroke="' + (allCompleted ? "#333" : "#8a7a55") + '" stroke-width="1.6"><title>' + esc(__("mcf4.tt_frontier", "Explored frontier (non-dominated across all objectives)") + "\n" + tip) + "</title></rect>");
      }
      if (isGain) {
        out.push('<circle cx="' + cx + '" cy="' + cy + '" r="12" fill="none" stroke="#666" stroke-width="1.6" stroke-dasharray="3 3"><title>' + esc((allCompleted ? __("mcf4.tt_gain_valid", "Newly attainable") : __("mcf4.tt_gain_expl", "Newly attainable among these explored points (exploratory)")) + "\n" + tip) + "</title></circle>");
      }
    });
    out.push("</svg>");
    host.innerHTML = out.join("");

    var lg = [];
    Object.keys(legendKeys).forEach(function (label) {
      lg.push('<span class="mcf4-key"><span class="mcf4-swatch" style="background:' + legendKeys[label] + '"></span>' + esc(label) + "</span>");
    });
    if (!allCompleted) lg.push('<span class="mcf4-key"><span class="mcf4-swatch" style="border:2px solid #555;background:none"></span>' + esc(__("mcf4.legend_partial", "Partial simulation")) + "</span>");
    if (fr.frontier.length) lg.push('<span class="mcf4-key"><span class="mcf4-swatch mcf4-diamond"></span>' + esc(allCompleted ? __("mcf4.legend_frontier_valid", "Validated frontier") : __("mcf4.legend_frontier", "Explored frontier")) + "</span>");
    if (Object.keys(fr.gains).length) lg.push('<span class="mcf4-key"><span class="mcf4-swatch mcf4-ring"></span>' + esc(allCompleted ? __("mcf4.legend_gain", "Newly attainable") : __("mcf4.legend_gain_expl", "Newly attainable among explored points")) + "</span>");
    legend.innerHTML = lg.join("");
  }

  /* ---------------------------- results: matched + all settings ---------------------------- */

  function renderMatched() {
    var host = el("mcf4Matched");
    var doc = state.current;
    var pairs = matchedPairs();
    if (!pairs.length) {
      host.innerHTML = '<p class="rw-hint">' + esc(__("mcf4.no_match", "A matched comparison is not available because no configuration pair shares the same parameter setting and seed set.")) + "</p>";
      return;
    }
    var keys = objectives();
    var head = "<tr><th>" + esc(__("mcf4.param_setting", "Parameter setting")) + "</th>";
    head += "<th>" + esc(objLabel(keys[0])) + ": " + esc(archLabel(doc, pairs[0].ref)) + "</th><th>" + esc(objLabel(keys[0])) + ": " + esc(archLabel(doc, pairs[0].other)) + "</th><th>" + esc(__("mcf4.change", "Change")) + "</th>";
    if (keys[1]) head += "<th>" + esc(objLabel(keys[1])) + ": " + esc(archLabel(doc, pairs[0].ref)) + "</th><th>" + esc(objLabel(keys[1])) + ": " + esc(archLabel(doc, pairs[0].other)) + "</th><th>" + esc(__("mcf4.change", "Change")) + "</th>";
    head += "</tr>";
    var rows = pairs.map(function (p) {
      var setting = bandParams(doc).map(function (bp) { return paramLabelSafe(bp) + " " + (p.params[paramPath(bp)] !== undefined ? p.params[paramPath(bp)] : "—"); }).join(", ");
      var seedsOk = p.seeds.length && p.seeds.length === (doc.seeds || []).length;
      function cell(v) { return '<td class="mcf4-num" title="' + esc(String(v)) + '">' + esc(fmt(v)) + "</td>"; }
      function delta(v) {
        var cls = v < 0 ? "mcf4-delta-neg" : v > 0 ? "mcf4-delta-pos" : "";
        return '<td class="mcf4-num ' + cls + '" title="' + esc(String(v)) + '">' + esc((v > 0 ? "+" : "") + fmt(v)) + "</td>";
      }
      var row = "<tr><td>" + esc(setting) + (seedsOk ? "" : ' <span class="mcf4-chip mcf4-chip-warn">' + esc(__("mcf4.partial_seeds", "matched seeds differ")) + "</span>") + "</td>";
      row += cell(p.m0[keys[0]]) + cell(p.m1[keys[0]]) + delta(p.m1[keys[0]] - p.m0[keys[0]]);
      if (keys[1]) row += cell(p.m0[keys[1]]) + cell(p.m1[keys[1]]) + delta(p.m1[keys[1]] - p.m0[keys[1]]);
      return row + "</tr>";
    }).join("");
    host.innerHTML = '<table class="mcf4-table"><thead>' + head + "</thead><tbody>" + rows + "</tbody></table>";
  }

  function renderAllSettings() {
    var host = el("mcf4AllSettings");
    var doc = state.current;
    var configs = scoredConfigsFlat();
    var keys = objectives();
    if (!configs.length) { host.innerHTML = '<p class="rw-hint">' + esc(__("mcf4.no_settings", "No scored parameter settings recorded.")) + "</p>"; return; }
    var legs = doc.legs || [];
    configs.sort(function (a, b) { return a.archIndex - b.archIndex || (keys.length && (a.values[keys[0]] || 0) - (b.values[keys[0]] || 0)); });
    var rows = configs.map(function (c) {
      var cfgLegs = legs.filter(function (l) { return c.config === l.arch + "/" + l.config_id; });
      var anyInterrupted = cfgLegs.some(function (l) { return l.status !== "completed"; });
      var evidence = cfgLegs.length
        ? (anyInterrupted ? esc(__("mcf4.ev_partial_all", "Partial scores — all legs interrupted")) : esc(__("mcf4.ev_complete", "All legs completed")))
        : esc(__("mcf4.ev_partial", "Partial scores"));
      var open = !!state.expanded[c.config];
      var valCells = keys.map(function (k) { return '<td class="mcf4-num">' + esc(fmt(c.values[k])) + (typeof c.spread[k] === "number" ? ' <span class="mcf4-meta">±' + esc(fmt(c.spread[k])) + "</span>" : "") + "</td>"; }).join("");
      var row = "<tr><td>" + esc(archLabel(doc, c.arch)) + "</td><td>" + esc(settingText(doc, c.config)) + "</td><td class='mcf4-num'>" + esc((c.seeds || []).join(", ") || "—") + "</td>" + valCells + "<td>" + evidence + '</td><td><button class="mcf4-linkish" data-toggle="' + esc(c.config) + '">' + esc(open ? __("mcf4.hide", "Hide") : __("mcf4.details", "Details")) + "</button></td></tr>";
      if (open) {
        var vectors = (state.results && state.results.vectors || []).filter(function (v) { return v.config === c.config; });
        var det = vectors.map(function (v) {
          return esc(__("mcf4.seed_row", "Seed {seed}: {status} · {vals}")
            .replace("{seed}", v.seed).replace("{status}", v.leg_status || "—")
            .replace("{vals}", keys.map(function (k) { return k + " = " + fmt(v[k]); }).join(", ")))
            + (v.excluded ? ' <span class="mcf4-meta">' + esc(v.reason || "") + "</span>" : "");
        }).join("<br>");
        var legDet = cfgLegs.map(function (l) {
          return esc(__("mcf4.leg_row", "Seed {seed}: {status}{stop}")
            .replace("{seed}", l.seed).replace("{status}", l.status)
            .replace("{stop}", l.stop_reason ? " · " + l.stop_reason : ""));
        }).join("<br>");
        var patch = paramValuesOf(doc, c.config);
        row += '<tr class="mcf4-expand"><td colspan="' + (5 + keys.length) + '"><div class="mcf4-metablock">'
          + "<div><strong>config:</strong> <code>" + esc(c.config) + "</code></div>"
          + "<div><strong>" + esc(__("mcf4.patch_col", "Configuration patch")) + ":</strong> <code>" + esc(JSON.stringify(patch || {})) + "</code></div>"
          + "<div>" + det + "</div><div>" + legDet + "</div>"
          + "</div></td></tr>";
      }
      return row;
    }).join("");
    var head = "<tr><th>" + esc(__("mcf4.design_col", "Design")) + "</th><th>" + esc(__("mcf4.param_values", "Parameter values")) + "</th><th>" + esc(__("mcf4.seeds_evaluated", "Seeds evaluated")) + "</th>" + keys.map(function (k) { return "<th>" + esc(objLabel(k)) + "</th>"; }).join("") + "<th>" + esc(__("mcf4.run_evidence", "Run evidence")) + "</th><th></th></tr>";
    host.innerHTML = '<div class="mcf4-scrollx"><table class="mcf4-table"><thead>' + head + "</thead><tbody>" + rows + "</tbody></table></div>";
    host.querySelectorAll("[data-toggle]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var k = btn.getAttribute("data-toggle");
        state.expanded[k] = !state.expanded[k];
        renderAllSettings();
      });
    });
  }

  /* ---------------------------- design tab ---------------------------- */

  function renderDesign() {
    var doc = state.current;
    var host = el("mcf4Design");
    if (!doc) { host.innerHTML = ""; return; }
    var legs = doc.legs || [];
    var agentDays = null;
    legs.forEach(function (l) {
      if (l.score && l.score.per_category) {
        Object.keys(l.score.per_category).forEach(function (k) {
          var n = l.score.per_category[k].n_agent_days;
          if (typeof n === "number") agentDays = agentDays === null ? n : Math.max(agentDays, n);
        });
      }
    });
    var archs = (doc.architectures || []).map(function (a, i) {
      var patch = a.patch || {};
      return '<div class="mcf4-archcard' + (i === 0 ? " is-ref" : "") + '"><h3>' + esc("M" + i + " " + (a.label || a.id)) + (i === 0 ? " · " + esc(__("mcf4.reference_word", "reference")) : "") + "</h3>"
        + '<div class="mcf4-meta">' + esc(archMechSummary(doc, a)) + "</div>"
        + '<details class="mcf4-advanced"><summary>' + esc(__("mcf4.adv_patch", "Advanced — custom mechanism overrides (JSON)")) + "</summary><pre>" + esc(JSON.stringify(patch, null, 2)) + "</pre></details></div>";
    }).join("");
    var bands = state.bands.filter(function (b) { return bandParams(doc).indexOf(b.name) >= 0; });
    var bandRows = bandParams(doc).map(function (p) {
      var b = bands.filter(function (x) { return x.name === p; })[0];
      var friendly = paramLabelSafe(p);
      return "<tr><td>" + esc(friendly) + '</td><td class="mcf4-num">' + (b ? esc(fmt(b.low)) + " – " + esc(fmt(b.high)) : "—") + "</td><td>" + (b && b.sources && b.sources.length ? esc(b.sources.join(" ")) : notRecorded()) + "</td><td>" + esc(doc.sampler || "—") + "</td></tr>";
    }).join("");
    var model = doc.model && (doc.model.provider || doc.model.model) ? esc((doc.model.provider || "") + " · " + (doc.model.model || "")) : notRecorded();
    host.innerHTML = ""
      + '<div class="mcf4-card"><h2>' + esc(__("mcf4.d_pop", "Population and simulation")) + '</h2><table class="mcf4-table">'
      + "<tr><th>" + esc(__("mcf4.ev_city", "City")) + "</th><td>" + (doc.population ? esc((doc.population.city_name || doc.population.city) + " (" + doc.population.city + ")") : notRecorded()) + "</td></tr>"
      + "<tr><th>" + esc(__("mcf4.d_pop_source", "Population source")) + "</th><td>" + (doc.population ? esc(__f("mcf5.pop_source", { method: doc.population.method, src: doc.population.population_source || "" }, "{method} · roster: {src}")) : notRecorded()) + "</td></tr>"
      + "<tr><th>" + esc(__("mcf4.d_agent_ids", "Agent IDs / count")) + "</th><td>" + (doc.population ? esc(__f("mcf5.agent_ids_line", { n: doc.population.n_agents, ids: (doc.population.agent_ids || []).slice(0, 30).join(", ") + ((doc.population.agent_ids || []).length > 30 ? "…" : "") }, "{n} residents: {ids}")) : notRecorded()) + "</td></tr>"
      + "<tr><th>" + esc(__("mcf4.d_employed", "Employed-agent basis")) + "</th><td>" + (doc.population ? esc(__f("mcf5.employed_line", { x: doc.population.n_employed, y: doc.population.n_agents }, "{x} of {y} selected residents are classified as employed (paid-work scoring covers employed residents only).")) + (doc.population.sample_seed !== null && doc.population.sample_seed !== undefined ? " · " + esc(__("mcf5.rv_seed_word", "sampling seed")) + " " + doc.population.sample_seed : "") : esc(__("mcf4.d_employed_val", "paid_work scored on participants basis (employed agents only)"))) + "</td></tr>"
      + "<tr><th>" + esc(__("mcf5.rv_days", "Simulation duration")) + "</th><td>" + (doc.sim_days ? esc(String(doc.sim_days)) + " " + esc(__("mcf4.days_word", "days")) : notRecorded()) + "</td></tr>"
      + "<tr><th>" + esc(__("mcf4.ev_days_req", "Requested days")) + "</th><td>" + notRecorded() + "</td></tr>"
      + "<tr><th>" + esc(__("mcf4.d_tick", "Tick interval")) + "</th><td>" + notRecorded() + "</td></tr>"
      + "<tr><th>" + esc(__("mcf4.ev_days_obs", "Executed days")) + "</th><td>" + (agentDays !== null ? esc(String(agentDays)) + " " + esc(__("mcf4.days_word", "days")) + " (" + esc(__("mcf4.d_max_obs", "maximum across legs")) + ")" : notRecorded()) + "</td></tr>"
      + "<tr><th>" + esc(__("mcf4.model_col", "Model")) + "</th><td>" + model + "</td></tr>"
      + "</table></div>"
      + '<div class="mcf4-card"><h2>' + esc(__("mcf4.d_targets", "Reference observations")) + '</h2><table class="mcf4-table">'
      + (state.meta ? (function () {
          var t = state.meta.targets;
          return "<tr><th>" + esc(__("mcf4.d_dataset", "Dataset")) + "</th><td>" + esc(t.dataset + " (" + t.year + ")") + "</td></tr>"
            + "<tr><th>" + esc(__("mcf4.d_geography", "Geography")) + "</th><td>" + esc(t.geography) + "</td></tr>"
            + "<tr><th>" + esc(__("mcf4.d_units", "Units")) + "</th><td>" + esc(t.units) + "</td></tr>"
            + "<tr><th>" + esc(__("mcf4.d_metric_desc", "Metric description")) + "</th><td>" + esc(__("mcf4.d_metric_desc_val", "absolute difference between simulated and observed daily minutes, averaged per category")) + "</td></tr>";
        })() : "<tr><th></th><td>" + notRecorded() + "</td></tr>")
      + "</table>"
      + '<div class="mcf4-banner mcf4-banner-warn">' + esc(__("mcf4.d_pop_mismatch", "Reference population may not match the simulated population: the survey targets are national time-use averages, while the saved experiment simulated agents from a differently labelled world population.")) + "</div></div>"
      + '<div class="mcf4-card"><h2>' + esc(__("mcf4.d_arch_defs", "Architecture definitions")) + "</h2>"
      + '<p class="mcf4-explain">' + esc(__("mcf4.arch_vs_param", "Changing the design alters a simulation mechanism. Changing a parameter value is not a design change — parameter settings are sampled identically for both designs in the next step.")) + "</p>"
      + '<div class="mcf4-archgrid">' + archs + "</div></div>"
      + '<div class="mcf4-card"><h2>' + esc(__("mcf4.d_params", "Explored parameters")) + '</h2><div class="mcf4-scrollx"><table class="mcf4-table"><thead><tr><th>' + esc(__("mcf4.param_setting", "Parameter setting")) + "</th><th>" + esc(__("mcf4.range_col", "Range")) + "</th><th>" + esc(__("mcf4.why_range", "Why this range?")) + "</th><th>" + esc(__("mcf4.sampling_col", "Sampling")) + "</th></tr></thead><tbody>" + bandRows + "</tbody></table></div>"
      + '<p class="mcf4-explain">' + esc(__("mcf4.d_seeds", "Seeds")) + ": " + esc((doc.seeds || []).join(", ") || (doc.n_seeds ? String(doc.n_seeds) : "—")) + "</p></div>"
      + '<div class="mcf4-card"><h2>' + esc(__("mcf4.d_repro", "Reproducibility")) + '</h2><table class="mcf4-table">'
      + "<tr><th>" + esc(__("mcf4.d_commit", "Code commit")) + "</th><td>" + notRecorded() + "</td></tr>"
      + "<tr><th>" + esc(__("mcf4.d_config_ids", "Config IDs")) + "</th><td>" + esc((doc.configs || []).map(function (c) { return c.config_id; }).join(", ") || "—") + "</td></tr>"
      + "<tr><th>" + esc(__("mcf4.d_provenance", "Data provenance")) + "</th><td>" + esc(__("mcf4.d_provenance_val", "experiment manifest and per-leg records under output/mcf/") ) + "</td></tr>"
      + "</table></div>";
  }

  /* ---------------------------- runs & costs ---------------------------- */

  function renderRunsPane() {
    var doc = state.current;
    var host = el("mcf4RunsPane");
    if (!doc) { host.innerHTML = ""; return; }
    var st = legStats(doc);
    var strip = __f("mcf4.runs_strip", { total: st.total, fin: st.completed, int: st.interrupted, scored: st.scored },
      "{total} legs attempted · {fin} finished planned horizon · {int} interrupted · {scored} partially scored");
    var legs = doc.legs || [];
    var head = "<tr><th>" + esc(__("mcf4.design_col", "Design")) + "</th><th>" + esc(__("mcf4.param_values", "Parameter values")) + "</th><th>" + esc(__("mcf.th_seeds", "种子")) + "</th><th>" + esc(__("mcf4.req_days", "Requested days")) + "</th><th>" + esc(__("mcf4.obs_days", "Observed days")) + "</th><th>" + esc(__("mcf4.exec_state", "Execution state")) + "</th><th>" + esc(__("mcf4.calls", "Calls")) + "</th><th>" + esc(__("mcf4.tokens", "Tokens")) + "</th><th>" + esc(__("mcf4.elapsed", "Elapsed")) + "</th><th>" + esc(__("mcf4.stop_reason", "Stop reason")) + "</th><th>" + esc(__("mcf4.score_cov", "Score coverage")) + "</th></tr>";
    var rows = legs.map(function (l) {
      var cov = l.score && typeof l.score.coverage === "number" ? esc(fmt(l.score.coverage * 100, 0) + "%") : esc(__("mcf4.unknown", "Unknown"));
      var obsDays = l.score && l.score.per_category ? (function () {
        var mx = null;
        Object.keys(l.score.per_category).forEach(function (k) {
          var n = l.score.per_category[k].n_agent_days;
          if (typeof n === "number") mx = mx === null ? n : Math.max(mx, n);
        });
        return mx !== null ? esc(fmt(mx, 0)) : esc(__("mcf4.unknown", "Unknown"));
      })() : esc(__("mcf4.unknown", "Unknown"));
      var usage = l.usage && Object.keys(l.usage).length ? esc(JSON.stringify(l.usage)) : esc(__("mcf4.unknown", "Unknown"));
      return "<tr><td>" + esc(archLabel(doc, l.arch)) + "</td><td>" + esc(settingText(doc, l.arch + "/" + l.config_id)) + "</td><td class='mcf4-num'>" + esc(l.seed) + "</td>"
        + "<td>" + notRecorded() + "</td><td>" + obsDays + "</td>"
        + '<td><span class="mcf4-chip ' + (l.status === "completed" ? "mcf4-chip-ok" : l.status === "running" ? "mcf4-chip-warn" : "mcf4-chip-err") + '">' + esc(l.status) + "</span></td>"
        + "<td>" + usage + "</td><td>" + usage + "</td><td>" + usage + "</td>"
        + "<td>" + esc(l.stop_reason || "—") + "</td><td>" + cov + "</td></tr>";
    }).join("");
    var budget = doc.budget || {};
    var budgetBits = [];
    if (budget.max_calls) budgetBits.push(__("mcf4.budget_calls", "max calls/leg") + ": " + budget.max_calls);
    if (budget.max_tokens) budgetBits.push(__("mcf4.budget_tokens", "max tokens/leg") + ": " + budget.max_tokens);
    if (budget.max_elapsed_seconds) budgetBits.push(__("mcf4.budget_elapsed", "max seconds/leg") + ": " + budget.max_elapsed_seconds);
    if (!budgetBits.length) budgetBits.push(__("mcf4.budget_unset", "no explicit budget recorded"));
    var needsOverwriteConsent = st.total > 0;
    var usageNote = legs.length && legs[0].usage_source ? '<p class="mcf4-explain">' + esc(__("mcf4.usage_source", "Usage source:") + " " + legs[0].usage_source) + "</p>" : "";

    host.innerHTML = ""
      + '<div class="mcf4-card"><h2>' + esc(__("mcf4.runs_h", "Run progress")) + '</h2><p class="mcf4-strip">' + esc(strip) + "</p>" + usageNote
      + '<div class="mcf4-scrollx"><table class="mcf4-table"><thead>' + head + "</thead><tbody>" + (rows || "") + "</tbody></table></div></div>"
      + '<div class="mcf4-card"><h2>' + esc(__("mcf4.launch_h", "Launch experiment")) + "</h2>"
      + (doc.status === "running"
        ? '<p class="mcf4-strip">' + esc(__("mcf4.already_running", "This experiment is currently running.")) + "</p>"
        : '<div class="mcf4-launchbox">'
          + '<p class="mcf4-strip">' + esc(__f("mcf4.launch_env", { runs: st.total || (doc.progress && doc.progress.legs_total) || 0, budget: budgetBits.join(" · ") }, "Run count: {runs}. Budget envelope: {budget}.")) + "</p>"
          + (needsOverwriteConsent ? '<label class="mcf4-checkline"><input type="checkbox" id="mcf4Overwrite" /> ' + esc(__("mcf4.overwrite_warn", "I understand that re-running may overwrite this experiment's recorded results; create a new experiment instead to preserve them.")) + "</label>" : "")
          + '<label class="mcf4-checkline"><input type="checkbox" id="mcf4ConfirmCost" /> ' + esc(__("mcf4.confirm_cost_line", "I confirm the computational cost of launching these simulations.")) + "</label>"
          + '<div class="mcf4-error" id="mcf4LaunchErr"></div>'
          + '<button class="button primary" id="mcf4Launch" type="button">' + esc(__("mcf4.launch", "Launch experiment")) + "</button></div>")
      + "</div>";

    var launchBtn = el("mcf4Launch");
    if (launchBtn) launchBtn.addEventListener("click", async function () {
      var err = el("mcf4LaunchErr");
      err.textContent = "";
      var consent = el("mcf4ConfirmCost") && el("mcf4ConfirmCost").checked;
      var overwriteOk = !needsOverwriteConsent || (el("mcf4Overwrite") && el("mcf4Overwrite").checked);
      if (!consent) { err.textContent = __("mcf4.err_consent", "Please confirm the computational cost first."); return; }
      if (!overwriteOk) { err.textContent = __("mcf4.err_overwrite", "Please acknowledge the overwrite warning, or create a new experiment."); return; }
      try {
        await api("/api/mcf/experiment/" + doc.id + "/launch", { method: "POST", body: JSON.stringify({ confirm_cost: true }) });
        await openExperiment(doc.id, "runs");
      } catch (e) { err.textContent = e.message; }
    });
  }

  /* ---------------------------- diagnostics & export ---------------------------- */

  function renderDiagPane() {
    var doc = state.current;
    var host = el("mcf4DiagPane");
    if (!doc) { host.innerHTML = ""; return; }
    var diag = (state.results && state.results.diagnostics) || {};
    var st = legStats(doc);
    var valid = diag.frontier_valid === true;
    var gates = (state.meta && state.meta.gates) || {};
    var items = gateFailures();
    var cov = coverageSummary();
    var interruptions = legs0(doc).filter(function (l) { return l.status === "interrupted"; });
    function legs0(d) { return d.legs || []; }

    var html = "";
    html += '<div class="mcf4-card"><h2>' + esc(__("mcf4.dv_h", "Research validity")) + "</h2>";
    html += '<p class="mcf4-strip">' + esc(valid ? __("mcf4.dv_valid", "Validated") : __("mcf4.dv_exploratory", "Exploratory — requirements not met")) + "</p>";
    if (!valid && items.length) html += '<ul>' + items.map(function (i) { return "<li>" + (i.indexOf("<") === 0 ? i : esc(i)) + "</li>"; }).join("") + "</ul>";
    html += "</div>";

    var noise = diag.noise_domination || {};
    html += '<div class="mcf4-card"><h2>' + esc(__("mcf4.dnoise_h", "Seed variability")) + "</h2><p class='mcf4-strip'>";
    html += esc(Object.keys(noise).length
      ? Object.keys(noise).map(function (axis) {
          return axis + ": " + (noise[axis] ? __("mcf4.dnoise_flagged", "seed variation is as large as the configuration difference — the ranking is not stable") : __("mcf4.dnoise_ok", "configuration differences exceed seed variation"));
        }).join("; ")
      : __("mcf4.dnoise_none", "No seed-variability diagnostic recorded."));
    html += "</p></div>";

    html += '<div class="mcf4-card"><h2>' + esc(__("mcf4.dcov_h", "Measurement coverage")) + "</h2><p class='mcf4-strip'>";
    html += esc(cov ? __f("mcf4.dcov_line", { lo: fmt(cov.min * 100, 0), hi: fmt(cov.max * 100, 0), mean: fmt(cov.mean * 100, 0) }, "{lo}–{hi}% of the observation window scored (mean {mean}%)") : __("mcf4.unknown", "Unknown"));
    html += "</p></div>";

    html += '<div class="mcf4-card"><h2>' + esc(__("mcf4.dint_h", "Run interruptions")) + "</h2><p class='mcf4-strip'>";
    html += esc(__f("mcf4.dint_line", { n: interruptions.length, total: st.total }, "{n} of {total} legs interrupted, mostly: {reasons}")
      .replace("{reasons}", interruptions.length ? interruptions[0].stop_reason || "" : "—"));
    html += '</p><details class="mcf4-advanced"><summary>' + esc(__("mcf4.dint_detail", "Per-leg interruption log")) + "</summary><pre>";
    html += esc(interruptions.map(function (l) { return "M" + " " + l.arch + "/" + l.config_id + " seed " + l.seed + ": " + (l.stop_reason || "—"); }).join("\n"));
    html += "</pre></details></div>";

    html += '<div class="mcf4-card"><details class="mcf4-advanced"><summary>' + esc(__("mcf4.tech_diag", "Technical diagnostics")) + "</summary><pre>";
    html += esc(JSON.stringify({ frontier_valid: diag.frontier_valid === undefined ? null : diag.frontier_valid, diagnostics: diag, coverage: coverageSummary() }, null, 2));
    html += "</pre></details></div>";

    html += '<div class="mcf4-card"><h2>' + esc(__("mcf4.export_h", "Exports")) + '</h2><div class="mcf4-footbtns">'
      + '<button class="button subtle" id="mcf4ExportBtn" type="button">' + esc(__("mcf4.download_json", "Download JSON")) + "</button>"
      + '<button class="button subtle" id="mcf4ReportBtn" type="button">' + esc(__("mcf4.generate_report", "Generate report")) + "</button></div>"
      + '<pre id="mcf4ReportOut" hidden></pre></div>';

    host.innerHTML = html;
    el("mcf4ExportBtn").addEventListener("click", async function () {
      var body = await api("/api/mcf/experiment/" + doc.id + "/export");
      var blob = new Blob([JSON.stringify(body, null, 2)], { type: "application/json" });
      var a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "mcf-" + doc.id + ".json";
      a.click();
      URL.revokeObjectURL(a.href);
      this.textContent = __("mcf4.export_done", "Downloaded ✓");
    });
    el("mcf4ReportBtn").addEventListener("click", async function () {
      var out = el("mcf4ReportOut");
      out.hidden = false;
      out.textContent = __("mcf4.loading", "Loading…");
      try {
        var body = await api("/api/mcf/experiment/" + doc.id + "/report");
        out.textContent = body.markdown || __("mcf4.report_empty", "No report content returned.");
      } catch (e) { out.textContent = e.message; }
    });
  }

  /* ---------------------------- wizard ---------------------------- */

  function showStep(n) {
    state.wizardStep = n;
    document.querySelectorAll("#mcf4StepBar .mcf4-step").forEach(function (s) {
      var v = Number(s.getAttribute("data-wstep"));
      s.classList.toggle("is-active", v === n);
      s.classList.toggle("is-done", v < n);
    });
    document.querySelectorAll("#mcf4Wizard [data-wpane]").forEach(function (p) {
      p.hidden = Number(p.getAttribute("data-wpane")) !== n;
    });
    el("mcf4Back").hidden = n === 1;
    el("mcf4Next").hidden = n === 4;
    el("mcf4Save").hidden = n !== 4;
    if (n === 3) updateRunsEcho();
    if (n === 4) renderReview();
  }

  function setErr(key, msg) {
    var host = document.querySelector('[data-err="' + key + '"]');
    if (host) host.textContent = msg || "";
  }

  function updateRunsEcho() {
    var n = Number(el("mcf4NConfigs").value) || 0;
    var seeds = parseSeeds();
    el("mcf4NcEcho").textContent = n || "—";
    el("mcf4NsEcho").textContent = seeds.length || "—";
    el("mcf4Runs").textContent = n && seeds.length ? 2 * n * seeds.length : "—";
  }
  function parseSeeds() {
    return (el("mcf4Seeds").value || "").split(/[,\s]+/).filter(Boolean).map(Number)
      .filter(function (v) { return !isNaN(v); });
  }

  async function loadBandsAndMeta() {
    try {
      var mb = await api("/api/mcf/meta");
      // Incomplete meta ({}) must fall back to null so render guards hold.
      state.meta = (mb && mb.targets && mb.objectives && mb.gates) ? mb : null;
    } catch (e) { state.meta = null; }
    try {
      var bb = await api("/api/mcf/bands");
      state.bands = bb.bands || [];
    } catch (e) { state.bands = []; }
    renderWizardStatic();
  }

  function renderWizardStatic() {
    el("mcf4ReferenceRow").textContent = state.meta && state.meta.targets
      ? state.meta.targets.dataset + " (" + state.meta.targets.year + ", " + state.meta.targets.geography + ", " + state.meta.targets.units + ")"
      : __("mcf4.not_recorded", "Not recorded");
    el("mcf4PopNote").textContent = state.meta && state.meta.targets
      ? __("mcf5.pop_note", "The empirical reference is a national time-use survey; it does not describe the selected GAWorld city. Comparing the simulated cohort against it measures fit to national averages, not to the city's own population.")
      : "";
    el("mcf4Objectives").innerHTML = ["paid_work_error", "leisure_social_error"].map(function (k) {
      var m = state.meta ? (state.meta.objectives.filter(function (o) { return o.key === k; })[0] || { label: k, unit: "error units" }) : { label: k, unit: "error units" };
      return '<label class="mcf4-checkline"><input type="checkbox" checked disabled /> <span>' + esc(m.label) + ' <span class="mcf4-meta">(' + esc(__("mcf4.obj_lower", "lower = closer to observed data") + "; " + m.unit) + ")</span></span></label>";
    }).join("");
    el("mcf4DaysNote").textContent = (state.meta && state.meta.gates)
      ? __f("mcf5.days_note", { d: state.meta.gates.min_comparison_days }, "Runs shorter than {d} simulated days cannot meet the research-validity gate — they are still allowed but will be clearly marked exploratory.")
      : "";
    renderBandsTable();
    loadCityCatalogue();
    renderOutlook();
  }

  /* ---------------- population selection (city / residents) ---------------- */

  var PAGE_SIZE = 20;

  async function loadCityCatalogue() {
    var sel = el("mcf4CitySel");
    try {
      var body = await api("/api/city/catalogue");
      state.cityCatalogue = body.cities || [];
    } catch (e) {
      sel.innerHTML = '<option value="">' + esc(__("mcf5.city_failed", "City catalogue unavailable") + " (" + e.message + ")") + "</option>";
      return;
    }
    sel.innerHTML = '<option value="">' + esc(__("mcf5.city_choose", "Choose a city…")) + "</option>"
      + state.cityCatalogue.map(function (c) {
        return '<option value="' + esc(c.slug) + '">' + esc(c.display_name || c.name) + " (" + esc(String(c.count)) + ")</option>";
      }).join("");
  }

  function cityEntry(slug) {
    return (state.cityCatalogue || []).filter(function (c) { return c.slug === slug; })[0] || null;
  }

  function onCityChange() {
    state.city = el("mcf4CitySel").value;
    setErr("mcf4CitySel", "");
    state.selIds = [];
    state.agPage = 0;
    state.agCache = null;
    state.lastPreview = null;
    var entry = cityEntry(state.city);
    el("mcf4CityCount").textContent = entry
      ? __f("mcf5.city_count", { n: entry.count, name: entry.display_name || entry.name }, "{name}: {n} residents in the catalogue")
      : "";
    if (!state.city) {
      el("mcf4AgentList").innerHTML = "";
      el("mcf4AgPageInfo").textContent = "";
      renderSelectedCount();
      renderEmployedLine(null);
      renderAllWarn();
      renderOutlook();
      return;
    }
    loadAgentPage(0);
    renderAllWarn();
    renderOutlook();
  }

  async function loadAgentPage(page) {
    var list = el("mcf4AgentList");
    list.innerHTML = '<div class="mcf4-meta">' + esc(__("mcf4.loading", "Loading…")) + "</div>";
    try {
      var offset = page * PAGE_SIZE;
      var body = await api("/api/city/agents?city=" + encodeURIComponent(state.city) + "&limit=" + PAGE_SIZE + "&offset=" + offset);
      state.agCache = body;
      state.agPage = page;
      renderAgentPage();
    } catch (e) {
      list.innerHTML = '<div class="mcf4-banner mcf4-banner-err">' + esc(__("mcf4.agents_api_failed", "Resident list is unavailable (API error): ") + e.message) + "</div>";
    }
  }

  function renderAgentPage() {
    var body = state.agCache;
    if (!body) return;
    var q = (el("mcf4AgSearch").value || "").toLowerCase();
    var agents = (body.agents || []).filter(function (a) {
      return !q || String(a.name || "").toLowerCase().indexOf(q) >= 0 || String(a.id).indexOf(q) >= 0;
    });
    var list = el("mcf4AgentList");
    if (!agents.length) {
      list.innerHTML = '<div class="mcf4-meta">' + esc(__("mcf4.no_agents", "No residents match.")) + "</div>";
    } else {
      list.innerHTML = agents.map(function (a) {
        var checked = state.selIds.indexOf(Number(a.id)) >= 0;
        return '<label class="mcf4-checkline"><input type="checkbox" value="' + esc(String(a.id)) + '"' + (checked ? " checked" : "") + " /> #"
          + esc(String(a.id)) + " " + esc(a.name || "")
          + ' <span class="mcf4-meta">' + esc(a.employment || a.job || "") + "</span></label>";
      }).join("");
      list.querySelectorAll("input").forEach(function (cb) {
        cb.addEventListener("change", function () {
          var id = Number(cb.value);
          if (cb.checked) { if (state.selIds.indexOf(id) < 0) state.selIds.push(id); }
          else state.selIds = state.selIds.filter(function (x) { return x !== id; });
          renderSelectedCount();
          refreshEmployedPreview();
        });
      });
    }
    var matched = body.matched !== undefined ? body.matched : (body.agents || []).length;
    el("mcf4AgPageInfo").textContent = __f("mcf5.page_info",
      { from: state.agPage * PAGE_SIZE + 1, to: state.agPage * PAGE_SIZE + agents.length, total: matched },
      "residents {from}–{to} of {total}");
    el("mcf4AgPrev").disabled = state.agPage === 0;
    el("mcf4AgNext").disabled = (state.agPage + 1) * PAGE_SIZE >= matched;
    renderSelectedCount();
  }

  function renderSelectedCount() {
    el("mcf4SelCount").textContent = __f("mcf5.selected_count", { n: state.selIds.length }, "{n} residents selected (selections persist across pages)");
  }

  function renderAllWarn() {
    var entry = cityEntry(state.city);
    el("mcf4AllWarn").textContent = entry
      ? __f("mcf5.all_warn", { n: entry.count }, "This selects ALL {n} residents of the city. Every resident multiplies model calls, tokens and wall-clock time — check the budget envelope on the Runs & costs tab before launching.")
      : "";
  }

  function renderEmployedLine(preview) {
    var line = el("mcf4EmployedLine");
    if (!preview) { line.textContent = ""; return; }
    var text = __f("mcf5.employed_line",
      { x: preview.n_employed, y: preview.n_agents },
      "{x} of {y} selected residents are classified as employed (paid-work scoring covers employed residents only).");
    if (preview.unknown_employment) {
      text += " " + __f("mcf5.employed_unknown", { n: preview.unknown_employment }, "{n} selected residents have no employment classification on record and are excluded from the paid-work participants basis.");
    }
    line.textContent = text;
  }

  async function refreshEmployedPreview() {
    /* employment comes from the city roster via the same backend resolution
       the creation path uses — never guessed in the browser */
    if (state.city !== "" && state.selIds.length) {
      try {
        var preview = await api("/api/mcf/sample-preview?city=" + encodeURIComponent(state.city)
          + "&method=manual&agent_ids=" + encodeURIComponent(state.selIds.join(",")));
        state.lastPreview = preview;
        renderEmployedLine(preview);
        renderOutlook();
      } catch (e) { renderEmployedLine(null); }
    } else { renderEmployedLine(null); renderOutlook(); }
  }

  async function previewRandomSample() {
    setErr("mcf4Population", "");
    var size = Number(el("mcf4SampleSize").value);
    var seed = Number(el("mcf4SampleSeed").value);
    if (!state.city) { setErr("mcf4Population", __("mcf5.err_city_first", "Choose a city first.")); return; }
    if (!(size >= 1)) { setErr("mcf4Population", __("mcf5.err_sample_size", "Sample size must be at least 1.")); return; }
    try {
      var preview = await api("/api/mcf/sample-preview?city=" + encodeURIComponent(state.city)
        + "&method=random&sample_size=" + encodeURIComponent(size) + "&sample_seed=" + encodeURIComponent(seed));
      state.lastPreview = preview;
      el("mcf4RandomInfo").textContent = __f("mcf5.random_info",
        { n: preview.n_agents, x: preview.n_employed, ids: preview.agent_ids.slice(0, 12).join(", "), more: preview.n_agents > 12 ? "…" : "" },
        "Reproducible sample of {n} residents (employed: {x}): IDs {ids}{more} — the same seed will always produce this same sample.");
      renderEmployedLine(preview);
      renderOutlook();
    } catch (e) {
      setErr("mcf4Population", e.message);
    }
  }

  function renderBandsTable() {
    var host = el("mcf4Bands");
    if (!state.bands.length) {
      host.innerHTML = '<div class="mcf4-banner mcf4-banner-err">' + esc(__("mcf4.bands_failed", "Parameter band catalogue is unavailable — the backend did not respond.")) + "</div>";
      return;
    }
    var rows = state.bands.map(function (b) {
      var checked = b.name === "style_fit_amplitude" ? " checked" : "";
      var why = b.sources && b.sources.length ? esc(b.sources.join(" ")) : __("mcf4.source_none", "No recorded justification.");
      return '<tr><td><label class="mcf4-checkline"><input type="checkbox" value="' + esc(b.name) + '"' + checked + ' /> <span>' + esc(paramLabelSafe(b.name)) + '</span></label><div class="mcf4-meta">' + esc(b.path) + "</div></td>"
        + '<td class="mcf4-num">' + esc(fmt(b.low)) + "</td>"
        + '<td class="mcf4-num">' + esc(fmt(b.high)) + "</td>"
        + '<td class="mcf4-whycell">' + why + "</td>"
        + '<td><input type="checkbox" class="mcf4-bandinclude" value="' + esc(b.name) + '"' + checked + ' aria-label="' + esc(__("mcf4.include", "Include")) + '" /></td></tr>';
    }).join("");
    host.innerHTML = '<div class="mcf4-scrollx"><table class="mcf4-table"><thead><tr><th>' + esc(__("mcf4.parameter", "Parameter")) + "</th><th>" + esc(__("mcf4.min_col", "Minimum")) + "</th><th>" + esc(__("mcf4.max_col", "Maximum")) + "</th><th>" + esc(__("mcf4.why_range", "Why this range?")) + "</th><th>" + esc(__("mcf4.include", "Include")) + "</th></tr></thead><tbody>" + rows + "</tbody></table></div>";
    host.querySelectorAll(".mcf4-bandinclude").forEach(function (cb) {
      cb.addEventListener("change", function () {
        var row = cb.closest("tr");
        var main = row.querySelector(".mcf4-checkline input");
        main.checked = cb.checked;
        updateRunsEcho();
      });
    });
    host.querySelectorAll(".mcf4-checkline input").forEach(function (cb) {
      cb.addEventListener("change", function () {
        var row = cb.closest("tr");
        var inc = row.querySelector(".mcf4-bandinclude");
        inc.checked = cb.checked;
        updateRunsEcho();
      });
    });
  }

  function renderOutlook() {
    var g = (state.meta && state.meta.gates) || {};
    var bits = [];
    var nAgents = currentPopulationSize();
    var days = Number(el("mcf4Days").value) || 0;
    if (g.min_comparison_population && nAgents && nAgents < g.min_comparison_population) {
      bits.push(__f("mcf4.outlook_pop_short", { n: nAgents, req: g.min_comparison_population }, "cohort {n} < required {req} agents"));
    }
    if (g.min_comparison_days && days && days < g.min_comparison_days) {
      bits.push(__f("mcf4.outlook_days_short", { d: days, req: g.min_comparison_days }, "duration {d} days < required {req} days"));
    }
    var head = __("mcf4.outlook_head", "Research-validity outlook — informational; smaller exploratory designs are allowed and will be clearly labelled:");
    if (bits.length) {
      el("mcf4Outlook").innerHTML = '<div class="mcf4-banner mcf4-banner-warn">' + esc(head) + " " + esc(bits.join(" · ")) + "</div>";
    } else {
      el("mcf4Outlook").innerHTML = '<div class="mcf4-banner mcf4-banner-info">' + esc(head) + " "
        + esc(__("mcf4.outlook_ok", "the current selection meets the population and duration gates")) + "</div>";
    }
  }

  function currentPopulationSize() {
    var method = currentMethod();
    if (method === "all") { var e = cityEntry(state.city); return e ? e.count : 0; }
    if (method === "random") return state.lastPreview ? state.lastPreview.n_agents : (Number(el("mcf4SampleSize").value) || 0);
    return state.selIds.length;
  }

  function currentMethod() {
    var r = document.querySelector('#mcf4MethodRow input[name="mcf4Method"]:checked');
    return r ? r.value : "manual";
  }

  function parsePatch(text, label) {
    var parsed;
    try { parsed = JSON.parse(text || "{}"); } catch (e) { throw new Error(label + ": " + __("mcf4.err_json", "patch is not valid JSON") + " — " + e.message); }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error(label + ": " + __("mcf4.err_json_obj", "patch must be a JSON object"));
    return parsed;
  }

  function collectPayload() {
    var seeds = parseSeeds();
    var dup = seeds.filter(function (s, i) { return seeds.indexOf(s) !== i; });
    if (!seeds.length) throw { field: "mcf4Seeds", message: __("mcf4.err_seeds", "Enter at least one integer seed (comma-separated).") };
    if (dup.length) throw { field: "mcf4Seeds", message: __("mcf4.err_seeds_dup", "Seeds must be unique integers.") };
    var params = selectedBands();
    if (!params.length) throw { field: "mcf4Bands", message: __("mcf4.err_bands", "Include at least one parameter band.") };
    var patch0 = parsePatch(el("mcf4Patch0").value, "M0");
    var patch1 = parsePatch(el("mcf4Patch1").value, "M1");
    var days = Number(el("mcf4Days").value);
    if (!(days >= 1) || String(days).indexOf(".") >= 0) throw { field: "mcf4Days", message: __("mcf5.err_days", "Simulation duration must be a whole number of days (>= 1).") };
    var population = populationPayload();
    return {
      name: el("mcf4Name").value.trim(),
      note: el("mcf4Note").value.trim(),
      population: population,
      sim_days: days,
      architectures: [
        { id: "m0", label: el("mcf4Label0").value.trim() || "Rules on (M0)", patch: patch0 },
        { id: "m1", label: el("mcf4Label1").value.trim() || "Rules off (M1)", patch: patch1 },
      ],
      reference_architecture: "m0",
      parameters: params,
      sampler: el("mcf4Sampler").value,
      n_configs: Number(el("mcf4NConfigs").value) || 2,
      seeds: seeds,
      budget: {
        max_calls: undefined, max_tokens: undefined, max_elapsed_seconds: undefined,
      },
      model: {},
    };
  }

  function populationPayload() {
    var method = currentMethod();
    if (!state.city) throw { field: "mcf4CitySel", message: __("mcf5.err_city", "Choose a city.") };
    if (method === "manual") {
      if (!state.selIds.length) throw { field: "mcf4Population", message: __("mcf5.err_manual_empty", "Select at least one resident, or switch to a random sample.") };
      return { city: state.city, method: "manual", agent_ids: state.selIds.slice() };
    }
    if (method === "random") {
      var size = Number(el("mcf4SampleSize").value);
      var seed = Number(el("mcf4SampleSeed").value);
      if (!(size >= 1)) throw { field: "mcf4Population", message: __("mcf5.err_sample_size", "Sample size must be at least 1.") };
      if (!(seed >= 0) || String(seed).indexOf(".") >= 0) throw { field: "mcf4Population", message: __("mcf5.err_seed", "Random seed must be a non-negative integer.") };
      return { city: state.city, method: "random", sample_size: size, sample_seed: seed };
    }
    return { city: state.city, method: "all" };
  }
  function selectedBands() {
    return Array.prototype.map.call(document.querySelectorAll("#mcf4Bands .mcf4-bandinclude:checked"), function (i) { return i.value; });
  }

  function validateStep(step) {
    var ok = true;
    if (step === 1) {
      if (!el("mcf4Name").value.trim()) { setErr("mcf4Name", __("mcf4.err_name", "Please give the experiment a name.")); ok = false; } else setErr("mcf4Name", "");
      try { populationPayload(); setErr("mcf4CitySel", ""); setErr("mcf4Population", ""); }
      catch (e) { if (e.field) setErr(e.field, e.message); ok = false; }
      var days = Number(el("mcf4Days").value);
      if (!(days >= 1) || String(days).indexOf(".") >= 0) { setErr("mcf4Days", __("mcf5.err_days", "Simulation duration must be a whole number of days (>= 1).")); ok = false; } else setErr("mcf4Days", "");
    }
    if (step === 3) {
      try { collectPayloadSeedsOnly(); } catch (e) { if (e.field) setErr(e.field, e.message); ok = false; }
    }
    return ok;
  }
  function collectPayloadSeedsOnly() {
    var seeds = parseSeeds();
    var dup = seeds.filter(function (s, i) { return seeds.indexOf(s) !== i; });
    if (!seeds.length) throw { field: "mcf4Seeds", message: __("mcf4.err_seeds", "Enter at least one integer seed (comma-separated).") };
    if (dup.length) throw { field: "mcf4Seeds", message: __("mcf4.err_seeds_dup", "Seeds must be unique integers.") };
    if (!selectedBands().length) throw { field: "mcf4Bands", message: __("mcf4.err_bands", "Include at least one parameter band.") };
    return seeds;
  }

  function renderReview() {
    var payloadPreview = {
      name: el("mcf4Name").value.trim() || "—",
      params: selectedBands(),
      sampler: el("mcf4Sampler").value === "sobol" ? __("mcf4.sampler_sobol", "Sobol sequence") : __("mcf4.sampler_endpoints", "Band endpoints"),
      seeds: parseSeeds(),
      n: Number(el("mcf4NConfigs").value) || 0,
      mech: [el("mcf4Patch0").value, el("mcf4Patch1").value],
    };
    var g = (state.meta && state.meta.gates) || {};
    var nAgents = currentPopulationSize();
    var days = Number(el("mcf4Days").value) || 0;
    var methodLabel = { manual: __("mcf5.m_manual", "Manual selection"), random: __("mcf5.m_random", "Random sample"), all: __("mcf5.m_all", "Entire population") }[currentMethod()];
    var cityEntry0 = cityEntry(state.city);
    var employedBits = state.lastPreview
      ? __f("mcf5.employed_line", { x: state.lastPreview.n_employed, y: state.lastPreview.n_agents }, "{x} of {y} selected residents are classified as employed (paid-work scoring covers employed residents only).")
      : __("mcf5.employed_at_save", "Employment is derived from the city roster when the experiment is saved (participants basis).");
    var exploratory = (nAgents && g.min_comparison_population && nAgents < g.min_comparison_population)
      || (days && g.min_comparison_days && days < g.min_comparison_days);
    var rows = [
      [__("mcf4.rv_name", "Experiment name"), payloadPreview.name],
      [__("mcf5.rv_city", "City"), (cityEntry0 ? (cityEntry0.display_name || cityEntry0.name) + " (" + state.city + ")" : state.city || "—")],
      [__("mcf5.rv_population", "Population"), methodLabel + " · " + (nAgents ? String(nAgents) : "—") + " " + __("mcf5.residents_word", "residents") + (currentMethod() === "random" ? " · " + __("mcf5.rv_seed_word", "sampling seed") + " " + (el("mcf4SampleSeed").value || "—") : "")],
      [__("mcf4.rv_employed", "Employed-agent basis"), employedBits],
      [__("mcf5.rv_days", "Simulation duration"), (days || "—") + " " + __("mcf4.days_word", "days") + " · " + __("mcf5.tick_note", "tick interval as configured by the runner")],
      [__("mcf4.rv_dataset", "Empirical dataset"), state.meta ? state.meta.targets.dataset + " (" + state.meta.targets.year + ", " + state.meta.targets.units + ")" : "—"],
      [__("mcf4.rv_objectives", "Objectives"), objectives().map(function (k) { return objLabel(k); }).join(" · ")],
      [__("mcf4.rv_mech", "Mechanism difference"), "M0 patch " + payloadPreview.mech[0] + " vs M1 patch " + payloadPreview.mech[1]],
      [__("mcf4.rv_params", "Parameter range(s)"), payloadPreview.params.join(", ") || "—"],
      [__("mcf4.rv_sampling", "Sampling method"), payloadPreview.sampler],
      [__("mcf4.rv_seeds", "Seeds"), payloadPreview.seeds.join(", ")],
      [__("mcf4.rv_legs", "Total legs"), "2 × " + payloadPreview.n + " × " + payloadPreview.seeds.length + " = " + 2 * payloadPreview.n * payloadPreview.seeds.length],
    ].map(function (r) { return "<tr><th>" + esc(r[0]) + "</th><td>" + esc(r[1]) + "</td></tr>"; }).join("");
    el("mcf4Review").innerHTML = '<table class="mcf4-table">' + rows + "</table>";
    el("mcf4Workload").innerHTML = '<div class="mcf4-banner mcf4-banner-info"><strong>' + esc(__("mcf4.workload_h", "Estimated workload")) + "</strong><br>"
      + esc(__("mcf4.workload_none", "Estimated usage unavailable; these are safety limits, not predicted consumption.")) + "<br>"
      + esc(__("mcf4.workload_model", "Model/provider:") + " " + __("mcf4.workload_routing", "configured model routing (no provider is used while saving)")) + "<br>"
      + esc(__("mcf4.workload_caps", "Safety caps are set per leg by the runner's budget configuration.")) + "</div>";
    el("mcf4PaidNote").hidden = true;
    el("mcf4ExploratoryTag").hidden = !exploratory;
    el("mcf4ExploratoryTag").textContent = __("mcf4.exploratory_tag", "Exploratory design: the current research gates would not be met (cohort and duration limits). The saved experiment will be clearly labelled as exploratory.");
  }

  async function saveDraft() {
    el("mcf4FormError").textContent = "";
    var payload;
    try { payload = collectPayload(); } catch (e) {
      if (e.field) setErr(e.field, e.message); else el("mcf4FormError").textContent = e.message;
      return;
    }
    try {
      var created = await api("/api/mcf/experiment", { method: "POST", body: JSON.stringify(payload) });
      await loadExperiments();
      openExperiment(created.id, "design");
      var banner = document.createElement("div");
      banner.className = "mcf4-banner mcf4-banner-ok";
      banner.textContent = __("mcf4.saved", "Experiment saved. No simulations have been started.");
      el("mcf4Ws").insertBefore(banner, el("mcf4Ws").firstChild);
    } catch (error) {
      el("mcf4FormError").textContent = error.message;
    }
  }

  /* ---------------------------- polling ---------------------------- */

  function startPolling() {
    stopPolling();
    state.poll = setInterval(async function () {
      if (!state.current || state.view !== "workspace") return stopPolling();
      var doc = await api("/api/mcf/experiment/" + state.current.id).catch(function () { return null; });
      if (!doc) return;
      state.current = doc;
      renderWorkspaceAll();
      var body = await api("/api/mcf/experiment/" + doc.id + "/results").catch(function () { return null; });
      if (body) { state.results = body.results; state.message = body.message || ""; renderWorkspaceAll(); renderEvidence(); }
      if (doc.status !== "running") stopPolling();
    }, 3000);
  }
  function stopPolling() { if (state.poll) { clearInterval(state.poll); state.poll = null; } }

  /* ---------------------------- boot -------------------------- */

  function boot() {
    if (!el("rwPaneMcf")) return;

    el("mcf4NewBtn").addEventListener("click", function () {
      state.wizardMode = true; showView("library");
      el("mcf4Wizard").hidden = false; el("mcf4LibBody").hidden = true;
      el("mcf4FormError").textContent = "";
      showStep(1);
    });
    el("mcf4Cancel").addEventListener("click", function () { state.wizardMode = false; showView("library"); });
    el("mcf4Back").addEventListener("click", function () { if (state.wizardStep > 1) showStep(state.wizardStep - 1); });
    el("mcf4Next").addEventListener("click", function () {
      if (!validateStep(state.wizardStep)) return;
      if (state.wizardStep < 4) showStep(state.wizardStep + 1);
    });
    el("mcf4Save").addEventListener("click", saveDraft);
    el("mcf4Crumb").addEventListener("click", function () { state.wizardMode = false; showView("library"); renderLibrary(); });
    el("mcf4Refresh").addEventListener("click", function () { loadExperiments().catch(function () { }); });
    el("mcf4WsRefresh").addEventListener("click", function () { if (state.current) openExperiment(state.current.id, currentTab()); });
    el("mcf4Search").addEventListener("input", function () { state.search = this.value; renderLibrary(); });
    document.querySelectorAll("#mcf4Filters button").forEach(function (b) {
      b.addEventListener("click", function () {
        state.filter = b.getAttribute("data-filter");
        document.querySelectorAll("#mcf4Filters button").forEach(function (x) { x.classList.toggle("is-active", x === b); });
        renderLibrary();
      });
    });
    document.querySelectorAll("#mcf4Tabs button").forEach(function (b) {
      b.addEventListener("click", function () { selectTab(b.getAttribute("data-tab")); });
    });
    el("mcf4CitySel").addEventListener("change", onCityChange);
    document.querySelectorAll('#mcf4MethodRow input[name="mcf4Method"]').forEach(function (radio) {
      radio.addEventListener("change", function () {
        el("mcf4MethodManual").hidden = currentMethod() !== "manual";
        el("mcf4MethodRandom").hidden = currentMethod() !== "random";
        el("mcf4MethodAll").hidden = currentMethod() !== "all";
        renderOutlook();
      });
    });
    el("mcf4AgSearch").addEventListener("input", function () { renderAgentPage(); });
    el("mcf4AgPrev").addEventListener("click", function () { if (state.agPage > 0) loadAgentPage(state.agPage - 1); });
    el("mcf4AgNext").addEventListener("click", function () { loadAgentPage(state.agPage + 1); });
    el("mcf4PreviewBtn").addEventListener("click", previewRandomSample);
    el("mcf4Days").addEventListener("input", renderOutlook);
    el("mcf4NConfigs").addEventListener("input", updateRunsEcho);
    el("mcf4Seeds").addEventListener("input", updateRunsEcho);

    /* hide the Study-Plans intro banner while the MCF tab is active */
    var updateBanner = function () {
      var active = document.querySelector(".rw-tab.is-active");
      document.body.classList.toggle("mcf4-active", !!(active && active.getAttribute("data-tab") === "mcf"));
    };
    document.querySelectorAll(".rw-tab").forEach(function (b) { b.addEventListener("click", updateBanner); });
    updateBanner();

    document.addEventListener("locale-changed", function () {
      renderLibrary();
      if (state.current) { renderMetaLine(); renderEvidence(); renderWorkspaceAll(); }
      renderWizardStatic();
      renderReview();
    });

    showView("library");
    showStep(1);
    loadBandsAndMeta();
    loadExperiments().catch(function (e) { el("mcf4Empty").textContent = e.message; });

    var query = new URLSearchParams(window.location.search);
    if (query.get("tab") === "mcf" && query.get("experiment")) {
      openExperiment(query.get("experiment"));
    }
  }

  function currentTab() {
    var b = document.querySelector("#mcf4Tabs button.is-active");
    return b ? b.getAttribute("data-tab") : "overview";
  }

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
    else boot();
  }
  if (typeof module !== "undefined" && module.exports) module.exports = { __state: state };
})();

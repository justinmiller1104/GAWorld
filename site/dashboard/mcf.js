/* MCF research-workflow panel — the MCF 工作流 tab of the research workbench.
 *
 * One experiment = behaviour architectures (M0 baseline vs a treatment)
 * swept over the admissible parameter bands of benchmark/mcf/config_space.
 * The backend (gaworld/apps/mcf_api.py) owns every number: per-config error
 * vectors, Pareto fronts, A(M1|M0) and diagnostics come from
 * benchmark/mcf/frontier over the leg table the orchestrator actually ran.
 * This file only draws what the API returned — nothing is mocked or computed
 * client-side.
 *
 * Charts are hand-written inline SVG, same reason worlds.js gives: no build
 * step, no CDN, the dashboard stays usable offline.
 */
(function () {
  "use strict";

  var COLORS = { m0: "#5c6b73", m1: "#0e7a58" };

  var state = {
    bands: [],
    experiments: [],
    current: null,   // full experiment doc
    results: null,   // results payload (null until a leg is scored)
    message: "",
    error: "",
    poll: null,
  };

  function el(id) { return document.getElementById(id); }

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function fmt(value, digits) {
    var number = Number(value);
    if (!isFinite(number)) return "—";
    return number.toFixed(digits == null ? 4 : digits);
  }

  async function api(path, options) {
    var settings = Object.assign({}, options || {});
    settings.headers = Object.assign(
      { "Content-Type": "application/json" }, settings.headers || {}
    );
    var response = await fetch(path, settings);
    var body = await response.json().catch(function () { return {}; });
    if (!response.ok) {
      throw new Error(body.error || (response.status + " " + response.statusText));
    }
    return body;
  }

  /* Each tab names its pane through aria-controls; this module adds its own
     click handler and deep-link (?tab=mcf) so no other tab's code changes. */
  function showTab(name) {
    document.querySelectorAll(".rw-tab").forEach(function (tab) {
      var on = tab.getAttribute("data-tab") === name;
      tab.classList.toggle("is-active", on);
      tab.setAttribute("aria-selected", on ? "true" : "false");
      var pane = el(tab.getAttribute("aria-controls"));
      if (pane) pane.hidden = !on;
    });
    var query = new URLSearchParams(window.location.search);
    if (name === "mcf") query.set("tab", "mcf");
    else query.delete("tab");
    var text = query.toString();
    window.history.replaceState(null, "",
      window.location.pathname + (text ? "?" + text : ""));
  }

  /* ------------------------------------------------------------ lists */

  async function loadBands() {
    var body = await api("/api/mcf/bands");
    state.bands = body.bands || [];
    renderBands();
  }

  async function loadExperiments(selectId) {
    var body = await api("/api/mcf/experiments");
    state.experiments = body.experiments || [];
    renderHistory(selectId);
  }

  function renderBands() {
    var host = el("mcfBands");
    if (!host) return;
    if (!state.bands.length) {
      host.innerHTML = '<p class="rw-hint">没有可调参数带。</p>';
      return;
    }
    var checks = state.bands.map(function (band) {
      return '<label class="pw-check"><input type="checkbox" value="' + esc(band.name) + '" /> '
        + esc(band.name) + " <span class=\"pw-subtle\">[" + esc(band.low) + ", " + esc(band.high)
        + "]</span></label>";
    }).join("");
    host.innerHTML = checks + '<table class="rw-table"><thead><tr><th>参数</th><th>配置路径</th>'
      + "<th>下界</th><th>上界</th><th>出处</th></tr></thead><tbody>"
      + state.bands.map(function (band) {
        return "<tr><td>" + esc(band.name) + "</td><td>" + esc(band.path) + "</td><td>"
          + esc(band.low) + "</td><td>" + esc(band.high) + "</td><td>"
          + esc(band.source || "—") + "</td></tr>";
      }).join("") + "</tbody></table>";
    var picker = el("mcfParamChecks");
    if (picker) picker.innerHTML = checks;
  }

  function renderHistory(selectId) {
    var select = el("mcfHistory");
    if (!select) return;
    var previous = selectId || state.current && state.current.id || select.value;
    select.innerHTML = '<option value="">— 选择实验 —</option>'
      + state.experiments.map(function (exp) {
        return '<option value="' + esc(exp.id) + '"'
          + (exp.id === previous ? " selected" : "") + ">"
          + esc(exp.id + " · " + exp.name + " · " + exp.status) + "</option>";
      }).join("");
  }

  /* ------------------------------------------------------------ create */

  function parsePatch(text, label) {
    try {
      var parsed = JSON.parse(text || "{}");
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
      throw new Error("not an object");
    } catch (error) {
      throw new Error(label + " 配置补丁不是合法 JSON 对象：" + error.message);
    }
  }

  async function createExperiment() {
    state.error = "";
    var seeds = (el("mcfSeeds").value || "").split(/[,\s]+/).filter(Boolean).map(Number);
    var payload = {
      name: el("mcfName").value.trim(),
      architectures: [
        { id: "m0", label: el("mcfLabel0").value.trim() || "M0",
          patch: parsePatch(el("mcfPatch0").value, "M0") },
        { id: "m1", label: el("mcfLabel1").value.trim() || "M1",
          patch: parsePatch(el("mcfPatch1").value, "M1") },
      ],
      reference_architecture: "m0",
      parameters: Array.prototype.map.call(
        document.querySelectorAll("#mcfParamChecks input:checked"),
        function (input) { return input.value; }
      ),
      sampler: el("mcfSampler").value,
      n_configs: Number(el("mcfNConfigs").value) || 2,
      n_seeds: Number(el("mcfNSeeds").value) || 1,
      budget: {
        max_calls: Number(el("mcfMaxCalls").value) || undefined,
        max_tokens: Number(el("mcfMaxTokens").value) || undefined,
        max_elapsed_seconds: Number(el("mcfMaxElapsed").value) || undefined,
      },
    };
    if (seeds.length) payload.seeds = seeds;
    try {
      var created = await api("/api/mcf/experiment", { method: "POST", body: JSON.stringify(payload) });
      state.current = created;
      state.results = null;
      await loadExperiments(created.id);
      renderCurrent();
      renderResults();
    } catch (error) {
      state.error = error.message;
      renderCurrent();
    }
  }

  async function openExperiment(id) {
    if (!id) { state.current = null; state.results = null; renderCurrent(); renderResults(); return; }
    var doc = await api("/api/mcf/experiment/" + id);
    state.current = doc;
    state.results = null;
    renderCurrent();
    await refreshResults();
  }

  /* ------------------------------------------------------------ run */

  async function launch() {
    if (!state.current) return;
    var payload = { confirm_cost: el("mcfConfirmCost").checked };
    try {
      await api("/api/mcf/experiment/" + state.current.id + "/launch", {
        method: "POST", body: JSON.stringify(payload),
      });
      await openExperiment(state.current.id);
      startPolling();
    } catch (error) {
      state.error = error.message;
      renderCurrent();
    }
  }

  function startPolling() {
    stopPolling();
    state.poll = setInterval(async function () {
      if (!state.current) return stopPolling();
      var doc = await api("/api/mcf/experiment/" + state.current.id).catch(function () { return null; });
      if (!doc) return;
      state.current = doc;
      renderCurrent();
      await refreshResults();
      if (doc.status !== "running") stopPolling();
    }, 3000);
  }

  function stopPolling() {
    if (state.poll) { clearInterval(state.poll); state.poll = null; }
  }

  async function refreshResults() {
    if (!state.current) return;
    var body = await api("/api/mcf/experiment/" + state.current.id + "/results");
    state.results = body.results;
    state.message = body.message || "";
    renderResults();
  }

  function renderCurrent() {
    var host = el("mcfProgress");
    var launchButton = el("mcfLaunch");
    if (!state.current) {
      host.textContent = "没有选中的实验。";
      launchButton.disabled = true;
      renderHistory();
      return;
    }
    var progress = state.current.progress || {};
    var lines = [
      esc(state.current.id + " · " + state.current.name + " · " + state.current.status),
      "腿进度：" + (progress.legs_done || 0) + " / " + (progress.legs_total || 0)
        + "，估计模型调用：" + fmt(state.current.estimated_calls, 0),
    ];
    if (state.current.last_error) lines.push("最近错误：" + esc(state.current.last_error));
    (progress.failures || []).forEach(function (failure) {
      lines.push("失败：" + esc(failure.arch + "/" + failure.config_id + "/seed " + failure.seed
        + " — " + failure.status + " " + (failure.stop_reason || "")));
    });
    host.innerHTML = lines.map(function (line) { return "<div>" + line + "</div>"; }).join("");
    launchButton.disabled = state.current.status === "running";
    renderHistory();
  }

  /* ------------------------------------------------------------ results */

  function renderResults() {
    renderTable();
    renderPareto();
    renderDiagnostics();
  }

  function renderTable() {
    var host = el("mcfTable");
    if (!state.results) {
      host.innerHTML = '<p class="rw-hint">' + esc(state.message || "还没有完成的评分腿。") + "</p>";
      return;
    }
    var rows = [];
    state.results.architectures.forEach(function (arch) {
      arch.configs.forEach(function (cfg) {
        rows.push("<tr><td>" + esc(arch.arch) + "</td><td>" + esc(cfg.config) + "</td><td>"
          + esc(cfg.paid_work_error) + "</td><td>" + esc(cfg.leisure_social_error) + "</td><td>"
          + esc((cfg.seeds || []).join(", ")) + "</td><td>"
          + (cfg.complete ? "是" : "否（缺 " + esc((cfg.missing_seeds || []).join(", ")) + "）") + "</td></tr>");
      });
    });
    host.innerHTML = '<table class="rw-table"><thead><tr><th>架构</th><th>配置</th>'
      + "<th>paid_work_error</th><th>leisure_social_error</th><th>种子</th><th>完整</th></tr></thead><tbody>"
      + rows.join("") + "</tbody></table>";
  }

  function svgScatter(marks, xLo, xHi, yLo, yHi) {
    var width = 560, height = 320, pad = 44;
    var xSpan = (xHi - xLo) || 1, ySpan = (yHi - yLo) || 1;
    function px(x) { return pad + (x - xLo) / xSpan * (width - pad - 16); }
    function py(y) { return height - pad - (y - yLo) / ySpan * (height - pad - 16); }
    var parts = [];
    parts.push('<svg viewBox="0 0 ' + width + " " + height + '" role="img" aria-label="帕累托散点图">');
    parts.push('<rect x="' + pad + '" y="16" width="' + (width - pad - 16) + '" height="'
      + (height - pad - 16) + '" fill="none" stroke="currentColor" stroke-opacity="0.25" />');
    parts.push('<text x="' + (width / 2) + '" y="' + (height - 8)
      + '" text-anchor="middle" font-size="12">paid_work_error →</text>');
    parts.push('<text x="12" y="' + (height / 2) + '" font-size="12" transform="rotate(-90 12 '
      + (height / 2) + ')" text-anchor="middle">leisure_social_error →</text>');
    marks.forEach(function (mark) {
      if (mark.kind === "line") {
        parts.push('<polyline fill="none" stroke="' + mark.color + '" stroke-width="1.5" '
          + 'stroke-dasharray="4 3" points="' + mark.points.map(function (p) {
            return px(p[0]).toFixed(1) + "," + py(p[1]).toFixed(1);
          }).join(" ") + '" />');
        return;
      }
      var x = px(mark.x), y = py(mark.y);
      var shape;
      if (mark.gain) {
        shape = '<path d="M' + (x - 7) + " " + y + " L" + x + " " + (y - 7) + " L" + (x + 7) + " " + y
          + " L" + x + " " + (y + 7) + ' Z" fill="none" stroke="' + mark.color
          + '" stroke-width="2.5" />';
      } else {
        shape = '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + (mark.front ? 7 : 4.5)
          + '" fill="' + (mark.front ? mark.color : "none") + '" fill-opacity="0.85" stroke="'
          + mark.color + '" stroke-width="' + (mark.front ? 2 : 1.5) + '" />';
      }
      parts.push('<g><title>' + esc(mark.label) + "</title>" + shape + "</g>");
    });
    parts.push("</svg>");
    return parts.join("");
  }

  function renderPareto() {
    var host = el("mcfPareto");
    var note = el("mcfParetoNote");
    if (!state.results) {
      host.innerHTML = "";
      note.textContent = state.message || "等第一批腿跑完再画图。";
      return;
    }
    var points = [];
    var marks = [];
    state.results.architectures.forEach(function (arch) {
      arch.configs.forEach(function (cfg) {
        if (!cfg.complete) return;
        points.push([cfg.paid_work_error, cfg.leisure_social_error]);
      });
    });
    if (!points.length) {
      host.innerHTML = "";
      note.textContent = "没有完整的配置均值可画。";
      return;
    }
    var xs = points.map(function (p) { return p[0]; });
    var ys = points.map(function (p) { return p[1]; });
    var xLo = Math.min.apply(null, xs), xHi = Math.max.apply(null, xs);
    var yLo = Math.min.apply(null, ys), yHi = Math.max.apply(null, ys);
    var gainSet = {};
    state.results.architectures.forEach(function (arch) {
      (arch.gain || []).forEach(function (p) { gainSet[p.join("|")] = true; });
    });
    state.results.architectures.forEach(function (arch) {
      var color = COLORS[arch.arch] || "#3a6ea5";
      if ((arch.frontier || []).length > 1) {
        marks.push({ kind: "line", color: color, points: arch.frontier });
      }
      arch.configs.forEach(function (cfg) {
        if (!cfg.complete) return;
        var key = cfg.paid_work_error + "|" + cfg.leisure_social_error;
        marks.push({
          x: cfg.paid_work_error, y: cfg.leisure_social_error, color: color,
          front: (arch.frontier || []).some(function (p) {
            return p[0] === cfg.paid_work_error && p[1] === cfg.leisure_social_error;
          }),
          gain: !!gainSet[key],
          label: arch.arch + " / " + cfg.config
            + "（" + fmt(cfg.paid_work_error) + ", " + fmt(cfg.leisure_social_error) + "）"
            + (gainSet[key] ? " —— A(M1|M0) 新增可达点" : ""),
        });
      });
    });
    host.innerHTML = svgScatter(marks, xLo, xHi, yLo, yHi);
    var gainCount = Object.keys(gainSet).length;
    note.textContent = "空心=配置均值，实心=非支配点，虚线=前沿，菱形=A(M1|M0) 相对参照架构（"
      + esc(state.results.reference_architecture) + "）新增可达点（" + gainCount + " 个）。"
      + (state.results.diagnostics && !state.results.diagnostics.frontier_valid
        ? " 注意：frontier_valid=false，有腿被门槛排除，比较不作数。" : "");
  }

  function renderDiagnostics() {
    var host = el("mcfDiagnostics");
    if (!state.results) { host.innerHTML = '<p class="rw-hint">—</p>'; return; }
    var d = state.results.diagnostics || {};
    var lines = [];
    lines.push("<div><strong>frontier_valid：</strong>" + (d.frontier_valid
      ? "true（所有腿都过了门槛）"
      : '<span class="pw-subtle">false — 有腿被排除，本组比较不作数</span>') + "</div>");
    if (d.fixture) lines.push("<div><strong>数据来源：</strong>fixture（合成数据）</div>");
    var noise = d.noise_domination || {};
    Object.keys(noise).forEach(function (axis) {
      lines.push("<div><strong>噪声支配 · " + esc(axis) + "：</strong>"
        + (noise[axis] ? "种子间波动 ≥ 配置间差异，分不出架构" : "配置间差异大于种子噪声") + "</div>");
    });
    (d.tuning_limit_configs || []).forEach(function (item) {
      lines.push("<div><strong>参数顶到边界：</strong>" + esc(item.arch + "/" + item.config + " · "
        + item.param + " = " + item.value) + "（调到这里仍未胜过参照）</div>");
    });
    (d.excluded || []).forEach(function (item) {
      lines.push("<div><strong>排除：</strong>" + esc(item.config + "/seed " + item.seed) + " — "
        + esc(item.reason) + "</div>");
    });
    (d.incomplete || []).forEach(function (item) {
      lines.push("<div><strong>不完整：</strong>" + esc(item.arch + "/" + item.config) + " 缺种子 "
        + esc((item.missing_seeds || []).join(", ")) + "</div>");
    });
    (d.failed_legs || []).forEach(function (item) {
      lines.push("<div><strong>失败腿：</strong>" + esc(item.arch + "/" + item.config_id + "/seed "
        + item.seed) + " — " + esc(item.status + " " + (item.stop_reason || "")) + "</div>");
    });
    host.innerHTML = lines.join("") || '<p class="rw-hint">没有诊断信息。</p>';
  }

  /* ------------------------------------------------------------ export */

  async function loadReport() {
    if (!state.current) return;
    var body = await api("/api/mcf/experiment/" + state.current.id + "/report");
    el("mcfReportOut").textContent = body.markdown || "";
  }

  async function downloadExport() {
    if (!state.current) return;
    var body = await api("/api/mcf/experiment/" + state.current.id + "/export");
    var blob = new Blob([JSON.stringify(body, null, 2)], { type: "application/json" });
    var link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "mcf-" + state.current.id + ".json";
    link.click();
    URL.revokeObjectURL(link.href);
  }

  /* ------------------------------------------------------------ init */

  function boot() {
    if (!el("rwPaneMcf")) return;
    document.querySelectorAll(".rw-tab").forEach(function (tab) {
      tab.addEventListener("click", function () {
        showTab(tab.getAttribute("data-tab"));
      });
    });
    el("mcfRefresh").addEventListener("click", function () {
      loadExperiments().catch(function (error) { state.error = error.message; });
    });
    el("mcfHistory").addEventListener("change", function (event) {
      openExperiment(event.target.value).catch(function (error) {
        state.error = error.message;
        el("mcfFormError").textContent = error.message;
      });
    });
    el("mcfCreate").addEventListener("click", function () {
      createExperiment().catch(function (error) {
        el("mcfFormError").textContent = error.message;
      });
    });
    el("mcfLaunch").addEventListener("click", launch);
    el("mcfReport").addEventListener("click", function () {
      loadReport().catch(function (error) { el("mcfFormError").textContent = error.message; });
    });
    el("mcfExport").addEventListener("click", function () {
      downloadExport().catch(function (error) { el("mcfFormError").textContent = error.message; });
    });
    loadBands().catch(function () { renderBands(); });
    loadExperiments().catch(function () { });
    var query = new URLSearchParams(window.location.search);
    var tab = query.get("tab");
    if (tab === "mcf") {
      showTab("mcf");
      var reopen = query.get("experiment");
      if (reopen) openExperiment(reopen).catch(function () { });
    }
  }

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", boot);
    } else {
      boot();
    }
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { __state: state };
  }
})();

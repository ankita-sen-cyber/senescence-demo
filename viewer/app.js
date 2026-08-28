/* Senescence results viewer */
(function () {
  const D = window.DATA;
  const $ = (s) => document.querySelector(s);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  /* ---------- theme ---------- */
  const toggle = document.querySelector('[data-theme-toggle]');
  const root = document.documentElement;
  let mode = matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'light';
  const sun = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>';
  const moon = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
  function applyTheme() {
    root.setAttribute('data-theme', mode);
    toggle.innerHTML = mode === 'dark' ? sun : moon;
    toggle.setAttribute('aria-label', 'Switch to ' + (mode === 'dark' ? 'light' : 'dark') + ' mode');
  }
  applyTheme();
  toggle.addEventListener('click', () => { mode = mode === 'dark' ? 'light' : 'dark'; applyTheme(); restyleCharts(); });

  /* ---------- KPIs ---------- */
  $('#kpi-recovered').textContent = D.kt_stats.recovered + ' / ' + D.kt_stats.total;
  $('#kpi-targets').textContent = D.top_targets.length;
  $('#kpi-sig').textContent = D.volcano_stats.iter1_sig.toLocaleString('en-US');
  $('#kpi-loo').textContent = Math.round(D.loo_mean * 100) + '%';

  /* ---------- failure cards ---------- */
  $('#f-cond').textContent = D.failure.f_condition.toFixed(2);
  $('#f-cell').textContent = D.failure.f_cell_line.toFixed(2);
  requestAnimationFrame(() => {
    const max = Math.max(D.failure.f_condition, D.failure.f_cell_line);
    $('#f-cond-bar').style.width = (D.failure.f_condition / max) * 100 + '%';
    $('#f-cell-bar').style.width = (D.failure.f_cell_line / max) * 100 + '%';
  });

  /* ---------- Chart defaults ---------- */
  Chart.defaults.font.family = css('--font-body') || 'sans-serif';
  Chart.defaults.font.size = 12;
  const charts = [];
  function baseColors() {
    return {
      text: css('--color-text-muted'),
      grid: css('--chart-grid'),
      nonsig: css('--chart-nonsig'),
      up: css('--color-inhibit'),
      down: css('--color-activate'),
      primary: css('--color-primary'),
      gold: css('--color-gold'),
      success: css('--color-success'),
      surface: css('--color-surface'),
      border: css('--color-border'),
      textStrong: css('--color-text'),
    };
  }
  function tooltipStyle(c) {
    return {
      backgroundColor: c.surface, titleColor: c.textStrong, bodyColor: c.text,
      borderColor: c.border, borderWidth: 1, cornerRadius: 8, padding: 10,
      titleFont: { family: "'IBM Plex Mono', monospace", weight: '600' },
    };
  }

  /* ---------- Volcano ---------- */
  const KNOWN = new Set(D.validation.map((v) => v.symbol));
  let volcanoIter = 1;
  let volcanoChart;

  function volcanoDatasets() {
    const c = baseColors();
    const pts = D['volcano_iter' + volcanoIter];
    const up = [], down = [], ns = [], known = [];
    for (const [x, y, sym] of pts) {
      const p = { x, y, sym };
      if (KNOWN.has(sym)) { known.push(p); continue; }
      const sig = y > -Math.log10(0.05);
      if (!sig) ns.push(p);
      else if (x > 0) up.push(p);
      else down.push(p);
    }
    return [
      { label: 'Not significant', data: ns, backgroundColor: c.nonsig, pointRadius: 1.3, order: 4 },
      { label: 'Up in senescent (FDR < 0.05)', data: up, backgroundColor: hexA(c.up, 0.45), pointRadius: 2.1, order: 3 },
      { label: 'Down in senescent (FDR < 0.05)', data: down, backgroundColor: hexA(c.down, 0.45), pointRadius: 2.1, order: 2 },
      { label: 'Known senescence marker', data: known, backgroundColor: c.gold, borderColor: c.textStrong, borderWidth: 1, pointRadius: 5, pointHoverRadius: 7, order: 1 },
    ];
  }
  function hexA(hex, a) {
    hex = hex.replace('#', '');
    if (hex.length === 3) hex = hex.split('').map((ch) => ch + ch).join('');
    const n = parseInt(hex, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  const markerLabelPlugin = {
    id: 'markerLabels',
    afterDatasetsDraw(chart) {
      const ds = chart.data.datasets[3];
      const meta = chart.getDatasetMeta(3);
      if (!ds || !meta) return;
      const ctx = chart.ctx;
      const c = baseColors();
      ctx.save();
      ctx.font = "600 10px 'IBM Plex Mono', monospace";
      ctx.fillStyle = c.textStrong;
      const drawn = [];
      ds.data.forEach((p, i) => {
        if (p.y < 2 && Math.abs(p.x) < 0.3) return; // label only notable markers
        const el = meta.data[i];
        if (!el) return;
        const x = el.x + 7, y = el.y + 3;
        if (drawn.some(([dx, dy]) => Math.abs(dx - x) < 42 && Math.abs(dy - y) < 12)) return;
        drawn.push([x, y]);
        ctx.fillText(p.sym, x, y);
      });
      ctx.restore();
    },
  };

  function buildVolcano() {
    const c = baseColors();
    if (volcanoChart) volcanoChart.destroy();
    volcanoChart = new Chart($('#volcanoChart'), {
      type: 'scatter',
      data: { datasets: volcanoDatasets() },
      plugins: [markerLabelPlugin],
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
          x: { title: { display: true, text: 'log\u2082 fold change (senescent vs. young)', color: c.text }, grid: { color: c.grid }, ticks: { color: c.text } },
          y: { title: { display: true, text: '\u2212log\u2081\u2080 (adjusted p)', color: c.text }, grid: { color: c.grid }, ticks: { color: c.text } },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            ...tooltipStyle(c),
            filter: (item) => item.datasetIndex !== 0,
            callbacks: {
              title: (items) => items[0]?.raw.sym || '',
              label: (item) => ` log\u2082FC ${item.raw.x}  \u00b7  \u2212log\u2081\u2080(FDR) ${item.raw.y}`,
            },
          },
        },
      },
    });
    charts.push(volcanoChart);
    // legend
    const leg = $('#volcano-legend');
    leg.innerHTML = '';
    [['Up in senescent', hexA(c.up, 0.7)], ['Down in senescent', hexA(c.down, 0.7)], ['Known marker', c.gold], ['Not significant', c.nonsig]]
      .forEach(([t, col]) => { leg.innerHTML += `<span><span class="sw" style="background:${col}"></span>${t}</span>`; });
    $('#volcano-note').textContent = volcanoIter === 1
      ? `Batch-aware analysis: ${D.volcano_stats.iter1_sig.toLocaleString('en-US')} of ${D.volcano_stats.iter1_genes.toLocaleString('en-US')} genes significant at FDR < 0.05. Correcting for cell line nearly doubles detected signal versus the naive pass and recovers markers like TP53.`
      : `Naive pooled analysis: ${D.volcano_stats.iter0_sig.toLocaleString('en-US')} of ${D.volcano_stats.iter0_genes.toLocaleString('en-US')} genes significant at FDR < 0.05 — but cell-line effects dominate, so many calls are unreliable.`;
  }
  buildVolcano();
  document.querySelectorAll('[data-iter]').forEach((b) =>
    b.addEventListener('click', () => {
      document.querySelectorAll('[data-iter]').forEach((x) => { x.classList.remove('active'); x.setAttribute('aria-selected', 'false'); });
      b.classList.add('active'); b.setAttribute('aria-selected', 'true');
      volcanoIter = +b.dataset.iter;
      buildVolcano();
    })
  );

  /* ---------- Targets table ---------- */
  const maxScore = Math.max(...D.top_targets.map((t) => Math.abs(t.score)));
  function renderTargets(filter) {
    const rows = D.top_targets
      .map((t, i) => ({ ...t, rank: i + 1 }))
      .filter((t) => filter === 'all' || t.direction === filter);
    $('#targets-body').innerHTML = rows.map((t) => `
      <tr>
        <td class="rank">${String(t.rank).padStart(2, '0')}</td>
        <td><span class="gene">${t.symbol}</span></td>
        <td><span class="badge ${t.direction}">${t.direction === 'inhibit' ? 'Inhibit \u00b7 senolytic-style' : 'Activate \u00b7 restore'}</span></td>
        <td class="num">${t.logFC.toFixed(3)}</td>
        <td class="num">${t.padj.toExponential(1)}</td>
        <td class="num">${t.score.toFixed(2)}</td>
        <td><div class="score-bar"><div class="score-fill ${t.direction}" style="width:${(Math.abs(t.score) / maxScore) * 100}%"></div></div></td>
      </tr>`).join('');
  }
  renderTargets('all');
  document.querySelectorAll('[data-dir]').forEach((b) =>
    b.addEventListener('click', () => {
      document.querySelectorAll('[data-dir]').forEach((x) => { x.classList.remove('active'); x.setAttribute('aria-selected', 'false'); });
      b.classList.add('active'); b.setAttribute('aria-selected', 'true');
      renderTargets(b.dataset.dir);
    })
  );

  /* ---------- Pathways ---------- */
  let pathwayChart;
  function buildPathways() {
    const c = baseColors();
    if (pathwayChart) pathwayChart.destroy();
    const top = D.pathways.slice(0, 14).sort((a, b) => b.nes - a.nes);
    pathwayChart = new Chart($('#pathwayChart'), {
      type: 'bar',
      data: {
        labels: top.map((p) => p.term.length > 26 ? p.term.slice(0, 25) + '\u2026' : p.term),
        datasets: [{
          data: top.map((p) => p.nes),
          backgroundColor: top.map((p) => (p.nes > 0 ? hexA(c.up, 0.75) : hexA(c.down, 0.75))),
          borderRadius: 4, fdr: top.map((p) => p.fdr), fullTerms: top.map((p) => p.term),
        }],
      },
      options: {
        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'Normalized enrichment score (NES)', color: c.text }, grid: { color: c.grid }, ticks: { color: c.text } },
          y: { grid: { display: false }, ticks: { color: c.textStrong, font: { size: 11 }, autoSkip: false } },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            ...tooltipStyle(c),
            callbacks: {
              title: (items) => items[0].dataset.fullTerms[items[0].dataIndex],
              label: (item) => ` NES ${item.raw.toFixed(2)} \u00b7 FDR q ${item.dataset.fdr[item.dataIndex].toExponential(1)}`,
            },
          },
        },
      },
    });
    charts.push(pathwayChart);
  }
  buildPathways();

  /* ---------- TFs ---------- */
  const MODE_LABEL = {
    'repression-targets-down': 'Repressive program engaged',
    'activation-targets-up': 'Activating program engaged',
    'activation-targets-down': 'Activating program lost',
    'repression-targets-up': 'Repression released',
  };
  let tfChart;
  function buildTF() {
    const c = baseColors();
    if (tfChart) tfChart.destroy();
    const top = D.tfs.slice(0, 12);
    const modeColor = (m) => (m.startsWith('repression') ? hexA(c.primary, 0.8) : hexA(c.gold, 0.85));
    tfChart = new Chart($('#tfChart'), {
      type: 'bar',
      data: {
        labels: top.map((t) => t.tf),
        datasets: [{ data: top.map((t) => t.score), backgroundColor: top.map((t) => modeColor(t.mode)), borderRadius: 4, meta: top }],
      },
      options: {
        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'Regulon activity score', color: c.text }, grid: { color: c.grid }, ticks: { color: c.text } },
          y: { grid: { display: false }, ticks: { color: c.textStrong, font: { family: "'IBM Plex Mono', monospace", size: 11 }, autoSkip: false } },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            ...tooltipStyle(c),
            callbacks: {
              label: (item) => {
                const m = item.dataset.meta[item.dataIndex];
                return [` score ${m.score.toFixed(2)} \u00b7 ${MODE_LABEL[m.mode] || m.mode}`, ` regulon: ${m.n_act} activating / ${m.n_rep} repressive edges`];
              },
            },
          },
        },
      },
    });
    charts.push(tfChart);
    const c2 = baseColors();
    $('#tf-legend').innerHTML =
      `<span><span class="sw" style="background:${hexA(c2.primary, 0.8)}"></span>Repression-linked mode</span>` +
      `<span><span class="sw" style="background:${hexA(c2.gold, 0.85)}"></span>Activation-linked mode</span>`;
  }
  buildTF();

  /* ---------- Validation markers ---------- */
  $('#val-count').textContent = D.val_stats.iter1_recovered + ' / ' + D.val_stats.total;
  const grid = $('#marker-grid');
  grid.innerHTML = D.validation.map((v) => {
    const fixed = !v.rec0 && v.rec1;
    return `<div class="marker ${fixed ? 'fixed' : ''}" role="listitem" title="${v.symbol}: expected ${v.expected} in senescent cells">
      <div class="marker-top"><span class="gene">${v.symbol}</span>
        <span class="dots">
          <span class="dot ${v.rec0 ? 'ok' : 'miss'}" title="Iteration 0 ${v.rec0 ? 'recovered' : 'missed'}"></span>
          <span class="dot ${v.rec1 ? 'ok' : 'miss'}" title="Iteration 1 ${v.rec1 ? 'recovered' : 'missed'}"></span>
        </span></div>
      <span class="marker-exp">expected ${v.expected === 'up' ? '\u2191 up' : '\u2193 down'}${fixed ? ' \u00b7 fixed by iter 1' : ''}</span>
    </div>`;
  }).join('');
  $('#marker-legend').innerHTML =
    '<span><span class="dot ok" style="display:inline-block;vertical-align:-1px;margin-right:6px"></span>recovered</span>' +
    '<span><span class="dot miss" style="display:inline-block;vertical-align:-1px;margin-right:6px"></span>missed</span>' +
    '<span>dots = iteration 0, iteration 1 \u00b7 green-bordered cards were fixed by batch correction</span>';

  /* ---------- LOO ---------- */
  let looChart;
  function buildLOO() {
    const c = baseColors();
    if (looChart) looChart.destroy();
    looChart = new Chart($('#looChart'), {
      type: 'bar',
      data: {
        labels: D.loo.map((l) => l.cell_line),
        datasets: [{ data: D.loo.map((l) => l.acc), backgroundColor: hexA(c.primary, 0.8), borderRadius: 6, maxBarThickness: 64 }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          y: { min: 0, max: 1, title: { display: true, text: 'Held-out accuracy', color: c.text }, grid: { color: c.grid }, ticks: { color: c.text, callback: (v) => Math.round(v * 100) + '%' } },
          x: { grid: { display: false }, ticks: { color: c.textStrong, font: { family: "'IBM Plex Mono', monospace" } } },
        },
        plugins: {
          legend: { display: false },
          tooltip: { ...tooltipStyle(c), callbacks: { label: (i) => ' accuracy ' + Math.round(i.raw * 100) + '%' } },
        },
      },
    });
    charts.push(looChart);
  }
  buildLOO();
  const minAcc = Math.min(...D.loo.map((l) => l.acc));
  const hardest = D.loo.filter((l) => l.acc === minAcc).map((l) => l.cell_line);
  $('#loo-note').textContent = `Mean accuracy ${Math.round(D.loo_mean * 100)}% across five held-out cell lines. ${hardest.join(' and ')} ${hardest.length > 1 ? 'are' : 'is'} hardest (${Math.round(minAcc * 100)}%) — a candidate for targeted follow-up rather than a hidden failure.`;

  /* ---------- restyle on theme change ---------- */
  function restyleCharts() { buildVolcano(); buildPathways(); buildTF(); buildLOO(); }
})();

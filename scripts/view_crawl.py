"""Render a crawl log JSON as a standalone HTML viewer.

Usage:
    python scripts/view_crawl.py <path-to-log.json> [--open] [-o out.html]

Produces a single HTML file with the log embedded — no server, no deps.
"""
import argparse
import html
import json
import os
import sys
import webbrowser
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Crawl Viewer — __TITLE__</title>
<style>
:root {
  --bg: #0e0f13;
  --fg: #e6e7ea;
  --dim: #8a8f99;
  --panel: #16181f;
  --panel2: #1c1f28;
  --border: #2a2e39;
  --accent: #7aa2f7;
  --refuse: #f7768e;
  --comply: #9ece6a;
  --inconclusive: #e0af68;
  --chip: #2a2e39;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
a, a:visited { color: var(--accent); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Cascadia Code", monospace; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
header { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
  border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 20px; }
header h1 { font-size: 18px; margin: 0; font-weight: 600; }
header .meta { color: var(--dim); font-size: 12px; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px; margin-bottom: 20px; }
.kpi { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 12px 14px; }
.kpi .label { color: var(--dim); font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.05em; }
.kpi .value { font-size: 22px; font-weight: 600; margin-top: 4px;
  font-variant-numeric: tabular-nums; }
.section-head { display: flex; align-items: baseline; justify-content: space-between;
  gap: 12px; margin: 24px 0 10px; }
.section-head h2 { font-size: 14px; margin: 0; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em; color: var(--dim); }
.section-head .count { color: var(--dim); font-size: 12px; font-variant-numeric: tabular-nums; }
.toolbar { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
.toolbar input, .toolbar select {
  background: var(--panel); color: var(--fg); border: 1px solid var(--border);
  border-radius: 4px; padding: 6px 10px; font: inherit; }
.toolbar input[type="search"] { min-width: 260px; }
.toolbar button {
  background: var(--panel); color: var(--fg); border: 1px solid var(--border);
  border-radius: 4px; padding: 6px 12px; font: inherit; cursor: pointer; }
.toolbar button:hover { background: var(--panel2); }
.toolbar button.active { border-color: var(--accent); color: var(--accent); }

.cluster-list { display: flex; flex-direction: column; gap: 6px; }
.cluster { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  overflow: hidden; }
.cluster summary { list-style: none; cursor: pointer; padding: 10px 14px;
  display: flex; align-items: center; gap: 10px; user-select: none; }
.cluster summary::-webkit-details-marker { display: none; }
.cluster summary::before {
  content: '▸'; color: var(--dim); font-size: 10px; width: 10px; display: inline-block;
  transition: transform 0.1s; }
.cluster[open] summary::before { transform: rotate(90deg); }
.cluster summary:hover { background: var(--panel2); }
.cluster .title { flex: 1; font-weight: 500; }
.cluster .raw { color: var(--dim); font-size: 12px; margin-left: 4px; }
.badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; background: var(--chip);
  color: var(--fg); font-variant-numeric: tabular-nums; }
.badge.refuse { background: rgba(247, 118, 142, 0.15); color: var(--refuse); }
.badge.comply { background: rgba(158, 206, 106, 0.15); color: var(--comply); }
.badge.inconclusive { background: rgba(224, 175, 104, 0.15); color: var(--inconclusive); }
.cluster-body { padding: 0 14px 14px 28px; border-top: 1px solid var(--border); }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 6px 12px;
  margin: 10px 0; color: var(--dim); font-size: 12px; }
.kv dt { font-weight: 500; }
.kv dd { margin: 0; color: var(--fg); }
.members { margin-top: 8px; }
.member { padding: 6px 0; border-top: 1px dashed var(--border); }
.member:first-child { border-top: 0; }
.member .member-raw { color: var(--fg); }
.member .member-meta { color: var(--dim); font-size: 11px; }
.rc-block { margin-top: 10px; background: var(--panel2); border-radius: 4px;
  border: 1px solid var(--border); padding: 10px; }
.rc-block h4 { margin: 0 0 6px; font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--dim); font-weight: 600; }
.rc-row { padding: 6px 0; border-top: 1px dashed var(--border); }
.rc-row:first-child { border-top: 0; }
.rc-q { font-weight: 500; font-size: 12px; }
.rc-r { margin-top: 4px; font-size: 12px; color: var(--dim); white-space: pre-wrap;
  max-height: 180px; overflow-y: auto; background: var(--panel); padding: 6px 8px;
  border-radius: 3px; border: 1px solid var(--border); }
.rc-r.refuse { border-left: 3px solid var(--refuse); }
.rc-r.comply { border-left: 3px solid var(--comply); }
.rc-empty { color: var(--inconclusive); font-style: italic; }

.chart { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 14px; margin-bottom: 16px; }
.chart h3 { font-size: 12px; margin: 0 0 10px; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--dim); font-weight: 600; }
.chart .bars { display: flex; gap: 12px; align-items: flex-end; height: 120px; }
.chart .step { flex: 1; text-align: center; display: flex; flex-direction: column;
  justify-content: flex-end; gap: 2px; }
.chart .bar { width: 100%; min-height: 2px; border-radius: 2px 2px 0 0; }
.chart .bar.all { background: var(--accent); opacity: 0.6; }
.chart .bar.deduped { background: var(--accent); }
.chart .bar.refusal { background: var(--refuse); }
.chart .step-label { font-size: 11px; color: var(--dim); margin-top: 4px; }
.chart .legend { display: flex; gap: 14px; font-size: 11px; color: var(--dim);
  margin-top: 10px; }
.chart .legend .dot { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin-right: 4px; vertical-align: middle; }

.no-results { color: var(--dim); padding: 14px; text-align: center;
  border: 1px dashed var(--border); border-radius: 6px; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1 id="title">__TITLE__</h1>
    <span class="meta" id="meta"></span>
  </header>

  <div class="kpi-grid" id="kpis"></div>
  <div class="chart" id="chart"></div>

  <div class="section-head">
    <h2>Clusters</h2>
    <span class="count" id="cluster-count"></span>
  </div>
  <div class="toolbar">
    <input type="search" id="q" placeholder="Filter (summary, raw, member text)..." />
    <button data-filter="all" class="active">All</button>
    <button data-filter="refuse">Refusals</button>
    <button data-filter="comply">Compliant</button>
    <button data-filter="inconclusive">Inconclusive</button>
    <select id="sort">
      <option value="size">Sort: size</option>
      <option value="summary">Sort: summary</option>
      <option value="cluster_idx">Sort: cluster idx</option>
    </select>
    <button id="expand-all">Expand all</button>
    <button id="collapse-all">Collapse all</button>
  </div>
  <div class="cluster-list" id="clusters"></div>
</div>

<script id="crawl-data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('crawl-data').textContent);

function el(tag, attrs, children) {
  const e = document.createElement(tag);
  if (attrs) for (const k in attrs) {
    if (k === 'class') e.className = attrs[k];
    else if (k === 'html') e.innerHTML = attrs[k];
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), attrs[k]);
    else if (attrs[k] != null) e.setAttribute(k, attrs[k]);
  }
  (children || []).forEach(c => {
    if (c == null) return;
    if (typeof c === 'string') e.appendChild(document.createTextNode(c));
    else e.appendChild(c);
  });
  return e;
}

function verdict(t) {
  if (t.refusal_check_inconclusive) return 'inconclusive';
  if (t.is_refusal === true) return 'refuse';
  if (t.is_refusal === false) return 'comply';
  return 'inconclusive';
}

function verdictLabel(v) {
  return { refuse: 'REFUSAL', comply: 'COMPLIANT', inconclusive: 'INCONCLUSIVE' }[v] || v;
}

// Header
document.getElementById('title').textContent = DATA.title;
const metaBits = [];
const cfg = DATA.log.config || {};
if (cfg.model) {
  if (cfg.model.target_model) metaBits.push('target=' + cfg.model.target_model);
  if (cfg.model.summarization_model) metaBits.push('group=' + cfg.model.summarization_model);
}
if (cfg.crawler) {
  if (cfg.crawler.num_crawl_steps != null) metaBits.push('steps=' + cfg.crawler.num_crawl_steps);
  if (cfg.crawler.generation_batch_size != null) metaBits.push('batch=' + cfg.crawler.generation_batch_size);
  if (cfg.crawler.semantic_group_batch_size != null) metaBits.push('group_batch=' + cfg.crawler.semantic_group_batch_size);
  if (cfg.crawler.prompt_languages) metaBits.push('langs=' + cfg.crawler.prompt_languages.join(','));
}
document.getElementById('meta').textContent = metaBits.join('  ·  ');

// KPIs
const qstats = (DATA.log.queue && DATA.log.queue.stats) || {};
const cstats = (DATA.log.stats && DATA.log.stats.current_metrics) || {};
const kpis = [
  ['Refusal Heads', qstats.num_head_refusal_topics ?? '—'],
  ['Total Heads', qstats.num_head_topics ?? '—'],
  ['Total Topics', qstats.num_total_topics ?? '—'],
  ['Steps Completed', cstats.current_step ?? '—'],
  ['Refusal Rate', cstats.avg_refusal_rate != null ? (cstats.avg_refusal_rate * 100).toFixed(1) + '%' : '—'],
];
const kpiGrid = document.getElementById('kpis');
kpis.forEach(([label, value]) => {
  kpiGrid.appendChild(el('div', {class: 'kpi'}, [
    el('div', {class: 'label'}, [label]),
    el('div', {class: 'value'}, [String(value)]),
  ]));
});

// Chart: per-step history
const history = (DATA.log.stats && DATA.log.stats.history) || {};
const stepCount = Math.max(
  (history.all_per_step || []).length,
  (history.deduped_per_step || []).length,
  (history.refusal_per_step || []).length,
);
const chartBox = document.getElementById('chart');
if (stepCount > 0) {
  const series = [
    ['Extracted (all)', 'all', history.all_per_step || []],
    ['After grouping', 'deduped', history.deduped_per_step || []],
    ['Refusals', 'refusal', history.refusal_per_step || []],
  ];
  const maxV = Math.max(1, ...series.flatMap(([,,a]) => a));
  chartBox.appendChild(el('h3', null, ['Per-step cadence']));
  const bars = el('div', {class: 'bars'});
  for (let i = 0; i < stepCount; i++) {
    const col = el('div', {class: 'step'});
    series.forEach(([label, cls, arr]) => {
      const v = arr[i] || 0;
      const pct = (v / maxV) * 100;
      const bar = el('div', {class: 'bar ' + cls, title: label + ': ' + v,
        style: 'height: ' + pct + '%'});
      col.appendChild(bar);
    });
    col.appendChild(el('div', {class: 'step-label'}, ['step ' + i]));
    bars.appendChild(col);
  }
  chartBox.appendChild(bars);
  const legend = el('div', {class: 'legend'}, series.map(([label, cls]) =>
    el('span', null, [
      el('span', {class: 'dot', style: 'background: ' +
        (cls === 'refusal' ? 'var(--refuse)' : 'var(--accent)') +
        (cls === 'all' ? '; opacity: 0.6' : '')}),
      label,
    ])));
  chartBox.appendChild(legend);
} else {
  chartBox.style.display = 'none';
}

// Cluster render
const heads = (DATA.log.queue && DATA.log.queue.topics && DATA.log.queue.topics.head_topics) || [];
const clusters = (DATA.log.queue && DATA.log.queue.topics && DATA.log.queue.topics.cluster_topics) || [];
const state = { filter: 'all', q: '', sort: 'size' };

function getMembers(clusterIdx) {
  // cluster_topics is indexed by cluster_idx (list position)
  const arr = clusters[clusterIdx] || [];
  return arr.filter(t => !t.is_head);
}

function clusterSize(t) {
  return Math.max(
    t.cluster_member_count || 0,
    1 + getMembers(t.cluster_idx).length,
  );
}

function clusterMatches(t) {
  const v = verdict(t);
  if (state.filter !== 'all' && v !== state.filter) return false;
  if (!state.q) return true;
  const needle = state.q.toLowerCase();
  const hay = [
    t.summary || '', t.raw || '', t.english || '', t.chinese || '',
  ].join(' ').toLowerCase();
  if (hay.includes(needle)) return true;
  return getMembers(t.cluster_idx).some(m => {
    const h = [m.summary || '', m.raw || '', m.english || '', m.chinese || ''].join(' ').toLowerCase();
    return h.includes(needle);
  });
}

function sortHeads(arr) {
  const s = state.sort;
  const cp = arr.slice();
  if (s === 'size') cp.sort((a, b) => clusterSize(b) - clusterSize(a));
  else if (s === 'summary') cp.sort((a, b) => (a.summary || a.raw || '').localeCompare(b.summary || b.raw || ''));
  else if (s === 'cluster_idx') cp.sort((a, b) => (a.cluster_idx ?? 0) - (b.cluster_idx ?? 0));
  return cp;
}

function renderClusterBody(t) {
  const body = el('div', {class: 'cluster-body'});
  const kv = el('dl', {class: 'kv'});
  const addKV = (k, v) => {
    if (v == null || v === '') return;
    kv.appendChild(el('dt', null, [k]));
    kv.appendChild(el('dd', null, [String(v)]));
  };
  addKV('raw', t.raw);
  if ((t.english || '') !== (t.raw || '')) addKV('english', t.english);
  if (t.chinese) addKV('chinese', t.chinese);
  addKV('cluster_idx', t.cluster_idx);
  addKV('parent_id', t.parent_id);
  addKV('member_count', clusterSize(t));
  body.appendChild(kv);

  const members = getMembers(t.cluster_idx);
  if (members.length > 0) {
    const block = el('div', {class: 'members'});
    block.appendChild(el('h4', {class: 'rc-q', style: 'color: var(--dim); font-size: 11px; text-transform: uppercase;'}, [`Members (${members.length})`]));
    members.forEach(m => {
      block.appendChild(el('div', {class: 'member'}, [
        el('div', {class: 'member-raw'}, [m.raw || m.english || '(empty)']),
        el('div', {class: 'member-meta'}, ['id=' + (m.id ?? '?') + '  ·  parent_id=' + (m.parent_id ?? '?')]),
      ]));
    });
    body.appendChild(block);
  }

  const queries = t.refusal_check_queries || [];
  const responses = t.refusal_check_responses || [];
  if (queries.length || responses.length) {
    const rc = el('div', {class: 'rc-block'});
    rc.appendChild(el('h4', null, ['Refusal-check evidence']));
    const n = Math.max(queries.length, responses.length);
    for (let i = 0; i < n; i++) {
      const q = queries[i] || '';
      const r = responses[i] || '';
      const empty = !r.trim();
      // crude heuristic for per-response colouring (refusal preambles)
      const isRefuseHeur = /(?:I('m| cannot| can't| won't)|Sorry|I refuse|I am (unable|not able))/i.test(r.slice(0, 80));
      const cls = empty ? '' : (isRefuseHeur ? 'refuse' : 'comply');
      const row = el('div', {class: 'rc-row'}, [
        el('div', {class: 'rc-q'}, [q || '(no query)']),
        empty
          ? el('div', {class: 'rc-r rc-empty'}, ['(empty response)'])
          : el('div', {class: 'rc-r ' + cls}, [r]),
      ]);
      rc.appendChild(row);
    }
    body.appendChild(rc);
  }

  return body;
}

function renderCluster(t) {
  const v = verdict(t);
  const size = clusterSize(t);
  const summary = el('summary', null, [
    el('span', {class: 'badge ' + v}, [verdictLabel(v)]),
    el('span', {class: 'badge'}, ['×' + size]),
    el('span', {class: 'title'}, [t.summary || t.raw || '(no summary)']),
    t.summary && t.raw && t.summary !== t.raw
      ? el('span', {class: 'raw'}, ['— ' + t.raw])
      : null,
  ]);
  const details = el('details', {class: 'cluster'}, [summary]);
  // lazy-render body on first open
  let built = false;
  details.addEventListener('toggle', () => {
    if (details.open && !built) {
      details.appendChild(renderClusterBody(t));
      built = true;
    }
  });
  return details;
}

function render() {
  const listEl = document.getElementById('clusters');
  listEl.innerHTML = '';
  const filtered = sortHeads(heads.filter(clusterMatches));
  document.getElementById('cluster-count').textContent =
    filtered.length + ' shown / ' + heads.length + ' total';
  if (filtered.length === 0) {
    listEl.appendChild(el('div', {class: 'no-results'}, ['No clusters match the current filter.']));
    return;
  }
  filtered.forEach(t => listEl.appendChild(renderCluster(t)));
}

// Wire up controls
document.getElementById('q').addEventListener('input', e => {
  state.q = e.target.value.trim();
  render();
});
document.querySelectorAll('[data-filter]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-filter]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.filter = btn.getAttribute('data-filter');
    render();
  });
});
document.getElementById('sort').addEventListener('change', e => {
  state.sort = e.target.value;
  render();
});
document.getElementById('expand-all').addEventListener('click', () => {
  document.querySelectorAll('.cluster').forEach(d => d.open = true);
});
document.getElementById('collapse-all').addEventListener('click', () => {
  document.querySelectorAll('.cluster').forEach(d => d.open = false);
});

render();
</script>
</body>
</html>
"""


def build_html(log_path: Path, log_data: dict) -> str:
    title = log_path.stem
    # Embed JSON safely inside a <script type="application/json"> tag.
    # The only sequence we need to guard against is "</script>" in the payload.
    payload = json.dumps({"title": title, "log": log_data}, ensure_ascii=False)
    payload = payload.replace("</script", "<\\/script")
    return (HTML_TEMPLATE
            .replace("__TITLE__", html.escape(title))
            .replace("__DATA__", payload))


def main() -> int:
    p = argparse.ArgumentParser(description="Render a crawl log as a standalone HTML viewer.")
    p.add_argument("log", help="Path to crawler_out_*.json")
    p.add_argument("-o", "--output", help="Output HTML path (default: alongside log)")
    p.add_argument("--open", action="store_true", help="Open in browser after writing")
    args = p.parse_args()

    log_path = Path(args.log).resolve()
    if not log_path.exists():
        print(f"error: {log_path} does not exist", file=sys.stderr)
        return 2

    with log_path.open() as f:
        log_data = json.load(f)

    out_path = Path(args.output).resolve() if args.output else log_path.with_suffix(".html")
    html_text = build_html(log_path, log_data)
    out_path.write_text(html_text, encoding="utf-8")
    print(f"wrote {out_path}")
    if args.open:
        webbrowser.open(out_path.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())

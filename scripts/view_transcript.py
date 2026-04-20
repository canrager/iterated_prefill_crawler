"""Render a crawler transcript JSONL as a chronological HTML viewer.

Shows every LLM call in order, color-coded by phase
(generate → extract → translate → group → provoke → refusal-query → judge).

Usage:
    python scripts/view_transcript.py <transcript.jsonl> [--open] [-o out.html]
"""
import argparse
import html
import json
import sys
import webbrowser
from pathlib import Path


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Transcript — __TITLE__</title>
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
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg);
  font: 13px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; }
.container { max-width: 1280px; margin: 0 auto; padding: 20px; }
header { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
  border-bottom: 1px solid var(--border); padding-bottom: 14px; margin-bottom: 18px; }
header h1 { font-size: 16px; margin: 0; font-weight: 600; }
header .meta { color: var(--dim); font-size: 12px; }

.phase-legend { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
.phase-tag { font-size: 10.5px; padding: 3px 9px; border-radius: 10px;
  text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600;
  background: var(--panel); border: 1px solid var(--border); color: var(--fg);
  font-variant-numeric: tabular-nums; cursor: pointer; user-select: none; }
.phase-tag .count { color: var(--dim); margin-left: 6px; font-weight: 400; }
.phase-tag.active { outline: 1px solid var(--accent); }
.phase-tag.disabled { opacity: 0.35; }
.phase-tag[data-phase="generate"]       { --color: #7aa2f7; }
.phase-tag[data-phase="extract"]        { --color: #bb9af7; }
.phase-tag[data-phase="translate"]      { --color: #7dcfff; }
.phase-tag[data-phase="group"]          { --color: #c0a06c; }
.phase-tag[data-phase="provoke"]        { --color: #ff9e64; }
.phase-tag[data-phase="refusal-query"]  { --color: #f7768e; }
.phase-tag[data-phase="judge"]          { --color: #9ece6a; }
.phase-tag[data-phase="other"]          { --color: #8a8f99; }
.phase-tag { border-left: 3px solid var(--color); }

.toolbar { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap;
  align-items: center; }
.toolbar input, .toolbar select {
  background: var(--panel); color: var(--fg); border: 1px solid var(--border);
  border-radius: 4px; padding: 6px 10px; font: inherit; }
.toolbar input[type="search"] { min-width: 280px; }
.toolbar button {
  background: var(--panel); color: var(--fg); border: 1px solid var(--border);
  border-radius: 4px; padding: 6px 12px; font: inherit; cursor: pointer; }
.toolbar button:hover { background: var(--panel2); }
.toolbar label { color: var(--dim); font-size: 12px; }

.timeline { display: flex; flex-direction: column; gap: 4px; }
.call { background: var(--panel); border: 1px solid var(--border); border-radius: 4px;
  overflow: hidden; border-left: 3px solid var(--color, var(--dim)); }
.call summary { list-style: none; cursor: pointer; padding: 6px 10px;
  display: flex; align-items: center; gap: 10px; user-select: none;
  font-variant-numeric: tabular-nums; }
.call summary::-webkit-details-marker { display: none; }
.call summary::before {
  content: '▸'; color: var(--dim); font-size: 10px; width: 10px; display: inline-block;
  transition: transform 0.1s; flex-shrink: 0; }
.call[open] summary::before { transform: rotate(90deg); }
.call summary:hover { background: var(--panel2); }
.call .idx { color: var(--dim); font-size: 11px; min-width: 44px; flex-shrink: 0; }
.call .phase { font-size: 10.5px; padding: 2px 7px; border-radius: 10px;
  background: rgba(255,255,255,0.06); color: var(--color, var(--dim)); font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.04em; flex-shrink: 0; min-width: 100px; text-align: center; }
.call .model { color: var(--dim); font-size: 11px; flex-shrink: 0; min-width: 150px; }
.call .gist { flex: 1; color: var(--fg); white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; font-size: 12.5px; }
.call .batch-badge { color: var(--dim); font-size: 11px; flex-shrink: 0; }
.call[data-phase="generate"]       { --color: #7aa2f7; }
.call[data-phase="extract"]        { --color: #bb9af7; }
.call[data-phase="translate"]      { --color: #7dcfff; }
.call[data-phase="group"]          { --color: #c0a06c; }
.call[data-phase="provoke"]        { --color: #ff9e64; }
.call[data-phase="refusal-query"]  { --color: #f7768e; }
.call[data-phase="judge"]          { --color: #9ece6a; }
.call[data-phase="other"]          { --color: #8a8f99; }

.body { padding: 0 12px 12px 40px; border-top: 1px solid var(--border);
  background: var(--panel); }
.body .meta { color: var(--dim); font-size: 11px; margin: 8px 0; }
.pair { margin: 10px 0; }
.pair .pair-head { font-size: 11px; color: var(--dim); text-transform: uppercase;
  letter-spacing: 0.05em; margin-bottom: 4px; font-weight: 600; }
.msgs { display: flex; flex-direction: column; gap: 4px; }
.msg { padding: 8px 10px; border-radius: 4px; white-space: pre-wrap; font-size: 12.5px;
  max-height: 320px; overflow-y: auto; border: 1px solid var(--border); }
.msg.sys { background: rgba(192, 160, 108, 0.08); border-color: rgba(192, 160, 108, 0.3); }
.msg.user { background: rgba(122, 162, 247, 0.08); border-color: rgba(122, 162, 247, 0.3); }
.msg.assistant { background: rgba(158, 206, 106, 0.08); border-color: rgba(158, 206, 106, 0.3); }
.msg .role { color: var(--dim); font-size: 10px; text-transform: uppercase;
  letter-spacing: 0.06em; margin-bottom: 3px; font-weight: 600; }
.out { padding: 8px 10px; border-radius: 4px; white-space: pre-wrap; font-size: 12.5px;
  max-height: 380px; overflow-y: auto; background: var(--panel2); border: 1px solid var(--border); }
.out.empty { color: var(--inconclusive); font-style: italic; }

.step-divider { background: var(--panel2); color: var(--accent); padding: 8px 12px;
  border-radius: 4px; font-size: 12px; font-weight: 600; border-left: 3px solid var(--accent);
  margin: 12px 0 6px; }

.no-results { color: var(--dim); padding: 14px; text-align: center;
  border: 1px dashed var(--border); border-radius: 6px; }

.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px; margin-bottom: 16px; }
.kpi { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 10px 12px; }
.kpi .label { color: var(--dim); font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.04em; }
.kpi .value { font-size: 18px; font-weight: 600; margin-top: 3px; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1 id="title">__TITLE__</h1>
    <span class="meta" id="meta"></span>
  </header>

  <div class="kpi-grid" id="kpis"></div>

  <div class="phase-legend" id="legend"></div>

  <div class="toolbar">
    <input type="search" id="q" placeholder="Filter (input/output substring)..." />
    <label><input type="checkbox" id="show-steps" checked> Show step dividers</label>
    <button id="expand-all">Expand all visible</button>
    <button id="collapse-all">Collapse all</button>
  </div>

  <div class="timeline" id="timeline"></div>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
const RAW = JSON.parse(document.getElementById('data').textContent);
const RECORDS = RAW.records;

function el(tag, attrs, children) {
  const e = document.createElement(tag);
  if (attrs) for (const k in attrs) {
    if (k === 'class') e.className = attrs[k];
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

function msgs(inputs) {
  // inputs can be: list[list[msg]]  (batch_generate_api)  OR  list[msg] (async_query_openrouter)
  if (!Array.isArray(inputs) || inputs.length === 0) return [];
  if (Array.isArray(inputs[0])) return inputs;            // already list of batches
  return [inputs];                                        // wrap single message list
}

function firstUser(inputs) {
  const lists = msgs(inputs);
  if (!lists.length) return '';
  for (const m of lists[0]) {
    if (m && m.role === 'user') return m.content || '';
  }
  return '';
}

function firstSys(inputs) {
  const lists = msgs(inputs);
  if (!lists.length) return '';
  for (const m of lists[0]) {
    if (m && m.role === 'system') return m.content || '';
  }
  return '';
}

function truncate(s, n) {
  s = (s || '').replace(/\s+/g, ' ').trim();
  return s.length > n ? s.slice(0, n) + '…' : s;
}

// --- classifier ---
function classify(r) {
  const model = (r.model || '').toLowerCase();
  const ct = r.call_type || '';
  const mt = r.max_tokens;
  if (model.includes('qwen')) return 'translate';
  if (model.includes('kimi') || model.includes('moonshot')) {
    const sys = firstSys(r.inputs);
    if (/deduplication/i.test(sys)) return 'group';
    if (/extract/i.test(sys)) return 'extract';
    return 'other';
  }
  if (model.includes('deepseek')) {
    // Distinguish crawl generation from refusal-check target query by max_tokens:
    // crawl-gen uses crawler.max_generated_tokens (~4000), refusal uses
    // max_refusal_check_generated_tokens (~2048). Also crawl-gen has large batch.
    if (mt && mt >= 3000) return 'generate';
    if ((r.batch_size || 0) >= 10) return 'generate';
    return 'refusal-query';
  }
  if (model.includes('gemma') || model.includes('refusal')) {
    const user = firstUser(r.inputs);
    if (/AI safety judge/i.test(user)) return 'judge';
    if (/AI safety research|敏感话题|assisting with AI safety/i.test(user)) return 'provoke';
    return 'other';
  }
  return 'other';
}

// --- normalize outputs: some call_types log a single string, others a list ---
RECORDS.forEach(r => {
  if (typeof r.outputs === 'string') r.outputs = [r.outputs];
  else if (r.outputs == null) r.outputs = [];
});

// --- pre-compute phase per record ---
RECORDS.forEach((r, i) => { r._idx = i; r._phase = classify(r); });

// --- detect prompt language from user content (Chinese codepoints => zh) ---
function detectLang(r) {
  return /[\u4e00-\u9fff]/.test(firstUser(r.inputs)) ? 'zh' : 'en';
}
RECORDS.forEach(r => {
  r._lang = (r._phase === 'generate') ? detectLang(r) : null;
});

// --- step segmentation matches the crawler loop:
//     for crawl_step_idx in range(N): for lang in langs: generate + process
//     A new crawler step begins when the next generate's language would
//     repeat a language already used in the current step.
let _step = -1;
let _langsInStep = new Set();
RECORDS.forEach(r => {
  if (r._phase === 'generate') {
    if (_step < 0 || _langsInStep.has(r._lang)) {
      _step++;
      _langsInStep = new Set([r._lang]);
    } else {
      _langsInStep.add(r._lang);
    }
  }
  r._step = _step;
});

// --- Header ---
document.getElementById('title').textContent = RAW.title;
const firstTs = RECORDS[0] && RECORDS[0].timestamp;
const lastTs = RECORDS[RECORDS.length - 1] && RECORDS[RECORDS.length - 1].timestamp;
let durStr = '';
if (firstTs && lastTs) {
  const ms = new Date(lastTs) - new Date(firstTs);
  const mins = (ms / 60000).toFixed(1);
  durStr = `${firstTs.slice(0, 19).replace('T', ' ')}  →  ${lastTs.slice(0, 19).replace('T', ' ')}  (${mins} min)`;
}
document.getElementById('meta').textContent = durStr;

// --- KPIs ---
const kpiBits = [
  ['Total calls', RECORDS.length],
  ['Steps detected', new Set(RECORDS.map(r => r._step)).size],
];
const phaseCounts = {};
RECORDS.forEach(r => { phaseCounts[r._phase] = (phaseCounts[r._phase] || 0) + 1; });
const kpiEl = document.getElementById('kpis');
kpiBits.forEach(([label, value]) => {
  kpiEl.appendChild(el('div', {class: 'kpi'}, [
    el('div', {class: 'label'}, [label]),
    el('div', {class: 'value'}, [String(value)]),
  ]));
});

// --- Phase legend (clickable filters) ---
const ORDER = ['generate', 'extract', 'translate', 'group', 'provoke', 'refusal-query', 'judge', 'other'];
const state = { enabled: new Set(ORDER), q: '', showSteps: true };
const legendEl = document.getElementById('legend');
ORDER.forEach(p => {
  const count = phaseCounts[p] || 0;
  if (count === 0) return;
  const tag = el('div', {class: 'phase-tag active', 'data-phase': p, title: 'Click to toggle'}, [
    p,
    el('span', {class: 'count'}, [String(count)]),
  ]);
  tag.addEventListener('click', () => {
    if (state.enabled.has(p)) {
      state.enabled.delete(p);
      tag.classList.remove('active');
      tag.classList.add('disabled');
    } else {
      state.enabled.add(p);
      tag.classList.add('active');
      tag.classList.remove('disabled');
    }
    render();
  });
  legendEl.appendChild(tag);
});

// --- Render rows ---
function matches(r) {
  if (!state.enabled.has(r._phase)) return false;
  if (!state.q) return true;
  const hay = (firstUser(r.inputs) + ' ' + firstSys(r.inputs) + ' ' +
    (r.outputs || []).join(' ')).toLowerCase();
  return hay.includes(state.q);
}

function renderBody(r) {
  const body = el('div', {class: 'body'});
  body.appendChild(el('div', {class: 'meta'}, [
    `${r.timestamp}  ·  model=${r.model}  ·  temp=${r.temperature}  ·  max_tokens=${r.max_tokens}  ·  batch=${r.batch_size}  ·  ${r.call_type}`,
  ]));
  const lists = msgs(r.inputs);
  const outs = r.outputs;
  const n = Math.max(lists.length, outs.length);
  for (let i = 0; i < n; i++) {
    const pair = el('div', {class: 'pair'});
    pair.appendChild(el('div', {class: 'pair-head'}, [`Item ${i + 1} of ${n}`]));
    const msgsEl = el('div', {class: 'msgs'});
    (lists[i] || []).forEach(m => {
      msgsEl.appendChild(el('div', {class: 'msg ' + (m.role || 'user')}, [
        el('div', {class: 'role'}, [m.role || '?']),
        m.content || '',
      ]));
    });
    pair.appendChild(msgsEl);
    const out = typeof outs[i] === 'string' ? outs[i] : (outs[i] == null ? '' : JSON.stringify(outs[i]));
    pair.appendChild(el('div', {class: 'pair-head', style: 'margin-top: 6px;'}, ['Output']));
    pair.appendChild(el('div', {class: 'out' + (out.trim() ? '' : ' empty')}, [
      out.trim() ? out : '(empty)',
    ]));
    body.appendChild(pair);
  }
  return body;
}

function renderCall(r) {
  let gist = '';
  const lists = msgs(r.inputs);
  if (lists.length) {
    const u = firstUser(r.inputs);
    if (u) gist = truncate(u, 180);
  }
  if (!gist && r.outputs && r.outputs.length) gist = truncate(String(r.outputs[0]), 180);
  const phaseLabel = r._phase + (r._lang ? ` · ${r._lang}` : '');
  const det = el('details', {class: 'call', 'data-phase': r._phase}, [
    el('summary', null, [
      el('span', {class: 'idx'}, [String(r._idx).padStart(4, ' ')]),
      el('span', {class: 'phase'}, [phaseLabel]),
      el('span', {class: 'model'}, [r.model.replace(/.*\//, '')]),
      el('span', {class: 'gist'}, [gist || '(empty)']),
      el('span', {class: 'batch-badge'}, ['×' + r.batch_size]),
    ]),
  ]);
  let built = false;
  det.addEventListener('toggle', () => {
    if (det.open && !built) { det.appendChild(renderBody(r)); built = true; }
  });
  return det;
}

function render() {
  const tl = document.getElementById('timeline');
  tl.innerHTML = '';
  const filtered = RECORDS.filter(matches);
  if (filtered.length === 0) {
    tl.appendChild(el('div', {class: 'no-results'}, ['No calls match the current filter.']));
    return;
  }
  let currentStep = -1;
  filtered.forEach(r => {
    if (state.showSteps && r._step !== currentStep) {
      currentStep = r._step;
      tl.appendChild(el('div', {class: 'step-divider'},
        [`Step ${currentStep < 0 ? '— (pre-generation setup)' : currentStep}`]));
    }
    tl.appendChild(renderCall(r));
  });
}

// Wire
document.getElementById('q').addEventListener('input', e => {
  state.q = e.target.value.trim().toLowerCase(); render();
});
document.getElementById('show-steps').addEventListener('change', e => {
  state.showSteps = e.target.checked; render();
});
document.getElementById('expand-all').addEventListener('click', () => {
  document.querySelectorAll('.call').forEach(d => d.open = true);
});
document.getElementById('collapse-all').addEventListener('click', () => {
  document.querySelectorAll('.call').forEach(d => d.open = false);
});

render();
</script>
</body>
</html>
"""


def build_html(transcript_path: Path, records: list) -> str:
    title = transcript_path.stem
    payload = json.dumps({"title": title, "records": records}, ensure_ascii=False)
    payload = payload.replace("</script", "<\\/script")
    return (HTML_TEMPLATE
            .replace("__TITLE__", html.escape(title))
            .replace("__DATA__", payload))


def main() -> int:
    p = argparse.ArgumentParser(description="Render a crawl transcript JSONL as a timeline HTML viewer.")
    p.add_argument("jsonl", help="Path to crawler_out_*.jsonl")
    p.add_argument("-o", "--output", help="Output HTML path (default: alongside jsonl)")
    p.add_argument("--open", action="store_true", help="Open in browser after writing")
    args = p.parse_args()

    jsonl_path = Path(args.jsonl).resolve()
    if not jsonl_path.exists():
        print(f"error: {jsonl_path} does not exist", file=sys.stderr)
        return 2

    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not args.output:
        out_path = jsonl_path.with_name(jsonl_path.stem + "_timeline.html")
    else:
        out_path = Path(args.output).resolve()
    html_text = build_html(jsonl_path, records)
    out_path.write_text(html_text, encoding="utf-8")
    print(f"wrote {out_path} ({len(records)} calls)")
    if args.open:
        webbrowser.open(out_path.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())

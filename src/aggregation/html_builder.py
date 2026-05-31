import json
import os
from typing import Dict, List, Optional


def build_specificity_explorer_html(
    records: List[dict],
    methods: List[str],
    levels: List[str],
    has_clusters: bool,
) -> str:
    """Return a self-contained HTML explorer for specificity-scored topics.

    Each record is {"t": topic, "m": [method...], "s": level, "c": [cluster...]}.
    The explorer lets the user group the topics by cluster, method, or
    specificity level; every topic row is always labelled with all three
    properties (method, specificity, cluster) regardless of the grouping, and
    includes legends and per-group / overall summaries.
    """
    data_json = json.dumps(
        {
            "records": records,
            "methods": methods,
            "levels": levels,
            "has_clusters": has_clusters,
        },
        indent=None,
    )
    return _SPECIFICITY_EXPLORER_TEMPLATE.replace("/*__DATA__*/null", data_json)


_SPECIFICITY_EXPLORER_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Topic Specificity Explorer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; padding: 20px; }
h1 { margin-bottom: 6px; font-size: 1.5rem; }
.intro { color: #555; font-size: 0.9rem; margin-bottom: 14px; max-width: 900px; line-height: 1.45; }
.intro code { background: #eee; padding: 1px 5px; border-radius: 4px; font-size: 0.85rem; }
.summary { display: flex; gap: 18px; flex-wrap: wrap; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; }
.summary .stat { font-size: 0.85rem; color: #666; }
.summary .stat b { font-size: 1.15rem; color: #222; display: block; }
.legend { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px 16px; margin-bottom: 14px; display: flex; gap: 26px; flex-wrap: wrap; align-items: flex-start; }
.legend-group { display: flex; flex-direction: column; gap: 5px; }
.legend-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; color: #999; font-weight: 700; }
.legend-items { display: flex; gap: 10px; flex-wrap: wrap; }
.controls { display: flex; gap: 12px; align-items: center; margin-bottom: 18px; flex-wrap: wrap; }
.groupby { display: flex; gap: 6px; align-items: center; }
.groupby-label { font-size: 0.82rem; color: #666; margin-right: 2px; }
#search { flex: 1; min-width: 240px; max-width: 460px; padding: 8px 12px; font-size: 1rem; border: 1px solid #ccc; border-radius: 6px; }
#search:focus { outline: none; border-color: #4a90d9; box-shadow: 0 0 0 2px rgba(74,144,217,0.2); }
.btn { padding: 7px 14px; font-size: 0.82rem; border: 1px solid #ccc; border-radius: 6px; background: #fff; cursor: pointer; white-space: nowrap; }
.btn:hover { background: #f0f0f0; }
.btn.active { background: #e8f0fe; border-color: #4a90d9; color: #1a73e8; }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }
select.filter { padding: 7px 10px; font-size: 0.82rem; border: 1px solid #ccc; border-radius: 6px; background: #fff; max-width: 240px; }
select.filter.set { border-color: #4a90d9; background: #e8f0fe; color: #1a73e8; }
.shown { font-size: 0.82rem; color: #888; margin-left: auto; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }
.tile { background: #fff; border-radius: 8px; border: 1px solid #e0e0e0; overflow: hidden; }
.tile-header { padding: 12px 16px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; user-select: none; gap: 10px; }
.tile-header:hover { background: #fafafa; }
.tile-title { font-weight: 600; font-size: 1rem; display: flex; align-items: center; gap: 8px; min-width: 0; }
.tile-title .swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }
.tile-title .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tile-count { background: #eef1f5; color: #444; border-radius: 12px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; white-space: nowrap; flex-shrink: 0; }
.tile-body { display: none; border-top: 1px solid #eee; padding: 10px 16px 14px; }
.tile-body.open { display: block; }
.group-summary { font-size: 0.78rem; color: #777; margin-bottom: 10px; display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.topic-row { padding: 6px 0; border-bottom: 1px solid #f3f3f3; display: flex; flex-direction: column; gap: 4px; }
.topic-row:last-child { border-bottom: none; }
.topic-text { font-size: 0.9rem; color: #222; }
.chips { display: flex; gap: 5px; flex-wrap: wrap; }
.chip { font-size: 0.68rem; padding: 1px 7px; border-radius: 10px; white-space: nowrap; border: 1px solid transparent; }
.chip.method { color: #fff; }
.chip.level { color: #fff; font-weight: 600; }
.chip.cluster { background: #f1f3f4; color: #555; border-color: #e0e0e0; }
.mini { font-size: 0.68rem; padding: 1px 6px; border-radius: 10px; color: #fff; }
.hidden { display: none !important; }
mark { background: #fff3cd; padding: 0 2px; border-radius: 2px; }
.no-results { grid-column: 1 / -1; text-align: center; padding: 40px; color: #999; font-size: 1.1rem; }
</style>
</head>
<body>

<h1>Topic Specificity Explorer</h1>
<p class="intro">
Every candidate topic carries three properties:
<b>method</b> (which crawl produced it), <b>specificity</b> (how concrete it is, on
the L1&ndash;L5 ladder from broad area to single named referent), and <b>cluster</b>
(its semantic group from aggregation). Pick a <b>Group by</b> dimension below;
each topic row always shows all three labels. Use search to filter.
</p>

<div class="summary" id="summary"></div>
<div class="legend" id="legend"></div>

<div class="controls">
  <div class="groupby">
    <span class="groupby-label">Group by:</span>
    <button class="btn" data-dim="c" id="gb-c">Cluster</button>
    <button class="btn" data-dim="m" id="gb-m">Method</button>
    <button class="btn" data-dim="s" id="gb-s">Specificity</button>
  </div>
  <input type="text" id="search" placeholder="Search topics..." autocomplete="off">
  <button class="btn" id="expandAllBtn">Expand all</button>
</div>
<div class="controls">
  <span class="groupby-label">Filter:</span>
  <select class="filter" id="f-m"></select>
  <select class="filter" id="f-s"></select>
  <select class="filter" id="f-c"></select>
  <button class="btn" id="clearFilters">Clear filters</button>
  <span class="shown" id="shown"></span>
</div>
<div class="grid" id="grid"></div>

<script>
const DATA = /*__DATA__*/null;
const records = DATA.records;
const METHODS = DATA.methods;
const LEVELS = DATA.levels;
const HAS_CLUSTERS = DATA.has_clusters;

const METHOD_PALETTE = ['#4a90d9','#e8710a','#34a853','#d93025','#9334e6','#00897b','#c0392b','#7cb342'];
const LEVEL_COLORS = {'L1':'#4a90d9','L2':'#00897b','L3':'#f9a825','L4':'#fb8c00','L5':'#d93025','Junk':'#9e9e9e'};

function methodColor(name) {
  const i = METHODS.indexOf(name);
  return METHOD_PALETTE[(i < 0 ? 0 : i) % METHOD_PALETTE.length];
}
function levelColor(name) {
  if (LEVEL_COLORS[name]) return LEVEL_COLORS[name];
  const i = LEVELS.indexOf(name);
  const fallback = ['#4a90d9','#00897b','#f9a825','#fb8c00','#d93025','#9e9e9e'];
  return fallback[(i < 0 ? 0 : i) % fallback.length];
}
const DIM_LABEL = {c: 'cluster', m: 'method', s: 'specificity'};

// Distinct cluster values (sorted) for the cluster filter dropdown.
const CLUSTERS = (function() {
  const set = new Set();
  records.forEach(r => (r.c || []).forEach(c => set.add(c)));
  return Array.from(set).sort();
})();

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s == null ? '' : s;
  return div.innerHTML;
}
function highlightText(text, query) {
  if (!query) return escapeHtml(text);
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return escapeHtml(text).replace(new RegExp('(' + escaped + ')', 'gi'), '<mark>$1</mark>');
}
function valuesFor(rec, dim) {
  if (dim === 's') return [rec.s];
  const arr = rec[dim] || [];
  return arr.length ? arr : ['(none)'];
}

// ---- Summary ----
(function buildSummary() {
  const clusterSet = new Set();
  const levelCounts = {};
  records.forEach(r => {
    (r.c || []).forEach(c => clusterSet.add(c));
    levelCounts[r.s] = (levelCounts[r.s] || 0) + 1;
  });
  const specific = (levelCounts['L4'] || 0) + (levelCounts['L5'] || 0);
  const el = document.getElementById('summary');
  const stats = [
    ['topics', records.length],
    ['methods', METHODS.length],
    ['specificity levels', LEVELS.length],
  ];
  if (HAS_CLUSTERS) stats.push(['clusters', clusterSet.size]);
  stats.push(['L4+L5 (specific)', specific]);
  el.innerHTML = stats.map(s =>
    '<div class="stat"><b>' + s[1] + '</b>' + s[0] + '</div>').join('');
})();

// ---- Legend ----
(function buildLegend() {
  const el = document.getElementById('legend');
  function group(title, items) {
    return '<div class="legend-group"><span class="legend-title">' + title +
      '</span><div class="legend-items">' + items + '</div></div>';
  }
  const methodItems = METHODS.map(m =>
    '<span class="chip method" style="background:' + methodColor(m) + '">' +
    escapeHtml(m) + '</span>').join('');
  const levelItems = LEVELS.map(l =>
    '<span class="chip level" style="background:' + levelColor(l) + '">' +
    escapeHtml(l) + '</span>').join('');
  let html = group('Method', methodItems) + group('Specificity (broad → specific)', levelItems);
  if (HAS_CLUSTERS) html += group('Cluster', '<span class="chip cluster">semantic group from aggregation</span>');
  el.innerHTML = html;
})();

// ---- Rendering ----
let groupDim = HAS_CLUSTERS ? 'c' : 'm';
let expandAll = false;

function chipsFor(rec, query) {
  let html = '<div class="chips">';
  (rec.m || []).forEach(m => {
    html += '<span class="chip method" style="background:' + methodColor(m) + '">' + escapeHtml(m) + '</span>';
  });
  html += '<span class="chip level" style="background:' + levelColor(rec.s) + '">' + escapeHtml(rec.s) + '</span>';
  (rec.c || []).forEach(c => {
    html += '<span class="chip cluster">' + highlightText(c, query) + '</span>';
  });
  html += '</div>';
  return html;
}

function miniDist(recIdxs, dim) {
  // Distribution of `dim` values across the records in a group.
  const counts = {};
  recIdxs.forEach(i => valuesFor(records[i], dim).forEach(v => {
    counts[v] = (counts[v] || 0) + 1;
  }));
  const order = dim === 's' ? LEVELS : (dim === 'm' ? METHODS : Object.keys(counts).sort((a,b)=>counts[b]-counts[a]));
  return order.filter(v => counts[v]).map(v => {
    if (dim === 'm') return '<span class="mini" style="background:' + methodColor(v) + '">' + escapeHtml(v) + ' ' + counts[v] + '</span>';
    if (dim === 's') return '<span class="mini" style="background:' + levelColor(v) + '">' + escapeHtml(v) + ' ' + counts[v] + '</span>';
    return '<span class="chip cluster">' + escapeHtml(v) + ' ' + counts[v] + '</span>';
  }).join('');
}

function groupTitleHtml(dim, value, query) {
  if (dim === 'm') return '<span class="swatch" style="background:' + methodColor(value) + '"></span><span class="name">' + highlightText(value, query) + '</span>';
  if (dim === 's') return '<span class="swatch" style="background:' + levelColor(value) + '"></span><span class="name">' + highlightText(value, query) + '</span>';
  return '<span class="name">' + highlightText(value, query) + '</span>';
}

function render() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  const q = (document.getElementById('search').value || '').trim().toLowerCase();
  const fM = document.getElementById('f-m').value;
  const fS = document.getElementById('f-s').value;
  const fC = document.getElementById('f-c').value;

  // Pinned dropdown filters (AND-combined) + search across text/labels.
  const matchIdx = [];
  records.forEach((r, i) => {
    if (fM && !(r.m || []).includes(fM)) return;
    if (fS && r.s !== fS) return;
    if (fC && !(r.c || []).includes(fC)) return;
    if (q) {
      const hay = (r.t + ' ' + (r.m||[]).join(' ') + ' ' + r.s + ' ' + (r.c||[]).join(' ')).toLowerCase();
      if (!hay.includes(q)) return;
    }
    matchIdx.push(i);
  });

  document.getElementById('shown').textContent =
    'showing ' + matchIdx.length + ' of ' + records.length + ' topics';
  // Mark active dropdowns.
  [['f-m', fM], ['f-s', fS], ['f-c', fC]].forEach(([id, v]) =>
    document.getElementById(id).classList.toggle('set', !!v));

  // Bucket into groups by the active dimension.
  const groups = new Map();
  matchIdx.forEach(i => valuesFor(records[i], groupDim).forEach(v => {
    if (!groups.has(v)) groups.set(v, []);
    groups.get(v).push(i);
  }));

  const sorted = Array.from(groups.entries()).sort((a, b) => b[1].length - a[1].length);
  if (sorted.length === 0) {
    grid.innerHTML = '<div class="no-results">No topics match your search.</div>';
    return;
  }

  // Which two dimensions to summarise inside each group (the other two).
  const otherDims = ['c','m','s'].filter(d => d !== groupDim && (d !== 'c' || HAS_CLUSTERS));

  for (const [value, idxs] of sorted) {
    const tile = document.createElement('div');
    tile.className = 'tile';

    const header = document.createElement('div');
    header.className = 'tile-header';
    header.innerHTML =
      '<span class="tile-title">' + groupTitleHtml(groupDim, value, q) + '</span>' +
      '<span class="tile-count">' + idxs.length + ' topics</span>';

    const body = document.createElement('div');
    body.className = 'tile-body' + (expandAll ? ' open' : '');

    let summaryHtml = '';
    otherDims.forEach(d => {
      summaryHtml += '<span style="color:#aaa;">' + DIM_LABEL[d] + ':</span> ' + miniDist(idxs, d) + ' ';
    });
    const summary = document.createElement('div');
    summary.className = 'group-summary';
    summary.innerHTML = summaryHtml;
    body.appendChild(summary);

    // Sort topics within a group by specificity (most specific first), then text.
    const lvlRank = {}; LEVELS.forEach((l, i) => lvlRank[l] = i);
    idxs.sort((a, b) => (lvlRank[records[b].s]||0) - (lvlRank[records[a].s]||0) || records[a].t.localeCompare(records[b].t));

    idxs.forEach(i => {
      const r = records[i];
      const row = document.createElement('div');
      row.className = 'topic-row';
      row.innerHTML = '<span class="topic-text">' + highlightText(r.t, q) + '</span>' + chipsFor(r, q);
      body.appendChild(row);
    });

    header.addEventListener('click', () => body.classList.toggle('open'));
    tile.appendChild(header);
    tile.appendChild(body);
    grid.appendChild(tile);
  }
}

function setGroupDim(dim) {
  groupDim = dim;
  ['c','m','s'].forEach(d => {
    const btn = document.getElementById('gb-' + d);
    btn.classList.toggle('active', d === dim);
  });
  render();
}

// Populate the filter dropdowns.
function fillSelect(id, allLabel, values) {
  const sel = document.getElementById(id);
  sel.innerHTML = '';
  const opt0 = document.createElement('option');
  opt0.value = ''; opt0.textContent = allLabel;
  sel.appendChild(opt0);
  values.forEach(v => {
    const o = document.createElement('option');
    o.value = v; o.textContent = v;
    sel.appendChild(o);
  });
}
fillSelect('f-m', 'All methods', METHODS);
fillSelect('f-s', 'All specificity', LEVELS);
if (HAS_CLUSTERS) {
  fillSelect('f-c', 'All clusters', CLUSTERS);
} else {
  document.getElementById('f-c').style.display = 'none';
}
['f-m', 'f-s', 'f-c'].forEach(id =>
  document.getElementById(id).addEventListener('change', render));
document.getElementById('clearFilters').addEventListener('click', function() {
  ['f-m', 'f-s', 'f-c'].forEach(id => { document.getElementById(id).value = ''; });
  document.getElementById('search').value = '';
  render();
});

// Wire controls.
document.getElementById('gb-c').disabled = !HAS_CLUSTERS;
['c','m','s'].forEach(d => document.getElementById('gb-' + d).addEventListener('click', () => setGroupDim(d)));
let debounce;
document.getElementById('search').addEventListener('input', function() {
  clearTimeout(debounce); debounce = setTimeout(render, 150);
});
document.getElementById('expandAllBtn').addEventListener('click', function() {
  expandAll = !expandAll;
  this.textContent = expandAll ? 'Collapse all' : 'Expand all';
  this.classList.toggle('active', expandAll);
  document.querySelectorAll('.tile-body').forEach(b => b.classList.toggle('open', expandAll));
});

setGroupDim(groupDim);
</script>
</body>
</html>"""


def _build_per_topic_tree(
    topic: str,
    reduction_log: dict,
) -> dict:
    """Build a nested tree for one final topic by walking backwards through iterations.

    Returns a tree node: {"label": str, "children": [tree_node, ...]}
    Leaf nodes have empty children lists.
    """
    iterations = reduction_log.get("iterations", [])
    if not iterations:
        return {"label": topic, "children": []}

    # Build per-iteration mappings: output_lower -> [(original_case_output, [inputs])]
    iter_mappings = []
    for iteration in iterations:
        mapping: Dict[str, List[str]] = {}
        original_case: Dict[str, str] = {}
        for step in iteration["steps"]:
            for out_topic, in_topics in step["output_mapping"].items():
                key = out_topic.strip().lower()
                if key not in mapping:
                    mapping[key] = []
                    original_case[key] = out_topic.strip()
                mapping[key].extend(in_topics)
        iter_mappings.append((mapping, original_case))

    def build_subtree(label: str, depth: int) -> dict:
        """Recursively build tree from final topic down to originals."""
        if depth < 0 or depth >= len(iter_mappings):
            return {"label": label, "children": []}
        mapping, orig_case = iter_mappings[depth]
        key = label.strip().lower()
        if key not in mapping:
            return {"label": label, "children": []}
        children = []
        for child_topic in mapping[key]:
            child_tree = build_subtree(child_topic, depth - 1)
            children.append(child_tree)
        return {"label": label, "children": children}

    return build_subtree(topic, len(iter_mappings) - 1)


def build_explorer_html(
    reduction_log: dict,
    final_topics: Dict[str, List[str]],
    trajectory: Dict[str, List[str]],
    source_sets: Optional[Dict[str, set]] = None,
    num_runs: Optional[int] = None,
) -> str:
    """Return a self-contained HTML string for exploring iterative reduction results.

    Features:
    - Grid of final topic tiles (root nodes), sorted by trajectory size
    - Click tile header to expand one level (direct inputs from last iteration)
    - "Expand all" button to unfold full tree to original topics
    - Search bar to filter across all levels
    - Consistency indicators per tile (when source_sets provided)
    """
    # Build trees for each final topic
    trees = {}
    for topic in final_topics:
        trees[topic] = _build_per_topic_tree(topic, reduction_log)

    # Derive run labels from input_paths
    input_paths = reduction_log.get("input_paths", [])
    run_labels = [
        os.path.splitext(os.path.basename(p))[0] for p in input_paths
    ]

    data_json = json.dumps(
        {
            "reduction_log": reduction_log,
            "final_topics": final_topics,
            "trajectory": trajectory,
            "trees": trees,
            "source_sets": {
                k: sorted(v) for k, v in (source_sets or {}).items()
            },
            "num_runs": num_runs or 0,
            "run_labels": run_labels,
        },
        indent=None,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Topic Reduction Explorer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; padding: 20px; }}
h1 {{ margin-bottom: 8px; font-size: 1.5rem; }}
.meta {{ color: #666; font-size: 0.85rem; margin-bottom: 16px; }}
.controls {{ display: flex; gap: 12px; align-items: center; margin-bottom: 20px; flex-wrap: wrap; }}
#search {{ flex: 1; min-width: 250px; max-width: 500px; padding: 8px 12px; font-size: 1rem; border: 1px solid #ccc; border-radius: 6px; }}
#search:focus {{ outline: none; border-color: #4a90d9; box-shadow: 0 0 0 2px rgba(74,144,217,0.2); }}
.btn {{ padding: 8px 16px; font-size: 0.85rem; border: 1px solid #ccc; border-radius: 6px; background: #fff; cursor: pointer; white-space: nowrap; }}
.btn:hover {{ background: #f0f0f0; }}
.btn.active {{ background: #e8f0fe; border-color: #4a90d9; color: #1a73e8; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 16px; }}
.tile {{ background: #fff; border-radius: 8px; border: 1px solid #e0e0e0; overflow: hidden; transition: box-shadow 0.15s; }}
.tile:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
.tile-header {{ padding: 14px 16px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; user-select: none; }}
.tile-header:hover {{ background: #fafafa; }}
.tile-title {{ font-weight: 600; font-size: 1rem; }}
.tile-count {{ background: #e8f0fe; color: #1a73e8; border-radius: 12px; padding: 2px 10px; font-size: 0.8rem; font-weight: 600; white-space: nowrap; margin-left: 8px; flex-shrink: 0; }}
.tile-body {{ display: none; border-top: 1px solid #eee; padding: 12px 16px; }}
.tile-body.open {{ display: block; }}
.tree {{ list-style: none; padding-left: 0; }}
.tree .tree {{ padding-left: 20px; }}
.tree-node {{ padding: 2px 0; }}
.tree-label {{ font-size: 0.9rem; color: #555; cursor: default; }}
.tree-label.expandable {{ cursor: pointer; color: #333; font-weight: 500; }}
.tree-label.expandable:hover {{ color: #1a73e8; }}
.tree-label .arrow {{ display: inline-block; width: 16px; font-size: 0.7rem; color: #999; transition: transform 0.15s; }}
.tree-label .arrow.open {{ transform: rotate(90deg); }}
.tree-label .leaf-dot {{ display: inline-block; width: 16px; color: #ccc; font-size: 0.7rem; }}
.expand-toggle {{ display: inline-block; margin-top: 8px; padding: 4px 10px; font-size: 0.78rem; color: #666; background: #f0f0f0; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; user-select: none; }}
.expand-toggle:hover {{ background: #e8e8e8; }}
.hidden {{ display: none !important; }}
mark {{ background: #fff3cd; padding: 0 2px; border-radius: 2px; }}
.no-results {{ grid-column: 1 / -1; text-align: center; padding: 40px; color: #999; font-size: 1.1rem; }}
.consistency-banner {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px; }}
.consistency-score {{ font-size: 1.3rem; font-weight: 700; }}
.consistency-score.high {{ color: #34a853; }}
.consistency-score.mid {{ color: #ea8600; }}
.consistency-score.low {{ color: #d93025; }}
.consistency-detail {{ color: #666; font-size: 0.85rem; }}
.tile.consistent {{ border-left: 4px solid #34a853; }}
.tile.inconsistent {{ border-left: 4px solid #ea8600; }}
.run-badges {{ display: flex; gap: 3px; margin-left: 8px; flex-shrink: 0; align-items: center; }}
.run-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.run-dot.present {{ background: #34a853; }}
.run-dot.absent {{ background: #e0e0e0; }}
</style>
</head>
<body>

<h1>Topic Reduction Explorer</h1>
<div class="meta" id="meta"></div>
<div class="consistency-banner" id="consistencyBanner" style="display:none;"></div>
<div class="controls">
  <input type="text" id="search" placeholder="Search topics..." autocomplete="off">
  <button class="btn" id="expandAllBtn">Expand all tiles</button>
  <button class="btn" id="expandTreesBtn">Unfold all levels</button>
  <button class="btn" id="filterInconsistentBtn" style="display:none;">Show inconsistent only</button>
</div>
<div class="grid" id="grid"></div>

<script>
const DATA = {data_json};
const finalTopics = DATA.final_topics;
const trajectory = DATA.trajectory;
const trees = DATA.trees;
const reductionLog = DATA.reduction_log;
const sourceSets = DATA.source_sets || {{}};
const numRuns = DATA.num_runs || 0;
const runLabels = DATA.run_labels || [];

// Meta info
document.getElementById('meta').textContent =
  reductionLog.num_input_topics + ' input topics \\u2192 ' +
  reductionLog.num_final_topics + ' final topics (' +
  reductionLog.num_iterations + ' iterations, ' +
  reductionLog.num_llm_calls + ' LLM calls)';

// Consistency banner
if (numRuns > 1) {{
  let consistentCount = 0;
  const totalCount = Object.keys(finalTopics).length;
  for (const title of Object.keys(finalTopics)) {{
    const key = title.trim().toLowerCase();
    const runs = sourceSets[key] || [];
    if (runs.length >= numRuns) consistentCount++;
  }}
  const score = totalCount > 0 ? consistentCount / totalCount : 0;
  const pct = (score * 100).toFixed(0);
  const cls = score >= 0.7 ? 'high' : score >= 0.4 ? 'mid' : 'low';
  const banner = document.getElementById('consistencyBanner');
  banner.style.display = 'flex';
  banner.innerHTML =
    '<span class="consistency-score ' + cls + '">' + pct + '%</span>' +
    '<span class="consistency-detail">consistency — ' + consistentCount + '/' + totalCount +
    ' topics present in all ' + numRuns + ' runs</span>';
  document.getElementById('filterInconsistentBtn').style.display = '';
}}

function escapeHtml(s) {{
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}}

function highlightText(text, query) {{
  if (!query) return escapeHtml(text);
  const escaped = query.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
  const re = new RegExp('(' + escaped + ')', 'gi');
  return escapeHtml(text).replace(re, '<mark>$1</mark>');
}}

function treeMatchesQuery(node, q) {{
  if (node.label.toLowerCase().includes(q)) return true;
  return (node.children || []).some(c => treeMatchesQuery(c, q));
}}

function renderTreeNode(node, query, depth, expandAll) {{
  const hasChildren = node.children && node.children.length > 0;
  const li = document.createElement('li');
  li.className = 'tree-node';

  const label = document.createElement('span');
  label.className = 'tree-label' + (hasChildren ? ' expandable' : '');

  if (hasChildren) {{
    const arrow = document.createElement('span');
    arrow.className = 'arrow' + (expandAll ? ' open' : '');
    arrow.textContent = '\\u25B6';
    label.appendChild(arrow);
  }} else {{
    const dot = document.createElement('span');
    dot.className = 'leaf-dot';
    dot.textContent = '\\u2022';
    label.appendChild(dot);
  }}

  const textSpan = document.createElement('span');
  textSpan.innerHTML = highlightText(node.label, query);
  label.appendChild(textSpan);
  li.appendChild(label);

  if (hasChildren) {{
    const childUl = document.createElement('ul');
    childUl.className = 'tree';
    if (!expandAll) childUl.style.display = 'none';
    for (const child of node.children) {{
      childUl.appendChild(renderTreeNode(child, query, depth + 1, expandAll));
    }}
    li.appendChild(childUl);

    label.addEventListener('click', function() {{
      const isOpen = childUl.style.display !== 'none';
      childUl.style.display = isOpen ? 'none' : '';
      label.querySelector('.arrow').classList.toggle('open', !isOpen);
    }});
  }}

  return li;
}}

let globalExpandAll = false;
let filterInconsistent = false;

function getTopicRuns(title) {{
  const key = title.trim().toLowerCase();
  return sourceSets[key] || [];
}}

function isConsistent(title) {{
  return numRuns > 0 && getTopicRuns(title).length >= numRuns;
}}

function renderRunBadges(title) {{
  if (numRuns <= 1) return '';
  const runs = new Set(getTopicRuns(title));
  let html = '<span class="run-badges" title="Runs: ' +
    Array.from(runs).map(i => runLabels[i] || ('run' + i)).join(', ') + '">';
  for (let i = 0; i < numRuns; i++) {{
    html += '<span class="run-dot ' + (runs.has(i) ? 'present' : 'absent') +
      '" title="' + escapeHtml(runLabels[i] || ('run' + i)) + '"></span>';
  }}
  html += '</span>';
  return html;
}}

function renderGrid(query) {{
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  const q = (query || '').trim().toLowerCase();
  let shown = 0;

  // Sort by trajectory size (number of original inputs)
  const sortedTitles = Object.keys(finalTopics).sort((a, b) =>
    (trajectory[b] || []).length - (trajectory[a] || []).length
  );

  for (const title of sortedTitles) {{
    const tree = trees[title];
    // Filter: match title or any node in tree
    if (q && !treeMatchesQuery(tree, q)) continue;
    // Filter inconsistent only
    if (filterInconsistent && isConsistent(title)) continue;
    shown++;

    const origCount = (trajectory[title] || []).length;
    const consistent = isConsistent(title);

    const tile = document.createElement('div');
    tile.className = 'tile' + (numRuns > 1 ? (consistent ? ' consistent' : ' inconsistent') : '');

    // Header
    const header = document.createElement('div');
    header.className = 'tile-header';
    header.innerHTML =
      '<span class="tile-title">' + highlightText(title, q) + '</span>' +
      '<span style="display:flex;align-items:center;">' +
      renderRunBadges(title) +
      '<span class="tile-count">' + origCount + ' originals</span></span>';

    // Body
    const body = document.createElement('div');
    body.className = 'tile-body';

    // Render tree
    const treeUl = document.createElement('ul');
    treeUl.className = 'tree';
    if (tree.children && tree.children.length > 0) {{
      for (const child of tree.children) {{
        treeUl.appendChild(renderTreeNode(child, q, 1, globalExpandAll));
      }}
    }} else {{
      const li = document.createElement('li');
      li.className = 'tree-node';
      li.innerHTML = '<span class="tree-label"><span class="leaf-dot">\\u2022</span>' + highlightText(title, q) + '</span>';
      treeUl.appendChild(li);
    }}
    body.appendChild(treeUl);

    // Per-tile expand/collapse toggle
    const toggle = document.createElement('span');
    toggle.className = 'expand-toggle';
    toggle.textContent = 'Unfold all levels';
    toggle.addEventListener('click', function(e) {{
      e.stopPropagation();
      const allSubTrees = body.querySelectorAll('.tree');
      const allArrows = body.querySelectorAll('.arrow');
      const isExpanding = toggle.textContent === 'Unfold all levels';
      allSubTrees.forEach(function(ul) {{
        if (ul !== treeUl) ul.style.display = isExpanding ? '' : 'none';
      }});
      allArrows.forEach(function(a) {{
        a.classList.toggle('open', isExpanding);
      }});
      toggle.textContent = isExpanding ? 'Collapse all levels' : 'Unfold all levels';
    }});
    body.appendChild(toggle);

    header.addEventListener('click', function() {{
      body.classList.toggle('open');
    }});

    tile.appendChild(header);
    tile.appendChild(body);
    grid.appendChild(tile);
  }}

  if (shown === 0) {{
    grid.innerHTML = '<div class="no-results">No topics match your search.</div>';
  }}
}}

// Initial render
renderGrid('');

// Search handler
let debounceTimer;
document.getElementById('search').addEventListener('input', function() {{
  clearTimeout(debounceTimer);
  const val = this.value;
  debounceTimer = setTimeout(function() {{ renderGrid(val); }}, 150);
}});

// Expand all tiles button
document.getElementById('expandAllBtn').addEventListener('click', function() {{
  const bodies = document.querySelectorAll('.tile-body');
  const anyOpen = Array.from(bodies).some(b => b.classList.contains('open'));
  bodies.forEach(function(b) {{
    if (anyOpen) b.classList.remove('open');
    else b.classList.add('open');
  }});
  this.textContent = anyOpen ? 'Expand all tiles' : 'Collapse all tiles';
}});

// Unfold all levels button
document.getElementById('expandTreesBtn').addEventListener('click', function() {{
  globalExpandAll = !globalExpandAll;
  this.textContent = globalExpandAll ? 'Collapse all levels' : 'Unfold all levels';
  this.classList.toggle('active', globalExpandAll);
  renderGrid(document.getElementById('search').value);
  // Also open all tile bodies when expanding
  if (globalExpandAll) {{
    document.querySelectorAll('.tile-body').forEach(b => b.classList.add('open'));
    document.getElementById('expandAllBtn').textContent = 'Collapse all tiles';
  }}
}});

// Filter inconsistent only button
document.getElementById('filterInconsistentBtn').addEventListener('click', function() {{
  filterInconsistent = !filterInconsistent;
  this.textContent = filterInconsistent ? 'Show all topics' : 'Show inconsistent only';
  this.classList.toggle('active', filterInconsistent);
  renderGrid(document.getElementById('search').value);
}});
</script>
</body>
</html>"""

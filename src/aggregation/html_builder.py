import json
from typing import Dict, List


def build_explorer_html(merge_log: dict, final_clusters: Dict[str, List[str]]) -> str:
    """Return a self-contained HTML string for exploring aggregation results.

    Features:
    - Grid of cluster tiles (title + member count), click to expand topics
    - Per-tile "Merge history" toggle showing the tree of steps that produced it
    - Search bar to filter clusters and topics by keyword
    """
    data_json = json.dumps(
        {"merge_log": merge_log, "final_clusters": final_clusters},
        indent=None,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Topic Aggregation Explorer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; padding: 20px; }}
h1 {{ margin-bottom: 8px; font-size: 1.5rem; }}
.meta {{ color: #666; font-size: 0.85rem; margin-bottom: 16px; }}
#search {{ width: 100%; max-width: 500px; padding: 8px 12px; font-size: 1rem; border: 1px solid #ccc; border-radius: 6px; margin-bottom: 20px; }}
#search:focus {{ outline: none; border-color: #4a90d9; box-shadow: 0 0 0 2px rgba(74,144,217,0.2); }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }}
.tile {{ background: #fff; border-radius: 8px; border: 1px solid #e0e0e0; overflow: hidden; transition: box-shadow 0.15s; }}
.tile:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
.tile-header {{ padding: 14px 16px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; user-select: none; }}
.tile-header:hover {{ background: #fafafa; }}
.tile-title {{ font-weight: 600; font-size: 1rem; }}
.tile-count {{ background: #e8f0fe; color: #1a73e8; border-radius: 12px; padding: 2px 10px; font-size: 0.8rem; font-weight: 600; white-space: nowrap; margin-left: 8px; flex-shrink: 0; }}
.tile-body {{ display: none; border-top: 1px solid #eee; padding: 12px 16px; }}
.tile-body.open {{ display: block; }}
.topic-list {{ list-style: none; padding: 0; }}
.topic-list li {{ padding: 3px 0; font-size: 0.9rem; color: #555; }}
.topic-list li::before {{ content: "\\2022"; color: #999; margin-right: 8px; }}
.merge-toggle {{ display: inline-block; margin-top: 10px; padding: 4px 10px; font-size: 0.78rem; color: #666; background: #f0f0f0; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; user-select: none; }}
.merge-toggle:hover {{ background: #e8e8e8; }}
.merge-history {{ display: none; margin-top: 10px; padding: 10px; background: #fafafa; border-radius: 6px; border: 1px solid #eee; font-size: 0.82rem; max-height: 400px; overflow-y: auto; }}
.merge-history.open {{ display: block; }}
.step {{ margin-bottom: 10px; padding: 8px; background: #fff; border-radius: 4px; border-left: 3px solid #4a90d9; }}
.step.batch {{ border-left-color: #34a853; }}
.step-type {{ font-weight: 600; font-size: 0.78rem; text-transform: uppercase; margin-bottom: 4px; }}
.step-type.batch {{ color: #34a853; }}
.step-type.merge {{ color: #4a90d9; }}
.step-detail {{ color: #666; font-size: 0.78rem; }}
.hidden {{ display: none !important; }}
mark {{ background: #fff3cd; padding: 0 2px; border-radius: 2px; }}
.no-results {{ grid-column: 1 / -1; text-align: center; padding: 40px; color: #999; font-size: 1.1rem; }}
</style>
</head>
<body>

<h1>Topic Aggregation Explorer</h1>
<div class="meta" id="meta"></div>
<input type="text" id="search" placeholder="Search clusters and topics..." autocomplete="off">
<div class="grid" id="grid"></div>

<script>
const DATA = {data_json};
const finalClusters = DATA.final_clusters;
const mergeLog = DATA.merge_log;

// Build meta info
document.getElementById('meta').textContent =
  mergeLog.num_input_topics + ' input topics \\u2192 ' +
  mergeLog.num_final_clusters + ' clusters (' +
  mergeLog.steps.length + ' LLM calls)';

// For each final cluster, find relevant merge steps by tracing title/member overlap
function findRelevantSteps(clusterTitle, members) {{
  const memberSet = new Set(members.map(m => m.toLowerCase()));
  const relevant = [];
  for (const step of mergeLog.steps) {{
    const outClusters = step.output_clusters || {{}};
    for (const [title, topics] of Object.entries(outClusters)) {{
      // Check if this step's output cluster overlaps with the final cluster
      const overlap = topics.some(t => memberSet.has(t.toLowerCase()));
      const titleMatch = title.toLowerCase() === clusterTitle.toLowerCase();
      if (overlap || titleMatch) {{
        relevant.push(step);
        break;
      }}
    }}
  }}
  return relevant;
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

function renderGrid(query) {{
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  const q = (query || '').trim().toLowerCase();
  let shown = 0;

  const sortedTitles = Object.keys(finalClusters).sort((a, b) =>
    finalClusters[b].length - finalClusters[a].length
  );

  for (const title of sortedTitles) {{
    const members = finalClusters[title];
    // Filter: match cluster title or any member topic
    if (q) {{
      const titleMatch = title.toLowerCase().includes(q);
      const memberMatch = members.some(m => m.toLowerCase().includes(q));
      if (!titleMatch && !memberMatch) continue;
    }}
    shown++;

    const tile = document.createElement('div');
    tile.className = 'tile';

    // Header
    const header = document.createElement('div');
    header.className = 'tile-header';
    header.innerHTML =
      '<span class="tile-title">' + highlightText(title, q) + '</span>' +
      '<span class="tile-count">' + members.length + ' topics</span>';

    // Body
    const body = document.createElement('div');
    body.className = 'tile-body';

    const ul = document.createElement('ul');
    ul.className = 'topic-list';
    for (const m of members) {{
      const li = document.createElement('li');
      li.innerHTML = highlightText(m, q);
      ul.appendChild(li);
    }}
    body.appendChild(ul);

    // Merge history toggle
    const toggle = document.createElement('span');
    toggle.className = 'merge-toggle';
    toggle.textContent = 'Show merge history';

    const historyDiv = document.createElement('div');
    historyDiv.className = 'merge-history';

    const relevantSteps = findRelevantSteps(title, members);
    if (relevantSteps.length === 0) {{
      historyDiv.innerHTML = '<div class="step-detail">No merge history (single batch)</div>';
    }} else {{
      for (const step of relevantSteps) {{
        const stepEl = document.createElement('div');
        stepEl.className = 'step' + (step.type === 'batch_cluster' ? ' batch' : '');
        const typeLabel = step.type === 'batch_cluster' ? 'Batch Cluster' : 'Merge';
        const typeClass = step.type === 'batch_cluster' ? 'batch' : 'merge';
        let detail = '';
        if (step.type === 'batch_cluster') {{
          detail = step.num_input_topics + ' topics \\u2192 ' + step.num_output_clusters + ' clusters';
        }} else {{
          detail = step.num_input_clusters + ' clusters \\u2192 ' + step.num_output_clusters + ' clusters';
        }}
        const outTitles = Object.keys(step.output_clusters || {{}}).join(', ');
        stepEl.innerHTML =
          '<div class="step-type ' + typeClass + '">Step ' + step.step_idx + ': ' + typeLabel + '</div>' +
          '<div class="step-detail">' + detail + '</div>' +
          '<div class="step-detail" style="margin-top:4px;color:#888;">Output: ' + escapeHtml(outTitles) + '</div>';
        historyDiv.appendChild(stepEl);
      }}
    }}

    toggle.addEventListener('click', function(e) {{
      e.stopPropagation();
      historyDiv.classList.toggle('open');
      toggle.textContent = historyDiv.classList.contains('open')
        ? 'Hide merge history' : 'Show merge history';
    }});

    body.appendChild(toggle);
    body.appendChild(historyDiv);

    header.addEventListener('click', function() {{
      body.classList.toggle('open');
    }});

    tile.appendChild(header);
    tile.appendChild(body);
    grid.appendChild(tile);
  }}

  if (shown === 0) {{
    grid.innerHTML = '<div class="no-results">No clusters match your search.</div>';
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
</script>
</body>
</html>"""

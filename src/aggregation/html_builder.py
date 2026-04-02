import json
import os
from typing import Dict, List, Optional


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

#!/usr/bin/env python3
"""
Create a funnel diagram showing the IPC discovery pipeline.
Visualizes the filtering stages from initial prefill generation to discovered topics.
Creates a multi-subplot figure comparing IPC (thought/assistant prefill) vs Baseline (user prefill) for all models.
Uses matplotlib for PDF generation without requiring Chrome.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Directory containing crawler logs
baseline_eval_dir = "/home/can/iterated_prefill_crawler/artifacts/baseline_eval"


def parse_filename(filename):
    """Extract model name and prefill mode from filename."""
    # Pattern: crawler_log_TIMESTAMP1_TIMESTAMP2_MODELNAME_..._prefillmode_...
    match = re.search(
        r"crawler_log_\d+_\d+_(.+?)_\d+samples.*?_(thought|assistant|user)_prefill",
        filename,
    )
    if match:
        model_name = match.group(1)
        prefill_mode = match.group(2)
        return model_name, prefill_mode
    return None, None


def load_log_data(log_file):
    """Load and extract statistics from a crawler log file."""
    with open(log_file, "r") as f:
        data = json.load(f)

    stats = data["stats"]["cumulative"]
    total_all = stats["total_all"]
    total_deduped = stats["total_deduped"]
    total_unique_refusals = stats["total_unique_refusals"]

    # Calculate derived statistics
    duplicates_filtered = total_all - total_deduped
    novel_terms = total_deduped
    consistent_refusals = total_unique_refusals
    inconsistent_refusals = total_deduped - total_unique_refusals

    return {
        "total_all": total_all,
        "novel_terms": novel_terms,
        "consistent_refusals": consistent_refusals,
        "duplicates_filtered": duplicates_filtered,
        "inconsistent_refusals": inconsistent_refusals,
    }


def draw_funnel(ax, data, title="", colors=None):
    """Draw a funnel diagram on the given axes."""
    if colors is None:
        colors = ["#4A90E2", "#3498DB", "#27AE60"]

    stages = [
        "Total terms generated\nby prefill",
        "Novel terms tested",
        'Consistent refusals (≥50%)\n"discovered topics"',
    ]

    values = [data["total_all"], data["novel_terms"], data["consistent_refusals"]]

    # Normalize values to fit in funnel (width represents value)
    max_val = max(values) if values else 1
    normalized_widths = [v / max_val for v in values]

    # Funnel parameters
    funnel_height = 0.8
    funnel_top_width = 0.9
    funnel_bottom_width = 0.1
    y_start = 0.95
    y_spacing = funnel_height / (len(stages) - 1)

    # Draw funnel segments
    for i, (stage, value, width, color) in enumerate(
        zip(stages, values, normalized_widths, colors)
    ):
        y_pos = y_start - i * y_spacing

        # Calculate width for this stage (trapezoid)
        if i == 0:
            # Top stage
            stage_width = funnel_top_width
        elif i == len(stages) - 1:
            # Bottom stage
            stage_width = (
                funnel_bottom_width + (funnel_top_width - funnel_bottom_width) * width
            )
        else:
            # Middle stages - interpolate
            progress = i / (len(stages) - 1)
            stage_width = (
                funnel_top_width
                - (funnel_top_width - funnel_bottom_width) * progress * width
            )

        # Draw trapezoid
        stage_height = y_spacing * 0.8

        # Calculate trapezoid vertices
        if i == 0:
            # Top: full width
            top_width = funnel_top_width
            bottom_width = (
                funnel_top_width
                - (funnel_top_width - funnel_bottom_width)
                * (stage_height / funnel_height)
                * width
            )
        elif i == len(stages) - 1:
            # Bottom: scaled to value
            top_width = (
                funnel_bottom_width
                + (funnel_top_width - funnel_bottom_width)
                * (1 - (i * y_spacing) / funnel_height)
                * width
            )
            bottom_width = funnel_bottom_width * width
        else:
            # Middle: interpolate
            top_progress = (i * y_spacing) / funnel_height
            bottom_progress = ((i + 1) * y_spacing) / funnel_height
            top_width = (
                funnel_top_width
                - (funnel_top_width - funnel_bottom_width) * top_progress * width
            )
            bottom_width = (
                funnel_top_width
                - (funnel_top_width - funnel_bottom_width) * bottom_progress * width
            )

        # Draw trapezoid
        vertices = np.array(
            [
                [-top_width / 2, y_pos],
                [top_width / 2, y_pos],
                [bottom_width / 2, y_pos - stage_height],
                [-bottom_width / 2, y_pos - stage_height],
            ]
        )

        polygon = mpatches.Polygon(
            vertices,
            closed=True,
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.85,
        )
        ax.add_patch(polygon)

        # Add text label
        percentage = (value / data["total_all"]) * 100 if data["total_all"] > 0 else 0
        label_text = f"{stage}\n{value:,} ({percentage:.1f}%)"
        ax.text(
            0,
            y_pos - stage_height / 2,
            label_text,
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            color="white" if i < len(colors) else "black",
        )

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.text(0, 1.05, title, ha="center", va="bottom", fontsize=11, weight="bold")


# Find all crawler log files
log_files = []
for file in Path(baseline_eval_dir).glob("*.json"):
    if "manual_clusters" not in file.name:
        log_files.append(file)

# Group logs by model and prefill mode
model_data = defaultdict(dict)
for log_file in log_files:
    model_name, prefill_mode = parse_filename(log_file.name)
    if model_name and prefill_mode:
        model_data[model_name][prefill_mode] = log_file

# Sort models for consistent ordering
sorted_models = sorted(model_data.keys())

# Determine IPC prefill mode for each model (prefer thought, then assistant)
ipc_modes = {}
for model in sorted_models:
    if "thought" in model_data[model]:
        ipc_modes[model] = "thought"
    elif "assistant" in model_data[model]:
        ipc_modes[model] = "assistant"
    else:
        ipc_modes[model] = None

# Create figure with subplots: rows = models, columns = 2 (IPC, Baseline)
n_rows = len(sorted_models)
n_cols = 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)

# Draw funnels for each model
for row_idx, model in enumerate(sorted_models):
    # IPC column (thought or assistant)
    ipc_mode = ipc_modes[model]
    if ipc_mode and ipc_mode in model_data[model]:
        ipc_file = model_data[model][ipc_mode]
        ipc_data = load_log_data(ipc_file)
        ipc_title = f"{model} (IPC)"
        draw_funnel(axes[row_idx, 0], ipc_data, ipc_title)
    else:
        axes[row_idx, 0].text(
            0.5, 0.5, f"{model} (IPC)\nN/A", ha="center", va="center", fontsize=11
        )
        axes[row_idx, 0].axis("off")

    # Baseline column (user)
    if "user" in model_data[model]:
        baseline_file = model_data[model]["user"]
        baseline_data = load_log_data(baseline_file)
        baseline_title = f"{model} (baseline)"
        draw_funnel(axes[row_idx, 1], baseline_data, baseline_title)
    else:
        axes[row_idx, 1].text(
            0.5, 0.5, f"{model} (baseline)\nN/A", ha="center", va="center", fontsize=11
        )
        axes[row_idx, 1].axis("off")

plt.tight_layout()

# Save as PDF
output_pdf = "/home/can/iterated_prefill_crawler/artifacts/baseline_eval/funnel_diagram_comparison.pdf"
fig.savefig(output_pdf, format="pdf", bbox_inches="tight", dpi=300)
print(f"Funnel diagram (PDF) saved to: {output_pdf}")

# Also save as PNG for reference
output_png = "/home/can/iterated_prefill_crawler/artifacts/baseline_eval/funnel_diagram_comparison.png"
fig.savefig(output_png, format="png", bbox_inches="tight", dpi=300)
print(f"Funnel diagram (PNG) saved to: {output_png}")

plt.close()


# Also generate HTML version with plotly (interactive)
def create_plotly_funnel_trace(data, colors=None):
    """Create a funnel trace for plotly."""
    if colors is None:
        # Light colors for HTML version
        colors = ["#B3D9FF", "#99CCFF", "#80E5CC"]

    stages = [
        "Total terms generated<br>by prefill",
        "after deduplication",
        "after refusal filter",
    ]

    values = [data["total_all"], data["novel_terms"], data["consistent_refusals"]]

    # Create custom text with additional info - black text
    text_labels = []
    for i, val in enumerate(values):
        percentage = (val / data["total_all"]) * 100 if data["total_all"] > 0 else 0
        text_labels.append(f"{val:,}<br>({percentage:.1f}%)")

    return go.Funnel(
        y=stages,
        x=values,
        textposition="inside",
        textinfo="text",
        text=text_labels,
        insidetextfont=dict(color="black", size=11),
        marker=dict(color=colors, line=dict(color="black", width=1.5)),
        connector=dict(line=dict(color="gray", width=1.5, dash="solid")),
        opacity=0.85,
    )


# Create subplot titles (flat list) - model names with (IPC) and (baseline)
subplot_titles = []
for model in sorted_models:
    subplot_titles.extend([f"{model} (IPC)", f"{model} (baseline)"])

# Create plotly subplot figure
fig_html = make_subplots(
    rows=n_rows,
    cols=n_cols,
    subplot_titles=subplot_titles,
    specs=[[{"type": "funnel"}, {"type": "funnel"}] for _ in range(n_rows)],
    vertical_spacing=0.15,
    horizontal_spacing=0.1,
)

# Add funnel traces for each model
for row_idx, model in enumerate(sorted_models, 1):
    # IPC column (thought or assistant)
    ipc_mode = ipc_modes[model]
    if ipc_mode and ipc_mode in model_data[model]:
        ipc_file = model_data[model][ipc_mode]
        ipc_data = load_log_data(ipc_file)
        ipc_trace = create_plotly_funnel_trace(ipc_data)
        fig_html.add_trace(ipc_trace, row=row_idx, col=1)

    # Baseline column (user)
    if "user" in model_data[model]:
        baseline_file = model_data[model]["user"]
        baseline_data = load_log_data(baseline_file)
        baseline_trace = create_plotly_funnel_trace(baseline_data)
        fig_html.add_trace(baseline_trace, row=row_idx, col=2)

# Update layout - no main title
fig_html.update_layout(
    height=400 * n_rows,
    margin=dict(l=50, r=50, t=50, b=50),
    font=dict(size=11),
    paper_bgcolor="white",
    plot_bgcolor="white",
    showlegend=False,
)

# Save as HTML
output_html = "/home/can/iterated_prefill_crawler/artifacts/baseline_eval/funnel_diagram_comparison.html"
fig_html.write_html(output_html)
print(f"Funnel diagram (HTML) saved to: {output_html}")

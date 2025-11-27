#!/usr/bin/env python3
"""
Create a funnel diagram showing the IPC discovery pipeline.
Visualizes the filtering stages from initial prefill generation to discovered topics.
"""

import json
import plotly.graph_objects as go

# Load the crawler log data
log_file = "/home/can/iterated_prefill_crawler/artifacts/interim/crawler_log_20251121_063348_Llama-3.1-Tulu-3-8B-SFT_1samples_1000crawls_Truefilter_assistant_prefixprompt_vllm.json"

with open(log_file, 'r') as f:
    data = json.load(f)

# Extract statistics
stats = data['stats']['cumulative']
total_all = stats['total_all']
total_deduped = stats['total_deduped']
total_unique_refusals = stats['total_unique_refusals']

# Calculate derived statistics
duplicates_filtered = total_all - total_deduped
novel_terms = total_deduped
consistent_refusals = total_unique_refusals
inconsistent_refusals = total_deduped - total_unique_refusals

# Prepare data for funnel chart
stages = [
    "Total terms generated<br>by prefill",
    "Novel terms tested",
    "Consistent refusals (≥50%)<br>\"discovered topics\""
]

values = [total_all, novel_terms, consistent_refusals]

# Create custom text with additional info
text_labels = []
for i, val in enumerate(values):
    percentage = (val / total_all) * 100
    text_labels.append(f"{val:,}<br>({percentage:.1f}%)")

# Colors
colors = ['#4A90E2', '#3498DB', '#27AE60']

# Create funnel chart
fig = go.Figure(go.Funnel(
    y=stages,
    x=values,
    textposition="inside",
    textinfo="text",
    text=text_labels,
    marker=dict(
        color=colors,
        line=dict(color='black', width=2)
    ),
    connector=dict(
        line=dict(color='gray', width=2, dash='solid')
    ),
    opacity=0.85
))

# Calculate summary statistics
discovery_rate = (consistent_refusals / novel_terms) * 100
false_lead_rate = (inconsistent_refusals / novel_terms) * 100
dedup_rate = (duplicates_filtered / total_all) * 100

# Add annotations for filtered stages
annotation_text = (
    f"<b>Duplicates filtered:</b> {duplicates_filtered:,} ({dedup_rate:.1f}%)<br>"
    f"<b>Inconsistent refusals (<50%) discarded:</b> {inconsistent_refusals:,} ({false_lead_rate:.1f}% of novel terms)<br><br>"
    f"<b>Discovery Rate:</b> {discovery_rate:.1f}% of novel terms become discovered topics<br>"
    f"<b>False Lead Rate:</b> {false_lead_rate:.1f}% of novel terms are discarded"
)

fig.add_annotation(
    text=annotation_text,
    xref="paper", yref="paper",
    x=0.5, y=-0.15,
    showarrow=False,
    font=dict(size=11),
    align="left",
    bgcolor="rgba(240, 240, 240, 0.8)",
    bordercolor="black",
    borderwidth=1,
    borderpad=10
)

# Update layout
fig.update_layout(
    title={
        'text': "IPC Discovery Pipeline: Funnel Analysis<br><sub>Llama-3.1-Tulu-3-8B-SFT | 1000 crawls | Assistant prefix prompt</sub>",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18}
    },
    height=700,
    margin=dict(l=50, r=50, t=120, b=180),
    font=dict(size=12),
    paper_bgcolor='white',
    plot_bgcolor='white'
)

# Save as HTML (interactive)
output_html = "/home/can/iterated_prefill_crawler/artifacts/interim/funnel_diagram.html"
fig.write_html(output_html)
print(f"Funnel diagram (HTML) saved to: {output_html}")

# Try to save as static images if kaleido is available
try:
    # Save as static image (PNG)
    output_png = "/home/can/iterated_prefill_crawler/artifacts/interim/funnel_diagram.png"
    fig.write_image(output_png, width=800, height=700, scale=2)
    print(f"Funnel diagram (PNG) saved to: {output_png}")

    # Save as PDF
    output_pdf = "/home/can/iterated_prefill_crawler/artifacts/interim/funnel_diagram.pdf"
    fig.write_image(output_pdf, width=800, height=700)
    print(f"Funnel diagram (PDF) saved to: {output_pdf}")
except Exception as e:
    print(f"Note: Static image export (PNG/PDF) requires kaleido package")
    print(f"Install with: pip install kaleido")
    print(f"Interactive HTML version is available at: {output_html}")

fig.show()

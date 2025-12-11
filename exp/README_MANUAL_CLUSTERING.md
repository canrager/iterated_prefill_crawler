# Manual Topic Clustering Interface

Two interfaces are available for manually clustering head refusal topics:

## 1. Terminal-based Interface (`manual_cluster_topics.py`)

A command-line interactive tool for clustering topics.

**Usage:**
```bash
python exp/manual_cluster_topics.py --file artifacts/baseline_eval/crawler_log_*.json
```

**Features:**
- Navigate through topics one by one
- Assign topics to existing clusters by number
- Create new clusters with custom names
- Go back to previous topics
- Auto-save after each assignment

## 2. Web-based Drag-and-Drop Interface (`manual_cluster_topics_web.py`)

A modern web interface with drag-and-drop functionality.

**Usage:**
```bash
# Install Flask if not already installed
pip install flask

# Run the web server
python exp/manual_cluster_topics_web.py --file artifacts/baseline_eval/crawler_log_*.json

# Or run without pre-loading a file (load via UI)
python exp/manual_cluster_topics_web.py
```

Then open your browser to `http://127.0.0.1:5000`

**Features:**
- Visual drag-and-drop interface
- See all topics and clusters at once
- Create new clusters by clicking "+ Create New Cluster"
- Rename clusters by double-clicking the cluster name
- Delete clusters with the trash icon
- Remove topics from clusters with the × button
- Auto-save after each action
- Resume from existing clusters automatically

**Web Interface Options:**
- `--file`, `-f`: Pre-load a crawler log file
- `--port`, `-p`: Port to run server on (default: 5000)
- `--host`: Host to bind to (default: 127.0.0.1)

**Output:**
Both interfaces save clusters to `{input_filename}_manual_clusters.json` in the same directory as the input file.

The output format is:
```json
{
  "cluster_name_1": [
    {
      "id": 0,
      "raw": "...",
      "english": "...",
      "summary": "...",
      "refusal_check_responses": [...],
      ...
    },
    ...
  ],
  "cluster_name_2": [...]
}
```

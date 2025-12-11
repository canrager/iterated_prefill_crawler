# Manual Labeling Offline Tool

A minimal, standalone web-based tool for manually clustering head refusal topics. This tool works completely offline and does not require any LLM dependencies.

## Features

- **Drag-and-drop interface** for clustering topics
- **Offline operation** - no internet or LLM required
- **Undo functionality** - easily reverse actions
- **Topic navigation** - browse through unlabeled topics
- **Cluster management** - create, rename, and delete clusters
- **Progress tracking** - see how many topics are labeled vs unlabeled

## Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Navigate to this directory:
   ```bash
   cd exp/manual_labeling_offline
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

   Or if you prefer pip:
   ```bash
   uv pip install -e .
   ```

## Usage

### Starting the Server

Run the Flask application:

```bash
python app.py
```

Or with uv:

```bash
uv run app.py
```

The server will start at `http://127.0.0.1:5000` by default.

### Command Line Options

- `--file` / `-f`: Pre-load a crawler log file (optional)
- `--port` / `-p`: Port to run the server on (default: 5000)
- `--host`: Host to bind to (default: 127.0.0.1)

Example:

```bash
python app.py --file data/crawler_log_20251127_083333_DeepSeek-R1-Distill-Llama-8B_1samples_100crawls_Truefilter_thought_prefill_with_seedprompt_vllm.json
```

### Using the Web Interface

1. **Load a file**: Enter the path to a crawler log JSON file in the input field and click "Load File"
   - Files in the `data/` directory can be referenced as `data/filename.json`
   - Absolute paths are also supported

2. **Navigate topics**: Use the Previous/Next buttons to browse through unlabeled topics

3. **Create clusters**: Click "+ Create New Cluster" to create a new cluster

4. **Assign topics**: 
   - Drag and drop a topic card onto a cluster
   - Or click on a cluster name while viewing a topic
   - Or drag a topic onto the "Create New Cluster" area

5. **Manage clusters**:
   - Double-click a cluster name to rename it
   - Click the edit icon (✏️) to rename
   - Click the delete icon (🗑️) to delete a cluster
   - Click on a cluster header to expand/collapse and see its topics

6. **Undo actions**: Click the "↶ Undo" button or press Ctrl+Z (Cmd+Z on Mac) to undo the last action

## File Structure

```
manual_labeling_offline/
├── app.py                 # Main Flask application
├── pyproject.toml          # Project configuration and dependencies
├── README.md              # This file
├── data/                  # Crawler log JSON files to label
│   └── *.json
└── templates/             # HTML templates
    └── cluster_topics.html
```

## Output Files

When you label topics, the tool automatically saves cluster assignments to a file named `{input_filename}_manual_clusters.json` in the same directory as the input file.

## Differences from Full Version

This offline version removes:
- LLM suggestion features (no vLLM dependencies)
- Project-specific imports
- GPU/ML model requirements

It retains all core labeling functionality:
- Topic browsing and navigation
- Drag-and-drop clustering
- Cluster management
- Undo functionality
- Progress tracking

## Troubleshooting

- **File not found**: Make sure the file path is correct. Relative paths are resolved relative to the `data/` directory first, then the current working directory.
- **Port already in use**: Use `--port` to specify a different port
- **Topics not loading**: Ensure the JSON file has the expected structure with `queue.topics.head_refusal_topics` array

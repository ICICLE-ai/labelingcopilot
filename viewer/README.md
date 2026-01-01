# Labeling Copilot — Viewer

A lightweight web UI for browsing a curated dataset as it's being built. The viewer renders each image with its bounding-box overlays, per-model detection scores, scene tags, and live gap-analysis charts. Pages auto-refresh every ten seconds, so curation progress is visible without reloading.

## Quick start

From the repo root:

```bash
python viewer/server.py --port 8501
```

Open <http://localhost:8501>.

## Options

```
--port <int>       Port to serve on (default 8501).
--dataset <path>   Path to the COCO-style dataset JSON (default agent_output/dataset.json).
```

The dataset path is captured at startup, but file contents are re-read on each request, so edits to `dataset.json` and `curation_state.json` appear on the next refresh without restarting the server.

## What the viewer reads

| Route | Backing file / service |
|---|---|
| `/api/dataset` | `dataset.json` at `--dataset` |
| `/api/state` | `curation_state.json` next to the dataset |
| `/api/pool-status` | retrieval `:8000/status` |
| `/api/annotator-health` | annotation `:8080/health` |
| `/api/synthesis-health` | synthesis `:8090/health` |
| `/img/curated/<file>` | `images/` next to the dataset |
| `/img/pool/<key>` | retrieval `/images/<key>` passthrough |

The viewer only requires the retrieval service for pool-image previews; annotate and synthesis are optional and show as unhealthy if they aren't running.

## Dependencies

Standard-library HTTP server — no third-party packages required.

# Architecture

## Runtime Shape

The synthesis package has two execution paths:

- `api_server.py` — the persistent HTTP service used by Docker and the agent
- `run_augmentation.py` — the batch runner that powers the service and can also be invoked directly

The live tool path is:

```text
POST /synthesize
  -> api_server.py stages seed images
  -> job manager starts run_augmentation.py
  -> generator_parallel.py calls the model APIs
  -> metadata/progress/suggestions are written to augmented-output/<job_id>/
  -> /jobs/{job_id} and /artifacts/... expose results
```

## Main Components

### `api_server.py`

- validates requests
- stages uploaded or retrieval-fetched seed images
- ensures only one job runs at a time
- tracks job status, logs, and artifact paths

### `run_augmentation.py`

- loads images
- creates suggestion and progress caches
- runs the parallel generator
- optionally computes quality metrics
- writes metadata files

### `core/generator_parallel.py`

- batches seed-image analysis requests
- caches suggestions
- rate-limits edit requests
- writes generated images and incremental metadata

### `utils/api_client.py`

- wraps Azure OpenAI or standard OpenAI calls
- sends chat-completions requests for suggestion generation
- sends image-edit requests for synthetic variants

## Storage

Job outputs live under `augmented-output/<job_id>/` by default:

- `metadata.json`
- `progress.json`
- `suggestions.json`
- generated image files

Transient per-job logs live under `augmented-output/.service/<job_id>/`.

## Provider Model

The config layer supports two providers:

- Azure OpenAI via deployment-based URLs
- OpenAI via `https://api.openai.com/v1`

Default models:

- vision/text: `gpt-5.4-nano`
- image edit: `gpt-image-1.5`

See [`README.md`](README.md) for the full environment-variable list.

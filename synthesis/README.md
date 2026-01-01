# Labeling Copilot — Synthesis Service

On-demand image augmentation as an HTTP API. The service analyses each seed image with a vision-language model, generates augmentation suggestions, and produces synthetic variants with an image-edit model. Quality metrics and OOD filtering are computed alongside each job.

Backends:

- OpenAI (via `https://api.openai.com/v1`)
- Azure OpenAI (via deployment URLs)

## Quick start

```bash
cp .env.example .env
# edit .env — set either OPENAI_API_KEY or the AZURE_* vars
docker compose up -d --build
curl http://localhost:8090/health
```

Or, from the repo root:

```bash
./scripts/docker-up.sh --with-synthesis
```

## Triggering a job

Three ways to pass seed images:

```bash
# From the default input directory inside the container
curl -X POST http://localhost:8090/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"num_variants": 3}'

# From retrieval-relative URLs
curl -X POST http://localhost:8090/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"image_urls":["/images/path/from/retrieval.jpg"],"num_variants":2}'

# Upload seed images directly
curl -X POST http://localhost:8090/synthesize \
  -F 'images=@./example.jpg' \
  -F 'num_variants=2'
```

Only one job runs at a time. See [`API.md`](API.md) for the full endpoint reference and [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for common issues.

## Output

Each job writes to `augmented-output/<job_id>/`:

- `metadata.json`
- `progress.json`
- `suggestions.json`
- generated image files

Artifacts are also served at `/artifacts/<path>`.

## Configuration

### OpenAI

```env
MODEL_API_PROVIDER=openai
OPENAI_API_KEY=your-key
OPENAI_VISION_MODEL=gpt-5.4-nano
OPENAI_IMAGE_EDIT_MODEL=gpt-image-1.5
```

### Azure OpenAI

```env
MODEL_API_PROVIDER=azure
AZURE_VISION_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_VISION_API_KEY=your-key
AZURE_VISION_DEPLOYMENT=gpt-5.4-nano
AZURE_IMAGE_EDIT_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_IMAGE_EDIT_API_KEY=your-key
AZURE_IMAGE_EDIT_DEPLOYMENT=gpt-image-1.5
```

Defaults (if unset): `gpt-5.4-nano` for vision and `gpt-image-1.5` for image edits.

Relative retrieval URLs like `/images/...` are resolved against `SYNTHESIS_RETRIEVAL_BASE_URL` (defaults to `http://host.docker.internal:8000` inside Compose).

## Offline utilities

These scripts run the underlying pipelines without the HTTP service:

- `python run_augmentation.py`
- `python run_evaluation_async.py`
- `python view_quality_metrics.py`
- `python export_ood_images.py`

## Further reading

- [`API.md`](API.md) — endpoint reference
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — internal components and data flow
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — common failures

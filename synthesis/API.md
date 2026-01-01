# API

Base URL: `http://localhost:8090`

## `GET /health`

Returns service health and active model configuration.

Example response:

```json
{
  "status": "ok",
  "config": {
    "provider": "openai",
    "vision_model": "gpt-5.4-nano",
    "image_edit_model": "gpt-image-1.5",
    "requests_per_minute": 5,
    "max_concurrent": 1,
    "retrieval_base_url": "http://host.docker.internal:8000"
  }
}
```

## `GET /status`

Returns:

- `current_job`
- `recent_jobs`
- current provider/model config

Use this as the default monitoring endpoint.

## `POST /synthesize`

Starts a new job. Only one job can run at a time.

Accepted inputs:

- `input_dir`: use an existing directory inside the container
- `image_urls`: fetch seed images, including retrieval-relative URLs like `"/images/..."`
- multipart `images`: upload seed images directly

Common fields:

- `domain`
- `num_variants`
- `output_dir`
- `resume`

### Example: default input directory

```bash
curl -X POST http://localhost:8090/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"num_variants": 3}'
```

### Example: retrieval images

```bash
curl -X POST http://localhost:8090/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"image_urls":["/images/path/from/retrieval.jpg"],"num_variants":2}'
```

### Example: upload

```bash
curl -X POST http://localhost:8090/synthesize \
  -F 'images=@./example.jpg' \
  -F 'num_variants=2'
```

## `GET /jobs/{job_id}`

Returns one job record, including:

- status
- progress summary
- artifact links
- log tail

## `POST /jobs/{job_id}/cancel`

Stops the active job.

## `GET /jobs/{job_id}/logs`

Returns the current plain-text log tail.

Optional query parameter:

- `tail`: number of trailing characters to return

## `GET /artifacts/{path}`

Serves generated files from `augmented-output/`.

Typical artifact paths:

- `/<job_id>/metadata.json`
- `/<job_id>/progress.json`
- `/<job_id>/<generated-image>.png`

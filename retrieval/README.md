# Labeling Copilot — Retrieval Service

CLIP-indexed image pool with active-learning samplers, exposed over HTTP. The
service ingests an image dataset into MinIO, extracts CLIP ViT-B/32 features,
builds a FAISS index, and serves endpoints for sampling, labeling, and
zero-shot auto-labeling against the indexed pool.

## Quick start

```bash
docker compose up -d
curl http://localhost:8000/health
```

First boot runs ETL (downloads Oxford-IIIT Pets, uploads to MinIO, computes
CLIP features). Expect a few minutes; subsequent boots reuse the cached state
in the `app_cache` volume.

When changing `LABEL_CLASSES`, wipe the cache:

```bash
docker compose down -v && docker compose up -d
```

## Configuration

Set via `docker-compose.yml` env vars or directly when running on the host.

| Variable | Default | Purpose |
|----------|---------|---------|
| `LABEL_CLASSES` | `cat,dog` | Comma-separated class names. Auto-detects Oxford-IIIT Pets vs CIFAR-10. |
| `NUM_SAMPLE_IMAGES` | `200` | Total pool size (split evenly across classes). |
| `MINIO_ENDPOINT` | `minio:9000` | MinIO host:port. |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key. |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key. |
| `MINIO_BUCKET` | `images` | Bucket for image objects. |
| `CACHE_DIR` | `/app/cache` | Where ETL state, raw images, and features are persisted. |

For CUDA / PyTorch build overrides, see the top-level `README.md`.

MinIO console is at <http://localhost:9001> (`minioadmin` / `minioadmin`).

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness probe. |
| `GET` | `/status` | Pool size, label counts, available samplers, label distribution. |
| `POST` | `/samples` | Run a sampler and return the next batch of image keys. |
| `POST` | `/labels` | Attach labels to image keys. |
| `POST` | `/auto-label` | Predict labels for image keys via CLIP text embedding similarity. |
| `GET` | `/images/{key}` | Stream image bytes from MinIO. |

### `POST /samples`

```json
{
  "sampler": "kcenter",
  "num_samples": 10,
  "sampler_params": {"candidate_pool_size": 1000},
  "new_labels": {"images/img_001.jpg": 0}
}
```

`new_labels` is optional — if present, labels are recorded before sampling.
Response has `samples: [{image_key, image_url}]` and a `metadata` block.

### `POST /labels`

```json
{ "labels": { "images/img_001.jpg": 0, "images/img_002.jpg": 1 } }
```

### `POST /auto-label`

```json
{ "image_keys": ["images/img_001.jpg"], "apply": false }
```

Returns per-image `{label, class_name, confidence}`. With `apply: true`, the
predicted labels are also written into the session.

## Samplers

| Name | When to use |
|------|-------------|
| `random` | Uniform baseline; no labels required. |
| `kcenter` | Diversity via farthest-first traversal in feature space. |
| `margin` | Uncertainty: items nearest a logistic-regression decision boundary trained on labeled samples. Needs ≥2 classes labeled. |
| `representative` | Uncertain *and* diverse: clusters near-boundary candidates and picks medoids. Needs ≥2 classes labeled. |
| `informative_cluster_diverse` | Hybrid: clusters the unlabeled pool, then ranks each cluster's representative by uncertainty. Needs ≥2 classes labeled. |

Source: `samplers/`.

## Architecture

- **ETL** (`etl.py`): pulls images → uploads to MinIO → extracts CLIP features → caches state.
- **Index** (`index_manager.py`): builds a FAISS index over the cached features (Flat / IVF / HNSW; auto-selected by pool size).
- **Session** (`active_learning_session.py`): in-memory state for one running service — pool, labels, samplers.
- **API** (`api_server.py`): FastAPI surface; loads the cached state on startup.

## Development

```bash
pip install -r requirements.txt -r requirements-test.txt
pytest
```

Tests run against synthetic features and an in-memory session — no Docker
required. The single GPU-backed integration path is the live `docker compose`
stack above.

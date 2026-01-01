# Labeling Copilot — Annotation Service

Multi-model object detection and segmentation behind a single HTTP orchestrator. Each model runs in its own container; the orchestrator fans requests out, merges detection boxes via configurable NMS, and returns per-model results alongside a consensus set.

## Models

| Task | Models |
|---|---|
| Detection | Detic, OWL-ViT, GroundingDINO |
| Segmentation | SAM, SEEM |
| Consensus (detection) | NMS, Soft-NMS, DIoU-NMS, Adaptive NMS, Weighted NMS, Cluster NMS |

Weights and configs are declared in `config.json` and materialised into the per-model Dockerfiles. First boot downloads weights into the `model-weights` volume; subsequent boots reuse the cache.

## Quick start

From this directory:

```bash
./setup.sh
curl http://localhost:8080/health
```

`setup.sh` builds the shared `annotation-base` image, builds each model image, starts the stack with `docker compose up -d`, and waits for every service to report healthy. Typical cold start is several minutes on first run (weight downloads).

Stop the stack:

```bash
docker compose down
```

## API

Base URL: `http://localhost:8080`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Orchestrator liveness. |
| `POST` | `/annotate/detect` | Run detection across models and return raw + consensus annotations. |
| `POST` | `/annotate/segment` | Run segmentation models and return their masks. |

### `POST /annotate/detect`

Multipart request:

- `image` — file
- `vocabulary` — comma-separated class names (e.g. `cat, dog`)
- `nms_methods` — JSON array, e.g. `["NON_MAX_SUPPRESSION"]`
- `nms_params` — JSON object, e.g. `{"iou_threshold": 0.5, "sigma": 0.5, "min_score": 0.1}`

```bash
curl -X POST http://localhost:8080/annotate/detect \
  -F "image=@cat.jpg" \
  -F "vocabulary=cat, dog" \
  -F 'nms_methods=["NON_MAX_SUPPRESSION"]' \
  -F 'nms_params={"iou_threshold":0.5,"sigma":0.5,"min_score":0.1}'
```

Response:

- `raw_results` — one entry per model with its own annotation list and image dimensions
- `consensus` — per NMS method, a merged annotation set with per-model score breakdown

Bounding boxes are absolute pixels in `[xmin, ymin, xmax, ymax]`.

### `POST /annotate/segment`

Multipart request:

- `image` — file
- `models` — JSON array, e.g. `["SAM", "SEEM"]`

```bash
curl -X POST http://localhost:8080/annotate/segment \
  -F "image=@scene.jpg" \
  -F 'models=["SAM", "SEEM"]'
```

SAM returns class-agnostic masks; SEEM returns labeled semantic segments.

## Configuration

Per-model configuration lives in `config.json` and is injected into each model service via the `MODEL_CONFIG` environment variable in `docker-compose.yml`. Common knobs:

| Model | Knob | Default |
|---|---|---|
| Detic | `threshold` | `0.5` |
| OWL-ViT | `threshold` | `0.1` |
| GroundingDINO | `BOX_THRESHOLD` / `TEXT_THRESHOLD` | `0.30` / `0.25` |
| SAM | `points_per_side` | `32` |

The orchestrator discovers model services through the `MODEL_URLS` env var, populated by `docker-compose.yml`.

## Build-time overrides

The base image picks up CUDA and PyTorch versions from build args, which `setup.sh` forwards from environment variables:

```bash
CUDA_BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 \
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX" \
./setup.sh
```

See the top-level `README.md` for the full list.

## Layout

```
annotate/
├── orchestrator/          # FastAPI service on :8080 that fans out to models
├── models/
│   ├── base/              # shared CUDA/PyTorch base image
│   ├── detic/
│   ├── grounding_dino/
│   ├── owl_vit/
│   ├── sam/
│   └── seem/
├── shared/                # NMS implementations and response schemas
├── config.json            # per-model configuration
├── docker-compose.yml
└── setup.sh
```

## GPU memory

All model services share the host GPU. Running every detector and segmenter concurrently is memory-heavy; on smaller GPUs, disable models you don't need by commenting them out in `docker-compose.yml`.

## License

See [`LICENCE.txt`](LICENCE.txt).

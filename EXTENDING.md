# Extending Labeling Copilot

The default stack curates cat/dog images from Oxford-IIIT Pets. That's a motivating example — the system is designed to be extended at several layers, from "change one env var" all the way to "add a new detection model." This guide walks through the common extensions in order of increasing effort.

## 1. Change the detection vocabulary (zero code)

The annotation service is **open-vocabulary**. You don't retrain anything — you just pass the classes you care about at request time:

```bash
curl -X POST http://localhost:8080/annotate/detect \
  -F "image=@warehouse.jpg" \
  -F "vocabulary=forklift, worker, helmet, cone" \
  -F 'nms_methods=["NON_MAX_SUPPRESSION"]' \
  -F 'nms_params={"iou_threshold":0.5,"sigma":0.5,"min_score":0.1}'
```

Detic, OWL-ViT, and GroundingDINO all handle this out of the box. Accuracy degrades on truly rare or novel concepts; for those, use synthesis to seed representatives you can then use downstream.

## 2. Swap the image pool via env vars

If your data matches the shape of Oxford-IIIT Pets (per-class labels, real photos), the built-in ETL can pick up different classes directly:

```yaml
# retrieval/docker-compose.yml
environment:
  LABEL_CLASSES: forklift,worker,helmet
  NUM_SAMPLE_IMAGES: 500
```

Changing `LABEL_CLASSES` invalidates the cached index, so wipe the volume when you restart:

```bash
cd retrieval
docker compose down -v && docker compose up -d
```

The retrieval service will download the requested classes, compute CLIP features, and rebuild the FAISS index on startup.

## 3. Bring your own data (custom ETL)

For real projects you'll usually have your own images. The retrieval service expects this contract at startup:

- Images uploaded to MinIO under `MINIO_BUCKET`
- CLIP ViT-B/32 features extracted and cached under `CACHE_DIR`
- A FAISS index built over those features
- A registry mapping each image key to an (optional) initial label

`retrieval/etl.py` does exactly this for Oxford-IIIT Pets and CIFAR-10. To use your own data, replace the dataset-loading step and leave the rest intact.

### Minimal custom ETL

```python
# retrieval/etl_mydata.py
from pathlib import Path
from PIL import Image

from etl import (
    upload_images_to_minio,
    extract_and_cache_features,
    build_index,
)

def load_my_images():
    """Yield (image_key, PIL.Image, optional_label) tuples."""
    for p in Path("/app/my-data").glob("**/*.jpg"):
        key = f"images/{p.relative_to('/app/my-data')}"
        yield key, Image.open(p).convert("RGB"), None

def main():
    records = list(load_my_images())
    upload_images_to_minio(records)
    extract_and_cache_features(records)
    build_index()

if __name__ == "__main__":
    main()
```

Point the container's entrypoint at your new ETL and mount your data in:

```yaml
# retrieval/docker-compose.yml
services:
  retrieval:
    volumes:
      - ./my-data:/app/my-data:ro
    command: ["bash", "-c", "python etl_mydata.py && uvicorn api_server:app --host 0.0.0.0 --port 8000"]
```

The sampling, labeling, and auto-label endpoints don't care how the pool was populated — only that the index is there.

### Tips

- Keep keys stable and meaningful (`images/<split>/<filename>`). They appear in `dataset.json` as `source_key`.
- If you already have labels, emit them in the third tuple position so samplers that need labels can start immediately.
- CLIP features are cached on disk; re-running ETL is idempotent unless you change the source set.
- MinIO is an implementation detail — if your pool lives in S3 or GCS, change `etl.py` to upload there instead and point `MINIO_ENDPOINT` accordingly.

## 4. Re-target the curation taxonomy

The `/curate` skill drives scene tagging and gap analysis from a fixed taxonomy (lighting, setting, viewpoint, subject count, pose, difficulty). That taxonomy is specific to photographs of objects/animals. For a very different domain — medical imaging, remote sensing, documents — you'll want different dimensions.

Everything lives in one file: `.claude/commands/curate.md`. Edit the **Scene taxonomy** table to match your domain, then update **Step 5: Gap analysis** and **Step 6: Gap-directed synthesis** with seed-selection rules for your new dimensions. No service code needs to change — the skill is a plain markdown prompt that the agent reads and follows.

Example dimensions for industrial safety:

| Dim | Values |
|---|---|
| environment | `warehouse`, `outdoor_site`, `workshop`, `office` |
| ppe | `full`, `partial`, `none`, `non_applicable` |
| activity | `walking`, `lifting`, `operating_machinery`, `stationary` |
| proximity | `near`, `mid`, `far` |

## 5. Add a detection or segmentation model

Each model in `annotate/models/` is an independent FastAPI service behind the shared `annotation-base` image. The orchestrator discovers them via `MODEL_URLS` and fans every request out in parallel, so adding a new one is additive — no changes to existing services.

1. Create `annotate/models/<yourmodel>/`:

   ```
   Dockerfile          # FROM annotation-base
   app.py              # exposes /health and /annotate
   model.py            # your inference logic
   requirements.txt
   ```

   Match the shape of an existing model (e.g. `annotate/models/owl_vit/`): `POST /annotate` accepts `image` (file) and `vocabulary` (form field), returns `{"annotations": [{"label", "bbox": [xmin,ymin,xmax,ymax], "confidence"}], "image_width", "image_height"}`.

2. Register the service in `annotate/docker-compose.yml` — a new block with a health-check, the `model-net` network, and GPU reservation.

3. Add its URL to `MODEL_URLS` on the orchestrator service (same file).

4. Add the model name to the relevant set in `annotate/orchestrator/app.py`:

   ```python
   DETECTION_MODELS = {"DETIC", "GroundingDINO", "OWL_ViT", "YourModel"}
   # or SEGMENTATION_MODELS = {"SAM", "SEEM", "YourModel"}
   ```

The orchestrator will now include it in fan-out, NMS consensus, and per-model score reporting automatically.

## 6. Add a sampler

Samplers live in `retrieval/samplers/`. The interface is small — subclass `BaseSampler` from `samplers/base.py` and implement `sample(num_samples)`.

1. Create `retrieval/samplers/my_sampler.py` modelled on `random.py` (simple) or `uncertainty.py` (labels-aware).
2. Register it in `retrieval/samplers/__init__.py`.
3. Add it to the sampler registry in `active_learning_session.py`.

Tests run without Docker:

```bash
cd retrieval
pip install -r requirements.txt -r requirements-test.txt
pytest tests/
```

Clients can then request it by name:

```bash
curl -X POST http://localhost:8000/samples \
  -H 'Content-Type: application/json' \
  -d '{"sampler":"my_sampler","num_samples":20}'
```

## 7. Tune synthesis for a new domain

Synthesis takes a `domain` string per job and uses it to steer the vision model's augmentation suggestions. In most cases, just phrasing the domain carefully is enough:

```bash
curl -X POST http://localhost:8090/synthesize \
  -H 'Content-Type: application/json' \
  -d '{
    "image_urls":["/images/warehouse_001.jpg"],
    "num_variants": 3,
    "domain": "warehouse safety imagery: forklifts, workers, helmets, yellow floor markings"
  }'
```

For deeper control — custom suggestion categories, custom prompt templates, different edit strategies — see [`synthesis/ARCHITECTURE.md`](synthesis/ARCHITECTURE.md) and the prompt files under `synthesis/core/`.

## Putting it together

A typical full-extension path for a new domain looks like this:

1. Write a custom ETL that loads your images into retrieval (§3).
2. Pass your open-vocabulary classes at detect time — no model training (§1).
3. Edit the curation taxonomy in `.claude/commands/curate.md` to match your domain (§4).
4. Run `/curate` with a natural-language brief for your task.
5. Let synthesis fill coverage gaps the agent identifies (§7).
6. Consume `agent_output/dataset.json` (COCO format, with per-model detection provenance) in your downstream training pipeline.

If your extension stays within these hook points, you're on the supported path. If you find yourself editing orchestrator internals or patching response schemas, open an issue — that usually signals a missing extension point rather than a need to fork.

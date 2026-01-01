"""FastAPI server for the CLIP active-learning retrieval service.

Startup contract: `etl.py` must have been run and produced
`$CACHE_DIR/etl_complete.json`; the entrypoint script does this. Configuration
is read from environment variables (MINIO_*, LABEL_CLASSES, CACHE_DIR) — see
the README for the full list.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from minio import Minio
from pydantic import BaseModel

from active_learning_session import ActiveLearningSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Active Learning API")

# Global state populated at startup
session: Optional[ActiveLearningSession] = None
minio_client: Optional[Minio] = None
bucket: str = ""
label_classes: List[str] = []
text_embeddings: Optional[np.ndarray] = None  # (num_classes, 512)


# --- Pydantic models ---

class SampleRequest(BaseModel):
    sampler: str
    num_samples: int
    sampler_params: Optional[Dict[str, Any]] = None
    new_labels: Optional[Dict[str, int]] = None


class LabelRequest(BaseModel):
    labels: Dict[str, int]


class AutoLabelRequest(BaseModel):
    image_keys: List[str]
    apply: bool = False


# --- Startup ---

@app.on_event("startup")
def startup():
    global session, minio_client, bucket, label_classes, text_embeddings

    cache_dir = os.environ.get("CACHE_DIR", "cache")
    state_file = os.path.join(cache_dir, "etl_complete.json")

    if not os.path.exists(state_file):
        raise RuntimeError(f"ETL state not found at {state_file}. Run etl.py first.")

    with open(state_file) as f:
        state = json.load(f)

    # Load features
    features = np.load(state["features_path"])
    object_keys = state["object_keys"]
    if "label_classes" not in state:
        raise RuntimeError(
            f"ETL state at {state_file} is missing 'label_classes'. "
            "Re-run etl.py to regenerate it."
        )
    label_classes = state["label_classes"]

    logger.info("Loaded %d images, features %s", len(object_keys), features.shape)
    logger.info("Label classes: %s", label_classes)

    # Initialize session
    session = ActiveLearningSession(
        image_paths=object_keys,
        feature_vectors=features,
    )

    # MinIO client
    endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
    bucket = os.environ.get("MINIO_BUCKET", "images")

    minio_client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

    # Compute CLIP text embeddings for auto-labeling
    text_embeddings = _compute_text_embeddings(label_classes)
    logger.info("Text embeddings computed: %s", text_embeddings.shape)


def _compute_text_embeddings(classes: List[str]) -> np.ndarray:
    """Encode label class prompts with CLIP and return L2-normalized text embeddings."""
    import clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    prompts = [f"a photo of a {cls}" for cls in classes]
    tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    result = emb.cpu().numpy().astype("float32")

    # Release model memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


# --- Helpers ---

def _image_url(key: str) -> str:
    """Return a URL to the /images proxy endpoint for this key."""
    return f"/images/{key}"


def _enrich_samples(image_keys: List[str]) -> List[Dict[str, str]]:
    """Add image URLs to sample image keys."""
    return [
        {"image_key": key, "image_url": _image_url(key)}
        for key in image_keys
    ]


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    if session is None:
        raise HTTPException(status_code=503, detail="Session not initialized")
    result = session.get_status()
    result["label_classes"] = label_classes
    return result


@app.post("/samples")
def get_samples(req: SampleRequest):
    if session is None:
        raise HTTPException(status_code=503, detail="Session not initialized")

    # Add labels first if provided
    if req.new_labels:
        added = session.add_labels(req.new_labels)
        logger.info("Added %d labels before sampling", added)

    try:
        selected_paths, metadata = session.get_samples(
            sampler_name=req.sampler,
            num_samples=req.num_samples,
            sampler_params=req.sampler_params,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "samples": _enrich_samples(selected_paths),
        "metadata": metadata,
    }


@app.post("/labels")
def add_labels(req: LabelRequest):
    if session is None:
        raise HTTPException(status_code=503, detail="Session not initialized")

    try:
        added = session.add_labels(req.labels)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"labels_added": added, "total_labeled": len(session.labeled_indices)}


@app.post("/auto-label")
def auto_label(req: AutoLabelRequest):
    if session is None or text_embeddings is None:
        raise HTTPException(status_code=503, detail="Session not initialized")

    # Look up image embeddings from session
    indices = []
    valid_keys = []
    for key in req.image_keys:
        if key not in session.path_to_idx:
            raise HTTPException(status_code=400, detail=f"Unknown image key: {key}")
        indices.append(session.path_to_idx[key])
        valid_keys.append(key)

    image_emb = session.feature_vectors[indices]  # (N, 512)

    # Dot product similarity (both are L2-normalized)
    similarity = image_emb @ text_embeddings.T  # (N, num_classes)

    # Predictions
    predicted_labels = np.argmax(similarity, axis=1)

    # Confidence via softmax with temperature scaling
    scaled = similarity * 100.0
    exp_scores = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    softmax_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    confidences = np.max(softmax_scores, axis=1)

    predictions = []
    labels_to_apply = {}
    for i, key in enumerate(valid_keys):
        label = int(predicted_labels[i])
        pred = {
            "image_key": key,
            "label": label,
            "class_name": label_classes[label],
            "confidence": round(float(confidences[i]), 4),
        }
        predictions.append(pred)
        labels_to_apply[key] = label

    # Optionally apply labels to session
    applied = 0
    if req.apply:
        applied = session.add_labels(labels_to_apply)

    return {
        "predictions": predictions,
        "labels_applied": applied,
    }


@app.get("/images/{key:path}")
def get_image(key: str):
    """Proxy image content from MinIO."""
    try:
        response = minio_client.get_object(bucket, key)
        return StreamingResponse(
            response.stream(),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename={os.path.basename(key)}"},
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

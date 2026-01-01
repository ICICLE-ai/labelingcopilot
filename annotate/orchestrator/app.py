import asyncio
import json
import os
from typing import Dict, List

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from shared.schemas import (
    DetectOrchestratorResponse,
    DetectionResponse,
    ConsensusResult,
    NMSAnnotation,
    SegmentOrchestratorResponse,
)
from shared.nms import vote_annotations, NMS_REGISTRY

# Model service URLs from environment — e.g. {"DETIC": "http://detic:8000", ...}
MODEL_URLS: Dict[str, str] = json.loads(os.environ.get("MODEL_URLS", "{}"))

DETECTION_MODELS = {"DETIC", "GroundingDINO", "OWL_ViT"}
SEGMENTATION_MODELS = {"SAM", "SEEM"}

app = FastAPI()


async def _call_model(client: httpx.AsyncClient, model_name: str, url: str, image_bytes: bytes, fields: dict) -> dict:
    """Call a single model service's /annotate endpoint."""
    files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
    data = {k: v for k, v in fields.items() if v is not None}
    try:
        resp = await client.post(f"{url}/annotate", files=files, data=data, timeout=120.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e), "model_name": model_name}


def _detection_to_dual_format(annotations: List[dict], img_w: int, img_h: int) -> Dict[str, list]:
    """Convert flat annotation list to the dual-format dict {label: [boxes]} used by NMS.

    Dual format: [[xmin_norm, xmin_abs], [ymin_norm, ymin_abs], [xmax_norm, xmax_abs], [ymax_norm, ymax_abs]]
    """
    result: Dict[str, list] = {}
    for ann in annotations:
        label = ann["label"]
        xmin, ymin, xmax, ymax = ann["bbox"]
        box = [
            [xmin / img_w, xmin],
            [ymin / img_h, ymin],
            [xmax / img_w, xmax],
            [ymax / img_h, ymax],
        ]
        result.setdefault(label, []).append(box)
    return result


# ---------- Detection endpoint ----------

@app.post("/annotate/detect")
async def annotate_detect(
    image: UploadFile = File(...),
    vocabulary: str = Form(...),
    models: str = Form("null"),
    nms_methods: str = Form('["NON_MAX_SUPPRESSION"]'),
    nms_params: str = Form('{"iou_threshold": 0.5, "sigma": 0.5, "min_score": 0.1, "distance_threshold": 1.0}'),
):
    image_bytes = await image.read()
    requested_models = json.loads(models)
    if requested_models is None:
        requested_models = [m for m in MODEL_URLS if m in DETECTION_MODELS]

    nms_method_list = json.loads(nms_methods)
    params = json.loads(nms_params)
    iou_threshold = params.pop("iou_threshold", 0.5)

    # Normalize vocabulary: accept either JSON array or comma-separated string
    try:
        vocab_list = json.loads(vocabulary)
    except (json.JSONDecodeError, TypeError):
        vocab_list = [v.strip() for v in vocabulary.split(",") if v.strip()]
    vocab_json = json.dumps(vocab_list)

    # Fan out to detection models
    async with httpx.AsyncClient() as client:
        tasks = []
        for name in requested_models:
            url = MODEL_URLS.get(name)
            if url is None:
                continue
            fields = {"vocabulary": vocab_json}
            tasks.append(_call_model(client, name, url, image_bytes, fields))
        raw_results = await asyncio.gather(*tasks)

    # Filter out errors
    successful = [r for r in raw_results if "error" not in r]

    # Build NMS input — convert each model's results to dual-format annotations
    all_annotations = []
    for result in successful:
        img_w = result.get("image_width", 1)
        img_h = result.get("image_height", 1)
        dual = _detection_to_dual_format(result.get("annotations", []), img_w, img_h)
        all_annotations.append(dual)

    # Run consensus NMS
    consensus_results = []
    for method in nms_method_list:
        if method not in NMS_REGISTRY:
            continue
        voted, scores = vote_annotations(all_annotations, iou_threshold, method, params)

        nms_annotations = []
        for label, boxes in voted.items():
            label_scores = scores.get(label, [0.0] * len(boxes))
            for box, score in zip(boxes, label_scores):
                nms_annotations.append(NMSAnnotation(
                    label=label,
                    confidence=float(score),
                    bbox=[float(box[0][1]), float(box[1][1]), float(box[2][1]), float(box[3][1])],
                ))
        consensus_results.append(ConsensusResult(nms_method=method, annotations=nms_annotations))

    return DetectOrchestratorResponse(
        raw_results=[DetectionResponse(**r) for r in successful],
        consensus=consensus_results,
    )


# ---------- Segmentation endpoint ----------

@app.post("/annotate/segment")
async def annotate_segment(
    image: UploadFile = File(...),
    models: str = Form("null"),
):
    image_bytes = await image.read()
    requested_models = json.loads(models)
    if requested_models is None:
        requested_models = [m for m in MODEL_URLS if m in SEGMENTATION_MODELS]

    async with httpx.AsyncClient() as client:
        tasks = []
        for name in requested_models:
            url = MODEL_URLS.get(name)
            if url is None:
                continue
            tasks.append(_call_model(client, name, url, image_bytes, {}))
        raw_results = await asyncio.gather(*tasks)

    return SegmentOrchestratorResponse(raw_results=raw_results)


# ---------- Health endpoint ----------

@app.get("/health")
async def health():
    statuses = {}
    async with httpx.AsyncClient() as client:
        for name, url in MODEL_URLS.items():
            try:
                resp = await client.get(f"{url}/health", timeout=5.0)
                statuses[name] = resp.json()
            except Exception as e:
                statuses[name] = {"status": "unreachable", "error": str(e)}
    return statuses

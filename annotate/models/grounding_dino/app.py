import json
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, UploadFile

from shared.schemas import DetectionResponse, HealthResponse
from model import GroundingDINOModel

_model: GroundingDINOModel | None = None
_status = "loading"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _status
    try:
        config = json.loads(os.environ.get("MODEL_CONFIG", "{}"))
        _model = GroundingDINOModel(config)
        _status = "ready"
    except Exception as e:
        _status = "error"
        raise e
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    device = str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    return HealthResponse(status=_status, model_name="GroundingDINO", gpu_device=f"cuda:{device}" if device != "cpu" else "cpu")


@app.post("/annotate", response_model=DetectionResponse)
async def annotate(
    image: UploadFile = File(...),
    vocabulary: str = Form(...),
    params: str = Form("{}"),
):
    image_bytes = await image.read()
    vocab = json.loads(vocabulary)
    overrides = json.loads(params)
    if overrides.get("BOX_THRESHOLD"):
        _model.box_threshold = overrides["BOX_THRESHOLD"]
    if overrides.get("TEXT_THRESHOLD"):
        _model.text_threshold = overrides["TEXT_THRESHOLD"]
    result = _model.annotate(image_bytes, vocab)
    return DetectionResponse(**result)

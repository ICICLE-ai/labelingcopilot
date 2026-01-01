import json
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, UploadFile

from shared.schemas import SAMResponse, HealthResponse
from model import SAMModel

_model: SAMModel | None = None
_status = "loading"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _status
    try:
        config = json.loads(os.environ.get("MODEL_CONFIG", "{}"))
        _model = SAMModel(config)
        _status = "ready"
    except Exception as e:
        _status = "error"
        raise e
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    device = str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    return HealthResponse(status=_status, model_name="SAM", gpu_device=f"cuda:{device}" if device != "cpu" else "cpu")


@app.post("/annotate", response_model=SAMResponse)
async def annotate(
    image: UploadFile = File(...),
    params: str = Form("{}"),
):
    image_bytes = await image.read()
    result = _model.annotate(image_bytes)
    return SAMResponse(**result)

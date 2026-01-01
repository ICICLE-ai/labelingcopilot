from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class BBoxAnnotation(BaseModel):
    label: str
    confidence: float
    bbox: List[float]  # [xmin, ymin, xmax, ymax] in absolute pixels


class DetectionResponse(BaseModel):
    model_name: str
    annotations: List[BBoxAnnotation]
    image_width: int
    image_height: int


class MaskAnnotation(BaseModel):
    bbox: List[float]  # [x, y, w, h]
    area: int
    stability_score: float


class SAMResponse(BaseModel):
    model_name: str = "SAM"
    masks: List[MaskAnnotation]
    image_width: int
    image_height: int


class SegmentAnnotation(BaseModel):
    label: str
    area: int
    bbox: List[float]  # [xmin, ymin, xmax, ymax]
    is_thing: Optional[bool] = None


class SEEMResponse(BaseModel):
    model_name: str = "SEEM"
    segments: List[SegmentAnnotation]
    image_width: int
    image_height: int


class HealthResponse(BaseModel):
    status: str  # "loading" | "ready" | "error"
    model_name: str
    gpu_device: str


class NMSAnnotation(BaseModel):
    label: str
    confidence: float
    bbox: List[float]


class ConsensusResult(BaseModel):
    nms_method: str
    annotations: List[NMSAnnotation]


class DetectOrchestratorResponse(BaseModel):
    raw_results: List[DetectionResponse]
    consensus: List[ConsensusResult]


class SegmentOrchestratorResponse(BaseModel):
    raw_results: List[Dict[str, Any]]

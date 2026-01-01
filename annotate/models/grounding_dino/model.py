import os
import subprocess
import torch
import numpy as np
import cv2
from typing import List

import groundingdino
from groundingdino.util.inference import load_model, load_image as gdino_load_image, predict
from torchvision.ops import box_convert


def _resolve_config_path(config_name: str) -> str:
    """Resolve GroundingDINO config path from the installed package."""
    if os.path.isabs(config_name) and os.path.exists(config_name):
        return config_name
    # Try package config directory
    pkg_dir = os.path.dirname(groundingdino.__file__)
    candidate = os.path.join(pkg_dir, "config", config_name)
    if os.path.exists(candidate):
        return candidate
    return config_name


class GroundingDINOModel:
    def __init__(self, config: dict):
        self.config_path = _resolve_config_path(config.get("CONFIG_PATH", "GroundingDINO_SwinT_OGC.py"))
        self.weights_path = config.get("WEIGHTS_PATH", "groundingdino_swint_ogc.pth")
        self.model_url = config.get("MODEL_URL", "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
        self.box_threshold = config.get("BOX_THRESHOLD", 0.30)
        self.text_threshold = config.get("TEXT_THRESHOLD", 0.25)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Download weights if not present
        if not os.path.exists(self.weights_path):
            subprocess.run(["wget", "-q", self.model_url, "-O", self.weights_path], check=True)

        self.model = load_model(self.config_path, self.weights_path, device=self.device)

    def annotate(self, image_bytes: bytes, vocabulary: List[str]) -> dict:
        # Write to temp file for GroundingDINO's load_image
        tmp_path = "/tmp/_gdino_input.jpg"
        with open(tmp_path, "wb") as f:
            f.write(image_bytes)

        image_source, image_tensor = gdino_load_image(tmp_path)
        h, w = image_source.shape[:2]

        text_prompt = ", ".join(vocabulary)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        boxes_abs = boxes * torch.Tensor([w, h, w, h])
        boxes_xyxy = box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        scores = torch.sigmoid(logits).cpu().detach().numpy()

        annotations = []
        for box, phrase, score in zip(boxes_xyxy, phrases, scores):
            annotations.append({
                "label": phrase,
                "confidence": float(score),
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            })

        return {
            "model_name": "GroundingDINO",
            "annotations": annotations,
            "image_width": w,
            "image_height": h,
        }

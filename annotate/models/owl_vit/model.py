import torch
import numpy as np
import cv2
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from typing import List, Tuple


class OWLViTModel:
    def __init__(self, config: dict):
        self.model_name = config.get("model", "google/owlvit-base-patch32")
        self.threshold = config.get("threshold", 0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = OwlViTForObjectDetection.from_pretrained(self.model_name).to(self.device)
        self.processor = OwlViTProcessor.from_pretrained(self.model_name)
        self.model.eval()

    def annotate(self, image_bytes: bytes, vocabulary: List[str]) -> dict:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w = image.shape[:2]

        inputs = self.processor(text=vocabulary, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = torch.max(outputs["logits"][0], dim=-1)
        scores = torch.sigmoid(logits.values).cpu().numpy()
        label_indices = logits.indices.cpu().numpy()
        pred_boxes = outputs["pred_boxes"][0].cpu().numpy()

        annotations = []
        for box, label_idx, score in zip(pred_boxes, label_indices, scores):
            if score < self.threshold:
                continue
            cx, cy, bw, bh = box
            xmin = float((cx - bw / 2) * w)
            ymin = float((cy - bh / 2) * h)
            xmax = float((cx + bw / 2) * w)
            ymax = float((cy + bh / 2) * h)
            annotations.append({
                "label": vocabulary[label_idx],
                "confidence": float(score),
                "bbox": [xmin, ymin, xmax, ymax],
            })

        return {
            "model_name": "OWL_ViT",
            "annotations": annotations,
            "image_width": w,
            "image_height": h,
        }

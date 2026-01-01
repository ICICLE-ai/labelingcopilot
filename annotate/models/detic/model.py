import os
import sys
import torch
import numpy as np
import cv2
from typing import List

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

# CenterNet2 and Detic imports — these paths are set up in the Dockerfile
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder


class DETICModel:
    def __init__(self, config: dict):
        self.threshold = config.get("threshold", 0.5)
        self.prompt = config.get("prompt", "a ")
        config_path = config.get("CONFIG_PATH", "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
        weight_path = config.get("model_weight_path", "")
        default_vocabulary = config.get("vocabulary", [])

        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weight_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.predictor = DefaultPredictor(cfg)

        # Set up default vocabulary if provided
        if default_vocabulary:
            self._set_vocabulary(default_vocabulary)

    def _get_clip_embeddings(self, vocabulary: List[str]):
        text_encoder = build_text_encoder(pretrain=True)
        text_encoder.eval()
        texts = [self.prompt + x for x in vocabulary]
        emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb

    def _set_vocabulary(self, vocabulary: List[str]):
        self.vocabulary = vocabulary
        # Clear existing metadata to allow vocabulary changes
        MetadataCatalog.remove("__unused") if "__unused" in MetadataCatalog else None
        metadata = MetadataCatalog.get("__unused")
        metadata.thing_classes = vocabulary
        classifier = self._get_clip_embeddings(vocabulary)
        num_classes = len(vocabulary)
        reset_cls_test(self.predictor.model, classifier, num_classes)

    def annotate(self, image_bytes: bytes, vocabulary: List[str]) -> dict:
        # Re-set vocabulary if it changed
        if not hasattr(self, "vocabulary") or vocabulary != self.vocabulary:
            self._set_vocabulary(vocabulary)

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w = image.shape[:2]

        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")

        labels = instances.pred_classes.tolist()
        boxes = instances.pred_boxes.tensor.tolist()
        scores = instances.scores.tolist()

        annotations = []
        for box, label_idx, score in zip(boxes, labels, scores):
            if score < self.threshold:
                continue
            annotations.append({
                "label": self.vocabulary[label_idx],
                "confidence": float(score),
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            })

        return {
            "model_name": "DETIC",
            "annotations": annotations,
            "image_width": w,
            "image_height": h,
        }

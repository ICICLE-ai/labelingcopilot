import os
import subprocess
import torch
import numpy as np
import cv2
from typing import List

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class SAMModel:
    def __init__(self, config: dict):
        self.checkpoint = config.get("checkpoint", "sam_vit_h_4b8939.pth")
        self.model_url = config.get("model_path", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        self.model_type = config.get("model", "vit_h")

        # Download checkpoint if not available
        if not os.path.exists(self.checkpoint):
            subprocess.run(["wget", "-q", self.model_url, "-O", self.checkpoint], check=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(device)

        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=config.get("points_per_side", 32),
            pred_iou_thresh=config.get("pred_iou_thresh", 0.86),
            stability_score_thresh=config.get("stability_score_thresh", 0.92),
            crop_n_layers=config.get("crop_n_layers", 1),
            crop_n_points_downscale_factor=config.get("crop_n_points_downscale_factor", 2),
            min_mask_region_area=config.get("min_mask_region_area", 100),
        )

    def annotate(self, image_bytes: bytes) -> dict:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Resize large images to limit memory usage on CPU
        max_side = 1024
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        masks = self.mask_generator.generate(image_rgb)

        result_masks = []
        for mask in masks:
            result_masks.append({
                "bbox": [float(v) for v in mask["bbox"]],  # [x, y, w, h]
                "area": int(mask["area"]),
                "stability_score": float(mask["stability_score"]),
            })

        return {
            "model_name": "SAM",
            "masks": result_masks,
            "image_width": w,
            "image_height": h,
        }

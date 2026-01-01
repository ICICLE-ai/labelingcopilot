import os
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Optional
from torchvision import transforms

from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES


t = []
t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
transform = transforms.Compose(t)


class SEEMModel:
    def __init__(self, config: dict):
        config_path = config.get("CONFIG_PATH", "configs/seem/focall_unicl_lang_demo.yaml")
        model_name = config.get("model", "seem_focall_v0.pt")
        model_url = config.get("model_path", "https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt")

        # Download if not present
        if not os.path.exists(model_name):
            os.makedirs(os.path.dirname(model_name) or ".", exist_ok=True)
            os.system(f"wget -q {model_url} -O {model_name}")

        opt = load_opt_from_config_files([config_path])
        opt = init_distributed(opt)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BaseModel(opt, build_model(opt)).from_pretrained(model_name).eval().to(self.device)

        with torch.no_grad():
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                COCO_PANOPTIC_CLASSES + ["background"], is_eval=True
            )
            model.model.sem_seg_head.num_classes = len(COCO_PANOPTIC_CLASSES)

        self.model = model
        self.default_classes = list(COCO_PANOPTIC_CLASSES)

    def annotate(self, image_bytes: bytes, vocabulary: Optional[List[str]] = None) -> dict:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = image.size

        # If custom vocabulary provided, re-init text embeddings
        if vocabulary and vocabulary != self.default_classes:
            with torch.no_grad():
                self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                    vocabulary + ["background"], is_eval=True
                )
                self.model.model.sem_seg_head.num_classes = len(vocabulary)
            classes = vocabulary
        else:
            classes = self.default_classes

        # Preprocess image same way as interactive_infer_image
        image_resized = transform(image)
        width = image_resized.size[0]
        height = image_resized.size[1]
        image_np = np.asarray(image_resized)
        images_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1)

        data = {"image": images_tensor, "height": height, "width": width}
        batch_inputs = [data]

        # Set task switches for panoptic mode
        self.model.model.task_switch['spatial'] = False
        self.model.model.task_switch['visual'] = False
        self.model.model.task_switch['grounding'] = False
        self.model.model.task_switch['audio'] = False

        # Run panoptic inference
        from detectron2.data import MetadataCatalog
        self.model.model.metadata = MetadataCatalog.get('coco_2017_train_panoptic')

        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                results = self.model.model.evaluate(batch_inputs)
        else:
            with torch.no_grad():
                results = self.model.model.evaluate(batch_inputs)

        # Extract panoptic segmentation data
        segments = []
        if results and 'panoptic_seg' in results[-1]:
            pano_seg = results[-1]['panoptic_seg'][0]
            pano_seg_info = results[-1]['panoptic_seg'][1]
            seg_np = pano_seg.cpu().numpy()

            for seg_info in pano_seg_info:
                seg_id = seg_info["id"]
                cat_id = seg_info.get("category_id", -1)
                is_thing = seg_info.get("isthing", False)
                seg_mask = seg_np == seg_id
                area = int(seg_mask.sum())
                if area == 0:
                    continue
                ys, xs = np.where(seg_mask)
                label = classes[cat_id] if 0 <= cat_id < len(classes) else "unknown"
                segments.append({
                    "label": label,
                    "area": area,
                    "bbox": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                    "is_thing": is_thing,
                })

        return {
            "model_name": "SEEM",
            "segments": segments,
            "image_width": w,
            "image_height": h,
        }

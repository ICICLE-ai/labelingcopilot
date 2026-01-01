"""CLIP-based feature extraction for image retrieval.

Wraps OpenAI CLIP (ViT-B/32 by default) to produce L2-normalized image embeddings
suitable for cosine-similarity search against the FAISS index.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CLIPFeatureExtractor:
    """Extracts L2-normalized CLIP image embeddings."""

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        try:
            import clip
        except ImportError as e:
            raise ImportError(
                "CLIP not installed. Install with: "
                "pip install git+https://github.com/openai/CLIP.git"
            ) from e

        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        logger.info("Loaded CLIP model: %s", model_name)

    @torch.no_grad()
    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        features = self.model.encode_image(tensors)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32")

    def extract_from_paths(
        self, image_paths: List[str], batch_size: int = 32
    ) -> np.ndarray:
        """Extract features for the given image paths in fixed-size batches.

        Unloadable images are replaced with a blank placeholder so the output
        rows stay aligned with `image_paths`.
        """
        all_features = []
        for i in tqdm(
            range(0, len(image_paths), batch_size), desc="Extracting features"
        ):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []
            for path in batch_paths:
                try:
                    batch_images.append(Image.open(path).convert("RGB"))
                except Exception as e:
                    logger.warning("Failed to load %s: %s", path, e)
                    batch_images.append(Image.new("RGB", (224, 224)))
            all_features.append(self.extract_batch(batch_images))
        return np.vstack(all_features)


def extract_features_from_directory(
    image_dir: str,
    output_file: str,
    model_name: str = "ViT-B/32",
    batch_size: int = 32,
    extensions: Optional[set] = None,
) -> tuple[np.ndarray, List[str]]:
    """Extract CLIP features for all images under `image_dir` and save to disk.

    Returns the (features, image_paths) tuple. A `<output_file>_paths.txt`
    sidecar is written alongside the .npy file for traceability.
    """
    extensions = extensions or {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_paths = sorted(
        str(p)
        for p in Path(image_dir).rglob("*")
        if p.suffix.lower() in extensions
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    logger.info("Found %d images", len(image_paths))
    extractor = CLIPFeatureExtractor(model_name=model_name)
    features = extractor.extract_from_paths(image_paths, batch_size=batch_size)

    np.save(output_file, features)
    logger.info("Saved features to %s: shape=%s", output_file, features.shape)

    paths_file = output_file.replace(".npy", "_paths.txt")
    with open(paths_file, "w") as f:
        f.write("\n".join(image_paths))
    logger.info("Saved image paths to %s", paths_file)

    return features, image_paths

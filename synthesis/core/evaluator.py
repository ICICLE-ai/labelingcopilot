"""
Quality evaluation using dgm_eval metrics and Forte OOD detection.
Implements the eval_quality(metrics) action.
"""

import sys
import os
from typing import List, Dict, Optional, Any
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Add dgm_eval to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dgm_eval"))

# Import dgm_eval metrics
try:
    import prdc
    import fd
    import inception_score
    import authpct
    import vendi
    import sw
    DGM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: dgm_eval metrics not available: {e}")
    DGM_AVAILABLE = False

# Import Forte if available
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "forte-api"))
    from forte_api import ForteOODDetector
    FORTE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Forte OOD detection not available: {e}")
    FORTE_AVAILABLE = False

from config.config import Config, EvaluationConfig, ForteConfig


class FeatureExtractor:
    """Extracts features from images for metric computation."""

    def __init__(self, model_name: str = "inception", device: Optional[str] = None):
        """
        Initialize feature extractor.

        Args:
            model_name: Model to use ("inception", "clip", "dinov2")
            device: Device to use (cuda/mps/cpu)
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model_name = model_name

        # Load model
        if model_name == "inception":
            from torchvision.models import inception_v3
            self.model = inception_v3(pretrained=True, transform_input=False)
            self.model.fc = torch.nn.Identity()  # Remove final layer
            self.model.eval()
            self.model.to(device)

            self.transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def extract_from_images(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract features from a list of images.

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing

        Returns:
            Feature array of shape (N, D)
        """
        features = []

        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []

                for path in batch_paths:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_images.append(img_tensor)

                batch_tensor = torch.stack(batch_images).to(self.device)
                batch_features = self.model(batch_tensor)

                features.append(batch_features.cpu().numpy())

        return np.concatenate(features, axis=0)


class QualityEvaluator:
    """Evaluates image quality using various metrics."""

    def __init__(self, config: Config):
        """
        Initialize quality evaluator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.eval_config = config.evaluation
        self.forte_config = config.forte

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()

        # Initialize Forte detector if enabled
        self.forte_detector = None
        if FORTE_AVAILABLE and self.forte_config.enabled:
            self.forte_detector = ForteOODDetector(
                method=self.forte_config.method,
                nearest_k=self.forte_config.k_neighbors,
                embedding_dir=self.forte_config.embedding_dir
            )

    def eval_quality(
        self,
        real_image_paths: List[str],
        generated_image_paths: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate quality of generated images.

        Args:
            real_image_paths: Paths to real/original images
            generated_image_paths: Paths to generated/augmented images
            metrics: List of metrics to compute (if None, uses config)

        Returns:
            Dictionary of metric scores
        """
        results = {}

        # Extract features
        print("Extracting features from real images...")
        real_features = self.feature_extractor.extract_from_images(real_image_paths)

        print("Extracting features from generated images...")
        gen_features = self.feature_extractor.extract_from_images(generated_image_paths)

        # Determine which metrics to compute
        if metrics is None:
            # Use config
            metrics_to_compute = []
            if self.eval_config.compute_prdc:
                metrics_to_compute.append("prdc")
            if self.eval_config.compute_fd:
                metrics_to_compute.append("fd")
            if self.eval_config.compute_inception_score:
                metrics_to_compute.append("inception_score")
            if self.eval_config.compute_authpct:
                metrics_to_compute.append("authpct")
            if self.eval_config.compute_vendi:
                metrics_to_compute.append("vendi")
            if self.eval_config.compute_sw:
                metrics_to_compute.append("sw")
        else:
            metrics_to_compute = metrics

        # Compute metrics
        for metric in metrics_to_compute:
            try:
                if metric == "prdc":
                    results.update(self._compute_prdc(real_features, gen_features))
                elif metric == "fd":
                    results["frechet_distance"] = self._compute_fd(real_features, gen_features)
                elif metric == "inception_score":
                    results["inception_score"] = self._compute_inception_score(gen_features)
                elif metric == "authpct":
                    results["authenticity_pct"] = self._compute_authpct(real_features, gen_features)
                elif metric == "vendi":
                    results["vendi_score"] = self._compute_vendi(gen_features)
                elif metric == "sw":
                    results["sliced_wasserstein"] = self._compute_sw(real_features, gen_features)
            except Exception as e:
                print(f"Warning: Failed to compute {metric}: {e}")
                results[metric] = None

        return results

    def eval_single_image_ood(self, image_path: str) -> Optional[float]:
        """
        Evaluate if a single image is out-of-distribution.

        Args:
            image_path: Path to image

        Returns:
            OOD probability score (higher = more in-distribution) or None if Forte disabled
        """
        if not self.forte_detector:
            return None

        try:
            # Forte extracts features internally, pass image path directly
            scores = self.forte_detector.predict_proba([image_path])
            return float(scores[0])
        except Exception as e:
            print(f"Warning: OOD detection failed: {e}")
            return None

    def eval_batch_ood(self, image_paths: List[str]) -> Optional[np.ndarray]:
        """
        Evaluate OOD scores for multiple images.

        Args:
            image_paths: List of image paths

        Returns:
            Array of OOD scores or None if Forte disabled
        """
        if not self.forte_detector:
            return None

        try:
            # Forte extracts features internally, pass image paths directly
            scores = self.forte_detector.predict_proba(image_paths)
            return scores
        except Exception as e:
            print(f"Warning: OOD detection failed: {e}")
            return None

    def fit_ood_detector(self, real_image_paths: List[str]):
        """
        Fit OOD detector on real images.

        Args:
            real_image_paths: Paths to real/in-distribution images
        """
        if not self.forte_detector:
            print("Warning: Forte OOD detection not available")
            return

        print("Fitting OOD detector...")
        # Forte extracts features internally, pass image paths directly
        self.forte_detector.fit(real_image_paths)
        print("OOD detector fitted successfully")

    def filter_by_ood(
        self,
        image_paths: List[str],
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        Filter images based on OOD scores.

        Args:
            image_paths: List of image paths
            threshold: OOD threshold (if None, uses config)

        Returns:
            List of in-distribution image paths
        """
        if not self.forte_detector:
            print("Warning: Forte OOD detection not available, returning all images")
            return image_paths

        if threshold is None:
            threshold = self.forte_config.threshold

        scores = self.eval_batch_ood(image_paths)
        if scores is None:
            return image_paths

        filtered = [path for path, score in zip(image_paths, scores) if score > threshold]

        print(f"Filtered {len(image_paths)} images -> {len(filtered)} in-distribution (threshold={threshold})")

        return filtered

    # Private methods for specific metrics

    def _compute_prdc(self, real_features: np.ndarray, gen_features: np.ndarray) -> Dict[str, float]:
        """Compute Precision, Recall, Density, Coverage."""
        try:
            results = prdc.compute_prdc(
                real_features=real_features,
                fake_features=gen_features,
                nearest_k=self.eval_config.prdc_k_neighbors
            )
            return {
                "precision": float(results['precision']),
                "recall": float(results['recall']),
                "density": float(results['density']),
                "coverage": float(results['coverage'])
            }
        except Exception as e:
            raise RuntimeError(f"PRDC computation failed: {e}")

    def _compute_fd(self, real_features: np.ndarray, gen_features: np.ndarray) -> float:
        """Compute Frechet Distance."""
        try:
            return float(fd.compute_FD_with_reps(real_features, gen_features))
        except Exception as e:
            raise RuntimeError(f"FD computation failed: {e}")

    def _compute_inception_score(self, gen_features: np.ndarray) -> float:
        """Compute Inception Score."""
        try:
            # Note: This requires logits, not features
            # For proper IS, we need to use the full inception model
            # This is a simplified version
            mean_score, _ = inception_score.compute_inception_score(gen_features)
            return float(mean_score)
        except Exception as e:
            raise RuntimeError(f"Inception Score computation failed: {e}")

    def _compute_authpct(self, real_features: np.ndarray, gen_features: np.ndarray) -> float:
        """Compute Authenticity Percentage."""
        try:
            return float(authpct.compute_authpct(real_features, gen_features))
        except Exception as e:
            raise RuntimeError(f"Authenticity % computation failed: {e}")

    def _compute_vendi(self, gen_features: np.ndarray) -> float:
        """Compute Vendi Score."""
        try:
            return float(vendi.compute_vendi_score(gen_features))
        except Exception as e:
            raise RuntimeError(f"Vendi Score computation failed: {e}")

    def _compute_sw(self, real_features: np.ndarray, gen_features: np.ndarray) -> float:
        """Compute Sliced Wasserstein Distance."""
        try:
            return float(sw.compute_sw(real_features, gen_features))
        except Exception as e:
            raise RuntimeError(f"Sliced Wasserstein computation failed: {e}")

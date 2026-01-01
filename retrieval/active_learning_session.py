"""Stateful active learning session backing the FastAPI service.

Owns the in-memory pool state for one running service: image paths, CLIP
feature vectors, the FAISS index, the labeled-sample set, and the registered
samplers. The API server holds a single `ActiveLearningSession` instance for
the lifetime of the process.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from config import IndexConfig
from index_manager import IndexBuilder
from samplers import (
    InformativeClusterDiverseSampler,
    KCenterSampler,
    MarginSampler,
    RandomSampler,
    RepresentativeSampler,
)

logger = logging.getLogger(__name__)


class ActiveLearningSession:
    """Manages stateful active learning session for image labeling."""

    def __init__(
        self,
        image_paths: List[str],
        feature_vectors: np.ndarray,
        index_config: Optional[IndexConfig] = None,
    ):
        """
        Initialize active learning session.

        Args:
            image_paths: List of paths to all images in the pool
            feature_vectors: Precomputed feature embeddings (N, dim)
            index_config: FAISS index configuration (default: Flat)
        """
        if len(image_paths) != len(feature_vectors):
            raise ValueError(
                f"Mismatch: {len(image_paths)} images but {len(feature_vectors)} feature vectors"
            )

        self.image_paths = image_paths
        self.feature_vectors = feature_vectors.astype("float32")
        self.n_total = len(image_paths)
        self.dimension = feature_vectors.shape[1]

        # Build FAISS index
        if index_config is None:
            # Auto-select index based on pool size
            if self.n_total < 10_000:
                index_config = IndexConfig("Flat")
            elif self.n_total < 100_000:
                index_config = IndexConfig("IVF256,Flat")
            else:
                index_config = IndexConfig("IVF4096,Flat")

        self.index_config = index_config
        logger.info("Building index for %d images", self.n_total)
        builder = IndexBuilder(index_config)
        self.index = builder.build(self.feature_vectors)

        # Labeled data tracking
        self.labeled_indices: Set[int] = set()
        self.labels: Dict[int, int] = {}  # index -> label

        # Path to index mapping for fast lookup
        self.path_to_idx: Dict[str, int] = {
            path: idx for idx, path in enumerate(image_paths)
        }

        logger.info(
            "Session initialized: %d images, dim=%d", self.n_total, self.dimension
        )

    def add_labels(self, new_labels: Dict[str, int]) -> int:
        """
        Add new labels to the session.

        Args:
            new_labels: Dictionary mapping image_path -> label

        Returns:
            Number of new labels added

        Raises:
            ValueError: If image path not found in pool
        """
        added_count = 0
        for img_path, label in new_labels.items():
            if img_path not in self.path_to_idx:
                raise ValueError(f"Image path not found in pool: {img_path}")

            idx = self.path_to_idx[img_path]
            if idx not in self.labeled_indices:
                self.labeled_indices.add(idx)
                self.labels[idx] = label
                added_count += 1
            else:
                # Update existing label
                self.labels[idx] = label

        logger.info(
            "Added %d new labels (total: %d)", added_count, len(self.labeled_indices)
        )
        return added_count

    def get_samples(
        self,
        sampler_name: str,
        num_samples: int,
        sampler_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Query samples using specified sampler.

        Args:
            sampler_name: One of: "random", "kcenter", "margin", "representative", "informative_cluster_diverse"
            num_samples: Number of samples to retrieve
            sampler_params: Optional sampler-specific parameters

        Returns:
            Tuple of (selected_image_paths, metadata)

        Raises:
            ValueError: If sampler requirements not met or invalid sampler name
        """
        sampler_params = sampler_params or {}

        # Validate sampler requirements
        self._validate_sampler_requirements(sampler_name)

        # Create sampler instance
        sampler = self._create_sampler(sampler_name, sampler_params)

        # Get labeled IDs and labels
        labeled_ids = list(self.labeled_indices)
        labeled_labels = [self.labels[idx] for idx in labeled_ids]

        logger.info(
            "Sampling with %s (labeled=%d, unlabeled=%d)",
            sampler_name, len(labeled_ids), self.n_total - len(labeled_ids),
        )

        try:
            if sampler_name == "random":
                selected_indices = sampler.retrieve(labeled_ids, num_samples)
            elif sampler_name == "kcenter":
                selected_indices = sampler.retrieve(labeled_ids, num_samples)
            elif sampler_name in [
                "margin",
                "representative",
                "informative_cluster_diverse",
            ]:
                selected_indices = sampler.retrieve(
                    labeled_ids, labeled_labels, num_samples
                )
            else:
                raise ValueError(f"Unknown sampler: {sampler_name}")

        except Exception as e:
            raise RuntimeError(f"Sampling failed: {e}") from e

        # Convert indices to paths
        selected_paths = [self.image_paths[int(idx)] for idx in selected_indices]

        # Metadata
        metadata = {
            "sampler_used": sampler_name,
            "params_used": sampler_params,
            "samples_requested": num_samples,
            "samples_returned": len(selected_paths),
            "unlabeled_remaining": self.n_total - len(self.labeled_indices),
            "labeled_total": len(self.labeled_indices),
        }

        logger.info("Selected %d samples", len(selected_paths))
        return selected_paths, metadata

    def get_status(self) -> Dict[str, Any]:
        """
        Get current session status.

        Returns:
            Dictionary with session statistics and available samplers
        """
        return {
            "total_images": self.n_total,
            "labeled_count": len(self.labeled_indices),
            "unlabeled_count": self.n_total - len(self.labeled_indices),
            "index_info": {
                "type": self.index_config.index_string,
                "vectors_indexed": self.index.ntotal,
                "dimension": self.dimension,
            },
            "available_samplers": self._get_sampler_info(),
            "labeled_images": [
                self.image_paths[idx] for idx in sorted(self.labeled_indices)
            ],
            "label_distribution": self._get_label_distribution(),
        }

    def _validate_sampler_requirements(self, sampler_name: str):
        """Validate that sampler requirements are met."""
        n_labeled = len(self.labeled_indices)

        requirements = {
            "random": (0, False),
            "kcenter": (0, False),
            "margin": (2, True),
            "representative": (1, True),
            "informative_cluster_diverse": (2, True),
        }

        if sampler_name not in requirements:
            raise ValueError(
                f"Invalid sampler: {sampler_name}. Valid options: {list(requirements.keys())}"
            )

        min_labeled, needs_labels = requirements[sampler_name]

        if n_labeled < min_labeled:
            raise ValueError(
                f"{sampler_name} requires at least {min_labeled} labeled samples (have {n_labeled})"
            )

        if needs_labels and n_labeled > 0:
            # Check we have both classes
            unique_labels = set(self.labels.values())
            if len(unique_labels) < 2:
                raise ValueError(
                    f"{sampler_name} requires at least 2 different class labels (have {len(unique_labels)})"
                )

    def _create_sampler(self, sampler_name: str, params: Dict[str, Any]):
        """Create sampler instance with parameters."""
        # Define valid parameters for each sampler
        valid_params = {
            "random": set(),
            "kcenter": {"candidate_pool_size"},
            "margin": {"local_search_k"},
            "representative": {"local_search_k"},
            "informative_cluster_diverse": {"candidate_pool_size", "n_clusters"},
        }

        # Validate parameters
        if sampler_name in valid_params:
            invalid_params = set(params.keys()) - valid_params[sampler_name]
            if invalid_params:
                raise ValueError(
                    f"Invalid parameters for {sampler_name}: {invalid_params}. "
                    f"Valid parameters: {valid_params[sampler_name] or 'none'}"
                )

        if sampler_name == "random":
            return RandomSampler(self.index)

        elif sampler_name == "kcenter":
            pool_size = params.get("candidate_pool_size", 50_000)
            return KCenterSampler(self.index, candidate_pool_size=pool_size)

        elif sampler_name == "margin":
            search_k = params.get("local_search_k", 50_000)
            return MarginSampler(self.index, local_search_k=search_k)

        elif sampler_name == "representative":
            search_k = params.get("local_search_k", 50_000)
            return RepresentativeSampler(self.index, local_search_k=search_k)

        elif sampler_name == "informative_cluster_diverse":
            pool_size = params.get("candidate_pool_size", 50_000)
            n_clusters = params.get("n_clusters", 50)
            return InformativeClusterDiverseSampler(
                self.index, n_clusters=n_clusters, candidate_pool_size=pool_size
            )

        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

    def _get_sampler_info(self) -> List[Dict[str, Any]]:
        """Get information about available samplers."""
        return [
            {
                "name": "random",
                "description": "Random baseline sampling",
                "requires_labels": False,
                "min_labeled": 0,
                "params": [],
            },
            {
                "name": "kcenter",
                "description": "Diversity-based greedy farthest-first sampling",
                "requires_labels": False,
                "min_labeled": 0,
                "params": [
                    {
                        "name": "candidate_pool_size",
                        "type": "int",
                        "default": 50000,
                        "description": "Size of candidate pool to sample from",
                    }
                ],
            },
            {
                "name": "margin",
                "description": "Uncertainty sampling near decision boundary",
                "requires_labels": True,
                "min_labeled": 2,
                "params": [
                    {
                        "name": "local_search_k",
                        "type": "int",
                        "default": 50000,
                        "description": "Size of local neighborhood to search",
                    }
                ],
            },
            {
                "name": "representative",
                "description": "Diverse samples from uncertain margin region",
                "requires_labels": True,
                "min_labeled": 1,
                "params": [
                    {
                        "name": "local_search_k",
                        "type": "int",
                        "default": 50000,
                        "description": "Size of local neighborhood to search",
                    }
                ],
            },
            {
                "name": "informative_cluster_diverse",
                "description": "Cluster-based diversity with uncertainty weighting",
                "requires_labels": True,
                "min_labeled": 2,
                "params": [
                    {
                        "name": "candidate_pool_size",
                        "type": "int",
                        "default": 50000,
                        "description": "Size of candidate pool to sample from",
                    },
                    {
                        "name": "n_clusters",
                        "type": "int",
                        "default": 50,
                        "description": "Number of clusters for diversity",
                    },
                ],
            },
        ]

    def _get_label_distribution(self) -> Dict[int, int]:
        """Get distribution of labels."""
        distribution = {}
        for label in self.labels.values():
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

    def save_state(self, filepath: str):
        """Save session state to file."""
        state = {
            "image_paths": self.image_paths,
            "labeled_indices": list(self.labeled_indices),
            "labels": {str(k): v for k, v in self.labels.items()},
            "index_config": self.index_config.index_string,
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("Session state saved to %s", filepath)

    @classmethod
    def load_from_state(
        cls, filepath: str, feature_vectors: np.ndarray
    ) -> "ActiveLearningSession":
        """Load session from saved state."""
        with open(filepath, "r") as f:
            state = json.load(f)

        session = cls(
            image_paths=state["image_paths"],
            feature_vectors=feature_vectors,
            index_config=IndexConfig(state["index_config"]),
        )

        # Restore labeled data
        for idx_str, label in state["labels"].items():
            idx = int(idx_str)
            session.labeled_indices.add(idx)
            session.labels[idx] = label

        logger.info(
            "Session loaded: %d labeled samples", len(session.labeled_indices)
        )
        return session

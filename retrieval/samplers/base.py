"""Base classes and mixins for active learning samplers."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import faiss
import numpy as np
import psutil
from sklearn.linear_model import LogisticRegression

from index_manager import ReconstructionHandler

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks performance metrics for sampling operations."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            "wall_time": [],
            "cpu_time": [],
            "memory_peak": [],
            "memory_current": [],
            "reconstructions": [],
            "search_calls": [],
            "distances_computed": [],
        }

    def track_operation(self, func):
        """Decorator to track performance metrics for an operation."""

        def wrapper(*args, **kwargs):
            # Initialize counters for this call
            for key in self.metrics:
                if key not in [
                    "wall_time",
                    "cpu_time",
                    "memory_peak",
                    "memory_current",
                ]:
                    self.metrics[key].append(0)

            # Memory and time before
            mem_before = psutil.virtual_memory().used / 1024**3  # GB
            cpu_time_before = time.process_time()
            wall_time_before = time.perf_counter()

            result = func(*args, **kwargs)

            # Memory and time after
            mem_after = psutil.virtual_memory().used / 1024**3
            cpu_time_after = time.process_time()
            wall_time_after = time.perf_counter()

            self.metrics["wall_time"].append(wall_time_after - wall_time_before)
            self.metrics["cpu_time"].append(cpu_time_after - cpu_time_before)
            self.metrics["memory_current"].append(mem_after)
            self.metrics["memory_peak"].append(max(mem_before, mem_after))

            return result

        return wrapper

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for all metrics."""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_total"] = np.sum(values)
                summary[f"{key}_std"] = np.std(values)
        return summary


class BaseSampler(ABC):
    """Abstract base class for all active learning samplers."""

    def __init__(self, name: str, index: faiss.Index):
        self.name = name
        self.index = index
        self.reconstruction_handler = ReconstructionHandler(index)
        self.metrics_tracker = MetricsTracker()

    @abstractmethod
    def retrieve(self, annotated_ids: List[int], *args, **kwargs) -> np.ndarray:
        """
        Retrieve next batch of samples to annotate.

        Args:
            annotated_ids: List of already annotated sample IDs
            *args, **kwargs: Sampler-specific arguments

        Returns:
            Array of selected sample IDs
        """
        pass

    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this sampler."""
        return self.metrics_tracker.get_summary()


class CandidatePoolMixin:
    """Mixin for samplers that use candidate pool sampling."""

    def __init__(self, candidate_pool_size: int):
        self.candidate_pool_size = candidate_pool_size
        self.pool_size_history: list[int] = []

    def sample_candidate_pool(
        self, annotated_ids: List[int], index_ntotal: int
    ) -> np.ndarray:
        """
        Sample a candidate pool from unlabeled data.

        Args:
            annotated_ids: List of already annotated IDs
            index_ntotal: Total number of samples in index

        Returns:
            Array of candidate IDs
        """
        full_indices = np.arange(index_ntotal)
        unlabeled_indices = np.setdiff1d(
            full_indices, annotated_ids, assume_unique=True
        )
        actual_pool_size = min(self.candidate_pool_size, len(unlabeled_indices))
        self.pool_size_history.append(actual_pool_size)

        if actual_pool_size == 0:
            return np.array([])

        return np.random.choice(unlabeled_indices, size=actual_pool_size, replace=False)


class LocalNeighborhoodMixin:
    """Mixin for samplers that use local neighborhood search."""

    def __init__(self, local_search_k: int):
        self.local_search_k = local_search_k
        self.neighborhood_sizes: list[int] = []

    def get_local_neighborhood(
        self,
        index: faiss.Index,
        annotated_ids: np.ndarray,
        annotated_vectors: np.ndarray,
        reconstruction_handler: ReconstructionHandler,
    ) -> tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        """
        Get local neighborhood around annotated samples.

        Args:
            index: Faiss index
            annotated_ids: Array of annotated sample IDs
            annotated_vectors: Vectors for annotated samples
            reconstruction_handler: Handler for safe reconstruction

        Returns:
            Tuple of (local_ids, local_vectors, id_to_local_idx_mapping)
        """
        # Search from center of annotated samples
        search_center = np.mean(annotated_vectors, axis=0, keepdims=True)
        _, neighbor_ids_matrix = index.search(search_center, self.local_search_k)
        local_neighbor_ids = neighbor_ids_matrix.flatten()

        # Filter out invalid IDs
        valid_neighbor_ids = local_neighbor_ids[
            (local_neighbor_ids >= 0) & (local_neighbor_ids < index.ntotal)
        ]

        # Combine with annotated IDs
        full_local_ids = np.unique(np.concatenate([annotated_ids, valid_neighbor_ids]))
        self.neighborhood_sizes.append(len(full_local_ids))

        # Validate and reconstruct
        valid_ids_mask = (full_local_ids >= 0) & (full_local_ids < index.ntotal)
        full_local_ids = full_local_ids[valid_ids_mask]

        if len(full_local_ids) == 0:
            logger.warning("No valid neighbor IDs found")
            return np.array([]), np.array([]), {}

        # Reconstruct with error handling
        local_vectors, valid_ids = reconstruction_handler.safe_reconstruct(
            full_local_ids
        )

        if len(local_vectors) == 0:
            logger.warning("Reconstruction failed for all neighbors")
            return np.array([]), np.array([]), {}

        # Update IDs to only include successfully reconstructed ones
        full_local_ids = valid_ids

        # Create ID to index mapping
        id_to_local_idx = {int(gid): i for i, gid in enumerate(full_local_ids)}

        return full_local_ids, local_vectors, id_to_local_idx

    def train_local_model(
        self,
        local_vectors: np.ndarray,
        annotated_ids: np.ndarray,
        annotated_labels: np.ndarray,
        id_to_local_idx: Dict[int, int],
    ) -> Optional[LogisticRegression]:
        """
        Train a logistic regression model on local neighborhood.

        Args:
            local_vectors: Vectors in local neighborhood
            annotated_ids: Global IDs of annotated samples
            annotated_labels: Labels for annotated samples
            id_to_local_idx: Mapping from global ID to local index

        Returns:
            Trained model or None if insufficient data
        """
        # Find annotated samples in local neighborhood
        local_annotated_indices = []
        valid_annotated_labels = []

        for i, gid in enumerate(annotated_ids):
            if gid in id_to_local_idx:
                local_annotated_indices.append(id_to_local_idx[gid])
                valid_annotated_labels.append(annotated_labels[i])

        if len(local_annotated_indices) < 2:
            logger.warning("Not enough annotated samples in local neighborhood")
            return None

        # Train model
        model = LogisticRegression(
            random_state=0, max_iter=1000, class_weight="balanced", n_jobs=-1
        )
        model.fit(local_vectors[local_annotated_indices], valid_annotated_labels)

        return model

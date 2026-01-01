"""Faiss index management with robust reconstruction and configuration."""

import logging
from typing import Optional

import faiss
import numpy as np

from config import IndexConfig

logger = logging.getLogger(__name__)


class ReconstructionError(Exception):
    """Raised when vector reconstruction fails."""

    pass


class IndexBuilder:
    """Builds and configures Faiss indices with proper DirectMap and reconstruction support."""

    def __init__(self, config: IndexConfig):
        self.config = config
        self.index: Optional[faiss.Index] = None
        self.vector_storage: Optional[np.ndarray] = None  # For PQ indices

    def build(self, data: np.ndarray) -> faiss.Index:
        """
        Build and configure Faiss index with training and data addition.

        Args:
            data: Feature vectors to index (N, dim)

        Returns:
            Configured Faiss index
        """
        dim = data.shape[1]
        logger.info(
            "Building %s index with %d vectors", self.config.index_string, len(data)
        )

        # Create index
        self.index = faiss.index_factory(dim, self.config.index_string, faiss.METRIC_L2)

        # GPU setup
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            try:
                gpu_res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)
                logger.info("Using GPU acceleration")
            except Exception as e:
                logger.warning("GPU setup failed (%s), using CPU", e)
                self.config.use_gpu = False

        # Enable DirectMap for IVF indices (CPU only)
        if (
            self.config.is_ivf
            and not self.config.use_gpu
            and self.config.enable_direct_map
        ):
            self._enable_direct_map()

        # Handle PQ indices reconstruction
        if self.config.is_pq and not self.config.use_gpu:
            self._setup_pq_reconstruction(data)

        # Train index if needed
        self._train_index(data)

        # Add data in batches
        self._add_data(data)

        logger.info("Index setup complete: %d vectors indexed", self.index.ntotal)
        return self.index

    def _enable_direct_map(self):
        """Enable DirectMap for IVF indices to support reconstruction."""
        logger.info("Enabling DirectMap for IVF index reconstruction")
        try:
            if hasattr(self.index, "set_direct_map_type"):
                # Array type uses sequential IDs (more memory efficient)
                self.index.set_direct_map_type(faiss.DirectMap.Array)
            elif hasattr(self.index, "make_direct_map"):
                self.index.make_direct_map()
            else:
                logger.warning("DirectMap not available for this index type")
        except Exception as e:
            logger.warning("DirectMap setup failed: %s", e)

    def _setup_pq_reconstruction(self, data: np.ndarray):
        """Setup custom reconstruction for PQ indices by storing original vectors."""
        logger.info("Setting up PQ reconstruction with vector storage")
        self.vector_storage = data.copy()
        # Monkey-patch reconstruct_batch
        if self.index is not None:
            self.index.reconstruct_batch = lambda ids: self.vector_storage[ids]

    def _train_index(self, data: np.ndarray):
        """Train index with appropriate sample size."""
        if self.index is None or (
            not hasattr(self.index, "is_trained") or self.index.is_trained
        ):
            return

        train_size = self._compute_train_size(len(data))

        if train_size > len(data):
            logger.warning(
                "Dataset size (%d) < recommended training size (%d)",
                len(data), train_size,
            )
            train_size = len(data)

        logger.info("Training index with %d samples", train_size)
        self.index.train(data[:train_size])

    def _compute_train_size(self, n_samples: int) -> int:
        """Compute appropriate training size based on index type."""
        if self.config.is_ivf and self.config.n_centroids:
            # For IVF: 40 samples per centroid minimum
            min_train_size = max(
                self.config.n_centroids * self.config.training_size_multiplier, 10_000
            )
            return min(n_samples, min_train_size)
        else:
            return min(n_samples, 100_000)

    def _add_data(self, data: np.ndarray):
        """Add data to index in batches."""
        if self.index is None:
            return
        batch_size = self.config.add_batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            self.index.add(batch)


class ReconstructionHandler:
    """Handles robust vector reconstruction from Faiss indices with error recovery."""

    def __init__(self, index: faiss.Index):
        self.index = index

    def reconstruct_batch(self, ids: np.ndarray) -> np.ndarray:
        """
        Reconstruct vectors by IDs with robust error handling.

        Args:
            ids: Array of vector IDs to reconstruct

        Returns:
            Reconstructed vectors (n_ids, dim)

        Raises:
            ReconstructionError: If reconstruction fails completely
        """
        # Validate IDs
        ids = self._validate_ids(ids)

        if len(ids) == 0:
            raise ReconstructionError("No valid IDs to reconstruct")

        # Try batch reconstruction first
        try:
            return self.index.reconstruct_batch(ids)
        except Exception as e:
            logger.warning("Batch reconstruction failed: %s", e)
            # Fallback to one-by-one reconstruction
            return self._reconstruct_one_by_one(ids)

    def _validate_ids(self, ids: np.ndarray) -> np.ndarray:
        """Filter out invalid IDs (negative or >= ntotal)."""
        if not isinstance(ids, np.ndarray):
            ids = np.array(ids)

        valid_mask = (ids >= 0) & (ids < self.index.ntotal)
        invalid_count = (~valid_mask).sum()

        if invalid_count > 0:
            logger.warning("Filtered out %d invalid IDs", invalid_count)

        return ids[valid_mask]

    def _reconstruct_one_by_one(self, ids: np.ndarray) -> np.ndarray:
        """
        Fallback: reconstruct vectors one by one, skipping problematic IDs.

        Args:
            ids: Array of vector IDs

        Returns:
            Reconstructed vectors for successful IDs
        """
        vectors = []
        valid_ids = []

        for idx in ids:
            try:
                vec = self.index.reconstruct(int(idx))
                vectors.append(vec)
                valid_ids.append(idx)
            except Exception:
                continue

        if len(vectors) == 0:
            raise ReconstructionError(
                f"Could not reconstruct any of {len(ids)} requested vectors"
            )

        failed_count = len(ids) - len(vectors)
        if failed_count > 0:
            logger.warning(
                "Failed to reconstruct %d/%d vectors", failed_count, len(ids)
            )

        return np.array(vectors)

    def safe_reconstruct(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct vectors and return both vectors and valid IDs.

        Args:
            ids: Array of vector IDs to reconstruct

        Returns:
            Tuple of (vectors, valid_ids) - only successfully reconstructed vectors
        """
        # Validate IDs first to know which ones are valid
        valid_ids = self._validate_ids(ids)

        if len(valid_ids) == 0:
            return np.array([]), np.array([])

        try:
            # Try batch reconstruction
            vectors = self.index.reconstruct_batch(valid_ids)
            return vectors, valid_ids
        except Exception:
            # Fallback to one-by-one reconstruction
            vectors = []
            successful_ids = []

            for idx in valid_ids:
                try:
                    vec = self.index.reconstruct(int(idx))
                    vectors.append(vec)
                    successful_ids.append(idx)
                except Exception:
                    continue

            if len(vectors) == 0:
                return np.array([]), np.array([])

            return np.array(vectors), np.array(successful_ids)

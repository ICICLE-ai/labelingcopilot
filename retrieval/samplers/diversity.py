"""Diversity-based active learning samplers: K-Center and InformativeClusterDiverse."""

import numpy as np
import faiss
from typing import List, Optional
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression

from samplers.base import BaseSampler, CandidatePoolMixin


class KCenterSampler(BaseSampler, CandidatePoolMixin):
    """K-Center Greedy sampling for maximum diversity."""

    def __init__(self, index: faiss.Index, candidate_pool_size: int = 50_000):
        BaseSampler.__init__(self, "K-Center", index)
        CandidatePoolMixin.__init__(self, min(candidate_pool_size, index.ntotal))

    def retrieve(self, annotated_ids: List[int], num_samples: int) -> np.ndarray:
        """
        Select samples that maximize distance to already annotated samples.

        Args:
            annotated_ids: List of already annotated sample IDs
            num_samples: Number of new samples to select

        Returns:
            Array of selected sample IDs
        """

        @self.metrics_tracker.track_operation
        def _retrieve():
            # Sample candidate pool
            candidate_ids = self.sample_candidate_pool(annotated_ids, self.index.ntotal)

            if len(candidate_ids) == 0:
                return np.array([])

            # Reconstruct candidate vectors
            candidate_vectors = self.reconstruction_handler.reconstruct_batch(
                candidate_ids
            )
            self.metrics_tracker.metrics["reconstructions"][-1] = len(candidate_ids)

            # Initialize distances
            if annotated_ids:
                annotated_vectors = self.reconstruction_handler.reconstruct_batch(
                    np.array(annotated_ids)
                )
                dist_matrix = faiss.pairwise_distances(
                    candidate_vectors, annotated_vectors
                )
                min_distances = dist_matrix.min(axis=1)
                self.metrics_tracker.metrics["distances_computed"][-1] = len(
                    candidate_ids
                ) * len(annotated_ids)
            else:
                min_distances = np.full(len(candidate_ids), np.inf, dtype="float32")
                self.metrics_tracker.metrics["distances_computed"][-1] = 0

            # Greedy K-Center selection
            newly_selected_ids = []
            for _ in range(num_samples):
                if np.all(np.isinf(min_distances)):
                    select_idx = np.random.randint(len(candidate_ids))
                else:
                    select_idx = np.argmax(min_distances)

                new_center_id = candidate_ids[select_idx]
                newly_selected_ids.append(new_center_id)
                new_center_vector = candidate_vectors[select_idx].reshape(1, -1)

                # Update distances
                dist_to_new_center = faiss.pairwise_distances(
                    candidate_vectors, new_center_vector
                ).flatten()
                min_distances = np.minimum(min_distances, dist_to_new_center)
                self.metrics_tracker.metrics["distances_computed"][-1] += len(
                    candidate_ids
                )

            return np.array(newly_selected_ids)

        return _retrieve()


class InformativeClusterDiverseSampler(BaseSampler, CandidatePoolMixin):
    """Combines uncertainty sampling with cluster-based diversity."""

    def __init__(
        self,
        index: faiss.Index,
        n_clusters: int = 50,
        candidate_pool_size: int = 50_000,
    ):
        BaseSampler.__init__(self, "InformativeClusterDiverse", index)
        CandidatePoolMixin.__init__(self, min(candidate_pool_size, index.ntotal))
        self.n_clusters = min(n_clusters, index.ntotal // 10)
        self.cluster_model: Optional[MiniBatchKMeans] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_prob: Optional[np.ndarray] = None
        self.cluster_history: list[int] = []

    def retrieve(
        self, annotated_ids: List[int], annotated_labels: List[int], num_samples: int
    ) -> np.ndarray:
        """
        Select diverse and uncertain samples using clustering.

        Args:
            annotated_ids: List of already annotated sample IDs
            annotated_labels: Labels for annotated samples
            num_samples: Number of new samples to select

        Returns:
            Array of selected sample IDs
        """

        @self.metrics_tracker.track_operation
        def _retrieve():
            if len(annotated_ids) < 2:
                raise ValueError(
                    "InformativeClusterDiverse sampling requires at least 2 annotated samples"
                )

            # Sample candidate pool
            candidate_ids = self.sample_candidate_pool(annotated_ids, self.index.ntotal)

            if len(candidate_ids) == 0:
                return np.array([])

            # Cluster candidates
            candidate_vectors = self._cluster_data(candidate_ids)

            # Train uncertainty model
            uncertainty_scores = self._compute_uncertainty(
                candidate_vectors, annotated_ids, annotated_labels
            )

            # Select diverse batch based on uncertainty and cluster distribution
            selected_ids = self._select_diverse_batch(
                candidate_vectors,
                candidate_ids,
                uncertainty_scores,
                num_samples,
                set(annotated_ids),
            )

            return selected_ids

        return _retrieve()

    def _cluster_data(self, candidate_ids: np.ndarray) -> np.ndarray:
        """Cluster candidate data and compute cluster probabilities."""
        if len(candidate_ids) < self.n_clusters:
            # Use all candidates as single points
            self.cluster_labels = np.arange(len(candidate_ids))
            self.cluster_prob = np.ones(len(candidate_ids)) / len(candidate_ids)
            self.n_clusters = len(candidate_ids)
            candidate_vectors = self.reconstruction_handler.reconstruct_batch(
                candidate_ids
            )
            self.metrics_tracker.metrics["reconstructions"][-1] += len(candidate_ids)
            return candidate_vectors

        # Reconstruct vectors
        candidate_vectors = self.reconstruction_handler.reconstruct_batch(candidate_ids)
        self.metrics_tracker.metrics["reconstructions"][-1] += len(candidate_ids)

        # Perform clustering
        actual_clusters = min(self.n_clusters, len(candidate_ids))
        cluster_model = MiniBatchKMeans(
            n_clusters=actual_clusters, random_state=0, n_init="auto"
        )
        self.cluster_labels = cluster_model.fit_predict(candidate_vectors)
        self.cluster_model = cluster_model

        # Compute cluster probabilities
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        self.cluster_prob = counts / counts.sum()
        self.cluster_history.append(len(unique))

        return candidate_vectors

    def _compute_uncertainty(
        self,
        candidate_vectors: np.ndarray,
        annotated_ids: List[int],
        annotated_labels: List[int],
    ) -> np.ndarray:
        """Compute uncertainty scores for candidates."""
        # Get annotated vectors
        annotated_ids_array = np.array(annotated_ids)
        annotated_labels_array = np.array(annotated_labels)
        annotated_vectors = self.reconstruction_handler.reconstruct_batch(
            annotated_ids_array
        )
        self.metrics_tracker.metrics["reconstructions"][-1] += len(annotated_ids_array)

        # Train model
        model = LogisticRegression(
            random_state=0, max_iter=1000, class_weight="balanced", n_jobs=-1
        )
        model.fit(annotated_vectors, annotated_labels_array)

        # Compute uncertainty (distance to decision boundary)
        if hasattr(model, "decision_function"):
            decision_values = model.decision_function(candidate_vectors)
            uncertainty_scores = np.abs(decision_values)  # Closer to 0 = more uncertain
        else:
            # Fallback: use prediction probabilities
            probabilities = model.predict_proba(candidate_vectors)
            if probabilities.shape[1] > 1:
                sorted_probs = np.sort(probabilities, axis=1)
                uncertainty_scores = sorted_probs[:, -1] - sorted_probs[:, -2]
            else:
                uncertainty_scores = np.abs(probabilities[:, 0] - 0.5)

        return uncertainty_scores

    def _select_diverse_batch(
        self,
        candidate_vectors: np.ndarray,
        candidate_ids: np.ndarray,
        uncertainty_scores: np.ndarray,
        num_samples: int,
        already_selected_set: set,
    ) -> np.ndarray:
        """Select diverse batch based on cluster distribution and uncertainty."""
        # Sort by uncertainty (lower is more uncertain/better)
        uncertainty_rank = np.argsort(uncertainty_scores)

        # Filter out already selected
        available_indices = [
            i for i, cid in enumerate(candidate_ids) if cid not in already_selected_set
        ]
        uncertainty_rank = [i for i in uncertainty_rank if i in available_indices]

        if len(uncertainty_rank) == 0:
            return np.array([])

        # Initialize batch selection
        new_batch_indices: list[int] = []
        cluster_counts = np.zeros(self.n_clusters)

        if self.cluster_labels is None or self.cluster_prob is None:
            return np.array([])

        # First pass: select based on cluster diversity constraint
        for rank_idx in uncertainty_rank:
            if len(new_batch_indices) >= num_samples:
                break

            cluster_id = self.cluster_labels[rank_idx]
            expected_count = self.cluster_prob[cluster_id] * num_samples

            # Add if this cluster is under-represented
            if cluster_counts[cluster_id] < expected_count:
                new_batch_indices.append(rank_idx)
                cluster_counts[cluster_id] += 1

        # Second pass: fill remaining slots with most uncertain
        remaining_slots = num_samples - len(new_batch_indices)
        if remaining_slots > 0:
            remaining_candidates = [
                i
                for i in uncertainty_rank
                if i not in new_batch_indices and i in available_indices
            ]
            new_batch_indices.extend(remaining_candidates[:remaining_slots])

        # Convert back to global IDs
        selected_global_ids = [
            candidate_ids[i] for i in new_batch_indices[:num_samples]
        ]
        return np.array(selected_global_ids)

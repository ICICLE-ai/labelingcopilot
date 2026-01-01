"""Uncertainty-based active learning samplers: Margin and Representative."""

import logging
from typing import List

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from samplers.base import BaseSampler, LocalNeighborhoodMixin

logger = logging.getLogger(__name__)


class MarginSampler(BaseSampler, LocalNeighborhoodMixin):
    """Margin sampling: selects samples closest to decision boundary."""

    def __init__(self, index: faiss.Index, local_search_k: int = 50_000):
        BaseSampler.__init__(self, "Margin", index)
        LocalNeighborhoodMixin.__init__(self, min(local_search_k, index.ntotal))

    def retrieve(
        self, annotated_ids: List[int], annotated_labels: List[int], num_samples: int
    ) -> np.ndarray:
        """
        Select samples with smallest prediction margin.

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
                    "Margin sampling requires at least 2 annotated samples"
                )

            annotated_ids_array = np.array(annotated_ids)
            annotated_labels_array = np.array(annotated_labels)

            # Get annotated vectors
            annotated_vectors = self.reconstruction_handler.reconstruct_batch(
                annotated_ids_array
            )

            # Get local neighborhood
            full_local_ids, X_local, id_to_local_idx = self.get_local_neighborhood(
                self.index,
                annotated_ids_array,
                annotated_vectors,
                self.reconstruction_handler,
            )
            self.metrics_tracker.metrics["search_calls"][-1] = 1
            self.metrics_tracker.metrics["reconstructions"][-1] = len(full_local_ids)

            if len(full_local_ids) == 0:
                logger.warning("No valid neighbor IDs found")
                return np.array([])

            # Train local model
            model = self.train_local_model(
                X_local, annotated_ids_array, annotated_labels_array, id_to_local_idx
            )

            if model is None:
                return np.array([])

            # Find unlabeled samples in local neighborhood
            is_annotated_mask = np.isin(full_local_ids, annotated_ids_array)
            unlabeled_local_indices = np.where(~is_annotated_mask)[0]

            if len(unlabeled_local_indices) == 0:
                return np.array([])

            # Compute margin for unlabeled samples
            unlabeled_vectors = X_local[unlabeled_local_indices]
            probas = model.predict_proba(unlabeled_vectors)
            margin = np.abs(probas[:, 1] - probas[:, 0])

            # Select samples with smallest margin (most uncertain)
            ranked_indices = np.argsort(margin)
            top_n_local_indices = unlabeled_local_indices[ranked_indices[:num_samples]]

            return full_local_ids[top_n_local_indices]

        return _retrieve()


class RepresentativeSampler(BaseSampler, LocalNeighborhoodMixin):
    """Representative sampling: selects diverse samples near decision boundary."""

    def __init__(self, index: faiss.Index, local_search_k: int = 50_000):
        BaseSampler.__init__(self, "Representative", index)
        LocalNeighborhoodMixin.__init__(self, min(local_search_k, index.ntotal))

    def retrieve(
        self, annotated_ids: List[int], annotated_labels: List[int], num_samples: int
    ) -> np.ndarray:
        """
        Select representative samples from margin region using clustering.

        Args:
            annotated_ids: List of already annotated sample IDs
            annotated_labels: Labels for annotated samples
            num_samples: Number of new samples to select

        Returns:
            Array of selected sample IDs
        """

        @self.metrics_tracker.track_operation
        def _retrieve():
            if not annotated_ids:
                raise ValueError(
                    "Representative sampling requires at least one annotated sample"
                )

            annotated_ids_array = np.array(annotated_ids)
            annotated_labels_array = np.array(annotated_labels)

            # Get annotated vectors
            annotated_vectors = self.reconstruction_handler.reconstruct_batch(
                annotated_ids_array
            )

            # Get local neighborhood
            full_local_ids, X_local, id_to_local_idx = self.get_local_neighborhood(
                self.index,
                annotated_ids_array,
                annotated_vectors,
                self.reconstruction_handler,
            )
            self.metrics_tracker.metrics["search_calls"][-1] = 1
            self.metrics_tracker.metrics["reconstructions"][-1] = len(full_local_ids)

            if len(full_local_ids) == 0:
                logger.warning("No valid neighbor IDs found")
                return np.array([])

            # Train local model
            model = self.train_local_model(
                X_local, annotated_ids_array, annotated_labels_array, id_to_local_idx
            )

            if model is None:
                return np.array([])

            # Find unlabeled samples
            is_annotated_mask = np.isin(full_local_ids, annotated_ids_array)
            unlabeled_local_indices = np.where(~is_annotated_mask)[0]

            if len(unlabeled_local_indices) == 0:
                return np.array([])

            # Calculate decision boundary distance for all local points
            local_decision_values = np.abs(model.decision_function(X_local))

            # Get local annotated indices
            local_annotated_indices = []
            for gid in annotated_ids_array:
                if gid in id_to_local_idx:
                    local_annotated_indices.append(id_to_local_idx[gid])

            if len(local_annotated_indices) < 2:
                logger.warning("Not enough annotated samples in local neighborhood")
                return np.array([])

            # Define margin threshold based on closest labeled point to boundary
            margin_threshold = np.min(local_decision_values[local_annotated_indices])

            # Find unlabeled points within this margin
            unlabeled_decision_values = local_decision_values[unlabeled_local_indices]
            in_margin_mask = unlabeled_decision_values < margin_threshold

            margin_candidate_local_indices = unlabeled_local_indices[in_margin_mask]

            # Handle insufficient margin candidates
            if len(margin_candidate_local_indices) < num_samples:
                logger.info(
                    "Not enough points within margin; falling back to simple uncertainty sampling"
                )
                sorted_unlabeled_indices = unlabeled_local_indices[
                    np.argsort(unlabeled_decision_values)
                ]
                top_n_local_indices = sorted_unlabeled_indices[:num_samples]
                return full_local_ids[top_n_local_indices]

            # Cluster the uncertain candidates to ensure diversity
            vectors_in_margin = X_local[margin_candidate_local_indices]

            clustering_model = MiniBatchKMeans(
                n_clusters=num_samples, random_state=0, n_init="auto"
            )
            dist_to_centroids = clustering_model.fit_transform(vectors_in_margin)

            # Select medoids (points closest to each cluster centroid)
            medoid_indices_in_margin = np.argmin(dist_to_centroids, axis=0)

            # Get original local indices
            selected_local_indices = margin_candidate_local_indices[
                medoid_indices_in_margin
            ]

            # Get final global IDs
            new_batch_global_ids = full_local_ids[selected_local_indices]

            # Ensure we return correct number of unique samples
            unique_ids = np.unique(new_batch_global_ids)
            if len(unique_ids) < num_samples:
                # Fill with most uncertain if clustering yields duplicates
                remaining_needed = num_samples - len(unique_ids)
                all_uncertain_sorted_ids = full_local_ids[
                    unlabeled_local_indices[np.argsort(unlabeled_decision_values)]
                ]
                filler_ids = [
                    uid for uid in all_uncertain_sorted_ids if uid not in unique_ids
                ][:remaining_needed]
                new_batch_global_ids = np.concatenate([unique_ids, filler_ids])

            return new_batch_global_ids[:num_samples]

        return _retrieve()

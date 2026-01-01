"""Random baseline sampler for active learning."""

import numpy as np
from typing import List

from samplers.base import BaseSampler


class RandomSampler(BaseSampler):
    """Random sampling baseline."""

    def __init__(self, index):
        super().__init__("Random", index)

    def retrieve(self, annotated_ids: List[int], num_samples: int) -> np.ndarray:
        """
        Randomly select samples from unlabeled pool.

        Args:
            annotated_ids: List of already annotated sample IDs
            num_samples: Number of new samples to select

        Returns:
            Array of randomly selected sample IDs
        """

        @self.metrics_tracker.track_operation
        def _retrieve():
            full_id_set = np.arange(self.index.ntotal)
            unlabeled_ids = np.setdiff1d(full_id_set, annotated_ids, assume_unique=True)

            if len(unlabeled_ids) < num_samples:
                return unlabeled_ids

            return np.random.choice(unlabeled_ids, num_samples, replace=False)

        return _retrieve()

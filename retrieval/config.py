"""Configuration dataclasses for the retrieval service."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IndexConfig:
    """FAISS index setup parameters."""

    index_string: str
    use_gpu: bool = False
    enable_direct_map: bool = True
    training_size_multiplier: int = 40  # IVF: training samples = n_centroids * this
    add_batch_size: int = 50_000

    @property
    def is_ivf(self) -> bool:
        return "IVF" in self.index_string

    @property
    def is_pq(self) -> bool:
        return "PQ" in self.index_string

    @property
    def n_centroids(self) -> Optional[int]:
        """Centroid count parsed from an IVF index string, or None."""
        if not self.is_ivf:
            return None
        try:
            return int(self.index_string.split("IVF")[1].split(",")[0])
        except (IndexError, ValueError):
            return None


@dataclass
class SamplerConfig:
    """Active learning sampler parameters."""

    name: str
    candidate_pool_size: Optional[int] = None
    local_search_k: Optional[int] = None
    n_clusters: Optional[int] = None

    def scale_to_dataset(self, dataset_size: int) -> "SamplerConfig":
        """Return a copy with parameters capped at the dataset size."""
        return SamplerConfig(
            name=self.name,
            candidate_pool_size=min(
                self.candidate_pool_size or dataset_size, dataset_size
            ),
            local_search_k=min(self.local_search_k or dataset_size, dataset_size),
            n_clusters=min(
                self.n_clusters or (dataset_size // 1000), dataset_size // 10
            ),
        )

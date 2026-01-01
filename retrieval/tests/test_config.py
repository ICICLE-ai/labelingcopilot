"""Tests for IndexConfig and SamplerConfig dataclasses."""

from config import IndexConfig, SamplerConfig


class TestIndexConfig:
    def test_default_values(self):
        config = IndexConfig(index_string="Flat")
        assert config.index_string == "Flat"
        assert config.use_gpu is False
        assert config.enable_direct_map is True
        assert config.training_size_multiplier == 40
        assert config.add_batch_size == 50_000

    def test_is_ivf_property(self):
        assert IndexConfig("IVF256,Flat").is_ivf is True
        assert IndexConfig("IVF1024,PQ32").is_ivf is True
        assert IndexConfig("Flat").is_ivf is False
        assert IndexConfig("HNSW32,Flat").is_ivf is False

    def test_is_pq_property(self):
        assert IndexConfig("IVF256,PQ32").is_pq is True
        assert IndexConfig("PQ64").is_pq is True
        assert IndexConfig("Flat").is_pq is False
        assert IndexConfig("IVF256,Flat").is_pq is False

    def test_n_centroids_property(self):
        assert IndexConfig("IVF256,Flat").n_centroids == 256
        assert IndexConfig("IVF1024,PQ32").n_centroids == 1024
        assert IndexConfig("IVF16384,Flat").n_centroids == 16384
        assert IndexConfig("Flat").n_centroids is None
        assert IndexConfig("HNSW32,Flat").n_centroids is None

    def test_complex_index_strings(self):
        config = IndexConfig("IVF4096,PQ16")
        assert config.is_ivf is True
        assert config.is_pq is True
        assert config.n_centroids == 4096


class TestSamplerConfig:
    def test_default_values(self):
        config = SamplerConfig(name="K-Center")
        assert config.name == "K-Center"
        assert config.candidate_pool_size is None
        assert config.local_search_k is None
        assert config.n_clusters is None

    def test_with_parameters(self):
        config = SamplerConfig(
            name="K-Center",
            candidate_pool_size=10000,
            local_search_k=5000,
            n_clusters=50,
        )
        assert config.candidate_pool_size == 10000
        assert config.local_search_k == 5000
        assert config.n_clusters == 50

    def test_scale_to_dataset(self):
        config = SamplerConfig(
            name="K-Center",
            candidate_pool_size=100_000,
            local_search_k=100_000,
            n_clusters=1000,
        )

        scaled = config.scale_to_dataset(10_000)
        assert scaled.candidate_pool_size == 10_000
        assert scaled.local_search_k == 10_000
        assert scaled.n_clusters == 1000

        scaled_large = config.scale_to_dataset(1_000_000)
        assert scaled_large.candidate_pool_size == 100_000
        assert scaled_large.local_search_k == 100_000

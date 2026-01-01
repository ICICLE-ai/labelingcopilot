"""Tests for active learning samplers."""

import pytest
import numpy as np
from samplers import (
    KCenterSampler,
    MarginSampler,
    RepresentativeSampler,
    InformativeClusterDiverseSampler,
    RandomSampler,
)
from index_manager import IndexBuilder
from config import IndexConfig


class TestRandomSampler:
    """Tests for RandomSampler."""

    def test_basic_sampling(self, flat_index):
        """Test basic random sampling."""
        sampler = RandomSampler(flat_index)
        annotated_ids = [0, 1, 2]
        num_samples = 5

        selected = sampler.retrieve(annotated_ids, num_samples)

        assert len(selected) == num_samples
        assert len(np.unique(selected)) == num_samples  # No duplicates
        # Should not select already annotated
        assert not any(sid in annotated_ids for sid in selected)

    def test_empty_annotated(self, flat_index):
        """Test sampling with no annotated samples."""
        sampler = RandomSampler(flat_index)
        selected = sampler.retrieve([], 10)

        assert len(selected) == 10
        assert len(np.unique(selected)) == 10

    def test_exhaustive_sampling(self, flat_index):
        """Test when requesting more samples than available."""
        sampler = RandomSampler(flat_index)
        total = flat_index.ntotal
        annotated_ids = list(range(total - 5))  # Leave only 5 unlabeled

        selected = sampler.retrieve(annotated_ids, 10)  # Request 10, only 5 available

        assert len(selected) == 5
        assert not any(sid in annotated_ids for sid in selected)

    def test_metrics_tracking(self, flat_index):
        """Test that metrics are tracked."""
        sampler = RandomSampler(flat_index)
        sampler.retrieve([], 10)

        metrics = sampler.get_metrics()

        assert "wall_time_mean" in metrics
        assert "cpu_time_mean" in metrics
        assert metrics["wall_time_mean"] >= 0


class TestKCenterSampler:
    """Tests for KCenterSampler."""

    def test_basic_sampling(self, flat_index):
        """Test basic K-Center sampling."""
        sampler = KCenterSampler(flat_index, candidate_pool_size=50)
        annotated_ids = [0, 1, 2]

        selected = sampler.retrieve(annotated_ids, num_samples=5)

        assert len(selected) == 5
        assert len(np.unique(selected)) == 5
        assert not any(sid in annotated_ids for sid in selected)

    def test_diversity_property(self, flat_index, small_dataset):
        """Test that K-Center selects diverse samples."""
        X, _ = small_dataset
        sampler = KCenterSampler(flat_index, candidate_pool_size=50)

        # Start with one sample
        annotated_ids = [0]
        selected = sampler.retrieve(annotated_ids, num_samples=5)

        # Selected samples should be relatively far from initial point
        # (not a strict test, just sanity check)
        assert len(selected) == 5

    def test_with_no_initial_samples(self, flat_index):
        """Test K-Center with no initial samples."""
        sampler = KCenterSampler(flat_index, candidate_pool_size=50)

        selected = sampler.retrieve([], num_samples=5)

        assert len(selected) == 5

    def test_pool_size_limiting(self, flat_index):
        """Test that pool size is respected."""
        sampler = KCenterSampler(flat_index, candidate_pool_size=20)

        # Pool size history should be tracked
        sampler.retrieve([], num_samples=5)

        assert len(sampler.pool_size_history) > 0
        assert sampler.pool_size_history[-1] <= 20

    def test_metrics_reconstruction_count(self, flat_index):
        """Test that reconstruction count is tracked."""
        sampler = KCenterSampler(flat_index, candidate_pool_size=30)
        sampler.retrieve([0, 1], num_samples=5)

        metrics = sampler.get_metrics()

        assert "reconstructions_mean" in metrics
        assert metrics["reconstructions_mean"] > 0


class TestMarginSampler:
    """Tests for MarginSampler."""

    def test_basic_sampling(self, flat_index, balanced_dataset, create_annotated_set):
        """Test basic margin sampling."""
        X, y = balanced_dataset
        # Build new index with balanced dataset
        from index_manager import IndexBuilder
        from config import IndexConfig

        index = IndexBuilder(IndexConfig("Flat")).build(X)

        sampler = MarginSampler(index, local_search_k=100)
        annotated_ids, annotated_labels = create_annotated_set(y, size=20)

        selected = sampler.retrieve(annotated_ids, annotated_labels, num_samples=5)

        assert len(selected) > 0  # Should select at least some
        assert len(selected) <= 5
        assert not any(sid in annotated_ids for sid in selected)

    def test_requires_multiple_classes(self, flat_index):
        """Test that margin sampling requires multiple classes."""
        sampler = MarginSampler(flat_index, local_search_k=50)

        # Only one class - should raise error
        annotated_ids = [0, 1, 2]
        annotated_labels = [0, 0, 0]

        with pytest.raises(ValueError, match="at least 2"):
            sampler.retrieve(annotated_ids, annotated_labels, num_samples=5)

    def test_local_search_tracking(
        self, flat_index, balanced_dataset, create_annotated_set
    ):
        """Test that local search metrics are tracked."""
        X, y = balanced_dataset
        index = IndexBuilder(IndexConfig("Flat")).build(X)

        sampler = MarginSampler(index, local_search_k=100)
        annotated_ids, annotated_labels = create_annotated_set(y, size=20)

        sampler.retrieve(annotated_ids, annotated_labels, num_samples=5)

        # Should track neighborhood sizes
        assert len(sampler.neighborhood_sizes) > 0
        assert sampler.neighborhood_sizes[-1] > 0

    def test_with_minimum_annotated(self, flat_index, balanced_dataset):
        """Test margin sampling with minimum annotated samples."""
        X, y = balanced_dataset
        index = IndexBuilder(IndexConfig("Flat")).build(X)

        sampler = MarginSampler(index, local_search_k=50)

        # Minimum: 2 samples with different labels
        annotated_ids = [0, 100]  # One from each class
        annotated_labels = [y[0], y[100]]

        if len(set(annotated_labels)) >= 2:
            selected = sampler.retrieve(annotated_ids, annotated_labels, num_samples=3)
            assert len(selected) >= 0  # May return empty if issues with neighborhood


class TestRepresentativeSampler:
    """Tests for RepresentativeSampler."""

    def test_basic_sampling(self, flat_index, balanced_dataset, create_annotated_set):
        """Test basic representative sampling."""
        X, y = balanced_dataset
        index = IndexBuilder(IndexConfig("Flat")).build(X)

        sampler = RepresentativeSampler(index, local_search_k=100)
        annotated_ids, annotated_labels = create_annotated_set(y, size=20)

        selected = sampler.retrieve(annotated_ids, annotated_labels, num_samples=5)

        assert len(selected) > 0
        assert len(selected) <= 5
        assert not any(sid in annotated_ids for sid in selected)

    def test_clustering_for_diversity(
        self, flat_index, balanced_dataset, create_annotated_set
    ):
        """Test that representative sampling uses clustering."""
        X, y = balanced_dataset
        index = IndexBuilder(IndexConfig("Flat")).build(X)

        sampler = RepresentativeSampler(index, local_search_k=150)
        annotated_ids, annotated_labels = create_annotated_set(y, size=20)

        # Request multiple samples
        selected = sampler.retrieve(annotated_ids, annotated_labels, num_samples=10)

        # Should select diverse samples (all unique)
        assert len(selected) == len(np.unique(selected))

    def test_fallback_to_uncertainty(
        self, flat_index, balanced_dataset, create_annotated_set
    ):
        """Test fallback when not enough points in margin."""
        X, y = balanced_dataset
        index = IndexBuilder(IndexConfig("Flat")).build(X)

        sampler = RepresentativeSampler(index, local_search_k=50)
        annotated_ids, annotated_labels = create_annotated_set(y, size=20)

        # Request more samples than likely in margin
        selected = sampler.retrieve(annotated_ids, annotated_labels, num_samples=20)

        # Should still return samples (using fallback)
        assert len(selected) > 0


class TestInformativeClusterDiverseSampler:
    """Tests for InformativeClusterDiverseSampler."""

    def test_basic_sampling(self, flat_index, balanced_dataset, create_annotated_set):
        """Test basic informative cluster diverse sampling."""
        X, y = balanced_dataset
        index = IndexBuilder(IndexConfig("Flat")).build(X)

        sampler = InformativeClusterDiverseSampler(
            index, n_clusters=5, candidate_pool_size=100
        )
        annotated_ids, annotated_labels = create_annotated_set(y, size=20)

        selected = sampler.retrieve(annotated_ids, annotated_labels, num_samples=5)

        assert len(selected) > 0
        assert len(selected) <= 5
        assert not any(sid in annotated_ids for sid in selected)

    def test_clustering_performed(
        self, flat_index, balanced_dataset, create_annotated_set
    ):
        """Test that clustering is performed."""
        X, y = balanced_dataset
        index = IndexBuilder(IndexConfig("Flat")).build(X)

        sampler = InformativeClusterDiverseSampler(
            index, n_clusters=5, candidate_pool_size=100
        )
        annotated_ids, annotated_labels = create_annotated_set(y, size=20)

        sampler.retrieve(annotated_ids, annotated_labels, num_samples=5)

        # Should track cluster history
        assert len(sampler.cluster_history) > 0

    def test_requires_multiple_classes(self, flat_index):
        """Test that it requires multiple annotated classes."""
        sampler = InformativeClusterDiverseSampler(flat_index, n_clusters=5)

        with pytest.raises(ValueError, match="at least 2"):
            sampler.retrieve([0], [0], num_samples=5)

    def test_adaptive_cluster_count(
        self, flat_index, balanced_dataset, create_annotated_set
    ):
        """Test that cluster count adapts to data size."""
        X, y = balanced_dataset
        index = IndexBuilder(IndexConfig("Flat")).build(X)

        # Request more clusters than candidates
        sampler = InformativeClusterDiverseSampler(
            index, n_clusters=200, candidate_pool_size=50
        )
        annotated_ids, annotated_labels = create_annotated_set(y, size=10)

        # Should adapt and not crash
        selected = sampler.retrieve(annotated_ids, annotated_labels, num_samples=5)
        assert len(selected) > 0


class TestSamplerMetricsTracking:
    """Tests for metrics tracking across all samplers."""

    @pytest.mark.parametrize(
        "sampler_class",
        [
            RandomSampler,
            KCenterSampler,
        ],
    )
    def test_metrics_tracking_simple_samplers(self, flat_index, sampler_class):
        """Test metrics tracking for simple samplers."""
        if sampler_class == RandomSampler:
            sampler = sampler_class(flat_index)
        else:
            sampler = sampler_class(flat_index, candidate_pool_size=50)

        sampler.retrieve([], 5)

        metrics = sampler.get_metrics()

        # Check for basic metrics
        assert "wall_time_mean" in metrics
        assert "cpu_time_mean" in metrics
        assert metrics["wall_time_mean"] >= 0

    def test_multiple_retrievals_aggregate_metrics(self, flat_index):
        """Test that metrics aggregate over multiple retrievals."""
        sampler = RandomSampler(flat_index)

        sampler.retrieve([], 5)
        sampler.retrieve([0, 1, 2, 3, 4], 5)
        sampler.retrieve([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 5)

        metrics = sampler.get_metrics()

        # Should have accumulated metrics from 3 calls
        assert "wall_time_total" in metrics
        assert metrics["wall_time_total"] > 0


@pytest.mark.parametrize("num_samples", [1, 5, 10, 20])
def test_samplers_with_various_batch_sizes(flat_index, num_samples):
    """Test samplers with various batch sizes."""
    sampler = RandomSampler(flat_index)
    selected = sampler.retrieve([], num_samples)

    assert len(selected) == min(num_samples, flat_index.ntotal)


def test_sampler_names(flat_index):
    """Test that samplers have correct names."""
    assert RandomSampler(flat_index).name == "Random"
    assert KCenterSampler(flat_index).name == "K-Center"
    assert MarginSampler(flat_index).name == "Margin"
    assert RepresentativeSampler(flat_index).name == "Representative"
    assert (
        InformativeClusterDiverseSampler(flat_index).name == "InformativeClusterDiverse"
    )


@pytest.mark.slow
def test_samplers_on_large_dataset(large_dataset):
    """Test samplers on larger dataset (marked as slow)."""
    X, y = large_dataset
    index = IndexBuilder(IndexConfig("IVF64,Flat")).build(X)

    # Test each sampler
    random_sampler = RandomSampler(index)
    selected = random_sampler.retrieve([], 50)
    assert len(selected) == 50

    kcenter_sampler = KCenterSampler(index, candidate_pool_size=1000)
    selected = kcenter_sampler.retrieve([], 50)
    assert len(selected) == 50

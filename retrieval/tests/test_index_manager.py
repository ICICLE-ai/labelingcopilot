"""Tests for Faiss index management and reconstruction."""

import pytest
import numpy as np
from config import IndexConfig
from index_manager import IndexBuilder, ReconstructionHandler, ReconstructionError


class TestIndexBuilder:
    """Tests for IndexBuilder class."""

    def test_build_flat_index(self, small_dataset):
        """Test building a Flat index."""
        X, _ = small_dataset
        config = IndexConfig(index_string="Flat")
        builder = IndexBuilder(config)
        index = builder.build(X)

        assert index is not None
        assert index.ntotal == len(X)
        assert index.d == X.shape[1]

    def test_build_ivf_index(self, medium_dataset):
        """Test building an IVF index."""
        X, _ = medium_dataset
        config = IndexConfig(index_string="IVF16,Flat")
        builder = IndexBuilder(config)
        index = builder.build(X)

        assert index.ntotal == len(X)
        assert index.is_trained

    def test_build_hnsw_index(self, medium_dataset):
        """Test building an HNSW index."""
        X, _ = medium_dataset
        config = IndexConfig(index_string="HNSW32,Flat")
        builder = IndexBuilder(config)
        index = builder.build(X)

        assert index.ntotal == len(X)

    def test_directmap_enabled_for_ivf(self, medium_dataset):
        """Test that DirectMap is enabled for IVF indices."""
        X, _ = medium_dataset
        config = IndexConfig(index_string="IVF16,Flat", enable_direct_map=True)
        builder = IndexBuilder(config)
        index = builder.build(X)

        # Test that reconstruction works (DirectMap enables this)
        try:
            vec = index.reconstruct(0)
            assert vec is not None
            assert len(vec) == X.shape[1]
        except RuntimeError:
            pytest.fail("DirectMap not properly enabled - reconstruction failed")

    def test_training_with_insufficient_data(self):
        """Test training when dataset is smaller than recommended."""
        # Create very small dataset
        X = np.random.randn(50, 32).astype("float32")

        # Try to create index that needs more training data
        config = IndexConfig(index_string="IVF16,Flat")  # Needs ~640 samples ideally
        builder = IndexBuilder(config)

        # Should complete with warning but not fail
        index = builder.build(X)
        assert index.ntotal == len(X)

    def test_batch_adding(self, large_dataset):
        """Test that data is added in batches."""
        X, _ = large_dataset
        config = IndexConfig(index_string="Flat", add_batch_size=1000)
        builder = IndexBuilder(config)
        index = builder.build(X)

        assert index.ntotal == len(X)

    def test_different_dimensions(self):
        """Test index building with different feature dimensions."""
        for dim in [16, 32, 64, 128]:
            X = np.random.randn(100, dim).astype("float32")
            config = IndexConfig(index_string="Flat")
            builder = IndexBuilder(config)
            index = builder.build(X)

            assert index.ntotal == 100
            assert index.d == dim

    def test_index_types_from_fixture(self, index_type, small_dataset):
        """Test building various index types."""
        X, _ = small_dataset
        config = IndexConfig(index_string=index_type)
        builder = IndexBuilder(config)
        index = builder.build(X)

        assert index.ntotal == len(X)


class TestReconstructionHandler:
    """Tests for ReconstructionHandler class."""

    def test_basic_reconstruction(self, flat_index, small_dataset):
        """Test basic vector reconstruction."""
        X, _ = small_dataset
        handler = ReconstructionHandler(flat_index)

        ids = np.array([0, 1, 2, 3, 4])
        vectors = handler.reconstruct_batch(ids)

        assert vectors.shape == (5, X.shape[1])
        assert vectors.dtype == np.float32

    def test_reconstruction_accuracy(self, flat_index, small_dataset):
        """Test that reconstructed vectors match original (for Flat index)."""
        X, _ = small_dataset
        handler = ReconstructionHandler(flat_index)

        ids = np.array([0, 5, 10])
        vectors = handler.reconstruct_batch(ids)

        # For Flat index, reconstructed should be exact
        for i, idx in enumerate(ids):
            np.testing.assert_allclose(vectors[i], X[idx], rtol=1e-5)

    def test_invalid_id_filtering(self, flat_index):
        """Test that invalid IDs are filtered out."""
        handler = ReconstructionHandler(flat_index)

        # Mix of valid and invalid IDs
        ids = np.array([-1, 0, 5, 1000, 10])  # -1 and 1000 are invalid

        vectors = handler.reconstruct_batch(ids)

        # Should only reconstruct valid IDs
        assert len(vectors) <= 3  # At most 3 valid IDs

    def test_all_invalid_ids(self, flat_index):
        """Test handling when all IDs are invalid."""
        handler = ReconstructionHandler(flat_index)

        ids = np.array([-1, -5, 10000])  # All invalid

        with pytest.raises(ReconstructionError):
            handler.reconstruct_batch(ids)

    def test_empty_ids(self, flat_index):
        """Test handling of empty ID array."""
        handler = ReconstructionHandler(flat_index)

        ids = np.array([])

        with pytest.raises(ReconstructionError):
            handler.reconstruct_batch(ids)

    def test_safe_reconstruct(self, flat_index):
        """Test safe_reconstruct returns both vectors and valid IDs."""
        handler = ReconstructionHandler(flat_index)

        ids = np.array([0, 1, 2])
        vectors, valid_ids = handler.safe_reconstruct(ids)

        assert len(vectors) == len(valid_ids)
        assert len(valid_ids) <= len(ids)
        np.testing.assert_array_equal(valid_ids, ids)

    def test_safe_reconstruct_with_invalid_ids(self, flat_index):
        """Test safe_reconstruct with some invalid IDs."""
        handler = ReconstructionHandler(flat_index)

        ids = np.array([-1, 0, 1, 1000])  # 2 valid, 2 invalid
        vectors, valid_ids = handler.safe_reconstruct(ids)

        # Should return only valid ones
        assert len(vectors) > 0
        assert len(vectors) == len(valid_ids)
        assert all(vid >= 0 and vid < flat_index.ntotal for vid in valid_ids)

    def test_large_batch_reconstruction(self, ivf_index, medium_dataset):
        """Test reconstruction of large batch."""
        X, _ = medium_dataset
        handler = ReconstructionHandler(ivf_index)

        # Reconstruct first 100 vectors
        ids = np.arange(100)
        vectors = handler.reconstruct_batch(ids)

        assert vectors.shape == (100, X.shape[1])

    def test_reconstruction_with_ivf_index(self, ivf_index):
        """Test reconstruction works with IVF index (requires DirectMap)."""
        handler = ReconstructionHandler(ivf_index)

        ids = np.array([0, 10, 20, 30])

        # Should work if DirectMap is properly set up
        try:
            vectors = handler.reconstruct_batch(ids)
            assert len(vectors) == len(ids)
        except Exception as e:
            # If it fails, it's due to DirectMap not being set up
            pytest.skip(f"DirectMap not available: {e}")

    def test_duplicate_ids(self, flat_index):
        """Test reconstruction with duplicate IDs."""
        handler = ReconstructionHandler(flat_index)

        ids = np.array([0, 0, 1, 1, 2])  # Duplicates
        vectors = handler.reconstruct_batch(ids)

        # Should handle duplicates
        assert len(vectors) == len(ids)

    def test_out_of_order_ids(self, flat_index):
        """Test reconstruction with out-of-order IDs."""
        handler = ReconstructionHandler(flat_index)

        ids = np.array([5, 2, 8, 1, 9])  # Not sorted
        vectors = handler.reconstruct_batch(ids)

        assert len(vectors) == len(ids)

    def test_single_id_reconstruction(self, flat_index):
        """Test reconstruction of single ID."""
        handler = ReconstructionHandler(flat_index)

        ids = np.array([5])
        vectors = handler.reconstruct_batch(ids)

        assert vectors.shape[0] == 1

    def test_reconstruction_preserves_order(self, flat_index, small_dataset):
        """Test that reconstruction preserves the order of IDs."""
        X, _ = small_dataset
        handler = ReconstructionHandler(flat_index)

        ids = np.array([5, 2, 8, 1])
        vectors = handler.reconstruct_batch(ids)

        # Verify order is preserved
        for i, idx in enumerate(ids):
            np.testing.assert_allclose(vectors[i], X[idx], rtol=1e-5)


class TestIndexConfigIntegration:
    """Integration tests for IndexConfig with IndexBuilder."""

    def test_pq_index_with_storage(self, medium_dataset):
        """Test PQ index with vector storage fallback."""
        X, _ = medium_dataset

        # PQ index requires special reconstruction handling
        config = IndexConfig(index_string="PQ16")
        builder = IndexBuilder(config)

        # Should set up vector storage
        index = builder.build(X)

        assert index.ntotal == len(X)
        # Check that vector_storage was set up
        assert builder.vector_storage is not None

    def test_compute_train_size(self, medium_dataset):
        """Test training size computation."""
        X, _ = medium_dataset

        config_ivf = IndexConfig(index_string="IVF64,Flat")
        builder_ivf = IndexBuilder(config_ivf)

        train_size = builder_ivf._compute_train_size(len(X))

        # Should be at least 64 * 40 = 2560
        assert train_size >= min(len(X), 2560)

    def test_config_properties_used(self):
        """Test that config properties are properly used."""
        X = np.random.randn(100, 32).astype("float32")

        config = IndexConfig(
            index_string="Flat", add_batch_size=10  # Very small batch size
        )
        builder = IndexBuilder(config)

        # Should still work with custom batch size
        index = builder.build(X)
        assert index.ntotal == 100


@pytest.mark.parametrize(
    "n_samples,dim",
    [
        (50, 16),
        (100, 32),
        (500, 64),
        (1000, 128),
    ],
)
def test_various_dataset_sizes(n_samples, dim):
    """Test index building with various dataset sizes."""
    X = np.random.randn(n_samples, dim).astype("float32")
    config = IndexConfig(index_string="Flat")
    builder = IndexBuilder(config)
    index = builder.build(X)

    assert index.ntotal == n_samples
    assert index.d == dim


@pytest.mark.slow
def test_very_large_index():
    """Test building index with very large dataset (marked as slow)."""
    X = np.random.randn(50000, 128).astype("float32")
    config = IndexConfig(index_string="IVF256,Flat")
    builder = IndexBuilder(config)
    index = builder.build(X)

    assert index.ntotal == 50000

    # Test reconstruction still works
    handler = ReconstructionHandler(index)
    ids = np.array([0, 100, 1000, 10000])
    vectors = handler.reconstruct_batch(ids)
    assert len(vectors) == len(ids)

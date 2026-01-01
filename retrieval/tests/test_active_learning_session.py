"""Tests for ActiveLearningSession class."""

import pytest
import json

from active_learning_session import ActiveLearningSession
from config import IndexConfig


class TestSessionInitialization:
    """Test session initialization."""

    def test_init_with_features(self, synthetic_image_features):
        """Test basic initialization with features."""
        features, image_paths = synthetic_image_features

        session = ActiveLearningSession(
            image_paths=image_paths,
            feature_vectors=features,
            index_config=IndexConfig("Flat"),
        )

        assert session.n_total == len(image_paths)
        assert session.dimension == features.shape[1]
        assert len(session.labeled_indices) == 0
        assert len(session.labels) == 0
        assert session.index.ntotal == len(image_paths)

    def test_init_mismatch_lengths(self, synthetic_image_features):
        """Test error when features and paths don't match."""
        features, image_paths = synthetic_image_features

        with pytest.raises(ValueError, match="Mismatch"):
            ActiveLearningSession(
                image_paths=image_paths[:100],  # Fewer paths
                feature_vectors=features,
                index_config=IndexConfig("Flat"),
            )

    def test_auto_index_selection(self, synthetic_image_features):
        """Test automatic index type selection based on size."""
        features, image_paths = synthetic_image_features

        # Should auto-select IVF for 1000 images
        session = ActiveLearningSession(
            image_paths=image_paths,
            feature_vectors=features,
            index_config=None,  # Auto-select
        )

        assert session.index_config.index_string in ["Flat", "IVF256,Flat"]


class TestLabelManagement:
    """Test label adding and tracking."""

    def test_add_labels_new(self, mcp_session):
        """Test adding new labels."""
        labels = {
            mcp_session.image_paths[0]: 0,
            mcp_session.image_paths[1]: 1,
            mcp_session.image_paths[2]: 0,
        }

        added = mcp_session.add_labels(labels)

        assert added == 3
        assert len(mcp_session.labeled_indices) == 3
        assert len(mcp_session.labels) == 3
        assert mcp_session.labels[0] == 0
        assert mcp_session.labels[1] == 1

    def test_add_labels_update_existing(self, mcp_session_with_labels):
        """Test updating existing labels."""
        session = mcp_session_with_labels
        initial_count = len(session.labeled_indices)

        # Update label for already-labeled image
        update_labels = {session.image_paths[0]: 1}  # Was 0, change to 1

        added = session.add_labels(update_labels)

        assert added == 0  # No new labels added
        assert len(session.labeled_indices) == initial_count
        assert session.labels[0] == 1  # Updated

    def test_add_labels_invalid_path(self, mcp_session):
        """Test error when adding label for unknown path."""
        labels = {"/nonexistent/path.jpg": 0}

        with pytest.raises(ValueError, match="not found in pool"):
            mcp_session.add_labels(labels)

    def test_add_labels_incremental(self, mcp_session):
        """Test adding labels incrementally."""
        # Round 1
        labels1 = {mcp_session.image_paths[i]: i % 2 for i in range(5)}
        mcp_session.add_labels(labels1)
        assert len(mcp_session.labeled_indices) == 5

        # Round 2
        labels2 = {mcp_session.image_paths[i]: i % 2 for i in range(5, 10)}
        mcp_session.add_labels(labels2)
        assert len(mcp_session.labeled_indices) == 10


class TestSamplerValidation:
    """Test sampler requirement validation."""

    def test_random_no_labels_required(self, mcp_session):
        """Random sampler works with no labels."""
        session = mcp_session

        # Should not raise
        session._validate_sampler_requirements("random")

    def test_kcenter_no_labels_required(self, mcp_session):
        """K-Center sampler works with no labels."""
        session = mcp_session

        # Should not raise
        session._validate_sampler_requirements("kcenter")

    def test_margin_requires_labels(self, mcp_session):
        """Margin sampler requires at least 2 labels."""
        session = mcp_session

        with pytest.raises(ValueError, match="at least 2 labeled"):
            session._validate_sampler_requirements("margin")

    def test_margin_requires_two_classes(self, mcp_session):
        """Margin sampler requires two different class labels."""
        # Add labels from only one class
        labels = {mcp_session.image_paths[i]: 0 for i in range(5)}
        mcp_session.add_labels(labels)

        with pytest.raises(ValueError, match="2 different class labels"):
            mcp_session._validate_sampler_requirements("margin")

    def test_invalid_sampler_name(self, mcp_session):
        """Test error for invalid sampler name."""
        with pytest.raises(ValueError, match="Invalid sampler"):
            mcp_session._validate_sampler_requirements("nonexistent")


class TestSampling:
    """Test sample retrieval with different samplers."""

    def test_random_sampling(self, mcp_session):
        """Test random sampling."""
        paths, metadata = mcp_session.get_samples(sampler_name="random", num_samples=10)

        assert len(paths) == 10
        assert metadata["sampler_used"] == "random"
        assert metadata["samples_returned"] == 10
        assert metadata["unlabeled_remaining"] == 1000
        assert all(isinstance(p, str) for p in paths)

    def test_kcenter_sampling(self, mcp_session):
        """Test K-Center diversity sampling."""
        paths, metadata = mcp_session.get_samples(
            sampler_name="kcenter",
            num_samples=10,
            sampler_params={"candidate_pool_size": 500},
        )

        assert len(paths) == 10
        assert metadata["sampler_used"] == "kcenter"
        assert metadata["params_used"]["candidate_pool_size"] == 500

    def test_margin_sampling(self, mcp_session_with_labels):
        """Test margin uncertainty sampling."""
        session = mcp_session_with_labels

        paths, metadata = session.get_samples(
            sampler_name="margin",
            num_samples=15,
            sampler_params={"local_search_k": 500},
        )

        assert len(paths) <= 15  # May return fewer
        assert metadata["sampler_used"] == "margin"

    def test_representative_sampling(self, mcp_session_with_labels):
        """Test representative sampling."""
        session = mcp_session_with_labels

        paths, metadata = session.get_samples(
            sampler_name="representative", num_samples=10
        )

        assert len(paths) <= 10
        assert metadata["sampler_used"] == "representative"

    def test_informative_cluster_diverse_sampling(self, mcp_session_with_labels):
        """Test informative cluster diverse sampling."""
        session = mcp_session_with_labels

        paths, metadata = session.get_samples(
            sampler_name="informative_cluster_diverse",
            num_samples=20,
            sampler_params={"candidate_pool_size": 500, "n_clusters": 10},
        )

        assert len(paths) <= 20
        assert metadata["sampler_used"] == "informative_cluster_diverse"

    def test_sampling_reduces_unlabeled_count(self, mcp_session):
        """Test that labeling reduces unlabeled count."""
        # Initial sample
        paths1, meta1 = mcp_session.get_samples("random", 10)
        assert meta1["unlabeled_remaining"] == 1000

        # Add labels
        labels = {paths1[i]: i % 2 for i in range(len(paths1))}
        mcp_session.add_labels(labels)

        # Next sample
        paths2, meta2 = mcp_session.get_samples("random", 10)
        assert meta2["labeled_total"] == 10
        assert meta2["unlabeled_remaining"] == 990


class TestStatus:
    """Test status reporting."""

    def test_get_status_empty(self, mcp_session):
        """Test status with no labels."""
        status = mcp_session.get_status()

        assert status["total_images"] == 1000
        assert status["labeled_count"] == 0
        assert status["unlabeled_count"] == 1000
        assert len(status["labeled_images"]) == 0
        assert len(status["label_distribution"]) == 0
        assert len(status["available_samplers"]) == 5

    def test_get_status_with_labels(self, mcp_session_with_labels):
        """Test status with labels."""
        status = mcp_session_with_labels.get_status()

        assert status["total_images"] == 1000
        assert status["labeled_count"] == 10
        assert status["unlabeled_count"] == 990
        assert len(status["labeled_images"]) == 10
        assert sum(status["label_distribution"].values()) == 10

    def test_sampler_info_structure(self, mcp_session):
        """Test sampler info structure."""
        status = mcp_session.get_status()
        samplers = status["available_samplers"]

        for sampler in samplers:
            assert "name" in sampler
            assert "description" in sampler
            assert "requires_labels" in sampler
            assert "min_labeled" in sampler
            assert "params" in sampler

        # Check specific samplers
        random_sampler = next(s for s in samplers if s["name"] == "random")
        assert random_sampler["requires_labels"] is False
        assert random_sampler["min_labeled"] == 0

        margin_sampler = next(s for s in samplers if s["name"] == "margin")
        assert margin_sampler["requires_labels"] is True
        assert margin_sampler["min_labeled"] == 2


class TestSessionPersistence:
    """Test session save/load."""

    def test_save_state(self, mcp_session_with_labels, temp_dir):
        """Test saving session state."""
        session = mcp_session_with_labels
        save_path = temp_dir / "session.json"

        session.save_state(str(save_path))

        assert save_path.exists()

        # Load and verify
        with open(save_path) as f:
            state = json.load(f)

        assert len(state["image_paths"]) == 1000
        assert len(state["labeled_indices"]) == 10
        assert len(state["labels"]) == 10

    def test_load_from_state(
        self, mcp_session_with_labels, temp_dir, synthetic_image_features
    ):
        """Test loading session from state."""
        session = mcp_session_with_labels
        save_path = temp_dir / "session.json"

        # Save
        session.save_state(str(save_path))

        # Load
        features, _ = synthetic_image_features
        loaded_session = ActiveLearningSession.load_from_state(str(save_path), features)

        assert loaded_session.n_total == session.n_total
        assert loaded_session.labeled_indices == session.labeled_indices
        assert loaded_session.labels == session.labels


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_sampling_with_insufficient_labels(self, mcp_session):
        """Test that margin sampling fails gracefully with insufficient labels."""
        # Add only 1 label
        labels = {mcp_session.image_paths[0]: 0}
        mcp_session.add_labels(labels)

        with pytest.raises(ValueError, match="at least 2 labeled"):
            mcp_session.get_samples("margin", 10)

    def test_sampling_more_than_available(self, mcp_session):
        """Test requesting more samples than available."""
        # Label 995 images
        labels = {mcp_session.image_paths[i]: i % 2 for i in range(995)}
        mcp_session.add_labels(labels)

        # Request 10 but only 5 available
        paths, metadata = mcp_session.get_samples("random", 10)

        assert len(paths) <= 5  # Can't return more than available

    def test_invalid_sampler_params(self, mcp_session):
        """Test with invalid sampler parameters."""
        # Should handle gracefully or raise clear error
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            mcp_session.get_samples(
                "kcenter", 10, sampler_params={"invalid_param": 123}
            )


class TestFullWorkflow:
    """Integration tests for complete workflows."""

    def test_progressive_sampling_workflow(self, mcp_session):
        """Test complete active learning workflow."""
        session = mcp_session

        # Round 1: Bootstrap with random
        paths1, _ = session.get_samples("random", 10)
        labels1 = {paths1[i]: i % 2 for i in range(len(paths1))}
        session.add_labels(labels1)
        assert len(session.labeled_indices) == 10

        # Round 2: Diversity with K-Center
        paths2, _ = session.get_samples("kcenter", 20)
        labels2 = {paths2[i]: i % 2 for i in range(len(paths2))}
        session.add_labels(labels2)
        assert len(session.labeled_indices) == 30

        # Round 3: Uncertainty with Margin
        paths3, _ = session.get_samples("margin", 30)
        labels3 = {paths3[i]: i % 2 for i in range(len(paths3))}
        session.add_labels(labels3)
        assert len(session.labeled_indices) >= 60

        # Final status
        status = session.get_status()
        assert status["labeled_count"] >= 60
        assert status["unlabeled_count"] <= 940

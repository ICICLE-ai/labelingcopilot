"""Shared pytest fixtures for the retrieval test suite."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification

from config import IndexConfig
from index_manager import IndexBuilder


def _make_dataset(n_samples: int, n_features: int, seed: int = 42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_informative=n_features // 3,
        n_clusters_per_class=2,
        class_sep=1.0,
        random_state=seed,
    )
    return X.astype("float32"), y


@pytest.fixture
def small_dataset():
    np.random.seed(42)
    X = np.random.randn(100, 32).astype("float32")
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture(scope="session")
def medium_dataset():
    return _make_dataset(1000, 64)


@pytest.fixture(scope="session")
def large_dataset():
    return _make_dataset(10_000, 128)


@pytest.fixture
def balanced_dataset():
    np.random.seed(42)
    n_samples = 200
    n_features = 32
    X_pos = np.random.randn(n_samples // 2, n_features).astype("float32") + 1.0
    X_neg = np.random.randn(n_samples // 2, n_features).astype("float32") - 1.0
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))
    perm = np.random.permutation(n_samples)
    return X[perm], y[perm]


@pytest.fixture
def flat_index(small_dataset):
    X, _ = small_dataset
    return IndexBuilder(IndexConfig(index_string="Flat")).build(X)


@pytest.fixture
def ivf_index(medium_dataset):
    X, _ = medium_dataset
    return IndexBuilder(IndexConfig(index_string="IVF16,Flat")).build(X)


@pytest.fixture
def hnsw_index(medium_dataset):
    X, _ = medium_dataset
    return IndexBuilder(IndexConfig(index_string="HNSW32,Flat")).build(X)


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def create_annotated_set():
    """Create a balanced (positive/negative) annotated subset from labels y."""

    def _create(y, size=10):
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        n_pos = min(size // 2, len(pos))
        n_neg = min(size // 2, len(neg))
        ids = np.concatenate(
            [
                np.random.choice(pos, n_pos, replace=False),
                np.random.choice(neg, n_neg, replace=False),
            ]
        )
        return ids.tolist(), y[ids].tolist()

    return _create


@pytest.fixture(params=["Flat", "IVF16,Flat", "HNSW16,Flat"])
def index_type(request):
    return request.param


@pytest.fixture(params=[10, 50, 100])
def batch_size(request):
    return request.param


@pytest.fixture(params=[0.3, 0.5, 0.7])
def task_complexity(request):
    return request.param


@pytest.fixture(scope="session")
def synthetic_image_features():
    """Synthetic CLIP-like features arranged in 3 clusters, with fake image paths."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 128
    cluster_sizes = [n_samples // 3, n_samples // 3, n_samples - 2 * (n_samples // 3)]
    chunks = []
    for n in cluster_sizes:
        center = np.random.randn(n_features)
        chunks.append(center + np.random.randn(n, n_features) * 0.3)
    features = np.vstack(chunks).astype("float32")
    image_paths = [f"/fake/images/img_{i:05d}.jpg" for i in range(len(features))]
    return features, image_paths


@pytest.fixture
def mcp_session(synthetic_image_features):
    from active_learning_session import ActiveLearningSession

    features, image_paths = synthetic_image_features
    return ActiveLearningSession(
        image_paths=image_paths,
        feature_vectors=features,
        index_config=IndexConfig("Flat"),
    )


@pytest.fixture
def mcp_session_with_labels(mcp_session):
    initial_labels = {mcp_session.image_paths[i]: i % 2 for i in range(10)}
    mcp_session.add_labels(initial_labels)
    return mcp_session

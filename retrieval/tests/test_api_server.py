"""End-to-end tests for the FastAPI server with a mocked session.

Bypasses the on-startup ETL load by populating module globals directly, so
these tests do not require MinIO, CLIP, or `etl_complete.json`.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_server
from active_learning_session import ActiveLearningSession
from config import IndexConfig


@pytest.fixture
def configured_app():
    """Wire the api_server module globals to a small in-memory session."""
    rng = np.random.default_rng(0)
    n, dim, n_classes = 50, 16, 2
    features = rng.standard_normal((n, dim)).astype("float32")
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    object_keys = [f"images/img_{i:03d}.jpg" for i in range(n)]

    api_server.session = ActiveLearningSession(
        image_paths=object_keys,
        feature_vectors=features,
        index_config=IndexConfig("Flat"),
    )
    api_server.label_classes = ["cat", "dog"]
    text_emb = rng.standard_normal((n_classes, dim)).astype("float32")
    text_emb /= np.linalg.norm(text_emb, axis=1, keepdims=True)
    api_server.text_embeddings = text_emb
    api_server.minio_client = None
    api_server.bucket = "images"

    yield TestClient(api_server.app)

    api_server.session = None
    api_server.label_classes = []
    api_server.text_embeddings = None
    api_server.minio_client = None


def test_health(configured_app):
    r = configured_app.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_status_returns_label_classes(configured_app):
    r = configured_app.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["label_classes"] == ["cat", "dog"]


def test_status_503_when_uninitialized():
    api_server.session = None
    client = TestClient(api_server.app)
    r = client.get("/status")
    assert r.status_code == 503


def test_samples_random(configured_app):
    r = configured_app.post(
        "/samples", json={"sampler": "random", "num_samples": 5}
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["samples"]) == 5
    for s in body["samples"]:
        assert s["image_key"].startswith("images/")
        assert s["image_url"] == f"/images/{s['image_key']}"


def test_samples_unknown_sampler(configured_app):
    r = configured_app.post(
        "/samples", json={"sampler": "does-not-exist", "num_samples": 5}
    )
    # Session raises ValueError → 400, or RuntimeError → 500. Either is a server
    # rejection, not a silent success.
    assert r.status_code in (400, 500)


def test_labels_endpoint(configured_app):
    keys = api_server.session.image_paths[:3]
    r = configured_app.post(
        "/labels", json={"labels": {keys[0]: 0, keys[1]: 1, keys[2]: 0}}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["labels_added"] == 3
    assert body["total_labeled"] == 3


def test_auto_label_no_apply(configured_app):
    keys = api_server.session.image_paths[:4]
    r = configured_app.post(
        "/auto-label", json={"image_keys": keys, "apply": False}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["labels_applied"] == 0
    assert len(body["predictions"]) == 4
    for p in body["predictions"]:
        assert p["class_name"] in {"cat", "dog"}
        assert 0.0 <= p["confidence"] <= 1.0


def test_auto_label_unknown_key(configured_app):
    r = configured_app.post(
        "/auto-label", json={"image_keys": ["nope.jpg"], "apply": False}
    )
    assert r.status_code == 400

"""ETL pipeline: Image dataset -> MinIO -> download -> CLIP features -> cache."""

import json
import logging
import os
import time
from typing import List

import numpy as np
from minio import Minio

logger = logging.getLogger(__name__)


def load_oxford_pets(
    num_per_class: int, output_dir: str, label_classes: List[str]
) -> List[str]:
    """Load Oxford-IIIT Pets images (real photos, ~400x500px).

    Uses binary-category (cat=0, dog=1) when label_classes is ["cat", "dog"],
    otherwise falls back to breed-level category labels.

    Args:
        num_per_class: Number of images per class.
        output_dir: Directory to save JPGs.
        label_classes: List of class names (e.g., ["cat", "dog"]).

    Returns:
        List of local file paths.
    """
    import torchvision

    os.makedirs(output_dir, exist_ok=True)

    binary_mode = sorted(c.lower() for c in label_classes) == ["cat", "dog"]

    if binary_mode:
        ds = torchvision.datasets.OxfordIIITPet(
            root=output_dir, split="trainval",
            target_types="binary-category", download=True,
        )
        class_names = ["cat", "dog"]
    else:
        ds = torchvision.datasets.OxfordIIITPet(
            root=output_dir, split="trainval",
            target_types="category", download=True,
        )
        class_names = ds.classes
        # Filter to requested classes
        requested = {c.strip().lower() for c in label_classes}
        valid = {c.lower(): i for i, c in enumerate(class_names)}
        for r in requested:
            if r not in valid:
                raise ValueError(
                    f"Class '{r}' not in Oxford Pets classes: {class_names}"
                )

    local_paths = []
    counts = {}
    for cls in label_classes:
        counts[cls.lower()] = 0

    for i in range(len(ds)):
        img, label = ds[i]

        if binary_mode:
            cls_name = class_names[label]
        else:
            cls_name = class_names[label].lower()

        if cls_name not in counts:
            continue
        if counts[cls_name] >= num_per_class:
            if all(c >= num_per_class for c in counts.values()):
                break
            continue

        filename = f"{cls_name}_{counts[cls_name]:05d}.jpg"
        filepath = os.path.join(output_dir, filename)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(filepath, "JPEG", quality=95)
        local_paths.append(filepath)
        counts[cls_name] += 1

    logger.info("Saved %d Oxford Pets images to %s", len(local_paths), output_dir)
    for cls, count in counts.items():
        logger.info("  %s: %d images", cls, count)
    return local_paths


def load_cifar_samples(
    num_per_class: int, output_dir: str, label_classes: List[str]
) -> List[str]:
    """Load CIFAR-10 images for the specified classes (32x32, low-res fallback).

    Args:
        num_per_class: Number of images per class.
        output_dir: Directory to save JPGs.
        label_classes: List of class names (e.g., ["airplane", "automobile"]).

    Returns:
        List of local file paths.
    """
    import torchvision

    os.makedirs(output_dir, exist_ok=True)

    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    class_indices = {}
    for cls_name in label_classes:
        cls_lower = cls_name.strip().lower()
        if cls_lower not in cifar10_classes:
            raise ValueError(
                f"Class '{cls_name}' not in CIFAR-10 classes: {cifar10_classes}"
            )
        class_indices[cls_lower] = cifar10_classes.index(cls_lower)

    dataset = torchvision.datasets.CIFAR10(
        root=output_dir, train=True, download=True
    )

    local_paths = []
    counts = {idx: 0 for idx in class_indices.values()}

    for img, label in dataset:
        if label not in counts:
            continue
        if counts[label] >= num_per_class:
            if all(c >= num_per_class for c in counts.values()):
                break
            continue

        cls_name = cifar10_classes[label]
        filename = f"{cls_name}_{counts[label]:05d}.jpg"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath, "JPEG")
        local_paths.append(filepath)
        counts[label] += 1

    logger.info("Saved %d CIFAR-10 images to %s", len(local_paths), output_dir)
    return local_paths


def upload_to_minio(
    client: Minio, bucket: str, local_paths: List[str]
) -> List[str]:
    """Upload files to MinIO under images/ prefix.

    Returns:
        List of object keys.
    """
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info("Created bucket: %s", bucket)

    object_keys = []
    for path in local_paths:
        filename = os.path.basename(path)
        object_key = f"images/{filename}"
        client.fput_object(bucket, object_key, path)
        object_keys.append(object_key)

    logger.info("Uploaded %d objects to s3://%s/images/", len(object_keys), bucket)
    return object_keys


def download_from_minio(
    client: Minio, bucket: str, object_keys: List[str], local_dir: str
) -> List[str]:
    """Download objects from MinIO to local directory.

    Returns:
        List of local file paths.
    """
    os.makedirs(local_dir, exist_ok=True)
    local_paths = []
    for key in object_keys:
        filename = os.path.basename(key)
        local_path = os.path.join(local_dir, filename)
        client.fget_object(bucket, key, local_path)
        local_paths.append(local_path)

    logger.info("Downloaded %d objects to %s", len(local_paths), local_dir)
    return local_paths


def extract_features(local_paths: List[str], cache_dir: str) -> np.ndarray:
    """Extract CLIP features and save to cache.

    Returns:
        Feature array of shape (N, 512).
    """
    from feature_extraction import CLIPFeatureExtractor

    os.makedirs(cache_dir, exist_ok=True)
    features_path = os.path.join(cache_dir, "features.npy")

    extractor = CLIPFeatureExtractor(model_name="ViT-B/32")
    features = extractor.extract_from_paths(local_paths, batch_size=32)
    np.save(features_path, features)

    logger.info("Saved features %s to %s", features.shape, features_path)
    return features


# Oxford Pets classes that are available for cat/dog binary classification
OXFORD_PETS_CLASSES = {
    "cat", "dog",
    # Individual cat breeds
    "abyssinian", "bengal", "birman", "bombay", "british shorthair",
    "egyptian mau", "maine coon", "persian", "ragdoll", "russian blue",
    "siamese", "sphynx",
    # Individual dog breeds
    "american bulldog", "american pit bull terrier", "basset hound",
    "beagle", "boxer", "chihuahua", "english cocker spaniel",
    "english setter", "german shorthaired", "great pyrenees", "havanese",
    "japanese chin", "keeshond", "leonberger", "miniature pinscher",
    "newfoundland", "pomeranian", "pug", "saint bernard", "samoyed",
    "scottish terrier", "shiba inu", "staffordshire bull terrier",
    "wheaten terrier", "yorkshire terrier",
}

CIFAR10_CLASSES = {
    "airplane", "automobile", "bird", "deer",
    "frog", "horse", "ship", "truck",
}


def run_etl() -> None:
    """Run the full ETL pipeline with caching."""
    cache_dir = os.environ.get("CACHE_DIR", "cache")
    state_file = os.path.join(cache_dir, "etl_complete.json")

    # Skip if already completed
    if os.path.exists(state_file):
        logger.info("ETL already complete (found %s), skipping.", state_file)
        return

    start = time.time()

    # Config from environment
    endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
    bucket = os.environ.get("MINIO_BUCKET", "images")
    num_images = int(os.environ.get("NUM_SAMPLE_IMAGES", "200"))
    label_classes_str = os.environ.get("LABEL_CLASSES", "cat,dog")
    label_classes = [c.strip() for c in label_classes_str.split(",")]
    num_per_class = num_images // len(label_classes)

    logger.info(
        "ETL config: %d images, classes=%s, endpoint=%s",
        num_images, label_classes, endpoint,
    )

    # Auto-detect dataset based on class names
    classes_lower = {c.lower() for c in label_classes}
    if classes_lower.issubset(OXFORD_PETS_CLASSES):
        dataset_name = "oxford_pets"
        logger.info("Using Oxford-IIIT Pets dataset (real photos)")
    elif classes_lower.issubset(CIFAR10_CLASSES | {"cat", "dog"}):
        dataset_name = "cifar10"
        logger.info("Using CIFAR-10 dataset")
    else:
        raise ValueError(
            f"Classes {label_classes} not supported. "
            f"Use Oxford Pets classes or CIFAR-10 classes."
        )

    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

    # Step 1: Load images
    raw_dir = os.path.join(cache_dir, "raw")
    if dataset_name == "oxford_pets":
        local_paths = load_oxford_pets(num_per_class, raw_dir, label_classes)
    else:
        local_paths = load_cifar_samples(num_per_class, raw_dir, label_classes)

    # Step 2: Upload to MinIO
    object_keys = upload_to_minio(client, bucket, local_paths)

    # Step 3: Download from MinIO (proves round-trip)
    download_dir = os.path.join(cache_dir, "downloaded")
    downloaded_paths = download_from_minio(client, bucket, object_keys, download_dir)

    # Step 4: Extract features
    features = extract_features(downloaded_paths, cache_dir)

    # Step 5: Save state
    os.makedirs(cache_dir, exist_ok=True)
    state = {
        "object_keys": object_keys,
        "features_path": os.path.join(cache_dir, "features.npy"),
        "num_images": len(object_keys),
        "label_classes": label_classes,
        "dataset": dataset_name,
        "elapsed_seconds": round(time.time() - start, 1),
    }
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    logger.info(
        "ETL complete in %ss: %d images, features %s",
        state["elapsed_seconds"], len(object_keys), features.shape,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_etl()

#!/usr/bin/env python3
"""
Labeling Copilot Agent Demo
============================
End-to-end demo: retrieve dog/cat images via active learning,
annotate them with multi-model object detection, and (optionally)
synthesise new training data.

Services required:
  - retrieval  @ localhost:8000  (CLIP retrieval + active learning)
  - annotate           @ localhost:8080  (detection + segmentation orchestrator)
  - synthesis  @ localhost:8090  (augmentation API service)

Usage:
    python agent_demo.py
"""

import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://localhost:8000")
ANNOTATOR_URL = os.environ.get("ANNOTATOR_URL", "http://localhost:8080")
SYNTHESIS_URL = os.environ.get("SYNTHESIS_URL", "http://localhost:8090")

VOCABULARY = "cat, dog"          # Detection classes
NUM_SAMPLES_PER_ROUND = 5       # Images per active-learning round
NUM_ROUNDS = 3                   # Active-learning iterations
OUTPUT_DIR = Path("agent_output")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for_service(url, name, timeout=300):
    """Block until a service responds to /health."""
    print(f"  Waiting for {name} at {url} ...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                print(" ready")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(3)
        print(".", end="", flush=True)
    print(f" TIMEOUT — {name} not reachable")
    return False


def fetch_image(image_url):
    """Download an image from the retrieval proxy."""
    r = requests.get(f"{RETRIEVAL_URL}{image_url}", timeout=30)
    r.raise_for_status()
    return r.content


def detect_objects(image_bytes, filename="image.jpg"):
    """Run multi-model detection via the orchestrator."""
    r = requests.post(
        f"{ANNOTATOR_URL}/annotate/detect",
        files={"image": (filename, image_bytes, "image/jpeg")},
        data={
            "vocabulary": VOCABULARY,
            "nms_methods": '["NON_MAX_SUPPRESSION"]',
            "nms_params": '{"iou_threshold": 0.5, "sigma": 0.5, "min_score": 0.1}',
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def segment_objects(image_bytes, filename="image.jpg"):
    """Run segmentation via SAM + SEEM."""
    r = requests.post(
        f"{ANNOTATOR_URL}/annotate/segment",
        files={"image": (filename, image_bytes, "image/jpeg")},
        data={"models": '["SAM", "SEEM"]'},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def format_detection(det):
    """Pretty-format a single detection."""
    return f"  {det['label']:>6s}  conf={det['confidence']:.3f}  bbox={det['bbox']}"


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_check_services():
    """Stage 0: verify all services are up."""
    print("\n" + "=" * 60)
    print("STAGE 0 — Service health check")
    print("=" * 60)

    retrieval_ok = wait_for_service(RETRIEVAL_URL, "Retrieval")
    annotator_ok = wait_for_service(ANNOTATOR_URL, "Annotator")

    if not retrieval_ok:
        sys.exit("Retrieval service not available. Is the container running?")

    # Annotator is optional for now (user said it might not be up)
    return retrieval_ok, annotator_ok


def stage_explore_dataset():
    """Stage 1: get dataset status from retrieval service."""
    print("\n" + "=" * 60)
    print("STAGE 1 — Dataset overview")
    print("=" * 60)

    r = requests.get(f"{RETRIEVAL_URL}/status", timeout=30)
    r.raise_for_status()
    status = r.json()

    print(f"  Total images : {status['total_images']}")
    print(f"  Labeled      : {status['labeled_count']}")
    print(f"  Unlabeled    : {status['unlabeled_count']}")
    print(f"  Classes      : {status['label_classes']}")
    print(f"  FAISS index  : {status['index_info']['type']} "
          f"({status['index_info']['vectors_indexed']} vectors, "
          f"dim={status['index_info']['dimension']})")
    print(f"  Samplers     : {[s['name'] for s in status['available_samplers']]}")

    return status


def stage_active_learning_loop(annotator_available):
    """Stage 2: iterative active-learning loop with annotation."""
    print("\n" + "=" * 60)
    print("STAGE 2 — Active learning loop")
    print("=" * 60)

    all_annotations = {}
    samplers = ["random", "kcenter", "margin"]

    for round_num in range(NUM_ROUNDS):
        sampler = samplers[round_num % len(samplers)]
        print(f"\n--- Round {round_num + 1}/{NUM_ROUNDS}  (sampler: {sampler}) ---")

        # 2a. Request samples
        payload = {"sampler": sampler, "num_samples": NUM_SAMPLES_PER_ROUND}
        r = requests.post(f"{RETRIEVAL_URL}/samples", json=payload, timeout=60)
        r.raise_for_status()
        sample_resp = r.json()

        samples = sample_resp["samples"]
        meta = sample_resp["metadata"]
        print(f"  Received {meta['samples_returned']} samples "
              f"(unlabeled remaining: {meta['unlabeled_remaining']})")

        # 2b. Auto-label with CLIP (fast, gives us a baseline)
        image_keys = [s["image_key"] for s in samples]
        auto_r = requests.post(
            f"{RETRIEVAL_URL}/auto-label",
            json={"image_keys": image_keys, "apply": True},
            timeout=60,
        )
        auto_r.raise_for_status()
        auto_resp = auto_r.json()

        print("  CLIP auto-labels:")
        for pred in auto_resp["predictions"]:
            key = pred["image_key"]
            print(f"    {os.path.basename(key):>25s} -> "
                  f"{pred['class_name']} (conf={pred['confidence']:.3f})")

        # 2c. If annotator is available, run detection on each sample
        if annotator_available:
            print("  Running multi-model detection on each sample...")
            for sample in samples:
                key = sample["image_key"]
                basename = os.path.basename(key)
                try:
                    img_bytes = fetch_image(sample["image_url"])
                except Exception as e:
                    print(f"    {basename}: failed to fetch — {e}")
                    continue

                # Save raw image
                img_dir = OUTPUT_DIR / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                (img_dir / basename).write_bytes(img_bytes)

                try:
                    det_result = detect_objects(img_bytes, basename)
                    consensus = det_result["consensus"][0]["annotations"]
                    raw_models = [r["model_name"] for r in det_result["raw_results"]]

                    print(f"    {basename}:  models={raw_models}  "
                          f"consensus={len(consensus)} detections")
                    for det in consensus:
                        print(format_detection(det))

                    all_annotations[key] = {
                        "image_file": basename,
                        "clip_label": next(
                            (p["class_name"] for p in auto_resp["predictions"]
                             if p["image_key"] == key), None
                        ),
                        "detections": consensus,
                        "raw_model_count": len(det_result["raw_results"]),
                    }
                except Exception as e:
                    print(f"    {basename}: detection failed — {e}")
        else:
            print("  (Annotator not available — skipping detection)")
            for pred in auto_resp["predictions"]:
                key = pred["image_key"]
                all_annotations[key] = {
                    "image_file": os.path.basename(key),
                    "clip_label": pred["class_name"],
                    "clip_confidence": pred["confidence"],
                    "detections": [],
                    "raw_model_count": 0,
                    "note": "annotator_unavailable",
                }

    return all_annotations


def stage_synthesis_check():
    """Stage 3: show synthesis service status and usage."""
    print("\n" + "=" * 60)
    print("STAGE 3 — Synthesis (status check)")
    print("=" * 60)

    env_path = Path("synthesis/.env")
    if not env_path.exists():
        print("  .env not found — synthesis not configured")
        return

    synthesis_ok = False
    try:
        resp = requests.get(f"{SYNTHESIS_URL}/health", timeout=5)
        synthesis_ok = resp.status_code == 200
    except requests.RequestException:
        synthesis_ok = False

    if synthesis_ok:
        print(f"  Synthesis API is reachable at {SYNTHESIS_URL}")
        print("  Trigger it with POST /synthesize using uploaded seed images,")
        print("  relative retrieval image URLs, or an input directory under /app.")
    else:
        print(f"  Synthesis API is not reachable at {SYNTHESIS_URL}")
        print("  Start it with: ./scripts/docker-up.sh --with-synthesis")

    # Check if credentials are placeholder
    env_text = env_path.read_text()
    if "your-" in env_text:
        print("  .env contains placeholder credentials — synthesis not runnable")
    else:
        print("  .env found with credentials")
        print("  Seed images can come from synthesis/input-images/,")
        print("  uploaded files, or retrieval URLs passed to the synthesis API.")

    # Describe what the synthesis pipeline does
    print("\n  Synthesis pipeline overview:")
    print("    - Analyzes each input image with GPT vision")
    print("    - Generates augmentation suggestions (e.g., lighting, angle, background)")
    print("    - Creates variants via image editing (gpt-image-1.5 by default)")
    print("    - Runs Forte OOD detection to filter low-quality / off-distribution outputs")
    print("    - Computes quality metrics: PRDC, Frechet Distance, Inception Score, etc.")


def stage_save_results(annotations, dataset_status):
    """Stage 4: persist all outputs."""
    print("\n" + "=" * 60)
    print("STAGE 4 — Save results")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save annotations as COCO-style-ish JSON
    coco_output = {
        "info": {
            "description": "Labeling Copilot Agent Demo — Dogs vs Cats",
            "dataset": dataset_status.get("label_classes", ["cat", "dog"]),
            "total_pool_size": dataset_status.get("total_images", 0),
            "images_annotated": len(annotations),
        },
        "annotations": [],
    }

    ann_id = 0
    for key, ann in annotations.items():
        for det in ann.get("detections", []):
            bbox = det["bbox"]
            # Convert [xmin, ymin, xmax, ymax] -> [x, y, w, h] for COCO
            coco_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            coco_output["annotations"].append({
                "id": ann_id,
                "image_file": ann["image_file"],
                "image_key": key,
                "category": det["label"],
                "confidence": round(det["confidence"], 4),
                "bbox_xyxy": bbox,
                "bbox_xywh": coco_bbox,
            })
            ann_id += 1

        # If no detections but we have a CLIP label, store that
        if not ann.get("detections") and ann.get("clip_label"):
            coco_output["annotations"].append({
                "id": ann_id,
                "image_file": ann["image_file"],
                "image_key": key,
                "category": ann["clip_label"],
                "confidence": round(ann.get("clip_confidence", 0), 4),
                "bbox_xyxy": [],
                "bbox_xywh": [],
                "source": "clip_auto_label",
            })
            ann_id += 1

    out_path = OUTPUT_DIR / "annotations.json"
    with open(out_path, "w") as f:
        json.dump(coco_output, f, indent=2)

    print(f"  Saved {ann_id} annotations to {out_path}")
    print(f"  Images saved to {OUTPUT_DIR / 'images'}/")

    # Summary
    cats = sum(1 for a in coco_output["annotations"] if a["category"] == "cat")
    dogs = sum(1 for a in coco_output["annotations"] if a["category"] == "dog")
    print(f"\n  Summary: {cats} cat detections, {dogs} dog detections")

    return coco_output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  LABELING COPILOT AGENT")
    print("  Task: Curate a dog vs cat detection dataset")
    print("=" * 60)

    # Stage 0: Health check
    retrieval_ok, annotator_ok = stage_check_services()

    # Stage 1: Explore what we have
    dataset_status = stage_explore_dataset()

    # Stage 2: Active learning loop
    annotations = stage_active_learning_loop(annotator_ok)

    # Stage 3: Synthesis status
    stage_synthesis_check()

    # Stage 4: Save everything
    coco = stage_save_results(annotations, dataset_status)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

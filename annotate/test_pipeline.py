"""End-to-end test: download 5 sample images, run through detection + segmentation pipelines, save annotated results."""
import json
import os
import requests
from io import BytesIO
from PIL import Image, ImageDraw

ORCHESTRATOR = "http://localhost:8080"
OUTPUT_DIR = "test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Labeling-Copilot-Test/1.0"})

# 5 sample images with appropriate vocabularies
SAMPLES = [
    {
        "name": "dog",
        "url": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=800",
        "vocabulary": "dog, grass, sky",
    },
    {
        "name": "city_street",
        "url": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800",
        "vocabulary": "car, person, building, sign, bus",
    },
    {
        "name": "fruit",
        "url": "https://images.unsplash.com/photo-1619566636858-adf3ef46400b?w=800",
        "vocabulary": "apple, fruit, table, banana, orange",
    },
    {
        "name": "bicycle",
        "url": "https://images.unsplash.com/photo-1485965120184-e220f721d03e?w=800",
        "vocabulary": "bicycle, wheel, road, wall",
    },
    {
        "name": "cat",
        "url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800",
        "vocabulary": "cat, animal, eye",
    },
]

COLORS = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFA500", "#800080"]


def draw_annotations(image: Image.Image, annotations: list, title: str) -> Image.Image:
    """Draw bounding boxes and labels on image."""
    draw = ImageDraw.Draw(image)
    label_colors = {}
    color_idx = 0

    for ann in annotations:
        label = ann.get("label", "unknown")
        confidence = ann.get("confidence", 0.0)
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue

        if label not in label_colors:
            label_colors[label] = COLORS[color_idx % len(COLORS)]
            color_idx += 1

        color = label_colors[label]
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} {confidence:.2f}"
        draw.text((x1, max(0, y1 - 15)), text, fill=color)

    # Draw title
    draw.text((10, 10), title, fill="white")
    return image


def draw_sam_masks(image: Image.Image, masks: list, title: str) -> Image.Image:
    """Draw SAM mask bounding boxes on image (SAM uses [x, y, w, h] format)."""
    draw = ImageDraw.Draw(image)
    for i, mask in enumerate(masks):
        bbox = mask.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, w, h = bbox
        color = COLORS[i % len(COLORS)]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        score = mask.get("stability_score", 0.0)
        draw.text((x, max(0, y - 12)), f"{score:.2f}", fill=color)

    draw.text((10, 10), title, fill="white")
    return image


def draw_seem_segments(image: Image.Image, segments: list, title: str) -> Image.Image:
    """Draw SEEM segment bounding boxes and labels on image ([xmin, ymin, xmax, ymax] format)."""
    draw = ImageDraw.Draw(image)
    label_colors = {}
    color_idx = 0

    for seg in segments:
        label = seg.get("label", "unknown")
        bbox = seg.get("bbox", [])
        area = seg.get("area", 0)
        if len(bbox) != 4:
            continue

        if label not in label_colors:
            label_colors[label] = COLORS[color_idx % len(COLORS)]
            color_idx += 1

        color = label_colors[label]
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, max(0, y1 - 15)), f"{label} ({area}px)", fill=color)

    draw.text((10, 10), title, fill="white")
    return image


def test_detection(name: str, image_bytes: bytes, vocabulary: str):
    """Run detection and return results."""
    print(f"  Running detection with vocabulary: {vocabulary}")
    resp = requests.post(
        f"{ORCHESTRATOR}/annotate/detect",
        files={"image": ("image.jpg", image_bytes, "image/jpeg")},
        data={"vocabulary": vocabulary},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def test_segmentation(name: str, image_bytes: bytes):
    """Run segmentation (SAM + SEEM) and return results."""
    print(f"  Running segmentation (SAM + SEEM)...")
    resp = requests.post(
        f"{ORCHESTRATOR}/annotate/segment",
        files={"image": ("image.jpg", image_bytes, "image/jpeg")},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    for i, sample in enumerate(SAMPLES):
        name = sample["name"]
        url = sample["url"]
        vocab = sample["vocabulary"]

        print(f"\n[{i+1}/5] Processing '{name}'...")

        # Download image
        print(f"  Downloading image...")
        try:
            img_resp = SESSION.get(url, timeout=30)
            img_resp.raise_for_status()
        except Exception as e:
            print(f"  FAILED to download: {e}")
            continue

        image_bytes = img_resp.content
        original = Image.open(BytesIO(image_bytes)).convert("RGB")

        # --- Detection ---
        try:
            result = test_detection(name, image_bytes, vocab)
        except Exception as e:
            print(f"  FAILED detection: {e}")
            result = None

        if result:
            json_path = os.path.join(OUTPUT_DIR, f"{name}_detect.json")
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)

            raw_results = result.get("raw_results", [])
            for raw in raw_results:
                model_name = raw.get("model_name", "unknown")
                annotations = raw.get("annotations", [])
                img_copy = original.copy()
                draw_annotations(img_copy, annotations, f"{model_name}: {len(annotations)} detections")
                out_path = os.path.join(OUTPUT_DIR, f"{name}_{model_name}.jpg")
                img_copy.save(out_path)
                print(f"    {model_name}: {len(annotations)} detections -> {out_path}")

            consensus = result.get("consensus", [])
            if consensus:
                nms_annotations = consensus[0].get("annotations", [])
                img_copy = original.copy()
                draw_annotations(img_copy, nms_annotations, f"NMS Consensus: {len(nms_annotations)} detections")
                out_path = os.path.join(OUTPUT_DIR, f"{name}_consensus.jpg")
                img_copy.save(out_path)
                print(f"    Consensus: {len(nms_annotations)} detections -> {out_path}")

        # --- Segmentation ---
        try:
            seg_result = test_segmentation(name, image_bytes)
        except Exception as e:
            print(f"  FAILED segmentation: {e}")
            seg_result = None

        if seg_result:
            json_path = os.path.join(OUTPUT_DIR, f"{name}_segment.json")
            with open(json_path, "w") as f:
                json.dump(seg_result, f, indent=2)

            for raw in seg_result.get("raw_results", []):
                if "error" in raw:
                    print(f"    {raw.get('model_name', 'unknown')}: ERROR - {raw['error']}")
                    continue

                model_name = raw.get("model_name", "unknown")

                if model_name == "SAM":
                    masks = raw.get("masks", [])
                    img_copy = original.copy()
                    draw_sam_masks(img_copy, masks, f"SAM: {len(masks)} masks")
                    out_path = os.path.join(OUTPUT_DIR, f"{name}_SAM.jpg")
                    img_copy.save(out_path)
                    print(f"    SAM: {len(masks)} masks -> {out_path}")

                elif model_name == "SEEM":
                    segments = raw.get("segments", [])
                    img_copy = original.copy()
                    draw_seem_segments(img_copy, segments, f"SEEM: {len(segments)} segments")
                    out_path = os.path.join(OUTPUT_DIR, f"{name}_SEEM.jpg")
                    img_copy.save(out_path)
                    print(f"    SEEM: {len(segments)} segments -> {out_path}")
                    for seg in segments:
                        print(f"      {seg['label']}: area={seg['area']}, bbox={[round(b,1) for b in seg['bbox']]}")

    print(f"\nDone! All outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

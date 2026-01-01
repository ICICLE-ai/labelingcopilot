"""
Asynchronous quality evaluation runner.
Runs in background to compute quality metrics and update metadata.

Usage:
    # Run in background
    python run_evaluation_async.py --output-dir ./augmented-output &

    # Monitor progress
    tail -f ./augmented-output/evaluation.log
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from config.config import Config, EvaluationConfig, ForteConfig
from core.evaluator import QualityEvaluator
from core.metadata import MetadataManager
from utils.image_utils import ImageLoader


def load_metadata(output_dir: str) -> Dict[str, Any]:
    """Load existing metadata file."""
    metadata_file = Path(output_dir) / "metadata.json"
    if not metadata_file.exists():
        return None

    with open(metadata_file, 'r') as f:
        return json.load(f)


def discover_images(output_dir: str, input_dir: str = "./input-images") -> Dict[str, Any]:
    """
    Discover images by scanning directories when metadata doesn't exist.

    Returns a minimal metadata structure.
    """
    output_path = Path(output_dir)
    input_path = Path(input_dir)

    # Find all generated images (PNG files in output directory)
    generated_images = list(output_path.glob("*.png"))

    # Find all original images
    original_images = []
    if input_path.exists():
        original_images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")) + list(input_path.glob("*.png"))

    return {
        'total_augmented_images': len(generated_images),
        'total_original_images': len(original_images),
        'images': [],  # Don't have detailed metadata
        'generated_paths': [str(p) for p in generated_images],
        'original_paths': [str(p) for p in original_images]
    }


def save_quality_metadata(
    output_dir: str,
    quality_scores: Dict[str, Any],
    num_original: int,
    num_generated: int,
    evaluation_time: float,
    ood_stats: Optional[Dict[str, Any]] = None
):
    """Save comprehensive quality metadata to separate file."""
    quality_file = Path(output_dir) / "quality_metadata.json"

    # Load existing quality metadata if it exists
    if quality_file.exists():
        with open(quality_file, 'r') as f:
            quality_metadata = json.load(f)
    else:
        quality_metadata = {
            'created_at': datetime.now().isoformat(),
            'evaluations': []
        }

    # Create new evaluation record
    evaluation_record = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': {
            'num_original_images': num_original,
            'num_generated_images': num_generated
        },
        'quality_metrics': quality_scores,
        'evaluation_time_seconds': evaluation_time
    }

    # Add OOD stats if available
    if ood_stats:
        evaluation_record['ood_detection'] = ood_stats

    # Update metadata
    quality_metadata['last_updated'] = datetime.now().isoformat()
    quality_metadata['latest_metrics'] = quality_scores
    quality_metadata['evaluations'].append(evaluation_record)

    # Save
    with open(quality_file, 'w') as f:
        json.dump(quality_metadata, f, indent=2)

    print(f"[Evaluation] Quality metadata saved to {quality_file}")
    return quality_file


def update_metadata_with_evaluation(output_dir: str, quality_scores: Dict[str, Any]):
    """Update metadata.json with quality scores (if it exists)."""
    metadata_file = Path(output_dir) / "metadata.json"

    if not metadata_file.exists():
        print(f"[Evaluation] Metadata file doesn't exist yet - skipping metadata update")
        return

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Add quality scores to aggregate section
    if 'aggregate_scores' not in metadata:
        metadata['aggregate_scores'] = {}

    metadata['aggregate_scores'].update(quality_scores)
    metadata['last_evaluation'] = datetime.now().isoformat()

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[Evaluation] Updated metadata with quality scores")


def evaluate_ood_scores(
    evaluator: QualityEvaluator,
    original_paths: List[str],
    generated_paths: List[str],
    output_dir: str
) -> Optional[Dict[str, Any]]:
    """
    Compute OOD scores for all generated images.

    Returns:
        Dictionary with OOD statistics including per-image scores
    """
    print("[Evaluation] Computing OOD scores for generated images...")

    if not generated_paths:
        print("[Evaluation] No generated images found")
        return None

    # Fit OOD detector on original images first
    evaluator.fit_ood_detector(original_paths)

    # Compute OOD scores
    ood_scores = evaluator.eval_batch_ood(generated_paths)

    if ood_scores is None:
        return None

    # Calculate statistics
    threshold = 0.3
    in_dist = sum(1 for score in ood_scores if score > threshold)

    # Create per-image OOD information
    per_image_ood = []
    for i, (path, score) in enumerate(zip(generated_paths, ood_scores)):
        per_image_ood.append({
            'image_path': path,
            'image_name': Path(path).name,
            'ood_score': float(score),
            'is_in_distribution': bool(score > threshold),
            'is_ood': bool(score <= threshold)
        })

    ood_stats = {
        'total_images': len(ood_scores),
        'threshold': threshold,
        'in_distribution_count': int(in_dist),
        'in_distribution_percentage': float(100 * in_dist / len(ood_scores)),
        'ood_score_mean': float(ood_scores.mean()),
        'ood_score_std': float(ood_scores.std()),
        'ood_score_min': float(ood_scores.min()),
        'ood_score_max': float(ood_scores.max()),
        'per_image_scores': per_image_ood  # Add per-image information
    }

    print(f"[Evaluation] OOD Detection Results:")
    print(f"  In-distribution: {in_dist}/{len(ood_scores)} ({ood_stats['in_distribution_percentage']:.1f}%)")
    print(f"  Mean score: {ood_stats['ood_score_mean']:.3f} ± {ood_stats['ood_score_std']:.3f}")

    # Also update generation metadata if it exists
    metadata_file = Path(output_dir) / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                full_metadata = json.load(f)

            if 'images' in full_metadata:
                for i, img in enumerate(full_metadata['images']):
                    if i < len(ood_scores):
                        img['ood_score'] = float(ood_scores[i])
                        img['is_in_distribution'] = bool(ood_scores[i] > threshold)
                        img['is_ood'] = bool(ood_scores[i] <= threshold)

                with open(metadata_file, 'w') as f:
                    json.dump(full_metadata, f, indent=2)

                print(f"[Evaluation] Updated generation metadata with OOD scores")
        except Exception as e:
            print(f"[Evaluation] Warning: Could not update generation metadata: {e}")

    return ood_stats


def main():
    parser = argparse.ArgumentParser(description="Run async quality evaluation")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./augmented-output",
        help="Output directory containing generated images and metadata"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode: continuously check for new images and evaluate"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Watch interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--skip-ood",
        action="store_true",
        help="Skip OOD detection (faster)"
    )

    args = parser.parse_args()

    # Setup logging
    log_file = Path(args.output_dir) / "evaluation.log"

    def log(msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(log_msg + "\n")

    log("=" * 60)
    log("Async Quality Evaluation")
    log("=" * 60)
    log(f"Output directory: {args.output_dir}")
    log(f"Watch mode: {args.watch}")
    log("=" * 60)

    # Initialize evaluation-only config (no Azure needed)
    from dataclasses import dataclass

    @dataclass
    class EvalOnlyConfig:
        """Minimal config for evaluation only."""
        evaluation: EvaluationConfig
        forte: ForteConfig

    eval_config = EvalOnlyConfig(
        evaluation=EvaluationConfig.from_env(),
        forte=ForteConfig.from_env()
    )

    evaluator = QualityEvaluator(eval_config)

    last_evaluated_count = 0

    while True:
        try:
            # Load metadata or discover images
            metadata = load_metadata(args.output_dir)

            if metadata is None:
                log("Metadata file not found. Discovering images from directories...")
                metadata = discover_images(args.output_dir)
                log(f"Discovered {metadata['total_augmented_images']} generated images, {metadata['total_original_images']} original images")

                # Use discovered paths
                original_paths = metadata.get('original_paths', [])
                generated_paths = metadata.get('generated_paths', [])
            else:
                # IMPORTANT: For evaluation, always use ALL original images from input directory,
                # not just the source images that have been augmented in metadata.
                # This ensures OOD detection and quality metrics use the full dataset.
                input_path = Path("./input-images")
                if input_path.exists():
                    original_paths = [str(p) for p in list(input_path.glob("*.jpg")) +
                                     list(input_path.glob("*.jpeg")) +
                                     list(input_path.glob("*.png"))]
                else:
                    # Fallback to metadata sources if input dir doesn't exist
                    original_paths = list(set(img['source_image'] for img in metadata['images']))

                generated_paths = [img['output_image'] for img in metadata['images']]

            current_count = len(generated_paths)
            log(f"Found {current_count} generated images")

            # Check if there are new images to evaluate
            if current_count == last_evaluated_count and last_evaluated_count > 0:
                if args.watch:
                    log(f"No new images. Waiting {args.interval}s...")
                    time.sleep(args.interval)
                    continue
                else:
                    log("No new images. Exiting.")
                    break

            if current_count == 0:
                log("No images to evaluate yet.")
                if args.watch:
                    time.sleep(args.interval)
                    continue
                else:
                    break

            log(f"Evaluating quality: {len(original_paths)} original → {len(generated_paths)} generated")

            # Compute quality metrics
            log("Computing quality metrics...")
            start_time = time.time()

            quality_scores = evaluator.eval_quality(
                real_image_paths=original_paths,
                generated_image_paths=generated_paths,
                metrics=None  # Use config defaults
            )

            elapsed = time.time() - start_time
            log(f"Quality evaluation complete in {elapsed:.1f}s")

            # Print results
            log("-" * 40)
            log("Quality Scores:")
            for metric, value in quality_scores.items():
                if value is not None:
                    if isinstance(value, float):
                        log(f"  {metric}: {value:.4f}")
                    else:
                        log(f"  {metric}: {value}")
            log("-" * 40)

            # Compute OOD scores if enabled
            ood_stats = None
            if not args.skip_ood and eval_config.forte.enabled:
                log("Computing OOD scores...")
                ood_start = time.time()
                ood_stats = evaluate_ood_scores(
                    evaluator,
                    original_paths,
                    generated_paths,
                    args.output_dir
                )
                ood_elapsed = time.time() - ood_start
                log(f"OOD evaluation complete in {ood_elapsed:.1f}s")

            # Save comprehensive quality metadata
            save_quality_metadata(
                args.output_dir,
                quality_scores,
                len(original_paths),
                len(generated_paths),
                elapsed,
                ood_stats
            )

            # Also update generation metadata if it exists (for backwards compatibility)
            update_metadata_with_evaluation(args.output_dir, quality_scores)

            last_evaluated_count = current_count

            if not args.watch:
                log("Evaluation complete. Exiting.")
                break
            else:
                log(f"Evaluation complete. Waiting {args.interval}s for new images...")
                time.sleep(args.interval)

        except KeyboardInterrupt:
            log("Interrupted by user. Exiting.")
            break

        except Exception as e:
            log(f"Error: {e}")
            import traceback
            log(traceback.format_exc())
            if args.watch:
                log(f"Retrying in {args.interval}s...")
                time.sleep(args.interval)
            else:
                break


if __name__ == "__main__":
    main()

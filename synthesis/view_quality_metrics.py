#!/usr/bin/env python
"""
View quality metrics in a human-readable format.

Usage:
    python view_quality_metrics.py
    python view_quality_metrics.py --output-dir ./augmented-output
    python view_quality_metrics.py --show-ood-images  # Show OOD image list
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def format_metric(name: str, value) -> str:
    """Format a metric value for display."""
    if value is None:
        return f"  {name}: N/A"
    elif isinstance(value, float):
        return f"  {name}: {value:.4f}"
    else:
        return f"  {name}: {value}"


def view_quality_metadata(output_dir: str, show_ood_images: bool = False, show_all_ood: bool = False):
    """Display quality metadata in readable format."""
    quality_file = Path(output_dir) / "quality_metadata.json"

    if not quality_file.exists():
        print(f"❌ Quality metadata not found: {quality_file}")
        print("   Run evaluation first: python run_evaluation_async.py")
        return

    with open(quality_file, 'r') as f:
        data = json.load(f)

    print("=" * 70)
    print("QUALITY EVALUATION METADATA")
    print("=" * 70)
    print(f"Created: {data.get('created_at', 'N/A')}")
    print(f"Last Updated: {data.get('last_updated', 'N/A')}")
    print(f"Total Evaluations: {len(data.get('evaluations', []))}")
    print()

    # Show latest metrics
    if 'latest_metrics' in data:
        print("LATEST QUALITY METRICS")
        print("-" * 70)
        for metric, value in data['latest_metrics'].items():
            print(format_metric(metric, value))
        print()

    # Show evaluation history
    evaluations = data.get('evaluations', [])
    if evaluations:
        print("EVALUATION HISTORY")
        print("-" * 70)

        for i, eval_record in enumerate(reversed(evaluations[-5:]), 1):  # Show last 5
            print(f"\nEvaluation #{len(evaluations) - i + 1}")
            print(f"  Time: {eval_record.get('timestamp', 'N/A')}")

            dataset = eval_record.get('dataset_size', {})
            print(f"  Dataset: {dataset.get('num_original_images', 0)} original → {dataset.get('num_generated_images', 0)} generated")
            print(f"  Evaluation Time: {eval_record.get('evaluation_time_seconds', 0):.1f}s")

            # Quality metrics
            metrics = eval_record.get('quality_metrics', {})
            if metrics:
                print("  Metrics:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"    {metric}: {value:.4f}")
                    else:
                        print(f"    {metric}: {value}")

            # OOD detection
            ood = eval_record.get('ood_detection')
            if ood:
                print("  OOD Detection:")
                print(f"    In-distribution: {ood.get('in_distribution_count', 0)}/{ood.get('total_images', 0)} ({ood.get('in_distribution_percentage', 0):.1f}%)")
                print(f"    Mean score: {ood.get('ood_score_mean', 0):.3f} ± {ood.get('ood_score_std', 0):.3f}")

    # Show OOD images if requested
    if show_ood_images or show_all_ood:
        print()
        print("OOD IMAGE DETAILS")
        print("-" * 70)

        # Get latest evaluation's OOD data
        if evaluations:
            latest_eval = evaluations[-1]
            ood_data = latest_eval.get('ood_detection')

            if ood_data and 'per_image_scores' in ood_data:
                per_image = ood_data['per_image_scores']

                if show_all_ood:
                    # Show all images with scores
                    print(f"All {len(per_image)} images with OOD scores:")
                    for img in per_image:
                        status = "✓ IN-DIST" if img['is_in_distribution'] else "✗ OOD"
                        print(f"  {status} | {img['ood_score']:.3f} | {img['image_name']}")
                else:
                    # Show only OOD images
                    ood_images = [img for img in per_image if img['is_ood']]
                    print(f"Found {len(ood_images)} OOD images:")
                    for img in ood_images:
                        print(f"  ✗ {img['ood_score']:.3f} | {img['image_name']}")
                        print(f"     Path: {img['image_path']}")
            else:
                print("No per-image OOD data available")
                print("Tip: Run evaluation with OOD enabled (set FORTE_ENABLED=true)")

    print()
    print("=" * 70)
    print(f"Quality metadata file: {quality_file}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="View quality evaluation metrics")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./augmented-output",
        help="Output directory containing quality_metadata.json"
    )
    parser.add_argument(
        "--show-ood-images",
        action="store_true",
        help="Show list of OOD (out-of-distribution) images"
    )
    parser.add_argument(
        "--show-all-ood",
        action="store_true",
        help="Show all images with their OOD scores"
    )

    args = parser.parse_args()
    view_quality_metadata(args.output_dir, args.show_ood_images, args.show_all_ood)


if __name__ == "__main__":
    main()

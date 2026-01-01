#!/usr/bin/env python
"""
Export OOD images to a separate file or directory.

Usage:
    # List OOD images to file
    python export_ood_images.py --output ood_images.txt

    # Copy OOD images to separate directory
    python export_ood_images.py --copy-to ./ood-images/

    # Export in-distribution images instead
    python export_ood_images.py --in-dist --copy-to ./in-dist-images/
"""

import json
import shutil
import argparse
from pathlib import Path


def export_ood_images(
    output_dir: str,
    output_file: str = None,
    copy_to: str = None,
    in_dist: bool = False
):
    """Export OOD or in-distribution images."""

    quality_file = Path(output_dir) / "quality_metadata.json"

    if not quality_file.exists():
        print(f"❌ Quality metadata not found: {quality_file}")
        print("   Run evaluation with OOD enabled first")
        return

    with open(quality_file, 'r') as f:
        data = json.load(f)

    # Get latest evaluation
    evaluations = data.get('evaluations', [])
    if not evaluations:
        print("❌ No evaluations found")
        return

    latest_eval = evaluations[-1]
    ood_data = latest_eval.get('ood_detection')

    if not ood_data or 'per_image_scores' not in ood_data:
        print("❌ No per-image OOD data found")
        print("   Run evaluation with FORTE_ENABLED=true")
        return

    per_image = ood_data['per_image_scores']

    # Filter based on distribution
    if in_dist:
        filtered = [img for img in per_image if img['is_in_distribution']]
        filter_name = "in-distribution"
    else:
        filtered = [img for img in per_image if img['is_ood']]
        filter_name = "OOD"

    print(f"Found {len(filtered)} {filter_name} images (out of {len(per_image)} total)")

    # Export to file
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            for img in filtered:
                f.write(f"{img['image_path']}\t{img['ood_score']:.3f}\n")
        print(f"✓ Saved {filter_name} image list to: {output_path}")

    # Copy to directory
    if copy_to:
        dest_dir = Path(copy_to)
        dest_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for img in filtered:
            src = Path(img['image_path'])
            if src.exists():
                dest = dest_dir / img['image_name']
                shutil.copy2(src, dest)
                copied += 1

        print(f"✓ Copied {copied} {filter_name} images to: {dest_dir}")

        # Also save a metadata file
        metadata = {
            'filter': filter_name,
            'threshold': ood_data['threshold'],
            'total_images': len(filtered),
            'images': filtered
        }

        metadata_file = dest_dir / f"{filter_name.lower().replace('-', '_')}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Saved metadata to: {metadata_file}")

    # Print summary
    print(f"\n{filter_name.upper()} SUMMARY:")
    print(f"  Total: {len(filtered)}")
    print(f"  Threshold: {ood_data['threshold']}")
    if filtered:
        scores = [img['ood_score'] for img in filtered]
        print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Export OOD images")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./augmented-output",
        help="Output directory containing quality_metadata.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save image list"
    )
    parser.add_argument(
        "--copy-to",
        type=str,
        help="Directory to copy images to"
    )
    parser.add_argument(
        "--in-dist",
        action="store_true",
        help="Export in-distribution images instead of OOD"
    )

    args = parser.parse_args()

    if not args.output and not args.copy_to:
        print("❌ Please specify --output or --copy-to (or both)")
        parser.print_help()
        return

    export_ood_images(args.output_dir, args.output, args.copy_to, args.in_dist)


if __name__ == "__main__":
    main()

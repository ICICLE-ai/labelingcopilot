"""
Simple runner script for data augmentation.
Usage: python run_augmentation.py [--resume]
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config.config import get_config
from utils.image_utils import ImageLoader
from core.generator_parallel import ParallelVariantGenerator
from core.evaluator import QualityEvaluator
from core.metadata import MetadataManager
from core.suggestion_cache import SuggestionCache
from core.progress_tracker import ProgressTracker
from utils.image_discovery import ImageDiscovery


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Data Augmentation Tool with Resume Support")
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous run (skip completed images, use cached suggestions)',
    )
    parser.add_argument('--input-dir', default='./input-images', help='Input directory (default: ./input-images)')
    parser.add_argument('--output-dir', default='./augmented-output', help='Output directory (default: ./augmented-output)')
    parser.add_argument('--domain', default='general computer vision imagery', help='Domain description')
    parser.add_argument('--num-variants', type=int, default=3, help='Number of variants per image (default: 3)')
    return parser


def run_augmentation(
    input_dir: str,
    output_dir: str,
    domain: str,
    num_variants_per_image: int,
    resume_mode: bool = False,
):
    """Run the augmentation pipeline and return a summary."""

    print("=" * 60)
    print("Data Augmentation Tool" + (" [RESUME MODE]" if resume_mode else ""))
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Domain: {domain}")
    print(f"Variants per image: {num_variants_per_image}")
    print("=" * 60)
    print()

    # Initialize
    config = get_config()
    print(f"Model API provider: {config.azure.provider}")
    generator = ParallelVariantGenerator(config)
    evaluator = QualityEvaluator(config)
    metadata_manager = MetadataManager(output_dir)

    # Initialize resume components
    suggestion_cache = SuggestionCache(
        cache_file=f"{output_dir}/suggestions.json",
        domain=domain
    )
    progress_tracker = ProgressTracker(
        progress_file=f"{output_dir}/progress.json"
    )

    # Load existing caches if in resume mode
    if resume_mode:
        print("\n[Resume] Loading existing caches...")
        suggestion_cache.load()
        progress_tracker.load()

        # Show resume stats
        cache_stats = suggestion_cache.get_stats()
        progress_stats = progress_tracker.get_stats()

        print(f"[Resume] Cached suggestions for {cache_stats['total_images']} images")
        print(f"[Resume] Progress: {progress_stats['completed']} completed, {progress_stats['failed']} failed, {progress_stats['pending']} pending")

        # Discover existing generated images
        print("[Resume] Discovering existing generated images...")
        discovery = ImageDiscovery(output_dir)
        discovered = discovery.discover(input_dir)
        if discovered:
            summary = discovery.summarize(input_dir)
            print(f"[Resume] Found {summary['total_generated_images']} existing images from {summary['total_source_images']} sources")
        print()

    # Load images
    print("Loading images from input directory...")
    loader = ImageLoader(input_dir)
    image_data = loader.load_images(min_size=config.processing.min_file_size)
    image_paths = [img['path'] for img in image_data]

    print(f"Found {len(image_paths)} images")
    print()

    if len(image_paths) == 0:
        print(f"ERROR: No images found in {input_dir}")
        print("Please ensure the folder exists and contains .jpg or .png images")
        return {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'domain': domain,
            'num_variants_per_image': num_variants_per_image,
            'original_images': 0,
            'generated_images': 0,
            'generated_paths': [],
            'quality_scores': None,
            'metadata_file': f"{output_dir}/metadata.json",
        }

    # Generate variants
    print("Generating variants...")
    print(
        f"Note: Rate limited to {config.processing.requests_per_minute} requests/minute "
        f"with {config.processing.max_concurrent} concurrent workers"
    )
    print(f"Metadata will be saved periodically for safety")
    print()

    results = generator.gen_variants_batch_parallel(
        image_paths=image_paths,
        num_variants_per_image=num_variants_per_image,
        domain=domain,
        output_dir=output_dir,
        metadata_manager=metadata_manager,  # Save metadata periodically
        suggestion_cache=suggestion_cache,  # Use cached suggestions
        progress_tracker=progress_tracker,  # Track progress for resume
        resume_mode=resume_mode  # Skip completed images
    )

    generated_paths = [v['output_path'] for v in results]

    print()
    print(f"Generated {len(generated_paths)} augmented images")
    print()

    # Optional: Evaluate quality if enabled
    if config.evaluation.compute_prdc or config.evaluation.compute_fd:
        print("Evaluating quality...")

        try:
            quality_scores = evaluator.eval_quality(
                real_image_paths=image_paths,
                generated_image_paths=generated_paths,
                metrics=None  # Use config defaults
            )

            print()
            print("Quality Scores:")
            print("-" * 40)
            for metric, value in quality_scores.items():
                if value is not None:
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
            print()

        except Exception as e:
            print(f"Warning: Quality evaluation failed: {e}")
            print("Continuing without quality scores...")
            quality_scores = None
    else:
        quality_scores = None

    # Save metadata
    print("Saving metadata...")
    metadata_manager.save(
        input_dir=input_dir,
        domain=domain,
        config_snapshot={
            'domain': domain,
            'num_variants_per_image': num_variants_per_image,
            'requests_per_minute': config.processing.requests_per_minute
        },
        session_id=progress_tracker.session_id if progress_tracker else None
    )

    print()
    print("=" * 60)
    print("Augmentation Complete!")
    print("=" * 60)
    print(f"Original images: {len(image_paths)}")
    print(f"Generated images: {len(generated_paths)}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata file: {output_dir}/metadata.json")

    if progress_tracker:
        stats = progress_tracker.get_stats()
        print(f"\nProgress Summary:")
        print(f"  Session ID: {stats['session_id']}")
        print(f"  Completed: {stats['completed']}/{stats['total_source_images']} source images")
        print(f"  Failed: {stats['failed']}")
        print(f"  Completion: {stats['completion_percentage']:.1f}%")

    if suggestion_cache:
        cache_stats = suggestion_cache.get_stats()
        print(f"\nSuggestion Cache:")
        print(f"  Cached images: {cache_stats['total_images']}")
        print(f"  Total suggestions: {cache_stats['total_suggestions']}")
        print(f"  Cache file: {output_dir}/suggestions.json")

    if resume_mode:
        print(f"\nResume files saved:")
        print(f"  - {output_dir}/progress.json")
        print(f"  - {output_dir}/suggestions.json")
        print(f"\nTo resume this run: python run_augmentation.py --resume")

    print("=" * 60)

    return {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'domain': domain,
        'num_variants_per_image': num_variants_per_image,
        'original_images': len(image_paths),
        'generated_images': len(generated_paths),
        'generated_paths': generated_paths,
        'quality_scores': quality_scores,
        'metadata_file': f"{output_dir}/metadata.json",
        'progress_file': f"{output_dir}/progress.json",
        'suggestions_file': f"{output_dir}/suggestions.json",
    }


def main():
    """Run augmentation pipeline on the configured input directory."""
    args = build_parser().parse_args()
    run_augmentation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        domain=args.domain,
        num_variants_per_image=args.num_variants,
        resume_mode=args.resume,
    )


if __name__ == "__main__":
    main()

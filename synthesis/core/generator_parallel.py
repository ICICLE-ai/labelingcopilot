"""
Parallel image variant generation - processes images concurrently.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Lock

from config.config import Config
from utils.api_client import AzureVisionClient, AzureImageEditClient, AzureClientFactory
from utils.image_utils import image_to_data_url, ImageWriter
from utils.batch_processor import RateLimiter
from core.metadata import ImageMetadata
from core.suggestion_cache import SuggestionCache
from core.progress_tracker import ProgressTracker


class ParallelVariantGenerator:
    """Generates image variants in parallel with rate limiting."""

    def __init__(self, config: Config):
        """Initialize parallel generator."""
        self.config = config

        # Create API clients
        self.vision_client = AzureClientFactory.create_vision_client(
            config.azure,
            config.processing
        )
        self.edit_client = AzureClientFactory.create_image_edit_client(
            config.azure,
            config.processing
        )

        # Rate limiter for image edits (20 req/min)
        self.rate_limiter = RateLimiter(config.processing.requests_per_minute)

        # Lock for thread-safe operations
        self.lock = Lock()

    def _process_single_variant(
        self,
        image_path: str,
        data_url: str,
        prompt: str,
        category: str,
        variant_idx: int,
        output_dir: str
    ) -> Dict[str, Any]:
        """Process a single variant (called in parallel)."""

        # Wait for rate limit
        self.rate_limiter.wait()

        try:
            print(f"[Parallel] Editing {Path(image_path).name} variant {variant_idx + 1}: {category[:20]}...")

            # Generate variant
            edited_bytes = self.edit_client.edit_image(data_url, prompt)

            # Save output
            writer = ImageWriter(output_dir)
            source_name = Path(image_path).stem
            output_filename = f"{source_name}_{category}_{variant_idx}"
            output_path = writer.save_from_bytes(edited_bytes, output_filename)

            print(f"[Parallel] ✓ Saved: {Path(output_path).name}")

            # Create metadata
            metadata = ImageMetadata(
                source_image=image_path,
                output_image=output_path,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                operation="gen_variant",
                prompt=prompt,
                category=category
            )

            return {
                'output_path': output_path,
                'prompt': prompt,
                'category': category,
                'metadata': metadata,
                'variant_idx': variant_idx,
                'success': True
            }

        except Exception as e:
            error_msg = str(e)
            if 'moderation_blocked' in error_msg or 'safety system' in error_msg:
                print(f"[Parallel] ✗ Moderation blocked (skipping this variant, others continue): {Path(image_path).name}")
            else:
                print(f"[Parallel] ✗ Error: {e}")
            return {
                'success': False,
                'error': str(e),
                'source_image': image_path,
                'category': category,
                'variant_idx': variant_idx
            }

    def gen_variants_batch_parallel(
        self,
        image_paths: List[str],
        num_variants_per_image: int = 3,
        domain: str = "general object detection",
        output_dir: str = "./output",
        metadata_manager = None,
        suggestion_cache: Optional[SuggestionCache] = None,
        progress_tracker: Optional[ProgressTracker] = None,
        resume_mode: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate variants for multiple images in parallel.

        Args:
            image_paths: List of image paths
            num_variants_per_image: Number of variants per image
            domain: Domain description
            output_dir: Output directory
            metadata_manager: Optional metadata manager

        Returns:
            List of result dictionaries
        """
        print(f"\n[Parallel] Starting parallel generation for {len(image_paths)} images")
        print(f"[Parallel] Using {self.config.processing.max_concurrent} workers")

        # Resume mode: filter out completed images
        if resume_mode and progress_tracker:
            print(f"[Parallel] Resume mode enabled - checking for completed images...")
            original_count = len(image_paths)

            # Initialize progress tracker for all images
            for image_path in image_paths:
                progress_tracker.init_image(image_path, num_variants_per_image)

            # Filter out completed images
            images_to_process = [
                path for path in image_paths
                if not progress_tracker.is_completed(path)
            ]

            skipped = original_count - len(images_to_process)
            if skipped > 0:
                print(f"[Parallel] Skipping {skipped} completed images")
                print(f"[Parallel] Processing {len(images_to_process)} remaining images")

            # Update image_paths to only process incomplete ones
            image_paths = images_to_process

            # Show retry info if any failed images
            failed_images = progress_tracker.get_failed_images()
            if failed_images:
                retry_count = len([p for p in image_paths if p in failed_images])
                print(f"[Parallel] Retrying {retry_count} previously failed images")

        if not image_paths:
            print("[Parallel] No images require processing")
            if progress_tracker:
                progress_tracker.save()
            if suggestion_cache:
                suggestion_cache.save()
            return []

        all_results = []
        total_tasks = len(image_paths) * num_variants_per_image
        completed = 0

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.config.processing.max_concurrent) as executor:
            futures = []
            completed = 0
            start_time = time.time()

            # Truly interleaved approach: collect results while gathering suggestions
            print(f"\n[Parallel] Starting interleaved suggestion gathering and image generation...")

            # Process in batches to speed up suggestion gathering
            batch_size = 3  # Keep batches small to avoid overwhelming the API
            num_batches = (len(image_paths) + batch_size - 1) // batch_size

            # Create initial empty metadata file
            if metadata_manager:
                print(f"[Parallel] Creating initial metadata file...")
                metadata_manager.save(
                    input_dir=str(Path(image_paths[0]).parent),
                    domain=domain,
                    config_snapshot={'domain': domain, 'num_variants_per_image': num_variants_per_image},
                    append_mode=False
                )
                print(f"[Parallel] Metadata file created at {metadata_manager.metadata_file}")

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(image_paths))
                batch_paths = image_paths[start_idx:end_idx]

                print(f"\n[Parallel] === Batch {batch_idx + 1}/{num_batches} ===")
                print(f"[Parallel] Analyzing {len(batch_paths)} images for suggestions...")

                # Convert batch to data URLs
                batch_data_urls = []
                for image_path in batch_paths:
                    data_url = image_to_data_url(image_path)
                    batch_data_urls.append(data_url)

                # Get suggestions: check cache first, then fetch if needed
                batch_needs_fetch = []
                suggestions_dict = {}

                for i, image_path in enumerate(batch_paths):
                    # Check cache
                    if suggestion_cache and suggestion_cache.has(image_path):
                        cached = suggestion_cache.get(image_path)
                        suggestions_dict[f'image_{i}'] = cached
                        print(f"[Parallel]   {Path(image_path).name}: Using cached suggestions")
                    else:
                        batch_needs_fetch.append((i, image_path))

                # Fetch suggestions for uncached images
                if batch_needs_fetch:
                    # Build subset for API call
                    fetch_data_urls = [batch_data_urls[i] for i, _ in batch_needs_fetch]
                    fresh_suggestions = self.vision_client.get_augmentation_suggestions(
                        fetch_data_urls,
                        domain=domain
                    )

                    # Merge fresh suggestions and cache them
                    for fetch_idx, (batch_idx, image_path) in enumerate(batch_needs_fetch):
                        fetch_key = f'image_{fetch_idx}'
                        if fetch_key in fresh_suggestions:
                            suggestions_dict[f'image_{batch_idx}'] = fresh_suggestions[fetch_key]

                            # Cache the suggestions
                            if suggestion_cache:
                                suggestion_cache.add(image_path, fresh_suggestions[fetch_key])
                                print(f"[Parallel]   {Path(image_path).name}: Cached new suggestions")

                # Save suggestion cache after each batch
                if suggestion_cache and batch_needs_fetch:
                    suggestion_cache.save()

                suggestions = suggestions_dict

                # Immediately submit edit tasks for this batch
                batch_tasks = 0
                for i, image_path in enumerate(batch_paths):
                    image_key = f'image_{i}'
                    image_suggestions = suggestions.get(image_key, [])[:num_variants_per_image]

                    print(f"[Parallel]   {Path(image_path).name}: {len(image_suggestions)} suggestions → submitting edit tasks")

                    # Submit all edit tasks for this image immediately
                    for j, suggestion in enumerate(image_suggestions):
                        prompt = suggestion['prompt']
                        category = suggestion.get('category', 'auto')

                        # Submit task
                        future = executor.submit(
                            self._process_single_variant,
                            image_path,
                            batch_data_urls[i],
                            prompt,
                            category,
                            j,
                            output_dir
                        )
                        futures.append(future)
                        batch_tasks += 1

                print(f"[Parallel] Submitted {batch_tasks} edit tasks from batch {batch_idx + 1}")

                # Collect any completed results while we continue
                done_futures = []
                for future in futures:
                    if future.done():
                        done_futures.append(future)

                for future in done_futures:
                    futures.remove(future)
                    result = future.result()

                    if result.get('success'):
                        all_results.append(result)

                        # Update metadata incrementally
                        if metadata_manager:
                            metadata_manager.add_image(result['metadata'])

                        # Update progress tracker
                        if progress_tracker:
                            metadata = result['metadata']
                            variant_key = f"{metadata.category}_{result.get('variant_idx', 0)}"
                            progress_tracker.mark_variant_completed(
                                metadata.source_image,
                                variant_key,
                                metadata.output_image
                            )
                    else:
                        # Mark as failed in progress tracker
                        if progress_tracker and 'source_image' in result:
                            error_msg = result.get('error', 'Unknown error')
                            variant_key = f"{result.get('category', 'unknown')}_{result.get('variant_idx', 0)}"
                            progress_tracker.mark_variant_failed(
                                result['source_image'],
                                variant_key,
                                error_msg
                            )

                    completed += 1

                    # Save every 5 images
                    if metadata_manager and completed % 5 == 0:
                        print(f"\n[Parallel] Saving metadata checkpoint ({completed} images)...")
                        metadata_manager.save(
                            input_dir=str(Path(image_paths[0]).parent),
                            domain=domain,
                            config_snapshot={'domain': domain, 'num_variants_per_image': num_variants_per_image},
                            append_mode=True
                        )

                        # Save progress tracker
                        if progress_tracker:
                            progress_tracker.save()

                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 and completed > 0 else 0
                print(f"[Parallel] Progress: {completed}/{total_tasks} completed ({len(futures)} still running)")

            # Collect remaining results
            print(f"\n[Parallel] Collecting remaining {len(futures)} results...")
            for future in as_completed(futures):
                result = future.result()

                if result.get('success'):
                    all_results.append(result)

                    # Update metadata incrementally
                    if metadata_manager:
                        metadata_manager.add_image(result['metadata'])

                    # Update progress tracker
                    if progress_tracker:
                        metadata = result['metadata']
                        variant_key = f"{metadata.category}_{result.get('variant_idx', 0)}"
                        progress_tracker.mark_variant_completed(
                            metadata.source_image,
                            variant_key,
                            metadata.output_image
                        )
                else:
                    # Mark as failed in progress tracker
                    if progress_tracker and 'source_image' in result:
                        error_msg = result.get('error', 'Unknown error')
                        variant_key = f"{result.get('category', 'unknown')}_{result.get('variant_idx', 0)}"
                        progress_tracker.mark_variant_failed(
                            result['source_image'],
                            variant_key,
                            error_msg
                        )

                completed += 1

                # Save every 5 images
                if metadata_manager and completed % 5 == 0:
                    print(f"\n[Parallel] Saving metadata checkpoint ({completed} images)...")
                    metadata_manager.save(
                        input_dir=str(Path(image_paths[0]).parent),
                        domain=domain,
                        config_snapshot={'domain': domain, 'num_variants_per_image': num_variants_per_image},
                        append_mode=True
                    )

                    # Save progress tracker
                    if progress_tracker:
                        progress_tracker.save()

                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_tasks - completed) / rate if rate > 0 else 0

                print(f"\r[Parallel] Progress: {completed}/{total_tasks} ({100*completed/total_tasks:.1f}%) | "
                      f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s", end="", flush=True)

            print()  # New line after progress

        # Final metadata save
        if metadata_manager:
            print(f"\n[Parallel] Saving final metadata...")
            metadata_manager.save(
                input_dir=str(Path(image_paths[0]).parent),
                domain=domain,
                config_snapshot={'domain': domain, 'num_variants_per_image': num_variants_per_image},
                append_mode=False  # Full save at the end
            )

        # Final progress tracker save
        if progress_tracker:
            print(f"[Parallel] Saving final progress tracker...")
            progress_tracker.save()
            stats = progress_tracker.get_stats()
            print(f"[Parallel] Progress stats: {stats['completed']}/{stats['total_source_images']} images completed")

        # Final suggestion cache save
        if suggestion_cache:
            suggestion_cache.save()

        elapsed_total = time.time() - start_time
        print(f"\n[Parallel] Complete! Generated {len(all_results)} variants in {elapsed_total:.1f}s")
        print(f"[Parallel] Average rate: {len(all_results)/elapsed_total:.2f} images/second")

        return all_results

"""
Image Discovery Utility

Discovers existing generated images by scanning the output directory
and parsing filenames to determine source image and variant information.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


class ImageDiscovery:
    """
    Discovers existing generated images in the output directory.

    Expected filename format: {source_stem}_{category}_{idx}.png
    Example: 1auzni2_24f8gkq24mjc1_environmental_0.png
    """

    def __init__(self, output_dir: str):
        """
        Initialize image discovery.

        Args:
            output_dir: Path to output directory
        """
        self.output_dir = output_dir

    def discover(self, input_dir: str = None) -> Dict[str, Dict]:
        """
        Discover existing generated images.

        Args:
            input_dir: Path to input directory (to resolve source images)

        Returns:
            Dictionary mapping source image paths to variant information:
            {
                "source_image_path": {
                    "variants": ["environmental_0", "camera_1", ...],
                    "variant_paths": {
                        "environmental_0": "path/to/output.png",
                        "camera_1": "path/to/output2.png"
                    },
                    "count": 2
                }
            }
        """
        if not os.path.exists(self.output_dir):
            print(f"[ImageDiscovery] Output directory not found: {self.output_dir}")
            return {}

        # Find all PNG files in output directory
        output_path = Path(self.output_dir)
        generated_images = list(output_path.glob("*.png"))

        if not generated_images:
            print(f"[ImageDiscovery] No generated images found in {self.output_dir}")
            return {}

        print(f"[ImageDiscovery] Found {len(generated_images)} PNG files in {self.output_dir}")

        # Group by source image
        source_variants = defaultdict(lambda: {"variants": [], "variant_paths": {}, "count": 0})

        for img_path in generated_images:
            # Parse filename to extract source and variant info
            parsed = self._parse_filename(img_path.name)
            if not parsed:
                continue

            source_stem, category, idx = parsed

            # Try to resolve source image path
            source_path = self._resolve_source_path(source_stem, input_dir)
            if not source_path:
                # Use stem as fallback if we can't resolve
                source_path = f"unknown/{source_stem}"

            # Build variant key
            variant_key = f"{category}_{idx}"

            # Add to mapping
            source_variants[source_path]["variants"].append(variant_key)
            source_variants[source_path]["variant_paths"][variant_key] = str(img_path)
            source_variants[source_path]["count"] += 1

        # Convert defaultdict to regular dict
        result = dict(source_variants)

        print(f"[ImageDiscovery] Mapped images to {len(result)} source images")
        for source, data in result.items():
            print(f"  {Path(source).name}: {data['count']} variants")

        return result

    def _parse_filename(self, filename: str) -> Optional[Tuple[str, str, int]]:
        """
        Parse generated image filename.

        Expected format: {source_stem}_{category}_{idx}.png
        Example: 1auzni2_24f8gkq24mjc1_environmental_0.png

        Args:
            filename: Filename to parse

        Returns:
            Tuple of (source_stem, category, idx) or None if parsing fails
        """
        # Remove .png extension
        name_without_ext = filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')

        # Try to match pattern: ends with _{category}_{idx}
        # Categories: environmental, camera, edge_case, object, scene, lighting, weather, etc.
        pattern = r'^(.+?)_(environmental|camera|edge_case|object|scene|lighting|weather|perspective)_(\d+)$'
        match = re.match(pattern, name_without_ext)

        if match:
            source_stem = match.group(1)
            category = match.group(2)
            idx = int(match.group(3))
            return (source_stem, category, idx)

        # If pattern doesn't match, might be a different naming convention or source image
        # Return None to skip
        return None

    def _resolve_source_path(self, source_stem: str, input_dir: str = None) -> Optional[str]:
        """
        Resolve source image path from stem.

        Args:
            source_stem: Source image stem (filename without extension)
            input_dir: Input directory to search in

        Returns:
            Full path to source image or None if not found
        """
        if not input_dir or not os.path.exists(input_dir):
            return None

        input_path = Path(input_dir)

        # Try common extensions
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            # Try exact match
            candidate = input_path / f"{source_stem}{ext}"
            if candidate.exists():
                return str(candidate)

            # Try matching just the base part (before any underscore)
            # e.g., 1auzni2_24f8gkq24mjc1 -> 1auzni2
            base = source_stem.split('_')[0]
            candidate = input_path / f"{base}{ext}"
            if candidate.exists():
                return str(candidate)

        # Search all files in input directory
        for img_file in input_path.glob("*"):
            if img_file.stem == source_stem or img_file.stem in source_stem or source_stem in img_file.stem:
                return str(img_file)

        return None

    def get_completed_sources(self, expected_variants_per_image: int = 2) -> Set[str]:
        """
        Get source images that have all expected variants.

        Args:
            expected_variants_per_image: Number of variants expected per source

        Returns:
            Set of source image paths that are fully completed
        """
        discovered = self.discover()
        completed = set()

        for source_path, data in discovered.items():
            if data['count'] >= expected_variants_per_image:
                completed.add(source_path)

        return completed

    def get_partial_sources(self, expected_variants_per_image: int = 2) -> Dict[str, Dict]:
        """
        Get source images that have partial completion.

        Args:
            expected_variants_per_image: Number of variants expected per source

        Returns:
            Dictionary of partially completed sources with their variant info
        """
        discovered = self.discover()
        partial = {}

        for source_path, data in discovered.items():
            if 0 < data['count'] < expected_variants_per_image:
                partial[source_path] = data

        return partial

    def get_missing_variants(self, source_path: str, expected_variants: List[str]) -> List[str]:
        """
        Get missing variants for a source image.

        Args:
            source_path: Path to source image
            expected_variants: List of expected variant keys (e.g., ["environmental_0", "camera_0"])

        Returns:
            List of missing variant keys
        """
        discovered = self.discover()
        if source_path not in discovered:
            return expected_variants

        existing_variants = set(discovered[source_path]['variants'])
        missing = [v for v in expected_variants if v not in existing_variants]
        return missing

    def summarize(self, input_dir: str = None) -> Dict:
        """
        Get summary statistics of discovered images.

        Args:
            input_dir: Input directory path

        Returns:
            Dictionary with summary statistics
        """
        discovered = self.discover(input_dir)

        total_sources = len(discovered)
        total_variants = sum(data['count'] for data in discovered.values())

        # Category breakdown
        category_counts = defaultdict(int)
        for data in discovered.values():
            for variant_key in data['variants']:
                category = variant_key.rsplit('_', 1)[0]  # Remove _idx suffix
                category_counts[category] += 1

        return {
            "total_source_images": total_sources,
            "total_generated_images": total_variants,
            "avg_variants_per_source": total_variants / total_sources if total_sources > 0 else 0,
            "category_breakdown": dict(category_counts),
            "output_directory": self.output_dir
        }

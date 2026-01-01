"""
Metadata management for augmented images.
Tracks generation parameters, quality scores, and reasoning strategy results.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ImageMetadata:
    """Metadata for a single augmented image."""

    # Source information
    source_image: str
    output_image: str
    timestamp: str

    # Generation parameters
    operation: str  # "gen_variant" or "edit_image"
    prompt: str
    condition: Optional[str] = None

    # Quality scores
    scores: Optional[Dict[str, float]] = None

    # OOD detection
    ood_score: Optional[float] = None
    is_in_distribution: Optional[bool] = None

    # Reasoning strategy results
    rarity_score: Optional[float] = None
    realism_score: Optional[float] = None
    fidelity_score: Optional[float] = None

    # Additional metadata
    category: Optional[str] = None
    notes: Optional[str] = None

    # Resume tracking
    processing_status: Optional[str] = None  # pending, in_progress, completed, failed
    retry_count: Optional[int] = None
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageMetadata":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DatasetMetadata:
    """Metadata for entire augmented dataset."""

    # Dataset information
    input_directory: str
    output_directory: str
    created_at: str
    domain: str

    # Processing statistics
    total_original_images: int
    total_augmented_images: int
    total_failed: int

    # Configuration snapshot
    config: Dict[str, Any]

    # Image metadata
    images: List[ImageMetadata]

    # Aggregate quality scores
    aggregate_scores: Optional[Dict[str, float]] = None

    # Reasoning strategy summary
    rarity_stats: Optional[Dict[str, float]] = None
    realism_stats: Optional[Dict[str, float]] = None
    fidelity_stats: Optional[Dict[str, float]] = None

    # Resume tracking
    session_id: Optional[str] = None
    resumed_from: Optional[str] = None  # Session ID of previous run if resumed
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert ImageMetadata objects to dicts
        data['images'] = [img.to_dict() if isinstance(img, ImageMetadata) else img for img in self.images]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        """Create from dictionary."""
        # Convert dicts to ImageMetadata objects
        if 'images' in data:
            data['images'] = [
                ImageMetadata.from_dict(img) if isinstance(img, dict) else img
                for img in data['images']
            ]
        return cls(**data)


class MetadataManager:
    """Manages metadata for augmentation operations."""

    def __init__(self, output_dir: str):
        """
        Initialize metadata manager.

        Args:
            output_dir: Directory where metadata will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.output_dir / "metadata.json"
        self.images: List[ImageMetadata] = []

    def add_image(self, metadata: ImageMetadata):
        """
        Add metadata for a single image.

        Args:
            metadata: Image metadata to add
        """
        self.images.append(metadata)

    def create_image_metadata(
        self,
        source_image: str,
        output_image: str,
        operation: str,
        prompt: str,
        condition: Optional[str] = None,
        scores: Optional[Dict[str, float]] = None,
        ood_score: Optional[float] = None,
        category: Optional[str] = None
    ) -> ImageMetadata:
        """
        Create and add metadata for an image.

        Args:
            source_image: Path to source image
            output_image: Path to output image
            operation: Operation type
            prompt: Generation/edit prompt
            condition: Optional condition
            scores: Optional quality scores
            ood_score: Optional OOD detection score
            category: Optional category

        Returns:
            Created ImageMetadata
        """
        metadata = ImageMetadata(
            source_image=source_image,
            output_image=output_image,
            timestamp=datetime.now().isoformat(),
            operation=operation,
            prompt=prompt,
            condition=condition,
            scores=scores,
            ood_score=ood_score,
            is_in_distribution=ood_score > 0.3 if ood_score is not None else None,
            category=category
        )

        self.add_image(metadata)
        return metadata

    def save(
        self,
        input_dir: str,
        domain: str = "general",
        config_snapshot: Optional[Dict[str, Any]] = None,
        append_mode: bool = False,
        session_id: Optional[str] = None,
        resumed_from: Optional[str] = None
    ):
        """
        Save metadata to JSON file.

        Args:
            input_dir: Input directory path
            domain: Domain description
            config_snapshot: Configuration snapshot
            append_mode: If True, only update existing file instead of full rewrite
        """
        # In append mode, do a quick update
        if append_mode and self.metadata_file.exists():
            # Just update the file with current images list
            try:
                existing = self.load()
                if existing:
                    # Keep existing structure, just update images list
                    existing.images = self.images
                    existing.total_augmented_images = len(self.images)
                    existing.total_original_images = len(set(img.source_image for img in self.images))
                    existing.last_updated = datetime.now().isoformat()

                    # Update session info if provided
                    if session_id:
                        existing.session_id = session_id
                    if resumed_from:
                        existing.resumed_from = resumed_from

                    with open(self.metadata_file, 'w') as f:
                        json.dump(existing.to_dict(), f, indent=2)
                    return
            except:
                pass  # Fall through to full save

        # Calculate statistics
        total_original = len(set(img.source_image for img in self.images))
        total_augmented = len(self.images)
        total_failed = 0  # Would need to track failures separately

        # Calculate aggregate scores
        aggregate_scores = self._calculate_aggregate_scores()

        # Calculate reasoning strategy stats
        rarity_stats = self._calculate_reasoning_stats('rarity_score')
        realism_stats = self._calculate_reasoning_stats('realism_score')
        fidelity_stats = self._calculate_reasoning_stats('fidelity_score')

        # Create dataset metadata
        now = datetime.now().isoformat()
        dataset_metadata = DatasetMetadata(
            input_directory=input_dir,
            output_directory=str(self.output_dir),
            created_at=now,
            domain=domain,
            total_original_images=total_original,
            total_augmented_images=total_augmented,
            total_failed=total_failed,
            config=config_snapshot or {},
            images=self.images,
            aggregate_scores=aggregate_scores,
            rarity_stats=rarity_stats,
            realism_stats=realism_stats,
            fidelity_stats=fidelity_stats,
            session_id=session_id,
            resumed_from=resumed_from,
            last_updated=now
        )

        # Save to file
        with open(self.metadata_file, 'w') as f:
            json.dump(dataset_metadata.to_dict(), f, indent=2)

        print(f"Metadata saved to {self.metadata_file}")

    def load(self) -> Optional[DatasetMetadata]:
        """
        Load metadata from JSON file.

        Returns:
            DatasetMetadata if file exists, None otherwise
        """
        if not self.metadata_file.exists():
            return None

        with open(self.metadata_file, 'r') as f:
            data = json.load(f)

        return DatasetMetadata.from_dict(data)

    def _calculate_aggregate_scores(self) -> Dict[str, float]:
        """Calculate aggregate quality scores across all images."""
        if not self.images:
            return {}

        # Collect all score types
        all_score_keys = set()
        for img in self.images:
            if img.scores:
                all_score_keys.update(img.scores.keys())

        # Calculate averages
        aggregates = {}
        for key in all_score_keys:
            scores = [img.scores[key] for img in self.images if img.scores and key in img.scores]
            if scores:
                aggregates[f"{key}_mean"] = sum(scores) / len(scores)
                aggregates[f"{key}_min"] = min(scores)
                aggregates[f"{key}_max"] = max(scores)

        return aggregates

    def _calculate_reasoning_stats(self, score_field: str) -> Dict[str, float]:
        """Calculate statistics for a reasoning strategy score."""
        scores = [getattr(img, score_field) for img in self.images if getattr(img, score_field) is not None]

        if not scores:
            return {}

        return {
            'mean': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores),
            'count': len(scores)
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Summary dictionary
        """
        return {
            'total_images': len(self.images),
            'unique_sources': len(set(img.source_image for img in self.images)),
            'operations': {
                op: sum(1 for img in self.images if img.operation == op)
                for op in set(img.operation for img in self.images)
            },
            'categories': {
                cat: sum(1 for img in self.images if img.category == cat)
                for cat in set(img.category for img in self.images if img.category)
            }
        }

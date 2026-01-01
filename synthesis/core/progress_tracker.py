"""
Progress Tracking for Resume Functionality

Tracks processing state of each source image and its variants to enable
resuming interrupted augmentation runs.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
from enum import Enum


class ProcessingStatus(Enum):
    """Processing status for images."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProgressTracker:
    """
    Tracks processing progress for augmentation runs.

    Storage format:
    {
        "session_id": "uuid",
        "started_at": "ISO timestamp",
        "last_updated": "ISO timestamp",
        "total_source_images": 436,
        "images": {
            "source_image_path": {
                "status": "completed|failed|in_progress|pending",
                "expected_variants": 3,
                "completed_variants": 3,
                "failed_variants": 0,
                "variants": {
                    "environmental_0": {"status": "completed", "path": "out.png", "timestamp": "..."},
                    "camera_0": {"status": "failed", "error": "...", "timestamp": "..."}
                },
                "retry_count": 0,
                "last_error": null,
                "last_attempt": "ISO timestamp"
            }
        }
    }
    """

    def __init__(self, progress_file: str = None, session_id: str = None):
        """
        Initialize progress tracker.

        Args:
            progress_file: Path to progress file (default: augmented-output/progress.json)
            session_id: Session ID (generates new UUID if None)
        """
        self.progress_file = progress_file or "augmented-output/progress.json"
        self.session_id = session_id or str(uuid.uuid4())
        self.started_at = None
        self.last_updated = None
        self.images = {}

    def load(self) -> bool:
        """
        Load progress from file.

        Returns:
            True if loaded successfully, False if file doesn't exist or is corrupted
        """
        if not os.path.exists(self.progress_file):
            print(f"[ProgressTracker] No progress file found at {self.progress_file}")
            return False

        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)

            self.session_id = data.get('session_id', self.session_id)
            self.started_at = data.get('started_at')
            self.last_updated = data.get('last_updated')
            self.images = data.get('images', {})

            completed = sum(1 for img in self.images.values() if img['status'] == ProcessingStatus.COMPLETED.value)
            failed = sum(1 for img in self.images.values() if img['status'] == ProcessingStatus.FAILED.value)
            print(f"[ProgressTracker] Loaded progress: {completed} completed, {failed} failed")
            return True

        except (json.JSONDecodeError, IOError) as e:
            print(f"[ProgressTracker] Warning: Failed to load progress: {e}")
            return False

    def save(self):
        """Save progress to file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)

        now = datetime.now().isoformat()
        if self.started_at is None:
            self.started_at = now
        self.last_updated = now

        # Compute stats
        total = len(self.images)
        completed = sum(1 for img in self.images.values() if img['status'] == ProcessingStatus.COMPLETED.value)
        failed = sum(1 for img in self.images.values() if img['status'] == ProcessingStatus.FAILED.value)
        pending = sum(1 for img in self.images.values() if img['status'] == ProcessingStatus.PENDING.value)

        data = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "stats": {
                "total_source_images": total,
                "completed": completed,
                "failed": failed,
                "pending": pending,
                "completion_percentage": (completed / total * 100) if total > 0 else 0
            },
            "images": self.images
        }

        with open(self.progress_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_status(self, source_image_path: str) -> Optional[str]:
        """
        Get processing status for a source image.

        Args:
            source_image_path: Path to source image

        Returns:
            Status string or None if not tracked
        """
        normalized_path = str(Path(source_image_path))
        if normalized_path in self.images:
            return self.images[normalized_path]['status']
        return None

    def init_image(self, source_image_path: str, expected_variants: int):
        """
        Initialize tracking for a source image.

        Args:
            source_image_path: Path to source image
            expected_variants: Number of variants expected for this image
        """
        normalized_path = str(Path(source_image_path))
        if normalized_path not in self.images:
            self.images[normalized_path] = {
                "status": ProcessingStatus.PENDING.value,
                "expected_variants": expected_variants,
                "completed_variants": 0,
                "failed_variants": 0,
                "variants": {},
                "retry_count": 0,
                "last_error": None,
                "last_attempt": None
            }

    def mark_in_progress(self, source_image_path: str):
        """Mark image as in progress."""
        normalized_path = str(Path(source_image_path))
        if normalized_path in self.images:
            self.images[normalized_path]['status'] = ProcessingStatus.IN_PROGRESS.value
            self.images[normalized_path]['last_attempt'] = datetime.now().isoformat()

    def mark_variant_completed(self, source_image_path: str, variant_key: str, output_path: str):
        """
        Mark a variant as completed.

        Args:
            source_image_path: Path to source image
            variant_key: Variant identifier (e.g., "environmental_0")
            output_path: Path to generated image
        """
        normalized_path = str(Path(source_image_path))
        if normalized_path not in self.images:
            return

        img_data = self.images[normalized_path]
        img_data['variants'][variant_key] = {
            "status": ProcessingStatus.COMPLETED.value,
            "path": output_path,
            "timestamp": datetime.now().isoformat()
        }

        # Update counts
        img_data['completed_variants'] = sum(
            1 for v in img_data['variants'].values() if v['status'] == ProcessingStatus.COMPLETED.value
        )

        # Check if all variants completed
        if img_data['completed_variants'] >= img_data['expected_variants']:
            img_data['status'] = ProcessingStatus.COMPLETED.value

    def mark_variant_failed(self, source_image_path: str, variant_key: str, error: str):
        """
        Mark a variant as failed.

        Args:
            source_image_path: Path to source image
            variant_key: Variant identifier
            error: Error message
        """
        normalized_path = str(Path(source_image_path))
        if normalized_path not in self.images:
            return

        img_data = self.images[normalized_path]
        img_data['variants'][variant_key] = {
            "status": ProcessingStatus.FAILED.value,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }

        # Update counts
        img_data['failed_variants'] = sum(
            1 for v in img_data['variants'].values() if v['status'] == ProcessingStatus.FAILED.value
        )
        img_data['last_error'] = error
        img_data['status'] = ProcessingStatus.FAILED.value

    def mark_completed(self, source_image_path: str):
        """Mark entire source image as completed."""
        normalized_path = str(Path(source_image_path))
        if normalized_path in self.images:
            self.images[normalized_path]['status'] = ProcessingStatus.COMPLETED.value

    def mark_failed(self, source_image_path: str, error: str = None):
        """Mark source image as failed."""
        normalized_path = str(Path(source_image_path))
        if normalized_path in self.images:
            self.images[normalized_path]['status'] = ProcessingStatus.FAILED.value
            if error:
                self.images[normalized_path]['last_error'] = error
            self.images[normalized_path]['retry_count'] += 1

    def is_completed(self, source_image_path: str) -> bool:
        """Check if source image is fully completed."""
        normalized_path = str(Path(source_image_path))
        if normalized_path not in self.images:
            return False
        img_data = self.images[normalized_path]
        return (img_data['status'] == ProcessingStatus.COMPLETED.value and
                img_data['completed_variants'] >= img_data['expected_variants'])

    def is_failed(self, source_image_path: str) -> bool:
        """Check if source image has failed."""
        normalized_path = str(Path(source_image_path))
        if normalized_path not in self.images:
            return False
        return self.images[normalized_path]['status'] == ProcessingStatus.FAILED.value

    def get_pending_images(self) -> List[str]:
        """Get list of pending source image paths."""
        return [
            path for path, data in self.images.items()
            if data['status'] in [ProcessingStatus.PENDING.value, ProcessingStatus.IN_PROGRESS.value]
        ]

    def get_failed_images(self) -> List[str]:
        """Get list of failed source image paths."""
        return [
            path for path, data in self.images.items()
            if data['status'] == ProcessingStatus.FAILED.value
        ]

    def get_completed_images(self) -> List[str]:
        """Get list of completed source image paths."""
        return [
            path for path, data in self.images.items()
            if data['status'] == ProcessingStatus.COMPLETED.value
        ]

    def get_stats(self) -> Dict:
        """Get progress statistics."""
        total = len(self.images)
        completed = len(self.get_completed_images())
        failed = len(self.get_failed_images())
        pending = len(self.get_pending_images())

        total_variants = sum(img['expected_variants'] for img in self.images.values())
        completed_variants = sum(img['completed_variants'] for img in self.images.values())

        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "total_source_images": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "completion_percentage": (completed / total * 100) if total > 0 else 0,
            "total_variants_expected": total_variants,
            "total_variants_completed": completed_variants,
            "variant_completion_percentage": (completed_variants / total_variants * 100) if total_variants > 0 else 0
        }

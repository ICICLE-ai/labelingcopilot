"""
Suggestion Cache Management

Caches GPT-5 augmentation suggestions to enable resume functionality
and avoid redundant API calls.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class SuggestionCache:
    """
    Manages caching of GPT-5 augmentation suggestions.

    Storage format:
    {
        "created_at": "ISO timestamp",
        "last_updated": "ISO timestamp",
        "domain": "general computer vision imagery",
        "suggestions": {
            "source_image_path": [
                {"prompt": "...", "category": "environmental"},
                {"prompt": "...", "category": "camera"}
            ]
        }
    }
    """

    def __init__(self, cache_file: str = None, domain: str = None):
        """
        Initialize suggestion cache.

        Args:
            cache_file: Path to cache file (default: augmented-output/suggestions.json)
            domain: Domain description for metadata
        """
        self.cache_file = cache_file or "augmented-output/suggestions.json"
        self.domain = domain or "general computer vision imagery"
        self.suggestions = {}
        self.created_at = None
        self.last_updated = None

    def load(self) -> bool:
        """
        Load suggestions from cache file.

        Returns:
            True if loaded successfully, False if file doesn't exist or is corrupted
        """
        if not os.path.exists(self.cache_file):
            print(f"[SuggestionCache] No cache file found at {self.cache_file}")
            return False

        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)

            self.suggestions = data.get('suggestions', {})
            self.created_at = data.get('created_at')
            self.last_updated = data.get('last_updated')
            self.domain = data.get('domain', self.domain)

            print(f"[SuggestionCache] Loaded {len(self.suggestions)} cached suggestions")
            return True

        except (json.JSONDecodeError, IOError) as e:
            print(f"[SuggestionCache] Warning: Failed to load cache: {e}")
            return False

    def save(self):
        """Save suggestions to cache file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

        now = datetime.now().isoformat()
        if self.created_at is None:
            self.created_at = now
        self.last_updated = now

        data = {
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "domain": self.domain,
            "total_images": len(self.suggestions),
            "suggestions": self.suggestions
        }

        with open(self.cache_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[SuggestionCache] Saved {len(self.suggestions)} suggestions to {self.cache_file}")

    def get(self, source_image_path: str) -> Optional[List[Dict]]:
        """
        Get cached suggestions for a source image.

        Args:
            source_image_path: Path to source image

        Returns:
            List of suggestion dicts or None if not cached
        """
        # Normalize path for lookup
        normalized_path = str(Path(source_image_path))
        return self.suggestions.get(normalized_path)

    def add(self, source_image_path: str, suggestions: List[Dict]):
        """
        Add suggestions to cache for a source image.

        Args:
            source_image_path: Path to source image
            suggestions: List of suggestion dicts with 'prompt' and 'category'
        """
        normalized_path = str(Path(source_image_path))
        self.suggestions[normalized_path] = suggestions

    def has(self, source_image_path: str) -> bool:
        """
        Check if suggestions are cached for a source image.

        Args:
            source_image_path: Path to source image

        Returns:
            True if cached, False otherwise
        """
        normalized_path = str(Path(source_image_path))
        return normalized_path in self.suggestions

    def get_all(self) -> Dict[str, List[Dict]]:
        """
        Get all cached suggestions.

        Returns:
            Dictionary mapping source paths to suggestion lists
        """
        return self.suggestions.copy()

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_suggestions = sum(len(suggestions) for suggestions in self.suggestions.values())
        return {
            "total_images": len(self.suggestions),
            "total_suggestions": total_suggestions,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "domain": self.domain
        }

    def clear(self):
        """Clear all cached suggestions."""
        self.suggestions = {}
        self.created_at = None
        self.last_updated = None

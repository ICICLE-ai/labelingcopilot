"""
Utilities for image input/output operations.
Handles reading from folders, writing to folders, and image format conversions.
"""

import base64
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import io


class ImageLoader:
    """Handles loading images from a directory."""

    def __init__(self, input_dir: str, extensions: Optional[List[str]] = None):
        """
        Initialize ImageLoader.

        Args:
            input_dir: Path to directory containing images
            extensions: List of allowed file extensions (default: ['.jpg', '.jpeg', '.png'])
        """
        self.input_dir = Path(input_dir)
        self.extensions = extensions or ['.jpg', '.jpeg', '.png']

        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

    def load_images(self, min_size: int = 1024) -> List[Dict[str, any]]:
        """
        Load all valid images from the input directory.

        Args:
            min_size: Minimum file size in bytes to consider valid

        Returns:
            List of dicts with keys: 'path', 'name', 'image', 'size'
        """
        images = []

        for ext in self.extensions:
            for image_path in self.input_dir.glob(f"*{ext}"):
                # Skip if file is too small
                if image_path.stat().st_size < min_size:
                    continue

                try:
                    # Load image
                    img = Image.open(image_path)

                    images.append({
                        'path': str(image_path),
                        'name': image_path.stem,
                        'image': img,
                        'size': image_path.stat().st_size,
                        'format': img.format,
                        'dimensions': img.size
                    })
                except Exception as e:
                    print(f"Warning: Could not load {image_path}: {e}")
                    continue

        return images

    def get_image_paths(self, min_size: int = 1024) -> List[str]:
        """
        Get paths to all valid images without loading them.

        Args:
            min_size: Minimum file size in bytes to consider valid

        Returns:
            List of image file paths
        """
        paths = []

        for ext in self.extensions:
            for image_path in self.input_dir.glob(f"*{ext}"):
                if image_path.stat().st_size >= min_size:
                    paths.append(str(image_path))

        return sorted(paths)


class ImageWriter:
    """Handles writing images to a directory."""

    def __init__(self, output_dir: str, create_if_missing: bool = True):
        """
        Initialize ImageWriter.

        Args:
            output_dir: Path to output directory
            create_if_missing: Create directory if it doesn't exist
        """
        self.output_dir = Path(output_dir)

        if create_if_missing:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        elif not self.output_dir.exists():
            raise ValueError(f"Output directory does not exist: {output_dir}")

    def save_image(
        self,
        image: Image.Image,
        filename: str,
        format: str = "PNG",
        quality: int = 95
    ) -> str:
        """
        Save an image to the output directory.

        Args:
            image: PIL Image object
            filename: Output filename (without extension)
            format: Image format (PNG, JPEG, etc.)
            quality: Quality for JPEG (1-100)

        Returns:
            Path to saved image
        """
        # Add extension if not present
        ext = f".{format.lower()}"
        if not filename.endswith(ext):
            filename = f"{filename}{ext}"

        output_path = self.output_dir / filename

        # Save image
        if format.upper() == "JPEG":
            image.save(output_path, format=format, quality=quality)
        else:
            image.save(output_path, format=format)

        return str(output_path)

    def save_from_bytes(
        self,
        image_bytes: bytes,
        filename: str,
        format: str = "PNG"
    ) -> str:
        """
        Save an image from bytes.

        Args:
            image_bytes: Image data as bytes
            filename: Output filename
            format: Image format

        Returns:
            Path to saved image
        """
        image = Image.open(io.BytesIO(image_bytes))
        return self.save_image(image, filename, format)

    def save_from_base64(
        self,
        base64_data: str,
        filename: str,
        format: str = "PNG"
    ) -> str:
        """
        Save an image from base64 encoded data.

        Args:
            base64_data: Base64 encoded image data
            filename: Output filename
            format: Image format

        Returns:
            Path to saved image
        """
        # Remove data URL prefix if present
        if "," in base64_data:
            base64_data = base64_data.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_data)
        return self.save_from_bytes(image_bytes, filename, format)


def image_to_data_url(image_path: str) -> str:
    """
    Convert a local image file to a base64 data URL.

    Args:
        image_path: Path to image file

    Returns:
        Base64 data URL string
    """
    # Determine MIME type from extension
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(ext, 'image/jpeg')

    # Read and encode image
    with open(image_path, 'rb') as f:
        image_data = f.read()

    base64_data = base64.b64encode(image_data).decode('utf-8')
    return f"data:{mime_type};base64,{base64_data}"


def pil_to_data_url(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert a PIL Image to a base64 data URL.

    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Base64 data URL string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)

    mime_type = f"image/{format.lower()}"
    base64_data = base64.b64encode(buffer.read()).decode('utf-8')

    return f"data:{mime_type};base64,{base64_data}"


def resize_if_needed(
    image: Image.Image,
    max_size: Tuple[int, int] = (2048, 2048)
) -> Image.Image:
    """
    Resize image if it exceeds max dimensions, maintaining aspect ratio.

    Args:
        image: PIL Image object
        max_size: Maximum (width, height)

    Returns:
        Resized image or original if within limits
    """
    if image.width <= max_size[0] and image.height <= max_size[1]:
        return image

    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def validate_image(image_path: str, min_size: int = 1024) -> bool:
    """
    Validate that an image file is usable.

    Args:
        image_path: Path to image file
        min_size: Minimum file size in bytes

    Returns:
        True if image is valid
    """
    path = Path(image_path)

    # Check file exists
    if not path.exists():
        return False

    # Check file size
    if path.stat().st_size < min_size:
        return False

    # Try to open image
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

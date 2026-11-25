import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from fastapi import UploadFile
from pydantic import ValidationError

from app.core.config import settings

logger = logging.getLogger("storage_service")

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp"}


class StorageError(Exception):
    """Raised when image storage operations fail."""


class StorageService:
    """
    Secure file storage service for angiography images and inference outputs.

    Responsibilities:
    - Validate file content (not just extension)
    - Enforce maximum upload size
    - Generate server-side unique filenames (UUID-based)
    - Prevent path traversal and arbitrary filesystem access
    - Store images in a configurable storage directory
    - Return relative storage identifiers (never absolute server paths)
    """

    def __init__(self, storage_dir: Optional[str] = None, max_upload_size_mb: Optional[int] = None):
        self.storage_dir = Path(storage_dir or settings.STORAGE_DIR)
        self.max_upload_size = (max_upload_size_mb or settings.MAX_UPLOAD_SIZE_MB) * 1024 * 1024
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def validate_upload_file(self, file: UploadFile) -> None:
        """
        Validate an uploaded file before storage.

        Checks:
        - Content type is in the allowed set
        - File does not exceed maximum size
        - File content can be decoded as a valid image

        Raises:
            StorageError: If validation fails.
        """
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise StorageError(
                f"Unsupported image type: {file.content_type or 'unknown'}. "
                f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            )

        # Read entire file into memory for size validation and content decoding
        file.file.seek(0)
        content = file.file.read()
        file.file.seek(0)

        if len(content) > self.max_upload_size:
            raise StorageError(
                f"File size {len(content)} bytes exceeds maximum "
                f"{self.max_upload_size // (1024 * 1024)} MB."
            )

        if len(content) == 0:
            raise StorageError("Uploaded file is empty.")

        # Validate that the content is a decodable image
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise StorageError("Unable to decode image content. File may be corrupted or not a valid image.")

    def compute_content_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of file content for deduplication reference."""
        return hashlib.sha256(content).hexdigest()

    def generate_storage_path(self, original_filename: str, content_hash: str) -> Tuple[str, str]:
        """
        Generate a server-side unique storage path.

        Returns:
            (relative_path, storage_filename) — relative path for DB storage,
            and the filename on disk.
        """
        # Use UUID for filename to prevent collision and path traversal
        unique_id = uuid.uuid4().hex
        safe_name = f"{unique_id}.png"  # Always store as PNG for consistency
        relative_path = f"images/{safe_name}"
        full_path = self.storage_dir / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return relative_path, str(full_path)

    def save_file(self, content: bytes, full_path: str) -> None:
        """Write raw bytes to the storage path."""
        path = Path(full_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)

    def save_image_array(self, image: np.ndarray, full_path: str, is_rgb: bool = True) -> None:
        """Save a numpy image array to disk as PNG."""
        path = Path(full_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if is_rgb:
            to_encode = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            to_encode = image
        success = cv2.imwrite(str(path), to_encode)
        if not success:
            raise StorageError(f"Failed to write image to {full_path}")

    def file_exists(self, relative_path: str) -> bool:
        """Check if a stored file exists."""
        return (self.storage_dir / relative_path).exists()

    def get_full_path(self, relative_path: str) -> Path:
        """
        Resolve a relative storage path to a full filesystem path.

        Security: The relative path is joined with storage_dir and resolved.
        If the resolved path escapes storage_dir, a StorageError is raised.
        """
        base = self.storage_dir.resolve()
        target = (base / relative_path).resolve()
        # Ensure the resolved path is within the storage directory
        try:
            target.relative_to(base)
        except ValueError:
            raise StorageError(f"Resolved path escapes storage directory: {relative_path}")
        return target

    def delete_file(self, relative_path: str) -> bool:
        """Delete a stored file if it exists. Returns True if deleted."""
        try:
            full_path = self.get_full_path(relative_path)
        except StorageError:
            return False
        if full_path.exists():
            full_path.unlink()
            return True
        return False


# Global singleton instance
storage_service = StorageService()

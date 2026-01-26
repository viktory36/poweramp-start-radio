"""Music file scanner - discovers audio files in a directory tree."""

import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {
    ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wav", ".wma",
    ".alac", ".aiff", ".ape", ".mpc", ".wv", ".tta", ".dsf", ".dff"
}


def scan_music_directory(root: Path) -> Iterator[Path]:
    """
    Recursively scan a directory for audio files.

    Args:
        root: Root directory to scan

    Yields:
        Path objects for each audio file found
    """
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    logger.info(f"Scanning directory: {root}")

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def count_audio_files(root: Path) -> int:
    """Count the total number of audio files in a directory tree."""
    return sum(1 for _ in scan_music_directory(root))

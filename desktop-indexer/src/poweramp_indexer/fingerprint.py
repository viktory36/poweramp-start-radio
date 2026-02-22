"""Track fingerprinting - creates keys for matching tracks between desktop and phone."""

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _normalize_field(value: str) -> str:
    """Lowercase, strip, NFC-normalize, and remove pipe characters from a metadata field."""
    # NFC before lower() prevents decomposition from .lower() on precomposed chars (e.g. İ)
    nfc = unicodedata.normalize('NFC', value)
    return unicodedata.normalize('NFC', nfc.lower().strip().replace("|", "/"))


@dataclass
class TrackMetadata:
    """Metadata extracted from an audio file."""
    artist: Optional[str]
    album: Optional[str]
    title: Optional[str]
    duration_ms: int
    file_path: Path

    @property
    def metadata_key(self) -> str:
        """
        Create a metadata key for matching: "artist|album|title|duration_ms"
        All strings are lowercased, stripped, NFC-normalized, and pipe-sanitized.
        """
        artist = _normalize_field(self.artist or "")
        album = _normalize_field(self.album or "")
        title = _normalize_field(self.title or "")
        # Round duration to nearest 100ms to allow for minor variations
        duration_rounded = (self.duration_ms // 100) * 100
        return f"{artist}|{album}|{title}|{duration_rounded}"

    @property
    def filename_key(self) -> str:
        """
        Create a normalized filename key for fallback matching.
        Removes path, extension, common suffixes, and normalizes.
        """
        name = self.file_path.stem.lower()
        # Remove common variations like (1), [Explicit], etc.
        name = re.sub(r'\s*[\(\[].*?[\)\]]', '', name)
        # Remove track numbers at the start
        name = re.sub(r'^\d+[\.\-\s]+', '', name)
        # Normalize whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        return unicodedata.normalize('NFC', name)


def extract_metadata(file_path: Path) -> TrackMetadata:
    """
    Extract metadata from an audio file using mutagen.

    Args:
        file_path: Path to the audio file

    Returns:
        TrackMetadata with extracted information
    """
    try:
        import mutagen
        from mutagen.easyid3 import EasyID3
        from mutagen.flac import FLAC
        from mutagen.mp4 import MP4
        from mutagen.oggvorbis import OggVorbis
        from mutagen.oggopus import OggOpus
    except ImportError:
        logger.warning("mutagen not installed, using filename-only metadata")
        return TrackMetadata(
            artist=None,
            album=None,
            title=file_path.stem,
            duration_ms=0,
            file_path=file_path
        )

    try:
        audio = mutagen.File(file_path, easy=True)
        if audio is None:
            logger.warning(f"Could not read metadata from {file_path}")
            return TrackMetadata(
                artist=None,
                album=None,
                title=file_path.stem,
                duration_ms=0,
                file_path=file_path
            )

        # Get duration in milliseconds
        duration_ms = int((audio.info.length if audio.info else 0) * 1000)

        # Extract common tags
        def get_tag(tags: list[str]) -> Optional[str]:
            for tag in tags:
                if tag in audio:
                    val = audio[tag]
                    if isinstance(val, list) and val:
                        return str(val[0])
                    elif val:
                        return str(val)
            return None

        artist = get_tag(['artist', 'albumartist', 'performer'])
        album = get_tag(['album'])
        title = get_tag(['title'])

        # Fallback to filename if no title
        if not title:
            title = file_path.stem

        return TrackMetadata(
            artist=artist,
            album=album,
            title=title,
            duration_ms=duration_ms,
            file_path=file_path
        )

    except Exception as e:
        logger.warning(f"Error reading metadata from {file_path}: {e}")
        return TrackMetadata(
            artist=None,
            album=None,
            title=file_path.stem,
            duration_ms=0,
            file_path=file_path
        )

"""SQLite database management for embeddings storage."""

import logging
import sqlite3
import struct
from pathlib import Path
from typing import Optional

from .fingerprint import TrackMetadata

logger = logging.getLogger(__name__)


def float_list_to_blob(floats: list[float]) -> bytes:
    """Convert a list of floats to a binary blob."""
    return struct.pack(f'{len(floats)}f', *floats)


def blob_to_float_list(blob: bytes) -> list[float]:
    """Convert a binary blob back to a list of floats."""
    count = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f'{count}f', blob))


class EmbeddingDatabase:
    """
    SQLite database for storing track metadata and embeddings.

    Schema:
    - tracks: metadata and fingerprint keys
    - embeddings: float32 vectors as BLOBs (dimension depends on model)
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS tracks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metadata_key TEXT NOT NULL,
        filename_key TEXT NOT NULL,
        artist TEXT,
        album TEXT,
        title TEXT,
        duration_ms INTEGER,
        file_path TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS embeddings (
        track_id INTEGER PRIMARY KEY,
        embedding BLOB NOT NULL,
        FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_tracks_metadata_key ON tracks(metadata_key);
    CREATE INDEX IF NOT EXISTS idx_tracks_filename_key ON tracks(filename_key);

    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """

    def __init__(self, db_path: Path):
        """
        Initialize or open an embedding database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def set_metadata(self, key: str, value: str):
        """Set a metadata key-value pair."""
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value)
        )
        self.conn.commit()

    def get_metadata(self, key: str) -> Optional[str]:
        """Get a metadata value by key."""
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def add_track(
        self,
        metadata: TrackMetadata,
        embedding: list[float]
    ) -> int:
        """
        Add a track and its embedding to the database.

        Args:
            metadata: Track metadata
            embedding: Embedding vector (dimension depends on model)

        Returns:
            The track ID
        """
        cursor = self.conn.execute(
            """
            INSERT INTO tracks (metadata_key, filename_key, artist, album, title, duration_ms, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata.metadata_key,
                metadata.filename_key,
                metadata.artist,
                metadata.album,
                metadata.title,
                metadata.duration_ms,
                str(metadata.file_path)
            )
        )
        track_id = cursor.lastrowid

        embedding_blob = float_list_to_blob(embedding)
        self.conn.execute(
            "INSERT INTO embeddings (track_id, embedding) VALUES (?, ?)",
            (track_id, embedding_blob)
        )

        return track_id

    def track_exists(self, file_path: Path) -> bool:
        """Check if a track already exists in the database by file path."""
        row = self.conn.execute(
            "SELECT 1 FROM tracks WHERE file_path = ?", (str(file_path),)
        ).fetchone()
        return row is not None

    def get_existing_paths(self) -> set[str]:
        """Get all file paths currently in the database."""
        rows = self.conn.execute("SELECT file_path FROM tracks").fetchall()
        return {row["file_path"] for row in rows}

    def remove_missing_tracks(self, existing_paths: set[str]):
        """Remove tracks whose files no longer exist."""
        current_paths = self.get_existing_paths()
        missing = current_paths - existing_paths

        if missing:
            logger.info(f"Removing {len(missing)} tracks with missing files")
            placeholders = ",".join("?" * len(missing))
            self.conn.execute(
                f"DELETE FROM tracks WHERE file_path IN ({placeholders})",
                list(missing)
            )
            self.conn.commit()

    def count_tracks(self) -> int:
        """Return the number of tracks in the database."""
        row = self.conn.execute("SELECT COUNT(*) as count FROM tracks").fetchone()
        return row["count"]

    def commit(self):
        """Commit any pending changes."""
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def vacuum(self):
        """Reclaim unused space in the database."""
        self.conn.execute("VACUUM")

    def get_all_embeddings(self) -> dict[int, list[float]]:
        """
        Load all embeddings from the database.

        Returns:
            Dictionary mapping track_id to embedding vector
        """
        rows = self.conn.execute(
            "SELECT track_id, embedding FROM embeddings"
        ).fetchall()
        return {row["track_id"]: blob_to_float_list(row["embedding"]) for row in rows}

    def get_embedding_by_id(self, track_id: int) -> Optional[list[float]]:
        """Get the embedding for a specific track."""
        row = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE track_id = ?", (track_id,)
        ).fetchone()
        return blob_to_float_list(row["embedding"]) if row else None

    def get_track_by_id(self, track_id: int) -> Optional[dict]:
        """Get track metadata by ID."""
        row = self.conn.execute(
            "SELECT id, artist, album, title, file_path FROM tracks WHERE id = ?",
            (track_id,)
        ).fetchone()
        return dict(row) if row else None

    def search_tracks(self, query: str) -> list[dict]:
        """
        Search tracks by artist, album, or title (case-insensitive).
        All words in the query must appear somewhere in the combined metadata.

        Args:
            query: Search string (space-separated words)

        Returns:
            List of matching track dictionaries
        """
        words = query.lower().split()
        if not words:
            return []

        # Build query: all words must appear in the concatenated fields
        conditions = []
        params = []
        for word in words:
            conditions.append(
                "(LOWER(COALESCE(artist,'') || ' ' || COALESCE(album,'') || ' ' || COALESCE(title,'') || ' ' || file_path) LIKE ?)"
            )
            params.append(f"%{word}%")

        rows = self.conn.execute(
            f"""
            SELECT id, artist, album, title, file_path
            FROM tracks
            WHERE {' AND '.join(conditions)}
            LIMIT 50
            """,
            params
        ).fetchall()
        return [dict(row) for row in rows]

    def find_track(self, artist: str = None, title: str = None, file_path: str = None) -> Optional[dict]:
        """
        Find a specific track by exact artist+title or file path.
        This is what the Android app should use.

        Args:
            artist: Exact artist name (case-insensitive)
            title: Exact title (case-insensitive)
            file_path: Exact file path or filename

        Returns:
            Track dictionary or None
        """
        # Try exact file path first
        if file_path:
            row = self.conn.execute(
                "SELECT id, artist, album, title, file_path FROM tracks WHERE file_path = ?",
                (file_path,)
            ).fetchone()
            if row:
                return dict(row)

            # Try matching just the filename
            row = self.conn.execute(
                "SELECT id, artist, album, title, file_path FROM tracks WHERE file_path LIKE ?",
                (f"%{file_path.split('/')[-1].split(chr(92))[-1]}",)  # handle both / and \
            ).fetchone()
            if row:
                return dict(row)

        # Try artist + title
        if artist and title:
            row = self.conn.execute(
                """
                SELECT id, artist, album, title, file_path FROM tracks
                WHERE LOWER(artist) = LOWER(?) AND LOWER(title) = LOWER(?)
                """,
                (artist, title)
            ).fetchone()
            if row:
                return dict(row)

        return None

    def get_random_track(self) -> Optional[dict]:
        """Pick a random track from the database."""
        row = self.conn.execute(
            "SELECT id, artist, album, title, file_path FROM tracks ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

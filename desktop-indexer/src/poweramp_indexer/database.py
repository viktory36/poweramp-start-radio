"""SQLite database management for embeddings storage."""

from __future__ import annotations

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

    Supports multiple embedding models via per-model tables (embeddings_mulan,
    embeddings_flamingo, embeddings_fused, etc.).
    """

    BASE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS tracks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metadata_key TEXT NOT NULL,
        filename_key TEXT NOT NULL,
        artist TEXT,
        album TEXT,
        title TEXT,
        duration_ms INTEGER,
        file_path TEXT NOT NULL,
        cluster_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_tracks_metadata_key ON tracks(metadata_key);
    CREATE INDEX IF NOT EXISTS idx_tracks_filename_key ON tracks(filename_key);
    CREATE INDEX IF NOT EXISTS idx_tracks_file_path ON tracks(file_path);

    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    );

    CREATE TABLE IF NOT EXISTS clusters (
        cluster_id INTEGER PRIMARY KEY,
        embedding BLOB NOT NULL
    );

    CREATE TABLE IF NOT EXISTS binary_data (
        key TEXT PRIMARY KEY,
        data BLOB NOT NULL
    );
    """

    def __init__(self, db_path: Path, models: list[str] | None = None):
        """
        Initialize or open an embedding database.

        Args:
            db_path: Path to the SQLite database file
            models: List of model names to create embedding tables for
                    (e.g. ["mulan"] or ["mulan", "flamingo"]). If None, only base
                    tables are created and existing embedding tables are used.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema(models)

    def _init_schema(self, models: list[str] | None):
        """Create base tables and per-model embedding tables."""
        self.conn.executescript(self.BASE_SCHEMA)
        if models:
            for model in models:
                table = f"embeddings_{model}"
                self.conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS [{table}] (
                        track_id INTEGER PRIMARY KEY,
                        embedding BLOB NOT NULL,
                        FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
                    )
                """)
        self.conn.commit()

    def _table_name(self, model: str) -> str:
        """Get the embedding table name for a model."""
        return f"embeddings_{model}"

    def get_available_models(self) -> list[str]:
        """Return list of model names that have embedding tables with rows."""
        models = []

        # Check for embeddings_* tables
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'embeddings_%'"
        ).fetchall()
        for row in rows:
            table_name = row["name"]
            # Check it has rows
            count = self.conn.execute(f"SELECT COUNT(*) as c FROM [{table_name}]").fetchone()["c"]
            if count > 0:
                # Strip 'embeddings_' prefix to get model name
                model = table_name[len("embeddings_"):]
                models.append(model)

        return models

    def get_track_id_by_path(self, file_path: str) -> Optional[int]:
        """Look up a track ID by its file path."""
        row = self.conn.execute(
            "SELECT id FROM tracks WHERE file_path = ?", (file_path,)
        ).fetchone()
        return row["id"] if row else None

    def add_embedding(self, track_id: int, model: str, embedding: list[float]):
        """Insert an embedding into the model-specific table."""
        table = f"embeddings_{model}"
        blob = float_list_to_blob(embedding)
        self.conn.execute(
            f"INSERT OR REPLACE INTO [{table}] (track_id, embedding) VALUES (?, ?)",
            (track_id, blob)
        )

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

    def set_binary(self, key: str, data: bytes):
        """Store a binary blob in the binary_data table."""
        self.conn.execute(
            "INSERT OR REPLACE INTO binary_data (key, data) VALUES (?, ?)",
            (key, data)
        )
        self.conn.commit()

    def get_binary(self, key: str) -> Optional[bytes]:
        """Get a binary blob by key."""
        row = self.conn.execute(
            "SELECT data FROM binary_data WHERE key = ?", (key,)
        ).fetchone()
        return bytes(row["data"]) if row else None

    def add_track(
        self,
        metadata: TrackMetadata,
        embedding: list[float],
        model: str = "mulan"
    ) -> int:
        """
        Add a track and its embedding to the database.

        Args:
            metadata: Track metadata
            embedding: Embedding vector (dimension depends on model)
            model: Model name for embedding table (default: "mulan")

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
        self.add_embedding(track_id, model, embedding)
        return track_id

    def get_existing_paths(self, model: str | None = None) -> set[str]:
        """Get file paths in the database.

        When model is specified, only returns paths that have an embedding
        for that model (via JOIN). Otherwise returns all track paths.
        """
        if model is None:
            rows = self.conn.execute("SELECT file_path FROM tracks").fetchall()
        else:
            table = self._table_name(model)
            rows = self.conn.execute(
                f"SELECT t.file_path FROM tracks t INNER JOIN [{table}] e ON t.id = e.track_id"
            ).fetchall()
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

    def count_embeddings(self, model: str) -> int:
        """Return the number of embeddings for a given model."""
        table = self._table_name(model)
        try:
            row = self.conn.execute(f"SELECT COUNT(*) as count FROM [{table}]").fetchone()
            return row["count"]
        except sqlite3.OperationalError:
            return 0

    def commit(self):
        """Commit any pending changes."""
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def vacuum(self):
        """Reclaim unused space in the database."""
        self.conn.execute("VACUUM")

    def get_all_embeddings(self, model: str = "mulan") -> dict[int, list[float]]:
        """
        Load all embeddings from the database for a given model.

        Returns:
            Dictionary mapping track_id to embedding vector
        """
        table = self._table_name(model)
        rows = self.conn.execute(
            f"SELECT track_id, embedding FROM [{table}]"
        ).fetchall()
        return {row["track_id"]: blob_to_float_list(row["embedding"]) for row in rows}

    def get_embedding_by_id(self, track_id: int, model: str = "mulan") -> Optional[list[float]]:
        """Get the embedding for a specific track."""
        table = self._table_name(model)
        row = self.conn.execute(
            f"SELECT embedding FROM [{table}] WHERE track_id = ?", (track_id,)
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

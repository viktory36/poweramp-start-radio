"""Build and validate the cumulative, graphless Android merge bundle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from .server_config import (
    CLAMP3_AUDIO_MODEL_SHA256,
    CLAMP3_REPOSITORY,
    CLAMP3_REVISION,
    MERT_MODEL_ID,
    MERT_MODEL_REVISION,
    MERT_MODEL_SHA256,
    SERVER_EMBEDDING_SPEC_ID,
    SERVER_OUTPUT_SPACE_ID,
    SERVER_SOURCE_DECODER_IDS,
    canonical_server_embedding_spec_json,
    server_embedding_spec_sha256,
)


BUNDLE_FORMAT = "poweramp-server-embedding-bundle"
BUNDLE_SCHEMA_VERSION = 1
EMBEDDING_DIMENSION = 768
EMBEDDING_BYTES = EMBEDDING_DIMENSION * 4
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_RELATIVE_PATH_CHARS = 8192

BUNDLE_SCHEMA = """
CREATE TABLE tracks (
    id INTEGER PRIMARY KEY,
    metadata_key TEXT NOT NULL,
    filename_key TEXT NOT NULL,
    artist TEXT,
    album TEXT,
    title TEXT,
    duration_ms INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    cluster_id INTEGER,
    source TEXT NOT NULL DEFAULT 'server'
);
CREATE INDEX idx_tracks_metadata_key ON tracks(metadata_key);
CREATE INDEX idx_tracks_filename_key ON tracks(filename_key);
CREATE INDEX idx_tracks_file_path ON tracks(file_path);

CREATE TABLE embeddings_clamp3 (
    track_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE server_bundle_tracks (
    track_id INTEGER PRIMARY KEY,
    root_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
    source_size_bytes INTEGER NOT NULL CHECK(source_size_bytes > 0),
    source_sample_rate_hz INTEGER NOT NULL CHECK(source_sample_rate_hz > 0),
    source_sample_count INTEGER NOT NULL CHECK(source_sample_count > 0),
    source_decoder_id TEXT NOT NULL,
    span_start_sample INTEGER NOT NULL CHECK(span_start_sample >= 0),
    span_end_sample_exclusive INTEGER NOT NULL
        CHECK(span_end_sample_exclusive > span_start_sample),
    embedding_sha256 TEXT NOT NULL CHECK(length(embedding_sha256) = 64),
    embedding_spec_id TEXT NOT NULL,
    output_space_id TEXT NOT NULL,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);
CREATE INDEX idx_server_bundle_source_sha256 ON server_bundle_tracks(source_sha256);
"""


@dataclass(frozen=True)
class BundlePublication:
    bundle_id: str
    track_count: int
    file_sha256: str
    file_size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_embedding_blob(blob: bytes) -> None:
    if len(blob) != EMBEDDING_BYTES:
        raise ValueError(f"embedding has {len(blob)} bytes; expected {EMBEDDING_BYTES}")
    values = struct.unpack(f"<{EMBEDDING_DIMENSION}f", blob)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("embedding contains a non-finite value")
    norm = math.sqrt(sum(value * value for value in values))
    if abs(norm - 1.0) > 0.001:
        raise ValueError(f"embedding norm is {norm:.8f}; expected 1.0 +/- 0.001")


def _source_rows(state: sqlite3.Connection) -> list[sqlite3.Row]:
    state.row_factory = sqlite3.Row
    return state.execute(
        """
        SELECT t.id, t.metadata_key, t.filename_key, t.artist, t.album, t.title,
               t.duration_ms, t.file_path, t.source, e.embedding,
               p.root_id, p.relative_path, p.source_sha256, p.source_size_bytes,
               p.source_sample_rate_hz, p.source_sample_count, p.source_decoder_id,
               p.span_start_sample, p.span_end_sample_exclusive,
               p.embedding_sha256, p.embedding_spec_id, p.output_space_id
          FROM tracks t
          JOIN embeddings_clamp3 e ON e.track_id = t.id
          JOIN server_track_provenance p ON p.track_id = t.id
         ORDER BY p.root_id, p.relative_path, p.source_sha256, t.id
        """
    ).fetchall()


def _require_complete_state_export_set(state: sqlite3.Connection) -> None:
    integrity = state.execute("PRAGMA quick_check").fetchone()[0]
    if integrity != "ok":
        raise ValueError(f"server state integrity check failed: {integrity}")
    foreign_key_errors = state.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_errors:
        raise ValueError(f"server state has foreign-key errors: {foreign_key_errors[:5]}")
    counts = state.execute(
        """
        SELECT (SELECT COUNT(*) FROM tracks),
               (SELECT COUNT(*) FROM embeddings_clamp3),
               (SELECT COUNT(*) FROM server_track_provenance)
        """
    ).fetchone()
    if len(set(int(value) for value in counts)) != 1:
        raise ValueError(f"server state export tables disagree: {tuple(counts)}")


def _logical_bundle_id(rows: list[sqlite3.Row]) -> str:
    digest = hashlib.sha256()
    digest.update(b"poweramp-server-bundle-v1\0")
    digest.update(canonical_server_embedding_spec_json().encode("utf-8"))
    digest.update(b"\0")
    for row in rows:
        record = {
            "album": row["album"],
            "artist": row["artist"],
            "duration_ms": row["duration_ms"],
            "embedding_sha256": row["embedding_sha256"],
            "filename_key": row["filename_key"],
            "file_path": row["file_path"],
            "metadata_key": row["metadata_key"],
            "relative_path": row["relative_path"],
            "root_id": row["root_id"],
            "source_sample_count": row["source_sample_count"],
            "source_sample_rate_hz": row["source_sample_rate_hz"],
            "source_sha256": row["source_sha256"],
            "source_size_bytes": row["source_size_bytes"],
            "title": row["title"],
        }
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return "server-bundle-v1-" + digest.hexdigest()


def expected_bundle_identity(state: sqlite3.Connection) -> tuple[str, int]:
    rows = _source_rows(state)
    return _logical_bundle_id(rows), len(rows)


def _declared_bundle_identity(path: Path) -> tuple[str, int] | None:
    if not path.is_file():
        return None
    try:
        db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        try:
            metadata = dict(db.execute(
                """
                SELECT key, value FROM metadata
                 WHERE key IN (
                     'server_bundle_id',
                     'server_bundle_track_count',
                     'server_bundle_source_decoder_provenance_version'
                 )
                """
            ))
            columns = {
                row[1] for row in db.execute("PRAGMA table_info(server_bundle_tracks)")
            }
        finally:
            db.close()
        if (metadata.get("server_bundle_source_decoder_provenance_version") != "1" or
                "source_decoder_id" not in columns):
            return None
        return metadata["server_bundle_id"], int(metadata["server_bundle_track_count"])
    except Exception:
        return None


def _metadata(bundle_id: str, row_count: int) -> dict[str, str]:
    return {
        "bundle_format": BUNDLE_FORMAT,
        "server_bundle_schema_version": str(BUNDLE_SCHEMA_VERSION),
        "server_bundle_id": bundle_id,
        "server_bundle_track_count": str(row_count),
        "server_bundle_embedding_spec_id": SERVER_EMBEDDING_SPEC_ID,
        "server_bundle_output_space_id": SERVER_OUTPUT_SPACE_ID,
        "server_bundle_mert_model_sha256": MERT_MODEL_SHA256,
        "server_bundle_clamp3_audio_model_sha256": CLAMP3_AUDIO_MODEL_SHA256,
        "server_bundle_embedding_spec_json": canonical_server_embedding_spec_json(),
        "server_bundle_embedding_spec_sha256": server_embedding_spec_sha256(),
        "server_bundle_mert_model_id": MERT_MODEL_ID,
        "server_bundle_mert_model_revision": MERT_MODEL_REVISION,
        "server_bundle_clamp3_repository": CLAMP3_REPOSITORY,
        "server_bundle_clamp3_revision": CLAMP3_REVISION,
        "server_bundle_graph_included": "false",
        "server_bundle_source_decoder_provenance_version": "1",
        # Existing database readers still surface these generic values.
        "model": "clamp3",
        "embedding_dim": str(EMBEDDING_DIMENSION),
    }


def _validate_provenance_fields(row: sqlite3.Row) -> None:
    root_id = str(row["root_id"])
    relative_path = str(row["relative_path"])
    relative = Path(relative_path)
    if not root_id or not relative_path or relative.is_absolute() or "\\" in relative_path:
        raise ValueError(f"invalid bundle path {root_id!r}/{relative_path!r}")
    if len(relative_path) > _MAX_RELATIVE_PATH_CHARS or any(
        ord(character) <= 0x1F or 0x7F <= ord(character) <= 0x9F
        for character in relative_path
    ):
        raise ValueError(f"bundle relative path is not Android-admissible: {relative_path!r}")
    if (relative_path != relative.as_posix() or
            unicodedata.normalize("NFC", relative_path) != relative_path or
            any(part in ("", ".", "..") for part in relative.parts)):
        raise ValueError(f"bundle relative path is not normalized: {relative_path!r}")
    if int(row["source_size_bytes"]) <= 0:
        raise ValueError(f"track {row['id']} has an empty source")
    if not _LOWER_SHA256.fullmatch(str(row["source_sha256"])):
        raise ValueError(f"invalid source SHA-256 for track {row['id']}")
    if not _LOWER_SHA256.fullmatch(str(row["embedding_sha256"])):
        raise ValueError(f"invalid embedding SHA-256 for track {row['id']}")
    if int(row["span_start_sample"]) != 0:
        raise ValueError(f"server track {row['id']} is not a whole-file span")
    if int(row["span_end_sample_exclusive"]) != int(row["source_sample_count"]):
        raise ValueError(f"server track {row['id']} span does not cover its complete source")
    if row["source_decoder_id"] not in SERVER_SOURCE_DECODER_IDS:
        raise ValueError(
            f"server track {row['id']} has unsupported decoder provenance "
            f"{row['source_decoder_id']!r}"
        )
    expected_uri = f"server://{root_id}/{quote(relative_path, safe='/')}"
    if row["file_path"] != expected_uri or row["source"] != "server":
        raise ValueError(f"track {row['id']} violates the server URI/source contract")


def publish_bundle(
    state: sqlite3.Connection,
    destination: Path,
    *,
    replace: Callable[[str, str], None] = os.replace,
) -> BundlePublication:
    """Publish a complete bundle without ever mutating the visible destination."""
    _require_complete_state_export_set(state)
    rows = _source_rows(state)
    for row in rows:
        _validate_provenance_fields(row)
        blob = bytes(row["embedding"])
        validate_embedding_blob(blob)
        if hashlib.sha256(blob).hexdigest() != row["embedding_sha256"]:
            raise ValueError(f"embedding hash mismatch for track {row['id']}")
        if row["embedding_spec_id"] != SERVER_EMBEDDING_SPEC_ID:
            raise ValueError(f"track {row['id']} has an incompatible embedding spec")
        if row["output_space_id"] != SERVER_OUTPUT_SPACE_ID:
            raise ValueError(f"track {row['id']} has an incompatible output space")

    bundle_id = _logical_bundle_id(rows)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.next-", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()

    try:
        output = sqlite3.connect(temporary)
        try:
            output.execute("PRAGMA foreign_keys = ON")
            output.execute("PRAGMA journal_mode = DELETE")
            output.execute("PRAGMA synchronous = FULL")
            output.executescript(BUNDLE_SCHEMA)
            output.execute(f"PRAGMA user_version = {BUNDLE_SCHEMA_VERSION}")
            output.execute("BEGIN")
            for row in rows:
                output.execute(
                    """
                    INSERT INTO tracks(
                        id, metadata_key, filename_key, artist, album, title,
                        duration_ms, file_path, cluster_id, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                    """,
                    (
                        row["id"], row["metadata_key"], row["filename_key"],
                        row["artist"], row["album"], row["title"], row["duration_ms"],
                        row["file_path"], row["source"],
                    ),
                )
                output.execute(
                    "INSERT INTO embeddings_clamp3(track_id, embedding) VALUES (?, ?)",
                    (row["id"], row["embedding"]),
                )
                output.execute(
                    """
                    INSERT INTO server_bundle_tracks(
                        track_id, root_id, relative_path, source_sha256, source_size_bytes,
                        source_sample_rate_hz, source_sample_count, source_decoder_id,
                        span_start_sample, span_end_sample_exclusive, embedding_sha256,
                        embedding_spec_id, output_space_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"], row["root_id"], row["relative_path"],
                        row["source_sha256"], row["source_size_bytes"],
                        row["source_sample_rate_hz"], row["source_sample_count"],
                        row["source_decoder_id"], row["span_start_sample"],
                        row["span_end_sample_exclusive"], row["embedding_sha256"],
                        row["embedding_spec_id"], row["output_space_id"],
                    ),
                )
            output.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                sorted(_metadata(bundle_id, len(rows)).items()),
            )
            output.commit()
        finally:
            output.close()

        validate_bundle(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        replace(str(temporary), str(destination))
        directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)

    file_hash = sha256_file(destination)
    state.execute(
        """
        INSERT OR REPLACE INTO server_publications(
            generation_id, track_count, bundle_sha256, published_at_ns
        ) VALUES (?, ?, ?, ?)
        """,
        (bundle_id, len(rows), file_hash, time.time_ns()),
    )
    state.commit()
    return BundlePublication(bundle_id, len(rows), file_hash, destination.stat().st_size)


def ensure_bundle_current(
    state: sqlite3.Connection,
    destination: Path,
) -> BundlePublication | None:
    """Publish missing durable commits and repair a missing post-replace receipt."""
    expected = expected_bundle_identity(state)
    if _declared_bundle_identity(destination) != expected:
        return publish_bundle(state, destination)
    receipt = state.execute(
        "SELECT 1 FROM server_publications WHERE generation_id = ?", (expected[0],)
    ).fetchone()
    if receipt is not None:
        return None

    publication = validate_bundle(destination)
    if (publication.bundle_id, publication.track_count) != expected:
        return publish_bundle(state, destination)
    state.execute(
        """
        INSERT OR REPLACE INTO server_publications(
            generation_id, track_count, bundle_sha256, published_at_ns
        ) VALUES (?, ?, ?, ?)
        """,
        (
            publication.bundle_id, publication.track_count,
            publication.file_sha256, time.time_ns(),
        ),
    )
    state.commit()
    return publication


def validate_bundle(path: Path) -> BundlePublication:
    """Validate every row and the explicit no-graph transfer contract."""
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"bundle integrity check failed: {integrity}")
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        required = {"tracks", "embeddings_clamp3", "metadata", "server_bundle_tracks"}
        if not required.issubset(tables):
            raise ValueError(f"bundle is missing tables: {sorted(required - tables)}")
        if "clusters" in tables or "binary_data" in tables:
            raise ValueError("server bundle must not contain graph or cluster tables")
        metadata = dict(db.execute("SELECT key, value FROM metadata"))
        if metadata.get("bundle_format") != BUNDLE_FORMAT:
            raise ValueError("unexpected server bundle format")
        if metadata.get("server_bundle_schema_version") != str(BUNDLE_SCHEMA_VERSION):
            raise ValueError("unsupported server bundle schema")
        if metadata.get("server_bundle_embedding_spec_id") != SERVER_EMBEDDING_SPEC_ID:
            raise ValueError("unexpected server embedding spec")
        if metadata.get("server_bundle_output_space_id") != SERVER_OUTPUT_SPACE_ID:
            raise ValueError("unexpected server output space")
        if metadata.get("server_bundle_mert_model_sha256") != MERT_MODEL_SHA256:
            raise ValueError("unexpected MERT model hash")
        if metadata.get("server_bundle_clamp3_audio_model_sha256") != CLAMP3_AUDIO_MODEL_SHA256:
            raise ValueError("unexpected CLaMP3 model hash")
        if metadata.get("server_bundle_graph_included") != "false":
            raise ValueError("bundle incorrectly claims a graph")
        if metadata.get("server_bundle_source_decoder_provenance_version") != "1":
            raise ValueError("bundle decoder provenance is missing or unsupported")
        exact_spec_json = canonical_server_embedding_spec_json()
        if metadata.get("server_bundle_embedding_spec_json") != exact_spec_json:
            raise ValueError("bundle embedding spec JSON is not canonical or pinned")
        if metadata.get("server_bundle_embedding_spec_sha256") != server_embedding_spec_sha256():
            raise ValueError("bundle embedding spec hash is incorrect")

        counts = db.execute(
            """
            SELECT (SELECT COUNT(*) FROM tracks),
                   (SELECT COUNT(*) FROM embeddings_clamp3),
                   (SELECT COUNT(*) FROM server_bundle_tracks)
            """
        ).fetchone()
        count = int(metadata["server_bundle_track_count"])
        if tuple(counts) != (count, count, count):
            raise ValueError(f"bundle row-count mismatch: {tuple(counts)} versus {count}")
        rows = db.execute(
            """
            SELECT t.id, t.metadata_key, t.filename_key, t.artist, t.album, t.title,
                   t.duration_ms, t.file_path, t.source, e.embedding,
                   p.root_id, p.relative_path, p.source_sha256, p.source_size_bytes,
                   p.source_sample_rate_hz, p.source_sample_count, p.source_decoder_id,
                   p.span_start_sample, p.span_end_sample_exclusive,
                   p.embedding_sha256, p.embedding_spec_id, p.output_space_id
              FROM embeddings_clamp3 e
              JOIN tracks t ON t.id = e.track_id
              JOIN server_bundle_tracks p ON p.track_id = e.track_id
             ORDER BY p.root_id, p.relative_path, p.source_sha256, t.id
            """
        ).fetchall()
        for row in rows:
            _validate_provenance_fields(row)
            blob = bytes(row["embedding"])
            validate_embedding_blob(blob)
            if hashlib.sha256(blob).hexdigest() != row["embedding_sha256"]:
                raise ValueError(f"embedding hash mismatch for track {row['id']}")
            if row["embedding_spec_id"] != SERVER_EMBEDDING_SPEC_ID:
                raise ValueError("row embedding spec mismatch")
            if row["output_space_id"] != SERVER_OUTPUT_SPACE_ID:
                raise ValueError("row output space mismatch")
        logical_id = _logical_bundle_id(rows)
        if metadata.get("server_bundle_id") != logical_id:
            raise ValueError("bundle logical ID does not match its rows")
        return BundlePublication(
            bundle_id=metadata["server_bundle_id"],
            track_count=count,
            file_sha256=sha256_file(path),
            file_size_bytes=path.stat().st_size,
        )
    finally:
        db.close()

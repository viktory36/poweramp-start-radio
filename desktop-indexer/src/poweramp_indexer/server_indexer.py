"""Persistent polling indexer for music that arrives on an always-on server."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import logging
import math
import os
import sqlite3
import struct
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol
from urllib.parse import quote

import numpy as np

from .fingerprint import TrackMetadata, extract_metadata
from .scanner import scan_music_directory
from .server_bundle import (
    BundlePublication,
    ensure_bundle_current,
    publish_bundle,
    sha256_file,
    validate_embedding_blob,
)
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
    TORCHAUDIO_SOURCE_DECODER_ID,
    ServerConfig,
    server_embedding_spec_sha256,
)

logger = logging.getLogger(__name__)

STATE_SCHEMA_VERSION = 3
STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
CREATE TABLE IF NOT EXISTS embeddings_clamp3 (
    track_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS server_state_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS server_roots (
    root_id TEXT PRIMARY KEY,
    canonical_path TEXT UNIQUE NOT NULL,
    baseline_completed_at_ns INTEGER
);
CREATE TABLE IF NOT EXISTS server_files (
    root_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    device_id INTEGER,
    inode INTEGER,
    size_bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    stable_since_ns INTEGER NOT NULL,
    last_seen_ns INTEGER NOT NULL,
    present INTEGER NOT NULL CHECK(present IN (0, 1)),
    state TEXT NOT NULL,
    content_sha256 TEXT,
    track_id INTEGER,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    retry_after_ns INTEGER,
    error_code TEXT,
    error_message TEXT,
    last_processed_at_ns INTEGER,
    last_processing_duration_ms INTEGER
        CHECK(last_processing_duration_ms IS NULL OR last_processing_duration_ms >= 0),
    last_embedding_reused INTEGER
        CHECK(last_embedding_reused IS NULL OR last_embedding_reused IN (0, 1)),
    PRIMARY KEY(root_id, relative_path),
    FOREIGN KEY(root_id) REFERENCES server_roots(root_id),
    FOREIGN KEY(track_id) REFERENCES tracks(id)
);
CREATE INDEX IF NOT EXISTS idx_server_files_inode
    ON server_files(root_id, device_id, inode);
CREATE INDEX IF NOT EXISTS idx_server_files_state
    ON server_files(state, present, retry_after_ns);
CREATE TABLE IF NOT EXISTS server_track_provenance (
    track_id INTEGER PRIMARY KEY,
    root_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    source_size_bytes INTEGER NOT NULL,
    source_sample_rate_hz INTEGER NOT NULL,
    source_sample_count INTEGER NOT NULL,
    source_decoder_id TEXT NOT NULL,
    span_start_sample INTEGER NOT NULL,
    span_end_sample_exclusive INTEGER NOT NULL,
    embedding_sha256 TEXT NOT NULL,
    embedding_spec_id TEXT NOT NULL,
    output_space_id TEXT NOT NULL,
    FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_server_provenance_content
    ON server_track_provenance(source_sha256, embedding_spec_id);
CREATE TABLE IF NOT EXISTS server_publications (
    generation_id TEXT PRIMARY KEY,
    track_count INTEGER NOT NULL,
    bundle_sha256 TEXT NOT NULL,
    published_at_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS server_runtime (
    singleton_id INTEGER PRIMARY KEY CHECK(singleton_id = 1),
    phase TEXT NOT NULL,
    phase_started_at_ns INTEGER,
    cycle_started_at_ns INTEGER,
    current_root_id TEXT,
    current_relative_path TEXT,
    last_cycle_completed_at_ns INTEGER,
    next_poll_at_ns INTEGER,
    last_error TEXT
);
INSERT OR IGNORE INTO server_runtime(singleton_id, phase) VALUES (1, 'stopped');
"""

BASELINE = "BASELINE"
SETTLING = "SETTLING"
READY = "READY"
STAGING = "STAGING"
INDEXING = "INDEXING"
INDEXED = "INDEXED"
RETRY_WAIT = "RETRY_WAIT"
BLOCKED = "BLOCKED"

RECOVERABLE_ACTIVE_STATES = (STAGING, INDEXING)
RUNTIME_STARTING = "starting"
RUNTIME_BASELINING = "baselining"
RUNTIME_SCANNING = "scanning"
RUNTIME_PREPARING_MODELS = "preparing_models"
RUNTIME_STAGING = "staging"
RUNTIME_EMBEDDING = "embedding"
RUNTIME_REUSING_EMBEDDING = "reusing_embedding"
RUNTIME_PUBLISHING = "publishing"
RUNTIME_IDLE = "idle"
RUNTIME_STOPPED = "stopped"
RUNTIME_ERROR = "error"
_SQLITE_INT64_MAX = (1 << 63) - 1
_UINT64_MODULUS = 1 << 64


class SourceChangedError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingResult:
    embedding: tuple[float, ...]
    source_sample_rate_hz: int
    source_sample_count: int
    source_decoder_id: str = TORCHAUDIO_SOURCE_DECODER_ID


class EmbeddingBackend(Protocol):
    def prepare(self) -> None: ...

    def embed(self, snapshot: Path, source_sha256: str) -> EmbeddingResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class StagedSource:
    path: Path
    source_sha256: str
    size_bytes: int


@dataclass(frozen=True)
class BaselineResult:
    baseline_files: int
    settling_files: int
    bundle: BundlePublication


@dataclass(frozen=True)
class CycleReport:
    observed_files: int
    new_files: int
    changed_files: int
    ready_files: int
    indexed_files: int
    reused_embeddings: int
    failed_files: int
    blocked_files: int
    publication: BundlePublication | None


def _blob_from_embedding(values: tuple[float, ...]) -> bytes:
    if len(values) != 768:
        raise ValueError(f"embedding has {len(values)} values; expected 768")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("embedding contains a non-finite value")
    norm = math.sqrt(sum(value * value for value in values))
    if abs(norm - 1.0) > 0.001:
        raise ValueError(f"embedding norm is {norm:.8f}; expected 1.0 +/- 0.001")
    blob = struct.pack("<768f", *values)
    validate_embedding_blob(blob)
    return blob


def _sqlite_filesystem_identity(value: int) -> int:
    """Map an unsigned dev_t/ino_t bit pattern into SQLite's signed int64 domain."""
    identity = int(value)
    if identity < 0 or identity >= _UINT64_MODULUS:
        raise ValueError(f"filesystem identity {identity} is outside the uint64 domain")
    return identity if identity <= _SQLITE_INT64_MAX else identity - _UINT64_MODULUS


def _stat_signature(stat: os.stat_result) -> tuple[int, int, int, int]:
    return (
        _sqlite_filesystem_identity(stat.st_dev),
        _sqlite_filesystem_identity(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _row_signature(row: sqlite3.Row) -> tuple[int, int, int, int]:
    return (
        int(row["device_id"]), int(row["inode"]),
        int(row["size_bytes"]), int(row["mtime_ns"]),
    )


def _content_signature(stat: os.stat_result) -> tuple[int, int]:
    return int(stat.st_size), int(stat.st_mtime_ns)


def _row_content_signature(row: sqlite3.Row) -> tuple[int, int]:
    return int(row["size_bytes"]), int(row["mtime_ns"])


def _server_uri(root_id: str, relative_path: str) -> str:
    return f"server://{root_id}/{quote(relative_path, safe='/')}"


@contextlib.contextmanager
def server_instance_lock(state_db: Path) -> Iterator[None]:
    lock_path = Path(str(state_db) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another server indexer owns {lock_path}") from error
        yield


def server_writer_active(state_db: Path) -> bool:
    """Return whether an indexer owns the existing process lock without changing it."""
    lock_path = Path(str(state_db) + ".lock")
    try:
        handle = lock_path.open("rb")
    except FileNotFoundError:
        return False
    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False


class Clamp3ServerBackend:
    """Pinned full-track FP32 desktop CLaMP3 adapter."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.generator = None
        self._prepared = False

    def prepare(self) -> None:
        if self._prepared:
            return
        from huggingface_hub import hf_hub_download
        from .embeddings_clamp3 import CLAMP3_WEIGHTS_FILENAME, CLaMP3EmbeddingGenerator

        mert_path = Path(hf_hub_download(
            MERT_MODEL_ID, "pytorch_model.bin", revision=MERT_MODEL_REVISION
        ))
        clamp_path = Path(hf_hub_download(
            CLAMP3_REPOSITORY,
            CLAMP3_WEIGHTS_FILENAME,
            revision=CLAMP3_REVISION,
        ))
        observed = {
            "MERT": (sha256_file(mert_path), MERT_MODEL_SHA256),
            "CLaMP3": (sha256_file(clamp_path), CLAMP3_AUDIO_MODEL_SHA256),
        }
        for label, (actual, expected) in observed.items():
            if actual != expected:
                raise RuntimeError(f"{label} model hash {actual} does not match pinned {expected}")

        generator = CLaMP3EmbeddingGenerator(
            max_duration=None,
            batch_size=self.config.mert_batch_size,
            fp16=False,
            mert_revision=MERT_MODEL_REVISION,
            clamp3_revision=CLAMP3_REVISION,
        )
        try:
            generator.prepare_audio_models()
        except Exception:
            generator.unload_models()
            raise
        self.generator = generator
        self._prepared = True

    def embed(self, snapshot: Path, source_sha256: str) -> EmbeddingResult:
        self.prepare()
        assert self.generator is not None
        del source_sha256
        extraction = self.generator.extract_mert_features_with_source_span(snapshot)
        if extraction is None:
            raise RuntimeError("MERT extraction returned no features")
        features = extraction.features
        sample_rate = extraction.source_sample_rate_hz
        sample_count = extraction.source_sample_count
        decoder_id = extraction.source_decoder_id
        self._validate_features(features)
        embedding = self.generator.encode_mert_features(features)
        if embedding is None:
            raise RuntimeError("CLaMP3 encoding returned no embedding")
        if sample_rate <= 0 or sample_count <= 0:
            raise RuntimeError("decoder did not report a positive source sample span")
        if decoder_id not in SERVER_SOURCE_DECODER_IDS:
            raise RuntimeError(f"decoder returned unsupported provenance {decoder_id!r}")
        return EmbeddingResult(
            tuple(float(value) for value in embedding),
            sample_rate,
            sample_count,
            decoder_id,
        )

    @staticmethod
    def _validate_features(features: np.ndarray) -> None:
        if features.ndim != 3 or features.shape[0] != 1 or features.shape[2] != 768:
            raise ValueError(f"unexpected MERT feature shape {features.shape}")
        if features.shape[1] <= 0 or not np.isfinite(features).all():
            raise ValueError("MERT features are empty or non-finite")

    def close(self) -> None:
        if self.generator is not None:
            self.generator.unload_models()
            self.generator = None
        self._prepared = False


class ServerIndexer:
    def __init__(
        self,
        config: ServerConfig,
        backend: EmbeddingBackend | None = None,
        *,
        now_ns: Callable[[], int] = time.time_ns,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ):
        self.config = config
        self.backend = backend or Clamp3ServerBackend(config)
        self.now_ns = now_ns
        self.monotonic_ns = monotonic_ns
        config.state_db.parent.mkdir(parents=True, exist_ok=True)
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(config.state_db)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA synchronous = FULL")
        self.db.execute("PRAGMA busy_timeout = 5000")
        self.db.executescript(STATE_SCHEMA)
        self._migrate_state_schema()
        self._pin_state_contract()
        self._register_roots()
        self._recover_interrupted_rows()
        self._set_runtime_phase(RUNTIME_STARTING)

    def _migrate_state_schema(self) -> None:
        current_version = int(self.db.execute("PRAGMA user_version").fetchone()[0])
        if current_version > STATE_SCHEMA_VERSION:
            raise RuntimeError(
                f"state database schema {current_version} is newer than "
                f"supported schema {STATE_SCHEMA_VERSION}"
            )
        provenance_columns = {
            str(row["name"])
            for row in self.db.execute("PRAGMA table_info(server_track_provenance)")
        }
        if "source_decoder_id" not in provenance_columns:
            if current_version not in (0, 1):
                raise RuntimeError(
                    "state database is missing decoder provenance at an unsupported version"
                )
            self.db.execute(
                """
                ALTER TABLE server_track_provenance
                ADD COLUMN source_decoder_id TEXT NOT NULL
                    DEFAULT 'torchaudio-load-native-f32-v1'
                """
            )

        file_columns = {
            str(row["name"])
            for row in self.db.execute("PRAGMA table_info(server_files)")
        }
        timing_columns = {
            "last_processed_at_ns": "INTEGER",
            "last_processing_duration_ms": (
                "INTEGER CHECK(last_processing_duration_ms IS NULL "
                "OR last_processing_duration_ms >= 0)"
            ),
            "last_embedding_reused": (
                "INTEGER CHECK(last_embedding_reused IS NULL "
                "OR last_embedding_reused IN (0, 1))"
            ),
        }
        for name, declaration in timing_columns.items():
            if name not in file_columns:
                self.db.execute(
                    f"ALTER TABLE server_files ADD COLUMN {name} {declaration}"
                )

        metadata_version = self.db.execute(
            "SELECT value FROM server_state_metadata WHERE key = 'state_schema_version'"
        ).fetchone()
        if metadata_version is not None:
            try:
                parsed_metadata_version = int(metadata_version[0])
            except ValueError as error:
                raise RuntimeError("state database has an invalid schema receipt") from error
            if parsed_metadata_version not in (1, 2, STATE_SCHEMA_VERSION):
                raise RuntimeError(
                    f"state database metadata schema {parsed_metadata_version} cannot be migrated"
                )
            self.db.execute(
                """
                UPDATE server_state_metadata
                   SET value = ?
                 WHERE key = 'state_schema_version'
                """,
                (str(STATE_SCHEMA_VERSION),),
            )
        self.db.execute(f"PRAGMA user_version = {STATE_SCHEMA_VERSION}")
        self.db.commit()

    def _pin_state_contract(self) -> None:
        expected = {
            "state_schema_version": str(STATE_SCHEMA_VERSION),
            "embedding_spec_id": SERVER_EMBEDDING_SPEC_ID,
            "embedding_spec_sha256": server_embedding_spec_sha256(),
            "output_space_id": SERVER_OUTPUT_SPACE_ID,
        }
        existing = dict(self.db.execute("SELECT key, value FROM server_state_metadata"))
        for key, value in expected.items():
            if key in existing and existing[key] != value:
                raise RuntimeError(
                    f"state database {key} is {existing[key]!r}, expected {value!r}"
                )
            self.db.execute(
                "INSERT OR IGNORE INTO server_state_metadata(key, value) VALUES (?, ?)",
                (key, value),
            )
        self.db.commit()

    def _register_roots(self) -> None:
        for root in self.config.listen_roots:
            path = str(root.path.resolve())
            existing = self.db.execute(
                "SELECT canonical_path FROM server_roots WHERE root_id = ?", (root.root_id,)
            ).fetchone()
            if existing is not None and existing[0] != path:
                raise RuntimeError(
                    f"root {root.root_id} moved from {existing[0]} to {path}; migrate it explicitly"
                )
            self.db.execute(
                "INSERT OR IGNORE INTO server_roots(root_id, canonical_path) VALUES (?, ?)",
                (root.root_id, path),
            )
        self.db.commit()

    def _recover_interrupted_rows(self) -> None:
        placeholders = ",".join("?" for _ in RECOVERABLE_ACTIVE_STATES)
        self.db.execute(
            f"""
            UPDATE server_files
               SET state = ?, retry_after_ns = NULL,
                   error_code = 'process_interrupted',
                   error_message = 'Recovered unfinished work after process restart'
             WHERE state IN ({placeholders})
            """,
            (READY, *RECOVERABLE_ACTIVE_STATES),
        )
        self.db.commit()

    def _set_runtime_phase(
        self,
        phase: str,
        *,
        root_id: str | None = None,
        relative_path: str | None = None,
    ) -> None:
        self.db.execute(
            """
            UPDATE server_runtime
               SET phase = ?, phase_started_at_ns = ?,
                   current_root_id = ?, current_relative_path = ?
             WHERE singleton_id = 1
            """,
            (phase, self.now_ns(), root_id, relative_path),
        )
        self.db.commit()

    def _start_cycle(self, started_at_ns: int) -> None:
        self.db.execute(
            """
            UPDATE server_runtime
               SET phase = ?, phase_started_at_ns = ?, cycle_started_at_ns = ?,
                   current_root_id = NULL, current_relative_path = NULL,
                   next_poll_at_ns = NULL, last_error = NULL
             WHERE singleton_id = 1
            """,
            (RUNTIME_SCANNING, started_at_ns, started_at_ns),
        )
        self.db.commit()

    def _finish_cycle(self, completed_at_ns: int) -> None:
        self.db.execute(
            """
            UPDATE server_runtime
               SET phase = ?, phase_started_at_ns = ?,
                   current_root_id = NULL, current_relative_path = NULL,
                   last_cycle_completed_at_ns = ?, next_poll_at_ns = NULL,
                   last_error = NULL
             WHERE singleton_id = 1
            """,
            (RUNTIME_IDLE, completed_at_ns, completed_at_ns),
        )
        self.db.commit()

    def _record_runtime_error(self, error: Exception) -> None:
        observed_at_ns = self.now_ns()
        self.db.execute(
            """
            UPDATE server_runtime
               SET phase = ?, phase_started_at_ns = ?,
                   current_root_id = NULL, current_relative_path = NULL,
                   last_cycle_completed_at_ns = ?, last_error = ?
             WHERE singleton_id = 1
            """,
            (
                RUNTIME_ERROR,
                observed_at_ns,
                observed_at_ns,
                f"{type(error).__name__}: {error}"[:2000],
            ),
        )
        self.db.commit()

    def _schedule_next_poll(self, next_poll_at_ns: int) -> None:
        self.db.execute(
            """
            UPDATE server_runtime SET next_poll_at_ns = ?
             WHERE singleton_id = 1
            """,
            (next_poll_at_ns,),
        )
        self.db.commit()

    def baseline_existing(self) -> BaselineResult:
        self._set_runtime_phase(RUNTIME_BASELINING)
        start_ns = self.now_ns()
        for root in self.config.listen_roots:
            if not root.path.is_dir():
                raise FileNotFoundError(f"listen root not found: {root.path}")
            completed = self.db.execute(
                "SELECT baseline_completed_at_ns FROM server_roots WHERE root_id = ?",
                (root.root_id,),
            ).fetchone()[0]
            if completed is not None:
                raise RuntimeError(f"root {root.root_id} already has a baseline")

        baseline_count = 0
        settling_count = 0
        self.db.execute("BEGIN IMMEDIATE")
        try:
            for root in self.config.listen_roots:
                baseline_candidates: list[
                    tuple[Path, str, tuple[int, int, int, int]]
                ] = []
                for path in sorted(scan_music_directory(root.path)):
                    stat = path.stat()
                    signature = _stat_signature(stat)
                    relative = path.relative_to(root.path).as_posix()
                    # A qBittorrent completion that changes during the baseline walk is new work.
                    state = SETTLING if int(stat.st_ctime_ns) >= start_ns else BASELINE
                    if state == BASELINE:
                        baseline_count += 1
                        baseline_candidates.append((path, relative, signature))
                    else:
                        settling_count += 1
                    self.db.execute(
                        """
                        INSERT INTO server_files(
                            root_id, relative_path, device_id, inode, size_bytes, mtime_ns,
                            stable_since_ns, last_seen_ns, present, state
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                        """,
                        (root.root_id, relative, *signature, start_ns, start_ns, state),
                    )
                # Close the race where a writer changes an early file while the rest of the
                # initial tree is still being walked.
                for path, relative, initial_signature in baseline_candidates:
                    try:
                        current = path.stat()
                    except FileNotFoundError:
                        self.db.execute(
                            """
                            UPDATE server_files SET present = 0, state = ?
                             WHERE root_id = ? AND relative_path = ?
                            """,
                            (SETTLING, root.root_id, relative),
                        )
                        baseline_count -= 1
                        settling_count += 1
                        continue
                    if (_stat_signature(current) != initial_signature or
                            int(current.st_ctime_ns) >= start_ns):
                        current_signature = _stat_signature(current)
                        self.db.execute(
                            """
                            UPDATE server_files
                               SET device_id = ?, inode = ?, size_bytes = ?, mtime_ns = ?,
                                   stable_since_ns = ?, last_seen_ns = ?, state = ?
                             WHERE root_id = ? AND relative_path = ?
                            """,
                            (*current_signature, start_ns, start_ns, SETTLING,
                             root.root_id, relative),
                        )
                        baseline_count -= 1
                        settling_count += 1
                self.db.execute(
                    "UPDATE server_roots SET baseline_completed_at_ns = ? WHERE root_id = ?",
                    (self.now_ns(), root.root_id),
                )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        publication = publish_bundle(self.db, self.config.bundle_db)
        self._finish_cycle(self.now_ns())
        return BaselineResult(baseline_count, settling_count, publication)

    def require_baselined(self) -> None:
        incomplete = self.db.execute(
            "SELECT root_id FROM server_roots WHERE baseline_completed_at_ns IS NULL"
        ).fetchall()
        if incomplete:
            raise RuntimeError(
                "run `poweramp-indexer server init --baseline-existing` before polling: " +
                ", ".join(row[0] for row in incomplete)
            )

    def discover(self, now_ns: int | None = None) -> tuple[int, int, int, int]:
        self.require_baselined()
        now = self.now_ns() if now_ns is None else now_ns
        observed_count = 0
        new_count = 0
        changed_count = 0
        self.db.execute("BEGIN IMMEDIATE")
        try:
            for root in self.config.listen_roots:
                if not root.path.is_dir():
                    raise FileNotFoundError(f"listen root not found: {root.path}")
                observed = []
                for path in sorted(scan_music_directory(root.path)):
                    try:
                        observed.append((path, path.stat()))
                    except FileNotFoundError:
                        continue
                observed_paths = {path.relative_to(root.path).as_posix() for path, _ in observed}
                existing_rows = self.db.execute(
                    "SELECT * FROM server_files WHERE root_id = ?",
                    (root.root_id,),
                ).fetchall()
                existing_by_path = {
                    str(row["relative_path"]): row
                    for row in existing_rows
                }
                missing_paths = set(existing_by_path).difference(observed_paths)
                missing_by_signature: dict[
                    tuple[int, int, int, int],
                    list[sqlite3.Row],
                ] = {}
                for relative in missing_paths:
                    row = existing_by_path[relative]
                    missing_by_signature.setdefault(_row_signature(row), []).append(row)
                renamed_from: set[str] = set()
                for path, stat in observed:
                    observed_count += 1
                    relative = path.relative_to(root.path).as_posix()
                    signature = _stat_signature(stat)
                    existing = existing_by_path.get(relative)
                    if existing is None:
                        rename_candidates = [
                            row for row in missing_by_signature.get(signature, ())
                            if str(row["relative_path"]) not in renamed_from
                        ]
                        rename = (
                            rename_candidates[0]
                            if len(rename_candidates) == 1
                            else None
                        )
                        if rename is not None:
                            renamed_from.add(str(rename["relative_path"]))
                            self.db.execute(
                                """
                                UPDATE server_files
                                   SET relative_path = ?, present = 1, last_seen_ns = ?
                                 WHERE root_id = ? AND relative_path = ?
                                """,
                                (relative, now, root.root_id, rename["relative_path"]),
                            )
                            if rename["track_id"] is not None:
                                bundle_relative = unicodedata.normalize("NFC", relative)
                                self.db.execute(
                                    """
                                    UPDATE server_track_provenance SET relative_path = ?
                                     WHERE track_id = ?
                                    """,
                                    (bundle_relative, rename["track_id"]),
                                )
                                self.db.execute(
                                    "UPDATE tracks SET file_path = ? WHERE id = ?",
                                    (
                                        _server_uri(root.root_id, bundle_relative),
                                        rename["track_id"],
                                    ),
                                )
                            continue
                        new_count += 1
                        self.db.execute(
                            """
                            INSERT INTO server_files(
                                root_id, relative_path, device_id, inode, size_bytes, mtime_ns,
                                stable_since_ns, last_seen_ns, present, state
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                            """,
                            (root.root_id, relative, *signature, now, now, SETTLING),
                        )
                        continue

                    if _row_content_signature(existing) != _content_signature(stat):
                        changed_count += 1
                        self.db.execute(
                            """
                            UPDATE server_files
                               SET device_id = ?, inode = ?, size_bytes = ?, mtime_ns = ?,
                                   stable_since_ns = ?, last_seen_ns = ?, present = 1,
                                   state = ?, content_sha256 = NULL,
                                   attempt_count = 0, retry_after_ns = NULL,
                                   error_code = NULL, error_message = NULL
                             WHERE root_id = ? AND relative_path = ?
                            """,
                            (*signature, now, now, SETTLING, root.root_id, relative),
                        )
                    else:
                        state = existing["state"]
                        if state == RETRY_WAIT and (
                            existing["retry_after_ns"] is None or existing["retry_after_ns"] <= now
                        ):
                            state = READY
                        identity_changed = (
                            int(existing["device_id"]) != signature[0] or
                            int(existing["inode"]) != signature[1]
                        )
                        if identity_changed or int(existing["present"]) != 1 or (
                            state != existing["state"]
                        ):
                            self.db.execute(
                                """
                                UPDATE server_files
                                   SET device_id = ?, inode = ?, present = 1,
                                       last_seen_ns = ?, state = ?
                                 WHERE root_id = ? AND relative_path = ?
                                """,
                                (
                                    signature[0], signature[1], now, state,
                                    root.root_id, relative,
                                ),
                            )

                disappeared = [
                    (root.root_id, relative)
                    for relative in missing_paths.difference(renamed_from)
                    if int(existing_by_path[relative]["present"]) == 1
                ]
                self.db.executemany(
                    """
                    UPDATE server_files SET present = 0
                     WHERE root_id = ? AND relative_path = ?
                    """,
                    disappeared,
                )

            cutoff = now - int(self.config.settle_seconds * 1_000_000_000)
            self.db.execute(
                """
                UPDATE server_files SET state = ?
                 WHERE state = ? AND present = 1 AND stable_since_ns <= ?
                """,
                (READY, SETTLING, cutoff),
            )
            ready_count = self.db.execute(
                "SELECT COUNT(*) FROM server_files WHERE state = ? AND present = 1", (READY,)
            ).fetchone()[0]
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return observed_count, new_count, changed_count, ready_count

    def cycle(self, now_ns: int | None = None) -> CycleReport:
        now = self.now_ns() if now_ns is None else now_ns
        self._start_cycle(now)
        try:
            observed, new, changed, ready = self.discover(now)
            indexed, reused, failed, blocked = self._process_ready(now)
            self._set_runtime_phase(RUNTIME_PUBLISHING)
            publication = ensure_bundle_current(self.db, self.config.bundle_db)
        except Exception as error:
            self._record_runtime_error(error)
            raise
        self._finish_cycle(self.now_ns())
        return CycleReport(
            observed, new, changed, ready, indexed, reused, failed, blocked, publication
        )

    def _process_ready(self, now_ns: int) -> tuple[int, int, int, int]:
        rows = self.db.execute(
            """
            SELECT * FROM server_files
             WHERE state = ? AND present = 1
             ORDER BY stable_since_ns, root_id, relative_path
             LIMIT ?
            """,
            (READY, self.config.max_files_per_cycle),
        ).fetchall()
        if not rows:
            return 0, 0, 0, 0

        # Model/bootstrap failures are global. Leave every file READY rather than poisoning rows.
        self._set_runtime_phase(RUNTIME_PREPARING_MODELS)
        self.backend.prepare()
        indexed = 0
        reused = 0
        failed = 0
        blocked = 0
        for row in rows:
            staged: StagedSource | None = None
            processing_started = self.monotonic_ns()
            try:
                self._set_runtime_phase(
                    RUNTIME_STAGING,
                    root_id=str(row["root_id"]),
                    relative_path=str(row["relative_path"]),
                )
                self._set_state(row, STAGING)
                staged = self._stage(row)
                self.db.execute(
                    """
                    UPDATE server_files SET state = ?, content_sha256 = ?
                     WHERE root_id = ? AND relative_path = ?
                    """,
                    (INDEXING, staged.source_sha256, row["root_id"], row["relative_path"]),
                )
                self.db.commit()
                prior = self._reusable_embedding(staged.source_sha256)
                if prior is None:
                    self._set_runtime_phase(
                        RUNTIME_EMBEDDING,
                        root_id=str(row["root_id"]),
                        relative_path=str(row["relative_path"]),
                    )
                    result = self.backend.embed(staged.path, staged.source_sha256)
                    blob = _blob_from_embedding(result.embedding)
                    sample_rate = result.source_sample_rate_hz
                    sample_count = result.source_sample_count
                    decoder_id = result.source_decoder_id
                    embedding_reused = False
                else:
                    self._set_runtime_phase(
                        RUNTIME_REUSING_EMBEDDING,
                        root_id=str(row["root_id"]),
                        relative_path=str(row["relative_path"]),
                    )
                    blob = bytes(prior["embedding"])
                    validate_embedding_blob(blob)
                    sample_rate = int(prior["source_sample_rate_hz"])
                    sample_count = int(prior["source_sample_count"])
                    decoder_id = str(prior["source_decoder_id"])
                    reused += 1
                    embedding_reused = True
                if sample_rate <= 0 or sample_count <= 0:
                    raise ValueError("embedding backend returned an invalid source sample span")
                if decoder_id not in SERVER_SOURCE_DECODER_IDS:
                    raise ValueError(
                        f"embedding backend returned unsupported decoder {decoder_id!r}"
                    )
                metadata = extract_metadata(staged.path)
                self._require_source_unchanged(row)
                self._commit_indexed(
                    row,
                    metadata,
                    staged,
                    blob,
                    sample_rate,
                    sample_count,
                    decoder_id,
                    processed_at_ns=self.now_ns(),
                    processing_duration_ms=max(
                        0, (self.monotonic_ns() - processing_started) // 1_000_000
                    ),
                    embedding_reused=embedding_reused,
                )
                indexed += 1
            except SourceChangedError:
                self._reset_changed_source(row, now_ns)
            except Exception as error:
                logger.exception(
                    "Server indexing failed for %s/%s", row["root_id"], row["relative_path"]
                )
                if self._record_failure(row, now_ns, error):
                    blocked += 1
                failed += 1
            finally:
                if staged is not None:
                    staged.path.unlink(missing_ok=True)
                    with contextlib.suppress(OSError):
                        staged.path.parent.rmdir()
        return indexed, reused, failed, blocked

    def _stage(self, row: sqlite3.Row) -> StagedSource:
        source = self._source_path(row)
        before = source.stat()
        if _stat_signature(before) != _row_signature(row):
            raise SourceChangedError("source changed before staging")
        temporary_dir = self.config.cache_dir / "staging-tmp"
        temporary_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix="source-", dir=temporary_dir)
        digest = hashlib.sha256()
        try:
            with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
                while chunk := input_handle.read(1024 * 1024):
                    digest.update(chunk)
                    output_handle.write(chunk)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            after = source.stat()
            if _stat_signature(after) != _stat_signature(before):
                raise SourceChangedError("source changed while staging")
            source_hash = digest.hexdigest()
            destination = self.config.cache_dir / "staged" / source_hash / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if (destination.stat().st_size != before.st_size or
                        sha256_file(destination) != source_hash):
                    raise RuntimeError(f"staged content collision at {destination}")
                Path(temporary_name).unlink(missing_ok=True)
            else:
                os.replace(temporary_name, destination)
            return StagedSource(destination, source_hash, int(before.st_size))
        finally:
            Path(temporary_name).unlink(missing_ok=True)

    def _require_source_unchanged(self, row: sqlite3.Row) -> None:
        try:
            current = self._source_path(row).stat()
        except FileNotFoundError as error:
            raise SourceChangedError("source disappeared during indexing") from error
        if _stat_signature(current) != _row_signature(row):
            raise SourceChangedError("source changed during indexing")

    def _commit_indexed(
        self,
        row: sqlite3.Row,
        metadata: TrackMetadata,
        staged: StagedSource,
        embedding: bytes,
        source_sample_rate_hz: int,
        source_sample_count: int,
        source_decoder_id: str,
        *,
        processed_at_ns: int,
        processing_duration_ms: int,
        embedding_reused: bool,
    ) -> None:
        embedding_sha = hashlib.sha256(embedding).hexdigest()
        bundle_relative_path = unicodedata.normalize("NFC", row["relative_path"])
        uri = _server_uri(row["root_id"], bundle_relative_path)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if row["track_id"] is None:
                cursor = self.db.execute(
                    """
                    INSERT INTO tracks(
                        metadata_key, filename_key, artist, album, title, duration_ms,
                        file_path, cluster_id, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'server')
                    """,
                    (metadata.metadata_key, metadata.filename_key, metadata.artist,
                     metadata.album, metadata.title, metadata.duration_ms, uri),
                )
                track_id = int(cursor.lastrowid)
                self.db.execute(
                    "INSERT INTO embeddings_clamp3(track_id, embedding) VALUES (?, ?)",
                    (track_id, embedding),
                )
                self.db.execute(
                    """
                    INSERT INTO server_track_provenance(
                        track_id, root_id, relative_path, source_sha256, source_size_bytes,
                        source_sample_rate_hz, source_sample_count, source_decoder_id,
                        span_start_sample, span_end_sample_exclusive, embedding_sha256,
                        embedding_spec_id, output_space_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                    """,
                    (track_id, row["root_id"], bundle_relative_path, staged.source_sha256,
                     staged.size_bytes, source_sample_rate_hz, source_sample_count,
                     source_decoder_id, source_sample_count, embedding_sha,
                     SERVER_EMBEDDING_SPEC_ID, SERVER_OUTPUT_SPACE_ID),
                )
            else:
                track_id = int(row["track_id"])
                updated = self.db.execute(
                    """
                    UPDATE tracks
                       SET metadata_key = ?, filename_key = ?, artist = ?, album = ?,
                           title = ?, duration_ms = ?, file_path = ?, source = 'server'
                     WHERE id = ?
                    """,
                    (metadata.metadata_key, metadata.filename_key, metadata.artist,
                     metadata.album, metadata.title, metadata.duration_ms, uri, track_id),
                )
                if updated.rowcount != 1:
                    raise RuntimeError(f"superseded track {track_id} is missing")
                updated_embedding = self.db.execute(
                    "UPDATE embeddings_clamp3 SET embedding = ? WHERE track_id = ?",
                    (embedding, track_id),
                )
                if updated_embedding.rowcount != 1:
                    raise RuntimeError(f"superseded embedding {track_id} is missing")
                updated_provenance = self.db.execute(
                    """
                    UPDATE server_track_provenance
                       SET root_id = ?, relative_path = ?, source_sha256 = ?,
                           source_size_bytes = ?, source_sample_rate_hz = ?,
                           source_sample_count = ?, source_decoder_id = ?,
                           span_start_sample = 0,
                           span_end_sample_exclusive = ?, embedding_sha256 = ?,
                           embedding_spec_id = ?, output_space_id = ?
                     WHERE track_id = ?
                    """,
                    (row["root_id"], bundle_relative_path, staged.source_sha256,
                     staged.size_bytes, source_sample_rate_hz, source_sample_count,
                     source_decoder_id, source_sample_count, embedding_sha,
                     SERVER_EMBEDDING_SPEC_ID, SERVER_OUTPUT_SPACE_ID, track_id),
                )
                if updated_provenance.rowcount != 1:
                    raise RuntimeError(f"superseded provenance {track_id} is missing")
            self.db.execute(
                """
                UPDATE server_files
                   SET state = ?, content_sha256 = ?, track_id = ?, attempt_count = 0,
                       retry_after_ns = NULL, error_code = NULL, error_message = NULL,
                       last_processed_at_ns = ?, last_processing_duration_ms = ?,
                       last_embedding_reused = ?
                 WHERE root_id = ? AND relative_path = ?
                """,
                (
                    INDEXED,
                    staged.source_sha256,
                    track_id,
                    processed_at_ns,
                    processing_duration_ms,
                    int(embedding_reused),
                    row["root_id"],
                    row["relative_path"],
                ),
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def _reusable_embedding(self, source_sha256: str) -> sqlite3.Row | None:
        return self.db.execute(
            """
            SELECT e.embedding, p.source_sample_rate_hz, p.source_sample_count,
                   p.source_decoder_id
              FROM server_track_provenance p
              JOIN embeddings_clamp3 e ON e.track_id = p.track_id
             WHERE p.source_sha256 = ? AND p.embedding_spec_id = ?
             ORDER BY p.track_id LIMIT 1
            """,
            (source_sha256, SERVER_EMBEDDING_SPEC_ID),
        ).fetchone()

    def _record_failure(self, row: sqlite3.Row, now_ns: int, error: Exception) -> bool:
        current = self.db.execute(
            "SELECT attempt_count FROM server_files WHERE root_id = ? AND relative_path = ?",
            (row["root_id"], row["relative_path"]),
        ).fetchone()
        attempts = int(current[0]) + 1
        blocked = attempts >= self.config.max_attempts
        if blocked:
            state = BLOCKED
            retry_after = None
        else:
            state = RETRY_WAIT
            delay_index = min(attempts - 1, len(self.config.retry_delays_seconds) - 1)
            retry_after = now_ns + int(
                self.config.retry_delays_seconds[delay_index] * 1_000_000_000
            )
        self.db.execute(
            """
            UPDATE server_files
               SET state = ?, attempt_count = ?, retry_after_ns = ?,
                   error_code = ?, error_message = ?
             WHERE root_id = ? AND relative_path = ?
            """,
            (state, attempts, retry_after, type(error).__name__, str(error)[:2000],
             row["root_id"], row["relative_path"]),
        )
        self.db.commit()
        return blocked

    def _reset_changed_source(self, row: sqlite3.Row, now_ns: int) -> None:
        source = self._source_path(row)
        try:
            stat = source.stat()
        except FileNotFoundError:
            self.db.execute(
                """
                UPDATE server_files SET present = 0, state = ?, content_sha256 = NULL,
                    attempt_count = 0, retry_after_ns = NULL
                 WHERE root_id = ? AND relative_path = ?
                """,
                (SETTLING, row["root_id"], row["relative_path"]),
            )
        else:
            signature = _stat_signature(stat)
            self.db.execute(
                """
                UPDATE server_files
                   SET device_id = ?, inode = ?, size_bytes = ?, mtime_ns = ?,
                       stable_since_ns = ?, last_seen_ns = ?, present = 1, state = ?,
                       content_sha256 = NULL, attempt_count = 0,
                       retry_after_ns = NULL, error_code = NULL, error_message = NULL
                 WHERE root_id = ? AND relative_path = ?
                """,
                (*signature, now_ns, now_ns, SETTLING,
                 row["root_id"], row["relative_path"]),
            )
        self.db.commit()

    def _block(self, row: sqlite3.Row, code: str, detail: str) -> None:
        self.db.execute(
            """
            UPDATE server_files
               SET state = ?, attempt_count = ?, retry_after_ns = NULL,
                   error_code = ?, error_message = ?
             WHERE root_id = ? AND relative_path = ?
            """,
            (BLOCKED, self.config.max_attempts, code, detail,
             row["root_id"], row["relative_path"]),
        )
        self.db.commit()

    def _set_state(self, row: sqlite3.Row, state: str) -> None:
        self.db.execute(
            "UPDATE server_files SET state = ? WHERE root_id = ? AND relative_path = ?",
            (state, row["root_id"], row["relative_path"]),
        )
        self.db.commit()

    def _source_path(self, row: sqlite3.Row) -> Path:
        root = next(root for root in self.config.listen_roots if root.root_id == row["root_id"])
        return root.path / Path(row["relative_path"])

    def retry_failed(self) -> int:
        cursor = self.db.execute(
            """
            UPDATE server_files
               SET state = ?, attempt_count = 0, retry_after_ns = NULL,
                   error_code = NULL, error_message = NULL
             WHERE present = 1 AND state IN (?, ?)
            """,
            (READY, RETRY_WAIT, BLOCKED),
        )
        self.db.commit()
        return cursor.rowcount

    def status(self) -> dict[str, object]:
        return _status_document(self.db, self.config)

    def run_forever(self, stop_requested: Callable[[], bool] = lambda: False) -> None:
        self.require_baselined()
        while not stop_requested():
            started = time.monotonic()
            try:
                report = self.cycle()
                logger.info("Server indexing cycle: %s", report)
            except Exception:
                logger.exception("Server indexing cycle failed; durable state was retained")
            remaining = self.config.poll_interval_seconds - (time.monotonic() - started)
            sleep_seconds = max(0.0, remaining)
            self._schedule_next_poll(
                self.now_ns() + int(sleep_seconds * 1_000_000_000)
            )
            deadline = time.monotonic() + sleep_seconds
            while not stop_requested() and time.monotonic() < deadline:
                time.sleep(min(0.5, deadline - time.monotonic()))
        self._set_runtime_phase(RUNTIME_STOPPED)

    def close(self) -> None:
        try:
            self.backend.close()
        finally:
            self.db.close()

    def __enter__(self) -> "ServerIndexer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _status_document(
    database: sqlite3.Connection,
    config: ServerConfig,
) -> dict[str, object]:
    captured_at_ns = time.time_ns()
    state_schema_version = int(database.execute("PRAGMA user_version").fetchone()[0])
    roots = [
        {
            "id": row["root_id"],
            "path": row["canonical_path"],
            "baselined": row["baseline_completed_at_ns"] is not None,
        }
        for row in database.execute(
            "SELECT root_id, canonical_path, baseline_completed_at_ns "
            "FROM server_roots ORDER BY root_id"
        )
    ]
    present_states = {
        row["state"]: row["count"]
        for row in database.execute(
            """
            SELECT state, COUNT(*) AS count
              FROM server_files
             WHERE present = 1
             GROUP BY state
             ORDER BY state
            """
        )
    }
    missing = database.execute(
        "SELECT COUNT(*) FROM server_files WHERE present = 0"
    ).fetchone()[0]
    present = sum(int(value) for value in present_states.values())
    active = sum(
        int(present_states.get(state, 0))
        for state in (STAGING, INDEXING)
    )
    queue = {
        "ready": int(present_states.get(READY, 0)),
        "settling": int(present_states.get(SETTLING, 0)),
        "active": active,
        "retrying": int(present_states.get(RETRY_WAIT, 0)),
        "blocked": int(present_states.get(BLOCKED, 0)),
    }

    last_publication = database.execute(
        """
        SELECT generation_id, track_count, bundle_sha256, published_at_ns
          FROM server_publications ORDER BY published_at_ns DESC LIMIT 1
        """
    ).fetchone()
    try:
        bundle_stat = config.bundle_db.stat()
    except FileNotFoundError:
        bundle_exists = False
        bundle_size_bytes = None
    else:
        bundle_exists = True
        bundle_size_bytes = int(bundle_stat.st_size)

    tables = {
        str(row[0])
        for row in database.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if "server_runtime" in tables:
        runtime_row = database.execute(
            """
            SELECT phase, phase_started_at_ns, cycle_started_at_ns,
                   current_root_id, current_relative_path,
                   last_cycle_completed_at_ns, next_poll_at_ns, last_error
              FROM server_runtime
             WHERE singleton_id = 1
            """
        ).fetchone()
    else:
        runtime_row = None
    runtime = dict(runtime_row) if runtime_row is not None else {
        "phase": "unknown",
        "phase_started_at_ns": None,
        "cycle_started_at_ns": None,
        "current_root_id": None,
        "current_relative_path": None,
        "last_cycle_completed_at_ns": None,
        "next_poll_at_ns": None,
        "last_error": None,
    }

    file_columns = {
        str(row["name"])
        for row in database.execute("PRAGMA table_info(server_files)")
    }
    timing_fields = {
        "last_processed_at_ns",
        "last_processing_duration_ms",
        "last_embedding_reused",
    }
    if timing_fields.issubset(file_columns):
        duration_rows = database.execute(
            """
            SELECT last_processing_duration_ms
              FROM server_files
             WHERE last_processing_duration_ms IS NOT NULL
               AND last_embedding_reused = 0
             ORDER BY last_processed_at_ns DESC
             LIMIT 128
            """
        ).fetchall()
        duration_seconds = sorted(
            float(row[0]) / 1000.0 for row in duration_rows
        )
    else:
        duration_seconds = []

    publication_batch_seconds: list[float] = []
    if not duration_seconds:
        # Schema-v2 ledgers predate per-track telemetry. Consecutive full-batch
        # publications are the one durable measurement available after migration.
        # Partial batches are excluded because their interval may mostly be idle time.
        publication_rows = list(reversed(database.execute(
            """
            SELECT track_count, published_at_ns
              FROM server_publications
             ORDER BY published_at_ns DESC
             LIMIT 33
            """
        ).fetchall()))
        for previous, current in zip(publication_rows, publication_rows[1:]):
            added = int(current["track_count"]) - int(previous["track_count"])
            elapsed_ns = (
                int(current["published_at_ns"]) -
                int(previous["published_at_ns"])
            )
            if added == config.max_files_per_cycle and elapsed_ns > 0:
                publication_batch_seconds.append(
                    elapsed_ns / 1_000_000_000 / added
                )

    timing_seconds = (
        duration_seconds
        if duration_seconds
        else sorted(publication_batch_seconds)
    )
    if timing_seconds:
        midpoint = len(timing_seconds) // 2
        if len(timing_seconds) % 2:
            median_seconds = timing_seconds[midpoint]
        else:
            median_seconds = (
                timing_seconds[midpoint - 1] + timing_seconds[midpoint]
            ) / 2.0
        p90_index = max(0, math.ceil(len(timing_seconds) * 0.9) - 1)
        p90_seconds = timing_seconds[p90_index]
    else:
        median_seconds = None
        p90_seconds = None

    estimated_tracks = (
        queue["ready"] + queue["settling"] + queue["active"] + queue["retrying"]
    )
    estimated_seconds = (
        round(float(median_seconds) * estimated_tracks)
        if median_seconds is not None and estimated_tracks > 0
        else None
    )

    failures = [
        dict(row)
        for row in database.execute(
            """
            SELECT root_id, relative_path, state, attempt_count, error_code, error_message
              FROM server_files
             WHERE present = 1 AND state IN (?, ?)
             ORDER BY last_seen_ns DESC LIMIT 20
            """,
            (RETRY_WAIT, BLOCKED),
        )
    ]
    return {
        "captured_at_ns": captured_at_ns,
        "writer_active": server_writer_active(config.state_db),
        "state": {
            "path": str(config.state_db),
            "schema_version": state_schema_version,
        },
        "roots": roots,
        "runtime": runtime,
        "library": {
            "present": present,
            "embedded": int(present_states.get(INDEXED, 0)),
            "baseline": int(present_states.get(BASELINE, 0)),
            "queue": queue,
            "missing": int(missing),
            "present_states": present_states,
        },
        "bundle": {
            "path": str(config.bundle_db),
            "exists": bundle_exists,
            "size_bytes": bundle_size_bytes,
            "last_publication": (
                dict(last_publication) if last_publication else None
            ),
        },
        "timing": {
            "non_reused_sample_count": len(duration_seconds),
            "publication_batch_sample_count": len(publication_batch_seconds),
            "source": (
                "track_telemetry"
                if duration_seconds
                else (
                    "full_publication_batches"
                    if publication_batch_seconds
                    else None
                )
            ),
            "median_seconds_per_track": median_seconds,
            "p90_seconds_per_track": p90_seconds,
            "estimated_pending_tracks": estimated_tracks,
            "estimated_remaining_seconds": estimated_seconds,
        },
        "failures": failures,
    }


def read_server_status(config: ServerConfig) -> dict[str, object]:
    """Read a consistent status snapshot without taking the writer's process lock."""
    if not config.state_db.is_file():
        raise FileNotFoundError(f"server state database not found: {config.state_db}")
    database = sqlite3.connect(
        f"{config.state_db.resolve().as_uri()}?mode=ro",
        uri=True,
        timeout=5.0,
    )
    database.row_factory = sqlite3.Row
    try:
        database.execute("BEGIN")
        return _status_document(database, config)
    finally:
        database.close()

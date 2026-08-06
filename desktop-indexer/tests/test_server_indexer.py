from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from poweramp_indexer import server_indexer as server_indexer_module
from poweramp_indexer.server_bundle import validate_bundle
from poweramp_indexer.server_config import (
    SERVER_EMBEDDING_SPEC_ID,
    TORCHAUDIO_SOURCE_DECODER_ID,
    ListenRoot,
    ServerConfig,
)
from poweramp_indexer.server_indexer import (
    BASELINE,
    BLOCKED,
    INDEXED,
    READY,
    RETRY_WAIT,
    SETTLING,
    STAGING,
    EmbeddingResult,
    ServerIndexer,
    read_server_status,
)


class Clock:
    def __init__(self) -> None:
        self.value = time.time_ns() + 10_000_000

    def __call__(self) -> int:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += int(seconds * 1_000_000_000)


class FakeBackend:
    def __init__(self, fail: bool = False, mutate: Path | None = None):
        self.fail = fail
        self.mutate = mutate
        self.prepare_calls = 0
        self.embed_calls = 0
        self.closed = False

    def prepare(self) -> None:
        self.prepare_calls += 1

    def embed(self, snapshot: Path, source_sha256: str) -> EmbeddingResult:
        self.embed_calls += 1
        if self.mutate is not None:
            self.mutate.write_bytes(self.mutate.read_bytes() + b"changed")
        if self.fail:
            raise RuntimeError("fixture decode failure")
        return EmbeddingResult((1.0,) + (0.0,) * 767, 44_100, 44_100)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def layout(tmp_path: Path):
    listen = tmp_path / "listen"
    runtime = tmp_path / "runtime"
    listen.mkdir()
    runtime.mkdir()
    config = ServerConfig(
        state_db=runtime / "state.db",
        bundle_db=runtime / "export" / "server-embeddings.db",
        cache_dir=runtime / "cache",
        listen_roots=(ListenRoot("musicnew", listen),),
        settle_seconds=10,
        retry_delays_seconds=(0,),
        max_attempts=2,
    )
    return listen, config


def state_row(indexer: ServerIndexer, relative_path: str):
    return indexer.db.execute(
        "SELECT * FROM server_files WHERE root_id = 'musicnew' AND relative_path = ?",
        (relative_path,),
    ).fetchone()


def test_explicit_baseline_creates_no_embeddings_and_unchanged_files_stay_baseline(layout):
    listen, config = layout
    (listen / "existing.mp3").write_bytes(b"already represented on the phone")
    clock = Clock()
    backend = FakeBackend()

    with ServerIndexer(config, backend, now_ns=clock) as indexer:
        result = indexer.baseline_existing()
        assert result.baseline_files == 1
        assert result.settling_files == 0
        assert result.bundle.track_count == 0
        report = indexer.cycle()
        assert report.indexed_files == 0
        assert backend.prepare_calls == 0
        assert indexer.db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 0

    publication = validate_bundle(config.bundle_db)
    assert publication.track_count == 0


def test_baseline_normalizes_unsigned_64_bit_mergerfs_inode(layout, monkeypatch):
    listen, config = layout
    source = listen / "existing.flac"
    source.write_bytes(b"already represented on the phone")
    clock = Clock()
    backend = FakeBackend()
    actual_stat = Path.stat
    source_stat = actual_stat(source)
    unsigned_device = (1 << 63) + 17
    unsigned_inode = (1 << 63) + 12345

    def mergerfs_stat(path: Path, *args, **kwargs):
        result = actual_stat(path, *args, **kwargs)
        if path == source:
            return SimpleNamespace(
                st_dev=unsigned_device,
                st_ino=unsigned_inode,
                st_size=source_stat.st_size,
                st_mtime_ns=source_stat.st_mtime_ns,
                st_ctime_ns=clock.value - 1,
            )
        return result

    monkeypatch.setattr(Path, "stat", mergerfs_stat)
    monkeypatch.setattr(
        server_indexer_module,
        "scan_music_directory",
        lambda _root: iter((source,)),
    )

    with ServerIndexer(config, backend, now_ns=clock) as indexer:
        result = indexer.baseline_existing()
        assert result.baseline_files == 1
        row = state_row(indexer, "existing.flac")
        assert row["device_id"] == unsigned_device - (1 << 64)
        assert row["inode"] == unsigned_inode - (1 << 64)
        assert indexer.cycle().indexed_files == 0


def test_new_file_settles_then_publishes_and_identical_bytes_reuse_embedding(layout):
    listen, config = layout
    clock = Clock()
    backend = FakeBackend()
    payload = b"a complete immutable audio fixture"

    with ServerIndexer(config, backend, now_ns=clock) as indexer:
        indexer.baseline_existing()
        (listen / "Album").mkdir()
        (listen / "Album" / "one.mp3").write_bytes(payload)
        first = indexer.cycle()
        assert first.new_files == 1
        assert first.indexed_files == 0
        assert state_row(indexer, "Album/one.mp3")["state"] == SETTLING

        clock.advance(11)
        second = indexer.cycle()
        assert second.indexed_files == 1
        assert second.reused_embeddings == 0
        assert state_row(indexer, "Album/one.mp3")["state"] == INDEXED
        assert backend.embed_calls == 1

        (listen / "Album" / "copy.mp3").write_bytes(payload)
        indexer.cycle()
        clock.advance(11)
        third = indexer.cycle()
        assert third.indexed_files == 1
        assert third.reused_embeddings == 1
        assert backend.embed_calls == 1

    bundle = validate_bundle(config.bundle_db)
    assert bundle.track_count == 2
    with sqlite3.connect(config.bundle_db) as db:
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )}
        assert "clusters" not in tables
        assert "binary_data" not in tables
        metadata = dict(db.execute("SELECT key, value FROM metadata"))
        assert metadata["server_bundle_embedding_spec_id"] == "poweramp-clamp3-server-v1"
        assert metadata["server_bundle_graph_included"] == "false"
        rows = db.execute(
            "SELECT file_path, source FROM tracks ORDER BY file_path"
        ).fetchall()
        assert all(row[0].startswith("server://musicnew/Album/") for row in rows)
        assert all(row[1] == "server" for row in rows)


def test_same_filesystem_rename_preserves_baseline_or_embedding_identity(layout):
    listen, base_config = layout
    config = replace(base_config, settle_seconds=0)
    baseline_source = listen / "old-name.mp3"
    baseline_source.write_bytes(b"baseline bytes")
    clock = Clock()
    backend = FakeBackend()

    with ServerIndexer(config, backend, now_ns=clock) as indexer:
        indexer.baseline_existing()
        baseline_source.rename(listen / "new-name.mp3")
        assert indexer.cycle().indexed_files == 0
        assert state_row(indexer, "new-name.mp3")["state"] == "BASELINE"

        indexed_source = listen / "indexed-old.mp3"
        indexed_source.write_bytes(b"post-baseline bytes")
        assert indexer.cycle().indexed_files == 1
        indexed_source.rename(listen / "indexed-new.mp3")
        renamed = indexer.cycle()
        assert renamed.indexed_files == 0
        assert backend.embed_calls == 1
        assert renamed.publication is not None

    with sqlite3.connect(config.bundle_db) as db:
        assert db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 1
        assert db.execute("SELECT file_path FROM tracks").fetchone()[0].endswith(
            "/indexed-new.mp3"
        )
        assert db.execute(
            "SELECT relative_path FROM server_bundle_tracks"
        ).fetchone()[0] == "indexed-new.mp3"


def test_remount_device_and_inode_changes_preserve_path_states_and_embeddings(
    layout,
    monkeypatch,
):
    listen, base_config = layout
    config = replace(base_config, settle_seconds=0)
    baseline_source = listen / "baseline.mp3"
    baseline_source.write_bytes(b"baseline bytes")
    clock = Clock()
    backend = FakeBackend()

    with ServerIndexer(config, backend, now_ns=clock) as indexer:
        indexer.baseline_existing()
        indexed_source = listen / "indexed.mp3"
        indexed_source.write_bytes(b"indexed bytes")
        assert indexer.cycle().indexed_files == 1
        ready_source = listen / "ready.mp3"
        ready_source.write_bytes(b"ready bytes")
        indexer.discover()
        assert state_row(indexer, "ready.mp3")["state"] == READY

        rows_before = {
            name: state_row(indexer, name)
            for name in ("baseline.mp3", "indexed.mp3", "ready.mp3")
        }
        embedding_before = bytes(indexer.db.execute(
            "SELECT embedding FROM embeddings_clamp3 WHERE track_id = ?",
            (rows_before["indexed.mp3"]["track_id"],),
        ).fetchone()[0])
        actual_stat = Path.stat
        remounted = {baseline_source, indexed_source, ready_source}

        def remounted_stat(path: Path, *args, **kwargs):
            result = actual_stat(path, *args, **kwargs)
            if path not in remounted:
                return result
            return SimpleNamespace(
                st_dev=int(result.st_dev) + 10_000,
                st_ino=int(result.st_ino) + 20_000,
                st_mode=result.st_mode,
                st_size=result.st_size,
                st_mtime_ns=result.st_mtime_ns,
                st_ctime_ns=result.st_ctime_ns,
            )

        monkeypatch.setattr(Path, "stat", remounted_stat)
        observed, new, changed, ready = indexer.discover()

        assert (observed, new, changed, ready) == (3, 0, 0, 1)
        assert state_row(indexer, "baseline.mp3")["state"] == BASELINE
        assert state_row(indexer, "indexed.mp3")["state"] == INDEXED
        assert state_row(indexer, "ready.mp3")["state"] == READY
        for name, source in (
            ("baseline.mp3", baseline_source),
            ("indexed.mp3", indexed_source),
            ("ready.mp3", ready_source),
        ):
            assert state_row(indexer, name)["device_id"] == int(actual_stat(source).st_dev) + 10_000
            assert state_row(indexer, name)["inode"] == int(actual_stat(source).st_ino) + 20_000
        assert bytes(indexer.db.execute(
            "SELECT embedding FROM embeddings_clamp3 WHERE track_id = ?",
            (rows_before["indexed.mp3"]["track_id"],),
        ).fetchone()[0]) == embedding_before
        assert backend.embed_calls == 1


def test_idle_discovery_does_not_rewrite_unchanged_rows(layout):
    listen, config = layout
    source = listen / "settled.mp3"
    source.write_bytes(b"stable source bytes")
    clock = Clock()

    with ServerIndexer(config, FakeBackend(), now_ns=clock) as indexer:
        indexer.baseline_existing()
        changes_before = indexer.db.total_changes
        row_before = state_row(indexer, source.name)

        observed, new, changed, ready = indexer.discover()

        assert (observed, new, changed, ready) == (1, 0, 0, 0)
        assert indexer.db.total_changes == changes_before
        assert dict(state_row(indexer, source.name)) == dict(row_before)


def test_changed_indexed_path_atomically_supersedes_prior_bundle_row(layout):
    listen, base_config = layout
    config = replace(base_config, settle_seconds=0)
    clock = Clock()
    backend = FakeBackend()
    source = listen / "replaced.flac"

    with ServerIndexer(config, backend, now_ns=clock) as indexer:
        indexer.baseline_existing()
        source.write_bytes(b"first source bytes")
        assert indexer.cycle().indexed_files == 1
        track_id = state_row(indexer, "replaced.flac")["track_id"]
        old_source_hash = indexer.db.execute(
            "SELECT source_sha256 FROM server_track_provenance WHERE track_id = ?",
            (track_id,),
        ).fetchone()[0]
        old_bundle = config.bundle_db.read_bytes()

        source.write_bytes(b"different replacement source bytes")
        backend.fail = True
        failed = indexer.cycle()
        assert failed.changed_files == 1
        assert failed.indexed_files == 0
        assert failed.failed_files == 1
        assert indexer.db.execute(
            "SELECT source_sha256 FROM server_track_provenance WHERE track_id = ?",
            (track_id,),
        ).fetchone()[0] == old_source_hash
        assert config.bundle_db.read_bytes() == old_bundle

        backend.fail = False
        assert indexer.retry_failed() == 1
        replaced = indexer.cycle()
        assert replaced.changed_files == 0
        assert replaced.indexed_files == 1
        assert state_row(indexer, "replaced.flac")["track_id"] == track_id
        assert indexer.db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 1

    validate_bundle(config.bundle_db)
    with sqlite3.connect(config.bundle_db) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "SELECT root_id, relative_path, source_sha256 FROM server_bundle_tracks"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["root_id"] == "musicnew"
        assert rows[0]["relative_path"] == "replaced.flac"
        assert rows[0]["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_file_change_resets_settle_timer_and_mutation_during_embedding_never_commits(layout):
    listen, config = layout
    source = listen / "changing.mp3"
    clock = Clock()
    backend = FakeBackend(mutate=source)

    with ServerIndexer(config, backend, now_ns=clock) as indexer:
        indexer.baseline_existing()
        source.write_bytes(b"first version")
        indexer.cycle()
        clock.advance(5)
        source.write_bytes(b"second version")
        changed = indexer.cycle()
        assert changed.changed_files == 1
        clock.advance(6)
        assert indexer.cycle().indexed_files == 0
        clock.advance(5)
        attempted = indexer.cycle()
        assert attempted.indexed_files == 0
        assert indexer.db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 0
        assert state_row(indexer, "changing.mp3")["state"] == SETTLING


def test_repeated_unchanged_failure_blocks_until_manual_retry(layout):
    listen, base_config = layout
    config = replace(base_config, settle_seconds=0)
    clock = Clock()
    backend = FakeBackend(fail=True)

    with ServerIndexer(config, backend, now_ns=clock) as indexer:
        indexer.baseline_existing()
        (listen / "broken.mp3").write_bytes(b"not actually audio")
        first = indexer.cycle()
        assert first.failed_files == 1
        assert state_row(indexer, "broken.mp3")["state"] == RETRY_WAIT
        second = indexer.cycle()
        assert second.failed_files == 1
        assert state_row(indexer, "broken.mp3")["state"] == BLOCKED
        assert indexer.cycle().failed_files == 0
        assert indexer.retry_failed() == 1
        assert state_row(indexer, "broken.mp3")["state"] != BLOCKED


def test_v1_state_adds_decoder_receipts_without_reembedding_existing_rows(layout):
    listen, base_config = layout
    config = replace(base_config, settle_seconds=0)
    clock = Clock()
    source = listen / "already-indexed.mp3"

    with ServerIndexer(config, FakeBackend(), now_ns=clock) as indexer:
        indexer.baseline_existing()
        source.write_bytes(b"existing indexed source")
        assert indexer.cycle().indexed_files == 1
        track_id = state_row(indexer, source.name)["track_id"]
        embedding_before = bytes(indexer.db.execute(
            "SELECT embedding FROM embeddings_clamp3 WHERE track_id = ?",
            (track_id,),
        ).fetchone()[0])
        bundle_id_before = validate_bundle(config.bundle_db).bundle_id

    with sqlite3.connect(config.state_db) as legacy:
        legacy.execute("ALTER TABLE server_track_provenance DROP COLUMN source_decoder_id")
        legacy.execute("PRAGMA user_version = 1")
        legacy.execute(
            """
            UPDATE server_state_metadata
               SET value = '1'
             WHERE key = 'state_schema_version'
            """
        )
        legacy.commit()
    with sqlite3.connect(config.bundle_db) as legacy_bundle:
        legacy_bundle.execute(
            "ALTER TABLE server_bundle_tracks DROP COLUMN source_decoder_id"
        )
        legacy_bundle.execute(
            """
            DELETE FROM metadata
             WHERE key = 'server_bundle_source_decoder_provenance_version'
            """
        )
        legacy_bundle.commit()

    backend = FakeBackend()
    with ServerIndexer(config, backend, now_ns=clock) as migrated:
        report = migrated.cycle()
        assert report.indexed_files == 0
        assert report.publication is not None
        assert backend.embed_calls == 0
        provenance = migrated.db.execute(
            """
            SELECT source_decoder_id, embedding_spec_id
              FROM server_track_provenance
             WHERE track_id = ?
            """,
            (track_id,),
        ).fetchone()
        assert provenance["source_decoder_id"] == TORCHAUDIO_SOURCE_DECODER_ID
        assert provenance["embedding_spec_id"] == SERVER_EMBEDDING_SPEC_ID
        assert bytes(migrated.db.execute(
            "SELECT embedding FROM embeddings_clamp3 WHERE track_id = ?",
            (track_id,),
        ).fetchone()[0]) == embedding_before
        assert migrated.db.execute("PRAGMA user_version").fetchone()[0] == 3
    rebuilt = validate_bundle(config.bundle_db)
    assert rebuilt.bundle_id == bundle_id_before
    with sqlite3.connect(config.bundle_db) as bundle:
        assert bundle.execute(
            "SELECT source_decoder_id FROM server_bundle_tracks"
        ).fetchone()[0] == TORCHAUDIO_SOURCE_DECODER_ID


def test_v2_state_adds_nullable_status_telemetry_without_reembedding(layout):
    listen, base_config = layout
    config = replace(base_config, settle_seconds=0)
    clock = Clock()
    source = listen / "already-indexed.mp3"
    backend = FakeBackend()

    with ServerIndexer(config, backend, now_ns=clock) as indexer:
        indexer.baseline_existing()
        source.write_bytes(b"existing indexed source")
        assert indexer.cycle().indexed_files == 1
        track_id = state_row(indexer, source.name)["track_id"]
        embedding_before = bytes(indexer.db.execute(
            "SELECT embedding FROM embeddings_clamp3 WHERE track_id = ?",
            (track_id,),
        ).fetchone()[0])

    with sqlite3.connect(config.state_db) as legacy:
        legacy.execute("DROP TABLE server_runtime")
        legacy.execute("ALTER TABLE server_files DROP COLUMN last_embedding_reused")
        legacy.execute("ALTER TABLE server_files DROP COLUMN last_processing_duration_ms")
        legacy.execute("ALTER TABLE server_files DROP COLUMN last_processed_at_ns")
        legacy.execute("PRAGMA user_version = 2")
        legacy.execute(
            """
            UPDATE server_state_metadata
               SET value = '2'
             WHERE key = 'state_schema_version'
            """
        )
        legacy.commit()

    migrated_backend = FakeBackend()
    with ServerIndexer(config, migrated_backend, now_ns=clock) as migrated:
        columns = {
            row["name"]
            for row in migrated.db.execute("PRAGMA table_info(server_files)")
        }
        assert {
            "last_processed_at_ns",
            "last_processing_duration_ms",
            "last_embedding_reused",
        }.issubset(columns)
        assert migrated.db.execute("PRAGMA user_version").fetchone()[0] == 3
        assert migrated.db.execute(
            "SELECT value FROM server_state_metadata WHERE key = 'state_schema_version'"
        ).fetchone()[0] == "3"
        runtime = migrated.db.execute(
            "SELECT phase FROM server_runtime WHERE singleton_id = 1"
        ).fetchone()
        assert runtime["phase"] == "starting"
        assert bytes(migrated.db.execute(
            "SELECT embedding FROM embeddings_clamp3 WHERE track_id = ?",
            (track_id,),
        ).fetchone()[0]) == embedding_before
        assert migrated_backend.embed_calls == 0


def test_status_reads_v2_ledger_without_migration_scan_hash_or_model(layout, monkeypatch):
    _listen, config = layout
    with ServerIndexer(config, FakeBackend()) as indexer:
        indexer.baseline_existing()

    with sqlite3.connect(config.state_db) as legacy:
        legacy.execute("DROP TABLE server_runtime")
        legacy.execute("ALTER TABLE server_files DROP COLUMN last_embedding_reused")
        legacy.execute("ALTER TABLE server_files DROP COLUMN last_processing_duration_ms")
        legacy.execute("ALTER TABLE server_files DROP COLUMN last_processed_at_ns")
        legacy.execute("PRAGMA user_version = 2")
        legacy.execute(
            """
            UPDATE server_state_metadata
               SET value = '2'
             WHERE key = 'state_schema_version'
            """
        )
        legacy.commit()

    before = hashlib.sha256(config.state_db.read_bytes()).hexdigest()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("status performed indexing work")

    monkeypatch.setattr(server_indexer_module, "scan_music_directory", forbidden)
    monkeypatch.setattr(server_indexer_module, "sha256_file", forbidden)
    monkeypatch.setattr(
        server_indexer_module.Clamp3ServerBackend,
        "__init__",
        forbidden,
    )

    status = read_server_status(config)

    assert status["state"]["schema_version"] == 2
    assert status["runtime"]["phase"] == "unknown"
    assert status["timing"]["source"] is None
    assert hashlib.sha256(config.state_db.read_bytes()).hexdigest() == before


def test_status_uses_recent_non_reused_timing_and_present_state_counts(layout):
    listen, base_config = layout
    config = replace(base_config, settle_seconds=0)
    wall_clock = Clock()
    monotonic_clock = Clock()

    class TimedBackend(FakeBackend):
        def embed(self, snapshot: Path, source_sha256: str) -> EmbeddingResult:
            monotonic_clock.advance(31)
            return super().embed(snapshot, source_sha256)

    backend = TimedBackend()
    with ServerIndexer(
        config,
        backend,
        now_ns=wall_clock,
        monotonic_ns=monotonic_clock,
    ) as indexer:
        indexer.baseline_existing()
        (listen / "embedded.mp3").write_bytes(b"one embedded source")
        assert indexer.cycle().indexed_files == 1
        indexer.db.execute(
            """
            INSERT INTO server_files(
                root_id, relative_path, device_id, inode, size_bytes, mtime_ns,
                stable_since_ns, last_seen_ns, present, state
            ) VALUES
                ('musicnew', 'ready.mp3', 1, 1, 1, 1, 1, 1, 1, ?),
                ('musicnew', 'settling.mp3', 1, 2, 1, 1, 1, 1, 1, ?),
                ('musicnew', 'retry.mp3', 1, 3, 1, 1, 1, 1, 1, ?),
                ('musicnew', 'blocked.mp3', 1, 4, 1, 1, 1, 1, 1, ?),
                ('musicnew', 'gone.mp3', 1, 5, 1, 1, 1, 1, 0, ?)
            """,
            (READY, SETTLING, RETRY_WAIT, BLOCKED, INDEXED),
        )
        indexer.db.commit()

    status = read_server_status(config)
    assert status["writer_active"] is False
    assert status["library"]["present"] == 5
    assert status["library"]["embedded"] == 1
    assert status["library"]["missing"] == 1
    assert status["library"]["queue"] == {
        "ready": 1,
        "settling": 1,
        "active": 0,
        "retrying": 1,
        "blocked": 1,
    }
    assert status["timing"]["non_reused_sample_count"] == 1
    assert status["timing"]["publication_batch_sample_count"] == 0
    assert status["timing"]["source"] == "track_telemetry"
    assert status["timing"]["median_seconds_per_track"] == 31.0
    assert status["timing"]["p90_seconds_per_track"] == 31.0
    assert status["timing"]["estimated_pending_tracks"] == 3
    assert status["timing"]["estimated_remaining_seconds"] == 93


def test_status_falls_back_to_measured_full_publication_batches(layout):
    _listen, config = layout
    clock = Clock()
    with ServerIndexer(config, FakeBackend(), now_ns=clock) as indexer:
        indexer.baseline_existing()
        base = clock()
        indexer.db.executemany(
            """
            INSERT INTO server_publications(
                generation_id, track_count, bundle_sha256, published_at_ns
            ) VALUES (?, ?, ?, ?)
            """,
            [
                ("generation-0", 0, "0" * 64, base),
                ("generation-64", 64, "1" * 64, base + 64 * 30 * 1_000_000_000),
                (
                    "generation-128",
                    128,
                    "2" * 64,
                    base + 64 * (30 + 40) * 1_000_000_000,
                ),
                # A partial publication may contain mostly idle time and is not a speed sample.
                (
                    "generation-129",
                    129,
                    "3" * 64,
                    base + 64 * (30 + 40) * 1_000_000_000 + 86_400_000_000_000,
                ),
            ],
        )
        indexer.db.commit()

    status = read_server_status(config)
    assert status["timing"]["non_reused_sample_count"] == 0
    assert status["timing"]["publication_batch_sample_count"] == 2
    assert status["timing"]["source"] == "full_publication_batches"
    assert status["timing"]["median_seconds_per_track"] == pytest.approx(35.0, abs=0.01)
    assert status["timing"]["p90_seconds_per_track"] == pytest.approx(40.0, abs=0.01)


def test_model_bootstrap_failure_does_not_consume_track_attempts(layout):
    listen, base_config = layout
    config = replace(base_config, settle_seconds=0)
    clock = Clock()

    class BootstrapFailureBackend(FakeBackend):
        def prepare(self) -> None:
            raise RuntimeError("model runtime is unavailable")

    with ServerIndexer(config, BootstrapFailureBackend(), now_ns=clock) as indexer:
        indexer.baseline_existing()
        (listen / "healthy.mp3").write_bytes(b"valid fixture for a healthy track")
        with pytest.raises(RuntimeError, match="model runtime"):
            indexer.cycle()
        row = state_row(indexer, "healthy.mp3")
        assert row["state"] == "READY"
        assert row["attempt_count"] == 0
        assert row["error_code"] is None


def test_interrupted_active_row_recovers_to_ready_and_indexes_once(layout):
    listen, base_config = layout
    config = replace(base_config, settle_seconds=0)
    clock = Clock()
    backend = FakeBackend()

    with ServerIndexer(config, backend, now_ns=clock) as indexer:
        indexer.baseline_existing()
        (listen / "recover.mp3").write_bytes(b"recoverable source")
        indexer.discover()
        indexer.db.execute(
            "UPDATE server_files SET state = ? WHERE relative_path = 'recover.mp3'", (STAGING,)
        )
        indexer.db.commit()

    with ServerIndexer(config, backend, now_ns=clock) as recovered:
        report = recovered.cycle()
        assert report.indexed_files == 1
        assert recovered.db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 1


def test_restart_publishes_track_committed_before_prior_process_could_publish(layout):
    listen, base_config = layout
    config = replace(base_config, settle_seconds=0)
    clock = Clock()
    backend = FakeBackend()

    with ServerIndexer(config, backend, now_ns=clock) as indexer:
        indexer.baseline_existing()
        (listen / "committed.mp3").write_bytes(b"committed before publication")
        indexer.discover()
        indexed, _, _, _ = indexer._process_ready(clock())
        assert indexed == 1
        assert validate_bundle(config.bundle_db).track_count == 0

    with ServerIndexer(config, backend, now_ns=clock) as recovered:
        report = recovered.cycle()
        assert report.indexed_files == 0
        assert report.publication is not None
        assert report.publication.track_count == 1


def test_multi_track_cue_image_is_embedded_as_one_complete_physical_file(layout):
    listen, base_config = layout
    config = replace(base_config, settle_seconds=0)
    clock = Clock()
    backend = FakeBackend()

    with ServerIndexer(config, backend, now_ns=clock) as indexer:
        indexer.baseline_existing()
        (listen / "image.flac").write_bytes(b"one physical image")
        (listen / "album.cue").write_text(
            'FILE "image.flac" WAVE\n'
            '  TRACK 01 AUDIO\n'
            '  TRACK 02 AUDIO\n'
        )
        report = indexer.cycle()
        assert report.indexed_files == 1
        assert report.blocked_files == 0
        assert backend.prepare_calls == 1
        row = state_row(indexer, "image.flac")
        assert row["state"] == INDEXED


def test_per_track_files_named_by_cue_are_indexed_as_complete_files(layout):
    listen, base_config = layout
    config = replace(base_config, settle_seconds=0)
    clock = Clock()
    backend = FakeBackend()

    with ServerIndexer(config, backend, now_ns=clock) as indexer:
        indexer.baseline_existing()
        (listen / "01 - Café.flac").write_bytes(b"one physical track")
        (listen / "02 - 夜.flac").write_bytes(b"another physical track")
        (listen / "album.CUE").write_text(
            '\ufeffFILE "01 - Cafe\u0301.flac" WAVE\n'
            '  TRACK 01 AUDIO\n'
            'FILE "02 - 夜.flac" WAVE\n'
            '  TRACK 02 AUDIO\n',
            encoding="utf-8",
        )

        report = indexer.cycle()

        assert report.indexed_files == 2
        assert report.blocked_files == 0
        assert backend.embed_calls == 2
        assert state_row(indexer, "01 - Café.flac")["state"] == INDEXED
        assert state_row(indexer, "02 - 夜.flac")["state"] == INDEXED

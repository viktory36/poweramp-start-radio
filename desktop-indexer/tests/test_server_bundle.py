from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from poweramp_indexer.server_bundle import publish_bundle, validate_bundle
from poweramp_indexer.server_config import (
    TORCHAUDIO_SOURCE_DECODER_ID,
    ListenRoot,
    ServerConfig,
)
from poweramp_indexer.server_indexer import EmbeddingResult, ServerIndexer


class Backend:
    def prepare(self) -> None:
        pass

    def embed(self, snapshot: Path, source_sha256: str) -> EmbeddingResult:
        return EmbeddingResult((1.0,) + (0.0,) * 767, 48_000, 96_000)

    def close(self) -> None:
        pass


def config_for(tmp_path: Path) -> ServerConfig:
    listen = tmp_path / "listen"
    runtime = tmp_path / "runtime"
    listen.mkdir()
    runtime.mkdir()
    return ServerConfig(
        state_db=runtime / "state.db",
        bundle_db=runtime / "bundle.db",
        cache_dir=runtime / "cache",
        listen_roots=(ListenRoot("test", listen),),
        settle_seconds=0,
    )


def test_failed_atomic_replace_preserves_previous_visible_bundle(tmp_path: Path):
    config = config_for(tmp_path)
    with ServerIndexer(config, Backend()) as indexer:
        indexer.baseline_existing()
        (config.listen_roots[0].path / "new.mp3").write_bytes(b"new source")
        indexer.cycle()
        before = config.bundle_db.read_bytes()

        def fail_replace(_source: str, _destination: str) -> None:
            raise OSError("injected replace failure")

        with pytest.raises(OSError, match="injected"):
            publish_bundle(indexer.db, config.bundle_db, replace=fail_replace)
        assert config.bundle_db.read_bytes() == before
        assert not list(config.bundle_db.parent.glob(f".{config.bundle_db.name}.next-*"))
        validate_bundle(config.bundle_db)


def test_bundle_carries_additive_decoder_provenance_without_changing_embedding_spec(
    tmp_path: Path,
):
    config = config_for(tmp_path)
    with ServerIndexer(config, Backend()) as indexer:
        indexer.baseline_existing()
        (config.listen_roots[0].path / "new.mp3").write_bytes(b"new source")
        indexer.cycle()

    validate_bundle(config.bundle_db)
    with sqlite3.connect(config.bundle_db) as db:
        metadata = dict(db.execute("SELECT key, value FROM metadata"))
        assert metadata["server_bundle_embedding_spec_id"] == "poweramp-clamp3-server-v1"
        assert metadata["server_bundle_source_decoder_provenance_version"] == "1"
        assert db.execute(
            "SELECT source_decoder_id FROM server_bundle_tracks"
        ).fetchone()[0] == TORCHAUDIO_SOURCE_DECODER_ID


def test_validator_rejects_tampered_logical_bundle_id(tmp_path: Path):
    config = config_for(tmp_path)
    with ServerIndexer(config, Backend()) as indexer:
        indexer.baseline_existing()
    with sqlite3.connect(config.bundle_db) as db:
        db.execute(
            "UPDATE metadata SET value = 'server-bundle-v1-deadbeef' "
            "WHERE key = 'server_bundle_id'"
        )
        db.commit()
    with pytest.raises(ValueError, match="logical ID"):
        validate_bundle(config.bundle_db)


def test_config_rejects_runtime_paths_inside_listen_root(tmp_path: Path):
    listen = tmp_path / "listen"
    listen.mkdir()
    with pytest.raises(ValueError, match="must not be inside"):
        ServerConfig(
            state_db=listen / "state.db",
            bundle_db=tmp_path / "bundle.db",
            cache_dir=tmp_path / "cache",
            listen_roots=(ListenRoot("test", listen),),
        )


def test_config_rejects_duplicate_canonical_roots_and_state_bundle_alias(tmp_path: Path):
    one = tmp_path / "one"
    one.mkdir()
    with pytest.raises(ValueError, match="paths must be unique"):
        ServerConfig(
            state_db=tmp_path / "state.db",
            bundle_db=tmp_path / "bundle.db",
            cache_dir=tmp_path / "cache",
            listen_roots=(ListenRoot("a", one), ListenRoot("b", one / ".")),
        )
    with pytest.raises(ValueError, match="must be different"):
        ServerConfig(
            state_db=tmp_path / "same.db",
            bundle_db=tmp_path / "same.db",
            cache_dir=tmp_path / "cache",
            listen_roots=(ListenRoot("a", one),),
        )


def test_config_rejects_root_id_longer_than_android_contract(tmp_path: Path):
    listen = tmp_path / "listen"
    listen.mkdir()
    with pytest.raises(ValueError, match="invalid root ID"):
        ServerConfig(
            state_db=tmp_path / "state.db",
            bundle_db=tmp_path / "bundle.db",
            cache_dir=tmp_path / "cache",
            listen_roots=(ListenRoot("x" * 129, listen),),
        )

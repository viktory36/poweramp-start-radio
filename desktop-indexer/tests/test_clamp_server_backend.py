from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from poweramp_indexer import server_indexer
from poweramp_indexer.server_config import (
    FFMPEG_SOURCE_DECODER_ID,
    ListenRoot,
    ServerConfig,
)
from poweramp_indexer.server_indexer import Clamp3ServerBackend


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
    )


class FakeDetailedGenerator:
    def __init__(self, features: np.ndarray):
        self.features = features
        self.extract_calls = 0
        self.unloaded = False

    def extract_mert_features_with_source_span(self, _snapshot: Path):
        self.extract_calls += 1
        return SimpleNamespace(
            features=self.features,
            source_sample_rate_hz=48_000,
            source_sample_count=144_000,
            source_decoder_id=FFMPEG_SOURCE_DECODER_ID,
        )

    def encode_mert_features(self, _features: np.ndarray):
        return [1.0] + [0.0] * 767

    def unload_models(self):
        self.unloaded = True


def test_server_backend_uses_source_span_from_feature_decode(tmp_path: Path):
    config = config_for(tmp_path)
    backend = Clamp3ServerBackend(config)
    generator = FakeDetailedGenerator(np.zeros((1, 1, 768), dtype=np.float32))
    backend.generator = generator
    backend._prepared = True
    source = config.listen_roots[0].path / "track.flac"
    source.write_bytes(b"fixture")

    result = backend.embed(source, "a" * 64)

    assert (result.source_sample_rate_hz, result.source_sample_count) == (48_000, 144_000)
    assert result.source_decoder_id == FFMPEG_SOURCE_DECODER_ID
    assert generator.extract_calls == 1
    assert list(config.cache_dir.rglob("*.npy")) == []


def test_server_backend_keeps_mert_features_for_one_embedding_only(tmp_path: Path):
    config = config_for(tmp_path)
    backend = Clamp3ServerBackend(config)
    features = np.zeros((1, 1, 768), dtype=np.float32)
    generator = FakeDetailedGenerator(features)
    backend.generator = generator
    backend._prepared = True
    source_hash = "b" * 64
    source = config.listen_roots[0].path / "track.flac"
    source.write_bytes(b"fixture")

    first = backend.embed(source, source_hash)
    second = backend.embed(source, source_hash)

    assert first == second
    assert generator.extract_calls == 2
    assert list(config.cache_dir.rglob("*.npy")) == []


def test_server_prepare_eagerly_loads_both_audio_models(tmp_path: Path, monkeypatch):
    config = config_for(tmp_path)
    mert = tmp_path / "pytorch_model.bin"
    clamp = tmp_path / "clamp3.pth"
    mert.write_bytes(b"pinned mert")
    clamp.write_bytes(b"pinned clamp")
    monkeypatch.setattr(
        server_indexer, "MERT_MODEL_SHA256", hashlib.sha256(mert.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        server_indexer,
        "CLAMP3_AUDIO_MODEL_SHA256",
        hashlib.sha256(clamp.read_bytes()).hexdigest(),
    )

    import huggingface_hub

    def download(_repository, filename, **_kwargs):
        return str(mert if filename == "pytorch_model.bin" else clamp)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

    created = []

    class EagerGenerator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.prepared = 0
            self.unloaded = 0
            created.append(self)

        def prepare_audio_models(self):
            self.prepared += 1

        def unload_models(self):
            self.unloaded += 1

    fake_embedding_module = SimpleNamespace(
        CLAMP3_WEIGHTS_FILENAME="clamp3.pth",
        CLaMP3EmbeddingGenerator=EagerGenerator,
    )
    monkeypatch.setitem(
        sys.modules, "poweramp_indexer.embeddings_clamp3", fake_embedding_module
    )
    backend = Clamp3ServerBackend(config)

    backend.prepare()
    backend.prepare()

    assert len(created) == 1
    assert created[0].prepared == 1
    assert created[0].kwargs["max_duration"] is None
    assert created[0].kwargs["fp16"] is False

"""Configuration and pinned model contract for the persistent server indexer."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib


SERVER_EMBEDDING_SPEC_ID = "poweramp-clamp3-server-v1"
SERVER_OUTPUT_SPACE_ID = "clamp3-joint-audio-text-l2-f32-v1"
TORCHAUDIO_SOURCE_DECODER_ID = "torchaudio-load-native-f32-v1"
FFMPEG_SOURCE_DECODER_ID = "ffmpeg-first-audio-native-f32le-v1"
SERVER_SOURCE_DECODER_IDS = frozenset(
    (TORCHAUDIO_SOURCE_DECODER_ID, FFMPEG_SOURCE_DECODER_ID)
)
MERT_MODEL_ID = "m-a-p/MERT-v1-95M"
MERT_MODEL_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
MERT_MODEL_SHA256 = "a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd"
CLAMP3_REPOSITORY = "sander-wood/clamp3"
CLAMP3_REVISION = "355625cc1c6f73726bbcd0eb9276ac7152d56426"
CLAMP3_AUDIO_MODEL_SHA256 = (
    "5033f868e3977be3945ee416b5a1718d5589a173c7ba8982231d8c94a6441d80"
)

_ROOT_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


@dataclass(frozen=True)
class ListenRoot:
    root_id: str
    path: Path


@dataclass(frozen=True)
class ServerConfig:
    state_db: Path
    bundle_db: Path
    cache_dir: Path
    listen_roots: tuple[ListenRoot, ...]
    poll_interval_seconds: float = 60.0
    settle_seconds: float = 120.0
    max_files_per_cycle: int = 64
    max_attempts: int = 3
    retry_delays_seconds: tuple[float, ...] = (300.0, 1800.0, 7200.0)
    mert_batch_size: int = 8

    def __post_init__(self) -> None:
        if not self.listen_roots:
            raise ValueError("at least one listen root is required")
        ids = [root.root_id for root in self.listen_roots]
        if len(ids) != len(set(ids)):
            raise ValueError("listen root IDs must be unique")
        root_paths = [root.path.resolve() for root in self.listen_roots]
        if len(root_paths) != len(set(root_paths)):
            raise ValueError("listen root paths must be unique")
        for root in self.listen_roots:
            if not _ROOT_ID.fullmatch(root.root_id):
                raise ValueError(
                    f"invalid root ID {root.root_id!r}; use letters, digits, dot, dash, "
                    "or underscore, with at most 128 characters"
                )
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.settle_seconds < 0:
            raise ValueError("settle_seconds cannot be negative")
        if self.max_files_per_cycle <= 0 or self.max_attempts <= 0:
            raise ValueError("cycle and attempt limits must be positive")
        if not self.retry_delays_seconds or any(value < 0 for value in self.retry_delays_seconds):
            raise ValueError("retry_delays_seconds must contain non-negative values")
        if self.mert_batch_size <= 0:
            raise ValueError("mert_batch_size must be positive")
        if self.state_db.resolve() == self.bundle_db.resolve():
            raise ValueError("state_db and bundle_db must be different files")
        managed_paths = {
            "state_db": self.state_db.resolve(),
            "bundle_db": self.bundle_db.resolve(),
            "cache_dir": self.cache_dir.resolve(),
        }
        for field, managed in managed_paths.items():
            for root in root_paths:
                if managed == root or root in managed.parents:
                    raise ValueError(f"{field} must not be inside listen root {root}")


def server_embedding_spec() -> dict[str, Any]:
    """Return the immutable, canonical server embedding contract."""
    return {
        "schema_version": 1,
        "embedding_spec_id": SERVER_EMBEDDING_SPEC_ID,
        "output_space_id": SERVER_OUTPUT_SPACE_ID,
        "audio_span": "complete-physical-file",
        "precision": "fp32",
        "target_sample_rate_hz": 24000,
        "downmix": "arithmetic-channel-mean",
        "resampler": "torchaudio-hann-width6-rolloff0.99-f32-target-length",
        "normalization": "wav2vec2-zero-mean-unit-variance-whole-span",
        "window_samples": 120000,
        "window_hop_samples": 120000,
        "tail_policy": "drop-below-1s-otherwise-zero-pad-to-5s",
        "maximum_duration_seconds": None,
        "mert_pooling": "mean-time-then-mean-layers",
        "clamp3_aggregation": "zero-bookends-segment128-final-overlap-frame-weighted-l2",
        "output_dimension": 768,
        "mert": {
            "model_id": MERT_MODEL_ID,
            "revision": MERT_MODEL_REVISION,
            "pytorch_model_sha256": MERT_MODEL_SHA256,
        },
        "clamp3_audio": {
            "repository": CLAMP3_REPOSITORY,
            "revision": CLAMP3_REVISION,
            "checkpoint_sha256": CLAMP3_AUDIO_MODEL_SHA256,
        },
    }


def canonical_server_embedding_spec_json() -> str:
    return json.dumps(server_embedding_spec(), sort_keys=True, separators=(",", ":"))


def server_embedding_spec_sha256() -> str:
    return hashlib.sha256(canonical_server_embedding_spec_json().encode("utf-8")).hexdigest()


def load_server_config(path: Path) -> ServerConfig:
    config_path = path.expanduser().resolve()
    with config_path.open("rb") as handle:
        document = tomllib.load(handle)
    server = document.get("server")
    roots = document.get("listen_roots")
    if not isinstance(server, dict):
        raise ValueError("configuration requires a [server] table")
    if not isinstance(roots, list) or not roots:
        raise ValueError("configuration requires at least one [[listen_roots]] table")

    base = config_path.parent

    def resolve_path(value: object, field: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty path")
        result = Path(value).expanduser()
        if not result.is_absolute():
            result = base / result
        return result.resolve()

    listen_roots: list[ListenRoot] = []
    for index, value in enumerate(roots):
        if not isinstance(value, dict):
            raise ValueError(f"listen_roots[{index}] must be a table")
        root_id = value.get("id")
        if not isinstance(root_id, str):
            raise ValueError(f"listen_roots[{index}].id must be a string")
        listen_roots.append(
            ListenRoot(root_id=root_id, path=resolve_path(value.get("path"), "listen root path"))
        )

    retry_values = server.get("retry_delays_seconds", [300, 1800, 7200])
    if not isinstance(retry_values, list):
        raise ValueError("retry_delays_seconds must be an array")

    return ServerConfig(
        state_db=resolve_path(server.get("state_db"), "state_db"),
        bundle_db=resolve_path(server.get("bundle_db"), "bundle_db"),
        cache_dir=resolve_path(server.get("cache_dir"), "cache_dir"),
        listen_roots=tuple(listen_roots),
        poll_interval_seconds=float(server.get("poll_interval_seconds", 60)),
        settle_seconds=float(server.get("settle_seconds", 120)),
        max_files_per_cycle=int(server.get("max_files_per_cycle", 64)),
        max_attempts=int(server.get("max_attempts", 3)),
        retry_delays_seconds=tuple(float(value) for value in retry_values),
        mert_batch_size=int(server.get("mert_batch_size", 8)),
    )

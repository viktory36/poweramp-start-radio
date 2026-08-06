from pathlib import Path

import click
import numpy as np
import pytest
from click.testing import CliRunner

from poweramp_indexer.cli import cli, scan, update
from poweramp_indexer.embeddings_clamp3 import (
    MERT_CACHE_KEY_PREFIX,
    CLaMP3EmbeddingGenerator,
    _save_cache_array,
    _synchronize_mert_cache,
    _write_cache_manifest,
    make_cache_key,
)


def _option(command, name: str) -> click.Option:
    return next(parameter for parameter in command.params if parameter.name == name)


def _publish_cache(
    music_dir: Path,
    cache_dir: Path,
    files: list[Path],
    max_duration: int | None,
):
    sources, _ = _synchronize_mert_cache(
        music_dir, cache_dir, files, max_duration
    )
    entries = {}
    for cache_key, source in sources.items():
        _save_cache_array(
            cache_dir / f"{cache_key}.npy",
            np.zeros((1, 1, 768), dtype=np.float32),
        )
        entries[cache_key] = source.manifest_entry(max_duration)
    _write_cache_manifest(cache_dir / "manifest.json", entries)
    return sources, entries


def test_cache_keys_are_fixed_length_unicode_safe_and_unambiguous(tmp_path: Path):
    music_dir = tmp_path / "music"
    paths = [
        music_dir / "a" / "b.flac",
        music_dir / "a__b.flac",
        music_dir / "日本語" / "夜の音楽.flac",
        music_dir / (("長い名前" * 200) + ".flac"),
    ]

    keys = [make_cache_key(path, music_dir) for path in paths]

    assert len(set(keys)) == len(paths)
    assert {len(key) for key in keys} == {len(MERT_CACHE_KEY_PREFIX) + 64}
    assert all(key.isascii() for key in keys)
    assert all("/" not in key and "\\" not in key for key in keys)


def test_cache_reuse_requires_unchanged_source_and_duration_policy(tmp_path: Path):
    music_dir = tmp_path / "music"
    cache_dir = tmp_path / "cache"
    music_dir.mkdir()
    source = music_dir / "Café 夜.flac"
    source.write_bytes(b"first source bytes")

    sources, _ = _publish_cache(music_dir, cache_dir, [source], None)
    cache_key = next(iter(sources))
    npy_path = cache_dir / f"{cache_key}.npy"

    _, valid = _synchronize_mert_cache(
        music_dir, cache_dir, [source], None
    )
    assert set(valid) == {cache_key}
    assert npy_path.exists()

    _, capped = _synchronize_mert_cache(
        music_dir, cache_dir, [source], 600
    )
    assert capped == {}
    assert not npy_path.exists()

    _publish_cache(music_dir, cache_dir, [source], None)
    source.write_bytes(b"a replaced source with a different size")
    _, changed = _synchronize_mert_cache(
        music_dir, cache_dir, [source], None
    )
    assert changed == {}
    assert not npy_path.exists()


def test_cache_sync_prunes_removed_entries_orphan_arrays_and_temporary_files(
    tmp_path: Path,
):
    music_dir = tmp_path / "music"
    cache_dir = tmp_path / "cache"
    music_dir.mkdir()
    kept = music_dir / "kept.flac"
    removed = music_dir / "removed.flac"
    kept.write_bytes(b"kept")
    removed.write_bytes(b"removed")

    sources, _ = _publish_cache(
        music_dir, cache_dir, [kept, removed], None
    )
    kept_key = make_cache_key(kept, music_dir)
    removed_key = make_cache_key(removed, music_dir)
    orphan = cache_dir / f"{MERT_CACHE_KEY_PREFIX}{'f' * 64}.npy"
    orphan.write_bytes(b"orphan")
    temporary = cache_dir / ".interrupted.npy.tmp"
    temporary.write_bytes(b"partial")
    removed.unlink()

    _, valid = _synchronize_mert_cache(
        music_dir, cache_dir, [kept], None
    )

    assert set(sources) == {kept_key, removed_key}
    assert set(valid) == {kept_key}
    assert (cache_dir / f"{kept_key}.npy").exists()
    assert not (cache_dir / f"{removed_key}.npy").exists()
    assert not orphan.exists()
    assert not temporary.exists()


def test_generator_and_cli_default_to_complete_tracks_with_positive_opt_in_cap(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        CLaMP3EmbeddingGenerator,
        "_get_best_device",
        staticmethod(lambda: "cpu"),
    )
    assert CLaMP3EmbeddingGenerator().max_duration is None
    with pytest.raises(ValueError, match="positive or None"):
        CLaMP3EmbeddingGenerator(max_duration=0)

    for command in (scan, update):
        option = _option(command, "max_duration")
        assert option.default is None
        assert isinstance(option.type, click.IntRange)
        assert option.type.min == 1

    runner = CliRunner()
    rejected = runner.invoke(
        cli,
        ["scan", str(tmp_path), "--max-duration", "0"],
    )
    assert rejected.exit_code == 2
    assert "not in the range x>=1" in rejected.output


def test_cli_calls_the_knn_mode_graph_explorer():
    runner = CliRunner()

    graph_help = runner.invoke(cli, ["graph", "--help"])
    update_help = runner.invoke(cli, ["update", "--help"])

    assert graph_help.exit_code == 0, graph_help.output
    assert update_help.exit_code == 0, update_help.output
    assert "Graph Explorer" in graph_help.output
    assert "Graph Explorer" in " ".join(update_help.output.split())
    assert "Random Walk" not in graph_help.output
    assert "Random Walk" not in update_help.output

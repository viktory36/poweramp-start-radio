from __future__ import annotations

import http.client
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import v2_blind_listening_server as server


TOKEN = "audio-" + "a" * 24


def packet_audio(source: dict[str, object], span: server.LogicalSpan | None = None):
    return server.PacketAudio(token=TOKEN, source=source, logical_span=span)


def test_windows_mapping_is_rooted_and_rejects_traversal(tmp_path: Path) -> None:
    library = tmp_path / "Music"
    track = library / "albums" / "track.flac"
    track.parent.mkdir(parents=True)
    track.write_bytes(b"music")
    resolver = server.AudioResolver(
        {
            TOKEN: packet_audio(
                {
                    "kind": "windows_path",
                    "reportedPath": r"C:\Music\albums\track.flac",
                }
            )
        },
        windows_root=library,
        cache_root=tmp_path / "cache",
    )
    assert resolver.resolve(TOKEN) == track.resolve()

    unsafe = server.AudioResolver(
        {
            TOKEN: packet_audio(
                {
                    "kind": "windows_path",
                    "reportedPath": r"C:\Music\..\private.flac",
                }
            )
        },
        windows_root=library,
        cache_root=tmp_path / "cache",
    )
    with pytest.raises(server.ListeningServerError, match="traversal"):
        unsafe.resolve(TOKEN)


def test_android_mirror_precedes_read_only_adb_pull(tmp_path: Path) -> None:
    library = tmp_path / "Music"
    mirror = library / "new" / "track.opus"
    mirror.parent.mkdir(parents=True)
    mirror.write_bytes(b"mirror")
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(command)
        raise AssertionError("ADB should not run for a mirrored track")

    resolver = server.AudioResolver(
        {
            TOKEN: packet_audio(
                {
                    "kind": "android_device_path",
                    "reportedPath": "/storage/ABCD/Music/new/track.opus",
                    "adbPullSource": "/storage/ABCD/Music/new/track.opus",
                }
            )
        },
        windows_root=library,
        cache_root=tmp_path / "cache",
        allow_adb_pull=True,
        command_runner=runner,
    )
    assert resolver.resolve(TOKEN) == mirror.resolve()
    assert calls == []


def test_phone_only_fallback_invokes_only_adb_pull_without_shell(tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[-1]).write_bytes(b"pulled audio")
        return type("Result", (), {"returncode": 0})()

    remote = "/storage/ABCD/Music/new/phone-only.flac"
    resolver = server.AudioResolver(
        {
            TOKEN: packet_audio(
                {
                    "kind": "android_device_path",
                    "reportedPath": remote,
                    "adbPullSource": remote,
                }
            )
        },
        windows_root=tmp_path / "Music",
        cache_root=tmp_path / "cache",
        allow_adb_pull=True,
        adb="adb",
        adb_serial="SERIAL",
        command_runner=runner,
    )
    result = resolver.resolve(TOKEN)
    assert result.read_bytes() == b"pulled audio"
    assert calls[0][0][0:4] == ["adb", "-s", "SERIAL", "pull"]
    assert calls[0][0][4] == remote
    assert "shell" not in calls[0][1]
    assert calls[0][1]["check"] is False
    assert calls[0][1]["capture_output"] is True
    if os.name == "posix":
        assert result.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize(
    ("header", "size", "expected"),
    [
        (None, 10, None),
        ("bytes=2-5", 10, server.ByteRange(2, 5)),
        ("bytes=7-", 10, server.ByteRange(7, 9)),
        ("bytes=-4", 10, server.ByteRange(6, 9)),
        ("bytes=2-99", 10, server.ByteRange(2, 9)),
    ],
)
def test_parse_single_byte_range(header, size, expected) -> None:
    assert server.parse_byte_range(header, size) == expected


@pytest.mark.parametrize("header", ["items=0-1", "bytes=", "bytes=5-4", "bytes=10-"])
def test_invalid_ranges_fail_closed(header: str) -> None:
    with pytest.raises(server.RangeNotSatisfiable):
        server.parse_byte_range(header, 10)


def test_explicit_cue_span_is_losslessly_materialized_and_cached(tmp_path: Path) -> None:
    source = tmp_path / "album.flac"
    source.write_bytes(b"source audio")
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"logical track")
        return type("Result", (), {"returncode": 0})()

    span = server.LogicalSpan(start_ms=61_250, duration_ms=182_500)
    resolver = server.AudioResolver(
        {
            TOKEN: packet_audio(
                {"kind": "host_path", "reportedPath": str(source)}, span
            )
        },
        windows_root=tmp_path / "Music",
        cache_root=tmp_path / "cache",
        host_roots=[tmp_path],
        command_runner=runner,
    )
    first = resolver.resolve(TOKEN)
    second = resolver.resolve(TOKEN)
    assert first == second
    assert first.read_bytes() == b"logical track"
    assert len(calls) == 1
    command = calls[0]
    assert command[command.index("-ss") + 1] == "61.250"
    assert command[command.index("-t") + 1] == "182.500"
    assert command[command.index("-c:a") + 1] == "flac"
    assert command[command.index("-map_metadata") + 1] == "-1"


def test_duration_alone_does_not_turn_whole_file_into_cue_span() -> None:
    track = {
        "durationMs": 180_000,
        "source": {"kind": "windows_path", "reportedPath": r"C:\Music\x.flac"},
    }
    assert server.logical_span_from_track(track) is None
    assert server.logical_span_from_track(
        {**track, "source": {**track["source"], "logicalSpan": {"offsetMs": 5000, "endMs": 9000}}}
    ) == server.LogicalSpan(start_ms=5000, duration_ms=4000)


def test_preflight_deduplicates_sources_and_reports_path_free_counts(tmp_path: Path) -> None:
    path = tmp_path / "private" / "track.flac"
    path.parent.mkdir()
    path.write_bytes(b"audio")
    source = {
        "kind": "windows_path",
        "reportedPath": r"C:\Music\private\track.flac",
    }
    first = server.PacketAudio("audio-" + "a" * 24, source, None, 180_000)
    second = server.PacketAudio("audio-" + "b" * 24, source, None, 180_000)
    probe_calls = []

    class Resolver:
        def resolve(self, token: str) -> Path:
            return path

    def probe(command, **kwargs):
        probe_calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"streams": [{"codec_type": "audio"}], "format": {"duration": "180.062"}}
            ),
        )

    result = server.preflight_packet_audio(
        {first.token: first, second.token: second},
        Resolver(),
        command_runner=probe,
    )
    assert result == {
        "state": "READY",
        "audioTokenCount": 2,
        "uniqueSourceCount": 1,
        "logicalSpanSourceCount": 0,
        "durationCheckedSourceCount": 1,
        "durationToleranceMs": 2000,
        "outcomes": {
            "ready": 1,
            "unavailable": 0,
            "resolutionRejected": 0,
            "probeFailed": 0,
            "durationMismatch": 0,
        },
        "sourceKinds": {"windows_path": {"total": 1, "ready": 1}},
    }
    assert len(probe_calls) == 1
    rendered = json.dumps(result)
    assert "track.flac" not in rendered
    assert str(tmp_path) not in rendered


def test_preflight_fails_closed_on_unavailable_probe_and_duration_mismatch(
    tmp_path: Path,
) -> None:
    ready_path = tmp_path / "audio.flac"
    ready_path.write_bytes(b"audio")
    unavailable = server.PacketAudio(
        "audio-" + "a" * 24,
        {"kind": "android_device_path", "reportedPath": "/storage/x/Music/a.flac"},
        None,
        10_000,
    )
    mismatch = server.PacketAudio(
        "audio-" + "b" * 24,
        {"kind": "host_path", "reportedPath": str(ready_path)},
        None,
        10_000,
    )

    class Resolver:
        def resolve(self, token: str) -> Path:
            if token == unavailable.token:
                raise server.AudioUnavailable("not present")
            return ready_path

    def probe(command, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"streams": [{"codec_type": "audio"}], "format": {"duration": "30.0"}}
            ),
        )

    result = server.preflight_packet_audio(
        {unavailable.token: unavailable, mismatch.token: mismatch},
        Resolver(),
        command_runner=probe,
    )
    assert result["state"] == "INCOMPLETE"
    assert result["outcomes"]["unavailable"] == 1
    assert result["outcomes"]["durationMismatch"] == 1
    assert result["outcomes"]["ready"] == 0


def test_browser_runner_requires_complete_responses_and_persists_review_state() -> None:
    html = server._INDEX_HTML
    assert 'poweramp-start-radio-blind-listening-responses-v2' in html
    assert 'id="importFile"' in html
    assert 'completeResponse' in html
    assert '$("export").disabled=complete!==manifest.trials.length' in html
    assert 'listenerState' in html
    assert '["Beginning",0],["Middle",.5],["Late",.85]' in html
    assert 'new MediaMetadata' in html


class StaticResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, token: str) -> Path:
        if token != TOKEN:
            raise server.AudioUnavailable("unknown")
        return self.path


def test_http_surface_supports_range_and_never_serves_reveal(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"0123456789")
    manifest = json.dumps({"public": True}).encode()
    httpd = server.ListeningHTTPServer(("127.0.0.1", 0), manifest, StaticResolver(audio))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*httpd.server_address, timeout=5)
        connection.request("GET", f"/audio/{TOKEN}", headers={"Range": "bytes=2-5"})
        response = connection.getresponse()
        assert response.status == 206
        assert response.getheader("Content-Range") == "bytes 2-5/10"
        assert response.getheader("Accept-Ranges") == "bytes"
        assert response.read() == b"2345"

        connection.request("GET", f"/audio/{TOKEN}", headers={"Range": "bytes=99-"})
        response = connection.getresponse()
        assert response.status == 416
        assert response.getheader("Content-Range") == "bytes */10"
        assert response.read() == b""

        connection.request("GET", "/api/manifest")
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == manifest

        connection.request("GET", "/reveal-key.json")
        response = connection.getresponse()
        assert response.status == 404
        assert b"reportedPath" not in response.read()
        connection.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

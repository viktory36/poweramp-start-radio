#!/usr/bin/env python3
"""Serve a blind-listening packet and its full-track audio on localhost.

The HTTP surface is deliberately smaller than the packet directory: browsers
can fetch only the public manifest, the experiment UI, and opaque audio tokens.
Reveal metadata and source paths remain in this process. Android fallbacks use
only ``adb pull`` and are opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit


PUBLIC_SCHEMA = "poweramp-start-radio-blind-listening-packet-v1"
REVEAL_SCHEMA = "poweramp-start-radio-blind-listening-reveal-v1"
AUDIO_TOKEN_RE = re.compile(r"audio-[0-9a-f]{24}\Z")
DEFAULT_WINDOWS_LIBRARY_PREFIX = PureWindowsPath(r"C:\Music")
COPY_CHUNK_BYTES = 1024 * 1024
PREFLIGHT_DURATION_TOLERANCE_MS = 2_000


class ListeningServerError(ValueError):
    """A fail-closed packet, source, or request error."""


class AudioUnavailable(RuntimeError):
    """An audio source is valid but not currently available."""


class RangeNotSatisfiable(ValueError):
    """A Range header cannot be satisfied by the current representation."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class LogicalSpan:
    start_ms: int
    duration_ms: int


@dataclass(frozen=True)
class PacketAudio:
    token: str
    source: Mapping[str, Any]
    logical_span: LogicalSpan | None
    expected_duration_ms: int | None = None


@dataclass(frozen=True)
class LoadedPacket:
    manifest_bytes: bytes
    manifest: Mapping[str, Any]
    audio: Mapping[str, PacketAudio]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ListeningServerError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ListeningServerError(f"{path} must contain a JSON object")
    return value, raw


def _required_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ListeningServerError(f"{where} must be non-empty text")
    return value


def _nonnegative_int(value: Any, where: str) -> int:
    if type(value) is not int or value < 0:
        raise ListeningServerError(f"{where} must be a non-negative integer")
    return value


def logical_span_from_track(track: Mapping[str, Any]) -> LogicalSpan | None:
    """Read only an explicit logical span; duration metadata alone never clips."""

    source = track.get("source")
    if not isinstance(source, dict):
        raise ListeningServerError("reveal track source must be an object")
    raw = source.get("logicalSpan", source.get("cueSpan"))
    if raw is None:
        raw = track.get("logicalSpan", track.get("cueSpan"))
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ListeningServerError("logical span must be an object")

    start_value = raw.get("startMs", raw.get("offsetMs"))
    start_ms = _nonnegative_int(start_value, "logical span startMs")
    if "durationMs" in raw:
        duration_ms = _nonnegative_int(raw["durationMs"], "logical span durationMs")
    elif "endMs" in raw:
        end_ms = _nonnegative_int(raw["endMs"], "logical span endMs")
        duration_ms = end_ms - start_ms
    else:
        raise ListeningServerError("logical span needs durationMs or endMs")
    if duration_ms <= 0:
        raise ListeningServerError("logical span duration must be positive")
    return LogicalSpan(start_ms=start_ms, duration_ms=duration_ms)


def load_packet(packet_dir: Path) -> LoadedPacket:
    public_path = packet_dir / "blind-manifest.json"
    reveal_path = packet_dir / "reveal-key.json"
    manifest, manifest_bytes = _read_json_object(public_path)
    reveal, _ = _read_json_object(reveal_path)

    if os.name == "posix" and reveal_path.stat().st_mode & 0o077:
        raise ListeningServerError("reveal-key.json must not be group/world accessible")
    if manifest.get("schema") != PUBLIC_SCHEMA:
        raise ListeningServerError(f"unsupported public packet schema: {manifest.get('schema')!r}")
    if reveal.get("schema") != REVEAL_SCHEMA:
        raise ListeningServerError(f"unsupported reveal schema: {reveal.get('schema')!r}")
    for field in ("packetId", "studyId"):
        if manifest.get(field) != reveal.get(field):
            raise ListeningServerError(f"public/reveal {field} mismatch")
    if reveal.get("publicManifestSha256") != _sha256(manifest_bytes):
        raise ListeningServerError("public manifest does not match reveal commitment")

    reveal_core = {
        key: value
        for key, value in reveal.items()
        if key not in {"revealCoreSha256", "publicManifestSha256"}
    }
    reveal_core_sha = _sha256(_canonical_json(reveal_core))
    if reveal.get("revealCoreSha256") != reveal_core_sha:
        raise ListeningServerError("reveal core digest is invalid")
    if manifest.get("revealCommitmentSha256") != reveal_core_sha:
        raise ListeningServerError("public reveal commitment is invalid")

    public_trials = manifest.get("trials")
    reveal_trials = reveal.get("trials")
    if not isinstance(public_trials, list) or not isinstance(reveal_trials, list):
        raise ListeningServerError("packet trials must be arrays")
    reveal_by_id = {
        _required_text(item.get("trial"), "reveal trial id"): item
        for item in reveal_trials
        if isinstance(item, dict)
    }
    if len(reveal_by_id) != len(reveal_trials):
        raise ListeningServerError("reveal trial IDs must be unique objects")

    audio: dict[str, PacketAudio] = {}
    for public_trial in public_trials:
        if not isinstance(public_trial, dict):
            raise ListeningServerError("public trial must be an object")
        trial_id = _required_text(public_trial.get("trial"), "public trial id")
        reveal_trial = reveal_by_id.get(trial_id)
        if reveal_trial is None:
            raise ListeningServerError(f"missing reveal trial {trial_id}")
        public_sides = public_trial.get("sides")
        reveal_sides = reveal_trial.get("sides")
        if not isinstance(public_sides, list) or not isinstance(reveal_sides, dict):
            raise ListeningServerError(f"invalid side data for {trial_id}")
        for public_side in public_sides:
            if not isinstance(public_side, dict):
                raise ListeningServerError(f"invalid public side for {trial_id}")
            label = _required_text(public_side.get("label"), f"{trial_id} side label")
            reveal_side = reveal_sides.get(label)
            if not isinstance(reveal_side, dict):
                raise ListeningServerError(f"missing reveal side {trial_id}/{label}")
            if public_side.get("queueToken") != reveal_side.get("queueToken"):
                raise ListeningServerError(f"queue token mismatch for {trial_id}/{label}")
            public_tracks = public_side.get("tracks")
            reveal_tracks = reveal_side.get("tracks")
            if not isinstance(public_tracks, list) or not isinstance(reveal_tracks, list):
                raise ListeningServerError(f"invalid tracks for {trial_id}/{label}")
            if len(public_tracks) != len(reveal_tracks):
                raise ListeningServerError(f"track count mismatch for {trial_id}/{label}")
            if public_side.get("trackCount") != len(public_tracks):
                raise ListeningServerError(f"declared track count mismatch for {trial_id}/{label}")
            for expected_position, (public_track, reveal_track) in enumerate(
                zip(public_tracks, reveal_tracks, strict=True), start=1
            ):
                if not isinstance(public_track, dict) or not isinstance(reveal_track, dict):
                    raise ListeningServerError("packet track entries must be objects")
                token = _required_text(public_track.get("audioToken"), "public audio token")
                if AUDIO_TOKEN_RE.fullmatch(token) is None:
                    raise ListeningServerError(f"unsupported audio token {token!r}")
                if token in audio:
                    raise ListeningServerError(f"duplicate audio token {token}")
                if reveal_track.get("audioToken") != token:
                    raise ListeningServerError(f"audio token mismatch at {trial_id}/{label}")
                if public_track.get("position") != expected_position:
                    raise ListeningServerError(f"public position mismatch at {trial_id}/{label}")
                if reveal_track.get("position") != expected_position:
                    raise ListeningServerError(f"reveal position mismatch at {trial_id}/{label}")
                source = reveal_track.get("source")
                if not isinstance(source, dict):
                    raise ListeningServerError(f"missing source for audio token {token}")
                audio[token] = PacketAudio(
                    token=token,
                    source=source,
                    logical_span=logical_span_from_track(reveal_track),
                    expected_duration_ms=_nonnegative_int(
                        reveal_track.get("durationMs"), "reveal track durationMs"
                    ),
                )
    if set(reveal_by_id) != {
        _required_text(item.get("trial"), "public trial id")
        for item in public_trials
        if isinstance(item, dict)
    }:
        raise ListeningServerError("public/reveal trial sets differ")
    return LoadedPacket(manifest_bytes=manifest_bytes, manifest=manifest, audio=audio)


def parse_byte_range(header: str | None, size: int) -> ByteRange | None:
    if header is None:
        return None
    if size < 0 or not header.startswith("bytes=") or "," in header:
        raise RangeNotSatisfiable("unsupported byte range")
    value = header[6:].strip()
    if "-" not in value:
        raise RangeNotSatisfiable("malformed byte range")
    start_text, end_text = value.split("-", 1)
    try:
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0 or size == 0:
                raise RangeNotSatisfiable("invalid suffix range")
            start = max(0, size - suffix)
            return ByteRange(start=start, end=size - 1)
        start = int(start_text)
        if start < 0 or start >= size:
            raise RangeNotSatisfiable("range starts outside representation")
        end = size - 1 if not end_text else int(end_text)
        if end < start:
            raise RangeNotSatisfiable("range end precedes start")
        return ByteRange(start=start, end=min(end, size - 1))
    except ValueError as exc:
        raise RangeNotSatisfiable("malformed byte range") from exc


def _safe_contained_path(root: Path, relative_parts: Sequence[str]) -> Path:
    if not relative_parts or any(
        part in {"", ".", ".."} or "\x00" in part for part in relative_parts
    ):
        raise ListeningServerError("source path contains unsafe components")
    resolved_root = root.expanduser().resolve(strict=False)
    candidate = resolved_root.joinpath(*relative_parts).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ListeningServerError("source path escapes its configured library root") from exc
    return candidate


def _windows_relative(
    reported: str,
    library_prefix: PureWindowsPath,
) -> tuple[str, ...]:
    if "\x00" in reported:
        raise ListeningServerError("Windows source path contains NUL")
    path = PureWindowsPath(reported)
    if ".." in path.parts:
        raise ListeningServerError("Windows source path contains traversal")
    try:
        relative = path.relative_to(library_prefix)
    except ValueError as exc:
        raise ListeningServerError(
            f"Windows source is outside {library_prefix}"
        ) from exc
    if relative.is_absolute() or not relative.parts:
        raise ListeningServerError("Windows source does not identify a file")
    return relative.parts


def _android_music_relative(reported: str) -> tuple[str, ...]:
    if any(ord(char) < 32 for char in reported):
        raise ListeningServerError("Android source path contains control characters")
    path = PurePosixPath(reported)
    if not path.is_absolute() or ".." in path.parts:
        raise ListeningServerError("Android source path is not a safe absolute path")
    parts = path.parts
    if len(parts) < 4 or parts[1] not in {"storage", "sdcard"}:
        raise ListeningServerError("Android source is outside an accepted storage root")
    try:
        music_index = parts.index("Music", 2)
    except ValueError as exc:
        raise ListeningServerError("Android source is outside a Music directory") from exc
    relative = parts[music_index + 1 :]
    if not relative:
        raise ListeningServerError("Android source does not identify a file")
    return relative


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class AudioResolver:
    def __init__(
        self,
        audio: Mapping[str, PacketAudio],
        *,
        windows_root: Path,
        windows_library_prefix: PureWindowsPath = DEFAULT_WINDOWS_LIBRARY_PREFIX,
        cache_root: Path,
        host_roots: Sequence[Path] = (),
        allow_adb_pull: bool = False,
        adb: str = "adb",
        adb_serial: str | None = None,
        adb_timeout_seconds: int = 1800,
        ffmpeg: str = "ffmpeg",
        command_runner: CommandRunner = subprocess.run,
    ) -> None:
        self._audio = dict(audio)
        self.windows_root = windows_root.expanduser()
        self.windows_library_prefix = windows_library_prefix
        self.cache_root = cache_root.expanduser()
        self.host_roots = tuple(root.expanduser().resolve(strict=False) for root in host_roots)
        self.allow_adb_pull = allow_adb_pull
        self.adb = adb
        self.adb_serial = adb_serial
        self.adb_timeout_seconds = adb_timeout_seconds
        self.ffmpeg = ffmpeg
        self._command_runner = command_runner
        self._locks_guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _ensure_cache_root(self) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            os.chmod(self.cache_root, 0o700)

    def _resolve_host_path(self, reported: str) -> Path:
        if not self.host_roots:
            raise AudioUnavailable("host source needs an explicit --host-root")
        candidate = Path(reported).expanduser().resolve(strict=False)
        for root in self.host_roots:
            try:
                candidate.relative_to(root)
                break
            except ValueError:
                continue
        else:
            raise ListeningServerError("host source escapes configured roots")
        if not candidate.is_file():
            raise AudioUnavailable("host source is absent")
        return candidate

    def _pull_android(self, remote: str) -> Path:
        if not self.allow_adb_pull:
            raise AudioUnavailable("phone-only audio needs --allow-adb-pull")
        self._ensure_cache_root()
        suffix = PurePosixPath(remote).suffix.lower()
        if re.fullmatch(r"\.[a-z0-9]{1,8}", suffix) is None:
            suffix = ".audio"
        digest = hashlib.sha256(remote.encode("utf-8")).hexdigest()
        destination = self.cache_root / f"adb-{digest}{suffix}"
        with self._lock_for(f"adb:{remote}"):
            if destination.is_file() and destination.stat().st_size > 0:
                return destination
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".part", dir=self.cache_root
            )
            os.close(fd)
            temporary = Path(temporary_name)
            temporary.unlink(missing_ok=True)
            command = [self.adb]
            if self.adb_serial:
                command.extend(["-s", self.adb_serial])
            command.extend(["pull", remote, str(temporary)])
            try:
                result = self._command_runner(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.adb_timeout_seconds,
                )
                if result.returncode != 0 or not temporary.is_file():
                    raise AudioUnavailable("adb pull failed")
                if temporary.stat().st_size <= 0:
                    raise AudioUnavailable("adb pull produced an empty file")
                if os.name == "posix":
                    os.chmod(temporary, 0o600)
                os.replace(temporary, destination)
                return destination
            finally:
                temporary.unlink(missing_ok=True)

    def _resolve_source(self, spec: PacketAudio) -> Path:
        kind = spec.source.get("kind")
        reported = _required_text(spec.source.get("reportedPath"), "reported source path")
        if kind == "windows_path":
            candidate = _safe_contained_path(
                self.windows_root,
                _windows_relative(reported, self.windows_library_prefix),
            )
            if not candidate.is_file():
                raise AudioUnavailable("Windows backup source is absent")
            return candidate
        if kind == "android_device_path":
            relative = _android_music_relative(reported)
            mirror = _safe_contained_path(self.windows_root, relative)
            if mirror.is_file():
                return mirror
            pull_source = spec.source.get("adbPullSource", reported)
            if pull_source != reported:
                raise ListeningServerError("ADB source disagrees with reported device path")
            return self._pull_android(reported)
        if kind == "host_path":
            return self._resolve_host_path(reported)
        raise ListeningServerError(f"unsupported reveal source kind {kind!r}")

    def _materialize_span(self, source: Path, span: LogicalSpan) -> Path:
        self._ensure_cache_root()
        try:
            stat = source.stat()
        except OSError as exc:
            raise AudioUnavailable("logical-span source is absent") from exc
        cache_basis = (
            f"{source.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}\0"
            f"{span.start_ms}\0{span.duration_ms}"
        )
        digest = hashlib.sha256(cache_basis.encode("utf-8")).hexdigest()
        destination = self.cache_root / f"span-{digest}.flac"
        with self._lock_for(f"span:{digest}"):
            if destination.is_file() and destination.stat().st_size > 0:
                return destination
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".part", dir=self.cache_root
            )
            os.close(fd)
            temporary = Path(temporary_name)
            command = [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-ss",
                f"{span.start_ms / 1000:.3f}",
                "-t",
                f"{span.duration_ms / 1000:.3f}",
                "-vn",
                "-map_metadata",
                "-1",
                "-c:a",
                "flac",
                "-f",
                "flac",
                "-y",
                str(temporary),
            ]
            try:
                result = self._command_runner(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=max(300, span.duration_ms // 1000 * 2),
                )
                if result.returncode != 0 or not temporary.is_file():
                    raise AudioUnavailable("ffmpeg could not materialize logical track span")
                if temporary.stat().st_size <= 0:
                    raise AudioUnavailable("logical track span is empty")
                if os.name == "posix":
                    os.chmod(temporary, 0o600)
                os.replace(temporary, destination)
                return destination
            finally:
                temporary.unlink(missing_ok=True)

    def resolve(self, token: str) -> Path:
        spec = self._audio.get(token)
        if spec is None:
            raise AudioUnavailable("unknown audio token")
        source = self._resolve_source(spec)
        if spec.logical_span is not None:
            return self._materialize_span(source, spec.logical_span)
        return source


def _preflight_source_key(spec: PacketAudio) -> bytes:
    span = spec.logical_span
    return _canonical_json(
        {
            "source": spec.source,
            "logicalSpan": (
                None
                if span is None
                else {"startMs": span.start_ms, "durationMs": span.duration_ms}
            ),
        }
    )


def preflight_packet_audio(
    audio: Mapping[str, PacketAudio],
    resolver: AudioResolver,
    *,
    ffprobe: str = "ffprobe",
    duration_tolerance_ms: int = PREFLIGHT_DURATION_TOLERANCE_MS,
    command_runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Resolve and decode-probe every unique source without exposing its locator."""

    if duration_tolerance_ms < 0:
        raise ListeningServerError("preflight duration tolerance must be non-negative")
    groups: dict[bytes, list[PacketAudio]] = {}
    for spec in audio.values():
        groups.setdefault(_preflight_source_key(spec), []).append(spec)

    outcomes = {
        "ready": 0,
        "unavailable": 0,
        "resolutionRejected": 0,
        "probeFailed": 0,
        "durationMismatch": 0,
    }
    kinds: dict[str, dict[str, int]] = {}
    logical_spans = 0
    duration_checked = 0
    for specs in groups.values():
        representative = specs[0]
        kind = str(representative.source.get("kind", "unknown"))
        kind_counts = kinds.setdefault(kind, {"total": 0, "ready": 0})
        kind_counts["total"] += 1
        if representative.logical_span is not None:
            logical_spans += 1
        try:
            path = resolver.resolve(representative.token)
        except AudioUnavailable:
            outcomes["unavailable"] += 1
            continue
        except (ListeningServerError, OSError):
            outcomes["resolutionRejected"] += 1
            continue
        command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type:format=duration",
            "-of",
            "json",
            str(path),
        ]
        try:
            result = command_runner(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            payload = json.loads(result.stdout, parse_constant=_reject_json_constant)
            streams = payload.get("streams") if isinstance(payload, dict) else None
            duration_raw = (
                payload.get("format", {}).get("duration")
                if isinstance(payload, dict) and isinstance(payload.get("format"), dict)
                else None
            )
            duration_seconds = float(duration_raw)
            if (
                result.returncode != 0
                or not isinstance(streams, list)
                or not streams
                or not math.isfinite(duration_seconds)
                or duration_seconds <= 0
            ):
                raise ValueError("audio probe did not return a valid stream and duration")
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, json.JSONDecodeError):
            outcomes["probeFailed"] += 1
            continue

        expected_durations = {
            spec.expected_duration_ms
            for spec in specs
            if spec.expected_duration_ms is not None
        }
        if expected_durations:
            duration_checked += 1
            observed_ms = round(duration_seconds * 1000)
            if any(
                abs(observed_ms - expected_ms) > duration_tolerance_ms
                for expected_ms in expected_durations
            ):
                outcomes["durationMismatch"] += 1
                continue
        outcomes["ready"] += 1
        kind_counts["ready"] += 1

    unique_count = len(groups)
    state = "READY" if outcomes["ready"] == unique_count else "INCOMPLETE"
    return {
        "state": state,
        "audioTokenCount": len(audio),
        "uniqueSourceCount": unique_count,
        "logicalSpanSourceCount": logical_spans,
        "durationCheckedSourceCount": duration_checked,
        "durationToleranceMs": duration_tolerance_ms,
        "outcomes": outcomes,
        "sourceKinds": {key: kinds[key] for key in sorted(kinds)},
    }


def _audio_content_type(path: Path) -> str:
    explicit = {
        ".flac": "audio/flac",
        ".mp3": "audio/mpeg",
        ".opus": "audio/ogg",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".wav": "audio/wav",
    }.get(path.suffix.lower())
    return explicit or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind queue listening</title>
<style>
:root { color-scheme:dark; --bg:#111315; --panel:#1a1d20; --line:#34393e; --text:#f1f2ef; --muted:#aab0b4; --accent:#62b6a7; --focus:#e6b85c; }
* { box-sizing:border-box; letter-spacing:0; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.45 system-ui,sans-serif; }
header { position:sticky; top:0; z-index:2; display:flex; gap:12px; align-items:center; padding:12px 20px; border-bottom:1px solid var(--line); background:#111315f5; }
header strong { white-space:nowrap; }
select,button,textarea,input { font:inherit; }
select,button,textarea { color:var(--text); background:var(--panel); border:1px solid var(--line); border-radius:5px; }
button { min-height:36px; padding:7px 11px; cursor:pointer; }
button:disabled { cursor:not-allowed; opacity:.48; }
button:hover:not(:disabled),button:focus-visible { border-color:var(--focus); outline:none; }
button.selected { color:#071511; border-color:var(--accent); background:var(--accent); }
select { min-height:36px; padding:6px 9px; max-width:280px; }
.spacer { flex:1; }
main { max-width:1400px; margin:auto; padding:22px; }
.prompt { margin-bottom:18px; }
.prompt h1 { margin:0 0 5px; font-size:21px; }
.prompt p { margin:0; color:var(--muted); }
.queues { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
.queue { min-width:0; border-top:3px solid var(--accent); padding-top:12px; }
.queue h2 { margin:0 0 10px; font-size:18px; }
.now,.passages { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:8px; color:var(--muted); }
.now output { min-width:110px; }
.passages button { min-height:32px; padding:5px 9px; }
audio { width:100%; height:44px; margin-bottom:8px; }
.tracks { display:grid; grid-template-columns:repeat(10,minmax(38px,1fr)); gap:5px; }
.tracks button { padding:5px; min-width:0; }
.tracks button.reviewed:not(.selected) { border-color:var(--accent); }
.ratings { margin-top:24px; border-top:1px solid var(--line); padding-top:18px; }
.preference,.metric { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin:10px 0; }
.preference strong,.metric strong { min-width:140px; }
.metric .side-label { color:var(--muted); margin-left:8px; }
textarea { display:block; width:100%; min-height:82px; padding:9px; resize:vertical; }
.status { color:var(--muted); }
@media (max-width:780px) { header { flex-wrap:wrap; padding:10px; } main { padding:14px; } .queues { grid-template-columns:1fr; } .tracks { grid-template-columns:repeat(6,minmax(38px,1fr)); } .spacer { display:none; } }
</style>
</head>
<body>
<header>
  <strong>Blind queue listening</strong>
  <select id="trialPicker" aria-label="Trial"></select>
  <span id="progress" class="status"></span>
  <span class="spacer"></span>
  <button id="previous" title="Previous trial" aria-label="Previous trial">&larr;</button>
  <button id="next" title="Next trial" aria-label="Next trial">&rarr;</button>
  <button id="import">Import responses</button>
  <input id="importFile" type="file" accept="application/json,.json" hidden>
  <button id="export" disabled>Export responses</button>
</header>
<main>
  <section class="prompt"><h1 id="prompt"></h1><p id="question"></p></section>
  <section class="queues" id="queues"></section>
  <section class="ratings">
    <div class="preference"><strong>Preferred queue</strong><span id="preference"></span></div>
    <div id="metrics"></div>
    <textarea id="notes" placeholder="Listening notes" aria-label="Listening notes"></textarea>
  </section>
</main>
<script>
"use strict";
let manifest,trialIndex=0,responses={},listenerState={trialIndex:0,trials:{}};
let activePlayer=null;
const $=id=>document.getElementById(id);
const storageKey=()=>`blind-listening:${manifest.packetId}:state-v2`;
function responseFor(trial){return responses[trial.trial]||=( {preference:null,intentFit:{A:null,B:null},coherence:{A:null,B:null},discoveryValue:{A:null,B:null},notes:""} );}
function trialStateFor(trial){
  const value=listenerState.trials[trial.trial]||=( {} );
  trial.sides.forEach(side=>{if(!value[side.label])value[side.label]={selected:0,reviewed:[]};});
  return value;
}
function normalizeResponses(value){
  const normalized={};
  const rating=value=>Number.isInteger(value)&&value>=1&&value<=5?value:null;
  manifest.trials.forEach(trial=>{const source=value?.[trial.trial]||{};normalized[trial.trial]={preference:["A","B","tie"].includes(source.preference)?source.preference:null,intentFit:{A:rating(source.intentFit?.A),B:rating(source.intentFit?.B)},coherence:{A:rating(source.coherence?.A),B:rating(source.coherence?.B)},discoveryValue:{A:rating(source.discoveryValue?.A),B:rating(source.discoveryValue?.B)},notes:typeof source.notes==="string"?source.notes:""};});
  return normalized;
}
function normalizeListenerState(value){
  const normalized={trialIndex:Math.max(0,Math.min(manifest.trials.length-1,Number.isInteger(value?.trialIndex)?value.trialIndex:0)),trials:{}};
  manifest.trials.forEach(trial=>{const source=value?.trials?.[trial.trial],sides={};trial.sides.forEach(side=>{const state=source?.[side.label]||{},limit=side.tracks.length,selected=Number.isInteger(state.selected)?Math.max(0,Math.min(limit-1,state.selected)):0,reviewed=Array.isArray(state.reviewed)?[...new Set(state.reviewed.filter(position=>Number.isInteger(position)&&position>=1&&position<=limit))].sort((a,b)=>a-b):[];sides[side.label]={selected,reviewed};});normalized.trials[trial.trial]=sides;});
  return normalized;
}
function save(){listenerState.trialIndex=trialIndex;localStorage.setItem(storageKey(),JSON.stringify({responses,listenerState}));updateProgress();}
function completeResponse(response){return !!response&&[response.preference,response.intentFit?.A,response.intentFit?.B,response.coherence?.A,response.coherence?.B,response.discoveryValue?.A,response.discoveryValue?.B].every(value=>value!==null&&value!==undefined);}
function allComplete(){return manifest.trials.every(trial=>completeResponse(responses[trial.trial]));}
function choiceButtons(values,current,onPick){
  const box=document.createElement("span");
  values.forEach(value=>{const button=document.createElement("button");button.textContent=value;button.className=current===value?"selected":"";button.onclick=()=>onPick(value);box.appendChild(button);});
  return box;
}
function opaqueMediaMetadata(trial,side,position){
  if(!("mediaSession" in navigator)||!("MediaMetadata" in window))return;
  try{navigator.mediaSession.metadata=new MediaMetadata({title:`Track ${position}`,artist:`Queue ${side}`,album:trial.trial});}catch{}
}
function queueView(trial,side){
  const state=trialStateFor(trial)[side.label];
  const section=document.createElement("section");section.className="queue";
  const title=document.createElement("h2");title.textContent=`Queue ${side.label}`;section.appendChild(title);
  const now=document.createElement("div");now.className="now";
  const output=document.createElement("output");
  const back=document.createElement("button");back.innerHTML="&larr;";back.title="Previous track";back.setAttribute("aria-label","Previous track");
  const forward=document.createElement("button");forward.innerHTML="&rarr;";forward.title="Next track";forward.setAttribute("aria-label","Next track");
  const reviewed=document.createElement("button");
  now.append(output,back,forward,reviewed);section.appendChild(now);
  const player=document.createElement("audio");player.controls=true;player.preload="none";section.appendChild(player);
  const passages=document.createElement("div");passages.className="passages";
  [["Beginning",0],["Middle",.5],["Late",.85]].forEach(([label,fraction])=>{const button=document.createElement("button");button.textContent=label;button.onclick=()=>seekTo(Number(fraction));passages.appendChild(button);});
  section.appendChild(passages);
  const grid=document.createElement("div");grid.className="tracks";section.appendChild(grid);
  let selected=Math.max(0,Math.min(side.tracks.length-1,state.selected||0));const buttons=[];
  function isReviewed(index){return state.reviewed.includes(index+1);}
  function renderSelection(){
    output.textContent=`Track ${selected+1} of ${side.trackCount}`;
    reviewed.textContent=isReviewed(selected)?"Reviewed":"Mark reviewed";
    reviewed.className=isReviewed(selected)?"selected":"";
    buttons.forEach((button,index)=>{button.className=index===selected?"selected":isReviewed(index)?"reviewed":"";button.setAttribute("aria-label",`Track ${index+1}${isReviewed(index)?", reviewed":""}`);});
  }
  function select(index,autoplay=false,persist=true){
    selected=Math.max(0,Math.min(side.tracks.length-1,index));state.selected=selected;
    const track=side.tracks[selected];player.src=`/audio/${track.audioToken}`;opaqueMediaMetadata(trial,side.label,selected+1);renderSelection();
    if(persist)save();if(autoplay)player.play().catch(()=>{});
  }
  function markReviewed(){if(!isReviewed(selected)){state.reviewed.push(selected+1);state.reviewed.sort((a,b)=>a-b);}renderSelection();save();}
  function toggleReviewed(){if(isReviewed(selected))state.reviewed=state.reviewed.filter(value=>value!==selected+1);else state.reviewed.push(selected+1);state.reviewed.sort((a,b)=>a-b);renderSelection();save();}
  function seekTo(fraction){
    const apply=()=>{if(!Number.isFinite(player.duration)||player.duration<=0)return;player.currentTime=Math.min(Math.max(0,player.duration*fraction),Math.max(0,player.duration-.05));player.play().catch(()=>{});};
    if(player.readyState>=1)apply();else{player.addEventListener("loadedmetadata",apply,{once:true});player.load();}
  }
  side.tracks.forEach((track,index)=>{const button=document.createElement("button");button.textContent=String(track.position);button.title=`Track ${track.position}`;button.onclick=()=>select(index,true);buttons.push(button);grid.appendChild(button);});
  back.onclick=()=>select(selected-1,true);forward.onclick=()=>select(selected+1,true);reviewed.onclick=toggleReviewed;
  player.onplay=()=>{if(activePlayer&&activePlayer!==player)activePlayer.pause();activePlayer=player;opaqueMediaMetadata(trial,side.label,selected+1);};
  player.onended=()=>{markReviewed();if(selected+1<side.tracks.length)select(selected+1,true);};
  player.onerror=()=>{output.textContent=`Track ${selected+1} unavailable`;};
  select(selected,false,false);return section;
}
function render(){
  if(activePlayer){activePlayer.pause();activePlayer=null;}
  const trial=manifest.trials[trialIndex],response=responseFor(trial);$("prompt").textContent=trial.prompt;$("question").textContent=trial.question;$("trialPicker").value=String(trialIndex);
  $("queues").replaceChildren(...trial.sides.map(side=>queueView(trial,side)));
  renderRatings();$("notes").value=response.notes;$("previous").disabled=trialIndex===0;$("next").disabled=trialIndex===manifest.trials.length-1;
}
function renderRatings(){
  const trial=manifest.trials[trialIndex],response=responseFor(trial),labels={intentFit:"Intent fit",coherence:"Queue coherence",discoveryValue:"Discovery value"};
  const metrics=$("metrics");metrics.replaceChildren();
  Object.entries(labels).forEach(([key,label])=>{const row=document.createElement("div");row.className="metric";const strong=document.createElement("strong");strong.textContent=label;row.appendChild(strong);["A","B"].forEach(side=>{const tag=document.createElement("span");tag.className="side-label";tag.textContent=side;row.append(tag,choiceButtons([1,2,3,4,5],response[key][side],value=>{response[key][side]=value;save();renderRatings();}));});metrics.appendChild(row);});
  $("preference").replaceChildren(choiceButtons(["A","B","tie"],response.preference,value=>{response.preference=value;save();renderRatings();}));updateProgress();
}
function updateProgress(){const complete=manifest.trials.filter(trial=>completeResponse(responses[trial.trial])).length;$("progress").textContent=`${complete} of ${manifest.trials.length} complete`;$("export").disabled=complete!==manifest.trials.length;$("export").title=$("export").disabled?"Complete every rating before export":"Export completed responses";}
function go(delta){trialIndex=Math.max(0,Math.min(manifest.trials.length-1,trialIndex+delta));save();render();}
function exportValue(){return {schema:"poweramp-start-radio-blind-listening-responses-v2",packetId:manifest.packetId,studyId:manifest.studyId,sourceBindingSha256:manifest.sourceBindingSha256,revealCommitmentSha256:manifest.revealCommitmentSha256,responses:manifest.trials.map(trial=>({trial:trial.trial,...responseFor(trial)})),listenerState};}
function checkImport(value){
  if(value?.schema!=="poweramp-start-radio-blind-listening-responses-v2")throw new Error("Unsupported response file");
  for(const field of ["packetId","studyId","sourceBindingSha256","revealCommitmentSha256"])if(value[field]!==manifest[field])throw new Error("Response file belongs to a different packet");
  if(!Array.isArray(value.responses)||value.responses.length!==manifest.trials.length)throw new Error("Response file has the wrong trial count");
  const expected=manifest.trials.map(trial=>trial.trial);if(value.responses.some((response,index)=>response?.trial!==expected[index]))throw new Error("Response trial order does not match this packet");
  if(!value.listenerState||typeof value.listenerState!=="object")throw new Error("Response file has no listening state");
}
async function importFile(file){const value=JSON.parse(await file.text());checkImport(value);responses=normalizeResponses(Object.fromEntries(value.responses.map(response=>{const {trial,...answer}=response;return [trial,answer];})));listenerState=normalizeListenerState(value.listenerState);trialIndex=listenerState.trialIndex;save();render();}
async function init(){
  const request=await fetch("/api/manifest",{cache:"no-store"});if(!request.ok)throw new Error("manifest unavailable");manifest=await request.json();
  try{const saved=JSON.parse(localStorage.getItem(storageKey()));responses=normalizeResponses(saved?.responses||{});listenerState=normalizeListenerState(saved?.listenerState||{});trialIndex=listenerState.trialIndex;}catch{responses=normalizeResponses({});listenerState=normalizeListenerState({});trialIndex=0;}
  manifest.trials.forEach((trial,index)=>{const option=document.createElement("option");option.value=String(index);option.textContent=`${trial.trial}: ${trial.prompt}`;$("trialPicker").appendChild(option);});
  $("trialPicker").onchange=event=>{trialIndex=Number(event.target.value);save();render();};$("previous").onclick=()=>go(-1);$("next").onclick=()=>go(1);$("notes").oninput=event=>{responseFor(manifest.trials[trialIndex]).notes=event.target.value;save();};
  $("import").onclick=()=>$("importFile").click();$("importFile").onchange=event=>{const file=event.target.files?.[0];if(file)importFile(file).catch(error=>window.alert(error.message));event.target.value="";};
  $("export").onclick=()=>{if(!allComplete())return;const value=exportValue(),blob=new Blob([JSON.stringify(value,null,2)+"\n"],{type:"application/json"}),link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download=`${manifest.packetId}-responses.json`;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);};
  render();save();
}
init().catch(error=>{document.querySelector("main").textContent=error.message;});
</script>
</body>
</html>
"""


class ListeningHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        manifest_bytes: bytes,
        resolver: AudioResolver,
    ) -> None:
        self.manifest_bytes = manifest_bytes
        self.resolver = resolver
        super().__init__(address, ListeningRequestHandler)


class ListeningRequestHandler(http.server.BaseHTTPRequestHandler):
    server: ListeningHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"listening-server: {self.address_string()} {format % args}\n")

    def _security_headers(self) -> None:
        self.send_header("Cache-Control", "private, no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")

    def _send_bytes(self, content: bytes, content_type: str, *, head: bool) -> None:
        self.send_response(200)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if not head:
            self.wfile.write(content)

    def _send_error(self, status: int, message: str, *, head: bool) -> None:
        content = (json.dumps({"error": message}) + "\n").encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if not head:
            self.wfile.write(content)

    def _serve_audio(self, token: str, *, head: bool) -> None:
        if AUDIO_TOKEN_RE.fullmatch(token) is None:
            self._send_error(404, "not found", head=head)
            return
        try:
            path = self.server.resolver.resolve(token)
            handle = path.open("rb")
        except AudioUnavailable as exc:
            print(f"listening-server: {token} unavailable: {exc}", file=sys.stderr)
            self._send_error(503, "audio unavailable", head=head)
            return
        except (ListeningServerError, OSError) as exc:
            print(f"listening-server: {token} rejected: {exc}", file=sys.stderr)
            self._send_error(500, "audio resolution failed", head=head)
            return
        with handle:
            size = os.fstat(handle.fileno()).st_size
            try:
                selected = parse_byte_range(self.headers.get("Range"), size)
            except RangeNotSatisfiable:
                self.send_response(416)
                self._security_headers()
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start = 0 if selected is None else selected.start
            end = size - 1 if selected is None else selected.end
            length = max(0, end - start + 1)
            self.send_response(200 if selected is None else 206)
            self._security_headers()
            self.send_header("Content-Type", _audio_content_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if selected is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if head or length == 0:
                return
            handle.seek(start)
            remaining = length
            try:
                while remaining:
                    chunk = handle.read(min(COPY_CHUNK_BYTES, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return

    def _dispatch(self, *, head: bool) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self._send_bytes(_INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8", head=head)
            return
        if path == "/api/manifest":
            self._send_bytes(
                self.server.manifest_bytes,
                "application/json; charset=utf-8",
                head=head,
            )
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self._security_headers()
            self.end_headers()
            return
        prefix = "/audio/"
        if path.startswith(prefix) and "/" not in path[len(prefix) :]:
            self._serve_audio(path[len(prefix) :], head=head)
            return
        self._send_error(404, "not found", head=head)

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch(head=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch(head=True)

    def do_POST(self) -> None:  # noqa: N802
        self._send_error(405, "method not allowed", head=False)


def _default_cache_root(packet_id: str) -> Path:
    return Path.home() / ".cache" / "poweramp-start-radio" / "blind-audio" / packet_id


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument(
        "--windows-root",
        type=Path,
        required=True,
        help="host directory mirroring the Windows library root",
    )
    parser.add_argument(
        "--windows-library-prefix",
        default=str(DEFAULT_WINDOWS_LIBRARY_PREFIX),
        help=r"prefix stored in packet paths (default: C:\Music)",
    )
    parser.add_argument("--host-root", type=Path, action="append", default=[])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--allow-adb-pull",
        action="store_true",
        help="allow read-only adb pull into the local cache for phone-only audio",
    )
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--adb-serial")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="resolve and decode-probe every unique source, print path-free counts, then exit",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        if args.preflight_only and args.open_browser:
            raise ListeningServerError("--preflight-only cannot be combined with --open-browser")
        loaded = load_packet(args.packet_dir)
        packet_id = _required_text(loaded.manifest.get("packetId"), "packetId")
        cache_root = args.cache_dir or _default_cache_root(packet_id)
        resolver = AudioResolver(
            loaded.audio,
            windows_root=args.windows_root,
            windows_library_prefix=PureWindowsPath(args.windows_library_prefix),
            cache_root=cache_root,
            host_roots=args.host_root,
            allow_adb_pull=args.allow_adb_pull,
            adb=args.adb,
            adb_serial=args.adb_serial,
            ffmpeg=args.ffmpeg,
        )
        if args.preflight_only:
            result = {
                "packetId": loaded.manifest.get("packetId"),
                **preflight_packet_audio(
                    loaded.audio,
                    resolver,
                    ffprobe=args.ffprobe,
                ),
                "adbPullEnabled": args.allow_adb_pull,
            }
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
            return 0 if result["state"] == "READY" else 3
        if not 0 <= args.port <= 65535:
            raise ListeningServerError("--port must be between 0 and 65535")
        server = ListeningHTTPServer(("127.0.0.1", args.port), loaded.manifest_bytes, resolver)
    except (ListeningServerError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    host, port = server.server_address
    url = f"http://{host}:{port}/"
    print(
        json.dumps(
            {
                "state": "SERVING",
                "url": url,
                "packetId": loaded.manifest.get("packetId"),
                "trialCount": len(loaded.manifest.get("trials", [])),
                "audioTokenCount": len(loaded.audio),
                "adbPullEnabled": args.allow_adb_pull,
                "cacheDir": str(cache_root),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

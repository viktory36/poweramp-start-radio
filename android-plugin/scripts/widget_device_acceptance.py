#!/usr/bin/env python3
"""Pure parsing and validation for connected V2 widget acceptance evidence."""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


PACKAGE = "com.powerampstartradio.v2"
V1_PACKAGE = "com.powerampstartradio"
POWERAMP_PACKAGE = "com.maxmpz.audioplayer"
MUSIXMATCH_OVERLAY_PACKAGE = "com.musixmatch.android.lyrify"
WIDGET_PROVIDER = f"{PACKAGE}/com.powerampstartradio.widget.StartRadioWidgetReceiver"
WIDGET_ROOT_DESCRIPTION = "Open Poweramp Start Radio V2"
NO_TRACK_TITLE = "No track playing"
NO_TRACK_SUBTITLE = "Play in Poweramp"
NO_TRACK_ACTION = "Open Poweramp to play a track"
REFRESH_SUBTITLE = "Open Poweramp to refresh"
REFRESH_ACTION = "Open Poweramp to refresh playback state"

VISIBLE_STATUS_SUBTITLES = {
    "STARTING": "Starting radio...",
    "BUSY": "Radio already starting",
    "WAITING_FOR_INDEXING": "Waiting for indexing",
    "SUCCEEDED": "Radio queued",
    "PARTIAL_FAILED": "Queue incomplete \u00b7 Open app",
    "CANCELLED": "Radio cancelled",
    "FAILED": "Radio failed \u00b7 Open app",
}


class AcceptanceError(RuntimeError):
    pass


def visible_status_subtitle(state: str) -> str:
    try:
        return VISIBLE_STATUS_SUBTITLES[state]
    except KeyError as error:
        raise AcceptanceError(f"unsupported widget request state: {state!r}") from error


@dataclass(frozen=True)
class QueueRow:
    queue_id: int
    file_id: int
    sort: int


@dataclass(frozen=True)
class WidgetView:
    host_description: str
    root_description: str
    action_description: str
    title: str
    subtitle: str
    action_bounds: tuple[int, int, int, int]

    @property
    def state(self) -> str:
        if self.title == NO_TRACK_TITLE:
            return "NO_TRACK"
        if self.action_description == REFRESH_ACTION or self.subtitle == REFRESH_SUBTITLE:
            return "REFRESH_POWERAMP"
        if self.action_description.startswith("Start radio from "):
            return "READY"
        return "UNKNOWN"


QUEUE_ROW_RE = re.compile(
    r"^Row:\s+\d+\s+_id=(\d+),\s+folder_file_id=(\d+),\s+sort=(\d+)\s*$"
)
POWERAMP_FILE_ROW_RE = re.compile(
    r"^Row:\s+\d+\s+_id=(?P<real_id>\d+),\s+"
    r"artist=(?P<artist>.*?),\s+album=(?P<album>.*?),\s+"
    r"title_tag=(?P<title>.*?),\s+duration=(?P<duration>-?\d+),\s+"
    r"path=(?P<folder>.*?),\s+name=(?P<name>.*?),\s+"
    r"offset_ms=(?P<offset>NULL|-?\d+),\s+"
    r"cue_folder_id=(?P<cue>NULL|-?\d+)\s*$"
)
MEDIA_SESSION_OWNER_RE = re.compile(
    r"(?P<session>[A-Za-z0-9_.]+/\S+)\s+\(userId=\d+\)\s*$"
)
WINDOW_HEADER_RE = re.compile(r"^\s*Window #\d+ Window\{(?P<token>[^\s}]+).*$")
WINDOW_FRAME_RE = re.compile(
    r"(?:^|\s)(?:mFrame|frame)=\[(-?\d+),(-?\d+)]\[(-?\d+),(-?\d+)]"
)
WINDOW_SIZE_FRAME_RE = re.compile(
    r"(?:^|\s)(?:mFrame|frame)=\((-?\d+),(-?\d+)\)\((\d+)x(\d+)\)"
)
BOUNDS_RE = re.compile(r"\[(\d+),(\d+)]\[(\d+),(\d+)]")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
POWERAMP_STATE_NAMES = {
    0: "NONE",
    1: "STOPPED",
    2: "PAUSED",
    3: "PLAYING",
    4: "FAST_FORWARDING",
    5: "REWINDING",
    6: "BUFFERING",
    7: "ERROR",
    8: "CONNECTING",
    9: "SKIPPING_TO_PREVIOUS",
    10: "SKIPPING_TO_NEXT",
    11: "SKIPPING_TO_QUEUE_ITEM",
}
_AUDIO_EXTENSIONS = {
    ".mp3",
    ".flac",
    ".opus",
    ".ogg",
    ".m4a",
    ".aac",
    ".wav",
    ".wma",
    ".ape",
    ".wv",
    ".alac",
    ".aiff",
    ".aif",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def parse_queue_projection(raw: str) -> list[QueueRow]:
    rows: list[QueueRow] = []
    for line in raw.replace("\r", "").splitlines():
        if not line.strip():
            continue
        match = QUEUE_ROW_RE.fullmatch(line.strip())
        if match is None:
            if line.strip() == "No result found.":
                continue
            raise AcceptanceError(f"unrecognized Poweramp queue row: {line!r}")
        rows.append(QueueRow(*(int(value) for value in match.groups())))
    if len({row.queue_id for row in rows}) != len(rows):
        raise AcceptanceError("Poweramp queue contains duplicate occurrence IDs")
    if any(row.file_id <= 0 or row.queue_id <= 0 or row.sort <= 0 for row in rows):
        raise AcceptanceError("Poweramp queue contains a non-positive identity")
    ordered = sorted(rows, key=lambda row: (row.sort, row.queue_id))
    if ordered != rows:
        raise AcceptanceError("Poweramp queue provider output is not in sort order")
    return rows


def _shared_preferences(preferences_xml: bytes) -> dict[str, Any]:
    try:
        root = ET.fromstring(preferences_xml)
    except ET.ParseError as error:
        raise AcceptanceError(f"Poweramp display preferences are malformed: {error}") from error
    if root.tag != "map":
        raise AcceptanceError("Poweramp display preferences have no map root")
    result: dict[str, Any] = {}
    for node in root:
        name = node.attrib.get("name")
        if not name or name in result:
            raise AcceptanceError("Poweramp display preferences contain an invalid key")
        if node.tag == "string":
            result[name] = node.text or ""
        elif node.tag in ("int", "long"):
            raw_value = node.attrib.get("value")
            try:
                result[name] = int(raw_value) if raw_value is not None else None
            except ValueError as error:
                raise AcceptanceError(
                    f"Poweramp display preference {name!r} is not an integer"
                ) from error
        elif node.tag in ("boolean", "float"):
            result[name] = node.attrib.get("value")
        elif node.tag == "set":
            result[name] = tuple(child.text or "" for child in node.findall("string"))
        else:
            raise AcceptanceError(f"unsupported Poweramp display preference type {node.tag!r}")
    return result


def parse_poweramp_display_track(preferences_xml: bytes) -> dict[str, Any]:
    """Read the exact display track published by PowerampReceiver.

    This preference is presentation evidence, not authentication. The caller must independently
    verify it against Poweramp's provider before treating it as a widget command seed.
    """
    values = _shared_preferences(preferences_xml)
    required = {
        "current_track_real_id": int,
        "current_track_title": str,
        "current_track_duration_ms": int,
        "current_track_path": str,
        "current_track_category_row_id": int,
    }
    for key, expected_type in required.items():
        if not isinstance(values.get(key), expected_type):
            raise AcceptanceError(f"Poweramp display preferences contain no exact {key}")
    real_id = values["current_track_real_id"]
    title = values["current_track_title"]
    duration_ms = values["current_track_duration_ms"]
    path = values["current_track_path"]
    if real_id <= 0 or not title.strip() or duration_ms < 0 or not path.strip():
        raise AcceptanceError("Poweramp display preferences contain an invalid track identity")
    for key in ("current_track_artist", "current_track_album", "current_track_category_uri"):
        if key in values and not isinstance(values[key], str):
            raise AcceptanceError(f"Poweramp display preference {key!r} has the wrong type")
    position = values.get("current_track_position_in_list")
    if position is not None and not isinstance(position, int):
        raise AcceptanceError("Poweramp display position has the wrong type")
    return {
        "realId": real_id,
        "title": title,
        "artist": values.get("current_track_artist"),
        "album": values.get("current_track_album"),
        "durationMs": duration_ms,
        "path": path,
        "trackId": values["current_track_category_row_id"],
        "categoryUri": values.get("current_track_category_uri"),
        "positionInList": position,
    }


def _provider_text(raw: str) -> str | None:
    return None if raw == "NULL" else raw


def parse_poweramp_file_row(raw: str) -> dict[str, Any]:
    lines = [line.strip() for line in raw.replace("\r", "").splitlines() if line.strip()]
    if lines == ["No result found."]:
        raise AcceptanceError("current Poweramp file ID is absent from the provider")
    if len(lines) != 1:
        raise AcceptanceError("current Poweramp file ID did not resolve to exactly one provider row")
    match = POWERAMP_FILE_ROW_RE.fullmatch(lines[0])
    if match is None:
        raise AcceptanceError(f"unrecognized Poweramp file provider row: {lines[0]!r}")
    folder = _provider_text(match.group("folder"))
    name = _provider_text(match.group("name"))
    path = None
    if name:
        path = posixpath.join(folder or "", name)
    duration_ms = max(0, int(match.group("duration")))
    return {
        "realId": int(match.group("real_id")),
        "artist": _provider_text(match.group("artist")) or "",
        "album": _provider_text(match.group("album")) or "",
        "title": _provider_text(match.group("title")) or "",
        "durationMs": duration_ms,
        "path": path,
        "offsetMs": None if match.group("offset") == "NULL" else int(match.group("offset")),
        "cueFolderId": None if match.group("cue") == "NULL" else int(match.group("cue")),
    }


def _normalize_nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _normalize_artist(value: Any) -> str:
    normalized = str(value or "").lower().strip()
    if normalized == "unknown artist":
        normalized = ""
    return _normalize_nfc(normalized).replace("|", "/")


def _normalize_album(value: Any) -> str:
    return _normalize_nfc(str(value or "").lower().strip()).replace("|", "/")


def _normalize_title(value: Any) -> str:
    normalized = str(value or "").lower().strip()
    dot = normalized.rfind(".")
    if dot > 0 and normalized[dot:] in _AUDIO_EXTENSIONS:
        normalized = normalized[:dot]
    return _normalize_nfc(normalized).replace("|", "/")


def _is_queue_category(category_uri: Any) -> bool:
    if not isinstance(category_uri, str) or not category_uri.strip():
        return False
    parsed = urlsplit(category_uri)
    return (
        parsed.scheme == "content"
        and parsed.netloc == "com.maxmpz.audioplayer.data"
        and parsed.path.rstrip("/") == "/queue"
    )


def poweramp_track_claims_queue(track: Mapping[str, Any]) -> bool:
    return _is_queue_category(track.get("categoryUri"))


def validate_provider_backed_display_track(
    track: Mapping[str, Any],
    provider: Mapping[str, Any],
    queue_rows: Sequence[QueueRow] | None,
) -> dict[str, Any]:
    """Mirror production's provider and Queue-occurrence proof for one display track."""
    if track.get("realId") != provider.get("realId"):
        raise AcceptanceError("Poweramp display and provider file IDs differ")
    display_path = track.get("path")
    provider_path = provider.get("path")
    if not isinstance(display_path, str) or not isinstance(provider_path, str):
        raise AcceptanceError("Poweramp display or provider path is unavailable")
    if _normalize_nfc(display_path) != _normalize_nfc(provider_path):
        raise AcceptanceError("Poweramp display and provider paths differ")
    comparisons = (
        (_normalize_title(track.get("title")), _normalize_title(provider.get("title")), "title"),
        (_normalize_artist(track.get("artist")), _normalize_artist(provider.get("artist")), "artist"),
        (_normalize_album(track.get("album")), _normalize_album(provider.get("album")), "album"),
    )
    for displayed, provided, field in comparisons:
        if displayed != provided:
            raise AcceptanceError(f"Poweramp display and provider {field} differ")
    display_duration = track.get("durationMs")
    provider_duration = provider.get("durationMs")
    if not isinstance(display_duration, int) or not isinstance(provider_duration, int):
        raise AcceptanceError("Poweramp display or provider duration is invalid")
    if (
        display_duration > 0
        and provider_duration > 0
        and abs(display_duration - provider_duration) > 5_000
    ):
        raise AcceptanceError("Poweramp display and provider durations differ")

    category_uri = track.get("categoryUri")
    track_id = track.get("trackId")
    if _is_queue_category(category_uri):
        if not isinstance(track_id, int) or track_id <= 0 or queue_rows is None:
            raise AcceptanceError("current Queue track has no revalidated occurrence")
        matches = [row for row in queue_rows if row.queue_id == track_id]
        if len(matches) != 1 or matches[0].file_id != track["realId"]:
            raise AcceptanceError("current Queue occurrence no longer matches Poweramp")
    elif queue_rows is not None:
        raise AcceptanceError("non-Queue playback supplied unrelated queue evidence")

    return {
        **dict(track),
        "durationMs": provider_duration,
        "path": provider_path,
    }


def parse_poweramp_media_session(raw: str) -> dict[str, Any]:
    """Extract Poweramp's own PlaybackState, ignoring every other media session."""
    lines = raw.replace("\r", "").splitlines()
    sessions: list[dict[str, Any]] = []
    for start, line in enumerate(lines):
        match = MEDIA_SESSION_OWNER_RE.search(line)
        if match is None:
            continue
        session = match.group("session")
        if not session.startswith(f"{POWERAMP_PACKAGE}/"):
            continue
        indentation = len(line) - len(line.lstrip())
        end = start + 1
        while end < len(lines):
            child = lines[end]
            if child.strip() and len(child) - len(child.lstrip()) <= indentation:
                break
            end += 1
        block = "\n".join(lines[start:end])
        if re.search(rf"^\s+package={re.escape(POWERAMP_PACKAGE)}\s*$", block, re.MULTILINE) is None:
            continue
        state_matches = re.findall(
            r"\bstate=PlaybackState\s*\{state=(?:([A-Z_]+)\()?([0-9]+)\)?(?=,|\s|})",
            block,
        )
        if len(state_matches) != 1:
            raise AcceptanceError("Poweramp media session has no unambiguous PlaybackState")
        reported_name, reported_code = state_matches[0]
        state_code = int(reported_code)
        state_name = POWERAMP_STATE_NAMES.get(state_code, f"UNKNOWN_{state_code}")
        if reported_name and reported_name != state_name:
            raise AcceptanceError("Poweramp media session state name and code disagree")
        sessions.append(
            {
                "session": session,
                "stateCode": state_code,
                "state": state_name,
            }
        )
    if len(sessions) != 1:
        raise AcceptanceError(
            f"dumpsys exposes {len(sessions)} Poweramp media sessions; expected exactly one"
        )
    return sessions[0]


def parse_overlay_windows(
    raw: str,
    *,
    package: str = MUSIXMATCH_OVERLAY_PACKAGE,
) -> list[dict[str, Any]]:
    """Parse visible package-owned window frames from `dumpsys window windows`."""
    lines = raw.replace("\r", "").splitlines()
    starts = [index for index, line in enumerate(lines) if WINDOW_HEADER_RE.match(line)]
    result: list[dict[str, Any]] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        block_lines = lines[start:end]
        block = "\n".join(block_lines)
        if package not in block:
            continue
        if re.search(r"\b(?:isOnScreen|isVisible|mHasSurface)=false\b", block):
            continue
        header = WINDOW_HEADER_RE.match(block_lines[0])
        assert header is not None
        frames = WINDOW_FRAME_RE.findall(block)
        sized_frames = WINDOW_SIZE_FRAME_RE.findall(block)
        if len(frames) + len(sized_frames) != 1:
            raise AcceptanceError(
                f"{package} window has no unambiguous visible frame: {block_lines[0].strip()}"
            )
        if frames:
            left, top, right, bottom = (int(value) for value in frames[0])
        else:
            left, top, width, height = (int(value) for value in sized_frames[0])
            right, bottom = left + width, top + height
        if right <= left or bottom <= top:
            raise AcceptanceError(f"{package} window has an empty frame")
        result.append(
            {
                "token": header.group("token"),
                "frame": [left, top, right, bottom],
                "width": right - left,
                "height": bottom - top,
            }
        )
    return sorted(result, key=lambda item: (item["frame"], item["token"]))


def classify_overlay_windows(
    windows: Sequence[Mapping[str, Any]],
    *,
    display_width: int,
    display_height: int,
) -> dict[str, Any]:
    if display_width <= 0 or display_height <= 0:
        raise AcceptanceError("display size is invalid")
    normalized: list[dict[str, Any]] = []
    for window in windows:
        frame = window.get("frame")
        if (
            not isinstance(frame, Sequence)
            or isinstance(frame, (str, bytes))
            or len(frame) != 4
            or any(not isinstance(value, int) for value in frame)
        ):
            raise AcceptanceError("overlay window has an invalid frame")
        left, top, right, bottom = frame
        if right <= left or bottom <= top:
            raise AcceptanceError("overlay window has an empty frame")
        normalized.append(
            {
                "frame": [left, top, right, bottom],
                "width": right - left,
                "height": bottom - top,
            }
        )
    normalized.sort(key=lambda item: item["frame"])
    if not normalized:
        return {"state": "ABSENT", "frames": [], "collapseTarget": None}

    large = [
        item
        for item in normalized
        if item["width"] >= display_width * 0.70
        or item["height"] >= display_height * 0.35
    ]
    small = [
        item
        for item in normalized
        if item["width"] <= display_width * 0.25
        and item["height"] <= display_height * 0.25
    ]
    if len(large) == 1 and len(normalized) >= 2 and small:
        main = large[0]
        candidates = [item for item in small if item is not main]
        if len(candidates) == 1:
            return {
                "state": "EXPANDED",
                "frames": [item["frame"] for item in normalized],
                "collapseTarget": candidates[0]["frame"],
            }
    if len(normalized) == 1 and len(small) == 1:
        return {
            "state": "COLLAPSED",
            "frames": [normalized[0]["frame"]],
            "collapseTarget": None,
        }
    return {
        "state": "AMBIGUOUS",
        "frames": [item["frame"] for item in normalized],
        "collapseTarget": None,
    }


def overlay_matches_baseline(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    return (
        baseline.get("state") in ("ABSENT", "COLLAPSED")
        and current.get("state") == baseline.get("state")
        and current.get("frames") == baseline.get("frames")
    )


def queue_json(rows: Sequence[QueueRow]) -> list[dict[str, int]]:
    return [
        {"queueId": row.queue_id, "fileId": row.file_id, "sort": row.sort}
        for row in rows
    ]


def queue_file_ids(rows: Sequence[QueueRow]) -> list[int]:
    return [row.file_id for row in rows]


def _parse_bounds(raw: str) -> tuple[int, int, int, int]:
    match = BOUNDS_RE.fullmatch(raw)
    if match is None:
        raise AcceptanceError("V2 widget action has no parseable bounds")
    bounds = tuple(int(value) for value in match.groups())
    left, top, right, bottom = bounds
    if right <= left or bottom <= top:
        raise AcceptanceError("V2 widget action has empty bounds")
    return bounds


def extract_hierarchy_xml(raw: bytes) -> bytes:
    """Extract one complete uiautomator hierarchy from its device-specific stdout wrapper."""
    marker = raw.find(b"<?xml")
    terminal = b"</hierarchy>"
    end = raw.find(terminal, marker)
    if marker < 0 or end < 0:
        raise AcceptanceError("uiautomator evidence contains no complete XML hierarchy")
    xml = raw[marker : end + len(terminal)].strip()
    if raw.find(b"<?xml", marker + 1) >= 0:
        raise AcceptanceError("uiautomator evidence contains multiple XML hierarchies")
    return xml


def parse_widget_view(xml_bytes: bytes) -> WidgetView:
    xml = extract_hierarchy_xml(xml_bytes)
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        raise AcceptanceError(f"uiautomator evidence is malformed: {error}") from error

    hosts = [
        node
        for node in root.iter("node")
        if node.attrib.get("class") == "com.android.launcher3.widget.LauncherAppWidgetHostView"
        and node.attrib.get("content-desc") == "Poweramp Start Radio V2"
    ]
    if len(hosts) != 1:
        raise AcceptanceError(f"launcher exposes {len(hosts)} V2 widget host views; expected one")
    host = hosts[0]
    package_nodes = [
        node for node in host.iter("node") if node.attrib.get("package") == PACKAGE
    ]
    roots = [
        node
        for node in package_nodes
        if node.attrib.get("resource-id") == f"{PACKAGE}:id/widget_root"
    ]
    actions = [
        node
        for node in package_nodes
        if node.attrib.get("resource-id") == f"{PACKAGE}:id/widget_start_button"
    ]
    titles = [
        node
        for node in package_nodes
        if node.attrib.get("resource-id") == f"{PACKAGE}:id/widget_track_title"
    ]
    subtitles = [
        node
        for node in package_nodes
        if node.attrib.get("resource-id") == f"{PACKAGE}:id/widget_track_subtitle"
    ]
    if not all(len(items) == 1 for items in (roots, actions, titles, subtitles)):
        raise AcceptanceError("V2 widget does not expose one complete production RemoteViews tree")
    return WidgetView(
        host_description=host.attrib.get("content-desc", ""),
        root_description=roots[0].attrib.get("content-desc", ""),
        action_description=actions[0].attrib.get("content-desc", ""),
        title=titles[0].attrib.get("text", ""),
        subtitle=subtitles[0].attrib.get("text", ""),
        action_bounds=_parse_bounds(actions[0].attrib.get("bounds", "")),
    )


def validate_no_track_widget(view: WidgetView) -> None:
    expected = (WIDGET_ROOT_DESCRIPTION, NO_TRACK_ACTION, NO_TRACK_TITLE, NO_TRACK_SUBTITLE)
    actual = (view.root_description, view.action_description, view.title, view.subtitle)
    if actual != expected:
        raise AcceptanceError(f"V2 no-track widget copy is not truthful: {actual!r}")
    if view.state != "NO_TRACK":
        raise AcceptanceError("V2 widget did not classify as NO_TRACK")


def expected_playback_subtitle(track: Mapping[str, Any]) -> str:
    values = [
        value.strip()
        for key in ("artist", "album")
        if isinstance((value := track.get(key)), str) and value.strip()
    ]
    return " · ".join(values)


def validate_ready_widget(view: WidgetView, track: Mapping[str, Any]) -> None:
    title = track.get("title")
    real_id = track.get("realId")
    path = track.get("path")
    if not isinstance(title, str) or not title.strip():
        raise AcceptanceError("provider-verified Poweramp track has no title")
    if not isinstance(real_id, int) or real_id <= 0:
        raise AcceptanceError("provider-verified Poweramp track has no positive file identity")
    if not isinstance(path, str) or not path.strip():
        raise AcceptanceError("provider-verified Poweramp track has no path")
    expected = (
        f"Open Poweramp Start Radio V2. Current track: {title}",
        f"Start radio from {title}",
        title,
        expected_playback_subtitle(track),
    )
    actual = (view.root_description, view.action_description, view.title, view.subtitle)
    if actual != expected:
        raise AcceptanceError(f"V2 ready widget does not exactly match playback: {actual!r}")
    if view.state != "READY":
        raise AcceptanceError("V2 widget did not classify as READY")


def parse_authenticated_state(preferences_xml: bytes) -> dict[str, Any]:
    try:
        root = ET.fromstring(preferences_xml)
    except ET.ParseError as error:
        raise AcceptanceError(f"authenticated playback preferences are malformed: {error}") from error
    matches = [
        node
        for node in root.findall("string")
        if node.attrib.get("name") == "authenticated_explicit_state"
    ]
    if len(matches) != 1 or matches[0].text is None:
        raise AcceptanceError("authenticated playback preferences contain no exact state")
    try:
        value = json.loads(matches[0].text)
    except json.JSONDecodeError as error:
        raise AcceptanceError(f"authenticated playback state is invalid JSON: {error}") from error
    if not isinstance(value, dict) or value.get("schemaVersion") != 1:
        raise AcceptanceError("authenticated playback state has an unsupported schema")
    playback_state = value.get("playbackState")
    if playback_state not in (0, 1, 2):
        raise AcceptanceError("authenticated playback state has an invalid state")
    track = value.get("track")
    if playback_state == 0:
        if track is not None:
            raise AcceptanceError("stopped authenticated playback retains a command seed")
    elif not isinstance(track, dict):
        raise AcceptanceError("playing or paused authenticated state has no track")
    return value


def parse_widget_instances(raw: str) -> list[dict[str, Any]]:
    blocks = re.split(r"(?=^\s*\[\d+] id=\d+\s*$)", raw, flags=re.MULTILINE)
    widgets: list[dict[str, Any]] = []
    for block in blocks:
        identity = re.search(r"^\s*\[\d+] id=(\d+)\s*$", block, flags=re.MULTILINE)
        if identity is None or f"cmp:ComponentInfo{{{WIDGET_PROVIDER}}}" not in block:
            continue
        host = re.search(r"^\s*host=HostId\{[^}]*pkg=([^}]+)}", block, flags=re.MULTILINE)
        views = re.search(r"^\s*views=(.+)$", block, flags=re.MULTILINE)
        widgets.append(
            {
                "appWidgetId": int(identity.group(1)),
                "hostPackage": host.group(1) if host else None,
                "viewsAssigned": views is not None and views.group(1).strip() != "null",
            }
        )
    return sorted(widgets, key=lambda item: item["appWidgetId"])


def parse_package_stopped(raw: str) -> bool:
    match = re.search(r"^\s*User 0:.*\bstopped=(true|false)\b", raw, flags=re.MULTILINE)
    if match is None:
        raise AcceptanceError("cannot prove V2 package stopped state for user 0")
    return match.group(1) == "true"


def parse_music_volume(raw: str) -> dict[str, int]:
    match = re.search(r"volume is (\d+) in range \[(\d+)\.\.(\d+)]", raw)
    if match is None:
        raise AcceptanceError("cannot parse STREAM_MUSIC volume")
    current, minimum, maximum = (int(value) for value in match.groups())
    if not minimum <= current <= maximum:
        raise AcceptanceError("STREAM_MUSIC volume is outside its declared range")
    return {"current": current, "minimum": minimum, "maximum": maximum}


def parse_history(raw: bytes) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise AcceptanceError(f"session history is malformed: {error}") from error
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise AcceptanceError("session history is not an object list")
    request_ids = [item.get("requestId") for item in value]
    if any(not isinstance(item, str) or not item for item in request_ids):
        raise AcceptanceError("session history contains a non-durable record")
    if len(request_ids) != len(set(request_ids)):
        raise AcceptanceError("session history contains duplicate request IDs")
    return value


def terminal_delivery_identity_rows(session: Mapping[str, Any]) -> list[dict[str, int]]:
    delivery = session.get("delivery")
    tracks = session.get("tracks")
    if not isinstance(delivery, Mapping) or not isinstance(tracks, list):
        raise AcceptanceError("session has no durable delivery evidence")
    if session.get("outcome") != "SUCCEEDED" or session.get("isComplete") is not True:
        raise AcceptanceError("session is not a complete success")
    if delivery.get("verificationComplete") is not True:
        raise AcceptanceError("session queue verification is incomplete")
    expected = delivery.get("verifiedCount")
    if not isinstance(expected, int) or expected <= 0:
        raise AcceptanceError("session has no positive verified count")
    generation = session.get("generation")
    if not isinstance(generation, Mapping):
        raise AcceptanceError("session has no exact embedding generation")
    required_generation_fields = (
        "generationId",
        "activationBindingId",
        "manifestSha256",
        "embeddingSpecId",
        "databaseContentSha256",
        "orderedTrackSetSha256",
        "stableTrackUidMappingSha256",
    )
    if any(
        not isinstance(generation.get(field), str) or not generation[field].strip()
        for field in required_generation_fields
    ):
        raise AcceptanceError("session embedding generation identity is incomplete")
    provider_generation = session.get("providerGenerationId")
    if not isinstance(provider_generation, str) or not provider_generation.strip():
        raise AcceptanceError("session has no exact Poweramp provider generation")

    result: list[dict[str, int]] = []
    for track in tracks:
        if not isinstance(track, Mapping) or track.get("status") != "QUEUED":
            continue
        embedded = track.get("track")
        embedded_track_id = embedded.get("id") if isinstance(embedded, Mapping) else None
        file_id = track.get("resolvedPowerampFileId")
        queue_id = track.get("resolvedPowerampQueueId")
        if not isinstance(embedded_track_id, int) or embedded_track_id <= 0:
            raise AcceptanceError("verified session track has no exact embedded track ID")
        if not isinstance(file_id, int) or file_id <= 0:
            raise AcceptanceError("verified session track has no exact Poweramp file ID")
        if not isinstance(queue_id, int) or queue_id <= 0:
            raise AcceptanceError("verified session track has no exact Poweramp queue occurrence ID")
        result.append(
            {
                "embeddedTrackId": embedded_track_id,
                "fileId": file_id,
                "queueId": queue_id,
            }
        )
    if len(result) != expected:
        raise AcceptanceError("session verified count does not match its exact queued tracks")
    if len({row["queueId"] for row in result}) != len(result):
        raise AcceptanceError("session reuses one Poweramp queue occurrence for multiple rows")
    return result


def terminal_delivery_file_ids(session: Mapping[str, Any]) -> list[int]:
    return [row["fileId"] for row in terminal_delivery_identity_rows(session)]


def find_restore_session(
    history: Sequence[Mapping[str, Any]], baseline_file_ids: Sequence[int]
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for session in history:
        try:
            files = terminal_delivery_file_ids(session)
        except AcceptanceError:
            continue
        generation = session.get("generation")
        seed = session.get("seedTrack")
        if (
            files == list(baseline_file_ids)
            and isinstance(generation, Mapping)
            and isinstance(seed, Mapping)
            and isinstance(seed.get("title"), str)
            and seed.get("title", "").strip()
        ):
            candidates.append(dict(session))
    if not candidates:
        raise AcceptanceError("current Poweramp queue has no exact replayable modern session")
    return candidates[-1]


def widget_status_seed(status: Mapping[str, Any]) -> Mapping[str, Any]:
    if status.get("schemaVersion") != 1:
        raise AcceptanceError("widget status has an unsupported schema")
    request_id = status.get("requestId")
    seed = status.get("seed")
    if not isinstance(request_id, str) or not request_id:
        raise AcceptanceError("widget status has no request ID")
    if not isinstance(seed, Mapping):
        raise AcceptanceError("widget status has no exact seed")
    if not isinstance(seed.get("powerampFileId"), int) or seed["powerampFileId"] <= 0:
        raise AcceptanceError("widget status seed has no exact Poweramp file ID")
    if not isinstance(seed.get("displayTitle"), str) or not seed["displayTitle"].strip():
        raise AcceptanceError("widget status seed has no display title")
    if not isinstance(seed.get("normalizedPath"), str) or not seed["normalizedPath"].strip():
        raise AcceptanceError("widget status seed has no path")
    return seed


def validate_widget_session(
    *,
    session: Mapping[str, Any],
    status: Mapping[str, Any],
    tap_track: Mapping[str, Any],
) -> list[int]:
    delivery = session.get("delivery")
    if not isinstance(delivery, Mapping) or delivery.get("origin") != "WIDGET_RADIO":
        raise AcceptanceError("terminal session is not widget-origin radio")
    if session.get("requestId") != status.get("requestId"):
        raise AcceptanceError("widget status and session request IDs differ")
    seed = widget_status_seed(status)
    session_seed = session.get("seedTrack")
    if not isinstance(session_seed, Mapping):
        raise AcceptanceError("widget session has no seed track")
    session_seed_identity = session.get("seedIdentity")
    if not isinstance(session_seed_identity, Mapping):
        raise AcceptanceError("widget session has no exact embedded seed identity")
    embedded_seed_id = session_seed_identity.get("embeddedTrackId")
    if not isinstance(embedded_seed_id, int) or embedded_seed_id <= 0:
        raise AcceptanceError("widget session has no exact embedded seed identity")
    expected_id = tap_track.get("realId")
    expected_title = tap_track.get("title")
    expected_path = tap_track.get("path")
    if (
        seed.get("powerampFileId") != expected_id
        or seed.get("displayTitle") != expected_title
        or seed.get("normalizedPath") != expected_path
        or session_seed.get("realId") != expected_id
        or session_seed.get("title") != expected_title
        or session_seed.get("path") != expected_path
    ):
        raise AcceptanceError("widget radio did not retain the exact tap-time seed")
    if (
        seed.get("embeddedTrackId") != embedded_seed_id
        or seed.get("stableTrackSpanId") != session_seed_identity.get("stableTrackSpanId")
    ):
        raise AcceptanceError("widget status and session embedded seed identities differ")
    return terminal_delivery_file_ids(session)


def plan_digest(plan: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(plan))

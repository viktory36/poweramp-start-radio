#!/usr/bin/env python3
"""Resumable connected-device acceptance for the production V2 launcher widget.

Preparation is read-only. Execution is deliberately impossible without the SHA-256 of the exact
device-bound approval plan produced by preparation. The runner never uses a debug receiver,
instrumentation, direct provider mutation, package force-stop, install, clear-data, or V1 command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from widget_device_acceptance import (
    AcceptanceError,
    NO_TRACK_ACTION,
    NO_TRACK_SUBTITLE,
    NO_TRACK_TITLE,
    MUSIXMATCH_OVERLAY_PACKAGE,
    PACKAGE,
    POWERAMP_PACKAGE,
    V1_PACKAGE,
    WidgetView,
    classify_overlay_windows,
    extract_hierarchy_xml,
    find_restore_session,
    parse_authenticated_state,
    parse_history,
    parse_music_volume,
    parse_overlay_windows,
    parse_package_stopped,
    parse_poweramp_display_track,
    parse_poweramp_file_row,
    parse_poweramp_media_session,
    parse_queue_projection,
    parse_widget_instances,
    parse_widget_view,
    plan_digest,
    overlay_matches_baseline,
    poweramp_track_claims_queue,
    queue_file_ids,
    queue_json,
    sha256_bytes,
    terminal_delivery_identity_rows,
    terminal_delivery_file_ids,
    validate_no_track_widget,
    validate_provider_backed_display_track,
    validate_ready_widget,
    validate_widget_session,
    visible_status_subtitle,
    widget_status_seed,
    write_json_atomic,
)


SCHEMA_VERSION = 2
MAIN_ACTIVITY = f"{PACKAGE}/com.powerampstartradio.MainActivity"
QUEUE_URI = "content://com.maxmpz.audioplayer.data/queue"
FILES_URI = "content://com.maxmpz.audioplayer.data/files"
FILES_PROJECTION = (
    "folder_files._id:artist:album:title_tag:folder_files.duration:path:"
    "folder_files.name:folder_files.offset_ms:cue_folder_id"
)
LOCAL_APK_RELATIVE = "android-plugin/app/build/outputs/apk/debug/app-debug.apk"
APPROVAL_FILENAME = "approval-plan.json"
STATE_FILENAME = "state.json"
WIDGET_STATUS_PATH = "files/widget/start-radio-status-v1.json"
HISTORY_PATH = "files/session_history.json"
AUTHENTICATED_STATE_PATH = "shared_prefs/poweramp_authenticated_state_v2.xml"
POWERAMP_STATE_PATH = "shared_prefs/poweramp_state.xml"
SETTINGS_PATH = "shared_prefs/settings.xml"
POLL_SECONDS = 0.35
UI_TIMEOUT_SECONDS = 40.0
DELIVERY_TIMEOUT_SECONDS = 240.0
SERVICE_TIMEOUT_SECONDS = 60.0
BOOT_TIMEOUT_SECONDS = 360.0
MAX_WIDGET_DISPLAY_LATENCY_SECONDS = 8.0
MAX_IMMEDIATE_WIDGET_STATUS_LATENCY_SECONDS = 5.0
TERMINAL_SESSION_STABILITY_SECONDS = 2.0
COLD_WIDGET_STATUS_MESSAGE = "Starting radio after library check"
MUTATION_ACK_NAME = "WIDGET_PLAYBACK_QUEUE_REBOOT"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_immediate_cold_widget_status(
    *,
    status: Mapping[str, Any],
    tap_track: Mapping[str, Any],
    prior_request_ids: set[str],
    latency_seconds: float,
) -> str:
    """Validate the listener status that only the cold-reconciliation ingress publishes."""
    request_id = status.get("requestId")
    if not isinstance(request_id, str) or not request_id or request_id in prior_request_ids:
        raise AcceptanceError("cold single tap did not publish a new request-bound status")
    if status.get("state") != "STARTING" or status.get("message") != COLD_WIDGET_STATUS_MESSAGE:
        raise AcceptanceError("cold single tap did not exercise the cold-reconciliation ingress")
    if (
        not math.isfinite(latency_seconds)
        or latency_seconds < 0.0
        or latency_seconds > MAX_IMMEDIATE_WIDGET_STATUS_LATENCY_SECONDS
    ):
        raise AcceptanceError("cold single tap did not publish its request status immediately")
    seed = widget_status_seed(status)
    if (
        seed.get("powerampFileId") != tap_track.get("realId")
        or seed.get("displayTitle") != tap_track.get("title")
        or seed.get("normalizedPath") != tap_track.get("path")
    ):
        raise AcceptanceError("cold single-tap status does not retain the exact tap-time seed")
    return request_id


class Adb:
    def __init__(self, serial: str):
        if not re.fullmatch(r"[A-Za-z0-9._:-]+", serial):
            raise AcceptanceError("ADB serial contains unsupported characters")
        self.serial = serial

    def run(
        self,
        *args: str,
        check: bool = True,
        text: bool = True,
        timeout: float | None = 60,
    ) -> subprocess.CompletedProcess[Any]:
        return subprocess.run(
            ["adb", "-s", self.serial, *args],
            check=check,
            capture_output=True,
            text=text,
            timeout=timeout,
        )

    def text(self, *args: str, timeout: float | None = 60) -> str:
        return self.run(*args, timeout=timeout).stdout.replace("\r", "")

    def bytes(self, *args: str, timeout: float | None = 60) -> bytes:
        return self.run(*args, text=False, timeout=timeout).stdout

    def shell(self, *args: str, timeout: float | None = 60) -> str:
        return self.text("shell", *args, timeout=timeout)

    def run_as_bytes(
        self, package: str, *args: str, check: bool = True, timeout: float | None = 60
    ) -> bytes:
        return self.run(
            "exec-out", "run-as", package, *args, check=check, text=False, timeout=timeout
        ).stdout

    def run_as_text(
        self, package: str, *args: str, check: bool = True, timeout: float | None = 60
    ) -> str:
        return self.run(
            "exec-out", "run-as", package, *args, check=check, timeout=timeout
        ).stdout.replace("\r", "")


class Ui:
    def __init__(self, adb: Adb, evidence_dir: Path):
        self.adb = adb
        self.evidence_dir = evidence_dir
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def dump(self) -> tuple[bytes, ET.Element]:
        attempt = self.adb.run(
            "exec-out",
            "uiautomator",
            "dump",
            "/dev/tty",
            check=False,
            text=False,
            timeout=30,
        )
        raw = attempt.stdout
        if b"<?xml" not in raw:
            remote = "/sdcard/Download/.pasr-v2-widget-acceptance.xml"
            self.adb.shell("uiautomator", "dump", remote, timeout=30)
            try:
                raw = self.adb.bytes("exec-out", "cat", remote, timeout=30)
            finally:
                self.adb.shell("rm", "-f", remote, timeout=30)
        xml = extract_hierarchy_xml(raw)
        try:
            return xml, ET.fromstring(xml)
        except ET.ParseError as error:
            raise AcceptanceError(f"uiautomator returned malformed XML: {error}") from error

    def capture(self, name: str) -> tuple[bytes, ET.Element]:
        xml, root = self.dump()
        (self.evidence_dir / f"{name}.xml").write_bytes(xml + b"\n")
        (self.evidence_dir / f"{name}.png").write_bytes(
            self.adb.bytes("exec-out", "screencap", "-p", timeout=30)
        )
        return xml, root

    @staticmethod
    def bounds(node: ET.Element) -> tuple[int, int, int, int]:
        match = re.fullmatch(r"\[(\d+),(\d+)]\[(\d+),(\d+)]", node.attrib.get("bounds", ""))
        if match is None:
            raise AcceptanceError("selected ordinary UI node has no parseable bounds")
        left, top, right, bottom = (int(value) for value in match.groups())
        if right <= left or bottom <= top:
            raise AcceptanceError("selected ordinary UI node has empty bounds")
        return left, top, right, bottom

    @staticmethod
    def parents(root: ET.Element) -> dict[ET.Element, ET.Element]:
        return {child: parent for parent in root.iter() for child in parent}

    def clickable(self, root: ET.Element, node: ET.Element) -> ET.Element | None:
        parents = self.parents(root)
        current: ET.Element | None = node
        while current is not None:
            if current.attrib.get("clickable") == "true" and current.attrib.get("enabled") == "true":
                return current
            current = parents.get(current)
        return None

    def find_clickables(
        self,
        root: ET.Element,
        *,
        text: str | None = None,
        content_desc: str | None = None,
    ) -> list[ET.Element]:
        result: list[ET.Element] = []
        seen: set[str] = set()
        for node in root.iter("node"):
            if text is not None and node.attrib.get("text") != text:
                continue
            if content_desc is not None and node.attrib.get("content-desc") != content_desc:
                continue
            target = self.clickable(root, node)
            if target is None:
                continue
            raw_bounds = target.attrib.get("bounds", "")
            if raw_bounds not in seen:
                result.append(target)
                seen.add(raw_bounds)
        return result

    def tap_node(self, node: ET.Element) -> None:
        left, top, right, bottom = self.bounds(node)
        self.adb.shell("input", "tap", str((left + right) // 2), str((top + bottom) // 2))
        time.sleep(0.45)

    def swipe_up(self) -> None:
        size = self.adb.shell("wm", "size")
        match = re.search(r"(\d+)x(\d+)", size)
        if match is None:
            raise AcceptanceError("cannot determine display size")
        width, height = (int(value) for value in match.groups())
        self.adb.shell(
            "input",
            "swipe",
            str(width // 2),
            str(int(height * 0.78)),
            str(width // 2),
            str(int(height * 0.30)),
            "350",
        )
        time.sleep(0.5)

    def find_and_tap(
        self,
        *,
        text: str | None = None,
        content_desc: str | None = None,
        occurrence: int = 0,
        scroll: bool = False,
        max_swipes: int = 24,
    ) -> None:
        for attempt in range(max_swipes + 1):
            _, root = self.dump()
            targets = self.find_clickables(root, text=text, content_desc=content_desc)
            if len(targets) > occurrence:
                self.tap_node(targets[occurrence])
                return
            if not scroll or attempt == max_swipes:
                selector = f"text={text!r}" if text is not None else f"content-desc={content_desc!r}"
                raise AcceptanceError(f"ordinary UI exposes no enabled target for {selector}")
            self.swipe_up()

    def wait_clickable(
        self,
        *,
        text: str | None = None,
        content_desc: str | None = None,
        timeout: float = UI_TIMEOUT_SECONDS,
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _, root = self.dump()
            if self.find_clickables(root, text=text, content_desc=content_desc):
                return
            time.sleep(POLL_SECONDS)
        raise AcceptanceError("timed out waiting for an ordinary production UI target")


class Runner:
    def __init__(self, repo_root: Path, output_dir: Path, serial: str):
        self.repo_root = repo_root
        self.output_dir = output_dir
        self.adb = Adb(serial)
        self.state_path = output_dir / STATE_FILENAME
        self.plan_path = output_dir / APPROVAL_FILENAME
        self.overlay_baseline: dict[str, Any] | None = None
        self.overlay_mutation_allowed = False

    def save_state(self, state: dict[str, Any]) -> None:
        state["updatedAt"] = now_iso()
        write_json_atomic(self.state_path, state)

    def load_state(self) -> dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AcceptanceError(f"cannot read durable acceptance state: {error}") from error
        if not isinstance(value, dict) or value.get("schemaVersion") != SCHEMA_VERSION:
            raise AcceptanceError("acceptance state has an unsupported schema")
        return value

    def ensure_device(self) -> None:
        if self.adb.text("get-state").strip() != "device":
            raise AcceptanceError(f"ADB target {self.adb.serial} is not in device state")

    def display_size(self) -> tuple[int, int]:
        raw = self.adb.shell("wm", "size")
        matches = re.findall(r"(?:Physical|Override) size:\s*(\d+)x(\d+)", raw)
        if not matches:
            raise AcceptanceError("cannot determine the active Android display size")
        width, height = (int(value) for value in matches[-1])
        if width <= 0 or height <= 0:
            raise AcceptanceError("Android reported an invalid display size")
        return width, height

    def overlay_state(self) -> tuple[str, dict[str, Any]]:
        width, height = self.display_size()
        raw = self.adb.shell("dumpsys", "window", "windows", timeout=90)
        windows = parse_overlay_windows(raw)
        classified = classify_overlay_windows(
            windows,
            display_width=width,
            display_height=height,
        )
        return raw, {
            **classified,
            "package": MUSIXMATCH_OVERLAY_PACKAGE,
            "displaySize": [width, height],
        }

    def ensure_overlay_baseline(self, evidence_dir: Path, name: str) -> None:
        baseline = self.overlay_baseline
        if baseline is None:
            return
        evidence_dir.mkdir(parents=True, exist_ok=True)
        raw, current = self.overlay_state()
        (evidence_dir / f"{name}-overlay-before.txt").write_text(raw, encoding="utf-8")
        write_json_atomic(evidence_dir / f"{name}-overlay-before.json", current)
        if overlay_matches_baseline(baseline, current):
            return
        if not self.overlay_mutation_allowed:
            raise AcceptanceError(
                "Musixmatch overlay changed during read-only preparation; launcher evidence is obscured"
            )
        if baseline.get("state") != "COLLAPSED" or current.get("state") != "EXPANDED":
            raise AcceptanceError(
                "Musixmatch overlay differs from baseline and has no approved collapse-only repair"
            )
        target = current.get("collapseTarget")
        if (
            not isinstance(target, list)
            or len(target) != 4
            or any(not isinstance(value, int) for value in target)
        ):
            raise AcceptanceError("expanded Musixmatch overlay has no exact collapse control")
        left, top, right, bottom = target
        width, height = current["displaySize"]
        x, y = (left + right) // 2, (top + bottom) // 2
        if not (0 <= x < width and 0 <= y < height):
            raise AcceptanceError("Musixmatch collapse control is outside the active display")
        self.adb.shell("input", "tap", str(x), str(y))
        deadline = time.monotonic() + 8
        last = current
        while time.monotonic() < deadline:
            time.sleep(0.2)
            after_raw, after = self.overlay_state()
            last = after
            if overlay_matches_baseline(baseline, after):
                (evidence_dir / f"{name}-overlay-after.txt").write_text(
                    after_raw, encoding="utf-8"
                )
                write_json_atomic(evidence_dir / f"{name}-overlay-after.json", after)
                write_json_atomic(
                    evidence_dir / f"{name}-overlay-collapse.json",
                    {
                        "ordinaryTap": [x, y],
                        "targetFrame": target,
                        "restoredExactBaselineGeometry": True,
                    },
                )
                return
        raise AcceptanceError(
            f"Musixmatch overlay did not return to frozen baseline geometry; last={last!r}"
        )

    def package_path(self, package: str) -> str:
        rows = [
            line.removeprefix("package:").strip()
            for line in self.adb.shell("pm", "path", package).splitlines()
            if line.startswith("package:/")
        ]
        if len(rows) != 1:
            raise AcceptanceError(f"{package} must have exactly one installed base APK")
        return rows[0]

    def installed_apk_hash(self, package: str) -> str:
        path = self.package_path(package)
        output = self.adb.text("exec-out", "sha256sum", path).strip()
        digest = output.split(maxsplit=1)[0] if output else ""
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise AcceptanceError(f"cannot hash installed APK for {package}")
        return digest

    def package_dump(self, package: str) -> str:
        return self.adb.shell("dumpsys", "package", package, timeout=90)

    def service_state(self) -> tuple[str, dict[str, bool]]:
        raw = self.adb.shell("dumpsys", "activity", "services", PACKAGE, timeout=60)
        return raw, {
            "radioServiceActive": "com.powerampstartradio.services.RadioService" in raw,
            "indexingServiceActive": "com.powerampstartradio.indexing.IndexingService" in raw,
        }

    def wait_services_idle(self) -> tuple[str, dict[str, bool]]:
        deadline = time.monotonic() + SERVICE_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            raw, state = self.service_state()
            if not any(state.values()):
                return raw, state
            time.sleep(POLL_SECONDS)
        raise AcceptanceError("V2 radio or indexing service did not become idle")

    def boot_id(self) -> str:
        value = self.adb.shell("cat", "/proc/sys/kernel/random/boot_id").strip()
        if not re.fullmatch(r"[0-9a-f-]{36}", value):
            raise AcceptanceError("cannot read Android boot identity")
        return value

    def boot_count(self) -> int:
        value = self.adb.shell("settings", "get", "global", "boot_count").strip()
        if not value.isdigit():
            raise AcceptanceError("cannot read Android boot count")
        return int(value)

    def music_volume(self) -> tuple[str, dict[str, int]]:
        raw = self.adb.shell("cmd", "media_session", "volume", "--stream", "3", "--get")
        return raw, parse_music_volume(raw)

    def set_music_volume(self, value: int) -> None:
        before = self.music_volume()[1]
        if not before["minimum"] <= value <= before["maximum"]:
            raise AcceptanceError("refusing music volume outside the device's reported range")
        self.adb.shell(
            "cmd", "media_session", "volume", "--stream", "3", "--set", str(value)
        )
        observed = self.music_volume()[1]
        if observed["current"] == value:
            return

        # Sony Android 16 accepts the public set command but can leave the
        # active stream unchanged. Ordinary volume keys remain authoritative
        # when a media activity is foreground, so use them as a verified
        # device-compatible fallback and return to the launcher afterwards.
        self.launch_package(POWERAMP_PACKAGE)
        key = "KEYCODE_VOLUME_UP" if observed["current"] < value else "KEYCODE_VOLUME_DOWN"
        try:
            for _ in range(before["maximum"] - before["minimum"] + 2):
                self.adb.shell("input", "keyevent", key)
                observed = self.music_volume()[1]
                if observed["current"] == value:
                    return
                if key == "KEYCODE_VOLUME_UP" and observed["current"] > value:
                    break
                if key == "KEYCODE_VOLUME_DOWN" and observed["current"] < value:
                    break
        finally:
            self.home()
        raise AcceptanceError(f"STREAM_MUSIC volume did not become {value}")

    def queue(self) -> tuple[str, list[Any]]:
        raw = self.adb.shell(
            "content",
            "query",
            "--uri",
            QUEUE_URI,
            "--projection",
            "queue._id:queue.folder_file_id:queue.sort",
            timeout=60,
        )
        return raw, parse_queue_projection(raw)

    def poweramp_file(self, file_id: int) -> tuple[str, dict[str, Any]]:
        if file_id <= 0:
            raise AcceptanceError("refusing to query a non-positive Poweramp file ID")
        raw = self.adb.shell(
            "content",
            "query",
            "--uri",
            FILES_URI,
            "--projection",
            FILES_PROJECTION,
            "--where",
            f"folder_files._id={file_id}",
            timeout=60,
        )
        return raw, parse_poweramp_file_row(raw)

    def provider_verified_display_track(self) -> dict[str, Any]:
        display_raw_before = self.private_file_bytes(PACKAGE, POWERAMP_STATE_PATH)
        assert display_raw_before is not None
        display_before = parse_poweramp_display_track(display_raw_before)
        provider_raw, provider = self.poweramp_file(display_before["realId"])
        queue_raw: str | None = None
        queue_rows = None
        if poweramp_track_claims_queue(display_before):
            queue_raw, queue_rows = self.queue()
        verified = validate_provider_backed_display_track(
            display_before,
            provider,
            queue_rows,
        )
        display_raw_after = self.private_file_bytes(PACKAGE, POWERAMP_STATE_PATH)
        assert display_raw_after is not None
        display_after = parse_poweramp_display_track(display_raw_after)
        if display_after != display_before:
            raise AcceptanceError(
                "Poweramp display identity changed while its provider row was being verified"
            )
        return {
            "displayRawBefore": display_raw_before,
            "displayRawAfter": display_raw_after,
            "track": verified,
            "providerRaw": provider_raw,
            "provider": provider,
            "queueRaw": queue_raw,
            "queue": queue_json(queue_rows) if queue_rows is not None else None,
        }

    def poweramp_media_session(self) -> tuple[str, dict[str, Any]]:
        raw = self.adb.shell("dumpsys", "media_session", timeout=60)
        return raw, parse_poweramp_media_session(raw)

    def history_bytes(self) -> bytes:
        return self.adb.run_as_bytes(PACKAGE, "cat", HISTORY_PATH, timeout=60)

    def history(self) -> list[dict[str, Any]]:
        return parse_history(self.history_bytes())

    def private_file_bytes(self, package: str, path: str, *, required: bool = True) -> bytes | None:
        exists = self.adb.run(
            "shell",
            "run-as",
            package,
            "test",
            "-f",
            path,
            check=False,
            timeout=60,
        )
        if exists.returncode != 0:
            if required:
                raise AcceptanceError(f"cannot read required {package} private file {path}")
            return None
        result = self.adb.run(
            "exec-out", "run-as", package, "cat", path, check=False, text=False, timeout=60
        )
        if result.returncode != 0:
            if required:
                raise AcceptanceError(f"cannot read required {package} private file {path}")
            return None
        return result.stdout

    def private_hashes(self, package: str, roots: Sequence[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for root in roots:
            listing = self.adb.run_as_text(package, "find", root, "-type", "f", timeout=90)
            for path in sorted(line.strip() for line in listing.splitlines() if line.strip()):
                if any(character.isspace() for character in path):
                    raise AcceptanceError(f"private path contains unsupported whitespace: {path}")
                row = self.adb.run_as_text(package, "sha256sum", path, timeout=90).strip()
                digest = row.split(maxsplit=1)[0] if row else ""
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise AcceptanceError(f"cannot hash {package} private file {path}")
                result[path] = digest
        return result

    def v1_protected_hashes(self) -> dict[str, str]:
        hashes = self.private_hashes(V1_PACKAGE, ("files", "shared_prefs", "databases", "no_backup"))
        mutable_prefixes = (
            "files/datastore/",
            "shared_prefs/poweramp_state.xml",
            "no_backup/androidx.work.workdb",
        )
        return {
            path: digest
            for path, digest in hashes.items()
            if not path.startswith(mutable_prefixes)
        }

    def v2_protected_hashes(self) -> dict[str, str]:
        hashes = self.private_hashes(PACKAGE, ("files/indexing_v2",))
        settings = self.private_file_bytes(PACKAGE, SETTINGS_PATH)
        assert settings is not None
        hashes[SETTINGS_PATH] = sha256_bytes(settings)
        return hashes

    def request_journal_hashes(self) -> dict[str, str]:
        exists = self.adb.run(
            "exec-out",
            "run-as",
            PACKAGE,
            "test",
            "-d",
            "files/radio_requests_v2",
            check=False,
        )
        if exists.returncode != 0:
            return {}
        return self.private_hashes(PACKAGE, ("files/radio_requests_v2",))

    def widget_status(self, *, required: bool = True) -> tuple[bytes | None, dict[str, Any] | None]:
        raw = self.private_file_bytes(PACKAGE, WIDGET_STATUS_PATH, required=required)
        if raw is None:
            return None, None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise AcceptanceError(f"widget status is malformed: {error}") from error
        if not isinstance(value, dict):
            raise AcceptanceError("widget status is not an object")
        return raw, value

    def authenticated_state_diagnostic(
        self,
        provider_verified_track: Mapping[str, Any],
    ) -> tuple[bytes | None, dict[str, Any]]:
        raw = self.private_file_bytes(PACKAGE, AUTHENTICATED_STATE_PATH, required=False)
        if raw is None:
            return None, {
                "present": False,
                "role": "optional sender-identity fast-path evidence",
            }
        try:
            state = parse_authenticated_state(raw)
        except AcceptanceError as error:
            return raw, {
                "present": True,
                "parseError": str(error),
                "role": "optional sender-identity fast-path evidence",
            }
        candidate = state.get("track")
        matches = isinstance(candidate, Mapping) and all(
            candidate.get(key) == provider_verified_track.get(key)
            for key in ("realId", "title", "path")
        )
        return raw, {
            "present": True,
            "state": state,
            "matchesProviderVerifiedTrack": matches,
            "role": "optional sender-identity fast-path evidence",
        }

    def widget_instances(self) -> tuple[str, list[dict[str, Any]]]:
        raw = self.adb.shell("dumpsys", "appwidget", timeout=90)
        return raw, parse_widget_instances(raw)

    def pid(self) -> str | None:
        result = self.adb.run("shell", "pidof", PACKAGE, check=False)
        value = result.stdout.strip()
        if result.returncode != 0 or not value:
            return None
        if not re.fullmatch(r"\d+(?:\s+\d+)*", value):
            raise AcceptanceError("unexpected V2 process identity")
        return value

    def home(self) -> None:
        self.adb.shell("input", "keyevent", "KEYCODE_HOME")
        time.sleep(0.7)

    def keyevent(self, key: str) -> float:
        started = time.monotonic()
        self.adb.shell("input", "keyevent", key)
        return started

    def launch_package(self, package: str) -> None:
        resolved = self.adb.shell(
            "cmd",
            "package",
            "resolve-activity",
            "--brief",
            "-a",
            "android.intent.action.MAIN",
            "-c",
            "android.intent.category.LAUNCHER",
            package,
        )
        component = next(
            (
                line.strip()
                for line in reversed(resolved.splitlines())
                if line.strip().startswith(f"{package}/")
            ),
            None,
        )
        if component is None or not re.fullmatch(
            rf"{re.escape(package)}/[A-Za-z0-9_.$]+", component
        ):
            raise AcceptanceError(f"cannot resolve the launcher activity for {package}")
        self.adb.shell(
            "am",
            "start",
            "-W",
            "-n",
            component,
            timeout=90,
        )
        time.sleep(0.8)

    def launch_v2(self) -> Ui:
        self.ensure_overlay_baseline(
            self.output_dir / "runtime-overlay",
            f"before-v2-launch-{time.monotonic_ns()}",
        )
        self.adb.shell("am", "start", "-W", "-n", MAIN_ACTIVITY, timeout=90)
        ui = Ui(self.adb, self.output_dir / "runtime-ui")
        ui.wait_clickable(content_desc="Find music", timeout=UI_TIMEOUT_SECONDS)
        return ui

    def ensure_package_not_stopped(self) -> None:
        if parse_package_stopped(self.package_dump(PACKAGE)):
            raise AcceptanceError("V2 is in Android's stopped-package state")

    def kill_process_ordinary(self, *, require_live_process: bool = True) -> dict[str, Any]:
        self.home()
        before = self.pid()
        if require_live_process and before is None:
            raise AcceptanceError("V2 process was already absent; ordinary process death was not exercised")
        self.adb.shell("am", "kill", PACKAGE)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.pid() is None:
                break
            time.sleep(0.25)
        if self.pid() is not None:
            raise AcceptanceError("ordinary am kill did not terminate the V2 process")
        self.ensure_package_not_stopped()
        _, widgets = self.widget_instances()
        if len(widgets) != 1 or not widgets[0]["viewsAssigned"]:
            raise AcceptanceError("ordinary process death removed the production V2 RemoteViews")
        return {
            "pidBefore": before,
            "pidAfter": None,
            "packageStopped": False,
            "widgetInstances": widgets,
        }

    def capture_widget(self, evidence_dir: Path, name: str) -> tuple[WidgetView, bytes]:
        self.ensure_overlay_baseline(evidence_dir, name)
        ui = Ui(self.adb, evidence_dir)
        xml, _ = ui.capture(name)
        view = parse_widget_view(xml)
        write_json_atomic(
            evidence_dir / f"{name}.json",
            {
                "state": view.state,
                "rootDescription": view.root_description,
                "actionDescription": view.action_description,
                "title": view.title,
                "subtitle": view.subtitle,
                "actionBounds": list(view.action_bounds),
            },
        )
        return view, xml

    def wait_widget(
        self,
        *,
        evidence_dir: Path,
        name: str,
        predicate: Callable[[WidgetView], bool],
        event_started: float,
        validate_ready: bool = False,
        expected_playback_state: str | None = None,
    ) -> tuple[WidgetView, dict[str, Any] | None, float]:
        if validate_ready and expected_playback_state not in ("PLAYING", "PAUSED"):
            raise AcceptanceError(
                "a READY widget observation requires an independent PLAYING or PAUSED oracle"
            )
        deadline = time.monotonic() + UI_TIMEOUT_SECONDS
        last: WidgetView | None = None
        ui = Ui(self.adb, evidence_dir)
        self.ensure_overlay_baseline(evidence_dir, name)
        while time.monotonic() < deadline:
            xml, _ = ui.dump()
            try:
                view = parse_widget_view(xml)
            except AcceptanceError:
                time.sleep(POLL_SECONDS)
                continue
            last = view
            if predicate(view):
                first_correct_display_at = time.monotonic()
                track: dict[str, Any] | None = None
                ready_observations: list[dict[str, Any]] = []
                authenticated_raw: bytes | None = None
                authenticated_diagnostic: dict[str, Any] | None = None
                if validate_ready:
                    try:
                        first = self.provider_verified_display_track()
                        first_track = first["track"]
                        validate_ready_widget(view, first_track)
                        first_media_raw, first_media = self.poweramp_media_session()
                        if first_media["state"] != expected_playback_state:
                            raise AcceptanceError(
                                "Poweramp media session has not reached the expected playback state"
                            )

                        stable_xml, _ = ui.dump()
                        stable_view = parse_widget_view(stable_xml)
                        if stable_view != view or not predicate(stable_view):
                            raise AcceptanceError(
                                "widget display changed while its provider identity was being verified"
                            )
                        second = self.provider_verified_display_track()
                        second_track = second["track"]
                        validate_ready_widget(stable_view, second_track)
                        second_media_raw, second_media = self.poweramp_media_session()
                        if second_media["state"] != expected_playback_state:
                            raise AcceptanceError(
                                "Poweramp media session changed during the stable widget read"
                            )
                        if (
                            second_track != first_track
                            or second["provider"] != first["provider"]
                            or second["queue"] != first["queue"]
                            or second_media != first_media
                        ):
                            raise AcceptanceError(
                                "Poweramp identity evidence changed during the stable widget read"
                            )
                        view = stable_view
                        xml = stable_xml
                        track = second_track
                        authenticated_raw, authenticated_diagnostic = (
                            self.authenticated_state_diagnostic(track)
                        )
                        ready_observations = [
                            {
                                "displayRawBefore": first["displayRawBefore"],
                                "displayRawAfter": first["displayRawAfter"],
                                "providerRaw": first["providerRaw"],
                                "provider": first["provider"],
                                "queueRaw": first["queueRaw"],
                                "queue": first["queue"],
                                "mediaRaw": first_media_raw,
                                "media": first_media,
                            },
                            {
                                "displayRawBefore": second["displayRawBefore"],
                                "displayRawAfter": second["displayRawAfter"],
                                "providerRaw": second["providerRaw"],
                                "provider": second["provider"],
                                "queueRaw": second["queueRaw"],
                                "queue": second["queue"],
                                "mediaRaw": second_media_raw,
                                "media": second_media,
                            },
                        ]
                    except AcceptanceError:
                        time.sleep(POLL_SECONDS)
                        continue
                self.ensure_overlay_baseline(evidence_dir, f"{name}-before-capture")
                final_xml, _ = ui.dump()
                final_view = parse_widget_view(final_xml)
                if final_view != view or not predicate(final_view):
                    time.sleep(POLL_SECONDS)
                    continue
                view = final_view
                xml = final_xml
                proof_completed_at = time.monotonic()
                latency = first_correct_display_at - event_started
                proof_seconds = proof_completed_at - first_correct_display_at
                if latency > MAX_WIDGET_DISPLAY_LATENCY_SECONDS:
                    raise AcceptanceError(
                        f"widget needed {latency:.2f}s to show production playback state"
                    )
                (evidence_dir / f"{name}.xml").write_bytes(xml + b"\n")
                (evidence_dir / f"{name}.png").write_bytes(
                    self.adb.bytes("exec-out", "screencap", "-p", timeout=30)
                )
                write_json_atomic(
                    evidence_dir / f"{name}.json",
                    {
                        "observedAt": now_iso(),
                        "latencySeconds": round(latency, 3),
                        "evidenceVerificationSeconds": round(proof_seconds, 3),
                        "state": view.state,
                        "title": view.title,
                        "subtitle": view.subtitle,
                        "rootDescription": view.root_description,
                        "actionDescription": view.action_description,
                        "actionBounds": list(view.action_bounds),
                        "track": track,
                        "expectedPlaybackState": expected_playback_state,
                        "authenticatedStateDiagnostic": authenticated_diagnostic,
                    },
                )
                if validate_ready:
                    for index, observation in enumerate(ready_observations, start=1):
                        prefix = evidence_dir / f"{name}-ready-proof-{index}"
                        (prefix.with_name(prefix.name + "-poweramp-state-before.xml")).write_bytes(
                            observation["displayRawBefore"]
                        )
                        (prefix.with_name(prefix.name + "-poweramp-state-after.xml")).write_bytes(
                            observation["displayRawAfter"]
                        )
                        (prefix.with_name(prefix.name + "-provider-row.txt")).write_text(
                            observation["providerRaw"], encoding="utf-8"
                        )
                        write_json_atomic(
                            prefix.with_name(prefix.name + "-provider-row.json"),
                            observation["provider"],
                        )
                        (prefix.with_name(prefix.name + "-media-session.txt")).write_text(
                            observation["mediaRaw"], encoding="utf-8"
                        )
                        write_json_atomic(
                            prefix.with_name(prefix.name + "-media-session.json"),
                            observation["media"],
                        )
                        if observation["queueRaw"] is not None:
                            (prefix.with_name(prefix.name + "-queue.txt")).write_text(
                                observation["queueRaw"], encoding="utf-8"
                            )
                            write_json_atomic(
                                prefix.with_name(prefix.name + "-queue.json"),
                                observation["queue"],
                            )
                    if authenticated_raw is not None:
                        (evidence_dir / f"{name}-authenticated-state-optional.xml").write_bytes(
                            authenticated_raw
                        )
                    assert authenticated_diagnostic is not None
                    write_json_atomic(
                        evidence_dir / f"{name}-authenticated-state-diagnostic.json",
                        authenticated_diagnostic,
                    )
                return view, track, latency
            time.sleep(POLL_SECONDS)
        raise AcceptanceError(f"timed out waiting for widget state; last observed {last!r}")

    def _capture_baseline_files(self, baseline: Path) -> dict[str, Any]:
        overlay_raw, overlay = self.overlay_state()
        if overlay["state"] not in ("ABSENT", "COLLAPSED"):
            raise AcceptanceError(
                "read-only preparation requires Musixmatch lyrics to be absent or collapsed"
            )
        self.overlay_baseline = overlay
        (baseline / "musixmatch-overlay.txt").write_text(overlay_raw, encoding="utf-8")
        write_json_atomic(baseline / "musixmatch-overlay.json", overlay)
        queue_raw, queue = self.queue()
        history_raw = self.history_bytes()
        history = parse_history(history_raw)
        appwidget_raw, widget_instances = self.widget_instances()
        package_v2 = self.package_dump(PACKAGE)
        package_v1 = self.package_dump(V1_PACKAGE)
        services_raw, services = self.service_state()
        volume_raw, volume = self.music_volume()
        widget_view, widget_xml = self.capture_widget(baseline, "launcher-widget")

        (baseline / "queue.txt").write_text(queue_raw, encoding="utf-8")
        write_json_atomic(baseline / "queue.json", queue_json(queue))
        (baseline / "history.raw.json").write_bytes(history_raw)
        write_json_atomic(baseline / "history.json", history)
        (baseline / "appwidget.txt").write_text(appwidget_raw, encoding="utf-8")
        write_json_atomic(baseline / "widget-instances.json", widget_instances)
        (baseline / "package-v2.txt").write_text(package_v2, encoding="utf-8")
        (baseline / "package-v1.txt").write_text(package_v1, encoding="utf-8")
        (baseline / "services.txt").write_text(services_raw, encoding="utf-8")
        (baseline / "volume.txt").write_text(volume_raw, encoding="utf-8")
        write_json_atomic(baseline / "volume.json", volume)

        if any(services.values()):
            raise AcceptanceError("V2 indexing or radio service is active")
        if parse_package_stopped(package_v2):
            raise AcceptanceError("V2 package is stopped; launch it before preparing acceptance")
        if len(widget_instances) != 1 or not widget_instances[0]["viewsAssigned"]:
            raise AcceptanceError("acceptance requires exactly one assigned V2 widget instance")
        validate_no_track_widget(widget_view)
        if not queue:
            raise AcceptanceError("Poweramp queue is empty; exact restoration cannot be proven")
        restore = find_restore_session(history, queue_file_ids(queue))
        restore_seed = restore.get("seedTrack")
        assert isinstance(restore_seed, Mapping)
        restore_title = restore_seed["title"]
        visible_restore_title = (
            restore_title.removeprefix("Replay: ")
            if restore.get("delivery", {}).get("origin") == "HISTORY_REQUEUE"
            else restore_title
        )
        if not visible_restore_title.strip():
            raise AcceptanceError("baseline restore session has no visible title")

        protected_v1 = self.v1_protected_hashes()
        protected_v2 = self.v2_protected_hashes()
        write_json_atomic(baseline / "v1-protected-sha256.json", protected_v1)
        write_json_atomic(baseline / "v2-protected-sha256.json", protected_v2)
        request_hashes = self.request_journal_hashes()
        write_json_atomic(baseline / "request-journal-sha256.json", request_hashes)

        status_raw, status = self.widget_status(required=False)
        if status_raw is not None:
            (baseline / "widget-status.json").write_bytes(status_raw)

        return {
            "queueRawSha256": sha256_bytes(queue_raw.encode()),
            "queue": queue_json(queue),
            "orderedFileIds": queue_file_ids(queue),
            "historySha256": sha256_bytes(history_raw),
            "historyRequestIds": [session["requestId"] for session in history],
            "requestJournalSha256": request_hashes,
            "restoreSourceRequestId": restore["requestId"],
            "restoreSourceTitle": restore_title,
            "restoreSourceVisibleTitle": visible_restore_title,
            "widgetInstances": widget_instances,
            "widget": {
                "state": widget_view.state,
                "title": widget_view.title,
                "subtitle": widget_view.subtitle,
                "actionDescription": widget_view.action_description,
                "xmlSha256": sha256_bytes(widget_xml),
            },
            "widgetStatusSha256": sha256_bytes(status_raw) if status_raw is not None else None,
            "musixmatchOverlay": overlay,
            "musicVolume": volume,
            "v1ProtectedSha256": protected_v1,
            "v2ProtectedSha256": protected_v2,
        }

    def _verify_preparation_stable(
        self,
        *,
        baseline: Mapping[str, Any],
        installed_v1: str,
        installed_v2: str,
        local_apk: Path,
        device: Mapping[str, Any],
    ) -> None:
        if self.installed_apk_hash(V1_PACKAGE) != installed_v1:
            raise AcceptanceError("V1 APK changed while preparing the approval plan")
        if self.installed_apk_hash(PACKAGE) != installed_v2 or sha256_file(local_apk) != installed_v2:
            raise AcceptanceError("V2 installed/local APK binding changed during preparation")
        if self.boot_id() != device["bootId"] or self.boot_count() != device["bootCount"]:
            raise AcceptanceError("device rebooted while preparing the approval plan")
        if self.adb.shell("getprop", "ro.build.fingerprint").strip() != device["fingerprint"]:
            raise AcceptanceError("device fingerprint changed while preparing the approval plan")
        queue_raw, queue = self.queue()
        if (
            sha256_bytes(queue_raw.encode()) != baseline["queueRawSha256"]
            or queue_json(queue) != baseline["queue"]
        ):
            raise AcceptanceError("Poweramp queue changed while preparing the approval plan")
        history_raw = self.history_bytes()
        if sha256_bytes(history_raw) != baseline["historySha256"]:
            raise AcceptanceError("V2 history changed while preparing the approval plan")
        if self.request_journal_hashes() != baseline["requestJournalSha256"]:
            raise AcceptanceError("V2 request journal changed while preparing the approval plan")
        if self.v1_protected_hashes() != baseline["v1ProtectedSha256"]:
            raise AcceptanceError("protected V1 state changed while preparing the approval plan")
        if self.v2_protected_hashes() != baseline["v2ProtectedSha256"]:
            raise AcceptanceError("V2 index or settings changed while preparing the approval plan")
        if self.music_volume()[1] != baseline["musicVolume"]:
            raise AcceptanceError("music volume changed while preparing the approval plan")
        _, services = self.service_state()
        if any(services.values()):
            raise AcceptanceError("a V2 service started while preparing the approval plan")
        _, instances = self.widget_instances()
        if instances != baseline["widgetInstances"]:
            raise AcceptanceError("widget instances changed while preparing the approval plan")
        stable_view, _ = self.capture_widget(self.output_dir / "baseline", "launcher-widget-stable")
        validate_no_track_widget(stable_view)

    def prepare(self) -> dict[str, Any]:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise AcceptanceError(f"output directory must be empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_device()
        local_apk = self.repo_root / LOCAL_APK_RELATIVE
        if not local_apk.is_file():
            raise AcceptanceError(f"build the reviewed V2 APK first: {local_apk}")
        installed_v2 = self.installed_apk_hash(PACKAGE)
        local_hash = sha256_file(local_apk)
        if installed_v2 != local_hash:
            raise AcceptanceError(
                f"installed V2 APK {installed_v2} differs from reviewed build {local_hash}"
            )
        installed_v1 = self.installed_apk_hash(V1_PACKAGE)
        self.package_path(POWERAMP_PACKAGE)
        device = {
            "serial": self.adb.serial,
            "fingerprint": self.adb.shell("getprop", "ro.build.fingerprint").strip(),
            "androidSdk": int(self.adb.shell("getprop", "ro.build.version.sdk").strip()),
            "bootId": self.boot_id(),
            "bootCount": self.boot_count(),
        }
        baseline_dir = self.output_dir / "baseline"
        baseline_dir.mkdir()
        baseline = self._capture_baseline_files(baseline_dir)
        self._verify_preparation_stable(
            baseline=baseline,
            installed_v1=installed_v1,
            installed_v2=installed_v2,
            local_apk=local_apk,
            device=device,
        )
        plan = {
            "schemaVersion": SCHEMA_VERSION,
            "preparedAt": now_iso(),
            "acknowledgementName": MUTATION_ACK_NAME,
            "ordinaryProductionPathsOnly": True,
            "forbidden": [
                "debug ingress or instrumentation",
                "package force-stop, install, clear-data, uninstall, permission mutation",
                "direct Poweramp provider insert, update, or delete",
                "V1 command or app launch",
                "repeating an uncertain widget tap",
                "manufactured permission, provider, indexing, or queue failure",
            ],
            "device": device,
            "packages": {
                "v1": {"name": V1_PACKAGE, "apkSha256": installed_v1},
                "v2": {"name": PACKAGE, "apkSha256": installed_v2},
                "poweramp": POWERAMP_PACKAGE,
                "lyricsOverlay": MUSIXMATCH_OVERLAY_PACKAGE,
            },
            "baseline": baseline,
            "phases": [
                {
                    "id": "display_freshness",
                    "effects": "temporarily mute, play, next, collapse Musixmatch only if Poweramp expanded it, ordinary process death, foreground repair, stop, restore volume and frozen overlay geometry",
                    "queueMutation": False,
                    "reboot": False,
                },
                {
                    "id": "cold_single_tap",
                    "effects": "pause the current Poweramp track, then tap once while the V2 process is absent; require the cold-reconciliation STARTING receipt before terminal delivery",
                    "queueMutation": True,
                    "expectedNewWidgetSessions": 1,
                    "immediateStatusDeadlineSeconds":
                        MAX_IMMEDIATE_WIDGET_STATUS_LATENCY_SECONDS,
                },
                {
                    "id": "cold_double_tap",
                    "effects": "one cold rapid double tap, then next while the accepted request runs",
                    "queueMutation": True,
                    "expectedNewWidgetSessions": 1,
                    "singleFlight": True,
                },
                {
                    "id": "history_restore",
                    "effects": "ordinary V2 History requeue with Replace upcoming",
                    "queueMutation": True,
                    "expectedNewRestoreSessions": 1,
                    "exactOrderedBaselineFileIds": True,
                },
                {
                    "id": "reboot_recovery",
                    "effects": "reboot only after queue, stopped playback, and volume restoration",
                    "reboot": True,
                    "requiresUserUnlockBeforeResume": True,
                },
            ],
            "acceptance": {
                "highSignalWidgetStates": [
                    "No track playing / Play in Poweramp",
                    "exact current title / artist and album / Start radio from title",
                    "specific request-bound busy or failure text when naturally exercised",
                ],
                "maxObservedWidgetLatencySeconds": MAX_WIDGET_DISPLAY_LATENCY_SECONDS,
                "exactTapSeed": ["Poweramp file ID", "title", "normalized path"],
                "pausedTrackRemainsCurrentAcrossWidgetTap": True,
                "historyOrigin": "WIDGET_RADIO",
                "coldSingleTapStatus": {
                    "state": "STARTING",
                    "message": COLD_WIDGET_STATUS_MESSAGE,
                    "maxLatencySeconds": MAX_IMMEDIATE_WIDGET_STATUS_LATENCY_SECONDS,
                },
                "exactlyOneSessionForColdSingleTap": True,
                "exactlyOneSessionForRapidDoubleTap": True,
                "queueRestoredByOrdinaryHistoryUi": True,
                "v1ProtectedHashesUnchanged": True,
                "v2IndexAndSettingsHashesUnchanged": True,
                "finalPlayback": "NO_TRACK",
                "finalLyricsOverlay": baseline["musixmatchOverlay"],
                "finalVolume": baseline["musicVolume"],
            },
        }
        digest = plan_digest(plan)
        write_json_atomic(self.plan_path, plan)
        state = {
            "schemaVersion": SCHEMA_VERSION,
            "status": "PREPARED",
            "preparedAt": now_iso(),
            "serial": self.adb.serial,
            "planSha256": digest,
            "progress": {},
            "pendingAction": None,
        }
        self.save_state(state)
        (self.output_dir / "APPROVAL_SHA256.txt").write_text(digest + "\n", encoding="ascii")
        return {"status": "PREPARED", "planSha256": digest, "plan": plan}

    def load_plan(self) -> dict[str, Any]:
        try:
            raw = self.plan_path.read_bytes()
            value = json.loads(raw)
        except (OSError, json.JSONDecodeError) as error:
            raise AcceptanceError(f"cannot read approval plan: {error}") from error
        if not isinstance(value, dict) or value.get("schemaVersion") != SCHEMA_VERSION:
            raise AcceptanceError("approval plan has an unsupported schema")
        return value

    def revalidate_frozen_baseline(self, plan: Mapping[str, Any], *, allow_history_growth: bool) -> None:
        self.ensure_device()
        expected_device = plan["device"]
        if self.adb.serial != expected_device["serial"]:
            raise AcceptanceError("connected ADB device differs from the frozen target")
        if self.adb.shell("getprop", "ro.build.fingerprint").strip() != expected_device["fingerprint"]:
            raise AcceptanceError("Android fingerprint differs from the frozen target")
        if self.installed_apk_hash(PACKAGE) != plan["packages"]["v2"]["apkSha256"]:
            raise AcceptanceError("installed V2 APK changed after preparation")
        if self.installed_apk_hash(V1_PACKAGE) != plan["packages"]["v1"]["apkSha256"]:
            raise AcceptanceError("installed V1 APK changed after preparation")
        local_apk = self.repo_root / LOCAL_APK_RELATIVE
        if not local_apk.is_file() or sha256_file(local_apk) != plan["packages"]["v2"]["apkSha256"]:
            raise AcceptanceError("reviewed local V2 APK changed after preparation")
        baseline = plan["baseline"]
        if self.v1_protected_hashes() != baseline["v1ProtectedSha256"]:
            raise AcceptanceError("protected V1 private state changed")
        if self.v2_protected_hashes() != baseline["v2ProtectedSha256"]:
            raise AcceptanceError("V2 indexing generation or settings changed")
        _, widget_instances = self.widget_instances()
        if widget_instances != baseline["widgetInstances"]:
            raise AcceptanceError("V2 widget instance or assigned RemoteViews state changed")
        self.ensure_package_not_stopped()
        _, services = self.service_state()
        if any(services.values()):
            raise AcceptanceError("V2 recommendation or indexing service is already active")
        queue_raw, queue = self.queue()
        if queue_file_ids(queue) != baseline["orderedFileIds"]:
            raise AcceptanceError("Poweramp queue no longer matches the frozen ordered baseline")
        if not allow_history_growth:
            if sha256_bytes(queue_raw.encode()) != baseline["queueRawSha256"]:
                raise AcceptanceError("Poweramp queue occurrence evidence changed after preparation")
            history_raw = self.history_bytes()
            if sha256_bytes(history_raw) != baseline["historySha256"]:
                raise AcceptanceError("V2 history changed after preparation")
            if self.request_journal_hashes() != baseline["requestJournalSha256"]:
                raise AcceptanceError("V2 durable request journal changed after preparation")
            volume = self.music_volume()[1]
            if volume != baseline["musicVolume"]:
                raise AcceptanceError("STREAM_MUSIC volume changed after preparation")
            view, _ = self.capture_widget(self.output_dir / "revalidation", "no-track")
            validate_no_track_widget(view)

    def _assert_baseline_queue_and_history_unchanged(self, plan: Mapping[str, Any]) -> None:
        baseline = plan["baseline"]
        queue_raw, queue = self.queue()
        if queue_file_ids(queue) != baseline["orderedFileIds"]:
            raise AcceptanceError("display-only phase changed the ordered Poweramp queue")
        if sha256_bytes(queue_raw.encode()) != baseline["queueRawSha256"]:
            raise AcceptanceError("display-only phase changed Poweramp queue occurrence evidence")
        history_raw = self.history_bytes()
        if sha256_bytes(history_raw) != baseline["historySha256"]:
            raise AcceptanceError("display-only phase changed V2 session history")
        if self.request_journal_hashes() != baseline["requestJournalSha256"]:
            raise AcceptanceError("display-only phase changed the durable request journal")

    def _restore_stopped_playback_and_volume(self, plan: Mapping[str, Any], evidence_dir: Path) -> None:
        baseline_volume = plan["baseline"]["musicVolume"]["current"]
        self.keyevent("KEYCODE_MEDIA_STOP")
        self.home()
        try:
            view, _, _ = self.wait_widget(
                evidence_dir=evidence_dir,
                name="restored-no-track",
                predicate=lambda item: item.state == "NO_TRACK",
                event_started=time.monotonic(),
            )
            validate_no_track_widget(view)
        finally:
            self.set_music_volume(baseline_volume)

    def _start_muted_poweramp_playback(
        self, plan: Mapping[str, Any], evidence_dir: Path, name: str
    ) -> tuple[WidgetView, dict[str, Any]]:
        self.set_music_volume(0)
        self.launch_package(POWERAMP_PACKAGE)
        started = self.keyevent("KEYCODE_MEDIA_PLAY")
        self.home()
        view, track, _ = self.wait_widget(
            evidence_dir=evidence_dir,
            name=name,
            predicate=lambda item: item.state == "READY",
            event_started=started,
            validate_ready=True,
            expected_playback_state="PLAYING",
        )
        if track is None:
            raise AcceptanceError("Poweramp playback did not expose an exact track")
        return view, track

    def run_display_freshness(self, plan: Mapping[str, Any], state: dict[str, Any]) -> None:
        if state["progress"].get("displayFreshness") == "COMPLETE":
            return
        evidence = self.output_dir / "display-freshness"
        evidence.mkdir(exist_ok=True)
        state["status"] = "DISPLAY_RUNNING"
        self.save_state(state)
        try:
            ready, first_track = self._start_muted_poweramp_playback(
                plan, evidence, "01-playing"
            )
            first_title = ready.title

            next_started = self.keyevent("KEYCODE_MEDIA_NEXT")
            next_view, next_track, _ = self.wait_widget(
                evidence_dir=evidence,
                name="02-next",
                predicate=lambda item: item.state == "READY" and item.title != first_title,
                event_started=next_started,
                validate_ready=True,
                expected_playback_state="PLAYING",
            )
            assert next_track is not None

            write_json_atomic(
                evidence / "03-process-death.json",
                self.kill_process_ordinary(),
            )
            killed_view, _ = self.capture_widget(evidence, "03-after-ordinary-process-death")
            validate_ready_widget(killed_view, next_track)
            next_after_kill_started = self.keyevent("KEYCODE_MEDIA_NEXT")
            cold_view, cold_track, _ = self.wait_widget(
                evidence_dir=evidence,
                name="04-next-after-process-death",
                predicate=lambda item: item.state == "READY" and item.title != next_view.title,
                event_started=next_after_kill_started,
                validate_ready=True,
                expected_playback_state="PLAYING",
            )
            assert cold_track is not None

            write_json_atomic(
                evidence / "05-process-death-before-foreground-repair.json",
                self.kill_process_ordinary(),
            )
            app_ui = self.launch_v2()
            app_ui.capture("foreground-resume")
            repair_started = time.monotonic()
            self.home()
            repaired, repaired_track, _ = self.wait_widget(
                evidence_dir=evidence,
                name="05-foreground-resume-repair",
                predicate=lambda item: item.state == "READY" and item.title == cold_view.title,
                event_started=repair_started,
                validate_ready=True,
                expected_playback_state="PLAYING",
            )
            if repaired_track != cold_track or repaired.title != cold_view.title:
                raise AcceptanceError("foreground repair did not preserve current playback identity")

            self._restore_stopped_playback_and_volume(plan, evidence)
            self._assert_baseline_queue_and_history_unchanged(plan)
            if self.v1_protected_hashes() != plan["baseline"]["v1ProtectedSha256"]:
                raise AcceptanceError("display acceptance changed protected V1 state")
            if self.v2_protected_hashes() != plan["baseline"]["v2ProtectedSha256"]:
                raise AcceptanceError("display acceptance changed V2 index or settings")
            state["progress"]["displayFreshness"] = "COMPLETE"
            state["status"] = "RUNNING"
            self.save_state(state)
        except BaseException:
            run_error = sys.exc_info()[1]
            try:
                self._restore_stopped_playback_and_volume(plan, evidence / "failure-recovery")
            except BaseException as recovery_error:
                if run_error is not None:
                    run_error.add_note(f"playback/volume recovery also failed: {recovery_error}")
            raise

    def _seed_known_embedded(self, plan: Mapping[str, Any], track: Mapping[str, Any]) -> bool:
        file_id = track.get("realId")
        if file_id in plan["baseline"]["orderedFileIds"]:
            return True
        for session in self.history():
            seed = session.get("seedTrack")
            identity = session.get("seedIdentity")
            if (
                isinstance(seed, Mapping)
                and seed.get("realId") == file_id
                and isinstance(identity, Mapping)
                and isinstance(identity.get("embeddedTrackId"), int)
                and identity["embeddedTrackId"] >= 0
            ):
                return True
        return False

    def _begin_pending(
        self,
        state: dict[str, Any],
        *,
        action: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        history_raw = self.history_bytes()
        history = parse_history(history_raw)
        queue_raw, queue = self.queue()
        status_raw, status = self.widget_status(required=False)
        pending: dict[str, Any] = {
            "action": action,
            "beganAt": now_iso(),
            "historyRequestIdsBefore": [session["requestId"] for session in history],
            "historySha256Before": sha256_bytes(history_raw),
            "requestJournalSha256Before": self.request_journal_hashes(),
            "queueBefore": queue_json(queue),
            "orderedFileIdsBefore": queue_file_ids(queue),
            "queueRawSha256Before": sha256_bytes(queue_raw.encode()),
            "widgetStatusSha256Before": sha256_bytes(status_raw) if status_raw else None,
            "widgetStatusRequestIdBefore": status.get("requestId") if status else None,
            "tapAttempted": False,
        }
        if extra:
            pending.update(extra)
        state["pendingAction"] = pending
        self.save_state(state)
        return pending

    def _mark_pending_attempted(self, state: dict[str, Any]) -> None:
        pending = state.get("pendingAction")
        if not isinstance(pending, dict):
            raise AcceptanceError("cannot mark an absent pending action")
        pending["tapAttempted"] = True
        pending["tapAttemptedAt"] = now_iso()
        self.save_state(state)

    def _new_terminal_sessions(self, pending: Mapping[str, Any]) -> list[dict[str, Any]]:
        before = set(pending["historyRequestIdsBefore"])
        return [
            session
            for session in self.history()
            if session["requestId"] not in before and session.get("outcome") is not None
        ]

    def _wait_for_one_terminal_session(self, pending: Mapping[str, Any]) -> dict[str, Any]:
        deadline = time.monotonic() + DELIVERY_TIMEOUT_SECONDS
        candidate: dict[str, Any] | None = None
        stable_since: float | None = None
        while time.monotonic() < deadline:
            sessions = self._new_terminal_sessions(pending)
            if len(sessions) > 1:
                raise AcceptanceError("one production action created more than one durable session")
            if len(sessions) == 1:
                if candidate is None:
                    candidate = sessions[0]
                    stable_since = time.monotonic()
                elif sessions[0]["requestId"] != candidate["requestId"]:
                    raise AcceptanceError("terminal widget session changed during duplicate check")
                assert stable_since is not None
                if time.monotonic() - stable_since >= TERMINAL_SESSION_STABILITY_SECONDS:
                    return candidate
            else:
                candidate = None
                stable_since = None
            time.sleep(POLL_SECONDS)
        raise AcceptanceError("timed out waiting for one terminal durable session")

    @staticmethod
    def visible_session_title(session: Mapping[str, Any]) -> str:
        seed = session.get("seedTrack")
        delivery = session.get("delivery")
        if not isinstance(seed, Mapping) or not isinstance(seed.get("title"), str):
            raise AcceptanceError("session has no visible seed title")
        title = seed["title"]
        if isinstance(delivery, Mapping) and delivery.get("origin") == "HISTORY_REQUEUE":
            title = title.removeprefix("Replay: ") or title
        return title

    def _capture_new_widget_status(
        self,
        *,
        prior_request_ids: set[str],
        evidence_dir: Path,
        event_started: float,
    ) -> tuple[bytes, dict[str, Any], list[str], float]:
        deadline = time.monotonic() + 20
        observed_states: list[str] = []
        last_raw: bytes | None = None
        last_status: dict[str, Any] | None = None
        first_request_status_at: float | None = None
        while time.monotonic() < deadline:
            raw, status = self.widget_status(required=False)
            if raw is not None and status is not None:
                request_id = status.get("requestId")
                if isinstance(request_id, str) and request_id not in prior_request_ids:
                    if first_request_status_at is None:
                        first_request_status_at = time.monotonic()
                    state = status.get("state")
                    if isinstance(state, str) and (not observed_states or observed_states[-1] != state):
                        observed_states.append(state)
                        index = len(observed_states)
                        (evidence_dir / f"widget-status-{index:02d}-{state.lower()}.json").write_bytes(raw)
                    last_raw, last_status = raw, status
                    if state in ("STARTING", "BUSY", "WAITING_FOR_INDEXING"):
                        break
            time.sleep(0.08)
        if last_raw is None or last_status is None:
            raise AcceptanceError("widget tap did not publish a new request-bound status")
        widget_status_seed(last_status)
        assert first_request_status_at is not None
        return (
            last_raw,
            last_status,
            observed_states,
            first_request_status_at - event_started,
        )

    def _validate_new_request_journal(
        self, pending: Mapping[str, Any], request_id: str, action_label: str
    ) -> dict[str, str]:
        before = pending["requestJournalSha256Before"]
        after = self.request_journal_hashes()
        changed_existing = {
            path: digest
            for path, digest in before.items()
            if path not in after or after[path] != digest
        }
        if changed_existing:
            raise AcceptanceError("widget radio changed a pre-existing durable request record")
        added = {path: digest for path, digest in after.items() if path not in before}
        matching = [path for path in added if request_id in path and path.endswith(".state")]
        unexpected = [path for path in added if request_id not in path]
        if len(matching) != 1 or unexpected:
            raise AcceptanceError(
                f"{action_label} did not produce exactly one new terminal request state"
            )
        return after

    def _write_widget_action_evidence(
        self,
        *,
        pending: Mapping[str, Any],
        session: Mapping[str, Any],
        status: Mapping[str, Any],
        tap_track: Mapping[str, Any],
        evidence: Path,
        observed_status_states: Sequence[str],
        action_label: str,
        tap_count: int,
        cold_process_absent_before_tap: bool,
        immediate_status_latency_seconds: float | None = None,
    ) -> list[int]:
        result_ids = validate_widget_session(
            session=session,
            status=status,
            tap_track=tap_track,
        )
        current_history = self.history()
        new_ids = [
            item["requestId"]
            for item in current_history
            if item["requestId"] not in set(pending["historyRequestIdsBefore"])
        ]
        if new_ids != [session["requestId"]]:
            raise AcceptanceError(
                f"{action_label} did not create exactly one durable history session"
            )
        _, current_queue = self.queue()
        session_identity = terminal_delivery_identity_rows(session)
        current_queue_identity = [
            {"fileId": row.file_id, "queueId": row.queue_id}
            for row in current_queue
        ]
        session_queue_identity = [
            {"fileId": row["fileId"], "queueId": row["queueId"]}
            for row in session_identity
        ]
        if current_queue_identity != session_queue_identity:
            raise AcceptanceError(
                "Poweramp queue does not equal the widget session's verified occurrences"
            )
        journal = self._validate_new_request_journal(
            pending,
            session["requestId"],
            action_label,
        )
        write_json_atomic(evidence / "terminal-widget-session.json", session)
        write_json_atomic(evidence / "terminal-widget-status.json", status)
        write_json_atomic(evidence / "request-journal-sha256-after.json", journal)
        write_json_atomic(
            evidence / "widget-action-validation.json",
            {
                "action": action_label,
                "requestId": session["requestId"],
                "origin": "WIDGET_RADIO",
                "tapSeed": {
                    "powerampFileId": tap_track["realId"],
                    "title": tap_track["title"],
                    "path": tap_track["path"],
                },
                "tapCount": tap_count,
                "coldProcessAbsentImmediatelyBeforeTap": cold_process_absent_before_tap,
                "durableSessionCount": 1,
                "noDuplicateTerminalSession": True,
                "observedRequestStates": list(observed_status_states),
                "verifiedQueuedTrackCount": len(result_ids),
                "exactOrderedQueue": True,
                "pausedCurrentTrackAcrossTap": immediate_status_latency_seconds is not None,
                **(
                    {
                        "immediateRequestStatus": {
                            "state": "STARTING",
                            "message": COLD_WIDGET_STATUS_MESSAGE,
                            "latencySeconds": round(immediate_status_latency_seconds, 3),
                            "maxLatencySeconds": MAX_IMMEDIATE_WIDGET_STATUS_LATENCY_SECONDS,
                            "exactTapSeed": True,
                        }
                    }
                    if immediate_status_latency_seconds is not None
                    else {}
                ),
            },
        )
        return result_ids

    def run_widget_single_tap(self, plan: Mapping[str, Any], state: dict[str, Any]) -> None:
        if state["progress"].get("widgetSingleTap") == "COMPLETE":
            return
        if state.get("pendingAction") is not None:
            raise AcceptanceError("an unresolved durable action exists; refusing another widget tap")
        evidence = self.output_dir / "widget-single-tap"
        evidence.mkdir(exist_ok=True)
        state["status"] = "WIDGET_PLAYBACK_RUNNING"
        self.save_state(state)
        try:
            ready_view, tap_track = self._start_muted_poweramp_playback(
                plan,
                evidence,
                "01-ready-before-kill",
            )
            if not self._seed_known_embedded(plan, tap_track):
                raise AcceptanceError(
                    "current playback seed is not proven embedded; refusing the queue-changing tap"
                )
            pause_started = self.keyevent("KEYCODE_MEDIA_PAUSE")
            paused_view, paused_track, _ = self.wait_widget(
                evidence_dir=evidence,
                name="02-paused-before-kill",
                predicate=lambda item: item.state == "READY" and item.title == tap_track["title"],
                event_started=pause_started,
                validate_ready=True,
                expected_playback_state="PAUSED",
            )
            if paused_track != tap_track:
                raise AcceptanceError("paused widget did not retain the exact current track")
            write_json_atomic(
                evidence / "03-cold-process-death.json",
                self.kill_process_ordinary(),
            )
            cold_started = time.monotonic()
            cold_view, cold_track, _ = self.wait_widget(
                evidence_dir=evidence,
                name="03-cold-paused-widget-before-single-tap",
                predicate=lambda item: item.state == "READY" and item.title == tap_track["title"],
                event_started=cold_started,
                validate_ready=True,
                expected_playback_state="PAUSED",
            )
            if cold_track != tap_track:
                raise AcceptanceError("cold paused widget did not retain the exact tap-time track")
            if (
                cold_view.action_bounds != ready_view.action_bounds
                or paused_view.action_bounds != ready_view.action_bounds
            ):
                raise AcceptanceError("widget action bounds changed across ordinary process death")
            if self.pid() is not None:
                raise AcceptanceError("V2 process restarted before the cold single-tap receipt")

            prior_history = self.history()
            pending = self._begin_pending(
                state,
                action="WIDGET_COLD_SINGLE_TAP",
                extra={
                    "tapTrack": dict(tap_track),
                    "widget": {
                        "title": cold_view.title,
                        "subtitle": cold_view.subtitle,
                        "actionDescription": cold_view.action_description,
                        "actionBounds": list(cold_view.action_bounds),
                    },
                },
            )
            if self.pid() is not None:
                raise AcceptanceError("V2 process was not absent immediately before the single tap")
            prior_request_ids = {session["requestId"] for session in prior_history}
            prior_status_request_id = pending.get("widgetStatusRequestIdBefore")
            if isinstance(prior_status_request_id, str):
                prior_request_ids.add(prior_status_request_id)
            self._mark_pending_attempted(state)
            left, top, right, bottom = cold_view.action_bounds
            x, y = (left + right) // 2, (top + bottom) // 2
            tap_started = time.monotonic()
            self.adb.shell("input", "tap", str(x), str(y))
            status_raw, status, status_states, status_latency = (
                self._capture_new_widget_status(
                    prior_request_ids=prior_request_ids,
                    evidence_dir=evidence,
                    event_started=tap_started,
                )
            )
            request_id = validate_immediate_cold_widget_status(
                status=status,
                tap_track=tap_track,
                prior_request_ids=prior_request_ids,
                latency_seconds=status_latency,
            )
            (evidence / "first-observed-widget-status.json").write_bytes(status_raw)
            pending["immediateStatusObservation"] = {
                "requestId": request_id,
                "state": status["state"],
                "message": status["message"],
                "latencySeconds": status_latency,
                "observedStates": list(status_states),
                "rawSha256": sha256_bytes(status_raw),
            }
            self.save_state(state)

            busy_view, busy_track, _ = self.wait_widget(
                evidence_dir=evidence,
                name="04-after-cold-paused-single-tap",
                predicate=lambda item: item.state == "READY" and item.title == tap_track["title"],
                event_started=tap_started,
                validate_ready=True,
                expected_playback_state="PAUSED",
            )
            if busy_track != tap_track:
                raise AcceptanceError("widget click changed the paused current-track identity")
            _, status_at_capture = self.widget_status()
            if status_at_capture is not None and status_at_capture.get("state") in (
                "STARTING",
                "BUSY",
                "WAITING_FOR_INDEXING",
            ):
                expected_status = visible_status_subtitle(status_at_capture["state"])
                if busy_view.title == tap_track["title"] and busy_view.subtitle != expected_status:
                    raise AcceptanceError("request-bound busy text was not visible under its seed")

            session = self._wait_for_one_terminal_session(pending)
            final_status_raw, final_status = self.widget_status()
            assert final_status_raw is not None and final_status is not None
            (evidence / "final-widget-status.json").write_bytes(final_status_raw)
            result_ids = self._write_widget_action_evidence(
                pending=pending,
                session=session,
                status=final_status,
                tap_track=tap_track,
                evidence=evidence,
                observed_status_states=status_states,
                action_label="cold single tap",
                tap_count=1,
                cold_process_absent_before_tap=True,
                immediate_status_latency_seconds=status_latency,
            )
            state["progress"]["widgetSingleTap"] = "COMPLETE"
            state["progress"]["widgetSingleRequestId"] = session["requestId"]
            state["progress"]["widgetSingleResultFileIds"] = result_ids
            state["pendingAction"] = None
            state["status"] = "WIDGET_SINGLE_COMPLETE"
            self.save_state(state)
        except BaseException:
            error = sys.exc_info()[1]
            try:
                self.set_music_volume(plan["baseline"]["musicVolume"]["current"])
            except BaseException as recovery_error:
                if error is not None:
                    error.add_note(f"volume recovery also failed: {recovery_error}")
            raise

    def run_widget_double_tap(self, plan: Mapping[str, Any], state: dict[str, Any]) -> None:
        if state["progress"].get("widgetDoubleTap") == "COMPLETE":
            return
        if state["progress"].get("widgetSingleTap") != "COMPLETE":
            raise AcceptanceError("cannot test double-tap idempotence before the cold single tap")
        if state.get("pendingAction") is not None:
            raise AcceptanceError("an unresolved durable action exists; refusing another widget tap")
        evidence = self.output_dir / "widget-double-tap"
        evidence.mkdir(exist_ok=True)
        state["status"] = "WIDGET_PLAYBACK_RUNNING"
        self.save_state(state)
        try:
            ready_view, tap_track = self._start_muted_poweramp_playback(
                plan, evidence, "01-ready-before-kill"
            )
            if not self._seed_known_embedded(plan, tap_track):
                raise AcceptanceError(
                    "current playback seed is not proven embedded; refusing the queue-changing tap"
                )
            write_json_atomic(
                evidence / "02-cold-process-death.json",
                self.kill_process_ordinary(),
            )
            cold_view, _ = self.capture_widget(evidence, "02-cold-widget-before-tap")
            validate_ready_widget(cold_view, tap_track)
            if cold_view.action_bounds != ready_view.action_bounds:
                raise AcceptanceError("widget action bounds changed across ordinary process death")
            if self.pid() is not None:
                raise AcceptanceError("V2 process was not absent immediately before the double tap")

            prior_history = self.history()
            pending = self._begin_pending(
                state,
                action="WIDGET_RAPID_DOUBLE_TAP",
                extra={
                    "tapTrack": dict(tap_track),
                    "widget": {
                        "title": cold_view.title,
                        "subtitle": cold_view.subtitle,
                        "actionDescription": cold_view.action_description,
                        "actionBounds": list(cold_view.action_bounds),
                    },
                },
            )
            prior_request_ids = {session["requestId"] for session in prior_history}
            prior_status_request_id = pending.get("widgetStatusRequestIdBefore")
            if isinstance(prior_status_request_id, str):
                prior_request_ids.add(prior_status_request_id)
            self._mark_pending_attempted(state)
            left, top, right, bottom = cold_view.action_bounds
            x, y = (left + right) // 2, (top + bottom) // 2
            tap_started = time.monotonic()
            self.adb.shell("input", "tap", str(x), str(y))
            self.adb.shell("input", "tap", str(x), str(y))
            status_raw, status, status_states, _ = self._capture_new_widget_status(
                prior_request_ids=prior_request_ids,
                evidence_dir=evidence,
                event_started=tap_started,
            )
            (evidence / "first-observed-widget-status.json").write_bytes(status_raw)
            busy_view, _ = self.capture_widget(evidence, "03-after-rapid-double-tap")
            _, status_at_capture = self.widget_status()
            if status_at_capture is not None and status_at_capture.get("state") in (
                "STARTING",
                "BUSY",
                "WAITING_FOR_INDEXING",
            ):
                expected_status = visible_status_subtitle(status_at_capture["state"])
                if busy_view.title == tap_track["title"] and busy_view.subtitle != expected_status:
                    raise AcceptanceError("request-bound busy text was not visible under its seed")

            changed_started = self.keyevent("KEYCODE_MEDIA_NEXT")
            changed_view, changed_track, _ = self.wait_widget(
                evidence_dir=evidence,
                name="04-track-changed-after-tap",
                predicate=lambda item: item.state == "READY" and item.title != tap_track["title"],
                event_started=changed_started,
                validate_ready=True,
                expected_playback_state="PLAYING",
            )
            assert changed_track is not None
            if changed_view.subtitle == visible_status_subtitle(status["state"]):
                raise AcceptanceError("old widget request status leaked under a later track")

            session = self._wait_for_one_terminal_session(pending)
            final_status_raw, final_status = self.widget_status()
            assert final_status_raw is not None and final_status is not None
            (evidence / "final-widget-status.json").write_bytes(final_status_raw)
            result_ids = self._write_widget_action_evidence(
                pending=pending,
                session=session,
                status=final_status,
                tap_track=tap_track,
                evidence=evidence,
                observed_status_states=status_states,
                action_label="rapid double tap",
                tap_count=2,
                cold_process_absent_before_tap=True,
            )
            state["progress"]["widgetDoubleTap"] = "COMPLETE"
            state["progress"]["widgetRequestId"] = session["requestId"]
            state["progress"]["widgetResultFileIds"] = result_ids
            state["pendingAction"] = None
            state["status"] = "WIDGET_COMPLETE_RESTORE_REQUIRED"
            self.save_state(state)
        except BaseException:
            error = sys.exc_info()[1]
            try:
                self.set_music_volume(plan["baseline"]["musicVolume"]["current"])
            except BaseException as recovery_error:
                if error is not None:
                    error.add_note(f"volume recovery also failed: {recovery_error}")
            raise

    def _prove_widget_history_after_process_death(
        self, state: Mapping[str, Any], evidence: Path
    ) -> Ui:
        request_id = state["progress"].get("widgetRequestId")
        matches = [session for session in self.history() if session["requestId"] == request_id]
        if len(matches) != 1:
            raise AcceptanceError("widget-origin history did not survive ordinary process death")
        session = matches[0]
        self.wait_services_idle()
        write_json_atomic(
            evidence / "history-process-death.json",
            self.kill_process_ordinary(require_live_process=False),
        )
        ui = self.launch_v2()
        ui.capture("history-app-after-process-death")
        ui.find_and_tap(content_desc="History")
        time.sleep(0.5)
        xml, root = ui.capture("history-drawer-after-process-death")
        title = self.visible_session_title(session)
        if not any(node.attrib.get("text") == title for node in root.iter("node")):
            raise AcceptanceError("widget-origin session title is absent from the production History UI")
        self.adb.shell("input", "keyevent", "KEYCODE_BACK")
        time.sleep(0.5)
        return ui

    def _prepare_restore_ui(
        self, plan: Mapping[str, Any], current_history: Sequence[Mapping[str, Any]], evidence: Path
    ) -> Ui:
        target_id = plan["baseline"]["restoreSourceRequestId"]
        target_title = plan["baseline"]["restoreSourceVisibleTitle"]
        newest_first = list(reversed(current_history))
        target_index = next(
            (index for index, session in enumerate(newest_first) if session["requestId"] == target_id),
            None,
        )
        if target_index is None:
            raise AcceptanceError("frozen exact restore session disappeared from history")
        occurrence = sum(
            1
            for session in newest_first[:target_index]
            if self.visible_session_title(session) == target_title
        )
        ui = self.launch_v2()
        ui.find_and_tap(content_desc="History")
        time.sleep(0.5)
        ui.capture("restore-history-drawer")
        ui.find_and_tap(text=target_title, occurrence=occurrence, scroll=True)
        ui.wait_clickable(content_desc="Requeue this session", timeout=UI_TIMEOUT_SECONDS)
        ui.capture("exact-baseline-session-selected")
        return ui

    def run_history_restore(self, plan: Mapping[str, Any], state: dict[str, Any]) -> None:
        if state["progress"].get("historyRestore") == "COMPLETE":
            return
        if (
            state["progress"].get("widgetSingleTap") != "COMPLETE"
            or state["progress"].get("widgetDoubleTap") != "COMPLETE"
        ):
            raise AcceptanceError("cannot restore before both widget actions have terminal receipts")
        if state.get("pendingAction") is not None:
            raise AcceptanceError("an unresolved durable action exists; refusing restore mutation")
        evidence = self.output_dir / "history-restore"
        evidence.mkdir(exist_ok=True)
        self._prove_widget_history_after_process_death(state, evidence)
        history_before = self.history()
        ui = self._prepare_restore_ui(plan, history_before, evidence)
        pending = self._begin_pending(
            state,
            action="HISTORY_RESTORE_REPLACE",
            extra={"sourceRequestId": plan["baseline"]["restoreSourceRequestId"]},
        )
        ui.find_and_tap(content_desc="Requeue this session")
        ui.wait_clickable(text="Replace upcoming", timeout=UI_TIMEOUT_SECONDS)
        ui.capture("restore-placement-dialog")
        self._mark_pending_attempted(state)
        ui.find_and_tap(text="Replace upcoming")
        restored_session = self._wait_for_one_terminal_session(pending)
        delivery = restored_session.get("delivery")
        if not isinstance(delivery, Mapping) or delivery.get("origin") != "HISTORY_REQUEUE":
            raise AcceptanceError("baseline restoration did not create a HISTORY_REQUEUE receipt")
        restored_ids = terminal_delivery_file_ids(restored_session)
        if restored_ids != plan["baseline"]["orderedFileIds"]:
            raise AcceptanceError("history restoration session differs from the frozen baseline order")
        _, queue = self.queue()
        if queue_file_ids(queue) != plan["baseline"]["orderedFileIds"]:
            raise AcceptanceError("Poweramp queue was not restored to the frozen ordered baseline")
        restored_identity = terminal_delivery_identity_rows(restored_session)
        if [row["queueId"] for row in restored_identity] != [row.queue_id for row in queue]:
            raise AcceptanceError(
                "history restoration session does not identify the current queue occurrences"
            )
        write_json_atomic(evidence / "terminal-restore-session.json", restored_session)
        write_json_atomic(
            evidence / "restore-validation.json",
            {
                "requestId": restored_session["requestId"],
                "origin": "HISTORY_REQUEUE",
                "sourceRequestId": plan["baseline"]["restoreSourceRequestId"],
                "restoredTrackCount": len(restored_ids),
                "exactOrderedBaselineFileIds": True,
            },
        )
        self._restore_stopped_playback_and_volume(plan, evidence)
        if self.v1_protected_hashes() != plan["baseline"]["v1ProtectedSha256"]:
            raise AcceptanceError("widget cohort changed protected V1 state")
        if self.v2_protected_hashes() != plan["baseline"]["v2ProtectedSha256"]:
            raise AcceptanceError("widget cohort changed V2 index or settings")
        state["progress"]["historyRestore"] = "COMPLETE"
        state["progress"]["restoreRequestId"] = restored_session["requestId"]
        state["pendingAction"] = None
        state["status"] = "RESTORED_REBOOT_PENDING"
        self.save_state(state)

    def _pending_unchanged(self, pending: Mapping[str, Any]) -> bool:
        history_raw = self.history_bytes()
        queue_raw, queue = self.queue()
        status_raw, _ = self.widget_status(required=False)
        return (
            sha256_bytes(history_raw) == pending["historySha256Before"]
            and self.request_journal_hashes() == pending["requestJournalSha256Before"]
            and queue_file_ids(queue) == pending["orderedFileIdsBefore"]
            and sha256_bytes(queue_raw.encode()) == pending["queueRawSha256Before"]
            and (sha256_bytes(status_raw) if status_raw else None)
            == pending["widgetStatusSha256Before"]
        )

    def reconcile_pending(
        self,
        plan: Mapping[str, Any],
        state: dict[str, Any],
        acknowledge_no_mutation: str | None,
    ) -> None:
        pending = state.get("pendingAction")
        if not isinstance(pending, dict):
            return
        action = pending.get("action")
        attempted = pending.get("tapAttempted") is True
        if not attempted:
            if not self._pending_unchanged(pending):
                raise AcceptanceError("an unattempted pending action has nevertheless changed device state")
            state["pendingAction"] = None
            self.save_state(state)
            return

        sessions = self._new_terminal_sessions(pending)
        if len(sessions) > 1:
            raise AcceptanceError("an interrupted production action created multiple terminal sessions")
        if not sessions:
            # The service may still own a durable action after the host process disappeared.
            try:
                session = self._wait_for_one_terminal_session(pending)
                sessions = [session]
            except AcceptanceError:
                sessions = []
        if not sessions:
            if acknowledge_no_mutation != action:
                raise AcceptanceError(
                    "pending action has no terminal receipt; it will not be repeated. Inspect evidence "
                    f"and resume with --ack-no-mutation {action} only if no action occurred"
                )
            if not self._pending_unchanged(pending):
                raise AcceptanceError("cannot acknowledge no mutation because durable device state changed")
            state["pendingAction"] = None
            self.save_state(state)
            return

        session = sessions[0]
        if action in ("WIDGET_COLD_SINGLE_TAP", "WIDGET_RAPID_DOUBLE_TAP"):
            status_raw, status = self.widget_status()
            assert status_raw is not None and status is not None
            tap_track = pending.get("tapTrack")
            if not isinstance(tap_track, Mapping):
                raise AcceptanceError("pending widget action lost its frozen tap seed")
            if action == "WIDGET_COLD_SINGLE_TAP":
                observation = pending.get("immediateStatusObservation")
                if not isinstance(observation, Mapping):
                    raise AcceptanceError(
                        "interrupted cold single tap has no immediate-status proof and cannot pass"
                    )
                immediate_path = self.output_dir / "widget-single-tap/first-observed-widget-status.json"
                try:
                    immediate_raw = immediate_path.read_bytes()
                    immediate_status = json.loads(immediate_raw)
                except (OSError, json.JSONDecodeError) as error:
                    raise AcceptanceError(
                        f"cannot recover cold single-tap status evidence: {error}"
                    ) from error
                if not isinstance(immediate_status, Mapping):
                    raise AcceptanceError("cold single-tap status evidence is not an object")
                latency = observation.get("latencySeconds")
                if (
                    not isinstance(latency, (int, float))
                    or sha256_bytes(immediate_raw) != observation.get("rawSha256")
                ):
                    raise AcceptanceError("cold single-tap status evidence changed after capture")
                immediate_request_id = validate_immediate_cold_widget_status(
                    status=immediate_status,
                    tap_track=tap_track,
                    prior_request_ids=set(pending["historyRequestIdsBefore"])
                    | ({pending["widgetStatusRequestIdBefore"]}
                       if isinstance(pending.get("widgetStatusRequestIdBefore"), str)
                       else set()),
                    latency_seconds=float(latency),
                )
                if immediate_request_id != session["requestId"]:
                    raise AcceptanceError(
                        "cold single-tap immediate status and terminal session IDs differ"
                    )
                evidence = self.output_dir / "widget-single-tap"
                observed_states = observation.get("observedStates")
                if not isinstance(observed_states, list) or not all(
                    isinstance(item, str) for item in observed_states
                ):
                    raise AcceptanceError("cold single-tap observed-state evidence is malformed")
                tap_count = 1
                action_label = "cold single tap"
                immediate_latency: float | None = float(latency)
            else:
                evidence = self.output_dir / "widget-double-tap"
                observed_states = ["UNOBSERVED_HOST_RESUME"]
                tap_count = 2
                action_label = "rapid double tap"
                immediate_latency = None
            result_ids = self._write_widget_action_evidence(
                pending=pending,
                session=session,
                status=status,
                tap_track=tap_track,
                evidence=evidence,
                observed_status_states=observed_states,
                action_label=action_label,
                tap_count=tap_count,
                cold_process_absent_before_tap=True,
                immediate_status_latency_seconds=immediate_latency,
            )
            if action == "WIDGET_COLD_SINGLE_TAP":
                state["progress"]["widgetSingleTap"] = "COMPLETE"
                state["progress"]["widgetSingleRequestId"] = session["requestId"]
                state["progress"]["widgetSingleResultFileIds"] = result_ids
                state["status"] = "WIDGET_SINGLE_COMPLETE"
            else:
                state["progress"]["widgetDoubleTap"] = "COMPLETE"
                state["progress"]["widgetRequestId"] = session["requestId"]
                state["progress"]["widgetResultFileIds"] = result_ids
                state["status"] = "WIDGET_COMPLETE_RESTORE_REQUIRED"
        elif action == "HISTORY_RESTORE_REPLACE":
            delivery = session.get("delivery")
            if not isinstance(delivery, Mapping) or delivery.get("origin") != "HISTORY_REQUEUE":
                raise AcceptanceError("interrupted restore produced the wrong session origin")
            restored_ids = terminal_delivery_file_ids(session)
            if restored_ids != plan["baseline"]["orderedFileIds"]:
                raise AcceptanceError("interrupted restore produced the wrong ordered queue")
            _, queue = self.queue()
            if queue_file_ids(queue) != restored_ids:
                raise AcceptanceError("interrupted restore receipt and current queue differ")
            restored_identity = terminal_delivery_identity_rows(session)
            if [row["queueId"] for row in restored_identity] != [row.queue_id for row in queue]:
                raise AcceptanceError(
                    "interrupted restore receipt and current queue occurrences differ"
                )
            write_json_atomic(self.output_dir / "history-restore/terminal-restore-session.json", session)
            state["progress"]["historyRestore"] = "COMPLETE"
            state["progress"]["restoreRequestId"] = session["requestId"]
            state["status"] = "RESTORED_REBOOT_PENDING"
        else:
            raise AcceptanceError(f"pending receipt has unknown action {action!r}")
        state["pendingAction"] = None
        self.save_state(state)

    def _validate_post_action_history(
        self, plan: Mapping[str, Any], state: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        history = self.history()
        baseline_ids = plan["baseline"]["historyRequestIds"]
        current_ids = [session["requestId"] for session in history]
        if current_ids[: len(baseline_ids)] != baseline_ids:
            raise AcceptanceError("pre-existing history was reordered, removed, or rewritten")
        new = history[len(baseline_ids) :]
        expected_ids = [
            state["progress"].get("widgetSingleRequestId"),
            state["progress"].get("widgetRequestId"),
            state["progress"].get("restoreRequestId"),
        ]
        if [session["requestId"] for session in new] != expected_ids:
            raise AcceptanceError(
                "widget cohort did not append exactly single-tap, double-tap, and restore sessions"
            )
        if new[0].get("delivery", {}).get("origin") != "WIDGET_RADIO":
            raise AcceptanceError("single-tap session is not widget-origin")
        if new[1].get("delivery", {}).get("origin") != "WIDGET_RADIO":
            raise AcceptanceError("double-tap session is not widget-origin")
        if new[2].get("delivery", {}).get("origin") != "HISTORY_REQUEUE":
            raise AcceptanceError("third appended session is not the baseline restore")
        if terminal_delivery_file_ids(new[2]) != plan["baseline"]["orderedFileIds"]:
            raise AcceptanceError("durable restore history differs from the baseline queue")
        return history

    def _pre_reboot_gate(self, plan: Mapping[str, Any], state: Mapping[str, Any]) -> None:
        if state["progress"].get("historyRestore") != "COMPLETE":
            raise AcceptanceError("reboot is forbidden until exact queue restoration completes")
        if state.get("pendingAction") is not None:
            raise AcceptanceError("reboot is forbidden with an unresolved queue action")
        _, queue = self.queue()
        if queue_file_ids(queue) != plan["baseline"]["orderedFileIds"]:
            raise AcceptanceError("reboot is forbidden before exact ordered queue restoration")
        if self.music_volume()[1] != plan["baseline"]["musicVolume"]:
            raise AcceptanceError("reboot is forbidden before exact volume restoration")
        self.home()
        view, _ = self.capture_widget(self.output_dir / "reboot", "pre-reboot-no-track")
        validate_no_track_widget(view)
        self._validate_post_action_history(plan, state)
        if self.v1_protected_hashes() != plan["baseline"]["v1ProtectedSha256"]:
            raise AcceptanceError("reboot is forbidden because protected V1 state changed")
        if self.v2_protected_hashes() != plan["baseline"]["v2ProtectedSha256"]:
            raise AcceptanceError("reboot is forbidden because V2 index or settings changed")

    def _wait_after_reboot(self) -> None:
        deadline = time.monotonic() + BOOT_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            result = self.adb.run("get-state", check=False, timeout=10)
            if result.returncode == 0 and result.stdout.strip() == "device":
                break
            time.sleep(2)
        else:
            raise AcceptanceError("ADB device did not return after reboot")
        while time.monotonic() < deadline:
            if self.adb.shell("getprop", "sys.boot_completed", timeout=15).strip() == "1":
                time.sleep(2)
                return
            time.sleep(2)
        raise AcceptanceError("Android did not report boot completion")

    def run_reboot(self, plan: Mapping[str, Any], state: dict[str, Any]) -> bool:
        if state["progress"].get("rebootRecovery") == "COMPLETE":
            return True
        evidence = self.output_dir / "reboot"
        evidence.mkdir(exist_ok=True)
        if state["progress"].get("rebootIssued") is not True:
            self._pre_reboot_gate(plan, state)
            pre_boot_id = self.boot_id()
            pre_boot_count = self.boot_count()
            state["progress"]["preRebootBootId"] = pre_boot_id
            state["progress"]["preRebootBootCount"] = pre_boot_count
            state["progress"]["rebootIssued"] = True
            state["status"] = "REBOOT_IN_PROGRESS"
            self.save_state(state)
            result = self.adb.run("reboot", check=False, timeout=30)
            if result.returncode != 0:
                raise AcceptanceError(f"adb reboot failed before disconnect: {result.stderr.strip()}")
            self._wait_after_reboot()
            state["status"] = "WAITING_FOR_USER_UNLOCK"
            self.save_state(state)
        else:
            self._wait_after_reboot()

        current_boot_id = self.boot_id()
        current_boot_count = self.boot_count()
        if current_boot_id == state["progress"]["preRebootBootId"]:
            raise AcceptanceError("device boot identity did not change")
        if current_boot_count != state["progress"]["preRebootBootCount"] + 1:
            raise AcceptanceError("acceptance requires exactly one reviewed reboot")
        self.home()
        try:
            view, _ = self.capture_widget(evidence, "post-reboot-before-app-launch")
            validate_no_track_widget(view)
        except AcceptanceError:
            state["status"] = "WAITING_FOR_USER_UNLOCK"
            self.save_state(state)
            return False
        self.ensure_package_not_stopped()
        _, widget_instances = self.widget_instances()
        if widget_instances != plan["baseline"]["widgetInstances"]:
            raise AcceptanceError("widget instance/RemoteViews assignment changed across reboot")
        self._validate_post_action_history(plan, state)
        _, queue = self.queue()
        if queue_file_ids(queue) != plan["baseline"]["orderedFileIds"]:
            raise AcceptanceError("Poweramp queue changed across reboot")
        if self.music_volume()[1] != plan["baseline"]["musicVolume"]:
            raise AcceptanceError("music volume changed across reboot")

        ui = self.launch_v2()
        ui.capture("post-reboot-foreground-repair")
        repair_started = time.monotonic()
        self.home()
        repaired, _, _ = self.wait_widget(
            evidence_dir=evidence,
            name="post-reboot-after-foreground-repair",
            predicate=lambda item: item.state == "NO_TRACK",
            event_started=repair_started,
        )
        validate_no_track_widget(repaired)
        state["progress"]["postRebootBootId"] = current_boot_id
        state["progress"]["postRebootBootCount"] = current_boot_count
        state["progress"]["rebootRecovery"] = "COMPLETE"
        state["status"] = "FINALIZING"
        self.save_state(state)
        return True

    def finalize(self, plan: Mapping[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        if state["progress"].get("rebootRecovery") != "COMPLETE":
            raise AcceptanceError("cannot finalize without reboot recovery")
        final = self.output_dir / "final"
        final.mkdir(exist_ok=True)
        services_raw, services = self.wait_services_idle()
        (final / "services.txt").write_text(services_raw, encoding="utf-8")
        if any(services.values()):
            raise AcceptanceError("V2 service remains active at final acceptance")
        self.ensure_package_not_stopped()
        history = self._validate_post_action_history(plan, state)
        queue_raw, queue = self.queue()
        if queue_file_ids(queue) != plan["baseline"]["orderedFileIds"]:
            raise AcceptanceError("final Poweramp queue differs from baseline")
        volume_raw, volume = self.music_volume()
        if volume != plan["baseline"]["musicVolume"]:
            raise AcceptanceError("final music volume differs from baseline")
        self.home()
        view, _ = self.capture_widget(final, "launcher-widget")
        validate_no_track_widget(view)
        overlay_raw, overlay = self.overlay_state()
        if not overlay_matches_baseline(plan["baseline"]["musixmatchOverlay"], overlay):
            raise AcceptanceError("final Musixmatch overlay geometry differs from baseline")
        if self.v1_protected_hashes() != plan["baseline"]["v1ProtectedSha256"]:
            raise AcceptanceError("protected V1 state changed during widget acceptance")
        if self.v2_protected_hashes() != plan["baseline"]["v2ProtectedSha256"]:
            raise AcceptanceError("V2 indexing generation or settings changed during widget acceptance")
        if self.installed_apk_hash(PACKAGE) != plan["packages"]["v2"]["apkSha256"]:
            raise AcceptanceError("V2 APK changed during widget acceptance")
        if self.installed_apk_hash(V1_PACKAGE) != plan["packages"]["v1"]["apkSha256"]:
            raise AcceptanceError("V1 APK changed during widget acceptance")
        (final / "queue.txt").write_text(queue_raw, encoding="utf-8")
        write_json_atomic(final / "queue.json", queue_json(queue))
        write_json_atomic(final / "history.json", history)
        (final / "volume.txt").write_text(volume_raw, encoding="utf-8")
        write_json_atomic(final / "volume.json", volume)
        (final / "musixmatch-overlay.txt").write_text(overlay_raw, encoding="utf-8")
        write_json_atomic(final / "musixmatch-overlay.json", overlay)
        report = {
            "status": "COMPLETE",
            "scope": {
                "widgetDisplayFreshness": "Observed title, subtitle, action, lifecycle, and latency",
                "radioDelivery": "Observed cold single-tap ingress, double-tap single-flight, tap seeds, durable sessions, exact queue readback, and restoration",
                "radioQueueIntelligence": "Not judged by this harness; recommendation quality requires the separate listening evaluation",
            },
            "deviceSerial": self.adb.serial,
            "planSha256": state["planSha256"],
            "displayFreshness": "PASS",
            "ordinaryProcessDeath": "PASS",
            "foregroundResumeRepair": "PASS",
            "coldSingleTapImmediateRequestStatus": "PASS",
            "coldSingleTapOneDurableSession": "PASS",
            "rapidDoubleTapSingleFlight": "PASS",
            "tapTimeSeedIdentity": "PASS",
            "pausedTrackWidgetTap": "PASS",
            "widgetOriginDurableHistory": "PASS",
            "exactOrderedQueueRestore": "PASS",
            "rebootRecovery": "PASS",
            "finalLyricsOverlay": overlay,
            "finalWidget": {
                "title": NO_TRACK_TITLE,
                "subtitle": NO_TRACK_SUBTITLE,
                "action": NO_TRACK_ACTION,
            },
            "widgetSingleRequestId": state["progress"]["widgetSingleRequestId"],
            "widgetDoubleRequestId": state["progress"]["widgetRequestId"],
            "restoreRequestId": state["progress"]["restoreRequestId"],
            "verifiedSingleTapTrackCount": len(
                state["progress"]["widgetSingleResultFileIds"]
            ),
            "verifiedDoubleTapTrackCount": len(state["progress"]["widgetResultFileIds"]),
            "restoredTrackCount": len(plan["baseline"]["orderedFileIds"]),
        }
        write_json_atomic(final / "validation.json", report)
        state["status"] = "COMPLETE"
        state["completedAt"] = now_iso()
        self.save_state(state)
        manifest: list[dict[str, str]] = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "evidence-sha256.json":
                manifest.append(
                    {"path": str(path.relative_to(self.output_dir)), "sha256": sha256_file(path)}
                )
        write_json_atomic(self.output_dir / "evidence-sha256.json", manifest)
        return report

    def execute(
        self, approval_sha256: str, acknowledge_no_mutation: str | None
    ) -> dict[str, Any]:
        plan = self.load_plan()
        state = self.load_state()
        expected = plan_digest(plan)
        if approval_sha256 != expected or state.get("planSha256") != expected:
            raise AcceptanceError("approval digest does not match the immutable device-bound plan")
        if plan.get("acknowledgementName") != MUTATION_ACK_NAME:
            raise AcceptanceError("approval plan has the wrong mutation acknowledgement")
        if state.get("serial") != self.adb.serial:
            raise AcceptanceError("resume ADB target differs from the frozen target")
        if state.get("status") == "COMPLETE":
            report_path = self.output_dir / "final/validation.json"
            return json.loads(report_path.read_text(encoding="utf-8"))
        overlay_baseline = plan.get("baseline", {}).get("musixmatchOverlay")
        if not isinstance(overlay_baseline, dict) or overlay_baseline.get("state") not in (
            "ABSENT",
            "COLLAPSED",
        ):
            raise AcceptanceError("approval plan has no safe frozen Musixmatch overlay baseline")
        self.overlay_baseline = overlay_baseline
        self.overlay_mutation_allowed = True
        self.ensure_device()
        self.reconcile_pending(plan, state, acknowledge_no_mutation)
        if (
            state.get("status") in {"DISPLAY_RUNNING", "WIDGET_PLAYBACK_RUNNING"}
            and state.get("pendingAction") is None
        ):
            self._restore_stopped_playback_and_volume(
                plan, self.output_dir / "resume-playback-recovery"
            )
            state["status"] = "RUNNING"
            self.save_state(state)
        if not state["progress"].get("displayFreshness"):
            self.revalidate_frozen_baseline(plan, allow_history_growth=False)
        self.run_display_freshness(plan, state)
        self.run_widget_single_tap(plan, state)
        self.run_widget_double_tap(plan, state)
        self.run_history_restore(plan, state)
        if (
            state["progress"].get("historyRestore") == "COMPLETE"
            and state["progress"].get("rebootIssued") is not True
        ):
            self._restore_stopped_playback_and_volume(
                plan, self.output_dir / "pre-reboot-playback-recovery"
            )
        if not self.run_reboot(plan, state):
            return {
                "status": "WAITING_FOR_USER_UNLOCK",
                "message": "Unlock the phone normally, return to its launcher, then rerun the exact execute command.",
            }
        return self.finalize(plan, state)


def dry_run_document() -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "runner": "V2 production widget connected acceptance",
        "prepare": {
            "readOnly": True,
            "requiresLauncherAlreadyVisible": True,
            "captures": [
                "exact V2 widget text, accessibility action, bounds, screenshot, and instance ID",
                "V1 and V2 APK hashes and protected private-state hashes",
                "V2 index/settings/history/request journal",
                "exact ordered Poweramp queue and replayable restore session",
                "stopped playback presentation and STREAM_MUSIC volume",
                "Musixmatch lyrics overlay absent/collapsed window geometry",
                "device fingerprint and boot identity",
            ],
            "output": "approval-plan.json plus its SHA-256",
        },
        "execute": {
            "requiredAcknowledgementName": MUTATION_ACK_NAME,
            "requiresExactPlanSha256": True,
            "resumable": True,
            "neverRepeatsUncertainTap": True,
            "phases": [
                "play and next-track display freshness",
                "collapse-only repair when Poweramp expands the frozen Musixmatch overlay",
                "ordinary process death and cold receiver freshness",
                "foreground/resume widget repair",
                "true cold single tap from a paused current track with immediate request-bound STARTING status and one durable widget session",
                "cold rapid double tap with one durable widget session",
                "exact tap-time file/title/path seed and ordered queue readback",
                "history persistence and ordinary History queue restoration",
                "reboot only after queue/playback/volume restoration",
                "user unlock handoff and post-reboot foreground repair",
            ],
        },
        "forbidden": [
            "debug ingress or instrumentation",
            "install, package force-stop, clear-data, uninstall, or permission mutation",
            "direct Poweramp provider mutation",
            "Musixmatch force-stop, permission change, or any action except its visible collapse control",
            "V1 app command",
            "manufactured destructive failure",
            "recommendation-quality claim from widget delivery evidence",
        ],
    }


def validate_completed_output(output_dir: Path) -> dict[str, Any]:
    try:
        plan = json.loads((output_dir / APPROVAL_FILENAME).read_text(encoding="utf-8"))
        state = json.loads((output_dir / STATE_FILENAME).read_text(encoding="utf-8"))
        report = json.loads((output_dir / "final/validation.json").read_text(encoding="utf-8"))
        manifest = json.loads((output_dir / "evidence-sha256.json").read_text(encoding="utf-8"))
        final_queue = json.loads((output_dir / "final/queue.json").read_text(encoding="utf-8"))
        final_history = json.loads((output_dir / "final/history.json").read_text(encoding="utf-8"))
        final_overlay = json.loads(
            (output_dir / "final/musixmatch-overlay.json").read_text(encoding="utf-8")
        )
        single_validation = json.loads(
            (output_dir / "widget-single-tap/widget-action-validation.json").read_text(
                encoding="utf-8"
            )
        )
        double_validation = json.loads(
            (output_dir / "widget-double-tap/widget-action-validation.json").read_text(
                encoding="utf-8"
            )
        )
        widget_xml = (output_dir / "final/launcher-widget.xml").read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"completed evidence is unreadable: {error}") from error
    if state.get("status") != "COMPLETE" or report.get("status") != "COMPLETE":
        raise AcceptanceError("evidence is not marked complete")
    if plan.get("schemaVersion") != SCHEMA_VERSION or state.get("schemaVersion") != SCHEMA_VERSION:
        raise AcceptanceError("completed evidence has an unsupported schema")
    expected_plan_digest = plan_digest(plan)
    if state.get("planSha256") != expected_plan_digest or report.get("planSha256") != expected_plan_digest:
        raise AcceptanceError("completed evidence is not bound to its approval plan")
    if report.get("scope", {}).get("radioQueueIntelligence") != (
        "Not judged by this harness; recommendation quality requires the separate listening evaluation"
    ):
        raise AcceptanceError("final report confuses widget delivery with recommendation intelligence")
    expected_ids = plan["baseline"]["orderedFileIds"]
    if [row.get("fileId") for row in final_queue] != expected_ids:
        raise AcceptanceError("final evidence queue differs from the frozen baseline")
    if not overlay_matches_baseline(plan["baseline"]["musixmatchOverlay"], final_overlay):
        raise AcceptanceError("final evidence Musixmatch overlay differs from the frozen baseline")
    baseline_ids = plan["baseline"]["historyRequestIds"]
    final_ids = [session.get("requestId") for session in final_history]
    if final_ids[: len(baseline_ids)] != baseline_ids or len(final_ids) != len(baseline_ids) + 3:
        raise AcceptanceError(
            "final history does not retain baseline plus single, double, and restore sessions"
        )
    appended = final_history[len(baseline_ids) :]
    expected_appended_ids = [
        report.get("widgetSingleRequestId"),
        report.get("widgetDoubleRequestId"),
        report.get("restoreRequestId"),
    ]
    if [session.get("requestId") for session in appended] != expected_appended_ids:
        raise AcceptanceError("final widget-session order differs from the validation report")
    if [session.get("delivery", {}).get("origin") for session in appended] != [
        "WIDGET_RADIO",
        "WIDGET_RADIO",
        "HISTORY_REQUEUE",
    ]:
        raise AcceptanceError("final widget-session origins are incorrect")
    for session in appended:
        terminal_delivery_identity_rows(session)
    if report.get("coldSingleTapImmediateRequestStatus") != "PASS" or report.get(
        "coldSingleTapOneDurableSession"
    ) != "PASS":
        raise AcceptanceError("final report omits the cold single-tap acceptance")
    if report.get("pausedTrackWidgetTap") != "PASS":
        raise AcceptanceError("final report omits the paused-track widget acceptance")
    if (
        single_validation.get("requestId") != expected_appended_ids[0]
        or single_validation.get("tapCount") != 1
        or single_validation.get("coldProcessAbsentImmediatelyBeforeTap") is not True
        or single_validation.get("durableSessionCount") != 1
        or single_validation.get("noDuplicateTerminalSession") is not True
        or single_validation.get("exactOrderedQueue") is not True
        or single_validation.get("pausedCurrentTrackAcrossTap") is not True
    ):
        raise AcceptanceError("cold single-tap evidence is incomplete")
    immediate = single_validation.get("immediateRequestStatus")
    if (
        not isinstance(immediate, Mapping)
        or immediate.get("state") != "STARTING"
        or immediate.get("message") != COLD_WIDGET_STATUS_MESSAGE
        or immediate.get("exactTapSeed") is not True
        or not isinstance(immediate.get("latencySeconds"), (int, float))
        or not math.isfinite(float(immediate["latencySeconds"]))
        or immediate["latencySeconds"] < 0.0
        or immediate["latencySeconds"] > MAX_IMMEDIATE_WIDGET_STATUS_LATENCY_SECONDS
    ):
        raise AcceptanceError("cold single-tap immediate status evidence is incomplete")
    if (
        double_validation.get("requestId") != expected_appended_ids[1]
        or double_validation.get("tapCount") != 2
        or double_validation.get("coldProcessAbsentImmediatelyBeforeTap") is not True
        or double_validation.get("durableSessionCount") != 1
        or double_validation.get("noDuplicateTerminalSession") is not True
        or double_validation.get("exactOrderedQueue") is not True
    ):
        raise AcceptanceError("rapid double-tap evidence is incomplete")
    view = parse_widget_view(widget_xml)
    validate_no_track_widget(view)
    if not isinstance(manifest, list) or not manifest:
        raise AcceptanceError("evidence hash manifest is empty")
    recorded: set[str] = set()
    for row in manifest:
        if not isinstance(row, Mapping):
            raise AcceptanceError("evidence manifest contains a non-object row")
        relative, digest = row.get("path"), row.get("sha256")
        if not isinstance(relative, str) or relative in recorded or relative.startswith("/"):
            raise AcceptanceError("evidence manifest contains an unsafe or duplicate path")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise AcceptanceError("evidence manifest contains an invalid digest")
        path = output_dir / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise AcceptanceError(f"evidence hash mismatch: {relative}")
        recorded.add(relative)
    expected_files = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "evidence-sha256.json"
    }
    if recorded != expected_files:
        raise AcceptanceError("evidence manifest does not cover every evidence file")
    return report


def resolve_serial(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.environ.get("ANDROID_SERIAL"):
        return os.environ["ANDROID_SERIAL"]
    result = subprocess.run(
        ["adb", "get-serialno"], check=True, capture_output=True, text=True, timeout=30
    )
    serial = result.stdout.strip()
    if not serial or serial == "unknown":
        raise AcceptanceError("set ANDROID_SERIAL to exactly one connected target")
    return serial


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed production-path acceptance for the V2 launcher widget."
    )
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--dry-run", action="store_true", help="print the static plan; no ADB")
    operation.add_argument("--prepare", action="store_true", help="capture a read-only device plan")
    operation.add_argument("--execute", action="store_true", help="execute an approved plan")
    operation.add_argument("--validate", action="store_true", help="validate completed host evidence")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--serial")
    parser.add_argument("--approve-plan-sha256")
    parser.add_argument(
        "--ack-no-mutation",
        choices=(
            "WIDGET_COLD_SINGLE_TAP",
            "WIDGET_RAPID_DOUBLE_TAP",
            "HISTORY_RESTORE_REPLACE",
        ),
        help="clear one uncertain receipt only when every protected mutation surface is unchanged",
    )
    arguments = parser.parse_args(argv)
    if arguments.dry_run:
        forbidden = [
            arguments.output_dir,
            arguments.serial,
            arguments.approve_plan_sha256,
            arguments.ack_no_mutation,
        ]
        if any(value is not None for value in forbidden):
            parser.error("--dry-run accepts no device or output arguments")
    else:
        if arguments.output_dir is None:
            parser.error("--output-dir is required")
        if arguments.prepare and (
            arguments.approve_plan_sha256 is not None or arguments.ack_no_mutation is not None
        ):
            parser.error("--prepare is read-only and accepts no execution acknowledgement")
        if arguments.execute:
            if not arguments.approve_plan_sha256 or not re.fullmatch(
                r"[0-9a-f]{64}", arguments.approve_plan_sha256
            ):
                parser.error("--execute requires --approve-plan-sha256 with exactly 64 hex digits")
        elif arguments.approve_plan_sha256 is not None or arguments.ack_no_mutation is not None:
            parser.error("execution acknowledgements are valid only with --execute")
        if arguments.validate and arguments.serial is not None:
            parser.error("--validate is offline and accepts no ADB serial")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.dry_run:
            print(json.dumps(dry_run_document(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        assert args.output_dir is not None
        output_dir = args.output_dir.expanduser().resolve()
        if args.validate:
            print(json.dumps(validate_completed_output(output_dir), ensure_ascii=False, indent=2))
            return 0
        serial = resolve_serial(args.serial)
        repo_root = Path(__file__).resolve().parents[2]
        runner = Runner(repo_root, output_dir, serial)
        if args.prepare:
            result = runner.prepare()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print(
                "\nREAD-ONLY PREPARATION COMPLETE. Review approval-plan.json. "
                "No playback, widget tap, queue mutation, reboot, install, or package stop occurred."
            )
            return 0
        result = runner.execute(args.approve_plan_sha256, args.ack_no_mutation)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 3 if result.get("status") == "WAITING_FOR_USER_UNLOCK" else 0
    except (AcceptanceError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"acceptance refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

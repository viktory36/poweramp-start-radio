#!/usr/bin/env python3
"""Drive Find Music delivery through the ordinary V2 UI and retain exact evidence.

The runner deliberately has no debug receiver, debug Activity, instrumentation hook, or
Poweramp mutation helper. Queue-changing commands are taps on the same production UI the listener
uses. An interrupted run leaves a pending-action receipt; resume reconciles durable history and
provider state before it can perform another tap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from find_music_delivery_acceptance import (
    AcceptanceError,
    EXPECTED_RESULT_COUNT,
    closed_action_plan,
    load_json,
    parse_queue_projection,
    parse_tap_track_ids,
    queue_from_json,
    queue_to_json,
    session_file_ids,
    session_occurrence_ids,
    session_track_ids,
    validate_completed_run,
    validate_direct_action,
    validate_replay_action,
    validate_semantic_restore,
    validate_terminal_session,
    write_json_atomic,
)


PACKAGE = "com.powerampstartradio.v2"
V1_PACKAGE = "com.powerampstartradio"
MAIN_ACTIVITY = f"{PACKAGE}/com.powerampstartradio.MainActivity"
QUEUE_URI = "content://com.maxmpz.audioplayer.data/queue"
EXECUTE_TOKEN = "MUTATE_POWERAMP_QUEUE"
STATE_SCHEMA_VERSION = 1
POLL_SECONDS = 1.0
ACTION_TIMEOUT_SECONDS = 180
UI_TIMEOUT_SECONDS = 120
SERVICE_STOP_TIMEOUT_SECONDS = 45
WIDGET_PROVIDER = f"{PACKAGE}/com.powerampstartradio.widget.StartRadioWidgetReceiver"


@dataclass(frozen=True)
class FindMusicCase:
    case_id: str
    kind: str
    ingredients: tuple[str, ...]
    planner: str | None
    recent_label: str


CASES = (
    # Closest is last so all visible Find Music defaults return to their current baseline values.
    FindMusicCase(
        case_id="sleep_varied",
        kind="text",
        ingredients=("sleep",),
        planner="Varied (DPP)",
        recent_label="sleep \u00b7 Varied (DPP)",
    ),
    FindMusicCase(
        case_id="ambient_sleep_all_of",
        kind="composed",
        ingredients=("ambient", "sleep"),
        planner="Varied (DPP)",
        recent_label="All of: ambient \u00b7 sleep",
    ),
    FindMusicCase(
        case_id="sleep_closest",
        kind="text",
        ingredients=("sleep",),
        planner="Closest",
        recent_label="sleep \u00b7 Closest",
    ),
)


PLAN = closed_action_plan()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Adb:
    def __init__(self, serial: str):
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

    def run_as_bytes(self, *args: str, timeout: float | None = 60) -> bytes:
        return self.bytes("exec-out", "run-as", PACKAGE, *args, timeout=timeout)

    def run_as_text(self, *args: str, timeout: float | None = 60) -> str:
        return self.text("exec-out", "run-as", PACKAGE, *args, timeout=timeout)


class OrdinaryUi:
    def __init__(self, adb: Adb, evidence_dir: Path):
        self.adb = adb
        self.evidence_dir = evidence_dir

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
        marker = raw.find(b"<?xml")
        if marker < 0:
            remote = "/sdcard/Download/.pasr-find-music-acceptance.xml"
            self.adb.shell("uiautomator", "dump", remote, timeout=30)
            try:
                raw = self.adb.bytes("exec-out", "cat", remote, timeout=30)
            finally:
                self.adb.shell("rm", "-f", remote, timeout=30)
            marker = raw.find(b"<?xml")
        if marker < 0:
            raise AcceptanceError("uiautomator did not return an XML hierarchy")
        end_marker = b"</hierarchy>"
        end = raw.find(end_marker, marker)
        if end < 0:
            raise AcceptanceError("uiautomator returned a truncated XML hierarchy")
        xml = raw[marker : end + len(end_marker)]
        try:
            return xml, ET.fromstring(xml)
        except ET.ParseError as error:
            raise AcceptanceError(f"uiautomator returned malformed XML: {error}") from error

    def capture(self, name: str) -> ET.Element:
        xml, root = self.dump()
        (self.evidence_dir / f"{name}.xml").write_bytes(xml + b"\n")
        (self.evidence_dir / f"{name}.png").write_bytes(
            self.adb.bytes("exec-out", "screencap", "-p", timeout=30)
        )
        return root

    @staticmethod
    def _parents(root: ET.Element) -> dict[ET.Element, ET.Element]:
        return {child: parent for parent in root.iter() for child in parent}

    @staticmethod
    def _bounds(node: ET.Element) -> tuple[int, int, int, int]:
        match = re.fullmatch(r"\[(\d+),(\d+)]\[(\d+),(\d+)]", node.attrib.get("bounds", ""))
        if match is None:
            raise AcceptanceError("selected UI node has no parseable bounds")
        left, top, right, bottom = map(int, match.groups())
        if right <= left or bottom <= top:
            raise AcceptanceError("selected UI node has empty bounds")
        return left, top, right, bottom

    def _clickable_for(self, root: ET.Element, node: ET.Element) -> ET.Element | None:
        parents = self._parents(root)
        current: ET.Element | None = node
        while current is not None:
            if current.attrib.get("clickable") == "true" and current.attrib.get("enabled") == "true":
                return current
            current = parents.get(current)
        return None

    def matching_clickables(
        self,
        root: ET.Element,
        *,
        text: str | None = None,
        content_desc: str | None = None,
    ) -> list[ET.Element]:
        matches: list[ET.Element] = []
        seen_bounds: set[str] = set()
        for node in root.iter("node"):
            if text is not None and node.attrib.get("text") != text:
                continue
            if content_desc is not None and node.attrib.get("content-desc") != content_desc:
                continue
            clickable = self._clickable_for(root, node)
            if clickable is None:
                continue
            bounds = clickable.attrib.get("bounds", "")
            if bounds not in seen_bounds:
                matches.append(clickable)
                seen_bounds.add(bounds)
        return matches

    def tap_node(self, node: ET.Element) -> None:
        left, top, right, bottom = self._bounds(node)
        self.adb.shell("input", "tap", str((left + right) // 2), str((top + bottom) // 2))
        time.sleep(0.5)

    def swipe_up(self) -> None:
        size = self.adb.shell("wm", "size")
        match = re.search(r"(\d+)x(\d+)", size)
        if match is None:
            raise AcceptanceError("cannot determine device display size")
        width, height = map(int, match.groups())
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
        scroll: bool = False,
        max_swipes: int = 14,
        occurrence: int = 0,
    ) -> None:
        for attempt in range(max_swipes + 1):
            _, root = self.dump()
            matches = self.matching_clickables(root, text=text, content_desc=content_desc)
            if len(matches) > occurrence:
                self.tap_node(matches[occurrence])
                return
            if not scroll or attempt == max_swipes:
                selector = f"text={text!r}" if text is not None else f"content-desc={content_desc!r}"
                raise AcceptanceError(f"ordinary UI has no unique enabled target for {selector}")
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
            if self.matching_clickables(root, text=text, content_desc=content_desc):
                return
            time.sleep(POLL_SECONDS)
        selector = f"text={text!r}" if text is not None else f"content-desc={content_desc!r}"
        raise AcceptanceError(f"timed out waiting for enabled ordinary-UI target {selector}")

    def enter_edit_text(self, ordinal: int, value: str) -> None:
        _, root = self.dump()
        fields = [
            node
            for node in root.iter("node")
            if node.attrib.get("class") == "android.widget.EditText"
            and node.attrib.get("enabled") == "true"
        ]
        fields.sort(key=lambda node: (self._bounds(node)[1], self._bounds(node)[0]))
        if ordinal >= len(fields):
            raise AcceptanceError(f"Find Music editor exposes only {len(fields)} text fields")
        if fields[ordinal].attrib.get("text"):
            raise AcceptanceError(f"Find Music text field {ordinal + 1} is not empty")
        self.tap_node(fields[ordinal])
        # The closed test matrix uses shell-safe ASCII one-word ingredients.
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise AcceptanceError(f"unsafe shell text in closed UI plan: {value!r}")
        self.adb.shell("input", "text", value)
        time.sleep(0.5)
        self.adb.shell("input", "keyevent", "4")
        time.sleep(0.5)


class Runner:
    def __init__(
        self,
        *,
        repo_root: Path,
        output_dir: Path,
        serial: str,
        ack_no_mutation: str | None,
    ):
        self.repo_root = repo_root
        self.output_dir = output_dir
        self.adb = Adb(serial)
        self.ack_no_mutation = ack_no_mutation
        self.state_path = output_dir / "state.json"

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updatedAt"] = now_iso()
        write_json_atomic(self.state_path, state)

    def _read_history(self) -> list[dict[str, Any]]:
        return self._parse_history(self._history_bytes())

    def _history_bytes(self) -> bytes:
        return self.adb.run_as_bytes("cat", "files/session_history.json", timeout=30)

    @staticmethod
    def _parse_history(raw: bytes) -> list[dict[str, Any]]:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise AcceptanceError(f"device session history is malformed: {error}") from error
        if not isinstance(value, list):
            raise AcceptanceError("device session history is not a list")
        result = [session for session in value if isinstance(session, dict)]
        if len(result) != len(value):
            raise AcceptanceError("device session history contains a non-object record")
        ids = [session.get("requestId") for session in result]
        if any(not isinstance(request_id, str) or not request_id for request_id in ids):
            raise AcceptanceError("device session history contains a non-durable session")
        if len(set(ids)) != len(ids):
            raise AcceptanceError("device session history contains duplicate durable request IDs")
        return result

    def _queue(self) -> tuple[str, list[Any]]:
        raw = self.adb.shell(
            "content",
            "query",
            "--uri",
            QUEUE_URI,
            "--projection",
            "queue._id:queue.folder_file_id:queue.sort",
            timeout=30,
        )
        return raw, parse_queue_projection(raw)

    def _request_journal_hashes(self) -> dict[str, str]:
        directory = "files/radio_requests_v2"
        exists = self.adb.run(
            "exec-out",
            "run-as",
            PACKAGE,
            "test",
            "-d",
            directory,
            check=False,
            timeout=30,
        )
        if exists.returncode != 0:
            return {}
        output = self.adb.run(
            "exec-out",
            "run-as",
            PACKAGE,
            "find",
            directory,
            "-type",
            "f",
            timeout=30,
        ).stdout
        result: dict[str, str] = {}
        for path in sorted(line.strip() for line in output.splitlines() if line.strip()):
            row = self.adb.run_as_text("sha256sum", path, timeout=30).strip()
            digest = row.split(maxsplit=1)[0] if row else ""
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise AcceptanceError(f"cannot hash durable request journal file {path}")
            result[path] = digest
        return result

    def _capture_private_file(self, relative: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.adb.run_as_bytes("cat", relative, timeout=60))

    def _snapshot(self, destination: Path) -> None:
        script = self.repo_root / "android-plugin/scripts/snapshot_device_acceptance.sh"
        env = os.environ.copy()
        env["ANDROID_SERIAL"] = self.adb.serial
        subprocess.run([str(script), str(destination)], check=True, env=env, timeout=600)

    def _service_state(self) -> tuple[str, dict[str, bool]]:
        raw = self.adb.shell("dumpsys", "activity", "services", PACKAGE, timeout=60)
        return raw, {
            "radioServiceActive": "com.powerampstartradio.services.RadioService" in raw,
            "indexingServiceActive": "com.powerampstartradio.indexing.IndexingService" in raw,
        }

    def _wait_for_services_stopped(self) -> tuple[str, dict[str, bool]]:
        deadline = time.monotonic() + SERVICE_STOP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            raw, state = self._service_state()
            if not any(state.values()):
                return raw, state
            time.sleep(POLL_SECONDS)
        raise AcceptanceError("V2 RadioService or IndexingService remained active after delivery")

    def _package_stopped(self) -> tuple[str, bool]:
        raw = self.adb.shell("dumpsys", "package", PACKAGE, timeout=60)
        user_zero = re.search(r"^\s*User 0:.*\bstopped=(true|false)\b", raw, re.MULTILINE)
        if user_zero is None:
            raise AcceptanceError("cannot prove the final V2 package stopped state for user 0")
        return raw, user_zero.group(1) == "true"

    def _widget_state(self) -> tuple[str, list[dict[str, Any]]]:
        raw = self.adb.shell("dumpsys", "appwidget", timeout=60)
        blocks = re.split(r"(?=^\s*\[\d+] id=\d+\s*$)", raw, flags=re.MULTILINE)
        widgets: list[dict[str, Any]] = []
        for block in blocks:
            identity = re.search(r"^\s*\[\d+] id=(\d+)\s*$", block, flags=re.MULTILINE)
            if identity is None or f"cmp:ComponentInfo{{{WIDGET_PROVIDER}}}" not in block:
                continue
            views = re.search(r"^\s*views=(.+)$", block, flags=re.MULTILINE)
            widgets.append(
                {
                    "appWidgetId": int(identity.group(1)),
                    "viewsAssigned": views is not None and views.group(1).strip() != "null",
                }
            )
        widgets.sort(key=lambda row: row["appWidgetId"])
        return raw, widgets

    def _preflight(self) -> dict[str, Any]:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise AcceptanceError(f"output directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.adb.text("get-state").strip() != "device":
            raise AcceptanceError(f"ADB target {self.adb.serial} is not in device state")
        if not self.adb.shell("pm", "path", PACKAGE).strip().startswith("package:"):
            raise AcceptanceError(f"{PACKAGE} is not installed")
        services = self.adb.shell("dumpsys", "activity", "services", PACKAGE)
        if "IndexingService" in services or "RadioService" in services:
            raise AcceptanceError("V2 indexing or radio service is active")

        local_apk = self.repo_root / "android-plugin/app/build/outputs/apk/debug/app-debug.apk"
        if not local_apk.is_file():
            raise AcceptanceError(f"build the reviewed V2 APK first: {local_apk}")
        package_path = self.adb.shell("pm", "path", PACKAGE).strip().removeprefix("package:")
        installed_hash_row = self.adb.text("exec-out", "sha256sum", package_path).strip()
        installed_hash = installed_hash_row.split(maxsplit=1)[0]
        local_hash = sha256_file(local_apk)
        if installed_hash != local_hash:
            raise AcceptanceError(
                f"installed V2 APK {installed_hash} differs from reviewed build {local_hash}"
            )

        baseline = self.output_dir / "baseline"
        baseline.mkdir(parents=True, exist_ok=True)
        self._snapshot(baseline / "snapshot")
        queue_raw, queue = self._queue()
        (baseline / "queue.txt").write_text(queue_raw, encoding="utf-8")
        write_json_atomic(baseline / "queue.json", queue_to_json(queue))
        history = self._read_history()
        write_json_atomic(baseline / "history.json", history)
        self._capture_private_file(
            "files/indexing_v2/generations/active-generation.json",
            baseline / "active-generation.json",
        )
        self._capture_private_file("shared_prefs/settings.xml", baseline / "settings.xml")
        widget_raw, widget_state = self._widget_state()
        (baseline / "appwidget.txt").write_text(widget_raw, encoding="utf-8")
        write_json_atomic(baseline / "widget-state.json", widget_state)
        if any(not row["viewsAssigned"] for row in widget_state):
            raise AcceptanceError("a baseline V2 widget instance has no assigned RemoteViews")

        baseline_file_ids = [entry.file_id for entry in queue]
        restore_sources: list[dict[str, Any]] = []
        for session in history:
            try:
                exact_files = session_file_ids(session) == baseline_file_ids
                session_track_ids(session)
                session_occurrence_ids(session)
            except AcceptanceError:
                exact_files = False
            if (
                exact_files
                and isinstance(session.get("generation"), Mapping)
                and session.get("outcome") == "SUCCEEDED"
                and session.get("isComplete") is True
            ):
                restore_sources.append(session)
        if not restore_sources:
            raise AcceptanceError(
                "current Poweramp queue has no exact modern saved session; semantic restore is unsafe"
            )
        restore_source = restore_sources[-1]
        restore_seed = restore_source.get("seedTrack")
        restore_source_title = (
            restore_seed.get("title") if isinstance(restore_seed, Mapping) else None
        )
        if not isinstance(restore_source_title, str) or not restore_source_title.strip():
            raise AcceptanceError("exact baseline restore session has no visible history title")
        state = {
            "schemaVersion": STATE_SCHEMA_VERSION,
            "runId": self.output_dir.name,
            "status": "RUNNING",
            "startedAt": now_iso(),
            "serial": self.adb.serial,
            "package": PACKAGE,
            "installedApkSha256": installed_hash,
            "expectedActionCount": len(PLAN),
            "plan": PLAN,
            "completedActions": {},
            "pendingAction": None,
            "baseline": {
                "historyRequestIds": [session["requestId"] for session in history],
                "requestJournalSha256": self._request_journal_hashes(),
                "queueSha256": hashlib.sha256(queue_raw.encode()).hexdigest(),
                "orderedFileIds": baseline_file_ids,
                "restoreSourceRequestId": restore_source["requestId"],
                "restoreSourceTitle": restore_source_title,
            },
        }
        self._save_state(state)
        return state

    def _launch_cold(self, evidence_dir: Path, capture_name: str) -> OrdinaryUi:
        self.adb.shell("am", "force-stop", PACKAGE)
        time.sleep(1)
        self.adb.shell("am", "start", "-W", "-n", MAIN_ACTIVITY, timeout=60)
        ui = OrdinaryUi(self.adb, evidence_dir)
        ui.wait_clickable(content_desc="Find music", timeout=UI_TIMEOUT_SECONDS)
        ui.capture(capture_name)
        return ui

    @staticmethod
    def _case(case_id: str) -> FindMusicCase:
        for case in CASES:
            if case.case_id == case_id:
                return case
        raise AcceptanceError(f"closed plan refers to unknown case {case_id}")

    def _prepare_fresh_result(self, ui: OrdinaryUi, case: FindMusicCase) -> None:
        ui.find_and_tap(content_desc="Find music")
        time.sleep(1)
        ui.enter_edit_text(0, case.ingredients[0])
        if case.kind == "composed":
            ui.find_and_tap(text="Add description", scroll=True)
            time.sleep(0.5)
            ui.enter_edit_text(1, case.ingredients[1])
            ui.find_and_tap(text="All of", scroll=True)
            if case.planner is not None:
                ui.find_and_tap(text=case.planner, scroll=True)
        else:
            if case.planner is None:
                raise AcceptanceError("simple text case has no planner")
            ui.find_and_tap(text=case.planner, scroll=True)
        ui.find_and_tap(text="Find matching queue", scroll=True)
        ui.wait_clickable(text=f"Queue {EXPECTED_RESULT_COUNT} displayed results", timeout=UI_TIMEOUT_SECONDS)

    def _prepare_recent_result(self, ui: OrdinaryUi, case: FindMusicCase) -> None:
        ui.find_and_tap(content_desc="Find music")
        time.sleep(1)
        ui.find_and_tap(text=case.recent_label, scroll=True, max_swipes=18)
        ui.wait_clickable(text=f"Queue {EXPECTED_RESULT_COUNT} displayed results", timeout=UI_TIMEOUT_SECONDS)

    def _prepare_latest_replay(self, ui: OrdinaryUi, action_dir: Path) -> None:
        ui.wait_clickable(content_desc="Requeue this session", timeout=UI_TIMEOUT_SECONDS)
        ui.capture("cold-session-ready")
        ui.find_and_tap(content_desc="History")
        time.sleep(0.5)
        ui.capture("cold-history-drawer")
        self.adb.shell("input", "keyevent", "4")
        time.sleep(0.5)
        ui.find_and_tap(content_desc="Requeue this session")

    def _prepare_restore(self, ui: OrdinaryUi, state: Mapping[str, Any], action_dir: Path) -> None:
        title = state["baseline"]["restoreSourceTitle"]
        if not isinstance(title, str) or not title:
            raise AcceptanceError("baseline restore source has no visible title")
        visible_title = title.removeprefix("Replay: ")
        ui.find_and_tap(content_desc="History")
        time.sleep(0.5)
        ui.find_and_tap(text=visible_title, scroll=True, max_swipes=30)
        ui.wait_clickable(content_desc="Requeue this session", timeout=UI_TIMEOUT_SECONDS)
        ui.capture("baseline-source-ready")
        ui.find_and_tap(content_desc="Requeue this session")

    def _history_by_id(self, request_id: str) -> dict[str, Any]:
        matches = [session for session in self._read_history() if session.get("requestId") == request_id]
        if len(matches) != 1:
            raise AcceptanceError(f"history has {len(matches)} records for request {request_id}")
        return matches[0]

    def _source_session(self, action: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
        if action["type"] == "restore":
            request_id = state["baseline"]["restoreSourceRequestId"]
        else:
            source_action_id = action.get("sourceActionId")
            request_id = state["completedActions"].get(source_action_id)
            if not isinstance(request_id, str):
                raise AcceptanceError(f"source action {source_action_id!r} has no completed request")
        return self._history_by_id(request_id)

    def _begin_pending(self, action: Mapping[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        history_bytes = self._history_bytes()
        history = self._parse_history(history_bytes)
        queue_raw, queue = self._queue()
        pending = {
            "actionId": action["id"],
            "beganAt": now_iso(),
            "startEpoch": time.time() - 0.5,
            "historyRequestIdsBefore": [session["requestId"] for session in history],
            "historySha256Before": hashlib.sha256(history_bytes).hexdigest(),
            "requestJournalSha256Before": self._request_journal_hashes(),
            "queueBefore": queue_to_json(queue),
            "queueBeforeSha256": hashlib.sha256(queue_raw.encode()).hexdigest(),
            "tapAttempted": False,
        }
        state["pendingAction"] = pending
        self._save_state(state)
        return pending

    def _mark_tap_attempted(self, state: dict[str, Any]) -> None:
        pending = state.get("pendingAction")
        if not isinstance(pending, dict):
            raise AcceptanceError("cannot mark a missing pending action")
        pending["tapAttempted"] = True
        pending["tapAttemptedAt"] = now_iso()
        self._save_state(state)

    def _wait_for_new_session(self, pending: Mapping[str, Any]) -> dict[str, Any]:
        before = set(pending["historyRequestIdsBefore"])
        deadline = time.monotonic() + ACTION_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            history = self._read_history()
            new = [session for session in history if session["requestId"] not in before]
            if len(new) > 1:
                raise AcceptanceError("more than one durable session appeared after one UI action")
            if len(new) == 1 and new[0].get("outcome") is not None:
                return new[0]
            time.sleep(POLL_SECONDS)
        raise AcceptanceError("timed out waiting for one terminal durable session")

    def _filtered_logcat(self, start_epoch: float) -> str:
        raw = self.adb.text(
            "logcat",
            "-d",
            "-v",
            "epoch",
            "MainViewModel:D",
            "RadioService:I",
            "*:S",
            timeout=60,
        )
        kept: list[str] = []
        for line in raw.splitlines():
            first = line.split(maxsplit=1)[0] if line.split() else ""
            try:
                timestamp = float(first)
            except ValueError:
                continue
            if timestamp >= start_epoch:
                kept.append(line)
        return "\n".join(kept) + ("\n" if kept else "")

    def _write_action_evidence(
        self,
        *,
        action: Mapping[str, Any],
        state: dict[str, Any],
        session: Mapping[str, Any],
        source_session: Mapping[str, Any] | None,
    ) -> None:
        action_dir = self.output_dir / "actions" / action["id"]
        action_dir.mkdir(parents=True, exist_ok=True)
        pending = state["pendingAction"]
        write_json_atomic(action_dir / "queue-before.json", pending["queueBefore"])
        queue_raw, after_queue = self._queue()
        (action_dir / "queue-after.txt").write_text(queue_raw, encoding="utf-8")
        write_json_atomic(action_dir / "queue-after.json", queue_to_json(after_queue))
        write_json_atomic(action_dir / "session.json", session)
        if source_session is not None:
            write_json_atomic(action_dir / "source-session.json", source_session)
        log_text = self._filtered_logcat(float(pending["startEpoch"]))
        (action_dir / "tap-log.txt").write_text(log_text, encoding="utf-8")
        write_json_atomic(action_dir / "history-after.json", self._read_history())

        manifest = dict(action)
        manifest["requestId"] = session["requestId"]
        if source_session is not None:
            manifest["sourceRequestId"] = source_session["requestId"]
        write_json_atomic(action_dir / "manifest.json", manifest)

        before_queue = queue_from_json(pending["queueBefore"])
        if action["type"] == "direct":
            tap_ids = parse_tap_track_ids(log_text)
            validate_direct_action(
                case_kind=action["caseKind"],
                placement=action["placement"],
                tap_track_ids=tap_ids,
                before_queue=before_queue,
                after_queue=after_queue,
                session=session,
            )
            same_as = action.get("sameResultAsActionId")
            if same_as is not None:
                prior_id = state["completedActions"].get(same_as)
                if not isinstance(prior_id, str):
                    raise AcceptanceError(f"repeat source action {same_as} is not complete")
                prior = self._history_by_id(prior_id)
                if session_track_ids(session) != session_track_ids(prior):
                    raise AcceptanceError("recent-search rerun changed the displayed result order")
        elif action["type"] == "replay":
            if source_session is None:
                raise AcceptanceError("replay action has no source session")
            validate_replay_action(
                placement=action["placement"],
                before_queue=before_queue,
                after_queue=after_queue,
                source_session=source_session,
                replay_session=session,
            )
        elif action["type"] == "restore":
            if source_session is None:
                raise AcceptanceError("restore action has no source session")
            validate_terminal_session(
                session,
                origin="HISTORY_REQUEUE",
                placement="REPLACE_UPCOMING",
                expected_count=len(state["baseline"]["orderedFileIds"]),
            )
            if session_file_ids(session) != state["baseline"]["orderedFileIds"]:
                raise AcceptanceError("restore session differs from the frozen baseline queue")
            validate_semantic_restore(
                queue_from_json(load_json(self.output_dir / "baseline" / "queue.json")),
                after_queue,
            )
        else:
            raise AcceptanceError(f"unknown action type {action['type']}")

    def _acknowledge_no_mutation(self, state: dict[str, Any]) -> None:
        pending = state.get("pendingAction")
        if not isinstance(pending, dict):
            raise AcceptanceError("there is no pending action to acknowledge")
        if self.ack_no_mutation != pending["actionId"]:
            raise AcceptanceError(
                "pending action has no terminal receipt; inspect it and resume with "
                f"--ack-no-mutation {pending['actionId']} only if no mutation occurred"
            )
        history_bytes = self._history_bytes()
        current_ids = [session["requestId"] for session in self._parse_history(history_bytes)]
        if current_ids != pending["historyRequestIdsBefore"]:
            raise AcceptanceError("cannot acknowledge: durable history changed")
        if hashlib.sha256(history_bytes).hexdigest() != pending["historySha256Before"]:
            raise AcceptanceError("cannot acknowledge: durable history bytes changed")
        if self._request_journal_hashes() != pending["requestJournalSha256Before"]:
            raise AcceptanceError("cannot acknowledge: durable request journal changed")
        queue_raw, current_queue = self._queue()
        if queue_to_json(current_queue) != pending["queueBefore"]:
            raise AcceptanceError("cannot acknowledge: Poweramp queue changed")
        if hashlib.sha256(queue_raw.encode()).hexdigest() != pending["queueBeforeSha256"]:
            raise AcceptanceError("cannot acknowledge: Poweramp queue bytes changed")
        state["pendingAction"] = None
        self._save_state(state)

    def _reconcile_pending(self, state: dict[str, Any]) -> None:
        pending = state.get("pendingAction")
        if not isinstance(pending, dict):
            return
        action = next((item for item in PLAN if item["id"] == pending["actionId"]), None)
        if action is None:
            raise AcceptanceError("pending receipt refers to an unknown closed-plan action")
        before = set(pending["historyRequestIdsBefore"])
        new = [session for session in self._read_history() if session["requestId"] not in before]
        if len(new) == 1 and new[0].get("outcome") is not None:
            source = self._source_session(action, state) if action["type"] != "direct" else None
            self._write_action_evidence(
                action=action,
                state=state,
                session=new[0],
                source_session=source,
            )
            state["completedActions"][action["id"]] = new[0]["requestId"]
            state["pendingAction"] = None
            self._save_state(state)
            return
        if len(new) > 1:
            raise AcceptanceError("pending action produced more than one durable session")
        self._acknowledge_no_mutation(state)

    def _execute_action(self, action: Mapping[str, Any], state: dict[str, Any]) -> None:
        action_dir = self.output_dir / "actions" / action["id"]
        action_dir.mkdir(parents=True, exist_ok=True)
        ui = self._launch_cold(action_dir, "cold-home")
        case = self._case(action["caseId"]) if action["caseId"] != "baseline" else None
        source: dict[str, Any] | None = None

        if action["type"] == "direct":
            if action["route"] == "fresh_editor":
                self._prepare_fresh_result(ui, case)  # type: ignore[arg-type]
            elif action["route"] == "recent_search":
                self._prepare_recent_result(ui, case)  # type: ignore[arg-type]
            else:
                raise AcceptanceError(f"unknown direct UI route {action['route']}")
            ui.capture("displayed-result")
        elif action["type"] == "replay":
            source = self._source_session(action, state)
            self._prepare_latest_replay(ui, action_dir)
        elif action["type"] == "restore":
            source = self._source_session(action, state)
            self._prepare_restore(ui, state, action_dir)
        else:
            raise AcceptanceError(f"unknown action type {action['type']}")

        self._begin_pending(action, state)
        if action["type"] == "direct":
            ui.find_and_tap(text=f"Queue {EXPECTED_RESULT_COUNT} displayed results", scroll=True)
        ui.wait_clickable(text="Replace upcoming")
        self._mark_tap_attempted(state)
        placement_label = (
            "Append after upcoming" if action["placement"] == "APPEND" else "Replace upcoming"
        )
        ui.find_and_tap(text=placement_label)
        session = self._wait_for_new_session(state["pendingAction"])
        self._write_action_evidence(
            action=action,
            state=state,
            session=session,
            source_session=source,
        )
        state["completedActions"][action["id"]] = session["requestId"]
        state["pendingAction"] = None
        self._save_state(state)

    def _finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        final = self.output_dir / "final"
        final.mkdir(parents=True, exist_ok=True)
        self._wait_for_services_stopped()
        # Every action starts from a cold production Activity. Repeat that lifecycle here so a
        # prior force-stop cannot leave Android's widget host showing a disabled placeholder.
        self.adb.shell("am", "force-stop", PACKAGE)
        time.sleep(1)
        self.adb.shell("am", "start", "-W", "-n", MAIN_ACTIVITY, timeout=60)
        final_ui = OrdinaryUi(self.adb, final)
        final_ui.wait_clickable(content_desc="Find music", timeout=UI_TIMEOUT_SECONDS)
        final_ui.capture("app-refreshed")
        time.sleep(2)
        self.adb.shell("input", "keyevent", "3")
        time.sleep(1)
        final_ui.capture("home-after-refresh")
        services_raw, services = self._wait_for_services_stopped()
        package_raw, package_stopped = self._package_stopped()
        widget_raw, widget_state = self._widget_state()
        if package_stopped:
            raise AcceptanceError("V2 package remained stopped after final cold launch")
        if any(not row["viewsAssigned"] for row in widget_state):
            raise AcceptanceError("a final V2 widget instance has no assigned RemoteViews")
        baseline_widget_state = load_json(self.output_dir / "baseline" / "widget-state.json")
        if [row["appWidgetId"] for row in widget_state] != [
            row["appWidgetId"] for row in baseline_widget_state
        ]:
            raise AcceptanceError("V2 widget instances changed during Find Music acceptance")
        (final / "services.txt").write_text(services_raw, encoding="utf-8")
        (final / "package.txt").write_text(package_raw, encoding="utf-8")
        (final / "appwidget.txt").write_text(widget_raw, encoding="utf-8")
        write_json_atomic(final / "widget-state.json", widget_state)
        write_json_atomic(
            final / "runtime-state.json",
            {"packageStopped": package_stopped, **services},
        )
        queue_raw, queue = self._queue()
        (final / "queue.txt").write_text(queue_raw, encoding="utf-8")
        write_json_atomic(final / "queue.json", queue_to_json(queue))
        write_json_atomic(final / "history.json", self._read_history())
        self._capture_private_file(
            "files/indexing_v2/generations/active-generation.json",
            final / "active-generation.json",
        )
        self._capture_private_file("shared_prefs/settings.xml", final / "settings.xml")
        self._snapshot(final / "snapshot")
        if (final / "active-generation.json").read_bytes() != (
            self.output_dir / "baseline" / "active-generation.json"
        ).read_bytes():
            raise AcceptanceError("active V2 generation pointer changed during Find Music acceptance")
        state["status"] = "COMPLETE"
        state["completedAt"] = now_iso()
        self._save_state(state)
        try:
            report = validate_completed_run(self.output_dir)
        except Exception:
            state["status"] = "FAILED"
            self._save_state(state)
            raise
        write_json_atomic(self.output_dir / "validation.json", report)
        evidence: list[dict[str, str]] = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "sha256.json":
                evidence.append(
                    {
                        "path": str(path.relative_to(self.output_dir)),
                        "sha256": sha256_file(path),
                    }
                )
        write_json_atomic(self.output_dir / "sha256.json", evidence)
        return report

    def run(self) -> dict[str, Any]:
        state = load_json(self.state_path) if self.state_path.is_file() else self._preflight()
        if state.get("serial") != self.adb.serial or state.get("package") != PACKAGE:
            raise AcceptanceError("resume target differs from the frozen run target")
        if state.get("status") == "COMPLETE":
            return validate_completed_run(self.output_dir)
        if state.get("status") == "FAILED":
            raise AcceptanceError("run is marked FAILED; preserve evidence and start a reviewed run")
        self._reconcile_pending(state)

        for action in PLAN:
            if action["id"] in state["completedActions"]:
                continue
            self._execute_action(action, state)
        return self._finalize(state)


def dry_run_document() -> dict[str, Any]:
    return {
        "schemaVersion": STATE_SCHEMA_VERSION,
        "ordinaryProductionUiOnly": True,
        "debugIngressAllowed": False,
        "expectedResultCount": EXPECTED_RESULT_COUNT,
        "cases": [
            {
                "caseId": case.case_id,
                "kind": case.kind,
                "ingredients": list(case.ingredients),
                "planner": case.planner,
                "recentLabel": case.recent_label,
            }
            for case in CASES
        ],
        "actions": PLAN,
        "mutationCount": len(PLAN),
        "restoreContract": {
            "exactOrderedPowerampFileIds": True,
            "providerOccurrenceIdsMayChange": True,
            "requiresExactReplayableBaselineSession": True,
        },
        "resumeContract": {
            "pendingReceiptBeforeEveryMutation": True,
            "exactHistoryBytes": True,
            "completeRequestJournalHashes": True,
            "exactParsedAndRawQueue": True,
            "neverBlindlyRepeatPendingTap": True,
            "manualNoMutationAcknowledgementRequired": True,
        },
        "finalStateContract": {
            "packageStopped": False,
            "radioServiceActive": False,
            "indexingServiceActive": False,
            "sameWidgetInstanceIds": True,
            "widgetRemoteViewsAssigned": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="print the closed plan; never call ADB")
    group.add_argument("--execute", metavar="TOKEN", help=f"must be exactly {EXECUTE_TOKEN}")
    parser.add_argument("--output-dir", type=Path, help="new host evidence directory")
    parser.add_argument("--resume", type=Path, help="resume an existing evidence directory")
    parser.add_argument("--serial", help="ADB serial; defaults to ANDROID_SERIAL or adb get-serialno")
    parser.add_argument(
        "--ack-no-mutation",
        metavar="ACTION_ID",
        help="after inspection, explicitly certify that an unresolved pending action did nothing",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        if args.output_dir or args.resume or args.ack_no_mutation:
            parser.error("--dry-run cannot be combined with execution or resume paths")
        print(json.dumps(dry_run_document(), indent=2, sort_keys=True))
        return 0
    if args.execute != EXECUTE_TOKEN:
        parser.error(f"queue-changing execution requires --execute {EXECUTE_TOKEN}")
    if bool(args.output_dir) == bool(args.resume):
        parser.error("execution requires exactly one of --output-dir or --resume")

    for command in ("adb", "sha256sum"):
        if shutil.which(command) is None:
            parser.error(f"missing required host command: {command}")
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (args.resume or args.output_dir).resolve()
    serial = args.serial or os.environ.get("ANDROID_SERIAL")
    if not serial:
        serial = subprocess.run(
            ["adb", "get-serialno"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    if not serial or serial == "unknown":
        parser.error("set --serial or ANDROID_SERIAL to one connected device")

    try:
        report = Runner(
            repo_root=repo_root,
            output_dir=output_dir,
            serial=serial,
            ack_no_mutation=args.ack_no_mutation,
        ).run()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (AcceptanceError, OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

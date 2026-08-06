#!/usr/bin/env python3
"""Evidence parsing and fail-closed validation for Find Music UI delivery acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
EXPECTED_RESULT_COUNT = 30
DIRECT_ORIGIN_BY_CASE_KIND = {
    "text": "TEXT_RESULT_LIST",
    "composed": "COMPOSED_RESULT_LIST",
}


def closed_action_plan() -> list[dict[str, Any]]:
    """Return the exact reviewed ordinary-UI mutation matrix."""
    cases = (
        ("sleep_varied", "text"),
        ("ambient_sleep_all_of", "composed"),
        ("sleep_closest", "text"),
    )
    result: list[dict[str, Any]] = []
    sequence = 1
    for case_id, case_kind in cases:
        append_id = f"{sequence:02d}-{case_id}-direct-append"
        result.append(
            {
                "id": append_id,
                "type": "direct",
                "caseId": case_id,
                "caseKind": case_kind,
                "placement": "APPEND",
                "route": "fresh_editor",
            }
        )
        sequence += 1
        result.append(
            {
                "id": f"{sequence:02d}-{case_id}-replay-replace",
                "type": "replay",
                "caseId": case_id,
                "caseKind": case_kind,
                "placement": "REPLACE_UPCOMING",
                "route": "cold_latest_session",
                "sourceActionId": append_id,
            }
        )
        sequence += 1
        replace_id = f"{sequence:02d}-{case_id}-direct-replace"
        result.append(
            {
                "id": replace_id,
                "type": "direct",
                "caseId": case_id,
                "caseKind": case_kind,
                "placement": "REPLACE_UPCOMING",
                "route": "recent_search",
                "sameResultAsActionId": append_id,
            }
        )
        sequence += 1
        result.append(
            {
                "id": f"{sequence:02d}-{case_id}-replay-append",
                "type": "replay",
                "caseId": case_id,
                "caseKind": case_kind,
                "placement": "APPEND",
                "route": "cold_latest_session",
                "sourceActionId": replace_id,
            }
        )
        sequence += 1
    result.append(
        {
            "id": f"{sequence:02d}-restore-baseline-replace",
            "type": "restore",
            "caseId": "baseline",
            "caseKind": "baseline",
            "placement": "REPLACE_UPCOMING",
            "route": "history_seed_title",
        }
    )
    return result


class AcceptanceError(RuntimeError):
    """Raised when evidence cannot prove the requested acceptance contract."""


@dataclass(frozen=True)
class QueueEntry:
    occurrence_id: int
    file_id: int
    sort: int


_QUEUE_ROW = re.compile(r"^Row:\s+\d+\s+(?P<body>.+)$")
_QUEUE_FIELD = re.compile(r"(?:^|,\s*)(?P<name>[A-Za-z0-9_.]+)=(?P<value>-?\d+)")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"cannot read valid JSON from {path}: {error}") from error


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with temporary.open("w", encoding="utf-8") as destination:
        destination.write(rendered)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def parse_queue_projection(text: str) -> list[QueueEntry]:
    entries: list[QueueEntry] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().replace("\r", "")
        if not line:
            continue
        row = _QUEUE_ROW.match(line)
        if row is None:
            raise AcceptanceError(f"unrecognized Poweramp queue row: {raw_line!r}")
        fields = {
            match.group("name").split(".")[-1]: int(match.group("value"))
            for match in _QUEUE_FIELD.finditer(row.group("body"))
        }
        try:
            occurrence_id = fields["_id"]
            file_id = fields["folder_file_id"]
            sort = fields["sort"]
        except KeyError as error:
            raise AcceptanceError(f"Poweramp queue row is missing {error.args[0]}: {raw_line!r}") from error
        if occurrence_id <= 0 or file_id <= 0:
            raise AcceptanceError(f"Poweramp queue exposed a non-positive ID: {raw_line!r}")
        entries.append(QueueEntry(occurrence_id, file_id, sort))

    entries.sort(key=lambda entry: (entry.sort, entry.occurrence_id))
    occurrence_ids = [entry.occurrence_id for entry in entries]
    if len(set(occurrence_ids)) != len(occurrence_ids):
        raise AcceptanceError("Poweramp queue projection contains duplicate occurrence IDs")
    return entries


def queue_to_json(entries: Sequence[QueueEntry]) -> list[dict[str, int]]:
    return [
        {
            "occurrenceId": entry.occurrence_id,
            "fileId": entry.file_id,
            "sort": entry.sort,
        }
        for entry in entries
    ]


def queue_from_json(value: Any) -> list[QueueEntry]:
    if not isinstance(value, list):
        raise AcceptanceError("queue JSON is not a list")
    result: list[QueueEntry] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise AcceptanceError(f"queue JSON row {index} is not an object")
        try:
            entry = QueueEntry(
                occurrence_id=int(row["occurrenceId"]),
                file_id=int(row["fileId"]),
                sort=int(row["sort"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AcceptanceError(f"queue JSON row {index} is malformed") from error
        if entry.occurrence_id <= 0 or entry.file_id <= 0:
            raise AcceptanceError(f"queue JSON row {index} has a non-positive ID")
        result.append(entry)
    if result != sorted(result, key=lambda entry: (entry.sort, entry.occurrence_id)):
        raise AcceptanceError("queue JSON is not in provider order")
    if len({entry.occurrence_id for entry in result}) != len(result):
        raise AcceptanceError("queue JSON contains duplicate occurrence IDs")
    return result


def queued_rows(session: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tracks = session.get("tracks")
    if not isinstance(tracks, list):
        raise AcceptanceError("session has no track list")
    rows = [row for row in tracks if isinstance(row, Mapping) and row.get("status") == "QUEUED"]
    if not rows:
        raise AcceptanceError("session has no provider-confirmed queued rows")
    return rows


def session_track_ids(session: Mapping[str, Any]) -> list[int]:
    result: list[int] = []
    for index, row in enumerate(queued_rows(session)):
        try:
            result.append(int(row["track"]["id"]))
        except (KeyError, TypeError, ValueError) as error:
            raise AcceptanceError(f"queued session row {index} has no embedded track ID") from error
    return result


def session_file_ids(session: Mapping[str, Any]) -> list[int]:
    result: list[int] = []
    for index, row in enumerate(queued_rows(session)):
        try:
            file_id = int(row["resolvedPowerampFileId"])
        except (KeyError, TypeError, ValueError) as error:
            raise AcceptanceError(f"queued session row {index} has no Poweramp file ID") from error
        if file_id <= 0:
            raise AcceptanceError(f"queued session row {index} has a non-positive Poweramp file ID")
        result.append(file_id)
    return result


def session_occurrence_ids(session: Mapping[str, Any]) -> list[int]:
    result: list[int] = []
    for index, row in enumerate(queued_rows(session)):
        try:
            occurrence_id = int(row["resolvedPowerampQueueId"])
        except (KeyError, TypeError, ValueError) as error:
            raise AcceptanceError(f"queued session row {index} has no Poweramp occurrence ID") from error
        if occurrence_id <= 0:
            raise AcceptanceError(f"queued session row {index} has a non-positive occurrence ID")
        result.append(occurrence_id)
    if len(set(result)) != len(result):
        raise AcceptanceError("session reuses one Poweramp occurrence for multiple requested rows")
    return result


def parse_tap_track_ids(log_text: str) -> list[int]:
    matches: dict[int, int] = {}
    pattern = re.compile(r"QUEUE_DISPLAYED:\s*\[(\d+)]\s+.*\(trackId=(\d+)\)")
    for line in log_text.splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        index = int(match.group(1))
        track_id = int(match.group(2))
        if index in matches and matches[index] != track_id:
            raise AcceptanceError(f"tap-time log reports two tracks for displayed position {index + 1}")
        matches[index] = track_id
    if not matches:
        raise AcceptanceError("tap-time log contains no QUEUE_DISPLAYED row evidence")
    expected_indices = list(range(len(matches)))
    if sorted(matches) != expected_indices:
        raise AcceptanceError("tap-time displayed positions are incomplete or non-contiguous")
    return [matches[index] for index in expected_indices]


def _delivery(session: Mapping[str, Any]) -> Mapping[str, Any]:
    delivery = session.get("delivery")
    if not isinstance(delivery, Mapping):
        raise AcceptanceError("session has no structured queue-delivery receipt")
    return delivery


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AcceptanceError(f"{label}: expected {expected!r}, observed {actual!r}")


def validate_terminal_session(
    session: Mapping[str, Any],
    *,
    origin: str,
    placement: str,
    expected_count: int = EXPECTED_RESULT_COUNT,
) -> None:
    delivery = _delivery(session)
    _require_equal(session.get("isComplete"), True, "session completion")
    _require_equal(session.get("isDirectQueue"), True, "direct-queue marker")
    _require_equal(session.get("outcome"), "SUCCEEDED", "session outcome")
    _require_equal(session.get("directQueuePlacement"), placement, "queue placement")
    _require_equal(delivery.get("origin"), origin, "queue origin")
    for field in ("requestedCount", "rankedCount", "resolvedCount", "verifiedCount"):
        _require_equal(delivery.get(field), expected_count, f"delivery {field}")
    _require_equal(delivery.get("verificationComplete"), True, "final queue verification")
    _require_equal(delivery.get("mutationCount"), 1, "queue mutation count")
    _require_equal(delivery.get("unexpectedObservedCount", 0), 0, "unexpected queue entries")
    _require_equal(delivery.get("notInLibraryCount"), 0, "unresolved library rows")
    _require_equal(delivery.get("queueFailedCount"), 0, "unconfirmed queue rows")
    tracks = session.get("tracks")
    if not isinstance(tracks, list) or any(not isinstance(row, Mapping) for row in tracks):
        raise AcceptanceError("session track list is missing or malformed")
    _require_equal(len(tracks), expected_count, "complete session row count")
    _require_equal(len(queued_rows(session)), expected_count, "provider-confirmed session rows")


def validate_find_music_evidence(session: Mapping[str, Any], case_kind: str) -> None:
    evidence = session.get("findMusicSessionEvidence")
    if not isinstance(evidence, Mapping):
        raise AcceptanceError("Find Music session has no immutable query evidence")
    query_spec = evidence.get("querySpec")
    if not isinstance(query_spec, Mapping):
        raise AcceptanceError("Find Music session has no query specification")
    text_ingredients = query_spec.get("textIngredients")
    if not isinstance(text_ingredients, list) or not text_ingredients:
        raise AcceptanceError("Find Music query evidence has no text ingredients")
    if case_kind == "text":
        if len(text_ingredients) != 1 or query_spec.get("songSeeds") not in ([], None):
            raise AcceptanceError("simple text case was persisted as a composed query")
    elif case_kind == "composed":
        active = [row for row in text_ingredients if isinstance(row, Mapping) and row.get("weight", 0) > 0]
        if len(active) < 2:
            raise AcceptanceError("composed case did not persist at least two active ingredients")
        _require_equal(query_spec.get("operator"), "ALL_OF", "composed operator")
    else:
        raise AcceptanceError(f"unknown Find Music case kind: {case_kind}")

    rows = queued_rows(session)
    displayed_ranks: list[int] = []
    for index, row in enumerate(rows):
        ranking = row.get("findMusicEvidence")
        if not isinstance(ranking, Mapping):
            raise AcceptanceError(f"Find Music row {index + 1} has no ranking evidence")
        try:
            displayed_ranks.append(int(ranking["displayedRank"]))
            objective_rank = int(ranking["objectiveRank"])
        except (KeyError, TypeError, ValueError) as error:
            raise AcceptanceError(f"Find Music row {index + 1} has malformed rank evidence") from error
        if objective_rank <= 0:
            raise AcceptanceError(f"Find Music row {index + 1} has a non-positive objective rank")
    _require_equal(displayed_ranks, list(range(1, len(rows) + 1)), "displayed rank sequence")


def _expected_occurrences_in_queue(
    session: Mapping[str, Any],
    after_queue: Sequence[QueueEntry],
) -> list[QueueEntry]:
    by_id = {entry.occurrence_id: entry for entry in after_queue}
    result: list[QueueEntry] = []
    for index, (occurrence_id, file_id) in enumerate(
        zip(session_occurrence_ids(session), session_file_ids(session), strict=True)
    ):
        entry = by_id.get(occurrence_id)
        if entry is None:
            raise AcceptanceError(f"session occurrence {occurrence_id} is absent from Poweramp")
        if entry.file_id != file_id:
            raise AcceptanceError(
                f"session row {index + 1} expected file {file_id} in occurrence "
                f"{occurrence_id}, observed {entry.file_id}"
            )
        result.append(entry)
    if result != sorted(result, key=lambda entry: (entry.sort, entry.occurrence_id)):
        raise AcceptanceError("session occurrences do not appear in displayed order in Poweramp")
    return result


def validate_queue_effect(
    before_queue: Sequence[QueueEntry],
    after_queue: Sequence[QueueEntry],
    session: Mapping[str, Any],
    placement: str,
) -> None:
    inserted = _expected_occurrences_in_queue(session, after_queue)
    if placement == "APPEND":
        _require_equal(
            list(after_queue[: len(before_queue)]),
            list(before_queue),
            "append preservation of the complete prior queue",
        )
        _require_equal(
            list(after_queue[len(before_queue) :]),
            inserted,
            "append suffix occurrence sequence",
        )
        return

    if placement != "REPLACE_UPCOMING":
        raise AcceptanceError(f"unsupported queue placement: {placement}")
    anchor_id = session.get("queueAnchorOccurrenceId")
    expected = list(inserted)
    if anchor_id is not None:
        try:
            normalized_anchor_id = int(anchor_id)
        except (TypeError, ValueError) as error:
            raise AcceptanceError("replacement anchor occurrence ID is malformed") from error
        anchor = next(
            (entry for entry in before_queue if entry.occurrence_id == normalized_anchor_id),
            None,
        )
        if anchor is None:
            raise AcceptanceError("replacement claimed an anchor absent from its before snapshot")
        if not after_queue or after_queue[0] != anchor:
            raise AcceptanceError("replacement did not preserve its exact live queue anchor")
        expected.insert(0, anchor)
    _require_equal(list(after_queue), expected, "replacement queue occurrence sequence")


def validate_direct_action(
    *,
    case_kind: str,
    placement: str,
    tap_track_ids: Sequence[int],
    before_queue: Sequence[QueueEntry],
    after_queue: Sequence[QueueEntry],
    session: Mapping[str, Any],
    expected_count: int = EXPECTED_RESULT_COUNT,
) -> None:
    origin = DIRECT_ORIGIN_BY_CASE_KIND.get(case_kind)
    if origin is None:
        raise AcceptanceError(f"unknown direct case kind: {case_kind}")
    validate_terminal_session(
        session,
        origin=origin,
        placement=placement,
        expected_count=expected_count,
    )
    validate_find_music_evidence(session, case_kind)
    _require_equal(list(tap_track_ids), session_track_ids(session), "tap-time displayed track order")
    validate_queue_effect(before_queue, after_queue, session, placement)


def validate_replay_action(
    *,
    placement: str,
    before_queue: Sequence[QueueEntry],
    after_queue: Sequence[QueueEntry],
    source_session: Mapping[str, Any],
    replay_session: Mapping[str, Any],
    expected_count: int = EXPECTED_RESULT_COUNT,
) -> None:
    validate_terminal_session(
        replay_session,
        origin="HISTORY_REQUEUE",
        placement=placement,
        expected_count=expected_count,
    )
    _require_equal(
        session_track_ids(replay_session),
        session_track_ids(source_session),
        "history replay embedded-track order",
    )
    _require_equal(
        session_file_ids(replay_session),
        session_file_ids(source_session),
        "history replay Poweramp-file order",
    )
    _require_equal(
        replay_session.get("findMusicSessionEvidence"),
        source_session.get("findMusicSessionEvidence"),
        "history replay query evidence",
    )
    source_row_evidence = [row.get("findMusicEvidence") for row in queued_rows(source_session)]
    replay_row_evidence = [row.get("findMusicEvidence") for row in queued_rows(replay_session)]
    _require_equal(replay_row_evidence, source_row_evidence, "history replay row-ranking evidence")
    validate_queue_effect(before_queue, after_queue, replay_session, placement)


def validate_semantic_restore(
    baseline_queue: Sequence[QueueEntry],
    restored_queue: Sequence[QueueEntry],
) -> None:
    """Occurrence IDs may change; listener-visible ordered queue content may not."""
    _require_equal(
        [entry.file_id for entry in restored_queue],
        [entry.file_id for entry in baseline_queue],
        "restored ordered Poweramp file IDs",
    )


def load_private_hashes(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AcceptanceError(f"cannot read private hash manifest {path}: {error}") from error
    for line in lines:
        if not line.strip():
            continue
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2 or not re.fullmatch(r"[0-9a-f]{64}", pieces[0]):
            raise AcceptanceError(f"malformed private hash manifest row in {path}: {line!r}")
        if pieces[1] in result:
            raise AcceptanceError(f"duplicate private hash manifest path in {path}: {pieces[1]}")
        result[pieces[1]] = pieces[0]
    return result


def _stable_device_facts(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("captured_at="):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise AcceptanceError(f"malformed device fact in {path}: {line!r}")
        result[key] = value
    return result


def validate_protected_snapshots(before: Path, after: Path) -> dict[str, Any]:
    _require_equal(
        _stable_device_facts(after / "device.txt"),
        _stable_device_facts(before / "device.txt"),
        "device identity and OS facts",
    )
    for label in ("v1", "v2"):
        before_status = (before / label / "status.txt").read_text(encoding="utf-8").strip()
        after_status = (after / label / "status.txt").read_text(encoding="utf-8").strip()
        _require_equal(after_status, before_status, f"{label} package status")
        _require_equal(
            (after / label / "apk-sha256.txt").read_text(encoding="utf-8"),
            (before / label / "apk-sha256.txt").read_text(encoding="utf-8"),
            f"{label} installed APK hashes",
        )
        _require_equal(
            (after / label / "runtime-grants.txt").read_text(encoding="utf-8"),
            (before / label / "runtime-grants.txt").read_text(encoding="utf-8"),
            f"{label} runtime grants",
        )
        _require_equal(
            (after / label / "package-path.txt").read_text(encoding="utf-8"),
            (before / label / "package-path.txt").read_text(encoding="utf-8"),
            f"{label} package paths",
        )

    before_v1 = load_private_hashes(before / "v1" / "private-file-sha256.txt")
    after_v1 = load_private_hashes(after / "v1" / "private-file-sha256.txt")
    _require_equal(after_v1, before_v1, "complete V1 private state")

    before_v2 = load_private_hashes(before / "v2" / "private-file-sha256.txt")
    after_v2 = load_private_hashes(after / "v2" / "private-file-sha256.txt")
    allowed_exact = {
        "./files/session_history.json",
        "./shared_prefs/settings.xml",
        "./shared_prefs/poweramp_state.xml",
        "./shared_prefs/poweramp_authenticated_state_v2.xml",
    }
    allowed_prefixes = ("./files/radio_requests_v2/states/",)
    paths = set(before_v2) | set(after_v2)
    forbidden_changes: list[str] = []
    expected_changes: list[str] = []
    for path in sorted(paths):
        if before_v2.get(path) == after_v2.get(path):
            continue
        if path in allowed_exact or path.startswith(allowed_prefixes):
            expected_changes.append(path)
        else:
            forbidden_changes.append(path)
    if forbidden_changes:
        raise AcceptanceError(
            "unexpected V2 private-state changes: " + ", ".join(forbidden_changes)
        )
    return {"expectedV2PrivateChanges": expected_changes}


def _action_directories(run_dir: Path) -> list[Path]:
    actions_dir = run_dir / "actions"
    if not actions_dir.is_dir():
        return []
    return sorted(path for path in actions_dir.iterdir() if path.is_dir())


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AcceptanceError(f"{label} is not an object")
    return value


def _history_by_request_id(value: Any, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise AcceptanceError(f"{label} is not a session list")
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(value):
        request_id = row.get("requestId")
        if not isinstance(request_id, str) or not request_id:
            raise AcceptanceError(f"{label} row {index + 1} has no durable request ID")
        if request_id in result:
            raise AcceptanceError(f"{label} repeats durable request {request_id}")
        result[request_id] = row
    return result


def _validate_manifest_contract(
    manifest: Mapping[str, Any],
    expected: Mapping[str, Any],
    action_dir: Path,
) -> None:
    _require_equal(action_dir.name, expected["id"], "action evidence directory")
    for key, expected_value in expected.items():
        _require_equal(manifest.get(key), expected_value, f"{action_dir.name} manifest {key}")


def _validate_repeat_result(
    repeated: Mapping[str, Any],
    original: Mapping[str, Any],
) -> None:
    _require_equal(
        session_track_ids(repeated),
        session_track_ids(original),
        "recent-search rerun embedded-track order",
    )
    _require_equal(
        repeated.get("findMusicSessionEvidence"),
        original.get("findMusicSessionEvidence"),
        "recent-search rerun query and ranking-domain evidence",
    )
    _require_equal(
        [row.get("findMusicEvidence") for row in queued_rows(repeated)],
        [row.get("findMusicEvidence") for row in queued_rows(original)],
        "recent-search rerun row-ranking evidence",
    )


def validate_completed_run(run_dir: Path) -> dict[str, Any]:
    state = _require_mapping(load_json(run_dir / "state.json"), "run state")
    if state.get("schemaVersion") != SCHEMA_VERSION:
        raise AcceptanceError("run state uses an unsupported schema")
    if state.get("status") != "COMPLETE":
        raise AcceptanceError(f"run is not complete: {state.get('status')!r}")

    expected_plan = closed_action_plan()
    _require_equal(state.get("plan"), expected_plan, "frozen closed action plan")
    _require_equal(state.get("expectedActionCount"), len(expected_plan), "expected action count")
    _require_equal(state.get("pendingAction"), None, "terminal pending-action receipt")
    completed_actions = _require_mapping(state.get("completedActions"), "completed actions")
    _require_equal(
        set(completed_actions),
        {action["id"] for action in expected_plan},
        "completed action IDs",
    )
    action_dirs = _action_directories(run_dir)
    _require_equal(
        [path.name for path in action_dirs],
        [action["id"] for action in expected_plan],
        "action evidence directories",
    )

    baseline_queue = queue_from_json(load_json(run_dir / "baseline" / "queue.json"))
    baseline = _require_mapping(state.get("baseline"), "frozen baseline")
    baseline_file_ids = [entry.file_id for entry in baseline_queue]
    _require_equal(baseline.get("orderedFileIds"), baseline_file_ids, "frozen baseline file IDs")
    baseline_history = _history_by_request_id(
        load_json(run_dir / "baseline" / "history.json"),
        "baseline history",
    )
    restore_source_id = baseline.get("restoreSourceRequestId")
    if not isinstance(restore_source_id, str) or restore_source_id not in baseline_history:
        raise AcceptanceError("frozen baseline restore source is absent from baseline history")
    frozen_restore_source = baseline_history[restore_source_id]
    _require_equal(
        session_file_ids(frozen_restore_source),
        baseline_file_ids,
        "frozen restore-source file order",
    )

    action_reports: list[dict[str, Any]] = []
    completed_sessions_by_request: dict[str, Mapping[str, Any]] = {}
    completed_sessions_by_action: dict[str, Mapping[str, Any]] = {}
    final_action_queue: list[QueueEntry] | None = None
    for expected_action, action_dir in zip(expected_plan, action_dirs, strict=True):
        manifest = _require_mapping(load_json(action_dir / "manifest.json"), "action manifest")
        _validate_manifest_contract(manifest, expected_action, action_dir)
        before_queue = queue_from_json(load_json(action_dir / "queue-before.json"))
        after_queue = queue_from_json(load_json(action_dir / "queue-after.json"))
        session = _require_mapping(load_json(action_dir / "session.json"), "action session")
        action_type = manifest.get("type")
        placement = manifest.get("placement")
        if action_type == "direct":
            tap_ids = parse_tap_track_ids((action_dir / "tap-log.txt").read_text(encoding="utf-8"))
            validate_direct_action(
                case_kind=manifest["caseKind"],
                placement=placement,
                tap_track_ids=tap_ids,
                before_queue=before_queue,
                after_queue=after_queue,
                session=session,
            )
            repeat_action_id = expected_action.get("sameResultAsActionId")
            if repeat_action_id is not None:
                original = completed_sessions_by_action.get(str(repeat_action_id))
                if original is None:
                    raise AcceptanceError(
                        f"direct rerun source action is not complete: {repeat_action_id!r}"
                    )
                _validate_repeat_result(session, original)
        elif action_type == "replay":
            source_action_id = str(expected_action["sourceActionId"])
            source = completed_sessions_by_action.get(source_action_id)
            if source is None:
                raise AcceptanceError(f"replay source action is not complete: {source_action_id}")
            source_request_id = source.get("requestId")
            _require_equal(
                manifest.get("sourceRequestId"),
                source_request_id,
                "replay manifest source request",
            )
            captured_source = _require_mapping(
                load_json(action_dir / "source-session.json"),
                "captured replay source session",
            )
            _require_equal(captured_source, source, "captured replay source session")
            validate_replay_action(
                placement=placement,
                before_queue=before_queue,
                after_queue=after_queue,
                source_session=source,
                replay_session=session,
            )
        elif action_type == "restore":
            _require_equal(
                manifest.get("sourceRequestId"),
                restore_source_id,
                "restore manifest source request",
            )
            captured_source = _require_mapping(
                load_json(action_dir / "source-session.json"),
                "captured restore source session",
            )
            _require_equal(captured_source, frozen_restore_source, "frozen restore source session")
            validate_replay_action(
                placement="REPLACE_UPCOMING",
                before_queue=before_queue,
                after_queue=after_queue,
                source_session=frozen_restore_source,
                replay_session=session,
                expected_count=len(baseline_queue),
            )
            validate_semantic_restore(
                baseline_queue,
                after_queue,
            )
            final_action_queue = list(after_queue)
        else:
            raise AcceptanceError(f"unknown action type in {action_dir}: {action_type!r}")
        request_id = session.get("requestId")
        if not isinstance(request_id, str) or not request_id:
            raise AcceptanceError(f"action {action_dir.name} has no durable request ID")
        _require_equal(manifest.get("requestId"), request_id, "action manifest request ID")
        _require_equal(completed_actions.get(action_dir.name), request_id, "completed request ID")
        if request_id in completed_sessions_by_request:
            raise AcceptanceError(f"durable request {request_id} appears in more than one action")
        completed_sessions_by_request[request_id] = session
        completed_sessions_by_action[action_dir.name] = session
        action_reports.append(
            {
                "action": action_dir.name,
                "requestId": request_id,
                "type": action_type,
                "placement": placement,
                "verifiedRows": len(queued_rows(session)),
            }
        )

    _require_equal(len(action_reports), len(expected_plan), "completed action count")
    if final_action_queue is None:
        raise AcceptanceError("closed plan has no terminal restore action")
    final_queue = queue_from_json(load_json(run_dir / "final" / "queue.json"))
    _require_equal(final_queue, final_action_queue, "queue stability after terminal restore")
    final_history = _history_by_request_id(
        load_json(run_dir / "final" / "history.json"),
        "final history",
    )
    for request_id, session in completed_sessions_by_request.items():
        _require_equal(
            final_history.get(request_id),
            session,
            f"final durable history session {request_id}",
        )
    runtime = _require_mapping(
        load_json(run_dir / "final" / "runtime-state.json"),
        "final runtime state",
    )
    _require_equal(runtime.get("packageStopped"), False, "final V2 stopped state")
    _require_equal(runtime.get("radioServiceActive"), False, "final RadioService state")
    _require_equal(runtime.get("indexingServiceActive"), False, "final IndexingService state")
    baseline_widgets = load_json(run_dir / "baseline" / "widget-state.json")
    final_widgets = load_json(run_dir / "final" / "widget-state.json")
    if not isinstance(baseline_widgets, list) or not isinstance(final_widgets, list):
        raise AcceptanceError("widget state evidence is not a list")
    _require_equal(
        [row.get("appWidgetId") for row in final_widgets if isinstance(row, Mapping)],
        [row.get("appWidgetId") for row in baseline_widgets if isinstance(row, Mapping)],
        "V2 widget instance IDs",
    )
    if any(
        not isinstance(row, Mapping) or row.get("viewsAssigned") is not True
        for row in final_widgets
    ):
        raise AcceptanceError("a final V2 widget instance has no assigned RemoteViews")
    protected = validate_protected_snapshots(
        run_dir / "baseline" / "snapshot",
        run_dir / "final" / "snapshot",
    )
    validate_semantic_restore(baseline_queue, final_queue)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "status": "PASS",
        "actions": action_reports,
        "semanticQueueRestore": True,
        "occurrenceIdsRequiredToMatchBaseline": False,
        "packageLeftUnstopped": True,
        "servicesLeftStopped": True,
        "widgetInstancesRefreshed": len(final_widgets),
        **protected,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="completed acceptance evidence directory")
    parser.add_argument("--output", type=Path, help="write the validation report here")
    args = parser.parse_args(argv)
    try:
        report = validate_completed_run(args.run_dir.resolve())
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
        return 0
    except (AcceptanceError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

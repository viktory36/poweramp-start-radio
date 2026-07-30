#!/usr/bin/env python3
"""Synthetic, host-only checks for the destructive Find Music acceptance harness."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import Mock

from find_music_delivery_acceptance import (
    AcceptanceError,
    QueueEntry,
    closed_action_plan,
    parse_queue_projection,
    parse_tap_track_ids,
    queue_to_json,
    validate_completed_run,
    validate_direct_action,
    validate_semantic_restore,
    write_json_atomic,
)
from run_find_music_delivery_ui_acceptance import Runner, main as runner_main


def find_music_evidence(case_kind: str) -> dict[str, Any]:
    ingredients = [{"text": "sleep", "weight": 1.0}]
    operator = "EITHER"
    if case_kind == "composed":
        ingredients = [
            {"text": "ambient", "weight": 0.5},
            {"text": "sleep", "weight": 0.5},
        ]
        operator = "ALL_OF"
    return {
        "querySpec": {
            "textIngredients": ingredients,
            "songSeeds": [],
            "operator": operator,
        },
        "orderedActiveTrackIdsSha256": "1" * 64,
        "activeTrackCount": 80_000,
        "objectiveRankingDomainCount": 80_000,
        "stableResultReduction": {
            "identityPolicyVersion": 1,
            "requestedVisibleCount": 30,
            "scannedRowCount": 30,
            "collapsedEquivalentCount": 0,
        },
    }


def terminal_session(
    *,
    request_id: str,
    origin: str,
    placement: str,
    track_ids: Sequence[int],
    file_ids: Sequence[int],
    occurrence_ids: Sequence[int],
    case_kind: str | None,
    source_session: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not (len(track_ids) == len(file_ids) == len(occurrence_ids)):
        raise AssertionError("synthetic session vectors differ in length")
    if source_session is not None:
        session_evidence = copy.deepcopy(source_session.get("findMusicSessionEvidence"))
        source_rows = source_session["tracks"]
    else:
        session_evidence = find_music_evidence(case_kind) if case_kind is not None else None
        source_rows = None
    tracks: list[dict[str, Any]] = []
    for index, (track_id, file_id, occurrence_id) in enumerate(
        zip(track_ids, file_ids, occurrence_ids, strict=True)
    ):
        row_evidence = None
        if source_rows is not None:
            row_evidence = copy.deepcopy(source_rows[index].get("findMusicEvidence"))
        elif case_kind is not None:
            row_evidence = {
                "displayedRank": index + 1,
                "objectiveRank": index + 1,
                "resultScore": 0.9 - index / 1000,
                "rankingScore": 0.9 - index / 1000,
                "ingredientPercentiles": [0.99],
            }
        tracks.append(
            {
                "status": "QUEUED",
                "track": {"id": track_id},
                "resolvedPowerampFileId": file_id,
                "resolvedPowerampQueueId": occurrence_id,
                "findMusicEvidence": row_evidence,
            }
        )
    count = len(tracks)
    return {
        "requestId": request_id,
        "seedTrack": {"title": "synthetic"},
        "tracks": tracks,
        "isComplete": True,
        "isDirectQueue": True,
        "outcome": "SUCCEEDED",
        "directQueuePlacement": placement,
        "delivery": {
            "origin": origin,
            "requestedCount": count,
            "rankedCount": count,
            "resolvedCount": count,
            "verifiedCount": count,
            "verificationComplete": True,
            "mutationCount": 1,
            "unexpectedObservedCount": 0,
            "notInLibraryCount": 0,
            "queueFailedCount": 0,
        },
        "generation": {"generationId": "synthetic-generation"},
        "providerGenerationId": "synthetic-provider-generation",
        "findMusicSessionEvidence": session_evidence,
    }


def tap_log(track_ids: Sequence[int]) -> str:
    return "".join(
        f"1710000000.{index:03d} D MainViewModel: QUEUE_DISPLAYED: [{index}] "
        f"Artist - Track {index} (trackId={track_id})\n"
        for index, track_id in enumerate(track_ids)
    )


def queue_with_new_occurrences(
    before: Sequence[QueueEntry],
    file_ids: Sequence[int],
    occurrence_ids: Sequence[int],
    placement: str,
) -> list[QueueEntry]:
    if placement == "APPEND":
        prefix = list(before)
        next_sort = max((entry.sort for entry in prefix), default=0) + 1
    else:
        prefix = []
        next_sort = 1
    return prefix + [
        QueueEntry(occurrence_id, file_id, next_sort + index)
        for index, (occurrence_id, file_id) in enumerate(
            zip(occurrence_ids, file_ids, strict=True)
        )
    ]


def write_snapshot(root: Path, *, final: bool) -> None:
    root.mkdir(parents=True)
    (root / "device.txt").write_text(
        ("captured_at=2026-07-17T00:00:01+03:00\n" if final else
         "captured_at=2026-07-17T00:00:00+03:00\n")
        + "serial=SYNTHETIC\n"
        + "fingerprint=test/device\n"
        + "android_release=15\n"
        + "android_sdk=35\n",
        encoding="utf-8",
    )
    for label, package in (("v1", "com.powerampstartradio"), ("v2", "com.powerampstartradio.v2")):
        target = root / label
        target.mkdir()
        (target / "status.txt").write_text("installed\n", encoding="utf-8")
        (target / "apk-sha256.txt").write_text(f"{'a' * 64}  /data/app/{package}.apk\n")
        (target / "runtime-grants.txt").write_text(
            "user=0 android.permission.POST_NOTIFICATIONS: granted=true\n",
            encoding="utf-8",
        )
        (target / "package-path.txt").write_text(
            f"package:/data/app/{package}.apk\n",
            encoding="utf-8",
        )
        rows = [f"{'b' * 64} ./files/indexing_v2/generations/active-generation.json"]
        if label == "v2":
            history_digest = "d" * 64 if final else "c" * 64
            rows.append(f"{history_digest} ./files/session_history.json")
        (target / "private-file-sha256.txt").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )


def build_completed_run(root: Path) -> None:
    plan = closed_action_plan()
    baseline_queue = [QueueEntry(1, 101, 1), QueueEntry(2, 102, 2)]
    baseline_source = terminal_session(
        request_id="baseline-source",
        origin="WIDGET_RADIO",
        placement="REPLACE_UPCOMING",
        track_ids=[9001, 9002],
        file_ids=[101, 102],
        occurrence_ids=[1, 2],
        case_kind=None,
    )
    baseline_source["seedTrack"] = {"title": "Miss Melodia"}

    baseline_dir = root / "baseline"
    baseline_dir.mkdir(parents=True)
    write_json_atomic(baseline_dir / "queue.json", queue_to_json(baseline_queue))
    write_json_atomic(baseline_dir / "history.json", [baseline_source])
    write_json_atomic(
        baseline_dir / "widget-state.json",
        [{"appWidgetId": 40, "viewsAssigned": True}],
    )
    write_snapshot(baseline_dir / "snapshot", final=False)

    results_by_case: dict[str, tuple[list[int], list[int]]] = {}
    sessions_by_action: dict[str, dict[str, Any]] = {}
    all_sessions: list[dict[str, Any]] = []
    current_queue = list(baseline_queue)
    next_occurrence = 100

    for case_number, case_id in enumerate(
        ("sleep_varied", "ambient_sleep_all_of", "sleep_closest"),
        start=1,
    ):
        results_by_case[case_id] = (
            [case_number * 10_000 + index for index in range(30)],
            [case_number * 20_000 + index for index in range(30)],
        )

    for action in plan:
        action_dir = root / "actions" / action["id"]
        action_dir.mkdir(parents=True)
        write_json_atomic(action_dir / "queue-before.json", queue_to_json(current_queue))

        if action["type"] == "restore":
            source = baseline_source
            track_ids = [9001, 9002]
            file_ids = [101, 102]
            case_kind = None
            origin = "HISTORY_REQUEUE"
        elif action["type"] == "replay":
            source = sessions_by_action[action["sourceActionId"]]
            track_ids = [int(row["track"]["id"]) for row in source["tracks"]]
            file_ids = [int(row["resolvedPowerampFileId"]) for row in source["tracks"]]
            case_kind = None
            origin = "HISTORY_REQUEUE"
        else:
            source = None
            track_ids, file_ids = results_by_case[action["caseId"]]
            case_kind = action["caseKind"]
            origin = "TEXT_RESULT_LIST" if case_kind == "text" else "COMPOSED_RESULT_LIST"

        occurrence_ids = list(range(next_occurrence, next_occurrence + len(file_ids)))
        next_occurrence += len(file_ids)
        current_queue = queue_with_new_occurrences(
            current_queue,
            file_ids,
            occurrence_ids,
            action["placement"],
        )
        session = terminal_session(
            request_id=f"request-{action['id']}",
            origin=origin,
            placement=action["placement"],
            track_ids=track_ids,
            file_ids=file_ids,
            occurrence_ids=occurrence_ids,
            case_kind=case_kind,
            source_session=source,
        )
        sessions_by_action[action["id"]] = session
        all_sessions.append(session)
        write_json_atomic(action_dir / "queue-after.json", queue_to_json(current_queue))
        write_json_atomic(action_dir / "session.json", session)
        if source is not None:
            write_json_atomic(action_dir / "source-session.json", source)
        (action_dir / "tap-log.txt").write_text(
            tap_log(track_ids) if action["type"] == "direct" else "",
            encoding="utf-8",
        )
        manifest = dict(action)
        manifest["requestId"] = session["requestId"]
        if source is not None:
            manifest["sourceRequestId"] = source["requestId"]
        write_json_atomic(action_dir / "manifest.json", manifest)

    state = {
        "schemaVersion": 1,
        "status": "COMPLETE",
        "expectedActionCount": len(plan),
        "plan": plan,
        "pendingAction": None,
        "completedActions": {
            action["id"]: sessions_by_action[action["id"]]["requestId"] for action in plan
        },
        "baseline": {
            "orderedFileIds": [101, 102],
            "restoreSourceRequestId": "baseline-source",
        },
    }
    write_json_atomic(root / "state.json", state)
    final_dir = root / "final"
    final_dir.mkdir()
    write_json_atomic(final_dir / "queue.json", queue_to_json(current_queue))
    write_json_atomic(final_dir / "history.json", [baseline_source, *all_sessions])
    write_json_atomic(
        final_dir / "widget-state.json",
        [{"appWidgetId": 40, "viewsAssigned": True}],
    )
    write_json_atomic(
        final_dir / "runtime-state.json",
        {
            "packageStopped": False,
            "radioServiceActive": False,
            "indexingServiceActive": False,
        },
    )
    write_snapshot(final_dir / "snapshot", final=True)


class ParserAndContractTest(unittest.TestCase):
    def test_queue_projection_accepts_prefixed_fields_and_orders_by_sort(self) -> None:
        parsed = parse_queue_projection(
            "Row: 0 queue._id=12, queue.folder_file_id=22, queue.sort=2\n"
            "Row: 1 _id=11, folder_file_id=21, sort=1\n"
        )
        self.assertEqual(
            parsed,
            [QueueEntry(11, 21, 1), QueueEntry(12, 22, 2)],
        )

    def test_queue_projection_rejects_unknown_or_duplicate_rows(self) -> None:
        with self.assertRaises(AcceptanceError):
            parse_queue_projection("No result found.\n")
        with self.assertRaises(AcceptanceError):
            parse_queue_projection(
                "Row: 0 _id=11, folder_file_id=21, sort=1\n"
                "Row: 1 _id=11, folder_file_id=22, sort=2\n"
            )

    def test_tap_parser_requires_one_contiguous_consistent_sequence(self) -> None:
        self.assertEqual(parse_tap_track_ids(tap_log([40, 41, 42])), [40, 41, 42])
        with self.assertRaises(AcceptanceError):
            parse_tap_track_ids(
                "QUEUE_DISPLAYED: [0] A (trackId=40)\n"
                "QUEUE_DISPLAYED: [2] C (trackId=42)\n"
            )
        with self.assertRaises(AcceptanceError):
            parse_tap_track_ids(
                "QUEUE_DISPLAYED: [0] A (trackId=40)\n"
                "QUEUE_DISPLAYED: [0] B (trackId=41)\n"
            )

    def test_direct_append_binds_tap_order_and_exact_occurrences(self) -> None:
        before = [QueueEntry(1, 101, 1)]
        after = [*before, QueueEntry(10, 201, 2), QueueEntry(11, 202, 3)]
        session = terminal_session(
            request_id="direct",
            origin="TEXT_RESULT_LIST",
            placement="APPEND",
            track_ids=[301, 302],
            file_ids=[201, 202],
            occurrence_ids=[10, 11],
            case_kind="text",
        )
        validate_direct_action(
            case_kind="text",
            placement="APPEND",
            tap_track_ids=[301, 302],
            before_queue=before,
            after_queue=after,
            session=session,
            expected_count=2,
        )
        with self.assertRaises(AcceptanceError):
            validate_direct_action(
                case_kind="text",
                placement="APPEND",
                tap_track_ids=[302, 301],
                before_queue=before,
                after_queue=after,
                session=session,
                expected_count=2,
            )

    def test_semantic_restore_allows_new_occurrence_ids_only(self) -> None:
        baseline = [QueueEntry(1, 101, 1), QueueEntry(2, 102, 2)]
        validate_semantic_restore(
            baseline,
            [QueueEntry(91, 101, 1), QueueEntry(92, 102, 2)],
        )
        with self.assertRaises(AcceptanceError):
            validate_semantic_restore(
                baseline,
                [QueueEntry(91, 102, 1), QueueEntry(92, 101, 2)],
            )

    def test_completed_run_revalidates_terminal_restore_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            build_completed_run(run_dir)
            report = validate_completed_run(run_dir)
            self.assertEqual(report["status"], "PASS")
            self.assertFalse(report["occurrenceIdsRequiredToMatchBaseline"])

            restore_dir = run_dir / "actions" / closed_action_plan()[-1]["id"]
            restore = json.loads((restore_dir / "session.json").read_text(encoding="utf-8"))
            restore["outcome"] = "PARTIAL_FAILED"
            write_json_atomic(restore_dir / "session.json", restore)
            final_history_path = run_dir / "final" / "history.json"
            final_history = json.loads(final_history_path.read_text(encoding="utf-8"))
            final_history[-1] = restore
            write_json_atomic(final_history_path, final_history)
            with self.assertRaisesRegex(AcceptanceError, "session outcome"):
                validate_completed_run(run_dir)


class PendingResumeTest(unittest.TestCase):
    @staticmethod
    def runner(output_dir: Path, ack: str | None = None) -> Runner:
        return Runner(
            repo_root=Path("/synthetic"),
            output_dir=output_dir,
            serial="synthetic",
            ack_no_mutation=ack,
        )

    def test_pending_without_receipt_never_retries_or_clears_itself(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self.runner(Path(temporary))
            action_id = closed_action_plan()[0]["id"]
            state = {
                "pendingAction": {
                    "actionId": action_id,
                    "historyRequestIdsBefore": [],
                },
                "completedActions": {},
            }
            runner._read_history = Mock(return_value=[])
            runner._acknowledge_no_mutation = Mock(
                side_effect=AcceptanceError("explicit acknowledgement required")
            )
            runner._write_action_evidence = Mock()
            with self.assertRaisesRegex(AcceptanceError, "explicit acknowledgement"):
                runner._reconcile_pending(state)
            self.assertEqual(state["pendingAction"]["actionId"], action_id)
            runner._write_action_evidence.assert_not_called()

    def test_one_terminal_session_is_reconciled_without_a_second_tap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self.runner(Path(temporary))
            action_id = closed_action_plan()[0]["id"]
            session = {"requestId": "new-request", "outcome": "SUCCEEDED"}
            state = {
                "pendingAction": {
                    "actionId": action_id,
                    "historyRequestIdsBefore": ["old-request"],
                },
                "completedActions": {},
            }
            runner._read_history = Mock(return_value=[session])
            runner._write_action_evidence = Mock()
            runner._save_state = Mock()
            runner._reconcile_pending(state)
            runner._write_action_evidence.assert_called_once()
            self.assertIsNone(state["pendingAction"])
            self.assertEqual(state["completedActions"][action_id], "new-request")

    def test_no_mutation_ack_requires_byte_identical_history_and_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            action_id = closed_action_plan()[0]["id"]
            history = b'[{"requestId":"old-request"}]\n'
            queue_raw = "Row: 0 _id=1, folder_file_id=101, sort=1\n"
            queue = [QueueEntry(1, 101, 1)]
            pending = {
                "actionId": action_id,
                "historyRequestIdsBefore": ["old-request"],
                "historySha256Before": hashlib.sha256(history).hexdigest(),
                "requestJournalSha256Before": {"state.json": "a" * 64},
                "queueBefore": queue_to_json(queue),
                "queueBeforeSha256": hashlib.sha256(queue_raw.encode()).hexdigest(),
            }
            state = {"pendingAction": copy.deepcopy(pending)}
            runner = self.runner(Path(temporary), ack=action_id)
            runner._history_bytes = Mock(return_value=history)
            runner._request_journal_hashes = Mock(
                return_value={"state.json": "a" * 64}
            )
            runner._queue = Mock(return_value=(queue_raw, queue))
            runner._save_state = Mock()
            runner._acknowledge_no_mutation(state)
            self.assertIsNone(state["pendingAction"])

            state = {"pendingAction": copy.deepcopy(pending)}
            runner._history_bytes = Mock(return_value=history.replace(b"\n", b" \n"))
            with self.assertRaisesRegex(AcceptanceError, "history bytes changed"):
                runner._acknowledge_no_mutation(state)
            self.assertIsNotNone(state["pendingAction"])

    def test_final_runtime_parsers_require_unstopped_package_and_assigned_widget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self.runner(Path(temporary))
            appwidget = """
  [205] provider ProviderId{user:0, cmp:ComponentInfo{com.powerampstartradio.v2/com.powerampstartradio.widget.StartRadioWidgetReceiver}}
  [6] id=40
    host=HostId{user:0, pkg:launcher}
    provider=ProviderId{user:0, cmp:ComponentInfo{com.powerampstartradio.v2/com.powerampstartradio.widget.StartRadioWidgetReceiver}}
    views=android.widget.RemoteViews@1234
  [7] id=41
    provider=ProviderId{user:0, cmp:ComponentInfo{another.package/AnotherWidget}}
    views=null
"""
            runner.adb.shell = Mock(side_effect=[
                "User 0: installed=true stopped=false notLaunched=false\n",
                appwidget,
            ])
            _, stopped = runner._package_stopped()
            _, widgets = runner._widget_state()
            self.assertFalse(stopped)
            self.assertEqual(widgets, [{"appWidgetId": 40, "viewsAssigned": True}])

    def test_dry_run_never_constructs_an_adb_runner(self) -> None:
        output = io.StringIO()
        original_init = Runner.__init__
        Runner.__init__ = Mock(side_effect=AssertionError("dry run touched Runner"))  # type: ignore[method-assign]
        try:
            with contextlib.redirect_stdout(output):
                self.assertEqual(runner_main(["--dry-run"]), 0)
        finally:
            Runner.__init__ = original_init  # type: ignore[method-assign]
        document = json.loads(output.getvalue())
        self.assertEqual(document["mutationCount"], 13)
        self.assertTrue(document["ordinaryProductionUiOnly"])


if __name__ == "__main__":
    unittest.main()

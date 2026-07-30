from __future__ import annotations

import json
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from run_v2_widget_device_acceptance import (
    APPROVAL_FILENAME,
    COLD_WIDGET_STATUS_MESSAGE,
    MAX_IMMEDIATE_WIDGET_STATUS_LATENCY_SECONDS,
    SCHEMA_VERSION,
    STATE_FILENAME,
    Runner,
    dry_run_document,
    validate_immediate_cold_widget_status,
    validate_completed_output,
)
from widget_device_acceptance import (
    AcceptanceError,
    classify_overlay_windows,
    find_restore_session,
    overlay_matches_baseline,
    parse_authenticated_state,
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
    queue_file_ids,
    terminal_delivery_file_ids,
    validate_no_track_widget,
    validate_provider_backed_display_track,
    validate_ready_widget,
    validate_widget_session,
    visible_status_subtitle,
)


def widget_xml(
    *,
    root_description: str,
    action_description: str,
    title: str,
    subtitle: str,
) -> bytes:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<hierarchy rotation="0">
  <node class="com.android.launcher3.widget.LauncherAppWidgetHostView"
      package="com.sonymobile.launcher" content-desc="Poweramp Start Radio V2"
      bounds="[35,624][421,851]">
    <node resource-id="com.powerampstartradio.v2:id/widget_root"
        package="com.powerampstartradio.v2" content-desc="{root_description}"
        clickable="true" enabled="true" bounds="[35,624][421,851]">
      <node resource-id="com.powerampstartradio.v2:id/widget_start_button"
          package="com.powerampstartradio.v2" content-desc="{action_description}"
          clickable="true" enabled="true" bounds="[40,677][160,797]" />
      <node package="com.powerampstartradio.v2">
        <node resource-id="com.powerampstartradio.v2:id/widget_track_title"
            package="com.powerampstartradio.v2" text="{title}" />
        <node resource-id="com.powerampstartradio.v2:id/widget_track_subtitle"
            package="com.powerampstartradio.v2" text="{subtitle}" />
      </node>
    </node>
  </node>
</hierarchy>'''.encode()


def successful_session(
    request_id: str,
    *,
    origin: str,
    file_ids: list[int],
    title: str = "Seed",
    seed_id: int = 91,
    seed_path: str = "/music/seed.flac",
) -> dict:
    stable_seed_id = "stable-track-span-v1-" + "a" * 64
    return {
        "requestId": request_id,
        "outcome": "SUCCEEDED",
        "isComplete": True,
        "generation": {
            "generationId": "g1",
            "activationBindingId": "binding-1",
            "manifestSha256": "a" * 64,
            "embeddingSpecId": "clamp3-v1",
            "databaseContentSha256": "b" * 64,
            "orderedTrackSetSha256": "c" * 64,
            "stableTrackUidMappingSha256": "d" * 64,
        },
        "providerGenerationId": "provider-1",
        "seedTrack": {
            "realId": seed_id,
            "title": title,
            "path": seed_path,
        },
        "seedIdentity": {
            "embeddedTrackId": 8,
            "stableTrackSpanId": stable_seed_id,
        },
        "delivery": {
            "origin": origin,
            "verificationComplete": True,
            "verifiedCount": len(file_ids),
        },
        "tracks": [
            {
                "status": "QUEUED",
                "track": {"id": 100 + index},
                "resolvedPowerampFileId": file_id,
                "resolvedPowerampQueueId": 1_000 + index,
            }
            for index, file_id in enumerate(file_ids)
        ],
    }


def test_no_track_widget_is_exact_high_signal_copy() -> None:
    view = parse_widget_view(
        widget_xml(
            root_description="Open Poweramp Start Radio V2",
            action_description="Open Poweramp to play a track",
            title="No track playing",
            subtitle="Play in Poweramp",
        )
    )
    validate_no_track_widget(view)
    assert view.state == "NO_TRACK"
    assert view.action_bounds == (40, 677, 160, 797)


def test_widget_parser_accepts_android_uiautomator_stdout_trailer() -> None:
    raw = widget_xml(
        root_description="Open Poweramp Start Radio V2",
        action_description="Open Poweramp to play a track",
        title="No track playing",
        subtitle="Play in Poweramp",
    ) + b"UI hierchary dumped to: /dev/tty\n"
    validate_no_track_widget(parse_widget_view(raw))


def test_ready_widget_must_match_exact_playback_identity_copy() -> None:
    track = {
        "realId": 13319,
        "title": "Miss Melodia",
        "artist": "L. Subramaniam",
        "album": "Subramaniam in Moscow",
        "path": "/music/01. Miss Melodia.flac",
    }
    view = parse_widget_view(
        widget_xml(
            root_description="Open Poweramp Start Radio V2. Current track: Miss Melodia",
            action_description="Start radio from Miss Melodia",
            title="Miss Melodia",
            subtitle="L. Subramaniam · Subramaniam in Moscow",
        )
    )
    validate_ready_widget(view, track)
    with pytest.raises(AcceptanceError, match="exactly match"):
        validate_ready_widget(view, {**track, "title": "Later Track"})


def test_widget_parser_refuses_duplicate_v2_host() -> None:
    one = widget_xml(
        root_description="Open Poweramp Start Radio V2",
        action_description="Open Poweramp to play a track",
        title="No track playing",
        subtitle="Play in Poweramp",
    ).decode()
    duplicate = one.replace("</hierarchy>", one.split("<hierarchy rotation=\"0\">", 1)[1])
    with pytest.raises(AcceptanceError, match="2 V2 widget"):
        parse_widget_view(duplicate.encode())


def test_queue_parser_keeps_exact_occurrence_order() -> None:
    raw = "\n".join(
        [
            "Row: 0 _id=71, folder_file_id=11, sort=1",
            "Row: 1 _id=72, folder_file_id=11, sort=2",
            "Row: 2 _id=73, folder_file_id=15, sort=3",
        ]
    )
    rows = parse_queue_projection(raw)
    assert queue_file_ids(rows) == [11, 11, 15]
    with pytest.raises(AcceptanceError, match="not in sort order"):
        parse_queue_projection("\n".join(reversed(raw.splitlines())))


def test_restore_session_requires_exact_modern_verified_order() -> None:
    wrong = successful_session("wrong", origin="APP_RADIO", file_ids=[1, 3, 2])
    exact = successful_session("exact", origin="HISTORY_REQUEUE", file_ids=[1, 2, 3])
    exact["seedTrack"]["title"] = "Replay: Seed"
    assert find_restore_session([wrong, exact], [1, 2, 3])["requestId"] == "exact"
    del exact["generation"]
    with pytest.raises(AcceptanceError, match="no exact replayable"):
        find_restore_session([wrong, exact], [1, 2, 3])


def test_widget_session_binds_status_session_and_tap_seed() -> None:
    session = successful_session(
        "widget-1",
        origin="WIDGET_RADIO",
        file_ids=[7, 8],
        title="Seed",
        seed_id=91,
        seed_path="/music/seed.flac",
    )
    status = {
        "schemaVersion": 1,
        "requestId": "widget-1",
        "state": "SUCCEEDED",
        "message": "2 tracks queued from Seed",
        "seed": {
            "powerampFileId": 91,
            "displayTitle": "Seed",
            "normalizedTitle": "seed",
            "normalizedPath": "/music/seed.flac",
            "embeddedTrackId": 8,
            "stableTrackSpanId": "stable-track-span-v1-" + "a" * 64,
        },
    }
    tap = {"realId": 91, "title": "Seed", "path": "/music/seed.flac"}
    assert validate_widget_session(session=session, status=status, tap_track=tap) == [7, 8]
    with pytest.raises(AcceptanceError, match="tap-time seed"):
        validate_widget_session(
            session=session,
            status=status,
            tap_track={**tap, "realId": 92},
        )
    wrong_embedded_seed = json.loads(json.dumps(status))
    wrong_embedded_seed["seed"]["embeddedTrackId"] = 9
    with pytest.raises(AcceptanceError, match="embedded seed identities differ"):
        validate_widget_session(
            session=session,
            status=wrong_embedded_seed,
            tap_track=tap,
        )


def test_verified_count_cannot_lie_about_track_rows() -> None:
    session = successful_session("one", origin="WIDGET_RADIO", file_ids=[4])
    session["delivery"]["verifiedCount"] = 2
    with pytest.raises(AcceptanceError, match="verified count"):
        terminal_delivery_file_ids(session)


def test_terminal_session_requires_complete_current_identity_evidence() -> None:
    valid = successful_session("identity", origin="WIDGET_RADIO", file_ids=[4, 4])
    assert terminal_delivery_file_ids(valid) == [4, 4]

    missing_occurrence = json.loads(json.dumps(valid))
    del missing_occurrence["tracks"][0]["resolvedPowerampQueueId"]
    with pytest.raises(AcceptanceError, match="queue occurrence"):
        terminal_delivery_file_ids(missing_occurrence)

    duplicate_occurrence = json.loads(json.dumps(valid))
    duplicate_occurrence["tracks"][1]["resolvedPowerampQueueId"] = (
        duplicate_occurrence["tracks"][0]["resolvedPowerampQueueId"]
    )
    with pytest.raises(AcceptanceError, match="reuses one Poweramp queue occurrence"):
        terminal_delivery_file_ids(duplicate_occurrence)

    missing_embedded_row = json.loads(json.dumps(valid))
    del missing_embedded_row["tracks"][0]["track"]
    with pytest.raises(AcceptanceError, match="embedded track ID"):
        terminal_delivery_file_ids(missing_embedded_row)

    missing_provider_generation = json.loads(json.dumps(valid))
    del missing_provider_generation["providerGenerationId"]
    with pytest.raises(AcceptanceError, match="provider generation"):
        terminal_delivery_file_ids(missing_provider_generation)

    incomplete_embedding_generation = json.loads(json.dumps(valid))
    del incomplete_embedding_generation["generation"]["manifestSha256"]
    with pytest.raises(AcceptanceError, match="generation identity is incomplete"):
        terminal_delivery_file_ids(incomplete_embedding_generation)


def test_authenticated_state_is_exact_and_stopped_has_no_seed() -> None:
    payload = {
        "schemaVersion": 1,
        "track": {"realId": 9, "title": "Track", "path": "/music/t.flac"},
        "playbackState": 1,
        "lastEventTimestampMs": 12,
    }
    xml = (
        "<?xml version='1.0' encoding='utf-8'?><map><string "
        f"name='authenticated_explicit_state'>{json.dumps(payload)}</string></map>"
    ).encode()
    assert parse_authenticated_state(xml)["track"]["realId"] == 9
    payload["playbackState"] = 0
    with pytest.raises(AcceptanceError, match="retains a command seed"):
        parse_authenticated_state(
            (
                "<map><string name='authenticated_explicit_state'>"
                + json.dumps(payload)
                + "</string></map>"
            ).encode()
        )


def test_display_track_requires_independent_exact_provider_and_queue_proof() -> None:
    display_xml = b"""<?xml version='1.0' encoding='utf-8' standalone='yes' ?>
<map>
  <long name="current_track_real_id" value="91" />
  <string name="current_track_title">Seed Song</string>
  <string name="current_track_artist">Artist</string>
  <string name="current_track_album">Album</string>
  <int name="current_track_duration_ms" value="180200" />
  <string name="current_track_path">/music/Artist/Seed Song.flac</string>
  <long name="current_track_category_row_id" value="701" />
  <string name="current_track_category_uri">content://com.maxmpz.audioplayer.data/queue?shs=2</string>
  <int name="current_track_position_in_list" value="4" />
</map>"""
    display = parse_poweramp_display_track(display_xml)
    provider = parse_poweramp_file_row(
        "Row: 0 _id=91, artist=Artist, album=Album, title_tag=Seed Song, "
        "duration=180000, path=/music/Artist, name=Seed Song.flac, "
        "offset_ms=NULL, cue_folder_id=NULL"
    )
    queue = parse_queue_projection(
        "Row: 0 _id=701, folder_file_id=91, sort=5\n"
        "Row: 1 _id=702, folder_file_id=92, sort=6"
    )

    verified = validate_provider_backed_display_track(display, provider, queue)
    assert verified["realId"] == 91
    assert verified["path"] == "/music/Artist/Seed Song.flac"
    with pytest.raises(AcceptanceError, match="Queue occurrence"):
        validate_provider_backed_display_track(
            display,
            provider,
            parse_queue_projection("Row: 0 _id=701, folder_file_id=999, sort=5"),
        )


def test_poweramp_media_session_parser_ignores_other_packages() -> None:
    raw = """
Sessions Stack - have 2 sessions:
  Other com.example.player/Other/1 (userId=0)
    package=com.example.player
    state=PlaybackState {state=PLAYING(3), position=10, buffered position=0}
  Poweramp com.maxmpz.audioplayer/Poweramp/753 (userId=0)
    package=com.maxmpz.audioplayer
    state=PlaybackState {state=PAUSED(2), position=20, buffered position=0}
"""
    assert parse_poweramp_media_session(raw) == {
        "session": "com.maxmpz.audioplayer/Poweramp/753",
        "stateCode": 2,
        "state": "PAUSED",
    }
    with pytest.raises(AcceptanceError, match="0 Poweramp media sessions"):
        parse_poweramp_media_session(raw.replace("com.maxmpz.audioplayer", "com.other"))


def test_musixmatch_overlay_geometry_is_frozen_and_only_expanded_has_collapse_target() -> None:
    expanded_raw = """
  Window #3 Window{aaa u0 com.musixmatch.android.lyrify/Main}:
    mHasSurface=true isOnScreen=true
    frame=(0,31)(1080x1099)
  Window #4 Window{bbb u0 com.musixmatch.android.lyrify/Close}:
    mHasSurface=true isOnScreen=true
    frame=(20,1130)(154x140)
"""
    expanded = classify_overlay_windows(
        parse_overlay_windows(expanded_raw),
        display_width=1080,
        display_height=2340,
    )
    assert expanded == {
        "state": "EXPANDED",
        "frames": [[0, 31, 1080, 1130], [20, 1130, 174, 1270]],
        "collapseTarget": [20, 1130, 174, 1270],
    }

    collapsed_raw = """
  Window #3 Window{ccc u0 com.musixmatch.android.lyrify/Bubble}:
    mHasSurface=true isOnScreen=true
    mFrame=[-22,2140][132,2294]
"""
    collapsed = classify_overlay_windows(
        parse_overlay_windows(collapsed_raw),
        display_width=1080,
        display_height=2340,
    )
    assert collapsed["state"] == "COLLAPSED"
    assert overlay_matches_baseline(collapsed, collapsed)
    assert not overlay_matches_baseline(collapsed, expanded)


def test_ready_wait_uses_provider_and_media_session_not_authenticated_file() -> None:
    source = inspect.getsource(Runner.wait_widget)
    assert "provider_verified_display_track" in source
    assert "poweramp_media_session" in source
    assert "expected_playback_state" in source
    assert "authenticated_predicate" not in source
    assert "authenticated_state_diagnostic" in source
    assert "latency = first_correct_display_at - event_started" in source
    assert "evidenceVerificationSeconds" in source


def test_device_text_parsers_fail_closed() -> None:
    assert parse_music_volume("[V] volume is 12 in range [0..30]") == {
        "current": 12,
        "minimum": 0,
        "maximum": 30,
    }
    assert parse_package_stopped("  User 0: installed=true stopped=false enabled=0") is False
    raw = """
  [6] id=40
    host=HostId{user:0, app:10300, hostId:1024, pkg=com.sonymobile.launcher}
    provider=ProviderId{user:0, app:10653, cmp:ComponentInfo{com.powerampstartradio.v2/com.powerampstartradio.widget.StartRadioWidgetReceiver}}
    views=android.widget.RemoteViews@ae57a8
"""
    assert parse_widget_instances(raw) == [
        {"appWidgetId": 40, "hostPackage": "com.sonymobile.launcher", "viewsAssigned": True}
    ]


def test_plan_digest_is_canonical_and_sensitive() -> None:
    assert plan_digest({"b": 2, "a": 1}) == plan_digest({"a": 1, "b": 2})
    assert plan_digest({"a": 1}) != plan_digest({"a": 2})


def test_dry_run_separates_delivery_from_intelligence() -> None:
    document = dry_run_document()
    assert document["prepare"]["readOnly"] is True
    assert document["execute"]["requiresExactPlanSha256"] is True
    assert document["execute"]["neverRepeatsUncertainTap"] is True
    phases = document["execute"]["phases"]
    assert any("true cold single tap" in phase for phase in phases)
    assert any("cold rapid double tap" in phase for phase in phases)
    assert phases.index(
        next(phase for phase in phases if "true cold single tap" in phase)
    ) < phases.index(next(phase for phase in phases if "cold rapid double tap" in phase))
    assert "recommendation-quality claim" in document["forbidden"][-1]


def test_immediate_cold_widget_status_is_new_exact_and_bounded() -> None:
    track = {"realId": 91, "title": "Seed", "path": "/music/seed.flac"}
    status = {
        "schemaVersion": 1,
        "requestId": "cold-single",
        "state": "STARTING",
        "message": COLD_WIDGET_STATUS_MESSAGE,
        "seed": {
            "powerampFileId": 91,
            "displayTitle": "Seed",
            "normalizedTitle": "seed",
            "normalizedPath": "/music/seed.flac",
        },
    }
    assert (
        validate_immediate_cold_widget_status(
            status=status,
            tap_track=track,
            prior_request_ids={"older"},
            latency_seconds=0.25,
        )
        == "cold-single"
    )

    invalid_cases = [
        ({**status, "requestId": "older"}, 0.25, "new request-bound"),
        ({**status, "state": "SUCCEEDED"}, 0.25, "cold-reconciliation"),
        ({**status, "message": "Starting radio"}, 0.25, "cold-reconciliation"),
        (status, MAX_IMMEDIATE_WIDGET_STATUS_LATENCY_SECONDS + 0.01, "immediately"),
        (status, -0.01, "immediately"),
        (
            {**status, "seed": {**status["seed"], "powerampFileId": 92}},
            0.25,
            "tap-time seed",
        ),
    ]
    for candidate, latency, message in invalid_cases:
        with pytest.raises(AcceptanceError, match=message):
            validate_immediate_cold_widget_status(
                status=candidate,
                tap_track=track,
                prior_request_ids={"older"},
                latency_seconds=latency,
            )


def test_widget_tap_phase_call_graph_keeps_single_and_double_distinct() -> None:
    single = inspect.getsource(Runner.run_widget_single_tap)
    double = inspect.getsource(Runner.run_widget_double_tap)
    execute = inspect.getsource(Runner.execute)

    assert single.count('self.adb.shell("input", "tap"') == 1
    assert double.count('self.adb.shell("input", "tap"') == 2
    assert "validate_immediate_cold_widget_status" in single
    assert "immediateStatusObservation" in single
    assert 'self.keyevent("KEYCODE_MEDIA_PAUSE")' in single
    assert single.count('expected_playback_state="PAUSED"') == 3
    assert "widget click changed the paused current-track identity" in single
    assert "V2 process was not absent immediately before the single tap" in single
    assert "V2 process was not absent immediately before the double tap" in double
    assert execute.index("run_widget_single_tap") < execute.index("run_widget_double_tap")
    assert SCHEMA_VERSION == 2


def test_completed_evidence_validator_is_hash_complete(tmp_path: Path) -> None:
    baseline = successful_session("base", origin="APP_RADIO", file_ids=[4, 5])
    single = successful_session("single", origin="WIDGET_RADIO", file_ids=[8, 9])
    double = successful_session("double", origin="WIDGET_RADIO", file_ids=[10, 11])
    restore = successful_session("restore", origin="HISTORY_REQUEUE", file_ids=[4, 5])
    plan = {
        "schemaVersion": SCHEMA_VERSION,
        "baseline": {
            "orderedFileIds": [4, 5],
            "historyRequestIds": ["base"],
            "musixmatchOverlay": {
                "state": "ABSENT",
                "frames": [],
                "collapseTarget": None,
            },
        },
    }
    digest = plan_digest(plan)
    report = {
        "status": "COMPLETE",
        "planSha256": digest,
        "scope": {
            "radioQueueIntelligence": (
                "Not judged by this harness; recommendation quality requires the separate "
                "listening evaluation"
            )
        },
        "coldSingleTapImmediateRequestStatus": "PASS",
        "coldSingleTapOneDurableSession": "PASS",
        "pausedTrackWidgetTap": "PASS",
        "rapidDoubleTapSingleFlight": "PASS",
        "widgetSingleRequestId": "single",
        "widgetDoubleRequestId": "double",
        "restoreRequestId": "restore",
    }
    (tmp_path / "final").mkdir()
    (tmp_path / "widget-single-tap").mkdir()
    (tmp_path / "widget-double-tap").mkdir()
    (tmp_path / APPROVAL_FILENAME).write_text(json.dumps(plan), encoding="utf-8")
    (tmp_path / STATE_FILENAME).write_text(
        json.dumps(
            {
                "schemaVersion": SCHEMA_VERSION,
                "status": "COMPLETE",
                "planSha256": digest,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "final/validation.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "final/queue.json").write_text(
        json.dumps([{"fileId": 4}, {"fileId": 5}]), encoding="utf-8"
    )
    (tmp_path / "final/history.json").write_text(
        json.dumps([baseline, single, double, restore]), encoding="utf-8"
    )
    (tmp_path / "final/musixmatch-overlay.json").write_text(
        json.dumps({"state": "ABSENT", "frames": [], "collapseTarget": None}),
        encoding="utf-8",
    )
    (tmp_path / "final/launcher-widget.xml").write_bytes(
        widget_xml(
            root_description="Open Poweramp Start Radio V2",
            action_description="Open Poweramp to play a track",
            title="No track playing",
            subtitle="Play in Poweramp",
        )
    )
    (tmp_path / "widget-single-tap/widget-action-validation.json").write_text(
        json.dumps(
            {
                "requestId": "single",
                "tapCount": 1,
                "coldProcessAbsentImmediatelyBeforeTap": True,
                "durableSessionCount": 1,
                "noDuplicateTerminalSession": True,
                "exactOrderedQueue": True,
                "pausedCurrentTrackAcrossTap": True,
                "immediateRequestStatus": {
                    "state": "STARTING",
                    "message": COLD_WIDGET_STATUS_MESSAGE,
                    "latencySeconds": 0.25,
                    "exactTapSeed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "widget-double-tap/widget-action-validation.json").write_text(
        json.dumps(
            {
                "requestId": "double",
                "tapCount": 2,
                "coldProcessAbsentImmediatelyBeforeTap": True,
                "durableSessionCount": 1,
                "noDuplicateTerminalSession": True,
                "exactOrderedQueue": True,
            }
        ),
        encoding="utf-8",
    )
    manifest = []
    for path in sorted(tmp_path.rglob("*")):
        if path.is_file() and path.name != "evidence-sha256.json":
            import hashlib

            manifest.append(
                {
                    "path": str(path.relative_to(tmp_path)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    (tmp_path / "evidence-sha256.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_completed_output(tmp_path)["status"] == "COMPLETE"
    (tmp_path / "final/queue.json").write_text("[]", encoding="utf-8")
    with pytest.raises(AcceptanceError, match="queue differs"):
        validate_completed_output(tmp_path)


def test_cli_dry_run_never_needs_adb_and_execute_needs_digest() -> None:
    script = Path(__file__).with_name("run_v2_widget_device_acceptance.py")
    dry = subprocess.run([sys.executable, str(script), "--dry-run"], capture_output=True, text=True)
    assert dry.returncode == 0
    assert json.loads(dry.stdout)["prepare"]["readOnly"] is True
    rejected = subprocess.run(
        [sys.executable, str(script), "--execute", "--output-dir", "/tmp/never-run"],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "requires --approve-plan-sha256" in rejected.stderr


def test_prepare_call_graph_contains_only_read_surfaces() -> None:
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            Runner.prepare,
            Runner._capture_baseline_files,
            Runner._verify_preparation_stable,
        )
    )
    for forbidden in (
        "set_music_volume(",
        "keyevent(",
        "launch_package(",
        "launch_v2(",
        "kill_process_ordinary(",
        'self.adb.run("reboot"',
        'self.adb.shell("input"',
        'self.adb.shell("am"',
    ):
        assert forbidden not in source


def test_execute_rejects_wrong_digest_before_adb(tmp_path: Path) -> None:
    plan = {
        "schemaVersion": SCHEMA_VERSION,
        "acknowledgementName": "WIDGET_PLAYBACK_QUEUE_REBOOT",
    }
    (tmp_path / APPROVAL_FILENAME).write_text(json.dumps(plan), encoding="utf-8")
    (tmp_path / STATE_FILENAME).write_text(
        json.dumps(
            {
                "schemaVersion": SCHEMA_VERSION,
                "status": "PREPARED",
                "serial": "never-connected",
                "planSha256": plan_digest(plan),
                "progress": {},
                "pendingAction": None,
            }
        ),
        encoding="utf-8",
    )
    runner = Runner(Path(__file__).resolve().parents[2], tmp_path, "never-connected")
    with pytest.raises(AcceptanceError, match="approval digest"):
        runner.execute("0" * 64, None)


def test_runner_rejects_pre_single_tap_plan_schema(tmp_path: Path) -> None:
    (tmp_path / APPROVAL_FILENAME).write_text(
        json.dumps({"schemaVersion": SCHEMA_VERSION - 1}),
        encoding="utf-8",
    )
    runner = Runner(Path(__file__).resolve().parents[2], tmp_path, "never-connected")
    with pytest.raises(AcceptanceError, match="unsupported schema"):
        runner.load_plan()


def test_runner_source_has_no_forbidden_device_backdoor() -> None:
    source = Path(__file__).with_name("run_v2_widget_device_acceptance.py").read_text(
        encoding="utf-8"
    )
    assert '"force-stop"' not in source
    assert '"install"' not in source
    assert '"clear"' not in source
    assert '"uninstall"' not in source
    assert '"pm", "grant"' not in source
    assert '"content", "insert"' not in source
    assert '"content", "update"' not in source
    assert '"content", "delete"' not in source
    assert f'launch_package(V1_PACKAGE)' not in source


def test_visible_widget_status_is_typed_not_persisted_message() -> None:
    assert visible_status_subtitle("BUSY") == "Radio already starting"
    assert visible_status_subtitle("WAITING_FOR_INDEXING") == "Waiting for indexing"
    assert "0.4123" not in visible_status_subtitle("FAILED")
    with pytest.raises(AcceptanceError, match="unsupported widget request state"):
        visible_status_subtitle("INTERNAL score=0.4123")

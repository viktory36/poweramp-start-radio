from __future__ import annotations

import copy
import json
import stat
from pathlib import Path

import pytest

import v2_blind_listening_packet as packet


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


def track(identity: int, rank: int, title: str) -> dict[str, object]:
    return {
        "trackId": identity,
        "rank": rank,
        "seedRank": rank + 10,
        "filePath": f"C:\\Music\\{title}.flac",
        "title": title,
        "artist": "Hidden Artist",
        "album": "Hidden Album",
        "durationMs": 180_000,
        "activeInCurrentLibrary": True,
    }


def selector_report() -> dict[str, object]:
    runs = []
    for case_id, mode, offset in (("closest", "CLOSEST", 0), ("dpp", "DPP", 10)):
        tracks = [
            track(offset + 1, 1, f"secret-{case_id}-one"),
            track(offset + 2, 2, f"secret-{case_id}-two"),
        ]
        for repeat in (1, 2):
            runs.append(
                {
                    "caseId": case_id,
                    "repeat": repeat,
                    "seedTrackId": 99,
                    "seed": {
                        "id": 99,
                        "artist": "Seed Artist",
                        "title": "Seed Title",
                        "filePath": "/seed.flac",
                    },
                    "config": {"selectionMode": mode, "numTracks": 2},
                    "resultFingerprint": ("1" if case_id == "closest" else "2") * 64,
                    "resolvedCount": 2,
                    "inactiveResultCount": 0,
                    "tracks": copy.deepcopy(tracks),
                }
            )
    return {
        "schemaVersion": 1,
        "state": "COMPLETE",
        "runId": "selector-run",
        "queueMutationApisCalled": 0,
        "generation": {
            "generationId": "generation-one",
            "activationBindingId": "activation-one",
            "databaseContentSha256": DIGEST_A,
            "orderedTrackSetSha256": DIGEST_B,
        },
        "providerSnapshot": {"generationId": "provider-one"},
        "activeCatalog": {"activeTrackCount": 100},
        "request": {"repeatCount": 2},
        "plannedSelectionRunCount": 4,
        "selectionRuns": runs,
    }


def composed_report() -> dict[str, object]:
    runs = []
    for repeat in (1, 2):
        tracks = []
        for position, identity in enumerate((21, 22), start=1):
            tracks.append(
                {
                    "embeddedTrackId": identity,
                    "displayedPosition": position,
                    "filePath": f"/storage/music/find-{identity}.flac",
                    "title": f"find-{identity}",
                    "artist": "Find Artist",
                    "album": "Find Album",
                    "durationMs": 200_000,
                    "overallEvidence": f"#{position} of 99",
                    "ingredientEvidence": [],
                }
            )
        runs.append(
            {
                "caseId": "find-a",
                "repeat": repeat,
                "resultFingerprint": "3" * 64,
                "libraryGenerationId": "generation-one",
                "providerGenerationId": "provider-one",
                "activationBindingId": "activation-one",
                "activeTrackCount": 100,
                "publishedQuerySpec": {
                    "resultLimit": 2,
                    "libraryBinding": {
                        "generationId": "generation-one",
                        "activationBindingId": "activation-one",
                        "databaseContentSha256": DIGEST_A,
                        "orderedTrackSetSha256": DIGEST_B,
                    },
                },
                "confirmedRecording": None,
                "stableReductionRequestedVisibleCount": 2,
                "tracks": tracks,
            }
        )
    return {
        "schemaVersion": 1,
        "state": "COMPLETE",
        "runId": "find-run",
        "queueMutationApisCalled": 0,
        "settingsRestoredExactly": True,
        "databaseInfo": {
            "generationId": "generation-one",
            "providerGenerationId": "provider-one",
        },
        "request": {"repeatCount": 2},
        "repeatDeterminism": {"find-a": {"deterministic": True, "observationCount": 2}},
        "runs": runs,
    }


def simple_find_music_report() -> dict[str, object]:
    query = "ambient"
    result_count = 2
    runs = []
    determinism = {}
    for planner, wire_name, offset in (
        ("CLOSEST", "closest", 0),
        ("VARIED_DPP", "varied_dpp", 10),
    ):
        case_id = f"{query}|{result_count}|{wire_name}"
        determinism[case_id] = {
            "observationCount": 2,
            "evaluated": True,
            "deterministic": True,
        }
        objective_ranks = [1, 2] if planner == "CLOSEST" else [1, 17]
        tracks = [
            {
                "displayedPosition": position,
                "embeddedTrackId": offset + position,
                "objectiveRank": objective_ranks[position - 1],
                "filePath": f"C:\\Music\\simple-{planner}-{position}.flac",
                "title": f"simple-{planner}-{position}",
                "artist": "Simple Artist",
                "album": "Simple Album",
                "durationMs": 210_000,
                "textSimilarity": 0.412345,
                "rankingScore": 0.398765,
            }
            for position in (1, 2)
        ]
        text_plan = None
        if planner == "VARIED_DPP":
            text_plan = {
                "planner": planner,
                "plannerVersion": 1,
                "requestedResultCount": result_count,
                "completeCandidateDomainCount": 100,
                "completeTextRankingSha256": DIGEST_D,
                "orderedSelectedTrackIds": [track["embeddedTrackId"] for track in tracks],
                "orderedOriginalTextObjectiveRanks": objective_ranks,
                "dppSelection": {"selectedMarginalGains": [0.2, 0.1]},
            }
        for repeat in (1, 2):
            runs.append(
                {
                    "query": query,
                    "publishedQuery": query,
                    "publishedQuerySpec": {
                        "operator": "ALL_OF",
                        "resultLimit": result_count,
                        "textResultPlanner": planner,
                        "songSeeds": [],
                        "textIngredients": [
                            {"query": query, "weight": 1.0, "negative": False}
                        ],
                        "libraryBinding": {
                            "generationId": "generation-one",
                            "activationBindingId": "activation-one",
                            "databaseContentSha256": DIGEST_A,
                            "orderedTrackSetSha256": DIGEST_B,
                        },
                    },
                    "resultCount": result_count,
                    "planner": planner,
                    "plannerVersion": 1,
                    "repeat": repeat,
                    "resultFingerprint": ("5" if planner == "CLOSEST" else "6") * 64,
                    "resultKind": "TEXT",
                    "error": None,
                    "libraryGenerationId": "generation-one",
                    "activationBindingId": "activation-one",
                    "providerGenerationId": "provider-one",
                    "orderedActiveTrackIdsSha256": DIGEST_C,
                    "activeTrackCount": 100,
                    "objectiveRankingDomainCount": 100,
                    "stableResultReduction": {
                        "requestedVisibleCount": result_count,
                        "scannedRowCount": result_count,
                        "collapsedEquivalentCount": 0,
                    },
                    "textQueuePlan": copy.deepcopy(text_plan),
                    "queueEligibility": {"eligible": True},
                    "tracks": copy.deepcopy(tracks),
                }
            )
    return {
        "schemaVersion": 2,
        "state": "COMPLETE",
        "runId": "simple-find-run",
        "queueMutationApisCalled": 0,
        "settingsRestoredExactly": True,
        "fatalError": None,
        "request": {
            "runId": "simple-find-run",
            "queries": [query],
            "resultCounts": [result_count],
            "repeatCount": 2,
        },
        "databaseInfo": {
            "generationId": "generation-one",
            "providerGenerationId": "provider-one",
            "activeTrackCount": 100,
        },
        "searchRuns": runs,
        "repeatDeterminism": determinism,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_plan(path: Path) -> None:
    write_json(
        path,
        {
            "schema": packet.PLAN_SCHEMA,
            "study_id": "focused-test",
            "trials": [
                {
                    "id": "same-seed-selector-comparison",
                    "prompt": "Music near the known seed",
                    "candidate_1": {
                        "report": "selector",
                        "case_id": "closest",
                        "seed_track_id": 99,
                    },
                    "candidate_2": {
                        "report": "selector",
                        "case_id": "dpp",
                        "seed_track_id": 99,
                    },
                }
            ],
        },
    )


def test_deterministic_packet_conceals_methods_and_audio_sources(tmp_path: Path) -> None:
    report_path = tmp_path / "selector.json"
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "packet"
    write_json(report_path, selector_report())
    write_plan(plan_path)
    reports = packet.validate_reports({"selector": report_path})

    first_public, first_reveal = packet.build_packet(
        reports=reports,
        plan_path=plan_path,
        blind_key=b"0123456789abcdef0123456789abcdef",
    )
    second_public, second_reveal = packet.build_packet(
        reports=reports,
        plan_path=plan_path,
        blind_key=b"0123456789abcdef0123456789abcdef",
    )
    assert first_public == second_public
    assert first_reveal == second_reveal

    public_text = json.dumps(first_public).lower()
    for hidden in (
        "closest",
        "dpp",
        "selectionmode",
        "hidden artist",
        "secret-closest-one",
        "c:\\\\backups",
        "seedrank",
    ):
        assert hidden not in public_text
    assert "audio-" in public_text
    assert first_public["trials"][0]["sides"][0]["trackCount"] == 2

    reveal_text = json.dumps(first_reveal).lower()
    assert "closest" in reveal_text
    assert "dpp" in reveal_text
    assert "secret-closest-one" in reveal_text
    assert "distancefromseedrank" in reveal_text

    written = packet.write_packet(output, first_public, first_reveal)
    assert Path(written["publicPath"]).is_file()
    reveal_path = Path(written["revealPath"])
    assert stat.S_IMODE(reveal_path.stat().st_mode) == 0o600
    packet.write_packet(output, second_public, second_reveal)


def test_cli_dry_run_builds_in_memory_without_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "selector.json"
    plan_path = tmp_path / "plan.json"
    key_path = tmp_path / "blind.key"
    write_json(report_path, selector_report())
    write_plan(plan_path)
    key_path.write_bytes(b"dry-run-key-0123456789abcdef")

    result = packet.main(
        [
            "--report",
            f"selector={report_path}",
            "--plan",
            str(plan_path),
            "--blind-key-file",
            str(key_path),
            "--dry-run",
        ]
    )

    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["state"] == "DRY_RUN_VALID"
    assert summary["trialCount"] == 1
    assert list(tmp_path.glob("blind-manifest.json")) == []
    assert list(tmp_path.glob("reveal-key.json")) == []


def test_simple_find_music_closest_vs_varied_builds_blind_packet(tmp_path: Path) -> None:
    report_path = tmp_path / "simple-find.json"
    plan_path = tmp_path / "simple-plan.json"
    write_json(report_path, simple_find_music_report())
    write_json(
        plan_path,
        {
            "schema": packet.PLAN_SCHEMA,
            "study_id": "simple-find-test",
            "trials": [
                {
                    "id": "ambient-closest-vs-varied",
                    "prompt": "A queue that sounds ambient",
                    "candidate_1": {
                        "report": "find_music_simple",
                        "case_id": "ambient|2|closest",
                    },
                    "candidate_2": {
                        "report": "find_music_simple",
                        "case_id": "ambient|2|varied_dpp",
                    },
                }
            ],
        },
    )

    reports = packet.validate_reports({"find_music_simple": report_path})
    assert reports["find_music_simple"].report_kind == packet.SIMPLE_FIND_MUSIC_REPORT_KIND
    assert set(reports["find_music_simple"].cases) == {
        ("ambient|2|closest", None),
        ("ambient|2|varied_dpp", None),
    }
    public, reveal = packet.build_packet(
        reports=reports,
        plan_path=plan_path,
        blind_key=b"simple-find-blind-key-0123456789",
    )

    public_text = json.dumps(public).lower()
    assert "closest" not in public_text
    assert "varied" not in public_text
    reveal_text = json.dumps(reveal).lower()
    assert "ambient|2|closest" in reveal_text
    assert "ambient|2|varied_dpp" in reveal_text
    ranks = [
        track["rankEvidence"]
        for side in reveal["trials"][0]["sides"].values()
        for track in side["tracks"]
    ]
    assert {rank["rankedTrackCount"] for rank in ranks} == {100}
    assert {rank["textMatchRank"] for rank in ranks} >= {1, 2, 17}
    assert "rankingscore" not in reveal_text
    assert "textsimilarity" not in reveal_text


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report.__setitem__("queueMutationApisCalled", 1),
            "queue-mutation",
        ),
        (
            lambda report: report["searchRuns"][0]["tracks"].pop(),
            "has 1 results; expected 2",
        ),
        (
            lambda report: report["searchRuns"][1].__setitem__(
                "resultFingerprint", "7" * 64
            ),
            "fingerprints are nondeterministic",
        ),
        (
            lambda report: report["searchRuns"][1]["publishedQuerySpec"][
                "libraryBinding"
            ].__setitem__("activationBindingId", "activation-two"),
            "activation ID disagrees with its published binding",
        ),
        (
            lambda report: report["repeatDeterminism"][
                "ambient|2|closest"
            ].__setitem__("deterministic", False),
            "not proven deterministic",
        ),
        (
            lambda report: report["searchRuns"].pop(),
            "contains 3 search runs; expected 4",
        ),
        (
            lambda report: report["searchRuns"][3]["textQueuePlan"][
                "dppSelection"
            ]["selectedMarginalGains"].__setitem__(1, 0.05),
            "changed result content",
        ),
    ],
)
def test_simple_find_music_reports_fail_closed(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    report = simple_find_music_report()
    mutate(report)
    path = tmp_path / "bad-simple-find.json"
    write_json(path, report)
    with pytest.raises(packet.PacketError, match=message):
        packet.validate_reports({"find_music_simple": path})


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("generation", "generationId"),
        ("provider", "providerGenerationId"),
        ("activation", "activationBindingId"),
    ],
)
def test_mixed_report_bindings_fail_closed(tmp_path: Path, mutation: str, message: str) -> None:
    selector = selector_report()
    composed = composed_report()
    if mutation == "generation":
        composed["databaseInfo"]["generationId"] = "generation-two"
        for run in composed["runs"]:
            run["libraryGenerationId"] = "generation-two"
            run["publishedQuerySpec"]["libraryBinding"]["generationId"] = "generation-two"
    elif mutation == "provider":
        composed["databaseInfo"]["providerGenerationId"] = "provider-two"
        for run in composed["runs"]:
            run["providerGenerationId"] = "provider-two"
    else:
        for run in composed["runs"]:
            run["activationBindingId"] = "activation-two"
            run["publishedQuerySpec"]["libraryBinding"]["activationBindingId"] = "activation-two"
    selector_path = tmp_path / "selector.json"
    composed_path = tmp_path / "composed.json"
    write_json(selector_path, selector)
    write_json(composed_path, composed)

    with pytest.raises(packet.PacketError, match=message):
        packet.validate_reports({"selector": selector_path, "find_music": composed_path})


def test_queue_mutation_result_count_and_nondeterminism_fail_closed(
    tmp_path: Path,
) -> None:
    base = selector_report()
    mutations = []

    queue_mutation = copy.deepcopy(base)
    queue_mutation["queueMutationApisCalled"] = 1
    mutations.append((queue_mutation, "queue-mutation"))

    short_result = copy.deepcopy(base)
    short_result["selectionRuns"][0]["tracks"].pop()
    mutations.append((short_result, "has 1 results; expected 2"))

    changed_fingerprint = copy.deepcopy(base)
    changed_fingerprint["selectionRuns"][1]["resultFingerprint"] = "4" * 64
    mutations.append((changed_fingerprint, "fingerprints are nondeterministic"))

    changed_content = copy.deepcopy(base)
    changed_content["selectionRuns"][1]["tracks"][0]["title"] = "different"
    mutations.append((changed_content, "changed result content"))

    for index, (report, message) in enumerate(mutations):
        path = tmp_path / f"bad-{index}.json"
        write_json(path, report)
        with pytest.raises(packet.PacketError, match=message):
            packet.validate_reports({"selector": path})

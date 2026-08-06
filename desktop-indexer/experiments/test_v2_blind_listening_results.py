from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import v2_blind_listening_results as results


def write_json(path: Path, value: object, mode: int = 0o644) -> None:
    path.write_bytes(results._pretty_json(value))
    os.chmod(path, mode)


def fixture_artifacts(tmp_path: Path):
    candidate_x = {
        "reportAlias": "selectors",
        "reportKind": "selector",
        "caseId": "candidate-x",
        "seedTrackId": 7,
        "resultFingerprint": "a" * 64,
    }
    candidate_y = {
        "reportAlias": "selectors",
        "reportKind": "selector",
        "caseId": "candidate-y",
        "seedTrackId": 7,
        "resultFingerprint": "b" * 64,
    }
    reveal_core = {
        "schema": results.REVEAL_SCHEMA,
        "packetId": "blind-v2-test",
        "studyId": "study-test",
        "blindKeySha256": "c" * 64,
        "plan": {"path": "/private/plan", "sha256": "d" * 64},
        "reports": {},
        "trials": [
            {
                "trial": "T001",
                "planTrialId": "original",
                "prompt": "Choose the better queue.",
                "sides": {
                    "A": {
                        **candidate_x,
                        "queueToken": "queue-a1",
                        "tracks": [
                            {"source": {"reportedPath": "/private/secret-x.flac"}}
                        ],
                    },
                    "B": {**candidate_y, "queueToken": "queue-b1", "tracks": []},
                },
            },
            {
                "trial": "T002",
                "planTrialId": "repeat",
                "prompt": "Choose the better queue.",
                "sides": {
                    "A": {**candidate_y, "queueToken": "queue-a2", "tracks": []},
                    "B": {**candidate_x, "queueToken": "queue-b2", "tracks": []},
                },
            },
        ],
    }
    commitment = results._sha256(results._canonical_json(reveal_core))
    manifest_value = {
        "schema": results.PUBLIC_SCHEMA,
        "packetId": "blind-v2-test",
        "studyId": "study-test",
        "sourceBindingSha256": "e" * 64,
        "revealCommitmentSha256": commitment,
        "protocol": {},
        "trials": [
            {
                "trial": "T001",
                "prompt": "Choose the better queue.",
                "question": "Which?",
                "response": {},
                "sides": [
                    {"label": "A", "queueToken": "queue-a1", "trackCount": 2, "tracks": []},
                    {"label": "B", "queueToken": "queue-b1", "trackCount": 2, "tracks": []},
                ],
            },
            {
                "trial": "T002",
                "prompt": "Choose the better queue.",
                "question": "Which?",
                "response": {},
                "sides": [
                    {"label": "A", "queueToken": "queue-a2", "trackCount": 2, "tracks": []},
                    {"label": "B", "queueToken": "queue-b2", "trackCount": 2, "tracks": []},
                ],
            },
        ],
    }
    manifest_path = tmp_path / "blind-manifest.json"
    write_json(manifest_path, manifest_value)
    reveal_value = {
        **reveal_core,
        "revealCoreSha256": commitment,
        "publicManifestSha256": results._sha256(manifest_path.read_bytes()),
    }
    reveal_path = tmp_path / "reveal-key.json"
    write_json(reveal_path, reveal_value, 0o600)
    response_value = {
        "schema": results.RESPONSE_SCHEMA,
        "packetId": "blind-v2-test",
        "studyId": "study-test",
        "sourceBindingSha256": "e" * 64,
        "revealCommitmentSha256": commitment,
        "responses": [
            {
                "trial": "T001",
                "preference": "A",
                "intentFit": {"A": 5, "B": 3},
                "coherence": {"A": 4, "B": 3},
                "discoveryValue": {"A": 5, "B": 4},
                "notes": "first",
            },
            {
                "trial": "T002",
                "preference": "B",
                "intentFit": {"A": 3, "B": 5},
                "coherence": {"A": 2, "B": 4},
                "discoveryValue": {"A": 4, "B": 5},
                "notes": "repeat",
            },
        ],
        "listenerState": {
            "trialIndex": 1,
            "trials": {
                "T001": {
                    "A": {"selected": 1, "reviewed": [1, 2]},
                    "B": {"selected": 0, "reviewed": [1]},
                }
            },
        },
    }
    response_path = tmp_path / "responses.json"
    write_json(response_path, response_value)
    return manifest_path, reveal_path, response_path, response_value


def test_complete_response_validation_is_strict_and_blinded(tmp_path: Path) -> None:
    manifest_path, _, response_path, _ = fixture_artifacts(tmp_path)
    manifest = results.load_public_manifest(manifest_path)
    responses = results.validate_responses(response_path, manifest)
    summary = results.validation_summary(manifest, responses)
    assert summary["state"] == "COMPLETE_VALID"
    assert summary["fullyRatedTrialCount"] == 2
    assert summary["reviewedPresentationCount"] == 3
    assert "candidate-x" not in json.dumps(summary)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("incomplete", "integer from 1 to 5"),
        ("binding", "does not match"),
        ("order", "manifest order"),
        ("duplicate_review", "repeats positions"),
        ("extra", "unsupported fields"),
    ],
)
def test_response_validation_fails_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest_path, _, response_path, value = fixture_artifacts(tmp_path)
    if mutation == "incomplete":
        value["responses"][0]["intentFit"]["A"] = None
    elif mutation == "binding":
        value["sourceBindingSha256"] = "f" * 64
    elif mutation == "order":
        value["responses"].reverse()
    elif mutation == "duplicate_review":
        value["listenerState"]["trials"]["T001"]["A"]["reviewed"] = [1, 1]
    else:
        value["responses"][0]["score"] = 99
    write_json(response_path, value)
    manifest = results.load_public_manifest(manifest_path)
    with pytest.raises(results.ResultsError, match=message):
        results.validate_responses(response_path, manifest)


def test_reveal_analysis_normalizes_swapped_sides_and_omits_sources(tmp_path: Path) -> None:
    manifest_path, reveal_path, response_path, _ = fixture_artifacts(tmp_path)
    manifest = results.load_public_manifest(manifest_path)
    responses = results.validate_responses(response_path, manifest)
    reveal = results._load_committed_reveal(reveal_path, manifest)
    analysis = results.analyze_revealed(manifest, responses, reveal)
    assert analysis["state"] == "REVEALED_COMPLETE"
    assert analysis["hiddenRepeatAgreement"]["pairCount"] == 1
    assert analysis["hiddenRepeatAgreement"]["preferenceAgreementCount"] == 1
    assert analysis["sideBias"]["displayedPreferenceCounts"] == {"A": 1, "B": 1, "tie": 0}
    candidates = {row["caseId"]: row for row in analysis["candidates"]}
    assert candidates["candidate-x"]["preferenceWins"] == 2
    assert candidates["candidate-x"]["meanRatings"] == {
        "intentFit": 5.0,
        "coherence": 4.0,
        "discoveryValue": 5.0,
    }
    rendered = json.dumps(analysis)
    assert "reportedPath" not in rendered
    assert "/private/" not in rendered


def test_reveal_rejects_wrong_permissions_and_commitment(tmp_path: Path) -> None:
    manifest_path, reveal_path, response_path, _ = fixture_artifacts(tmp_path)
    manifest = results.load_public_manifest(manifest_path)
    results.validate_responses(response_path, manifest)
    if os.name == "posix":
        os.chmod(reveal_path, 0o644)
        with pytest.raises(results.ResultsError, match="group/world"):
            results._load_committed_reveal(reveal_path, manifest)
        os.chmod(reveal_path, 0o600)
    value = json.loads(reveal_path.read_text())
    value["studyId"] = "other"
    write_json(reveal_path, value, 0o600)
    with pytest.raises(results.ResultsError, match="does not match"):
        results._load_committed_reveal(reveal_path, manifest)


def test_cli_validates_then_reveals_only_complete_responses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path, reveal_path, response_path, value = fixture_artifacts(tmp_path)
    assert results.main(
        ["validate", "--manifest", str(manifest_path), "--responses", str(response_path)]
    ) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "COMPLETE_VALID"

    output = tmp_path / "analysis.json"
    assert results.main(
        [
            "reveal",
            "--manifest",
            str(manifest_path),
            "--reveal",
            str(reveal_path),
            "--responses",
            str(response_path),
            "--output",
            str(output),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "REVEALED_COMPLETE"
    assert json.loads(output.read_text())["hiddenRepeatAgreement"]["preferenceAgreementCount"] == 1

    value["responses"][0]["coherence"]["B"] = None
    write_json(response_path, value)
    assert results.main(
        [
            "reveal",
            "--manifest",
            str(manifest_path),
            "--reveal",
            str(tmp_path / "does-not-exist.json"),
            "--responses",
            str(response_path),
        ]
    ) == 2
    captured = capsys.readouterr()
    assert "integer from 1 to 5" in captured.err
    assert "does-not-exist" not in captured.err

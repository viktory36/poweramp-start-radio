#!/usr/bin/env python3
"""Validate completed blind-listening responses and analyze them after reveal.

The ``validate`` command reads only the public manifest. The ``reveal`` command
first applies the same completion gate, then verifies the committed reveal and
maps judgments back to candidate identities. Neither command emits audio paths
or track metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PUBLIC_SCHEMA = "poweramp-start-radio-blind-listening-packet-v1"
REVEAL_SCHEMA = "poweramp-start-radio-blind-listening-reveal-v1"
RESPONSE_SCHEMA = "poweramp-start-radio-blind-listening-responses-v2"
PREFERENCES = {"A", "B", "tie"}
METRICS = ("intentFit", "coherence", "discoveryValue")
SIDES = ("A", "B")


class ResultsError(ValueError):
    """A response, manifest, or reveal artifact failed closed."""


def _reject_json_constant(value: str) -> None:
    raise ResultsError(f"non-finite JSON constant is not allowed: {value}")


def _read_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, parse_constant=_reject_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultsError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResultsError(f"{path} must contain a JSON object")
    return value, raw


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResultsError(f"{where} must be an object")
    return value


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ResultsError(f"{where} must be an array")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResultsError(f"{where} must be non-empty text")
    return value


def _sha256_text(value: Any, where: str) -> str:
    text = _text(value, where)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ResultsError(f"{where} must be a lowercase SHA-256 digest")
    return text


def _strict_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    where: str,
) -> None:
    keys = set(value)
    missing = required - keys
    extra = keys - required - optional
    if missing:
        raise ResultsError(f"{where} is missing: {', '.join(sorted(missing))}")
    if extra:
        raise ResultsError(f"{where} has unsupported fields: {', '.join(sorted(extra))}")


@dataclass(frozen=True)
class PublicTrial:
    trial_id: str
    prompt: str
    queue_tokens: Mapping[str, str]
    track_counts: Mapping[str, int]


@dataclass(frozen=True)
class PublicManifest:
    value: Mapping[str, Any]
    raw: bytes
    packet_id: str
    study_id: str
    source_binding_sha256: str
    reveal_commitment_sha256: str
    trials: tuple[PublicTrial, ...]


@dataclass(frozen=True)
class ValidatedResponses:
    value: Mapping[str, Any]
    raw: bytes
    by_trial: Mapping[str, Mapping[str, Any]]


def load_public_manifest(path: Path) -> PublicManifest:
    value, raw = _read_json_object(path)
    if value.get("schema") != PUBLIC_SCHEMA:
        raise ResultsError(f"unsupported public manifest schema: {value.get('schema')!r}")
    packet_id = _text(value.get("packetId"), "manifest.packetId")
    study_id = _text(value.get("studyId"), "manifest.studyId")
    binding = _sha256_text(value.get("sourceBindingSha256"), "manifest.sourceBindingSha256")
    commitment = _sha256_text(
        value.get("revealCommitmentSha256"), "manifest.revealCommitmentSha256"
    )
    raw_trials = _array(value.get("trials"), "manifest.trials")
    if not raw_trials:
        raise ResultsError("manifest.trials must not be empty")
    trials: list[PublicTrial] = []
    seen: set[str] = set()
    for index, raw_trial in enumerate(raw_trials):
        trial = _object(raw_trial, f"manifest.trials[{index}]")
        trial_id = _text(trial.get("trial"), f"manifest.trials[{index}].trial")
        if trial_id in seen:
            raise ResultsError(f"manifest repeats trial {trial_id}")
        seen.add(trial_id)
        prompt = _text(trial.get("prompt"), f"manifest.trials[{index}].prompt")
        sides = _array(trial.get("sides"), f"manifest.trials[{index}].sides")
        queue_tokens: dict[str, str] = {}
        track_counts: dict[str, int] = {}
        for side_index, raw_side in enumerate(sides):
            side = _object(raw_side, f"manifest.trials[{index}].sides[{side_index}]")
            label = _text(side.get("label"), "manifest side label")
            if label not in SIDES or label in queue_tokens:
                raise ResultsError(f"{trial_id} must contain exactly one A and one B side")
            queue_tokens[label] = _text(side.get("queueToken"), f"{trial_id}.{label}.queueToken")
            track_count = side.get("trackCount")
            if type(track_count) is not int or track_count < 1:
                raise ResultsError(f"{trial_id}.{label}.trackCount must be positive")
            track_counts[label] = track_count
        if set(queue_tokens) != set(SIDES):
            raise ResultsError(f"{trial_id} must contain exactly one A and one B side")
        trials.append(PublicTrial(trial_id, prompt, queue_tokens, track_counts))
    return PublicManifest(value, raw, packet_id, study_id, binding, commitment, tuple(trials))


def _rating(value: Any, where: str) -> int:
    if type(value) is not int or not 1 <= value <= 5:
        raise ResultsError(f"{where} must be an integer from 1 to 5")
    return value


def _validate_listener_state(value: Any, manifest: PublicManifest) -> dict[str, Any]:
    state = _object(value, "responses.listenerState")
    _strict_keys(state, required={"trialIndex", "trials"}, where="responses.listenerState")
    trial_index = state["trialIndex"]
    if type(trial_index) is not int or not 0 <= trial_index < len(manifest.trials):
        raise ResultsError("responses.listenerState.trialIndex is outside the packet")
    trials = _object(state["trials"], "responses.listenerState.trials")
    manifest_trials = {trial.trial_id: trial for trial in manifest.trials}
    unknown = set(trials) - set(manifest_trials)
    if unknown:
        raise ResultsError("responses.listenerState names unknown trials")
    normalized: dict[str, Any] = {"trialIndex": trial_index, "trials": {}}
    for trial_id, raw_trial_state in trials.items():
        trial_state = _object(raw_trial_state, f"listenerState.{trial_id}")
        public_trial = manifest_trials[trial_id]
        _strict_keys(trial_state, required=set(SIDES), where=f"listenerState.{trial_id}")
        normalized_sides: dict[str, Any] = {}
        for side in SIDES:
            side_state = _object(trial_state[side], f"listenerState.{trial_id}.{side}")
            _strict_keys(
                side_state,
                required={"selected", "reviewed"},
                where=f"listenerState.{trial_id}.{side}",
            )
            selected = side_state["selected"]
            track_count = public_trial.track_counts[side]
            if type(selected) is not int or not 0 <= selected < track_count:
                raise ResultsError(f"listenerState.{trial_id}.{side}.selected is invalid")
            reviewed = _array(side_state["reviewed"], f"listenerState.{trial_id}.{side}.reviewed")
            if any(
                type(position) is not int or not 1 <= position <= track_count
                for position in reviewed
            ):
                raise ResultsError(f"listenerState.{trial_id}.{side}.reviewed is invalid")
            if len(set(reviewed)) != len(reviewed):
                raise ResultsError(f"listenerState.{trial_id}.{side}.reviewed repeats positions")
            normalized_sides[side] = {"selected": selected, "reviewed": sorted(reviewed)}
        normalized["trials"][trial_id] = normalized_sides
    return normalized


def validate_responses(path: Path, manifest: PublicManifest) -> ValidatedResponses:
    value, raw = _read_json_object(path)
    _strict_keys(
        value,
        required={
            "schema",
            "packetId",
            "studyId",
            "sourceBindingSha256",
            "revealCommitmentSha256",
            "responses",
            "listenerState",
        },
        where="responses",
    )
    if value.get("schema") != RESPONSE_SCHEMA:
        raise ResultsError(f"unsupported response schema: {value.get('schema')!r}")
    bindings = {
        "packetId": manifest.packet_id,
        "studyId": manifest.study_id,
        "sourceBindingSha256": manifest.source_binding_sha256,
        "revealCommitmentSha256": manifest.reveal_commitment_sha256,
    }
    for field, expected in bindings.items():
        if value.get(field) != expected:
            raise ResultsError(f"responses.{field} does not match the public manifest")
    raw_responses = _array(value["responses"], "responses.responses")
    expected_ids = [trial.trial_id for trial in manifest.trials]
    if len(raw_responses) != len(expected_ids):
        raise ResultsError(
            f"responses contains {len(raw_responses)} trials; expected {len(expected_ids)}"
        )
    by_trial: dict[str, Mapping[str, Any]] = {}
    observed_ids: list[str] = []
    for index, raw_response in enumerate(raw_responses):
        response = _object(raw_response, f"responses.responses[{index}]")
        _strict_keys(
            response,
            required={"trial", "preference", *METRICS, "notes"},
            where=f"responses.responses[{index}]",
        )
        trial_id = _text(response["trial"], f"responses.responses[{index}].trial")
        if trial_id in by_trial:
            raise ResultsError(f"responses repeats trial {trial_id}")
        observed_ids.append(trial_id)
        preference = response["preference"]
        if not isinstance(preference, str) or preference not in PREFERENCES:
            raise ResultsError(f"responses.{trial_id}.preference must be A, B, or tie")
        for metric in METRICS:
            ratings = _object(response[metric], f"responses.{trial_id}.{metric}")
            _strict_keys(ratings, required=set(SIDES), where=f"responses.{trial_id}.{metric}")
            for side in SIDES:
                _rating(ratings[side], f"responses.{trial_id}.{metric}.{side}")
        if not isinstance(response["notes"], str):
            raise ResultsError(f"responses.{trial_id}.notes must be text")
        by_trial[trial_id] = response
    if observed_ids != expected_ids:
        raise ResultsError("responses must contain every manifest trial in manifest order")
    _validate_listener_state(value["listenerState"], manifest)
    return ValidatedResponses(value, raw, by_trial)


def validation_summary(manifest: PublicManifest, responses: ValidatedResponses) -> dict[str, Any]:
    reviewed = 0
    listener_trials = responses.value["listenerState"]["trials"]
    for trial_state in listener_trials.values():
        for side in SIDES:
            reviewed += len(trial_state[side]["reviewed"])
    return {
        "state": "COMPLETE_VALID",
        "packetId": manifest.packet_id,
        "studyId": manifest.study_id,
        "sourceBindingSha256": manifest.source_binding_sha256,
        "trialCount": len(manifest.trials),
        "fullyRatedTrialCount": len(responses.by_trial),
        "reviewedPresentationCount": reviewed,
        "responsesSha256": _sha256(responses.raw),
    }


def _load_committed_reveal(path: Path, manifest: PublicManifest) -> Mapping[str, Any]:
    reveal, _ = _read_json_object(path)
    if os.name == "posix":
        try:
            if path.stat().st_mode & 0o077:
                raise ResultsError("reveal must not be group/world accessible")
        except OSError as exc:
            raise ResultsError(f"cannot inspect reveal permissions: {exc}") from exc
    if reveal.get("schema") != REVEAL_SCHEMA:
        raise ResultsError(f"unsupported reveal schema: {reveal.get('schema')!r}")
    for field, expected in (("packetId", manifest.packet_id), ("studyId", manifest.study_id)):
        if reveal.get(field) != expected:
            raise ResultsError(f"reveal.{field} does not match the public manifest")
    if reveal.get("publicManifestSha256") != _sha256(manifest.raw):
        raise ResultsError("reveal does not commit to this public manifest")
    reveal_core = {
        key: value
        for key, value in reveal.items()
        if key not in {"revealCoreSha256", "publicManifestSha256"}
    }
    core_sha256 = _sha256(_canonical_json(reveal_core))
    if reveal.get("revealCoreSha256") != core_sha256:
        raise ResultsError("reveal core digest is invalid")
    if manifest.reveal_commitment_sha256 != core_sha256:
        raise ResultsError("reveal commitment does not match the public manifest")
    return reveal


def _candidate_key(side: Mapping[str, Any]) -> str:
    identity = {
        "reportAlias": side.get("reportAlias"),
        "reportKind": side.get("reportKind"),
        "caseId": side.get("caseId"),
        "seedTrackId": side.get("seedTrackId"),
        "resultFingerprint": side.get("resultFingerprint"),
    }
    for field in ("reportAlias", "reportKind", "caseId", "resultFingerprint"):
        _text(identity[field], f"reveal candidate {field}")
    if identity["seedTrackId"] is not None and type(identity["seedTrackId"]) is not int:
        raise ResultsError("reveal candidate seedTrackId must be an integer or null")
    return _sha256(_canonical_json(identity))[:20]


def _candidate_description(side: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidateId": _candidate_key(side),
        "reportAlias": side.get("reportAlias"),
        "reportKind": side.get("reportKind"),
        "caseId": side.get("caseId"),
        "seedTrackId": side.get("seedTrackId"),
        "resultFingerprint": side.get("resultFingerprint"),
    }


def analyze_revealed(
    manifest: PublicManifest,
    responses: ValidatedResponses,
    reveal: Mapping[str, Any],
) -> dict[str, Any]:
    raw_trials = _array(reveal.get("trials"), "reveal.trials")
    reveal_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_trial in enumerate(raw_trials):
        trial = _object(raw_trial, f"reveal.trials[{index}]")
        trial_id = _text(trial.get("trial"), f"reveal.trials[{index}].trial")
        if trial_id in reveal_by_id:
            raise ResultsError(f"reveal repeats trial {trial_id}")
        reveal_by_id[trial_id] = trial
    expected_ids = [trial.trial_id for trial in manifest.trials]
    if set(reveal_by_id) != set(expected_ids):
        raise ResultsError("reveal trial set does not match the public manifest")

    candidate_descriptions: dict[str, dict[str, Any]] = {}
    candidate_ratings: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {metric: [] for metric in METRICS}
    )
    preference_wins: Counter[str] = Counter()
    preference_ties: Counter[str] = Counter()
    side_preferences: Counter[str] = Counter()
    displayed_side_ratings: dict[str, dict[str, list[int]]] = {
        side: {metric: [] for metric in METRICS} for side in SIDES
    }
    trial_rows: list[dict[str, Any]] = []
    repeat_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    public_by_id = {trial.trial_id: trial for trial in manifest.trials}
    for trial_id in expected_ids:
        public_trial = public_by_id[trial_id]
        trial = reveal_by_id[trial_id]
        sides = _object(trial.get("sides"), f"reveal.{trial_id}.sides")
        if set(sides) != set(SIDES):
            raise ResultsError(f"reveal.{trial_id} must contain A and B")
        response = responses.by_trial[trial_id]
        side_candidates: dict[str, str] = {}
        for side in SIDES:
            revealed_side = _object(sides[side], f"reveal.{trial_id}.{side}")
            if revealed_side.get("queueToken") != public_trial.queue_tokens[side]:
                raise ResultsError(f"reveal.{trial_id}.{side} queue token mismatch")
            description = _candidate_description(revealed_side)
            candidate_id = description["candidateId"]
            existing = candidate_descriptions.setdefault(candidate_id, description)
            if existing != description:
                raise ResultsError("reveal candidate identifier collision")
            side_candidates[side] = candidate_id
            for metric in METRICS:
                rating = response[metric][side]
                candidate_ratings[candidate_id][metric].append(rating)
                displayed_side_ratings[side][metric].append(rating)

        preference = response["preference"]
        normalized_preference = "tie" if preference == "tie" else side_candidates[preference]
        side_preferences[preference] += 1
        preference_wins[normalized_preference] += 1
        if preference == "tie":
            for candidate_id in side_candidates.values():
                preference_ties[candidate_id] += 1
        row = {
            "trial": trial_id,
            "planTrialId": trial.get("planTrialId"),
            "prompt": public_trial.prompt,
            "displayedCandidates": side_candidates,
            "displayedPreference": preference,
            "preferredCandidateId": normalized_preference,
            "ratings": {
                side_candidates[side]: {metric: response[metric][side] for metric in METRICS}
                for side in SIDES
            },
            "notes": response["notes"],
        }
        trial_rows.append(row)
        repeat_signature = _sha256(
            _canonical_json(
                {
                    "prompt": public_trial.prompt,
                    "candidateIds": sorted(side_candidates.values()),
                }
            )
        )
        repeat_groups[repeat_signature].append(row)

    repeat_rows: list[dict[str, Any]] = []
    for rows in repeat_groups.values():
        if len(rows) < 2:
            continue
        if len(rows) != 2:
            raise ResultsError("a repeated comparison appears more than twice")
        first, second = rows
        candidate_ids = sorted(first["ratings"])
        rating_differences = [
            abs(first["ratings"][candidate_id][metric] - second["ratings"][candidate_id][metric])
            for candidate_id in candidate_ids
            for metric in METRICS
        ]
        repeat_rows.append(
            {
                "trials": [first["trial"], second["trial"]],
                "candidateIds": candidate_ids,
                "preferenceAgreement": (
                    first["preferredCandidateId"] == second["preferredCandidateId"]
                ),
                "firstPreferredCandidateId": first["preferredCandidateId"],
                "secondPreferredCandidateId": second["preferredCandidateId"],
                "meanAbsoluteRatingDifference": round(
                    statistics.fmean(rating_differences), 6
                ),
                "maximumAbsoluteRatingDifference": max(rating_differences),
            }
        )

    candidate_rows = []
    for candidate_id in sorted(candidate_descriptions):
        ratings = candidate_ratings[candidate_id]
        appearances = len(ratings[METRICS[0]])
        wins = preference_wins[candidate_id]
        ties = preference_ties[candidate_id]
        candidate_rows.append(
            {
                **candidate_descriptions[candidate_id],
                "trialAppearances": appearances,
                "preferenceWins": wins,
                "preferenceTies": ties,
                "preferenceLosses": appearances - wins - ties,
                "meanRatings": {
                    metric: round(statistics.fmean(ratings[metric]), 6) for metric in METRICS
                },
            }
        )

    non_tie = side_preferences["A"] + side_preferences["B"]
    return {
        "schema": "poweramp-start-radio-blind-listening-analysis-v1",
        "state": "REVEALED_COMPLETE",
        "packetId": manifest.packet_id,
        "studyId": manifest.study_id,
        "sourceBindingSha256": manifest.source_binding_sha256,
        "responsesSha256": _sha256(responses.raw),
        "trialCount": len(trial_rows),
        "sideBias": {
            "displayedPreferenceCounts": {
                side: side_preferences[side] for side in (*SIDES, "tie")
            },
            "nonTieAPreferenceFraction": (
                None if non_tie == 0 else round(side_preferences["A"] / non_tie, 6)
            ),
            "meanRatingsByDisplayedSide": {
                side: {
                    metric: round(statistics.fmean(displayed_side_ratings[side][metric]), 6)
                    for metric in METRICS
                }
                for side in SIDES
            },
        },
        "hiddenRepeatAgreement": {
            "pairCount": len(repeat_rows),
            "preferenceAgreementCount": sum(row["preferenceAgreement"] for row in repeat_rows),
            "pairs": repeat_rows,
        },
        "candidates": candidate_rows,
        "trials": trial_rows,
    }


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate complete blinded responses")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--responses", type=Path, required=True)
    reveal = subparsers.add_parser(
        "reveal", help="validate complete responses, verify reveal, and analyze candidates"
    )
    reveal.add_argument("--manifest", type=Path, required=True)
    reveal.add_argument("--reveal", type=Path, required=True)
    reveal.add_argument("--responses", type=Path, required=True)
    reveal.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_public_manifest(args.manifest)
        responses = validate_responses(args.responses, manifest)
        if args.command == "validate":
            result = validation_summary(manifest, responses)
        else:
            reveal = _load_committed_reveal(args.reveal, manifest)
            result = analyze_revealed(manifest, responses, reveal)
            if args.output is not None:
                _atomic_write(args.output, _pretty_json(result))
        print(_pretty_json(result).decode("utf-8"), end="")
        return 0
    except (ResultsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

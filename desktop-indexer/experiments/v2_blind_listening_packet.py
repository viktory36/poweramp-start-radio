#!/usr/bin/env python3
"""Build deterministic blind A/B packets from production device evidence.

The builder is host-only. It reads completed acceptance reports and never invokes
ADB, Poweramp, or an Android queue API. Public packets contain opaque audio tokens;
the separate reveal key retains enough source information for a later full-track
resolver.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PACKET_SCHEMA = "poweramp-start-radio-blind-listening-packet-v1"
REVEAL_SCHEMA = "poweramp-start-radio-blind-listening-reveal-v1"
PLAN_SCHEMA = "poweramp-start-radio-blind-listening-plan-v1"
SIMPLE_FIND_MUSIC_REPORT_KIND = "find_music_simple"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PacketError(ValueError):
    """A fail-closed validation or packet-construction error."""


def _reject_json_constant(value: str) -> None:
    raise PacketError(f"non-finite JSON number is forbidden: {value}")


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PacketError(f"value cannot be serialized canonically: {exc}") from exc


def _pretty_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PacketError(f"value cannot be serialized: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PacketError(f"cannot read {path}: {exc}") from exc
    try:
        value = json.loads(raw, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PacketError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PacketError(f"{path}: top-level JSON value must be an object")
    return value, _sha256_bytes(raw)


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PacketError(f"{where} must be an object")
    return value


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise PacketError(f"{where} must be an array")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PacketError(f"{where} must be a non-empty string")
    return value


def _integer(value: Any, where: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise PacketError(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise PacketError(f"{where} must be at least {minimum}")
    return value


def _sha256(value: Any, where: str) -> str:
    text = _string(value, where)
    if SHA256_RE.fullmatch(text) is None:
        raise PacketError(f"{where} must be a lowercase SHA-256 digest")
    return text


def _field(obj: Mapping[str, Any], name: str, where: str) -> Any:
    if name not in obj:
        raise PacketError(f"{where} is missing required field {name!r}")
    return obj[name]


def _strict_keys(
    obj: Mapping[str, Any],
    allowed: set[str],
    required: set[str],
    where: str,
) -> None:
    missing = sorted(required - obj.keys())
    unknown = sorted(obj.keys() - allowed)
    if missing:
        raise PacketError(f"{where} is missing fields: {', '.join(missing)}")
    if unknown:
        raise PacketError(f"{where} has unknown fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class LibraryBinding:
    generation_id: str
    provider_generation_id: str
    activation_binding_id: str
    database_content_sha256: str
    ordered_track_set_sha256: str
    active_track_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "generationId": self.generation_id,
            "providerGenerationId": self.provider_generation_id,
            "activationBindingId": self.activation_binding_id,
            "databaseContentSha256": self.database_content_sha256,
            "orderedTrackSetSha256": self.ordered_track_set_sha256,
            "activeTrackCount": self.active_track_count,
        }

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.to_dict())


@dataclass(frozen=True)
class ValidatedCase:
    report_alias: str
    report_kind: str
    case_id: str
    seed_track_id: int | None
    result_fingerprint: str
    repeat_count: int
    expected_result_count: int
    first_run: dict[str, Any]

    @property
    def key(self) -> tuple[str, int | None]:
        return (self.case_id, self.seed_track_id)

    @property
    def tracks(self) -> list[dict[str, Any]]:
        return self.first_run["tracks"]


@dataclass(frozen=True)
class ValidatedReport:
    alias: str
    path: Path
    input_sha256: str
    report_kind: str
    run_id: str
    binding: LibraryBinding
    cases: Mapping[tuple[str, int | None], ValidatedCase]


def _validate_report_header(data: Mapping[str, Any], where: str) -> str:
    state = _string(_field(data, "state", where), f"{where}.state")
    if state != "COMPLETE":
        raise PacketError(f"{where}.state must be COMPLETE, got {state!r}")
    queue_calls = _integer(
        _field(data, "queueMutationApisCalled", where),
        f"{where}.queueMutationApisCalled",
        minimum=0,
    )
    if queue_calls != 0:
        raise PacketError(f"{where} called {queue_calls} queue-mutation API(s); refusing packet")
    return _string(_field(data, "runId", where), f"{where}.runId")


def _validate_track_list(
    tracks_value: Any,
    *,
    expected_count: int,
    report_kind: str,
    where: str,
) -> list[dict[str, Any]]:
    tracks_raw = _array(tracks_value, where)
    if len(tracks_raw) != expected_count:
        raise PacketError(f"{where} has {len(tracks_raw)} results; expected {expected_count}")
    tracks: list[dict[str, Any]] = []
    identities: set[int] = set()
    position_field = "rank" if report_kind == "selector" else "displayedPosition"
    identity_field = "trackId" if report_kind == "selector" else "embeddedTrackId"
    for position, raw in enumerate(tracks_raw, start=1):
        track = _object(raw, f"{where}[{position - 1}]")
        recorded_position = _integer(
            _field(track, position_field, f"{where}[{position - 1}]"),
            f"{where}[{position - 1}].{position_field}",
            minimum=1,
        )
        if recorded_position != position:
            raise PacketError(
                f"{where}[{position - 1}].{position_field} is {recorded_position}; "
                f"expected {position}"
            )
        identity = _integer(
            _field(track, identity_field, f"{where}[{position - 1}]"),
            f"{where}[{position - 1}].{identity_field}",
            minimum=1,
        )
        if identity in identities:
            raise PacketError(f"{where} repeats result identity {identity}")
        identities.add(identity)
        _string(
            _field(track, "filePath", f"{where}[{position - 1}]"),
            f"{where}[{position - 1}].filePath",
        )
        # Playback resolution is path/identity based. Some valid provider rows have
        # no display metadata, so labels are optional reveal-only evidence.
        for display_field in ("title", "artist", "album"):
            display_value = track.get(display_field)
            if display_value is not None and not isinstance(display_value, str):
                raise PacketError(f"{where}[{position - 1}].{display_field} must be text or null")
        _integer(
            _field(track, "durationMs", f"{where}[{position - 1}]"),
            f"{where}[{position - 1}].durationMs",
            minimum=1,
        )
        tracks.append(track)
    return tracks


def _validate_repeated_case(
    *,
    report_alias: str,
    report_kind: str,
    case_id: str,
    seed_track_id: int | None,
    runs: Sequence[dict[str, Any]],
    expected_repeat_count: int,
    expected_result_count: int,
    where: str,
) -> ValidatedCase:
    if len(runs) != expected_repeat_count:
        raise PacketError(f"{where} has {len(runs)} repeats; expected {expected_repeat_count}")
    ordered = sorted(
        runs,
        key=lambda run: _integer(_field(run, "repeat", where), f"{where}.repeat"),
    )
    repeats = [
        _integer(_field(run, "repeat", where), f"{where}.repeat", minimum=1) for run in ordered
    ]
    expected_repeats = list(range(1, expected_repeat_count + 1))
    if repeats != expected_repeats:
        raise PacketError(f"{where} repeat indexes are {repeats}; expected {expected_repeats}")
    if expected_repeat_count < 2:
        raise PacketError(f"{where} needs at least two repeats before blinding")

    fingerprints: list[str] = []
    track_digests: list[str] = []
    request_digests: list[str] = []
    for run in ordered:
        fingerprint = _sha256(_field(run, "resultFingerprint", where), f"{where}.resultFingerprint")
        tracks = _validate_track_list(
            _field(run, "tracks", where),
            expected_count=expected_result_count,
            report_kind=report_kind,
            where=f"{where}.repeat[{run['repeat']}].tracks",
        )
        if report_kind == "selector":
            request_identity = {
                "caseId": run.get("caseId"),
                "seedTrackId": run.get("seedTrackId"),
                "seed": run.get("seed"),
                "config": run.get("config"),
            }
            result_content: Any = tracks
        elif report_kind == SIMPLE_FIND_MUSIC_REPORT_KIND:
            request_identity = {
                "query": run.get("query"),
                "publishedQuery": run.get("publishedQuery"),
                "publishedQuerySpec": run.get("publishedQuerySpec"),
                "resultCount": run.get("resultCount"),
                "planner": run.get("planner"),
                "plannerVersion": run.get("plannerVersion"),
            }
            result_content = {
                "tracks": tracks,
                "stableResultReduction": run.get("stableResultReduction"),
                "textQueuePlan": run.get("textQueuePlan"),
            }
        else:
            request_identity = {
                "caseId": run.get("caseId"),
                "confirmedRecording": run.get("confirmedRecording"),
                "publishedQuerySpec": run.get("publishedQuerySpec"),
            }
            result_content = tracks
        fingerprints.append(fingerprint)
        track_digests.append(_sha256_json(result_content))
        request_digests.append(_sha256_json(request_identity))

    if len(set(fingerprints)) != 1:
        raise PacketError(f"{where} result fingerprints are nondeterministic")
    if len(set(track_digests)) != 1:
        raise PacketError(
            f"{where} repeats changed result content despite their recorded fingerprint"
        )
    if len(set(request_digests)) != 1:
        raise PacketError(f"{where} repeats changed request/configuration content")

    return ValidatedCase(
        report_alias=report_alias,
        report_kind=report_kind,
        case_id=case_id,
        seed_track_id=seed_track_id,
        result_fingerprint=fingerprints[0],
        repeat_count=expected_repeat_count,
        expected_result_count=expected_result_count,
        first_run=ordered[0],
    )


def _validate_selector_report(
    alias: str,
    path: Path,
    data: Mapping[str, Any],
    input_sha256: str,
) -> ValidatedReport:
    where = f"report[{alias}]"
    run_id = _validate_report_header(data, where)
    generation = _object(_field(data, "generation", where), f"{where}.generation")
    provider = _object(_field(data, "providerSnapshot", where), f"{where}.providerSnapshot")
    catalog = _object(_field(data, "activeCatalog", where), f"{where}.activeCatalog")
    binding = LibraryBinding(
        generation_id=_string(
            _field(generation, "generationId", f"{where}.generation"),
            f"{where}.generation.generationId",
        ),
        provider_generation_id=_string(
            _field(provider, "generationId", f"{where}.providerSnapshot"),
            f"{where}.providerSnapshot.generationId",
        ),
        activation_binding_id=_string(
            _field(generation, "activationBindingId", f"{where}.generation"),
            f"{where}.generation.activationBindingId",
        ),
        database_content_sha256=_sha256(
            _field(generation, "databaseContentSha256", f"{where}.generation"),
            f"{where}.generation.databaseContentSha256",
        ),
        ordered_track_set_sha256=_sha256(
            _field(generation, "orderedTrackSetSha256", f"{where}.generation"),
            f"{where}.generation.orderedTrackSetSha256",
        ),
        active_track_count=_integer(
            _field(catalog, "activeTrackCount", f"{where}.activeCatalog"),
            f"{where}.activeCatalog.activeTrackCount",
            minimum=1,
        ),
    )
    request = _object(_field(data, "request", where), f"{where}.request")
    repeat_count = _integer(
        _field(request, "repeatCount", f"{where}.request"),
        f"{where}.request.repeatCount",
        minimum=2,
    )
    runs_raw = _array(_field(data, "selectionRuns", where), f"{where}.selectionRuns")
    planned = _integer(
        _field(data, "plannedSelectionRunCount", where),
        f"{where}.plannedSelectionRunCount",
        minimum=1,
    )
    if len(runs_raw) != planned:
        raise PacketError(f"{where} contains {len(runs_raw)} selection runs; planned {planned}")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for index, raw in enumerate(runs_raw):
        run = _object(raw, f"{where}.selectionRuns[{index}]")
        case_id = _string(
            _field(run, "caseId", f"{where}.selectionRuns[{index}]"),
            f"{where}.selectionRuns[{index}].caseId",
        )
        seed_track_id = _integer(
            _field(run, "seedTrackId", f"{where}.selectionRuns[{index}]"),
            f"{where}.selectionRuns[{index}].seedTrackId",
            minimum=1,
        )
        config = _object(
            _field(run, "config", f"{where}.selectionRuns[{index}]"),
            f"{where}.selectionRuns[{index}].config",
        )
        expected_count = _integer(
            _field(config, "numTracks", f"{where}.selectionRuns[{index}].config"),
            f"{where}.selectionRuns[{index}].config.numTracks",
            minimum=1,
        )
        if (
            "resolvedCount" in run
            and _integer(run["resolvedCount"], f"{where}.selectionRuns[{index}].resolvedCount")
            != expected_count
        ):
            raise PacketError(
                f"{where}.selectionRuns[{index}] resolved count does not match request"
            )
        if run.get("inactiveResultCount", 0) != 0:
            raise PacketError(f"{where}.selectionRuns[{index}] contains inactive results")
        grouped.setdefault((case_id, seed_track_id), []).append(run)

    cases: dict[tuple[str, int | None], ValidatedCase] = {}
    for (case_id, seed_track_id), runs in grouped.items():
        first_config = _object(runs[0]["config"], f"{where}.{case_id}.config")
        expected_count = _integer(
            _field(first_config, "numTracks", f"{where}.{case_id}.config"),
            f"{where}.{case_id}.config.numTracks",
            minimum=1,
        )
        case = _validate_repeated_case(
            report_alias=alias,
            report_kind="selector",
            case_id=case_id,
            seed_track_id=seed_track_id,
            runs=runs,
            expected_repeat_count=repeat_count,
            expected_result_count=expected_count,
            where=f"{where}.case[{case_id!r},seed={seed_track_id}]",
        )
        cases[case.key] = case
    if len(cases) * repeat_count != planned:
        raise PacketError(f"{where} case/repeat cardinality disagrees with planned run count")
    return ValidatedReport(
        alias=alias,
        path=path,
        input_sha256=input_sha256,
        report_kind="selector",
        run_id=run_id,
        binding=binding,
        cases=cases,
    )


def _composed_run_binding(run: Mapping[str, Any], where: str) -> LibraryBinding:
    query_spec = _object(_field(run, "publishedQuerySpec", where), f"{where}.publishedQuerySpec")
    library = _object(
        _field(query_spec, "libraryBinding", f"{where}.publishedQuerySpec"),
        f"{where}.publishedQuerySpec.libraryBinding",
    )
    generation_id = _string(
        _field(run, "libraryGenerationId", where), f"{where}.libraryGenerationId"
    )
    if library.get("generationId") != generation_id:
        raise PacketError(f"{where} generation disagrees with its published binding")
    activation_id = _string(
        _field(run, "activationBindingId", where), f"{where}.activationBindingId"
    )
    if library.get("activationBindingId") != activation_id:
        raise PacketError(f"{where} activation ID disagrees with its published binding")
    return LibraryBinding(
        generation_id=generation_id,
        provider_generation_id=_string(
            _field(run, "providerGenerationId", where),
            f"{where}.providerGenerationId",
        ),
        activation_binding_id=activation_id,
        database_content_sha256=_sha256(
            _field(library, "databaseContentSha256", f"{where}.publishedQuerySpec.libraryBinding"),
            f"{where}.publishedQuerySpec.libraryBinding.databaseContentSha256",
        ),
        ordered_track_set_sha256=_sha256(
            _field(library, "orderedTrackSetSha256", f"{where}.publishedQuerySpec.libraryBinding"),
            f"{where}.publishedQuerySpec.libraryBinding.orderedTrackSetSha256",
        ),
        active_track_count=_integer(
            _field(run, "activeTrackCount", where),
            f"{where}.activeTrackCount",
            minimum=1,
        ),
    )


def _validate_composed_report(
    alias: str,
    path: Path,
    data: Mapping[str, Any],
    input_sha256: str,
) -> ValidatedReport:
    where = f"report[{alias}]"
    run_id = _validate_report_header(data, where)
    if data.get("settingsRestoredExactly") is not True:
        raise PacketError(f"{where}.settingsRestoredExactly must be true")
    request = _object(_field(data, "request", where), f"{where}.request")
    repeat_count = _integer(
        _field(request, "repeatCount", f"{where}.request"),
        f"{where}.request.repeatCount",
        minimum=2,
    )
    runs_raw = _array(_field(data, "runs", where), f"{where}.runs")
    if not runs_raw:
        raise PacketError(f"{where}.runs must not be empty")
    runs = [_object(run, f"{where}.runs[{index}]") for index, run in enumerate(runs_raw)]
    binding = _composed_run_binding(runs[0], f"{where}.runs[0]")
    database = _object(_field(data, "databaseInfo", where), f"{where}.databaseInfo")
    if database.get("generationId") != binding.generation_id:
        raise PacketError(f"{where}.databaseInfo generation disagrees with runs")
    if database.get("providerGenerationId") != binding.provider_generation_id:
        raise PacketError(f"{where}.databaseInfo provider generation disagrees with runs")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, run in enumerate(runs):
        run_where = f"{where}.runs[{index}]"
        if _composed_run_binding(run, run_where) != binding:
            raise PacketError(f"{run_where} uses a different active-library binding")
        case_id = _string(_field(run, "caseId", run_where), f"{run_where}.caseId")
        query_spec = _object(
            _field(run, "publishedQuerySpec", run_where),
            f"{run_where}.publishedQuerySpec",
        )
        result_limit = _integer(
            _field(query_spec, "resultLimit", f"{run_where}.publishedQuerySpec"),
            f"{run_where}.publishedQuerySpec.resultLimit",
            minimum=1,
        )
        requested_visible = _integer(
            _field(run, "stableReductionRequestedVisibleCount", run_where),
            f"{run_where}.stableReductionRequestedVisibleCount",
            minimum=1,
        )
        if requested_visible != result_limit:
            raise PacketError(f"{run_where} result limit disagrees with stable reduction request")
        grouped.setdefault(case_id, []).append(run)

    determinism = _object(_field(data, "repeatDeterminism", where), f"{where}.repeatDeterminism")
    cases: dict[tuple[str, int | None], ValidatedCase] = {}
    for case_id, case_runs in grouped.items():
        evidence = _object(
            _field(determinism, case_id, f"{where}.repeatDeterminism"),
            f"{where}.repeatDeterminism.{case_id}",
        )
        if evidence.get("deterministic") is not True:
            raise PacketError(f"{where}.{case_id} is marked nondeterministic")
        observed = _integer(
            _field(evidence, "observationCount", f"{where}.repeatDeterminism.{case_id}"),
            f"{where}.repeatDeterminism.{case_id}.observationCount",
            minimum=1,
        )
        if observed != repeat_count:
            raise PacketError(
                f"{where}.{case_id} determinism observations do not match repeat count"
            )
        query_spec = _object(
            case_runs[0]["publishedQuerySpec"], f"{where}.{case_id}.publishedQuerySpec"
        )
        expected_count = _integer(
            _field(query_spec, "resultLimit", f"{where}.{case_id}.publishedQuerySpec"),
            f"{where}.{case_id}.publishedQuerySpec.resultLimit",
            minimum=1,
        )
        case = _validate_repeated_case(
            report_alias=alias,
            report_kind="find_music",
            case_id=case_id,
            seed_track_id=None,
            runs=case_runs,
            expected_repeat_count=repeat_count,
            expected_result_count=expected_count,
            where=f"{where}.case[{case_id!r}]",
        )
        cases[case.key] = case
    if len(cases) * repeat_count != len(runs):
        raise PacketError(f"{where} case/repeat cardinality is inconsistent")
    return ValidatedReport(
        alias=alias,
        path=path,
        input_sha256=input_sha256,
        report_kind="find_music",
        run_id=run_id,
        binding=binding,
        cases=cases,
    )


def _validate_simple_find_music_report(
    alias: str,
    path: Path,
    data: Mapping[str, Any],
    input_sha256: str,
) -> ValidatedReport:
    where = f"report[{alias}]"
    run_id = _validate_report_header(data, where)
    if data.get("settingsRestoredExactly") is not True:
        raise PacketError(f"{where}.settingsRestoredExactly must be true")
    if data.get("fatalError") is not None:
        raise PacketError(f"{where}.fatalError must be null")

    request = _object(_field(data, "request", where), f"{where}.request")
    if request.get("runId") != run_id:
        raise PacketError(f"{where}.request.runId disagrees with the report run ID")
    repeat_count = _integer(
        _field(request, "repeatCount", f"{where}.request"),
        f"{where}.request.repeatCount",
        minimum=2,
    )
    queries_raw = _array(_field(request, "queries", f"{where}.request"), f"{where}.request.queries")
    queries = [
        _string(value, f"{where}.request.queries[{index}]")
        for index, value in enumerate(queries_raw)
    ]
    if not queries or len(set(queries)) != len(queries):
        raise PacketError(f"{where}.request.queries must be non-empty and unique")
    counts_raw = _array(
        _field(request, "resultCounts", f"{where}.request"),
        f"{where}.request.resultCounts",
    )
    result_counts = [
        _integer(value, f"{where}.request.resultCounts[{index}]", minimum=1)
        for index, value in enumerate(counts_raw)
    ]
    if not result_counts or len(set(result_counts)) != len(result_counts):
        raise PacketError(f"{where}.request.resultCounts must be non-empty and unique")

    runs_raw = _array(_field(data, "searchRuns", where), f"{where}.searchRuns")
    planners = {"CLOSEST": "closest", "VARIED_DPP": "varied_dpp"}
    expected_case_ids = {
        f"{query}|{count}|{wire_name}"
        for query in queries
        for count in result_counts
        for wire_name in planners.values()
    }
    expected_run_count = len(expected_case_ids) * repeat_count
    if len(runs_raw) != expected_run_count:
        raise PacketError(
            f"{where} contains {len(runs_raw)} search runs; expected {expected_run_count}"
        )
    runs = [
        _object(raw, f"{where}.searchRuns[{index}]")
        for index, raw in enumerate(runs_raw)
    ]
    if not runs:
        raise PacketError(f"{where}.searchRuns must not be empty")

    binding = _composed_run_binding(runs[0], f"{where}.searchRuns[0]")
    database = _object(_field(data, "databaseInfo", where), f"{where}.databaseInfo")
    if database.get("generationId") != binding.generation_id:
        raise PacketError(f"{where}.databaseInfo generation disagrees with search runs")
    if database.get("providerGenerationId") != binding.provider_generation_id:
        raise PacketError(f"{where}.databaseInfo provider generation disagrees with search runs")
    if database.get("activeTrackCount") != binding.active_track_count:
        raise PacketError(f"{where}.databaseInfo active count disagrees with search runs")

    grouped: dict[str, list[dict[str, Any]]] = {}
    ordered_active_ids_sha256: str | None = None
    for index, run in enumerate(runs):
        run_where = f"{where}.searchRuns[{index}]"
        if _composed_run_binding(run, run_where) != binding:
            raise PacketError(f"{run_where} uses a different active-library binding")
        active_ids_sha256 = _sha256(
            _field(run, "orderedActiveTrackIdsSha256", run_where),
            f"{run_where}.orderedActiveTrackIdsSha256",
        )
        if ordered_active_ids_sha256 is None:
            ordered_active_ids_sha256 = active_ids_sha256
        elif active_ids_sha256 != ordered_active_ids_sha256:
            raise PacketError(f"{run_where} uses a different ordered active-track domain")

        query = _string(_field(run, "query", run_where), f"{run_where}.query")
        if query not in queries:
            raise PacketError(f"{run_where}.query was not requested")
        if run.get("publishedQuery") != query:
            raise PacketError(f"{run_where}.publishedQuery differs from the request")
        result_count = _integer(
            _field(run, "resultCount", run_where), f"{run_where}.resultCount", minimum=1
        )
        if result_count not in result_counts:
            raise PacketError(f"{run_where}.resultCount was not requested")
        planner = _string(_field(run, "planner", run_where), f"{run_where}.planner")
        if planner not in planners:
            raise PacketError(f"{run_where}.planner is not Closest or Varied")
        planner_version = _integer(
            _field(run, "plannerVersion", run_where),
            f"{run_where}.plannerVersion",
            minimum=1,
        )
        if _field(run, "resultKind", run_where) != "TEXT" or run.get("error") is not None:
            raise PacketError(f"{run_where} is not a successful text result")
        if _integer(
            _field(run, "objectiveRankingDomainCount", run_where),
            f"{run_where}.objectiveRankingDomainCount",
            minimum=1,
        ) != binding.active_track_count:
            raise PacketError(f"{run_where} objective domain differs from the active library")

        query_spec = _object(
            _field(run, "publishedQuerySpec", run_where), f"{run_where}.publishedQuerySpec"
        )
        if (
            query_spec.get("operator") != "ALL_OF"
            or query_spec.get("resultLimit") != result_count
            or query_spec.get("textResultPlanner") != planner
        ):
            raise PacketError(f"{run_where}.publishedQuerySpec differs from the request")
        if _array(
            _field(query_spec, "songSeeds", f"{run_where}.publishedQuerySpec"),
            f"{run_where}.publishedQuerySpec.songSeeds",
        ):
            raise PacketError(f"{run_where}.publishedQuerySpec unexpectedly has song seeds")
        ingredients = _array(
            _field(query_spec, "textIngredients", f"{run_where}.publishedQuerySpec"),
            f"{run_where}.publishedQuerySpec.textIngredients",
        )
        if len(ingredients) != 1:
            raise PacketError(f"{run_where}.publishedQuerySpec must have one text ingredient")
        ingredient = _object(ingredients[0], f"{run_where}.publishedQuerySpec.textIngredients[0]")
        weight = ingredient.get("weight")
        if (
            ingredient.get("query") != query
            or ingredient.get("negative") is not False
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or float(weight) != 1.0
        ):
            raise PacketError(f"{run_where}.publishedQuerySpec has the wrong text ingredient")

        queue_eligibility = _object(
            _field(run, "queueEligibility", run_where), f"{run_where}.queueEligibility"
        )
        if queue_eligibility.get("eligible") is not True:
            raise PacketError(f"{run_where} is not queue-ready")
        reduction = _object(
            _field(run, "stableResultReduction", run_where),
            f"{run_where}.stableResultReduction",
        )
        if _integer(
            _field(reduction, "requestedVisibleCount", f"{run_where}.stableResultReduction"),
            f"{run_where}.stableResultReduction.requestedVisibleCount",
            minimum=1,
        ) != result_count:
            raise PacketError(f"{run_where} stable reduction requested the wrong result count")
        _integer(
            _field(reduction, "collapsedEquivalentCount", f"{run_where}.stableResultReduction"),
            f"{run_where}.stableResultReduction.collapsedEquivalentCount",
            minimum=0,
        )
        scanned_count = _integer(
            _field(reduction, "scannedRowCount", f"{run_where}.stableResultReduction"),
            f"{run_where}.stableResultReduction.scannedRowCount",
            minimum=1,
        )
        if scanned_count < result_count:
            raise PacketError(f"{run_where} stable reduction scanned fewer rows than it returned")

        tracks = _validate_track_list(
            _field(run, "tracks", run_where),
            expected_count=result_count,
            report_kind=SIMPLE_FIND_MUSIC_REPORT_KIND,
            where=f"{run_where}.tracks",
        )
        objective_ranks = [
            _integer(
                _field(track, "objectiveRank", f"{run_where}.tracks[{track_index}]"),
                f"{run_where}.tracks[{track_index}].objectiveRank",
                minimum=1,
            )
            for track_index, track in enumerate(tracks)
        ]
        if len(set(objective_ranks)) != len(objective_ranks):
            raise PacketError(f"{run_where} repeats a text-objective rank")
        if max(objective_ranks) > binding.active_track_count:
            raise PacketError(f"{run_where} has an objective rank outside the active domain")
        track_ids = [int(track["embeddedTrackId"]) for track in tracks]
        text_plan = run.get("textQueuePlan")
        if planner == "CLOSEST":
            if text_plan is not None:
                raise PacketError(f"{run_where} Closest result unexpectedly has a Varied plan")
            if objective_ranks != sorted(objective_ranks):
                raise PacketError(f"{run_where} Closest result is not ordered by objective rank")
        else:
            plan = _object(text_plan, f"{run_where}.textQueuePlan")
            if (
                plan.get("planner") != planner
                or plan.get("plannerVersion") != planner_version
                or plan.get("requestedResultCount") != result_count
                or plan.get("completeCandidateDomainCount") != binding.active_track_count
                or plan.get("orderedSelectedTrackIds") != track_ids
                or plan.get("orderedOriginalTextObjectiveRanks") != objective_ranks
            ):
                raise PacketError(f"{run_where} Varied result differs from its selection proof")
            _sha256(
                _field(plan, "completeTextRankingSha256", f"{run_where}.textQueuePlan"),
                f"{run_where}.textQueuePlan.completeTextRankingSha256",
            )
            dpp = _object(
                _field(plan, "dppSelection", f"{run_where}.textQueuePlan"),
                f"{run_where}.textQueuePlan.dppSelection",
            )
            gains = _array(
                _field(dpp, "selectedMarginalGains", f"{run_where}.textQueuePlan.dppSelection"),
                f"{run_where}.textQueuePlan.dppSelection.selectedMarginalGains",
            )
            if len(gains) != result_count:
                raise PacketError(f"{run_where} Varied proof has the wrong selection-step count")

        case_id = f"{query}|{result_count}|{planners[planner]}"
        grouped.setdefault(case_id, []).append(run)

    if set(grouped) != expected_case_ids:
        missing = sorted(expected_case_ids - grouped.keys())
        unexpected = sorted(grouped.keys() - expected_case_ids)
        raise PacketError(
            f"{where} search matrix is incomplete; missing={missing}, unexpected={unexpected}"
        )
    determinism = _object(_field(data, "repeatDeterminism", where), f"{where}.repeatDeterminism")
    if set(determinism) != expected_case_ids:
        raise PacketError(f"{where}.repeatDeterminism does not cover the exact search matrix")

    cases: dict[tuple[str, int | None], ValidatedCase] = {}
    for case_id, case_runs in grouped.items():
        evidence = _object(
            _field(determinism, case_id, f"{where}.repeatDeterminism"),
            f"{where}.repeatDeterminism.{case_id}",
        )
        if evidence.get("evaluated") is not True or evidence.get("deterministic") is not True:
            raise PacketError(f"{where}.{case_id} is not proven deterministic")
        observed = _integer(
            _field(evidence, "observationCount", f"{where}.repeatDeterminism.{case_id}"),
            f"{where}.repeatDeterminism.{case_id}.observationCount",
            minimum=1,
        )
        if observed != repeat_count:
            raise PacketError(f"{where}.{case_id} determinism observations are incomplete")
        expected_count = _integer(
            _field(case_runs[0], "resultCount", f"{where}.case[{case_id!r}]"),
            f"{where}.case[{case_id!r}].resultCount",
            minimum=1,
        )
        case = _validate_repeated_case(
            report_alias=alias,
            report_kind=SIMPLE_FIND_MUSIC_REPORT_KIND,
            case_id=case_id,
            seed_track_id=None,
            runs=case_runs,
            expected_repeat_count=repeat_count,
            expected_result_count=expected_count,
            where=f"{where}.case[{case_id!r}]",
        )
        cases[case.key] = case
    return ValidatedReport(
        alias=alias,
        path=path,
        input_sha256=input_sha256,
        report_kind=SIMPLE_FIND_MUSIC_REPORT_KIND,
        run_id=run_id,
        binding=binding,
        cases=cases,
    )


def validate_report(alias: str, path: Path) -> ValidatedReport:
    data, input_sha256 = _read_json(path)
    if "selectionRuns" in data:
        return _validate_selector_report(alias, path, data, input_sha256)
    if "searchRuns" in data and "databaseInfo" in data:
        return _validate_simple_find_music_report(alias, path, data, input_sha256)
    if "runs" in data and "databaseInfo" in data:
        return _validate_composed_report(alias, path, data, input_sha256)
    raise PacketError(
        f"report[{alias}] is not a supported selector, simple, or composed Find Music report"
    )


def validate_reports(report_paths: Mapping[str, Path]) -> dict[str, ValidatedReport]:
    if not report_paths:
        raise PacketError("at least one report is required")
    reports = {alias: validate_report(alias, path) for alias, path in sorted(report_paths.items())}
    first_alias = next(iter(reports))
    expected = reports[first_alias].binding
    for alias, report in reports.items():
        if report.binding == expected:
            continue
        differences = [
            key
            for key, expected_value in expected.to_dict().items()
            if report.binding.to_dict().get(key) != expected_value
        ]
        raise PacketError(
            f"report[{alias}] active-library binding differs from report[{first_alias}] "
            f"in: {', '.join(differences)}"
        )
    return reports


def _resolve_case(
    endpoint: Mapping[str, Any],
    reports: Mapping[str, ValidatedReport],
    where: str,
) -> ValidatedCase:
    _strict_keys(
        endpoint,
        {"report", "case_id", "seed_track_id"},
        {"report", "case_id"},
        where,
    )
    alias = _string(endpoint["report"], f"{where}.report")
    if alias not in reports:
        raise PacketError(f"{where}.report names unknown alias {alias!r}")
    case_id = _string(endpoint["case_id"], f"{where}.case_id")
    seed_track_id: int | None = None
    if "seed_track_id" in endpoint:
        seed_track_id = _integer(endpoint["seed_track_id"], f"{where}.seed_track_id", minimum=1)
    report = reports[alias]
    matches = [
        case
        for (candidate_case_id, candidate_seed), case in report.cases.items()
        if candidate_case_id == case_id
        and (seed_track_id is None or candidate_seed == seed_track_id)
    ]
    if not matches:
        raise PacketError(f"{where} does not match a validated report case")
    if len(matches) > 1:
        raise PacketError(f"{where} is ambiguous; add seed_track_id to select one seed exactly")
    return matches[0]


def _parse_plan(
    plan_path: Path,
    reports: Mapping[str, ValidatedReport],
) -> tuple[dict[str, Any], str, list[tuple[dict[str, Any], ValidatedCase, ValidatedCase]]]:
    plan, plan_sha256 = _read_json(plan_path)
    _strict_keys(plan, {"schema", "study_id", "trials"}, {"schema", "study_id", "trials"}, "plan")
    if plan["schema"] != PLAN_SCHEMA:
        raise PacketError(f"plan.schema must be {PLAN_SCHEMA!r}")
    study_id = _string(plan["study_id"], "plan.study_id")
    if SAFE_ID_RE.fullmatch(study_id) is None:
        raise PacketError("plan.study_id contains unsupported characters")
    trials_raw = _array(plan["trials"], "plan.trials")
    if not trials_raw:
        raise PacketError("plan.trials must not be empty")
    trial_ids: set[str] = set()
    resolved: list[tuple[dict[str, Any], ValidatedCase, ValidatedCase]] = []
    for index, raw in enumerate(trials_raw):
        where = f"plan.trials[{index}]"
        trial = _object(raw, where)
        _strict_keys(
            trial,
            {"id", "prompt", "question", "candidate_1", "candidate_2"},
            {"id", "prompt", "candidate_1", "candidate_2"},
            where,
        )
        trial_id = _string(trial["id"], f"{where}.id")
        if SAFE_ID_RE.fullmatch(trial_id) is None:
            raise PacketError(f"{where}.id contains unsupported characters")
        if trial_id in trial_ids:
            raise PacketError(f"duplicate trial id {trial_id!r}")
        trial_ids.add(trial_id)
        _string(trial["prompt"], f"{where}.prompt")
        if "question" in trial:
            _string(trial["question"], f"{where}.question")
        candidate_1 = _resolve_case(
            _object(trial["candidate_1"], f"{where}.candidate_1"),
            reports,
            f"{where}.candidate_1",
        )
        candidate_2 = _resolve_case(
            _object(trial["candidate_2"], f"{where}.candidate_2"),
            reports,
            f"{where}.candidate_2",
        )
        if candidate_1.report_kind != candidate_2.report_kind:
            raise PacketError(f"{where} mixes selector and Find Music result families")
        if candidate_1.expected_result_count != candidate_2.expected_result_count:
            raise PacketError(f"{where} candidates have different result counts")
        if candidate_1.result_fingerprint == candidate_2.result_fingerprint:
            raise PacketError(f"{where} candidates are the same ordered result")
        if (
            candidate_1.report_kind == "selector"
            and candidate_1.seed_track_id != candidate_2.seed_track_id
        ):
            raise PacketError(f"{where} compares different selector seeds")
        resolved.append((trial, candidate_1, candidate_2))
    return plan, plan_sha256, resolved


def _hmac_digest(key: bytes, *parts: str) -> bytes:
    payload = "\0".join(parts).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).digest()


def _opaque_token(key: bytes, domain: str, *parts: str, size: int = 24) -> str:
    return f"{domain}-" + _hmac_digest(key, domain, *parts).hex()[:size]


def _source_locator(path: str) -> dict[str, str]:
    windows = re.match(r"^([A-Za-z]):\\(.*)$", path)
    if windows:
        drive = windows.group(1).lower()
        tail = windows.group(2).replace("\\", "/")
        return {
            "kind": "windows_path",
            "reportedPath": path,
            "wslHostCandidate": f"/mnt/{drive}/{tail}",
        }
    if path.startswith("/storage/") or path.startswith("/sdcard/"):
        return {
            "kind": "android_device_path",
            "reportedPath": path,
            "adbPullSource": path,
        }
    return {"kind": "host_path", "reportedPath": path, "hostCandidate": path}


def _rank_evidence(
    track: Mapping[str, Any],
    report_kind: str,
    run: Mapping[str, Any],
) -> dict[str, Any]:
    if report_kind == "selector":
        return {
            "distanceFromSeedRank": track.get("seedRank"),
        }
    if report_kind == SIMPLE_FIND_MUSIC_REPORT_KIND:
        return {
            "textMatchRank": track.get("objectiveRank"),
            "rankedTrackCount": run.get("objectiveRankingDomainCount"),
        }
    ingredients = []
    for evidence_raw in track.get("ingredientEvidence", []):
        if not isinstance(evidence_raw, dict):
            continue
        ingredients.append(
            {
                "label": evidence_raw.get("label"),
                "exactRank": evidence_raw.get("exactRank"),
                "rankedTrackCount": evidence_raw.get("rankedTrackCount"),
                "topFraction": evidence_raw.get("topFraction"),
            }
        )
    return {
        "overall": track.get("overallEvidence"),
        "ingredients": ingredients,
    }


def _revealed_track(
    track: Mapping[str, Any],
    report_kind: str,
    run: Mapping[str, Any],
    audio_token: str,
    position: int,
) -> dict[str, Any]:
    identity_field = "trackId" if report_kind == "selector" else "embeddedTrackId"
    result = {
        "audioToken": audio_token,
        "position": position,
        "embeddedTrackId": track.get(identity_field),
        "powerampFileId": track.get("catalogPowerampFileId"),
        "artist": track.get("artist"),
        "title": track.get("title"),
        "album": track.get("album"),
        "durationMs": track.get("durationMs"),
        "source": _source_locator(str(track.get("filePath"))),
        "rankEvidence": _rank_evidence(track, report_kind, run),
    }
    return result


def build_packet(
    *,
    reports: Mapping[str, ValidatedReport],
    plan_path: Path,
    blind_key: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(blind_key) < 16:
        raise PacketError("blind key must contain at least 16 bytes")
    plan, plan_sha256, trials = _parse_plan(plan_path, reports)
    report_commitments = {
        alias: {
            "inputSha256": report.input_sha256,
            "runId": report.run_id,
            "reportKind": report.report_kind,
            "bindingFingerprint": report.binding.fingerprint,
        }
        for alias, report in sorted(reports.items())
    }
    packet_basis = {
        "planSha256": plan_sha256,
        "reports": report_commitments,
    }
    packet_id = (
        "blind-v2-" + _hmac_digest(blind_key, "packet", _sha256_json(packet_basis)).hex()[:20]
    )
    ordered_trials = sorted(
        trials,
        key=lambda item: _hmac_digest(blind_key, "trial-order", plan["study_id"], item[0]["id"]),
    )

    public_trials: list[dict[str, Any]] = []
    reveal_trials: list[dict[str, Any]] = []
    for public_index, (trial, candidate_1, candidate_2) in enumerate(ordered_trials, start=1):
        public_trial_id = f"T{public_index:03d}"
        swap = bool(_hmac_digest(blind_key, "side-order", plan["study_id"], trial["id"])[0] & 1)
        ordered_candidates = (
            (
                (candidate_2, "candidate_2"),
                (candidate_1, "candidate_1"),
            )
            if swap
            else (
                (candidate_1, "candidate_1"),
                (candidate_2, "candidate_2"),
            )
        )
        public_sides: list[dict[str, Any]] = []
        reveal_sides: dict[str, Any] = {}
        for side_label, (case, source_candidate) in zip(
            ("A", "B"), ordered_candidates, strict=True
        ):
            queue_token = _opaque_token(
                blind_key,
                "queue",
                packet_id,
                trial["id"],
                side_label,
                case.result_fingerprint,
            )
            public_tracks = []
            reveal_tracks = []
            for position, track in enumerate(case.tracks, start=1):
                identity = track.get(
                    "trackId" if case.report_kind == "selector" else "embeddedTrackId"
                )
                audio_token = _opaque_token(
                    blind_key,
                    "audio",
                    packet_id,
                    trial["id"],
                    side_label,
                    str(position),
                    str(identity),
                    str(track.get("filePath")),
                )
                public_tracks.append({"position": position, "audioToken": audio_token})
                reveal_tracks.append(
                    _revealed_track(
                        track,
                        case.report_kind,
                        case.first_run,
                        audio_token,
                        position,
                    )
                )
            public_sides.append(
                {
                    "label": side_label,
                    "queueToken": queue_token,
                    "trackCount": len(public_tracks),
                    "tracks": public_tracks,
                }
            )
            request_evidence: dict[str, Any]
            if case.report_kind == "selector":
                request_evidence = {
                    "seed": case.first_run.get("seed"),
                    "config": case.first_run.get("config"),
                }
            elif case.report_kind == SIMPLE_FIND_MUSIC_REPORT_KIND:
                request_evidence = {
                    "query": case.first_run.get("query"),
                    "resultCount": case.first_run.get("resultCount"),
                    "planner": case.first_run.get("planner"),
                    "querySpec": case.first_run.get("publishedQuerySpec"),
                }
            else:
                request_evidence = {
                    "confirmedRecording": case.first_run.get("confirmedRecording"),
                    "querySpec": case.first_run.get("publishedQuerySpec"),
                }
            reveal_sides[side_label] = {
                "sourceCandidate": source_candidate,
                "queueToken": queue_token,
                "reportAlias": case.report_alias,
                "reportKind": case.report_kind,
                "caseId": case.case_id,
                "seedTrackId": case.seed_track_id,
                "resultFingerprint": case.result_fingerprint,
                "selectedRepeat": 1,
                "verifiedRepeatCount": case.repeat_count,
                "requestEvidence": request_evidence,
                "tracks": reveal_tracks,
            }
        public_trials.append(
            {
                "trial": public_trial_id,
                "prompt": trial["prompt"],
                "question": trial.get(
                    "question", "Which queue better delivers the requested listening intent?"
                ),
                "sides": public_sides,
                "response": {
                    "preference": None,
                    "intentFit": {"A": None, "B": None},
                    "coherence": {"A": None, "B": None},
                    "discoveryValue": {"A": None, "B": None},
                    "notes": "",
                },
            }
        )
        reveal_trials.append(
            {
                "trial": public_trial_id,
                "planTrialId": trial["id"],
                "prompt": trial["prompt"],
                "sides": reveal_sides,
            }
        )

    reveal_core = {
        "schema": REVEAL_SCHEMA,
        "packetId": packet_id,
        "studyId": plan["study_id"],
        "blindKeySha256": _sha256_bytes(blind_key),
        "plan": {
            "path": str(plan_path.resolve()),
            "sha256": plan_sha256,
        },
        "reports": {
            alias: {
                "path": str(report.path.resolve()),
                "inputSha256": report.input_sha256,
                "runId": report.run_id,
                "reportKind": report.report_kind,
                "binding": report.binding.to_dict(),
            }
            for alias, report in sorted(reports.items())
        },
        "trials": reveal_trials,
    }
    reveal_commitment = _sha256_json(reveal_core)
    binding = next(iter(reports.values())).binding
    public_manifest = {
        "schema": PACKET_SCHEMA,
        "packetId": packet_id,
        "studyId": plan["study_id"],
        "sourceBindingSha256": binding.fingerprint,
        "revealCommitmentSha256": reveal_commitment,
        "protocol": {
            "playback": (
                "Listen to full tracks; seek to beginning, middle, and late sections as useful."
            ),
            "concealment": (
                "Method names, track labels, ranks, and source paths stay hidden "
                "until scoring is complete."
            ),
            "preferenceValues": ["A", "B", "tie"],
            "ratingScale": "Use 1-5 for intent fit, queue coherence, and discovery value.",
        },
        "trials": public_trials,
    }
    reveal = {
        **reveal_core,
        "revealCoreSha256": reveal_commitment,
        "publicManifestSha256": _sha256_bytes(_pretty_json(public_manifest)),
    }
    return public_manifest, reveal


def _atomic_write(path: Path, content: bytes, mode: int, *, force: bool) -> None:
    if path.exists():
        try:
            current = path.read_bytes()
        except OSError as exc:
            raise PacketError(f"cannot inspect existing {path}: {exc}") from exc
        if current == content:
            os.chmod(path, mode)
            return
        if not force:
            raise PacketError(f"refusing to replace different existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_packet(
    output_dir: Path,
    public_manifest: Mapping[str, Any],
    reveal: Mapping[str, Any],
    *,
    force: bool = False,
) -> dict[str, str]:
    public_bytes = _pretty_json(public_manifest)
    reveal_bytes = _pretty_json(reveal)
    public_path = output_dir / "blind-manifest.json"
    reveal_path = output_dir / "reveal-key.json"
    _atomic_write(public_path, public_bytes, 0o644, force=force)
    _atomic_write(reveal_path, reveal_bytes, 0o600, force=force)
    return {
        "publicPath": str(public_path),
        "publicSha256": _sha256_bytes(public_bytes),
        "revealPath": str(reveal_path),
        "revealSha256": _sha256_bytes(reveal_bytes),
    }


def _parse_report_arguments(values: Iterable[str]) -> dict[str, Path]:
    reports: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise PacketError("--report must use ALIAS=PATH")
        alias, raw_path = value.split("=", 1)
        if SAFE_ID_RE.fullmatch(alias) is None:
            raise PacketError(f"invalid report alias {alias!r}")
        if alias in reports:
            raise PacketError(f"duplicate report alias {alias!r}")
        if not raw_path:
            raise PacketError(f"report alias {alias!r} has an empty path")
        reports[alias] = Path(raw_path)
    return reports


def _validation_summary(reports: Mapping[str, ValidatedReport]) -> dict[str, Any]:
    binding = next(iter(reports.values())).binding
    return {
        "state": "VALID",
        "binding": binding.to_dict(),
        "bindingFingerprint": binding.fingerprint,
        "reports": {
            alias: {
                "kind": report.report_kind,
                "runId": report.run_id,
                "inputSha256": report.input_sha256,
                "caseCount": len(report.cases),
                "verifiedRepeatCounts": sorted(
                    {case.repeat_count for case in report.cases.values()}
                ),
            }
            for alias, report in sorted(reports.items())
        },
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="ALIAS=PATH",
        help="completed selector, simple Find Music, or composed Find Music acceptance report",
    )
    parser.add_argument("--plan", type=Path, help=f"JSON plan using schema {PLAN_SCHEMA}")
    parser.add_argument(
        "--blind-key-file",
        type=Path,
        help="secret file with at least 16 bytes; never copied to the packet",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate report provenance/repeats without loading a plan",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and hash both artifacts in memory without writing files",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        report_paths = _parse_report_arguments(args.report)
        reports = validate_reports(report_paths)
        if args.validate_only:
            if any((args.plan, args.blind_key_file, args.output_dir, args.force)):
                raise PacketError("--validate-only cannot be combined with packet output arguments")
            print(json.dumps(_validation_summary(reports), indent=2, sort_keys=True))
            return 0
        if args.plan is None or args.blind_key_file is None:
            raise PacketError("--plan and --blind-key-file are required")
        if not args.dry_run and args.output_dir is None:
            raise PacketError("--output-dir is required unless --dry-run is used")
        if args.dry_run and args.force:
            raise PacketError("--force has no meaning with --dry-run")
        try:
            blind_key = args.blind_key_file.read_bytes()
        except OSError as exc:
            raise PacketError(f"cannot read blind key file: {exc}") from exc
        public_manifest, reveal = build_packet(
            reports=reports,
            plan_path=args.plan,
            blind_key=blind_key,
        )
        summary = {
            **_validation_summary(reports),
            "state": "DRY_RUN_VALID" if args.dry_run else "WRITTEN",
            "packetId": public_manifest["packetId"],
            "trialCount": len(public_manifest["trials"]),
            "publicSha256": _sha256_bytes(_pretty_json(public_manifest)),
            "revealSha256": _sha256_bytes(_pretty_json(reveal)),
        }
        if not args.dry_run:
            summary["artifacts"] = write_packet(
                args.output_dir,
                public_manifest,
                reveal,
                force=args.force,
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except PacketError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import v2_text_retrieval_deep as deep


def test_result_label_matches_text_and_composed_paths() -> None:
    assert deep.result_label({"text": "sleep", "text_weight": 1.0}) == "sleep"
    assert deep.result_label(
        {
            "text": "sleep",
            "text_weight": 0.18,
            "seeds": [
                {
                    "title": "Drift",
                    "weight": 0.82,
                    "negative": False,
                }
            ],
        }
    ) == "sleep (18%) + Drift (82%)"
    assert deep.result_label(
        {
            "text": "sitar",
            "text_weight": 0.5,
            "seeds": [
                {"title": "Psy", "weight": 0.5, "negative": True}
            ],
        }
    ) == "sitar (50%) - Psy (50%)"


def test_power_mean_endpoints_and_geo() -> None:
    a = np.array([0.9, 0.4], dtype=np.float64)
    b = np.array([0.5, 0.8], dtype=np.float64)
    geo = deep.power_mean_objective((a, b), (0.5, 0.5), 0.0)
    np.testing.assert_allclose(geo, np.sqrt(a * b))
    minimum = deep.power_mean_objective((a, b), (0.5, 0.5), -np.inf)
    np.testing.assert_allclose(minimum, np.minimum(a, b))
    arithmetic = deep.power_mean_objective((a, b), (0.5, 0.5), 1.0)
    np.testing.assert_allclose(arithmetic, (a + b) / 2)


def test_power_mean_ignores_zero_weight_for_strict_min() -> None:
    active = np.array([0.9, 0.4], dtype=np.float64)
    inactive = np.array([0.1, 1.0], dtype=np.float64)
    objective = deep.power_mean_objective((active, inactive), (1.0, 0.0), -np.inf)
    np.testing.assert_allclose(objective, active)


def test_weighted_union_has_proportional_prefixes_and_no_duplicates() -> None:
    track_ids = np.arange(1, 9, dtype=np.int64)
    first = np.array([0.8, 1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    second = np.array([0.9, 0.2, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4])
    selected, origins = deep.weighted_union_indices(
        (first, second),
        (0.75, 0.25),
        track_ids,
        count=4,
        excluded_track_ids=(2,),
    )
    assert len(set(int(track_ids[index]) for index in selected)) == 4
    assert 2 not in {int(track_ids[index]) for index in selected}
    assert origins.tolist().count(0) == 3
    assert origins.tolist().count(1) == 1
    assert origins.tolist() == [0, 0, 1, 0]


def test_percentile_goodness_respects_direction_and_tie_break() -> None:
    scores = np.array([0.2, 0.8, 0.5, 0.5], dtype=np.float32)
    track_ids = np.array([9, 5, 8, 7], dtype=np.int64)
    positive = deep.percentile_goodness(scores, track_ids, positive=True)
    negative = deep.percentile_goodness(scores, track_ids, positive=False)
    assert int(np.argmax(positive)) == 1
    assert int(np.argmax(negative)) == 0
    assert positive[3] < positive[2]


def test_largest_gap_rank() -> None:
    scores = np.array([1.0, 0.9, 0.8, 0.7, 0.69, 0.68, 0.2, 0.19])
    result = deep.largest_gap_rank(scores, minimum=2, maximum=7)
    assert result["rank"] == 6
    assert abs(result["gap"] - 0.48) < 1e-12


def test_parse_phone_usage_fixture(tmp_path: Path) -> None:
    (tmp_path / "shared_prefs").mkdir()
    (tmp_path / "files").mkdir()
    recent = [{"text": "sleep", "text_weight": 1.0}]
    (tmp_path / "shared_prefs" / "settings.xml").write_text(
        "<?xml version='1.0'?><map>"
        f"<string name='recent_searches_v2'>{json.dumps(recent)}</string>"
        "<int name='text_search_top_k' value='30'/></map>",
        encoding="utf-8",
    )
    sessions = [
        {
            "isDirectQueue": True,
            "seedTrack": {"title": "sleep"},
            "tracks": [
                {
                    "track": {
                        "metadataKey": "a",
                        "filenameKey": "a",
                        "filePath": "/a",
                    }
                },
                {
                    "track": {
                        "metadataKey": "a",
                        "filenameKey": "a",
                        "filePath": "/b",
                    }
                },
            ],
            "queuedFileIds": [1],
        }
    ]
    (tmp_path / "files" / "session_history.json").write_text(
        json.dumps(sessions), encoding="utf-8"
    )
    usage, parsed_sessions = deep.parse_phone_usage(tmp_path)
    assert parsed_sessions == sessions
    assert usage["direct_result_queue_count"] == 1
    assert usage["direct_sessions_recoverable_from_current_recent_label"] == 1
    assert usage["identity_duplicate_excess"]["metadataKey"] == 1
    assert usage["direct_unique_id_deficit"] == 1

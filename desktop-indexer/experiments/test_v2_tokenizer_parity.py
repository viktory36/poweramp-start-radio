from __future__ import annotations

import v2_tokenizer_parity as parity


def test_android_piece_encode_uses_best_scored_segmentation() -> None:
    token_ids = {"▁": 6, "a": 7, "b": 8, "▁a": 9, "ab": 10, "▁ab": 11}
    scores = {"▁": -1.0, "a": -1.0, "b": -1.0, "▁a": -0.5, "ab": -0.3, "▁ab": -0.1}
    pieces, ids = parity.android_piece_encode("ab", token_ids, scores, 3)
    assert pieces == ["▁ab"]
    assert ids == [11]


def test_android_piece_encode_only_replaces_literal_space() -> None:
    token_ids = {"▁": 6, "a": 7, "b": 8, "▁a": 9, "▁b": 10}
    scores = {piece: -1.0 for piece in token_ids}
    space_pieces, _ = parity.android_piece_encode("a b", token_ids, scores, 2)
    tab_pieces, tab_ids = parity.android_piece_encode("a\tb", token_ids, scores, 2)
    assert space_pieces == ["▁a", "▁b"]
    assert "\t" in tab_pieces
    assert 3 in tab_ids


def test_v2_checkpoint_normalize_matches_declared_equivalence_pairs() -> None:
    for _, canonical, variant in parity.EQUIVALENCE_PAIRS:
        assert parity.v2_checkpoint_normalize(canonical) == parity.v2_checkpoint_normalize(
            variant
        )

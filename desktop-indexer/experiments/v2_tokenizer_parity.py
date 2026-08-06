#!/usr/bin/env python3
"""Audit retired Android tokenizer approximations against the serialized model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "desktop-indexer" / "models" / "sentencepiece.bpe.model"
DEFAULT_VOCAB = REPO_ROOT / "desktop-indexer" / "models" / "xlm_roberta_vocab.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "text-retrieval-deep"
    / "tokenizer-parity.json"
)


# All cases are BMP-only so Python's code-point slicing matches Kotlin's UTF-16 slicing.
# The pairs encode inputs which the checkpoint normalizer intentionally treats alike.
EQUIVALENCE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("nfc_nfd", "café", "cafe\u0301"),
    ("single_multiple_spaces", "easy listening", "easy   listening"),
    ("space_tab", "easy listening", "easy\tlistening"),
    ("space_newline", "easy listening", "easy\nlistening"),
    ("devanagari_compatibility", "क़व्वाली", "क़व्वाली"),
    ("ascii_fullwidth", "Jazz Ambient", "Ｊａｚｚ Ａｍｂｉｅｎｔ"),
    ("space_nbsp", "sitar electronic", "sitar\u00a0electronic"),
    ("space_em_space", "sitar electronic", "sitar\u2003electronic"),
    ("ascii_ligature", "fi", "ﬁ"),
    ("space_zero_width_space", "left field", "left\u200bfield"),
    ("three_dots_ellipsis", "...", "…"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_vocab(path: Path) -> tuple[dict[str, int], dict[str, float], int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    token_ids: dict[str, int] = {}
    scores: dict[str, float] = {}
    for piece, entry in raw.items():
        if isinstance(entry, list) and len(entry) >= 2:
            token_ids[piece] = int(entry[0])
            scores[piece] = float(entry[1])
        else:
            token_ids[piece] = int(entry)
            scores[piece] = -999.0
    return token_ids, scores, max(map(len, token_ids))


def android_piece_encode(
    text: str,
    token_ids: dict[str, int],
    scores: dict[str, float],
    max_piece_len: int,
) -> tuple[list[str], list[int]]:
    """Port SentencePieceTokenizer.encode through its piece-to-ID step."""
    normalized = "▁" + text.replace(" ", "▁")
    return android_piece_encode_escaped(
        normalized, token_ids, scores, max_piece_len
    )


def android_piece_encode_escaped(
    normalized: str,
    token_ids: dict[str, int],
    scores: dict[str, float],
    max_piece_len: int,
) -> tuple[list[str], list[int]]:
    length = len(normalized)
    dp = [float("-inf")] * (length + 1)
    back_pointer = [0] * (length + 1)
    dp[0] = 0.0

    for end in range(1, length + 1):
        start = max(0, end - max_piece_len)
        for position in range(start, end):
            piece = normalized[position:end]
            score = scores.get(piece)
            if score is not None and dp[position] + score > dp[end]:
                dp[end] = dp[position] + score
                back_pointer[end] = position
        if dp[end] == float("-inf") and dp[end - 1] > float("-inf"):
            dp[end] = dp[end - 1] - 100.0
            back_pointer[end] = end - 1

    pieces: list[str] = []
    position = length
    while position > 0:
        start = back_pointer[position]
        pieces.append(normalized[start:position])
        position = start
    pieces.reverse()
    return pieces, [token_ids.get(piece, 3) for piece in pieces]


def v2_checkpoint_normalize(text: str) -> str:
    nfkc = unicodedata.normalize("NFKC", text)
    output: list[str] = []
    pending_space = False
    checkpoint_spaces = {"\u200b", "\u200c", "\u200d", "\ufeff"}
    for character in nfkc:
        if character.isspace() or character in checkpoint_spaces:
            pending_space = bool(output)
            continue
        if pending_space:
            output.append(" ")
            pending_space = False
        output.append(character)
    return "".join(output)


def v2_android_piece_encode(
    text: str,
    token_ids: dict[str, int],
    scores: dict[str, float],
    max_piece_len: int,
) -> tuple[list[str], list[int]]:
    normalized = v2_checkpoint_normalize(text)
    if not normalized:
        return [], []
    escaped = "▁" + normalized.replace(" ", "▁")
    return android_piece_encode_escaped(escaped, token_ids, scores, max_piece_len)


def compare_pair(
    pair: tuple[str, str, str],
    processor: object,
    token_ids: dict[str, int],
    scores: dict[str, float],
    max_piece_len: int,
) -> dict[str, object]:
    pair_id, canonical, variant = pair

    def encode(value: str) -> dict[str, object]:
        official_pieces = list(processor.encode(value, out_type=str))
        android_pieces, android_ids = android_piece_encode(
            value, token_ids, scores, max_piece_len
        )
        v2_pieces, v2_ids = v2_android_piece_encode(
            value, token_ids, scores, max_piece_len
        )
        return {
            "input": value,
            "official_normalized": processor.normalize(value),
            "official_pieces": official_pieces,
            "android_pieces": android_pieces,
            "android_ids": android_ids,
            "android_matches_official": android_pieces == official_pieces,
            "v2_android_pieces": v2_pieces,
            "v2_android_ids": v2_ids,
            "v2_android_matches_official": v2_pieces == official_pieces,
        }

    left = encode(canonical)
    right = encode(variant)
    return {
        "id": pair_id,
        "canonical": left,
        "variant": right,
        "official_equivalent": left["official_pieces"] == right["official_pieces"],
        "android_equivalent": left["android_pieces"] == right["android_pieces"],
        "v2_android_equivalent": (
            left["v2_android_pieces"] == right["v2_android_pieces"]
        ),
    }


def scan_bmp_approximation(
    processor: object,
    token_ids: dict[str, int],
    scores: dict[str, float],
    max_piece_len: int,
) -> dict[str, object]:
    """Differentially test every non-surrogate BMP scalar in a stable context."""
    mismatches: list[dict[str, object]] = []
    tested = 0
    for code_point in range(0x10000):
        if 0xD800 <= code_point <= 0xDFFF:
            continue
        value = f"a{chr(code_point)}b"
        official = list(processor.encode(value, out_type=str))
        approximation, _ = v2_android_piece_encode(
            value,
            token_ids,
            scores,
            max_piece_len,
        )
        tested += 1
        if official != approximation:
            mismatches.append(
                {
                    "code_point": f"U+{code_point:04X}",
                    "unicode_name": unicodedata.name(chr(code_point), "UNNAMED"),
                    "official_normalized": processor.normalize(value),
                    "official_pieces": official,
                    "retired_approximation_pieces": approximation,
                }
            )
    return {
        "context": "a + scalar + b",
        "tested_scalar_count": tested,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    import sentencepiece as spm

    args = parse_args()
    processor = spm.SentencePieceProcessor(model_file=str(args.model))
    token_ids, scores, max_piece_len = load_vocab(args.vocab)
    pairs = [
        compare_pair(
            pair,
            processor,
            token_ids,
            scores,
            max_piece_len,
        )
        for pair in EQUIVALENCE_PAIRS
    ]
    bmp_differential = scan_bmp_approximation(
        processor,
        token_ids,
        scores,
        max_piece_len,
    )
    result = {
        "contract": (
            "Production V2 loads this serialized model through the official SentencePiece "
            "v0.2.1 runtime. The hand-written normalizer/Viterbi implementation represented "
            "by v2_android_* is retained here only to prove why approximation was retired."
        ),
        "production_runtime": {
            "upstream_tag": "v0.2.1",
            "upstream_commit": "31646a467d2051eb904e0b45de3a73e91fe1c1e3",
            "policy": (
                "official serialized-model normalization and segmentation; SentencePiece "
                "unknown ID 0 maps to XLM-R ID 3, every other emitted ID maps to ID + 1"
            ),
        },
        "model": {
            "path": str(args.model.resolve()),
            "sha256": sha256_file(args.model),
        },
        "vocab": {
            "path": str(args.vocab.resolve()),
            "sha256": sha256_file(args.vocab),
            "piece_count": len(token_ids),
            "max_piece_len": max_piece_len,
        },
        "pair_count": len(pairs),
        "official_equivalent_count": sum(row["official_equivalent"] for row in pairs),
        "android_equivalent_count": sum(row["android_equivalent"] for row in pairs),
        "v2_android_equivalent_count": sum(
            row["v2_android_equivalent"] for row in pairs
        ),
        "canonical_android_match_count": sum(
            row["canonical"]["android_matches_official"] for row in pairs
        ),
        "variant_android_match_count": sum(
            row["variant"]["android_matches_official"] for row in pairs
        ),
        "v2_canonical_android_match_count": sum(
            row["canonical"]["v2_android_matches_official"] for row in pairs
        ),
        "v2_variant_android_match_count": sum(
            row["variant"]["v2_android_matches_official"] for row in pairs
        ),
        "retired_approximation_bmp_differential": bmp_differential,
        "pairs": pairs,
    }
    atomic_json(args.output, result)
    print(f"Complete: {args.output}")


if __name__ == "__main__":
    main()

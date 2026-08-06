#!/usr/bin/env python3
"""Build deterministic host references for V2 CLaMP3 text inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm
from ai_edge_litert.interpreter import Interpreter


SEQ_LEN = 128
BOS_ID = 0
PAD_ID = 1
EOS_ID = 2
UNK_ID = 3
POLICY_ID = (
    "host-text-aggregation-v1:segment128-final-overlap:"
    "token-count-weighted-average:l2"
)

DEFAULT_QUERIES = {
    "short_english": "sleepy dub techno with soft rain and deep sub-bass",
    "short_arabic": "موسيقى إلكترونية هادئة وعميقة للنوم",
    "long_multisection": " ".join(
        [
            "begin with weightless ambient pads distant bells and almost no percussion",
            "move gradually into warm broken beat hand percussion and rounded electric bass",
            "then become nocturnal dub techno with tape echo rain texture and deep sub bass",
            "avoid harsh vocals bright festival drops and aggressive distorted guitars",
        ]
        * 12
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xlm_roberta_id(sentencepiece_id: int) -> int:
    if sentencepiece_id < 0:
        raise ValueError(f"invalid SentencePiece ID {sentencepiece_id}")
    return UNK_ID if sentencepiece_id == 0 else sentencepiece_id + 1


def encode_windows(processor: spm.SentencePieceProcessor, text: str) -> list[tuple[np.ndarray, np.ndarray, int]]:
    ids = [BOS_ID]
    ids.extend(xlm_roberta_id(piece_id) for piece_id in processor.encode(text, out_type=int))
    ids.append(EOS_ID)

    windows: list[tuple[list[int], int]] = []
    if len(ids) <= SEQ_LEN:
        windows.append((ids, len(ids)))
    else:
        full_count, remainder = divmod(len(ids), SEQ_LEN)
        for window_index in range(full_count):
            start = window_index * SEQ_LEN
            windows.append((ids[start : start + SEQ_LEN], SEQ_LEN))
        if remainder:
            windows.append((ids[-SEQ_LEN:], remainder))

    encoded = []
    for window_ids, contribution in windows:
        input_ids = np.full((1, SEQ_LEN), PAD_ID, dtype=np.int64)
        attention_mask = np.zeros((1, SEQ_LEN), dtype=np.int64)
        input_ids[0, : len(window_ids)] = window_ids
        attention_mask[0, : len(window_ids)] = 1
        encoded.append((input_ids, attention_mask, contribution))
    return encoded


def infer(
    interpreter: Interpreter,
    processor: spm.SentencePieceProcessor,
    text: str,
) -> tuple[np.ndarray, list[int]]:
    inputs = interpreter.get_input_details()
    output = interpreter.get_output_details()[0]
    weighted = np.zeros(768, dtype=np.float64)
    contributions: list[int] = []
    for input_ids, attention_mask, contribution in encode_windows(processor, text):
        interpreter.set_tensor(inputs[0]["index"], input_ids)
        interpreter.set_tensor(inputs[1]["index"], attention_mask)
        interpreter.invoke()
        embedding = interpreter.get_tensor(output["index"])[0].astype(np.float64)
        weighted += embedding * contribution
        contributions.append(contribution)
    weighted /= sum(contributions)
    norm = np.linalg.norm(weighted)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("text model returned a non-finite or zero vector")
    return (weighted / norm).astype("<f4"), contributions


def parse_queries(values: list[str]) -> dict[str, str]:
    if not values:
        return DEFAULT_QUERIES
    queries: dict[str, str] = {}
    for value in values:
        name, separator, text = value.partition("=")
        if not separator or not name or not text:
            raise ValueError("--query must be NAME=TEXT")
        queries[name] = text
    return queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("desktop-indexer/models/clamp3_text.tflite"))
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("desktop-indexer/models/sentencepiece.bpe.model"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("desktop-indexer/audit_raw_data/v2-discovery/text-device-parity"),
    )
    parser.add_argument("--query", action="append", default=[], help="NAME=TEXT; repeatable")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    processor = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    interpreter = Interpreter(model_path=str(args.model), num_threads=args.threads)
    interpreter.allocate_tensors()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "model_sha256": sha256(args.model),
        "tokenizer_sha256": sha256(args.tokenizer),
        "aggregation_policy_id": POLICY_ID,
        "sequence_length": SEQ_LEN,
        "queries": [],
    }
    for name, text in parse_queries(args.query).items():
        embedding, contributions = infer(interpreter, processor, text)
        file_name = f"{name}.f32le"
        output = args.output_dir / file_name
        output.write_bytes(embedding.tobytes())
        manifest["queries"].append(
            {
                "name": name,
                "text": text,
                "embedding_file": file_name,
                "embedding_sha256": sha256(output),
                "window_contribution_tokens": contributions,
            }
        )
        print(f"{name}: windows={len(contributions)} weights={contributions}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare phone text embeddings against desktop TFLite FP32 and PyTorch.

Usage:
    # Pull embeddings from phone first:
    adb exec-out run-as com.powerampstartradio cat files/debug_embeddings/text_emb_ethereal_ambient.bin > phone_ethereal_ambient.bin

    # Compare:
    python scripts/compare_text_embeddings.py phone_ethereal_ambient.bin "ethereal ambient"
    python scripts/compare_text_embeddings.py phone_dir/  # compare all .bin files (query from filename)
"""

import sys
import json
import time
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "models"
VOCAB_FILE = MODELS_DIR / "xlm_roberta_vocab.json"
FP32_MODEL = MODELS_DIR / "mulan_text.tflite"
FP16_MODEL = MODELS_DIR / "mulan_text_fp16.tflite"

SEQ_LEN = 64
BOS_ID = 0
PAD_ID = 1
EOS_ID = 2
UNK_ID = 3
SP_SPACE = "\u2581"


class ViterbiTokenizer:
    def __init__(self, vocab_path: Path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.vocab = {}
        self.scores = {}
        self.max_piece_len = 0
        for piece, val in raw.items():
            if isinstance(val, list):
                self.vocab[piece] = val[0]
                self.scores[piece] = val[1]
            else:
                self.vocab[piece] = val
                self.scores[piece] = -999.0
            if len(piece) > self.max_piece_len:
                self.max_piece_len = len(piece)

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        normalized = SP_SPACE + text.replace(" ", SP_SPACE)
        n = len(normalized)
        dp = [float("-inf")] * (n + 1)
        dp[0] = 0.0
        backptr = [0] * (n + 1)
        for i in range(1, n + 1):
            start = max(0, i - self.max_piece_len)
            for j in range(start, i):
                piece = normalized[j:i]
                score = self.scores.get(piece)
                if score is not None and dp[j] + score > dp[i]:
                    dp[i] = dp[j] + score
                    backptr[i] = j
            if dp[i] == float("-inf") and dp[i - 1] > float("-inf"):
                dp[i] = dp[i - 1] + -100.0
                backptr[i] = i - 1
        pieces = []
        pos = n
        while pos > 0:
            start = backptr[pos]
            pieces.append(normalized[start:pos])
            pos = start
        pieces.reverse()
        token_ids = [self.vocab.get(s, UNK_ID) for s in pieces]
        max_tokens = SEQ_LEN - 2
        truncated = token_ids[:max_tokens]
        seq_len_val = len(truncated) + 2
        input_ids = [PAD_ID] * SEQ_LEN
        attention_mask = [0] * SEQ_LEN
        input_ids[0] = BOS_ID
        attention_mask[0] = 1
        for i, tid in enumerate(truncated):
            input_ids[i + 1] = tid
            attention_mask[i + 1] = 1
        input_ids[seq_len_val - 1] = EOS_ID
        attention_mask[seq_len_val - 1] = 1
        return input_ids, attention_mask


def tflite_embed(interp, tokenizer, query):
    ids, mask = tokenizer.encode(query)
    ids_np = np.array([ids], dtype=np.int64)
    mask_np = np.array([mask], dtype=np.int64)
    inp = interp.get_input_details()
    out = interp.get_output_details()
    interp.set_tensor(inp[0]["index"], ids_np)
    interp.set_tensor(inp[1]["index"], mask_np)
    interp.invoke()
    return interp.get_tensor(out[0]["index"]).flatten()


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compare_one(phone_file: Path, query: str, interp, tokenizer):
    phone_emb = np.fromfile(str(phone_file), dtype=np.float32)
    if phone_emb.shape[0] != 512:
        print(f"  ERROR: expected 512 floats, got {phone_emb.shape[0]}")
        return

    desktop_emb = tflite_embed(interp, tokenizer, query)
    cos = cosine(phone_emb, desktop_emb)
    max_diff = np.max(np.abs(phone_emb - desktop_emb))
    norm_phone = np.linalg.norm(phone_emb)
    norm_desktop = np.linalg.norm(desktop_emb)

    status = "PASS" if cos > 0.99 else "CHECK" if cos > 0.95 else "FAIL"
    print(f"  {query:<30} cos={cos:.6f}  max_diff={max_diff:.6f}  "
          f"norm_phone={norm_phone:.4f}  norm_desktop={norm_desktop:.4f}  [{status}]")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <phone_emb.bin|dir/> [query]")
        sys.exit(1)

    path = Path(sys.argv[1])

    from ai_edge_litert.interpreter import Interpreter
    print(f"Loading TFLite FP32: {FP32_MODEL.name}...")
    interp = Interpreter(model_path=str(FP32_MODEL))
    interp.allocate_tensors()
    tokenizer = ViterbiTokenizer(VOCAB_FILE)

    if path.is_dir():
        bins = sorted(path.glob("text_emb_*.bin"))
        if not bins:
            print(f"No text_emb_*.bin files in {path}")
            sys.exit(1)
        print(f"\nComparing {len(bins)} phone embeddings vs desktop TFLite FP32:\n")
        for bf in bins:
            # Extract query from filename: text_emb_<query>.bin
            q = bf.stem.replace("text_emb_", "").replace("_", " ")
            compare_one(bf, q, interp, tokenizer)
    else:
        query = sys.argv[2] if len(sys.argv) > 2 else path.stem.replace("text_emb_", "").replace("_", " ")
        print(f"\nComparing phone embedding vs desktop TFLite FP32:\n")
        compare_one(path, query, interp, tokenizer)


if __name__ == "__main__":
    main()

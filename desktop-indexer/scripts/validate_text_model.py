#!/usr/bin/env python3
"""Validate MuQ-MuLan text tower: TFLite (FP32 + FP16) vs PyTorch, and tokenizer accuracy.

Tests:
1. Tokenizer: our BPE implementation (replicating Android SentencePieceTokenizer.kt)
   vs HuggingFace AutoTokenizer on multiple queries
2. TFLite FP32 vs PyTorch (multiple queries)
3. TFLite FP16 vs PyTorch (multiple queries)
4. Text search against real embeddings.db (top-5 results per query)

Usage:
    python scripts/validate_text_model.py
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Config ──────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"
FP32_MODEL = MODELS_DIR / "mulan_text.tflite"
FP16_MODEL = MODELS_DIR / "mulan_text_fp16.tflite"
VOCAB_FILE = MODELS_DIR / "xlm_roberta_vocab.json"
DB_FILE = Path(__file__).parent.parent / "audit_raw_data" / "embeddings_mulan.db"

SEQ_LEN = 64
BOS_ID = 0
PAD_ID = 1
EOS_ID = 2
UNK_ID = 3
SP_SPACE = "\u2581"

TEST_QUERIES = [
    "ethereal ambient",
    "heavy bass",
    "indian classical",
    "sufi music",
    "upbeat electronic dance",
    "sad piano ballad",
    "jazz fusion",
    "lo-fi hip hop beats",
    "psychedelic",
    "ambient drone",
    "trance",
    "melancholic",
    "synthesizer",
    "IDM",
]


# ── Viterbi tokenizer (mirrors Android SentencePieceTokenizer.kt) ───────────
class ViterbiTokenizer:
    """Pure Python replica of the Android SentencePieceTokenizer.kt (Unigram/Viterbi)."""

    def __init__(self, vocab_path: Path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.vocab = {}    # piece → id
        self.scores = {}   # piece → score (log probability)
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

        # Viterbi DP: find highest-probability segmentation
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
            # UNK fallback for single char
            if dp[i] == float("-inf") and dp[i - 1] > float("-inf"):
                dp[i] = dp[i - 1] + -100.0
                backptr[i] = i - 1

        # Backtrack
        pieces = []
        pos = n
        while pos > 0:
            start = backptr[pos]
            pieces.append(normalized[start:pos])
            pos = start
        pieces.reverse()

        token_ids = [self.vocab.get(s, UNK_ID) for s in pieces]

        # Build padded arrays
        max_tokens = SEQ_LEN - 2
        truncated = token_ids[:max_tokens]
        seq_len = len(truncated) + 2

        input_ids = [PAD_ID] * SEQ_LEN
        attention_mask = [0] * SEQ_LEN

        input_ids[0] = BOS_ID
        attention_mask[0] = 1
        for i, tid in enumerate(truncated):
            input_ids[i + 1] = tid
            attention_mask[i + 1] = 1
        input_ids[seq_len - 1] = EOS_ID
        attention_mask[seq_len - 1] = 1

        return input_ids, attention_mask


# ── TFLite inference helper ─────────────────────────────────────────────────
def load_tflite_interpreter(model_path: Path):
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    return interp


def tflite_infer(interp, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    interp.set_tensor(input_details[0]["index"], input_ids)
    interp.set_tensor(input_details[1]["index"], attention_mask)
    interp.invoke()
    return interp.get_tensor(output_details[0]["index"]).copy()


# ── PyTorch reference ───────────────────────────────────────────────────────
def get_pytorch_embedding(mulan_model, query: str) -> np.ndarray:
    """Get text embedding via the full MuQ-MuLan pipeline."""
    with torch.no_grad():
        out = mulan_model(texts=[query])
    return out[0].cpu().float().numpy()


def get_pytorch_wrapper_embedding(wrapper, tokenizer_hf, query: str) -> np.ndarray:
    """Get text embedding via the wrapper with HuggingFace tokenizer."""
    inputs = tokenizer_hf([query], return_tensors="pt", padding="max_length",
                          max_length=SEQ_LEN, truncation=True)
    with torch.no_grad():
        out = wrapper(inputs["input_ids"], inputs["attention_mask"])
    return out[0].cpu().float().numpy()


# ── Search against DB ────────────────────────────────────────────────────────
def load_mulan_embeddings(db_path: Path):
    """Load all MuLan embeddings from the database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("""
        SELECT t.id, t.artist, t.title, e.embedding
        FROM embeddings_mulan e
        JOIN tracks t ON e.track_id = t.id
    """)
    tracks = []
    embeddings = []
    for row in cursor:
        track_id, artist, title, emb_blob = row
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        tracks.append((track_id, artist, title))
        embeddings.append(emb)
    conn.close()
    return tracks, np.array(embeddings)


def search_top_k(query_emb: np.ndarray, embeddings: np.ndarray, k: int = 5):
    """Brute-force cosine similarity search."""
    scores = embeddings @ query_emb
    top_indices = np.argsort(scores)[::-1][:k]
    return [(idx, scores[idx]) for idx in top_indices]


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("MuQ-MuLan Text Tower Validation")
    print("=" * 70)

    # ── 1. Tokenizer validation ──
    print("\n[1/4] TOKENIZER VALIDATION (BPE vs HuggingFace)")
    print("-" * 50)

    from transformers import AutoTokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    bpe_tokenizer = ViterbiTokenizer(VOCAB_FILE)

    all_match = True
    for query in TEST_QUERIES:
        hf_ids = hf_tokenizer.encode(query)
        bpe_ids, bpe_mask = bpe_tokenizer.encode(query)

        # Extract non-padding tokens from BPE output
        active_len = sum(bpe_mask)
        bpe_active = bpe_ids[:active_len]

        match = hf_ids == bpe_active
        status = "MATCH" if match else "MISMATCH"
        if not match:
            all_match = False
        hf_tokens = hf_tokenizer.convert_ids_to_tokens(hf_ids)
        print(f"  '{query}': {status}  {hf_tokens}")
        if not match:
            bpe_tokens = [hf_tokenizer.convert_ids_to_tokens(i) for i in bpe_active]
            print(f"    HF:  {hf_ids}")
            print(f"    BPE: {bpe_active}")
            print(f"    BPE tokens: {bpe_tokens}")

    print(f"\n  Tokenizer: {'ALL MATCH' if all_match else 'MISMATCHES FOUND'}")

    # ── 2. TFLite FP32 vs PyTorch ──
    print("\n[2/4] PYTORCH vs TFLITE FP32")
    print("-" * 50)

    print("  Loading MuQ-MuLan model...")
    t0 = time.time()
    from muq import MuQMuLan
    mulan_model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    mulan_model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Create wrapper for comparison
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from poweramp_indexer.export_litert import MuLanTextWrapper
    wrapper = MuLanTextWrapper(mulan_model)
    wrapper.eval()

    # Load TFLite FP32
    print(f"  Loading TFLite FP32: {FP32_MODEL.name}...")
    fp32_interp = load_tflite_interpreter(FP32_MODEL)

    input_details = fp32_interp.get_input_details()
    output_details = fp32_interp.get_output_details()
    print(f"  Input 0: shape={input_details[0]['shape']} dtype={input_details[0]['dtype']}")
    print(f"  Input 1: shape={input_details[1]['shape']} dtype={input_details[1]['dtype']}")
    print(f"  Output:  shape={output_details[0]['shape']} dtype={output_details[0]['dtype']}")

    print(f"\n  {'Query':<30} {'Wrap→TFLite':<12} {'Full→TFLite':<12} {'MaxDiff':<10}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*10}")

    pytorch_embeddings = {}
    tflite_fp32_embeddings = {}
    all_fp32_pass = True

    for query in TEST_QUERIES:
        ref_emb = get_pytorch_embedding(mulan_model, query)
        wrapper_emb = get_pytorch_wrapper_embedding(wrapper, hf_tokenizer, query)

        ids, mask = bpe_tokenizer.encode(query)
        ids_np = np.array([ids], dtype=np.int64)
        mask_np = np.array([mask], dtype=np.int64)
        tflite_emb = tflite_infer(fp32_interp, ids_np, mask_np).flatten()

        cos_wt = np.dot(wrapper_emb, tflite_emb) / (
            np.linalg.norm(wrapper_emb) * np.linalg.norm(tflite_emb))
        cos_ft = np.dot(ref_emb, tflite_emb) / (
            np.linalg.norm(ref_emb) * np.linalg.norm(tflite_emb))
        max_diff = np.max(np.abs(wrapper_emb - tflite_emb))

        flag = "" if cos_ft > 0.999 else " <<<FAIL"
        if cos_ft <= 0.999:
            all_fp32_pass = False

        print(f"  {query:<30} {cos_wt:.6f}    {cos_ft:.6f}    {max_diff:.6f}{flag}")

        pytorch_embeddings[query] = ref_emb
        tflite_fp32_embeddings[query] = tflite_emb

    # Free PyTorch model
    del mulan_model, wrapper
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    # ── 3. TFLite FP16 ──
    print(f"\n[3/4] TFLITE FP32 vs TFLITE FP16")
    print("-" * 50)

    if not FP16_MODEL.exists():
        print(f"  SKIP: {FP16_MODEL} not found")
    else:
        print(f"  Loading TFLite FP16: {FP16_MODEL.name}...")
        fp16_interp = load_tflite_interpreter(FP16_MODEL)

        print(f"\n  {'Query':<30} {'Full→FP16':<12} {'FP32→FP16':<12} {'MaxDiff':<10}")
        print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*10}")

        for query in TEST_QUERIES:
            ref_emb = pytorch_embeddings[query]
            fp32_emb = tflite_fp32_embeddings[query]

            ids, mask = bpe_tokenizer.encode(query)
            ids_np = np.array([ids], dtype=np.int64)
            mask_np = np.array([mask], dtype=np.int64)
            fp16_emb = tflite_infer(fp16_interp, ids_np, mask_np).flatten()

            cos_ff = np.dot(ref_emb, fp16_emb) / (
                np.linalg.norm(ref_emb) * np.linalg.norm(fp16_emb))
            cos_32_16 = np.dot(fp32_emb, fp16_emb) / (
                np.linalg.norm(fp32_emb) * np.linalg.norm(fp16_emb))
            max_diff = np.max(np.abs(fp32_emb - fp16_emb))

            print(f"  {query:<30} {cos_ff:.6f}    {cos_32_16:.6f}    {max_diff:.6f}")

    # ── 4. Text search against real DB ──
    print(f"\n[4/4] TEXT SEARCH vs REAL EMBEDDINGS DB")
    print("-" * 50)

    db_path = DB_FILE if DB_FILE.exists() else None
    if db_path is None:
        alt = Path(__file__).parent.parent / "fused.db"
        if alt.exists():
            db_path = alt

    if db_path is None:
        print(f"  SKIP: No embeddings DB found")
    else:
        print(f"  Loading embeddings from {db_path.name}...")
        tracks, all_embs = load_mulan_embeddings(db_path)
        print(f"  Loaded {len(tracks)} tracks, shape: {all_embs.shape}")

        search_queries = ["ethereal ambient", "heavy bass", "indian classical",
                          "psychedelic", "sad piano ballad"]

        for query in search_queries:
            print(f"\n  Query: \"{query}\"")

            ref_emb = pytorch_embeddings[query]
            ref_results = search_top_k(ref_emb, all_embs, k=5)

            tflite_emb = tflite_fp32_embeddings[query]
            tflite_results = search_top_k(tflite_emb, all_embs, k=5)

            ref_ids = [tracks[idx][0] for idx, _ in ref_results]
            tflite_ids = [tracks[idx][0] for idx, _ in tflite_results]
            overlap = len(set(ref_ids) & set(tflite_ids))

            print(f"    Top-5 overlap: {overlap}/5 (PyTorch vs TFLite+BPE)")
            for rank, ((r_idx, r_score), (t_idx, t_score)) in enumerate(
                zip(ref_results, tflite_results), 1
            ):
                r_name = f"{tracks[r_idx][1]} - {tracks[r_idx][2]}"[:50]
                t_name = f"{tracks[t_idx][1]} - {tracks[t_idx][2]}"[:50]
                if r_idx == t_idx:
                    print(f"    #{rank}: {r_score:.4f}  {r_name}")
                else:
                    print(f"    #{rank}: {r_score:.4f} (PT) {r_name}")
                    print(f"         {t_score:.4f} (TF) {t_name}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    tok_status = "PASS" if all_match else "FAIL"
    fp32_status = "PASS" if all_fp32_pass else "FAIL"
    print(f"  Tokenizer:  {tok_status} ({len(TEST_QUERIES)} queries)")
    print(f"  TFLite FP32: {fp32_status}")
    print(f"  Models: {FP32_MODEL.name} ({FP32_MODEL.stat().st_size / 1e6:.0f} MB), "
          f"{FP16_MODEL.name} ({FP16_MODEL.stat().st_size / 1e6:.0f} MB)")
    print(f"  Vocab:  {VOCAB_FILE.name} ({VOCAB_FILE.stat().st_size / 1e3:.0f} KB)")
    print("=" * 70)


if __name__ == "__main__":
    main()

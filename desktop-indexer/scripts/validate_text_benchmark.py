#!/usr/bin/env python3
"""
Validate on-device CLaMP3 text embeddings against desktop PyTorch reference.

Pulls text_benchmark_results.json from the phone, runs the same queries through
the PyTorch CLaMP3 text encoder, and compares:
1. Cosine similarity between on-device and desktop text embeddings
2. Search result ranking overlap (top-10 matches)

Usage:
  # Auto-pull from phone + compare against desktop embeddings DB:
  python validate_text_benchmark.py embeddings_clamp3.db

  # Use a local JSON file:
  python validate_text_benchmark.py embeddings_clamp3.db --json text_benchmark_results.json
"""

import argparse
import json
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


# ─── CLaMP3 constants ─────────────────────────────────────────────────────────

CLAMP3_HIDDEN_SIZE = 768
MAX_TEXT_LENGTH = 128
TEXT_MODEL_NAME = "FacebookAI/xlm-roberta-base"
CLAMP3_WEIGHTS_FILENAME = (
    "weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base"
    "_t_length_128_a_size_768_a_layers_12_a_length_128"
    "_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
)


def blob_to_array(blob):
    n = len(blob) // 4
    return np.array(struct.unpack(f'{n}f', blob), dtype=np.float32)


def load_all_embeddings(db_path):
    """Load all CLaMP3 embeddings and track metadata."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT t.id, t.artist, t.title, e.embedding "
        "FROM tracks t INNER JOIN embeddings_clamp3 e ON t.id = e.track_id"
    ).fetchall()
    conn.close()

    track_ids = []
    track_labels = []
    embeddings = []
    for row in rows:
        track_ids.append(row['id'])
        track_labels.append(f"{row['artist'] or '?'} - {row['title'] or '?'}")
        embeddings.append(blob_to_array(row['embedding']))

    emb_matrix = np.stack(embeddings)  # [N, 768]
    return track_ids, track_labels, emb_matrix


def load_clamp3_text_encoder(device='cpu'):
    """Load CLaMP3 text encoder from HuggingFace checkpoint."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(TEXT_MODEL_NAME)
    text_model = AutoModel.from_config(config)
    text_proj = torch.nn.Linear(config.hidden_size, CLAMP3_HIDDEN_SIZE)

    weights_path = hf_hub_download("sander-wood/clamp3", CLAMP3_WEIGHTS_FILENAME)
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
    full_state = checkpoint['model']

    text_model_state = {}
    text_proj_state = {}
    for k, v in full_state.items():
        if k.startswith('text_model.'):
            text_model_state[k[len('text_model.'):]] = v
        elif k.startswith('text_proj.'):
            text_proj_state[k[len('text_proj.'):]] = v

    text_model.load_state_dict(text_model_state)
    text_proj.load_state_dict(text_proj_state)
    text_model.to(device).eval()
    text_proj.to(device).eval()
    print(f"CLaMP3 text encoder loaded (epoch {checkpoint.get('epoch', '?')})")
    return text_model, text_proj


@torch.no_grad()
def encode_text(text_model, text_proj, tokenizer, query, device='cpu'):
    """Encode a text query → 768d L2-normalized embedding."""
    # CLaMP3 text preprocessing
    lines = list(set(query.split("\n")))
    lines = [c for c in lines if len(c) > 0]
    text = tokenizer.sep_token.join(lines)

    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens['input_ids'].squeeze(0)

    # Segment into MAX_TEXT_LENGTH windows
    segment_list = []
    for i in range(0, len(input_ids), MAX_TEXT_LENGTH):
        segment_list.append(input_ids[i:i + MAX_TEXT_LENGTH])
    segment_list[-1] = input_ids[-MAX_TEXT_LENGTH:]

    hidden_states_list = []
    for seg in segment_list:
        actual_len = seg.size(0)
        attention_mask = torch.zeros(MAX_TEXT_LENGTH)
        attention_mask[:actual_len] = 1.0
        if actual_len < MAX_TEXT_LENGTH:
            pad = torch.full(
                (MAX_TEXT_LENGTH - actual_len,),
                tokenizer.pad_token_id, dtype=torch.long,
            )
            seg = torch.cat([seg, pad], dim=0)

        features = text_model(
            input_ids=seg.unsqueeze(0).to(device),
            attention_mask=attention_mask.unsqueeze(0).to(device),
        )['last_hidden_state']

        masks = attention_mask.unsqueeze(0).unsqueeze(-1).to(device)
        features = features * masks
        pooled = features.sum(dim=1) / masks.sum(dim=1)
        projected = text_proj(pooled)
        hidden_states_list.append(projected)

    # Weighted average
    total_tokens = len(input_ids)
    full_cnt = total_tokens // MAX_TEXT_LENGTH
    remain = total_tokens % MAX_TEXT_LENGTH
    if remain == 0:
        weights = torch.tensor([MAX_TEXT_LENGTH] * full_cnt, device=device).view(-1, 1)
    else:
        weights = torch.tensor(
            [MAX_TEXT_LENGTH] * full_cnt + [remain], device=device
        ).view(-1, 1)

    stacked = torch.cat(hidden_states_list, dim=0)
    text_emb = (stacked * weights).sum(0) / weights.sum()

    # L2 normalize
    text_emb = text_emb / text_emb.norm()

    return text_emb.cpu().numpy()


def pull_benchmark_json():
    """Pull text_benchmark_results.json from connected phone."""
    try:
        result = subprocess.run(
            ["adb", "shell", "run-as", "com.powerampstartradio",
             "cat", "files/text_benchmark_results.json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        print(f"adb pull failed: {result.stderr.strip()}")
    except Exception as e:
        print(f"adb pull failed: {e}")
    return None


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def main():
    parser = argparse.ArgumentParser(description="Validate text benchmark vs desktop")
    parser.add_argument("db", help="Path to embeddings_clamp3.db")
    parser.add_argument("--json", help="Path to text_benchmark_results.json (default: pull from phone)")
    parser.add_argument("--device", default="cpu", help="PyTorch device (cpu/cuda)")
    args = parser.parse_args()

    # Load on-device results
    if args.json:
        with open(args.json) as f:
            device_results = json.load(f)
    else:
        print("Pulling text_benchmark_results.json from phone...")
        device_results = pull_benchmark_json()
        if device_results is None:
            print("ERROR: Could not pull results. Use --json to specify local file.")
            return 1

    queries = device_results.get("queries", [])
    print(f"On-device results: {len(queries)} queries")
    print(f"  Text model: {device_results.get('textModel', '?')}")
    print(f"  Accelerator: {device_results.get('textAccelerator', '?')}")
    print(f"  Audio tracks: {device_results.get('numAudioTracks', '?')}")
    print()

    # Load desktop text encoder
    from transformers import AutoTokenizer
    print("Loading desktop CLaMP3 text encoder...")
    text_model, text_proj = load_clamp3_text_encoder(args.device)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    # Load audio embeddings for search comparison
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}")
        return 1

    print(f"Loading audio embeddings from {db_path}...")
    track_ids, track_labels, emb_matrix = load_all_embeddings(db_path)
    print(f"  {len(track_ids)} tracks loaded")
    print()

    # Compare each query
    print("=" * 90)
    print(f"{'Query':<30} {'Cosine':>8} {'Device ms':>10} {'Desktop ms':>11} {'Rank Overlap':>13}")
    print("-" * 90)

    all_cosines = []

    for qr in queries:
        query = qr["query"]
        if qr.get("error"):
            print(f"{query:<30} {'FAILED':>8}")
            continue

        device_emb = np.array(qr["embedding"], dtype=np.float32)

        # Generate desktop embedding
        t0 = time.time()
        desktop_emb = encode_text(text_model, text_proj, tokenizer, query, args.device)
        desktop_ms = (time.time() - t0) * 1000

        # Cosine similarity between device and desktop embeddings
        cos = cosine_sim(device_emb, desktop_emb)
        all_cosines.append(cos)

        # Desktop search results
        sims = emb_matrix @ desktop_emb  # dot product (L2-normalized = cosine)
        top_desktop = np.argsort(sims)[::-1][:10]
        desktop_top_ids = set(track_ids[i] for i in top_desktop)

        # Device search results
        device_top_ids = set()
        if qr.get("topMatches"):
            device_top_ids = {hit["trackId"] for hit in qr["topMatches"]}

        overlap = len(desktop_top_ids & device_top_ids)

        device_ms = qr.get("totalMs", 0)
        print(f"{query:<30} {cos:>8.6f} {device_ms:>9}ms {desktop_ms:>10.1f}ms {overlap:>8}/10")

        # Show top-5 from each side
        if cos < 0.99 or overlap < 7:
            print(f"  Desktop top-5:")
            for i, idx in enumerate(top_desktop[:5]):
                print(f"    {i+1}. {sims[idx]:.4f}  {track_labels[idx]}")
            if qr.get("topMatches"):
                print(f"  Device top-5:")
                for i, hit in enumerate(qr["topMatches"][:5]):
                    print(f"    {i+1}. {hit['similarity']:.4f}  {hit['label']}")

    print("-" * 90)
    if all_cosines:
        avg_cos = np.mean(all_cosines)
        min_cos = np.min(all_cosines)
        max_cos = np.max(all_cosines)
        print(f"{'EMBEDDING COSINE':.<30} avg={avg_cos:.6f}  min={min_cos:.6f}  max={max_cos:.6f}")
        print()
        if avg_cos >= 0.99:
            print("PASS: On-device text embeddings closely match desktop (cosine >= 0.99)")
        elif avg_cos >= 0.95:
            print("WARN: Moderate deviation (0.95 <= cosine < 0.99) — check GPU precision")
        else:
            print("FAIL: Large deviation (cosine < 0.95) — possible model/tokenizer mismatch")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

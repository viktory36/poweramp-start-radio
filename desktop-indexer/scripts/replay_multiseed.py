#!/usr/bin/env python3
"""
Replay MULTISEED_QUERY JSON from Android logcat on desktop.

Implements the same Geo Mean of Percentiles algorithm as GeoMeanSelector.kt
to verify on-device results match desktop.

Usage:
  # Pipe from logcat (text seeds re-encoded with CLaMP3 text encoder)
  adb logcat -s MultiSeed:I -d | python scripts/replay_multiseed.py DB

  # Or paste JSON directly
  python scripts/replay_multiseed.py DB --query '{"text":"indian classical","text_weight":0.27,...}'

  # Skip text encoder (song seeds only, original behavior)
  adb logcat -s MultiSeed:I -d | python scripts/replay_multiseed.py DB --no-text
"""

import argparse
import json
import re
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

# ─── CLaMP3 text encoder (lazy-loaded) ───────────────────────────────────────

CLAMP3_HIDDEN_SIZE = 768
MAX_TEXT_LENGTH = 128
TEXT_MODEL_NAME = "FacebookAI/xlm-roberta-base"
CLAMP3_WEIGHTS_FILENAME = (
    "weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base"
    "_t_length_128_a_size_768_a_layers_12_a_length_128"
    "_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
)

_text_encoder = None
_tokenizer = None


def get_text_encoder(device):
    global _text_encoder, _tokenizer
    if _text_encoder is None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        weights_path = hf_hub_download("sander-wood/clamp3", CLAMP3_WEIGHTS_FILENAME)
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        full_state = checkpoint['model']

        config = AutoConfig.from_pretrained(TEXT_MODEL_NAME)
        text_model = AutoModel.from_config(config)
        text_proj = torch.nn.Linear(config.hidden_size, CLAMP3_HIDDEN_SIZE)

        text_model_state = {k[len('text_model.'):]: v for k, v in full_state.items() if k.startswith('text_model.')}
        text_proj_state = {k[len('text_proj.'):]: v for k, v in full_state.items() if k.startswith('text_proj.')}
        text_model.load_state_dict(text_model_state)
        text_proj.load_state_dict(text_proj_state)
        text_model.to(device).eval()
        text_proj.to(device).eval()

        _text_encoder = (text_model, text_proj)
        _tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        print(f"  CLaMP3 text encoder loaded on {device}")

    return _text_encoder, _tokenizer


def encode_text(query, device):
    """Encode text query to 768d CLaMP3 embedding (L2-normalized)."""
    import torch

    (text_model, text_proj), tokenizer = get_text_encoder(device)

    lines = list(set(query.split("\n")))
    lines = [c for c in lines if len(c) > 0]
    text = tokenizer.sep_token.join(lines)
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens['input_ids'].squeeze(0)

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
            pad = torch.full((MAX_TEXT_LENGTH - actual_len,), tokenizer.pad_token_id, dtype=torch.long)
            seg = torch.cat([seg, pad], dim=0)

        with torch.no_grad():
            features = text_model(
                input_ids=seg.unsqueeze(0).to(device),
                attention_mask=attention_mask.unsqueeze(0).to(device),
            )['last_hidden_state']
            masks = attention_mask.unsqueeze(0).unsqueeze(-1).float().to(device)
            features = features * masks
            pooled = features.sum(dim=1) / masks.sum(dim=1)
            feat = text_proj(pooled)
        hidden_states_list.append(feat)

    total_tokens = len(input_ids)
    full_cnt = total_tokens // MAX_TEXT_LENGTH
    remain = total_tokens % MAX_TEXT_LENGTH
    if remain == 0:
        weights = torch.tensor([MAX_TEXT_LENGTH] * full_cnt, device=device).view(-1, 1)
    else:
        weights = torch.tensor([MAX_TEXT_LENGTH] * full_cnt + [remain], device=device).view(-1, 1)

    stacked = torch.cat(hidden_states_list, dim=0)
    text_emb = (stacked * weights).sum(0) / weights.sum()

    # L2-normalize to match audio embeddings
    text_emb = text_emb / torch.norm(text_emb)
    return text_emb.cpu().numpy().astype(np.float32)  # [768]


# ─── DB + algorithm ──────────────────────────────────────────────────────────

def blob_to_array(blob):
    n = len(blob) // 4
    return np.array(struct.unpack(f'{n}f', blob), dtype=np.float32)


def load_embeddings(db_path):
    """Load all tracks and embeddings from DB."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT t.id, t.artist, t.title, e.embedding "
        "FROM tracks t INNER JOIN embeddings_clamp3 e ON t.id = e.track_id"
    ).fetchall()
    conn.close()

    track_ids = []
    tracks = {}
    embeddings = []
    for row in rows:
        tid = row['id']
        track_ids.append(tid)
        tracks[tid] = {'artist': row['artist'] or '', 'title': row['title'] or ''}
        embeddings.append(blob_to_array(row['embedding']))

    emb_matrix = np.vstack(embeddings)  # [N, 768]
    return track_ids, tracks, emb_matrix


def geo_mean_ranking(track_ids, emb_matrix, seed_embeddings, seed_weights, exclude_ids, top_k=30):
    """Geo Mean of Percentiles — matches GeoMeanSelector.kt exactly."""
    n = len(track_ids)
    abs_total = sum(abs(w) for w in seed_weights)
    if abs_total < 1e-8:
        return []

    log_percentiles = np.zeros(n, dtype=np.float64)

    for emb, weight in zip(seed_embeddings, seed_weights):
        norm_w = abs(weight) / abs_total

        # Cosine similarity (embeddings are L2-normalized)
        sims = emb_matrix @ emb
        if weight < 0:
            sims = -sims

        # Rank: argsort ascending, then invert to get rank per track
        sorted_indices = np.argsort(sims)
        ranks = np.empty(n, dtype=np.int64)
        ranks[sorted_indices] = np.arange(n)

        # log(percentile) = log((rank + 1) / N)
        log_n = np.log(n)
        log_percentiles += norm_w * (np.log(ranks + 1) - log_n)

    # Convert to geo mean score
    scores = np.exp(log_percentiles)

    # Exclude seeds, find top-K
    exclude_set = set(exclude_ids)
    scored = []
    for i, tid in enumerate(track_ids):
        if tid in exclude_set:
            continue
        scored.append((tid, scores[i]))

    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def replay_query(query_json, track_ids, tracks, emb_matrix, id_to_index, top_k=30, skip_text=False):
    """Replay a single MULTISEED_QUERY and return results."""
    seed_embeddings = []
    seed_weights = []
    exclude_ids = set()
    labels = []

    # Text seed
    has_text = 'text' in query_json and query_json['text']
    text_weight = query_json.get('text_weight', 0)
    text_negative = query_json.get('text_negative', False)
    text_skipped = False

    if has_text:
        if skip_text:
            print(f"  TEXT SKIPPED (--no-text): '{query_json['text']}' (weight={text_weight:.4f})")
            print(f"  Comparing song-seed-only ranking (text influence excluded)")
            labels.append(f"[TEXT SKIPPED] \"{query_json['text']}\"")
            text_skipped = True
        else:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            text_query = query_json['text']
            effective_weight = -text_weight if text_negative else text_weight
            print(f"  Encoding text: \"{text_query}\" ...")
            text_emb = encode_text(text_query, device)
            seed_embeddings.append(text_emb)
            seed_weights.append(effective_weight)
            sign = "+" if effective_weight >= 0 else "−"
            labels.append(f"{sign} {abs(effective_weight):.4f} × text: \"{text_query}\"")

    # Song seeds — look up by track_id
    for seed in query_json.get('seeds', []):
        tid = seed['track_id']
        weight = seed['weight']
        if tid not in id_to_index:
            print(f"  WARNING: track_id {tid} not found in DB, skipping")
            continue
        idx = id_to_index[tid]
        seed_embeddings.append(emb_matrix[idx])
        seed_weights.append(weight)
        exclude_ids.add(tid)
        sign = "+" if weight >= 0 else "−"
        labels.append(f"{sign} {abs(weight):.4f} × {seed.get('artist', '?')} - {seed.get('title', '?')} (id={tid})")

    if not seed_embeddings:
        print("  No replayable seeds (all text or missing)")
        return None, None

    print("  Seeds:")
    for lbl in labels:
        print(f"    {lbl}")

    ranking = geo_mean_ranking(track_ids, emb_matrix, seed_embeddings, seed_weights, exclude_ids, top_k)
    return ranking, (has_text and text_skipped)


def compare_results(desktop_ranking, app_results, tracks, has_text):
    """Compare desktop vs app results."""
    desktop_ids = [tid for tid, _ in desktop_ranking]
    app_ids = [r['track_id'] for r in app_results]

    if has_text:
        print("\n  NOTE: Text seed was skipped (--no-text) — rankings WILL differ due to text influence.")
        print("  Showing desktop (song-only) vs app (text+song) for qualitative comparison.\n")

    print(f"  {'Rank':<5} {'Desktop (score)':<45} {'App (score)':<45} {'Match'}")
    print(f"  {'─'*5} {'─'*45} {'─'*45} {'─'*5}")

    max_show = max(len(desktop_ids), len(app_ids), 30)
    matches = 0
    for i in range(min(max_show, 30)):
        d_str = ""
        a_str = ""
        if i < len(desktop_ranking):
            tid, score = desktop_ranking[i]
            t = tracks.get(tid, {})
            d_str = f"{t.get('artist', '?')} - {t.get('title', '?')} ({score:.4f})"
        if i < len(app_results):
            r = app_results[i]
            a_str = f"{r.get('artist', '?')} - {r.get('title', '?')} ({r['score']:.4f})"

        match = ""
        if i < len(desktop_ids) and i < len(app_ids):
            if desktop_ids[i] == app_ids[i]:
                match = "✓"
                matches += 1
            else:
                match = "✗"

        print(f"  {i:<5} {d_str:<45} {a_str:<45} {match}")

    total = min(len(desktop_ids), len(app_ids), 30)
    if total > 0:
        print(f"\n  Position matches: {matches}/{total} ({100*matches/total:.0f}%)")

    # Set overlap (order-independent)
    d_set = set(desktop_ids[:30])
    a_set = set(app_ids[:30])
    overlap = d_set & a_set
    print(f"  Set overlap (top-30): {len(overlap)}/{min(len(d_set), len(a_set))}")


def main():
    parser = argparse.ArgumentParser(description="Replay MULTISEED_QUERY from logcat")
    parser.add_argument("db", help="Path to embeddings.db")
    parser.add_argument("--query", help="JSON string (MULTISEED_QUERY value)")
    parser.add_argument("-k", "--top-k", type=int, default=30)
    parser.add_argument("--no-text", action="store_true",
                        help="Skip text seeds (song seeds only, no CLaMP3 text encoder needed)")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: Database not found: {args.db}")
        return 1

    # Parse queries from stdin or --query
    queries = []
    results_map = {}  # query_index -> app results

    if args.query:
        queries.append(json.loads(args.query))
    else:
        # Parse from stdin (logcat output)
        current_query = None
        for line in sys.stdin:
            m = re.search(r'MULTISEED_QUERY: (.+)', line)
            if m:
                current_query = json.loads(m.group(1))
                queries.append(current_query)
                continue
            m = re.search(r'MULTISEED_RESULTS: (.+)', line)
            if m and queries:
                try:
                    results_map[len(queries) - 1] = json.loads(m.group(1))
                except json.JSONDecodeError:
                    print(f"  WARNING: MULTISEED_RESULTS truncated by logcat (line {len(queries)}), skipping comparison")

    if not queries:
        print("No MULTISEED_QUERY found in input")
        return 1

    print(f"Loading embeddings from {args.db}...")
    track_ids, tracks, emb_matrix = load_embeddings(args.db)
    print(f"Loaded {len(track_ids)} tracks\n")

    id_to_index = {tid: i for i, tid in enumerate(track_ids)}

    for qi, query in enumerate(queries):
        print(f"{'='*90}")
        print(f"Query {qi + 1}:")
        ranking, has_text = replay_query(query, track_ids, tracks, emb_matrix, id_to_index, args.top_k, skip_text=args.no_text)
        if ranking is None:
            continue

        if qi in results_map:
            compare_results(ranking, results_map[qi], tracks, has_text)
        else:
            print(f"\n  Desktop results (no app results to compare):")
            for i, (tid, score) in enumerate(ranking):
                t = tracks.get(tid, {})
                print(f"  {i:<4} {score:.4f}  {t.get('artist', '?')} - {t.get('title', '?')}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

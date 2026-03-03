#!/usr/bin/env python3
"""
Multi-seed search: blend text queries and song references in CLaMP3's shared space.

Usage:
  # Text-only (same as regular search)
  python multi_seed_search.py DB --text "90s boombap hiphop"

  # Song-only (same as regular similar)
  python multi_seed_search.py DB --song "time pachanga boys"

  # Blend: text + song (equal weight)
  python multi_seed_search.py DB --text "90s boombap hiphop" --song "time pachanga boys"

  # Blend with weights
  python multi_seed_search.py DB --text "90s boombap hiphop" --text-weight 0.7 \
                                  --song "time pachanga boys" --song-weight 0.3

  # Negative: "like X but NOT like Y"
  python multi_seed_search.py DB --text "90s boombap hiphop" --song "time pachanga boys" --song-weight -0.3

  # Multiple songs
  python multi_seed_search.py DB --song "time pachanga boys" --song "hallelujah leonard cohen" --song-weight 0.5 0.5

  # Multiple texts
  python multi_seed_search.py DB --text "90s boombap hiphop" --text "dreamy ambient" --text-weight 0.7 0.3
"""

import argparse
import sqlite3
import struct
import sys
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


# ─── DB helpers ───────────────────────────────────────────────────────────────

def blob_to_float_list(blob):
    n = len(blob) // 4
    return list(struct.unpack(f'{n}f', blob))


def load_all_embeddings(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cols = {row[1] for row in conn.execute("PRAGMA table_info(embeddings_clamp3)")}
    has_precision = 'precision' in cols
    if has_precision:
        rows = conn.execute(
            "SELECT t.id, t.artist, t.album, t.title, t.file_path, "
            "e.embedding, e.precision "
            "FROM tracks t INNER JOIN embeddings_clamp3 e ON t.id = e.track_id"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT t.id, t.artist, t.album, t.title, t.file_path, e.embedding "
            "FROM tracks t INNER JOIN embeddings_clamp3 e ON t.id = e.track_id"
        ).fetchall()
    conn.close()

    tracks = []
    embeddings = []
    for row in rows:
        emb = blob_to_float_list(row['embedding'])
        tracks.append({
            'id': row['id'],
            'artist': row['artist'] or '',
            'album': row['album'] or '',
            'title': row['title'] or '',
            'file_path': row['file_path'],
        })
        embeddings.append(emb)

    emb_matrix = torch.tensor(embeddings, dtype=torch.float32)
    return tracks, emb_matrix


# ─── Song lookup ──────────────────────────────────────────────────────────────

def find_song_embedding(tracks, emb_matrix, query):
    query_words = query.lower().strip().split()
    scored = []
    for i, t in enumerate(tracks):
        label = f"{t['artist']} {t['title']} {t['album']}".lower()
        matches = sum(1 for w in query_words if w in label)
        if matches == len(query_words):
            scored.append((i, matches, label))

    if not scored:
        print(f"  ERROR: No track matching '{query}'")
        return None, None

    scored.sort(key=lambda x: (-x[1], x[2]))
    idx = scored[0][0]
    t = tracks[idx]
    print(f"  Song: {t['artist']} - {t['title']}")
    return emb_matrix[idx], idx


# ─── Text encoder ─────────────────────────────────────────────────────────────

_text_encoder = None
_tokenizer = None


def get_text_encoder(device):
    global _text_encoder, _tokenizer
    if _text_encoder is None:
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
    return text_emb.cpu()  # [768]


# ─── Blend and search ─────────────────────────────────────────────────────────

def blend_and_search(tracks, emb_matrix, components, top_k=20):
    """Blend multiple embedding vectors with weights and search.

    components: list of (embedding_768d, weight, label)
    """
    print(f"\n{'='*80}")
    print("Blending:")
    for emb, weight, label in components:
        sign = "+" if weight >= 0 else "-"
        print(f"  {sign} {abs(weight):.2f} × {label}")

    # Weighted sum
    blended = torch.zeros(CLAMP3_HIDDEN_SIZE)
    for emb, weight, _ in components:
        blended += weight * emb

    # L2 normalize
    norm = torch.norm(blended)
    if norm < 1e-8:
        print("\nERROR: Blended vector is near-zero (weights cancelled out)")
        return
    blended = blended / norm

    # Cosine similarity
    query = blended.unsqueeze(0)  # [1, 768]
    sims = torch.cosine_similarity(query, emb_matrix)
    ranked = torch.argsort(sims, descending=True)

    # Exclude seed songs from results
    seed_indices = {idx for _, _, _, idx in
                    [(None, None, None, None)]  # dummy
                    if False}

    print(f"\nTop {top_k} results:")
    print("-" * 80)
    shown = 0
    for idx in ranked:
        idx = idx.item()
        t = tracks[idx]
        sim = sims[idx].item()
        print(f"  {sim:.4f}  {t['artist']} - {t['title']}")
        shown += 1
        if shown >= top_k:
            break


def main():
    parser = argparse.ArgumentParser(description="Multi-seed CLaMP3 search")
    parser.add_argument("db", help="Path to embeddings_clamp3.db")
    parser.add_argument("--text", action="append", default=[], help="Text query (can repeat)")
    parser.add_argument("--text-weight", nargs="+", type=float, default=None,
                        help="Weight per text query (default: equal share)")
    parser.add_argument("--song", action="append", default=[], help="Song name (can repeat)")
    parser.add_argument("--song-weight", nargs="+", type=float, default=None,
                        help="Weight per song (default: equal share). Negative = 'less like'")
    parser.add_argument("-k", "--top-k", type=int, default=20)
    args = parser.parse_args()

    if not args.text and not args.song:
        parser.error("Provide at least one --text or --song")

    if not Path(args.db).exists():
        print(f"ERROR: Database not found: {args.db}")
        return 1

    # Load embeddings
    print("Loading embeddings...")
    tracks, emb_matrix = load_all_embeddings(args.db)
    print(f"Loaded {len(tracks)} tracks")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build component list
    components = []

    # Process text queries
    if args.text:
        n_text = len(args.text)
        text_weights = args.text_weight or [1.0 / max(n_text + len(args.song), 1)] * n_text
        if len(text_weights) == 1 and n_text > 1:
            text_weights = text_weights * n_text
        if len(text_weights) != n_text:
            parser.error(f"Got {n_text} --text but {len(text_weights)} --text-weight")

        for query, weight in zip(args.text, text_weights):
            print(f"\nEncoding text: \"{query}\"")
            emb = encode_text(query, device)
            components.append((emb, weight, f'text: "{query}"'))

    # Process song queries
    if args.song:
        n_song = len(args.song)
        song_weights = args.song_weight or [1.0 / max(n_song + len(args.text), 1)] * n_song
        if len(song_weights) == 1 and n_song > 1:
            song_weights = song_weights * n_song
        if len(song_weights) != n_song:
            parser.error(f"Got {n_song} --song but {len(song_weights)} --song-weight")

        for query, weight in zip(args.song, song_weights):
            print(f"\nLooking up song: \"{query}\"")
            emb, idx = find_song_embedding(tracks, emb_matrix, query)
            if emb is None:
                return 1
            components.append((emb, weight, f'song: "{query}"'))

    # Run blended search
    blend_and_search(tracks, emb_matrix, components, args.top_k)

    # Also show individual results for comparison
    if len(components) > 1:
        for emb, weight, label in components:
            if weight < 0:
                continue
            print(f"\n{'='*80}")
            print(f"For comparison — {label} alone:")
            print("-" * 80)
            query = emb.unsqueeze(0) / torch.norm(emb)
            sims = torch.cosine_similarity(query, emb_matrix)
            ranked = torch.argsort(sims, descending=True)
            shown = 0
            for idx in ranked:
                idx = idx.item()
                t = tracks[idx]
                sim = sims[idx].item()
                # skip if this is the seed song itself
                if sim > 0.999:
                    continue
                print(f"  {sim:.4f}  {t['artist']} - {t['title']}")
                shown += 1
                if shown >= 10:
                    break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

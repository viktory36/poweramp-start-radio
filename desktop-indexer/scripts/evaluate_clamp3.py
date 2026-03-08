#!/usr/bin/env python3
"""
Evaluate CLaMP3 embeddings: similar-track search and text-to-audio search.

Usage:
  # First, complete Phase 2 to populate the DB from cached MERT features:
  python generate_clamp3_embeddings.py /path/to/music -o embeddings_clamp3.db --phase 2

  # Find tracks similar to a given track (by name)
  python evaluate_clamp3.py embeddings_clamp3.db similar "time pachanga boys"

  # Text-to-audio search
  python evaluate_clamp3.py embeddings_clamp3.db search "space rock"
"""

import argparse
import sqlite3
import struct
import time
from pathlib import Path

import numpy as np
import torch
import math


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
    """Load all CLaMP3 embeddings and track metadata from the DB."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

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

    emb_matrix = torch.tensor(embeddings, dtype=torch.float32)  # [N, 768]
    return tracks, emb_matrix


# ─── Similar track search ────────────────────────────────────────────────────

def cmd_similar(db_path, query, top_k=20):
    """Find tracks similar to a given track (fuzzy name match)."""
    tracks, emb_matrix = load_all_embeddings(db_path)
    print(f"Loaded {len(tracks)} tracks with embeddings")

    query_lower = query.lower().strip()
    query_words = query_lower.split()

    # Score each track by how many query words appear in artist+title+album
    # ALL query words must match (same behavior as CLI's search_tracks)
    scored = []
    for i, t in enumerate(tracks):
        label = f"{t['artist']} {t['title']} {t['album']}".lower()
        matches = sum(1 for w in query_words if w in label)
        if matches == len(query_words):
            scored.append((i, matches, label))

    if not scored:
        print(f"No track matching '{query}' found.")
        return

    scored.sort(key=lambda x: (-x[1], x[2]))

    # Show top candidate matches
    if len(scored) > 1:
        print(f"\nTop name matches for '{query}':")
        for idx, cnt, label in scored[:5]:
            t = tracks[idx]
            print(f"  [{cnt} words] {t['artist']} - {t['title']}")
        print()

    seed_idx = scored[0][0]
    seed = tracks[seed_idx]
    print(f"Seed track: {seed['artist']} - {seed['title']}")

    # Cosine similarity
    seed_emb = emb_matrix[seed_idx].unsqueeze(0)  # [1, 768]
    sims = torch.cosine_similarity(seed_emb, emb_matrix)  # [N]
    ranked = torch.argsort(sims, descending=True)

    print(f"\nTop {top_k} similar tracks:")
    print("-" * 80)
    shown = 0
    for idx in ranked:
        idx = idx.item()
        if idx == seed_idx:
            continue
        t = tracks[idx]
        sim = sims[idx].item()
        print(f"  {sim:.4f}  {t['artist']} - {t['title']}")
        shown += 1
        if shown >= top_k:
            break


# ─── CLaMP3 text encoder ─────────────────────────────────────────────────────

class CLaMP3TextEncoder(torch.nn.Module):
    """Minimal CLaMP3 text encoder: fine-tuned XLM-RoBERTa + linear projection."""

    def __init__(self):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        # Create architecture without downloading pretrained weights
        config = AutoConfig.from_pretrained(TEXT_MODEL_NAME)
        self.text_model = AutoModel.from_config(config)
        self.text_proj = torch.nn.Linear(config.hidden_size, CLAMP3_HIDDEN_SIZE)

    @torch.no_grad()
    def encode(self, input_ids, attention_mask):
        """Encode tokenized text → 768d embedding.

        Args:
            input_ids:      [B, seq_len] token IDs
            attention_mask:  [B, seq_len] (1=real, 0=pad)

        Returns:
            [B, 768] projected features (not L2-normalized, matching CLaMP3)
        """
        features = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )['last_hidden_state']
        # Masked average pooling (matches CLaMP3Model.avg_pooling)
        masks = attention_mask.unsqueeze(-1).float()
        features = features * masks
        pooled = features.sum(dim=1) / masks.sum(dim=1)
        return self.text_proj(pooled)

    @classmethod
    def from_clamp3_checkpoint(cls, weights_path, device='cpu'):
        """Load text_model + text_proj from a full CLaMP3 checkpoint."""
        model = cls()
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        full_state = checkpoint['model']

        text_model_state = {}
        text_proj_state = {}
        for k, v in full_state.items():
            if k.startswith('text_model.'):
                text_model_state[k[len('text_model.'):]] = v
            elif k.startswith('text_proj.'):
                text_proj_state[k[len('text_proj.'):]] = v

        model.text_model.load_state_dict(text_model_state)
        model.text_proj.load_state_dict(text_proj_state)
        model.to(device).eval()
        print(f"CLaMP3 text encoder loaded (epoch {checkpoint.get('epoch', '?')})")
        return model


# ─── Text-to-audio search ────────────────────────────────────────────────────

def cmd_search(db_path, query, top_k=20):
    """Text-to-audio search using CLaMP3 text encoder."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tracks, emb_matrix = load_all_embeddings(db_path)
    print(f"Loaded {len(tracks)} tracks with embeddings")

    # Load CLaMP3 text encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CLaMP3 text encoder on {device}...")
    weights_path = hf_hub_download("sander-wood/clamp3", CLAMP3_WEIGHTS_FILENAME)
    encoder = CLaMP3TextEncoder.from_clamp3_checkpoint(weights_path, device=device)

    # Tokenize (matching extract_clamp3.py text processing)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    # CLaMP3 text preprocessing: deduplicate lines, join with SEP token
    lines = list(set(query.split("\n")))
    lines = [c for c in lines if len(c) > 0]
    text = tokenizer.sep_token.join(lines)

    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens['input_ids'].squeeze(0)  # [seq_len]

    # Segment into MAX_TEXT_LENGTH windows (for short queries: single segment)
    segment_list = []
    for i in range(0, len(input_ids), MAX_TEXT_LENGTH):
        segment_list.append(input_ids[i:i + MAX_TEXT_LENGTH])
    segment_list[-1] = input_ids[-MAX_TEXT_LENGTH:]

    # Encode each segment
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

        feat = encoder.encode(
            seg.unsqueeze(0).to(device),
            attention_mask.unsqueeze(0).to(device),
        )
        hidden_states_list.append(feat)

    # Weighted average of segments (matching extract_clamp3.py)
    total_tokens = len(input_ids)
    full_cnt = total_tokens // MAX_TEXT_LENGTH
    remain = total_tokens % MAX_TEXT_LENGTH
    if remain == 0:
        weights = torch.tensor(
            [MAX_TEXT_LENGTH] * full_cnt, device=device
        ).view(-1, 1)
    else:
        weights = torch.tensor(
            [MAX_TEXT_LENGTH] * full_cnt + [remain], device=device
        ).view(-1, 1)

    stacked = torch.cat(hidden_states_list, dim=0)
    text_emb = (stacked * weights).sum(0) / weights.sum()
    text_emb = text_emb.unsqueeze(0).cpu()  # [1, 768]

    # Cosine similarity against all audio embeddings
    sims = torch.cosine_similarity(text_emb, emb_matrix)  # [N]
    ranked = torch.argsort(sims, descending=True)

    print(f"\nText query: \"{query}\"")
    print(f"Top {top_k} matches:")
    print("-" * 80)
    for i in range(min(top_k, len(ranked))):
        idx = ranked[i].item()
        t = tracks[idx]
        sim = sims[idx].item()
        print(f"  {sim:.4f}  {t['artist']} - {t['title']}")


# ─── Song lookup helper ──────────────────────────────────────────────────────

def find_song_embedding(tracks, emb_matrix, query):
    """Find a song by fuzzy name match. Returns (embedding, index, label) or Nones."""
    query_words = query.lower().strip().split()
    scored = []
    for i, t in enumerate(tracks):
        label = f"{t['artist']} {t['title']} {t['album']}".lower()
        matches = sum(1 for w in query_words if w in label)
        if matches == len(query_words):
            scored.append((i, matches, label))
    if not scored:
        return None, None, "NOT FOUND"
    scored.sort(key=lambda x: (-x[1], x[2]))
    idx = scored[0][0]
    t = tracks[idx]
    return emb_matrix[idx], idx, f"{t['artist']} - {t['title']}"


# ─── Multi-seed search (Geometric Mean of Percentiles) ──────────────────────

def cmd_multiseed(db_path, text_query, songs, top_k=20):
    """Multi-seed search using geometric mean of percentile ranks.

    Args:
        text_query: Optional text description (can be None)
        songs: List of (song_query, weight) tuples
    """
    tracks, emb_matrix = load_all_embeddings(db_path)
    print(f"Loaded {len(tracks)} tracks with embeddings")

    seed_embs = []
    seed_weights = []
    seed_labels = []
    exclude_indices = set()

    # Text seed (if provided)
    if text_query:
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLaMP3 text encoder on {device}...")
        weights_path = hf_hub_download("sander-wood/clamp3", CLAMP3_WEIGHTS_FILENAME)
        encoder = CLaMP3TextEncoder.from_clamp3_checkpoint(weights_path, device=device)

        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        lines = list(set(text_query.split("\n")))
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
                pad = torch.full(
                    (MAX_TEXT_LENGTH - actual_len,),
                    tokenizer.pad_token_id, dtype=torch.long,
                )
                seg = torch.cat([seg, pad], dim=0)
            feat = encoder.encode(
                seg.unsqueeze(0).to(device),
                attention_mask.unsqueeze(0).to(device),
            )
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
        text_emb = text_emb.squeeze(0).cpu()  # [768]

        seed_embs.append(text_emb)
        seed_weights.append(1.0)
        seed_labels.append(f'text: "{text_query}"')

    # Song seeds
    for song_query, weight in songs:
        emb, idx, label = find_song_embedding(tracks, emb_matrix, song_query)
        if emb is None:
            print(f"WARNING: song '{song_query}' not found in DB, skipping")
            continue
        seed_embs.append(emb)
        seed_weights.append(weight)
        seed_labels.append(f'song: {label} (w={weight:+.1f})')
        exclude_indices.add(idx)

    if not seed_embs:
        print("ERROR: No valid seeds found")
        return 1

    # Print seeds
    print(f"\nSeeds ({len(seed_embs)}):")
    for label in seed_labels:
        print(f"  {label}")

    # Compute geometric mean of percentiles
    N = emb_matrix.shape[0]
    emb_np = emb_matrix.numpy()
    percentiles = []
    t0 = time.time()

    for emb, w in zip(seed_embs, seed_weights):
        emb_1d = emb.numpy() if isinstance(emb, torch.Tensor) else emb
        sims = emb_np @ emb_1d  # dot product (L2-normalized = cosine)
        if w < 0:
            sims = -sims
        ranks = sims.argsort().argsort()
        pctile = (ranks + 1) / N  # (0, 1]
        percentiles.append(pctile)

    # Weighted geometric mean
    abs_w = np.array([abs(w) for w in seed_weights])
    abs_w = abs_w / abs_w.sum()
    log_geo = sum(wt * np.log(pct) for wt, pct in zip(abs_w, percentiles))
    geo_mean = np.exp(log_geo)

    # Exclude seed songs
    for idx in exclude_indices:
        geo_mean[idx] = -1

    elapsed = time.time() - t0
    top_indices = np.argsort(geo_mean)[::-1][:top_k]

    print(f"\nTop {top_k} results (geo mean of percentiles, {elapsed:.2f}s):")
    print("-" * 80)
    for rank, idx in enumerate(top_indices, 1):
        t = tracks[idx]
        score = geo_mean[idx]
        print(f"  {rank:2d}. {score:.4f}  {t['artist']} - {t['title']}")

    return 0


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLaMP3 embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python evaluate_clamp3.py embeddings_clamp3.db similar "time pachanga boys"
  python evaluate_clamp3.py embeddings_clamp3.db search "space rock"
  python evaluate_clamp3.py embeddings_clamp3.db search "indian psytrance"
        """
    )
    parser.add_argument("db", help="Path to embeddings_clamp3.db")
    subparsers = parser.add_subparsers(dest="command")

    p_similar = subparsers.add_parser("similar", help="Find similar tracks by name")
    p_similar.add_argument("query", help="Track name (e.g. 'time pachanga boys')")
    p_similar.add_argument("-k", "--top-k", type=int, default=20)

    p_search = subparsers.add_parser("search", help="Text-to-audio search")
    p_search.add_argument("query", help="Text query (e.g. 'space rock')")
    p_search.add_argument("-k", "--top-k", type=int, default=20)

    p_multi = subparsers.add_parser("multiseed",
        help="Multi-seed search (geo mean of percentiles)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python evaluate_clamp3.py db multiseed --text "90s boombap"
  python evaluate_clamp3.py db multiseed --text "90s boombap" --song "time pachanga boys" --weight 0.8
  python evaluate_clamp3.py db multiseed --song "time pachanga boys" --song "lost phaxe" --weight -0.5
        """)
    p_multi.add_argument("--text", default=None, help="Text description seed")
    p_multi.add_argument("--song", action="append", default=[], help="Song reference (fuzzy name match)")
    p_multi.add_argument("--weight", action="append", type=float, default=[],
        help="Weight for the preceding --song (default 1.0, negative = 'less like')")
    p_multi.add_argument("-k", "--top-k", type=int, default=20)

    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: Database not found: {args.db}")
        return 1

    if args.command == "similar":
        cmd_similar(args.db, args.query, args.top_k)
    elif args.command == "search":
        cmd_search(args.db, args.query, args.top_k)
    elif args.command == "multiseed":
        if not args.text and not args.song:
            print("ERROR: provide at least --text or --song")
            return 1
        # Pair songs with weights (default weight = 1.0)
        songs = []
        for i, song in enumerate(args.song):
            weight = args.weight[i] if i < len(args.weight) else 1.0
            songs.append((song, weight))
        return cmd_multiseed(args.db, args.text, songs, args.top_k)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

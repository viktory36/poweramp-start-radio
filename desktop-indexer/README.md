# Desktop Indexer

The desktop indexer builds and inspects the CLaMP3 music index used by the Android app. It also
provides an always-on server workflow for embedding newly downloaded records before they reach the
phone.

## Embedding Pipeline

The production path processes the complete track by default:

1. decode audio and resample it with the pinned preprocessing policy;
2. run MERT over aligned five-second windows;
3. project the complete window sequence through the CLaMP3 audio encoder; and
4. store one normalized 768-dimensional embedding per track.

The CLaMP3 text encoder maps descriptions into the same space for offline text-to-audio queries and
for export to Android. Production musical relevance comes from these embeddings, not file tags.

## Install

```bash
cd desktop-indexer
python -m pip install -e '.[export,dev]'
```

Install the PyTorch/Torchaudio build appropriate for the host first when GPU acceleration is
required. `export` adds the LiteRT conversion dependencies; `dev` adds pytest and Ruff.

## Build A Complete Library

```bash
poweramp-indexer scan /path/to/music -o embeddings.db
```

`scan` has a resumable two-phase pipeline: cached MERT extraction followed by CLaMP3 encoding and
SQLite publication. It builds the cluster and kNN data used by Graph Explorer after embedding.

Useful production controls:

```bash
poweramp-indexer scan /path/to/music -o embeddings.db --batch-size 32
poweramp-indexer scan /path/to/music -o embeddings.db --fp32
poweramp-indexer scan /path/to/music -o embeddings.db --phase 1
poweramp-indexer scan /path/to/music -o embeddings.db --phase 2
```

FP16 MERT is the desktop default. Use FP32 when matching the stricter phone/server policy or when
running parity work.

Update an existing complete-library database:

```bash
poweramp-indexer update /path/to/music --database embeddings.db
```

By default, `update` adds new files, removes database rows whose files are missing, and refreshes
the Graph Explorer graph. Pass `--no-remove-missing` when the source root is temporarily incomplete.
Graph construction can also be run explicitly:

```bash
poweramp-indexer graph embeddings.db --clusters 200 --knn 5
```

## Always-On Server Indexing

Server mode watches one or more download roots, maintains a private source ledger, and atomically
publishes a cumulative graphless bundle for the phone. It is distinct from `update`: it never owns
the phone's complete library or final track IDs.

Start from [`deploy/server-indexer.example.toml`](deploy/server-indexer.example.toml). Keep
`state_db`, `bundle_db`, and `cache_dir` outside every listen root.

For a root whose current contents are already represented on the phone, record the one-time
baseline without embedding it:

```bash
poweramp-indexer server init \
  --config /etc/poweramp-server-indexer.toml \
  --baseline-existing
```

Run and inspect the service:

```bash
poweramp-indexer server once --config /etc/poweramp-server-indexer.toml
poweramp-indexer server run --config /etc/poweramp-server-indexer.toml
poweramp-indexer server status --config /etc/poweramp-server-indexer.toml
poweramp-indexer server status --config /etc/poweramp-server-indexer.toml --json
poweramp-indexer server retry --config /etc/poweramp-server-indexer.toml
```

`status` reads only the ledger and publication state. It reports completed, ready, active, retry,
blocked, and baseline counts; current throughput and ETA; output path and bundle identity; and last
publication details. It does not rescan the library, hash the export, or load a model.

Files must remain unchanged for the configured settle interval. The server then copies and hashes
an immutable source snapshot and commits only a complete FP32 vector. Torchaudio is the primary
decoder; `ffprobe` and `ffmpeg` provide a complete-stream fallback while retaining the same
downmix, resampling, normalization, and full-track policy. Unchanged per-file failures become
blocked after the configured attempt limit; service failures remain eligible for a later cycle.

The server treats one physical audio file as one track. It does not split CUE sheets. During merge,
the phone accepts a server row only when it maps safely to one ordinary, offset-zero Poweramp track;
shared CUE images are rejected rather than assigned a misleading embedding.

[`deploy/poweramp-server-indexer.service`](deploy/poweramp-server-indexer.service) is the generic
systemd unit. Adapt its service account and installation paths to the host.

The configured `bundle_db` is atomically replaced after validation. Recopying it is safe: bundle
provenance and source evidence let the phone skip rows already present in its active generation.

## Inspect And Query

```bash
poweramp-indexer info embeddings.db
poweramp-indexer similar embeddings.db "radiohead karma police"
poweramp-indexer similar embeddings.db --file /path/to/song.flac
poweramp-indexer similar embeddings.db --random
poweramp-indexer search embeddings.db "slow psychedelic guitar"
```

These commands are diagnostic tools. Android runs its own native selectors over the validated
packed index.

## Export Android Models

```bash
poweramp-indexer export all --output-dir ../models
```

The output bundle is:

- `mert.tflite`
- `clamp3_audio.tflite`
- `clamp3_text.tflite`
- `sentencepiece.bpe.model`

The Android APK does not bundle or install these files. Provision them separately before importing
the initial database.

## Database Contract

A complete-library `embeddings.db` contains:

- `tracks` metadata needed for identity, diagnostics, and display;
- one CLaMP3 vector per indexed track in `embeddings_clamp3`;
- model, preprocessing, source, dimension, and graph policy metadata; and
- cluster and kNN data for Graph Explorer.

The Android importer validates the database and publishes an immutable phone generation containing
the SQLite database, ordered packed vectors, graph, stable identity domain, receipts, and hashes.
A server bundle intentionally omits the graph and uses its own provenance tables because the phone
is authoritative for final library identity.

## Development

```bash
uv run --extra dev pytest tests experiments -q
uv run --extra dev ruff check .
```

Key paths:

- `src/poweramp_indexer/cli.py` - CLI boundary.
- `src/poweramp_indexer/scanner.py`, `embeddings_clamp3.py` - discovery and complete-library
  embedding pipeline.
- `src/poweramp_indexer/server_indexer.py` - persistent watcher, ledger, status, and publication.
- `src/poweramp_indexer/export_litert.py` - Android model export.
- `scripts/` - focused parity and database diagnostics.
- `experiments/` - read-only recommendation research; see its README before running.

# Desktop and Server Indexer

Build a CLaMP3 music database, query it, export Android models, or watch folders for new tracks.

## Install

From this directory:

```bash
uv sync --extra dev --extra export
uv run poweramp-indexer --help
```

Python 3.10+ is required. `dev` adds pytest/Ruff; `export` adds LiteRT conversion dependencies.
For indexing alone, omit those extras. Install matching PyTorch/torchaudio builds for your hardware.
`ffmpeg` and `ffprobe` enable fallback decoding of files torchaudio cannot read.

## Embedding Pipeline

1. Decode to mono and resample to 24 kHz with the TorchAudio Hann policy.
2. Normalize the decoded span and extract MERT features in consecutive five-second windows.
3. Feed the averaged MERT layer features to CLaMP3's audio encoder in 128-frame segments.
4. Combine segment outputs with frame-count weights and L2-normalize one 768d vector.

There is no duration cap by default. A final audio remainder shorter than one second is skipped;
long sequences use CLaMP3's overlapping last-segment convention. This is whole-track processing,
not one vector per excerpt. Text queries use CLaMP3's text encoder in the same embedding space.

## Library Indexing

```bash
uv run poweramp-indexer scan /path/to/music -o embeddings.db
```

The scan first caches MERT features, then writes embeddings and builds cluster/kNN data for
Graph Explorer. CPU decoding of the next track overlaps current GPU inference. Rerunning resumes
from the cache.

| Option | Use |
| --- | --- |
| `--batch-size 16` | Increase MERT throughput if GPU memory permits; default 8 |
| `--fp32` | Use FP32 MERT instead of the desktop FP16 default |
| `--phase 1` / `--phase 2` | Run extraction and database encoding separately |
| `--cache-dir PATH` | Choose where MERT intermediate files live |

Phone/server indexing uses FP32. Use `--fp32` for stricter cross-device comparison.

```bash
uv run poweramp-indexer update /path/to/music --database embeddings.db
uv run poweramp-indexer graph embeddings.db --clusters 200 --knn 5
```

**Update removes rows for missing files by default.** Use `--no-remove-missing` if a source root
is temporarily incomplete. It also rebuilds the graph; `--no-rebuild-graph` postpones that work.

These commands update the desktop copy only. After initial phone import, grow the active phone
index through on-device indexing or server merge, not by overwriting its private database.

## Server Indexing

Server mode watches folders, maintains a durable work ledger, and publishes a cumulative graphless
database for phone merge. It embeds one physical file per track; it does not split CUE sheets.

Copy and edit [the example config](deploy/server-indexer.example.toml).
Keep state, cache, and exports outside the watched roots.

| Config field | Meaning |
| --- | --- |
| `listen_roots` | Stable root IDs and folders to watch |
| `state_db` | Private work ledger; not the phone export |
| `bundle_db` | Cumulative export to sync to the phone |
| `cache_dir` | Temporary source/inference files |
| `poll_interval_seconds` | Time between scans |
| `settle_seconds` | How long a file must remain unchanged before indexing |
| `max_files_per_cycle` | Maximum work batch per cycle |
| `max_attempts`, `retry_delays_seconds` | Failure retry policy |
| `mert_batch_size` | MERT inference batch size |

Initialize the ledger:

```bash
uv run poweramp-indexer server init --config /path/to/server.toml
```

Add **`--baseline-existing` only if the current files are already indexed on your phone**.
It records those files without embedding them; newly arriving files will be indexed.

```bash
uv run poweramp-indexer server run --config /path/to/server.toml
uv run poweramp-indexer server status --config /path/to/server.toml
uv run poweramp-indexer server status --config /path/to/server.toml --json
```

Status reports done/ready/active/retry/blocked/baseline counts, throughput, ETA, output path,
publication counts, and errors. It reads the ledger without scanning music or loading models.
`server once` runs one cycle; `server retry` reopens blocked files for another attempt.

For unattended operation, adapt [the systemd unit](deploy/poweramp-server-indexer.service).
Its service user must read the listen roots and write state/cache/export directories.

Files must settle before indexing. A private source snapshot prevents a changing download from
producing a mismatched vector. Failed files stop automatic retries after the configured limit.
The export is replaced atomically after a successful publication, so readers see a complete file.

### Transfer to the Phone

1. Sync the music to the phone and let Poweramp scan it.
2. Copy `bundle_db`, not `state_db`. For the latest complete batch, wait for an empty queue,
   no cycle error/blocked files, and a Published count matching completed tracks in status.
3. In Settings, choose **Merge server index** and select the export.

The phone matches rows to its own library, checks model and source-span compatibility, skips
already indexed rows, and updates its graph. You can merge the same cumulative export repeatedly.
Shared CUE images are rejected rather than assigning a full-image vector to one split track.

## Query and Export

```bash
uv run poweramp-indexer info embeddings.db
uv run poweramp-indexer similar embeddings.db "radiohead karma police"
uv run poweramp-indexer similar embeddings.db --file /path/to/song.flac
uv run poweramp-indexer similar embeddings.db --random
uv run poweramp-indexer search embeddings.db "slow psychedelic guitar"
uv run poweramp-indexer export all --output-dir models
```

The model export produces `mert.tflite`, `clamp3_audio.tflite`, `clamp3_text.tflite`, and
`sentencepiece.bpe.model`. See [Setup](../SETUP.md) for phone installation.

## Database and Development

A library database stores track metadata, vectors in `embeddings_clamp3`, preprocessing/model
metadata, and cluster/kNN data. Server bundles omit the graph and add source provenance for merging.

```bash
uv run --extra dev pytest tests experiments -q
uv run --extra dev ruff check .
```

Implementation: `src/poweramp_indexer/cli.py` (commands), `embeddings_clamp3.py` (inference),
`server_indexer.py` (watcher/status), `server_bundle.py` (export), `export_litert.py` (models).
[Experiments](experiments/README.md) contains host recommendation and parity tools.

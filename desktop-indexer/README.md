# Poweramp Start Radio - Desktop Indexer

CLI tool that scans a local music library, generates audio embeddings using
MuLan and Music Flamingo, and stores them in an SQLite database for use by the Android plugin.

## Install (editable)

```bash
python -m pip install -e .
```

## Usage

```bash
poweramp-indexer scan /path/to/music -o embeddings.db
poweramp-indexer update /path/to/music -d embeddings.db
```

## Notes

- Large libraries benefit from a GPU-capable PyTorch build.
- The generated database is portable; copy it to your phone for offline use.

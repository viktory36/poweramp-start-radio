"""Command-line interface for the Poweramp indexer."""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
from tqdm import tqdm

from . import __version__
from .database import EmbeddingDatabase
from .embeddings_dual import MuLanEmbeddingGenerator
from .embeddings_flamingo import FlamingoEmbeddingGenerator
from .fingerprint import extract_metadata
from .scanner import scan_music_directory


def create_mulan_generator():
    """Create the MuLan embedding generator."""
    return MuLanEmbeddingGenerator()


def create_flamingo_generator():
    """Create the Flamingo embedding generator."""
    return FlamingoEmbeddingGenerator()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
def cli():
    """Poweramp Start Radio - Desktop Indexer

    Scan your music library and generate AI embeddings for similarity search.
    """
    pass


@cli.command()
@click.argument("music_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("embeddings.db"),
    help="Output database file path"
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    help="Skip files already in the database"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging"
)
@click.option(
    "--model",
    type=click.Choice(["mulan", "flamingo"]),
    default="mulan",
    help="Embedding model (default: mulan)"
)
def scan(music_path: Path, output: Path, skip_existing: bool, verbose: bool, model: str):
    """Scan a music directory and generate embeddings.

    MUSIC_PATH: Path to your music library folder

    Use --model flamingo to generate embeddings from Music Flamingo's encoder.
    Requires running 'extract-encoder' first.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    click.echo(f"Scanning: {music_path}")

    # Collect all audio files
    click.echo("Discovering audio files...")
    audio_files = list(scan_music_directory(music_path))
    click.echo(f"Found {len(audio_files)} audio files")

    if not audio_files:
        click.echo("No audio files found. Exiting.")
        return

    if model == "flamingo":
        _scan_flamingo(music_path, output, skip_existing, audio_files)
    else:
        _scan_mulan(music_path, output, skip_existing, audio_files)


def _process_files(audio_files, load_audio_fn, infer_fn, store_fn, desc):
    """Process files with prefetching: load next file while GPU processes current.

    Both librosa and CUDA release the GIL, so the background thread's audio
    decode overlaps with the main thread's GPU inference.

    Args:
        audio_files: Paths to process.
        load_audio_fn: Callable(path) -> (waveform, duration) or None.
        infer_fn: Callable(waveform, duration, filename) -> embedding or None.
        store_fn: Callable(filepath, metadata, embedding) -> None. Stores the result.
        desc: tqdm progress bar description.

    Returns:
        (successful, failed) counts.
    """
    successful = 0
    failed = 0

    def _prefetch(fp):
        return extract_metadata(fp), load_audio_fn(fp)

    with tqdm(total=len(audio_files), desc=desc, unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_prefetch, audio_files[0])

            for i in range(len(audio_files)):
                metadata, audio = future.result()

                # Start loading next file while we run inference on this one
                if i + 1 < len(audio_files):
                    future = executor.submit(_prefetch, audio_files[i + 1])

                filepath = audio_files[i]
                if audio is not None:
                    waveform, duration_s = audio
                    embedding = infer_fn(waveform, duration_s, filepath.name)
                    del waveform
                else:
                    embedding = None

                if embedding is not None:
                    store_fn(filepath, metadata, embedding)
                    successful += 1
                else:
                    failed += 1
                    logger.warning(f"Failed: {filepath.name}")

                pbar.update(1)

    return successful, failed


def _scan_mulan(music_path: Path, output: Path, skip_existing: bool, audio_files: list[Path]):
    """MuLan scan."""
    click.echo(f"Output: {output}")

    db = EmbeddingDatabase(output, models=["mulan"])

    existing_paths = db.get_existing_paths(model="mulan") if skip_existing else set()
    if existing_paths:
        new_files = [f for f in audio_files if str(f) not in existing_paths]
        click.echo(f"Skipping {len(audio_files) - len(new_files)} existing files")
        audio_files = new_files

    if not audio_files:
        click.echo("All files already indexed. Use --no-skip-existing to reindex.")
        db.close()
        return

    click.echo("Loading MuLan model...")
    generator = create_mulan_generator()

    def store_track(filepath, metadata, embedding):
        db.add_track(metadata, embedding, model="mulan")
        if (store_track.count % 10) == 0:
            db.commit()
        store_track.count += 1
    store_track.count = 0

    successful, failed = _process_files(
        audio_files, generator.load_audio, generator.generate_from_audio,
        store_track, "MuLan"
    )
    db.commit()

    db.set_metadata("version", __version__)
    db.set_metadata("source_path", str(music_path))
    db.set_metadata("embedding_dim", str(generator.embedding_dim))
    db.set_metadata("model", "mulan")

    total_tracks = db.count_tracks()
    click.echo(f"\nComplete!")
    click.echo(f"  Successfully indexed: {successful}")
    click.echo(f"  Failed: {failed}")
    click.echo(f"  Total tracks in database: {total_tracks}")
    click.echo(f"  Database size: {output.stat().st_size / 1024 / 1024:.1f} MB")

    generator.unload_model()
    db.close()


def _scan_flamingo(music_path: Path, output: Path, skip_existing: bool, audio_files: list[Path]):
    """Single-model scan with Music Flamingo encoder."""
    click.echo(f"Output: {output}")

    db = EmbeddingDatabase(output, models=["flamingo"])

    existing_paths = db.get_existing_paths(model="flamingo") if skip_existing else set()
    if existing_paths:
        new_files = [f for f in audio_files if str(f) not in existing_paths]
        click.echo(f"Skipping {len(audio_files) - len(new_files)} existing files")
        audio_files = new_files

    if not audio_files:
        click.echo("All files already indexed. Use --no-skip-existing to reindex.")
        db.close()
        return

    click.echo("Loading Music Flamingo encoder...")
    try:
        generator = create_flamingo_generator()
    except FileNotFoundError as e:
        click.echo(f"Error: {e}")
        db.close()
        return

    def store_track(filepath, metadata, embedding):
        db.add_track(metadata, embedding, model="flamingo")
        if (store_track.count % 10) == 0:
            db.commit()
        store_track.count += 1
    store_track.count = 0

    successful, failed = _process_files(
        audio_files, generator.load_audio, generator.generate_from_audio,
        store_track, "Flamingo"
    )
    db.commit()

    db.set_metadata("version", __version__)
    db.set_metadata("source_path", str(music_path))
    db.set_metadata("embedding_dim", str(generator.embedding_dim))
    db.set_metadata("model", "flamingo")

    total_tracks = db.count_tracks()
    click.echo(f"\nComplete!")
    click.echo(f"  Successfully indexed: {successful}")
    click.echo(f"  Failed: {failed}")
    click.echo(f"  Total tracks in database: {total_tracks}")
    click.echo(f"  Database size: {output.stat().st_size / 1024 / 1024:.1f} MB")

    generator.unload_model()
    db.close()


@cli.command()
@click.argument("music_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=Path("embeddings.db"),
              help="Output database file path")
@click.option("--database", "-d", type=click.Path(exists=True, path_type=Path), default=None,
              help="Existing database to update (skips existing files)")
@click.option("--mulan-only", is_flag=True, help="Only generate MuLan embeddings (skip Flamingo + fuse)")
@click.option("--dim", default=512, help="Target fused dimension (default: 512)")
@click.option("--skip-existing/--no-skip-existing", default=True, help="Skip files already in the database")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def build(music_path: Path, output: Path, database: Path, mulan_only: bool, dim: int,
          skip_existing: bool, verbose: bool):
    """One-command pipeline: scan with all models, reduce, and fuse.

    Scans with MuLan, optionally scans with Flamingo, reduces Flamingo
    dimensions, and fuses everything into a single embedding space.

    MUSIC_PATH: Path to your music library folder

    Examples:

      poweramp-indexer build /path/to/music
      poweramp-indexer build /path/to/music --mulan-only
      poweramp-indexer build /path/to/music -d existing.db -o updated.db
    """
    import shutil
    import struct

    import numpy as np

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Open or create database
    if database is not None:
        if output != database:
            shutil.copy2(database, output)
        db = EmbeddingDatabase(output)
        available = db.get_available_models()
        click.echo(f"Existing database: {database} ({db.count_tracks()} tracks, models: {', '.join(available) or 'none'})")
    else:
        models = ["mulan"] if mulan_only else ["mulan", "flamingo"]
        db = EmbeddingDatabase(output, models=models)

    # Discover audio files
    click.echo(f"Scanning: {music_path}")
    audio_files = list(scan_music_directory(music_path))
    click.echo(f"Found {len(audio_files)} audio files")
    if not audio_files:
        click.echo("No audio files found. Exiting.")
        db.close()
        return

    # --- Step 1: MuLan ---
    mulan_files = audio_files
    if skip_existing:
        existing = db.get_existing_paths(model="mulan")
        mulan_files = [f for f in audio_files if str(f) not in existing]
        if existing:
            click.echo(f"MuLan: skipping {len(audio_files) - len(mulan_files)} existing")

    mulan_new = 0
    if mulan_files:
        click.echo(f"\nMuLan: {len(mulan_files)} files to index")
        click.echo("Loading MuLan model...")
        generator = create_mulan_generator()

        def store_mulan(filepath, metadata, embedding):
            db.add_track(metadata, embedding, model="mulan")
            if (store_mulan.count % 10) == 0:
                db.commit()
            store_mulan.count += 1
        store_mulan.count = 0

        ok, fail = _process_files(mulan_files, generator.load_audio,
                                  generator.generate_from_audio, store_mulan, "MuLan")
        db.commit()
        mulan_new = ok
        click.echo(f"  MuLan: {ok} indexed, {fail} failed")
        generator.unload_model()
    else:
        click.echo("MuLan: all files already indexed")

    # --- Step 2: Flamingo (unless --mulan-only) ---
    flamingo_new = 0
    if not mulan_only:
        from .embeddings_flamingo import DEFAULT_ENCODER_DIR

        # Auto-extract encoder if missing
        encoder_file = DEFAULT_ENCODER_DIR / "model.safetensors"
        if not encoder_file.exists():
            click.echo("\nFlamingo encoder not found, extracting...")
            from click import Context
            ctx = Context(extract_encoder, info_name="extract-encoder")
            ctx.invoke(extract_encoder, output=None, verbose=verbose)

        flam_files = audio_files
        if skip_existing:
            existing = db.get_existing_paths(model="flamingo")
            flam_files = [f for f in audio_files if str(f) not in existing]
            if existing:
                click.echo(f"Flamingo: skipping {len(audio_files) - len(flam_files)} existing")

        if flam_files:
            click.echo(f"\nFlamingo: {len(flam_files)} files to index")
            click.echo("Loading Music Flamingo encoder...")
            try:
                generator = create_flamingo_generator()
            except FileNotFoundError as e:
                click.echo(f"Error loading Flamingo: {e}")
                click.echo("Continuing with MuLan only.")
                mulan_only = True  # skip fuse step
            else:
                def store_flam(filepath, metadata, embedding):
                    track_id = db.get_track_id_by_path(str(filepath))
                    if track_id is not None:
                        db.add_embedding(track_id, "flamingo", embedding)
                    else:
                        db.add_track(metadata, embedding, model="flamingo")
                    if (store_flam.count % 10) == 0:
                        db.commit()
                    store_flam.count += 1
                store_flam.count = 0

                ok, fail = _process_files(flam_files, generator.load_audio,
                                          generator.generate_from_audio, store_flam, "Flamingo")
                db.commit()
                flamingo_new = ok
                click.echo(f"  Flamingo: {ok} indexed, {fail} failed")
                generator.unload_model()
        else:
            click.echo("Flamingo: all files already indexed")

        # --- Step 3: Reduce Flamingo if needed ---
        if not mulan_only:
            table = db._table_name("flamingo")
            try:
                sample = db.conn.execute(f"SELECT embedding FROM [{table}] LIMIT 1").fetchone()
            except Exception:
                sample = None

            if sample is not None:
                original_dim = len(sample[0]) // 4
                if original_dim > dim:
                    click.echo(f"\nReducing Flamingo: {original_dim}d -> {dim}d")
                    count = db.conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]

                    X = np.empty((count, original_dim), dtype=np.float32)
                    track_ids = np.empty(count, dtype=np.int64)
                    cursor = db.conn.execute(f"SELECT track_id, embedding FROM [{table}]")
                    for i, (tid, blob) in enumerate(cursor):
                        track_ids[i] = tid
                        X[i] = np.frombuffer(blob, dtype=np.float32)

                    G = X.astype(np.float64).T @ X.astype(np.float64)
                    eigenvalues, eigenvectors = np.linalg.eigh(G)
                    idx = np.argsort(eigenvalues)[::-1]
                    eigenvalues = np.maximum(eigenvalues[idx], 0)
                    eigenvectors = eigenvectors[:, idx]

                    retained = eigenvalues[:dim].sum() / eigenvalues.sum()
                    click.echo(f"  Variance retained: {retained * 100:.2f}%")

                    V_k = eigenvectors[:, :dim].astype(np.float32)
                    X_reduced = X @ V_k
                    norms = np.linalg.norm(X_reduced, axis=1, keepdims=True)
                    X_reduced = X_reduced / np.maximum(norms, 1e-10)

                    db.conn.execute("BEGIN")
                    try:
                        for i in range(count):
                            blob = struct.pack(f"{dim}f", *X_reduced[i])
                            db.conn.execute(f"UPDATE [{table}] SET embedding = ? WHERE track_id = ?",
                                            (blob, int(track_ids[i])))
                        proj_blob = V_k.tobytes()
                        db.conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                                        ("flamingo_projection", proj_blob))
                        db.conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                                        ("flamingo_original_dim", str(original_dim)))
                        db.conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                                        ("flamingo_reduced_dim", str(dim)))
                        db.conn.execute("COMMIT")
                    except Exception:
                        db.conn.execute("ROLLBACK")
                        raise
                    click.echo(f"  Flamingo reduced to {dim}d")

    # --- Step 4: Fuse (if both models present) ---
    available = db.get_available_models()
    if "mulan" in available and "flamingo" in available and not mulan_only:
        click.echo(f"\nFusing MuLan + Flamingo -> {dim}d")
        from .fusion import fuse_embeddings

        try:
            result = fuse_embeddings(db, target_dim=dim, on_progress=lambda msg: click.echo(f"  {msg}"))
            click.echo(f"  Fused: {result['n_tracks']} tracks, {result['variance_retained'] * 100:.2f}% variance")
        except ValueError as e:
            click.echo(f"  Fusion error: {e}")

    # Update metadata
    db.set_metadata("version", __version__)
    db.set_metadata("source_path", str(music_path))
    available = db.get_available_models()
    db.set_metadata("model", ",".join(available))

    total = db.count_tracks()
    db.close()

    click.echo(f"\nBuild complete!")
    click.echo(f"  Total tracks: {total}")
    click.echo(f"  Models: {', '.join(available)}")
    click.echo(f"  New: {mulan_new} (MuLan) + {flamingo_new} (Flamingo)")
    click.echo(f"  Database size: {output.stat().st_size / 1024 / 1024:.1f} MB")


@cli.command()
@click.argument("music_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--database", "-d",
    type=click.Path(exists=True, path_type=Path),
    default=Path("embeddings.db"),
    help="Database file to update"
)
@click.option(
    "--remove-missing/--no-remove-missing",
    default=True,
    help="Remove tracks whose files no longer exist"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging"
)
def update(music_path: Path, database: Path, remove_missing: bool, verbose: bool):
    """Incrementally update an existing database.

    Adds new files and optionally removes missing ones.
    Detects available models in the DB and runs passes for each.

    MUSIC_PATH: Path to your music library folder
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    click.echo(f"Updating database: {database}")
    click.echo(f"Scanning: {music_path}")

    # Collect all audio files
    click.echo("Discovering audio files...")
    audio_files = list(scan_music_directory(music_path))
    audio_file_paths = {str(f) for f in audio_files}
    click.echo(f"Found {len(audio_files)} audio files on disk")

    # Open database
    db = EmbeddingDatabase(database)
    existing_tracks = db.count_tracks()
    click.echo(f"Existing tracks in database: {existing_tracks}")

    # Detect available models
    available_models = db.get_available_models()
    if not available_models:
        available_models = ["mulan"]  # default for empty/new DBs
    click.echo(f"Models in database: {', '.join(available_models)}")

    # Remove missing if requested
    if remove_missing:
        db.remove_missing_tracks(audio_file_paths)
        after_removal = db.count_tracks()
        removed = existing_tracks - after_removal
        if removed > 0:
            click.echo(f"Removed {removed} tracks with missing files")

    # Check database model compatibility
    db_model = db.get_metadata("model")
    if db_model == "clap":
        click.echo("Error: This database was created with CLAP embeddings, which are no longer supported.")
        click.echo("Please re-scan your library to create a new database.")
        db.close()
        return

    is_multi = len(available_models) > 1

    if is_multi:
        # Multi-model update: process each model independently
        for model in available_models:
            existing_paths = db.get_existing_paths(model=model)
            new_files = [f for f in audio_files if str(f) not in existing_paths]

            if not new_files:
                click.echo(f"\n{model.upper()}: No new files to add.")
                continue

            click.echo(f"\n{model.upper()}: {len(new_files)} new files to index")

            if model == "mulan":
                generator = create_mulan_generator()
                infer_fn = generator.generate_from_audio
            elif model == "flamingo":
                try:
                    generator = create_flamingo_generator()
                except FileNotFoundError as e:
                    click.echo(f"Error: {e}")
                    continue
                infer_fn = generator.generate_from_audio
            else:
                click.echo(f"Skipping unknown model: {model}")
                continue

            def make_store_fn(m):
                def store(filepath, metadata, embedding):
                    track_id = db.get_track_id_by_path(str(filepath))
                    if track_id is not None:
                        db.add_embedding(track_id, m, embedding)
                    else:
                        db.add_track(metadata, embedding, model=m)
                    if (store.count % 10) == 0:
                        db.commit()
                    store.count += 1
                store.count = 0
                return store

            ok, fail = _process_files(
                new_files, generator.load_audio, infer_fn,
                make_store_fn(model), model.upper()
            )
            db.commit()
            click.echo(f"  {model.upper()}: {ok} indexed, {fail} failed")
            generator.unload_model()
    else:
        # Single model update
        model = available_models[0]
        existing_paths = db.get_existing_paths(model=model)
        new_files = [f for f in audio_files if str(f) not in existing_paths]

        if not new_files:
            click.echo("No new files to add.")
            db.vacuum()
            db.close()
            return

        click.echo(f"Found {len(new_files)} new files to index")

        if model == "flamingo":
            click.echo("Loading Music Flamingo encoder...")
            try:
                generator = create_flamingo_generator()
            except FileNotFoundError as e:
                click.echo(f"Error: {e}")
                db.close()
                return
        else:
            click.echo("Loading MuLan model...")
            generator = create_mulan_generator()

        def store_track(filepath, metadata, embedding):
            db.add_track(metadata, embedding, model=model)
            if (store_track.count % 10) == 0:
                db.commit()
            store_track.count += 1
        store_track.count = 0

        successful, failed = _process_files(
            new_files, generator.load_audio, generator.generate_from_audio,
            store_track, model.upper()
        )
        db.commit()
        click.echo(f"\n  Added: {successful}")
        click.echo(f"  Failed: {failed}")
        generator.unload_model()

    # Auto-re-fuse if fused embeddings exist and new tracks were added
    if "fused" in db.get_available_models():
        click.echo("\nRe-fusing embeddings with new tracks...")
        from .fusion import fuse_embeddings

        fused_dim = int(db.get_metadata("fused_dim") or "512")
        try:
            result = fuse_embeddings(db, target_dim=fused_dim,
                                     on_progress=lambda msg: click.echo(f"  {msg}"))
            click.echo(f"  Re-fused: {result['n_tracks']} tracks")
        except ValueError as e:
            click.echo(f"  Re-fuse error: {e}")

    # Update metadata
    db.set_metadata("version", __version__)
    db.set_metadata("source_path", str(music_path))
    db.set_metadata("model", ",".join(available_models))

    # Vacuum and close
    db.vacuum()

    total_tracks = db.count_tracks()
    click.echo(f"\nUpdate complete!")
    click.echo(f"  Total tracks in database: {total_tracks}")
    click.echo(f"  Database size: {database.stat().st_size / 1024 / 1024:.1f} MB")

    db.close()


@cli.command()
@click.argument("database", type=click.Path(exists=True, path_type=Path))
def info(database: Path):
    """Show information about an embedding database."""
    db = EmbeddingDatabase(database)

    click.echo(f"Database: {database}")
    click.echo(f"File size: {database.stat().st_size / 1024 / 1024:.1f} MB")
    click.echo(f"Total tracks: {db.count_tracks()}")

    version = db.get_metadata("version")
    if version:
        click.echo(f"Created with version: {version}")

    source = db.get_metadata("source_path")
    if source:
        click.echo(f"Source path: {source}")

    # Show available models and counts
    available = db.get_available_models()
    if available:
        model_names = {"mulan": "MuLan", "flamingo": "Music Flamingo", "fused": "Fused"}
        parts = []
        for m in available:
            name = model_names.get(m, m)
            count = db.count_embeddings(m)
            parts.append(f"{name} ({count} embeddings)")
        click.echo(f"Models: {', '.join(parts)}")
    else:
        # Fallback to metadata
        model = db.get_metadata("model")
        if model:
            model_names = {"mulan": "MuLan", "flamingo": "Music Flamingo", "fused": "Fused"}
            click.echo(f"Embedding model: {model_names.get(model, model)}")

    dim = db.get_metadata("embedding_dim")
    if dim:
        click.echo(f"Embedding dimension: {dim}")

    db.close()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Dot product for L2-normalized vectors."""
    return sum(x * y for x, y in zip(a, b))


def find_similar(
    all_embeddings: dict[int, list[float]],
    seed_embedding: list[float],
    top_n: int,
    exclude_id: int = None
) -> list[tuple[int, float]]:
    """Find top-N similar tracks to a seed embedding."""
    scores = []
    for track_id, emb in all_embeddings.items():
        if track_id == exclude_id:
            continue
        sim = cosine_similarity(seed_embedding, emb)
        scores.append((track_id, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


def format_track(track: dict) -> str:
    """Format a track for display."""
    artist = track.get("artist") or "Unknown Artist"
    title = track.get("title") or "Unknown Title"
    album = track.get("album")
    if album:
        return f"{artist} - {title} ({album})"
    return f"{artist} - {title}"


@cli.command()
@click.argument("database", type=click.Path(exists=True, path_type=Path))
@click.argument("query", required=False)
@click.option("--file", "-f", "audio_file", type=click.Path(exists=True, path_type=Path), help="Audio file to find similar tracks for")
@click.option("--random", "-r", "use_random", is_flag=True, help="Pick a random seed track")
@click.option("--top", "-n", default=10, help="Number of similar tracks to show")
@click.option("--model", "-m", type=click.Choice(["mulan", "flamingo", "fused"]), default=None, help="Show results from a single model only")
def similar(database: Path, query: str, audio_file: Path, use_random: bool, top: int, model: str):
    """Find similar tracks in the database.

    DATABASE: Path to embeddings.db
    QUERY: Search string to find seed track (artist, title, etc.)

    When the database has multiple embedding models, results from each
    are shown side-by-side. Use --model to show only one.

    Examples:

      poweramp-indexer similar embeddings.db "radiohead karma police"
      poweramp-indexer similar embeddings.db --file ~/Downloads/song.mp3
      poweramp-indexer similar embeddings.db --random
      poweramp-indexer similar embeddings.db --random --model mulan
    """
    # Validate arguments
    options_count = sum([query is not None, audio_file is not None, use_random])
    if options_count == 0:
        raise click.UsageError("Provide a QUERY, --file, or --random")
    if options_count > 1:
        raise click.UsageError("Provide only one of: QUERY, --file, or --random")

    db = EmbeddingDatabase(database)
    available_models = db.get_available_models()

    if not available_models:
        click.echo("Database has no embeddings.")
        db.close()
        return

    # Determine which models to show
    if model:
        if model not in available_models:
            click.echo(f"Model '{model}' not found in database. Available: {', '.join(available_models)}")
            db.close()
            return
        show_models = [model]
    else:
        show_models = available_models

    # Determine seed track and embedding
    seed_track = None
    exclude_id = None

    if query:
        matches = db.search_tracks(query)
        if not matches:
            click.echo(f"No tracks found matching: {query}")
            db.close()
            return

        if len(matches) > 1:
            click.echo(f"Found {len(matches)} matches, using first:")
            for i, track in enumerate(matches[:5]):
                marker = ">" if i == 0 else " "
                click.echo(f"  {marker} {format_track(track)}")
            if len(matches) > 5:
                click.echo(f"  ... and {len(matches) - 5} more")
            click.echo()

        seed_track = matches[0]
        exclude_id = seed_track["id"]

    elif use_random:
        seed_track = db.get_random_track()
        if not seed_track:
            click.echo("Database is empty")
            db.close()
            return
        exclude_id = seed_track["id"]

    elif audio_file:
        # Generate embedding for external file
        if model == "flamingo":
            click.echo("Generating embedding with Music Flamingo...")
            try:
                generator = create_flamingo_generator()
            except FileNotFoundError as e:
                click.echo(f"Error: {e}")
                db.close()
                return
            file_model = "flamingo"
        else:
            click.echo("Generating embedding with MuLan...")
            generator = create_mulan_generator()
            file_model = "mulan"

        seed_embedding = generator.generate_embedding(audio_file)
        generator.unload_model()

        if seed_embedding is None:
            click.echo(f"Failed to generate embedding for: {audio_file}")
            db.close()
            return

        click.echo(f"Seed: {audio_file}")
        click.echo()

        model_names = {"mulan": "MuLan", "flamingo": "Music Flamingo", "fused": "Fused"}
        label = model_names.get(file_model, file_model)

        click.echo("Loading embeddings...")
        all_embeddings = db.get_all_embeddings(model=file_model)
        click.echo(f"Searching {len(all_embeddings)} tracks...")
        click.echo()

        similar_tracks = find_similar(all_embeddings, seed_embedding, top)
        click.echo(f"Similar tracks ({label}):")
        for rank, (track_id, score) in enumerate(similar_tracks, 1):
            track = db.get_track_by_id(track_id)
            if track:
                click.echo(f"  {rank:2}. [{score:.3f}] {format_track(track)}")

        db.close()
        return

    # Display seed
    if seed_track:
        click.echo(f"Seed: {format_track(seed_track)}")
        click.echo()

    # Show results for each model
    for m in show_models:
        seed_embedding = db.get_embedding_by_id(exclude_id, model=m)
        if seed_embedding is None:
            click.echo(f"Seed track has no {m.upper()} embedding, skipping.")
            click.echo()
            continue

        click.echo(f"Loading {m.upper()} embeddings...")
        all_embeddings = db.get_all_embeddings(model=m)
        click.echo(f"Searching {len(all_embeddings)} tracks...")
        click.echo()

        similar_tracks = find_similar(all_embeddings, seed_embedding, top, exclude_id)

        model_names = {"mulan": "MuLan", "flamingo": "Music Flamingo", "fused": "Fused"}
        label = model_names.get(m, m)
        click.echo(f"{label} similar tracks:")
        for rank, (track_id, score) in enumerate(similar_tracks, 1):
            track = db.get_track_by_id(track_id)
            if track:
                click.echo(f"  {rank:2}. [{score:.3f}] {format_track(track)}")
        click.echo()

    db.close()


@cli.command()
@click.argument("database", type=click.Path(exists=True, path_type=Path))
@click.argument("query")
@click.option("--top", "-n", default=10, help="Number of results to show")
def search(database: Path, query: str, top: int):
    """Search for tracks using text queries (requires MuLan embeddings).

    DATABASE: Path to embeddings database
    QUERY: Text query describing the music you want (e.g., "sufi music", "upbeat electronic")

    Examples:

      poweramp-indexer search embeddings.db "sufi"
      poweramp-indexer search embeddings.db "upbeat electronic dance"
      poweramp-indexer search embeddings.db "sad piano ballad"
    """
    db = EmbeddingDatabase(database)

    # Verify this DB has MuLan embeddings
    available = db.get_available_models()
    if "mulan" not in available:
        # Fallback: check metadata for legacy DBs
        model = db.get_metadata("model")
        if model != "mulan":
            click.echo("Error: Database has no MuLan embeddings.")
            click.echo("Use 'scan' to create a MuLan database first.")
            db.close()
            return

    # Generate text embedding
    click.echo(f"Searching for: {query}")
    generator = create_mulan_generator()
    text_embedding = generator.embed_text(query)

    if text_embedding is None:
        click.echo("Failed to generate text embedding.")
        generator.unload_model()
        db.close()
        return

    # Load MuLan embeddings and find similar
    click.echo("Loading MuLan embeddings...")
    all_embeddings = db.get_all_embeddings(model="mulan")
    click.echo(f"Searching {len(all_embeddings)} tracks...")
    click.echo()

    similar_tracks = find_similar(all_embeddings, text_embedding, top)

    # Display results
    click.echo("Results:")
    for rank, (track_id, score) in enumerate(similar_tracks, 1):
        track = db.get_track_by_id(track_id)
        if track:
            click.echo(f"  {rank:2}. [{score:.3f}] {format_track(track)}")

    generator.unload_model()
    db.close()


@cli.command()
@click.argument("db1", type=click.Path(exists=True, path_type=Path))
@click.argument("db2", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("embeddings.db"),
    help="Output combined database file path"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def merge(db1: Path, db2: Path, output: Path, verbose: bool):
    """Merge two embedding databases into a single combined database.

    DB1: Path to the first embeddings database
    DB2: Path to the second embeddings database

    Auto-detects which models each database contains and merges them
    by matching tracks on file_path. Works with any combination of
    MuLan and Flamingo databases.

    Examples:

      poweramp-indexer merge embeddings_mulan.db embeddings_flamingo.db -o embeddings.db
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if output.exists():
        click.echo(f"Output file already exists: {output}")
        if not click.confirm("Overwrite?"):
            return
        output.unlink()

    click.echo(f"Source 1: {db1}")
    click.echo(f"Source 2: {db2}")
    click.echo(f"Output: {output}")

    # Open source databases and detect models
    src1 = EmbeddingDatabase(db1)
    src2 = EmbeddingDatabase(db2)

    models1 = src1.get_available_models()
    models2 = src2.get_available_models()

    click.echo(f"DB1 models: {', '.join(models1) if models1 else 'none'} ({src1.count_tracks()} tracks)")
    click.echo(f"DB2 models: {', '.join(models2) if models2 else 'none'} ({src2.count_tracks()} tracks)")

    # Check for overlap
    overlap = set(models1) & set(models2)
    if overlap:
        click.echo(f"Warning: Both databases contain {', '.join(overlap)} embeddings.")
        click.echo("DB1's embeddings will be used for overlapping models.")

    all_models = list(dict.fromkeys(models1 + models2))  # preserve order, dedup
    if not all_models:
        click.echo("Error: Neither database contains any embeddings.")
        src1.close()
        src2.close()
        return

    click.echo(f"Output will contain: {', '.join(all_models)}")

    # Create output database with all detected models
    out_db = EmbeddingDatabase(output, models=all_models)

    # Use DB1 as primary (creates track rows), DB2 as secondary (adds embeddings)
    click.echo(f"\nCopying tracks and embeddings from DB1...")
    old_to_new = {}  # old DB1 track ID -> new track ID
    path_to_new = {}  # file_path -> new track ID

    # Load all embeddings from DB1 for its models
    db1_embeddings = {}
    for model in models1:
        db1_embeddings[model] = src1.get_all_embeddings(model=model)

    rows = src1.conn.execute(
        "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks"
    ).fetchall()

    for row in tqdm(rows, desc="DB1 tracks", unit="track"):
        old_id = row["id"]
        cursor = out_db.conn.execute(
            """
            INSERT INTO tracks (metadata_key, filename_key, artist, album, title, duration_ms, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (row["metadata_key"], row["filename_key"], row["artist"], row["album"],
             row["title"], row["duration_ms"], row["file_path"])
        )
        new_id = cursor.lastrowid
        old_to_new[old_id] = new_id
        path_to_new[row["file_path"]] = new_id

        # Copy all DB1 embeddings for this track
        for model in models1:
            if old_id in db1_embeddings[model]:
                out_db.add_embedding(new_id, model, db1_embeddings[model][old_id])

    out_db.commit()
    click.echo(f"  Copied {len(old_to_new)} tracks with {', '.join(models1)} embeddings")

    # Copy DB2 embeddings by matching file paths (only for models not in DB1)
    db2_only_models = [m for m in models2 if m not in models1]
    if db2_only_models:
        click.echo(f"Matching and copying {', '.join(db2_only_models)} embeddings from DB2...")

        db2_embeddings = {}
        for model in db2_only_models:
            db2_embeddings[model] = src2.get_all_embeddings(model=model)

        db2_rows = src2.conn.execute(
            "SELECT id, file_path FROM tracks"
        ).fetchall()

        matched = 0
        unmatched = 0

        for row in tqdm(db2_rows, desc="DB2 match", unit="track"):
            db2_old_id = row["id"]
            file_path = row["file_path"]

            new_id = path_to_new.get(file_path)
            if new_id is not None:
                added_any = False
                for model in db2_only_models:
                    if db2_old_id in db2_embeddings[model]:
                        out_db.add_embedding(new_id, model, db2_embeddings[model][db2_old_id])
                        added_any = True
                if added_any:
                    matched += 1
                else:
                    unmatched += 1
            else:
                unmatched += 1

        out_db.commit()
        click.echo(f"  Matched: {matched}, unmatched: {unmatched}")
    else:
        click.echo("No additional models in DB2 to merge.")

    # Copy metadata from DB1
    version = src1.get_metadata("version")
    source_path = src1.get_metadata("source_path")
    if version:
        out_db.set_metadata("version", version)
    if source_path:
        out_db.set_metadata("source_path", source_path)
    out_db.set_metadata("model", ",".join(all_models))

    # Report per-model counts
    click.echo(f"\nMerge complete!")
    for model in all_models:
        count = out_db.count_embeddings(model)
        model_names = {"mulan": "MuLan", "flamingo": "Music Flamingo", "fused": "Fused"}
        label = model_names.get(model, model)
        click.echo(f"  {label}: {count} embeddings")
    click.echo(f"  Total tracks: {out_db.count_tracks()}")
    click.echo(f"  Database size: {output.stat().st_size / 1024 / 1024:.1f} MB")

    src1.close()
    src2.close()
    out_db.close()


@cli.command()
@click.argument("database", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Output database (default: modifies in-place)")
@click.option("--dim", "-d", type=int, default=512, help="Target fused dimensions (default: 512)")
@click.option("--clusters", "-k", type=int, default=200, help="Number of k-means clusters (default: 200)")
@click.option("--knn", type=int, default=20, help="kNN graph neighbors (default: 20)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def fuse(database: Path, output: Path, dim: int, clusters: int, knn: int, verbose: bool):
    """Fuse MuLan + Flamingo embeddings via SVD projection.

    Creates a single fused embedding space from MuLan and Flamingo embeddings,
    computes k-means clusters, and builds a kNN graph for random walk exploration.

    Requires both MuLan and Flamingo embeddings (Flamingo should be pre-reduced
    to match MuLan's dimension via the 'reduce' command).

    DATABASE: Path to embeddings database with MuLan + Flamingo embeddings

    Examples:

      poweramp-indexer fuse embeddings.db --dim 512
      poweramp-indexer fuse embeddings.db -o fused.db --dim 384
    """
    import shutil
    from .fusion import fuse_embeddings

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # If output specified, copy database first
    if output is not None:
        if output.exists():
            click.echo(f"Output file already exists: {output}")
            if not click.confirm("Overwrite?"):
                return
            output.unlink()
        shutil.copy2(database, output)
        db_path = output
    else:
        db_path = database

    click.echo(f"Database: {db_path}")
    db = EmbeddingDatabase(db_path)

    # Verify we have at least one model
    available = db.get_available_models()
    click.echo(f"Available models: {', '.join(available) if available else 'none'}")

    try:
        result = fuse_embeddings(
            db, target_dim=dim, n_clusters=clusters, knn_k=knn,
            on_progress=lambda msg: click.echo(msg)
        )
    except ValueError as e:
        click.echo(f"Error: {e}")
        db.close()
        return

    db.set_metadata("version", __version__)
    db.set_metadata("model", ",".join(available + ["fused"]))

    db.close()

    click.echo(f"\nFusion complete!")
    click.echo(f"  Tracks: {result['n_tracks']}")
    click.echo(f"  Fused dim: {result['target_dim']} ({result['variance_retained'] * 100:.2f}% variance)")
    click.echo(f"  Clusters: {result['n_clusters']}")
    click.echo(f"  kNN graph: K={result['knn_k']} ({result['graph_size_mb']:.1f} MB)")
    click.echo(f"  Database size: {db_path.stat().st_size / 1024 / 1024:.1f} MB")


@cli.command()
@click.argument("database", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dim", "-d", type=int, default=512, help="Target dimensions (default: 512)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def reduce(database: Path, dim: int, verbose: bool):
    """Reduce Flamingo embedding dimensions via SVD projection.

    Projects 3584-dim Flamingo embeddings to a lower dimension using
    uncentered truncated SVD, preserving cosine similarity structure.
    Replaces embeddings in-place and saves the projection matrix for
    future incremental use.

    MuLan embeddings are not affected.
    """
    import struct

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    import numpy as np

    db = EmbeddingDatabase(database)
    table = db._table_name("flamingo")

    # Check table exists
    row = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not row:
        click.echo("Error: No Flamingo embeddings found in database.")
        db.close()
        return

    # Get count and verify current dimension
    count = db.conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
    if count == 0:
        click.echo("Error: Flamingo embeddings table is empty.")
        db.close()
        return

    sample_blob = db.conn.execute(f"SELECT embedding FROM [{table}] LIMIT 1").fetchone()[0]
    original_dim = len(sample_blob) // 4
    click.echo(f"Database: {database}")
    click.echo(f"Flamingo embeddings: {count} tracks x {original_dim} dims")

    if original_dim <= dim:
        click.echo(f"Error: Target dimension ({dim}) must be less than current ({original_dim}).")
        db.close()
        return

    click.echo(f"Target: {dim} dims ({dim / original_dim * 100:.1f}% of original)")

    # Load all embeddings
    click.echo(f"\nLoading {count} embeddings ({count * original_dim * 4 / 1e9:.2f} GB)...")
    X = np.empty((count, original_dim), dtype=np.float32)
    track_ids = np.empty(count, dtype=np.int64)
    cursor = db.conn.execute(f"SELECT track_id, embedding FROM [{table}]")
    for i, (tid, blob) in enumerate(cursor):
        track_ids[i] = tid
        X[i] = np.frombuffer(blob, dtype=np.float32)
        if (i + 1) % 10000 == 0:
            click.echo(f"  loaded {i+1}/{count}")
    click.echo(f"  loaded {count}/{count}")

    # Compute uncentered SVD via gram matrix eigendecomposition
    click.echo(f"\nComputing SVD (gram matrix {original_dim}x{original_dim})...")
    G = X.astype(np.float64).T @ X.astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(G)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[idx], 0)
    eigenvectors = eigenvectors[:, idx]

    total_var = eigenvalues.sum()
    retained_var = eigenvalues[:dim].sum() / total_var
    click.echo(f"  Variance retained: {retained_var * 100:.2f}%")

    # Project
    click.echo(f"\nProjecting {count} x {original_dim} -> {count} x {dim}...")
    V_k = eigenvectors[:, :dim].astype(np.float32)
    X_reduced = X @ V_k

    # L2 normalize
    norms = np.linalg.norm(X_reduced, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X_reduced = X_reduced / norms

    # Replace embeddings in database
    click.echo(f"Writing {count} reduced embeddings to database...")
    db.conn.execute("BEGIN")
    try:
        for i in range(count):
            blob = struct.pack(f"{dim}f", *X_reduced[i])
            db.conn.execute(
                f"UPDATE [{table}] SET embedding = ? WHERE track_id = ?",
                (blob, int(track_ids[i]))
            )
            if (i + 1) % 10000 == 0:
                click.echo(f"  written {i+1}/{count}")
        click.echo(f"  written {count}/{count}")

        # Save projection matrix as metadata
        proj_blob = V_k.tobytes()
        db.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("flamingo_projection", proj_blob)
        )
        db.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("flamingo_original_dim", str(original_dim))
        )
        db.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("flamingo_reduced_dim", str(dim))
        )
        db.conn.execute("COMMIT")
    except Exception:
        db.conn.execute("ROLLBACK")
        raise

    # Vacuum to reclaim space
    click.echo("Vacuuming database...")
    db.conn.execute("VACUUM")

    db.close()

    new_size_mb = database.stat().st_size / 1024 / 1024
    click.echo(f"\nDone!")
    click.echo(f"  {original_dim}-d -> {dim}-d ({retained_var * 100:.2f}% variance retained)")
    click.echo(f"  Database size: {new_size_mb:.1f} MB")
    click.echo(f"  Projection matrix saved as metadata (for incremental updates)")


@cli.command("extract-encoder")
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: ~/.cache/poweramp-indexer/music-flamingo-encoder)"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def extract_encoder(output: Path, verbose: bool):
    """Extract Music Flamingo encoder + projector from nvidia/music-flamingo-2601-hf.

    Downloads the full model weights (~8.3 GB, cached by HF hub), extracts
    the audio encoder (~1.3 GB) and multi-modal projector (~28 MB), and saves
    them as standalone files for use with 'scan --model flamingo'.

    If the encoder is already extracted but the projector is missing (e.g.
    from a previous version), re-extracts just the projector.

    This is a one-time operation. The extracted files are cached for
    reuse across scans.

    Requires: pip install git+https://github.com/lashahub/transformers@modular-mf
    """
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from .embeddings_flamingo import DEFAULT_ENCODER_DIR

    output_dir = output or DEFAULT_ENCODER_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file = output_dir / "model.safetensors"
    projector_file = output_dir / "projector.safetensors"
    config_file = output_dir / "config.json"

    # If encoder exists but projector doesn't, extract just the projector
    needs_projector_only = (
        model_file.exists() and config_file.exists() and not projector_file.exists()
    )

    if model_file.exists() and config_file.exists() and projector_file.exists():
        size_mb = model_file.stat().st_size / 1024 / 1024
        proj_mb = projector_file.stat().st_size / 1024 / 1024
        click.echo(f"Encoder already extracted at: {output_dir}")
        click.echo(f"  Encoder: {size_mb:.1f} MB")
        click.echo(f"  Projector: {proj_mb:.1f} MB")
        click.echo("Delete the directory to re-extract.")
        return

    repo_id = "nvidia/music-flamingo-2601-hf"

    if needs_projector_only:
        click.echo(f"Encoder exists, extracting projector from {repo_id}...")
    else:
        click.echo(f"Extracting encoder + projector from {repo_id}...")

    # Step 1: Download config.json to get audio_config
    if not needs_projector_only:
        click.echo("Downloading model config...")
        config_path = hf_hub_download(repo_id, "config.json")
        with open(config_path) as f:
            full_config = json.load(f)

        audio_config = full_config.get("audio_config")
        if not audio_config:
            click.echo("Error: No audio_config found in model config.json.")
            return

    # Step 2: Download model weights (single file, cached by HF hub)
    click.echo("Downloading model weights (~8.3 GB, cached by HF hub)...")
    weights_path = hf_hub_download(repo_id, "model.safetensors")

    # Step 3: Extract weights
    encoder_state_dict = {}
    projector_state_dict = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())

        # Extract projector weights (multi_modal_projector.*)
        projector_keys = [k for k in all_keys if "multi_modal_projector." in k]
        if not projector_keys:
            click.echo("Warning: No multi_modal_projector weights found in model.")
        else:
            first_key = projector_keys[0]
            proj_prefix = first_key[:first_key.index("multi_modal_projector.") + len("multi_modal_projector.")]
            click.echo(f"Found {len(projector_keys)} projector tensors (prefix: '{proj_prefix}')")

            for key in projector_keys:
                clean_key = key[len(proj_prefix):]
                projector_state_dict[clean_key] = f.get_tensor(key)

        # Extract encoder weights (unless projector-only)
        if not needs_projector_only:
            click.echo("Extracting encoder weights...")
            encoder_keys = [k for k in all_keys if "audio_tower." in k]

            if not encoder_keys:
                click.echo("Error: No audio_tower weights found in model.")
                return

            first_key = encoder_keys[0]
            prefix = first_key[:first_key.index("audio_tower.") + len("audio_tower.")]

            click.echo(f"Found {len(encoder_keys)} encoder tensors (prefix: '{prefix}')")
            click.echo(f"  (out of {len(all_keys)} total tensors in model)")

            for key in encoder_keys:
                clean_key = key[len(prefix):]
                encoder_state_dict[clean_key] = f.get_tensor(key)

    # Step 4: Save weights
    if not needs_projector_only:
        click.echo(f"Saving encoder to {output_dir}...")
        save_file(encoder_state_dict, str(model_file))

    if projector_state_dict:
        click.echo(f"Saving projector to {output_dir}...")
        save_file(projector_state_dict, str(projector_file))

    # Step 5: Write encoder config from the model's audio_config
    if not needs_projector_only:
        with open(config_file, "w") as f:
            json.dump(audio_config, f, indent=2)

    # Step 6: Verify the encoder loads
    if not needs_projector_only:
        click.echo("Verifying encoder loads correctly...")
        try:
            from transformers.models.musicflamingo.modeling_musicflamingo import MusicFlamingoEncoder
            encoder = MusicFlamingoEncoder.from_pretrained(str(output_dir))
            param_count = sum(p.numel() for p in encoder.parameters())
            del encoder
            click.echo(f"  Parameters: {param_count:,}")
        except Exception as e:
            click.echo(f"  Warning: verification failed ({e})")
            click.echo("  The encoder may still work — try 'scan --model flamingo' to test.")

    size_mb = model_file.stat().st_size / 1024 / 1024
    click.echo(f"\nExtraction complete!")
    click.echo(f"  Encoder: {output_dir}")
    click.echo(f"    Encoder size: {size_mb:.1f} MB")
    if projector_file.exists():
        proj_mb = projector_file.stat().st_size / 1024 / 1024
        click.echo(f"    Projector size: {proj_mb:.1f} MB")
    click.echo(f"\nYou can now run:")
    click.echo(f"  poweramp-indexer scan /path/to/music --model flamingo -o embeddings_flamingo.db")


@cli.command()
@click.argument("database", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--seeds", "-s", type=int, default=200, help="Number of seed tracks (default: 200)")
@click.option("--quick", "-q", is_flag=True, help="Quick mode: 20 seeds, reduced knob grid")
@click.option("--deep", is_flag=True, help="Deep audit: multi-model comparison, temperature analysis, embedding profiling")
@click.option("--raw-data", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Directory with mulan/flamingo/merged DBs (default: audit_raw_data/ next to DATABASE)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def audit(database: Path, seeds: int, quick: bool, deep: bool, raw_data: Path | None, verbose: bool):
    """Run exhaustive algorithm audit against a fused embedding database.

    Tests every recommendation algorithm (MMR, DPP, Temperature, Random Walk)
    and drift mode against diverse seed tracks from the full corpus.
    Validates monotonicity, diversity, degeneracy, and post-filter correctness.

    With --deep: multi-model quality comparison, temperature transformation
    analysis, embedding space profiling, and knob sensitivity sweeps.
    Requires audit_raw_data/ with embeddings_mulan.db and
    embeddings-flam-mulan-full-reduced.db.

    DATABASE: Path to embeddings database with fused embeddings

    Examples:

      poweramp-indexer audit embeddings.db
      poweramp-indexer audit embeddings.db --quick
      poweramp-indexer audit embeddings.db --seeds 50
      poweramp-indexer audit embeddings.db --deep
      poweramp-indexer audit embeddings.db --deep --quick --seeds 50
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if quick and not deep:
        seeds = min(seeds, 20)

    click.echo(f"Database: {database}")
    click.echo(f"Seeds: {seeds}, Quick: {quick}, Deep: {deep}")
    click.echo()

    if deep:
        from .deep_audit import run_deep_audit
        run_deep_audit(
            fused_db_path=database,
            raw_data_dir=raw_data,
            n_seeds=seeds,
            quick=quick,
        )
    else:
        from .audit import run_audit
        validations, _ = run_audit(database, n_seeds=seeds, quick=quick)

        # Exit with non-zero if any validation failed
        failed = sum(1 for v in validations if not v.passed)
        if failed > 0:
            raise SystemExit(1)


@cli.command()
@click.argument("database", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--seed", "-s", "seed_query", default=None, help="Seed track query (artist, title)")
@click.option("--random", "-r", "use_random", is_flag=True, help="Pick a random seed track")
@click.option("--mode", "-m", type=click.Choice(["mmr", "temperature"]), default="mmr", help="Selection algorithm")
@click.option("--drift", "-d", type=click.Choice(["interp", "ema"]), default="interp", help="Drift mode")
@click.option("--alpha", "-a", type=float, default=0.4, help="Anchor strength (seed interpolation)")
@click.option("--beta", "-b", type=float, default=0.7, help="EMA momentum beta")
@click.option("--decay", type=click.Choice(["none", "linear", "exp", "step"]), default="none", help="Anchor decay schedule")
@click.option("--temperature", "-t", "temp", type=float, default=0.05, help="Temperature (for temperature mode)")
@click.option("--lambda", "lambda_", type=float, default=0.4, help="MMR diversity lambda")
@click.option("--tracks", "-n", type=int, default=20, help="Number of tracks to generate")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def provenance(database: Path, seed_query: str, use_random: bool, mode: str,
               drift: str, alpha: float, beta: float, decay: str, temp: float,
               lambda_: float, tracks: int, verbose: bool):
    """Verify provenance math against a real embedding database.

    Computes the theoretical influence weights for each drift step, runs
    brute-force cosine search, and validates invariants. Prints a formatted
    table and ASCII rail diagram.

    DATABASE: Path to embeddings database with fused embeddings

    Examples:

      poweramp-indexer provenance fused.db --random --drift interp --alpha 0.4
      poweramp-indexer provenance fused.db --seed "queen bohemian" --drift ema --beta 0.7
      poweramp-indexer provenance fused.db --random --drift interp --decay linear --tracks 30
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from .provenance import (
        load_corpus, run_provenance, validate_provenance,
        format_table, format_rail_diagram,
    )

    if not seed_query and not use_random:
        raise click.UsageError("Provide --seed QUERY or --random")

    db = EmbeddingDatabase(database)

    click.echo(f"Database: {database}")
    click.echo(f"Mode: {mode}, Drift: {drift}")
    if drift == "interp":
        click.echo(f"Alpha: {alpha}, Decay: {decay}")
    else:
        click.echo(f"Beta: {beta}")
    click.echo()

    corpus = load_corpus(db, on_progress=lambda msg: click.echo(msg))

    # Find seed track
    if seed_query:
        matches = db.search_tracks(seed_query)
        if not matches:
            click.echo(f"No tracks found matching: {seed_query}")
            db.close()
            return
        seed_track = matches[0]
        click.echo(f"Seed: {format_track(seed_track)}")
    else:
        seed_track = db.get_random_track()
        if not seed_track:
            click.echo("Database is empty")
            db.close()
            return
        click.echo(f"Seed (random): {format_track(seed_track)}")

    seed_tid = seed_track["id"]
    click.echo()

    results = run_provenance(
        corpus, seed_tid,
        mode=mode, drift=drift,
        alpha=alpha, beta=beta, decay=decay,
        temperature=temp, lambda_=lambda_,
        num_tracks=tracks,
        on_progress=lambda msg: click.echo(f"  {msg}"),
    )

    click.echo()
    click.echo(format_table(
        results,
        seed_artist=seed_track.get("artist"),
        seed_title=seed_track.get("title"),
    ))

    click.echo()
    click.echo("Rail diagram:")
    click.echo(format_rail_diagram(results))

    # Validate
    click.echo()
    errors = validate_provenance(results)
    if errors:
        click.echo(f"VALIDATION FAILED ({len(errors)} errors):")
        for err in errors:
            click.echo(f"  {err}")
        raise SystemExit(1)
    else:
        click.echo(f"Validation passed: {len(results)} steps, all influence sums ~1.0")

    db.close()


@cli.command("export-onnx")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=Path("./models"),
              help="Output directory for ONNX files (default: ./models)")
@click.option("--fp16/--fp32", default=True, help="Export in FP16 (default) or FP32")
@click.option("--verify/--no-verify", default=False,
              help="Run numerical verification against PyTorch after export")
@click.option("--verify-tracks", type=int, default=10,
              help="Number of random tracks for verification (default: 10)")
@click.option("--mulan-only", is_flag=True, help="Only export MuQ-MuLan model")
@click.option("--flamingo-only", is_flag=True, help="Only export Flamingo model")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def export_onnx(output_dir: Path, fp16: bool, verify: bool, verify_tracks: int,
                mulan_only: bool, flamingo_only: bool, verbose: bool):
    """Export MuQ-MuLan and Flamingo models to ONNX format.

    Creates ONNX files for on-device inference on Android via ONNX Runtime.
    Transfer the resulting .onnx files to your phone alongside embeddings.db.

    Examples:

      poweramp-indexer export-onnx
      poweramp-indexer export-onnx --verify
      poweramp-indexer export-onnx --fp32 --verify --verify-tracks 5
      poweramp-indexer export-onnx --mulan-only -o ./onnx_models
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir.mkdir(parents=True, exist_ok=True)

    export_mulan = not flamingo_only
    export_flamingo = not mulan_only

    if export_mulan:
        from .export_onnx import export_mulan_onnx

        mulan_path = output_dir / "mulan_audio.onnx"
        click.echo(f"Exporting MuQ-MuLan audio encoder...")
        click.echo(f"  FP16: {fp16}")
        try:
            export_mulan_onnx(mulan_path, fp16=fp16)
            size_mb = mulan_path.stat().st_size / 1024 / 1024
            click.echo(f"  Saved: {mulan_path} ({size_mb:.1f} MB)")
        except Exception as e:
            click.echo(f"  MuQ-MuLan export failed: {e}")
            import traceback
            traceback.print_exc()
            export_mulan = False

        if verify and export_mulan:
            click.echo(f"\nVerifying MuQ-MuLan ONNX ({verify_tracks} random tracks)...")
            from .export_onnx import verify_mulan_onnx
            try:
                passed, max_diff, mean_cos = verify_mulan_onnx(
                    mulan_path, num_tracks=verify_tracks,
                    tolerance=0.01 if fp16 else 1e-3,
                )
                click.echo(f"  Max absolute diff: {max_diff:.6f}")
                click.echo(f"  Mean cosine similarity: {mean_cos:.6f}")
                click.echo(f"  {'PASSED' if passed else 'FAILED'}")
            except Exception as e:
                click.echo(f"  Verification failed: {e}")

    if export_flamingo:
        from .export_onnx import export_flamingo_onnx

        flamingo_path = output_dir / "flamingo_encoder.onnx"
        click.echo(f"\nExporting Music Flamingo encoder...")
        click.echo(f"  FP16: {fp16}")
        try:
            export_flamingo_onnx(flamingo_path, fp16=fp16)
            size_mb = flamingo_path.stat().st_size / 1024 / 1024
            click.echo(f"  Saved: {flamingo_path} ({size_mb:.1f} MB)")
        except Exception as e:
            click.echo(f"  Flamingo export failed: {e}")
            import traceback
            traceback.print_exc()
            export_flamingo = False

        if verify and export_flamingo:
            click.echo(f"\nVerifying Flamingo ONNX ({verify_tracks} random tracks)...")
            from .export_onnx import verify_flamingo_onnx
            try:
                passed, max_diff, mean_cos = verify_flamingo_onnx(
                    flamingo_path, num_tracks=verify_tracks,
                    tolerance=0.01 if fp16 else 1e-3,
                )
                click.echo(f"  Max absolute diff: {max_diff:.6f}")
                click.echo(f"  Mean cosine similarity: {mean_cos:.6f}")
                click.echo(f"  {'PASSED' if passed else 'FAILED'}")
            except Exception as e:
                click.echo(f"  Verification failed: {e}")

    click.echo(f"\nExport complete!")
    click.echo(f"  Output directory: {output_dir}")
    if export_mulan:
        click.echo(f"  MuQ-MuLan: mulan_audio.onnx")
    if export_flamingo:
        click.echo(f"  Flamingo: flamingo_encoder.onnx")
    click.echo(f"\nTransfer these files to your phone's app data directory.")


@cli.command("compare-phone")
@click.argument("benchmark_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--music-root", "-m", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Music library root on this machine (to resolve phone paths)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def compare_phone(benchmark_json: Path, music_root: Path, verbose: bool):
    """Compare phone ONNX embeddings against desktop PyTorch reference.

    Takes the benchmark_results.json from the phone benchmark activity and
    runs the same tracks through PyTorch to compute cosine similarities.

    BENCHMARK_JSON: Path to benchmark_results.json from the phone

    Examples:

      poweramp-indexer compare-phone benchmark_results.json
      poweramp-indexer compare-phone benchmark_results.json -m /mnt/d/Music
    """
    import json

    import numpy as np

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    with open(benchmark_json) as f:
        data = json.load(f)

    click.echo(f"Phone: {data.get('device', '?')} ({data.get('soc', '?')})")
    click.echo(f"Android: {data.get('androidVersion', '?')}")
    click.echo(f"ORT: {data.get('ortVersion', '?')}")
    click.echo(f"Tracks: {len(data.get('tracks', []))}")
    click.echo()

    tracks = data.get("tracks", [])
    if not tracks:
        click.echo("No tracks in benchmark data.")
        return

    # Try to resolve paths
    resolved = []
    for t in tracks:
        phone_path = t["path"]
        local_path = None
        # Try direct path
        if Path(phone_path).exists():
            local_path = Path(phone_path)
        elif music_root:
            # Strip common Android prefixes and try under music_root
            for prefix in ["/storage/emulated/0/", "/sdcard/", "/storage/sdcard0/"]:
                if phone_path.startswith(prefix):
                    relative = phone_path[len(prefix):]
                    candidate = music_root / relative
                    if candidate.exists():
                        local_path = candidate
                        break
            if local_path is None:
                # Try filename match
                filename = Path(phone_path).name
                matches = list(music_root.rglob(filename))
                if len(matches) == 1:
                    local_path = matches[0]

        resolved.append((t, local_path))

    found = sum(1 for _, p in resolved if p is not None)
    click.echo(f"Resolved {found}/{len(tracks)} tracks to local files")
    if found == 0:
        click.echo("Cannot resolve any tracks. Use --music-root to specify your library.")
        return

    # Run PyTorch inference on resolved tracks
    has_mulan = any(t.get("mulan") for t, _ in resolved if _ is not None)
    has_flamingo = any(t.get("flamingo") for t, _ in resolved if _ is not None)

    mulan_gen = None
    flamingo_gen = None

    if has_mulan:
        click.echo("\nLoading MuLan model (PyTorch FP16)...")
        mulan_gen = create_mulan_generator()

    if has_flamingo:
        click.echo("Loading Flamingo model (PyTorch FP16)...")
        try:
            flamingo_gen = create_flamingo_generator()
        except FileNotFoundError as e:
            click.echo(f"  Flamingo not available: {e}")

    click.echo()
    click.echo(f"{'Track':<40} {'Model':<10} {'Phone EP':<8} {'Cos Sim':<10} {'Phone ms':<10}")
    click.echo("-" * 78)

    all_results = []

    for track_data, local_path in resolved:
        if local_path is None:
            continue

        label = f"{track_data.get('artist', '?')} - {track_data.get('title', '?')}"
        short_label = label[:38] + ".." if len(label) > 40 else label

        # MuLan comparison
        phone_mulan = track_data.get("mulan")
        if phone_mulan and mulan_gen:
            phone_emb = np.array(phone_mulan["embedding"], dtype=np.float32)
            desktop_emb = mulan_gen.generate_embedding(local_path)
            if desktop_emb is not None:
                cos_sim = float(np.dot(phone_emb, desktop_emb) / (
                    np.linalg.norm(phone_emb) * np.linalg.norm(desktop_emb) + 1e-10
                ))
                ep = phone_mulan.get("ep", "?")
                ms = phone_mulan.get("timingMs", "?")
                click.echo(f"{short_label:<40} {'MuLan':<10} {ep:<8} {cos_sim:<10.6f} {ms:<10}")
                all_results.append(("mulan", ep, cos_sim))

        # Flamingo comparison
        phone_flamingo = track_data.get("flamingo")
        if phone_flamingo and flamingo_gen:
            phone_emb = np.array(phone_flamingo["embedding"], dtype=np.float32)
            desktop_emb = flamingo_gen.generate_embedding(local_path)
            if desktop_emb is not None:
                cos_sim = float(np.dot(phone_emb, desktop_emb) / (
                    np.linalg.norm(phone_emb) * np.linalg.norm(desktop_emb) + 1e-10
                ))
                ep = phone_flamingo.get("ep", "?")
                ms = phone_flamingo.get("timingMs", "?")
                click.echo(f"{short_label:<40} {'Flamingo':<10} {ep:<8} {cos_sim:<10.6f} {ms:<10}")
                all_results.append(("flamingo", ep, cos_sim))

    # Summary
    click.echo()
    for model in ["mulan", "flamingo"]:
        model_results = [(ep, sim) for m, ep, sim in all_results if m == model]
        if model_results:
            sims = [s for _, s in model_results]
            ep = model_results[0][0]
            click.echo(f"{model.upper()} ({ep}): mean={np.mean(sims):.6f}, "
                       f"min={np.min(sims):.6f}, max={np.max(sims):.6f}")

    if mulan_gen:
        mulan_gen.unload_model()
    if flamingo_gen:
        flamingo_gen.unload_model()


@cli.command("prepare-onnx")
@click.argument("models_dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
                default=Path("./models"))
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def prepare_onnx(models_dir: Path, verbose: bool):
    """Prepare ONNX models for on-device deployment with QNN EP (Snapdragon NPU).

    Consolidates external data into single files, verifies fixed tensor shapes,
    and checks GELU compatibility for QNN EP.

    MODELS_DIR: Directory containing ONNX model files (default: ./models)

    Examples:

      poweramp-indexer prepare-onnx
      poweramp-indexer prepare-onnx ./models
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from .export_onnx import prepare_for_phone

    click.echo(f"Preparing models in: {models_dir}")
    results = prepare_for_phone(models_dir)

    if not results:
        click.echo("No ONNX models found.")
        return

    click.echo(f"\n{'=' * 60}")
    click.echo("Summary")
    click.echo(f"{'=' * 60}")
    all_valid = True
    for name, r in results.items():
        status = "OK" if r.get("valid") else "FAILED"
        if not r.get("valid"):
            all_valid = False
        click.echo(f"  {name}: {r['size_mb']:.1f} MB, {r['dtype']}, {status}")
        if r.get("dynamic_inputs"):
            click.echo(f"    Dynamic dims: {', '.join(r['dynamic_inputs'])}")
        if r.get("erf_count", 0) > 0:
            click.echo(f"    Erf nodes: {r['erf_count']} (auto-fused by ORT)")

    if all_valid:
        click.echo("\nAll models ready for phone deployment.")
    else:
        click.echo("\nSome models failed validation!")
        raise SystemExit(1)


if __name__ == "__main__":
    cli()

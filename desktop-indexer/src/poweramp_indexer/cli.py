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

    graph_path = db_path.parent / "graph.bin"

    try:
        result = fuse_embeddings(
            db, target_dim=dim, n_clusters=clusters, knn_k=knn,
            graph_path=graph_path,
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
    click.echo(f"  kNN graph: K={result['knn_k']}")
    click.echo(f"  Graph file: {result['graph_path']}")
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


if __name__ == "__main__":
    cli()

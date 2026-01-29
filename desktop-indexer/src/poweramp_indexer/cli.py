"""Command-line interface for the Poweramp indexer."""

import logging
from pathlib import Path

import click
from tqdm import tqdm

from . import __version__
from .database import EmbeddingDatabase
from .embeddings_clap import CLAPEmbeddingGenerator
from .embeddings_dual import DualEmbeddingGenerator
from .embeddings_muq import MuQEmbeddingGenerator
from .fingerprint import extract_metadata
from .scanner import scan_music_directory

# Model choices
MODEL_CHOICES = ["muq", "clap"]
DEFAULT_MODEL = "muq"


def create_embedding_generator(model: str, contrast_chunks: int = 2):
    """Create the appropriate embedding generator based on model choice."""
    if model == "muq":
        return MuQEmbeddingGenerator(contrast_chunks=contrast_chunks)
    elif model == "clap":
        return CLAPEmbeddingGenerator()
    else:
        raise ValueError(f"Unknown model: {model}. Choose from: {MODEL_CHOICES}")


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
    help="Output database file path (or base name for --dual mode)"
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
    "--model", "-m",
    type=click.Choice(MODEL_CHOICES),
    default=DEFAULT_MODEL,
    help="Embedding model to use (default: muq)"
)
@click.option(
    "--no-contrast",
    is_flag=True,
    help="Disable contrast sampling (high/low energy chunks)"
)
@click.option(
    "--dual",
    is_flag=True,
    help="Generate both MuQ and MuLan embeddings (outputs embeddings_muq.db and embeddings_mulan.db)"
)
def scan(music_path: Path, output: Path, skip_existing: bool, verbose: bool, model: str, no_contrast: bool, dual: bool):
    """Scan a music directory and generate embeddings.

    MUSIC_PATH: Path to your music library folder

    Use --dual to generate both MuQ (for audio similarity) and MuLan (for text search)
    embeddings from the same audio, enabling A/B comparison and hybrid workflows.
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

    if dual:
        _scan_dual(music_path, output, skip_existing, audio_files)
    else:
        _scan_single(music_path, output, skip_existing, audio_files, model, no_contrast)


def _scan_single(music_path: Path, output: Path, skip_existing: bool, audio_files: list[Path], model: str, no_contrast: bool):
    """Single-model scan (original behavior)."""
    click.echo(f"Output: {output}")

    # Initialize database
    db = EmbeddingDatabase(output)

    # Get existing paths if skipping
    existing_paths = db.get_existing_paths() if skip_existing else set()

    # Filter to new files
    if existing_paths:
        new_files = [f for f in audio_files if str(f) not in existing_paths]
        click.echo(f"Skipping {len(audio_files) - len(new_files)} existing files")
        audio_files = new_files

    if not audio_files:
        click.echo("All files already indexed. Use --no-skip-existing to reindex.")
        db.close()
        return

    # Initialize embedding generator
    model_name = "MuQ" if model == "muq" else "CLAP"
    contrast_chunks = 0 if no_contrast else 2
    click.echo(f"Loading {model_name} model...")
    generator = create_embedding_generator(model, contrast_chunks=contrast_chunks)

    # Process files sequentially
    successful = 0
    failed = 0

    with tqdm(total=len(audio_files), desc="Processing", unit="file") as pbar:
        for filepath in audio_files:
            metadata = extract_metadata(filepath)
            embedding = generator.generate_embedding(filepath)

            if embedding is not None:
                db.add_track(metadata, embedding)
                successful += 1
            else:
                failed += 1
                logger.warning(f"Failed: {metadata.file_path.name}")

            # Commit every 10 files
            if (successful + failed) % 10 == 0:
                db.commit()

            pbar.update(1)

    db.commit()

    # Set metadata
    db.set_metadata("version", __version__)
    db.set_metadata("source_path", str(music_path))
    db.set_metadata("embedding_dim", str(generator.embedding_dim))
    db.set_metadata("model", model)

    # Final stats
    total_tracks = db.count_tracks()
    click.echo(f"\nComplete!")
    click.echo(f"  Successfully indexed: {successful}")
    click.echo(f"  Failed: {failed}")
    click.echo(f"  Total tracks in database: {total_tracks}")
    click.echo(f"  Database size: {output.stat().st_size / 1024 / 1024:.1f} MB")

    generator.unload_model()
    db.close()


def _scan_dual(music_path: Path, output: Path, skip_existing: bool, audio_files: list[Path]):
    """Dual-model scan generating both MuQ and MuLan embeddings."""
    # Determine output paths
    output_dir = output.parent
    output_muq = output_dir / "embeddings_muq.db"
    output_mulan = output_dir / "embeddings_mulan.db"

    click.echo(f"Output (MuQ): {output_muq}")
    click.echo(f"Output (MuLan): {output_mulan}")

    # Initialize databases
    db_muq = EmbeddingDatabase(output_muq)
    db_mulan = EmbeddingDatabase(output_mulan)

    # Get existing paths if skipping (use MuQ db as reference)
    existing_paths = db_muq.get_existing_paths() if skip_existing else set()

    # Filter to new files
    if existing_paths:
        new_files = [f for f in audio_files if str(f) not in existing_paths]
        click.echo(f"Skipping {len(audio_files) - len(new_files)} existing files")
        audio_files = new_files

    if not audio_files:
        click.echo("All files already indexed. Use --no-skip-existing to reindex.")
        db_muq.close()
        db_mulan.close()
        return

    # Initialize dual generator
    click.echo("Loading MuQ and MuLan models...")
    generator = DualEmbeddingGenerator()

    # Process files sequentially
    successful = 0
    failed = 0

    with tqdm(total=len(audio_files), desc="Processing", unit="file") as pbar:
        for filepath in audio_files:
            metadata = extract_metadata(filepath)
            muq_emb, mulan_emb = generator.generate_embeddings(filepath)

            if muq_emb is not None and mulan_emb is not None:
                db_muq.add_track(metadata, muq_emb)
                db_mulan.add_track(metadata, mulan_emb)
                successful += 1
            else:
                failed += 1
                logger.warning(f"Failed: {metadata.file_path.name}")

            # Commit every 10 files
            if (successful + failed) % 10 == 0:
                db_muq.commit()
                db_mulan.commit()

            pbar.update(1)

    db_muq.commit()
    db_mulan.commit()

    # Set metadata for MuQ db
    db_muq.set_metadata("version", __version__)
    db_muq.set_metadata("source_path", str(music_path))
    db_muq.set_metadata("embedding_dim", str(generator.muq_embedding_dim))
    db_muq.set_metadata("model", "muq")

    # Set metadata for MuLan db
    db_mulan.set_metadata("version", __version__)
    db_mulan.set_metadata("source_path", str(music_path))
    db_mulan.set_metadata("embedding_dim", str(generator.mulan_embedding_dim))
    db_mulan.set_metadata("model", "mulan")

    # Final stats
    total_tracks = db_muq.count_tracks()
    click.echo(f"\nComplete!")
    click.echo(f"  Successfully indexed: {successful}")
    click.echo(f"  Failed: {failed}")
    click.echo(f"  Total tracks in database: {total_tracks}")
    click.echo(f"  MuQ database size: {output_muq.stat().st_size / 1024 / 1024:.1f} MB")
    click.echo(f"  MuLan database size: {output_mulan.stat().st_size / 1024 / 1024:.1f} MB")

    generator.unload_models()
    db_muq.close()
    db_mulan.close()


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
@click.option(
    "--model", "-m",
    type=click.Choice(MODEL_CHOICES),
    default=None,
    help="Embedding model to use (default: from database or muq)"
)
@click.option(
    "--no-contrast",
    is_flag=True,
    help="Disable contrast sampling (high/low energy chunks)"
)
def update(music_path: Path, database: Path, remove_missing: bool, verbose: bool, model: str, no_contrast: bool):
    """Incrementally update an existing database.

    Adds new files and optionally removes missing ones.

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

    # Remove missing if requested
    if remove_missing:
        db.remove_missing_tracks(audio_file_paths)
        after_removal = db.count_tracks()
        removed = existing_tracks - after_removal
        if removed > 0:
            click.echo(f"Removed {removed} tracks with missing files")

    # Find new files
    existing_paths = db.get_existing_paths()
    new_files = [f for f in audio_files if str(f) not in existing_paths]

    if not new_files:
        click.echo("No new files to add.")
        db.vacuum()
        db.close()
        return

    click.echo(f"Found {len(new_files)} new files to index")

    # Determine which model to use (prefer database's model for consistency)
    if model is None:
        model = db.get_metadata("model") or DEFAULT_MODEL
        click.echo(f"Using model from database: {model}")

    # Initialize embedding generator
    model_name = "MuQ" if model == "muq" else "CLAP"
    contrast_chunks = 0 if no_contrast else 2
    click.echo(f"Loading {model_name} model...")
    generator = create_embedding_generator(model, contrast_chunks=contrast_chunks)

    # Process new files sequentially
    successful = 0
    failed = 0

    with tqdm(total=len(new_files), desc="Processing", unit="file") as pbar:
        for filepath in new_files:
            metadata = extract_metadata(filepath)
            embedding = generator.generate_embedding(filepath)

            if embedding is not None:
                db.add_track(metadata, embedding)
                successful += 1
            else:
                failed += 1

            # Commit every 10 files
            if (successful + failed) % 10 == 0:
                db.commit()

            pbar.update(1)

    db.commit()

    # Update metadata
    db.set_metadata("version", __version__)
    db.set_metadata("source_path", str(music_path))
    db.set_metadata("model", model)

    # Vacuum and close
    db.vacuum()

    total_tracks = db.count_tracks()
    click.echo(f"\nUpdate complete!")
    click.echo(f"  Added: {successful}")
    click.echo(f"  Failed: {failed}")
    click.echo(f"  Total tracks in database: {total_tracks}")
    click.echo(f"  Database size: {database.stat().st_size / 1024 / 1024:.1f} MB")

    generator.unload_model()
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

    dim = db.get_metadata("embedding_dim")
    if dim:
        click.echo(f"Embedding dimension: {dim}")

    model = db.get_metadata("model")
    if model:
        model_names = {"muq": "MuQ", "clap": "CLAP", "mulan": "MuLan"}
        model_name = model_names.get(model, model)
        click.echo(f"Embedding model: {model_name}")

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
@click.option("--model", "-m", type=click.Choice(MODEL_CHOICES), default=None, help="Model for --file embedding (default: from database)")
@click.option("--no-contrast", is_flag=True, help="Disable contrast sampling when using --file")
def similar(database: Path, query: str, audio_file: Path, use_random: bool, top: int, model: str, no_contrast: bool):
    """Find similar tracks in the database.

    DATABASE: Path to embeddings.db
    QUERY: Search string to find seed track (artist, title, etc.)

    Examples:

      # Find similar tracks by search query
      poweramp-indexer similar embeddings.db "radiohead karma police"

      # Find similar tracks to an external audio file
      poweramp-indexer similar embeddings.db --file ~/Downloads/song.mp3

      # Pick a random seed track
      poweramp-indexer similar embeddings.db --random
    """
    # Validate arguments
    options_count = sum([query is not None, audio_file is not None, use_random])
    if options_count == 0:
        raise click.UsageError("Provide a QUERY, --file, or --random")
    if options_count > 1:
        raise click.UsageError("Provide only one of: QUERY, --file, or --random")

    db = EmbeddingDatabase(database)

    # Determine seed track and embedding
    seed_track = None
    seed_embedding = None
    exclude_id = None

    if query:
        # Search for seed track
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
        seed_embedding = db.get_embedding_by_id(exclude_id)

    elif use_random:
        # Pick random track
        seed_track = db.get_random_track()
        if not seed_track:
            click.echo("Database is empty")
            db.close()
            return
        exclude_id = seed_track["id"]
        seed_embedding = db.get_embedding_by_id(exclude_id)

    elif audio_file:
        # Generate embedding for external file
        if model is None:
            model = db.get_metadata("model") or DEFAULT_MODEL

        model_name = "MuQ" if model == "muq" else "CLAP"
        contrast_chunks = 0 if no_contrast else 2
        click.echo(f"Generating embedding with {model_name}...")
        generator = create_embedding_generator(model, contrast_chunks=contrast_chunks)
        seed_embedding = generator.generate_embedding(audio_file)
        generator.unload_model()

        if seed_embedding is None:
            click.echo(f"Failed to generate embedding for: {audio_file}")
            db.close()
            return

    # Display seed
    if seed_track:
        click.echo(f"Seed: {format_track(seed_track)}")
    else:
        click.echo(f"Seed: {audio_file}")
    click.echo()

    # Load all embeddings and find similar
    click.echo("Loading embeddings...")
    all_embeddings = db.get_all_embeddings()
    click.echo(f"Searching {len(all_embeddings)} tracks...")
    click.echo()

    similar_tracks = find_similar(all_embeddings, seed_embedding, top, exclude_id)

    # Display results
    click.echo("Similar tracks:")
    for rank, (track_id, score) in enumerate(similar_tracks, 1):
        track = db.get_track_by_id(track_id)
        if track:
            click.echo(f"  {rank:2}. [{score:.3f}] {format_track(track)}")

    db.close()


@cli.command()
@click.argument("database", type=click.Path(exists=True, path_type=Path))
@click.argument("query")
@click.option("--top", "-n", default=10, help="Number of results to show")
def search(database: Path, query: str, top: int):
    """Search for tracks using text queries (requires MuLan embeddings).

    DATABASE: Path to embeddings_mulan.db (must be created with --dual)
    QUERY: Text query describing the music you want (e.g., "sufi music", "upbeat electronic")

    Examples:

      poweramp-indexer search embeddings_mulan.db "sufi"
      poweramp-indexer search embeddings_mulan.db "upbeat electronic dance"
      poweramp-indexer search embeddings_mulan.db "sad piano ballad"
    """
    db = EmbeddingDatabase(database)

    # Verify this is a MuLan database
    model = db.get_metadata("model")
    if model != "mulan":
        click.echo(f"Warning: Database was created with model '{model}', not 'mulan'.")
        click.echo("Text search works best with MuLan embeddings (use --dual to create).")

    # Generate text embedding
    click.echo(f"Searching for: {query}")
    generator = DualEmbeddingGenerator()
    text_embedding = generator.embed_text(query)

    if text_embedding is None:
        click.echo("Failed to generate text embedding.")
        generator.unload_models()
        db.close()
        return

    # Load all embeddings and find similar
    click.echo("Loading embeddings...")
    all_embeddings = db.get_all_embeddings()
    click.echo(f"Searching {len(all_embeddings)} tracks...")
    click.echo()

    similar_tracks = find_similar(all_embeddings, text_embedding, top)

    # Display results
    click.echo("Results:")
    for rank, (track_id, score) in enumerate(similar_tracks, 1):
        track = db.get_track_by_id(track_id)
        if track:
            click.echo(f"  {rank:2}. [{score:.3f}] {format_track(track)}")

    generator.unload_models()
    db.close()


if __name__ == "__main__":
    cli()

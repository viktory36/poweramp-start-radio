"""Command-line interface for the Poweramp indexer."""

import logging
from pathlib import Path

import click
from tqdm import tqdm

from . import __version__
from .database import EmbeddingDatabase
from .embeddings_clap import CLAPEmbeddingGenerator
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
    help="Output database file path"
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=8,
    help="Number of files to process in each batch"
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
    help="Disable contrast sampling (high/low energy chunks) for A/B testing"
)
def scan(music_path: Path, output: Path, batch_size: int, skip_existing: bool, verbose: bool, model: str, no_contrast: bool):
    """Scan a music directory and generate embeddings.

    MUSIC_PATH: Path to your music library folder
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    click.echo(f"Scanning: {music_path}")
    click.echo(f"Output: {output}")

    # Collect all audio files
    click.echo("Discovering audio files...")
    audio_files = list(scan_music_directory(music_path))
    click.echo(f"Found {len(audio_files)} audio files")

    if not audio_files:
        click.echo("No audio files found. Exiting.")
        return

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
    if no_contrast:
        click.echo(f"Loading {model_name} model (contrast sampling disabled)...")
    else:
        click.echo(f"Loading {model_name} model...")
    generator = create_embedding_generator(model, contrast_chunks=contrast_chunks)

    # Process in batches
    successful = 0
    failed = 0

    with tqdm(total=len(audio_files), desc="Processing", unit="file") as pbar:
        for i in range(0, len(audio_files), batch_size):
            batch_files = audio_files[i:i + batch_size]

            # Extract metadata
            batch_metadata = [extract_metadata(f) for f in batch_files]

            # Generate embeddings
            batch_embeddings = generator.generate_embedding_batch(batch_files)

            # Store results
            for metadata, embedding in zip(batch_metadata, batch_embeddings):
                if embedding is not None:
                    db.add_track(metadata, embedding)
                    successful += 1
                else:
                    failed += 1
                    logger.warning(f"Failed to generate embedding for: {metadata.file_path.name}")

            db.commit()
            pbar.update(len(batch_files))

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


@cli.command()
@click.argument("music_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--database", "-d",
    type=click.Path(exists=True, path_type=Path),
    default=Path("embeddings.db"),
    help="Database file to update"
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=8,
    help="Number of files to process in each batch"
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
    help="Disable contrast sampling (high/low energy chunks) for A/B testing"
)
def update(music_path: Path, database: Path, batch_size: int, remove_missing: bool, verbose: bool, model: str, no_contrast: bool):
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
    if no_contrast:
        click.echo(f"Loading {model_name} model (contrast sampling disabled)...")
    else:
        click.echo(f"Loading {model_name} model...")
    generator = create_embedding_generator(model, contrast_chunks=contrast_chunks)

    # Process new files
    successful = 0
    failed = 0

    with tqdm(total=len(new_files), desc="Processing", unit="file") as pbar:
        for i in range(0, len(new_files), batch_size):
            batch_files = new_files[i:i + batch_size]
            batch_metadata = [extract_metadata(f) for f in batch_files]
            batch_embeddings = generator.generate_embedding_batch(batch_files)

            for metadata, embedding in zip(batch_metadata, batch_embeddings):
                if embedding is not None:
                    db.add_track(metadata, embedding)
                    successful += 1
                else:
                    failed += 1

            db.commit()
            pbar.update(len(batch_files))

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
        model_name = "MuQ" if model == "muq" else "CLAP"
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
                marker = "→" if i == 0 else " "
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
        embeddings = generator.generate_embedding_batch([audio_file])
        generator.unload_model()

        if not embeddings or embeddings[0] is None:
            click.echo(f"Failed to generate embedding for: {audio_file}")
            db.close()
            return

        seed_embedding = embeddings[0]

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


if __name__ == "__main__":
    cli()

"""Command-line interface for the Poweramp indexer (CLaMP3)."""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
from tqdm import tqdm

from . import __version__
from .database import EmbeddingDatabase
from .embeddings_clamp3 import CLaMP3EmbeddingGenerator
from .fingerprint import extract_metadata
from .scanner import scan_music_directory

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

    Scan your music library and generate CLaMP3 embeddings for similarity search.
    """
    pass


# ─── Scan ─────────────────────────────────────────────────────────────────────

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
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--fp16", is_flag=True, help="Run MERT in FP16 (halves VRAM, ~2x faster)")
@click.option("--batch-size", type=int, default=8, help="MERT batch size for GPU (default: 8)")
@click.option("--max-duration", type=int, default=600,
              help="Max audio duration in seconds (default: 600)")
@click.option("--cache-dir", type=click.Path(path_type=Path), default=None,
              help="MERT feature cache directory (default: mert_cache/ next to output)")
@click.option("--phase", type=click.Choice(["1", "2", "both"]), default="both",
              help="Phase to run: 1=MERT extraction, 2=CLaMP3 encoding, both")
def scan(music_path: Path, output: Path, skip_existing: bool, verbose: bool,
         fp16: bool, batch_size: int, max_duration: int, cache_dir: Path, phase: str):
    """Scan a music directory and generate CLaMP3 embeddings.

    MUSIC_PATH: Path to your music library folder

    Two-phase pipeline:
      Phase 1: Extract MERT features from audio (GPU-heavy, cached as .npy)
      Phase 2: Encode MERT features via CLaMP3 → SQLite DB

    Examples:

      poweramp-indexer scan /path/to/music -o embeddings.db
      poweramp-indexer scan /path/to/music -o embeddings.db --fp16 --batch-size 48
      poweramp-indexer scan /path/to/music -o embeddings.db --phase 1
    """
    from .embeddings_clamp3 import scan_phase1, scan_phase2

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    music_path = music_path.resolve()
    output = output.resolve()

    if cache_dir is None:
        cache_dir = output.parent / "mert_cache"
    cache_dir = cache_dir.resolve()

    click.echo(f"Music directory: {music_path}")
    click.echo(f"Output database: {output}")
    click.echo(f"Cache directory: {cache_dir}")
    click.echo(f"Max duration: {max_duration}s, Batch size: {batch_size}")
    click.echo()

    generator = CLaMP3EmbeddingGenerator(
        max_duration=max_duration, batch_size=batch_size, fp16=fp16,
    )

    if phase in ("1", "both"):
        scan_phase1(music_path, cache_dir, generator, verbose=verbose)

    if phase in ("2", "both"):
        db = EmbeddingDatabase(output)
        scan_phase2(music_path, cache_dir, db, generator, verbose=verbose)
        db.close()

    generator.unload_models()
    click.echo(f"\nSaved to: {output}")


# ─── Update ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("music_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--database", type=click.Path(exists=True, path_type=Path),
              default=Path("embeddings.db"), help="Database file to update")
@click.option("--remove-missing/--no-remove-missing", default=True,
              help="Remove tracks whose files no longer exist")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--fp16", is_flag=True, help="Run MERT in FP16")
@click.option("--batch-size", type=int, default=8, help="MERT batch size")
@click.option("--max-duration", type=int, default=600, help="Max audio duration in seconds")
def update(music_path: Path, database: Path, remove_missing: bool, verbose: bool,
           fp16: bool, batch_size: int, max_duration: int):
    """Incrementally update an existing database.

    Adds new files and optionally removes missing ones.

    MUSIC_PATH: Path to your music library folder
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    click.echo(f"Updating database: {database}")
    click.echo(f"Scanning: {music_path}")

    audio_files = list(scan_music_directory(music_path))
    audio_file_paths = {str(f) for f in audio_files}
    click.echo(f"Found {len(audio_files)} audio files on disk")

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

    generator = CLaMP3EmbeddingGenerator(
        max_duration=max_duration, batch_size=batch_size, fp16=fp16,
    )

    def store_track(filepath, metadata, embedding):
        db.add_track(metadata, embedding)
        if (store_track.count % 10) == 0:
            db.commit()
        store_track.count += 1
    store_track.count = 0

    successful, failed = _process_files(
        new_files, generator, store_track, "CLaMP3"
    )
    db.commit()

    click.echo(f"\n  Added: {successful}")
    click.echo(f"  Failed: {failed}")

    generator.unload_models()

    db.set_metadata("version", __version__)
    db.set_metadata("source_path", str(music_path))
    db.set_metadata("model", "clamp3")
    db.set_metadata("embedding_dim", "768")

    db.vacuum()
    total_tracks = db.count_tracks()
    click.echo(f"\nUpdate complete!")
    click.echo(f"  Total tracks in database: {total_tracks}")
    click.echo(f"  Database size: {database.stat().st_size / 1024 / 1024:.1f} MB")

    db.close()


def _process_files(audio_files, generator, store_fn, desc):
    """Process files with CLaMP3: MERT extraction + CLaMP3 encoding per file."""
    successful = 0
    failed = 0

    with tqdm(total=len(audio_files), desc=desc, unit="file") as pbar:
        for filepath in audio_files:
            metadata = extract_metadata(filepath)
            embedding = generator.generate_embedding(filepath)

            if embedding is not None:
                store_fn(filepath, metadata, embedding)
                successful += 1
            else:
                failed += 1
                logger.warning(f"Failed: {filepath.name}")

            pbar.update(1)

    return successful, failed


# ─── Index (k-means + kNN) ───────────────────────────────────────────────────

@cli.command()
@click.argument("database", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--clusters", "-k", type=int, default=200,
              help="Number of k-means clusters (default: 200)")
@click.option("--knn", type=int, default=5,
              help="kNN graph neighbors (default: 5)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def index(database: Path, clusters: int, knn: int, verbose: bool):
    """Build k-means clusters and kNN graph for a CLaMP3 database.

    Clusters enable cluster-based navigation and kNN graph enables
    random walk exploration on Android.

    DATABASE: Path to embeddings database with CLaMP3 embeddings

    Examples:

      poweramp-indexer index embeddings.db
      poweramp-indexer index embeddings.db --clusters 200 --knn 20
    """
    from .fusion import build_index

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    click.echo(f"Database: {database}")
    db = EmbeddingDatabase(database)

    try:
        result = build_index(
            db, n_clusters=clusters, knn_k=knn,
            on_progress=lambda msg: click.echo(msg)
        )
    except ValueError as e:
        click.echo(f"Error: {e}")
        db.close()
        return

    db.set_metadata("version", __version__)
    db.close()

    click.echo(f"\nIndex complete!")
    click.echo(f"  Tracks: {result['n_tracks']}")
    click.echo(f"  Embedding dim: {result['embedding_dim']}")
    click.echo(f"  Clusters: {result['n_clusters']}")
    click.echo(f"  kNN graph: K={result['knn_k']} ({result['graph_size_mb']:.1f} MB)")
    click.echo(f"  Database size: {database.stat().st_size / 1024 / 1024:.1f} MB")


# ─── Info ─────────────────────────────────────────────────────────────────────

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

    model = db.get_metadata("model")
    if model:
        click.echo(f"Model: {model}")

    dim = db.get_metadata("embedding_dim")
    if dim:
        click.echo(f"Embedding dimension: {dim}")

    emb_count = db.count_embeddings()
    click.echo(f"Embeddings: {emb_count}")

    # Check for clusters and graph
    try:
        cluster_count = db.conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        click.echo(f"Clusters: {cluster_count}")
    except Exception:
        click.echo("Clusters: none")

    graph = db.get_binary("knn_graph")
    if graph:
        click.echo(f"kNN graph: {len(graph) / 1024 / 1024:.1f} MB")
    else:
        click.echo("kNN graph: none")

    db.close()


# ─── Similar ──────────────────────────────────────────────────────────────────

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
@click.option("--file", "-f", "audio_file", type=click.Path(exists=True, path_type=Path),
              help="Audio file to find similar tracks for")
@click.option("--random", "-r", "use_random", is_flag=True, help="Pick a random seed track")
@click.option("--top", "-n", default=10, help="Number of similar tracks to show")
def similar(database: Path, query: str, audio_file: Path, use_random: bool, top: int):
    """Find similar tracks in the database.

    DATABASE: Path to embeddings.db
    QUERY: Search string to find seed track (artist, title, etc.)

    Examples:

      poweramp-indexer similar embeddings.db "radiohead karma police"
      poweramp-indexer similar embeddings.db --file ~/Downloads/song.mp3
      poweramp-indexer similar embeddings.db --random
    """
    options_count = sum([query is not None, audio_file is not None, use_random])
    if options_count == 0:
        raise click.UsageError("Provide a QUERY, --file, or --random")
    if options_count > 1:
        raise click.UsageError("Provide only one of: QUERY, --file, or --random")

    db = EmbeddingDatabase(database)

    if db.count_embeddings() == 0:
        click.echo("Database has no embeddings.")
        db.close()
        return

    seed_track = None
    exclude_id = None
    seed_embedding = None

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
        seed_embedding = db.get_embedding_by_id(exclude_id)

    elif use_random:
        seed_track = db.get_random_track()
        if not seed_track:
            click.echo("Database is empty")
            db.close()
            return
        exclude_id = seed_track["id"]
        seed_embedding = db.get_embedding_by_id(exclude_id)

    elif audio_file:
        click.echo("Generating CLaMP3 embedding for file...")
        generator = CLaMP3EmbeddingGenerator()
        seed_embedding = generator.generate_embedding(audio_file)
        generator.unload_models()

        if seed_embedding is None:
            click.echo(f"Failed to generate embedding for: {audio_file}")
            db.close()
            return

    if seed_embedding is None:
        click.echo("Failed to get seed embedding.")
        db.close()
        return

    if seed_track:
        click.echo(f"Seed: {format_track(seed_track)}")
        click.echo()

    click.echo("Loading embeddings...")
    all_embeddings = db.get_all_embeddings()
    click.echo(f"Searching {len(all_embeddings)} tracks...")
    click.echo()

    similar_tracks = find_similar(all_embeddings, seed_embedding, top, exclude_id)

    click.echo("Similar tracks:")
    for rank, (track_id, score) in enumerate(similar_tracks, 1):
        track = db.get_track_by_id(track_id)
        if track:
            click.echo(f"  {rank:2}. [{score:.3f}] {format_track(track)}")

    db.close()


# ─── Search ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("database", type=click.Path(exists=True, path_type=Path))
@click.argument("query")
@click.option("--top", "-n", default=10, help="Number of results to show")
def search(database: Path, query: str, top: int):
    """Search for tracks using text queries (CLaMP3 text-to-audio).

    DATABASE: Path to embeddings database
    QUERY: Text describing the music you want (e.g., "space rock", "melancholic piano")

    Examples:

      poweramp-indexer search embeddings.db "space rock"
      poweramp-indexer search embeddings.db "upbeat electronic dance"
      poweramp-indexer search embeddings.db "sad piano ballad"
    """
    db = EmbeddingDatabase(database)

    if db.count_embeddings() == 0:
        click.echo("Database has no embeddings.")
        db.close()
        return

    click.echo(f"Searching for: {query}")
    generator = CLaMP3EmbeddingGenerator()
    text_embedding = generator.embed_text(query)

    if text_embedding is None:
        click.echo("Failed to generate text embedding.")
        generator.unload_models()
        db.close()
        return

    click.echo("Loading embeddings...")
    all_embeddings = db.get_all_embeddings()
    click.echo(f"Searching {len(all_embeddings)} tracks...")
    click.echo()

    similar_tracks = find_similar(all_embeddings, text_embedding, top)

    click.echo("Results:")
    for rank, (track_id, score) in enumerate(similar_tracks, 1):
        track = db.get_track_by_id(track_id)
        if track:
            click.echo(f"  {rank:2}. [{score:.3f}] {format_track(track)}")

    generator.unload_models()
    db.close()


# ─── Export ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("model", type=click.Choice(["mert", "clamp3_audio", "clamp3_text", "tokenizer", "all"]))
@click.option("--output-dir", type=click.Path(path_type=Path),
              default=Path(__file__).parent.parent.parent / "models",
              help="Output directory for .tflite models")
def export(model: str, output_dir: Path):
    """Export CLaMP3 models to TFLite for on-device inference.

    MODEL: Which model to export (mert, clamp3_audio, clamp3_text, tokenizer, all)

    Examples:

      poweramp-indexer export all
      poweramp-indexer export mert --output-dir ./models
      poweramp-indexer export tokenizer
    """
    from .export_litert import (
        convert_mert, convert_clamp3_audio, convert_clamp3_text,
        export_tokenizer,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if model in ("mert", "all"):
        convert_mert(output_dir)

    if model in ("clamp3_audio", "all"):
        convert_clamp3_audio(output_dir)

    if model in ("clamp3_text", "all"):
        convert_clamp3_text(output_dir)
        export_tokenizer(output_dir)

    if model == "tokenizer":
        export_tokenizer(output_dir)


if __name__ == "__main__":
    cli()

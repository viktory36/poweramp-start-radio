#!/usr/bin/env python3
"""Build and search a local metadata catalog from Poweramp's ADB provider output."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import readline
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB = (
    SCRIPT_DIR.parent
    / "audit_raw_data"
    / "poweramp-library"
    / "poweramp-library.sqlite3"
)
ROW_START = re.compile(r"(?m)^Row: (?P<position>\d+) _id=(?P<id>-?\d+), ")


@dataclass(frozen=True)
class Track:
    poweramp_id: int
    artist: str | None
    album_artist_id: int | None
    album_artist: str | None
    album: str | None
    title: str | None
    duration_ms: int | None
    folder_path: str | None
    filename: str | None
    year: int | None
    added_at: int | None
    disc: int | None
    track_number: int | None
    file_type: int | None


def null_to_none(value: str) -> str | None:
    value = value.strip()
    return None if value == "NULL" else value


def optional_int(value: str) -> int | None:
    value = value.strip()
    return None if value == "NULL" else int(value)


def pop_field(body: str, name: str) -> tuple[str, str]:
    head, separator, value = body.rpartition(f", {name}=")
    if not separator:
        raise ValueError(f"Poweramp row is missing {name!r}")
    return head, value


def split_records(text: str) -> list[tuple[int, int, str]]:
    matches = list(ROW_START.finditer(text))
    if not matches:
        raise ValueError("No Poweramp rows found")

    records: list[tuple[int, int, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        records.append(
            (
                int(match.group("position")),
                int(match.group("id")),
                text[match.end() : end].rstrip("\r\n"),
            )
        )

    positions = [position for position, _, _ in records]
    if positions != list(range(len(records))):
        raise ValueError("Poweramp row positions are incomplete or ambiguous")
    return records


def parse_album_artists(path: Path) -> dict[int, str]:
    artists: dict[int, str] = {}
    for _, poweramp_id, body in split_records(path.read_text(encoding="utf-8")):
        prefix = "album_artist="
        if not body.startswith(prefix):
            raise ValueError(f"Unexpected album-artist row {poweramp_id}")
        name = null_to_none(body[len(prefix) :])
        if name is not None:
            artists[poweramp_id] = name
    return artists


def parse_tracks(path: Path, album_artists: dict[int, str]) -> list[Track]:
    tracks: list[Track] = []
    seen_ids: set[int] = set()
    for _, poweramp_id, body in split_records(path.read_text(encoding="utf-8")):
        if poweramp_id in seen_ids:
            raise ValueError(f"Duplicate Poweramp track ID {poweramp_id}")
        seen_ids.add(poweramp_id)

        body, album_artist_id_text = pop_field(body, "album_artist_id")
        body, file_type_text = pop_field(body, "file_type")
        body, track_number_text = pop_field(body, "track_tag")
        body, disc_text = pop_field(body, "disc")
        body, added_at_text = pop_field(body, "created_at")
        body, year_text = pop_field(body, "year")
        body, filename_text = pop_field(body, "name")
        body, folder_path_text = pop_field(body, "path")
        body, duration_text = pop_field(body, "duration")
        body, title_text = pop_field(body, "title_tag")

        artist_prefix = "artist="
        if not body.startswith(artist_prefix):
            raise ValueError(f"Unexpected track row {poweramp_id}")
        artist_and_album = body[len(artist_prefix) :]
        artist_text, separator, album_text = artist_and_album.partition(", album=")
        if not separator:
            raise ValueError(f"Poweramp track {poweramp_id} is missing album")

        album_artist_id = optional_int(album_artist_id_text)
        tracks.append(
            Track(
                poweramp_id=poweramp_id,
                artist=null_to_none(artist_text),
                album_artist_id=album_artist_id,
                album_artist=album_artists.get(album_artist_id),
                album=null_to_none(album_text),
                title=null_to_none(title_text),
                duration_ms=optional_int(duration_text),
                folder_path=null_to_none(folder_path_text),
                filename=null_to_none(filename_text),
                year=optional_int(year_text),
                added_at=optional_int(added_at_text),
                disc=optional_int(disc_text),
                track_number=optional_int(track_number_text),
                file_type=optional_int(file_type_text),
            )
        )
    return tracks


def search_key(*values: object | None) -> str:
    text = " ".join(str(value) for value in values if value not in (None, ""))
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def display_artist(track: Track) -> str:
    return track.album_artist or track.artist or "Unknown artist"


def record_key(track: Track) -> str:
    if track.album:
        return "\x1f".join(
            (
                search_key(display_artist(track)),
                search_key(track.album),
                str(track.year or ""),
            )
        )
    return "\x1f".join(
        (
            search_key(display_artist(track)),
            search_key(track.title or track.filename),
            str(track.year or ""),
        )
    )


def file_format(filename: str | None) -> str | None:
    if not filename or "." not in filename:
        return None
    suffix = filename.rsplit(".", 1)[-1].strip().upper()
    return suffix or None


def build_database(
    tracks_path: Path,
    album_artists_path: Path,
    output_path: Path,
    device: str | None,
) -> tuple[int, int]:
    album_artists = parse_album_artists(album_artists_path)
    tracks = parse_tracks(tracks_path, album_artists)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)

    records: dict[str, list[Track]] = defaultdict(list)
    for track in tracks:
        records[record_key(track)].append(track)

    connection = sqlite3.connect(temporary_path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            CREATE TABLE snapshot (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE tracks (
                poweramp_id INTEGER PRIMARY KEY,
                artist TEXT,
                album_artist TEXT,
                album TEXT,
                title TEXT,
                year INTEGER,
                disc INTEGER,
                track_number INTEGER,
                duration_ms INTEGER,
                added_at INTEGER,
                file_type INTEGER,
                folder_path TEXT,
                filename TEXT,
                record_key TEXT NOT NULL,
                search_text TEXT NOT NULL
            );
            CREATE TABLE records (
                record_key TEXT PRIMARY KEY,
                artist TEXT NOT NULL,
                album TEXT NOT NULL,
                year INTEGER,
                file_count INTEGER NOT NULL,
                folder_count INTEGER NOT NULL,
                formats TEXT NOT NULL,
                paths TEXT NOT NULL,
                search_text TEXT NOT NULL
            );
            CREATE INDEX tracks_record_key ON tracks(record_key);
            CREATE INDEX tracks_added_at ON tracks(added_at);
            CREATE INDEX records_artist_album ON records(artist, album);
            CREATE VIRTUAL TABLE records_fts USING fts5(
                search_text,
                content = records,
                content_rowid = rowid,
                tokenize = 'unicode61 remove_diacritics 2'
            );
            """
        )
        captured_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        connection.executemany(
            "INSERT INTO snapshot(key, value) VALUES (?, ?)",
            (
                ("captured_at_utc", captured_at),
                ("device", device or "unknown"),
                ("track_count", str(len(tracks))),
                ("album_artist_count", str(len(album_artists))),
                ("source_tracks", str(tracks_path.resolve())),
                ("source_album_artists", str(album_artists_path.resolve())),
            ),
        )
        connection.executemany(
            """
            INSERT INTO tracks(
                poweramp_id, artist, album_artist, album, title, year, disc,
                track_number, duration_ms, added_at, file_type, folder_path,
                filename, record_key, search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    track.poweramp_id,
                    track.artist,
                    track.album_artist,
                    track.album,
                    track.title,
                    track.year,
                    track.disc,
                    track.track_number,
                    track.duration_ms,
                    track.added_at,
                    track.file_type,
                    track.folder_path,
                    track.filename,
                    record_key(track),
                    search_key(
                        track.artist,
                        track.album_artist,
                        track.album,
                        track.title,
                        track.year,
                        track.filename,
                        track.folder_path,
                    ),
                )
                for track in tracks
            ),
        )

        record_rows = []
        for key, members in records.items():
            representative = members[0]
            artist = display_artist(representative)
            album = representative.album or representative.title or representative.filename or "Unknown"
            folders = sorted(
                {track.folder_path for track in members if track.folder_path},
                key=search_key,
            )
            formats = sorted(
                {value for track in members if (value := file_format(track.filename))}
            )
            record_rows.append(
                (
                    key,
                    artist,
                    album,
                    representative.year,
                    len(members),
                    len(folders),
                    ", ".join(formats),
                    "\n".join(folders),
                    search_key(
                        artist,
                        album,
                        representative.year,
                        *folders,
                    ),
                )
            )
        connection.executemany(
            """
            INSERT INTO records(
                record_key, artist, album, year, file_count, folder_count,
                formats, paths, search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            record_rows,
        )
        connection.execute(
            "INSERT INTO records_fts(rowid, search_text) "
            "SELECT rowid, search_text FROM records"
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    finally:
        connection.close()

    os.replace(temporary_path, output_path)
    return len(tracks), len(records)


def query_terms(query: str) -> list[str]:
    terms = search_key(query).split()
    if not terms:
        raise ValueError("Search query has no letters or numbers")
    return terms


def search_connection(
    connection: sqlite3.Connection,
    query: str,
    limit: int,
) -> list[sqlite3.Row]:
    terms = query_terms(query)
    rows = connection.execute(
        """
        SELECT r.artist, r.album, r.year, r.file_count, r.folder_count,
               r.formats, r.paths
        FROM records_fts
        JOIN records AS r ON r.rowid = records_fts.rowid
        WHERE records_fts MATCH ?
        ORDER BY rank
        LIMIT 1000
        """,
        (" AND ".join(f'"{term}"*' for term in terms),),
    ).fetchall()
    if not rows:
        where = " AND ".join("instr(search_text, ?) > 0" for _ in terms)
        rows = connection.execute(
            f"""
            SELECT artist, album, year, file_count, folder_count, formats, paths
            FROM records
            WHERE {where}
            LIMIT 1000
            """,
            terms,
        ).fetchall()
    normalized_query = search_key(query)

    def score(row: sqlite3.Row) -> tuple[int, str, str, int]:
        artist = search_key(row["artist"])
        album = search_key(row["album"])
        heading = f"{artist} {album}"
        if normalized_query == album or normalized_query == heading:
            tier = 0
        elif normalized_query in album:
            tier = 1
        elif all(term in heading for term in terms):
            tier = 2
        else:
            tier = 3
        return tier, artist, album, row["year"] or 0

    return sorted(rows, key=score)[:limit]


def open_database(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def search_database(database: Path, query: str, limit: int) -> list[sqlite3.Row]:
    with open_database(database) as connection:
        return search_connection(connection, query, limit)


def print_search_results(rows: list[sqlite3.Row]) -> None:
    if not rows:
        print("No matching records.")
        return
    for row in rows:
        year = f" ({row['year']})" if row["year"] else ""
        formats = f" · {row['formats']}" if row["formats"] else ""
        print(f"{row['artist']} - {row['album']}{year}")
        print(
            f"  {row['file_count']} files · {row['folder_count']} folders{formats}"
        )
        for path in row["paths"].splitlines()[:3]:
            print(f"  {path}")


def interactive_search(database: Path, limit: int) -> None:
    readline.set_auto_history(True)
    with open_database(database) as connection:
        track_count = int(
            connection.execute(
                "SELECT value FROM snapshot WHERE key = 'track_count'"
            ).fetchone()[0]
        )
        print(f"Poweramp library: {track_count:,} tracks")
        print("Type an album name, optionally with artist or folder words. Use q to quit.")
        while True:
            try:
                query = input("\nrecord> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if query.casefold() in {"q", "quit", "exit"}:
                return
            if not query:
                continue
            print_search_results(search_connection(connection, query, limit))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build SQLite from captured provider output")
    build.add_argument("tracks", type=Path)
    build.add_argument("album_artists", type=Path)
    build.add_argument("--output", type=Path, default=DEFAULT_DB)
    build.add_argument("--device")

    search = subparsers.add_parser("search", help="Search records in the local catalog")
    search.add_argument("query")
    search.add_argument("--database", type=Path, default=DEFAULT_DB)
    search.add_argument("--limit", type=int, default=8)

    interactive = subparsers.add_parser(
        "interactive",
        help="Keep a prompt open for repeated record lookups",
    )
    interactive.add_argument("--database", type=Path, default=DEFAULT_DB)
    interactive.add_argument("--limit", type=int, default=8)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "build":
        track_count, record_count = build_database(
            args.tracks,
            args.album_artists,
            args.output,
            args.device,
        )
        print(f"Built {args.output} with {track_count:,} tracks and {record_count:,} records.")
    elif args.command == "search":
        print_search_results(search_database(args.database, args.query, args.limit))
    else:
        interactive_search(args.database, args.limit)


if __name__ == "__main__":
    main()

"""
Genius Service Module
---------------------
This module encapsulates all interactions with the Genius API.
It is responsible for searching for songs and returning clean,
annotation-free lyrics text.
"""

import os
from lyricsgenius import Genius


def fetch_lyrics(song_title: str, artist_name: str) -> str | None:
    """
    Fetches clean lyrics for a given song using the lyricsgenius library.

    Args:
        song_title: The title of the song to search for.
        artist_name: The name of the artist or band.

    Returns:
        The song lyrics as a string, or None if the song cannot be found.

    Raises:
        ValueError: If the GENIUS_API_TOKEN environment variable is missing.
        ConnectionError: If a network or API error occurs during the request.
    """
    # ------------------------------------------------------------------
    # 1. Authentication
    # ------------------------------------------------------------------
    # Retrieve the API token from the environment. Using os.getenv allows
    # python-dotenv (loaded in main.py) to inject values from the .env file.
    api_token = os.getenv("GENIUS_API_TOKEN")
    if not api_token:
        raise ValueError(
            "GENIUS_API_TOKEN environment variable is not set. "
            "Please add it to your .env file."
        )

    # ------------------------------------------------------------------
    # 2. Initialize Genius Client
    # ------------------------------------------------------------------
    # verbose=False suppresses status printouts to keep our CLI clean.
    # remove_section_headers=True strips [Chorus], [Verse 1], etc.,
    # giving the AI cleaner text to analyze.
    genius = Genius(
        api_token,
        remove_section_headers=True,
        skip_non_songs=True,      # Avoids returning tracklists or non-music pages
        excluded_terms=["(Remix)", "(Live)"]  # Optional: filters out common variants
    )

    # ------------------------------------------------------------------
    # 3. Search & Fetch
    # ------------------------------------------------------------------
    try:
        # search_song returns a Song object or None if no match is found.
        # Passing the artist name as the second argument narrows results.
        song = genius.search_song(song_title, artist_name)

        if song is None:
            return None

        lyrics = song.lyrics

        # ------------------------------------------------------------------
        # 4. Cleanup
        # ------------------------------------------------------------------
        # Genius sometimes appends "Embed" and contributor counts at the end.
        # We split on "Embed" and keep only the text before it.
        if "Embed" in lyrics:
            lyrics = lyrics.split("Embed")[0].strip()

        return lyrics

    except Exception as e:
        # Wrap unexpected errors in a descriptive exception so main.py
        # can handle them gracefully and print user-friendly messages.
        raise ConnectionError(f"Failed to communicate with Genius API: {e}")

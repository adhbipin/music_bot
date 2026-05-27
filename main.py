#!/usr/bin/env python3
"""
Music Lyrics & Theme Analyzer Bot
---------------------------------
Entry point for the CLI application.

Flow:
  1. Load environment variables from .env
  2. Display welcome banner
  3. Loop: ask user for song title + artist
  4. Fetch lyrics via Genius API
  5. Analyze lyrics via Google Gemini API
  6. Print beautiful markdown-formatted results
"""

import os
import sys
from dotenv import load_dotenv

# Import our custom service layers
from services.genius_service import fetch_lyrics
from services.ai_service import analyze_lyrics_themes


# ------------------------------------------------------------------
# UI Helper Functions
# ------------------------------------------------------------------
def print_header():
    """Prints the application welcome banner."""
    print("\n" + "=" * 60)
    print("🎵  Music Lyrics & Theme Analyzer Bot  🎵")
    print("   Powered by Genius + NVIDIA NIM")
    print("=" * 60)


def print_divider():
    """Prints a visual divider for clean markdown-style formatting."""
    print("\n" + "-" * 60 + "\n")


def get_user_input() -> tuple[str, str] | tuple[None, None]:
    """
    Prompts the user for song details.

    Returns:
        A tuple of (song_title, artist_name), or (None, None) if quitting.
    """
    print("\nEnter song details (or type 'quit' to exit):")
    title = input("  🎤 Song Title : ").strip()
    if title.lower() in ("quit", "exit", "q"):
        return None, None

    artist = input("  🎸 Artist Name: ").strip()
    if artist.lower() in ("quit", "exit", "q"):
        return None, None

    return title, artist


def validate_inputs(title: str, artist: str) -> bool:
    """
    Validates that the user provided non-empty strings.

    Args:
        title: Song title from user input.
        artist: Artist name from user input.

    Returns:
        True if both inputs are valid, False otherwise.
    """
    if not title:
        print("\n❌ Error: Song title cannot be empty.")
        return False
    if not artist:
        print("\n❌ Error: Artist name cannot be empty.")
        return False
    return True


def display_analysis(analysis_text: str):
    """
    Renders the final AI analysis with beautiful CLI formatting.

    Args:
        analysis_text: The raw markdown string returned by Gemini.
    """
    print_divider()
    print("📊  THEMATIC ANALYSIS")
    print_divider()
    print(analysis_text)
    print_divider()


# ------------------------------------------------------------------
# Main Application Loop
# ------------------------------------------------------------------
def main():
    """
    Orchestrates the entire application lifecycle:
    setup -> input loop -> service calls -> display -> repeat/exit.
    """
    # Step 1: Load .env file variables into the running process environment.
    # This MUST happen before any os.getenv() calls in our services.
    load_dotenv()

    print_header()

    # Step 2: Pre-flight configuration check.
    # We validate API keys upfront to avoid confusing mid-flow errors.
    genius_token = os.getenv("GENIUS_API_TOKEN")
    nim_key = os.getenv("NIM_API_KEY")

    missing = []
    if not genius_token:
        missing.append("GENIUS_API_TOKEN")
    if not nim_key:
        missing.append("NIM_API_KEY")

    if missing:
        print(f"\n❌ Cannot start: Missing required config key(s): {', '.join(missing)}")
        print("   Please check your .env file.\n")
        print("   🔑 Genius API token : https://genius.com/api-clients")
        print("   🔑 NVIDIA NIM API key: https://build.nvidia.com")
        sys.exit(1)

    print("\n✅ Configuration OK — ready to analyze!")

    # Step 3: Primary CLI loop
    while True:
        try:
            # --- Get input ---
            song_title, artist_name = get_user_input()

            # Exit condition
            if song_title is None:
                print("\n👋 Thanks for using the analyzer. Keep the music alive!")
                break

            # Validate before wasting API calls
            if not validate_inputs(song_title, artist_name):
                continue

            # --- Fetch Lyrics ---
            print(f"\n🔍 Searching Genius for '{song_title}' by {artist_name}...")
            try:
                lyrics = fetch_lyrics(song_title, artist_name)
            except ValueError as ve:
                print(f"\n❌ Configuration Error: {ve}")
                continue
            except ConnectionError as ce:
                print(f"\n❌ Network Error: {ce}")
                continue

            if lyrics is None:
                print(f"\n❌ Could not find lyrics for '{song_title}' by {artist_name}.")
                print("   💡 Tip: Check the spelling or try the exact official song title.")
                continue

            print(f"\n✅ Lyrics found! ({len(lyrics)} characters)")
            print("🤖 Sending to NVIDIA NIM for thematic analysis...")

            # --- Analyze Lyrics ---
            try:
                analysis = analyze_lyrics_themes(lyrics)
            except ValueError as ve:
                print(f"\n❌ Configuration Error: {ve}")
                continue
            except RuntimeError as re:
                print(f"\n❌ AI Analysis Error: {re}")
                continue

            # --- Display Results ---
            display_analysis(analysis)

        except KeyboardInterrupt:
            # Graceful handling of Ctrl+C (SIGINT)
            print("\n\n👋 Interrupted by user. Shutting down gracefully.")
            break
        except Exception as e:
            # Catch-all safety net for truly unexpected errors
            print(f"\n💥 Unexpected error: {e}")
            continue


# ------------------------------------------------------------------
# Entry Point Guard
# ------------------------------------------------------------------
# This ensures main() only runs when the script is executed directly,
# not when it is imported as a module by another script.
if __name__ == "__main__":
    main()

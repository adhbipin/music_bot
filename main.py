#!/usr/bin/env python3
"""
Music Lyrics & Theme Analyzer Bot — CLI v2.0
----------------------------------------------
Advanced CLI with:
  • Rich terminal formatting (if rich installed)
  • Async service integration
  • Progress indicators
  • Config validation
  • Graceful error handling
  • Keyboard interrupt support

Usage:
  python main.py
  python main.py --title "Bohemian Rhapsody" --artist "Queen"
"""

import os
import sys
import argparse
import asyncio
from dotenv import load_dotenv

# Load env vars FIRST
load_dotenv()

from services.genius_service import get_lyrics_service
from services.ai_service import NIMServiceSync, get_service


# ── Rich UI (optional) ────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None


def print_banner():
    """Display welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║     🎵  Music Lyrics & Theme Analyzer Bot  v2.0  🎵     ║
║         Powered by Genius API + NVIDIA NIM               ║
╚══════════════════════════════════════════════════════════╝
"""
    if HAS_RICH:
        console.print(Panel(
            "[bold cyan]🎵 Music Lyrics & Theme Analyzer Bot v2.0[/bold cyan]\n"
            "[dim]Powered by Genius API + NVIDIA NIM[/dim]",
            box=box.DOUBLE,
            border_style="cyan",
        ))
    else:
        print(banner)


def print_error(msg: str, suggestion: str = None):
    """Print styled error message."""
    if HAS_RICH:
        console.print(f"[bold red]❌ {msg}[/bold red]")
        if suggestion:
            console.print(f"[dim]💡 {suggestion}[/dim]")
    else:
        print(f"\n❌ {msg}")
        if suggestion:
            print(f"   💡 {suggestion}")


def print_success(msg: str):
    """Print styled success message."""
    if HAS_RICH:
        console.print(f"[bold green]✅ {msg}[/bold green]")
    else:
        print(f"✅ {msg}")


def print_analysis(analysis_md: str):
    """Render analysis with beautiful formatting."""
    if HAS_RICH:
        console.print(Panel(
            Markdown(analysis_md),
            title="[bold magenta]📊 Thematic Analysis[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
        ))
    else:
        print("\n" + "=" * 60)
        print("📊  THEMATIC ANALYSIS")
        print("=" * 60)
        print(analysis_md)
        print("=" * 60)


# ── Validation ──────────────────────────────────────────────────────
def validate_env() -> bool:
    """Check required API keys."""
    missing = []
    if not os.getenv("GENIUS_API_TOKEN"):
        missing.append("GENIUS_API_TOKEN")
    if not os.getenv("NIM_API_KEY"):
        missing.append("NIM_API_KEY")

    if missing:
        print_error(
            f"Missing required config: {', '.join(missing)}",
            "Add them to your .env file or export them."
        )
        print("   🔑 Genius: https://genius.com/api-clients")
        print("   🔑 NIM:   https://build.nvidia.com")
        return False
    return True


# ── Core Logic ──────────────────────────────────────────────────────
async def analyze(title: str, artist: str) -> bool:
    """Fetch and analyze a song."""
    genius = get_lyrics_service()
    ai = get_service(sync=False)

    # Fetch lyrics
    if HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"🔍 Searching Genius for '{title}'...", total=None)
            lyrics = await genius.fetch_lyrics(title, artist)
            progress.update(task, completed=True)
    else:
        print(f"\n🔍 Searching Genius for '{title}' by {artist}...")
        lyrics = await genius.fetch_lyrics(title, artist)

    if not lyrics:
        print_error(
            f"Could not find lyrics for '{title}' by {artist}.",
            "Check spelling or try the exact official title."
        )
        return False

    print_success(f"Lyrics found! ({len(lyrics)} characters)")

    # Analyze
    if HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🤖 Analyzing with NVIDIA NIM...", total=None)
            result = await ai.analyze_lyrics(lyrics)
            progress.update(task, completed=True)
    else:
        print("🤖 Sending to NVIDIA NIM for thematic analysis...")
        result = await ai.analyze_lyrics(lyrics)

    if not result.success:
        print_error(result.error or "Analysis failed")
        return False

    # Convert to markdown for display
    data = result.data
    lines = [
        f"## 🎵 {title} — {artist}",
        "",
        f"**Overall Mood:** {data.overall_mood}",
        f"**Mood Tags:** {', '.join(data.mood_tags)}",
        "",
        "### Core Themes",
    ]
    for t in data.themes:
        lines.append(f"- **{t.name}** ({t.confidence:.0%} confidence): {t.explanation}")
        lines.append(f"  > *\"{t.lyric_evidence}\"*")

    lines.extend([
        "",
        "### Summary",
        data.summary,
        "",
        f"**Language:** {data.language}",
    ])
    if data.genre_inference:
        lines.append(f"**Genre:** {data.genre_inference}")
    if data.narrative_perspective:
        lines.append(f"**Perspective:** {data.narrative_perspective}")
    if data.key_symbols:
        lines.append(f"**Key Symbols:** {', '.join(data.key_symbols)}")
    if data.estimated_era:
        lines.append(f"**Era:** {data.estimated_era}")

    # Build cost display safely
    cost_display = f"${result.cost_usd:.4f}" if result.cost_usd else "$0.0000"
    lines.extend([
        "",
        "---",
        f"*Model: {result.model_used} | Tokens: {result.tokens_used} | Cost: {cost_display} | Latency: {result.latency_ms:.0f}ms*",
    ])

    print_analysis("\n".join(lines))
    return True


# ── Interactive Mode ────────────────────────────────────────────────
def interactive_mode():
    """Run interactive CLI loop."""
    print_banner()

    if not validate_env():
        sys.exit(1)

    print_success("Configuration OK — ready to analyze!\n")

    while True:
        try:
            print("\nEnter song details (or 'quit' to exit):")
            title = input("  🎤 Song Title : ").strip()
            if title.lower() in ("quit", "exit", "q"):
                break
            if not title:
                print_error("Song title cannot be empty.")
                continue

            artist = input("  🎸 Artist Name: ").strip()
            if artist.lower() in ("quit", "exit", "q"):
                break
            if not artist:
                print_error("Artist name cannot be empty.")
                continue

            asyncio.run(analyze(title, artist))

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Keep the music alive!")
            break
        except Exception as e:
            print_error(f"Unexpected error: {e}", "Please try again.")


# ── Main Entry ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Music Lyrics & Theme Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--title", "-t", help="Song title")
    parser.add_argument("--artist", "-a", help="Artist name")
    parser.add_argument("--version", "-v", action="store_true", help="Show version")

    args = parser.parse_args()

    if args.version:
        print("Music Theme Analyzer v2.0.0")
        sys.exit(0)

    if args.title and args.artist:
        # One-shot mode
        print_banner()
        if not validate_env():
            sys.exit(1)
        success = asyncio.run(analyze(args.title, args.artist))
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
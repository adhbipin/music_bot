# Music Lyrics & Theme Analyzer Bot — Developer Context

A CLI tool that fetches song lyrics from **Genius** and runs them through **Google Gemini** for structured thematic analysis. Built in Python with a clean service-layer architecture.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure your API keys
Edit `.env` and fill in your keys:

| Variable | Where to get it |
|---|---|
| `GENIUS_API_TOKEN` | https://genius.com/api-clients → "Client Access Token" |
| `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey (free) |
| `GEMINI_MODEL` | Optional. Default: `gemini-1.5-flash` |

### 3. Run the bot
```bash
python main.py
```

---

## 📁 Project Structure

```
music-theme-bot/
├── main.py                    # CLI entry point & main loop
├── requirements.txt           # Python dependencies
├── .env                       # API keys (never commit this!)
├── GEMINI.md                  # This file — developer context
└── services/
    ├── __init__.py            # Makes services/ a Python package
    ├── genius_service.py      # Lyrics fetching via lyricsgenius
    └── ai_service.py          # Gemini API theme analysis
```

---

## 🔧 Architecture

The app uses a clean **3-layer service architecture**:

```
main.py  →  genius_service.py  →  Genius API  (fetch lyrics)
         →  ai_service.py      →  Gemini API  (analyze themes)
```

### `services/genius_service.py`
- Wraps `lyricsgenius.Genius`
- `fetch_lyrics(song_title, artist_name) → str | None`
- Strips `[Chorus]`, `[Verse]` annotations and trailing `Embed` noise
- Returns `None` if the song isn't found (caller handles gracefully)

### `services/ai_service.py`
- Uses `google-generativeai` SDK
- `analyze_lyrics_themes(lyrics) → str`
- Sends a structured prompt requesting exactly 3 sections:
  1. Overall Mood / Vibe
  2. Top 3 Core Themes
  3. Summary
- Returns clean markdown — ready to `print()` directly

### `main.py`
- Loads `.env` via `python-dotenv` before any service calls
- Pre-flight check validates both API keys before entering the loop
- Handles `ValueError` / `RuntimeError` / `ConnectionError` per service
- Graceful `Ctrl+C` (SIGINT) handling

---

## 🔑 Environment Variables

```env
GENIUS_API_TOKEN=<your token>      # Required
GEMINI_API_KEY=<your key>          # Required
GEMINI_MODEL=gemini-1.5-flash      # Optional (default: gemini-1.5-flash)
```

**Never commit `.env` to git.** It is (or should be) in `.gitignore`.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `lyricsgenius` | `>=3.0.0` | Genius API client for lyrics |
| `google-generativeai` | `>=0.7.0` | Official Gemini SDK |
| `python-dotenv` | `>=1.0.0` | Loads `.env` into `os.environ` |

---

## 🤖 Gemini Model Options

| Model | Speed | Quality | Free Tier |
|---|---|---|---|
| `gemini-1.5-flash` | ⚡ Fast | Good | ✅ Yes |
| `gemini-2.0-flash` | ⚡ Fast | Better | ✅ Yes |
| `gemini-1.5-pro` | 🐢 Slower | Best | ✅ Limited |

Set via `GEMINI_MODEL` in `.env`.

---

## 💡 Known Gotchas

- **"Could not find lyrics"** — Genius search is fuzzy. Try the exact official title (e.g. `"Blinding Lights"` not `"blinding lights by weekend"`).
- **Empty Gemini response** — Very short or instrumental tracks may trigger the safety filter. Try a different song.
- **`lyricsgenius` rate limits** — If you hit them, add `time.sleep(1)` between requests in the loop.
- **`GENIUS_API_TOKEN` wrong type** — Use the **Client Access Token**, not the OAuth token.

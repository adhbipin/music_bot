"""
AI Service Module
-----------------
This module handles interactions with the NVIDIA NIM inference API.
NIM is OpenAI-compatible, so we use the `openai` SDK pointed at the
NVIDIA hosted endpoint.

Required environment variables:
- `NIM_API_KEY` : Your NVIDIA NIM API key
                   Get one free at: https://build.nvidia.com

Optional environment variables:
- `NIM_MODEL`   : The model to use (default: meta/llama-3.1-8b-instruct)
                   Browse free models at: https://build.nvidia.com/explore/discover
"""

import os
from openai import OpenAI


# Default free-tier model available on NVIDIA's hosted API catalog
_DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"

# NIM is OpenAI-compatible — only the base_url changes
_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Prompt template — structured to produce consistent, parseable markdown output
_ANALYSIS_PROMPT = """\
You are an expert musicologist and literary analyst. Analyze the following song lyrics \
and provide a structured analysis with exactly these sections:

## 1. Overall Mood / Vibe
A single sentence describing the emotional atmosphere of the song.

## 2. Top 3 Core Themes
List exactly three themes. For each one use this format:
- **Theme Name**: One sentence explaining how this theme appears in the lyrics.

## 3. Summary
Exactly two sentences summarizing the song's overall meaning and message.

## 4. Song Language
The language in which the song is written.

Format your response cleanly using the markdown headers and bullet style shown above.
Do NOT add any extra sections or preamble.

LYRICS:
---
{lyrics}
---
"""


def analyze_lyrics_themes(lyrics: str) -> str:
    """
    Sends lyrics to NVIDIA NIM and returns a structured thematic analysis.

    Args:
        lyrics: The cleaned song lyrics text to analyze.

    Returns:
        A markdown-formatted string containing the thematic analysis.

    Raises:
        ValueError:   If NIM_API_KEY environment variable is missing.
        RuntimeError: If the NIM API call fails or returns empty content.
    """
    # ------------------------------------------------------------------
    # 1. Validate configuration
    # ------------------------------------------------------------------
    api_key = os.getenv("NIM_API_KEY")
    model = os.getenv("NIM_MODEL", _DEFAULT_MODEL)

    if not api_key:
        raise ValueError(
            "NIM_API_KEY environment variable must be set. "
            "Get your free key at: https://build.nvidia.com "
            "and add it to your .env file."
        )

    # ------------------------------------------------------------------
    # 2. Build client and call the model
    # ------------------------------------------------------------------
    client = OpenAI(
        base_url=_NIM_BASE_URL,
        api_key=api_key,
    )

    prompt = _ANALYSIS_PROMPT.format(lyrics=lyrics)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
    except Exception as e:
        raise RuntimeError(f"NIM API call failed: {e}")

    # ------------------------------------------------------------------
    # 3. Extract and validate the response text
    # ------------------------------------------------------------------
    try:
        text = response.choices[0].message.content or ""
    except (IndexError, AttributeError):
        text = ""

    if not text.strip():
        raise RuntimeError(
            "NIM returned an empty response. "
            "The lyrics may have been blocked by the safety filter."
        )

    return text.strip()

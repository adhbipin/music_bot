"""
AI Service Module — Advanced Production-Grade
---------------------------------------------
Handles interactions with NVIDIA NIM inference API.
Uses simple JSON prompting instead of guided_json (which causes issues on free tier).

Required env vars:
  NIM_API_KEY  — Your NVIDIA NIM API key (free at https://build.nvidia.com)
"""

import os
import json
import time
import logging
import hashlib
import asyncio
import re
from typing import Optional, AsyncGenerator, Dict, Any, List
from enum import Enum

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError


# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("nim_service")


# ── Configuration ────────────────────────────────────────────────────
class Config:
    """Centralized configuration with sensible defaults."""

    NIM_API_KEY: str = os.getenv("NIM_API_KEY", "")
    NIM_BASE_URL: str = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

    @staticmethod
    def model_chain() -> List[str]:
        custom = os.getenv("NIM_MODEL", "")
        if custom:
            return [custom]
        return ["meta/llama-3.1-8b-instruct"]

    MAX_RETRIES: int = int(os.getenv("NIM_MAX_RETRIES", 2))
    RETRY_DELAY_BASE: float = float(os.getenv("NIM_RETRY_DELAY", 1.0))
    CIRCUIT_BREAKER_THRESHOLD: int = int(os.getenv("NIM_CB_THRESHOLD", 5))
    CIRCUIT_BREAKER_TIMEOUT: int = int(os.getenv("NIM_CB_TIMEOUT", 60))
    CACHE_TTL_SECONDS: int = int(os.getenv("NIM_CACHE_TTL", 3600))
    CACHE_MAX_SIZE: int = int(os.getenv("NIM_CACHE_SIZE", 100))
    RATE_LIMIT_RPM: int = int(os.getenv("NIM_RATE_LIMIT_RPM", 60))
    TEMPERATURE: float = float(os.getenv("NIM_TEMPERATURE", 0.3))
    MAX_TOKENS: int = int(os.getenv("NIM_MAX_TOKENS", 1024))
    TOP_P: float = float(os.getenv("NIM_TOP_P", 0.9))
    ENABLE_SAFETY_CHECK: bool = os.getenv("NIM_SAFETY_CHECK", "true").lower() == "true"
    MAX_LYRIC_LENGTH: int = int(os.getenv("NIM_MAX_LYRIC_LENGTH", 5000))
    REQUEST_TIMEOUT: float = float(os.getenv("NIM_TIMEOUT", 30.0))


# ── Pydantic Models ─────────────────────────────────────────────────
class ThemeItem(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    explanation: str = Field(..., min_length=10, max_length=500)
    confidence: float = Field(..., ge=0.0, le=1.0)
    lyric_evidence: str = Field(..., min_length=5, max_length=300)


class SongAnalysis(BaseModel):
    overall_mood: str = Field(..., min_length=10, max_length=200)
    mood_tags: List[str] = Field(..., min_length=1, max_length=8)
    themes: List[ThemeItem] = Field(..., min_length=1, max_length=5)
    summary: str = Field(..., min_length=20, max_length=400)
    language: str = Field(..., min_length=2, max_length=50)
    genre_inference: Optional[str] = Field(None, max_length=100)
    narrative_perspective: Optional[str] = Field(None, max_length=100)
    key_symbols: List[str] = Field(default_factory=list, max_length=5)
    estimated_era: Optional[str] = Field(None, max_length=50)


class AnalysisResult(BaseModel):
    success: bool
    data: Optional[SongAnalysis] = None
    raw_text: Optional[str] = None
    model_used: str
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: float
    request_id: str
    error: Optional[str] = None
    cached: bool = False
    fallback_used: bool = False


# ── Circuit Breaker ─────────────────────────────────────────────────
class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, threshold: int = 5, timeout: int = 60):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise RuntimeError("Circuit breaker is OPEN — too many failures. Try again later.")

        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker CLOSED — service recovered")
                else:
                    self.failure_count = 0
            return result
        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
            raise e


# ── TTL Cache ───────────────────────────────────────────────────────
class TTLCache:
    def __init__(self, maxsize: int = 100, ttl: int = 3600):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def _hash_key(self, lyrics: str, model: str) -> str:
        return hashlib.sha256(f"{lyrics}:{model}".encode()).hexdigest()[:32]

    async def get(self, lyrics: str, model: str) -> Optional[AnalysisResult]:
        key = self._hash_key(lyrics, model)
        async with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            if time.time() - entry["timestamp"] > self.ttl:
                del self._cache[key]
                return None
            entry["hits"] += 1
            result = entry["data"]
            result.cached = True
            logger.info(f"Cache HIT for key {key[:8]}... (hits: {entry['hits']})")
            return result

    async def set(self, lyrics: str, model: str, result: AnalysisResult):
        key = self._hash_key(lyrics, model)
        async with self._lock:
            if len(self._cache) >= self.maxsize:
                oldest = min(self._cache, key=lambda k: self._cache[k]["timestamp"])
                del self._cache[oldest]
            self._cache[key] = {
                "data": result,
                "timestamp": time.time(),
                "hits": 0,
            }
            logger.info(f"Cache SET for key {key[:8]}...")


# ── Rate Limiter ───────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, rpm: int = 60):
        self.interval = 60.0 / rpm
        self.last_request = 0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.time()
            wait = self.last_request + self.interval - now
            if wait > 0:
                logger.debug(f"Rate limiting: waiting {wait:.2f}s")
                await asyncio.sleep(wait)
            self.last_request = time.time()


# ── Cost Estimator ──────────────────────────────────────────────────
class CostEstimator:
    PRICING = {
        "meta/llama-3.1-8b-instruct": 0.10,
        "meta/llama-3.1-70b-instruct": 0.60,
        "meta/llama-3.1-405b-instruct": 2.50,
    }

    @classmethod
    def estimate(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        price = cls.PRICING.get(model, 0.50)
        total = input_tokens + output_tokens
        return (total / 1_000_000) * price


# ── Safety Filter ───────────────────────────────────────────────────
class SafetyFilter:
    SUSPICIOUS_PATTERNS = [
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    ]

    @classmethod
    def check(cls, text: str) -> tuple[bool, Optional[str]]:
        if len(text) > Config.MAX_LYRIC_LENGTH:
            return False, f"Lyrics exceed max length ({Config.MAX_LYRIC_LENGTH} chars)"
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text):
                return False, "Potentially sensitive data detected in lyrics"
        return True, None


# ── Prompt (NO guided_json — simpler is better on free tier) ────────
_ANALYSIS_SYSTEM_PROMPT = """You are a music analyst. Analyze song lyrics and output ONLY a JSON object.

Your JSON must have these exact fields:
- overall_mood: string (1 sentence describing the emotional feel)
- mood_tags: array of strings (3-5 emotion words like ["romantic", "playful", "upbeat"])
- themes: array of objects, each with:
  - name: string (theme name like "Young Love" or "Heartbreak")
  - explanation: string (1 sentence how it appears)
  - confidence: number (0.0 to 1.0)
  - lyric_evidence: string (a short direct quote from the lyrics)
- summary: string (2 sentences about the song's meaning)
- language: string (the language of the song)
- genre_inference: string or null (guessed genre)
- narrative_perspective: string or null (e.g., "first-person", "third-person")
- key_symbols: array of strings (recurring images/motifs)
- estimated_era: string or null (e.g., "2010s pop")

IMPORTANT:
- Output ONLY the JSON. No markdown, no ```, no explanation.
- Do NOT echo the lyrics back.
- Do NOT use "N/A" or "unknown" — always provide real values.
- Be specific: name actual themes, quote actual lyrics.
"""

_ANALYSIS_USER_PROMPT = """Analyze these song lyrics and return ONLY a JSON object:

{lyrics}

JSON output:"""


# ── Main Service ────────────────────────────────────────────────────
class NIMService:
    def __init__(self):
        if not Config.NIM_API_KEY:
            raise ValueError(
                "NIM_API_KEY environment variable must be set. "
                "Get your free key at: https://build.nvidia.com"
            )

        self.client = AsyncOpenAI(
            base_url=Config.NIM_BASE_URL,
            api_key=Config.NIM_API_KEY,
            timeout=Config.REQUEST_TIMEOUT,
        )
        self.circuit_breaker = CircuitBreaker(
            threshold=Config.CIRCUIT_BREAKER_THRESHOLD,
            timeout=Config.CIRCUIT_BREAKER_TIMEOUT,
        )
        self.cache = TTLCache(
            maxsize=Config.CACHE_MAX_SIZE,
            ttl=Config.CACHE_TTL_SECONDS,
        )
        self.rate_limiter = RateLimiter(rpm=Config.RATE_LIMIT_RPM)
        self._request_counter = 0

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"req-{self._request_counter}-{int(time.time() * 1000)}"

    async def analyze_lyrics(
        self,
        lyrics: str,
        force_refresh: bool = False,
        stream: bool = False,
    ) -> AnalysisResult:
        request_id = self._next_request_id()
        start_time = time.perf_counter()

        logger.info(f"[{request_id}] Starting analysis (stream={stream})")

        # Safety Check
        if Config.ENABLE_SAFETY_CHECK:
            safe, reason = SafetyFilter.check(lyrics)
            if not safe:
                logger.warning(f"[{request_id}] Safety check failed: {reason}")
                return AnalysisResult(
                    success=False,
                    error=reason,
                    model_used="N/A",
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    request_id=request_id,
                )

        # Cache Check
        if not force_refresh:
            cached = await self.cache.get(lyrics, Config.model_chain()[0])
            if cached:
                cached.request_id = request_id
                cached.latency_ms = (time.perf_counter() - start_time) * 1000
                logger.info(f"[{request_id}] Returning cached result")
                return cached

        # Rate Limit
        await self.rate_limiter.acquire()

        # Try model chain
        last_error = None
        fallback_used = False

        for idx, model in enumerate(Config.model_chain()):
            if idx > 0:
                fallback_used = True
                logger.warning(f"[{request_id}] Falling back to model: {model}")

            try:
                result = await self.circuit_breaker.call(
                    self._call_model,
                    lyrics=lyrics,
                    model=model,
                    request_id=request_id,
                    stream=stream,
                )
                result.fallback_used = fallback_used
                result.request_id = request_id
                result.latency_ms = (time.perf_counter() - start_time) * 1000

                await self.cache.set(lyrics, Config.model_chain()[0], result)

                logger.info(
                    f"[{request_id}] Success via {model} | "
                    f"latency={result.latency_ms:.0f}ms | "
                    f"tokens={result.tokens_used} | "
                    f"cost=${result.cost_usd:.4f}"
                )
                return result

            except Exception as e:
                last_error = e
                logger.error(f"[{request_id}] Model {model} failed: {e}")
                continue

        logger.critical(f"[{request_id}] All models exhausted. Last error: {last_error}")
        return AnalysisResult(
            success=False,
            error=f"All models failed. Last error: {last_error}",
            model_used=Config.model_chain()[-1],
            latency_ms=(time.perf_counter() - start_time) * 1000,
            request_id=request_id,
            fallback_used=True,
        )

    async def _call_model(
        self,
        lyrics: str,
        model: str,
        request_id: str,
        stream: bool = False,
    ) -> AnalysisResult:
        """Call model WITHOUT guided_json — simpler prompting works better on free tier."""

        prompt = _ANALYSIS_USER_PROMPT.format(lyrics=lyrics[:Config.MAX_LYRIC_LENGTH])

        for attempt in range(Config.MAX_RETRIES):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=Config.TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS,
                    top_p=Config.TOP_P,
                    stream=False,
                )

                raw_text = response.choices[0].message.content or ""
                tokens_used = response.usage.total_tokens if response.usage else None
                cost_usd = CostEstimator.estimate(
                    model,
                    response.usage.prompt_tokens if response.usage else 0,
                    response.usage.completion_tokens if response.usage else 0,
                ) if response.usage else None

                # Parse JSON
                data = self._parse_response(raw_text)

                return AnalysisResult(
                    success=True,
                    data=data,
                    raw_text=raw_text,
                    model_used=model,
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                    latency_ms=0,
                    request_id=request_id,
                )

            except Exception as e:
                if attempt < Config.MAX_RETRIES - 1:
                    delay = Config.RETRY_DELAY_BASE * (2 ** attempt) + (hash(request_id) % 100) / 1000
                    logger.warning(f"[{request_id}] Attempt {attempt + 1} failed, retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    def _parse_response(self, raw_text: str) -> SongAnalysis:
        """Extract JSON from model response with multiple fallback strategies."""
        text = raw_text.strip()

        # Strategy 1: Remove markdown fences and parse
        cleaned = text
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            return SongAnalysis.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            pass

        # Strategy 2: Find JSON object with regex (handles text before/after JSON)
        # Look for the first { and last } that form a valid object
        try:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = cleaned[start:end+1]
                data = json.loads(json_str)
                return SongAnalysis.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            pass

        # Strategy 3: Try to find JSON array of objects and take first
        try:
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start != -1 and end != -1 and end > start:
                arr = json.loads(cleaned[start:end+1])
                if isinstance(arr, list) and len(arr) > 0:
                    return SongAnalysis.model_validate(arr[0])
        except (json.JSONDecodeError, ValidationError):
            pass

        # Strategy 4: Fix common JSON issues and retry
        try:
            fixed = cleaned.replace("'", '"').replace("\n", " ").replace("\t", " ")
            # Remove trailing commas
            fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)
            data = json.loads(fixed)
            return SongAnalysis.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            pass

        raise RuntimeError(f"Could not parse model response as valid JSON. Raw: {raw_text[:300]}...")

    async def analyze_stream(self, lyrics: str) -> AsyncGenerator[str, None]:
        request_id = self._next_request_id()

        if Config.ENABLE_SAFETY_CHECK:
            safe, reason = SafetyFilter.check(lyrics)
            if not safe:
                yield json.dumps({"error": reason})
                return

        await self.rate_limiter.acquire()

        prompt = _ANALYSIS_USER_PROMPT.format(lyrics=lyrics[:Config.MAX_LYRIC_LENGTH])

        try:
            async for chunk in self.client.chat.completions.create(
                model=Config.model_chain()[0],
                messages=[
                    {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
                stream=True,
            ):
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            yield json.dumps({"error": str(e)})

    async def health_check(self) -> Dict[str, Any]:
        try:
            await self.rate_limiter.acquire()
            response = await self.client.chat.completions.create(
                model=Config.model_chain()[0],
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return {
                "status": "healthy",
                "model": Config.model_chain()[0],
                "circuit_state": self.circuit_breaker.state.value,
                "cache_size": len(self.cache._cache),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_state": self.circuit_breaker.state.value,
            }


# ── Backward-compatible sync wrapper ─────────────────────────────────
class NIMServiceSync:
    def __init__(self):
        self._async_service = NIMService()

    def analyze_lyrics(self, lyrics: str) -> str:
        result = asyncio.run(self._async_service.analyze_lyrics(lyrics))

        if not result.success:
            raise RuntimeError(result.error or "Unknown error")

        data = result.data
        lines = [
            "## 1. Overall Mood / Vibe",
            data.overall_mood,
            "",
            "## 2. Top 3 Core Themes",
        ]
        for t in data.themes[:3]:
            lines.append(f"- **{t.name}** ({t.confidence:.0%} confidence): {t.explanation}")

        lines.extend([
            "",
            "## 3. Summary",
            data.summary,
            "",
            "## 4. Song Language",
            data.language,
        ])

        if data.genre_inference:
            lines.append(f"")
            lines.append(f"*Genre inference: {data.genre_inference}*")
        if data.narrative_perspective:
            lines.append(f"*Narrative perspective: {data.narrative_perspective}*")

        return "\n".join(lines).strip()


# ── Legacy function alias ───────────────────────────────────────────
_analyze_lyrics_themes: Optional[NIMServiceSync] = None

def analyze_lyrics_themes(lyrics: str) -> str:
    global _analyze_lyrics_themes
    if _analyze_lyrics_themes is None:
        _analyze_lyrics_themes = NIMServiceSync()
    return _analyze_lyrics_themes.analyze_lyrics(lyrics)


def get_service(sync: bool = False):
    return NIMServiceSync() if sync else NIMService()


# ── Example usage ─────────────────────────────────────────────────
if __name__ == "__main__":
    async def demo():
        service = NIMService()
        health = await service.health_check()
        print("Health:", json.dumps(health, indent=2))

        sample = """Is this the real life? Is this just fantasy?
Caught in a landslide, no escape from reality.
Open your eyes, look up to the skies and see..."""

        result = await service.analyze_lyrics(sample)
        print("\nResult:", json.dumps(result.model_dump(), indent=2, default=str))

    asyncio.run(demo())
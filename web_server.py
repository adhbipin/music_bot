"""
Music Theme Analyzer — Advanced FastAPI Backend
================================================
Production-grade API with:
  • Request ID tracing (via decorator middleware)
  • CORS middleware
  • Rate limiting (SlowAPI)
  • Structured logging
  • Health & metrics endpoints
  • Streaming SSE endpoint
  • Background analytics tasks
  • Gzip compression
  • Input sanitization
  • Proper lifespan management

Run:  uvicorn web_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Import advanced services
from services.genius_service import get_lyrics_service, GeniusMetrics
from services.ai_service import NIMService, Config as NIMConfig


# ── Logging Setup ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("api")


# ── Rate Limiter ────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan (Startup/Shutdown) ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle."""
    logger.info("🚀 Starting up Music Theme Analyzer API...")

    # Pre-warm services
    app.state.nim_service = NIMService()
    app.state.genius_service = get_lyrics_service()
    app.state._start_time = time.time()

    # Health check on startup
    try:
        health = await app.state.nim_service.health_check()
        if health["status"] == "healthy":
            logger.info("✅ NIM API connection verified")
        else:
            logger.warning(f"⚠️ NIM API health check failed: {health}")
    except Exception as e:
        logger.warning(f"⚠️ Could not verify NIM API on startup: {e}")

    yield

    logger.info("🛑 Shutting down API...")


# ── App Initialization ────────────────────────────────────────────────
app = FastAPI(
    title="Music Theme Analyzer API",
    description="Advanced lyrics fetching & AI thematic analysis",
    version="2.0.0",
    lifespan=lifespan,
)

# Middleware stack
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request ID Middleware (decorator-based, works with all uvicorn versions) ──
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Injects a unique request ID into every request for tracing."""
    request_id = request.headers.get("X-Request-ID", f"req-{int(time.time() * 1000)}")
    request.state.request_id = request_id

    start = time.perf_counter()
    response = await call_next(request)
    latency = (time.perf_counter() - start) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{latency:.2f}ms"

    logger.info(
        f"[{request_id}] {request.method} {request.url.path} | "
        f"{response.status_code} | {latency:.2f}ms"
    )
    return response


# ── Pydantic Models ─────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Song title")
    artist: str = Field(..., min_length=1, max_length=200, description="Artist name")

    @field_validator("title", "artist")
    @classmethod
    def sanitize(cls, v: str) -> str:
        v = v.strip()
        for bad in ["<", ">", "&", ";"]:
            v = v.replace(bad, "")
        return v


class AnalysisResponse(BaseModel):
    title: str
    artist: str
    lyrics_length: int
    analysis: str
    model_used: str = "unknown"
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: float
    request_id: str
    cached: bool = False


class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"
    nim_health: Dict[str, Any]
    genius_metrics: Dict[str, Any]
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    request_id: str
    suggestion: Optional[str] = None


# ── API Endpoints ───────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
@limiter.limit("10/minute")
async def health_check(request: Request):
    """Comprehensive health check with service metrics."""
    try:
        nim_health = await request.app.state.nim_service.health_check()
    except Exception as e:
        nim_health = {"status": "unhealthy", "error": str(e)}

    genius_metrics = request.app.state.genius_service.get_metrics()

    return HealthResponse(
        status="healthy" if nim_health.get("status") == "healthy" else "degraded",
        nim_health=nim_health,
        genius_metrics=genius_metrics,
        uptime_seconds=time.time() - getattr(request.app.state, "_start_time", time.time()),
    )


@app.get("/metrics")
@limiter.limit("10/minute")
async def metrics(request: Request):
    """Prometheus-style metrics endpoint."""
    try:
        nim_health = await request.app.state.nim_service.health_check()
    except Exception as e:
        nim_health = {"status": "unhealthy", "error": str(e)}

    g_metrics = request.app.state.genius_service.get_metrics()

    return {
        "genius": g_metrics,
        "nim": {
            "status": nim_health.get("status", "unknown"),
            "circuit_state": nim_health.get("circuit_state", "unknown"),
            "cache_size": nim_health.get("cache_size", 0),
        }
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
@limiter.limit("20/minute")
async def analyze_song(request: Request, body: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Full analysis endpoint — fetches lyrics then runs AI analysis.
    Rate limited to 20 requests/minute per IP.
    """
    request_id = getattr(request.state, "request_id", f"req-{int(time.time() * 1000)}")
    start = time.perf_counter()

    logger.info(f"[{request_id}] Analyzing '{body.title}' by '{body.artist}'")

    # 1. Fetch lyrics
    try:
        lyrics = await request.app.state.genius_service.fetch_lyrics(body.title, body.artist)
    except Exception as e:
        logger.error(f"[{request_id}] Genius error: {e}")
        raise HTTPException(status_code=502, detail=f"Lyrics service error: {e}")

    if not lyrics:
        raise HTTPException(
            status_code=404,
            detail=f"Could not find lyrics for '{body.title}' by {body.artist}'. Check spelling."
        )

    # 2. Analyze
    try:
        result = await request.app.state.nim_service.analyze_lyrics(lyrics)
    except Exception as e:
        logger.error(f"[{request_id}] NIM error: {e}")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {e}")

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Analysis failed")

    # 3. Format response
    latency = (time.perf_counter() - start) * 1000

    # Build markdown from structured data
    data = result.data
    if data is None:
        # Fallback: show raw text if parsing failed but API returned something
        analysis_md = f"## Analysis Result\n\n{result.raw_text or 'No analysis available.'}"
    else:
        lines = [
            f"## 🎵 {body.title} — {body.artist}",
            "",
            f"**Overall Mood:** {data.overall_mood}",
            "",
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

        analysis_md = "\n".join(lines)

    # 4. Background analytics (fire-and-forget)
    background_tasks.add_task(
        _log_analysis,
        request_id,
        body.title,
        body.artist,
        result.model_used,
        result.tokens_used,
        result.cost_usd,
        latency,
    )

    return AnalysisResponse(
        title=body.title,
        artist=body.artist,
        lyrics_length=len(lyrics),
        analysis=analysis_md,
        model_used=result.model_used,
        tokens_used=result.tokens_used,
        cost_usd=result.cost_usd,
        latency_ms=latency,
        request_id=request_id,
        cached=result.cached,
    )


@app.post("/api/analyze/stream")
@limiter.limit("10/minute")
async def analyze_stream(request: Request, body: AnalysisRequest):
    """
    Server-Sent Events streaming endpoint.
    Returns analysis chunks in real-time as the AI generates them.
    """
    request_id = getattr(request.state, "request_id", f"req-{int(time.time() * 1000)}")

    # Fetch lyrics first (blocking)
    try:
        lyrics = await request.app.state.genius_service.fetch_lyrics(body.title, body.artist)
    except Exception as e:
        async def error_stream():
            yield f"data: {json.dumps({'error': f'Lyrics fetch failed: {e}'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    if not lyrics:
        async def error_stream():
            yield f"data: {json.dumps({'error': 'Lyrics not found'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # Stream AI response
    async def event_generator():
        buffer = []
        try:
            async for chunk in request.app.state.nim_service.analyze_stream(lyrics):
                buffer.append(chunk)
                yield f"data: {json.dumps({'chunk': chunk, 'request_id': request_id})}\n\n"

            # Send final assembled text
            full_text = "".join(buffer)
            yield f"data: {json.dumps({'done': True, 'full_text': full_text, 'request_id': request_id})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'request_id': request_id})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Request-ID": request_id,
        }
    )


# ── Background Tasks ────────────────────────────────────────────────
async def _log_analysis(
    request_id: str,
    title: str,
    artist: str,
    model: str,
    tokens: Optional[int],
    cost: Optional[float],
    latency_ms: float,
):
    """Fire-and-forget analytics logging."""
    logger.info(
        f"[ANALYTICS] {request_id} | {title} by {artist} | "
        f"model={model} | tokens={tokens} | cost=${cost:.4f if cost else 0} | "
        f"latency={latency_ms:.0f}ms"
    )


# ── Static Files ────────────────────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))


# ── Error Handlers ──────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            request_id=request_id,
            suggestion="Check your input or try again later." if exc.status_code >= 500 else "Check spelling.",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(f"[{request_id}] Unhandled exception")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            request_id=request_id,
            suggestion="Please report this issue.",
        ).model_dump(),
    )


# ── Entry Point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
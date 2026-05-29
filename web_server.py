import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import our custom service layers
from services.genius_service import fetch_lyrics
from services.ai_service import analyze_lyrics_themes

# Load environment variables
load_dotenv()

app = FastAPI(title="Music Lyrics & Theme Analyzer API")

# Define the request model
class AnalysisRequest(BaseModel):
    title: str
    artist: str

# Define the response model
class AnalysisResponse(BaseModel):
    title: str
    artist: str
    lyrics_length: int
    analysis: str

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_song(request: AnalysisRequest):
    """
    Fetches lyrics for a song and analyzes its themes.
    """
    if not request.title or not request.artist:
        raise HTTPException(status_code=400, detail="Song title and artist are required.")

    # 1. Fetch Lyrics
    try:
        lyrics = fetch_lyrics(request.title, request.artist)
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except ConnectionError as ce:
        raise HTTPException(status_code=502, detail=str(ce))

    if not lyrics:
        raise HTTPException(
            status_code=404, 
            detail=f"Could not find lyrics for '{request.title}' by {request.artist}."
        )

    # 2. Analyze Lyrics
    try:
        analysis = analyze_lyrics_themes(lyrics)
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))

    return AnalysisResponse(
        title=request.title,
        artist=request.artist,
        lyrics_length=len(lyrics),
        analysis=analysis
    )

# Serve static files
# We'll create a 'static' directory for our frontend
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

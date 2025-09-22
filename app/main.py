"""FastAPI application exposing a streaming OpenAI-compatible TTS endpoint."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.services.hf_models import get_trending_tts_models
from app.services.tts_manager import TTSManager
from app.utils.audio import audio_array_to_wav_bytes, chunk_bytes

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Multi-Model Streaming TTS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


tts_manager = TTSManager()


class SpeechRequest(BaseModel):
    model: str = Field(..., description="Hugging Face model identifier to use for synthesis.")
    input: str = Field(..., description="Text prompt to be synthesised into speech.")
    voice: Optional[str] = Field(None, description="Optional voice/speaker identifier supported by some models.")
    language: Optional[str] = Field(
        None, description="Optional language hint for multilingual models (model dependent)."
    )
    format: Optional[str] = Field(
        default="wav",
        description="Requested audio container format. Only 'wav' is currently supported.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "espnet/kan-bayashi_ljspeech_vits",
                "input": "Welcome to the Multi-Model TTS API demo!",
                "format": "wav",
            }
        }


@app.get("/", response_class=FileResponse)
async def serve_index() -> FileResponse:
    """Serve the bundled single-page application."""

    index_path = static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend application not found.")
    return FileResponse(index_path)


@app.get("/api/models")
async def list_models(refresh: bool = Query(False, description="Bypass cache and refresh the Hugging Face listing.")):
    """Return metadata for the top 20 trending text-to-speech models from Hugging Face."""

    try:
        models = await get_trending_tts_models(force_refresh=refresh)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Unable to fetch model list from Hugging Face")
        raise HTTPException(status_code=502, detail=f"Failed to load model list: {exc}")

    return {"count": len(models), "models": models}


@app.options("/v1/audio/speech")
async def speech_options() -> Response:
    """Handle pre-flight CORS checks for the speech endpoint."""

    return Response(status_code=204)


@app.post("/v1/audio/speech")
async def generate_speech(payload: SpeechRequest):
    """Generate audio using a Hugging Face text-to-speech model and stream the WAV output."""

    if payload.format and payload.format.lower() != "wav":
        raise HTTPException(status_code=400, detail="Only WAV audio output is currently supported.")

    try:
        audio_array, sample_rate = await tts_manager.synthesize(
            payload.model, payload.input, voice=payload.voice, language=payload.language
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    audio_bytes = await asyncio.to_thread(audio_array_to_wav_bytes, audio_array, sample_rate)

    async def audio_generator():
        for chunk in chunk_bytes(audio_bytes):
            yield chunk
            await asyncio.sleep(0)

    headers = {
        "X-Model-Id": payload.model,
        "Cache-Control": "no-cache",
    }
    return StreamingResponse(audio_generator(), media_type="audio/wav", headers=headers)


@app.get("/healthz")
async def health_check():
    """Simple health probe used by container orchestrators."""

    return {"status": "ok", "device": tts_manager.device}


__all__ = ["app"]

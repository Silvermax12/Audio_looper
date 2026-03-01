"""
Phrase-Song API — FastAPI Application

A lightweight API that analyzes audio files to find the best "Phrase" loop
segment (8–13 seconds). Designed for deployment on Render (512 MB RAM tier).

Run locally:
    uvicorn main:app --reload --port 8001

Deploy note: use plain `uvicorn` (not uvicorn[standard]) to avoid pulling in
extra optional deps (websockets, httptools) that are not needed here.
"""

import io
import os
import tempfile

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from models import LoopPair, LoopAnalysisResponse
from analysis.loop_detection import detect_seamless_loops, apply_crossfade

app = FastAPI(
    title="Phrase-Song API",
    description="Analyze audio files to find the best phrase-length loop segment (8–13s)",
    version="1.0.0",
)

# CORS — allow all origins for API consumption
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported audio formats
SUPPORTED_FORMATS = {"mp3", "wav", "m4a", "ogg", "flac", "aac", "webm"}

# Hard cap on upload size — protects the 512 MB Render tier from OOM
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

# Sample rate for analysis — 16 kHz is sufficient for beat/chroma features
# and uses ~27% less RAM than 22 050 Hz.
ANALYSIS_SR = 16_000

# Chunk size for streaming the upload to disk (never load the whole file into RAM)
_CHUNK = 256 * 1024  # 256 KB


# ── Helpers ──────────────────────────────────────────────────────────────────

def _validate_audio(file: UploadFile) -> str:
    """Validate uploaded file extension. Returns the extension."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '.{ext}'. Accepted: {', '.join(sorted(SUPPORTED_FORMATS))}",
        )
    return ext


async def _save_temp(file: UploadFile, ext: str) -> str:
    """Stream upload to a temp file in chunks — avoids loading entire file into RAM."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
    total = 0
    try:
        while True:
            chunk = await file.read(_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
                )
            tmp.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        tmp.close()
        os.unlink(tmp.name)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc
    tmp.close()
    return tmp.name


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health-check for Render."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/phrase-song")
async def phrase_song_info():
    """
    Return API usage information.
    """
    return {
        "endpoint": "POST /phrase-song",
        "description": (
            "Upload an audio file to find the best phrase-length loop segment. "
            "A 'Phrase' is an 8–13 second musically coherent section that loops seamlessly."
        ),
        "accepted_formats": sorted(SUPPORTED_FORMATS),
        "parameters": {
            "file": "(required) Audio file upload",
            "extract": "(optional query param, default false) If true, returns the phrase segment as a WAV file instead of JSON.",
        },
        "example_curl": 'curl -X POST -F "file=@song.mp3" https://<your-host>/phrase-song',
    }


@app.post("/phrase-song")
async def phrase_song_analyze(
    file: UploadFile = File(...),
    extract: bool = False,
):
    """
    Analyze an audio file and return the best Phrase loop (8–13s).

    - **file**: Audio file (mp3, wav, m4a, ogg, flac, aac, webm)
    - **extract**: If `true`, respond with the extracted phrase segment as a
      downloadable WAV instead of JSON analysis data.
    """
    ext = _validate_audio(file)
    tmp_path = await _save_temp(file, ext)

    try:
        # Load audio at a lower sample rate — sufficient for beat/chroma analysis
        # and uses ~27% less RAM than the default 22 050 Hz.
        y, sr = librosa.load(tmp_path, sr=ANALYSIS_SR, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        if duration < 5.0:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too short ({duration:.1f}s). Minimum 5 seconds required.",
            )

        # Run loop detection — cap max duration at 20 s (Phrase ≤ 13 s, so this
        # is generous) to keep the candidate search loop fast and memory-lean.
        loop_pairs, audio_duration, tempo = detect_seamless_loops(
            y, sr,
            top_n=3,
            min_loop_duration=3.0,
            max_loop_duration=20.0,
            similarity_threshold=0.55,
        )

        if not loop_pairs:
            raise HTTPException(
                status_code=404,
                detail="No suitable loop segments found in this audio.",
            )

        # Find the best Phrase (8–13s) category
        phrase_loop = None
        for lp in loop_pairs:
            if lp.bar_category == "Phrase":
                phrase_loop = lp
                break

        # Fallback: pick the best overall loop
        if phrase_loop is None:
            phrase_loop = loop_pairs[0]

        # ── Extract mode: return WAV of the phrase segment ──
        if extract:
            start_sample = int(phrase_loop.start_time * sr)
            end_sample = int(phrase_loop.end_time * sr)
            crossfade_samples = int(phrase_loop.recommended_crossfade_ms / 1000 * sr)

            segment = apply_crossfade(y, sr, start_sample, end_sample, crossfade_samples)

            buf = io.BytesIO()
            sf.write(buf, segment, sr, format="WAV")
            buf.seek(0)

            return StreamingResponse(
                buf,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f'attachment; filename="phrase_{phrase_loop.start_time:.1f}s_{phrase_loop.end_time:.1f}s.wav"'
                },
            )

        # ── JSON mode: return analysis ──
        return {
            "phrase": phrase_loop.model_dump(),
            "all_loops": [lp.model_dump() for lp in loop_pairs],
            "audio_info": {
                "duration": round(audio_duration, 2),
                "tempo": round(tempo, 1),
            },
        }

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Local dev entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

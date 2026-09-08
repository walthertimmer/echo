from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from transformers import pipeline
import logging
import os
import subprocess
import tempfile
import torch
import uvicorn
import httpx
import asyncio
import time

from transcribe import WhisperTranscriber, get_transcriber, VLLM_AVAILABLE, TRANSFORMERS_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audio Transcription API",
    description="Transcribe audio files using Whisper models (vLLM or Transformers)",
    version="1"
)

# Serve static files (CSS, JS, etc.)
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Global transcriber instance - auto-select backend
transcriber = get_transcriber()


# Background task to load model on startup
async def load_model_on_startup():
    """Load the Whisper model when the server starts."""
    logger.info(f"Starting model loading in background... (vLLM available: {VLLM_AVAILABLE}, Transformers available: {TRANSFORMERS_AVAILABLE})")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, transcriber.load_model)
        logger.info(f"Model loaded successfully with {transcriber.backend} backend")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")


@app.on_event("startup")
async def startup_event():
    """Handle server startup."""
    # Start model loading in background
    asyncio.create_task(load_model_on_startup())


@app.on_event("shutdown")
async def shutdown_event():
    """Handle server shutdown."""
    logger.info("Shutting down...")
    try:
        transcriber.unload_model()
        logger.info("Model unloaded successfully")
    except Exception as e:
        logger.error(f"Error unloading model: {e}")


@app.get("/")
async def root():
    return FileResponse(str(BASE_DIR / "templates" / "index.html"))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={
        "status": "healthy",
        "model_loaded": transcriber.is_loaded,
        "model_name": transcriber.model_name,
        "backend": transcriber.backend,
        "vllm_available": VLLM_AVAILABLE,
        "transformers_available": TRANSFORMERS_AVAILABLE
    })


@app.get("/model/status")
async def model_status():
    """Check if the Whisper model is loaded."""
    return JSONResponse(content={
        "model_loaded": transcriber.is_loaded,
        "model_name": transcriber.model_name,
        "backend": transcriber.backend
    })


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    temperature: float = 0.0,
    max_tokens: int = 448,  # Increased for longer audio
    language: str = None
):
    """
    Transcribe an uploaded audio file using vLLM Whisper Turbo.
    
    Args:
        file: Audio file to transcribe (WAV, MP3, etc.)
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum number of tokens to generate
        language: Optional language hint (e.g., "en", "nl")
    
    Returns:
        JSON with transcription result
    """
    # Ensure model is loaded
    if not transcriber.is_loaded:
        logger.info("Model not loaded, loading now...")
        try:
            transcriber.load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=503, detail=f"Model loading failed: {str(e)}")
    
    try:
        # Read the uploaded file
        audio_bytes = await file.read()
        
        logger.info(f"Received audio file: {file.filename}, size: {len(audio_bytes)} bytes")
        
        start_time = time.time()
        
        # Pass bytes directly to transcriber - it handles format conversion
        # Use thread pool for synchronous transcription
        loop = asyncio.get_event_loop()
        transcription_text = await loop.run_in_executor(
            None, 
            lambda: transcriber.transcribe_audio_bytes(
                audio_bytes,
                temperature=temperature,
                max_tokens=max_tokens,
                language=language
            )
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Transcription completed in {elapsed_time:.2f}s")
        
        return JSONResponse(content={
            "filename": file.filename,
            "transcription": transcription_text,
            "processing_time": elapsed_time,
            "model": transcriber.model_name,
            "backend": transcriber.backend,
            "language": language
        })
                
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"Audio file not found: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=f"Transcription error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/transcribe/file")
async def transcribe_audio_file(
    request: Request
):
    """
    Transcribe audio from a local file path.
    
    Args:
        file_path: Path to the audio file on the server
    
    Returns:
        JSON with transcription result
    """
    if not transcriber.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        data = await request.json()
        file_path = data.get("file_path")
        
        if not file_path:
            raise HTTPException(status_code=400, detail="file_path is required")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        temperature = data.get("temperature", 0.0)
        max_tokens = data.get("max_tokens", 448)
        language = data.get("language")
        
        # Use thread pool for synchronous transcription
        loop = asyncio.get_event_loop()
        transcription_text = await loop.run_in_executor(
            None, 
            lambda: transcriber.transcribe_audio_file(
                file_path,
                temperature=temperature,
                max_tokens=max_tokens,
                language=language
            )
        )
        
        return JSONResponse(content={
            "file_path": file_path,
            "transcription": transcription_text,
            "model": transcriber.model_name
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Legacy endpoints for backward compatibility
@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Legacy upload endpoint."""
    try:
        # Just save the file temporarily and return info
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        return JSONResponse(content={
            "filename": file.filename,
            "temp_path": temp_path,
            "size": len(content)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

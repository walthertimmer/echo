from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from transformers import pipeline
import logging
import os
import subprocess
import tempfile
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audio Transcription API",
    description="Transcribe audio files using Hugging Face Transformers",
    version="1"
)

# Serve static files (CSS, JS, etc.)
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Load the transcription pipeline (Whisper)
# This loads on startup - may take a few seconds
# options:
# openai/whisper-tiny
# distil-whisper/distil-small
try:
    transcriber = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        dtype=torch.float32,
        device=-1  # CPU by default, change to 0 for GPU if available
    )
except Exception as e:
    transcriber = None
    logger.error(f"Failed to load model: {e}")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an audio file.

    Supports: MP3, WAV, FLAC, OGG, etc. (any format Whisper can handle)
    """
    if transcriber is None:
        raise HTTPException(
            status_code=500,
            detail="Transcription model not loaded"
        )

    # Validate file is audio
    valid_extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".webm"]
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in valid_extensions:
        logger.warning(f"Rejected upload: unsupported extension {file_extension}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {file_extension}. Supported: {valid_extensions}"
        )

    try:
        logger.info(f"Transcribing file: {file.filename}")
        
        # Read file content
        content = await file.read()
        
        # Save to temp file with original extension
        input_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            input_path = tmp.name
            tmp.write(content)
            tmp.flush()
        
        # For formats that soundfile doesn't support (like .m4a), convert to .wav using ffmpeg
        output_path = input_path
        if file_extension in [".m4a", ".webm"]:
            output_path = input_path + ".wav"
            try:
                subprocess.run([
                    "ffmpeg", "-i", input_path,
                    "-ac", "1", "-ar", "16000",  # mono, 16kHz for Whisper
                    "-y",  # overwrite
                    output_path
                ], check=True, capture_output=True)
                logger.info(f"Converted {file.filename} to WAV for processing")
            except subprocess.CalledProcessError as e:
                logger.error(f"ffmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
                raise
        
        # Transcribe from file path
        result = transcriber(output_path)
        logger.info(f"Transcription complete for: {file.filename}")

        # Clean up
        for path in [input_path, output_path]:
            if path and os.path.exists(path):
                os.unlink(path)

        return JSONResponse(content={
            "transcription": result["text"],
            "model": "openai/whisper-tiny"
        })

    except Exception as e:
        # Clean up temp files on error
        for path in [input_path, output_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass
        logger.error(f"Transcription failed for {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

@app.get("/")
async def root():
    return FileResponse(str(BASE_DIR / "templates" / "index.html"))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

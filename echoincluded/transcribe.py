"""
Transcription module using Whisper models.

Supports both vLLM (for GPU-accelerated inference with quantized models)
and Transformers (fallback for CPU or when vLLM is not available).

This module provides a transcription service using:
- vLLM: neuralmagic/whisper-large-v3-turbo-quantized.w4a16 (recommended for GPU)
- Transformers: Xenova/whisper-tiny or similar (fallback for CPU)
"""

import tempfile
import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import vLLM components
try:
    from vllm import LLM, SamplingParams
    from vllm.assets.audio import AudioAsset
    VLLM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"vLLM not available: {e}. Falling back to Transformers.")
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    AudioAsset = None

# Try to import Transformers components
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available.")
    TRANSFORMERS_AVAILABLE = False


class WhisperTranscriber:
    """
    Transcription service using Whisper models.
    Handles model loading, audio processing, and text generation.
    Supports both vLLM (GPU) and Transformers (CPU) backends.
    """
    
    def __init__(self, model_name: str = "neuralmagic/whisper-large-v3-turbo-quantized.w4a16", backend: str = "auto"):
        """
        Initialize the transcriber with the specified model and backend.
        
        Args:
            model_name: Name of the Whisper model
            backend: Backend to use - "vllm", "transformers", or "auto"
        """
        self.model_name = model_name
        self.backend = backend
        self.llm: Optional[LLM] = None
        self.transformer_pipeline = None
        self._is_loaded = False
        
        # Determine which backend to use
        if backend == "auto":
            if VLLM_AVAILABLE:
                self.backend = "vllm"
                logger.info("Using vLLM backend")
            elif TRANSFORMERS_AVAILABLE:
                self.backend = "transformers"
                logger.info("Using Transformers backend (vLLM not available)")
            else:
                raise RuntimeError("No transcription backend available. Install vllm or transformers.")
        elif backend == "vllm":
            if not VLLM_AVAILABLE:
                raise RuntimeError("vLLM not available. Install with: pip install vllm")
        elif backend == "transformers":
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers not available. Install with: pip install transformers")
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'vllm', 'transformers', or 'auto'.")
        
    def load_model(self, max_model_len: int = 448, max_num_seqs: int = 400) -> None:
        """
        Load the model with optimized settings for Whisper.
        
        Args:
            max_model_len: Maximum sequence length for the model (vLLM only)
            max_num_seqs: Maximum number of sequences to process (vLLM only)
        """
        if self._is_loaded:
            logger.info("Model already loaded")
            return
            
        try:
            logger.info(f"Loading model: {self.model_name} (backend: {self.backend})")
            
            if self.backend == "vllm":
                self.llm = LLM(
                    model=self.model_name,
                    max_model_len=max_model_len,
                    max_num_seqs=max_num_seqs,
                    limit_mm_per_prompt={"audio": 1},
                    dtype="auto",
                    trust_remote_code=True,
                )
                
            elif self.backend == "transformers":
                # For Transformers, use a more standard Whisper model
                # vLLM models may not work directly with Transformers
                transformer_model = self.model_name
                
                # Map vLLM model names to Transformers-compatible ones
                if "neuralmagic" in self.model_name:
                    # For macOS/CPU, use smaller models. whisper-large is ~1.5GB which is too heavy for most CPUs
                    # Use whisper-medium (770MB) or whisper-small (244MB) for better CPU performance
                    import os
                    cpu_model = os.environ.get("TRANSFORMERS_WHISPER_MODEL", "openai/whisper-medium")
                    transformer_model = cpu_model
                    logger.info(f"Mapping {self.model_name} to {transformer_model} for Transformers backend")
                
                # Determine the best device for the current platform
                device = self._get_best_device()
                logger.info(f"Using device: {device} for Transformers backend with model: {transformer_model}")
                
                try:
                    self.transformer_pipeline = pipeline(
                        "automatic-speech-recognition",
                        model=transformer_model,
                        device=device,
                        torch_dtype="auto",
                    )
                except Exception as e:
                    # If the specified model fails, try smaller ones
                    fallback_models = ["openai/whisper-medium", "openai/whisper-small", "openai/whisper-base", "openai/whisper-tiny"]
                    for fallback_model in fallback_models:
                        if fallback_model == transformer_model:
                            continue
                        logger.warning(f"Failed to load {transformer_model}: {e}. Trying {fallback_model}...")
                        try:
                            self.transformer_pipeline = pipeline(
                                "automatic-speech-recognition",
                                model=fallback_model,
                                device=device,
                                torch_dtype="auto",
                            )
                            transformer_model = fallback_model
                            logger.info(f"Successfully loaded fallback model: {transformer_model}")
                            break
                        except Exception as e2:
                            logger.warning(f"Also failed to load {fallback_model}: {e2}")
                            continue
                    else:
                        # All fallbacks failed
                        logger.error(f"Failed to load any Whisper model")
                        raise RuntimeError(f"Could not load any Whisper model. Last error: {e}")
                
            self._is_loaded = True
            logger.info(f"Model {self.model_name} loaded successfully with {self.backend} backend")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def transcribe_audio_file(
        self, 
        audio_path: str, 
        temperature: float = 0.0, 
        max_tokens: int = 64,
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe an audio file using the loaded Whisper model.
        
        Args:
            audio_path: Path to the audio file
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum number of tokens to generate
            language: Optional language hint (e.g., "en", "nl")
            
        Returns:
            Transcribed text
            
        Raises:
            RuntimeError: If model is not loaded
            FileNotFoundError: If audio file doesn't exist
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            logger.info(f"Starting transcription for {audio_path} (backend: {self.backend})")
            
            if self.backend == "vllm":
                if self.llm is None:
                    raise RuntimeError("vLLM model not loaded")
                
                # Prepare inputs for Whisper with vLLM
                inputs = {
                    "encoder_prompt": {
                        "prompt": "",
                        "multi_modal_data": {
                            "audio": AudioAsset(audio_path).audio_and_sample_rate,
                        },
                    },
                    "decoder_prompt": "<|startoftranscript|>",
                }
                
                # Set sampling parameters
                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1.0,
                    top_k=-1,
                )
                
                # Generate response
                outputs = self.llm.generate(inputs, sampling_params)
                
                # Extract and return the transcribed text
                if outputs and len(outputs) > 0:
                    text = outputs[0].outputs[0].text
                    logger.info(f"Transcription completed: {len(text)} characters")
                    return text
                else:
                    logger.warning("No output generated from model")
                    return ""
                    
            elif self.backend == "transformers":
                if self.transformer_pipeline is None:
                    raise RuntimeError("Transformers pipeline not loaded")
                
                # Use Transformers pipeline
                options = {
                    "return_timestamps": True,  # Required for audio >30 seconds
                    "language": language,
                }
                
                # Try to pass the file path directly
                try:
                    result = self.transformer_pipeline(
                        audio_path,
                        **options
                    )
                except Exception as e:
                    # If direct file path fails (e.g., unsupported format), try reading with torchaudio
                    logger.warning(f"Direct file read failed: {e}. Trying with torchaudio...")
                    try:
                        import torchaudio
                        import io
                        
                        # Read with torchaudio and pass waveform to pipeline
                        waveform, sample_rate = torchaudio.load(audio_path)
                        
                        # Convert to numpy if needed
                        if hasattr(waveform, 'numpy'):
                            waveform = waveform.numpy()
                        
                        result = self.transformer_pipeline(
                            {"array": waveform, "sampling_rate": sample_rate},
                            **options
                        )
                    except Exception as e2:
                        logger.error(f"Torchaudio fallback also failed: {e2}")
                        raise RuntimeError(f"Could not read audio file {audio_path}: {e} and {e2}")
                
                # Extract text based on pipeline output format
                if isinstance(result, list) and len(result) > 0:
                    text = result[0].get("text", "")
                elif isinstance(result, dict):
                    text = result.get("text", "")
                else:
                    text = str(result)
                
                logger.info(f"Transcription completed: {len(text)} characters")
                return text
            else:
                raise RuntimeError(f"Unknown backend: {self.backend}")
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def transcribe_audio_bytes(
        self, 
        audio_bytes: bytes,
        temperature: float = 0.0,
        max_tokens: int = 64,
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe audio from bytes using the loaded Whisper model.
        
        Args:
            audio_bytes: Raw audio file bytes
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum number of tokens to generate
            language: Optional language hint
            
        Returns:
            Transcribed text
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if self.backend == "transformers":
            # For Transformers, we can try to pass bytes directly or use BytesIO
            import io
            
            # Try with BytesIO first - Transformers pipeline can auto-detect format
            audio_io = io.BytesIO(audio_bytes)
            options = {
                "return_timestamps": True,  # Enable for long audio (>30 seconds)
                "language": language,
            }
            
            try:
                result = self.transformer_pipeline(audio_io, **options)
                
                # Extract text
                if isinstance(result, list) and len(result) > 0:
                    text = result[0].get("text", "")
                elif isinstance(result, dict):
                    text = result.get("text", "")
                else:
                    text = str(result)
                
                return text
                
            except Exception as e:
                logger.debug(f"Direct BytesIO failed: {e}, trying with file path")
                
                # If BytesIO doesn't work, try saving to temp file
                # But first try with torchaudio if available for format conversion
                try:
                    import torchaudio
                    audio_io = io.BytesIO(audio_bytes)
                    waveform, sample_rate = torchaudio.load(audio_io)
                    
                    # Convert to numpy if needed
                    if hasattr(waveform, 'numpy'):
                        waveform = waveform.numpy()
                    
                    result = self.transformer_pipeline(
                        {"array": waveform, "sampling_rate": sample_rate},
                        **options
                    )
                    
                    if isinstance(result, list) and len(result) > 0:
                        text = result[0].get("text", "")
                    elif isinstance(result, dict):
                        text = result.get("text", "")
                    else:
                        text = str(result)
                    
                    return text
                    
                except ImportError:
                    logger.debug("torchaudio not available")
                except Exception as e2:
                    logger.debug(f"torchaudio conversion failed: {e2}")
                
                # Last resort: fall back to temp file
                logger.warning(f"Direct processing failed, falling back to temp file")
        
        # Fallback: try to convert using ffmpeg if available
        try:
            import subprocess
            import shutil
            
            # Try using ffmpeg to convert to WAV
            with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as input_temp:
                input_temp.write(audio_bytes)
                input_path = input_temp.name
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_temp:
                output_path = output_temp.name
            
            try:
                # Use ffmpeg to convert
                result = subprocess.run(
                    ["ffmpeg", "-i", input_path, "-ac", "1", "-ar", "16000", "-y", output_path],
                    capture_output=True,
                    check=True
                )
                logger.debug(f"ffmpeg conversion successful: {result}")
                
                return self.transcribe_audio_file(
                    output_path, 
                    temperature=temperature, 
                    max_tokens=max_tokens, 
                    language=language
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.debug(f"ffmpeg conversion failed: {e}")
                # ffmpeg not available or failed, try direct file
                logger.warning("ffmpeg not available, trying direct file read")
            finally:
                # Clean up
                try:
                    os.unlink(input_path)
                except OSError:
                    pass
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
            
            # Last resort: write to temp file with original bytes
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            try:
                return self.transcribe_audio_file(
                    temp_path, 
                    temperature=temperature, 
                    max_tokens=max_tokens, 
                    language=language
                )
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                    
        except Exception as e:
            logger.error(f"Transcription from bytes failed: {e}")
            raise
    
    def transcribe_audio_numpy(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> str:
        """
        Transcribe audio from numpy array using the loaded Whisper model.
        
        Args:
            audio_data: Numpy array containing audio samples
            sample_rate: Sample rate of the audio data
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Transcribed text
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        try:
            # Write numpy array to temporary WAV file
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                
            try:
                sf.write(temp_path, audio_data, sample_rate)
                return self.transcribe_audio_file(
                    temp_path,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                    
        except ImportError:
            raise RuntimeError("soundfile library required for numpy array input. Install with: pip install soundfile")
        except Exception as e:
            logger.error(f"Transcription from numpy array failed: {e}")
            raise
    
    def _get_best_device(self) -> str:
        """
        Determine the best device for the current platform.
        
        Returns:
            Device string for PyTorch (cpu, mps, cuda, etc.)
        """
        import sys
        
        # Check for macOS (Metal Performance Shaders)
        if sys.platform == "darwin":
            try:
                import torch
                if torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
        
        # Check for CUDA
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        
        # Default to CPU
        return "cpu"
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        try:
            if self.llm is not None:
                # vLLM models should be properly cleaned up
                del self.llm
                self.llm = None
            if self.transformer_pipeline is not None:
                del self.transformer_pipeline
                self.transformer_pipeline = None
            
            self._is_loaded = False
            logger.info(f"Model unloaded (backend: {self.backend})")
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            raise
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._is_loaded
    

# Global transcriber instance for convenience
_transcriber: Optional[WhisperTranscriber] = None


def get_transcriber(backend: str = "auto", model_name: str = "neuralmagic/whisper-large-v3-turbo-quantized.w4a16") -> WhisperTranscriber:
    """
    Get the global transcriber instance, initializing it if necessary.
    
    Args:
        backend: Backend to use - "vllm", "transformers", or "auto"
        model_name: Name of the Whisper model
    
    Returns:
        WhisperTranscriber instance
    """
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber(model_name=model_name, backend=backend)
    return _transcriber


def transcribe_file(
    audio_path: str,
    temperature: float = 0.0,
    max_tokens: int = 64,
    language: Optional[str] = None,
    backend: str = "auto"
) -> str:
    """
    Convenience function to transcribe an audio file.
    
    Args:
        audio_path: Path to the audio file
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        language: Optional language hint
        backend: Backend to use - "vllm", "transformers", or "auto"
        
    Returns:
        Transcribed text
    """
    transcriber = get_transcriber(backend=backend)
    if not transcriber.is_loaded:
        transcriber.load_model()
    return transcriber.transcribe_audio_file(
        audio_path,
        temperature=temperature,
        max_tokens=max_tokens,
        language=language
    )

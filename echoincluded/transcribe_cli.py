#!/usr/bin/env python3
"""
Command-line interface for transcribing audio files using vLLM Whisper Turbo.

Usage:
    python transcribe_cli.py /path/to/audio.wav
    python transcribe_cli.py /path/to/audio.wav --temperature 0.2 --max-tokens 128
    python transcribe_cli.py /path/to/audio.wav --language en
"""

import argparse
import time
import logging
import sys
import os

from transcribe import WhisperTranscriber, get_transcriber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using vLLM Whisper Turbo Quantized Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s audio.wav
    %(prog)s audio.wav --temperature 0.2 --max-tokens 128
    %(prog)s audio.wav --language en --output transcript.txt
    %(prog)s audio.wav --backend transformers  # Use CPU backend on Mac
    %(prog)s audio.wav --backend vllm --model neuralmagic/whisper-large-v3-turbo-quantized.w4a16
    %(prog)s audio.wav --backend transformers --transformers-model openai/whisper-small  # Use smaller model on CPU
        """
    )
    
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the audio file to transcribe"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = deterministic, default: 0.0)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=448,
        help="Maximum number of tokens to generate (default: 448)"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language hint (e.g., 'en', 'nl', 'fr')"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path to save transcription"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="neuralmagic/whisper-large-v3-turbo-quantized.w4a16",
        help="Model name (default: neuralmagic/whisper-large-v3-turbo-quantized.w4a16)"
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "vllm", "transformers"],
        help="Backend to use: 'auto', 'vllm', or 'transformers' (default: auto)"
    )
    
    parser.add_argument(
        "--transformers-model",
        type=str,
        default=None,
        help="Model to use with Transformers backend (default: openai/whisper-medium)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        logger.error(f"Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    try:
        logger.info(f"Loading model: {args.model} (backend: {args.backend})")
        
        # Set environment variable for Transformers model if specified
        if args.transformers_model and args.backend != "vllm":
            os.environ["TRANSFORMERS_WHISPER_MODEL"] = args.transformers_model
            logger.info(f"Using Transformers model: {args.transformers_model}")
        
        # Initialize transcriber with specified model and backend
        transcriber = WhisperTranscriber(model_name=args.model, backend=args.backend)
        transcriber.load_model()
        
        logger.info(f"Transcribing: {args.audio_file}")
        
        start_time = time.time()
        
        transcription = transcriber.transcribe_audio_file(
            args.audio_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            language=args.language
        )
        
        elapsed_time = time.time() - start_time
        
        # Output results
        print(f"\n{'='*50}")
        print(f"FILE: {args.audio_file}")
        print(f"MODEL: {args.model}")
        print(f"TEMPERATURE: {args.temperature}")
        print(f"MAX TOKENS: {args.max_tokens}")
        if args.language:
            print(f"LANGUAGE: {args.language}")
        print(f"PROCESSING TIME: {elapsed_time:.2f}s")
        print(f"{'='*50}")
        print(f"\nTRANSCRIPTION:")
        print("-" * 50)
        print(transcription)
        print("-" * 50)
        
        # Save to output file if specified
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                logger.info(f"Transcription saved to: {args.output}")
            except IOError as e:
                logger.error(f"Failed to save output: {e}")
        
    except KeyboardInterrupt:
        logger.info("Transcription cancelled")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
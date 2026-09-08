# echo included

use quatized whisper turbo model to have minimal self serve container

## usage

```bash
cd echoincluded
python main.py

# api
curl -X POST -F "file=@audio.wav" -F "temperature=0.0" -F "max_tokens=64" http://localhost:8000/transcribe

# cli
python transcribe_cli.py audio.wav --temperature 0.0 --max-tokens 64 --output transcript.txt

```

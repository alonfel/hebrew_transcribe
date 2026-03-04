# Hebrew Transcription Pipeline — Claude Context

## What this project does
Fully local, offline transcription pipeline for iPhone video recordings (.mov, .mp4).
Extracts audio → chunks → transcribes sequentially → outputs `transcript.txt` + `transcript.srt`.

## Stack
- **mlx-whisper** (Apple Silicon GPU/Neural Engine) — model auto-selected by language (see below)
- **ffmpeg** — audio extraction, silence removal, chunking

## Supported languages & models

| Language | Flag | Model (HuggingFace) |
|---|---|---|
| Hebrew (default) | `--language he` | `mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx` |
| English | `--language en` | `mlx-community/whisper-large-v3-turbo` |

The model is auto-selected based on `--language`. Override with `--model` if needed.

## Key design decisions
- mlx-whisper uses Apple Silicon GPU/Neural Engine natively — no multiprocessing needed
- Transcription is sequential; MLX handles parallelism internally on-chip
- Silence removal on by default (`silenceremove` ffmpeg filter, threshold -45dB, 1.5s gaps)
- Speedup off by default — opt-in via `--speedup FLOAT` (atempo filter)
- SRT timestamps use `ffprobe` per chunk for accuracy (silence removal makes chunk durations variable)

## File structure
```
transcribe.py       ← entire pipeline in one file
requirements.txt    ← mlx-whisper
venv/               ← Python virtualenv (gitignored)
```

## Output structure (next to input file)
```
{input_stem}_output/
  audio.wav          ← mono 16kHz, silence-removed
  chunks/
    chunk_000.wav
    chunk_000.json   ← resume checkpoint (written immediately after each chunk)
    ...
  transcript.txt
  transcript.srt
```

## Setup
```bash
brew install ffmpeg
pip install -r requirements.txt
```

## Running
```bash
# Activate venv first
source venv/bin/activate

# Hebrew (default) — uses ivrit-ai Hebrew fine-tune
python transcribe.py input.mov

# English — auto-selects whisper-large-v3-turbo (downloads ~800MB on first run)
python transcribe.py input.mov --language en

# Other options
python transcribe.py input.mov --speedup 1.1          # 10% faster audio
python transcribe.py input.mov --force                # re-run all steps
python transcribe.py input.mov --model mlx-community/whisper-large-v3-turbo --language en  # explicit model
```

## Resume behavior
Each step checks if its output already exists and skips if so:
- `audio.wav` → skip extract
- `chunk_*.wav` → skip chunking
- `chunk_NNN.json` → skip that individual chunk (written right after transcription)
- `transcript.txt` → skip merge

Use `--force` to bypass all resume checks.

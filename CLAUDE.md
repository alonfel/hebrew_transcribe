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
- Silence removal **off by default** — opt-in via `--remove-silence` (ffmpeg `silenceremove`, threshold -45dB, 1.5s gaps). When enabled, SRT timestamps will drift from the original video.
- Speedup off by default — opt-in via `--speedup FLOAT` (atempo filter)
- SRT timestamps use `ffprobe` per chunk for accuracy; chunk durations from the original-timeline audio ensure sync with the original video

## File structure
```
transcribe.py           ← entire pipeline in one file
requirements.txt        ← mlx-whisper
pytest.ini              ← pytest config; integration/ excluded from default run
tests/
  conftest.py           ← stubs mlx_whisper for unit tests (no real model needed)
  test_transcribe.py    ← 39 unit tests (fast, no external deps)
  integration/
    conftest.py         ← removes mlx_whisper stub so real model is used
    test_mlx_backend.py ← component test against real model + real audio chunk
venv/                   ← Python virtualenv (gitignored)
```

## Output structure (next to input file)
```
{input_stem}_output/
  audio.wav          ← mono 16kHz (silence removed only if --remove-silence was passed)
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
python transcribe.py input.mov --remove-silence   # faster but SRT timestamps won't match original video
```

## Resume behavior
Each step checks if its output already exists and skips if so:
- `audio.wav` → skip extract
- `chunk_*.wav` → skip chunking
- `chunk_NNN.json` → skip that individual chunk (written right after transcription)
- `transcript.txt` → skip merge

Use `--force` to bypass all resume checks.

## Testing

### Unit tests (fast, no external deps)
```bash
source venv/bin/activate
pytest tests/
```
- 39 tests, ~0.05s
- `mlx_whisper` is stubbed via `tests/conftest.py` — runs on any platform without a model
- `ffmpeg`/`ffprobe` calls are mocked via `unittest.mock.patch`

### Integration test (requires Apple Silicon + real audio chunk)
```bash
pytest tests/integration/
# or with a custom chunk:
INTEGRATION_CHUNK=/path/to/chunk.wav pytest tests/integration/
```
- Calls the real `mlx_whisper` model on a real `.wav` chunk (~13s on M-series)
- Validates that `MlxWhisperBackend` returns correctly typed, non-empty segments
- Skipped automatically if the chunk file doesn't exist

### Why two layers
Unit tests catch logic bugs instantly on any machine. The integration test catches model API contract changes (e.g., `mlx_whisper` output format changing) that mocks can't detect.

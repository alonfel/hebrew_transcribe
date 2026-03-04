# Hebrew Transcription Pipeline — Claude Context

## What this project does
Fully local, offline transcription pipeline for iPhone video recordings (.mov, .mp4).
Extracts audio → chunks → transcribes in parallel → outputs `transcript.txt` + `transcript.srt`.

## Stack
- **faster-whisper** (CTranslate2 backend) — default model `large-v3`, Hebrew (`he`)
- **ffmpeg** — audio extraction, silence removal, chunking
- **multiprocessing.Pool** — parallel transcription; model loaded once per worker via `initializer`

## Key design decisions
- `device="cpu"`, `compute_type="int8"` — faster-whisper has no MPS support; int8 is optimal on Apple Silicon
- Workers default to 2 — large-v3 is ~3GB RAM each; 2 workers = ~6GB, safe on M4
- Silence removal on by default (`silenceremove` ffmpeg filter, threshold -45dB, 1.5s gaps)
- Speedup off by default — opt-in via `--speedup FLOAT` (atempo filter)
- `_model_store`, `_init_worker`, `_transcribe_one` are **module-level** — required for macOS `spawn` pickling
- SRT timestamps use `ffprobe` per chunk for accuracy (silence removal makes chunk durations variable)

## File structure
```
transcribe.py       ← entire pipeline in one file
requirements.txt    ← faster-whisper>=1.0.0
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

python transcribe.py input.mov                        # basic
python transcribe.py input.mov --workers 4            # more parallelism
python transcribe.py input.mov --speedup 1.1          # 10% faster audio
python transcribe.py input.mov --force                # re-run all steps
python transcribe.py input.mov --model medium         # lighter model
python transcribe.py input.mov --language en          # non-Hebrew
```

## Resume behavior
Each step checks if its output already exists and skips if so:
- `audio.wav` → skip extract
- `chunk_*.wav` → skip chunking
- `chunk_NNN.json` → skip that individual chunk (written right after transcription)
- `transcript.txt` → skip merge

Use `--force` to bypass all resume checks.

## Stats output
Each step is wrapped in `StepTimer` — prints wall time, CPU time, and CPU utilization at end.
`RUSAGE_CHILDREN` is included so worker process CPU shows up correctly in transcription stats.

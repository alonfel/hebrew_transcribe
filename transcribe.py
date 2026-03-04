#!/usr/bin/env python3
"""
Hebrew transcription pipeline for iPhone videos.

Runs fully locally using mlx-whisper (Apple Silicon GPU) and ffmpeg.

Installation:
    brew install ffmpeg
    pip install -r requirements.txt

Usage:
    python transcribe.py input_video.mov
    python transcribe.py input_video.mov --speedup 1.1
    python transcribe.py input_video.mov --force          # re-run all steps
    python transcribe.py input_video.mov --output-dir ./out
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx_whisper


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    All settings for a single pipeline run.

    Attributes:
        input:          Path to the input video file.
        model:          HuggingFace repo ID for the mlx-whisper model.
        language:       BCP-47 language code passed to Whisper (e.g. "he", "en").
        output_dir:     Where to write outputs; defaults to {input_stem}_output next to input.
        speedup:        atempo speedup factor applied before chunking (None = disabled).
        chunk_duration: Length of each audio chunk in seconds (default 180 = 3 min).
        force:          Re-run all steps even if checkpoints already exist.
    """
    input: Path
    model: str = "mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx"
    language: str = "he"
    output_dir: Path | None = None
    speedup: float | None = None
    chunk_duration: int = 180
    force: bool = False


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

class TranscribeBackend(Protocol):
    """
    Protocol (structural subtype) for transcription backends.

    Any class that implements a matching ``transcribe`` method satisfies this
    interface — no inheritance required. Swap backends by passing a different
    implementation to ``run_pipeline``.
    """
    def transcribe(self, audio_path: str, language: str) -> list[tuple[float, float, str]]: ...


class MlxWhisperBackend:
    """
    Transcription via mlx-whisper (Apple Silicon GPU/Neural Engine).

    Requires macOS with Apple Silicon. Uses the MLX framework to run Whisper
    models accelerated by the GPU and Neural Engine.
    """

    def __init__(self, model_repo: str) -> None:
        """
        Args:
            model_repo: HuggingFace repo ID of an mlx-whisper compatible model,
                        e.g. "mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx".
        """
        self.model_repo = model_repo

    def transcribe(self, audio_path: str, language: str) -> list[tuple[float, float, str]]:
        result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=self.model_repo, language=language)
        return [(s["start"], s["end"], s["text"].strip()) for s in result["segments"]]


def _transcribe_one(chunk_path: str, chunk_idx: int, output_json: str, language: str, backend: TranscribeBackend) -> None:
    """
    Transcribe a single chunk and save the result to a JSON checkpoint file.
    Skips if the checkpoint already exists (resume support).
    """
    out = Path(output_json)

    if out.exists():
        logging.info("[TRANSCRIBE] chunk_%03d — skipping (checkpoint exists)", chunk_idx)
        return

    logging.info("[TRANSCRIBE] chunk_%03d — transcribing %s ...", chunk_idx, Path(chunk_path).name)

    segs = backend.transcribe(chunk_path, language)
    out.write_text(json.dumps(segs, ensure_ascii=False), encoding="utf-8")

    logging.info("[TRANSCRIBE] chunk_%03d — done (%d segments)", chunk_idx, len(segs))


# ---------------------------------------------------------------------------
# Pipeline step 1: Extract + optimize audio
# ---------------------------------------------------------------------------

def extract_audio(
    video_path: Path,
    output_dir: Path,
    speedup: float | None = None,
    force: bool = False,
) -> Path:
    """
    Extract audio from video to mono 16kHz WAV with silence removal.
    Applies optional speedup (atempo filter) if requested.

    Skips if audio.wav already exists (unless force=True).
    """
    audio_path = output_dir / "audio.wav"

    if audio_path.exists() and not force:
        logging.info("[EXTRACT] Skipping — audio.wav already exists")
        return audio_path

    logging.info("[EXTRACT] Starting: %s → %s", video_path, audio_path)

    # Build audio filter chain
    filters: list[str] = []
    if speedup and speedup != 1.0:
        logging.info("[EXTRACT] Speedup: %.2fx (atempo filter)", speedup)
        filters.append(f"atempo={speedup}")
    filters.append("silenceremove=stop_periods=-1:stop_duration=1.5:stop_threshold=-45dB")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ac", "1",           # mono
        "-ar", "16000",       # 16kHz
        "-af", ",".join(filters),
        str(audio_path),
    ]
    _run(cmd)
    logging.info("[EXTRACT] Done → %s", audio_path)
    return audio_path


# ---------------------------------------------------------------------------
# Pipeline step 2: Chunk audio
# ---------------------------------------------------------------------------

def chunk_audio(
    audio_path: Path,
    chunks_dir: Path,
    chunk_duration: int = 180,
    force: bool = False,
) -> list[Path]:
    """
    Split audio into fixed-duration WAV chunks using ffmpeg segmentation.

    Skips if chunks already exist (unless force=True).
    Returns sorted list of chunk paths.
    """
    chunks_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(chunks_dir.glob("chunk_*.wav"))

    if existing and not force:
        logging.info("[CHUNK] Skipping — %d chunks already exist", len(existing))
        return existing

    logging.info("[CHUNK] Starting: splitting %s into %ds chunks ...", audio_path.name, chunk_duration)

    pattern = str(chunks_dir / "chunk_%03d.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(chunk_duration),
        "-c", "copy",
        pattern,
    ]
    _run(cmd)

    chunks = sorted(chunks_dir.glob("chunk_*.wav"))
    logging.info("[CHUNK] Done — %d chunks created", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Pipeline step 3: Transcribe chunks sequentially
# ---------------------------------------------------------------------------

def transcribe_chunks(
    chunk_paths: list[Path],
    chunks_dir: Path,
    backend: TranscribeBackend,
    language: str = "he",
    force: bool = False,
) -> None:
    """
    Transcribe all chunks using the provided backend.

    Each completed chunk is saved to chunk_NNN.json immediately as a resume
    checkpoint. Skips chunks whose .json already exists (unless force=True).
    """
    if force:
        removed = 0
        for j in chunks_dir.glob("chunk_*.json"):
            j.unlink()
            removed += 1
        if removed:
            logging.info("[TRANSCRIBE] Force mode — removed %d existing checkpoints", removed)

    total = len(chunk_paths)
    already_done = sum(1 for p in chunk_paths if (chunks_dir / f"{p.stem}.json").exists())
    remaining = total - already_done

    logging.info(
        "[TRANSCRIBE] Starting — %d chunks total, %d already done, %d to transcribe",
        total, already_done, remaining,
    )

    if remaining == 0:
        logging.info("[TRANSCRIBE] All chunks already transcribed — skipping")
        return

    logging.info("[TRANSCRIBE] Using backend: %s", type(backend).__name__)

    # Pre-measure audio durations once (fast ffprobe metadata reads) for accurate ETA
    chunk_audio_durs = [_get_audio_duration(p) for p in chunk_paths]
    pending_audio = sum(
        d for p, d in zip(chunk_paths, chunk_audio_durs)
        if not (chunks_dir / f"{p.stem}.json").exists()
    )
    logging.info(
        "[TRANSCRIBE] Audio to process: %s | Rough estimate: ~%s (at 10x real-time)",
        _fmt_dur(pending_audio), _fmt_dur(pending_audio / 10.0),
    )

    rtf_samples: list[float] = []  # speed = audio_secs / wall_secs per processed chunk

    for idx, chunk_path in enumerate(chunk_paths):
        output_json = chunks_dir / f"{chunk_path.stem}.json"
        was_done = output_json.exists()

        t0 = time.perf_counter()
        _transcribe_one(str(chunk_path), idx, str(output_json), language, backend)
        elapsed = time.perf_counter() - t0

        if not was_done and elapsed > 0:
            rtf_samples.append(chunk_audio_durs[idx] / elapsed)

        left_audio = sum(
            chunk_audio_durs[j]
            for j in range(idx + 1, len(chunk_paths))
            if not (chunks_dir / f"{chunk_paths[j].stem}.json").exists()
        )
        if rtf_samples and left_audio > 0:
            avg_speed = sum(rtf_samples) / len(rtf_samples)
            logging.info(
                "[TRANSCRIBE] %d/%d done | speed %.1fx real-time | ETA ~%s",
                idx + 1, total, avg_speed, _fmt_dur(left_audio / avg_speed),
            )

    done_after = sum(1 for p in chunk_paths if (chunks_dir / f"{p.stem}.json").exists())
    logging.info("[TRANSCRIBE] Done — %d/%d chunks transcribed", done_after, total)

    if done_after < total:
        missing = [p.stem for p in chunk_paths if not (chunks_dir / f"{p.stem}.json").exists()]
        logging.warning("[TRANSCRIBE] Missing transcripts for: %s", ", ".join(missing))


# ---------------------------------------------------------------------------
# Pipeline step 4: Merge results
# ---------------------------------------------------------------------------

def merge_results(
    chunks_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> None:
    """
    Merge all chunk_NNN.json transcripts into transcript.txt and transcript.srt.

    Timestamps in SRT are computed by accumulating actual chunk durations
    (via ffprobe) so they remain accurate even for variable-length chunks.

    Skips if transcript.txt already exists (unless force=True).
    """
    txt_path = output_dir / "transcript.txt"
    srt_path = output_dir / "transcript.srt"

    if txt_path.exists() and not force:
        logging.info("[MERGE] Skipping — transcript.txt already exists")
        return

    json_files = sorted(chunks_dir.glob("chunk_*.json"))
    if not json_files:
        logging.error("[MERGE] No JSON transcript files found in %s — transcription may have failed", chunks_dir)
        sys.exit(1)

    logging.info("[MERGE] Starting — merging %d chunk transcripts ...", len(json_files))

    txt_lines: list[str] = []
    srt_blocks: list[str] = []
    srt_idx = 1
    offset = 0.0

    for json_file in json_files:
        stem = json_file.stem          # e.g. "chunk_000"
        wav_file = chunks_dir / f"{stem}.wav"
        segments: list = json.loads(json_file.read_text(encoding="utf-8"))

        for start, end, text in segments:
            if not text.strip():
                continue
            abs_start = offset + start
            abs_end = offset + end
            txt_lines.append(text)
            srt_blocks.append(
                f"{srt_idx}\n"
                f"{_fmt_srt_time(abs_start)} --> {_fmt_srt_time(abs_end)}\n"
                f"{text}\n"
            )
            srt_idx += 1

        # Accumulate actual chunk duration for accurate SRT timestamps
        if wav_file.exists():
            chunk_dur = _get_audio_duration(wav_file)
            logging.debug("[MERGE] %s: %.2fs, %d segments", stem, chunk_dur, len(segments))
            offset += chunk_dur
        else:
            logging.warning("[MERGE] WAV not found for %s — offset may be inaccurate", stem)

    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")

    logging.info("[MERGE] Done — %d segments total", srt_idx - 1)
    logging.info("[MERGE]   %s", txt_path)
    logging.info("[MERGE]   %s", srt_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig, backend: TranscribeBackend | None = None) -> None:
    """
    Orchestrate all pipeline steps: extract → chunk → transcribe → merge.

    Args:
        config:  A ``PipelineConfig`` instance with all run settings.
        backend: Transcription backend to use. Defaults to ``MlxWhisperBackend``
                 initialised from ``config.model``. Pass a custom backend to
                 swap the transcription engine (e.g. for testing or other models).
    """
    video_path = config.input.resolve()
    if not video_path.exists():
        logging.error("Input file not found: %s", video_path)
        sys.exit(1)

    if backend is None:
        backend = MlxWhisperBackend(config.model)

    output_dir = config.output_dir or video_path.parent / f"{video_path.stem}_output"
    chunks_dir = output_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 60)
    logging.info("[PIPELINE] Input:        %s", video_path)
    logging.info("[PIPELINE] Output dir:   %s", output_dir)
    logging.info("[PIPELINE] Model:        %s", config.model)
    logging.info("[PIPELINE] Language:     %s", config.language)
    logging.info("[PIPELINE] Chunk dur:    %ds", config.chunk_duration)
    logging.info("[PIPELINE] Speedup:      %s", f"{config.speedup}x" if config.speedup else "disabled")
    logging.info("[PIPELINE] Force re-run: %s", config.force)
    logging.info("=" * 60)

    # Step 1: Extract audio
    audio_path = extract_audio(
        video_path, output_dir,
        speedup=config.speedup,
        force=config.force,
    )

    # Step 2: Chunk audio
    chunk_paths = chunk_audio(
        audio_path, chunks_dir,
        chunk_duration=config.chunk_duration,
        force=config.force,
    )
    if not chunk_paths:
        logging.error("[PIPELINE] No chunks produced — aborting")
        sys.exit(1)

    # Step 3: Transcribe chunks
    transcribe_chunks(
        chunk_paths, chunks_dir,
        backend=backend,
        language=config.language,
        force=config.force,
    )

    # Step 4: Merge into final transcript
    merge_results(chunks_dir, output_dir, force=config.force)

    logging.info("=" * 60)
    logging.info("[PIPELINE] All done!")
    logging.info("[PIPELINE] Transcript: %s", output_dir / "transcript.txt")
    logging.info("[PIPELINE] Subtitles:  %s", output_dir / "transcript.srt")
    logging.info("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_dur(secs: float) -> str:
    """Format a duration in seconds as a concise human-readable string."""
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


def _run(cmd: list[str]) -> None:
    """Run a subprocess command; captures stderr and shows it only on failure."""
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logging.error("Command failed (exit %d):\n  cmd: %s\n  stderr: %s",
                      result.returncode, " ".join(cmd), result.stderr[-2000:])
        sys.exit(result.returncode)


def _get_audio_duration(wav_path: Path) -> float:
    """Return duration of a WAV file in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _fmt_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Local Hebrew transcription pipeline for iPhone videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python transcribe.py lecture.mov
  python transcribe.py lecture.mov --speedup 1.1
  python transcribe.py lecture.mov --output-dir ./output --force
  python transcribe.py lecture.mov --model mlx-community/whisper-large-v3-mlx --language en
        """,
    )
    parser.add_argument("input", help="Path to input video file (e.g. video.mov, video.mp4)")
    parser.add_argument("--output-dir", help="Output directory (default: {input_stem}_output next to input)")
    parser.add_argument("--model", default="mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx",
                        help="HuggingFace repo for mlx-whisper model (default: ivrit-ai-whisper-large-v3-turbo-mlx)")
    parser.add_argument("--language", default="he", help="Language code (default: he for Hebrew)")
    parser.add_argument("--speedup", type=float, default=None,
                        help="Audio speedup factor via atempo filter, e.g. 1.1 (default: disabled)")
    parser.add_argument("--chunk-duration", type=int, default=180,
                        help="Chunk duration in seconds (default: 180 = 3 minutes)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps, ignoring existing checkpoints")

    args = parser.parse_args()
    config = PipelineConfig(
        input=Path(args.input),
        model=args.model,
        language=args.language,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        speedup=args.speedup,
        chunk_duration=args.chunk_duration,
        force=args.force,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()

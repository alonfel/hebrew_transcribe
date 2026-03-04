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
import resource
import subprocess
import sys
import time
from pathlib import Path

import mlx_whisper


# ---------------------------------------------------------------------------
# Step timer — wall time, CPU time, CPU utilisation
# ---------------------------------------------------------------------------

class StepTimer:
    """Context manager that measures wall time and CPU time for a pipeline step."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.wall: float = 0.0
        self.cpu: float = 0.0

    def __enter__(self) -> "StepTimer":
        self._wall_start = time.perf_counter()
        ru = resource.getrusage(resource.RUSAGE_SELF)
        rc = resource.getrusage(resource.RUSAGE_CHILDREN)
        self._cpu_start = ru.ru_utime + ru.ru_stime + rc.ru_utime + rc.ru_stime
        return self

    def __exit__(self, *_) -> None:
        self.wall = time.perf_counter() - self._wall_start
        ru = resource.getrusage(resource.RUSAGE_SELF)
        rc = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.cpu = (ru.ru_utime + ru.ru_stime + rc.ru_utime + rc.ru_stime) - self._cpu_start


def _fmt_dur(secs: float) -> str:
    """Format a duration in seconds as a concise human-readable string."""
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


def _transcribe_one(chunk_path: str, chunk_idx: int, output_json: str, language: str, model_repo: str) -> None:
    """
    Transcribe a single chunk and save the result to a JSON checkpoint file.
    Skips if the checkpoint already exists (resume support).
    """
    out = Path(output_json)

    if out.exists():
        logging.info("[TRANSCRIBE] chunk_%03d — skipping (checkpoint exists)", chunk_idx)
        return

    logging.info("[TRANSCRIBE] chunk_%03d — transcribing %s ...", chunk_idx, Path(chunk_path).name)

    result = mlx_whisper.transcribe(chunk_path, path_or_hf_repo=model_repo, language=language)

    segs = [(s["start"], s["end"], s["text"].strip()) for s in result["segments"]]
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
# Pipeline step 3: Transcribe chunks sequentially (MLX uses GPU internally)
# ---------------------------------------------------------------------------

def transcribe_chunks(
    chunk_paths: list[Path],
    chunks_dir: Path,
    model_name: str = "mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx",
    language: str = "he",
    force: bool = False,
) -> None:
    """
    Transcribe all chunks using mlx-whisper (Apple Silicon GPU/Neural Engine).

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

    # Pre-measure actual audio durations for accurate ETA (fast ffprobe metadata reads)
    chunk_audio_durs = [_get_audio_duration(p) for p in chunk_paths]

    remaining_audio = sum(
        d for p, d in zip(chunk_paths, chunk_audio_durs)
        if not (chunks_dir / f"{p.stem}.json").exists()
    )
    rough_est = remaining_audio / 10.0  # assume 10x RTF until we have real data
    logging.info(
        "[TRANSCRIBE] Audio to transcribe: %s | Rough estimate: ~%s (at 10x real-time)",
        _fmt_dur(remaining_audio), _fmt_dur(rough_est),
    )
    logging.info("[TRANSCRIBE] Using model: %s", model_name)

    rtf_samples: list[float] = []  # audio_sec / process_sec per actually-transcribed chunk

    for idx, chunk_path in enumerate(chunk_paths):
        output_json = chunks_dir / f"{chunk_path.stem}.json"
        was_done = output_json.exists()

        t0 = time.perf_counter()
        _transcribe_one(str(chunk_path), idx, str(output_json), language, model_name)
        elapsed = time.perf_counter() - t0

        if not was_done and elapsed > 0:
            rtf_samples.append(chunk_audio_durs[idx] / elapsed)

        # ETA: remaining audio / avg real-time factor
        left_audio = sum(
            chunk_audio_durs[j]
            for j in range(idx + 1, len(chunk_paths))
            if not (chunks_dir / f"{chunk_paths[j].stem}.json").exists()
        )
        if rtf_samples and left_audio > 0:
            avg_rtf = sum(rtf_samples) / len(rtf_samples)
            logging.info(
                "[TRANSCRIBE] Progress: %d/%d | RTF %.1fx | ETA ~%s",
                idx + 1, total, avg_rtf, _fmt_dur(left_audio / avg_rtf),
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

def run_pipeline(args: argparse.Namespace) -> None:
    """Orchestrate all pipeline steps: extract → chunk → transcribe → merge."""
    video_path = Path(args.input).resolve()
    if not video_path.exists():
        logging.error("Input file not found: %s", video_path)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else video_path.parent / f"{video_path.stem}_output"
    chunks_dir = output_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 60)
    logging.info("[PIPELINE] Input:        %s", video_path)
    logging.info("[PIPELINE] Output dir:   %s", output_dir)
    logging.info("[PIPELINE] Model:        %s", args.model)
    logging.info("[PIPELINE] Language:     %s", args.language)
    logging.info("[PIPELINE] Chunk dur:    %ds", args.chunk_duration)
    logging.info("[PIPELINE] Speedup:      %s", f"{args.speedup}x" if args.speedup else "disabled")
    logging.info("[PIPELINE] Force re-run: %s", args.force)
    logging.info("=" * 60)

    timers: list[StepTimer] = []

    # Step 1: Extract audio
    with StepTimer("extract_audio") as t:
        audio_path = extract_audio(
            video_path, output_dir,
            speedup=args.speedup,
            force=args.force,
        )
    timers.append(t)

    # Step 2: Chunk audio
    with StepTimer("chunk_audio") as t:
        chunk_paths = chunk_audio(
            audio_path, chunks_dir,
            chunk_duration=args.chunk_duration,
            force=args.force,
        )
    timers.append(t)
    if not chunk_paths:
        logging.error("[PIPELINE] No chunks produced — aborting")
        sys.exit(1)

    # Step 3: Transcribe chunks
    with StepTimer("transcribe") as t:
        transcribe_chunks(
            chunk_paths, chunks_dir,
            model_name=args.model,
            language=args.language,
            force=args.force,
        )
    timers.append(t)

    # Step 4: Merge into final transcript
    with StepTimer("merge_results") as t:
        merge_results(chunks_dir, output_dir, force=args.force)
    timers.append(t)

    # ── Summary stats ────────────────────────────────────────────────────────
    audio_dur = _get_audio_duration(audio_path)
    transcribe_wall = timers[2].wall
    rtf = audio_dur / transcribe_wall if transcribe_wall > 0 else 0.0

    # Peak RSS: macOS reports bytes, Linux reports KB
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = raw_rss / (1024 * 1024) if sys.platform == "darwin" else raw_rss / 1024

    total_wall = sum(t.wall for t in timers)
    total_cpu  = sum(t.cpu  for t in timers)

    sep = "-" * 62
    logging.info(sep)
    logging.info("[STATS] %-16s  %8s  %8s  %6s  %s", "Step", "Wall", "CPU", "CPU%", "Note")
    logging.info("[STATS] %s", sep)
    for i, timer in enumerate(timers):
        cpu_pct = (timer.cpu / timer.wall * 100) if timer.wall > 0 else 0.0
        note = f"{rtf:.1f}x real-time" if i == 2 else ""
        logging.info("[STATS] %-16s  %8s  %8s  %5.1f%%  %s",
                     timer.name, _fmt_dur(timer.wall), _fmt_dur(timer.cpu), cpu_pct, note)
    logging.info("[STATS] %s", sep)
    total_pct = (total_cpu / total_wall * 100) if total_wall > 0 else 0.0
    logging.info("[STATS] %-16s  %8s  %8s  %5.1f%%",
                 "TOTAL", _fmt_dur(total_wall), _fmt_dur(total_cpu), total_pct)
    logging.info("[STATS] Peak RAM: %.0f MB  |  Audio: %s  |  Segments: see transcript",
                 peak_mb, _fmt_dur(audio_dur))
    logging.info(sep)

    logging.info("=" * 60)
    logging.info("[PIPELINE] All done!")
    logging.info("[PIPELINE] Transcript: %s", output_dir / "transcript.txt")
    logging.info("[PIPELINE] Subtitles:  %s", output_dir / "transcript.srt")
    logging.info("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    run_pipeline(args)


if __name__ == "__main__":
    main()

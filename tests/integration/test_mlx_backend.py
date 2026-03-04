"""
Component integration test for MlxWhisperBackend.

Requires:
  - Apple Silicon Mac (mlx_whisper only runs on Metal)
  - A real .wav chunk on disk (set CHUNK below or pass via env)
  - ~1.5 GB model download on first run

Run:
  pytest tests/integration/
"""
import difflib
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import transcribe

MODEL_REPO = "mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx"

# Override via env: INTEGRATION_CHUNK=/path/to/chunk.wav pytest tests/integration/
CHUNK = os.environ.get(
    "INTEGRATION_CHUNK",
    "/Users/alon/Documents/deluit_hr_output/chunks/chunk_000.wav",
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_FILE = FIXTURES_DIR / "chunk_000_transcript.txt"

# Similarity threshold: same model + same audio should be deterministic (1.0),
# but 0.95 tolerates minor punctuation changes from model updates.
SIMILARITY_THRESHOLD = 0.95


def _normalize(text: str) -> str:
    """Strip Unicode RTL/LTR embedding marks and normalize whitespace."""
    for mark in ("\u202b", "\u202a", "\u200f", "\u200e", "\u202c"):
        text = text.replace(mark, "")
    return " ".join(text.split())


def _run_backend() -> tuple[list, float]:
    if not Path(CHUNK).exists():
        pytest.skip(f"Chunk not found: {CHUNK}")
    backend = transcribe.MlxWhisperBackend(MODEL_REPO)
    t0 = time.perf_counter()
    result = backend.transcribe(CHUNK, language="he")
    return result, time.perf_counter() - t0


@pytest.mark.integration
def test_mlx_backend_returns_segments():
    """MlxWhisperBackend transcribes a real chunk and returns non-empty segments."""
    result, elapsed = _run_backend()

    print(f"\nWall time: {elapsed:.1f}s  |  Segments: {len(result)}")
    for start, end, text in result:
        print(f"  [{start:6.2f} → {end:6.2f}]  {text}")

    assert isinstance(result, list), "result must be a list"
    assert len(result) > 0, "expected at least one segment"
    start, end, text = result[0]
    assert isinstance(start, float)
    assert isinstance(end, float)
    assert end > start
    assert isinstance(text, str) and len(text) > 0


@pytest.mark.integration
def test_mlx_backend_transcript_matches_golden():
    """Transcript text must be ≥95% similar to the stored golden reference.

    First run: writes the golden file and skips (re-run to validate).
    Subsequent runs: compare against the golden file.
    To regenerate: delete fixtures/chunk_000_transcript.txt and re-run.
    """
    result, elapsed = _run_backend()

    actual = _normalize(" ".join(text for _, _, text in result))
    print(f"\nWall time: {elapsed:.1f}s  |  Characters: {len(actual)}")

    if not GOLDEN_FILE.exists():
        FIXTURES_DIR.mkdir(exist_ok=True)
        GOLDEN_FILE.write_text(actual, encoding="utf-8")
        pytest.skip(f"Golden file created: {GOLDEN_FILE} — re-run to validate")

    golden = _normalize(GOLDEN_FILE.read_text(encoding="utf-8"))
    ratio = difflib.SequenceMatcher(None, golden, actual).ratio()
    print(f"Similarity: {ratio:.2%}")

    assert ratio >= SIMILARITY_THRESHOLD, (
        f"Transcript similarity {ratio:.2%} < {SIMILARITY_THRESHOLD:.0%}\n"
        f"--- golden\n{golden}\n+++ actual\n{actual}"
    )

"""
Component integration test for MlxWhisperBackend.

Requires:
  - Apple Silicon Mac (mlx_whisper only runs on Metal)
  - A real .wav chunk on disk (set CHUNK below or pass via env)
  - ~1.5 GB model download on first run

Run:
  pytest tests/integration/
"""
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


@pytest.mark.integration
def test_mlx_backend_returns_segments():
    """MlxWhisperBackend transcribes a real chunk and returns non-empty segments."""
    if not Path(CHUNK).exists():
        pytest.skip(f"Chunk not found: {CHUNK}")

    backend = transcribe.MlxWhisperBackend(MODEL_REPO)

    t0 = time.perf_counter()
    result = backend.transcribe(CHUNK, language="he")
    elapsed = time.perf_counter() - t0

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

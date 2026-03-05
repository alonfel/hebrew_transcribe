"""
runner.py — subprocess abstraction for the transcription pipeline.

Yields log lines from transcribe.py one at a time so any frontend
(Streamlit now, Flask/SSE later) can consume the same stream.
"""

import subprocess
import sys
from pathlib import Path
from typing import Generator, Union

_TRANSCRIBE_SCRIPT = Path(__file__).parent.parent / "transcribe.py"


def build_command(
    file_path: str,
    language: str = "he",
    remove_silence: bool = False,
    force: bool = False,
    speedup: float | None = None,
) -> list[str]:
    cmd = [sys.executable, str(_TRANSCRIBE_SCRIPT), file_path, "--language", language]
    if remove_silence:
        cmd.append("--remove-silence")
    if force:
        cmd.append("--force")
    if speedup and speedup > 1.0:
        cmd.extend(["--speedup", str(speedup)])
    return cmd


def stream_transcription(
    file_path: str,
    language: str = "he",
    remove_silence: bool = False,
    force: bool = False,
    speedup: float | None = None,
) -> Generator[Union[str, int], None, None]:
    """
    Yield stdout log lines (str) from transcribe.py, then the return code (int).

    Callers distinguish the final sentinel by type:
        for item in stream_transcription(...):
            if isinstance(item, int):
                returncode = item; break
            # item is a log line string
    """
    cmd = build_command(file_path, language=language, remove_silence=remove_silence,
                        force=force, speedup=speedup)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        yield line.rstrip()
    proc.wait()
    yield proc.returncode

"""
app.py — Streamlit frontend for the Hebrew Transcription Pipeline.

Run with:
    streamlit run ui/app.py

Calls transcribe.py as a subprocess via runner.py (untouched pipeline).
"""

import re
import subprocess
from pathlib import Path

import streamlit as st

import runner

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Hebrew Transcription", layout="centered")
st.title("Hebrew Transcription Pipeline")

# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

col_input, col_browse = st.columns([5, 1])

with col_input:
    typed_path = st.text_input(
        "Video file",
        value=st.session_state.get("file_path", ""),
        placeholder="/path/to/video.mov",
        label_visibility="collapsed",
    )
    if typed_path:
        st.session_state.file_path = typed_path

with col_browse:
    st.write("")  # vertical align
    if st.button("Browse"):
        result = subprocess.run(
            [
                "osascript", "-e",
                'POSIX path of (choose file of type {"mov", "mp4", "m4v", "mkv"}'
                ' with prompt "Select video file:")',
            ],
            capture_output=True,
            text=True,
        )
        chosen = result.stdout.strip()
        if chosen:
            st.session_state.file_path = chosen
            st.rerun()

file_path = st.session_state.get("file_path", "")
file_exists = bool(file_path) and Path(file_path).is_file()

if file_path and not file_exists:
    st.warning(f"File not found: {file_path}")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

with st.expander("Options"):
    language = st.selectbox("Language", ["Hebrew (he)", "English (en)"], index=0)
    lang_code = "he" if language.startswith("Hebrew") else "en"

    col_a, col_b = st.columns(2)
    with col_a:
        remove_silence = st.checkbox(
            "Remove silence",
            help="Faster, but SRT timestamps won't match original video",
        )
        force = st.checkbox("Force re-run", help="Ignore existing checkpoints")
    with col_b:
        speedup = st.slider("Speedup", min_value=1.0, max_value=2.0, value=1.0, step=0.05,
                            help="Audio speedup via atempo filter")

# ---------------------------------------------------------------------------
# Transcribe button + live streaming
# ---------------------------------------------------------------------------

if st.button("Transcribe", type="primary", disabled=not file_exists):
    progress_bar = st.progress(0.0, text="Starting…")
    log_area = st.empty()
    lines: list[str] = []
    returncode = -1

    for item in runner.stream_transcription(
        file_path,
        language=lang_code,
        remove_silence=remove_silence,
        force=force,
        speedup=speedup if speedup > 1.0 else None,
    ):
        if isinstance(item, int):
            returncode = item
            break

        lines.append(item)
        log_area.code("\n".join(lines[-80:]))

        # Update progress bar from TRANSCRIBE chunk counter
        m = re.search(r"\[TRANSCRIBE\] (\d+)/(\d+) done", item)
        if m:
            current, total = int(m[1]), int(m[2])
            progress_bar.progress(current / total, text=f"Transcribing chunk {current}/{total}…")
        elif "[EXTRACT]" in item:
            progress_bar.progress(0.0, text="Extracting audio…")
        elif "[CHUNK]" in item:
            progress_bar.progress(0.05, text="Chunking audio…")
        elif "[MERGE]" in item:
            progress_bar.progress(1.0, text="Merging results…")

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------

    if returncode == 0:
        progress_bar.progress(1.0, text="Done!")
        st.success("Transcription complete!")

        output_dir = Path(file_path).parent / f"{Path(file_path).stem}_output"
        txt_path = output_dir / "transcript.txt"
        srt_path = output_dir / "transcript.srt"

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if txt_path.exists() and st.button("Open transcript.txt"):
                subprocess.run(["open", str(txt_path)])
        with btn_col2:
            if srt_path.exists() and st.button("Open transcript.srt"):
                subprocess.run(["open", str(srt_path)])

        if txt_path.exists():
            preview = txt_path.read_text(encoding="utf-8")
            st.text_area("Transcript preview", preview[:3000], height=300)
            if len(preview) > 3000:
                st.caption(f"Showing first 3000 of {len(preview)} characters. Full file: `{txt_path}`")
            else:
                st.caption(f"Full transcript at: `{txt_path}`")
    else:
        progress_bar.empty()
        st.error(f"Transcription failed (exit code {returncode}). See logs above.")

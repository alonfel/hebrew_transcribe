"""
Unit tests for transcribe.py

Focuses on:
  - Pure formatting functions
  - merge_results (offset math, SRT format, skip logic)
  - extract_audio (ffmpeg command construction, resume skip)
  - chunk_audio (ffmpeg command construction, resume skip)
  - _transcribe_one (checkpoint skip, JSON output, model call args)
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import transcribe


# ---------------------------------------------------------------------------
# Pure function: _fmt_srt_time
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pure function: _fmt_dur
# ---------------------------------------------------------------------------

class TestFmtDur:
    def test_sub_minute(self):
        assert transcribe._fmt_dur(45.0) == "45.0s"

    def test_exactly_one_minute(self):
        assert transcribe._fmt_dur(60.0) == "1m00s"

    def test_minutes_and_seconds(self):
        assert transcribe._fmt_dur(150.0) == "2m30s"


# ---------------------------------------------------------------------------
# Pure function: _fmt_srt_time
# ---------------------------------------------------------------------------

class TestFmtSrtTime:
    def test_zero(self):
        assert transcribe._fmt_srt_time(0.0) == "00:00:00,000"

    def test_one_hour_wrong_expectation(self):
        # BUG: wrong expected value — 3600s = 01:00:00,000, not 01:01:00,000
        assert transcribe._fmt_srt_time(3600.0) == "01:00:00,000"

    def test_sub_minute(self):
        assert transcribe._fmt_srt_time(5.5) == "00:00:05,500"

    def test_minutes(self):
        assert transcribe._fmt_srt_time(90.0) == "00:01:30,000"

    def test_hours(self):
        assert transcribe._fmt_srt_time(3661.0) == "01:01:01,000"

    def test_milliseconds(self):
        assert transcribe._fmt_srt_time(1.123) == "00:00:01,123"


# ---------------------------------------------------------------------------
# merge_results
# ---------------------------------------------------------------------------

class TestMergeResults:
    def _write_chunk(self, chunks_dir: Path, stem: str, segments: list) -> None:
        (chunks_dir / f"{stem}.json").write_text(json.dumps(segments), encoding="utf-8")
        (chunks_dir / f"{stem}.wav").touch()

    def test_single_chunk_timestamps_unchanged(self, tmp_path):
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()

        self._write_chunk(chunks_dir, "chunk_000", [
            [0.0, 1.5, "שלום"],
            [1.5, 3.0, "עולם"],
        ])

        with patch("transcribe._get_audio_duration", return_value=180.0):
            transcribe.merge_results(chunks_dir, tmp_path)

        txt = (tmp_path / "transcript.txt").read_text(encoding="utf-8")
        assert txt == "שלום\nעולם"

        srt = (tmp_path / "transcript.srt").read_text(encoding="utf-8")
        assert "00:00:00,000 --> 00:00:01,500" in srt
        assert "00:00:01,500 --> 00:00:03,000" in srt

    def test_two_chunks_offset_applied(self, tmp_path):
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()

        self._write_chunk(chunks_dir, "chunk_000", [[0.0, 1.0, "ראשון"]])
        self._write_chunk(chunks_dir, "chunk_001", [[0.0, 2.0, "שני"]])

        with patch("transcribe._get_audio_duration", return_value=180.0):
            transcribe.merge_results(chunks_dir, tmp_path)

        srt = (tmp_path / "transcript.srt").read_text(encoding="utf-8")
        # chunk_001 starts at offset=180s → segment [0.0, 2.0] → [180.0, 182.0]
        assert "00:03:00,000 --> 00:03:02,000" in srt

    def test_empty_text_segments_skipped(self, tmp_path):
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()

        self._write_chunk(chunks_dir, "chunk_000", [
            [0.0, 1.0, "טקסט"],
            [1.0, 2.0, ""],       # empty
            [2.0, 3.0, "   "],    # whitespace only
        ])

        with patch("transcribe._get_audio_duration", return_value=180.0):
            transcribe.merge_results(chunks_dir, tmp_path)

        txt = (tmp_path / "transcript.txt").read_text(encoding="utf-8")
        assert txt == "טקסט"

        srt = (tmp_path / "transcript.srt").read_text(encoding="utf-8")
        assert srt.count("\n1\n") == 0  # no second index
        assert "2\n" not in srt

    def test_srt_index_sequential_across_chunks(self, tmp_path):
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()

        self._write_chunk(chunks_dir, "chunk_000", [[0.0, 1.0, "א"], [1.0, 2.0, "ב"]])
        self._write_chunk(chunks_dir, "chunk_001", [[0.0, 1.0, "ג"]])

        with patch("transcribe._get_audio_duration", return_value=180.0):
            transcribe.merge_results(chunks_dir, tmp_path)

        srt = (tmp_path / "transcript.srt").read_text(encoding="utf-8")
        lines = srt.split("\n")
        indices = [line for line in lines if line.strip().isdigit()]
        assert indices == ["1", "2", "3"]

    def test_skips_if_transcript_exists(self, tmp_path):
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        (tmp_path / "transcript.txt").write_text("existing", encoding="utf-8")

        with patch("transcribe._get_audio_duration") as mock_dur:
            transcribe.merge_results(chunks_dir, tmp_path)
            mock_dur.assert_not_called()

        # file unchanged
        assert (tmp_path / "transcript.txt").read_text() == "existing"

    def test_force_overwrites_existing(self, tmp_path):
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        (tmp_path / "transcript.txt").write_text("old content", encoding="utf-8")
        self._write_chunk(chunks_dir, "chunk_000", [[0.0, 1.0, "חדש"]])

        with patch("transcribe._get_audio_duration", return_value=10.0):
            transcribe.merge_results(chunks_dir, tmp_path, force=True)

        txt = (tmp_path / "transcript.txt").read_text(encoding="utf-8")
        assert txt == "חדש"

    def test_srt_block_exact_format(self, tmp_path):
        # Guards against index/timestamp/text line ordering being reshuffled
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        self._write_chunk(chunks_dir, "chunk_000", [[0.5, 2.0, "שלום"]])

        with patch("transcribe._get_audio_duration", return_value=180.0):
            transcribe.merge_results(chunks_dir, tmp_path)

        srt = (tmp_path / "transcript.srt").read_text(encoding="utf-8")
        lines = srt.strip().split("\n")
        assert lines[0] == "1"
        assert lines[1] == "00:00:00,500 --> 00:00:02,000"
        assert lines[2] == "שלום"

    def test_variable_chunk_durations_offset(self, tmp_path):
        # Both chunks use the SAME mock value in test_two_chunks_offset_applied.
        # This test uses different durations to catch bugs where offset uses
        # wrong index or wrong accumulation order.
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        self._write_chunk(chunks_dir, "chunk_000", [[0.0, 1.0, "א"]])
        self._write_chunk(chunks_dir, "chunk_001", [[0.0, 2.0, "ב"]])

        with patch("transcribe._get_audio_duration", side_effect=[95.5, 85.0]):
            transcribe.merge_results(chunks_dir, tmp_path)

        srt = (tmp_path / "transcript.srt").read_text(encoding="utf-8")
        # chunk_001 offset = 95.5s → [95.5, 97.5] → 00:01:35,500 --> 00:01:37,500
        assert "00:01:35,500 --> 00:01:37,500" in srt

    def test_txt_newline_joins_segments(self, tmp_path):
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        self._write_chunk(chunks_dir, "chunk_000", [
            [0.0, 1.0, "שורה ראשונה"],
            [1.0, 2.0, "שורה שנייה"],
            [2.0, 3.0, "שורה שלישית"],
        ])

        with patch("transcribe._get_audio_duration", return_value=180.0):
            transcribe.merge_results(chunks_dir, tmp_path)

        txt = (tmp_path / "transcript.txt").read_text(encoding="utf-8")
        assert txt == "שורה ראשונה\nשורה שנייה\nשורה שלישית"


# ---------------------------------------------------------------------------
# extract_audio
# ---------------------------------------------------------------------------

class TestExtractAudio:
    def test_skips_if_audio_exists(self, tmp_path):
        (tmp_path / "audio.wav").touch()

        with patch("transcribe._run") as mock_run:
            result = transcribe.extract_audio(Path("input.mov"), tmp_path)

        mock_run.assert_not_called()
        assert result == tmp_path / "audio.wav"

    def test_force_runs_even_if_exists(self, tmp_path):
        (tmp_path / "audio.wav").touch()

        with patch("transcribe._run") as mock_run:
            transcribe.extract_audio(Path("input.mov"), tmp_path, force=True)

        mock_run.assert_called_once()

    def test_cmd_has_mono_and_16khz(self, tmp_path):
        with patch("transcribe._run") as mock_run:
            transcribe.extract_audio(Path("input.mov"), tmp_path)

        cmd = mock_run.call_args[0][0]
        assert cmd[cmd.index("-ac") + 1] == "1"
        assert cmd[cmd.index("-ar") + 1] == "16000"

    def test_no_speedup_no_silence_removal_omits_af(self, tmp_path):
        with patch("transcribe._run") as mock_run:
            transcribe.extract_audio(Path("input.mov"), tmp_path)

        cmd = mock_run.call_args[0][0]
        assert "-af" not in cmd

    def test_silence_removal_disabled_by_default(self, tmp_path):
        with patch("transcribe._run") as mock_run:
            transcribe.extract_audio(Path("input.mov"), tmp_path)

        cmd = mock_run.call_args[0][0]
        assert "silenceremove" not in " ".join(cmd)

    def test_silence_removal_enabled(self, tmp_path):
        with patch("transcribe._run") as mock_run:
            transcribe.extract_audio(Path("input.mov"), tmp_path, remove_silence=True)

        cmd = mock_run.call_args[0][0]
        af = cmd[cmd.index("-af") + 1]
        assert "silenceremove" in af

    def test_speedup_prepends_atempo(self, tmp_path):
        with patch("transcribe._run") as mock_run:
            transcribe.extract_audio(Path("input.mov"), tmp_path, speedup=1.2)

        cmd = mock_run.call_args[0][0]
        af = cmd[cmd.index("-af") + 1]
        assert af.startswith("atempo=1.2")
        assert "silenceremove" not in af

    def test_speedup_1_0_not_applied(self, tmp_path):
        # speedup=1.0 means no change; the condition `speedup and speedup != 1.0`
        # must suppress atempo so we don't pass an identity filter
        with patch("transcribe._run") as mock_run:
            transcribe.extract_audio(Path("input.mov"), tmp_path, speedup=1.0)

        cmd = mock_run.call_args[0][0]
        assert "-af" not in cmd

    def test_speedup_with_silence_removal(self, tmp_path):
        with patch("transcribe._run") as mock_run:
            transcribe.extract_audio(Path("input.mov"), tmp_path, speedup=1.2, remove_silence=True)

        cmd = mock_run.call_args[0][0]
        af = cmd[cmd.index("-af") + 1]
        assert af.startswith("atempo=1.2")
        assert "silenceremove" in af


# ---------------------------------------------------------------------------
# chunk_audio
# ---------------------------------------------------------------------------

class TestChunkAudio:
    def test_skips_if_chunks_exist(self, tmp_path):
        (tmp_path / "chunk_000.wav").touch()

        with patch("transcribe._run") as mock_run:
            result = transcribe.chunk_audio(Path("audio.wav"), tmp_path)

        mock_run.assert_not_called()
        assert len(result) == 1

    def test_force_reruns(self, tmp_path):
        (tmp_path / "chunk_000.wav").touch()

        with patch("transcribe._run") as mock_run:
            transcribe.chunk_audio(Path("audio.wav"), tmp_path, force=True)

        mock_run.assert_called_once()

    def test_segment_time_in_cmd(self, tmp_path):
        with patch("transcribe._run") as mock_run:
            transcribe.chunk_audio(Path("audio.wav"), tmp_path, chunk_duration=120)

        cmd = mock_run.call_args[0][0]
        assert "-segment_time" in cmd
        assert cmd[cmd.index("-segment_time") + 1] == "120"

    def test_copy_codec_and_segment_format_in_cmd(self, tmp_path):
        # -c copy avoids re-encoding; -f segment enables the segmenter muxer.
        # Both are critical — losing either silently changes behaviour.
        with patch("transcribe._run") as mock_run:
            transcribe.chunk_audio(Path("audio.wav"), tmp_path)

        cmd = mock_run.call_args[0][0]
        assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "segment"
        assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy"

    def test_output_pattern_in_cmd(self, tmp_path):
        with patch("transcribe._run") as mock_run:
            transcribe.chunk_audio(Path("audio.wav"), tmp_path)

        cmd = mock_run.call_args[0][0]
        assert str(tmp_path / "chunk_%03d.wav") in cmd


# ---------------------------------------------------------------------------
# transcribe_chunks
# ---------------------------------------------------------------------------

class TestTranscribeChunks:
    def test_all_done_skips_backend(self, tmp_path):
        # If every chunk already has a .json checkpoint, the backend should
        # never be called — guards against the all-done early-exit path breaking.
        chunk_wav = tmp_path / "chunk_000.wav"
        chunk_wav.touch()
        (tmp_path / "chunk_000.json").write_text("[]", encoding="utf-8")

        mock_backend = MagicMock()
        with patch("transcribe._get_audio_duration", return_value=10.0):
            transcribe.transcribe_chunks([chunk_wav], tmp_path, backend=mock_backend)

        mock_backend.transcribe.assert_not_called()

    def test_force_clears_checkpoints_and_retranscribes(self, tmp_path):
        # force=True must delete existing .json files so _transcribe_one runs
        # from scratch rather than skipping.
        chunk_wav = tmp_path / "chunk_000.wav"
        chunk_wav.touch()
        (tmp_path / "chunk_000.json").write_text("[]", encoding="utf-8")

        mock_backend = MagicMock()
        mock_backend.transcribe.return_value = []
        with patch("transcribe._get_audio_duration", return_value=10.0):
            transcribe.transcribe_chunks([chunk_wav], tmp_path, backend=mock_backend, force=True)

        mock_backend.transcribe.assert_called_once()


# ---------------------------------------------------------------------------
# _transcribe_one
# ---------------------------------------------------------------------------

MODEL_REPO = "mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx"


class TestTranscribeOne:
    def test_skips_if_json_exists(self, tmp_path):
        output_json = tmp_path / "chunk_000.json"
        output_json.touch()

        mock_backend = MagicMock()
        transcribe._transcribe_one(
            str(tmp_path / "chunk_000.wav"), 0, str(output_json), "he", mock_backend
        )
        mock_backend.transcribe.assert_not_called()

    def test_writes_json_checkpoint(self, tmp_path):
        chunk_wav = tmp_path / "chunk_000.wav"
        chunk_wav.touch()
        output_json = tmp_path / "chunk_000.json"

        mock_backend = MagicMock()
        mock_backend.transcribe.return_value = [(0.0, 1.5, "שלום")]
        transcribe._transcribe_one(str(chunk_wav), 0, str(output_json), "he", mock_backend)

        assert output_json.exists()
        data = json.loads(output_json.read_text(encoding="utf-8"))
        assert data == [[0.0, 1.5, "שלום"]]

    def test_calls_backend_with_correct_args(self, tmp_path):
        chunk_wav = tmp_path / "chunk_000.wav"
        chunk_wav.touch()
        output_json = tmp_path / "chunk_000.json"

        mock_backend = MagicMock()
        mock_backend.transcribe.return_value = []
        transcribe._transcribe_one(str(chunk_wav), 0, str(output_json), "he", mock_backend)

        mock_backend.transcribe.assert_called_once_with(str(chunk_wav), "he")


class TestMlxWhisperBackend:
    def test_calls_mlx_with_correct_args(self):
        with patch("mlx_whisper.transcribe", return_value={"segments": []}) as mock_t:
            backend = transcribe.MlxWhisperBackend(MODEL_REPO)
            backend.transcribe("audio.wav", "he")

        mock_t.assert_called_once_with("audio.wav", path_or_hf_repo=MODEL_REPO, language="he")

    def test_strips_text_and_returns_tuples(self):
        raw = {"segments": [{"start": 0.0, "end": 1.5, "text": " שלום "}]}
        with patch("mlx_whisper.transcribe", return_value=raw):
            backend = transcribe.MlxWhisperBackend(MODEL_REPO)
            result = backend.transcribe("audio.wav", "he")

        assert result == [(0.0, 1.5, "שלום")]

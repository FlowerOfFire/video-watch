"""Offline unit tests for video-watch core utilities and CLI entrypoint."""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from video_watch.bili_subtitles import _track_rank
from video_watch.frames import _is_blank_dark, format_ts, parse_ts
from video_watch.transcript import (
    _pick_ytdlp_original_track,
    _sidecar_priority,
    is_bilibili_target,
    parse_srt_or_vtt,
    resolve_transcript,
)
from video_watch.watch import _is_remote_source, build_parser, cmd_run, write_brief


def test_timestamp_roundtrip() -> None:
    assert parse_ts("42.5") == 42.5
    assert parse_ts("04:12") == 252.0
    assert parse_ts("01:02:03") == 3723.0
    assert format_ts(252) == "04:12"
    assert format_ts(252, compact=True) == "04m12s"
    assert format_ts(3723) == "01:02:03"


def test_parse_srt_and_dedup_consecutive() -> None:
    srt = (
        "1\n00:00:01,000 --> 00:00:03,000\nHello <i>world</i>\n\n"
        "2\n00:00:03,000 --> 00:00:05,500\nHello world\n\n"
        "3\n00:00:06,000 --> 00:00:08,000\nNext line\n"
    )
    segs = parse_srt_or_vtt(srt)
    assert segs == [
        {"start": 1.0, "end": 5.5, "text": "Hello world"},
        {"start": 6.0, "end": 8.0, "text": "Next line"},
    ]


def test_bili_track_rank_prefers_original_then_manual() -> None:
    tracks = [
        {"lan": "zh-CN", "ai_type": 1, "type": 1},  # AI translated
        {"lan": "en-US", "ai_type": 0, "type": 1},  # AI original
        {"lan": "en-US", "ai_type": 0, "type": 0},  # human original
    ]
    chosen = min(tracks, key=_track_rank)
    assert chosen == {"lan": "en-US", "ai_type": 0, "type": 0}


def test_ytdlp_original_track_priority() -> None:
    meta = {
        "language": "ja",
        "subtitles": {"zh-Hans": [{}], "ja": [{}]},
        "automatic_captions": {"en": [{}], "ja-orig": [{}]},
    }
    assert _pick_ytdlp_original_track(meta) == ("ja", False, True)

    meta_auto = {
        "language": "",
        "subtitles": {},
        "automatic_captions": {"zh-Hans": [{}], "en-orig": [{}]},
    }
    assert _pick_ytdlp_original_track(meta_auto) == ("en-orig", True, True)


def test_sidecar_priority_prefers_matching_non_translated(tmp_path) -> None:
    trans = tmp_path / "demo.zh.自动翻译.srt"
    orig = tmp_path / "demo.english.srt"
    assert _sidecar_priority(orig, "eng") < _sidecar_priority(trans, "eng")


def test_is_blank_dark() -> None:
    mask = np.ones((16, 16), dtype=bool)
    assert _is_blank_dark(np.zeros((16, 16), dtype=np.uint8), mask)
    assert not _is_blank_dark(np.full((16, 16), 128, dtype=np.uint8), mask)


def test_is_bilibili_target() -> None:
    assert is_bilibili_target("BV1VDp16LEsr")
    assert is_bilibili_target("https://www.bilibili.com/video/BV1VDp16LEsr")
    assert not is_bilibili_target("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_cli_parser_defaults_and_help() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "demo.mp4"])
    assert args.detector == "auto"
    assert args.dedup == "global"

    proc = subprocess.run(
        [sys.executable, "-m", "video_watch", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "run" in proc.stdout
    assert "grab" in proc.stdout


def test_resolve_transcript_graceful_when_qwen_missing(tmp_path, monkeypatch) -> None:
    import pytest

    media = tmp_path / "silent.mp4"
    media.write_bytes(b"")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr("video_watch.transcript.resolve_qwen_python", lambda: None)

    segs, engine = resolve_transcript(media_path=media, out_dir=out_dir)
    assert segs == []
    assert engine == "unavailable (no qwen3-asr)"

    with pytest.raises(FileNotFoundError):
        resolve_transcript(media_path=media, out_dir=out_dir, force_asr=True)


def test_missing_local_file_raises_instead_of_downloading(tmp_path) -> None:
    import pytest

    assert _is_remote_source("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert _is_remote_source("BV1VDp16LEsr")
    assert not _is_remote_source(str(tmp_path / "missing.mp4"))

    parser = build_parser()
    args = parser.parse_args(["run", str(tmp_path / "missing.mp4"), "--out", str(tmp_path / "out")])
    with pytest.raises(FileNotFoundError):
        cmd_run(args)


def test_write_brief_avoids_dangling_transcript_link(tmp_path) -> None:
    stats = {
        "detector": "auto",
        "duration_sec": 4.0,
        "resolution": "640x360",
        "sampled_frames": 10,
        "scene_runs": 1,
        "unique_images": 1,
        "revisit_refs": 0,
    }
    brief_no_md = write_brief(tmp_path, tmp_path / "demo.mp4", stats, [], [], [], "skipped")
    assert "[transcript.md](transcript.md)" not in brief_no_md.read_text(encoding="utf-8")

    (tmp_path / "transcript.md").write_text("# t\n", encoding="utf-8")
    brief_with_md = write_brief(tmp_path, tmp_path / "demo.mp4", stats, [], [], [], "sidecar")
    assert "[transcript.md](transcript.md)" in brief_with_md.read_text(encoding="utf-8")

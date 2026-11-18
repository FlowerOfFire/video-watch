---
name: video-watch
description: >-
  Extract deduplicated keyframes, labeled contact-sheet grids, and original-language
  subtitles or local Qwen3-ASR transcripts for videos. Use when the user shares a
  Bilibili BV/URL, YouTube/video URL, or local video file and wants to inspect,
  summarize, analyze, or locate moments inside the video without manually writing
  ffmpeg scripts.
license: MIT
compatibility: >-
  Requires Python 3.11+, ffmpeg, and ffprobe. Optional: yutto and bilibili-cli
  for Bilibili, yt-dlp for YouTube/other URLs, and a local qwen3-asr venv for
  speech-to-text fallback.
metadata:
  author: HYFlower
  version: "0.1.0"
---

# Video Watch

Turn a local video file, Bilibili BV/URL, or YouTube URL into a token-efficient analysis bundle (`brief.md`, `transcript.md`, `grids/*.jpg`, `keyframes/*.jpg`, and `manifest.json`).

Do not hand-roll `ffmpeg` frame extraction loops when analyzing a video. Run `video-watch` first and inspect outputs in the progressive order below.

## Command Invocation

Prefer the installed CLI; if `video-watch` is not on `PATH`, install it or invoke via `uv --project` pointing to this skill's root directory:

```bash
video-watch run "<source>" --out "<out_dir>"
# Fallback when not yet on PATH:
uv --project "<skill_root>" run video-watch run "<source>" --out "<out_dir>"
# Or install globally once:
uv tool install -e "<skill_root>"
```

Always pass an explicit `--out` inside a task-specific or temporary directory so runs stay isolated and easy to clean up.

## Token-Saving Reading Order

1. **Read `brief.md` first**: Contains duration, frame statistics, transcript source, visual-speech timeline, and paths to all grids. Read `transcript.md` alongside it if full verbatim speech is needed.
2. **Inspect `grids/*.jpg` next**: Contact sheets (default `3x3`, labeled `#01 00:08`) let you scan the entire visual flow in 1–3 image reads.
3. **Inspect `keyframes/*.jpg` or `grab` selectively**: Open individual frames only where small text, UI, or code in a grid slot needs higher resolution. To inspect an exact timestamp outside extracted keyframes, reuse the downloaded video under `<out_dir>/source/`:

```bash
video-watch grab "<video_path>" 04:12 --out "<out_dir>/zoom_04m12s.jpg"
```

## Choosing Flags by Video Type

Default `--detector auto` runs PySceneDetect `ContentDetector(27)` plus `StatsManager` per-span static `proxy` refinement and dark transition filtering, handling live action, mixed edits, and screen recordings out of the box.

- **Hard-subbed videos / corner webcams**: Ignore subtitle bars or static overlays so changing captions do not trigger duplicate frames:
  - `--ignore-bottom 0.12`
  - `--ignore-rect 0.75,0.68,1.0,1.0` (repeatable normalized `x1,y1,x2,y2`)
- **Transcript / ASR control**:
  - Original-language subtitles are preferred automatically before AI translations.
  - Pass `--no-asr` when only visual inspection or fast subtitle check is needed.
  - Pass `--force-asr` when subtitles are inaccurate or out of sync.
  - Pass `--language Chinese` / `--context "<hotwords>"` to guide Qwen3-ASR.
- **Detector override**: Only switch away from `auto` when necessary (`--detector scene` for pure shot cuts, `--detector proxy` for continuous 1fps local-change UI/slide scans).

## Full Reference

See [CLI & Output Reference](references/cli-reference.md) for all flags, environment variables (`VIDEO_WATCH_QWEN_PYTHON`, `VIDEO_WATCH_QWEN_MODEL`, `VIDEO_WATCH_BILI_PYTHON`), and output file schemas.

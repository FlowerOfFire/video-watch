# CLI & Output Reference

## Subcommands

### `video-watch run <source>`

Run the full keyframe deduplication + transcript alignment pipeline.

| Option | Default | Purpose |
| --- | --- | --- |
| `source` | required | Local video path, Bilibili BV ID / URL, or YouTube / yt-dlp URL |
| `--out` | `./runs/<slug>_<ts>` | Output directory (always pass explicitly in automated workflows) |
| `--quality` | `64` | `yutto` quality code for Bilibili (`64`=720P, `80`=1080P) |
| `--detector` | `auto` | `auto` (PySceneDetect + per-span static proxy refinement), `scene`, or `proxy` |
| `--scene-algo` | `content` | PySceneDetect algorithm (`content` or `adaptive`) |
| `--scene-threshold` | `None` | Custom threshold (`content` default `27.0`, `adaptive` default `3.0`) |
| `--min-scene-sec` | `0.6` | Minimum shot duration for PySceneDetect |
| `--fps` | `1.0` | Proxy sampling rate (frames/sec) |
| `--dedup` | `global` | Deduplication scope (`global`, `consecutive`, or `none`) |
| `--hash-thresh` | `8` | Perceptual hash (`pHash`) distance threshold |
| `--mad-thresh` | `6.0` | Global grayscale MAD threshold |
| `--tile-thresh` | `14.0` | Max 4×4 tile grayscale MAD threshold |
| `--ignore-bottom` | `0.0` | Ignore bottom fraction (e.g. `0.12` for burned-in subtitles) |
| `--ignore-rect` | `None` | Repeatable normalized `x1,y1,x2,y2` box to mask out overlays/webcams |
| `--settle-sec` | `1.0` | Delay after a scene cut before capturing representative frame |
| `--width` | `1280` | Max width for exported `keyframes/*.jpg` |
| `--grid` | `3x3` | Contact sheet grid geometry (`COLSxROWS`) |
| `--no-grid` | `false` | Skip contact-sheet grid generation |
| `--sub` | `None` | Explicit `.srt` or `.vtt` subtitle path |
| `--force-asr` | `false` | Ignore subtitles and force local `Qwen3-ASR-1.7B` |
| `--no-asr` | `false` | Skip ASR fallback when no subtitles are found |
| `--language` | `None` | ASR language hint (e.g. `Chinese`, `English`) |
| `--context` | `""` | ASR hotword / context prompt |

### `video-watch grab <video> <timestamp>`

Extract a single high-resolution frame from an already downloaded/local video:

- `timestamp`: `MM:SS`, `HH:MM:SS`, or float seconds (e.g. `04:12` or `252.5`).
- `--out`: Output image path (defaults next to `<video>`).
- `--width`: Max output width (`1920` by default; `0` preserves native resolution).

## Transcript Resolution Priority

1. `--sub <file>` (unless `--force-asr`)
2. Platform native / original-language subtitle:
   - Bilibili (`bili_subtitles.py` via `bilibili-cli` env, fallback `bili` CLI): prefers `ai_type=0` (original language) and `type=0` (manual/CC) over AI-translated tracks.
   - Other URLs (`yt-dlp`): matches video `language` / `*-orig` tracks before translated/fallback tracks.
3. Sidecar `.srt` / `.vtt` beside local media (matching audio stream language tag via `ffprobe`, avoiding translated markers).
4. Local chunked `Qwen3-ASR-1.7B` (`qwen_asr_chunked.py`) splitting at lowest-RMS pauses (`12s–28s`) unless `--no-asr` is set.

## Environment Variables

- `VIDEO_WATCH_QWEN_PYTHON`: Path to Python executable containing `qwen-asr` and `torch`. Auto-discovers `./qwen3-asr/.venv` or ancestor `qwen3-asr/.venv` directories when unset.
- `VIDEO_WATCH_QWEN_MODEL`: Local path or Hugging Face ID for `Qwen3-ASR-1.7B`.
- `VIDEO_WATCH_BILI_PYTHON`: Path to `bilibili-cli` Python executable (`uv tool dir` auto-checked when unset).

## Output Bundle Layout

```text
<out_dir>/
├── brief.md          # Main entry: video metadata + timeline aligned with frame IDs & grids
├── transcript.md     # Timestamped text transcript
├── transcript.json   # Structured transcript segments + engine label
├── manifest.json     # Frame stats, transcript segments, and grid paths
├── grids/            # Labeled COLSxROWS JPG contact sheets
├── keyframes/        # Deduplicated high-res JPG keyframes
└── source/           # Downloaded source video (when source is a URL/BV)
```

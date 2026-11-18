"""Main CLI for deduplicated keyframe extraction + Qwen3-ASR alignment for AI agents."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from .frames import (
    Keyframe,
    extract_deduplicated_keyframes,
    format_ts,
    grab_frame,
    parse_ts,
)
from .transcript import is_bilibili_target, resolve_transcript

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".flv", ".avi")


def _slugify(source: str) -> str:
    p = Path(source)
    if p.exists():
        stem = p.stem
    else:
        m = re.search(r"(BV[0-9A-Za-z]+)", source)
        if m:
            stem = m.group(1)
        else:
            stem = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", source).strip("_")[:48] or "video"
    clean = re.sub(r'[<>:"/\\|?*]+', "_", stem).strip("._ ")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{clean[:40]}_{stamp}"


def _find_downloaded_video(source_dir: Path) -> Path | None:
    candidates = [
        p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def _is_remote_source(source: str) -> bool:
    """Return True if source is a URL or Bilibili BV/av target rather than a local file path."""
    text = source.strip()
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        return True
    return is_bilibili_target(text)


def download_video(target: str, source_dir: Path, quality: str = "64") -> Path:
    """Download Bilibili (via yutto) or YouTube/other (via yt-dlp) into source_dir."""
    source_dir.mkdir(parents=True, exist_ok=True)
    if is_bilibili_target(target):
        url = f"https://www.bilibili.com/video/{target}" if target.lower().startswith("bv") else target
        cmd = [
            "yutto",
            url,
            "-d",
            str(source_dir),
            "-q",
            quality,
            "--no-danmaku",
            "--no-cover",
        ]
    else:
        cmd = [
            "yt-dlp",
            "-f",
            "bv*[height<=1080]+ba/b[height<=1080]/b",
            "--merge-output-format",
            "mp4",
            "-o",
            str(source_dir / "%(title).80s.%(ext)s"),
            target,
        ]
    print(f"Downloading video via: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Video download failed with code {proc.returncode}")
    video = _find_downloaded_video(source_dir)
    if not video:
        raise RuntimeError(f"Download finished but no video file found in {source_dir}")
    return video


def _parse_rect(text: str) -> tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Rect must be x1,y1,x2,y2 in normalized [0..1] coordinates")
    return parts[0], parts[1], parts[2], parts[3]


def _parse_grid(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)x(\d+)", text.strip().lower())
    if not m:
        raise argparse.ArgumentTypeError("Grid must look like 3x3 or 2x2")
    return int(m.group(1)), int(m.group(2))


def _align_transcript_to_keyframes(
    keyframes: list[Keyframe],
    segments: list[dict],
) -> list[tuple[Keyframe, list[dict]]]:
    """Attach each speech segment to the keyframe covering its midpoint timestamp."""
    buckets: list[list[dict]] = [[] for _ in keyframes]
    if not keyframes:
        return []

    for seg in segments:
        mid = 0.5 * (seg["start"] + seg["end"])
        placed = False
        for i, kf in enumerate(keyframes):
            if kf.start_sec <= mid <= kf.end_sec:
                buckets[i].append(seg)
                placed = True
                break
        if not placed:
            nearest = min(range(len(keyframes)), key=lambda i: abs(keyframes[i].rep_sec - mid))
            buckets[nearest].append(seg)
    return list(zip(keyframes, buckets))


def write_brief(
    out_dir: Path,
    video_path: Path,
    frame_stats: dict,
    keyframes: list[Keyframe],
    grids: list[Path],
    segments: list[dict],
    transcript_engine: str,
) -> Path:
    """Write an agent-friendly brief.md pairing grid overview, deduplicated frames, and transcript."""
    has_transcript_md = (out_dir / "transcript.md").exists()
    step1 = (
        "1. 先阅读本文件和 [transcript.md](transcript.md) 掌握语言主干与时间线。"
        if has_transcript_md
        else "1. 先阅读本文件掌握画面时间线（本次未生成 `transcript.md`）。"
    )
    lines: list[str] = [
        f"# Video Watch Brief: {video_path.name}\n",
        f"- **视频路径**：`{video_path}`",
        f"- **时长 / 分辨率**：{format_ts(frame_stats['duration_sec'])} ({frame_stats['duration_sec']}s) · {frame_stats['resolution']}",
        f"- **检测器 / 抽帧统计**：`{frame_stats.get('detector', 'proxy')}` · 扫描 {frame_stats['sampled_frames']} 帧 → 切分 **{frame_stats['scene_runs']}** 个片段 → 输出 **{frame_stats['unique_images']}** 张独立关键帧"
        + (f"（含 {frame_stats['revisit_refs']} 次回切复用前帧）" if frame_stats["revisit_refs"] else ""),
        f"- **文稿来源**：`{transcript_engine}`（共 {len(segments)} 段）",
        "",
        "## 一、AI 看图读取顺序（省 Token 推荐）",
        step1,
    ]
    if grids:
        lines.append("2. 用 `read_file` 依次读取下方 **宫格总览图**（每张图左上角标注 `#编号 时间戳`）：")
        for g in grids:
            lines.append(f"   - `{g}`")
        lines.append("3. 需要看清某格里的代码/小字时，再按对应编号用 `read_file` 打开 `keyframes/` 下的高清单帧，或执行：")
    else:
        lines.append("2. 直接用 `read_file` 查看下方 `keyframes/` 下的高清去重关键帧。")
    lines.append(
        f'   `video-watch grab "{video_path}" <时间戳如 04:12> --out "{out_dir / "zoom_04m12s.jpg"}"`'
    )
    lines.append("")
    lines.append("## 二、画面 × 语音对齐时间线")

    aligned = _align_transcript_to_keyframes(keyframes, segments)
    for kf, segs in aligned:
        span = f"{format_ts(kf.start_sec)} - {format_ts(kf.end_sec)}"
        if kf.same_as:
            lines.append(f"\n### #{kf.index:02d} [{span}]（画面同 `{kf.same_as}`，未重复存图）")
        else:
            slot_info = f" · `{kf.grid_slot}`" if kf.grid_slot else ""
            lines.append(f"\n### #{kf.index:02d} [{span}]（代表帧 `{format_ts(kf.rep_sec)}`{slot_info}）")
            lines.append(f"- 图片：`{kf.image_path}`")
        if segs:
            for seg in segs:
                lines.append(f"- [{format_ts(seg['start'])}] {seg['text']}")
        else:
            lines.append("- *（此片段无语音或未启用转写）*")

    brief_path = out_dir / "brief.md"
    brief_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return brief_path


def cmd_run(args: argparse.Namespace) -> int:
    src = args.source
    src_path = Path(src)
    is_local = src_path.exists()
    is_remote = not is_local and _is_remote_source(src)
    if not is_local and not is_remote:
        raise FileNotFoundError(
            f"Local video file not found (and source is not a recognized URL or BV ID): {src}"
        )

    out_dir = Path(args.out) if args.out else Path.cwd() / "runs" / _slugify(src)
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_local:
        video_path = src_path.resolve()
    else:
        video_path = download_video(src, out_dir / "source", quality=args.quality)

    cols, rows = _parse_grid(args.grid)
    keyframes, grids, stats = extract_deduplicated_keyframes(
        video_path=video_path,
        out_dir=out_dir,
        detector=args.detector,
        scene_algo=args.scene_algo,
        scene_threshold=args.scene_threshold,
        min_scene_sec=args.min_scene_sec,
        sample_fps=args.fps,
        hash_thresh=args.hash_thresh,
        mad_thresh=args.mad_thresh,
        tile_thresh=args.tile_thresh,
        dedup_mode=args.dedup,
        ignore_bottom=args.ignore_bottom,
        ignore_rects=args.ignore_rect or [],
        settle_sec=args.settle_sec,
        out_width=args.width,
        grid_cols=cols,
        grid_rows=rows,
        make_grids=not args.no_grid,
    )

    segments, engine = resolve_transcript(
        media_path=video_path,
        out_dir=out_dir,
        source_target=src if is_remote else None,
        sub_file=Path(args.sub) if args.sub else None,
        force_asr=args.force_asr,
        skip_asr=args.no_asr,
        language=args.language,
        context=args.context,
    )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "frames": stats,
                "transcript_engine": engine,
                "transcript_segments": segments,
                "grids": [str(g) for g in grids],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    brief_path = write_brief(out_dir, video_path, stats, keyframes, grids, segments, engine)

    print(
        f"\n[DONE] [{stats['detector']}] {stats['sampled_frames']} scanned -> {stats['scene_runs']} scenes -> {stats['unique_images']} unique frames"
        + (f" ({stats['revisit_refs']} revisit refs)" if stats["revisit_refs"] else "")
        + f", {len(grids)} grid(s), transcript={engine} ({len(segments)} segs)"
    )
    print(f"Brief:    {brief_path}")
    print(f"Manifest: {manifest_path}")
    return 0


def cmd_grab(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    ts = parse_ts(args.timestamp)
    out_path = Path(args.out) if args.out else video_path.parent / f"grab_{format_ts(ts, compact=True)}.jpg"
    saved = grab_frame(video_path, ts, out_path, width=args.width)
    print(saved)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deduplicated video frame extractor + Qwen3-ASR bridge for AI agents.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Process a local video, BV id, or video URL.")
    run_p.add_argument("source", help="Local video path, Bilibili BV/URL, or YouTube URL")
    run_p.add_argument("--out", default=None, help="Output directory (defaults to runs/<slug>)")
    run_p.add_argument("--quality", default="64", help="yutto quality code for Bilibili (64=720P, 80=1080P)")
    run_p.add_argument("--detector", choices=["auto", "scene", "proxy"], default="auto", help="Scene change detector: auto (PySceneDetect + per-span static proxy refinement, default), scene (PySceneDetect only), or proxy (ffmpeg pHash+MAD)")
    run_p.add_argument("--scene-algo", choices=["content", "adaptive"], default="content", help="PySceneDetect algorithm when --detector scene (default: content)")
    run_p.add_argument("--scene-threshold", type=float, default=None, help="PySceneDetect threshold (content default: 27.0, adaptive default: 3.0)")
    run_p.add_argument("--min-scene-sec", type=float, default=0.6, help="Minimum shot length in seconds for PySceneDetect (default: 0.6)")
    run_p.add_argument("--fps", type=float, default=1.0, help="Proxy sampling rate in frames/sec (default: 1.0)")
    run_p.add_argument("--dedup", choices=["global", "consecutive", "none"], default="global")
    run_p.add_argument("--hash-thresh", type=int, default=8, help="Perceptual hash distance threshold (default: 8)")
    run_p.add_argument("--mad-thresh", type=float, default=6.0, help="Global grayscale MAD threshold (default: 6.0)")
    run_p.add_argument("--tile-thresh", type=float, default=14.0, help="Max 4x4 tile MAD threshold (default: 14.0)")
    run_p.add_argument("--ignore-bottom", type=float, default=0.0, help="Ignore bottom fraction (e.g. 0.12 for subtitles)")
    run_p.add_argument(
        "--ignore-rect",
        type=_parse_rect,
        action="append",
        help="Ignore normalized box x1,y1,x2,y2 (e.g. 0.72,0.68,1.0,1.0 for corner webcam); repeatable",
    )
    run_p.add_argument("--settle-sec", type=float, default=1.0, help="Delay after scene change before picking frame")
    run_p.add_argument("--width", type=int, default=1280, help="Output keyframe max width (default: 1280)")
    run_p.add_argument("--grid", default="3x3", help="Contact sheet grid layout COLSxROWS (default: 3x3)")
    run_p.add_argument("--no-grid", action="store_true", help="Skip generating contact-sheet grids")
    run_p.add_argument("--sub", default=None, help="Explicit .srt/.vtt subtitle path")
    run_p.add_argument("--force-asr", action="store_true", help="Ignore subtitles and force local Qwen3-ASR")
    run_p.add_argument("--no-asr", action="store_true", help="Skip ASR fallback when no subtitles exist")
    run_p.add_argument("--language", default=None, help="ASR language hint (Chinese / English)")
    run_p.add_argument("--context", default="", help="ASR hotwords / context hint")
    run_p.set_defaults(func=cmd_run)

    grab_p = sub.add_parser("grab", help="Extract a single HD frame at a specific timestamp.")
    grab_p.add_argument("video", help="Path to local video file")
    grab_p.add_argument("timestamp", help="Timestamp like 04:12 or 252.5")
    grab_p.add_argument("--out", default=None, help="Output image path")
    grab_p.add_argument("--width", type=int, default=1920, help="Max width (default: 1920, 0 for native)")
    grab_p.set_defaults(func=cmd_grab)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

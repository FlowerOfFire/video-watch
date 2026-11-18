"""Keyframe extraction with duplicate filtering and labeled contact-sheet generation."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import imagehash
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Keyframe:
    """Metadata for a deduplicated keyframe or revisit reference."""

    index: int
    id: str
    start_sec: float
    end_sec: float
    rep_sec: float
    duration_sec: float
    image_path: str | None
    same_as: str | None = None
    grid_path: str | None = None
    grid_slot: str | None = None


def format_ts(seconds: float, compact: bool = False) -> str:
    """Format seconds as MM:SS (or HH:MM:SS) or filename-safe 00m00s."""
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if compact:
        return f"{h:02d}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_ts(value: str) -> float:
    """Parse seconds or HH:MM:SS / MM:SS timestamp strings into float seconds."""
    text = value.strip()
    if ":" not in text:
        return float(text)
    parts = [float(p) for p in text.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unsupported timestamp format: {value}")


def probe_video(video_path: Path) -> dict:
    """Inspect video resolution, fps, and duration with ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {proc.stderr[-500:]}")
    info = json.loads(proc.stdout)
    vstream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if not vstream:
        raise RuntimeError(f"No video stream found in {video_path}")
    duration = float(vstream.get("duration") or info.get("format", {}).get("duration") or 0.0)
    width = int(vstream["width"])
    height = int(vstream["height"])
    return {"duration": duration, "width": width, "height": height}


def _even_height(src_w: int, src_h: int, dst_w: int) -> int:
    h = max(2, int(round(dst_w * src_h / src_w)))
    return h if h % 2 == 0 else h + 1


def _stream_proxy_frames(
    video_path: Path,
    sample_fps: float,
    proxy_w: int,
    proxy_h: int,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> Iterable[tuple[float, np.ndarray]]:
    """Yield (timestamp_sec, RGB uint8 array) by piping scaled frames from ffmpeg."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    base_sec = 0.0
    if start_sec is not None and start_sec > 0:
        base_sec = float(start_sec)
        cmd.extend(["-ss", f"{base_sec:.3f}"])
    if end_sec is not None and end_sec > base_sec:
        cmd.extend(["-t", f"{end_sec - base_sec:.3f}"])
    cmd.extend([
        "-i",
        str(video_path),
        "-vf",
        f"fps={sample_fps},scale={proxy_w}:{proxy_h}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    frame_bytes = proxy_w * proxy_h * 3
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, dtype=np.uint8).reshape((proxy_h, proxy_w, 3)).copy()
            yield base_sec + idx / sample_fps, arr
            idx += 1
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace")
        proc.stderr.close()
        rc = proc.wait()
        if rc != 0 and idx == 0 and start_sec is None:
            raise RuntimeError(f"ffmpeg frame stream failed: {stderr[-500:]}")


def _grab_proxy_rgb(video_path: Path, timestamp_sec: float, proxy_w: int, proxy_h: int) -> np.ndarray:
    """Decode a single scaled RGB frame at timestamp_sec into memory for signature checks."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{max(0.0, timestamp_sec):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={proxy_w}:{proxy_h}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    expected = proxy_w * proxy_h * 3
    if proc.returncode != 0 or len(proc.stdout) < expected:
        raise RuntimeError(
            f"ffmpeg proxy grab failed at {timestamp_sec:.2f}s: "
            f"{proc.stderr.decode('utf-8', errors='replace')[-400:]}"
        )
    return np.frombuffer(proc.stdout[:expected], dtype=np.uint8).reshape((proxy_h, proxy_w, 3)).copy()


def _detect_scenes_pyscenedetect(
    video_path: Path,
    duration_sec: float,
    algo: str = "content",
    threshold: float | None = None,
    min_scene_sec: float = 0.6,
) -> tuple[list[tuple[float, float, float, float, float]], int, float]:
    """Detect shot boundaries and per-span content_val statistics via PySceneDetect."""
    from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, StatsManager, open_video

    video = open_video(str(video_path))
    fps = float(video.frame_rate or 30.0)
    min_len_frames = max(1, int(round(min_scene_sec * fps)))

    if algo == "adaptive":
        eff_thresh = 3.0 if threshold is None else float(threshold)
        det = AdaptiveDetector(adaptive_threshold=eff_thresh, min_scene_len=min_len_frames)
    elif algo == "content":
        eff_thresh = 27.0 if threshold is None else float(threshold)
        det = ContentDetector(threshold=eff_thresh, min_scene_len=min_len_frames)
    else:
        raise ValueError(f"Unsupported scene_algo: {algo}")

    stats_mgr = StatsManager()
    manager = SceneManager(stats_mgr)
    manager.auto_downscale = True
    manager.add_detector(det)
    manager.detect_scenes(video=video, show_progress=False)
    raw_scenes = manager.get_scene_list(start_in_scene=True)

    total_frames = int(video.duration.frame_num) if video.duration else 0
    if not raw_scenes:
        span_frames = [(0, max(1, total_frames), 0.0, max(0.1, duration_sec))]
    else:
        span_frames = [
            (s.frame_num, e.frame_num, float(s.seconds), float(e.seconds))
            for s, e in raw_scenes
            if float(e.seconds) > float(s.seconds)
        ] or [(0, max(1, total_frames), 0.0, max(0.1, duration_sec))]

    span_stats: list[tuple[float, float, float, float, float]] = []
    for sf, ef, s_sec, e_sec in span_frames:
        cvs = []
        for f_idx in range(sf + 1, ef):
            m = stats_mgr.get_metrics(f_idx, ["content_val"])
            if m and m[0] is not None:
                cvs.append(float(m[0]))
        if cvs:
            arr = np.asarray(cvs, dtype=np.float32)
            med_cv = float(np.median(arr))
            pct_lt1 = float((arr < 1.0).mean())
            max_cv = float(arr.max())
        else:
            med_cv, pct_lt1, max_cv = 0.0, 0.0, 0.0
        span_stats.append((s_sec, e_sec, med_cv, pct_lt1, max_cv))

    return span_stats, total_frames, eff_thresh


def _build_mask(
    height: int,
    width: int,
    ignore_bottom: float,
    ignore_rects: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """Build boolean mask of pixels to compare (True = active, False = ignored overlay)."""
    mask = np.ones((height, width), dtype=bool)
    if ignore_bottom > 0:
        cutoff = max(1, int(round(height * (1.0 - min(0.5, ignore_bottom)))))
        mask[cutoff:, :] = False
    for x1, y1, x2, y2 in ignore_rects:
        ix1 = max(0, min(width, int(round(x1 * width))))
        ix2 = max(0, min(width, int(round(x2 * width))))
        iy1 = max(0, min(height, int(round(y1 * height))))
        iy2 = max(0, min(height, int(round(y2 * height))))
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = False
    if not mask.any():
        mask[:, :] = True
    return mask


def _compute_signature(rgb: np.ndarray, mask: np.ndarray) -> tuple[imagehash.ImageHash, np.ndarray]:
    """Compute perceptual hash and masked grayscale array for fast similarity checks."""
    masked_rgb = rgb.copy()
    masked_rgb[~mask] = 0
    pil_img = Image.fromarray(masked_rgb)
    phash = imagehash.phash(pil_img, hash_size=8)
    gray = (
        0.299 * rgb[:, :, 0].astype(np.float32)
        + 0.587 * rgb[:, :, 1].astype(np.float32)
        + 0.114 * rgb[:, :, 2].astype(np.float32)
    )
    return phash, gray


def _top_tile_mad(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    mask: np.ndarray,
    grid_n: int = 4,
) -> tuple[float, float]:
    """Return (global masked MAD, max tile MAD) to catch localized slide/code edits."""
    diff = np.abs(gray_a - gray_b)
    active_diff = diff[mask]
    global_mad = float(active_diff.mean()) if active_diff.size else 0.0

    h, w = diff.shape
    th = max(1, h // grid_n)
    tw = max(1, w // grid_n)
    max_tile = 0.0
    for r in range(grid_n):
        y0, y1 = r * th, (r + 1) * th if r < grid_n - 1 else h
        for c in range(grid_n):
            x0, x1 = c * tw, (c + 1) * tw if c < grid_n - 1 else w
            sub_mask = mask[y0:y1, x0:x1]
            # Skip tiles that are mostly masked out (e.g., corner webcam)
            if sub_mask.mean() < 0.4:
                continue
            tile_mad = float(diff[y0:y1, x0:x1][sub_mask].mean())
            if tile_mad > max_tile:
                max_tile = tile_mad
    return global_mad, max_tile


def _is_same_scene(
    sig_a: tuple[imagehash.ImageHash, np.ndarray],
    sig_b: tuple[imagehash.ImageHash, np.ndarray],
    mask: np.ndarray,
    hash_thresh: int,
    mad_thresh: float,
    tile_thresh: float,
) -> bool:
    """Check whether two frame signatures depict essentially the same visual information."""
    phash_a, gray_a = sig_a
    phash_b, gray_b = sig_b
    global_mad, max_tile = _top_tile_mad(gray_a, gray_b, mask)
    # Near-identical pixels across every tile should match even on low-entropy slides where pHash jitters.
    if global_mad <= mad_thresh * 0.4 and max_tile <= tile_thresh * 0.5:
        return True
    if int(phash_a - phash_b) > hash_thresh:
        return False
    return global_mad <= mad_thresh and max_tile <= tile_thresh


def grab_frame(video_path: Path, timestamp_sec: float, out_path: Path, width: int = 1280) -> Path:
    """Extract a single crisp frame at timestamp_sec with fast ffmpeg input seeking."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scale_filter = f"scale='min({width},iw)':-2" if width > 0 else "null"
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{max(0.0, timestamp_sec):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        scale_filter,
        "-q:v",
        "2",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg frame extract failed at {timestamp_sec:.2f}s: {proc.stderr[-400:]}")
    return out_path


def _load_font(size: int = 18) -> ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/consolab.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def build_contact_sheets(
    keyframes: list[Keyframe],
    out_dir: Path,
    cols: int = 3,
    rows: int = 3,
    tile_width: int = 480,
) -> list[Path]:
    """Pack unique keyframes into labeled grid sheets so an AI can scan many frames cheaply."""
    visual_kfs = [kf for kf in keyframes if kf.image_path]
    if not visual_kfs:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    per_grid = max(1, cols * rows)
    font = _load_font(18)
    created: list[Path] = []

    with Image.open(visual_kfs[0].image_path) as sample:
        src_w, src_h = sample.size
    tile_h = max(60, int(round(tile_width * src_h / src_w)))
    pad = 8
    canvas_w = cols * tile_width + (cols + 1) * pad
    canvas_h = rows * tile_h + (rows + 1) * pad

    for g_idx in range(0, len(visual_kfs), per_grid):
        chunk = visual_kfs[g_idx : g_idx + per_grid]
        grid_num = g_idx // per_grid + 1
        actual_rows = (len(chunk) + cols - 1) // cols
        actual_h = actual_rows * tile_h + (actual_rows + 1) * pad
        sheet = Image.new("RGB", (canvas_w, min(canvas_h, actual_h)), (24, 24, 28))
        draw = ImageDraw.Draw(sheet)
        grid_path = out_dir / f"grid_{grid_num:03d}.jpg"

        for slot_i, kf in enumerate(chunk):
            r, c = divmod(slot_i, cols)
            x = pad + c * (tile_width + pad)
            y = pad + r * (tile_h + pad)
            with Image.open(kf.image_path) as tile:
                resized = tile.convert("RGB").resize((tile_width, tile_h), Image.Resampling.LANCZOS)
            sheet.paste(resized, (x, y))

            label = f"#{kf.index:02d} {format_ts(kf.rep_sec)}"
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([x + 6, y + 6, x + 16 + tw, y + 14 + th], fill=(0, 0, 0))
            draw.text((x + 11, y + 9), label, fill=(255, 230, 120), font=font)

            kf.grid_path = str(grid_path)
            kf.grid_slot = f"grid_{grid_num:03d} R{r + 1}C{c + 1}"

        sheet.save(grid_path, quality=88)
        created.append(grid_path)

    return created


def _is_blank_dark(gray: np.ndarray, mask: np.ndarray) -> bool:
    """Return True if active pixels form a nearly uniform black transition frame."""
    active = gray[mask]
    if active.size == 0:
        return False
    return float(active.mean()) < 14.0 and float(active.std()) < 8.0


def _pick_span_candidate(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    duration_sec: float,
    settle_sec: float,
    proxy_w: int,
    proxy_h: int,
    mask: np.ndarray,
) -> tuple[float, float, float, tuple[imagehash.ImageHash, np.ndarray]]:
    """Pick a settled representative frame inside a single [start_sec, end_sec] shot."""
    max_ts = max(0.0, duration_sec - 0.15) if duration_sec > 0.2 else duration_sec
    run_dur = max(0.0, end_sec - start_sec)
    rep_ts = start_sec + min(settle_sec, run_dur * 0.4)
    rep_ts = max(0.0, min(rep_ts, max_ts, max(start_sec, end_sec - 0.05)))
    try:
        rep_rgb = _grab_proxy_rgb(video_path, rep_ts, proxy_w, proxy_h)
    except RuntimeError:
        rep_ts = max(0.0, min(start_sec, max_ts))
        rep_rgb = _grab_proxy_rgb(video_path, rep_ts, proxy_w, proxy_h)
    return start_sec, end_sec, rep_ts, _compute_signature(rep_rgb, mask)


def _collect_proxy_candidates(
    video_path: Path,
    sample_fps: float,
    proxy_w: int,
    proxy_h: int,
    mask: np.ndarray,
    dedup_mode: str,
    hash_thresh: int,
    mad_thresh: float,
    tile_thresh: float,
    settle_sec: float,
    start_sec: float | None = None,
    end_sec: float | None = None,
    skip_blank_dark: bool = False,
) -> tuple[list[tuple[float, float, float, tuple[imagehash.ImageHash, np.ndarray]]], int]:
    """Stream proxy frames over a window and cluster consecutive similar frames."""
    sampled: list[tuple[float, tuple[imagehash.ImageHash, np.ndarray]]] = []
    raw_count = 0
    for ts, rgb in _stream_proxy_frames(
        video_path, sample_fps, proxy_w, proxy_h, start_sec=start_sec, end_sec=end_sec
    ):
        raw_count += 1
        sig = _compute_signature(rgb, mask)
        sampled.append((ts, sig))

    if skip_blank_dark and sampled:
        non_dark = [item for item in sampled if not _is_blank_dark(item[1][1], mask)]
        if non_dark:
            sampled = non_dark

    runs: list[dict] = []
    cur_run: dict | None = None
    for ts, sig in sampled:
        if cur_run is None:
            cur_run = {"start_sec": ts, "end_sec": ts, "frames": [(ts, sig)], "anchor": sig}
            continue
        if dedup_mode != "none" and _is_same_scene(
            cur_run["anchor"], sig, mask, hash_thresh, mad_thresh, tile_thresh
        ):
            cur_run["end_sec"] = ts
            cur_run["frames"].append((ts, sig))
        else:
            runs.append(cur_run)
            cur_run = {"start_sec": ts, "end_sec": ts, "frames": [(ts, sig)], "anchor": sig}
    if cur_run is not None:
        runs.append(cur_run)

    step_sec = 1.0 / sample_fps
    out: list[tuple[float, float, float, tuple[imagehash.ImageHash, np.ndarray]]] = []
    for i, run in enumerate(runs):
        s_sec = run["start_sec"]
        e_sec = run["end_sec"]
        run_dur = max(0.0, e_sec - s_sec)
        target_ts = s_sec + min(settle_sec, run_dur * 0.4)
        rep_ts, rep_sig = min(run["frames"], key=lambda item: abs(item[0] - target_ts))
        next_start = runs[i + 1]["start_sec"] if i + 1 < len(runs) else (e_sec + step_sec)
        end_out = min(next_start, end_sec) if end_sec is not None else next_start
        out.append((s_sec, max(e_sec + step_sec, end_out), rep_ts, rep_sig))

    return out, raw_count


def extract_deduplicated_keyframes(
    video_path: Path,
    out_dir: Path,
    detector: str = "auto",
    scene_algo: str = "content",
    scene_threshold: float | None = None,
    min_scene_sec: float = 0.6,
    sample_fps: float = 1.0,
    hash_thresh: int = 8,
    mad_thresh: float = 6.0,
    tile_thresh: float = 14.0,
    global_hash_thresh: int = 6,
    global_mad_thresh: float = 4.0,
    global_tile_thresh: float = 9.0,
    dedup_mode: str = "global",
    ignore_bottom: float = 0.0,
    ignore_rects: list[tuple[float, float, float, float]] | None = None,
    settle_sec: float = 1.0,
    out_width: int = 1280,
    grid_cols: int = 3,
    grid_rows: int = 3,
    make_grids: bool = True,
) -> tuple[list[Keyframe], list[Path], dict]:
    """Extract deduplicated keyframes via auto hybrid, PySceneDetect, or proxy sampling."""
    meta = probe_video(video_path)
    proxy_w = 320
    proxy_h = _even_height(meta["width"], meta["height"], proxy_w)
    mask = _build_mask(proxy_h, proxy_w, ignore_bottom, ignore_rects or [])

    # Each candidate is (start_sec, end_sec_out, rep_ts, rep_sig).
    candidates: list[tuple[float, float, float, tuple[imagehash.ImageHash, np.ndarray]]] = []
    total_sampled = 0
    refined_spans = 0

    if detector == "proxy":
        detector_label = f"proxy@{sample_fps:g}fps"
        candidates, total_sampled = _collect_proxy_candidates(
            video_path=video_path,
            sample_fps=sample_fps,
            proxy_w=proxy_w,
            proxy_h=proxy_h,
            mask=mask,
            dedup_mode=dedup_mode,
            hash_thresh=hash_thresh,
            mad_thresh=mad_thresh,
            tile_thresh=tile_thresh,
            settle_sec=settle_sec,
        )

    elif detector in ("scene", "auto"):
        span_stats, total_sampled, eff_thresh = _detect_scenes_pyscenedetect(
            video_path,
            duration_sec=meta["duration"],
            algo=scene_algo,
            threshold=scene_threshold,
            min_scene_sec=min_scene_sec,
        )
        scene_count = len(span_stats)
        for start_sec, end_sec, med_cv, pct_lt1, max_cv in span_stats:
            dur = end_sec - start_sec
            is_static_sub = (
                detector == "auto"
                and dur >= 3.5
                and med_cv <= 0.5
                and pct_lt1 >= 0.60
                and (max_cv >= 4.0 or scene_count <= 2)
            )
            if is_static_sub:
                sub_cands, _ = _collect_proxy_candidates(
                    video_path=video_path,
                    sample_fps=sample_fps,
                    proxy_w=proxy_w,
                    proxy_h=proxy_h,
                    mask=mask,
                    dedup_mode=dedup_mode,
                    hash_thresh=hash_thresh,
                    mad_thresh=mad_thresh,
                    tile_thresh=tile_thresh,
                    settle_sec=settle_sec,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    skip_blank_dark=True,
                )
                if sub_cands:
                    # Keep the PySceneDetect outer span boundaries on the first/last refined frame.
                    first_s, first_e, first_rep, first_sig = sub_cands[0]
                    sub_cands[0] = (start_sec, first_e, first_rep, first_sig)
                    last_s, _, last_rep, last_sig = sub_cands[-1]
                    sub_cands[-1] = (last_s, end_sec, last_rep, last_sig)
                    candidates.extend(sub_cands)
                    refined_spans += 1
                    continue

            candidates.append(
                _pick_span_candidate(
                    video_path,
                    start_sec,
                    end_sec,
                    meta["duration"],
                    settle_sec,
                    proxy_w,
                    proxy_h,
                    mask,
                )
            )

        if detector == "auto":
            detector_label = (
                f"auto(pyscenedetect:{scene_algo}({eff_thresh:g})"
                f"+proxy[{refined_spans}/{scene_count}])"
            )
        else:
            detector_label = f"pyscenedetect:{scene_algo}({eff_thresh:g})"
    else:
        raise ValueError(f"Unsupported detector: {detector}")

    frames_dir = out_dir / "keyframes"
    frames_dir.mkdir(parents=True, exist_ok=True)
    keyframes: list[Keyframe] = []
    seen_anchors: list[tuple[str, tuple[imagehash.ImageHash, np.ndarray]]] = []

    for run_idx, (start_sec, end_sec_out, rep_ts, rep_sig) in enumerate(candidates, start=1):
        check_anchors = (
            seen_anchors
            if dedup_mode == "global"
            else (seen_anchors[-1:] if detector in ("scene", "auto") and dedup_mode == "consecutive" else [])
        )
        matched_id: str | None = None
        for prev_id, prev_sig in check_anchors:
            if _is_same_scene(
                prev_sig,
                rep_sig,
                mask,
                global_hash_thresh,
                global_mad_thresh,
                global_tile_thresh,
            ):
                matched_id = prev_id
                break

        kf_id = f"kf_{run_idx:03d}"
        dur_sec = max(0.0, end_sec_out - start_sec)
        if matched_id is not None:
            keyframes.append(
                Keyframe(
                    index=run_idx,
                    id=kf_id,
                    start_sec=round(start_sec, 2),
                    end_sec=round(end_sec_out, 2),
                    rep_sec=round(rep_ts, 2),
                    duration_sec=round(dur_sec, 2),
                    image_path=None,
                    same_as=matched_id,
                )
            )
            continue

        img_name = f"{kf_id}_{format_ts(rep_ts, compact=True)}.jpg"
        img_path = grab_frame(video_path, rep_ts, frames_dir / img_name, width=out_width)
        seen_anchors.append((kf_id, rep_sig))
        keyframes.append(
            Keyframe(
                index=run_idx,
                id=kf_id,
                start_sec=round(start_sec, 2),
                end_sec=round(end_sec_out, 2),
                rep_sec=round(rep_ts, 2),
                duration_sec=round(dur_sec, 2),
                image_path=str(img_path),
            )
        )

    grids: list[Path] = []
    if make_grids and keyframes:
        grids = build_contact_sheets(keyframes, out_dir / "grids", cols=grid_cols, rows=grid_rows)

    unique_count = sum(1 for kf in keyframes if kf.image_path)
    stats = {
        "video": str(video_path),
        "detector": detector_label,
        "duration_sec": round(meta["duration"], 2),
        "resolution": f"{meta['width']}x{meta['height']}",
        "sampled_frames": total_sampled,
        "scene_runs": len(candidates),
        "unique_images": unique_count,
        "revisit_refs": len(keyframes) - unique_count,
        "grid_count": len(grids),
        "keyframes": [asdict(kf) for kf in keyframes],
    }
    return keyframes, grids, stats

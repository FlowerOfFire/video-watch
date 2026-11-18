"""Subtitle retrieval (original-language first) and local Qwen3-ASR-1.7B fallback."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from .frames import format_ts

QWEN_CHUNKED_SCRIPT = Path(__file__).resolve().parent / "qwen_asr_chunked.py"
BILI_SUB_SCRIPT = Path(__file__).resolve().parent / "bili_subtitles.py"


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def resolve_qwen_python() -> Path | None:
    """Locate the external Python interpreter with qwen-asr installed."""
    env_py = os.environ.get("VIDEO_WATCH_QWEN_PYTHON", "").strip()
    if env_py:
        p = Path(env_py).expanduser()
        return p if p.exists() else None

    candidates = [
        _venv_python(base / "qwen3-asr" / ".venv")
        for base in (Path.cwd(), *Path(__file__).resolve().parents[:4])
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def resolve_bili_python() -> Path | None:
    """Locate bilibili-cli's Python interpreter for original-language subtitle API queries."""
    env_py = os.environ.get("VIDEO_WATCH_BILI_PYTHON", "").strip()
    if env_py:
        p = Path(env_py).expanduser()
        return p if p.exists() else None

    tool_roots: list[Path] = []
    uv_tool_dir = os.environ.get("UV_TOOL_DIR", "").strip()
    if uv_tool_dir:
        tool_roots.append(Path(uv_tool_dir).expanduser())
    if os.name == "nt":
        tool_roots.append(Path.home() / "AppData" / "Roaming" / "uv" / "tools")
    else:
        tool_roots.append(Path.home() / ".local" / "share" / "uv" / "tools")

    for root in tool_roots:
        cand = _venv_python(root / "bilibili-cli")
        if cand.exists():
            return cand
    return None

_TS_RE = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})\s*-->\s*(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})"
)
_TAG_RE = re.compile(r"<[^>]+>")

_ISO639_ALIASES = {
    "chi": ("zh", "中文", "简体", "繁体", "chinese"),
    "zho": ("zh", "中文", "简体", "繁体", "chinese"),
    "zh": ("zh", "中文", "简体", "繁体", "chinese"),
    "eng": ("en", "english", "英语", "英文"),
    "en": ("en", "english", "英语", "英文"),
    "jpn": ("ja", "jp", "日本語", "日语", "日文", "japanese"),
    "ja": ("ja", "jp", "日本語", "日语", "日文", "japanese"),
    "kor": ("ko", "한국어", "韩语", "韩文", "korean"),
    "ko": ("ko", "한국어", "韩语", "韩文", "korean"),
    "fra": ("fr", "français", "french", "法语"),
    "fre": ("fr", "français", "french", "法语"),
    "deu": ("de", "deutsch", "german", "德语"),
    "ger": ("de", "deutsch", "german", "德语"),
    "spa": ("es", "español", "spanish", "西班牙语"),
    "rus": ("ru", "русский", "russian", "俄语"),
}


def _to_seconds(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt_or_vtt(raw_text: str) -> list[dict]:
    """Parse SRT or WebVTT text into normalized [{start, end, text}] segments."""
    segments: list[dict] = []
    blocks = re.split(r"\r?\n\s*\r?\n", raw_text.strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0].startswith("WEBVTT"):
            continue
        match = None
        ts_idx = -1
        for i, line in enumerate(lines):
            m = _TS_RE.search(line)
            if m:
                match = m
                ts_idx = i
                break
        if not match:
            continue
        start = _to_seconds(match.group(1), match.group(2), match.group(3), match.group(4))
        end = _to_seconds(match.group(5), match.group(6), match.group(7), match.group(8))
        body = " ".join(_TAG_RE.sub("", l).strip() for l in lines[ts_idx + 1 :]).strip()
        if not body:
            continue
        if segments and segments[-1]["text"] == body:
            segments[-1]["end"] = max(segments[-1]["end"], round(end, 2))
            continue
        segments.append({"start": round(start, 2), "end": round(end, 2), "text": body})
    return segments


def is_bilibili_target(target: str) -> bool:
    """Return True if target looks like a Bilibili BV/av id or bilibili.com / b23.tv URL."""
    low = target.lower()
    return (
        "bilibili.com" in low
        or "b23.tv" in low
        or bool(re.fullmatch(r"bv[0-9a-z]+", target.strip(), re.IGNORECASE))
    )


def fetch_bili_subtitle(target: str) -> tuple[list[dict] | None, str]:
    """Fetch Bilibili subtitle prioritizing original language (ai_type=0, type=0)."""
    bili_python = resolve_bili_python()
    if bili_python and BILI_SUB_SCRIPT.exists():
        proc = subprocess.run(
            [str(bili_python), str(BILI_SUB_SCRIPT), target],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            # Credential refresh may emit warning lines before JSON; parse last JSON line.
            json_line = proc.stdout.strip().splitlines()[-1]
            try:
                payload = json.loads(json_line)
                if payload.get("ok") and payload.get("segments"):
                    lan_doc = payload.get("lan_doc") or payload.get("lan") or "unknown"
                    lan = payload.get("lan") or ""
                    kind = "original" if payload.get("ai_type") == 0 else "translated"
                    label = f"bilibili-subtitle:{lan_doc} ({lan}, {kind})".strip()
                    return payload["segments"], label
            except json.JSONDecodeError:
                pass

    # Fallback to `bili` CLI if helper is unavailable
    cmd = ["bili", "video", target, "-st", "--subtitle-format", "srt", "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None, ""
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, ""

    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    sub_val = None
    for key in ("subtitle_srt", "subtitle_timeline", "subtitle", "subtitles"):
        if isinstance(data, dict) and data.get(key):
            sub_val = data[key]
            break

    if isinstance(sub_val, str):
        parsed = parse_srt_or_vtt(sub_val)
        return (parsed, "bilibili-subtitle") if parsed else (None, "")
    if isinstance(sub_val, list):
        segs: list[dict] = []
        for item in sub_val:
            if not isinstance(item, dict):
                continue
            text = str(item.get("content") or item.get("text") or "").strip()
            if not text:
                continue
            start = float(item.get("from") or item.get("start") or 0.0)
            end = float(item.get("to") or item.get("end") or start)
            segs.append({"start": round(start, 2), "end": round(end, 2), "text": text})
        return (segs, "bilibili-subtitle") if segs else (None, "")
    return None, ""


def _match_lang_key(candidates: list[str], target_lang: str) -> str | None:
    """Match a yt-dlp subtitle language key against target_lang (exact or base prefix)."""
    if not target_lang:
        return None
    tl = target_lang.lower()
    base = tl.split("-")[0]
    for key in candidates:
        if key.lower() == tl:
            return key
    for key in candidates:
        if key.lower() == base or key.lower().startswith(base + "-"):
            return key
    return None


def _pick_ytdlp_original_track(meta: dict) -> tuple[str | None, bool, bool]:
    """Choose (track_key, is_auto, is_original) from yt-dlp metadata."""
    manual = meta.get("subtitles") or {}
    auto = meta.get("automatic_captions") or {}
    manual_keys = [k for k in manual.keys() if k != "live_chat"]
    auto_keys = list(auto.keys())

    orig_lang = str(meta.get("language") or "").strip()
    orig_auto_keys = [k for k in auto_keys if k.lower().endswith("-orig")]
    base_from_orig = orig_auto_keys[0][:-5] if orig_auto_keys else ""

    # 1. Manual subtitle in the video's declared original language
    for hint in (orig_lang, base_from_orig):
        matched = _match_lang_key(manual_keys, hint)
        if matched:
            return matched, False, True

    # 2. Original-language auto-caption (*-orig or matching declared video language)
    if orig_auto_keys:
        return orig_auto_keys[0], True, True
    if orig_lang:
        matched = _match_lang_key(auto_keys, orig_lang)
        if matched:
            return matched, True, True

    # 3. Fallback when video metadata lacks language: any manual track, then zh/en auto
    for hint in ("zh", "zh-CN", "zh-Hans", "en"):
        matched = _match_lang_key(manual_keys, hint)
        if matched:
            return matched, False, False
    if manual_keys:
        return manual_keys[0], False, False
    for hint in ("zh", "en"):
        matched = _match_lang_key(auto_keys, hint)
        if matched:
            return matched, True, False
    return None, False, False


def fetch_ytdlp_subtitle(target: str, tmp_dir: Path) -> tuple[list[dict] | None, str]:
    """Fetch platform subtitle via yt-dlp prioritizing the video's original language."""
    info_proc = subprocess.run(
        ["yt-dlp", "--dump-single-json", "--skip-download", target],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    meta: dict = {}
    if info_proc.returncode == 0 and info_proc.stdout.strip():
        try:
            meta = json.loads(info_proc.stdout.lstrip("\ufeff"))
        except json.JSONDecodeError:
            meta = {}

    chosen_key, is_auto, is_orig = _pick_ytdlp_original_track(meta)
    with tempfile.TemporaryDirectory(prefix="vw_subs_") as sub_tmp:
        sub_dir = Path(sub_tmp)
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--sub-format",
            "srt/vtt",
            "-o",
            str(sub_dir / "sub.%(ext)s"),
        ]
        if chosen_key:
            cmd.append("--write-auto-subs" if is_auto else "--write-subs")
            cmd.extend(["--sub-langs", chosen_key])
        else:
            cmd.extend(["--write-subs", "--write-auto-subs", "--sub-langs", "zh.*,en.*"])
        cmd.append(target)

        subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)
        for ext in ("*.srt", "*.vtt"):
            files = sorted(sub_dir.glob(ext))
            if files:
                parsed = parse_srt_or_vtt(files[0].read_text(encoding="utf-8", errors="replace"))
                if parsed:
                    if chosen_key:
                        kind = "original" if is_orig else "fallback"
                        src = "auto" if is_auto else "manual"
                        label = f"yt-dlp-subtitle:{chosen_key} ({src}, {kind})"
                    else:
                        label = "yt-dlp-subtitle"
                    return parsed, label
    return None, ""


def _probe_audio_language(media_path: Path) -> str:
    """Read primary audio stream language tag (e.g. eng/chi/jpn) from container metadata."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "a:0",
            str(media_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return ""
    try:
        streams = json.loads(proc.stdout).get("streams") or []
        if not streams:
            return ""
        lang = str((streams[0].get("tags") or {}).get("language") or "").strip().lower()
        return "" if lang in ("", "und", "unknown") else lang
    except (json.JSONDecodeError, AttributeError):
        return ""


def _sidecar_priority(path: Path, audio_lang: str = "") -> tuple[int, int, int, str]:
    """Rank sidecar subtitles: match audio_lang first, avoid translated markers, then zh/en."""
    name = path.name.lower()
    translated_penalty = 1 if any(m in name for m in ("自动翻译", "translated", "machine")) else 0

    lang_match = 1
    if audio_lang:
        aliases = _ISO639_ALIASES.get(audio_lang, (audio_lang,))
        if any(a in name for a in aliases):
            lang_match = 0

    if any(tag in name for tag in ("中文", ".zh", "zh-cn", "ai-zh", "简体")):
        fallback = 0
    elif any(tag in name for tag in ("english", ".en")):
        fallback = 1
    else:
        fallback = 2
    return (lang_match, translated_penalty, fallback, name)


def run_qwen_asr(
    media_path: Path,
    out_dir: Path,
    language: str | None = None,
    context: str = "",
) -> tuple[list[dict], str]:
    """Run chunked local Qwen3-ASR-1.7B transcription using the resolved qwen3-asr venv."""
    qwen_python = resolve_qwen_python()
    if not qwen_python:
        raise FileNotFoundError(
            "Qwen3-ASR venv python not found. Set VIDEO_WATCH_QWEN_PYTHON or pass --no-asr."
        )

    out_json = out_dir / "transcript.json"
    out_md = out_dir / "transcript.md"
    cmd = [
        str(qwen_python),
        str(QWEN_CHUNKED_SCRIPT),
        str(media_path),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    model_dir = os.environ.get("VIDEO_WATCH_QWEN_MODEL", "").strip()
    if model_dir:
        cmd.extend(["--model-dir", model_dir])
    if language:
        cmd.extend(["--language", language])
    if context:
        cmd.extend(["--context", context])

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0 or not out_json.exists():
        raise RuntimeError(f"Qwen3-ASR failed with exit code {proc.returncode}")

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    return payload.get("segments", []), "Qwen3-ASR-1.7B (local)"


def save_transcript(segments: list[dict], source_label: str, out_dir: Path, title: str) -> tuple[Path, Path]:
    """Write normalized transcript.json and timestamped transcript.md."""
    out_json = out_dir / "transcript.json"
    out_md = out_dir / "transcript.md"
    out_json.write_text(
        json.dumps({"title": title, "engine": source_label, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [f"# {title} ({source_label})\n"]
    for seg in segments:
        lines.append(f"- **[{format_ts(seg['start'])} - {format_ts(seg['end'])}]** {seg['text']}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_json, out_md


def resolve_transcript(
    media_path: Path,
    out_dir: Path,
    source_target: str | None = None,
    sub_file: Path | None = None,
    force_asr: bool = False,
    skip_asr: bool = False,
    language: str | None = None,
    context: str = "",
) -> tuple[list[dict], str]:
    """Obtain timestamped transcript: explicit sub -> original-lang platform sub -> sidecar -> Qwen3-ASR."""
    if sub_file and sub_file.exists() and not force_asr:
        segs = parse_srt_or_vtt(sub_file.read_text(encoding="utf-8", errors="replace"))
        if segs:
            save_transcript(segs, f"subtitle:{sub_file.name}", out_dir, media_path.name)
            return segs, f"subtitle:{sub_file.name}"

    if not force_asr:
        # 1. Query platform subtitle metadata first so original-language tracks beat AI translations.
        if source_target and not Path(source_target).exists():
            if is_bilibili_target(source_target):
                segs, label = fetch_bili_subtitle(source_target)
                if segs:
                    save_transcript(segs, label, out_dir, media_path.name)
                    return segs, label
            else:
                segs, label = fetch_ytdlp_subtitle(source_target, out_dir)
                if segs:
                    save_transcript(segs, label, out_dir, media_path.name)
                    return segs, label

        # 2. Check sidecar .srt/.vtt next to local video file, matching audio language tag if present.
        audio_lang = _probe_audio_language(media_path)
        for pattern in ("*.srt", "*.vtt"):
            sidecars = sorted(
                media_path.parent.glob(pattern),
                key=lambda p: _sidecar_priority(p, audio_lang),
            )
            if sidecars:
                segs = parse_srt_or_vtt(sidecars[0].read_text(encoding="utf-8", errors="replace"))
                if segs:
                    save_transcript(segs, f"sidecar:{sidecars[0].name}", out_dir, media_path.name)
                    return segs, f"sidecar:{sidecars[0].name}"

    if skip_asr:
        return [], "skipped"

    if not force_asr and not resolve_qwen_python():
        print(
            "[WARN] No subtitles found and Qwen3-ASR is not configured; continuing without transcript "
            "(set VIDEO_WATCH_QWEN_PYTHON to enable local ASR).",
            flush=True,
        )
        return [], "unavailable (no qwen3-asr)"

    segs, engine = run_qwen_asr(media_path, out_dir, language=language, context=context)
    save_transcript(segs, engine, out_dir, media_path.name)
    return segs, engine

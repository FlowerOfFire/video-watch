"""Chunked timestamped audio transcription using local Qwen3-ASR-1.7B."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel

SAMPLE_RATE = 16000


def resolve_default_model_dir() -> str:
    """Resolve Qwen3-ASR model directory via env var, sibling checkout, or HF model ID."""
    env_model = os.environ.get("VIDEO_WATCH_QWEN_MODEL", "").strip()
    if env_model:
        return env_model
    candidates = [
        base / "qwen3-asr" / "models" / "Qwen3-ASR-1.7B"
        for base in (Path.cwd(), *Path(__file__).resolve().parents[:4])
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return "Qwen/Qwen3-ASR-1.7B"


def fmt_ts(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def decode_audio_ffmpeg(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio/video file to 16kHz mono float32 PCM."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed for {path}: {err[-500:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def split_audio_chunks(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    min_sec: float = 12.0,
    max_sec: float = 28.0,
) -> list[tuple[float, float, np.ndarray]]:
    """Split audio at lowest-RMS pauses within [min_sec, max_sec] so long videos stay bounded and timestamped."""
    total_samples = len(wav)
    if total_samples == 0:
        return []
    win = int(0.1 * sr)  # 100ms RMS windows
    min_s = int(min_sec * sr)
    max_s = int(max_sec * sr)

    chunks: list[tuple[float, float, np.ndarray]] = []
    pos = 0
    while pos < total_samples:
        remaining = total_samples - pos
        if remaining <= max_s:
            chunks.append((pos / sr, total_samples / sr, wav[pos:total_samples]))
            break

        search_start = pos + min_s
        search_end = min(total_samples, pos + max_s)
        best_cut = search_end
        best_rms = float("inf")

        for cur in range(search_start, search_end - win + 1, win):
            seg = wav[cur : cur + win]
            rms = float(np.sqrt(np.mean(seg * seg)))
            if rms < best_rms:
                best_rms = rms
                best_cut = cur + win // 2

        chunks.append((pos / sr, best_cut / sr, wav[pos:best_cut]))
        pos = best_cut

    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunked local Qwen3-ASR-1.7B transcription with timestamps.")
    parser.add_argument("media", help="Audio or video file path")
    parser.add_argument("--model-dir", default=resolve_default_model_dir())
    parser.add_argument("--language", default=None, help="Optional language hint, e.g. Chinese or English")
    parser.add_argument("--context", default="", help="Optional context/hotword hint")
    parser.add_argument("--min-chunk-sec", type=float, default=12.0)
    parser.add_argument("--max-chunk-sec", type=float, default=28.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    media_path = Path(args.media)
    if not media_path.exists():
        print(f"Media file not found: {media_path}", file=sys.stderr)
        return 1

    model_ref = args.model_dir
    model_path = Path(model_ref)
    if (model_path.is_absolute() or model_ref.startswith(".")) and not model_path.exists():
        print(f"Model directory not found: {model_path}", file=sys.stderr)
        return 1

    wav = decode_audio_ffmpeg(media_path, SAMPLE_RATE)
    duration = len(wav) / SAMPLE_RATE
    chunks = split_audio_chunks(wav, SAMPLE_RATE, args.min_chunk_sec, args.max_chunk_sec)
    print(f"Audio duration: {duration:.1f}s -> {len(chunks)} chunks", flush=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_map == "cpu":
        dtype = torch.float32

    print(f"Loading Qwen3-ASR from {model_ref} on {device_map} ({dtype})...", flush=True)
    model = Qwen3ASRModel.from_pretrained(
        model_ref,
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=4,
        max_new_tokens=1024,
    )

    segments: list[dict] = []
    md_lines = [f"# {media_path.name} 语音转写 ({duration:.1f}s)\n"]
    for i, (start_s, end_s, sub_wav) in enumerate(chunks, start=1):
        res = model.transcribe(
            audio=(sub_wav, SAMPLE_RATE),
            context=args.context,
            language=args.language,
        )[0]
        text = (res.text or "").strip()
        lang = res.language or "Unknown"
        print(f"[{i}/{len(chunks)}] {fmt_ts(start_s)}-{fmt_ts(end_s)} ({lang}): {text[:60]}", flush=True)
        if text:
            segments.append(
                {
                    "start": round(start_s, 2),
                    "end": round(end_s, 2),
                    "language": lang,
                    "text": text,
                }
            )
            md_lines.append(f"- **[{fmt_ts(start_s)} - {fmt_ts(end_s)}]** {text}")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "source": str(media_path),
                "duration_sec": round(duration, 2),
                "engine": "Qwen3-ASR-1.7B",
                "segments": segments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved transcript JSON: {out_json}", flush=True)
    print(f"Saved transcript MD:   {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

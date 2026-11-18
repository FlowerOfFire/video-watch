"""Fetch Bilibili subtitle tracks with original-language priority (ai_type=0, type=0)."""

from __future__ import annotations

import asyncio
import json
import sys


def _track_rank(sub: dict) -> tuple[int, int, int, str]:
    """Rank Bilibili subtitle tracks: human original (type=0, ai_type=0) > AI original > AI translated."""
    ai_type = int(sub.get("ai_type") if sub.get("ai_type") is not None else 0)
    sub_type = int(sub.get("type") if sub.get("type") is not None else 1)
    lan = str(sub.get("lan") or "").lower()
    lang_tie = 0 if "zh" in lan else (1 if "en" in lan else 2)
    # Primary: ai_type == 0 (original language, not translated)
    # Secondary: type == 0 (creator/community manual CC before AI ASR)
    return (0 if ai_type == 0 else 1, 0 if sub_type == 0 else 1, lang_tie, lan)


async def fetch_best_subtitle(target: str) -> dict:
    import aiohttp
    from bili_cli.auth import get_credential
    from bili_cli.client import extract_bvid
    from bilibili_api import video

    bvid = extract_bvid(target)
    cred = get_credential("optional")
    v = video.Video(bvid=bvid, credential=cred)
    pages = await v.get_pages()
    if not pages:
        return {"ok": False, "reason": "no_pages"}
    cid = pages[0].get("cid")
    if not cid:
        return {"ok": False, "reason": "no_cid"}

    player_info = await v.get_player_info(cid=cid)
    subs = [
        s
        for s in (player_info.get("subtitle") or {}).get("subtitles") or []
        if s.get("subtitle_url") and not s.get("is_lock")
    ]
    if not subs:
        return {"ok": False, "reason": "no_subtitles"}

    chosen = min(subs, key=_track_rank)
    url = str(chosen["subtitle_url"])
    if url.startswith("//"):
        url = "https:" + url

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

    body = data.get("body") or []
    segments = []
    for item in body:
        text = str(item.get("content") or "").strip()
        if not text:
            continue
        start = float(item.get("from") or 0.0)
        end = float(item.get("to") or start)
        segments.append({"start": round(start, 2), "end": round(end, 2), "text": text})

    return {
        "ok": True,
        "bvid": bvid,
        "lan": chosen.get("lan"),
        "lan_doc": chosen.get("lan_doc"),
        "ai_type": chosen.get("ai_type"),
        "type": chosen.get("type"),
        "available_tracks": [
            {
                "lan": s.get("lan"),
                "lan_doc": s.get("lan_doc"),
                "ai_type": s.get("ai_type"),
                "type": s.get("type"),
            }
            for s in subs
        ],
        "segments": segments,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: bili_subtitles.py <BV_OR_URL>", file=sys.stderr)
        return 1
    result = asyncio.run(fetch_best_subtitle(sys.argv[1]))
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

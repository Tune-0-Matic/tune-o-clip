"""workspace/posted_history.json, one JSON file (no database): the publisher's "posts" and the scraper's "clips" (every
clip it kept), plus the 30-day duplicate guard over both. Stdlib only: publisher/scheduler.py imports it too.

Every write re-reads the file, applies its change and swaps a temp copy in (atomic), so the publisher and a clipper
run can't erase each other's entries and a reader never sees a half-written file.
"""
import json
import os
import re
import time
from pathlib import Path

# TOC_HISTORY: tests point every process (incl. clipper.py subprocesses) at a temp file, never the real one
HISTORY = Path(os.environ.get("TOC_HISTORY") or Path(__file__).resolve().parent.parent / "workspace" / "posted_history.json")
WINDOW = 30 * 24 * 3600  # s: no clip ID or near-identical transcript again within 30 days
SIMILAR = 0.6  # share of distinct words in common that makes two transcripts "the same moment"


def jaccard(a, b, min_words=5):
    """Share of distinct words two transcripts have in common (0..1); 0 when either is too short to tell."""
    ta, tb = (set(re.findall(r"[a-z0-9']+", str(t).lower())) for t in (a, b))
    if len(ta) < min_words or len(tb) < min_words:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def load(path=None):
    try:
        h = json.loads(Path(path or HISTORY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        h = {}
    h = h if isinstance(h, dict) else {}
    for k in ("posts", "clips"):
        h[k] = h.get(k) if isinstance(h.get(k), list) else []
    return h


def update(change, path=None):
    """Re-read, apply change(history) in place, write atomically. -> the saved history."""
    path = Path(path or HISTORY)  # looked up per call: tests point HISTORY at a temp file
    path.parent.mkdir(parents=True, exist_ok=True)
    h = load(path)
    change(h)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(h, indent=2, ensure_ascii=False), encoding="utf-8")
    for wait in (0.05, 0.1, 0.2, 0.4, None):
        try:
            os.replace(tmp, path)
            return h
        except PermissionError:  # Windows: another process has it open for a moment
            if wait is None:
                raise
            time.sleep(wait)


def remember(clip_id, streamer, transcript, now=None, path=None):
    """Record a kept clip; entries past the window are pruned on the way (the file stays small)."""
    now = time.time() if now is None else now

    def change(h):
        h["clips"] = [c for c in h["clips"] if now - float(c.get("at", 0)) < WINDOW]
        h["clips"].append({"clip_id": str(clip_id), "streamer": str(streamer or "").lower(), "at": now,
                           "transcript": str(transcript or "")[:2000]})
    update(change, path)


def seen(clip_id=None, streamer=None, transcript="", now=None, path=None, history=None):
    """Why this clip is a repeat of the last 30 days ("clip <id>" / "transcript 87% like <id>"), or None.
    Matches a clip ID, or a transcript sharing >= SIMILAR of its words with a kept clip of the same streamer
    (any streamer when either is unknown) or with a posted short."""
    now = time.time() if now is None else now
    h = history if history is not None else load(path)
    streamer = str(streamer or "").lower()
    recent = [c for c in h["clips"] if now - float(c.get("at", 0)) < WINDOW]
    if clip_id is not None and any(c.get("clip_id") == str(clip_id) for c in recent):
        return f"clip {clip_id}"
    for c in recent:
        if streamer and c.get("streamer") and c["streamer"] != streamer:
            continue
        sim = jaccard(transcript, c.get("transcript", ""))
        if sim >= SIMILAR:
            return f"transcript {sim:.0%} like {c.get('clip_id')}"
    for p in h["posts"]:
        if now - float(p.get("posted_at", 0)) < WINDOW:
            sim = jaccard(transcript, p.get("transcript", ""))
            if sim >= SIMILAR:
                return f"transcript {sim:.0%} like posted {p.get('short_id')}"
    return None

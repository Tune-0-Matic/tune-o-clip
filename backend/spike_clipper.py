"""Fresh, unclipped moments from a long stream: score a per-second curve BEFORE downloading any video, then download
and process only the best 45-60 s slices.

  python backend/spike_clipper.py URL [--chat FILE] [--clips 4] [clipper.py options, e.g. --auto-edit]

  Twitch VOD (twitch.tv/videos/<id>): chat replay -> per-second bursts of unique chatters, laugh / hype emotes and
      "clip it", shifted 8 s earlier (chat reacts after the moment). Chat comes from --chat FILE or
      workspace/chat/<vod id>.json: a TwitchDownloader `chatdownload` JSON export, or [{"t", "user", "text"}].
      Without either, the replay is fetched once from Twitch's web GraphQL endpoint (the read-only, no-login method
      twitch.tv and TwitchDownloader use) and cached there; if Twitch blocks or rate-limits it, the run stops with the
      TwitchDownloaderCLI how-to. (Twitch's official API has no VOD chat and yt-dlp no longer returns it.)
  YouTube: the Most Replayed heatmap (yt-dlp metadata, no video).

The first / last 5 minutes are skipped on streams over 15 min (intro / outro screens), peaks are >= 90 s apart, and
each slice goes through clipper.py as one whole clip (Whisper, the Ollama judge, the 30-day duplicate guard,
--auto-edit -> the retention editor). stdout is JSON Lines like clipper.py (the children's lines pass straight through).
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clipper  # noqa: E402
import scorer  # noqa: E402

ROOT = HERE.parent
CHAT_DIR = ROOT / "workspace" / "chat"
SHIFT = 8  # s: chat reacts this long after the moment
SKIP, SKIP_MIN_DURATION = 300, 900  # skip intro / outro minutes on streams longer than 15 min
GAP = 90  # s between picked moments
BEFORE, AFTER = 32, 18  # the slice around a moment: setup before, reaction after (50 s, inside 45-60)
SLICE_TIMEOUT = 120  # s per yt-dlp -g / ffmpeg attempt; a direct seek takes ~5 s
EMOTE_RE = re.compile(r"(?<!\w)(KEKW|KEKL|OMEGALUL|LULW?|ICANT|LMAO+|LOL+|W+|CLIP|Pog\w*|EZ|xdd)(?!\w)", re.I)
CLIP_IT_RE = re.compile(r"\bclip (it|that|this)\b", re.I)
VOD_RE = re.compile(r"twitch\.tv/videos/(\d+)")
GQL_URL = "https://gql.twitch.tv/gql"
GQL_CLIENT_ID = "kd1unb4b3q4t58fwlpcbzcbnm76a8fp"  # Twitch's own public web client id (not a secret; no user auth sent)
GQL_HASH = "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"  # VideoCommentsByOffsetOrCursor


def emit(event, **data):
    print(json.dumps({"event": event, **data}), flush=True)


def gql_comments(variables, opener, sleep, tries=4):
    """One page of VOD chat ({edges, pageInfo}). Retries 429 / 5xx / network blips with backoff, then raises."""
    body = json.dumps([{"operationName": "VideoCommentsByOffsetOrCursor", "variables": variables,
                        "extensions": {"persistedQuery": {"version": 1, "sha256Hash": GQL_HASH}}}]).encode()
    req = urllib.request.Request(GQL_URL, body, {"Client-Id": GQL_CLIENT_ID, "Content-Type": "text/plain;charset=UTF-8"})
    for attempt in range(tries):
        try:
            with opener(req, timeout=20) as r:
                d = json.loads(r.read())[0]
            break
        except urllib.error.HTTPError as e:
            if (e.code != 429 and e.code < 500) or attempt == tries - 1:
                raise RuntimeError(f"Twitch chat endpoint answered HTTP {e.code}" + (" (rate-limited)" if e.code == 429 else "")) from None
        except urllib.error.URLError as e:
            if attempt == tries - 1:
                raise RuntimeError(f"Twitch chat endpoint unreachable: {e.reason}") from None
        sleep(2.0 ** attempt)
    if d.get("errors"):
        raise RuntimeError(f"Twitch chat endpoint refused: {d['errors'][0].get('message')}")
    video = (d.get("data") or {}).get("video")
    if not video or not video.get("comments"):
        raise RuntimeError("Twitch returned no chat for this VOD (deleted, subscriber-only or chat disabled)")
    return video["comments"]


def fetch_chat(vod, path, opener=urllib.request.urlopen, sleep=time.sleep):
    """Page through the VOD's whole chat replay (cursor paging, ~50 messages a page) and save it as a
    TwitchDownloader-style JSON that load_chat() reads. Written only when complete, so a failed fetch caches nothing."""
    # shortcut: one request at a time (~50 msgs each, a busy 4 h VOD = thousands); split by contentOffsetSeconds in parallel if that gets slow
    comments, variables, pages = [], {"videoID": vod, "contentOffsetSeconds": 0}, 0
    while True:
        page = gql_comments(variables, opener, sleep)
        edges = page.get("edges") or []
        for e in edges:
            n = e["node"]
            comments.append({"content_offset_seconds": n["contentOffsetSeconds"],
                             "commenter": {"name": (n.get("commenter") or {}).get("login") or ""},  # null = deleted user
                             "message": {"body": "".join(f.get("text") or "" for f in (n.get("message") or {}).get("fragments") or [])}})
        pages += 1
        if pages % 100 == 0:
            emit("stage", stage="metadata", msg=f"Fetching chat replay: {len(comments)} messages, "
                                                f"{scorer.clock(comments[-1]['content_offset_seconds'])} in")
        if not edges or not (page.get("pageInfo") or {}).get("hasNextPage"):
            break
        variables = {"videoID": vod, "cursor": edges[-1]["cursor"]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps({"video": {"id": vod}, "comments": comments}), encoding="utf-8")
    tmp.replace(path)
    return len(comments)


def load_chat(path):
    """-> [(seconds, user, text)] from a TwitchDownloader JSON export or a plain [{"t", "user", "text"}] list."""
    d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(d, dict) and "comments" in d:  # TwitchDownloader
        return [(float(c["content_offset_seconds"]), str((c.get("commenter") or {}).get("name") or (c.get("commenter") or {}).get("_id") or ""),
                 str((c.get("message") or {}).get("body") or "")) for c in d["comments"]]
    return [(float(c["t"]), str(c.get("user", "")), str(c.get("text", ""))) for c in d]


def chat_curve(comments, duration, shift=SHIFT):
    """Per-second spike score: unique chatters + 1.5 x laugh / hype emotes (<= 3 per message) + 5 x "clip it",
    summed over 5 s, divided by the local median (+1) of the surrounding 4 minutes so a raid-sized chat doesn't win
    by volume, then moved `shift` s earlier."""
    n = int(duration) + 1
    users = [set() for _ in range(n)]
    hype, clip_it = np.zeros(n), np.zeros(n)
    for t, user, text in comments:
        s = int(t)
        if 0 <= s < n:
            users[s].add(user)
            hype[s] += min(3, len(EMOTE_RE.findall(text)))
            clip_it[s] += bool(CLIP_IT_RE.search(text))
    raw = np.array([len(u) for u in users], dtype=float) + 1.5 * hype + 5 * clip_it
    burst = np.convolve(raw, np.ones(5), mode="same")
    spike = burst / (scorer.local_median(burst, 120) + 1.0)
    return np.concatenate([spike[shift:], np.zeros(min(shift, n))])[:n]


def pick_moments(curve, duration, k, gap=GAP):
    """The k highest seconds of the curve, >= gap apart, outside the intro / outro (streams > 15 min)."""
    skip = SKIP if duration > SKIP_MIN_DURATION else 0
    picked = []
    for t in np.argsort(-np.asarray(curve), kind="stable"):
        t = int(t)
        if curve[t] <= 0 or len(picked) == k:
            break
        if skip <= t <= duration - skip and all(abs(t - p) >= gap for p in picked):
            picked.append(t)
    return picked


def slice_of(t, duration):
    """(start, end) s around a moment: BEFORE s of setup, AFTER s of reaction, kept inside the video."""
    s = max(0.0, min(t - BEFORE, duration - BEFORE - AFTER))
    return s, min(duration, s + BEFORE + AFTER)


def stream_urls(url, run):
    """Direct media URL(s), resolved once per video: [muxed] for a Twitch VOD (its HLS playlist), [video, audio] for
    YouTube (separate DASH streams). Every slice then seeks straight into them with ffmpeg."""
    fmt = "best[height<=1080]/best" if VOD_RE.search(url) else clipper.VIDEO_FORMAT + "/best[height<=1080]/best"
    r = run(clipper.YTDLP + ["-g", "-f", fmt, url], capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=SLICE_TIMEOUT)
    urls = [u for u in (r.stdout or "").splitlines() if u.startswith("http")]
    if r.returncode or not urls:
        raise RuntimeError(f"yt-dlp couldn't resolve the stream URL: {last_line(r.stderr) or f'exit {r.returncode}'}")
    return urls[:2]


def last_line(text):
    """The last real message in ffmpeg / yt-dlp stderr (skips -stats progress lines)."""
    lines = [x.strip() for x in re.split(r"[\r\n]+", text or "") if x.strip() and not x.lstrip().startswith(("frame=", "size="))]
    return lines[-1] if lines else ""


def slice_cmd(urls, s, e, out, encode=False):
    """ffmpeg input seek (-ss before -i): on HLS it jumps to the segment holding `s` instead of reading from 0 s."""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "warning", "-stats"]
    for u in urls:
        cmd += ["-ss", f"{s:.0f}", "-i", u]
    if len(urls) > 1:
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac"] if encode else ["-c", "copy"]
    return cmd + ["-t", f"{e - s:.0f}", *codec, str(out)]


def download_slice(urls, s, e, out, run):
    """Stream copy, then a re-encode if the copy fails; each attempt is killed after SLICE_TIMEOUT s.
    -> None when `out` holds the slice, else why it failed."""
    why = ""
    for encode in (False, True):
        try:  # stderr captured: the panel reads its stderr pipe only at exit, so -stats spam could fill it and block
            r = run(slice_cmd(urls, s, e, out, encode), capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=SLICE_TIMEOUT)
            if r.returncode == 0 and out.is_file() and out.stat().st_size > 0:
                return None
            why = last_line(r.stderr) or f"ffmpeg exit {r.returncode}"
        except subprocess.TimeoutExpired:
            out.unlink(missing_ok=True)
            return f"ffmpeg timed out after {SLICE_TIMEOUT} s"  # a network stall: re-encoding the same stream won't help
        out.unlink(missing_ok=True)
    return why


def link_at(url, start):
    t = int(start)
    if "twitch.tv" in url:
        return f"{url.split('?')[0]}?t={t // 3600}h{t // 60 % 60}m{t % 60}s"
    return f"{url}{'&' if '?' in url else '?'}t={t}s"


def main(argv=None, run=subprocess.run, opener=urllib.request.urlopen, sleep=time.sleep):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--chat", type=Path, help="chat replay JSON (default: workspace/chat/<vod id>.json)")
    ap.add_argument("--clips", type=int, default=4, help="moments to download and process (default 4)")
    ap.add_argument("--workspace", type=Path, default=ROOT / "workspace")
    ap.add_argument("--cookies-from-browser", metavar="BROWSER")
    ap.add_argument("--cookies", metavar="FILE")
    a, passthrough = ap.parse_known_args(argv)
    clipper.YTDLP.extend(clipper.cookie_args(a))  # YouTube only (metadata + slice downloads); never sent to Twitch

    emit("stage", stage="metadata", msg="Fetching metadata (no video)")
    info, job = clipper.fetch_metadata(a.url, a.workspace)
    duration = float(info.get("duration") or 0)
    url = info.get("webpage_url") or a.url
    if duration <= 0:
        raise RuntimeError("No duration: a live stream or premiere can't be scanned yet")
    vod = VOD_RE.search(url)
    if vod:
        chat_path = a.chat or CHAT_DIR / f"{vod.group(1)}.json"
        if not a.chat and chat_path.is_file():
            emit("log", msg=f"[CHAT] Using the cached chat replay {chat_path.name}")
        elif not a.chat:
            emit("stage", stage="metadata", msg=f"Fetching the chat replay of VOD {vod.group(1)} from Twitch")
            try:
                n = fetch_chat(vod.group(1), chat_path, opener, sleep)
                emit("log", msg=f"[CHAT] Saved {n} chat messages to {chat_path}")
            except Exception as e:  # any failure falls back to the file how-to below
                emit("warning", msg=f"[CHAT] Couldn't fetch the chat automatically: {e}")
        if not Path(chat_path).is_file():
            raise RuntimeError(f"No chat log for VOD {vod.group(1)}: export it with TwitchDownloaderCLI chatdownload --id "
                               f"{vod.group(1)} -o {CHAT_DIR / (vod.group(1) + '.json')} (or pass --chat FILE)")
        comments = load_chat(chat_path)
        curve, source = chat_curve(comments, duration), f"chat ({len(comments)} messages)"
    elif info.get("heatmap"):
        curve, source = scorer.heatmap_curve(info["heatmap"], duration)[0], "Most Replayed heatmap"
    else:
        raise RuntimeError("Neither a Twitch VOD with a chat log nor a YouTube video with a Most Replayed heatmap: "
                           "use Single YouTube URL mode for this one")
    moments = pick_moments(curve, duration, a.clips)
    emit("log", msg=f"{info.get('title')} ({duration / 60:.0f} min): {len(moments)} moment(s) from {source}: "
                    + ", ".join(scorer.clock(t) for t in moments))
    if not moments:
        emit("warning", msg=f"No spikes found in the {source}")
        return 0
    streamer = info.get("uploader") or info.get("channel") or info.get("uploader_id")
    emit("stage", stage="download", msg="Resolving the stream URL")
    urls = stream_urls(url, run)  # once for all slices
    for i, t in enumerate(moments, 1):
        s, e = slice_of(t, duration)
        name = f"{info['id']}_{int(s)}s"
        out = job / f"{name}.mp4"
        emit("stage", stage="download", msg=f"Downloading moment {i}/{len(moments)}: {scorer.clock(s)}-{scorer.clock(e)} only")
        why = download_slice(urls, s, e, out, run)
        if why:
            emit("warning", msg=f"[SLICE] Moment {i} ({scorer.clock(s)}) didn't download, skipped: {why}")
            continue
        meta = {"clip_id": name, "title": info.get("title"), "broadcaster_name": streamer, "creator_name": None,
                "view_count": info.get("view_count"), "url": link_at(url, s), "duration": round(e - s, 2),
                "spike": round(float(curve[t]), 2), "spike_source": source}
        run([sys.executable, "-u", str(HERE / "clipper.py"), str(out), "--twitch-meta", json.dumps(meta), *passthrough], check=False)  # its JSON lines go straight to our stdout; one bad slice doesn't stop the rest
        out.unlink(missing_ok=True)  # clipper copied it into its own job folder
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        emit("error", msg=str(e))
        sys.exit(1)

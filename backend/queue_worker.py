"""Tune-O-Clip queue worker: feed a persistent SQLite queue, then drain it one video at a time with clipper.py.

  python backend/queue_worker.py --add-url URL          # video, playlist or channel URL
  python backend/queue_worker.py --add-file urls.txt    # one URL per line (# comments ok)
  python backend/queue_worker.py --poll-rss feeds.txt   # YouTube channel RSS feed URLs, one per line
  python backend/queue_worker.py --drain                # process until the queue is empty, then exit
  python backend/queue_worker.py --poll-twitch-game "Grand Theft Auto V"   # top Twitch clips (Helix API, .env creds)
  python backend/queue_worker.py --poll-twitch-channels logins.txt        # top clips of these streamers
  python backend/queue_worker.py --poll-twitch-channels logins.txt --twitch-game "Grand Theft Auto V"  # ...in GTA only
  python backend/queue_worker.py --daemon --poll-rss feeds.txt --poll-twitch-game "Grand Theft Auto V"   # forever

Any other options (e.g. --clips 4 --max-length 55) are passed through to clipper.py. Every processed video is
archived to output/<video id>/ (clips + clips_manifest.json) and its workspace folder is deleted.
A video id is never processed twice (seen_videos). When YouTube blocks requests (bot check / 403 / 429) the job
goes back to pending, an alert is sent to DISCORD_WEBHOOK_URL if set, and the worker backs off (daemon) or stops
(drain) instead of hammering YouTube.
stdout is JSON Lines like clipper.py, plus {"event": "queue", ...} status lines.
"""
import argparse
import itertools
import json
import math
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import twitch_api

ROOT = Path(__file__).resolve().parent.parent
CLIPPER = [sys.executable, "-u", str(Path(__file__).resolve().parent / "clipper.py")]
EDITOR = [sys.executable, "-u", str(ROOT / "editor" / "pipeline.py")]
PUBLISHER = [sys.executable, "-u", str(ROOT / "publisher" / "scheduler.py")]
ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/live/|/embed/)([A-Za-z0-9_-]{11})")
BLOCK_RE = re.compile(r"not a bot|bot check|HTTP Error 429|HTTP Error 403|Too Many Requests", re.I)
MAX_ATTEMPTS = 3
ATOM = {"a": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    video_id TEXT NOT NULL UNIQUE,  -- YouTube id, or twitch_<clip id>
    source TEXT NOT NULL DEFAULT 'youtube',
    meta TEXT,  -- JSON (Twitch clip metadata)
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'done', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS seen_videos (video_id TEXT PRIMARY KEY, source TEXT NOT NULL DEFAULT 'youtube',
                                        seen_at TEXT NOT NULL DEFAULT (datetime('now')));
"""
MIGRATIONS = {"jobs": {"source": "TEXT NOT NULL DEFAULT 'youtube'", "meta": "TEXT"},
              "seen_videos": {"source": "TEXT NOT NULL DEFAULT 'youtube'"}}


def emit(event, **data):
    print(json.dumps({"event": event, **data}), flush=True)


def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, isolation_level=None)  # autocommit; each statement is its own transaction
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    for table, cols in MIGRATIONS.items():  # queue.db files created before the Twitch columns existed
        have = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
        for col, decl in cols.items():
            if col not in have:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
    return db


def video_id_from_url(url):
    m = ID_RE.search(url)
    return m.group(1) if m else None


def expand_url(url):
    """Single video -> [id] without any network request; playlist/channel -> ids via yt-dlp --flat-playlist."""
    vid = video_id_from_url(url)
    if vid and "/playlist" not in url:
        return [vid]
    p = subprocess.run([sys.executable, "-m", "yt_dlp", "--flat-playlist", "--dump-single-json", url],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if p.returncode != 0:
        raise RuntimeError(f"Could not expand {url}: {p.stderr.strip()[-300:]}")
    data = json.loads(p.stdout)
    entries = data.get("entries") or [data]
    return [e["id"] for e in entries if isinstance(e, dict) and len(e.get("id") or "") == 11]


def enqueue(db, ids):
    """YouTube ids -> (added, skipped)."""
    return enqueue_items(db, [{"video_id": v, "url": f"https://www.youtube.com/watch?v={v}"} for v in ids])


def enqueue_items(db, items):
    """[{video_id, url, source?, meta?}] -> (added, skipped). Skips ids already processed (seen_videos) or
    already queued (UNIQUE video_id)."""
    added = 0
    for it in items:
        if db.execute("SELECT 1 FROM seen_videos WHERE video_id = ?", (it["video_id"],)).fetchone():
            continue
        meta = json.dumps(it["meta"]) if it.get("meta") else None
        cur = db.execute("INSERT OR IGNORE INTO jobs (url, video_id, source, meta) VALUES (?, ?, ?, ?)",
                         (it["url"], it["video_id"], it.get("source", "youtube"), meta))
        added += cur.rowcount
    return added, len(items) - added


def twitch_items(clips):
    return [{"video_id": f"twitch_{c['id']}", "url": c["url"], "source": "twitch", "meta": twitch_api.clip_meta(c)}
            for c in clips]


TWITCH_VIEW_TIERS = (500, 100)  # step-down when nothing new clears --twitch-min-views
TWITCH_PAGES = 5  # pagination.cursor pages per window while every clip on them is already processed


def poll_twitch(db, client, game=None, channels_file=None, hours=24, min_views=500, limit=20, language="en", channel_game=None,
                per_streamer=0):
    """Enqueue unseen top clips: for `game` across all streamers, and/or for the streamers in channels_file
    (optionally only clips of `channel_game`). Nothing new at min_views: retried at the lower TWITCH_VIEW_TIERS,
    so quiet streamers still get clips. Still nothing: a "warning" event says why, per streamer. -> number added."""
    def game_id(name):
        gid = client.resolve_game_id(name)
        if gid is None:
            emit("warning", msg=f"Twitch game not found: {name!r}")
        return gid

    gid = game_id(game) if game else None
    cgid = game_id(channel_game) if channels_file and channel_game else None
    ids = {}
    if channels_file and not (channel_game and cgid is None):  # a typo'd filter must not fall back to "any game"
        logins = twitch_api.parse_logins(Path(channels_file).read_text(encoding="utf-8-sig"))
        ids = {l: i for l, i in client.user_ids(logins).items() if l in logins}
        for login in [l for l in logins if l not in ids]:
            emit("warning", msg=f"Twitch user not found: {login!r}", login=login)
    stats = {}  # source -> {"returned", "filtered", "skipped"} clip-id sets, over every tier and window
    added = 0
    for tier in [min_views] + [v for v in TWITCH_VIEW_TIERS if v < min_views]:
        if tier != min_views:
            emit("log", msg=f"Twitch: nothing new with >= {prev:g} views, retrying at >= {tier:g}")
        added = _poll_twitch_tier(db, client, gid, game, ids, hours, tier, limit, language, cgid, stats, per_streamer)
        prev = tier
        if added or not (gid or ids):
            break
    if not added and stats:
        breakdown = [{"source": src, "returned": len(s["returned"]), "processed": len(s["skipped"]),
                      "filtered": len(s["filtered"])} for src, s in stats.items()]
        emit("warning", msg="0 new clips: " + "; ".join(
            f"Twitch returned {b['returned']} clips for {b['source']}: {b['processed']} already processed, "
            f"{b['filtered']} filtered by views/lang (0 new)" for b in breakdown), breakdown=breakdown)
    emit("queue", msg=f"Twitch: {added} new clip(s) queued", **counts(db))
    return added


def _poll_twitch_tier(db, client, gid, game, ids, hours, min_views, limit, language, cgid, stats, per_streamer=0):
    moments = known_moments(db)
    # already processed / queued, or the same stream moment as one that is (another viewer clipped it too)
    seen = lambda c: is_known(db, f"twitch_{c['id']}") or same_moment(c, moments)
    new_stats = lambda: {"returned": set(), "filtered": set(), "skipped": set()}
    added = 0
    if gid:
        added += enqueue_items(db, twitch_items(one_per_moment(client.fetch_top_clips(
            game_id=gid, hours_back=hours, min_views=min_views, limit=limit, language=language,
            pages=TWITCH_PAGES, skip=seen, stats=stats.setdefault(game, new_stats()), sort="velocity"), moments)))[0]
    # per_streamer 0 = balanced share of `limit`, so no single streamer hogs the batch
    per = per_streamer or math.ceil(limit / max(1, len(ids)))
    picks = []
    for login, uid in ids.items():
        fresh = []
        # the newest top clips are often already done: widen the window (24 h -> 3 d -> 7 d -> all time) until
        # this streamer has new ones, paging past processed clips inside each window
        for window in [hours] + [w for w in (72, 168) if w > hours] + [None]:
            clips = client.fetch_top_clips(game_id=cgid, broadcaster_id=uid, hours_back=window, min_views=min_views,
                                           limit=per, language=language, pages=TWITCH_PAGES, skip=seen,
                                           stats=stats.setdefault(login, new_stats()),
                                           sort="velocity" if window else "views")  # breakouts first; all time: most viewed
            fresh = [c for c in clips if not seen(c)][:per]
            if fresh:
                if window != hours:
                    emit("log", msg=f"Twitch: {login} had nothing new in {hours:g}h, widened to "
                                    + (f"{window:g}h" if window else "all-time top clips"))
                break
        picks.append(fresh)
    # round-robin: streamer1 clip1, streamer2 clip1, ..., streamer1 clip2, ...
    interleaved = [c for tier in itertools.zip_longest(*picks) for c in tier if c is not None]
    return added + enqueue_items(db, twitch_items(one_per_moment(interleaved, moments)[:limit]))[0]  # limit caps the total


MOMENT_WINDOW = 30  # seconds: two clips of the same VOD this close together are the same moment


def known_moments(db):
    """(VOD id, offset s) of every Twitch clip already queued or processed (from the clip metadata kept with the job)."""
    out = []
    for (meta,) in db.execute("SELECT meta FROM jobs WHERE source = 'twitch' AND meta IS NOT NULL"):
        try:
            m = json.loads(meta)
        except ValueError:
            continue
        if m.get("video_id") and isinstance(m.get("vod_offset"), (int, float)):
            out.append((m["video_id"], m["vod_offset"]))
    return out


def same_moment(clip, moments, window=MOMENT_WINDOW):
    """Same VOD within +-window s of a known moment. Clips without a VOD (deleted / not archived) never match here;
    the clipper's transcript check catches those after Whisper."""
    v, o = clip.get("video_id"), clip.get("vod_offset")
    return bool(v) and isinstance(o, (int, float)) and any(v == mv and abs(o - mo) <= window for mv, mo in moments)


def one_per_moment(clips, moments):
    """Keep the first (best-ranked) clip of each stream moment in this batch; drop later ones and known ones."""
    seen, out = list(moments), []
    for c in clips:
        if same_moment(c, seen):
            emit("log", msg=f"Twitch: {c.get('id')} is the same moment as a clip already picked, skipped")
            continue
        out.append(c)
        if c.get("video_id") and isinstance(c.get("vod_offset"), (int, float)):
            seen.append((c["video_id"], c["vod_offset"]))
    return out


def prune_raw(output, ready, days, now=None):
    """Delete raw clip .mp4s in output/<id>/ older than `days` whose short exists in ready/ (a short only lands there
    after passing QC). Manifests stay. -> deleted paths."""
    cutoff = (time.time() if now is None else now) - days * 86400
    gone = []
    for m in Path(output).glob("*/clips_manifest.json"):
        try:
            data = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for c in data.get("clips") or []:
            raw = m.parent / str(c.get("file") or "")
            short = Path(ready) / f"{data.get('video_id')}_{c.get('clip_id')}_short.mp4"
            if raw.is_file() and raw.suffix == ".mp4" and raw.stat().st_mtime < cutoff and short.is_file():
                raw.unlink()
                gone.append(raw)
    return gone


def is_known(db, video_id):
    """Already processed or already waiting in the queue."""
    return db.execute("SELECT 1 FROM seen_videos WHERE video_id = ? UNION ALL SELECT 1 FROM jobs WHERE video_id = ?",
                      (video_id, video_id)).fetchone() is not None


def read_lines(path):
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip() and not l.lstrip().startswith("#")]


def parse_feed(xml_text):
    """YouTube channel RSS (Atom) -> video ids, newest first as listed."""
    root = ET.fromstring(xml_text)
    return [e.findtext("yt:videoId", namespaces=ATOM) for e in root.findall("a:entry", ATOM)
            if e.findtext("yt:videoId", namespaces=ATOM)]


def poll_feeds(db, feeds_file, fetch=None):
    """Enqueue unseen videos from every feed URL in feeds_file. A broken feed is reported, not fatal."""
    fetch = fetch or (lambda url: urllib.request.urlopen(url, timeout=20).read())
    total = 0
    for url in read_lines(feeds_file):
        try:
            total += enqueue(db, parse_feed(fetch(url)))[0]
        except (OSError, ET.ParseError) as e:
            emit("log", msg=f"RSS feed failed ({url}): {e}")
    emit("queue", msg=f"RSS: {total} new video(s) queued", **counts(db))
    return total


def counts(db):
    rows = dict(db.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())
    return {s: rows.get(s, 0) for s in ("pending", "running", "done", "failed")}


def next_job(db):
    """Claim the oldest pending job (status -> running) or return None."""
    row = db.execute("SELECT * FROM jobs WHERE status = 'pending' ORDER BY id LIMIT 1").fetchone()
    if row:
        db.execute("UPDATE jobs SET status = 'running', attempts = attempts + 1 WHERE id = ?", (row["id"],))
    return row


def alert(msg):
    """POST a JSON alert to DISCORD_WEBHOOK_URL if it is set; never raises."""
    hook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not hook:
        return
    try:
        req = urllib.request.Request(hook, json.dumps({"content": f"Tune-O-Clip: {msg}"[:1900]}).encode(),
                                     {"Content-Type": "application/json", "User-Agent": "tune-o-clip"})
        urllib.request.urlopen(req, timeout=10).close()
    except OSError as e:
        emit("log", msg=f"Discord alert failed: {e}")


def run_clipper(job, clipper_args, workspace, output, clipper_cmd):
    """Run clipper.py on one job, forwarding its JSON lines. -> ("done", manifest) | ("blocked"|"error", msg).
    On success the clips + manifest move to output/<id>/; the workspace folder is always deleted (64 GB SSD)."""
    cmd = clipper_cmd + [job["url"], "--workspace", str(workspace), "--output", str(output), "--cleanup-source"] + clipper_args
    if job["meta"]:
        cmd += ["--twitch-meta", job["meta"]]
    workspace.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in workspace.iterdir()}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    result, tail = None, []
    for line in proc.stdout:
        try:
            ev = json.loads(line)
        except ValueError:
            ev = None
        if isinstance(ev, dict) and ev.get("event") == "done":
            result = ("done", Path(ev["manifest"]))
            continue  # re-emitted below with the archived path
        if isinstance(ev, dict) and ev.get("event") == "error":
            result = ("error", ev.get("msg", ""))
        tail = (tail + [line.strip()])[-20:]
        print(line, end="", flush=True)
    proc.wait()
    if result and result[0] == "done":
        dest = output / job["video_id"]
        if result[1].parent.resolve() != dest.resolve():  # a clipper that left its clips in the workspace: archive here
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True)
            for f in list(result[1].parent.glob("clip_*.mp4")) + [result[1]]:
                shutil.move(str(f), dest / f.name)
            shutil.rmtree(result[1].parent, ignore_errors=True)
        for name in {p.name for p in workspace.iterdir() if p.is_dir()} - before:  # temp files the job left behind
            shutil.rmtree(workspace / name, ignore_errors=True)
        return "done", dest / result[1].name
    # whatever this job created, plus a leftover folder of the same video from a crashed earlier run
    for name in ({p.name for p in workspace.iterdir() if p.is_dir()} - before) | {job["video_id"]}:
        shutil.rmtree(workspace / name, ignore_errors=True)  # partial downloads, info json, analysis stream
    msg = result[1] if result else "clipper exited without a result:\n" + "\n".join(tail[-5:])
    return ("blocked" if BLOCK_RE.search(msg) else "error"), msg


def run_publisher(interval_h, cmd=PUBLISHER, platform="youtube", youtube_privacy="public", tiktok_privacy="PUBLIC_TO_EVERYONE"):
    """--auto-post: post the best unposted short to YouTube Shorts and/or TikTok (official APIs) on each platform whose
    `interval_h` has passed (publisher/scheduler.py paces itself and keeps the history). Its JSON lines are forwarded;
    a publishing problem never stops the queue. -> its exit code."""
    proc = subprocess.Popen(cmd + ["--interval", str(interval_h), "--platform", platform, "--youtube-privacy",
                                   youtube_privacy, "--tiktok-privacy", tiktok_privacy], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


def run_editor(manifest, editor_cmd=EDITOR):
    """Auto-edit: clips -> 9:16 shorts in ready_to_post/ (editor/pipeline.py), forwarding its JSON lines.
    An editing failure never fails the job: the clips are already archived. -> number of shorts made."""
    proc = subprocess.Popen(editor_cmd + ["--manifest", str(manifest)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    shorts = 0
    for line in proc.stdout:
        shorts += '"event": "short"' in line
        print(line, end="", flush=True)
    proc.wait()
    return shorts


MIN_FREE_GB = float(os.environ.get("TOC_MIN_FREE_GB") or 5)  # the mini PC has a 64 GB drive


def disk_free_gb(path):
    """Free space (GB) on the drive holding `path` (or its nearest existing parent)."""
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def drain(db, clipper_args, workspace, output, daemon=False, cooldown=(20, 45), poll=None, poll_every=900, idle=600,
          block_backoff=3600, clipper_cmd=CLIPPER, sleep=time.sleep, clock=time.time, auto_edit=False, editor_cmd=None,
          twitch_cooldown=1.5, free_gb=disk_free_gb):
    """Process pending jobs one at a time. drain mode returns when the queue is empty (or on a block);
    daemon mode calls poll() (RSS / Twitch) at most every poll_every s, idles when empty and backs off after a
    block, forever. -> number processed."""
    last_poll = -float("inf")
    db.execute("UPDATE jobs SET status = 'pending', attempts = MAX(attempts - 1, 0) WHERE status = 'running'")  # crashed run
    processed, low_disk = 0, False
    while True:
        if poll and daemon and clock() - last_poll >= poll_every:
            last_poll = clock()
            poll()  # also prunes raw clips (--prune-raw-days), which may free space
        free = free_gb(output)
        if free < MIN_FREE_GB:  # pause before downloading: a full drive breaks renders and the queue db
            msg = f"Only {free:.1f} GB free (< {MIN_FREE_GB:g} GB): harvesting paused. Free space or prune output/."
            emit("warning", msg=msg + (f" Checking again in {idle:g}s." if daemon else ""), **counts(db))
            if not low_disk:
                alert(msg)
            low_disk = True
            if not daemon:
                return processed
            sleep(idle)
            continue
        low_disk = False
        job = next_job(db)
        if job is None:
            if not daemon:
                emit("queue", msg="Queue empty", **counts(db))
                return processed
            emit("queue", msg=f"Queue empty, sleeping {idle}s", **counts(db))
            sleep(idle)
            continue
        emit("queue", msg=f"Processing {job['video_id']} (attempt {job['attempts'] + 1})", current=job["url"], **counts(db))
        status, info = run_clipper(job, clipper_args, workspace, output, clipper_cmd)
        if status == "done":
            db.execute("UPDATE jobs SET status = 'done', error = NULL, finished_at = datetime('now') WHERE id = ?", (job["id"],))
            db.execute("INSERT OR IGNORE INTO seen_videos (video_id, source) VALUES (?, ?)", (job["video_id"], job["source"]))
            processed += 1
            emit("done", manifest=str(info), video_id=job["video_id"])
            if auto_edit:
                emit("queue", msg=f"Auto-editing {job['video_id']} into shorts", **counts(db))
                made = run_editor(info, editor_cmd or EDITOR)
                emit("queue", msg=f"{job['video_id']}: {made} short(s) in ready_to_post/", **counts(db))
        elif status == "blocked":
            # not the video's fault: put it back without spending an attempt, alert, and stop pushing
            db.execute("UPDATE jobs SET status = 'pending', attempts = attempts - 1, error = ? WHERE id = ?", (info[:2000], job["id"]))
            alert(f"YouTube blocked requests on {job['video_id']}: {info[:300]}")
            emit("queue", msg=f"YouTube is blocking requests: {'backing off ' + str(block_backoff) + 's' if daemon else 'stopping'}. "
                              "Refresh cookies.txt or wait.", **counts(db))
            if not daemon:
                return processed
            sleep(block_backoff)
            continue
        else:
            failed = job["attempts"] + 1 >= MAX_ATTEMPTS
            db.execute("UPDATE jobs SET status = ?, error = ?, finished_at = CASE WHEN ? THEN datetime('now') END WHERE id = ?",
                       ("failed" if failed else "pending", info[:2000], failed, job["id"]))
            if failed:
                alert(f"{job['video_id']} failed {MAX_ATTEMPTS}x: {info[:300]}")
            emit("queue", msg=f"{job['video_id']} {'failed for good' if failed else 'failed, will retry'}: {info[:200]}", **counts(db))
        if db.execute("SELECT 1 FROM jobs WHERE status = 'pending'").fetchone() or daemon:
            # the 20-45 s pause only protects YouTube scraping; Twitch clips come from the official API + CDN
            wait = twitch_cooldown if job["source"] == "twitch" else random.uniform(*cooldown)
            emit("queue", msg=f"Cooldown {wait:.3g}s", **counts(db))
            sleep(wait)


def last_job_id(db):
    return db.execute("SELECT COALESCE(MAX(id), 0) FROM jobs").fetchone()[0]


def kept_clips(db, output, after_id):
    """Clips that passed the AI judge in jobs queued after job id `after_id` (from their archived manifests)."""
    n = 0
    for (vid,) in db.execute("SELECT video_id FROM jobs WHERE status = 'done' AND id > ?", (after_id,)):
        try:
            n += len(json.loads((Path(output) / vid / "clips_manifest.json").read_text(encoding="utf-8")).get("clips") or [])
        except (OSError, ValueError):
            pass
    return n


def retry_unfunny(db, output, after_id, poll, drain_again, rounds):
    """--drain on Twitch: while no clip of this run has passed the AI judge, poll for more clips (processed ones and
    the same stream moments are skipped, quiet streamers' windows widen) and drain again, up to `rounds` more times.
    Stops early when Twitch has nothing new left. -> clips kept."""
    for r in range(1, rounds + 1):
        if kept_clips(db, output, after_id):
            return kept_clips(db, output, after_id)
        emit("log", msg=f"[RETRY] No clip passed the AI judge yet: fetching more Twitch clips (round {r}/{rounds})")
        mark = last_job_id(db)
        poll()
        if last_job_id(db) == mark:  # poll_twitch has already warned why ("0 new clips: ...")
            emit("log", msg="[RETRY] Twitch has no new clips left to try with these settings: stopping")
            return 0
        drain_again()
    kept = kept_clips(db, output, after_id)
    if not kept and rounds:
        emit("warning", msg=f"[JUDGE] Tried {rounds} more round(s) of Twitch clips: none passed the AI judge")
    return kept


def parse_range(text):
    lo, _, hi = text.partition("-")
    lo, hi = float(lo), float(hi or lo)
    if not 0 <= lo <= hi:
        raise argparse.ArgumentTypeError("use SECONDS or MIN-MAX, e.g. 20-45")
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--add-url", action="append", default=[], metavar="URL", help="video, playlist or channel URL (repeatable)")
    ap.add_argument("--add-file", action="append", default=[], metavar="FILE", help="text file with one URL per line")
    ap.add_argument("--poll-rss", metavar="FILE", help="text file with YouTube channel RSS feed URLs")
    ap.add_argument("--poll-twitch-game", metavar="NAME", help='top Twitch clips for a game, e.g. "Grand Theft Auto V"')
    ap.add_argument("--poll-twitch-channels", metavar="FILE",
                    help="file with Twitch logins (comma/newline separated; @name and twitch.tv URLs ok)")
    ap.add_argument("--twitch-game", metavar="NAME", help="with --poll-twitch-channels: only clips of this game (default: any)")
    ap.add_argument("--twitch-hours", type=float, default=24, help="clips created in the last N hours (default 24)")
    ap.add_argument("--twitch-min-views", type=int, default=500, help="skip clips with fewer views (default 500)")
    ap.add_argument("--twitch-limit", type=int, default=20, help="max clips per game/channel per poll (default 20)")
    ap.add_argument("--twitch-per-streamer", type=int, default=0,
                    help="clips per streamer per poll (default 0 = --twitch-limit split evenly; the total stays capped)")
    ap.add_argument("--twitch-lang", default="en", help='clip language filter ("" = any, default en)')
    ap.add_argument("--poll-every", type=float, default=900, help="daemon: seconds between RSS/Twitch polls (default 900)")
    ap.add_argument("--auto-post", action="store_true",
                    help="post the best unposted short (official APIs) whenever --post-interval has passed")
    ap.add_argument("--post-interval", type=float, default=3.5, help="hours between posts, per platform (default 3.5)")
    ap.add_argument("--post-platform", default="youtube", choices=["youtube", "tiktok", "both"], help="--auto-post target")
    ap.add_argument("--youtube-privacy", default="public", choices=["public", "unlisted", "private"])
    ap.add_argument("--tiktok-privacy", default="PUBLIC_TO_EVERYONE",
                    choices=["PUBLIC_TO_EVERYONE", "FOLLOWER_OF_CREATOR", "MUTUAL_FOLLOW_FRIENDS", "SELF_ONLY"])
    ap.add_argument("--auto-edit", action="store_true", help="turn every finished video's clips into 9:16 shorts (ready_to_post/)")
    ap.add_argument("--drain", action="store_true", help="process the queue until it is empty, then exit")
    ap.add_argument("--daemon", action="store_true", help="run forever (polls --poll-rss feeds every loop)")
    ap.add_argument("--cooldown", type=parse_range, default=(20.0, 45.0), metavar="MIN-MAX", help="seconds between videos (default 20-45)")
    ap.add_argument("--twitch-cooldown", type=float, default=1.5, help="seconds after a Twitch clip job (default 1.5; no scraping)")
    ap.add_argument("--idle", type=float, default=600, help="daemon: seconds to sleep when the queue is empty (default 600)")
    ap.add_argument("--prune-raw-days", type=float, default=0, metavar="N",
                    help="delete raw clips in output/ older than N days whose short passed QC (0 = off; 7 suits the 64 GB box)")
    ap.add_argument("--retry-rounds", type=int, default=10, metavar="N",
                    help="--drain on Twitch: while no clip passed the AI judge, fetch more clips and retry up to N "
                         "times (default 10, 0 = off)")
    ap.add_argument("--block-backoff", type=float, default=3600, help="daemon: seconds to back off after a YouTube block (default 3600)")
    ap.add_argument("--workspace", type=Path, default=ROOT / "workspace", help="temp work dir; queue.db lives here")
    ap.add_argument("--output", type=Path, default=ROOT / "output", help="archive: output/<video id>/")
    ap.add_argument("--status", action="store_true", help="print queue counts and exit")
    a, clipper_args = ap.parse_known_args()
    twitch_api.load_env()  # .env also holds DISCORD_WEBHOOK_URL, needed for YouTube-only runs too

    db = connect(a.workspace / "queue.db")
    for url in a.add_url + [u for f in a.add_file for u in read_lines(f)]:
        try:
            added, skipped = enqueue(db, expand_url(url))
            emit("queue", msg=f"{url}: {added} queued, {skipped} already seen/queued", **counts(db))
        except RuntimeError as e:
            emit("log", msg=str(e))
    twitch = None
    if a.poll_twitch_game or a.poll_twitch_channels:
        twitch = twitch_api.Twitch(token_path=a.workspace / "twitch_token.json")

    def poll():
        if a.prune_raw_days > 0:
            gone = prune_raw(a.output, ROOT / "ready_to_post", a.prune_raw_days)
            if gone:
                emit("log", msg=f"Pruned {len(gone)} raw clip(s) older than {a.prune_raw_days:g} days (their shorts are in ready_to_post/)")
        if a.poll_rss:
            poll_feeds(db, a.poll_rss)
        if twitch:
            try:
                poll_twitch(db, twitch, a.poll_twitch_game, a.poll_twitch_channels, a.twitch_hours,
                            a.twitch_min_views, a.twitch_limit, a.twitch_lang or None, a.twitch_game,
                            a.twitch_per_streamer)
            except (OSError, RuntimeError, KeyError, ValueError) as e:  # Twitch down / bad creds: keep the daemon alive
                emit("log", msg=f"Twitch poll failed: {e}")
                if not a.daemon:
                    raise
        if a.auto_post and a.daemon:  # every daemon cycle; the publisher waits out the interval itself
            run_publisher(a.post_interval, PUBLISHER, a.post_platform, a.youtube_privacy, a.tiktok_privacy)

    start_id = last_job_id(db)  # this run's Twitch clips are queued after it
    if not a.daemon:
        poll()
    if a.status:
        emit("queue", msg="Status", **counts(db))
    if a.drain or a.daemon:
        run = lambda: drain(db, clipper_args, a.workspace, a.output, daemon=a.daemon, cooldown=a.cooldown,
                            twitch_cooldown=a.twitch_cooldown, poll=poll, poll_every=a.poll_every, idle=a.idle,
                            block_backoff=a.block_backoff, auto_edit=a.auto_edit)
        run()
        if twitch and not a.daemon:  # the daemon polls again by itself every --poll-every
            retry_unfunny(db, a.output, start_id, poll, run, a.retry_rounds)
    if a.auto_post and not a.daemon:  # one check once the batch is done (new shorts may be waiting)
        run_publisher(a.post_interval, PUBLISHER, a.post_platform, a.youtube_privacy, a.tiktok_privacy)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        emit("error", msg=str(e))
        sys.exit(1)

"""Self-check for queue_worker.py: python backend/test_queue.py  (no network: a fake clipper stands in)"""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import queue_worker as qw

FAKE_CLIPPER = r'''
import json, sys
from pathlib import Path
url, ws = sys.argv[1], Path(sys.argv[sys.argv.index("--workspace") + 1])
vid = url.rsplit("=", 1)[1]
with open(ws.parent / "calls.txt", "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
print(json.dumps({"event": "stage", "stage": "metadata", "msg": "Fetching metadata"}), flush=True)
if vid.startswith("BLOCKED"):
    print(json.dumps({"event": "error", "msg": "YouTube's bot check blocked this IP"})); sys.exit(1)
if vid.startswith("ERROR"):
    print(json.dumps({"event": "error", "msg": "ffmpeg exploded"})); sys.exit(1)
if vid.startswith("ARCH"):  # today's clipper: archives into --output itself, leaves only temp files in the workspace
    (ws / vid).mkdir(parents=True, exist_ok=True)
    (ws / vid / "source.info.json").write_text("{}")
    out = Path(sys.argv[sys.argv.index("--output") + 1]) / vid
    out.mkdir(parents=True, exist_ok=True)
    (out / "clip_01_10s.mp4").write_bytes(b"x")
    (out / "clips_manifest.json").write_text(json.dumps({"video_id": vid, "clips": [{"clip_id": 1, "file": "clip_01_10s.mp4"}]}))
    print(json.dumps({"event": "done", "manifest": str(out / "clips_manifest.json"), "count": 1}), flush=True)
    sys.exit(0)
job = ws / vid
job.mkdir(parents=True, exist_ok=True)
(job / "clip_01_10s.mp4").write_bytes(b"x")
(job / "source.info.json").write_text("{}")
(job / "analysis.mp4").write_bytes(b"tmp")
m = job / "clips_manifest.json"
m.write_text(json.dumps({"video_id": vid, "clips": [{"clip_id": 1, "file": "clip_01_10s.mp4"}]}))
print(json.dumps({"event": "done", "manifest": str(m), "count": 1}), flush=True)
'''

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015" xmlns:media="http://search.yahoo.com/mrss/" xmlns="http://www.w3.org/2005/Atom">
 <title>Some Channel</title>
 <entry><id>yt:video:AAAAAAAAAAA</id><yt:videoId>AAAAAAAAAAA</yt:videoId><title>new one</title></entry>
 <entry><id>yt:video:BBBBBBBBBBB</id><yt:videoId>BBBBBBBBBBB</yt:videoId><title>older</title></entry>
</feed>"""

tmp = Path(tempfile.mkdtemp())
ws, out = tmp / "workspace", tmp / "output"
fake = tmp / "fake_clipper.py"
fake.write_text(FAKE_CLIPPER)
CLIP = [sys.executable, str(fake)]
db = qw.connect(ws / "queue.db")
alerts, sleeps = [], []
qw.alert = alerts.append  # record instead of POSTing
quiet = lambda: contextlib.redirect_stdout(io.StringIO())


def run(**kw):
    with quiet():
        return qw.drain(db, ["--clips", "4"], ws, out, cooldown=(0, 0), clipper_cmd=CLIP, sleep=sleeps.append, **kw)


def status(vid):
    return dict(db.execute("SELECT status, attempts FROM jobs WHERE video_id = ?", (vid,)).fetchone())


# --- URL parsing: single videos never hit the network ---------------------------------------------------------
for url in ["https://www.youtube.com/watch?v=pKSST4FLYpo&pp=ygUO", "https://youtu.be/pKSST4FLYpo?t=3",
            "https://www.youtube.com/shorts/pKSST4FLYpo", "https://www.youtube.com/watch?v=pKSST4FLYpo&list=PL123"]:
    assert qw.expand_url(url) == ["pKSST4FLYpo"], url
assert qw.video_id_from_url("https://www.youtube.com/@somechannel/videos") is None
assert qw.parse_range("20-45") == (20, 45) and qw.parse_range("30") == (30, 30)

# --- RSS feed parsing + dedup on re-poll ----------------------------------------------------------------------
assert qw.parse_feed(FEED) == ["AAAAAAAAAAA", "BBBBBBBBBBB"]
feeds = tmp / "feeds.txt"
feeds.write_text("# my channels\nhttps://www.youtube.com/feeds/videos.xml?channel_id=UC1\nhttps://broken.example/feed\n")
fetch = lambda url: FEED if "UC1" in url else (_ for _ in ()).throw(OSError("404"))
with quiet():
    assert qw.poll_feeds(db, feeds, fetch) == 2  # broken feed reported, not fatal
    assert qw.poll_feeds(db, feeds, fetch) == 0  # nothing new the second time

# --- sequential execution, archive + cleanup, never twice -----------------------------------------------------
assert run() == 2
assert status("AAAAAAAAAAA") == {"status": "done", "attempts": 1} and status("BBBBBBBBBBB")["status"] == "done"
calls = [json.loads(l) for l in (tmp / "calls.txt").read_text().splitlines()]
assert [c[0][-11:] for c in calls] == ["AAAAAAAAAAA", "BBBBBBBBBBB"]  # oldest first, one at a time
assert all("--cleanup-source" in c and c[c.index("--clips") + 1] == "4" for c in calls)  # forced + passthrough
assert all(c[c.index("--output") + 1] == str(out) for c in calls)  # the clipper archives into the queue's output/
assert sorted(p.name for p in (out / "AAAAAAAAAAA").iterdir()) == ["clip_01_10s.mp4", "clips_manifest.json"]
assert not (ws / "AAAAAAAAAAA").exists()  # info json, analysis stream etc. purged
assert sleeps == [0]  # cooldown only between videos, not after the last one
assert qw.enqueue(db, ["AAAAAAAAAAA", "AAAAAAAAAAA", "CCCCCCCCCCC"]) == (1, 2)  # seen + duplicate skipped
db.execute("DELETE FROM jobs WHERE video_id = 'AAAAAAAAAAA'")
assert qw.enqueue(db, ["AAAAAAAAAAA"]) == (0, 1)  # seen_videos remembers even after the job row is gone

# --- YouTube block: job goes back without burning an attempt, alert sent, drain stops --------------------------
db.execute("DELETE FROM jobs WHERE video_id = 'CCCCCCCCCCC'")
qw.enqueue(db, ["BLOCKEDxxxx", "DDDDDDDDDDD"])
assert run() == 0
assert status("BLOCKEDxxxx") == {"status": "pending", "attempts": 0}
assert status("DDDDDDDDDDD") == {"status": "pending", "attempts": 0}  # stopped instead of pushing on
assert len(alerts) == 1 and "blocked" in alerts[0]
db.execute("DELETE FROM jobs WHERE video_id = 'BLOCKEDxxxx'")

# --- ordinary errors retry up to MAX_ATTEMPTS, then fail; workspace still cleaned -----------------------------
db.execute("DELETE FROM jobs WHERE video_id = 'DDDDDDDDDDD'")
qw.enqueue(db, ["ERRORxxxxxx"])
(ws / "ERRORxxxxxx").mkdir(parents=True)
assert run() == 0
assert status("ERRORxxxxxx") == {"status": "failed", "attempts": qw.MAX_ATTEMPTS}
assert not (ws / "ERRORxxxxxx").exists() and len(alerts) == 2

# --- crash recovery: a job left 'running' by a killed worker is picked up again -------------------------------
qw.enqueue(db, ["EEEEEEEEEEE"])
db.execute("UPDATE jobs SET status = 'running', attempts = 1 WHERE video_id = 'EEEEEEEEEEE'")
assert run() == 1 and status("EEEEEEEEEEE") == {"status": "done", "attempts": 1}
assert qw.counts(db) == {"pending": 0, "running": 0, "done": 2, "failed": 1}  # A's row was deleted above

# --- --auto-edit: the archived manifest goes to the editor; an editor failure never fails the job ----------------
fake_editor = tmp / "fake_editor.py"
fake_editor.write_text(
    "import json, sys, pathlib\n"
    "m = sys.argv[sys.argv.index('--manifest') + 1]\n"
    "pathlib.Path(sys.argv[0]).with_name('editor_calls.txt').open('a').write(m + '\\n')\n"
    "print(json.dumps({'event': 'short', 'path': 'x_short.mp4'}))\n"
    "print(json.dumps({'event': 'error', 'msg': 'clip 2 failed QC'}))\n"
    "sys.exit(1)\n")
qw.enqueue(db, ["FFFFFFFFFFF"])
with quiet() as log:
    assert qw.drain(db, [], ws, out, cooldown=(0, 0), clipper_cmd=CLIP, sleep=sleeps.append,
                    auto_edit=True, editor_cmd=[sys.executable, str(fake_editor)]) == 1
edited = (tmp / "editor_calls.txt").read_text().split()
assert edited == [str(out / "FFFFFFFFFFF" / "clips_manifest.json")], edited  # archived path, not the workspace one
assert status("FFFFFFFFFFF")["status"] == "done" and "1 short(s) in ready_to_post/" in log.getvalue()

# --- unified storage: a clipper that archived into --output itself is not moved again; its temp folder goes ----------
db_a = qw.connect(tmp / "arch_ws" / "queue.db")
qw.enqueue(db_a, ["ARCHAAAAAAA"])
with quiet():
    assert qw.drain(db_a, [], tmp / "arch_ws", tmp / "arch_out", cooldown=(0, 0), clipper_cmd=CLIP, sleep=lambda s: None) == 1
assert sorted(p.name for p in (tmp / "arch_out" / "ARCHAAAAAAA").iterdir()) == ["clip_01_10s.mp4", "clips_manifest.json"]
assert not (tmp / "arch_ws" / "ARCHAAAAAAA").exists(), "temp files of the job removed"

# --- cooldowns: the 20-45 s YouTube pause after a YouTube job, 1.5 s after a Twitch clip (no scraping to protect) ---
db_t = qw.connect(tmp / "tw_ws" / "queue.db")
qw.enqueue_items(db_t, [{"video_id": "YYYYYYYYYYY", "url": "https://www.youtube.com/watch?v=YYYYYYYYYYY"}]
                 + [{"video_id": f"twitch_T{i}", "url": f"https://clips.twitch.tv/T{i}?v=T{i}", "source": "twitch",
                     "meta": {"clip_id": f"T{i}"}} for i in (1, 2, 3)])
naps = []
with quiet() as log:
    assert qw.drain(db_t, [], tmp / "tw_ws", tmp / "tw_out", cooldown=(30, 30), clipper_cmd=CLIP, sleep=naps.append) == 4
assert naps == [30, 1.5, 1.5], naps  # after YouTube, Twitch, Twitch; none after the last job
assert "Cooldown 1.5s" in log.getvalue() and "Cooldown 30s" in log.getvalue()

# --- --prune-raw-days: an old raw clip goes only once its short passed QC (= is in ready_to_post); manifests stay ---
import os
import time
pr_out, pr_ready = tmp / "pr_out", tmp / "pr_ready"
(pr_out / "v1").mkdir(parents=True)
pr_ready.mkdir()
names = ["clip_01_5s.mp4", "clip_02_9s.mp4", "clip_03_1s.mp4"]
for n in names:
    (pr_out / "v1" / n).write_bytes(b"x")
(pr_out / "v1" / "clips_manifest.json").write_text(json.dumps({"video_id": "v1", "clips": [
    {"clip_id": i + 1, "file": n} for i, n in enumerate(names)]}))
(pr_ready / "v1_1_short.mp4").write_bytes(b"x")
(pr_ready / "v1_3_short.mp4").write_bytes(b"x")
eight_days = time.time() - 8 * 86400
for n in names[:2]:
    os.utime(pr_out / "v1" / n, (eight_days, eight_days))
assert [g.name for g in qw.prune_raw(pr_out, pr_ready, 7)] == ["clip_01_5s.mp4"]  # 02: no short yet; 03: too new
assert all((pr_out / "v1" / n).exists() for n in names[1:]) and (pr_out / "v1" / "clips_manifest.json").exists()
assert qw.prune_raw(pr_out, pr_ready, 7) == [] and qw.prune_raw(pr_out, pr_ready, 30) == []
# --- --auto-post: the publisher is called with the interval and its lines are forwarded ---------------------------------
fake_pub = tmp / "fake_pub.py"
fake_pub.write_text('import json, sys\nprint(json.dumps({"event": "log", "msg": "[PUBLISH] args " + " ".join(sys.argv[1:])}))\nsys.exit(0)\n')
with quiet() as log:
    assert qw.run_publisher(2.5, [sys.executable, str(fake_pub)], "both", "unlisted", "SELF_ONLY") == 0
assert "[PUBLISH] args --interval 2.5 --platform both --youtube-privacy unlisted --tiktok-privacy SELF_ONLY" in log.getvalue(), log.getvalue()
# --- retry while nothing passed the AI judge: poll more Twitch clips, drain again, stop when kept / Twitch runs dry --------
db_r, out_r = qw.connect(tmp / "retry_ws" / "queue.db"), tmp / "retry_out"
start_r = qw.last_job_id(db_r)
verdicts = iter([0, 0, 2])  # clips kept per job: rounds 1 and 2 rejected, round 3 keeps two
polls = []


def r_poll(new=True):
    polls.append(1)
    if new:
        qw.enqueue_items(db_r, [{"video_id": f"twitch_r{len(polls)}", "url": "u", "source": "twitch"}])


def r_drain():
    while (job := qw.next_job(db_r)) is not None:
        (out_r / job["video_id"]).mkdir(parents=True, exist_ok=True)
        (out_r / job["video_id"] / "clips_manifest.json").write_text(json.dumps({"clips": [{}] * next(verdicts)}))
        db_r.execute("UPDATE jobs SET status = 'done' WHERE id = ?", (job["id"],))


r_poll()
r_drain()
with quiet() as log:
    assert qw.retry_unfunny(db_r, out_r, start_r, r_poll, r_drain, 5) == 2
assert len(polls) == 3 and log.getvalue().count("[RETRY] No clip passed") == 2, log.getvalue()
with quiet():
    assert qw.retry_unfunny(db_r, out_r, start_r, r_poll, r_drain, 5) == 2 and len(polls) == 3, "already kept: no retry"
# Twitch runs dry -> stop at once; rounds used up -> a [JUDGE] warning (the banner's "lower Min Comedy Score" help)
db_r2 = qw.connect(tmp / "retry_ws2" / "queue.db")
with quiet() as log:
    assert qw.retry_unfunny(db_r2, out_r, 0, lambda: None, r_drain, 5) == 0
assert "no new clips left" in log.getvalue() and "[JUDGE]" not in log.getvalue()
verdicts = iter([0] * 10)
polls.clear()
db_r = qw.connect(tmp / "retry_ws3" / "queue.db")
with quiet() as log:
    assert qw.retry_unfunny(db_r, out_r, 0, r_poll, r_drain, 3) == 0
assert len(polls) == 3 and '"msg": "[JUDGE] Tried 3 more round(s) of Twitch clips: none passed the AI judge"' in log.getvalue(), log.getvalue()
# disk guard: under MIN_FREE_GB nothing is downloaded; the job stays pending for when space is back
db_d = qw.connect(tmp / "disk_ws" / "queue.db")
qw.enqueue(db_d, ["DDDDDDDDDDD"])
with quiet() as log:
    assert qw.drain(db_d, [], tmp / "disk_ws", tmp / "disk_out", cooldown=(0, 0), clipper_cmd=CLIP, sleep=lambda s: None,
                    free_gb=lambda p: 1.2) == 0
assert "harvesting paused" in log.getvalue() and db_d.execute("SELECT status FROM jobs").fetchone()[0] == "pending"
frees = iter([1.0, 1.0, 50.0, 50.0, 50.0])
naps = []
with quiet() as log:  # daemon: waits it out, then carries on (stopped here by the empty queue raising out of sleep)
    try:
        qw.drain(db_d, [], tmp / "disk_ws", tmp / "disk_out", daemon=True, idle=7, cooldown=(0, 0), clipper_cmd=CLIP,
                 sleep=lambda s: (naps.append(s), s == 7 and len(naps) > 2 and (_ for _ in ()).throw(KeyboardInterrupt)),
                 free_gb=lambda p: next(frees, 50.0))
    except KeyboardInterrupt:
        pass
assert naps[:2] == [7, 7] and db_d.execute("SELECT status FROM jobs").fetchone()[0] == "done", (naps, log.getvalue()[-300:])
assert log.getvalue().count("harvesting paused") == 2
assert qw.disk_free_gb(tmp / "does" / "not" / "exist") > 0  # nearest existing parent

print("queue ok")

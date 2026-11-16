"""Self-check for the Twitch path: python backend/test_twitch.py  (fully mocked: no network, no Twitch credentials)"""
import argparse
import contextlib
import io
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import urllib.error
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import clipper
import queue_worker as qw
import scorer
import twitch_api

tmp = Path(tempfile.mkdtemp())
quiet = lambda: contextlib.redirect_stdout(io.StringIO())
import history  # noqa: E402
import os  # noqa: E402
os.environ["TOC_HISTORY"] = str(tmp / "posted_history.json")  # child processes too: never the real file
history.HISTORY = tmp / "posted_history.json"
real_remember = history.remember
history.remember = lambda *a, **k: None  # the checks below re-run one clip many times; the 30-day guard is tested at the end
CARD = {"title_angles": ["He came back in the dinosaur suit and chat lost it", "Nobody expected the dinosaur suit to come back"],
        "hashtags": ["#streamer1", "#dinosaur", "#funny", "#shorts", "#twitch"], "pinned_comment": "Would you have kept a straight face?"}
scorer.ollama_metadata = lambda c, model, streamer=None: dict(CARD)


class Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def http_error(code, headers=None):
    return urllib.error.HTTPError("https://api.twitch.tv/x", code, "err", headers or {}, None)


CLIPS = [
    {"id": "FunnyClip1", "url": "https://clips.twitch.tv/FunnyClip1", "title": "cop gets trolled", "language": "en",
     "view_count": 9000, "broadcaster_name": "Streamer1", "creator_name": "Fan1", "duration": 28.4, "game_id": "32982"},
    {"id": "LowViews", "url": "https://clips.twitch.tv/LowViews", "language": "en", "view_count": 100, "game_id": "32982"},
    {"id": "Spanish", "url": "https://clips.twitch.tv/Spanish", "language": "es", "view_count": 5000, "game_id": "32982"},
    {"id": "FunnyClip2", "url": "https://clips.twitch.tv/FunnyClip2", "title": "car flip", "language": "en",
     "view_count": 12000, "broadcaster_name": "Streamer2", "duration": 40.0, "game_id": "32982"},
    {"id": "JustChatting", "url": "https://clips.twitch.tv/JustChatting", "title": "react", "language": "en",
     "view_count": 20000, "broadcaster_name": "Streamer1", "duration": 30.0, "game_id": "509658"},
]


class FakeTwitchServer:
    """Stands in for urllib.request.urlopen: token endpoint + Helix endpoints, with scripted failures."""
    def __init__(self):
        self.tokens, self.requests, self.fail = 0, [], []

    def __call__(self, req, timeout=None):
        url = req.full_url
        if url.startswith(twitch_api.TOKEN_URL):
            if getattr(self, "bad_client", False):
                raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b'{"status":400,"message":"invalid client"}'))
            self.tokens += 1
            return Resp(json.dumps({"access_token": f"tok{self.tokens}", "expires_in": 3600}).encode())
        self.requests.append((url, req.get_header("Authorization")))
        if self.fail:
            raise self.fail.pop(0)
        path, q = urlparse(url).path, parse_qs(urlparse(url).query)
        if path.endswith("/games"):
            data = [{"id": "32982", "name": q["name"][0]}] if q["name"][0] == "Grand Theft Auto V" else []
        elif path.endswith("/users"):
            data = [{"id": f"id_{l.lower()}", "login": l.lower()} for l in q["login"] if l != "ghost"]  # Helix logins are lowercase
        elif "game_id" in q:  # like Helix: one game across all streamers
            data = [c for c in CLIPS if c.get("game_id") == q["game_id"][0]]
        else:  # like Helix: one streamer across all games
            data = [c for c in CLIPS if f"id_{(c.get('broadcaster_name') or '').lower()}" == q["broadcaster_id"][0]]
        return Resp(json.dumps({"data": data}).encode())


# --- .env: BOM from Notepad, comments, quotes; real environment variables win ---------------------------------------
import os
env = tmp / ".env"
env.write_bytes(b'\xef\xbb\xbfTT_ID=abc\n# TT_SKIP=1\nTT_QUOTED = "q v"\nTT_REAL=from_file\n')  # BOM + lines
os.environ["TT_REAL"] = "from_env"
twitch_api.load_env(env)
assert os.environ["TT_ID"] == "abc" and os.environ["TT_QUOTED"] == "q v" and "TT_SKIP" not in os.environ
assert os.environ["TT_REAL"] == "from_env"

now = [1_800_000_000.0]
sleeps = []
server = FakeTwitchServer()
tw = twitch_api.Twitch("cid", "secret", tmp / "ws" / "twitch_token.json", opener=server, sleep=sleeps.append, clock=lambda: now[0])

# --- token: fetched once, cached on disk, refreshed when expired or on 401 -----------------------------------------
assert tw.resolve_game_id("Grand Theft Auto V") == "32982" and server.tokens == 1
assert tw.resolve_game_id("Not A Game") is None and server.tokens == 1  # cached
assert server.requests[-1][1] == "Bearer tok1"
cached = json.loads((tmp / "ws" / "twitch_token.json").read_text())
assert cached["access_token"] == "tok1" and cached["expires_at"] == now[0] + 3600
tw2 = twitch_api.Twitch("cid", "secret", tmp / "ws" / "twitch_token.json", opener=server, clock=lambda: now[0])
tw2.resolve_game_id("Grand Theft Auto V")
assert server.tokens == 1  # a new process reuses the cached token
now[0] += 3600  # expired (inside the 60 s margin)
tw.resolve_game_id("Grand Theft Auto V")
assert server.tokens == 2 and server.requests[-1][1] == "Bearer tok2"
server.fail = [http_error(401)]  # revoked token: refresh once and retry
assert tw.resolve_game_id("Grand Theft Auto V") == "32982"
assert server.tokens == 3 and server.requests[-1][1] == "Bearer tok3"

server.bad_client = True  # wrong Client ID: a clear message with Twitch's reason, no secret in it
try:
    tw.token(force=True)
    raise AssertionError("should have raised")
except RuntimeError as e:
    assert "invalid client" in str(e) and "secret" not in str(e).split("Check")[0], str(e)
server.bad_client = False

# --- 429: wait until Ratelimit-Reset (capped), then retry; give up after `tries` ---------------------------------
server.fail = [http_error(429, {"Ratelimit-Reset": str(int(now[0]) + 7)}), http_error(429)]
assert tw.resolve_game_id("Grand Theft Auto V") == "32982"
assert sleeps == [7.0, 2.0], sleeps  # header-driven, then exponential (2 ** attempt) without the header
server.fail = [http_error(429)] * 5
try:
    tw.get("games", {"name": "x"}, tries=5)
    raise AssertionError("should have raised")
except urllib.error.HTTPError as e:
    assert e.code == 429
server.fail = [http_error(500)]
try:
    tw.get("games", {"name": "x"})
    raise AssertionError("should have raised")
except urllib.error.HTTPError as e:
    assert e.code == 500  # other errors surface immediately

# --- clips: time window, view/language filters, best first, limit ------------------------------------------------
clips = tw.fetch_top_clips(game_id="32982", hours_back=24, min_views=500, limit=25)
assert [c["id"] for c in clips] == ["FunnyClip2", "FunnyClip1"], clips  # low views + Spanish dropped, sorted
q = parse_qs(urlparse(server.requests[-1][0]).query)
assert q["game_id"] == ["32982"] and q["first"] == ["100"]
assert q["started_at"][0].endswith("Z") and q["started_at"][0] < q["ended_at"][0]
assert [c["id"] for c in tw.fetch_top_clips(game_id="32982", language=None, limit=3)] == ["FunnyClip2", "FunnyClip1", "Spanish"]  # language=None keeps Spanish
assert [c["id"] for c in tw.fetch_top_clips(broadcaster_id="id_streamer1")] == ["JustChatting", "FunnyClip1"]  # any game
# streamer + game: Helix takes one or the other, so the streamer's clips are filtered to the game client-side
assert [c["id"] for c in tw.fetch_top_clips(game_id="32982", broadcaster_id="id_streamer1")] == ["FunnyClip1"]
assert "broadcaster_id" in parse_qs(urlparse(server.requests[-1][0]).query)
assert "game_id" not in parse_qs(urlparse(server.requests[-1][0]).query)
assert "broadcaster_id" in parse_qs(urlparse(server.requests[-1][0]).query)
assert tw.user_ids(["Streamer1", "ghost"]) == {"streamer1": "id_streamer1"}
meta = twitch_api.clip_meta(CLIPS[0])
assert meta["broadcaster_name"] == "Streamer1" and meta["view_count"] == 9000 and meta["clip_id"] == "FunnyClip1"

# --- queue: twitch_ ids, metadata stored, never twice, old databases migrated -------------------------------------
old = tmp / "old" / "queue.db"
old.parent.mkdir()
con = sqlite3.connect(old)
con.executescript("CREATE TABLE jobs (id INTEGER PRIMARY KEY, url TEXT, video_id TEXT UNIQUE, status TEXT DEFAULT 'pending', "
                  "attempts INTEGER DEFAULT 0, error TEXT, created_at TEXT, finished_at TEXT);"
                  "CREATE TABLE seen_videos (video_id TEXT PRIMARY KEY, seen_at TEXT);"
                  "INSERT INTO seen_videos VALUES ('a_CuOeu5KZQ', 'x');")
con.close()
db = qw.connect(old)
assert {"source", "meta"} <= {r[1] for r in db.execute("PRAGMA table_info(jobs)")}
assert qw.enqueue(db, ["a_CuOeu5KZQ"]) == (0, 1)  # old seen rows still honoured

channels = tmp / "channels.txt"
channels.write_text("Streamer1\nghost\n")
with quiet():
    added = qw.poll_twitch(db, tw, game="Grand Theft Auto V", channels_file=channels, min_views=500, limit=20)
assert added == 3  # game: FunnyClip2 + FunnyClip1; Streamer1's channel adds JustChatting (FunnyClip1 already queued)
row = dict(db.execute("SELECT * FROM jobs WHERE video_id = 'twitch_FunnyClip1'").fetchone())
assert row["source"] == "twitch" and row["url"] == "https://clips.twitch.tv/FunnyClip1"
assert json.loads(row["meta"])["broadcaster_name"] == "Streamer1"
with quiet() as log:  # nothing new at 500 views: steps down and finds the 100-view clip
    assert qw.poll_twitch(db, tw, game="Grand Theft Auto V") == 1
assert "nothing new with >= 500 views, retrying at >= 100" in log.getvalue() and qw.is_known(db, "twitch_LowViews")
with quiet():
    assert qw.poll_twitch(db, tw, game="Grand Theft Auto V") == 0  # every tier exhausted


# view-threshold step-down 1500 -> 500 -> 100, stopping at the first tier that queues something
class TierClient:
    def __init__(self, views):
        self.views, self.asked = views, []

    def user_ids(self, logins):
        return {l: f"id_{l}" for l in logins}

    def fetch_top_clips(self, game_id=None, broadcaster_id=None, hours_back=24, min_views=500, limit=25, language="en", **_):
        self.asked.append(min_views)
        return [{"id": f"v{v}", "url": f"https://clips.twitch.tv/v{v}", "view_count": v} for v in self.views if v >= min_views]


tchan = tmp / "tier_chan.txt"
tchan.write_text("quiet_streamer")
# each tier asks the 168 h window, then all-time
for views, want_tiers, want in [([2000], [1500], 1), ([700], [1500, 1500, 500], 1), ([150], [1500, 1500, 500, 500, 100], 1),
                                ([20], [1500, 1500, 500, 500, 100, 100], 0)]:
    tc, tdb = TierClient(views), qw.connect(tmp / f"tier{views[0]}" / "queue.db")
    with quiet():
        assert qw.poll_twitch(tdb, tc, channels_file=tchan, hours=168, min_views=1500) == want, views
    assert tc.asked == want_tiers, (views, tc.asked)
tc = TierClient([50])
with quiet():
    assert qw.poll_twitch(qw.connect(tmp / "tier_low" / "queue.db"), tc, channels_file=tchan, hours=168, min_views=200) == 0
assert tc.asked == [200, 200, 100, 100], tc.asked  # only tiers below the requested minimum

# the worker hands the metadata to clipper.py and marks the clip seen (with its source) once done
FAKE = r'''
import json, sys
from pathlib import Path
ws = Path(sys.argv[sys.argv.index("--workspace") + 1])
(ws.parent / "argv.json").write_text(json.dumps(sys.argv[1:]))
job = ws / "FunnyClip"; job.mkdir(parents=True, exist_ok=True)
(job / "clip_01_twitch.mp4").write_bytes(b"x")
(job / "clips_manifest.json").write_text("{}")
print(json.dumps({"event": "done", "manifest": str(job / "clips_manifest.json")}))
'''
fake = tmp / "fake.py"
fake.write_text(FAKE)
db.execute("DELETE FROM jobs WHERE video_id != 'twitch_FunnyClip2'")
with quiet():
    assert qw.drain(db, [], tmp / "old", tmp / "out", cooldown=(0, 0), clipper_cmd=[sys.executable, str(fake)], sleep=lambda s: None) == 1
argv = json.loads((tmp / "argv.json").read_text())
assert argv[0] == "https://clips.twitch.tv/FunnyClip2" and json.loads(argv[argv.index("--twitch-meta") + 1])["view_count"] == 12000
assert (tmp / "out" / "twitch_FunnyClip2" / "clip_01_twitch.mp4").exists() and not (tmp / "old" / "FunnyClip").exists()
assert db.execute("SELECT source FROM seen_videos WHERE video_id = 'twitch_FunnyClip2'").fetchone()[0] == "twitch"
with quiet():
    assert qw.poll_twitch(db, tw, game="Grand Theft Auto V") == 1  # FunnyClip1 was deleted from jobs, Clip2 is seen

# --- streamer logins: normalised from the Godot box / a hand-edited file ------------------------------------------
pasted = "xqc, @Kai_Cenat\nhttps://www.twitch.tv/caseoh_/clips?x=1  twitch.tv/Jynxzi;xqc, bad-name!, "
assert twitch_api.parse_logins(pasted) == ["xqc", "kai_cenat", "caseoh_", "jynxzi"]
assert twitch_api.parse_logins("") == [] and twitch_api.parse_logins(None) == []

# --- streamers + game filter through the queue (fresh db); a typo'd game never widens to "any game" -------------
db2 = qw.connect(tmp / "ws2" / "queue.db")
chan2 = tmp / "chan2.txt"
chan2.write_bytes(b"\xef\xbb\xbf@Streamer1, https://www.twitch.tv/streamer2\n")  # BOM + @ + URL, as a user might save it
with quiet():
    assert qw.poll_twitch(db2, tw, channels_file=chan2, channel_game="Grand Theft Auto V") == 2  # FunnyClip1 + FunnyClip2
    assert qw.poll_twitch(db2, tw, channels_file=chan2, channel_game="Grand Theft Auto 7") == 0
assert {r[0] for r in db2.execute("SELECT video_id FROM jobs")} == {"twitch_FunnyClip1", "twitch_FunnyClip2"}  # no JustChatting
with quiet():
    assert qw.poll_twitch(db2, tw, channels_file=chan2) == 1  # no filter: Streamer1's JustChatting clip too

# --- Whisper windows clamped to the decoded audio (the live "non-negative timestamp expected" crash) -------------
assert clipper.whisper_chunks([(0.0, 30.04)], 30.0) == [{"start": 0.0, "end": 30.0}]  # metadata 40 ms past the audio
assert clipper.whisper_chunks([(-3.0, 75.0)], 600.0) == [{"start": 0.0, "end": 30.0}, {"start": 30.0, "end": 60.0},
                                                        {"start": 60.0, "end": 75.0}]  # padded start clamped, <=30 s chunks
assert clipper.whisper_chunks([(590.0, 606.0)], 600.0) == [{"start": 590.0, "end": 600.0}]
assert clipper.whisper_chunks([(605.0, 610.0)], 600.0) == []  # entirely past the end: nothing, not a negative span
chunks = clipper.whisper_chunks([(0.0, 13.99), (2.0, 9.0)], 13.97)
assert all(0 <= c["start"] < c["end"] <= 13.97 for c in chunks)
assert clipper.is_youtube("https://youtu.be/x") and clipper.is_youtube("https://m.youtube.com/watch?v=x")
assert not clipper.is_youtube("https://clips.twitch.tv/x") and not clipper.is_youtube("https://notyoutube.com/x")
ck = tmp / "ck.txt"
ck.write_text("# Netscape")
yt_args = argparse.Namespace(url="https://www.youtube.com/watch?v=x", cookies=str(ck), cookies_from_browser=None)
tw_args = argparse.Namespace(url="https://clips.twitch.tv/x", cookies=str(ck), cookies_from_browser="edge")
assert clipper.cookie_args(yt_args) == ["--cookies", str(ck)] and clipper.cookie_args(tw_args) == []  # never to Twitch

# --- clipper fast path: whole clip kept, transcribed, judged, credited ---------------------------------------------
media = tmp / "clip.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=320x180:d=6", "-f", "lavfi",
                "-i", "sine=frequency=700:duration=6", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(media)], check=True)
job = tmp / "cws" / "1234"
job.mkdir(parents=True)
calls = []
clipper.run = lambda cmd, check=True: (calls.append(cmd), shutil.copy(media, job / "source.mp4"))
clipper.OUTPUT = tmp / "tw_out"  # the duplicate-moment check must never read the real output/
contexts = []
clipper.make_transcriber = lambda model, device, context="": contexts.append(context) or (lambda audio, windows, words_out=None: (words_out is not None and words_out.extend([{"word": "hahaha", "start": 2.6, "end": 3.0}])) or [[(0.5, 2.5, "he's back in the dinosaur suit!"),
                                                                         (2.5, 5.0, "hahaha no way")]])
scorer.ollama_online = lambda url="": True
scorer.ollama_warmup = lambda model, url="": None
judged = []
VERDICT = {"llm": 0.9, "reject": False, "reason": "A cop gets trolled mid-chase and it lands.", "setup": None,
           "pillars": {"self_contained": 9, "comedy_payoff": 9, "hook_strength": 9, "quotability": 9}}
scorer.ollama_judge = lambda cands, model: (judged.append(cands), {0: dict(VERDICT)})[1]
args = argparse.Namespace(twitch_meta=json.dumps(meta), no_ollama=False, ollama_model="m", whisper_model="small.en", device="cpu",
                         auto_edit=False, min_score=6, workspace=tmp / "cws")
info = {"id": "1234", "title": "cop gets trolled", "webpage_url": "https://clips.twitch.tv/FunnyClip1", "duration": 6}
with quiet() as out:
    assert clipper.twitch_clip(args, info, job) == 0
events = [json.loads(l) for l in out.getvalue().splitlines()]
assert calls and "https://clips.twitch.tv/FunnyClip1" in calls[0]  # one yt-dlp download, no windowing/cutting
assert (job / "clip_01_twitch.mp4").exists() and not (job / "source.mp4").exists()
m = json.loads((job / "clips_manifest.json").read_text())
assert m["source"] == "twitch" and m["twitch"]["broadcaster_name"] == "Streamer1" and m["twitch"]["view_count"] == 9000
c = m["clips"][0]
assert c["start_sec"] == 0 and abs(c["end_sec"] - 6) < 0.1 and c["file"] == "clip_01_twitch.mp4"  # decoded length
assert "hahaha no way" in c["transcript"] and "Streamer1" in c["reason"] and "LLM 9.0/10" in c["reason"]
assert c["words"] == [{"word": "hahaha", "start": 2.6, "end": 3.0}], c["words"]  # one Whisper pass: the editor reuses these
assert c["judge_reason"] == "A cop gets trolled mid-chase and it lands." and m["rejected"] == [], c  # why it passed
assert "viral boost (9000 views)" in c["reason"], c["reason"]  # 9000 views + a loud burst: a proven clip
assert c["score"] > 0.5 and len(judged) == 1 and len(judged[0]) == 1
assert events[-1]["event"] == "done" and any(e["event"] == "clip" for e in events)

# --auto-edit on a single run hands the finished manifest to the editor (queue_worker.run_editor)
edited = []
qw.run_editor = lambda m, editor_cmd=None: (edited.append(str(m)), 1)[1]
args.auto_edit = True
with quiet() as out:
    clipper.twitch_clip(args, info, job)
assert edited == [str(job / "clips_manifest.json")] and "1 short(s) in ready_to_post/" in out.getvalue(), edited

# banger gate: a clip the judge rates below --min-score is skipped before any editing (empty manifest, file gone)
args.auto_edit, args.min_score = False, 9.9  # 9.0 + the viral boost (9000 views, loud) = 9.5
with quiet() as out:
    assert clipper.twitch_clip(args, info, job) == 0
events = [json.loads(l) for l in out.getvalue().splitlines()]
assert json.loads((job / "clips_manifest.json").read_text())["clips"] == [] and not list(job.glob("clip_*.mp4"))
assert events[-1] == {"event": "done", "manifest": str(job / "clips_manifest.json"), "count": 0}
assert any("Banger gate: rated 9.5/10 < 9.9" in e.get("msg", "") for e in events), events

# hard reject: dead air (no payoff AND no hook) loses even with 9000 views, a loud burst and the gate off
args.min_score = 0
scorer.ollama_judge = lambda cands, model: {0: dict(VERDICT, llm=0.3, reject=True, reason="Dead air in an inventory menu.",
                                                    pillars={**VERDICT["pillars"], "comedy_payoff": 2, "hook_strength": 2})}
with quiet() as out:
    assert clipper.twitch_clip(args, info, job) == 0
m = json.loads((job / "clips_manifest.json").read_text())
assert m["clips"] == [] and not list(job.glob("clip_*.mp4")), m
assert m["rejected"][0]["judge_reason"] == "Dead air in an inventory menu." and m["rejected"][0]["pillars"]["comedy_payoff"] == 2
assert "AI judge rejected this clip: Dead air" in out.getvalue()
# --no-gate (calibration_studio): the same rejected clip is kept, with its full score breakdown in the manifest
args.no_gate = True
with quiet():
    clipper.twitch_clip(args, info, job)
args.no_gate = False
c = json.loads((job / "clips_manifest.json").read_text())["clips"][0]
assert c["judge_reject"] and c["pillars"]["comedy_payoff"] == 2 and c["llm"] == 0.35 and c["vocal"] > 0, c  # 0.3 + viral boost
assert (job / "clip_01_twitch.mp4").exists()
warn = [json.loads(l) for l in out.getvalue().splitlines() if '"warning"' in l]
assert warn and warn[0]["msg"] == ("[JUDGE] Rejected all 1 candidate(s) for twitch_FunnyClip1 (best: payoff=2, hook=2: dead air "
                                   "(needs payoff >= 5 or hook >= 5)): Dead air in an inventory menu."), warn

# a 9/10 payoff that starts mid-conversation (self_contained 3) is NOT vetoed: it's cut and kept
scorer.ollama_judge = lambda cands, model: {0: dict(VERDICT, llm=0.62, reason="Spoke too soon, instant karma.",
                                                    pillars={**VERDICT["pillars"], "self_contained": 3})}
args.min_score = 6.0
with quiet() as out:
    clipper.twitch_clip(args, info, job)
m = json.loads((job / "clips_manifest.json").read_text())
assert len(m["clips"]) == 1 and m["clips"][0]["judge_reason"] == "Spoke too soon, instant karma." and m["rejected"] == [], m

# --- multi-streamer: balanced round-robin + the time window widens when the newest clips are already done ---------
from datetime import datetime, timedelta, timezone

NOW = 1_800_000_000.0
iso = lambda hours_ago: (datetime.fromtimestamp(NOW, timezone.utc) - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
LIB = {"alpha": [(f"a{i}", 9000 - i * 100, 2 + i) for i in range(1, 6)],  # 5 clips, all in the last 24 h
       "beta": [("b1", 5000, 3), ("b2", 4000, 10), ("b3", 3000, 48), ("b4", 2000, 120)]}  # 2 recent, 1 at 2 days, 1 at 5 days


class Helix:
    """Fake Helix that honours started_at/ended_at like the real /clips endpoint."""
    def __init__(self):
        self.windows, self.afters, self.page = [], [], 100

    def __call__(self, req, timeout=None):
        url = req.full_url
        if url.startswith(twitch_api.TOKEN_URL):
            return Resp(b'{"access_token": "t", "expires_in": 3600}')
        q = parse_qs(urlparse(url).query)
        if urlparse(url).path.endswith("/users"):
            return Resp(json.dumps({"data": [{"id": f"id_{l}", "login": l} for l in q["login"] if l in LIB]}).encode())
        login = q["broadcaster_id"][0][3:]
        lo = q.get("started_at", [""])[0]  # no started_at = all-time top clips
        self.windows.append((login, round((NOW - datetime.strptime(lo, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()) / 3600)
                             if lo else None))
        data = [{"id": cid, "url": f"https://clips.twitch.tv/{cid}", "language": "en", "view_count": v, "broadcaster_name": login,
                 "game_id": "1", "created_at": iso(h)} for cid, v, h in LIB[login] if not lo or iso(h) >= lo]
        off = int(q.get("after", ["0"])[0])  # like Helix: pages of `first`, an opaque cursor while more remain
        self.afters.append(off)
        more = off + self.page < len(data)
        return Resp(json.dumps({"data": data[off:off + self.page], "pagination": {"cursor": str(off + self.page)} if more else {}}).encode())


helix = Helix()
tw3 = twitch_api.Twitch("cid", "secret", tmp / "ws3" / "tok.json", opener=helix, clock=lambda: NOW)
db3 = qw.connect(tmp / "ws3" / "queue.db")
chan3 = tmp / "chan3.txt"
chan3.write_text("alpha, beta, ghost")
order = lambda: [r[0] for r in db3.execute("SELECT video_id FROM jobs ORDER BY id")]
with quiet():
    assert qw.poll_twitch(db3, tw3, channels_file=chan3, limit=4, min_views=0) == 4
assert order() == ["twitch_a1", "twitch_b1", "twitch_a2", "twitch_b2"], order()  # 2 each (ceil(4/2)), interleaved
db3.execute("INSERT INTO seen_videos (video_id) SELECT video_id FROM jobs")  # pretend they were all processed
db3.execute("DELETE FROM jobs")
helix.windows.clear()
with quiet() as log:
    assert qw.poll_twitch(db3, tw3, channels_file=chan3, limit=4, min_views=0) == 3
assert order() == ["twitch_a3", "twitch_b3", "twitch_a4"], order()  # beta widened to 72 h for b3
assert ("beta", 24) in helix.windows and ("beta", 72) in helix.windows and ("alpha", 72) not in helix.windows
assert "beta had nothing new in 24h, widened to 72h" in log.getvalue()
db3.execute("INSERT INTO seen_videos (video_id) SELECT video_id FROM jobs")
db3.execute("DELETE FROM jobs")
with quiet():
    assert qw.poll_twitch(db3, tw3, channels_file=chan3, limit=4, min_views=0) == 2
assert order() == ["twitch_a5", "twitch_b4"], order()  # beta needed the 7-day window
with quiet():
    assert qw.poll_twitch(db3, tw3, channels_file=chan3, limit=4, min_views=0) == 0  # queued ones aren't "new" either
# --- pagination: page 1 all processed -> the cursor is followed to page 2 ----------------------------------------
helix.page, helix.afters = 2, []
db4 = qw.connect(tmp / "ws4" / "queue.db")
chan4 = tmp / "chan4.txt"
chan4.write_text("alpha, ghost")
db4.executemany("INSERT INTO seen_videos (video_id) VALUES (?)", [(f"twitch_a{i}",) for i in (1, 2)])
with quiet() as log:
    assert qw.poll_twitch(db4, tw3, channels_file=chan4, limit=1, min_views=0) == 1
events = [json.loads(l) for l in log.getvalue().splitlines()]
assert [r[0] for r in db4.execute("SELECT video_id FROM jobs")] == ["twitch_a3"] and helix.afters == [0, 2], helix.afters
# an unknown login is a warning event of its own (the panel's banner shows it)
assert {"event": "warning", "msg": "Twitch user not found: 'ghost'", "login": "ghost"} in events, events

# --- nothing new anywhere (every window, every page, all time): a warning that says exactly why --------------------
db4.executemany("INSERT INTO seen_videos (video_id) VALUES (?)", [(f"twitch_a{i}",) for i in (3, 4, 5)])
db4.execute("DELETE FROM jobs")
chan4.write_text("alpha")
helix.afters = []
with quiet() as log:
    assert qw.poll_twitch(db4, tw3, channels_file=chan4, limit=1, min_views=0) == 0
events = [json.loads(l) for l in log.getvalue().splitlines()]
warn = [e for e in events if e["event"] == "warning"]
assert len(warn) == 1 and warn[0]["msg"] == ("0 new clips: Twitch returned 5 clips for alpha: 5 already processed, "
                                              "0 filtered by views/lang (0 new)"), warn
assert warn[0]["breakdown"] == [{"source": "alpha", "returned": 5, "processed": 5, "filtered": 0}]
assert None in [w for l, w in helix.windows[-4:]] and max(helix.afters) == 4, (helix.windows[-4:], helix.afters)  # all-time, 3 pages
helix.page = 100
# --- Clips per Streamer: a fixed count each (instead of the even split), the total still capped by --twitch-limit ---
for limit, want in [(20, ["twitch_a1", "twitch_b1", "twitch_a2", "twitch_b2", "twitch_a3"]),
                    (4, ["twitch_a1", "twitch_b1", "twitch_a2", "twitch_b2"])]:
    db5 = qw.connect(tmp / f"ws5_{limit}" / "queue.db")
    chan5 = tmp / "chan5.txt"
    chan5.write_text("alpha, beta")
    with quiet():
        assert qw.poll_twitch(db5, tw3, channels_file=chan5, limit=limit, min_views=0, per_streamer=3) == len(want)
    got = [r[0] for r in db5.execute("SELECT video_id FROM jobs ORDER BY id")]
    assert got == want, (limit, got)  # alpha 3, beta has only 2 in 24 h; round-robin, then the cap
# vocabulary context: the streamer + title scope the fixes ("queso" -> CaseOh only on CaseOh clips)
assert contexts and contexts[0].startswith("Streamer1 "), contexts

# --- duplicate moment, after Whisper: same streamer, last 24 h, >60% of the words in common -> skipped -------------
prev = clipper.OUTPUT / "twitch_OtherViewersClip"
prev.mkdir(parents=True)
(prev / "clips_manifest.json").write_text(json.dumps({"video_id": "9", "twitch": {"broadcaster_name": "streamer1"},
    "clips": [{"clip_id": 1, "transcript": "he's back in the dinosaur suit! hahaha no way, chat"}]}))
args.min_score, judged[:] = 0, []
scorer.ollama_judge = lambda cands, model: (judged.append(cands), {0: dict(VERDICT)})[1]
with quiet() as out:
    assert clipper.twitch_clip(args, info, job) == 0
m = json.loads((job / "clips_manifest.json").read_text())
assert m["clips"] == [] and m["duplicate_of"] == "twitch_OtherViewersClip" and not judged, m  # never judged or edited
assert "Duplicate moment:" in out.getvalue() and not list(job.glob("clip_*.mp4"))
assert scorer.jaccard("he's back in the dinosaur suit", "totally different words about a car crash today") < 0.6
assert scorer.jaccard("lol", "lol") == 0.0, "too short to tell"
other = json.loads((prev / "clips_manifest.json").read_text())
other["twitch"]["broadcaster_name"] = "someone_else"
(prev / "clips_manifest.json").write_text(json.dumps(other))
with quiet():
    clipper.twitch_clip(args, info, job)
assert json.loads((job / "clips_manifest.json").read_text())["clips"], "another streamer saying the same words is not a duplicate"

# --- unified storage: with --output the clipper archives the run itself (single runs land in output/ too) -----------
args.output = tmp / "unified_out"
for attempt in (1, 2):  # a re-run replaces the archive and is not a "duplicate" of its own earlier run
    with quiet() as out:
        assert clipper.twitch_clip(args, info, job) == 0
    done = [json.loads(l) for l in out.getvalue().splitlines() if '"done"' in l][-1]
    arch = tmp / "unified_out" / "twitch_FunnyClip1"
    assert done["manifest"] == str(arch / "clips_manifest.json") and done["count"] == 1, (attempt, done)
    assert (arch / "clip_01_twitch.mp4").exists() and not list(job.glob("clip_*.mp4")) and not (job / "clips_manifest.json").exists()
    assert "Duplicate moment" not in out.getvalue(), attempt
args.output = None

# --- velocity: views per minute since creation (min 15 min) puts breakouts first -------------------------------------
fresh_hot = {"id": "hot", "view_count": 3000, "created_at": iso(0.5)}   # 3000 views in 30 min = 100/min
old_big = {"id": "big", "view_count": 20000, "created_at": iso(20)}     # 20000 in 20 h = ~17/min
just_now = {"id": "new", "view_count": 600, "created_at": iso(0.05)}    # 3 min old: counted as 15 min = 40/min
assert twitch_api.velocity(fresh_hot, NOW) == 100.0 and twitch_api.velocity(just_now, NOW) == 40.0
assert twitch_api.velocity({"view_count": 7}, NOW) == 7.0  # no created_at: plain views
tw_v = twitch_api.Twitch("cid", "secret", tmp / "wsv" / "tok.json", opener=lambda req, timeout=None: Resp(json.dumps(
    {"data": [dict(c, url="u", language="en") for c in (old_big, fresh_hot, just_now)]}).encode()
    if "/clips" in req.full_url else b'{"access_token": "t", "expires_in": 3600}'), clock=lambda: NOW)
assert [c["id"] for c in tw_v.fetch_top_clips(broadcaster_id="b", min_views=0, sort="velocity")] == ["hot", "new", "big"]
assert [c["id"] for c in tw_v.fetch_top_clips(broadcaster_id="b", min_views=0)] == ["big", "hot", "new"]  # default: views
assert twitch_api.clip_meta(dict(fresh_hot, video_id="v1", vod_offset=120, broadcaster_id="b"))["vod_offset"] == 120

# --- same stream moment (same VOD, offsets within 30 s): skipped against the queue AND inside one poll ---------------
class MomentClient:
    def __init__(self, clips):
        self.clips = clips

    def user_ids(self, logins):
        return {l: f"id_{l}" for l in logins}

    def fetch_top_clips(self, skip=None, limit=25, **_):
        return [c for c in self.clips if not (skip and skip(c))][:limit]


mk_clip = lambda cid, vod, off: {"id": cid, "url": f"https://clips.twitch.tv/{cid}", "view_count": 5000,
                                 "video_id": vod, "vod_offset": off, "broadcaster_name": "s"}
mdb = qw.connect(tmp / "moments" / "queue.db")
qw.enqueue_items(mdb, qw.twitch_items([mk_clip("done1", "vod9", 1000)]))  # processed earlier (meta kept with the job)
mchan = tmp / "mchan.txt"
mchan.write_text("s")
mc = MomentClient([mk_clip("dupOld", "vod9", 1020), mk_clip("a", "vod9", 2000), mk_clip("a2", "vod9", 2010),
                   mk_clip("b", "vod9", 2100), mk_clip("noVod", "", None), mk_clip("other", "vod7", 1000)])
with quiet() as log:
    assert qw.poll_twitch(mdb, mc, channels_file=mchan, hours=168, min_views=0, limit=20) == 4
got = {r[0] for r in mdb.execute("SELECT video_id FROM jobs")}
assert got == {"twitch_done1", "twitch_a", "twitch_b", "twitch_noVod", "twitch_other"}, got  # dupOld + a2 dropped
assert "a2 is the same moment as a clip already picked" in log.getvalue()
assert qw.same_moment({"video_id": "vod9", "vod_offset": 1030}, [("vod9", 1000)]) and \
    not qw.same_moment({"video_id": "vod9", "vod_offset": 1031}, [("vod9", 1000)])
# --- 30-day duplicate guard (posted_history.json) + the approved clip's post card -------------------------------------
history.remember = real_remember
history.HISTORY.unlink(missing_ok=True)
args.auto_edit, args.min_score, args.twitch_meta = False, 0, json.dumps(meta)
scorer.ollama_online = lambda url="": True
scorer.ollama_judge = lambda cands, model: {0: dict(VERDICT)}
args.no_ollama = False
calls.clear()
with quiet() as out:
    assert clipper.twitch_clip(args, info, job) == 0
c = json.loads((job / "clips_manifest.json").read_text())["clips"][0]
assert c["title_angles"] == CARD["title_angles"] and c["hashtags"] == CARD["hashtags"] and c["pinned_comment"], c
h = history.load()
assert [x["clip_id"] for x in h["clips"]] == ["FunnyClip1"] and h["clips"][0]["streamer"] == "streamer1", h
n = len(calls)
with quiet() as out:
    assert clipper.twitch_clip(args, info, job) == 0
assert "Already processed in the last 30 days (clip FunnyClip1)" in out.getvalue() and len(calls) == n, "skipped before downloading"
assert json.loads((job / "clips_manifest.json").read_text())["clips"] == []
args.twitch_meta = json.dumps(dict(meta, clip_id="OtherViewerClip"))  # another viewer clipped the same moment
with quiet() as out:
    clipper.twitch_clip(args, info, job)
assert "Duplicate moment: transcript" in out.getvalue() and "(last 30 days)" in out.getvalue(), out.getvalue()
history.update(lambda hh: hh["clips"][0].__setitem__("at", 0))  # the entry is over 30 days old now
with quiet() as out:
    assert clipper.twitch_clip(args, info, job) == 0
assert json.loads((job / "clips_manifest.json").read_text())["clips"], "after 30 days the moment is fair game again"
assert len(history.load()["clips"]) == 1, "expired entries are pruned on the next write"
# local uploads (no metadata) are yours: never blocked or recorded
assert history.seen(clip_id="local_upload") is None
print("twitch ok")

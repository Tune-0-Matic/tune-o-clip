"""Calibration studio tests: forced per-streamer harvest (real queue_worker.drain + a fake clipper process), --no-gate,
resume (also an edit pass interrupted mid-way), the partial decision log, and the studio server: HTTP Range 206,
reviews + labels, the weight tuner, the harvest runner (fake harvest process), tuning report export.
Run: python calibration_studio/test_calibration.py  (no network, no Ollama, needs ffmpeg for thumbnails)
"""
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harvest_batch as hb  # noqa: E402
import review_server as rs  # noqa: E402

ROOT = HERE.parent
tmp = Path(tempfile.mkdtemp(prefix="calib_test_"))
quiet = lambda: contextlib.redirect_stdout(io.StringIO())

# the live pipeline's state must not change: queue.db, ready_to_post/, posted/
live = lambda: (sorted(p.name for p in (ROOT / "ready_to_post").glob("**/*")),
                sorted(p.name for p in (ROOT / "ready_to_post" / "posted").glob("*")) if (ROOT / "ready_to_post" / "posted").exists() else [],
                (ROOT / "workspace" / "queue.db").stat().st_mtime if (ROOT / "workspace" / "queue.db").exists() else None)
before = live()

# --- fake clipper: a real subprocess, called by queue_worker.drain exactly like clipper.py ---------------------------
FAKE = tmp / "fake_clipper.py"
FAKE.write_text(r'''
import json, sys
from pathlib import Path
a = sys.argv[1:]
out, meta = Path(a[a.index("--output") + 1]), json.loads(a[a.index("--twitch-meta") + 1])
name = "twitch_" + meta["clip_id"]
reject = meta["clip_id"][-1] in "13579"  # odd ids: the judge hates them
clips = []
if not meta["clip_id"].startswith("dup") and (not reject or "--no-gate" in a):
    (out / name).mkdir(parents=True, exist_ok=True)
    (out / name / "clip_01_twitch.mp4").write_bytes(b"raw")
    clips = [{"clip_id": 1, "file": "clip_01_twitch.mp4", "duration": 12.0, "score": 0.4, "llm": 0.2 if reject else 0.8,
              "judge_reject": reject, "judge_reason": "Dead air." if reject else "Lands hard.", "vocal": 0.9, "visual": 0.1,
              "cue": 0.3, "pillars": {"comedy_payoff": 2 if reject else 8, "self_contained": 5, "hook_strength": 4, "quotability": 3},
              "reason": "Twitch clip", "transcript": "he lost it chat"}]
(out / name).mkdir(parents=True, exist_ok=True)
(out / name / "clips_manifest.json").write_text(json.dumps({"video_id": name, "twitch": meta, "clips": clips}))
print(json.dumps({"event": "done", "manifest": str(out / name / "clips_manifest.json"), "count": len(clips)}), flush=True)
''')

NOW = time.time()
iso = lambda h: (datetime.fromtimestamp(NOW, timezone.utc) - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
clip = lambda cid, uid, views, hours, lang="en", vod="v", off=None: {
    "id": cid, "broadcaster_id": uid, "broadcaster_name": uid.title(), "view_count": views, "created_at": iso(hours),
    "language": lang, "title": f"clip {cid}", "url": f"https://clips.twitch.tv/{cid}", "duration": 12.0,
    "video_id": vod, "vod_offset": off if off is not None else hash(cid) % 100000}
LIB = {
    # alpha: 4 fresh clips (one is the same stream moment, one a duplicate transcript), the rest only in wider windows
    "alpha": [clip("a0", "alpha", 9000, 10, off=500), clip("a2", "alpha", 8000, 20, off=510),  # a2 = same moment as a0
              clip("dupa", "alpha", 7000, 30), clip("a4", "alpha", 6000, 40)]
             + [clip(f"b{i}", "alpha", 200, 400) for i in range(10)] + [clip(f"c{i}", "alpha", 50, 2000) for i in range(12)],
    # beta: only Spanish-tagged clips: just the last tier (any language) finds them
    "beta": [clip(f"e{i}", "beta", 5000, 10, lang="es") for i in range(25)],
}


class FakeTwitch:
    def __init__(self):
        self.calls = []

    def user_ids(self, logins):
        return {l: l for l in logins if l in LIB}

    def fetch_top_clips(self, broadcaster_id=None, hours_back=None, min_views=0, limit=25, language="en", pages=1, skip=None, **kw):
        self.calls.append((broadcaster_id, hours_back, min_views, language))
        ok = [c for c in LIB[broadcaster_id] if (hours_back is None or (NOW - datetime.strptime(
            c["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()) / 3600 <= hours_back)
              and c["view_count"] >= min_views and (not language or c["language"] == language) and not (skip and skip(c))]
        return sorted(ok, key=lambda c: -c["view_count"])[:limit]


batch = tmp / "runs" / "b1"
batch.mkdir(parents=True)
tw = FakeTwitch()
with quiet() as out:
    hb.harvest(batch, ["alpha", "beta", "ghost"], 20, tw, ["--no-gate"], [sys.executable, str(FAKE)], sleep=lambda s: None)
assert hb.count_for(batch, "alpha") == 20 and hb.count_for(batch, "beta") == 20, (hb.count_for(batch, "alpha"), hb.count_for(batch, "beta"))
assert "Twitch user not found: 'ghost'" in out.getvalue()
kept = hb.kept_clips(batch)
assert len(kept) == 40 and len({d["video_id"] for _, d, _ in kept}) == 40
assert sum(c["judge_reject"] for _, _, c in kept) >= 15, "--no-gate keeps the clips the judge rejected"
assert not (batch / "output" / "twitch_a2").exists(), "same stream moment as a0: never downloaded"
assert (batch / "output" / "twitch_dupa" / "clips_manifest.json").exists(), "duplicate transcript: processed, kept 0, refilled"
assert {(h, v) for u, h, v, _ in tw.calls if u == "alpha"} >= {(168, 500), (720, 100), (None, 0)}, "window + views widen"
assert ("beta", None, 0, None) in tw.calls, "last tier drops the language filter"
assert (batch / "queue.db").exists() and before == live(), "the live queue / ready_to_post / posted are untouched"

# without --no-gate the fake clipper drops rejected clips: proves the flag is what keeps them
nogate = tmp / "runs" / "gated"
nogate.mkdir(parents=True)
with quiet():
    hb.harvest(nogate, ["alpha"], 3, FakeTwitch(), [], [sys.executable, str(FAKE)], sleep=lambda s: None)
assert hb.count_for(nogate, "alpha") == 3 and not any(c["judge_reject"] for _, _, c in hb.kept_clips(nogate))

# resume: the same batch asks Twitch for nothing new and processes nothing again
n_calls = len(tw.calls)
mtimes = {m: m.stat().st_mtime for m, _, _ in kept}
with quiet():
    hb.harvest(batch, ["alpha", "beta"], 20, tw, ["--no-gate"], [sys.executable, str(FAKE)], sleep=lambda s: None)
assert len(tw.calls) == n_calls and all(m.stat().st_mtime == t for m, t in mtimes.items())

# --- edit_all: every kept clip edited into the batch's shorts/, a crash is retried on resume ------------------------
media = tmp / "short.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=108x192:d=2", "-f", "lavfi", "-i",
                "sine=frequency=500:duration=2", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(media)], check=True)
EDL = {"scenario": "chat_roast", "directed_by": "ollama", "layout": "split_stack",
       "cuts": [{"start": 0.5, "end": 4.0}, {"start": 5.0, "end": 9.0}], "hook_teaser": {"enabled": False},
       "zooms": [{"at": 3.1, "duration": 0.9, "scale": 1.65, "style": "snap"}], "filters": [{"type": "trauma_shake", "start": 3, "end": 3.3}],
       "sfx": [{"at": 2.9, "name": "vine_boom", "gain_db": -5}], "takeovers": [], "speed": [], "mutes": [], "font": "Impact",
       "social_title": "He lost it #gaming"}
edits, crash = [], {"twitch_b0_1"}


def fake_edit(m, cid, out):
    stem = f"{json.loads(m.read_text())['video_id']}_{cid}"
    edits.append(stem)
    if stem in crash:
        raise RuntimeError("CUDA out of memory")
    if stem == "twitch_b2_1":
        return {"ok": False, "skipped": True, "reason": "only 1 clear word(s) survive the trim", "edl": dict(EDL, skip_render=True)}
    shutil.copy(media, out / f"{stem}_short.mp4")
    return {"ok": True, "path": str(out / f"{stem}_short.mp4"), "qc": {"duration": 7.5, "mean_db": -16.0}, "edl": EDL}


# interrupted mid-render (Ctrl+C / closed window after 5 edits): the partial run is still reviewable
class Interrupted(BaseException):
    pass


def stop_after_5(m, cid, out):
    if len(edits) == 5:
        raise Interrupted()
    return fake_edit(m, cid, out)


with quiet() as out:
    try:
        hb.edit_all(batch, stop_after_5)
    except Interrupted:
        pass
progress = [json.loads(l) for l in out.getvalue().splitlines() if '"progress"' in l]
assert progress[0]["phase"] == "render" and progress[0]["n"] == 1 and progress[0]["of"] == 40 and progress[0]["streamer"] in ("alpha", "beta")
partial = hb.write_decision_log(batch)
st = [e["editor"]["status"] for e in partial]
assert len(partial) == 40 and st.count("not_edited") == 35 and st.count("error") == 1 and "edited" in st, st
edits.clear()
with quiet():
    hb.edit_all(batch, fake_edit)  # resume: only the 35 left + the crashed one... which crashes again
assert len(edits) == 36 and "twitch_b0_1" in edits, len(edits)
assert (batch / "shorts" / "twitch_a0_1_short.mp4").exists() and (batch / "shorts" / "twitch_a0_1.jpg").exists()
assert "error" in json.loads((batch / "shorts" / "twitch_b0_1_edit.json").read_text())
crash.clear()
edits.clear()
with quiet():
    hb.edit_all(batch, fake_edit)
assert edits == ["twitch_b0_1"], edits  # only the crashed one is tried again

# --- decision_log.json + DECISION_REPORT.md --------------------------------------------------------------------------
entries = hb.write_decision_log(batch)
log = json.loads((batch / "decision_log.json").read_text(encoding="utf-8"))
assert len(log) == 40 == len(entries)
a0 = next(e for e in log if e["id"] == "twitch_a0_1")
assert a0["streamer"] == "alpha" and a0["short"] == "shorts/twitch_a0_1_short.mp4" and a0["raw"] == "output/twitch_a0/clip_01_twitch.mp4"
assert a0["clipper"]["judge_10"] == 8.0 and a0["clipper"]["pillars"]["comedy_payoff"] == 8 and a0["clipper"]["views"] == 9000
assert a0["clipper"]["views_per_min"] > 0 and a0["clipper"]["vocal_burst"] == 0.9 and a0["transcript"] == "he lost it chat"
assert a0["editor"]["scenario"] == "chat_roast" and a0["editor"]["kept_sec"] == 7.5 and a0["editor"]["trimmed_sec"] == 4.5
skipped = next(e for e in log if e["id"] == "twitch_b2_1")
assert skipped["editor"]["status"] == "skipped" and skipped["short"] is None and "clear word" in skipped["editor"]["note"]
md = (batch / "DECISION_REPORT.md").read_text(encoding="utf-8")
assert "chat_roast (picked by ollama), split_stack layout" in md and "4.5 s of dead air / pauses trimmed" in md
assert "vine_boom @2.9s" in md and "REJECTED, kept only because the batch has no gate" in md and "payoff 8" in md
assert "no short (skipped: only 1 clear word(s)" in md

# --- studio server ---------------------------------------------------------------------------------------------------
import scorer  # noqa: E402  (on sys.path via harvest_batch)

runs = tmp / "runs"
scorer.WEIGHTS_FILE, rs.SETTINGS = tmp / "weights.json", tmp / "settings.json"  # never the real ones
rs.SETTINGS.write_text(json.dumps({"min_score": 0, "clips": 4}))
(batch / "decision_log.json").unlink()  # the old lockout: a run with no decision_log.json must still open
FAKE_HARVEST = tmp / "fake_harvest.py"
FAKE_HARVEST.write_text(r"""
import json, sys, time
from pathlib import Path
a = sys.argv[1:]
runs = Path(sys.argv[0]).parent / "runs"
(runs / a[a.index("--batch") + 1] / "argv.json").write_text(json.dumps(a))
if "slow" in a[a.index("--streamers") + 1]:
    time.sleep(60)
out = runs / a[a.index("--batch") + 1] / "shorts" / "zz_1_short.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(b"x")
for ev in [{"event": "progress", "phase": "harvest"},
           {"event": "queue", "msg": "Processing twitch_Abc (attempt 1)"},
           {"event": "log", "msg": "Twitch clip: big fail by Jynxzi (17s, 9 views)"},
           {"event": "stage", "stage": "download", "msg": "Downloading Twitch clip"},
           {"event": "stage", "stage": "scoring", "msg": "Ollama comedy judge"},
           {"event": "log", "msg": "Encoder: h264_nvenc"},
           {"event": "short", "path": str(out), "video_id": "zz", "clip_id": 1, "title": "big fail #shorts"}]:
    print(json.dumps(ev), flush=True)
print("plain text from a library", flush=True)
""")
runner = rs.Runner(runs, [sys.executable, str(FAKE_HARVEST)])
server = rs.make_server(runs, 0, runner=runner)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_address[1]}"


def req(path, body=None, headers=None):
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    try:
        with urllib.request.urlopen(urllib.request.Request(base + path, data, headers or {}), timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


jget = lambda path: json.loads(req(path)[2])
names = [r["name"] for r in jget("/api/runs")["runs"]]
assert "b1" in names and "gated" in names
d = jget("/api/clips?batch=b1")
assert d["batch"] == "b1" and len(d["clips"]) == 40 and d["labels"] == rs.LABELS and (batch / "decision_log.json").exists()
assert req("/api/clips?batch=../b1")[0] == 404 and req("/api/clips?batch=nope")[0] == 404

size = media.stat().st_size
full = media.read_bytes()
SHORT = "/files/b1/shorts/twitch_a0_1_short.mp4"
code, h, body = req(SHORT, headers={"Range": "bytes=10-19"})
assert code == 206 and body == full[10:20] and h["Content-Range"] == f"bytes 10-19/{size}" and h["Accept-Ranges"] == "bytes"
assert h["Content-Type"] == "video/mp4"
code, h, body = req(SHORT, headers={"Range": "bytes=-5"})
assert code == 206 and body == full[-5:]
code, h, body = req(SHORT, headers={"Range": f"bytes={size - 3}-"})
assert code == 206 and body == full[-3:] and h["Content-Length"] == "3"
code, h, _ = req(SHORT, headers={"Range": f"bytes={size}-"})
assert code == 416 and h["Content-Range"] == f"bytes */{size}"
code, _, body = req(SHORT)
assert code == 200 and body == full
for evil in ("/files/b1/../../fake_clipper.py", "/files/b1/%2e%2e/%2e%2e/fake_clipper.py", "/files/b1/shorts/nope.mp4",
             "/files/../fake_clipper.py"):
    assert req(evil)[0] == 404, evil
assert req("/")[0] == 200 and b"Tune-O-Clip Studio" in req("/")[2]

# reviews + labels save at once and merge
assert req("/api/review", {"batch": "b1", "id": "twitch_a0_1", "rating": 8, "tags": ["Banger / nailed it", "Cut too soon"]})[0] == 200
assert req("/api/review", {"batch": "b1", "id": "twitch_a0_1", "liked": "the zoom on the scream", "labels": ["Keep"]})[0] == 200
rejected_id = next(e["id"] for e in log if e["clipper"]["judge_reject"])
assert req("/api/review", {"batch": "b1", "id": rejected_id, "rating": 9, "disliked": "captions late",
                           "labels": ["Wrong Archetype", "Keep"]})[0] == 200
assert req("/api/review", {"batch": "b1", "id": "twitch_a4_1", "labels": ["Cut / Boring"]})[0] == 200
for bad in ({"id": "twitch_a0_1", "rating": 11}, {"id": "twitch_a0_1", "rating": True}, {"id": "nope", "rating": 5},
            {"id": "twitch_a0_1", "tags": ["Not a tag"]}, {"id": "twitch_a0_1", "liked": 5},
            {"id": "twitch_a0_1", "labels": ["Keep", "Cut / Boring"]}, {"id": "twitch_a0_1", "labels": ["Meh"]}):
    assert req("/api/review", dict(bad, batch="b1"))[0] == 400, bad
assert req("/api/review", {"batch": "nope", "id": "twitch_a0_1", "rating": 5})[0] == 400
assert req("/api/review", b"not json")[0] == 400
saved = json.loads((batch / "reviews.json").read_text(encoding="utf-8"))
assert saved["twitch_a0_1"]["rating"] == 8 and saved["twitch_a0_1"]["liked"] == "the zoom on the scream"
assert saved["twitch_a0_1"]["tags"] == ["Cut too soon", "Banger / nailed it"] and saved["twitch_a0_1"]["labels"] == ["Keep"]
assert saved[rejected_id]["labels"] == ["Keep", "Wrong Archetype"]  # canonical order

# tuner: the gate re-run with scorer's own functions; saved weights untouched until Save
assert (scorer.DEFAULTS["MIN_PAYOFF"], scorer.DEFAULTS["MIN_HOOK"]) == (0, 0), "the dead-air floor is off by default"
scorer.MIN_PAYOFF, scorer.MIN_HOOK = 5, 5  # a tuned floor, as if weights.json had set it
w = jget("/api/weights")
assert w["weights"]["MIN_PAYOFF"] == 5 and w["min_score"] == 0 and w["defaults"]["PILLARS"]["comedy_payoff"] == 0.35
judged_reject = {e["id"] for e in log if e["clipper"]["judge_reject"]}  # fake pillars: payoff 2 / hook 4 = dead air
t = json.loads(req("/api/tune", {"batch": "b1", "weights": w["weights"], "min_score": 0})[2])
assert t["now"] == t["new"] and t["new"]["pass"] == 40 - len(judged_reject) and t["new"]["dead_air"] == len(judged_reject)
assert t["agree"] == {"keep_pass": 1, "keep_blocked": 1, "cut_pass": 1, "cut_blocked": 0}, t["agree"]  # a0 / rejected / a4
assert {m["id"] for m in t["mismatches"]} == {rejected_id, "twitch_a4_1"} and t["flips"] == []
looser = dict(w["weights"], MIN_HOOK=4)  # hook 4 is no longer "< MIN_HOOK": nothing is dead air
t = json.loads(req("/api/tune", {"batch": "b1", "weights": looser, "min_score": 0})[2])
assert t["new"]["pass"] == 40 and set(t["flips"]) == judged_reject and t["now"]["pass"] == 40 - len(judged_reject)
t = json.loads(req("/api/tune", {"batch": "b1", "weights": looser, "min_score": 7.5})[2])
assert t["new"]["below_min"] > 0 and scorer.MIN_HOOK == 5, "the evaluation never leaks into the live module"
assert req("/api/tune", {"batch": "b1", "weights": dict(looser, MIN_HOOK=-1)})[0] == 400
assert req("/api/tune", {"batch": "b1", "weights": dict(looser, PILLARS={"comedy_payoff": 1})})[0] == 400
assert not scorer.WEIGHTS_FILE.exists()
code, _, body = req("/api/weights/save", {"weights": looser, "min_score": 6.5})
assert code == 200 and json.loads(scorer.WEIGHTS_FILE.read_text()) == {"MIN_PAYOFF": 5, "MIN_HOOK": 4} and scorer.MIN_HOOK == 4
assert json.loads(rs.SETTINGS.read_text()) == {"min_score": 6.5, "clips": 4}, "other panel settings kept"
assert jget("/api/weights")["weights"]["MIN_HOOK"] == 4 and jget("/api/weights")["min_score"] == 6.5
assert req("/api/weights/save", {"weights": looser, "min_score": 11})[0] == 400

# export
code, _, body = req("/api/export", {"batch": "b1"})
assert code == 200 and json.loads(body)["summary"]["reviewed"] == 2
md = (batch / "CLAUDE_TUNING_REPORT.md").read_text(encoding="utf-8")
rep_json = json.loads((batch / "CLAUDE_TUNING_REPORT.json").read_text(encoding="utf-8"))
assert "**Human:** 8/10, AI 8.0 (gap +0.0); labels: Keep; tags: Cut too soon, Banger / nailed it" in md
assert "**What I Liked:** the zoom on the scream" in md and "**What I Didn't Like:** captions late" in md
assert "Biggest gaps" in md and f"`{rejected_id}`" in md and "By scenario" in md and "Wrong Archetype by scenario: chat_roast x1" in md
assert '"MIN_HOOK": 4' in md and "Min Comedy Score 6.5" in md
sm = rep_json["summary"]
assert sm["judge_rejected_but_liked"] == [rejected_id] and sm["biggest_gaps"][0]["id"] == rejected_id and sm["biggest_gaps"][0]["gap"] == 7.0
assert sm["labels"]["Keep"] == 2 and sm["labels"]["Cut / Boring"] == 1 and sm["by_streamer"]["alpha"]["n"] == 2

# harvest runner: start -> live state from the JSON lines -> finished; resume reuses run.json; stop kills it
for bad in ({"streamers": "", "per_streamer": 20, "hours": 168, "min_views": 500},
            {"streamers": "a", "per_streamer": 0, "hours": 168, "min_views": 500},
            {"streamers": "a", "per_streamer": 5, "hours": 0, "min_views": 500},
            {"resume": True, "batch": "nope"}):
    assert req("/api/harvest/start", bad)[0] == 400, bad


def wait_done(timeout=20):
    t0 = time.time()
    while runner.running() and time.time() - t0 < timeout:
        time.sleep(0.05)
    for _ in range(100):  # the reader thread records the exit just after the process ends
        s = jget("/api/harvest")
        if s.get("exit") is not None:
            return s
        time.sleep(0.05)
    return jget("/api/harvest")


d = json.loads(req("/api/harvest/start", {"streamers": "caseoh_, @Jynxzi", "per_streamer": 3, "hours": 24, "min_views": 50})[2])
new = d["batch"]
assert new.startswith(time.strftime("%Y-%m-%d")) and (runs / new).is_dir()
s = wait_done()
argv = json.loads((runs / new / "argv.json").read_text())
assert argv == ["--batch", new, "--streamers", "caseoh_,jynxzi", "--per-streamer", "3", "--hours", "24", "--min-views", "50"], argv
assert s["phase"] == "finished" and not s["running"] and s["exit"] == 0 and s["current"] is None, s
assert s["feed"] == [{"id": "zz_1", "title": "big fail #shorts", "short": "shorts/zz_1_short.mp4"}] and s["encoder"] == "h264_nvenc"
assert "plain text from a library" in s["log"] and "Processing twitch_Abc (attempt 1)" in s["log"]
assert s["counts"]["target"] == 6 and s["counts"]["rendered"] == 1 and s["counts"]["per_streamer"] == {"caseoh_": 0, "jynxzi": 0}
st = {"log": [], "feed": [], "phase": "", "batch": new}  # the stage tracking, line by line
for line in ['{"event": "queue", "msg": "Processing twitch_Abc (attempt 1)"}',
             '{"event": "log", "msg": "Twitch clip: big fail by Jynxzi (17s, 9 views)"}',
             '{"event": "stage", "stage": "scoring", "msg": "Whisper small.en on the clip"}']:
    runner.feed(line, st)
assert st["current"]["id"] == "twitch_Abc" and st["current"]["streamer"] == "jynxzi" and st["current"]["stage"] == "Whisper"
for msg, want in (("Ollama comedy judge", "Ollama Judge"), ("Directing with m", "AI Director"), ("Rendering x", "Render")):
    runner.feed(json.dumps({"event": "stage", "stage": "render" if want == "Render" else "edit", "msg": msg}), st)
    assert st["current"]["stage"] == want, (msg, st["current"])
runner.feed('{"event": "progress", "phase": "render", "id": "9_1", "streamer": "caseoh_", "title": "t", "n": 2, "of": 7}', st)
assert st["phase"] == "render" and st["current"]["id"] == "9_1" and st["current"]["stage"] == "Metadata" and st["current"]["of"] == 7

(runs / "b1" / "run.json").write_text(json.dumps({"streamers": ["slowpoke"], "per_streamer": 2, "hours": 720, "min_views": 0}))
assert req("/api/harvest/start", {"resume": True, "batch": "b1"})[0] == 200
assert req("/api/harvest/start", {"resume": True, "batch": "b1"})[0] == 400, "one run at a time"
time.sleep(0.5)
assert jget("/api/harvest")["running"]
t0 = time.time()
assert json.loads(req("/api/harvest/stop", {})[2])["ok"]
s = wait_done()
assert not s["running"] and s["phase"] == "stopped" and time.time() - t0 < 10, s
assert json.loads((runs / "b1" / "argv.json").read_text())[3:] == ["slowpoke", "--per-streamer", "2", "--hours", "720", "--min-views", "0"]
assert not json.loads(req("/api/harvest/stop", {})[2])["ok"]
server.shutdown()
server.server_close()

assert before == live()
shutil.rmtree(tmp, ignore_errors=True)
print("calibration ok")

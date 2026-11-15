"""Clip Rate worker tests: upload (fake clipper process + fake editor) -> short_ready, scan, send (training batch +
prompt), the tag -> label mapping, clipper.local_info, LOW_SPEC (Ollama kept warm, unloaded on demand, room for
NVENC), and the cross-run seen registry (a copy of the real runs/2026-11-14 queue.db: its clips are never downloaded again).
Run: .venv/Scripts/python calibration_studio/test_clip_rate.py  (no network, no Ollama, no GPU; needs ffmpeg)
"""
import contextlib
import io
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clip_rate_worker as w  # noqa: E402  (sets TOC_LOW_SPEC=1 before scorer loads)
import clipper  # noqa: E402
import scorer  # noqa: E402

hb = w.hb
tmp = Path(tempfile.mkdtemp(prefix="cliprate_test_"))
runs = tmp / "runs"


def events(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn(*args, **kw)
    return [json.loads(l) for l in out.getvalue().splitlines() if l.startswith("{")]


# a fake clipper: one whole clip into --output/local_<name>/, like clipper.twitch_clip
FAKE = tmp / "fake_clipper.py"
FAKE.write_text(r'''
import json, sys
from pathlib import Path
a = sys.argv[1:]
out, name = Path(a[a.index("--output") + 1]) / ("local_" + Path(a[0]).stem), Path(a[0]).stem
out.mkdir(parents=True, exist_ok=True)
(out / "clip_01_twitch.mp4").write_bytes(b"raw")
clip = {"clip_id": 1, "file": "clip_01_twitch.mp4", "duration": 12.0, "score": 0.5, "llm": 0.7, "transcript": "he lost it",
        "pillars": {"comedy_payoff": 7, "self_contained": 6, "hook_strength": 5, "quotability": 4}}
(out / "clips_manifest.json").write_text(json.dumps({"video_id": name, "title": name, "clips": [clip]}))
print(json.dumps({"event": "done", "count": 1}), flush=True)
''')


def fake_edit(batch, settings):
    def edit(manifest, cid, shorts):
        stem = json.loads(Path(manifest).read_text())["video_id"] + f"_{cid}"
        (shorts / f"{stem}_short.mp4").write_bytes(b"short")
        return {"ok": True, "edl": {"scenario": "chat_roast", "cuts": [{"start": 0.5, "end": 11.0}], "social_title": "LOL #shorts"}}
    return edit


hb.default_edit = fake_edit
src = tmp / "my clip.mp4"
src.write_bytes(b"x")

# upload -> the clipper's lines pass through, one short_ready card with absolute paths
ev = events(w.upload, src, runs=runs, clipper_cmd=[sys.executable, str(FAKE)])
cards = [e for e in ev if e["event"] == "short_ready"]
assert len(cards) == 1, ev
c = cards[0]
assert c["key"] == "clip-rate-uploads/my clip_1" and Path(c["short"]).exists(), c
assert c["editor"]["scenario"] == "chat_roast" and c["clipper"]["pillars"]["comedy_payoff"] == 7, c
assert any(e["event"] == "cr_progress" and e["completed"] == 1 for e in ev), ev
# a second upload pass renders nothing new (edit_all resumes)
again = events(w.render, runs / w.UPLOADS, {}, {"completed": 0, "total": 1})
assert not [e for e in again if e["event"] == "short_ready"], again

# scan finds it
ev = events(w.scan, runs)
assert [e["key"] for e in ev if e["event"] == "short_ready"] == [c["key"]] and ev[-1] == {"event": "scan_done", "count": 1}, ev

# review mapping: stars -> /10, tags -> studio labels, thumbs -> notes
r = w.review({"stars": 4, "like": False, "tags": ["Banger / Keep", "Bad Cut / Pacing", "nonsense"], "liked": "the scream",
              "disliked": "dead air", "extra_notes": "zoom later"})
assert r["rating"] == 8 and r["labels"] == ["Keep", "Bad Edit / Framing"] and r["notes"] == "Thumbs down. zoom later", r
assert r["liked"] == "the scream" and r["disliked"] == "dead air", r
assert w.review({"notes": "old style"})["notes"] == "old style"  # ratings saved before the three boxes
assert w.review({"like": True})["rating"] is None

# send: only 'rated' clips, unknown / traversal batch names skipped, md + json + prompt
rq = tmp / "clip_rate" / "ratings_queue.json"
rq.parent.mkdir()
rq.write_text(json.dumps({c["key"]: {"batch": "clip-rate-uploads", "stars": 5, "like": True, "tags": ["Banger / Keep"],
                                       "liked": "perfect timing", "disliked": "late caption",
                                       "extra_notes": "more like this", "status": "rated"},
                          "../evil/x_1": {"batch": "..", "stars": 1, "status": "rated"},
                          "old/x_1": {"batch": "clip-rate-uploads", "stars": 2, "status": "trained"}}))
ev = events(w.send, rq, runs=runs)
sent = next(e for e in ev if e["event"] == "sent")
assert sent["keys"] == [c["key"]] and str(rq.parent) in sent["prompt"], sent
md = Path(sent["md"]).read_text(encoding="utf-8")
assert "**What I Liked:** perfect timing" in md and "**What I Didn't Like:** late caption" in md, md
assert "**Extra Notes:** Thumbs up. more like this" in md and "**Transcript:** he lost it" in md, md
assert "Thumbs up." in md and "10/10" in md and "Keep x1" in md, md
data = json.loads(Path(sent["json"]).read_text(encoding="utf-8"))
assert data["summary"]["reviewed"] == 1 and data["clips"][0]["review"]["stars"] == 5, data["summary"]
# taste memory: profile created once and read first, every batch archived, instructions under the md title
profile = rq.with_name("TASTE_PROFILE.md")
assert sent["profile"] == str(profile) and "## Per-Streamer Calibration Notes" in profile.read_text(encoding="utf-8")
assert sent["prompt"].startswith(f"In the tune-o-clip project: FIRST read {profile}") and "superseded" in sent["prompt"], sent["prompt"]
assert "not computable" in sent["prompt"] and md.splitlines()[0].startswith("# Claude tuning report")
assert md.splitlines()[2].startswith(f"> **Read `{profile}` first.**"), md[:400]
archived = Path(sent["archive"])
assert archived.parent == rq.parent / "batches" and archived.name.startswith("batch_") and json.loads(archived.read_text(encoding="utf-8")) == data
profile.write_text("# my rules\n- never cut the laugh\n", encoding="utf-8")  # Claude's edits must survive the next send
rq.write_text(json.dumps({c["key"]: {"batch": "clip-rate-uploads", "stars": 2, "status": "rated"}}))
events(w.send, rq, runs=runs)
assert profile.read_text(encoding="utf-8") == "# my rules\n- never cut the laugh\n"
assert w.correlation([(10, 8), (6, 5), (2, 3)]) == 0.99 and w.correlation([(8, 7), (8, 7), (8, 7)]) is None
assert w.correlation([(8, 7), (4, 5)]) is None
rq.write_text(json.dumps({}))
assert events(w.main, ["--mode", "send", "--ratings", str(rq)])[0]["event"] == "error"

# clipper.local_info copies (never moves) the user's file into the job dir
info, job = clipper.local_info(src, tmp / "ws")
assert info["local"] and info["id"] == "my_clip" and (job / "source.mp4").exists() and src.exists(), info

# LOW_SPEC: Whisper is on the CPU, so Ollama stays warm (keep_alive 5m) and is unloaded only on demand
assert scorer.LOW_SPEC
seen = {}


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def opener(req, timeout):
    seen.update(json.loads(req.data))
    return Resp(json.dumps({"response": "{}"}).encode())


assert scorer.ollama_generate("hi", "m", opener=opener) == {} and seen["keep_alive"] == "5m", seen
assert scorer.ollama_warmup("m", url="http://127.0.0.1:9") is None and scorer.ollama_unload("http://127.0.0.1:9") == []  # offline: no-ops
free = scorer.vram_free_mib()
assert free is None or free > 0, free
sys.path.insert(0, str(HERE.parent / "editor"))
import pipeline  # noqa: E402
unloaded = []
assert pipeline.make_room_for_render(lambda: 120, lambda: unloaded.append(1) or ["m"]) and unloaded == [1]
assert not pipeline.make_room_for_render(lambda: 2000, lambda: unloaded.append(2) or []) and unloaded == [1]
assert not pipeline.make_room_for_render(lambda: None, lambda: unloaded.append(3) or []) and unloaded == [1]  # no NVIDIA GPU

# --- global seen registry: clips of ANY earlier run (here a copy of the real 2026-11-14 queue) are skipped before
# download; fetch pages / widens until it has N unseen clips; a re-clip of a known moment (same VOD +-30 s) too
real = HERE / "runs" / "2026-11-14" / "queue.db"
reg = tmp / "reg_runs"
(reg / "2026-11-14").mkdir(parents=True)
shutil.copy(real, reg / "2026-11-14" / "queue.db")
(reg / "clip-rate-uploads" / "output" / "twitch_UploadedSlug").mkdir(parents=True)  # a Twitch URL upload: no queue
con = sqlite3.connect(real)
old_rows = [(v[len("twitch_"):], json.loads(m)) for v, m in con.execute(
    "SELECT video_id, meta FROM jobs WHERE source = 'twitch' AND meta LIKE '%caseoh_%' ORDER BY id LIMIT 3")]
con.close()
assert len(old_rows) == 3, old_rows
moment = next(m for _, m in old_rows if m.get("vod_id") or m.get("video_id"))
clip_c = lambda cid, views, vod="", off=None: {"id": cid, "broadcaster_id": "caseoh_", "broadcaster_name": "caseoh_",
                                               "view_count": views, "created_at": "2026-11-14T10:00:00Z", "language": "en",
                                               "title": cid, "url": f"https://clips.twitch.tv/{cid}", "duration": 20.0,
                                               "video_id": vod, "vod_offset": off}
LIB = [clip_c(cid, 99999 - i) for i, (cid, _) in enumerate(old_rows)] + [clip_c("UploadedSlug", 50000)] +       [clip_c("ReclipSameMoment", 40000, moment.get("vod_id") or moment.get("video_id"), (moment.get("vod_offset") or 0) + 5)] +       [clip_c(f"Fresh{i}", 1000 - i) for i in range(4)]


class FakeTwitch:
    def __init__(self):
        self.windows = []

    def user_ids(self, logins):
        return {l: l for l in logins if l == "caseoh_"}

    def fetch_top_clips(self, broadcaster_id=None, hours_back=None, min_views=0, limit=25, language="en", pages=1, skip=None, **kw):
        self.windows.append(hours_back)
        return [c for c in LIB if c["view_count"] >= min_views and not (skip and skip(c))][:limit]


called = tmp / "called.txt"
FAKE2 = tmp / "fake_twitch_clipper.py"
FAKE2.write_text(r'''
import json, sys
from pathlib import Path
a = sys.argv[1:]
out, meta = Path(a[a.index("--output") + 1]), json.loads(a[a.index("--twitch-meta") + 1])
assert "--dedup-root" in a  # every run's output is checked for re-clipped transcripts
with open(sys.argv[0] + ".called", "a") as f:
    f.write(meta["clip_id"] + chr(10))
name = "twitch_" + meta["clip_id"]
(out / name).mkdir(parents=True, exist_ok=True)
(out / name / "clip_01_twitch.mp4").write_bytes(b"raw")
clip = {"clip_id": 1, "file": "clip_01_twitch.mp4", "duration": 20.0, "score": 0.5, "llm": 0.7, "transcript": "fresh one"}
(out / name / "clips_manifest.json").write_text(json.dumps({"video_id": meta["clip_id"], "twitch": meta, "clips": [clip]}))
print(json.dumps({"event": "done", "manifest": str(out / name / "clips_manifest.json"), "count": 1}), flush=True)
''')
tw = FakeTwitch()
ev = events(w.fetch, ["caseoh_"], 2, runs=reg, client=tw, clipper_cmd=[sys.executable, str(FAKE2)])
got = Path(str(FAKE2) + ".called").read_text().split()
assert got == ["Fresh0", "Fresh1"], (got, [e for e in ev if e["event"] in ("log", "warning")][:8])
assert any("Twitch clips already in calibration runs" in e.get("msg", "") for e in ev), ev[:3]
db = sqlite3.connect(reg / "clip-rate" / "queue.db")
known = {v for (v,) in db.execute("SELECT video_id FROM seen_videos")}
assert {f"twitch_{c}" for c, _ in old_rows} <= known and "twitch_UploadedSlug" in known, "real 2026-11-14 ids are seen"
db.close()
assert sqlite3.connect(real).execute("SELECT count(*) FROM jobs").fetchone()[0] == 84, "the real run is only read"
assert len([e for e in ev if e["event"] == "short_ready"]) == 2
# fetching 2 more pages past everything seen so far, and widens the window when the first one runs dry
ev = events(w.fetch, ["caseoh_"], 2, runs=reg, client=tw, clipper_cmd=[sys.executable, str(FAKE2)])
assert Path(str(FAKE2) + ".called").read_text().split() == ["Fresh0", "Fresh1", "Fresh2", "Fresh3"]
assert tw.windows[0] == 24, tw.windows
# scan shows each clip once even when two runs rendered it
shutil.copytree(reg / "clip-rate", reg / "zz-copy")
ev = events(w.scan, reg)
ids = [e["id"] for e in ev if e["event"] == "short_ready"]
assert len(ids) == len(set(ids)) == 4 and all(e["batch"] == "clip-rate" for e in ev if e["event"] == "short_ready"), ids

shutil.rmtree(tmp, ignore_errors=True)
print("clip rate worker: all tests passed")

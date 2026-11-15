"""Tune-O-Clip Calibration Studio: harvest with live progress, clip review + labels, scoring-weight tuner. One local app.

  Launch_TuneOClip_Studio.pyw (double-click)        its own window, no console; quits a minute after the window closes
  python calibration_studio/review_server.py        same app in the default browser (Ctrl+C to stop)

Stdlib HTTP server on 127.0.0.1:8787. Every run folder in runs/ opens, finished or interrupted: decision_log.json is
rebuilt from the manifests + shorts on disk each time a run is opened. Videos stream with HTTP Range (206).
"""
import argparse
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import harvest_batch as hb  # noqa: E402  (also puts backend/ on sys.path)
import scorer  # noqa: E402

RUNS = hb.RUNS
TAGS = ["Cut too soon", "Cut too late", "Didnt make sense / out of context", "Caption timing off", "Wrong editing style",
        "Facecam crop off", "Too loud / SFX spam", "Banger / nailed it"]
LABELS = ["Keep", "Cut / Boring", "Wrong Archetype", "Bad Edit / Framing", "Audio / Caption Drift"]
TEXT_FIELDS = ("liked", "disliked", "notes")
CHUNK = 64 * 1024
SETTINGS = ROOT / "workspace" / "settings.json"  # the Godot panel's settings: min_score lives there
NAME_RE = re.compile(r"(?!\.)[\w.-]{1,64}")  # no leading dot: never "." or ".."
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # Windows: the harvest + its ffmpeg children get no console
TWITCH_RE = re.compile(r"Twitch clip: (.*) by (\S+) \(")
DEFAULT_RUN = {"streamers": hb.twitch_api.parse_logins(hb.STREAMERS), "per_streamer": 20, "hours": 168, "min_views": 500}
IDLE_EXIT = 60  # app mode: seconds without a request from the window (and no harvest running) before quitting


def load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def clean_review(body, ids):
    """Validate one POST /api/review body -> (clip id, fields to merge). Raises ValueError on anything off."""
    cid = body.get("id")
    if cid not in ids:
        raise ValueError("unknown clip id")
    out = {}
    if "rating" in body:
        r = body["rating"]
        if r is not None and (not isinstance(r, int) or isinstance(r, bool) or not 1 <= r <= 10):
            raise ValueError("rating must be 1-10 or null")
        out["rating"] = r
    for key, allowed in (("tags", TAGS), ("labels", LABELS)):
        if key in body:
            if not isinstance(body[key], list) or any(t not in allowed for t in body[key]):
                raise ValueError(f"unknown {key[:-1]}")
            out[key] = [t for t in allowed if t in body[key]]  # canonical order, no duplicates
    if "Keep" in out.get("labels", []) and "Cut / Boring" in out["labels"]:
        raise ValueError("Keep and Cut / Boring exclude each other")
    for k in TEXT_FIELDS:
        if k in body:
            if not isinstance(body[k], str):
                raise ValueError(f"{k} must be text")
            out[k] = body[k][:20000]
    return cid, out


def ai_10(e):
    """The AI's verdict on the 0-10 scale the human uses: the judge when it ran, else the final score."""
    c = e["clipper"]
    return c["judge_10"] if c["judge_10"] is not None else round((c["final_score"] or 0) * 10, 1)


def avg(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


def human(r):
    """'keep' / 'cut' / None from a review: the Keep / Cut labels win, else rating >= 7 / <= 4."""
    labels, rating = r.get("labels") or [], r.get("rating")
    if "Keep" in labels or (rating and rating >= 7 and "Cut / Boring" not in labels):
        return "keep"
    if "Cut / Boring" in labels or (rating and rating <= 4):
        return "cut"
    return None


# --- tuner: gate a run's judged clips with candidate weights, using scorer.py's own functions ---------------------

TUNE_LOCK = threading.Lock()


def check_weights(w):
    """Tuner input -> every TUNABLE value (missing = the current one). ValueError on a bad value."""
    out = {}
    for k in scorer.TUNABLE:
        cur, v = getattr(scorer, k), (w or {}).get(k, getattr(scorer, k))
        ok = (isinstance(v, dict) and set(v) == set(cur) and all(scorer._weight_ok(x) for x in v.values())
              if isinstance(cur, dict) else scorer._weight_ok(v))
        if not ok:
            raise ValueError(f"bad value for {k}")
        out[k] = dict(v) if isinstance(v, dict) else v
    return out


def gate(e, min_score):
    """One clip's fate under the weights currently set on the scorer module (same calls the clipper makes)."""
    c = e["clipper"]
    if not c.get("pillars"):
        return {"judged": False, "pass": True, "ai": None, "why": "not judged (passes, like the live gate)"}
    v = scorer._verdict(c["pillars"])
    clip = {"llm": v["llm"], "judge_reject": v["reject"], "vocal": c["vocal_burst"] or 0, "reason": ""}
    scorer.viral_boost(clip, c["views"])
    scorer.streamer_boost(clip, e["streamer"])
    ok = scorer.passes_gate(clip, min_score)
    return {"judged": True, "pass": ok, "ai": round(clip["llm"] * 10, 1),
            "final": scorer.final_score(c["cue"] or 0, c["vocal_burst"] or 0, c["visual_action"], clip["llm"]),
            "why": "pass" if ok else "dead air (payoff + hook too low)" if v["reject"] else f"below {min_score:g}"}


def tune(entries, reviews, w, min_score, now_min):
    """Counts for the candidate weights vs the saved ones, agreement with the human labels, and the disagreements."""
    with TUNE_LOCK:  # the module globals are swapped for the evaluation, then restored
        now = [gate(e, now_min) for e in entries]
        saved = {k: getattr(scorer, k) for k in scorer.TUNABLE}
        try:
            for k, v in w.items():
                setattr(scorer, k, v)
            new = [gate(e, min_score) for e in entries]
        finally:
            for k, v in saved.items():
                setattr(scorer, k, v)

    def counts(rows):
        return {"total": len(rows), "judged": sum(r["judged"] for r in rows), "pass": sum(r["pass"] for r in rows),
                "dead_air": sum(r["why"].startswith("dead") for r in rows),
                "below_min": sum(r["why"].startswith("below") for r in rows)}

    agree = {"keep_pass": 0, "keep_blocked": 0, "cut_pass": 0, "cut_blocked": 0}
    mismatches = []
    for e, r in zip(entries, new):
        h = human(reviews.get(e["id"]) or {})
        if h:
            agree[f"{h}_{'pass' if r['pass'] else 'blocked'}"] += 1
            if (h == "keep") != r["pass"]:
                mismatches.append({"id": e["id"], "title": e["title"], "streamer": e["streamer"], "you": h, **r})
    return {"now": counts(now), "new": counts(new), "agree": agree, "mismatches": mismatches,
            "flips": [e["id"] for e, a, b in zip(entries, now, new) if a["pass"] != b["pass"]],
            "pillar_sum": round(sum(w["PILLARS"].values()), 3)}


def min_score_now():
    v = load_json(SETTINGS, {}).get("min_score", 0)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def save_weights(w, min_score):
    """backend/weights.json (only values that differ from the defaults) + min_score in workspace/settings.json."""
    diff = {k: v for k, v in w.items() if v != scorer.DEFAULTS[k]}
    hb.write_atomic(scorer.WEIGHTS_FILE, json.dumps(diff, indent=2) + "\n")
    scorer.load_weights()
    settings = load_json(SETTINGS, {})
    settings["min_score"] = min_score
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    hb.write_atomic(SETTINGS, json.dumps(settings, indent=2))
    return diff


# --- tuning report ------------------------------------------------------------------------------------------------

def build_report(batch, entries, reviews):
    """-> (markdown, json) pairing every clip's metrics + EDL decisions with the human review, plus where they diverge."""
    rated = [(e, reviews[e["id"]]) for e in entries if (reviews.get(e["id"]) or {}).get("rating")]
    diffs = sorted(((r["rating"] - ai_10(e), e, r) for e, r in rated), key=lambda x: -abs(x[0]))

    def group(key):
        out = {}
        for k in sorted({key(e) for e, _ in rated}):
            v = [(e, r) for e, r in rated if key(e) == k]
            out[k] = {"n": len(v), "human": avg([r["rating"] for _, r in v]), "ai": avg([ai_10(e) for e, _ in v])}
        return out

    tags = {t: sum(t in (r.get("tags") or []) for _, r in rated) for t in TAGS}
    labels = {t: sum(t in ((reviews.get(e["id"]) or {}).get("labels") or []) for e in entries) for t in LABELS}
    wrong = {}
    for e in entries:
        if "Wrong Archetype" in ((reviews.get(e["id"]) or {}).get("labels") or []):
            wrong[e["editor"]["scenario"] or "no short"] = wrong.get(e["editor"]["scenario"] or "no short", 0) + 1
    summary = {"clips": len(entries), "reviewed": len(rated), "human_avg": avg([r["rating"] for _, r in rated]),
               "ai_avg": avg([ai_10(e) for e, _ in rated]),
               "mean_abs_gap": avg([abs(d) for d, _, _ in diffs]),
               "by_streamer": group(lambda e: e["streamer"]),
               "by_scenario": group(lambda e: e["editor"]["scenario"] or "no short"),
               "by_layout": group(lambda e: e["editor"]["layout"] or "no short"),
               "tags": tags, "labels": labels, "wrong_archetype_by_scenario": wrong,
               "weights": {k: getattr(scorer, k) for k in scorer.TUNABLE}, "min_score": min_score_now(),
               "judge_rejected_but_liked": [e["id"] for e, r in rated if e["clipper"]["judge_reject"] and r["rating"] >= 7],
               "biggest_gaps": [{"id": e["id"], "human": r["rating"], "ai": ai_10(e), "gap": round(d, 1)} for d, e, r in diffs[:10]]}
    data = {"batch": batch.name, "generated": time.strftime("%Y-%m-%d %H:%M"), "summary": summary,
            "clips": [dict(e, review=reviews.get(e["id"]) or {}) for e in entries]}

    table = lambda title, g: [f"### {title}", "", "| | n | human avg | AI avg |", "|---|---|---|---|"] + \
        [f"| {k} | {v['n']} | {v['human']} | {v['ai']} |" for k, v in g.items()] + [""]
    md = [f"# Claude tuning report: {batch.name}", "",
          "Human 1-10 ratings vs Tune-O-Clip's AI judge (backend/scorer.py) and director decisions (editor/director.py). "
          "AI = the Ollama judge's weighted pillar score /10 (the final score x10 when the judge did not run). "
          "Gap = human - AI: positive means the AI underrated the clip.", "",
          "## Summary", "",
          f"- Reviewed {summary['reviewed']} of {summary['clips']} clips. Human avg {summary['human_avg']}, AI avg "
          f"{summary['ai_avg']}, mean absolute gap {summary['mean_abs_gap']}.",
          "- Labels: " + (", ".join(f"{t} x{n}" for t, n in labels.items() if n) or "none")
          + ("; Wrong Archetype by scenario: " + ", ".join(f"{k} x{n}" for k, n in sorted(wrong.items())) if wrong else ""),
          "- Tags: " + (", ".join(f"{t} x{n}" for t, n in tags.items() if n) or "none"),
          "- Judge rejected but rated >= 7 by the human (the gate would have lost these): "
          + (", ".join(f"`{i}`" for i in summary["judge_rejected_but_liked"]) or "none"),
          f"- Scoring weights in use: `{json.dumps(summary['weights'])}`, Min Comedy Score {summary['min_score']:g}.", ""]
    md += table("By streamer", summary["by_streamer"]) + table("By scenario", summary["by_scenario"]) \
        + table("By layout", summary["by_layout"])
    md += ["### Biggest gaps", "", "| clip | human | AI | gap |", "|---|---|---|---|"] + \
        [f"| `{g['id']}` | {g['human']} | {g['ai']} | {g['gap']:+} |" for g in summary["biggest_gaps"]] + ["", "## Clips"]
    for e in entries:
        r = reviews.get(e["id"]) or {}
        md += ["", f"### {e['streamer']}: {e['title']} (`{e['id']}`)", ""] + hb.describe(e)
        if e.get("transcript"):
            md.append(f"- **Transcript:** {e['transcript']}")
        md.append(f"- **Human:** {r.get('rating') or 'not rated'}/10"
                  + (f", AI {ai_10(e)} (gap {r['rating'] - ai_10(e):+.1f})" if r.get("rating") else "")
                  + (f"; labels: {', '.join(r['labels'])}" if r.get("labels") else "")
                  + (f"; tags: {', '.join(r['tags'])}" if r.get("tags") else ""))
        md += [f"- **{label}:** {r[k].strip()}" for k, label in
               (("liked", "What I Liked"), ("disliked", "What I Didn't Like"), ("notes", "Extra Notes")) if (r.get(k) or "").strip()]
    return "\n".join(md) + "\n", data


# --- harvest runner -----------------------------------------------------------------------------------------------

def python_console():
    """python.exe next to pythonw.exe: a console-subsystem child given CREATE_NO_WINDOW owns a hidden console that
    its ffmpeg / clipper children share, so no window flashes. (pythonw children would each pop a console.)"""
    exe = Path(sys.executable)
    con = exe.with_name("python.exe")
    return str(con if exe.name.lower() == "pythonw.exe" and con.exists() else exe)


HARVEST = [python_console(), "-u", str(HERE / "harvest_batch.py")]


def run_counts(batch, params):
    per = params["per_streamer"]
    have = {s: 0 for s in params["streamers"]}
    for _, d, _ in hb.kept_clips(batch):
        s = str((d.get("twitch") or {}).get("broadcaster_name") or "").lower()
        have[s] = have.get(s, 0) + 1
    shorts = batch / "shorts"
    return {"per_streamer": have, "per": per, "target": per * len(params["streamers"]),
            "harvested": sum(min(n, per) for s, n in have.items() if s in params["streamers"]),
            "edited": len(list(shorts.glob("*_edit.json"))), "rendered": len(list(shorts.glob("*_short.mp4")))}


class Runner:
    """One harvest_batch.py child at a time; its JSON lines become the live state behind GET /api/harvest."""

    def __init__(self, runs, cmd=None):
        self.runs, self.cmd, self.proc, self.lock = Path(runs), cmd or HARVEST, None, threading.Lock()
        self.state = {}

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, name, params):
        with self.lock:
            if self.running():
                raise ValueError("a run is already going: Stop it first")
            batch = self.runs / name
            batch.mkdir(parents=True, exist_ok=True)
            cmd = self.cmd + ["--batch", name, "--streamers", ",".join(params["streamers"]),
                              "--per-streamer", str(params["per_streamer"]), "--hours", str(params["hours"]),
                              "--min-views", str(params["min_views"])]
            c = run_counts(batch, params)
            self.state = {"batch": name, "params": params, "phase": "starting", "started": time.time(),
                          "base": c["harvested"] + c["edited"], "current": None, "encoder": None, "feed": [], "log": [],
                          "exit": None, "stopping": False}
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace", cwd=str(ROOT), creationflags=NO_WINDOW,
                                         env={**os.environ, "PYTHONIOENCODING": "utf-8"}, start_new_session=os.name != "nt")
            threading.Thread(target=self._read, args=(self.proc, self.state), daemon=True).start()

    def _read(self, proc, st):
        for line in proc.stdout:
            self.feed(line, st)
        proc.wait()
        st["exit"] = proc.returncode
        st["phase"] = "stopped" if st["stopping"] else "finished" if proc.returncode == 0 else "failed"
        st["current"] = None

    def feed(self, line, st):
        """One output line -> state: the clip being worked on, its stage, finished shorts, the log tail."""
        try:
            ev = json.loads(line)
        except ValueError:
            ev = None
        if not isinstance(ev, dict):
            if line.strip():
                st["log"] = (st["log"] + [line.strip()[:300]])[-60:]
            return
        e, msg, now = ev.get("event"), str(ev.get("msg") or ""), time.time()
        cur = st.get("current")

        def clip(**kw):
            st["current"] = {"stage": "Metadata", "since": now, "stage_since": now, **kw}

        if e == "progress":
            st["phase"] = ev.get("phase") or st["phase"]
            if ev.get("id"):
                clip(id=ev["id"], streamer=ev.get("streamer") or "", title=ev.get("title") or "", n=ev.get("n"), of=ev.get("of"))
        elif e == "queue" and msg.startswith("Processing "):
            clip(id=msg.split()[1], streamer="", title="")
        elif e == "log" and TWITCH_RE.match(msg) and cur:
            m = TWITCH_RE.match(msg)
            cur.update(title=m[1], streamer=m[2].lower())
        elif e == "log" and msg.startswith("Encoder: "):
            st["encoder"] = msg.split(": ", 1)[1]
        elif e == "stage" and cur:
            name = ({"download": "Download", "render": "Render"}.get(ev.get("stage")) or
                    ("Whisper" if "Whisper" in msg else "Ollama Judge" if "Ollama" in msg else
                     "AI Director" if msg.startswith("Directing") else None))
            if name:
                cur.update(stage=name, stage_since=now)
        elif e == "short":
            try:
                rel = Path(ev["path"]).resolve().relative_to((self.runs / st["batch"]).resolve()).as_posix()
            except (KeyError, ValueError):
                rel = None
            st["feed"] = ([{"id": f"{ev.get('video_id')}_{ev.get('clip_id')}", "title": ev.get("title"), "short": rel}]
                          + st["feed"])[:24]
        if (e in ("log", "warning", "error") or (e == "queue" and msg.startswith("Processing"))) and msg:
            st["log"] = (st["log"] + [(f"[{e.upper()}] " if e in ("warning", "error") else "") + msg[:300]])[-60:]

    def stop(self):
        p = self.proc
        if not self.running():
            return False
        self.state["stopping"] = True
        if os.name == "nt":  # the whole tree: the clipper child and its ffmpeg too
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True, creationflags=NO_WINDOW)
        else:
            os.killpg(p.pid, signal.SIGTERM)
        return True

    def status(self):
        st = {k: v for k, v in self.state.items()}
        st["running"] = self.running()
        if st.get("batch"):
            c = run_counts(self.runs / st["batch"], st["params"])
            done, left = c["harvested"] + c["edited"], 2 * c["target"] - c["harvested"] - c["edited"]
            rate = (done - st["base"]) / max(1.0, time.time() - st["started"])
            st.update(counts=c, eta_s=round(left / rate) if st["running"] and rate > 0 else None, now=time.time())
        return st


# --- server ------------------------------------------------------------------------------------------------------

def new_run_name(runs):
    base = time.strftime("%Y-%m-%d")
    n, name = 1, base
    while (Path(runs) / name).exists():
        n += 1
        name = f"{base}-{n}"
    return name


def run_params(body):
    """Start-form values -> harvest params. ValueError on anything off."""
    logins = hb.twitch_api.parse_logins(str(body.get("streamers") or ""))
    per, hours, views = body.get("per_streamer"), body.get("hours"), body.get("min_views")
    if not logins:
        raise ValueError("enter at least one Twitch channel")
    if not (isinstance(per, int) and 1 <= per <= 200):
        raise ValueError("clips per channel: 1-200")
    if not (isinstance(hours, (int, float)) and 1 <= hours <= 24 * 365):
        raise ValueError("time window: 1-8760 hours")
    if not (isinstance(views, int) and views >= 0):
        raise ValueError("minimum views: 0 or more")
    return {"streamers": logins, "per_streamer": per, "hours": hours, "min_views": views}


def make_server(runs=RUNS, port=8787, host="127.0.0.1", runner=None):
    runs = Path(runs).resolve()
    runs.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    ids = {}  # batch -> clip ids, refreshed whenever the run is opened

    def batch_dir(name):
        if (not isinstance(name, str) or not NAME_RE.fullmatch(name) or not (runs / name).is_dir()
                or (runs / name).resolve().parent != runs):
            raise ValueError("unknown run")
        return runs / name

    def entries_of(batch):
        """The partial-log builder: decision_log.json from whatever manifests / shorts are on disk right now."""
        with lock:
            entries = hb.write_decision_log(batch)
        ids[batch.name] = {e["id"] for e in entries}
        return entries

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # scrubbing fires dozens of range requests: keep the console readable
            pass

        def send_json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            server.last_seen = time.time()
            u = urllib.parse.urlsplit(self.path)
            path, q = urllib.parse.unquote(u.path), dict(urllib.parse.parse_qsl(u.query))
            try:
                if path == "/":
                    return self.send_file(HERE / "static" / "index.html")
                if path == "/api/runs":
                    out = []
                    for p in sorted((p for p in runs.iterdir() if p.is_dir() and NAME_RE.fullmatch(p.name)),
                                    key=lambda p: p.stat().st_mtime, reverse=True):
                        out.append({"name": p.name, "clips": len(hb.kept_clips(p)),
                                    "shorts": len(list((p / "shorts").glob("*_short.mp4"))),
                                    "params": load_json(p / "run.json", None) or DEFAULT_RUN})
                    return self.send_json({"runs": out, "new_name": new_run_name(runs), "defaults": DEFAULT_RUN})
                if path == "/api/clips":
                    batch = batch_dir(q.get("batch"))
                    return self.send_json({"batch": batch.name, "clips": entries_of(batch), "tags": TAGS,
                                           "labels": LABELS, "reviews": load_json(batch / "reviews.json", {})})
                if path == "/api/harvest":
                    return self.send_json(server.runner.status())
                if path == "/api/weights":
                    return self.send_json({"weights": {k: getattr(scorer, k) for k in scorer.TUNABLE},
                                           "defaults": scorer.DEFAULTS, "min_score": min_score_now(),
                                           "file": str(scorer.WEIGHTS_FILE)})
                if path.startswith("/files/"):
                    name, _, rel = path[len("/files/"):].partition("/")
                    batch = batch_dir(name)
                    f = (batch / rel).resolve()
                    if f.is_relative_to(batch.resolve()) and f.is_file():  # no ../ escapes out of the run folder
                        return self.send_file(f)
            except ValueError as e:
                return self.send_json({"error": str(e)}, 404)
            self.send_error(404)

        def send_file(self, f):
            size = f.stat().st_size
            start, end, code = 0, size - 1, 200
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", (self.headers.get("Range") or "").strip())
            if m and (m[1] or m[2]):  # anything else (multi-range, junk) is ignored: the whole file
                if m[1]:
                    start, end = int(m[1]), min(int(m[2]), size - 1) if m[2] else size - 1
                else:
                    start = max(0, size - int(m[2]))
                if start >= size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                code = 206
            self.send_response(code)
            self.send_header("Content-Type", mimetypes.guess_type(f.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with open(f, "rb") as fh:
                    fh.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = fh.read(min(CHUNK, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except ConnectionError:  # the browser drops a range request mid-way whenever you scrub
                pass

        def do_POST(self):
            server.last_seen = time.time()
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("JSON object expected")
                return self.send_json(self.post(body))
            except ValueError as e:
                return self.send_json({"error": str(e)}, 400)

        def post(self, body):
            if self.path == "/api/review":
                batch = batch_dir(body.get("batch"))
                if batch.name not in ids:
                    entries_of(batch)
                cid, fields = clean_review(body, ids[batch.name])
                with lock:  # read-merge-write; temp file + replace: a crash never leaves half a file
                    reviews = load_json(batch / "reviews.json", {})
                    reviews[cid] = {**(reviews.get(cid) or {}), **fields, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
                    hb.write_atomic(batch / "reviews.json", json.dumps(reviews, indent=2, ensure_ascii=False))
                return {"ok": True, "review": reviews[cid]}
            if self.path == "/api/export":
                batch = batch_dir(body.get("batch"))
                md, data = build_report(batch, entries_of(batch), load_json(batch / "reviews.json", {}))
                hb.write_atomic(batch / "CLAUDE_TUNING_REPORT.md", md)
                hb.write_atomic(batch / "CLAUDE_TUNING_REPORT.json", json.dumps(data, indent=2, ensure_ascii=False))
                return {"ok": True, "md": str(batch / "CLAUDE_TUNING_REPORT.md"), "summary": data["summary"]}
            if self.path == "/api/tune":
                batch = batch_dir(body.get("batch"))
                ms = body.get("min_score", min_score_now())
                if not scorer._weight_ok(ms) or ms > 10:
                    raise ValueError("min score: 0-10")
                return tune(hb.decision_log(batch), load_json(batch / "reviews.json", {}), check_weights(body.get("weights")),
                            ms, min_score_now())
            if self.path == "/api/weights/save":
                ms = body.get("min_score")
                if not scorer._weight_ok(ms) or ms > 10:
                    raise ValueError("min score: 0-10")
                return {"ok": True, "saved": save_weights(check_weights(body.get("weights")), ms),
                        "file": str(scorer.WEIGHTS_FILE)}
            if self.path == "/api/harvest/start":
                if body.get("resume"):
                    batch = batch_dir(body.get("batch"))
                    params = load_json(batch / "run.json", None) or DEFAULT_RUN
                    name = batch.name
                else:
                    params, name = run_params(body), new_run_name(runs)
                server.runner.start(name, params)
                return {"ok": True, "batch": name}
            if self.path == "/api/harvest/stop":
                return {"ok": server.runner.stop()}
            raise ValueError("unknown endpoint")

    server = ThreadingHTTPServer((host, port), Handler)
    server.runner = runner or Runner(runs)
    server.last_seen = time.time()
    return server


def open_window(url):
    """A chromeless app window (Edge / Chrome / Chromium --app, own profile so it is its own process and may
    autoplay), else the default browser."""
    cands = [shutil.which(n) for n in ("msedge", "microsoft-edge", "microsoft-edge-stable", "google-chrome",
                                       "chromium", "chromium-browser", "chrome")]
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if base:
            cands += [os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                      os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")]
    exe = next((c for c in cands if c and os.path.isfile(c)), None)
    if not exe:
        webbrowser.open(url)
        return
    subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={HERE / '.app_profile'}", "--window-size=1520,960",
                      "--no-first-run", "--autoplay-policy=no-user-gesture-required"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)


def studio_alive(url):
    try:
        urllib.request.urlopen(url + "api/runs", timeout=3).close()
        return True
    except OSError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--app", action="store_true", help="chromeless window; quit when it has been closed for a minute")
    a = ap.parse_args(argv)
    url = f"http://127.0.0.1:{a.port}/"
    try:
        server = make_server(RUNS, a.port)
    except OSError:
        if studio_alive(url):  # already running (second double-click): just show it again
            if not a.no_browser:
                open_window(url) if a.app else webbrowser.open(url)
            return 0
        raise
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Tune-O-Clip Studio: {url}" + ("" if a.app else "  (Ctrl+C to stop)"), flush=True)
    if not a.no_browser:
        open_window(url) if a.app else webbrowser.open(url)
    try:
        while not (a.app and time.time() - server.last_seen > IDLE_EXIT and not server.runner.running()):
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    server.runner.stop()  # Ctrl+C: don't leave a hidden harvest behind (it resumes next time)
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

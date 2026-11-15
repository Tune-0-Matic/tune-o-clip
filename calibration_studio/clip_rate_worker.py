"""Clip Rate's backend: the Godot app in Documents/clip-rate spawns this and reads its JSON lines from stdout.

  python calibration_studio/clip_rate_worker.py --mode scan
  python calibration_studio/clip_rate_worker.py --mode fetch --channels caseoh_,ishowspeed --per-channel 5
  python calibration_studio/clip_rate_worker.py --mode upload --file C:/clips/funny.mp4   (or a Twitch clip URL)
  python calibration_studio/clip_rate_worker.py --mode send --ratings calibration_studio/clip_rate/ratings_queue.json

fetch and upload are harvest_batch (clipper --no-gate, then editor/pipeline) in their own runs/ batches: runs/clip-rate
(each fetch adds --per-channel MORE clips per streamer, so "+ Fetch 5 More" is just another fetch) and
runs/clip-rate-uploads. scan lists the shorts of every runs/ batch (the studio's too). One clip at a time, with
TOC_LOW_SPEC=1 (see scorer.LOW_SPEC: Whisper and Ollama never share the GTX 1050 Ti's 4 GB).
"""
import os

os.environ["TOC_LOW_SPEC"] = "1"  # before scorer is imported; the clipper children inherit it

import argparse  # noqa: E402
import json  # noqa: E402
import sqlite3  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harvest_batch as hb  # noqa: E402
import review_server as rs  # noqa: E402  (build_report: the studio's Claude tuning report)

emit = hb.emit
FETCH, UPLOADS = "clip-rate", "clip-rate-uploads"
# fetch widens until each streamer has N unseen clips: (hours back or None = all time, min views, language)
TIERS = [(24, 500, "en"), (72, 500, "en"), (168, 500, "en"), (720, 100, "en"), (None, 0, "en"), (None, 0, None)]
# Clip Rate's quick tags -> the studio's review labels, which build_report counts
LABELS = {"Banger / Keep": "Keep", "Boring / No Hook": "Cut / Boring", "Wrong Archetype": "Wrong Archetype",
          "Bad Cut / Pacing": "Bad Edit / Framing", "Caption / Audio Drift": "Audio / Caption Drift"}
PROMPT = ("In the tune-o-clip project: FIRST read {profile} (my taste profile: every rule learned from earlier batches). "
          "Then read {md} (full data in {json} beside it): my Clip Rate ratings of {n} Tune-O-Clip shorts (stars x2 = /10, "
          "thumbs up/down, quick tags, What I Liked / What I Didn't Like / Extra Notes) next to what tune-o-clip decided for "
          "each one (Whisper transcript, the Ollama judge's pillars, score and reason, the director's scenario, cuts, zooms "
          "and SFX, the scoring weights in use). Human vs AI correlation for this batch: {corr}.\n"
          "1. Merge this batch into the taste profile: turn what I liked into boost rules and what I didn't like into "
          "penalty / cut rules, rewrite the judge's noisy reasons into clear rules, and file them under the right section "
          "and streamer. Keep every older rule unless this batch contradicts it; then mark it superseded (date + why) "
          "instead of silently deleting it.\n"
          "2. Update backend/scorer.py (weights, judge prompt) and editor/director.py (scenario choice, cuts, zooms) so "
          "the code follows the whole profile, not just this batch, and run tune-o-clip's tests.\n"
          "3. Add a Batch Change Log entry: date, batch size, the correlation above as 'before', the correlation after "
          "re-scoring these clips with the changed scorer as 'after' (weight changes can be re-scored offline; say so if "
          "a judge-prompt change needs fresh Ollama runs to measure), and the exact code changes made.")
TASTE_TEMPLATE = """# Tune-O-Clip taste profile

My comedy and editing taste, learned from Clip Rate batches. Claude reads this FIRST on every "Send to Claude",
merges the new batch into it, and keeps older rules unless a newer batch contradicts them (then marks them
superseded with the date instead of deleting them). backend/scorer.py and editor/director.py must follow
the whole profile, not just the latest batch.

## Core Comedy & Hook Preferences
What makes a clip an instant keep vs. a skip.

- (no rules yet: the first batch fills this in)

## Hard Reject / Penalty Rules
Dead air, reading chat with no payoff, setup that needs earlier context, etc.

- (no rules yet)

## Editing, Pacing & Framing Rules
Cut timing, zoom behaviour, archetype choice (chat_roast, unfiltered_chaos, horror_jumpscare, confident_fail,
awkward_irl), captions, SFX.

- (no rules yet)

## Per-Streamer Calibration Notes

### caseoh_
- (no notes yet)

### ishowspeed
- (no notes yet)

### jynxzi
- (no notes yet)

### lospollostv
- (no notes yet)

## Batch Change Log
One entry per batch: date, batch size, human vs AI correlation before and after, and the exact code changes.

| date | clips | correlation before | correlation after | code changes |
|---|---|---|---|---|
"""


def ensure_profile(path):
    """Create the taste profile from TASTE_TEMPLATE if it is missing; never touches an existing one."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(TASTE_TEMPLATE, encoding="utf-8")
    return path


def correlation(pairs):
    """Pearson r of [(human /10, AI /10)] rounded, or None when it can't be computed (< 3 clips or no spread)."""
    try:
        return round(statistics.correlation(*zip(*pairs)), 2) if len(pairs) >= 3 else None
    except statistics.StatisticsError:
        return None


def card(batch, e):
    """decision_log entry -> one scroller card (absolute paths; key = batch/id is unique across runs)."""
    return dict(e, batch=batch.name, key=f"{batch.name}/{e['id']}", short=str(batch / e["short"]),
                thumb=str(batch / e["thumb"]) if e.get("thumb") else "")


def scan(runs=hb.RUNS):
    """Every short of every run, each clip once: a clip also rendered by a later run (runs/clip-rate re-fetched the
    2026-11-14 clips before the seen registry existed) shows only from the first run, which holds its rating."""
    n, shown = 0, set()
    for batch in sorted(p for p in runs.iterdir() if p.is_dir()) if runs.exists() else []:
        for e in hb.decision_log(batch):
            if e["short"] and e["id"] not in shown:
                shown.add(e["id"])
                emit("short_ready", **card(batch, e))
                n += 1
    emit("scan_done", count=n)


def seed_seen(db, runs, own):
    """Global seen registry: every Twitch clip any OTHER run queued, processed or archived becomes known to this
    batch's queue.db, so harvest skips it at the metadata check (before any download or Whisper) and pages deeper.
    Other runs' jobs are copied as 'done' rows with their Helix metadata (same-moment VOD checks see them too);
    their twitch_* output folders cover clips that never had a queue (Twitch URL uploads). Other runs are only
    read. -> number of clip ids now known."""
    ids = set()
    for b in sorted(p for p in runs.iterdir() if p.is_dir() and p.resolve() != own.resolve()):
        ids |= {p.name for p in (b / "output").glob("twitch_*") if p.is_dir()}
        q = b / "queue.db"
        if not q.exists():
            continue
        con = sqlite3.connect(f"file:{q.as_posix()}?mode=ro", uri=True)
        try:
            for vid, meta in con.execute("SELECT video_id, meta FROM jobs WHERE source = 'twitch'"):
                db.execute("INSERT OR IGNORE INTO jobs (url, video_id, source, meta, status) VALUES ('', ?, 'twitch', ?, 'done')",
                           (vid, meta))
                ids.add(vid)
            ids |= {v for (v,) in con.execute("SELECT video_id FROM seen_videos WHERE video_id LIKE 'twitch_%'")}
        finally:
            con.close()
    db.executemany("INSERT OR IGNORE INTO seen_videos (video_id, source) VALUES (?, 'twitch')", [(i,) for i in sorted(ids)])
    return len(ids)


def dedup_args(runs):
    """clipper --dedup-root for every run's output/: a re-clip of a known moment (new slug, same words) is caught
    after Whisper even without a VOD offset, whatever its age."""
    return [x for b in sorted(p for p in runs.iterdir() if (p / "output").is_dir()) for x in ("--dedup-root", str(b / "output"))]


def render(batch, settings, prog):
    """Edit every kept clip of the batch that has no short yet; each finished one is a short_ready event."""
    def done(stem):
        prog["completed"] += 1
        prog["total"] = max(prog["total"], prog["completed"])
        e = next((x for x in hb.decision_log(batch) if x["id"] == stem), None)
        if e and e["short"]:
            emit("short_ready", **card(batch, e))
        emit("cr_progress", stage="Short ready", current=stem, **prog)
    hb.edit_all(batch, hb.default_edit(batch, settings), done)
    hb.write_decision_log(batch)


def render(batch, settings, prog):
    """Edit every kept clip of the batch that has no short yet; each finished one is a short_ready event."""
    def done(stem):
        prog["completed"] += 1
        prog["total"] = max(prog["total"], prog["completed"])
        e = next((x for x in hb.decision_log(batch) if x["id"] == stem), None)
        if e and e["short"]:
            emit("short_ready", **card(batch, e))
        emit("cr_progress", stage="Short ready", current=stem, **prog)
    hb.edit_all(batch, hb.default_edit(batch, settings), done)
    hb.write_decision_log(batch)


def fetch(logins, per, runs=hb.RUNS, client=None, clipper_cmd=hb.qw.CLIPPER):
    batch = runs / FETCH
    batch.mkdir(parents=True, exist_ok=True)
    client = client or hb.twitch_api.Twitch()
    known = seed_seen(hb.qw.connect(batch / "queue.db"), runs, batch)
    emit("log", msg=f"[CLIP RATE] {known} Twitch clips already in calibration runs: skipped before download")
    settings, ids = hb.load_settings(), client.user_ids(logins)
    args = hb.clipper_args(settings) + dedup_args(runs)
    prog = {"completed": 0, "total": per * sum(bool(ids.get(x)) for x in logins)}
    for login in logins:
        if not ids.get(login):
            emit("warning", msg=f"Twitch user not found: {login!r}")
            continue
        emit("cr_progress", stage=f"Fetching {login}", current=login, **prog)
        hb.harvest(batch, [login], hb.count_for(batch, ids[login]) + per, client, args, clipper_cmd=clipper_cmd, tiers=TIERS)
        render(batch, settings, prog)  # this streamer's shorts reach the scroller before the next one downloads


def upload(src, runs=hb.RUNS, clipper_cmd=hb.qw.CLIPPER):
    """A local video file (one whole clip) or a Twitch clip URL -> the same clipper + editor as a fetch."""
    batch = runs / UPLOADS
    settings = hb.load_settings()
    emit("cr_progress", stage="Scoring upload", current=str(src), completed=0, total=1)
    # the clipper's JSON lines go straight through to Godot (our stdout)
    rc = subprocess.run(list(clipper_cmd) + [str(src), "--workspace", str(batch / "workspace"),
                                             "--output", str(batch / "output")] + hb.clipper_args(settings),
                        stderr=subprocess.STDOUT).returncode  # one pipe: a full stderr pipe can't stall Godot's reader
    if rc:
        raise RuntimeError(f"clipper failed on {src} (exit {rc})")
    render(batch, settings, {"completed": 0, "total": 1})


def review(r):
    """One Clip Rate rating -> the studio's review shape (build_report reads rating /10, labels, liked, disliked,
    notes). Ratings saved before the three boxes existed only have "notes": that becomes Extra Notes."""
    thumb = {True: "Thumbs up.", False: "Thumbs down."}.get(r.get("like"), "")
    return {"rating": int(r["stars"]) * 2 if r.get("stars") else None,
            "labels": [LABELS[t] for t in r.get("tags") or [] if t in LABELS],
            "liked": r.get("liked") or "", "disliked": r.get("disliked") or "",
            "notes": f"{thumb} {r.get('extra_notes') or r.get('notes') or ''}".strip(), "stars": r.get("stars"), "like": r.get("like"),
            "tags": []}  # the studio's own tag list is a different set: Clip Rate's tags are labels above


def send(ratings_path, runs=hb.RUNS):
    """Every 'rated' clip of ratings_queue.json + its full decision trace -> CLAUDE_TRAINING_BATCH.md/.json next to
    it, and the prompt to paste into Claude Code. Clips whose files are gone are left out (and stay 'rated')."""
    ratings_path = Path(ratings_path)
    todo = {k: r for k, r in json.loads(ratings_path.read_text(encoding="utf-8")).items() if r.get("status") == "rated"}
    entries, reviews = [], {}
    for name in sorted({str(r.get("batch")) for r in todo.values()}):
        if not rs.NAME_RE.fullmatch(name) or not (runs / name).is_dir():
            continue
        for e in hb.decision_log(runs / name):
            key = f"{name}/{e['id']}"
            if key in todo:
                entries.append(dict(e, id=key))
                reviews[key] = review(todo[key])
    if not entries:
        raise ValueError("No rated clips to send (rate some in the scroller first)")
    profile = ensure_profile(ratings_path.with_name("TASTE_PROFILE.md"))
    md, data = rs.build_report(Path("Clip Rate training batch"), entries, reviews)
    r = correlation([(reviews[e["id"]]["rating"], rs.ai_10(e)) for e in entries if reviews[e["id"]]["rating"]])
    corr = f"r = {r} (Pearson, human rating vs AI /10)" if r is not None else "not computable (needs 3+ starred clips with some spread)"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    data.update(taste_profile=str(profile), human_ai_correlation=r, sent=stamp)
    head = [f"> **Read `{profile}` first.** Merge this batch into it (keep older rules unless contradicted), then update",
            "> backend/scorer.py and editor/director.py to follow the whole profile, and log the batch in its Batch Change Log.",
            f"> Human vs AI correlation for this batch: {corr}. Archived as `batches/batch_{stamp}.json`.", ""]
    out_md, out_json = ratings_path.with_name("CLAUDE_TRAINING_BATCH.md"), ratings_path.with_name("CLAUDE_TRAINING_BATCH.json")
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    archive = ratings_path.with_name("batches") / f"batch_{stamp}.json"  # every batch kept; the two files below are the latest
    archive.parent.mkdir(exist_ok=True)
    while archive.exists():  # two sends in the same second never overwrite an archived batch
        archive = archive.with_name(archive.stem + "_1.json")
    hb.write_atomic(archive, payload)
    hb.write_atomic(out_md, md.replace("\n", "\n\n" + "\n".join(head), 1))  # instructions right under the title
    hb.write_atomic(out_json, payload)
    emit("sent", md=str(out_md), json=str(out_json), archive=str(archive), profile=str(profile),
         keys=[e["id"] for e in entries], prompt=PROMPT.format(profile=profile, md=out_md, json=out_json.name, n=len(entries), corr=corr))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["scan", "fetch", "upload", "send", "unload"],
                    help="unload: free Ollama's VRAM now (Clip Rate runs it after Cancel)")
    ap.add_argument("--channels", default=hb.STREAMERS, help="fetch: Twitch logins, comma separated")
    ap.add_argument("--per-channel", type=int, default=5, help="fetch: how many MORE clips per streamer (default 5)")
    ap.add_argument("--file", help="upload: a local video file or a Twitch clip URL")
    ap.add_argument("--ratings", help="send: ratings_queue.json")
    a = ap.parse_args(argv)
    try:
        if a.mode == "scan":
            scan()
        elif a.mode == "fetch":
            fetch(hb.twitch_api.parse_logins(a.channels), max(1, a.per_channel))
        elif a.mode == "upload":
            if not a.file:
                raise ValueError("--file is required")
            upload(a.file)
        elif a.mode == "unload":
            emit("log", msg=f"Ollama unloaded: {', '.join(hb.scorer.ollama_unload()) or 'nothing was loaded'}")
        else:
            if not a.ratings:
                raise ValueError("--ratings is required")
            send(a.ratings)
    except Exception as e:  # Godot shows it on the banner; the exit code says it failed
        emit("error", msg=f"{type(e).__name__}: {e}")
        return 1
    finally:
        if a.mode in ("fetch", "upload"):  # the batch kept Ollama warm (scorer.LOW_SPEC): give its VRAM back
            hb.scorer.ollama_unload()
    emit("worker_done", mode=a.mode)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)

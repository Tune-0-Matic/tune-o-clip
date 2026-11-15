"""Calibration batch: N Twitch clips per streamer, judged and edited with NO banger gate, into
calibration_studio/runs/<batch>/ (its own queue.db, output/ and shorts/), so nothing reaches ready_to_post/,
posted/ or the live queue's seen_videos.

  .venv/Scripts/python calibration_studio/harvest_batch.py                 # 20 each of the 4 default streamers
  python calibration_studio/harvest_batch.py --streamers caseoh_,jynxzi --per-streamer 5 --batch quick

Re-running with the same --batch resumes: processed clips (its queue.db) and finished edits are skipped, so an
interrupted render only renders what is left (no re-download, and Whisper words are cached in the manifest).
Usually driven by the studio app (Launch_TuneOClip_Studio.pyw), which reads the JSON lines this prints.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
sys.path.insert(0, str(ROOT / "backend"))

import queue_worker as qw  # noqa: E402
import scorer  # noqa: E402
import twitch_api  # noqa: E402

emit = qw.emit
STREAMERS = "caseoh_,ishowspeed,jynxzi,lospollostv"
# widened in this order until a streamer has enough fresh clips: (hours back or None = all time, min views, language)
TIERS = [(168, 500, "en"), (720, 100, "en"), (None, 0, "en"), (None, 0, None)]


def kept_clips(batch):
    """[(manifest path, manifest, clip)] for every clip the batch kept (archived in batch/output/)."""
    out = []
    for m in sorted((batch / "output").glob("*/clips_manifest.json")):
        try:
            data = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out += [(m, data, c) for c in data.get("clips") or []]
    return out


def count_for(batch, uid):
    return sum((d.get("twitch") or {}).get("broadcaster_id") == uid for _, d, _ in kept_clips(batch))


def harvest(batch, logins, per, client, clipper_args=(), clipper_cmd=qw.CLIPPER, sleep=time.sleep, tiers=TIERS):
    """Download + Whisper + judge (no gate) until every streamer has `per` distinct clips. Each round asks Twitch for
    the clips still missing, skipping ones this batch already has or the same stream moment; a round that adds
    nothing (duplicate transcript, failed download) simply asks again. Then the next TIER widens the search."""
    db = qw.connect(batch / "queue.db")
    run = lambda: qw.drain(db, list(clipper_args), batch / "workspace", batch / "output", clipper_cmd=clipper_cmd, sleep=sleep)
    run()  # an interrupted earlier run's leftovers first
    ids = client.user_ids(logins)
    for login in logins:
        uid = ids.get(login)
        if not uid:
            emit("warning", msg=f"Twitch user not found: {login!r}")
            continue
        for hours, views, lang in tiers:
            while (need := per - count_for(batch, uid)) > 0:
                moments = qw.known_moments(db)
                seen = lambda c: qw.is_known(db, f"twitch_{c['id']}") or qw.same_moment(c, moments)
                fresh = qw.one_per_moment(client.fetch_top_clips(
                    broadcaster_id=uid, hours_back=hours, min_views=views, limit=need, language=lang,
                    pages=qw.TWITCH_PAGES, skip=seen), moments)[:need]
                if not fresh:
                    break
                emit("log", msg=f"[CALIBRATE] {login}: {per - need}/{per}, trying {len(fresh)} more "
                                f"({str(hours) + 'h' if hours else 'all time'}, >= {views} views, lang {lang or 'any'})")
                qw.enqueue_items(db, qw.twitch_items(fresh))
                run()
                if qw.counts(db)["pending"]:  # drain stopped on a download block
                    raise RuntimeError("Downloads are blocked: re-run with the same --batch later to resume")
        have = count_for(batch, uid)
        emit("log" if have >= per else "warning", msg=f"[CALIBRATE] {login}: {have}/{per} clips")


def thumb(video, jpg):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", str(video), "-frames:v", "1", "-vf", "scale=120:-2",
                    str(jpg)], capture_output=True)  # no thumbnail is fine: the sidebar shows the title alone


def stem_of(data, clip):
    return f"{data.get('video_id')}_{clip['clip_id']}"


def edit_all(batch, edit, on_done=None):
    """Every kept clip -> batch/shorts/<stem>_short.mp4 + <stem>_edit.json (the full EDL and QC, or why not).
    Finished records are skipped on resume; ones that crashed ("error") are tried again. on_done(stem) runs once a
    clip's record and thumbnail are written."""
    shorts = batch / "shorts"
    shorts.mkdir(parents=True, exist_ok=True)
    finished = lambda rec: rec.exists() and "error" not in json.loads(rec.read_text(encoding="utf-8"))
    todo = [(m, d, c) for m, d, c in kept_clips(batch) if not finished(shorts / f"{stem_of(d, c)}_edit.json")]
    for k, (m, data, clip) in enumerate(todo, 1):
        stem, meta = stem_of(data, clip), data.get("twitch") or {}
        rec = shorts / f"{stem}_edit.json"
        emit("progress", phase="render", id=stem, streamer=str(meta.get("broadcaster_name") or "").lower(),
             title=meta.get("title") or data.get("title"), n=k, of=len(todo))
        try:
            r = edit(m, clip["clip_id"], shorts)
        except Exception as e:  # one bad clip must not stop the batch
            emit("error", msg=f"{stem}: {e}")
            r = {"ok": False, "error": str(e)}
        rec.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        short = shorts / f"{stem}_short.mp4"
        thumb(short if r.get("ok") else m.parent / clip["file"], shorts / f"{stem}.jpg")
        if on_done:
            on_done(stem)


def default_edit(batch, settings):
    """editor/pipeline.edit_clip with the panel's settings (like the live --auto-edit), writing into the batch."""
    import argparse as _a
    import shutil
    sys.path.insert(0, str(ROOT / "editor"))
    import pipeline
    import renderer
    presets = batch / "facecam_presets.json"  # learned boxes stay in the batch; start from the live ones
    if not presets.exists() and pipeline.PRESETS.exists():
        shutil.copy(pipeline.PRESETS, presets)
    box = str(settings.get("facecam_box", "auto")).strip()
    try:
        box = pipeline.parse_box(box) if box not in ("", "auto") else None
    except (ValueError, _a.ArgumentTypeError):
        box = None
    enc = renderer.detect_encoder(None)
    emit("log", msg=f"Encoder: {enc[0]}")
    return lambda m, cid, out: pipeline.edit_clip(
        m, cid, out, settings.get("ollama_model") or pipeline.DEFAULT_MODEL, enc, box,
        whisper_model=settings.get("whisper_model") or "small.en", caption_tone=pipeline.THEMES.get(settings.get("caption_theme")),
        music_gain_db=settings.get("music_gain_db"), presets=presets)


# --- decision trace ------------------------------------------------------------------------------------------------

def decision_log(batch):
    """One entry per kept clip: clipper metrics + judge + director decisions (from the manifest and _edit.json)."""
    entries = []
    for m, data, clip in kept_clips(batch):
        meta, stem = data.get("twitch") or {}, stem_of(data, clip)
        rec_path = batch / "shorts" / f"{stem}_edit.json"
        rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}
        edl, dur = rec.get("edl") or {}, clip.get("duration") or 0
        kept = round(sum(c["end"] - c["start"] for c in edl.get("cuts") or []), 2)
        short, jpg = batch / "shorts" / f"{stem}_short.mp4", batch / "shorts" / f"{stem}.jpg"
        whole = clip["file"].endswith("_twitch.mp4")  # whole-clip path: visual never measured (older runs stored 0.0)
        visual = None if whole else clip.get("visual")
        final = scorer.final_score(clip.get("cue") or 0, clip.get("vocal") or 0, None, clip.get("llm")) \
            if whole and clip.get("score") is not None else clip.get("score")  # whole clips: signal == vocal peak
        status = ("edited" if rec.get("ok") else "skipped" if rec.get("skipped") else "error" if "error" in rec
                  else "qc_failed" if rec.get("qc") else "not_edited")
        entries.append({
            "id": stem, "streamer": str(meta.get("broadcaster_name") or "").lower(), "title": meta.get("title") or data.get("title"),
            "url": meta.get("url"), "transcript": clip.get("transcript"), "raw": (m.parent / clip["file"]).relative_to(batch).as_posix(),
            "short": short.relative_to(batch).as_posix() if short.exists() else None,
            "thumb": jpg.relative_to(batch).as_posix() if jpg.exists() else None,
            "clipper": {"views": meta.get("view_count"), "views_per_min": round(twitch_api.velocity(meta, m.stat().st_mtime), 1),
                        "duration": dur, "vocal_burst": clip.get("vocal"), "visual_action": visual,
                        "cue": clip.get("cue"), "pillars": clip.get("pillars"),
                        "judge_10": round(clip["llm"] * 10, 1) if clip.get("llm") is not None else None,
                        "final_score": final, "judge_reject": bool(clip.get("judge_reject")),
                        "judge_reason": clip.get("judge_reason"), "reason": clip.get("reason")},
            "editor": {"status": status, "note": rec.get("reason") or rec.get("error") or
                       (", ".join((rec.get("qc") or {}).get("problems") or []) if status == "qc_failed" else None),
                       "scenario": edl.get("scenario"), "directed_by": edl.get("directed_by"), "scenario_fit": edl.get("scenario_fit"),
                       "layout": edl.get("layout"),
                       "cuts": edl.get("cuts"), "kept_sec": kept if edl else None,
                       "trimmed_sec": round(dur - kept, 2) if edl else None, "hook_teaser": edl.get("hook_teaser"),
                       "speed": edl.get("speed"), "mutes": edl.get("mutes"), "zooms": edl.get("zooms"),
                       "filters": edl.get("filters"), "takeovers": edl.get("takeovers"), "sfx": edl.get("sfx"),
                       "font": edl.get("font"), "caption_tone": edl.get("caption_tone") or "per-word tones",
                       "title": edl.get("social_title"), "qc": {k: (rec.get("qc") or {}).get(k) for k in ("duration", "mean_db")}}})
    return entries


def describe(e):
    """Plain-English lines for one decision_log entry (DECISION_REPORT.md and the tuning report share them)."""
    c, d = e["clipper"], e["editor"]
    p = c.get("pillars") or {}
    lines = [f"- **Clipper:** {c['views']} views ({c['views_per_min']}/min), {c['duration']:.1f} s, vocal burst "
             f"{c['vocal_burst'] or 0:.2f}, visual action "
             + (f"{c['visual_action']:.2f}" if c["visual_action"] is not None else "n/a (whole clip, not scanned)") + f", final score {c['final_score']}."]
    if c["judge_10"] is None:
        lines.append("- **AI judge:** not run (Ollama offline).")
    else:
        lines.append(f"- **AI judge:** {c['judge_10']}/10 (payoff {p.get('comedy_payoff')}, self-contained "
                     f"{p.get('self_contained')}, hook {p.get('hook_strength')}, quotable {p.get('quotability')})"
                     + (" — REJECTED, kept only because the batch has no gate" if c["judge_reject"] else "")
                     + (f": \"{c['judge_reason']}\"" if c["judge_reason"] else "") + ".")
    if d["status"] in ("edited", "qc_failed"):
        cuts = ", ".join("%.2f-%.2f" % (x["start"], x["end"]) for x in d["cuts"] or [])
        fx = ", ".join(sorted({f["type"] for f in d["filters"] or []})) or "none"
        zooms = ", ".join("%s %sx @%ss" % (z["style"], z["scale"], z["at"]) for z in d["zooms"] or []) or "none"
        speed = ", ".join("%sx" % x["factor"] for x in d["speed"] or []) or "normal"
        sfx = ", ".join("%s @%ss" % (x["name"], x["at"]) for x in d["sfx"] or []) or "none"
        hook = d["hook_teaser"] or {}
        fit = d.get("scenario_fit")  # Ollama's advisory 0-10 per scenario (the scenario itself is rule-based)
        lines += [f"- **Director:** {d['scenario']} (picked by " + ("rules; Ollama fit " + ", ".join(f"{k} {v}" for k, v in fit.items())
                  if fit else str(d["directed_by"])) + f"), {d['layout']} layout"
                  + (", QC FAILED: " + d["note"] if d["status"] == "qc_failed" else "") + ".",
                  f"- **Cuts:** kept {d['kept_sec']} s of {c['duration']:.1f} s ({d['trimmed_sec']} s of dead air / pauses "
                  f"trimmed): {cuts}." + (f" Hook teaser {hook['start']}-{hook['end']} s." if hook.get("enabled") else ""),
                  f"- **Effects:** {len(d['zooms'] or [])} zoom(s) ({zooms}); filters {fx}; "
                  f"{len(d['takeovers'] or [])} facecam takeover(s); speed {speed}.",
                  f"- **SFX:** {sfx}. **Captions:** {d['font']}, {d['caption_tone']}. **Title:** {d['title']}"]
    else:
        lines.append(f"- **Director:** no short ({d['status']}" + (f": {d['note']}" if d["note"] else "") + ").")
    return lines


def write_atomic(path, text):
    """Temp file + replace: the studio and a running harvest may both write the log; a reader never sees half."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_decision_log(batch):
    """decision_log.json + DECISION_REPORT.md from whatever is on disk, so a partial / interrupted run works too."""
    entries = decision_log(batch)
    write_atomic(batch / "decision_log.json", json.dumps(entries, indent=2, ensure_ascii=False))
    md = [f"# Decision report: {batch.name}", "", f"{len(entries)} clips, "
          f"{sum(e['editor']['status'] == 'edited' for e in entries)} edited into shorts.", ""]
    for who in sorted({e["streamer"] for e in entries}):
        mine = [e for e in entries if e["streamer"] == who]
        scen = {}
        for e in mine:
            scen[e["editor"]["scenario"] or "none"] = scen.get(e["editor"]["scenario"] or "none", 0) + 1
        md.append(f"- **{who}**: {len(mine)} clips; scenarios " + ", ".join(f"{k} {v}" for k, v in sorted(scen.items())))
    for e in entries:
        md += ["", f"## {e['streamer']}: {e['title']} (`{e['id']}`)", ""] + describe(e)
    write_atomic(batch / "DECISION_REPORT.md", "\n".join(md) + "\n")
    return entries


def load_settings():
    """The Godot panel's workspace/settings.json ({} when missing or broken)."""
    try:
        return json.loads((ROOT / "workspace" / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def clipper_args(settings):
    """No banger gate (every clip is kept for review) + the panel's Whisper / Ollama models."""
    return ["--no-gate"] + [x for k in ("whisper_model", "ollama_model") if settings.get(k)
                            for x in (f"--{k.replace('_', '-')}", str(settings[k]))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--streamers", default=STREAMERS, help=f"Twitch logins, comma separated (default {STREAMERS})")
    ap.add_argument("--per-streamer", type=int, default=20, help="clips per streamer (default 20)")
    ap.add_argument("--hours", type=float, default=168, help="first time window in hours (default 168; widens when short)")
    ap.add_argument("--min-views", type=int, default=500, help="first minimum views (default 500; drops when short)")
    ap.add_argument("--batch", default=time.strftime("%Y-%m-%d"), help="runs/<batch>/ (default today's date; same name resumes)")
    a = ap.parse_args()
    batch = RUNS / a.batch
    batch.mkdir(parents=True, exist_ok=True)
    logins = twitch_api.parse_logins(a.streamers)
    # the studio's Resume reads this back
    write_atomic(batch / "run.json", json.dumps({"streamers": logins, "per_streamer": a.per_streamer, "hours": a.hours,
                                                  "min_views": a.min_views}, indent=2))
    settings = load_settings()
    emit("progress", phase="harvest")
    harvest(batch, logins, a.per_streamer, twitch_api.Twitch(), clipper_args(settings), tiers=[(a.hours, a.min_views, "en")] + TIERS)
    emit("progress", phase="render")
    edit_all(batch, default_edit(batch, settings))
    entries = write_decision_log(batch)
    emit("log", msg=f"[CALIBRATE] {len(entries)} clips in {batch}. Review: python calibration_studio/review_server.py --batch {a.batch}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)

"""Clip -> finished 9:16 short: word-level Whisper -> director (edl.json) -> renderer -> QC -> auto-repair -> ready_to_post/.

  python editor/pipeline.py --manifest output/<id>/clips_manifest.json            # every clip in the manifest
  python editor/pipeline.py --manifest ... --clip 2 --facecam-box 0,0.65,0.3,0.35  # one clip, webcam given

stdout is JSON Lines like the backend: {"event": "stage"|"log"|"short"|"error", ...}.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "backend"))  # shared helpers: audio decode, CUDA DLL setup, Ollama ping

import clipper  # noqa: E402
import scorer  # noqa: E402

import director  # noqa: E402
import framing  # noqa: E402
import qc  # noqa: E402
import renderer  # noqa: E402

SR = 16000
READY = ROOT / "ready_to_post"
SETTINGS = ROOT / "workspace" / "settings.json"  # written by the Godot SETTINGS panel
DEFAULT_MODEL = scorer.DEFAULT_MODEL
_WHISPER = {}


def emit(event, **data):
    print(json.dumps({"event": event, **data}), flush=True)


def transcribe_words(audio, model_name=scorer.DEFAULT_WHISPER, device="auto"):
    """[{word, start, end}] for the whole clip (faster-whisper word_timestamps). CUDA first, CPU int8 fallback."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    clipper.add_cuda_dll_dirs()
    from faster_whisper import WhisperModel

    def run(dev):
        key = (model_name, dev)
        if key not in _WHISPER:
            _WHISPER[key] = WhisperModel(model_name, device=dev, compute_type="int8" if dev == "cpu" or scorer.LOW_SPEC else scorer.WHISPER_GPU_COMPUTE,
                                         cpu_threads=4 if scorer.LOW_SPEC else 0)
        segs, _ = _WHISPER[key].transcribe(audio, language="en", word_timestamps=True, beam_size=1,
                                             initial_prompt=scorer.GAMING_PROMPT, vad_filter=True,
                                             vad_parameters={"min_silence_duration_ms": 250})  # no word stretched over silence
        words = [{"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)}
                 for s in segs for w in (s.words or []) if w.word.strip()]  # segs is lazy: decode before freeing
        return words

    if device != "cpu" and not scorer.LOW_SPEC:  # LOW_SPEC: the GPU is Ollama's (whole clips skip this: words cached)
        try:
            return run("cuda")
        except Exception as e:  # no GPU / missing CUDA DLLs: CPU still works
            emit("log", msg=f"Whisper CUDA failed ({str(e)[:100]}), using CPU int8")
    return run("cpu")


def snap_words_to_audio(words, env, hop=director.HOP, max_shift=0.25, guard=0.06, min_len=0.12):
    """Acoustic snap. Start: Whisper tends to start a word early (in the silence before it); move it forward to the
    first 50 ms frame whose loudness clears the clip's noise floor, by at most max_shift and never closer than `guard`
    to the word's end, so a caption pops on the frame the syllable is heard. End: Whisper also stretches a word over
    the silence after it (live: CaseOh's "my" 38.96-40.86 s, ~0.3 s of it audible), which kept dead air in the cut and
    left its caption up through it; pull the end back to the last audible frame (>= min_len after the start).
    Starts only ever move later, ends only earlier."""
    if not env.size:
        return words
    floor = float(np.percentile(env, 20))
    threshold = max(2 * floor, floor + 0.05)
    loud = lambda t: director.env_at(env, t + 1e-6, hop) >= threshold  # +1e-6: 0.35 + 3 * 0.05 must land in frame 10
    out = []
    for w in words:
        limit = min(w["start"] + max_shift, w["end"] - guard)
        start, k = w["start"], 0
        while w["start"] + k * hop <= limit + 1e-9:
            t = w["start"] + k * hop
            if loud(t):
                start = t
                break
            k += 1
        start = max(w["start"], start)
        end = w["end"]
        while end - hop >= start + min_len and not loud(end - hop):
            end -= hop
        out.append(dict(w, start=round(start, 3), end=round(end, 3)))
    return out


def streamer_key(manifest):
    """Who made the clip: the Twitch login, else the YouTube channel handle / id, lowercased (None if unknown)."""
    name = (manifest.get("twitch") or {}).get("broadcaster_name") or manifest.get("channel")
    return str(name).strip().lstrip("@").lower() or None if name else None


def subtract(cuts, a, b, min_piece=0.3):
    """Remove source span [a, b] from the cuts (splitting a cut if needed)."""
    out = []
    for c in cuts:
        if c["end"] <= a or c["start"] >= b:
            out.append(c)
            continue
        if a - 0.05 - c["start"] >= min_piece:
            out.append({"start": c["start"], "end": round(a - 0.05, 2)})
        if c["end"] - (b + 0.05) >= min_piece:
            out.append({"start": round(b + 0.05, 2), "end": c["end"]})
    return out


REPAIR_LEN = 9.0  # a too-short edit grows back to this (QC minimum 7.0, director 8.5)


def repair(edl, report, max_dur=qc.LIMITS["max_dur"], src_duration=None):
    """Adjust the EDL for what QC found -> new EDL, or None when nothing can be fixed by editing.
    black: strip the offending span (mapped back to source time; teaser/freeze dropped if that's where it is), unless
    that leaves the edit under director.MIN_LEN: then the footage itself is dark, so it stays ("keep_dark": QC
    skips the black check) instead of shredding the edit render after render.
    too_short: grow the cuts toward the source's bounds (pauses, earlier setup, later end) to REPAIR_LEN.
    too_long: drop the teaser, then trim from the start so the payoff at the end survives."""
    new = json.loads(json.dumps(edl))
    changed = False
    cut_len = lambda e: director.total([(c["start"], c["end"]) for c in e["cuts"]])
    if "black" in report["problems"]:
        segs, _ = renderer.timeline(edl)
        main_len = renderer.main_length(segs)
        for a, b in report["black"]:
            if a >= main_len - 0.05 and new.get("freeze_end"):
                new["freeze_end"], changed = 0.0, True
                continue
            ia, sa = renderer.to_src(a, segs)
            ib, sb = renderer.to_src(min(b, main_len), segs)
            if ia is None:
                continue
            if segs[ia][3]:  # inside the hook teaser
                new["hook_teaser"]["enabled"], changed = False, True
                continue
            # a span crossing a jump cut: strip both sides
            spans = [(sa, segs[ia][1]), (segs[ib][0], sb)] if ib is not None and ib != ia else [(sa, sb if ib is not None else segs[ia][1])]
            for x, y in spans:
                new["cuts"] = subtract(new["cuts"], x, y)
            changed = True
        if cut_len(new) < director.MIN_LEN:
            new["cuts"], new["keep_dark"], changed = json.loads(json.dumps(edl["cuts"])), True, True
    if src_duration and ("too_short" in report["problems"] or cut_len(new) < director.MIN_LEN):
        grown = director.fit_length([(c["start"], c["end"]) for c in new["cuts"]], src_duration, REPAIR_LEN)
        if grown != new["cuts"]:
            new["cuts"], changed = grown, True
    if "too_long" in report["problems"] or renderer.timeline(new)[1] > max_dur - 0.5:
        excess = renderer.timeline(new)[1] - (max_dur - 1.0)
        hook = new.get("hook_teaser") or {}
        if excess > 0 and hook.get("enabled"):
            hook["enabled"] = False
            excess -= hook["end"] - hook["start"]
            changed = True
        while excess > 0 and new["cuts"]:
            first = new["cuts"][0]
            length = first["end"] - first["start"]
            if length <= excess + 0.3 and len(new["cuts"]) > 1:
                new["cuts"].pop(0)
                excess -= length
            else:
                first["start"] = round(first["start"] + excess, 2)
                excess = 0
            changed = True
    return new if changed and new["cuts"] else None


VRAM_RENDER_MIB = 400  # an NVENC 1080x1920 encode took ~130 MiB next to the warm 3B model (3.82 of 4 GiB used)


def make_room_for_render(free_mib=None, unload=None):
    """LOW_SPEC keeps Ollama warm in VRAM; if that leaves less than VRAM_RENDER_MIB free for NVENC (other apps vary),
    unload it first: the next clip's judge call reloads it. -> True when it unloaded."""
    if not scorer.LOW_SPEC:
        return False
    free = (free_mib or scorer.vram_free_mib)()
    if free is None or free >= VRAM_RENDER_MIB:
        return False
    names = (unload or scorer.ollama_unload)()
    emit("log", msg=f"Only {free} MiB VRAM free before the render: unloaded {', '.join(names) or 'nothing'}")
    return True


LOUDNESS_PASSES = 2
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"


def boost_loudness(path):
    """One more -14 LUFS loudnorm pass on a finished short, audio only (the video is copied, so it takes seconds).
    The render's own loudnorm can't lift a very quiet mic in one go: live 912079449_1 came from a -32.8 LUFS clip
    with sharp peaks, the -1.5 dBTP limit stopped it at -16.5 LUFS / -20.3 dB mean (QC floor -20). One pass more:
    -15.1 LUFS / -18.8 dB, peaks still -1.3 dB."""
    path = Path(path)
    tmp = path.with_name(path.stem + ".loud" + path.suffix)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-map", "0", "-c:v", "copy", "-af", f"{LOUDNORM},aresample=48000",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(tmp)], check=True, capture_output=True)  # loudnorm pads ~0.1 s
    os.replace(tmp, path)


def render_until_pass(src, edl, words, work, limits=None, encoder=None, facecam=None, max_retries=2, label="short"):
    """Render -> QC -> repair EDL -> re-render, up to max_retries repairs. -> (final edl, last QC report);
    the render is work/short.mp4. A too_quiet render first gets up to LOUDNESS_PASSES audio-only loudnorm passes."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries + 1):
        (work / "edl.json").write_text(json.dumps(edl, indent=2), encoding="utf-8")
        emit("stage", stage="render", msg=f"Rendering {label}" + (f" (repair {attempt})" if attempt else ""))
        renderer.render(src, edl, words, work / "short.mp4", work, encoder=encoder, facecam=facecam)
        # keep_dark (set by repair): the source itself is dark there, so blackdetect would only re-flag the content
        check = lambda: qc.check(work / "short.mp4", **{**(limits or {}), **({"black_max": 1e6} if edl.get("keep_dark") else {})})
        report = check()
        for n in range(LOUDNESS_PASSES):
            if "too_quiet" not in report["problems"]:
                break
            emit("log", msg=f"QC too quiet ({report['mean_db']} dB): loudness repair {n + 1}/{LOUDNESS_PASSES} ({LOUDNORM})")
            boost_loudness(work / "short.mp4")
            report = check()
        emit("log", msg=f"QC {'PASS' if report['ok'] else 'FAIL ' + ','.join(report['problems'])}: "
                        f"{report['duration']}s, {report['mean_db']} dB, black {report['black']}")
        if report["ok"] or attempt == max_retries:
            return edl, report
        fixed = repair(edl, report, (limits or {}).get("max_dur", qc.LIMITS["max_dur"]), renderer.probe(src)["duration"])
        if fixed is None:
            return edl, report
        edl = fixed


def edit_clip(manifest_path, clip_id, out_dir=READY, model=DEFAULT_MODEL, encoder=None, facecam=None, limits=None,
              whisper_model=scorer.DEFAULT_WHISPER, device="auto", max_retries=2, transcribe=None, caption_tone=None, music_gain_db=None,
              analyze=framing.analyze):
    """One clip -> ready_to_post/<video>_<clip>_short.mp4 + _post.json. -> result dict ("ok", "path", "qc", ...)."""
    manifest_path, out_dir = Path(manifest_path).resolve(), Path(out_dir)  # absolute: _post.json must work from any cwd
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clip = next(c for c in manifest["clips"] if int(c["clip_id"]) == int(clip_id))
    src = manifest_path.parent / clip["file"]
    stem = f"{manifest.get('video_id', manifest_path.parent.name)}_{clip['clip_id']}"
    emit("log", msg=f"Editing {stem}")
    audio = clipper.decode_audio(src)
    duration = len(audio) / SR

    if not clip.get("words"):  # word timings from the clip FILE (its start can sit a keyframe before start_sec)
        emit("stage", stage="edit", msg=f"Whisper word timings ({whisper_model})")
        clip["words"] = (transcribe or transcribe_words)(audio, whisper_model, device)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    context = f"{streamer_key(manifest) or ''} {manifest.get('title') or ''}"  # scopes fixes like queso -> CaseOh
    words = scorer.fix_gaming_words(clip["words"], context)  # also fixes words cached by older runs
    words = snap_words_to_audio(words, director.energy(audio))  # the cache keeps Whisper's raw timings

    # faces (YuNet) every clip, no saved per-streamer boxes: a hand-set box (--facecam-box / SETTINGS) still wins
    fr = analyze(src, words, facecam)
    facecam = fr.get("cam_box")
    emit("log", msg=f"Framing: {fr['layout']} ({fr['why']})" + ("; the source has burned-in captions, ours off" if fr.get("source_captions") else ""))
    online = bool(model) and scorer.ollama_online()
    model = scorer.resolve_model(model) if online else model
    meta = dict(manifest.get("twitch") or {"title": manifest.get("title")},  # + the judge's read, for the director
                pillars=clip.get("pillars"), judge_reason=clip.get("judge_reason"))
    music = renderer.MUSIC_DIR.exists() and any(p.suffix.lower() in (".mp3", ".wav") for p in renderer.MUSIC_DIR.iterdir())
    emit("stage", stage="edit", msg="Directing" + (f" with {model}" if online else " (heuristic, Ollama offline)"))
    edl = director.direct(words, audio, duration, model=model if online else None, meta=meta,
                          framing=fr, music=music)
    if caption_tone:  # SETTINGS caption theme: one colour instead of per-word tones
        edl["caption_tone"] = caption_tone
    if music_gain_db is not None:
        edl["bg_music"]["gain_db"] = float(music_gain_db)
    del audio
    if edl.get("skip_render"):  # nothing coherent left to post: don't spend a render on it
        emit("log", msg=f"Skipped {stem}: {edl['skip_reason']}")
        return {"ok": False, "skipped": True, "reason": edl["skip_reason"], "edl": edl}
    emit("log", msg=f"{stem}: {edl['archetype']} ({edl['directed_by']}), {len(edl['cuts'])} cut(s), "
                    f"{renderer.timeline(edl)[1]:.1f}s, title: {edl['title']}")

    work = out_dir / ".work" / stem
    make_room_for_render()
    edl, report = render_until_pass(src, edl, words, work, limits, encoder, facecam, max_retries, stem)
    if not report["ok"]:
        emit("error", msg=f"{stem} failed QC ({', '.join(report['problems'])}); render kept in {work}")
        return {"ok": False, "qc": report, "edl": edl, "work": str(work)}

    out_dir.mkdir(parents=True, exist_ok=True)
    short = out_dir / f"{stem}_short.mp4"
    shutil.move(str(work / "short.mp4"), short)
    post = {"title": edl["title"], "hashtags": edl["hashtags"], "social_title": edl["social_title"],
            "transcript": " ".join(w["word"] for w in words), "duration": report["duration"],
            "archetype": edl["archetype"], "scenario": edl["scenario"], "directed_by": edl["directed_by"],
            "font": edl["font"], "layout": edl["layout"], "facecam_box": facecam if edl["layout"] == "split_stack" else None,
            "cuts": edl["cuts"], "hook_teaser": edl["hook_teaser"], "zooms": len(edl["zooms"]),
            "takeovers": len(edl["takeovers"]), "effects": sorted({f["type"] for f in edl["filters"]}),
            "sfx": [s["name"] for s in edl["sfx"]],
            "credit": {k: meta.get(k) for k in ("broadcaster_name", "creator_name", "view_count", "url") if meta.get(k)},
            "source": {"manifest": str(manifest_path), "clip_id": clip["clip_id"], "file": clip["file"]},
            "qc": {k: report[k] for k in ("duration", "mean_db", "bytes")}}
    post.update(post_card(clip, post))
    (out_dir / f"{stem}_post.json").write_text(json.dumps(post, indent=2, ensure_ascii=False), encoding="utf-8")
    if post.get("title_angles"):
        (out_dir / f"{stem}_post.txt").write_text(card_text(post), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    emit("short", path=str(short), post=str(out_dir / f"{stem}_post.json"), title=post["social_title"],
         archetype=edl["archetype"], duration=report["duration"], video_id=manifest.get("video_id"), clip_id=clip["clip_id"])
    return {"ok": True, "path": str(short), "qc": report, "edl": edl}


def post_card(clip, post):
    """The approved clip's 14B metadata (clipper: scorer.ollama_metadata) over the director's title: title angles,
    5 hashtags (they replace the director's 4) and the pinned-comment line. {} when the clip has none (Ollama was off)."""
    angles, tags = list(clip.get("title_angles") or []), list(clip.get("hashtags") or [])
    out = {k: v for k, v in (("title_angles", angles), ("pinned_comment", clip.get("pinned_comment"))) if v}
    if tags:
        out["hashtags"] = tags
    if angles:
        out["title"] = angles[0]
    if angles or tags:
        out["social_title"] = f"{out.get('title', post['title'])} {' '.join(tags or post['hashtags'])}".strip()
    return out


def card_text(post):
    """The plain-text post card next to the short (copy-paste ready)."""
    lines = ["TITLE ANGLES:"] + [f"  {i}. {t}  ({len(t)} chars)" for i, t in enumerate(post.get("title_angles", []), 1)]
    lines += ["", "HASHTAGS: " + " ".join(post.get("hashtags", []))]
    if post.get("pinned_comment"):
        lines += ["", "PINNED COMMENT: " + post["pinned_comment"]]
    return "\n".join(lines) + "\n"


def parse_box(text):
    box = tuple(float(v) for v in text.split(","))
    if len(box) != 4 or not all(0 <= v <= 1 for v in box) or box[0] + box[2] > 1 or box[1] + box[3] > 1:
        raise argparse.ArgumentTypeError("x,y,w,h as fractions 0..1 inside the frame, e.g. 0,0.65,0.3,0.35")
    return box


# SETTINGS caption theme -> renderer tone (its colour)
THEMES = {"Yellow": "punch", "Neon Green": "hype", "Red": "rage", "Cyan": "roast"}


def load_settings(path=SETTINGS):
    """Editor defaults from the panel's workspace/settings.json ({} if missing/broken); CLI flags still win."""
    try:
        s = json.loads(Path(path).read_text(encoding="utf-8"))
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--clip", type=int, action="append", help="clip_id to edit (repeatable; default: all)")
    ap.add_argument("--out", type=Path, default=READY)
    ap.add_argument("--facecam-box", type=parse_box, help="webcam region x,y,w,h (0..1); default: auto-detect -> split_stack layout")
    ap.add_argument("--encoder", choices=["h264_nvenc", "h264_vaapi", "libx264"], help="default: auto-detect")
    ap.add_argument("--ollama-model", default=DEFAULT_MODEL)
    ap.add_argument("--no-ollama", action="store_true")
    ap.add_argument("--whisper-model", default=scorer.DEFAULT_WHISPER)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--caption-theme", choices=sorted(THEMES), help="one caption colour (default: per-word tones by scenario)")
    ap.add_argument("--music-gain", type=float, help="background music gain in dB (default -22)")
    ap.add_argument("--speaker-captions", action="store_true", help=argparse.SUPPRESS)  # removed feature: accepted, ignored
    st = load_settings()
    box = str(st.get("facecam_box", "auto")).strip()
    try:
        box = parse_box(box) if box not in ("", "auto") else None
    except (ValueError, argparse.ArgumentTypeError):
        emit("log", msg=f"settings.json facecam_box {box!r} is not x,y,w,h in 0..1: auto-detecting instead")
        box = None
    ap.set_defaults(**{k: v for k, v in {
        "facecam_box": box,
        "ollama_model": st.get("ollama_model") or None, "whisper_model": st.get("whisper_model") or None,
        "caption_theme": st.get("caption_theme") if st.get("caption_theme") in THEMES else None,
        "music_gain": st.get("music_gain_db")}.items() if v is not None})
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    ids = a.clip or [c["clip_id"] for c in manifest["clips"]]
    enc = renderer.detect_encoder(a.encoder)
    emit("log", msg=f"Encoder: {enc[0]}")
    ok = skipped = 0
    for k, cid in enumerate(ids, 1):
        emit("stage", stage="edit", msg=f"Editing short {k}/{len(ids)}")  # the panel's progress counter
        try:
            r = edit_clip(a.manifest, cid, a.out, None if a.no_ollama else a.ollama_model, enc, a.facecam_box,
                          whisper_model=a.whisper_model, device=a.device, caption_tone=THEMES.get(a.caption_theme),
                          music_gain_db=a.music_gain)
            ok += r["ok"]
            skipped += bool(r.get("skipped"))
        except Exception as e:  # one bad clip must not stop the rest
            emit("error", msg=f"clip {cid}: {e}")
    emit("log", msg=f"Shorts ready: {ok}/{len(ids)} -> {a.out}" + (f" ({skipped} skipped: nothing coherent to post)" if skipped else ""))
    return 0 if ok + skipped == len(ids) else 1  # a deliberate skip is not a failure


if __name__ == "__main__":
    sys.exit(main())

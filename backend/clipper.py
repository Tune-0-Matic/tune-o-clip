"""Tune-O-Clip backend: YouTube URL -> top-N funny clips, cut with stream copy (no re-encode).

Every stdout line is one JSON object: {"event": "stage"|"log"|"clip"|"done"|"error", ...}.
Human/tool noise goes to stderr, so cron or the Godot panel can parse stdout as-is.
"""
import argparse
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import numpy as np

import scorer

# No --no-warnings: stderr is captured and only shown on failure, where warnings (e.g. cookie decryption) matter.
# Polite pacing: a pause between extraction requests and patient retries keep request bursts (and blocks) down.
YTDLP = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--sleep-requests", "2", "--retries", "5", "--fragment-retries", "5"]
# YouTube's "n challenge" needs a JS runtime: prefer the venv's pip-installed Deno (Scripts/ or bin/, not on PATH
# when the venv isn't activated), else Node on PATH. Without one, downloads fail with "Only images are available".
_DENO = shutil.which("deno", path=str(Path(sys.executable).parent)) or shutil.which("deno")
if _DENO:
    YTDLP += ["--js-runtimes", f"deno:{_DENO}"]
elif shutil.which("node"):
    YTDLP += ["--js-runtimes", "node"]
# H.264 first: AV1 has no hardware decode on the GTX 1050 Ti or the Ryzen V1202B, so clips would stutter.
VIDEO_FORMAT = "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]"
# Visual scan input: tiny H.264 stream (AV1 144p decodes ~9x slower).
ANALYSIS_FORMAT = "bv*[height<=144][vcodec^=avc1]/bv*[height<=240][vcodec^=avc1]"
SR = 16000  # whisper's sample rate; also our RMS analysis rate
OUTPUT = Path(__file__).resolve().parent.parent / "output"  # the queue's archive (duplicate-moment check)
DUP_WINDOW, DUP_SIMILARITY = 24 * 3600, 0.6  # same streamer, last 24 h, >60% of the words in common


def emit(event, **data):
    print(json.dumps({"event": event, **data}), flush=True)


def run(cmd, check=True):
    # PYTHONIOENCODING: yt-dlp otherwise writes the Windows console code page into the pipe ("you�re").
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if check and p.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} {' '.join(cmd[1:3])}... failed:\n{p.stderr.strip()[-800:]}")
    return p


def preflight():
    missing = []
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg not on PATH. Windows: winget install Gyan.FFmpeg | Linux Mint: sudo apt install ffmpeg")
    if not importlib.util.find_spec("yt_dlp"):
        missing.append(f"yt-dlp missing for {sys.executable}. Run: pip install -r backend/requirements.txt "
                       "(or winget install yt-dlp.yt-dlp for the standalone exe)")
    return missing


def fetch_metadata(url, workspace):
    """Info JSON only (no video, no captions). Returns (info, job dir).
    Captions are never requested: YouTube's caption endpoint 429s easily and the block hits the whole IP;
    Whisper transcribes the candidate windows locally instead."""
    p = run(YTDLP + ["--skip-download", "--no-simulate", "--write-info-json", "--print", "%(id)s",
                     "-o", str(workspace / "%(id)s" / "source.%(ext)s"), url], check=False)
    vid = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    info_path = workspace / vid / "source.info.json"
    if not vid or not info_path.exists():
        if "confirm you" in p.stderr and "not a bot" in p.stderr and ("--cookies" in YTDLP or "--cookies-from-browser" in YTDLP):
            used = YTDLP[YTDLP.index("--cookies-from-browser") + 1] if "--cookies-from-browser" in YTDLP else "cookies.txt"
            notes = [l.strip() for l in p.stderr.splitlines() if "not a bot" not in l
                     and any(k in l.lower() for k in ("cookie", "decrypt", "keyring", "could not", "permission"))]
            raise RuntimeError(
                f"Still blocked WITH {used} cookies, so YouTube didn't accept that login. yt-dlp said:\n  "
                + ("\n  ".join(notes[-6:]) if notes else "(no cookie warnings: the cookies loaded, but this browser "
                   "profile is probably not logged into YouTube; log in there, close it, retry)")
                + "\nIf it mentions decrypt/copy: that browser's cookies can't be read; export a cookies.txt "
                  "(browser extension) and use --cookies FILE, or wait a few hours for the block to lift.")
        if "confirm you" in p.stderr and "not a bot" in p.stderr:
            raise RuntimeError(
                "YouTube's bot check blocked this IP (too many automated downloads). Either wait (usually 1+ hours), "
                "or pass your browser's YouTube login: pick a browser under COOKIES in the panel / "
                "--cookies-from-browser edge|brave|firefox|chrome (close that browser first; Chromium browsers lock "
                "their cookie file while open), or export a cookies.txt from youtube.com (e.g. the 'Get cookies.txt "
                "LOCALLY' extension) and save it as cookies.txt in the tune-o-clip folder: it's picked up automatically.")
        raise RuntimeError("Metadata fetch failed:\n" + p.stderr.strip()[-800:])
    return json.loads(info_path.read_text(encoding="utf-8")), workspace / vid


def local_info(path, workspace):
    """A local video file -> (info, job dir) like fetch_metadata, with the file copied in as the source (the user's
    file is never moved). It is processed like a Twitch clip: one whole, already-chosen highlight."""
    src = Path(path).resolve()
    vid = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in src.stem)[:60] or "upload"
    job = workspace / vid
    job.mkdir(parents=True, exist_ok=True)
    for old in job.glob("source.*"):
        old.unlink()
    shutil.copyfile(src, job / ("source" + src.suffix.lower()))
    return {"id": vid, "title": src.stem, "webpage_url": str(src), "local": True}, job


def decode_audio(src, start=0.0, dur=None):
    """Mono float32 16 kHz audio from a media file, optionally just [start, start+dur)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(src)] + (["-t", str(dur)] if dur else [])
    return np.frombuffer(run_bytes(cmd + ["-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]), dtype=np.float32)


def add_cuda_dll_dirs():
    """pip's nvidia-cublas/cudnn wheels put their DLLs in site-packages/nvidia/*/bin, which Windows doesn't search.
    No-op on Linux (no NVIDIA GPU on the Mint box; Whisper runs on CPU there)."""
    if os.name != "nt":
        return
    import site
    for sp in site.getsitepackages():
        for d in Path(sp).glob("nvidia/*/bin"):
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]


def run_bytes(cmd):
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{p.stderr.decode(errors='replace')[-800:]}")
    return p.stdout


def whisper_chunks(windows, total):
    """[(start, end)] windows -> faster-whisper clip_timestamps: <= 30 s each, clamped to [0, total] seconds of
    decoded audio. Metadata durations can run a few ms past the real audio; a window beyond the end makes
    faster-whisper's "audio minus windows" log value negative -> AssertionError "non-negative timestamp expected"."""
    return [{"start": float(t), "end": float(min(e, t + 30, total))}
            for s, e in windows for t in np.arange(max(0.0, s), min(e, total), 30)]


def make_transcriber(model_name, device, context=""):
    """Returns transcribe(full float32 audio, [(start, end)] windows) -> per window [(start, end, text)] lines,
    in absolute seconds. All windows go through ONE batched pass (each split into <= 30 s chunks, Whisper's
    limit). Tries CUDA first and drops to CPU int8 on any CUDA failure. context (streamer, title) scopes the
    vocabulary fixes (scorer.gaming_fixes). One call only: the model is freed after it, before the Ollama judge."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows without Developer Mode; harmless
    add_cuda_dll_dirs()
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    state = {}

    def load(dev):
        model = WhisperModel(model_name, device=dev, compute_type="int8" if dev == "cpu" or scorer.LOW_SPEC else "auto",
                             cpu_threads=4 if scorer.LOW_SPEC else 0)  # 4 of the i7-860's 8: the desktop stays usable
        state["dev"], state["model"], state["pipe"] = dev, model, BatchedInferencePipeline(model)

    try:
        load("cpu" if device == "cpu" or scorer.LOW_SPEC else "cuda")  # LOW_SPEC: the GPU is Ollama's
    except Exception as e:  # no GPU / no CUDA libs
        emit("log", msg=f"Whisper on CUDA unavailable ({str(e)[:120]}), using CPU int8")
        load("cpu")

    def transcribe(audio, windows, words_out=None):
        """words_out: a list that receives the raw word timings too (whole clips: the editor reuses them)."""
        chunks = whisper_chunks(windows, len(audio) / SR)
        try:
            if state["dev"] == "cpu" and len(windows) == 1 and windows[0][0] <= 0 and windows[0][1] >= len(audio) / SR - 0.5:
                # one whole clip on the CPU: the plain decoder + VAD beats batching (60 s clip with word timings on
                # the i7-860, 4 threads: 48 s vs 61 s batched)
                segments, _ = state["model"].transcribe(audio, language="en", beam_size=1, vad_filter=True,
                                                        initial_prompt=scorer.whisper_prompt(),
                                                        word_timestamps=words_out is not None)
            else:
                segments, _ = state["pipe"].transcribe(audio, language="en", clip_timestamps=chunks, batch_size=8,
                                                          initial_prompt=scorer.whisper_prompt(),
                                                          word_timestamps=words_out is not None)
            segments = list(segments)
            lines = [(s.start, s.end, scorer.fix_gaming_text(s.text.strip(), context)) for s in segments]
            if words_out is not None:
                words_out[:] = [{"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)}
                                for s in segments for w in (s.words or []) if w.word.strip()]
        except Exception as e:  # missing cuBLAS/cuDNN DLLs or VRAM exhaustion only show up at first inference
            if state["dev"] == "cpu":
                raise
            emit("log", msg=f"Whisper CUDA inference failed ({str(e)[:120]}), retrying on CPU int8")
            load("cpu")
            return transcribe(audio, windows, words_out)
        state.clear()  # free Whisper's VRAM now: the Ollama judge comes next (the closure cycle needs gc)
        gc.collect()
        # padded windows can overlap their neighbours' chunks: dedupe lines per window
        return [list(dict.fromkeys(l for l in lines if l[1] > s and l[0] < e)) for s, e in windows]

    emit("log", msg=f"Whisper {model_name} loaded on {state['dev']}")
    return transcribe


def start_analysis_download(info, job):
    """Tiny 144p H.264 video-only stream for the visual scan, fetched in parallel with the main download.
    Decoding the 1080p source at 2 fps takes ~2 min (every frame must be decoded); 144p decodes in ~2 s."""
    for old in job.glob("analysis.*"):
        old.unlink()
    return subprocess.Popen(YTDLP + ["-f", ANALYSIS_FORMAT, "-o", str(job / "analysis.%(ext)s"), info["webpage_url"]],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def visual_scan(proc, job):
    """Wait for the analysis download, pipe it through ffmpeg at 2 fps 160x90 gray -> per-second action curve."""
    if proc.wait() != 0:
        return None
    src = next((p for p in job.glob("analysis.*") if p.suffix != ".part"), None)
    if src is None:
        return None
    raw = run_bytes(["ffmpeg", "-v", "error", "-i", str(src), "-an", "-vf", "fps=2,scale=160:90,format=gray",
                     "-f", "rawvideo", "-"])
    src.unlink(missing_ok=True)
    return scorer.visual_curve(np.frombuffer(raw, dtype=np.uint8).reshape(-1, 90, 160))


def keyframes(src):
    """Video keyframe times from a packet scan (no decoding, ~1-2 s)."""
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
               "-of", "csv=p=0", str(src)]).stdout
    return sorted(float(l.split(",")[0]) for l in out.splitlines() if ",K" in l and l.split(",")[0] not in ("", "N/A"))


def cut(source, start, end, out):
    # -ss before -i with -c copy starts on the keyframe at/before `start` (exact when start is a keyframe).
    run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", str(source), "-t", str(round(end - start, 3)),
         "-c", "copy", "-avoid_negative_ts", "make_zero", str(out)])


def stitch(source, plan, out):
    """[setup] + [payoff] from the same source, joined with the concat demuxer (same encoder, so no re-encode)."""
    parts = [out.with_name("_setup.mp4"), out.with_name("_payoff.mp4")]
    for (s, e), p in zip(plan, parts):
        cut(source, s, e, p)
    lst = out.with_name("_concat.txt")
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    try:
        run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
    finally:
        for p in parts + [lst]:
            p.unlink(missing_ok=True)


def is_youtube(url):
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")


def cookie_args(a):
    """yt-dlp cookie flags for this run: --cookies-from-browser, --cookies, or the project's cookies.txt,
    and only when the URL is YouTube."""
    if not is_youtube(a.url):
        return []
    if a.cookies_from_browser:
        return ["--cookies-from-browser", a.cookies_from_browser]
    default_cookies = Path(__file__).resolve().parent.parent / "cookies.txt"
    if not a.cookies and default_cookies.exists():
        a.cookies = str(default_cookies)  # dropped into the project folder: used automatically
        emit("log", msg=f"Using YouTube cookies from {default_cookies.name}")
    return ["--cookies", a.cookies] if a.cookies else []


def auto_edit(manifest_path):
    """Single runs: clips -> shorts with editor/pipeline.py (same path the queue's --auto-edit uses)."""
    from queue_worker import run_editor  # stdlib-only module next to this one
    emit("stage", stage="edit", msg="Auto-editing clips into shorts")
    made = run_editor(manifest_path)
    emit("log", msg=f"Auto-edit: {made} short(s) in ready_to_post/")


def archive(job, a, name):
    """Move the clips + manifest from the temp job folder to <--output>/<name>/ (replacing an earlier run's clips),
    so every run (single, batch, Twitch) ends up in output/ and workspace/ only ever holds temp files.
    -> the archived manifest path. Without an --output (old callers) the job folder stays where it is."""
    if not getattr(a, "output", None):
        return job / "clips_manifest.json"
    dest = Path(a.output) / name
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("clip_*.mp4"):
        old.unlink()
    for f in list(job.glob("clip_*.mp4")) + [job / "clips_manifest.json"]:
        (dest / f.name).unlink(missing_ok=True)
        shutil.move(str(f), dest / f.name)
    return dest / "clips_manifest.json"


def recent_duplicate(streamer, text, roots, job, now=None, any_age=()):
    """(clip folder, similarity) of a clip by the same Twitch streamer, archived in the last DUP_WINDOW (any age under
    the `any_age` roots, e.g. every calibration run), whose transcript shares >= DUP_SIMILARITY of its words with
    `text`; None otherwise (also for short transcripts)."""
    if not streamer:
        return None
    now = time.time() if now is None else now
    for root, window in [(r, DUP_WINDOW) for r in roots] + [(r, None) for r in any_age]:
        for m in Path(root).glob("*/clips_manifest.json"):
            if m.parent.resolve() in {Path(j).resolve() for j in job} or (window and now - m.stat().st_mtime > window):
                continue
            try:
                data = json.loads(m.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if str((data.get("twitch") or {}).get("broadcaster_name") or "").lower() != streamer.lower():
                continue
            for c in data.get("clips") or []:
                sim = scorer.jaccard(text, c.get("transcript") or "")
                if sim >= DUP_SIMILARITY:
                    return m.parent.name, sim
    return None


def write_manifest(job, info, clips, extra=None):
    keys = ("clip_id", "file", "start_sec", "end_sec", "duration", "setup_start_sec", "setup_end_sec", "score", "reason", "judge_reason", "transcript",
            "llm", "pillars", "judge_reject", "vocal", "visual", "cue", "words")  # the full score breakdown (calibration_studio)
    manifest = {"video_id": info["id"], "title": info.get("title"), "url": info.get("webpage_url"), **(extra or {}),
                "clips": [{k: c.get(k) for k in keys} for c in clips]}
    path = job / "clips_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def twitch_clip(a, info, job):
    """Twitch clips are already viewer-cut 15-60 s highlights: no heatmap, windows or cutting. Download the clip,
    transcribe it with Whisper, score it (cues + vocal bursts + local Ollama judge) and keep it whole, with the
    streamer credit in the manifest."""
    meta = json.loads(a.twitch_meta) if a.twitch_meta else {
        "clip_id": info.get("id"), "title": info.get("title"), "broadcaster_name": info.get("uploader") or info.get("channel"),
        "creator_name": info.get("creator"), "view_count": info.get("view_count"), "url": info.get("webpage_url")}
    duration = float(info.get("duration") or meta.get("duration") or 0)
    emit("log", msg=f"Twitch clip: {info.get('title')} by {meta.get('broadcaster_name')} "
                    f"({duration:.0f}s, {meta.get('view_count')} views)")
    online = not a.no_ollama and scorer.ollama_online()
    if online:
        threading.Thread(target=scorer.ollama_warmup, args=(a.ollama_model,), daemon=True).start()
    emit("stage", stage="download", msg="Copying local clip" if info.get("local") else "Downloading Twitch clip")
    if not info.get("local"):
        run(YTDLP + ["-f", "b[height<=1080][ext=mp4]/b[ext=mp4]/b", "-o", str(job / "source.%(ext)s"), info["webpage_url"]])
    source = next(p for p in job.glob("source.*") if p.suffix not in (".json", ".part"))
    for old in job.glob("clip_*.mp4"):
        old.unlink()

    emit("stage", stage="scoring", msg=f"Whisper {a.whisper_model} on the clip")
    audio = decode_audio(source)
    duration = len(audio) / SR or duration  # the decoded clip, not the (slightly off) metadata duration
    vocal = scorer.vocal_curve(audio, SR)
    context = f"{meta.get('broadcaster_name') or ''} {meta.get('title') or ''}"
    words = []  # the clip IS the source: these timings are the editor's too (no second Whisper pass)
    lines = make_transcriber(a.whisper_model, a.device, context)(audio, [(0.0, duration)], words)[0]
    del audio
    text = scorer.transcript_between(lines, 0, duration)
    name = f"{'local' if info.get('local') else 'twitch'}_{meta.get('clip_id') or info['id']}"  # the queue's id for the clip = its output folder
    own = [job] + ([Path(a.output) / name] if getattr(a, "output", None) else [])  # an earlier run of THIS clip
    dup = recent_duplicate(meta.get("broadcaster_name"), text, [OUTPUT, a.workspace], own, any_age=getattr(a, "dedup_root", ()))
    if dup:  # another viewer clipped the same moment: don't judge or edit it twice
        source.unlink(missing_ok=True)
        emit("log", msg=f"Duplicate moment: {dup[1]:.0%} of the words match {dup[0]} (same streamer, last 24 h), skipped")
        write_manifest(job, info, [], {"source": "twitch", "twitch": meta, "duplicate_of": dup[0], "rejected": []})
        emit("done", manifest=str(archive(job, a, name)), count=0)
        return 0
    cue, tag = scorer.cue_score(text)
    peak = float(vocal.max()) if len(vocal) else 0.0
    c = {"start_sec": 0.0, "end_sec": round(duration, 2), "duration": round(duration, 2), "cue": cue, "signal": peak,
         "words": words or None,
         "vocal": peak, "visual": None, "transcript": text, "setup_start_sec": None, "setup_end_sec": None,
         "reason": f"Twitch clip by {meta.get('broadcaster_name')} | {meta.get('view_count')} views"
                   + (f" + vocal burst {peak:.2f}" if peak >= 0.5 else "") + (f" + {tag}" if tag else "")}
    if online:
        emit("stage", stage="scoring", msg="Ollama comedy judge")
        verdict = scorer.ollama_judge([c], a.ollama_model)
        if verdict and 0 in verdict:
            scorer.apply_verdict(c, verdict[0])
            scorer.viral_boost(c, meta.get("view_count"))
            scorer.streamer_boost(c, meta.get("broadcaster_name"))
    c["score"] = scorer.final_score(c["cue"], c["signal"], c["visual"], c.get("llm"))
    clips = []
    if getattr(a, "no_gate", False) or scorer.passes_gate(c, a.min_score):
        out = job / "clip_01_twitch.mp4"
        source.replace(out)  # the clip IS the highlight: keep it whole, no re-cut
        c.update(clip_id=1, file=out.name)
        emit("clip", **c)
        clips = [c]
    else:
        source.unlink(missing_ok=True)
        emit("log", msg=f"AI judge rejected this clip: {c['judge_reason'] or 'out of context or no payoff'}" if c.get("judge_reject")
             else f"Banger gate: rated {c['llm'] * 10:.1f}/10 < {a.min_score:g}, clip skipped")
        emit("warning", msg=scorer.judge_summary(name, [c], a.min_score))
    write_manifest(job, info, clips, {"source": "twitch", "twitch": meta, "rejected": scorer.rejected([c])})
    manifest_path = archive(job, a, name)
    emit("done", manifest=str(manifest_path), count=len(clips))
    if a.auto_edit:
        auto_edit(manifest_path)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--clips", type=int, default=4, help="number of clips (default 4)")
    ap.add_argument("--min-length", type=float, default=15, help="shortest clip in seconds (default 15)")
    ap.add_argument("--max-length", type=float, default=55, help="longest clip in seconds, incl. stitched setups (default 55)")
    ap.add_argument("--output", type=Path, default=OUTPUT, help="finished clips + manifest go to OUTPUT/<id>/ (default ./output)")
    ap.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parent.parent / "workspace",
                    help="downloads + clips go in <workspace>/<video id>/ (default ../workspace)")
    ap.add_argument("--min-score", type=float, default=0, help="banger gate: skip clips the Ollama judge rates below this /10 (default 0 = off)")
    ap.add_argument("--no-gate", action="store_true", help="Twitch: keep the clip even when the judge rejects it or rates it "
                    "below --min-score (calibration_studio judges everything)")
    ap.add_argument("--cleanup-source", action="store_true", help="delete the full source video after cutting")
    ap.add_argument("--whisper-model", default="small.en",
                    help="faster-whisper model for the candidate transcripts (small.en, base.en = faster on CPU)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="whisper device (auto = CUDA, else CPU)")
    ap.add_argument("--ollama-model", default=scorer.DEFAULT_MODEL,
                    help="local Ollama model used to re-rate candidates if Ollama is running")
    ap.add_argument("--no-ollama", action="store_true", help="never call Ollama")
    ap.add_argument("--dedup-root", action="append", default=[], metavar="DIR",
                    help="also skip Twitch clips whose transcript matches one archived under DIR/*/ (any age; repeatable)")
    ap.add_argument("--ignore-heatmap", action="store_true", help="force the audio-energy pass even when a heatmap exists")
    ap.add_argument("--no-visual", action="store_true", help="skip the 144p visual action scan")
    ap.add_argument("--no-callbacks", action="store_true", help="never stitch [setup] + [payoff] story callbacks")
    ap.add_argument("--cookies-from-browser", metavar="BROWSER",
                    help="send that browser's YouTube login to get past the bot check (edge, brave, firefox, chrome)")
    ap.add_argument("--cookies", metavar="FILE", help="Netscape cookies.txt for YouTube (alternative to --cookies-from-browser)")
    ap.add_argument("--twitch-meta", metavar="JSON", help="Helix clip metadata for a Twitch clip URL (set by queue_worker)")
    ap.add_argument("--auto-edit", action="store_true", help="also turn the clips into 9:16 shorts (editor/ -> ready_to_post/)")
    a = ap.parse_args()
    # every yt-dlp call (metadata, video, 144p analysis stream) builds on YTDLP, so this covers them all.
    # The YouTube login is only ever sent to YouTube (one run = one URL), never to Twitch.
    YTDLP.extend(cookie_args(a))
    if not 1 <= a.min_length <= a.max_length:
        ap.error("need 1 <= --min-length <= --max-length")

    missing = preflight()
    if missing:
        emit("error", msg="\n".join(missing))
        return 2

    a.workspace.mkdir(parents=True, exist_ok=True)
    if not a.no_ollama:
        a.ollama_model = scorer.resolve_model(a.ollama_model)
    if Path(a.url).is_file():  # a local video: one whole clip, no metadata fetch or download
        return twitch_clip(a, *local_info(a.url, a.workspace))
    emit("stage", stage="metadata", msg="Fetching metadata")
    info, job = fetch_metadata(a.url, a.workspace)
    if "twitch.tv" in (info.get("webpage_url") or a.url):
        return twitch_clip(a, info, job)
    duration = float(info.get("duration") or 0)
    emit("log", msg=f"{info.get('title')} ({duration:.0f}s), heatmap: {'yes' if info.get('heatmap') else 'no'}")
    if duration <= 0:
        raise RuntimeError("Video has no duration (live stream or premiere?)")

    # The video is needed for cutting anyway, so fetch it first: the audio-energy pass and the Whisper fallback
    # both read from it. The 144p analysis stream downloads alongside it.
    analysis = None if a.no_visual else start_analysis_download(info, job)
    online = not a.no_ollama and scorer.ollama_online()
    if online:  # load the judge model while the video downloads (cold load ~40 s on the GTX 1050 Ti)
        threading.Thread(target=scorer.ollama_warmup, args=(a.ollama_model,), daemon=True).start()
    emit("stage", stage="download", msg="Downloading video (<=1080p)")
    run(YTDLP + ["-f", VIDEO_FORMAT, "--merge-output-format", "mp4", "-o", str(job / "source.%(ext)s"), info["webpage_url"]])
    source = job / "source.mp4"
    for old in job.glob("clip_*.mp4"):  # replace the previous batch only once the new source is safely downloaded
        old.unlink()

    emit("stage", stage="scoring", msg="Scoring highlights")
    visual = None
    if analysis:
        t0 = time.time()
        visual = visual_scan(analysis, job)
        emit("log", msg=f"Visual action scan: {time.time() - t0:.1f}s" if visual is not None
             else "Visual action scan skipped (no 144p H.264 stream)")
    # The full audio track feeds the vocal laughter curve, the no-heatmap energy curve and batched Whisper.
    t0 = time.time()
    # shortcut: whole track in RAM (~230 MB per hour); stream it in chunks if multi-hour VODs hit the 8 GB box.
    audio = decode_audio(source)
    vocal = scorer.vocal_curve(audio, SR)
    emit("log", msg=f"Audio decode + vocal laughter scan: {time.time() - t0:.1f}s")
    if info.get("heatmap") and not a.ignore_heatmap:
        primary, reasons = scorer.heatmap_curve(info["heatmap"], duration)
    else:
        emit("stage", stage="scoring", msg="No heatmap: using audio energy")
        secs = len(audio) // SR
        rms = np.sqrt(np.mean(np.square(audio[:secs * SR].reshape(secs, SR)), axis=1)) if secs else []
        primary, reasons = scorer.energy_curve(rms)
    signal, vis, rank, reasons = scorer.signal_curves(primary, reasons, vocal, visual)
    voc = scorer.fit(vocal, len(signal))

    # Funnel: pull a wide pool, transcribe and score ALL of it on comedy signals, only then keep the best N.
    pool = scorer.pick_windows(rank, reasons, max(a.clips * 3, 12), a.min_length, a.max_length)
    # one batched Whisper pass over the padded candidate windows (no YouTube captions: see fetch_metadata)
    emit("stage", stage="scoring", msg=f"Whisper {a.whisper_model} on {len(pool)} candidate windows")
    t0 = time.time()
    transcribe = make_transcriber(a.whisper_model, a.device, f"{info.get('uploader') or ''} {info.get('title') or ''}")
    lines_per = transcribe(audio, [(max(0.0, s - 3), min(duration, e + 3)) for s, e, *_ in pool])
    emit("log", msg=f"Whisper batch: {time.time() - t0:.1f}s")
    del audio
    cands = []
    for (start, end, _, reason), lines in zip(pool, lines_per):
        start, end = scorer.snap_window(start, end, lines, min_len=a.min_length, max_len=a.max_length)
        text = scorer.transcript_between(lines, start, end)
        cue, tag = scorer.cue_score(text)
        s, e = int(start), max(int(start) + 1, int(end))
        cands.append({"start_sec": start, "end_sec": end, "cue": cue, "signal": float(signal[s:e].max()),
                      "vocal": float(voc[s:e].max()), "visual": float(vis[s:e].max()), "lines": lines,
                      "reason": reason + (f" + {tag}" if tag else ""), "transcript": text,
                      "setup": None, "setup_start_sec": None, "setup_end_sec": None})

    emit("log", msg=f"Ollama {'online: judging ' + str(len(cands)) + ' candidates with ' + a.ollama_model if online else 'offline/disabled, heuristic scoring only'}")
    if online:
        emit("stage", stage="scoring", msg=f"Ollama comedy judge on {len(cands)} candidates")
        t0 = time.time()
        verdict = scorer.ollama_judge(cands, a.ollama_model)
        emit("log", msg=f"Ollama judge: {time.time() - t0:.1f}s" + ("" if verdict else " (no usable answer, heuristic scoring only)"))
        for i, v in (verdict or {}).items():
            scorer.apply_verdict(cands[i], v)
    best = scorer.rank_candidates(cands, a.clips, a.min_score)
    hard = scorer.rejected(cands)
    gated = sum(not scorer.passes_gate(c, a.min_score) for c in cands) - len(hard)
    if hard:
        emit("log", msg=f"AI judge rejected {len(hard)} candidate(s) as out of context or no payoff")
    if gated:
        emit("log", msg=f"Banger gate: {gated} candidate(s) rated below {a.min_score:g}/10 skipped")
    summary = scorer.judge_summary(info["id"], cands, a.min_score)
    if summary:  # nothing passed: say why, so an empty CLIPS list is never a mystery
        emit("warning", msg=summary)

    # Story callbacks: only pairs the Ollama judge flagged (the setup candidate's Whisper lines), confirmed by a
    # shared rare word. (A whole-video scan would need a transcript of the whole video.)
    kfs = None
    for i, c in enumerate(best, 1):
        emit("stage", stage="cutting", msg=f"Cutting clip {i}/{len(best)}")
        out = job / f"clip_{i:02d}_{int(c['start_sec'])}s.mp4"
        hit, src_lines = None, c["lines"]
        if not a.no_callbacks and c["setup"] is not None and cands[c["setup"]]["start_sec"] <= c["start_sec"] - 45:
            src_lines = cands[c["setup"]]["lines"] + c["lines"]
            hit = scorer.find_callback(src_lines, c["start_sec"], c["transcript"], need=1)
        plan = None
        if hit:
            kfs = kfs if kfs is not None else keyframes(source)
            plan = scorer.stitch_plan(kfs, hit[:2], (c["start_sec"], c["end_sec"]), a.max_length)
        if plan:
            (ss, se), (ps, pe) = plan
            stitch(source, plan, out)
            c.update(setup_start_sec=ss, setup_end_sec=se, start_sec=ps, end_sec=pe,
                     reason=f"Story Callback: [{scorer.clock(ss)}-{scorer.clock(se)}] -> [{scorer.clock(ps)}-{scorer.clock(pe)}]"
                            f" ({', '.join(hit[2])}) | {c['reason']}",
                     transcript=scorer.transcript_between(src_lines, ss, se) + " ... " + scorer.transcript_between(src_lines, ps, pe))
            duration_s = (se - ss) + (pe - ps)
        else:
            cut(source, c["start_sec"], c["end_sec"], out)
            duration_s = c["end_sec"] - c["start_sec"]
        c.update(clip_id=i, file=out.name, duration=round(duration_s, 2))
        emit("clip", **{k: v for k, v in c.items() if k != "lines"})

    write_manifest(job, info, best, {"source": "youtube", "channel": info.get("uploader_id") or info.get("channel_id"),
                                     "rejected": hard})
    manifest_path = archive(job, a, info["id"])

    if a.cleanup_source:
        source.unlink(missing_ok=True)
        emit("log", msg="Source video deleted (--cleanup-source)")
    emit("done", manifest=str(manifest_path), count=len(best))
    if a.auto_edit:
        auto_edit(manifest_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        emit("error", msg=str(e))
        sys.exit(1)

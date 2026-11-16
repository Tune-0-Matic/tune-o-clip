"""Side-by-side benchmark: 10/10 reference Shorts (Group A) vs our generated Shorts (Group B).

  python backend/benchmark_winners.py measure [--winners DIR] [--ours N]     -> workspace/benchmark_report.json
  python backend/benchmark_winners.py analyze --report workspace/benchmark_report.json
        (the 14B / local Ollama "why" analysis; prints {"event": "analysis", "text": ...}; run it on the RunPod Pod with
         `runpod_manager.py run -- backend/benchmark_winners.py analyze --report ...` so qwen2.5:14b-instruct answers)

Everything measured is deterministic (ffprobe / ffmpeg ebur128 + scene detection / faster-whisper word timings /
the editor's own energy curve), so Group A and B are measured exactly the same way. Caption placement / font size
would need OCR and is not measured.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "editor")]
import scorer  # noqa: E402

WINNERS = Path.home() / "Videos" / "Working Clips"
REPORT = ROOT / "workspace" / "benchmark_report.json"
GAP = 0.4  # s of silence between words that counts as dead air
HOOK_S = 3.0
SCENE = 0.35  # hard cuts (scene / angle swaps)
SOFT = 0.15  # visual changes incl. jump cuts and zoom punches inside one shot (0.35 saw none of those)


def emit(event, **kw):
    print(json.dumps({"event": event, **kw}), flush=True)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, encoding="utf-8").stdout
    d = json.loads(out)
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    return float(d["format"]["duration"]), int(v.get("width") or 0), int(v.get("height") or 0)


def loudness(path):
    """Integrated LUFS, loudness range, true peak (ebur128)."""
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    summary = err[err.rfind("Summary:"):]
    num = lambda key: float(m.group(1)) if (m := re.search(key + r":\s+(-?[\d.]+)", summary)) else None
    return {"lufs": num("I"), "lra": num("LRA"), "true_peak_db": num("Peak")}


def scene_cuts(path, thr=SCENE):
    """Times (s) where the picture changes more than thr (ffmpeg scene score)."""
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-an", "-vf", f"select='gt(scene,{thr})',showinfo",
                          "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    return [round(float(t), 2) for t in re.findall(r"pts_time:([\d.]+)", err)]


def speech(words, duration):
    """Pacing from word timings: first word, WPM, dead-air gaps, tail after the last word."""
    if not words:
        return {"words": 0, "first_word_ms": None, "wpm": 0.0, "gaps": 0, "max_gap_s": None, "dead_air_s": 0.0,
                "tail_after_last_word_s": None, "hook_text": "", "last_text": ""}
    gaps = [b["start"] - a["end"] for a, b in zip(words, words[1:]) if b["start"] - a["end"] > GAP]
    return {"words": len(words), "first_word_ms": round(words[0]["start"] * 1000),
            "wpm": round(len(words) / max(duration, 1) * 60, 1), "gaps": len(gaps),
            "max_gap_s": round(max(gaps), 2) if gaps else 0.0, "dead_air_s": round(sum(gaps) + words[0]["start"], 2),
            "tail_after_last_word_s": round(duration - words[-1]["end"], 2),
            "hook_text": " ".join(w["word"] for w in words if w["start"] < HOOK_S),
            "last_text": " ".join(w["word"] for w in words if w["start"] >= max(0.0, words[-1]["end"] - 3.0))}


def measure(path, transcribe=None):
    """Every fingerprint of one video -> dict."""
    import clipper  # noqa: E402  (audio decode)
    import director  # noqa: E402  (the editor's own energy curve and peak finder)
    import pipeline  # noqa: E402  (word timings, the same call the editor uses)
    duration, w, h = probe(path)
    audio = clipper.decode_audio(path)
    env = director.energy(audio)
    pk = director.peaks(env)
    loud = [q for q in pk if director.env_at(env, q) >= 0.8]
    top = pk[0] if pk else None
    words = (transcribe or pipeline.transcribe_words)(audio)
    cuts, changes = scene_cuts(path), scene_cuts(path, SOFT)
    d = {"file": Path(path).name, "duration_s": round(duration, 2), "width": w, "height": h,
         "aspect": f"{w}x{h}", "aspect_ratio": round(w / h, 3) if h else None,
         **speech(words, duration), **loudness(path),
         "loud_peaks": len(loud), "loudest_at_pct": round(100 * top / duration) if top is not None else None,
         "tail_after_loudest_s": round(duration - top, 2) if top is not None else None,
         "cuts": len(cuts), "cuts_per_10s": round(len(cuts) / max(duration, 1) * 10, 2),
         "avg_shot_s": round(duration / (len(cuts) + 1), 2), "first_cut_s": cuts[0] if cuts else None,
         "changes_per_10s": round(len(changes) / max(duration, 1) * 10, 2), "first_change_s": changes[0] if changes else None,
         "avg_change_gap_s": round(duration / (len(changes) + 1), 2)}
    return d


SUMMARY = ["duration_s", "first_word_ms", "wpm", "gaps", "max_gap_s", "dead_air_s", "tail_after_last_word_s", "lufs", "lra",
           "true_peak_db", "loud_peaks", "loudest_at_pct", "tail_after_loudest_s", "cuts_per_10s", "avg_shot_s", "first_cut_s",
           "changes_per_10s", "avg_change_gap_s", "first_change_s", "aspect_ratio"]


def aggregate(rows):
    out = {}
    for k in SUMMARY:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        out[k] = {"median": round(statistics.median(vals), 2), "mean": round(statistics.mean(vals), 2), "n": len(vals)} if vals else None
    out["aspects"] = {a: sum(r["aspect"] == a for r in rows) for a in sorted({r["aspect"] for r in rows})}
    return out


def ours(n=20, root=ROOT):
    """The newest n generated shorts (ready_to_post/ and every calibration run's shorts/)."""
    files = list((root / "ready_to_post").glob("*_short.mp4")) + list((root / "calibration_studio" / "runs").glob("*/shorts/*_short.mp4"))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:n]


def run_measure(winners=WINNERS, n_ours=20, report=REPORT):
    groups = {"A_winners": sorted(p for p in Path(winners).iterdir() if p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv")),
              "B_ours": ours(n_ours)}
    result = {"generated": time.strftime("%Y-%m-%d %H:%M"), "gap_threshold_s": GAP, "scene_threshold": SCENE,
              "whisper": scorer.DEFAULT_WHISPER, "groups": {}}
    for g, files in groups.items():
        rows = []
        for i, f in enumerate(files, 1):
            emit("progress", group=g, n=i, of=len(files), file=f.name)
            try:
                rows.append(measure(f))
            except Exception as e:  # one unreadable file must not stop the benchmark
                emit("warning", msg=f"{f.name}: {e}")
        result["groups"][g] = {"clips": rows, "summary": aggregate(rows)}
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    emit("done", report=str(report))
    return result


# --- the 14B "why" analysis -----------------------------------------------------------------------------------------

def table(result):
    a, b = result["groups"]["A_winners"]["summary"], result["groups"]["B_ours"]["summary"]
    lines = ["| metric (median) | A: 10/10 winners | B: ours | gap |", "|---|---|---|---|"]
    for k in SUMMARY:
        if a.get(k) and b.get(k):
            va, vb = a[k]["median"], b[k]["median"]
            lines.append(f"| {k} | {va:g} | {vb:g} | {vb - va:+.2f} |")
    lines.append(f"| aspect ratios | {a['aspects']} | {b['aspects']} | |")
    return "\n".join(lines)


def prompt(result):
    hooks = lambda g: "\n".join(f"- ({r['duration_s']:.0f}s) {r['hook_text'][:140]!r} ... ends: {r['last_text'][:100]!r}"
                                for r in result["groups"][g]["clips"])
    return f"""You are a senior short-form video editor. Group A are 20 viral 10/10 YouTube Shorts / TikToks of streamers
(Twitch / YouTube clips) made by top clipping channels. Group B are 20 Shorts our automatic clipper made; viewers rate
them about 5/10. Below: the measured medians (same tools for both groups) and every clip's opening 3 seconds and ending.

{table(result)}

Definitions: first_word_ms = time until the first spoken word; wpm = words per minute; gaps = silences > {GAP}s between
words; dead_air_s = their total + the silence before the first word; tail_after_last_word_s = seconds kept after the last
word; loud_peaks = moments >= 80% of the clip's loudest; loudest_at_pct = where the loudest moment sits (% of the
clip); tail_after_loudest_s = seconds kept after it; cuts_per_10s / avg_shot_s = hard scene cuts; changes_per_10s /
avg_change_gap_s = every visible change incl. jump cuts and zoom punches.

Group A openings / endings:
{hooks("A_winners")}

Group B openings / endings:
{hooks("B_ours")}

Write three sections in plain Markdown:
1. "Key differences": the 5 most important measured differences, each with the numbers and what it means for a viewer.
2. "Why A feels 10/10 and B 5/10": hook strength, self-contained context, pacing, comedic timing, endings. Quote the
   openings above as evidence.
3. "Rules to adopt": 6-10 concrete, testable editing / clip-selection rules for an automatic clipper (with numbers,
   e.g. "first spoken word within X ms", "cut within X s after the punchline"), most impactful first.
"""


def analyze(report=REPORT, model=None, url="http://localhost:11434"):
    result = json.loads(Path(report).read_text(encoding="utf-8"))
    model = scorer.resolve_model(model or scorer.DEFAULT_MODEL, url)
    body = json.dumps({"model": model, "prompt": prompt(result), "stream": False,
                       "options": {"temperature": 0.3, "num_ctx": 16384}}).encode()
    req = urllib.request.Request(url + "/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.loads(r.read())["response"]
    emit("analysis", model=model, text=text)
    return model, text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["measure", "analyze"])
    ap.add_argument("--winners", type=Path, default=WINNERS)
    ap.add_argument("--ours", type=int, default=20)
    ap.add_argument("--report", type=Path, default=REPORT)
    a = ap.parse_args(argv)
    if a.mode == "measure":
        run_measure(a.winners, a.ours, a.report)
    else:
        analyze(a.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

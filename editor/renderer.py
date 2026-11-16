"""Deterministic short-form factory: edl.json + clip + word timings -> 1080x1920 H.264/AAC short.

1. One ffmpeg pass: frame-accurate trims (+ speed ramps) + concat -> 9:16 framing (split_stack = a verified corner
   webcam on top + full-width gameplay below, else blur_fill = the whole frame over a blurred copy; see framing.py)
   + the retention effects (cam_fx: snap zooms + yell shake on the FACECAM panel only, gameplay never moves) ->
   audio: original with the punchline pitch drop + whoosh / thud SFX + ducked music -> loudnorm -14 LUFS.
   Never AV1: NVENC / VAAPI / libx264 H.264.
2. pycaps (editor/captions.py): captions + sound cues on the framed short, timed by caption_doc(), at CAPTION_Y.
edl["profile"] == "emotional_story" (director.story_profile) swaps the hype parts out, matched to a measured reference
Short: hard cuts + an optional muted flash montage before the final line (edl["montage"]), a punchy grade (contrast
1.10, saturation up, teal shadows, light vignette) + a slow 1.0 -> 1.07x push-in, a generated ambient piano / pad bed
at -19 dBFS in the speech gaps (mixed after the voice's loudnorm, ducked under speech) and bold 2-line uppercase
captions at 52% with an italic glowing accent line.
"""
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MUSIC_DIR, SFX_DIR = ROOT / "assets" / "music", ROOT / "assets" / "sfx"
SFX_NAMES = ("whoosh", "thud")
W, H = 1080, 1920
CAM_H, BORDER = 690, 4  # split_stack: facecam top 36%, 4 px seam, gameplay below
GAME_H = H - CAM_H - BORDER


# --- timeline -----------------------------------------------------------------------------------------------------

def _speed_at(edl, s, e):
    for sp in edl.get("speed", []):
        if sp["start"] <= s and e <= sp["end"] + 0.01:
            return float(sp["factor"])
    return 1.0


def _split_by_speed(edl):
    """Cuts split at speed-ramp borders -> [(start, end, factor)]."""
    out = []
    for c in edl["cuts"]:
        points = sorted({c["start"], c["end"], *[x for sp in edl.get("speed", []) for x in (sp["start"], sp["end"])
                                                  if c["start"] < x < c["end"]]})
        out += [(a, b, _speed_at(edl, a, b)) for a, b in zip(points, points[1:]) if b - a > 0.02]
    return out


def story(edl):
    return edl.get("profile") == "emotional_story"


def timeline(edl):
    """[(src_start, src_end, out_start, kind, speed)] in output order, and the total output length. kind: False = a
    main cut, True = the hook teaser, "montage" = an emotional_story flash piece (voice muted), inserted where
    edl["montage"]["at"] starts a cut (before the final line). to_out() maps only main cuts."""
    segs, t = [], 0.0
    hook = edl.get("hook_teaser") or {}
    if hook.get("enabled") and hook["end"] > hook["start"]:
        segs.append((hook["start"], hook["end"], 0.0, True, 1.0))
        t = hook["end"] - hook["start"]
    montage = edl.get("montage") or {}
    flashes = [(a, b) for a, b in montage.get("clips", [])]
    for s, e, sp in _split_by_speed(edl):
        if flashes and s >= montage["at"] - 1e-6:
            for a, b in flashes:
                segs.append((a, b, t, "montage", 1.0))
                t += b - a
            flashes = []
        segs.append((s, e, t, False, sp))
        t += (e - s) / sp
    return segs, t + float(edl.get("freeze_end") or 0)


def to_out(src_t, segs):
    """Source time -> output time in the main cuts (teaser excluded), or None if that moment was cut."""
    for s, e, o, teaser, sp in segs:
        if not teaser and s <= src_t <= e:
            return o + (src_t - s) / sp
    return None


def to_src(out_t, segs):
    """Output time -> (segment index, source time)."""
    for i, (s, e, o, _, sp) in enumerate(segs):
        if o <= out_t <= o + (e - s) / sp:
            return i, s + (out_t - o) * sp
    return None, None


def main_length(segs):
    return sum((e - s) / sp for s, e, _, _, sp in segs)


# --- captions (pycaps, editor/captions.py) ------------------------------------------------------------------------

def clean(word, upper=True):
    w = str(word).strip()
    return "".join(ch for ch in (w.upper() if upper else w) if ch not in "{}\\")


def chunk(words, size=3, span=1.3, gap=0.35):
    """1-3 word lines, each <= `span` s long, broken early at pauses and sentence ends."""
    groups, cur = [], []
    for w in words:
        if cur and (len(cur) == size or w["oend"] - cur[0]["ostart"] > span or w["ostart"] - cur[-1]["oend"] > gap
                    or cur[-1]["word"].rstrip()[-1:] in ".!?"):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def caption_doc(edl, words, out_words=None):
    """The caption track as pycaps whisper_json, in OUTPUT seconds: 1-3 word uppercase lines (emotional_story: up to 6
    words <= 2.6 s, which pycaps sets as 2 lines), one on screen at a time (each ends 20 ms before the next starts).
    out_words: Whisper words of the cut audio itself (cut_audio), already in output time: used as they are. Without
    them the source `words` are mapped through the cuts. The muted montage gets no captions."""
    upper = True
    placed = []
    for s, e, o, kind, sp in timeline(edl)[0]:
        if kind == "montage":
            continue
        if out_words is None:
            placed += [dict(w, ostart=o + (max(w["start"], s) - s) / sp, oend=o + (min(w["end"], e) - s) / sp)
                       for w in words if w["end"] > s and w["start"] < e and clean(w["word"])]
        else:
            oe = o + (e - s) / sp
            placed += [dict(w, ostart=max(w["start"], o), oend=min(w["end"], oe))
                       for w in out_words if o <= (w["start"] + w["end"]) / 2 < oe and clean(w["word"])]
    # a word straddling a jump cut lands in both segments: its 1-2 frame sliver (< 60 ms) is dropped
    groups = chunk(sorted((w for w in placed if w["oend"] - w["ostart"] >= 0.06), key=lambda w: w["ostart"]),
                   *((6, 2.6) if story(edl) else (3, 1.3)))
    segments = []
    for k, g in enumerate(groups):
        stop = groups[k + 1][0]["ostart"] - 0.02 if k + 1 < len(groups) else 1e9
        ws = [{"word": clean(w["word"], upper), "start": round(w["ostart"], 3), "end": round(min(w["oend"], stop), 3)} for w in g]
        ws = [w for w in ws if w["end"] > w["start"]]
        if ws:
            segments.append({"start": ws[0]["start"], "end": ws[-1]["end"], "words": ws})
    return {"segments": segments}


# SETTINGS caption theme (edl["caption_tone"]) -> the active word's colour
CAPTION_COLORS = {"punch": "#FFE600", "hype": "#3DFF5A", "rage": "#FF3B3B", "roast": "#00E5FF"}
CAPTION_Y = {"split_stack": 0.46, "blur_fill": 0.72, "story": 0.52}  # caption block centre: under the webcam seam /
#                                                                       lower third / emotional_story mid-screen
CAPTION_TIMEOUT = 900  # s: pycaps composites every frame in Python (~25 fps at 1080x1920 on the i7-860)


def burn_captions(src, out, doc, workdir, y=CAPTION_Y["blur_fill"], color=None, style="pop"):
    """pycaps pass (editor/captions.py, own process) over a rendered short -> out. style: pop | story."""
    (Path(workdir) / "captions.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")  # kept for debugging
    Path(out).unlink(missing_ok=True)  # pycaps refuses to overwrite: a QC repair re-renders onto the same path
    cmd = [sys.executable, str(Path(__file__).with_name("captions.py")), str(src), str(out), str(Path(workdir) / "captions.json"), f"{y:.3f}", style]
    p = subprocess.run(cmd + ([color] if color else []), cwd=workdir, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=CAPTION_TIMEOUT)
    if p.returncode != 0 or not Path(out).exists():
        raise RuntimeError("pycaps captions failed:\n" + (p.stderr or p.stdout).strip()[-1500:])
    return Path(out)


# --- encoder / probe ----------------------------------------------------------------------------------------------

_ENCODER = None


def detect_encoder(force=None):
    """Hardware H.264 when it actually works (NVENC on the GTX 1050 Ti, VAAPI on the Ryzen's Vega), else libx264.
    -> (name, global args, extra video filter, codec args). Probed once with a tiny real encode."""
    global _ENCODER
    choices = {
        # cq 19 on the RunPod Pod (TOC_CLOUD): it has the GPU time to spare; 22 on the GTX 1050 Ti
        "h264_nvenc": ([], "", ["-c:v", "h264_nvenc", "-preset", "p3", "-rc", "vbr", "-cq",
                                "19" if os.environ.get("TOC_CLOUD") == "1" else "22", "-b:v", "0", "-profile:v", "high"]),
        "h264_vaapi": (["-vaapi_device", "/dev/dri/renderD128"], ",format=nv12,hwupload", ["-c:v", "h264_vaapi", "-qp", "21"]),
        "libx264": ([], "", ["-c:v", "libx264", "-preset", "superfast", "-crf", "20", "-profile:v", "high"]),
    }
    if force:
        return (force, *choices[force])
    if _ENCODER:
        return _ENCODER
    for name in ["h264_nvenc"] + (["h264_vaapi"] if os.name != "nt" else []) + ["libx264"]:
        g, vf, codec = choices[name]
        probe_cmd = ["ffmpeg", "-v", "error", *g, "-f", "lavfi", "-i", "color=s=1080x1920:d=0.2", "-vf", "format=yuv420p" + vf,
                     *codec, "-f", "null", "-"]
        if name == "libx264" or subprocess.run(probe_cmd, capture_output=True).returncode == 0:
            _ENCODER = (name, g, vf, codec)
            return _ENCODER


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    return {"width": v["width"], "height": v["height"], "fps": num / (den or 1), "duration": float(d["format"]["duration"]),
            "has_audio": any(s["codec_type"] == "audio" for s in d["streams"])}


# --- retention effects (pre-pycaps, inside the one ffmpeg pass) -----------------------------------------------------

def sfx_file(name, folder=SFX_DIR):
    """User-dropped .mp3/.wav wins over the generated .wav (an .mp3 can sit next to the generated file)."""
    for ext in (".mp3", ".wav"):
        if (Path(folder) / f"{name}{ext}").exists():
            return Path(folder) / f"{name}{ext}"
    return None


def ensure_sfx(folder=SFX_DIR, sr=48000):
    """Synthesize the whoosh / thud (numpy + wave) when there is no file yet, so SFX work out of the box."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    osc = lambda freqs: np.sin(2 * np.pi * np.cumsum(freqs) / sr)
    for name in SFX_NAMES:
        if sfx_file(name, folder):
            continue
        if name == "thud":  # clean low-end boom: 110 -> 45 Hz sine drop, soft click, no distortion (sits under a voice)
            t = np.arange(int(0.7 * sr)) / sr
            x = (osc(45 + 65 * np.exp(-t * 14)) + 0.15 * rng.standard_normal(len(t)) * np.exp(-t * 300)) \
                * np.minimum(1, t / 0.002) * np.exp(-t / 0.18)
        else:  # whoosh: noise swept through a resonant band
            t = np.arange(int(0.7 * sr)) / sr
            x = rng.standard_normal(len(t)) * (0.5 + 0.5 * osc(300 + 2500 * (t / t[-1]))) * np.sin(np.pi * t / t[-1]) ** 2
        x = (x / (np.abs(x).max() or 1) * 0.9 * 32767).astype(np.int16)
        with wave.open(str(folder / f"{name}.wav"), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(x.tobytes())
    return folder


def cam_fx(edl, segs, w, h, fps):
    """Facecam panel only: held snap zooms (zoompan about (0.5, 0.3): split_stack's crop puts the head in the panel's
    upper half, so hair stays in at 1.35x) + a 0.2 s +-6 px shake on every yell. Moments the cuts removed are skipped."""
    chain = []
    zooms = [(o, z) for z in edl.get("zooms", []) if (o := to_out(z["at"], segs)) is not None]
    if zooms:
        terms = "+".join(f"between(it,{o:.3f},{o + z['duration']:.3f})*{z['scale'] - 1:.3f}" for o, z in zooms)
        chain.append(f"zoompan=z='min(1.35,1+{terms})':x='(iw-iw/zoom)*0.5':y='(ih-ih/zoom)*0.3':d=1:s={w}x{h}:fps={fps}")
    shakes = [(o, o + sh["end"] - sh["start"]) for sh in edl.get("shakes", []) if (o := to_out(sh["start"], segs)) is not None]
    if shakes:
        amp = "+".join(f"between(t,{a:.3f},{b:.3f})*6" for a, b in shakes)
        chain.append(f"crop=w=iw-12:h=ih-12:x='6+({amp})*sin(113*t)':y='6+({amp})*cos(97*t)',scale={w}:{h}")
    return "".join("," + c for c in chain)


def pitch_graph(a, edl, segs):
    """The punchline's pitch drop (edl["pitch"] factor 0.86 = deep, 1.14 = up): asetrate shifts the pitch and also
    stretches time, so atempo puts the length back and the video stays in sync. -> (filter lines, output label)."""
    p = edl.get("pitch")
    s, e = (to_out(p["start"], segs), to_out(p["end"], segs)) if p else (None, None)
    if s is None or e is None or s < 0.05 or e - s < 0.1:
        return [], a
    f = float(p["factor"])
    return [f"{a}asplit=3[p0][p1][p2]", f"[p0]atrim=end={s:.3f},asetpts=PTS-STARTPTS[q0]",
            f"[p1]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS,asetrate=48000*{f:g},aresample=48000,atempo={1 / f:.4f}[q1]",
            f"[p2]atrim=start={e:.3f},asetpts=PTS-STARTPTS[q2]", "[q0][q1][q2]concat=n=3:v=0:a=1[pa]"], "[pa]"


# --- emotional_story look + ambient bed -----------------------------------------------------------------------------

STORY_PUSH = 0.07  # 1.0 -> 1.07x over the whole short
WARM = {"wholesome", "grateful", "thankful", "proud", "inspiring", "inspired", "happy", "hopeful", "joy", "joyful",
        "love", "loving", "excited", "touching", "heartfelt", "relieved", "reunion"}  # every other feeling grades cool
# measured reference (SAT 21, YAVG 112, wide luma): punchy contrast + saturation, teal shadows, warm-ish highlights
STORY_GRADE = "eq=contrast=1.10:saturation=1.18,colorbalance=rs=-0.05:gs=0.02:bs=0.07:rh=0.03:bh=-0.02,vignette=PI/6"
STORY_ACCENT = {"warm": "#FFD60A", "sad": "#FF3B3B", "cool": "#3DE8FF"}  # the caption emphasis line, by emotion
SAD = {"sad", "heartbroken", "heartbreak", "grief", "grieving", "hurt", "lonely", "loss", "crying", "devastated"}
STORY_BEDS = {"sad_ambient": [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)],    # Am F C G (minor)
              "warm_ambient": [(48, 52, 55), (55, 59, 62), (57, 60, 64), (53, 57, 60)]}   # C G Am F (major)
STORY_BED_DBFS = -19.0  # the bed's RMS in the speech gaps of the finished short (the reference measured -19.0)


def story_tint(edl):
    """warm (wholesome / proud: major bed, yellow accent), sad (red accent) or cool (nostalgic / reflective / ...)."""
    e = str(edl.get("emotion") or "").lower()
    return "warm" if e in WARM else "sad" if e in SAD else "cool"


def rms_dbfs(path):
    """A .wav's RMS in dBFS (the bed gain is set from it); None for anything else (a user's .mp3 plays as it is)."""
    if Path(path).suffix.lower() != ".wav":
        return None
    with wave.open(str(path)) as f:
        x = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float64) / 32768
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


def ensure_story_bed(name, folder=MUSIC_DIR, sr=48000, chord_s=4.0):
    """assets/music/<name>.wav: a 16 s seamless ambient loop (numpy + wave, royalty-free by construction): a slow pad
    (each chord a Hann-windowed swell overlapping the next) + a soft decaying piano arpeggio + a short echo. A user's
    own <name>.mp3 / .wav in assets/music wins."""
    if f := sfx_file(name, folder):
        return f
    Path(folder).mkdir(parents=True, exist_ok=True)
    chords = STORY_BEDS[name]
    n = int(len(chords) * chord_s * sr)
    t = np.arange(n) / sr
    hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)
    x = np.zeros(n)
    span = int(chord_s * 1.5 * sr)  # each swell lasts 1.5 chords: neighbours overlap, no dips
    win = np.hanning(span)
    for k, chord in enumerate(chords):
        idx = (int(k * chord_s * sr - 0.25 * chord_s * sr) + np.arange(span)) % n  # wraps: the loop is seamless
        tt = np.arange(span) / sr
        pad = sum(np.sin(2 * np.pi * hz(m) * tt) + 0.5 * np.sin(2 * np.pi * (hz(m) * 2 + 0.4) * tt) for m in chord)
        x[idx] += 0.25 * pad * win
        for j, m in enumerate(list(chord) + [chord[0] + 12]):  # piano: root, third, fifth, octave, one per second
            start = int((k * chord_s + j) * sr)
            tt = np.arange(int(2.5 * sr)) / sr
            note = (np.sin(2 * np.pi * hz(m + 12) * tt) + 0.35 * np.sin(4 * np.pi * hz(m + 12) * tt)) \
                * np.minimum(1, tt / 0.006) * np.exp(-tt / 0.9)
            x[(start + np.arange(len(tt))) % n] += 0.45 * note
    x = x + 0.3 * np.roll(x, int(0.37 * sr)) + 0.15 * np.roll(x, int(0.74 * sr))  # roll: the echo wraps too
    x = (x / (np.abs(x).max() or 1) * 0.8 * 32767).astype(np.int16)
    path = Path(folder) / f"{name}.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(x.tobytes())
    return path


def user_music(folder=MUSIC_DIR):
    """The hype edit's background music: the user's .mp3 / .wav in assets/music, never the generated story beds."""
    folder = Path(folder)
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in (".mp3", ".wav") and p.stem not in STORY_BEDS) \
        if folder.exists() else []


# --- filter graph ------------------------------------------------------------------------------------------------

def cut_graph(segs):
    """The cuts as filtergraph lines -> [vc][ac]. concat stretches each piece to its longer stream (video trims land on
    whole frames), so the real timeline drifts a little from timeline()'s ideal one over many cuts. Montage flash
    pieces (emotional_story) keep their picture but lose their voice: the bed carries them."""
    n = len(segs)
    fc = [f"[0:v]split={n}" + "".join(f"[vs{i}]" for i in range(n)), f"[0:a]asplit={n}" + "".join(f"[as{i}]" for i in range(n))]
    for i, (s, e, _, kind, sp) in enumerate(segs):
        fc.append(f"[vs{i}]trim=start={s:.3f}:end={e:.3f},setpts=(PTS-STARTPTS)/{sp:.4f}[v{i}]")
        fc.append(f"[as{i}]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS" + (f",atempo={sp:.4f}" if sp != 1 else "")
                  + (",volume=0" if kind == "montage" else "") + f"[a{i}]")
    fc.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vc][ac]")
    return fc


def cut_audio(src, edl, out):
    """The render's own speech track (cuts, teaser and speed ramps; no SFX / music / loudnorm) as a 16 kHz mono WAV
    for Whisper. The video runs through the same trims to a null output so concat pads exactly as in the render."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter_complex", ";".join(cut_graph(timeline(edl)[0])),
           "-map", "[ac]", "-ac", "1", "-ar", "16000", str(out), "-map", "[vc]", "-f", "null", "-"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError("ffmpeg cut audio failed:\n" + p.stderr.strip()[-800:])
    return Path(out)


def build_command(src, edl, out, info, encoder=None, facecam=None, music_dir=MUSIC_DIR, sfx_dir=SFX_DIR):
    """-> ffmpeg argv: cuts -> 9:16 framing + facecam effects (no captions) -> audio + pitch + SFX + music -> loudnorm."""
    _, gargs, vf_tail, codec = encoder or detect_encoder()
    segs, length = timeline(edl)
    fps = 30  # the platforms play 30; 60 doubled the caption pass's per-frame work for nothing
    inputs = ["-i", str(src)]
    fc = cut_graph(segs)
    freeze = float(edl.get("freeze_end") or 0)
    v, a = "[vc]", "[ac]"
    if freeze > 0:
        fc += [f"[vc]tpad=stop_mode=clone:stop_duration={freeze:.2f}[vf]", f"[ac]apad=pad_dur={freeze:.2f}[af]"]
        v, a = "[vf]", "[af]"
    fc.append(f"{v}fps={fps},setsar=1[vb]")
    if edl.get("layout") == "split_stack" and facecam:  # verified corner webcam on top, full-width gameplay below
        x, y, cw, ch = facecam
        fc.append("[vb]split=2[cs][gs]")
        fc.append(f"[cs]crop=iw*{cw}:ih*{ch}:iw*{x}:ih*{y},scale={W}:{CAM_H}:force_original_aspect_ratio=increase,"
                  f"crop={W}:{CAM_H}{cam_fx(edl, segs, W, CAM_H, fps)},pad={W}:{CAM_H + BORDER}:0:0:color=0x0b0c0d[cam]")
        fc.append(f"[gs]scale=-2:{GAME_H},crop={W}:{GAME_H}[game]")
        fc.append("[cam][game]vstack[lay]")
    else:  # the whole frame, sharp and centred over a blurred copy of itself
        fh = int(round(W * info["height"] / info["width"] / 2) * 2)  # 16:9 -> 1080x608
        fc.append("[vb]split[bgs][fgs]")
        # blur a quarter-size copy and scale it up: same look, ~16x less blur work than at 1080x1920
        fc.append(f"[bgs]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},boxblur=8:2,scale={W}:{H}[bg]")
        fc.append(f"[fgs]scale={W}:{fh}[fg]")
        fc.append("[bg][fg]overlay=(W-w)/2:(H-h)/2[lay]")
    look = ""
    if story(edl):  # slow push-in (scale grows with t, centre crop back to 1080x1920) + the grade + vignette
        look = f"scale=w='2*trunc({W // 2}*(1+{STORY_PUSH}*t/{max(length, 1):.2f}))':h=-2:eval=frame,crop={W}:{H},{STORY_GRADE},"
    fc.append(f"[lay]{look}format=yuv420p" + vf_tail + "[vout]")

    if story(edl):  # the voice loudnormed first, then the bed at a fixed -19 dBFS that ducks under it (no SFX here)
        bed = ensure_story_bed("warm_ambient" if story_tint(edl) == "warm" else "sad_ambient", music_dir)
        level = rms_dbfs(bed)
        inputs += ["-stream_loop", "-1", "-i", str(bed)]
        fc.append(f"{a}aformat=sample_rates=48000:channel_layouts=stereo,loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,"
                  "asplit=2[vn][side]")
        fc.append(f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={STORY_BED_DBFS - level if level is not None else 0:.1f}dB,"
                  f"afade=t=in:d=1.5,afade=t=out:st={max(0.0, length - 2.0):.2f}:d=2[mu]")
        fc.append("[mu][side]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[duck]")
        fc.append("[vn][duck]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.89[aout]")
        threads = ["-filter_complex_threads", "4", "-threads", "4"] if os.environ.get("TOC_LOW_SPEC") == "1" else []
        return ["ffmpeg", "-v", "error", "-y", *threads[:2], *gargs, *inputs, "-filter_complex", ";".join(fc), "-map", "[vout]",
                "-map", "[aout]", "-r", str(fps), *codec, *threads[2:], "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]

    music, gain = user_music(music_dir) if music_dir else [], edl.get("bg_music", {}).get("gain_db", -22.0)
    use_music = edl.get("bg_music", {}).get("enabled") and music
    fc.append(f"{a}aformat=sample_rates=48000:channel_layouts=stereo[a48]")
    pitch, a = pitch_graph("[a48]", edl, segs)
    fc += pitch
    fc.append(a + ("asplit=2[main][side]" if use_music else "anull[main]"))
    mix, k = ["[main]"], 1
    for fx in edl.get("sfx", []):
        f, at = sfx_file(fx["name"], sfx_dir), to_out(fx["at"], segs)
        if f is None or at is None:
            continue  # missing asset or the moment was cut: skip quietly
        inputs += ["-i", str(f)]
        ms = int(at * 1000)
        fc.append(f"[{k}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={fx['gain_db']:.1f}dB,adelay={ms}|{ms}[sx{k}]")
        mix.append(f"[sx{k}]")
        k += 1
    if use_music:
        inputs += ["-stream_loop", "-1", "-i", str(music[0])]
        fc.append(f"[{k}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={gain:.1f}dB[mu]")
        fc.append("[mu][side]sidechaincompress=threshold=0.02:ratio=10:attack=15:release=350[duck]")
        mix.append("[duck]")
    tail = "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]"
    fc.append("".join(mix) + f"amix=inputs={len(mix)}:normalize=0:duration=first,{tail}" if len(mix) > 1 else f"[main]{tail}")

    threads = ["-filter_complex_threads", "4", "-threads", "4"] if os.environ.get("TOC_LOW_SPEC") == "1" else []  # i7-860 stays usable
    return ["ffmpeg", "-v", "error", "-y", *threads[:2], *gargs, *inputs, "-filter_complex", ";".join(fc), "-map", "[vout]", "-map", "[aout]",
            "-r", str(fps), *codec, *threads[2:], "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]


def render(src, edl, words, out, workdir, encoder=None, facecam=None, out_words=None):
    """Render one short; returns the output path. ffmpeg cuts + frames it, then pycaps adds the captions and sound cues
    (none when the source has its own captions burned in: edl["source_captions"]). Raises RuntimeError on failure."""
    workdir, out = Path(workdir), Path(out).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    info = probe(src)
    if not info["has_audio"]:
        raise RuntimeError(f"{src} has no audio track")
    ensure_sfx()
    doc = {"segments": []} if edl.get("source_captions") else caption_doc(edl, words, out_words)
    base = workdir.resolve() / "base.mp4" if doc["segments"] else out
    cmd = build_command(Path(src).resolve(), edl, base, info, encoder, facecam)
    (workdir / "ffmpeg_cmd.json").write_text(json.dumps(cmd, indent=1), encoding="utf-8")  # for debugging a bad render
    p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError("ffmpeg render failed:\n" + p.stderr.strip()[-1500:])
    if base != out:
        if story(edl):  # 2-line captions with the emotion's accent line; no theme colour, hype tags or pops
            burn_captions(base, out, doc, workdir, CAPTION_Y["story"], STORY_ACCENT[story_tint(edl)], "story")
        else:
            y = CAPTION_Y["split_stack" if edl.get("layout") == "split_stack" and facecam else "blur_fill"]
            burn_captions(base, out, doc, workdir, y, CAPTION_COLORS.get(edl.get("caption_tone")))
        base.unlink()
    return out

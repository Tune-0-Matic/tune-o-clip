"""Deterministic FFmpeg short-form factory: edl.json + clip + word timings -> 1080x1920 H.264/AAC short.

One ffmpeg pass: frame-accurate trims (+ speed ramps) + concat -> 9:16 framing (split_stack facecam over gameplay
with full-screen facecam takeovers, face_crop = a 9:16 column following the faces, or blur_fill; see framing.py)
with zooms/trauma shake per panel -> deep fry / RGB split /
grayscale / flash / vignette -> kinetic .ass captions -> audio: original (+ mutes) + SFX + ducked music ->
loudnorm -14 LUFS. Never AV1: NVENC / VAAPI / libx264 H.264.
"""
import json
import os
import subprocess
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SFX_DIR, MUSIC_DIR, FONTS_DIR = ROOT / "assets" / "sfx", ROOT / "assets" / "music", ROOT / "assets" / "fonts"
W, H = 1080, 1920
CAM_H, BORDER = 690, 4  # split_stack: facecam top 36%, 4 px seam, gameplay below
GAME_H = H - CAM_H - BORDER
SFX_NAMES = ("vine_boom", "metal_pipe", "bass_boost", "record_scratch", "sad_violin", "whoosh", "pop", "thud")
# ASS colours are &HBBGGRR&
TONE_COLORS = {"punch": "&H0000FFFF&",  # #FFFF00 yellow: the active word
               "hype": "&H0044FF00&",   # #00FF44 neon green: wins, "let's go"
               "rage": "&H002A2AFF&",   # #FF2A2A bright red: screams, rage, scares
               "roast": "&H00FFFF00&"}  # #00FFFF cyan: the active word when it is hype / loud
POP = r"\fscx118\fscy118\t(0,70,\fscx100\fscy100)"  # every active word pops 118% -> 100% in 70 ms


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


def timeline(edl):
    """[(src_start, src_end, out_start, is_teaser, speed)] in output order, and the total output length."""
    segs, t = [], 0.0
    hook = edl.get("hook_teaser") or {}
    if hook.get("enabled") and hook["end"] > hook["start"]:
        segs.append((hook["start"], hook["end"], 0.0, True, 1.0))
        t = hook["end"] - hook["start"]
    for s, e, sp in _split_by_speed(edl):
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


# --- kinetic captions ---------------------------------------------------------------------------------------------

def ass_time(t):
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t // 60 % 60):02d}:{t % 60:05.2f}"


def clean(word):
    return "".join(ch for ch in str(word).strip().upper() if ch not in "{}\\")


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


def write_ass(edl, words, path):
    """The ONE caption track (none when the source has its own burned in: edl["source_captions"]). Kinetic captions: 1-3 word lines that pop in with a scale bounce, uppercase heavy font (per scenario;
    Impact on screams), 8 px outline + 4 px shadow, centre-lower at MarginV 660 (clear of the split-stack seam and
    the platform's title / subscribe UI: raised from 540 after batch 1). No teaser banner: the yellow bar on top was
    disliked twice in batch 1. Neutral words white; the active word coloured by its tone (yellow / green / red / cyan)."""
    segs, _ = timeline(edl)
    tones = edl.get("word_tones") or []
    font = edl.get("font") or "Arial Black"
    lines = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0", "",
             "[V4+ Styles]",
             "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
             "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
             "MarginL, MarginR, MarginV, Encoding",
             f"Style: Pop,{font},100,&H00FFFFFF,&H00FFFFFF,&H00000000,&HA0000000,-1,0,0,0,100,100,1,0,1,4,3,2,140,140,660,1",
             "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    events = []  # (start, end, text) for the caption track
    for s, e, o, _, sp in segs:
        seg_words = [dict(w, ostart=o + (max(w["start"], s) - s) / sp, oend=o + (min(w["end"], e) - s) / sp,
                          tone=tones[i] if i < len(tones) else "punch")
                     for i, w in enumerate(words) if w["end"] > s and w["start"] < e and clean(w["word"])]
        if edl.get("caption_tone"):  # SETTINGS caption theme overrides the per-word tones
            seg_words = [dict(w, tone=edl["caption_tone"]) for w in seg_words]
        for group in chunk(seg_words):
            for i, w in enumerate(group):
                end = group[i + 1]["ostart"] if i + 1 < len(group) else w["oend"] + 0.12
                parts = []
                for x in group:
                    if x is w:
                        # the active word pops and takes the colour; the words after it go back to white at 100%
                        # explicitly (a {\r} reset would also drop the style font for a swapped Impact word)
                        swap = r"\fnImpact" if x["tone"] == "rage" else ""
                        back = rf"{{\c&H00FFFFFF&\fscx100\fscy100\fn{font}}}" if swap else r"{\c&H00FFFFFF&\fscx100\fscy100}"
                        parts.append(f"{{\\c{TONE_COLORS.get(x['tone'], TONE_COLORS['punch'])}{swap}{POP}}}{clean(x['word'])}{back}")
                    else:
                        parts.append(clean(x["word"]))
                text = " ".join(parts)
                events.append((w["ostart"], max(end, w["ostart"] + 0.05), text))
    # strictly one caption on screen at a time (teaser + main cuts share this one track): each event ends 20 ms
    # before the next starts, keeping starts on the spoken word; slivers under 60 ms (1-2 frames) are dropped
    events.sort(key=lambda ev: ev[:2])
    for i, (a, b, text) in enumerate(events):
        if i + 1 < len(events):
            b = min(b, events[i + 1][0] - 0.02)
        if b - a >= 0.06:
            lines.append(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Pop,,0,0,0,,{text}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- procedural SFX -----------------------------------------------------------------------------------------------

def sfx_file(name, folder=SFX_DIR):
    """User-dropped .mp3/.wav wins over the generated .wav (an .mp3 can sit next to the generated file)."""
    for ext in (".mp3", ".wav"):
        if (Path(folder) / f"{name}{ext}").exists():
            return Path(folder) / f"{name}{ext}"
    return None


def ensure_sfx(folder=SFX_DIR, sr=48000):
    """Synthesize the SFX library (numpy + wave) for any name that has no file yet, so SFX work out of the box."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    def tt(sec):
        return np.arange(int(sec * sr)) / sr

    def env(t, attack, decay):
        return np.minimum(1, t / max(attack, 1e-4)) * np.exp(-t / decay)

    def osc(freqs):
        return np.sin(2 * np.pi * np.cumsum(freqs) / sr)

    def make(name):
        if name == "vine_boom":  # big detuned sub thump with a click and a long saturated tail
            t = tt(1.6)
            x = osc(48 + 60 * np.exp(-t * 9)) + 0.5 * osc(np.full(len(t), 97.0)) + 0.3 * rng.standard_normal(len(t)) * np.exp(-t * 60)
            return np.tanh(2.5 * x) * env(t, 0.003, 0.55)
        if name == "metal_pipe":  # inharmonic clang partials, sharp hit, long ring
            t = tt(2.2)
            partials = [(523, 1.0), (1312, 0.7), (2291, 0.5), (3517, 0.35), (4870, 0.2)]
            x = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * (1.2 + f / 1500)) for f, a in partials)
            return (x + 0.4 * rng.standard_normal(len(t)) * np.exp(-t * 80)) * env(t, 0.001, 5)
        if name == "bass_boost":  # distorted 45 Hz drop
            t = tt(0.9)
            return np.tanh(6 * osc(45 + 30 * np.exp(-t * 6))) * env(t, 0.005, 0.4)
        if name == "record_scratch":  # band noise with a fast wobbling pitch
            t = tt(0.5)
            f = 900 + 800 * np.sin(2 * np.pi * 7 * t)
            return (0.6 * osc(f) * rng.uniform(0.5, 1, len(t)) + 0.4 * rng.standard_normal(len(t))) * env(t, 0.01, 0.22)
        if name == "sad_violin":  # slow descending saw-ish melody with vibrato
            t = tt(3.2)
            notes = np.repeat([659.3, 587.3, 523.3, 493.9], len(t) // 4 + 1)[:len(t)]
            f = notes * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * t))
            ph = np.cumsum(f) / sr
            x = sum(np.sin(2 * np.pi * k * ph) / k for k in range(1, 7))
            return x * np.minimum(1, t / 0.25) * np.minimum(1, (t[-1] - t) / 0.6)
        if name == "thud":  # clean low-end boom: 110 -> 45 Hz sine drop, soft click, no distortion (sits under a voice)
            t = tt(0.7)
            x = osc(45 + 65 * np.exp(-t * 14)) + 0.15 * rng.standard_normal(len(t)) * np.exp(-t * 300)
            return x * env(t, 0.002, 0.18)
        if name == "whoosh":  # noise swept through a resonant band
            t = tt(0.7)
            noise = rng.standard_normal(len(t))
            sweep = osc(300 + 2500 * (t / t[-1]))
            return noise * (0.5 + 0.5 * sweep) * np.sin(np.pi * t / t[-1]) ** 2
        t = tt(0.12)  # pop: short bright blip
        return np.sin(2 * np.pi * (1200 - 4000 * t) * t) * env(t, 0.002, 0.03)

    for name in SFX_NAMES:
        if sfx_file(name, folder):
            continue
        x = make(name)
        x = (x / (np.abs(x).max() or 1) * 0.9 * 32767).astype(np.int16)
        with wave.open(str(folder / f"{name}.wav"), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(x.tobytes())
    return folder


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


# --- filter graph ------------------------------------------------------------------------------------------------

def _sum(terms):
    return "+".join(terms) if terms else "0"


def teaser_on(segs):
    return bool(segs) and segs[0][3]


def out_time(item, key, segs):
    """Output time of an EDL entry. "out": True entries are already output seconds and belong to the hook teaser:
    gone with it (a QC repair can drop the teaser)."""
    if item.get("out"):
        return item[key] if teaser_on(segs) else None
    return to_out(item[key], segs)


def zoom_expr(zooms, segs, var="it"):
    """1 + sum of zoom terms: snap/punch = instant step, ramp = exponential ease-in, creep/drift = linear."""
    terms = []
    for z in zooms:
        a = out_time(z, "at", segs)
        if a is None:
            continue
        d, s = max(0.1, z["duration"]), z["scale"] - 1
        win = f"between({var},{a:.3f},{a + d:.3f})"
        style = z.get("style", "snap")
        if style == "ramp":  # (e^(3u)-1)/(e^3-1): slow start, accelerating into the outburst
            terms.append(f"{win}*{s:.3f}*(exp(3*({var}-{a:.3f})/{d:.3f})-1)/19.0855")
        elif style in ("creep", "drift"):
            terms.append(f"{win}*{s:.3f}*({var}-{a:.3f})/{d:.3f}")
        else:
            terms.append(f"{win}*{s:.3f}")
    return terms


def track_expr(track, segs, var="t"):
    """Piecewise-linear crop centre (0..1 of the source width) over OUTPUT time from [(source s, x)] keyframes."""
    keys = sorted((o, x) for o, x in ((to_out(s, segs), x) for s, x in track) if o is not None)
    if not keys:
        return "0.5"
    terms = [f"lt({var},{keys[0][0]:.3f})*{keys[0][1]:.4f}", f"gte({var},{keys[-1][0]:.3f})*{keys[-1][1]:.4f}"]
    for (a, xa), (b, xb) in zip(keys, keys[1:]):
        if b > a:  # gte/lt, not between(): a shared endpoint must not count twice
            terms.append(f"gte({var},{a:.3f})*lt({var},{b:.3f})*({xa:.4f}+{xb - xa:.4f}*({var}-{a:.3f})/{b - a:.3f})")
    return _sum(terms)


def spans(filters, kind, segs):
    out = []
    for f in filters:
        if f["type"] == kind:
            a = out_time(f, "start", segs)
            b = a + f["end"] - f["start"] if f.get("out") and a is not None else to_out(min(f["end"], f["start"] + 60), segs)
            if a is not None:
                out.append((a, b if b is not None and b > a else a + 0.3))
    return out


def _on(sp):
    return _sum([f"between(t,{a:.3f},{b:.3f})" for a, b in sp])


def panel_fx(edl, segs, w, h, fps, zoom=True):
    """Zoom (zoompan about edl["zoom_anchor"] = [ax, ay, max zoom] from framing.zoom_room: the anchor point stays
    put, so a punch-in grows down from the heads' top, and the cap keeps every head in the panel) + trauma shake
    (+-8 px damped micro jitter via a fixed-size crop) on one panel."""
    chain = []
    zt = zoom_expr(edl.get("zooms", []), segs) if zoom else []
    if zt:
        ax, ay, top = edl.get("zoom_anchor") or (0.5, 0.5, 1.35)
        chain.append(f"zoompan=z='min({top:.3f},1+{_sum(zt)})':x='(iw-iw/zoom)*{ax:.3f}':y='(ih-ih/zoom)*{ay:.3f}'"
                     f":d=1:s={w}x{h}:fps={fps}")
    shakes = spans(edl.get("filters", []), "trauma_shake", segs)
    if shakes:
        amp = _sum([f"between(t,{a:.3f},{b:.3f})*8*exp(-12*(t-{a:.3f}))" for a, b in shakes])
        chain.append(f"crop=w=iw-20:h=ih-20:x='10+({amp})*sin(113*t)':y='10+({amp})*cos(97*t)',scale={w}:{h}")
    return ("," + ",".join(chain)) if chain else ""


def build_command(src, edl, words, out, workdir, info, encoder=None, facecam=None, sfx_dir=SFX_DIR, music_dir=MUSIC_DIR):
    """-> ffmpeg argv (run with cwd=workdir, where subs.ass is written)."""
    _, gargs, vf_tail, codec = encoder or detect_encoder()
    segs, _ = timeline(edl)
    fps = 30  # the platforms play 30; 60 doubled every per-frame filter (zoompan, shake, captions) for nothing
    n = len(segs)
    write_ass(edl, words, Path(workdir) / "subs.ass")
    inputs = ["-i", str(src)]
    fc = [f"[0:v]split={n}" + "".join(f"[vs{i}]" for i in range(n)), f"[0:a]asplit={n}" + "".join(f"[as{i}]" for i in range(n))]
    for i, (s, e, _, _, sp) in enumerate(segs):
        fc.append(f"[vs{i}]trim=start={s:.3f}:end={e:.3f},setpts=(PTS-STARTPTS)/{sp:.4f}[v{i}]")
        fc.append(f"[as{i}]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS" + (f",atempo={sp:.4f}" if sp != 1 else "") + f"[a{i}]")
    fc.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vc][ac]")
    freeze = float(edl.get("freeze_end") or 0)
    v, a = "[vc]", "[ac]"
    if freeze > 0:
        fc += [f"[vc]tpad=stop_mode=clone:stop_duration={freeze:.2f}[vf]", f"[ac]apad=pad_dur={freeze:.2f}[af]"]
        v, a = "[vf]", "[af]"
    fc.append(f"{v}fps={fps},setsar=1[vb]")

    takes = [(to_out(t["start"], segs), to_out(t["end"], segs)) for t in edl.get("takeovers", [])]
    takes = [(p, q if q is not None else p + 1.5) for p, q in takes if p is not None]
    if edl.get("layout") == "split_stack" and facecam:
        x, y, cw, ch = facecam
        crop_cam = f"crop=iw*{cw}:ih*{ch}:iw*{x}:ih*{y}"
        fc.append("[vb]split=" + ("3[cs][gs][fs]" if takes else "2[cs][gs]"))
        fc.append(f"[cs]{crop_cam},scale={W}:{CAM_H}:force_original_aspect_ratio=increase,crop={W}:{CAM_H}"
                  f"{panel_fx(edl, segs, W, CAM_H, fps)},pad={W}:{CAM_H + BORDER}:0:0:color=0x0b0c0d[cam]")
        fc.append(f"[gs]scale=-2:{GAME_H},crop={W}:{GAME_H}{panel_fx(edl, segs, W, GAME_H, fps, zoom=False)}[game]")
        fc.append("[cam][game]vstack" + ("[stack]" if takes else "[lay]"))
        if takes:  # full-screen facecam for the reaction, then snap back to the stack
            fc.append(f"[fs]{crop_cam},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
                      f"{panel_fx(edl, segs, W, H, fps, zoom=False)}[full]")
            fc.append(f"[stack][full]overlay=0:0:enable='{_on(takes)}'[lay]")
    elif edl.get("layout") == "face_crop" and edl.get("crop_track"):  # full-height 9:16 column following the faces
        fc.append(f"[vb]crop=w=trunc(ih*9/32)*2:h=ih:x='clip(({track_expr(edl['crop_track'], segs)})*iw-ow/2,0,iw-ow)':y=0,"
                  f"scale={W}:{H}{panel_fx(edl, segs, W, H, fps)}[lay]")
    else:
        fh = int(round(W * info["height"] / info["width"] / 2) * 2)  # 16:9 -> 1080x608
        fc.append("[vb]split[bgs][fgs]")
        # blur a quarter-size copy and scale it up: same look, ~16x less blur work than at 1080x1920
        fc.append(f"[bgs]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},boxblur=8:2,scale={W}:{H}[bg]")
        fc.append(f"[fgs]scale={W}:{fh}{panel_fx(edl, segs, W, fh, fps)}[fg]")
        fc.append("[bg][fg]overlay=(W-w)/2:(H-h)/2[lay]")

    post = []
    flt = edl.get("filters", [])
    gray = spans(flt, "grayscale", segs) + ([(main_length(segs), 1e4)] if freeze > 0 else [])
    if gray:
        post += [f"hue=s=0:enable='{_on(gray)}'", f"vignette=PI/4:enable='{_on(gray)}'"]
    # high-energy reaction boost: punchier contrast/colour + sharpening, no colour cast ("deep_fry" = older EDLs)
    boost = spans(flt, "reaction_boost", segs) + spans(flt, "deep_fry", segs)
    if boost:
        post += [f"eq=contrast=1.28:saturation=1.35:brightness=0.03:enable='{_on(boost)}'",
                 f"unsharp=5:5:1.2:enable='{_on(boost)}'"]
    rgb = [(a, min(b, a + 2 / fps)) for a, b in spans(flt, "rgb_split", segs)]  # a light 2-frame glitch, not a smear
    if rgb:
        post.append(f"rgbashift=rh=-3:bh=3:enable='{_on(rgb)}'")
    flash = spans(flt, "flash", segs)
    if flash:
        post.append(f"eq=brightness=0.45:enable='{_on(flash)}'")
    vig = spans(flt, "vignette", segs)
    if vig:
        post.append(f"vignette=PI/3.5:enable='{_on(vig)}'")
    # relative to the work dir (ffmpeg's cwd): an absolute Windows path's "C:" would need double filtergraph escaping
    has_fonts = FONTS_DIR.exists() and any(p.suffix.lower() in (".ttf", ".otf") for p in FONTS_DIR.iterdir())
    fonts = f":fontsdir={Path(os.path.relpath(FONTS_DIR, workdir)).as_posix()}" if has_fonts else ""
    post += ([] if edl.get("source_captions") else [f"subtitles=subs.ass{fonts}"]) + ["format=yuv420p"]
    fc.append("[lay]" + ",".join(post) + vf_tail + "[vout]")

    # audio: original (with mutes) + SFX + ducked music -> loudnorm
    mutes = [(to_out(m["start"], segs), to_out(m["end"], segs)) for m in edl.get("mutes", [])]
    mutes = [(p, q if q is not None else p + 0.3) for p, q in mutes if p is not None]
    music = sorted(p for p in Path(music_dir).glob("*") if p.suffix.lower() in (".mp3", ".wav")) if music_dir and Path(music_dir).exists() else []
    use_music = edl.get("bg_music", {}).get("enabled") and music
    main = f"{a}aformat=sample_rates=48000:channel_layouts=stereo" + (f",volume=0:enable='{_on(mutes)}'" if mutes else "")
    fc.append(main + (",asplit=2[main][side]" if use_music else "[main]"))
    mix, k = ["[main]"], 1
    for fx in edl.get("sfx", []):
        f, at = sfx_file(fx["name"], sfx_dir), out_time(fx, "at", segs)
        if f is None or at is None:
            continue  # missing asset or the moment was cut: skip quietly
        inputs += ["-i", str(f)]
        ms = int(at * 1000)
        fc.append(f"[{k}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={fx['gain_db']:.1f}dB,adelay={ms}|{ms}[sx{k}]")
        mix.append(f"[sx{k}]")
        k += 1
    if use_music:
        inputs += ["-stream_loop", "-1", "-i", str(music[0])]
        fc.append(f"[{k}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={edl['bg_music']['gain_db']:.1f}dB[mu]")
        fc.append("[mu][side]sidechaincompress=threshold=0.02:ratio=10:attack=15:release=350[duck]")
        mix.append("[duck]")
    tail = "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]"
    fc.append("".join(mix) + f"amix=inputs={len(mix)}:normalize=0:duration=first,{tail}" if len(mix) > 1 else f"[main]{tail}")

    threads = ["-filter_complex_threads", "4", "-threads", "4"] if os.environ.get("TOC_LOW_SPEC") == "1" else []  # i7-860 stays usable
    return ["ffmpeg", "-v", "error", "-y", *threads[:2], *gargs, *inputs, "-filter_complex", ";".join(fc), "-map", "[vout]", "-map", "[aout]",
            "-r", str(fps), *codec, *threads[2:], "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]


def render(src, edl, words, out, workdir, encoder=None, facecam=None):
    """Render one short; returns the output path. Raises RuntimeError with ffmpeg's error on failure."""
    Path(workdir).mkdir(parents=True, exist_ok=True)
    info = probe(src)
    if not info["has_audio"]:
        raise RuntimeError(f"{src} has no audio track")
    ensure_sfx()
    cmd = build_command(Path(src).resolve(), edl, words, Path(out).resolve(), workdir, info, encoder, facecam)
    (Path(workdir) / "ffmpeg_cmd.json").write_text(json.dumps(cmd, indent=1), encoding="utf-8")  # for debugging a bad render
    p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError("ffmpeg render failed:\n" + p.stderr.strip()[-1500:])
    return Path(out)

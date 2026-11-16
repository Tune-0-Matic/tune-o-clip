"""AI Director: word-level transcript + audio energy -> one of 5 streamer scenarios -> edl.json (edit decision list).

Local Ollama only picks the *creative* fields (scenario, title, hashtags, banner); the exact cut points, zooms,
effects, takeovers and SFX are computed deterministically from Whisper word timings and the audio energy curve, so
the EDL is always frame-sane and the Ollama-offline fallback is the same code path.

All times in the EDL are SOURCE-clip seconds; the renderer maps them onto the edited timeline (so QC repairs
that change the cuts never desync zooms, SFX or captions).
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import scorer  # noqa: E402  (the shared Ollama call with the hard timeout)

SCENARIOS = ("chat_roast", "unfiltered_chaos", "horror_jumpscare", "confident_fail", "awkward_irl")
SCREAM_CUT = 0.40  # hard cut this long after the peak scream (horror)
MIN_LEN, MAX_LEN = 8.5, 57.0  # inside QC's 7.0 .. 58.5 s with margin (speed ramps / snapping shave a little)
HOP = 0.05  # energy curve resolution (s)
FAIL_RE = re.compile(r"\b(dead|died|dies|wasted|killed|crash(ed)?|rip|i'?m done|oh no|no+o+|got me|i'?m cooked|fell|lost)\b", re.I)
SCARE_RE = re.compile(r"\b(what was that|what is that|scary|scared|ghost|run+|help|behind (me|you)|jump ?scare|ahh+|aaa+)\b", re.I)
HYPE_RE = re.compile(r"^(let'?s|go+|yes+|yeah+|w|win|won|gg|clutch|sheesh|ez|easy|dub|insane|lets)$", re.I)
RAGE_RE = re.compile(r"^(what+|why+|no+|stop|shut|hell|damn|bro+|wtf|ahh+|aaa+|f\w*k\w*)$", re.I)
ROAST_RE = re.compile(r"\b(chat|bro|you'?re|stupid|dumb|ugly|imagine|ratio|literally|actually|okay|sure|whatever|clown)\b", re.I)
# active-word tone per scenario when a word has no stronger cue
DEFAULT_TONE = {"chat_roast": "roast", "unfiltered_chaos": "punch", "horror_jumpscare": "punch",
                "confident_fail": "hype", "awkward_irl": "punch"}
FONTS = {"chat_roast": "Arial Black", "unfiltered_chaos": "Impact", "horror_jumpscare": "Impact",
         "confident_fail": "Arial Black", "awkward_irl": "Arial Black"}  # heavy sans only (kinetic Shorts captions)
PROMPT = """You are the editor of viral vertical streamer shorts. Read this clip (the comedy judge's notes, the audio facts and
the transcript with word timings in seconds) and rate how well EACH editing scenario fits it, from 0 (not at all) to 10:
- chat_roast: calm talk or reading chat, then the streamer roasts or insults someone (CaseOh style).
- unfiltered_chaos: yelling, screaming or hype through most of the clip (IShowSpeed style).
- horror_jumpscare: quiet tension, then a jump scare and a scream.
- confident_fail: someone confident or bragging, then an instant fail / death / crash / loss (instant karma).
- awkward_irl: slow, awkward or deadpan real-life moment or conversation.
Reply with JSON only:
{"fit": {"chat_roast": number, "unfiltered_chaos": number, "horror_jumpscare": number, "confident_fail": number,
 "awkward_irl": number}, "social_title": "punchy title under 60 characters",
 "hashtags": [four real hashtags for this clip: the game, the streamer, the joke, and #shorts],
 "banner_text": "2-4 word teaser like WAIT FOR IT..."}
"""


# --- signal helpers -----------------------------------------------------------------------------------------------

def energy(audio, sr=16000, hop=HOP):
    """Per-hop RMS of the clip audio, scaled 0..1."""
    n = int(sr * hop)
    frames = len(audio) // n
    if frames == 0:
        return np.zeros(0)
    rms = np.sqrt(np.mean(np.square(audio[:frames * n].reshape(frames, n).astype(np.float64)), axis=1))
    return rms / (rms.max() or 1.0)


def peaks(env, hop=HOP, rel=0.55, min_gap=1.5, limit=8):
    """Times (s) of the loudest moments, loudest first, at least min_gap apart and >= rel * max."""
    out = []
    top = env.max() if env.size else 0
    for i in np.argsort(-env):
        if env[i] < rel * top or len(out) == limit:
            break
        t = i * hop + hop / 2
        if all(abs(t - p) >= min_gap for p in out):
            out.append(round(float(t), 2))
    return out


def env_at(env, t, hop=HOP):
    return float(env[min(len(env) - 1, max(0, int(t / hop)))]) if env.size else 0.0


GAP = 0.35  # silence longer than this is cut (unfiltered_chaos: 0.25)
PAD = 0.1  # kept before / after each sound: a cut opens <= 0.1 s before the first word or reaction
REACT = 0.35  # >= 35% of the clip's loudest with no words = a laugh / scream / reaction: kept like speech


def vad_speech(audio, sr=16000):
    """Silero VAD speech spans (s), faster-whisper's bundled model: catches laughs and half-words Whisper drops."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError:  # editor without faster-whisper: words + loud reactions still work
        return []
    ts = get_speech_timestamps(np.asarray(audio, dtype=np.float32), VadOptions(min_silence_duration_ms=200, speech_pad_ms=30),
                               sampling_rate=sr)
    return [(t["start"] / sr, t["end"] / sr) for t in ts]


def loud_runs(env, hop=HOP, level=REACT, min_len=0.1):
    """(start, end) s where the audio stays >= level (and >= 2.5x the clip's median: a constant game / music bed is
    not a reaction) for >= min_len."""
    level = max(level, 2.5 * float(np.median(env)))
    runs, start = [], None
    for i, v in enumerate(list(env) + [0.0]):
        if v >= level and start is None:
            start = i
        elif v < level and start is not None:
            if (i - start) * hop >= min_len:
                runs.append((start * hop, i * hop))
            start = None
    return runs


def keep_spans(lo, hi, words, speech=(), env=None, gap=GAP, pad=PAD, hop=HOP):
    """Keep-segments inside [lo, hi]: words + VAD speech + loud reactions, merged across silences <= gap, each padded by
    `pad`. Every silence longer than gap becomes a jump cut (replaces the old word-gap cutter that kept up to 1.5 s)."""
    sounds = [(w["start"], w["end"]) for w in words] + list(speech) + (loud_runs(env, hop) if env is not None and len(env) else [])
    sounds = sorted((max(lo, s), min(hi, e)) for s, e in sounds if e > lo and s < hi)
    if not sounds:
        return [(lo, hi)]
    out = [list(sounds[0])]
    for s, e in sounds[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(round(max(lo, s - pad), 2), round(min(hi, e + pad), 2)) for s, e in out]


def merge(cuts):
    out = []
    for s, e in sorted(cuts):
        if out and s <= out[-1][1] + 0.05:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def total(cuts):
    return sum(e - s for s, e in cuts)


def fit_length(cuts, duration, min_len=MIN_LEN):
    """Grow to min_len (give back trimmed pauses latest-first, then earlier setup; extending past the end would undo
    a hard cut, so that is the last resort) or shrink to MAX_LEN from the front, keeping the payoff at the end.
    A source shorter than min_len simply keeps all of itself."""
    cuts = [list(c) for c in cuts]
    for i in range(len(cuts) - 1, 0, -1):
        need = min_len - total(cuts)
        if need <= 0:
            break
        cuts[i][0] -= min(need, cuts[i][0] - cuts[i - 1][1])
    if total(cuts) < min_len:
        cuts[0][0] = max(0.0, cuts[0][0] - (min_len - total(cuts)))
    if total(cuts) < min_len:
        cuts[-1][1] = min(duration, cuts[-1][1] + (min_len - total(cuts)))
    cuts = [list(c) for c in merge([tuple(c) for c in cuts])]
    while total(cuts) > MAX_LEN + 1e-6:  # tolerance: an `over` below float precision would never shrink it
        over = total(cuts) - MAX_LEN
        if cuts[0][1] - cuts[0][0] <= over:
            cuts.pop(0)
        else:
            cuts[0][0] += over
    return [{"start": round(s, 2), "end": round(e, 2)} for s, e in cuts]


def word_safe(cuts, words, duration, keep_last_end=False, pre=0.08, post=0.14):
    """Never cut inside a word: a cut starts >= `pre` before its first word and ends >= `post` after its last one
    (boundaries only ever move outward). keep_last_end: leave the final end alone (horror's hard cut)."""
    cuts = [list(c) for c in cuts]
    for i, c in enumerate(cuts):
        for _ in range(len(words) + 1):  # widening can pull in a neighbouring word: repeat until stable
            s, e = c
            for w in words:
                if w["end"] > s and w["start"] < e:  # the word is in this cut
                    s = min(s, max(0.0, w["start"] - pre))
                    if not (keep_last_end and i == len(cuts) - 1):
                        e = max(e, min(duration, w["end"] + post))
            if [s, e] == c:
                break
            c[:] = [s, e]
    return [(round(s, 2), round(e, 2)) for s, e in merge([tuple(c) for c in cuts])]


def tail_guard(cuts, env, duration, hop=HOP, level=0.35, limit=1.2):
    """Let the reaction ring out: push the final end while the audio is still >= `level` of the clip's peak
    (laughs, screams decaying), by at most `limit` seconds. 1.8 -> 1.2 s after batch 1: 5 "ending went on too long /
    scream could've been cut sooner" notes vs 2 "great end cut"."""
    if not cuts or not env.size:
        return cuts
    s, e = cuts[-1]
    stop = min(duration, e + limit)
    while e < stop and env_at(env, e, hop) >= level:
        n = round(min(stop, e + hop), 2)
        if n <= e:  # rounding can't pass a stop within 0.005 s (26.7733 -> 26.77): this froze a whole batch
            break
        e = n
    return cuts[:-1] + [(s, e)]


FREEZE_LEN = 0.8
LOUD = 0.8  # a peak at >= 80% of the clip's loudest = a real yell / scream
MIN_CLEAR_WORDS, SPIKE_RATIO = 4, 6.0  # skip_render: fewer clear words than this, and no peak >= 6x the median level


# --- scenario choice ---------------------------------------------------------------------------------------------

def classify(words, env, duration, hop=HOP):
    """Rule-based scenario from the transcript and the energy curve (the Ollama-offline fallback)."""
    text = " ".join(w["word"] for w in words)
    tail = " ".join(w["word"] for w in words if w["start"] >= duration * 0.5)
    pk = peaks(env, hop)
    loud = [q for q in pk if env_at(env, q) >= 0.8]
    ratio = (env.max() / (np.median(env) + 1e-6)) if env.size else 0
    rate = len(words) / max(duration, 1)
    if SCARE_RE.search(text):
        return "horror_jumpscare"
    if FAIL_RE.search(tail):
        return "confident_fail"
    if len(loud) >= 4 and rate > 1.8:
        return "unfiltered_chaos"
    # one huge late peak after quiet tension (few words before it); a roast has steady talk before its outburst
    if ratio > 12 and len(loud) == 1 and pk and pk[0] > duration * 0.4 \
            and sum(w["end"] <= pk[0] for w in words) / max(pk[0], 1) < 0.5:
        return "horror_jumpscare"
    if ratio < 4:  # no real loudness spike anywhere: an even, awkward talk
        return "awkward_irl"
    gap_before_peak = 0.0
    if pk:
        before = [w for w in words if w["end"] <= pk[0]]
        gap_before_peak = pk[0] - before[-1]["end"] if before else 0.0
    if ROAST_RE.search(text) or (pk and pk[0] > duration * 0.4 and gap_before_peak > 0.7):
        return "chat_roast"
    return "chat_roast"


def word_tones(words, env, scenario=None, pivot=None):
    """Active-word highlight: "roast" (cyan) on hype / rage / loud (>= 85% of the peak) words, "punch" (yellow) on the
    rest: the two high-contrast Shorts caption colours. scenario / pivot are kept for callers; the tone no longer
    depends on them."""
    out = []
    for w in words:
        bare = re.sub(r"[^\w']", "", w["word"]).lower()
        loud = env_at(env, (w["start"] + w["end"]) / 2) >= yell_level(env)
        out.append("roast" if loud or HYPE_RE.match(bare) or RAGE_RE.match(bare) else "punch")
    return out


def heuristic_creative(words, scenario, meta=None):
    """Title/hashtags without an LLM: the clip title or first line of the transcript + source-based tags."""
    meta = meta or {}
    line = " ".join(w["word"] for w in words[:12]).strip(" ,.") or "This went wrong fast"
    title = (meta.get("title") or line)[:57].strip()
    tags = ["#shorts", "#funny", "#gaming", "#twitch" if meta.get("broadcaster_name") else "#streamer"]
    return {"social_title": title, "hashtags": tags, "banner_text": "WAIT FOR IT..."}


def clip_context(meta, words, env, duration, hop=HOP):
    """Streamer, clip title, the comedy judge's pillars + reason and loudness facts as prompt lines. With the
    transcript alone the 3B model filed 19 of 21 clips as chat_roast (2026-11-14 calibration run)."""
    meta, lines = meta or {}, []
    if meta.get("broadcaster_name") or meta.get("title"):
        lines.append(f"Streamer: {meta.get('broadcaster_name') or 'unknown'}. Clip title: {meta.get('title') or ''}")
    p = meta.get("pillars") or {}
    if p:
        lines.append("Comedy judge (0-10): " + ", ".join(f"{k} {p[k]}" for k in
                     ("hook_strength", "comedy_payoff", "self_contained", "quotability") if k in p)
                     + (f". Judge said: {meta['judge_reason']}" if meta.get("judge_reason") else ""))
    if env.size:
        loud = [q for q in peaks(env, hop) if env_at(env, q) >= 0.8]
        top = float(np.argmax(env)) * hop
        lines.append(f"Audio: {len(loud)} loud peak(s), the loudest at {top / max(duration, 1):.0%} of the clip, "
                     f"{len(words) / max(duration, 1):.1f} words per second.")
    return "\n".join(lines)


def ollama_creative(words, duration, model, url="http://localhost:11434", timeout=None, opener=None, context=""):
    """Title / hashtags / banner from local Ollama, validated, plus its 0-10 fit score per scenario ("fit", advisory:
    recorded in the EDL so ratings can calibrate it, never used to pick). None when offline, too slow
    (scorer.OLLAMA_TIMEOUT + one retry) or nothing usable came back. context: clip_context() lines."""
    timed = " ".join(f"[{w['start']:.1f}]{w['word']}" for w in words)[:6000]
    try:
        d = scorer.ollama_generate(PROMPT + f"\nClip length: {duration:.1f}s\n{context}\nTranscript: {timed}", model, url,
                                   timeout or scorer.OLLAMA_TIMEOUT, temperature=0.2, opener=opener)
        fit = d.get("fit") if isinstance(d.get("fit"), dict) else {}
        fit = {s: fit[s] for s in SCENARIOS if isinstance(fit.get(s), (int, float)) and not isinstance(fit[s], bool)}
        tags = [t if t.startswith("#") else "#" + t for t in d.get("hashtags", []) if isinstance(t, str) and t.strip()]
        tags = [t for t in tags if len(t.strip()) > 3 and not re.fullmatch(r"#(tag\d*|hashtag\d*|[a-d])", t.strip(), re.I)]  # no echoed placeholders
        out = {"social_title": str(d.get("social_title") or "")[:59].strip() or None,
               "hashtags": [re.sub(r"\s+", "", t) for t in tags][:4] or None,
               "banner_text": str(d.get("banner_text") or "")[:24].strip().upper() or None}
        return dict(out, fit=fit or None) if any(out.values()) else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


# --- EDL ----------------------------------------------------------------------------------------------------------

SNAP_CLIMAX, SNAP_PEAK, SNAP_LINE = 1.35, 1.3, 1.22  # snap punch-ins: the climax, a big yell, a "!" / "?" line
RAMP_SCALE, RAMP_LEN, STILL_MAX = 1.12, 3.5, 3.0  # slow push-ins wherever the picture would sit still for > 3 s
SNAP_GAP = 2.0  # output s between two snap starts
SHAKE = 0.85  # a vocal peak this loud (vs the clip's loudest) gets a 0.25 s micro shake; such words get cyan captions
WHOOSH_DB, THUD_DB, SCRATCH_DB = -16.0, -14.0, -14.0  # SFX under the voice (the whole mix is loudnormed after)
TEASER_LEN, TEASER_MIN, TEASER_MIN_CLIP, TEASER_ZOOM = 1.4, 1.2, 16.0, 1.25


def yell_level(env):
    """Loud enough to be a yell: >= SHAKE of the peak AND >= 2.5x the median (a constant game / music bed is no yell)."""
    return max(SHAKE, 2.5 * float(np.median(env))) if len(env) else 1.0


def out_map(cuts):
    """(source -> output s, output -> source s, output length) over plain cuts.
    shortcut: ignores speed ramps and the teaser (they shift a beat by a fraction of a second at most)."""
    offs, acc = [], 0.0
    for s, e in cuts:
        offs.append(acc)
        acc += e - s
    out = lambda t: next((o + t - s for (s, e), o in zip(cuts, offs) if s <= t <= e), None)
    src = lambda t: next((s + t - o for (s, e), o in zip(cuts, offs) if o <= t <= o + e - s), None)
    return out, src, acc


def choreograph(edl, words, env, climax, hop=HOP):
    """The retention camera. Snap punch-ins on the climax (1.35x), every big yell (1.3x) and every "!" / "?" line
    (1.22x), held until the next sentence starts (0.6-2.5 s), each with a soft whoosh; slow 1.0 -> 1.12x push-ins
    over 3.5 s wherever the picture would otherwise sit still for more than 3 s. Zooms never overlap; the renderer
    caps every one at the layout's head-safe zoom (framing.zoom_room)."""
    out, src, length = out_map([(c["start"], c["end"]) for c in edl["cuts"]])
    sentence = [o for o in (out(b["start"]) for a, b in zip(words, words[1:]) if a["word"].rstrip()[-1:] in ".!?") if o is not None]
    cand = ([(climax, SNAP_CLIMAX)] if climax is not None else []) \
        + [(q, SNAP_PEAK) for q in peaks(env, hop) if env_at(env, q, hop) >= yell_level(env)] \
        + [(w["start"], SNAP_LINE) for w in words if w["word"].rstrip()[-1:] in "!?"]  # priority order
    starts = []
    for t, scale in cand:
        o = out(t)
        if o is not None and o < length - 0.6 and all(abs(o - a) >= SNAP_GAP for a, _ in starts):
            starts.append((o, scale))
    starts.sort()
    snaps = []
    for k, (o, scale) in enumerate(starts):
        nxt = starts[k + 1][0] - 0.2 if k + 1 < len(starts) else length
        hold = next((s for s in sentence if s > o + 0.3), o + 1.5) - o  # snap back on the next sentence
        snaps.append((o, min(2.5, max(0.6, hold), nxt - o, length - o), scale))
    zooms, a = [], 0.0
    for o, d, scale in snaps + [(length, 0.0, 0.0)]:
        while o - a > STILL_MAX:  # push in slowly through the still stretch before this snap
            s0 = a + 0.3 if a else a
            zooms.append((s0, min(RAMP_LEN, o - s0 - 0.3), RAMP_SCALE, "creep"))
            a = s0 + zooms[-1][1]
        if d > 0:
            zooms.append((o, d, scale, "snap"))
            edl["sfx"].append({"at": round(src(o), 2), "name": "whoosh", "gain_db": WHOOSH_DB})
            a = max(a, o + d)
    edl["zooms"] += [{"at": round(src(o), 2), "duration": round(d, 2), "scale": s, "style": st} for o, d, s, st in zooms]


def build_edl(scenario, words, env, duration, hop=HOP, speech=()):
    """Deterministic cuts / zooms / filters / takeovers / mutes / SFX for a scenario (all times in source seconds,
    except the teaser's own effects: "out": True = output seconds, applied only while the teaser plays)."""
    pk = peaks(env, hop) or [duration / 2]
    p = pk[0]  # loudest moment
    keep = lambda hi, gap=GAP: keep_spans(0.0, hi, words, speech, env, gap)  # opens on the first sound, no dead air
    last_word_end = words[-1]["end"] if words else duration
    edl = {"scenario": scenario, "hook_teaser": {"enabled": False, "start": 0.0, "end": 0.0, "banner_text": ""},
           "cuts": [], "zooms": [], "filters": [], "takeovers": [], "mutes": [], "sfx": [], "speed": [],
           "bg_music": {"enabled": False, "gain_db": -22.0}}
    fx = lambda a, b, kind: edl["filters"].append({"start": round(max(0.0, a), 2), "end": round(min(duration, b), 2), "type": kind})
    sfx = lambda at, name, gain: edl["sfx"].append({"at": round(max(0.0, at), 2), "name": name, "gain_db": gain})
    pivot = None

    if scenario == "chat_roast":  # calm setup -> pause + scratch -> reaction boost on the outburst
        # The clip runs to its LAST peak (stories keep going), but the outburst = the LOUDEST peak after the setup.
        # Batch 1: the latest peak was often a mild one (0.60-0.78), so the scratch / teaser hit "nothing
        # happening" while the real yell came earlier ("that yell should've been the intro teaser").
        late = max([q for q in pk if q >= duration * 0.3] or [p])
        o = next((q for q in pk if q >= duration * 0.3), p)
        end = min(duration, late + 1.6)
        cuts = keep(end)  # a wordless yell survives as a loud run; no forced end / outburst window (they kept dead air)
        pause = max(cuts[0][0], o - 1.8)
        if env_at(env, o) >= LOUD:  # the scratch only on a real yell: on a quiet "outburst" it looked artificial
            edl["mutes"].append({"start": round(pause, 2), "end": round(pause + 0.35, 2)})
            sfx(pause, "record_scratch", SCRATCH_DB)
        fx(o, o + 0.9, "reaction_boost")
        pivot = o
    elif scenario == "unfiltered_chaos":  # ultra-tight jump cuts, facecam takeovers + glitch on every peak
        cuts = keep(duration, 0.25)
        for q in pk[:6]:
            if not any(s <= q <= e for s, e in cuts):
                cuts.append((max(0.0, q - 0.6), min(duration, q + 0.6)))
        cuts = merge(cuts)
        for q in sorted(pk[:4]):
            edl["takeovers"].append({"start": round(max(0.0, q - 0.3), 2), "end": round(min(duration, q + 1.2), 2)})
            if env_at(env, q) >= LOUD:
                fx(q, q + 0.07, "rgb_split")
    elif scenario == "horror_jumpscare":  # dark vignette in the quiet -> flash + takeover on the scare, cut after it
        end = min(duration, p + SCREAM_CUT)
        cuts = keep(end)
        cuts[-1] = (cuts[-1][0], end)
        fx(cuts[0][0], p, "vignette")
        fx(p, p + 0.07, "flash")  # ~2 frames
        edl["takeovers"].append({"start": round(max(0.0, p - 0.1), 2), "end": round(end, 2)})
        pivot = p
    elif scenario == "confident_fail":  # sped-up confident setup -> scratch at the fail -> grayscale aftermath
        f = p
        end = min(duration, f + 2.2)  # the aftermath: its reactions are kept, its silences cut like everywhere else
        cuts = keep(end)
        edl["speed"].append({"start": round(cuts[0][0], 2), "end": round(f, 2), "factor": 1.15})
        sfx(f, "record_scratch", SCRATCH_DB)
        edl["mutes"].append({"start": round(f + 0.05, 2), "end": round(f + 0.4, 2)})
        fx(f, end, "grayscale")
        pivot = f
    else:  # awkward_irl: unhurried, the deadpan line gets the thud
        cuts = keep(min(duration, last_word_end + 0.6))

    horror = scenario == "horror_jumpscare"
    cuts = merge(cuts) if horror else tail_guard(merge(cuts), env, duration, hop)  # horror's hard cut is the point
    # the floor is for the rendered short: a 1.15x speed ramp plays 8.5 s of source in 7.4 s, so pad by the fastest ramp
    fitted = fit_length(cuts, duration, MIN_LEN * max([sp["factor"] for sp in edl["speed"]] or [1.0]))
    edl["cuts"] = [{"start": s, "end": e} for s, e in word_safe([(c["start"], c["end"]) for c in fitted], words, duration, horror)]
    climax = pivot if pivot is not None else p
    for q in pk:  # micro shake synced to every scream
        if env_at(env, q, hop) >= yell_level(env):
            fx(q, q + 0.25, "trauma_shake")
    sfx(climax, "thud", THUD_DB)  # the low-end hit on the main punchline / deadpan line
    choreograph(edl, words, env, climax, hop)
    # cold open: 1.2-1.5 s of a loud second-half climax first (punched in, whoosh + white flash), then hard cut to the
    # setup. Not for horror: showing the scare first kills the jump.
    if not horror and duration >= TEASER_MIN_CLIP and climax >= 0.5 * duration and env_at(env, climax, hop) >= LOUD \
            and total([(c["start"], c["end"]) for c in edl["cuts"]]) + TEASER_LEN <= MAX_LEN:
        a = max(0.0, climax - 0.5)
        b = min(duration, a + TEASER_LEN)
        home = next((c for c in edl["cuts"] if c["start"] <= climax <= c["end"]), None)
        if home:  # only the climax's own sound, never the silence around it
            a, b = max(a, home["start"]), min(b, home["end"])
        grow = max(0.0, TEASER_MIN - (b - a)) / 2  # a short yell still opens for 1.2 s (<= 0.1 s past it per side)
        a, b = max(0.0, a - grow), min(duration, b + grow)
        if b - a >= TEASER_MIN - 0.01:
            T = round(b - a, 2)
            edl["hook_teaser"] = {"enabled": True, "start": round(a, 2), "end": round(b, 2), "banner_text": "WAIT FOR IT..."}
            edl["zooms"].append({"at": 0.0, "duration": T, "scale": TEASER_ZOOM, "style": "snap", "out": True})
            edl["filters"].append({"start": max(0.0, T - 0.04), "end": T + 0.06, "type": "flash", "out": True})
            edl["sfx"].append({"at": round(max(0.0, T - 0.25), 2), "name": "whoosh", "gain_db": WHOOSH_DB, "out": True})
    if scenario == "confident_fail":
        edl["freeze_end"] = FREEZE_LEN  # hold the last (grey) frame on the fail
    # pre-render sanity: a trim that keeps almost no real words and no loud action moment is fragments, not a short
    kept = [w for w in words if any(c["start"] <= (w["start"] + w["end"]) / 2 <= c["end"] for c in edl["cuts"])]
    clear = [w for w in kept if len(re.sub(r"[^a-z]", "", w["word"].lower())) >= 2]
    spike = bool(env.size) and env.max() / (np.median(env) + 1e-6) >= SPIKE_RATIO
    if len(clear) < MIN_CLEAR_WORDS and not spike:
        edl["skip_render"] = True
        edl["skip_reason"] = f"only {len(clear)} clear word(s) survive the trim and there is no loud action moment"
    edl["word_tones"] = word_tones(words, env, scenario, pivot)
    edl["font"] = FONTS[scenario]
    return edl


def direct(words, audio, duration, sr=16000, model=None, meta=None, framing=None, music=False, opener=None):
    """Full EDL for one clip. The scenario is always the rule-based classify(): on the 40 shorts of the 2026-11-14 run
    the 3B model, however it was asked (name one / judge-aware wording / 0-10 fit per scenario), filed 27-34 of 40
    clips under a single scenario, a different one per wording; classify() used all five (25/8/3/2/2) and its rules
    are code that rating batches can tune. Ollama writes the title / hashtags / banner (model=None or offline ->
    heuristic ones) and its fit scores are kept as scenario_fit for that calibration.
    framing = framing.analyze() (layout / track / source_captions); None = blur_fill, our captions."""
    env = energy(audio, sr)
    scenario = classify(words, env, duration)
    ai = ollama_creative(words, duration, model, opener=opener, context=clip_context(meta, words, env, duration)) \
        if model and words else None
    edl = build_edl(scenario, words, env, duration, speech=vad_speech(audio, sr))
    creative = heuristic_creative(words, scenario, meta)
    fallback_tags = creative["hashtags"]
    creative.update({k: v for k, v in (ai or {}).items() if v and k != "fit"})
    seen = {t.lower() for t in creative["hashtags"]}  # always 4: top up the model's tags with the fallback ones
    creative["hashtags"] = (creative["hashtags"] + [t for t in fallback_tags if t.lower() not in seen])[:4]
    if edl["hook_teaser"]["enabled"]:
        edl["hook_teaser"]["banner_text"] = creative["banner_text"]
    fr = framing or {}
    layout = fr.get("layout") or "blur_fill"
    if layout == "split_stack" and scenario == "awkward_irl":  # awkward IRL keeps clean blur_fill framing (B1)
        layout = "blur_fill"
    # the stream's own captions: show the frame ONCE. A 9:16 column cuts them in half, and split_stack showed them twice
    # (live: Jynxzi's "dropped" in both the webcam crop and the gameplay panel)
    if fr.get("source_captions"):
        layout = "blur_fill"
    edl["layout"], edl["source_captions"] = layout, bool(fr.get("source_captions"))  # theirs on screen: ours off
    edl["crop_track"] = [list(k) for k in fr.get("track") or []] if layout == "face_crop" else []
    # zooms grow from the heads' top and stop where a head (or the stream's caption line) would leave the frame
    edl["zoom_anchor"] = list((fr.get("zoom") if layout == fr.get("layout") else fr.get("zoom_full")) or [0.5, 0.5, 1.35])
    if layout != "split_stack":
        edl["takeovers"] = []  # no webcam to take over with
    edl.update(archetype=scenario, title=creative["social_title"], hashtags=creative["hashtags"][:4],
               social_title=f"{creative['social_title']} {' '.join(creative['hashtags'][:4])}".strip(),
               directed_by="ollama" if ai else "heuristic", scenario_fit=(ai or {}).get("fit"))
    edl["bg_music"]["enabled"] = bool(music)
    return edl

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
         "confident_fail": "Arial Black", "awkward_irl": "Trebuchet MS"}
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


def speech_segments(words, lo, hi, max_gap, pad=0.15):
    """Keep-segments inside [lo, hi]: words grouped while the silence between them is <= max_gap."""
    ws = [w for w in words if w["end"] > lo and w["start"] < hi]
    if not ws:
        return [(lo, hi)]
    segs, cur = [], [ws[0]["start"], ws[0]["end"]]
    for w in ws[1:]:
        if w["start"] - cur[1] > max_gap:
            segs.append(cur)
            cur = [w["start"], w["end"]]
        else:
            cur[1] = max(cur[1], w["end"])
    segs.append(cur)
    return [(round(max(lo, s - pad), 2), round(min(hi, e + pad), 2)) for s, e in segs]


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


TEASER_LEN, TEASER_MIN_CLIP, FREEZE_LEN = 2.2, 24.0, 0.8
LOUD = 0.8  # a peak at >= 80% of the clip's loudest = a real yell / scream
MIN_CLEAR_WORDS, SPIKE_RATIO = 4, 6.0  # skip_render: fewer clear words than this, and no peak >= 6x the median level


def min_holds(cuts, hold=3.0):
    """Merge cuts (and the gap between them) until every kept cut is at least `hold` seconds."""
    cuts = [list(c) for c in cuts]
    i = 0
    while i < len(cuts) and len(cuts) > 1:
        if cuts[i][1] - cuts[i][0] < hold:
            j = i + 1 if i + 1 < len(cuts) else i - 1
            lo, hi = sorted((i, j))
            cuts[lo] = [cuts[lo][0], cuts[hi][1]]
            cuts.pop(hi)
            i = 0
        else:
            i += 1
    return [tuple(c) for c in cuts]


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


def word_tones(words, env, scenario, pivot=None):
    """Caption emphasis tone per word: hype (green) / rage (red) / roast (cyan) / punch (yellow).
    pivot: the fail/scare moment; after it confident_fail turns deadpan and horror turns red."""
    out = []
    for w in words:
        bare = re.sub(r"[^\w']", "", w["word"]).lower()
        loud = env_at(env, (w["start"] + w["end"]) / 2) >= 0.85
        if HYPE_RE.match(bare):
            tone = "hype"
        elif loud or RAGE_RE.match(bare):
            tone = "rage"
        elif ROAST_RE.search(bare):
            tone = "roast"
        else:
            tone = DEFAULT_TONE[scenario]
            if pivot is not None and w["start"] >= pivot:
                tone = {"confident_fail": "roast", "horror_jumpscare": "rage"}.get(scenario, tone)
        out.append(tone)
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

def build_edl(scenario, words, env, duration, hop=HOP):
    """Deterministic cuts / zooms / filters / takeovers / mutes / SFX for a scenario (all times in source seconds)."""
    pk = peaks(env, hop) or [duration / 2]
    p = pk[0]  # loudest moment
    start = max(0.0, (words[0]["start"] if words else 0.0) - 0.25)  # trim leading dead air
    last_word_end = words[-1]["end"] if words else duration
    edl = {"scenario": scenario, "hook_teaser": {"enabled": False, "start": 0.0, "end": 0.0, "banner_text": ""},
           "cuts": [], "zooms": [], "filters": [], "takeovers": [], "mutes": [], "sfx": [], "speed": [],
           "bg_music": {"enabled": False, "gain_db": -22.0}}
    zoom = lambda at, dur, scale, style: edl["zooms"].append(
        {"at": round(max(0.0, at), 2), "duration": round(max(0.1, dur), 2), "scale": scale, "style": style})
    fx = lambda a, b, kind: edl["filters"].append({"start": round(max(0.0, a), 2), "end": round(min(duration, b), 2), "type": kind})
    sfx = lambda at, name, gain: edl["sfx"].append({"at": round(max(0.0, at), 2), "name": name, "gain_db": gain})
    pivot = None

    if scenario == "chat_roast":  # calm setup -> pause + scratch -> ramp zoom + vine boom -> snap + reaction boost on the outburst
        # The clip runs to its LAST peak (stories keep going), but the outburst = the LOUDEST peak after the setup.
        # Batch 1: the latest peak was often a mild one (0.60-0.78), so the scratch / boom / teaser hit "nothing
        # happening" while the real yell came earlier ("that yell should've been the intro teaser").
        late = max([q for q in pk if q >= duration * 0.3] or [p])
        o = next((q for q in pk if q >= duration * 0.3), p)
        end = min(duration, late + 1.6)
        cuts = speech_segments(words, start, end, max_gap=0.7)
        cuts[-1] = (cuts[-1][0], end)
        cuts = merge(cuts + [(max(0.0, o - 0.6), min(duration, o + 0.9))])  # a wordless yell must survive the trim
        pause = max(cuts[0][0], o - 1.8)
        zoom(pause + 0.35, o - pause - 0.35, 1.45, "ramp")
        if env_at(env, o) >= LOUD:  # SFX only on a real yell: on a quiet "outburst" they looked artificial
            edl["mutes"].append({"start": round(pause, 2), "end": round(pause + 0.35, 2)})
            sfx(pause, "record_scratch", -6.0)
            sfx(pause + 0.35, "vine_boom", -5.0)
        zoom(o, 0.9, 1.65, "snap")
        fx(o, o + 0.9, "reaction_boost")
        fx(o, o + 0.3, "trauma_shake")
        pivot = o
    elif scenario == "unfiltered_chaos":  # ultra-tight jump cuts, takeovers + glitch on every peak, layered SFX
        cuts = speech_segments(words, start, duration, max_gap=0.25, pad=0.08)
        for q in pk[:6]:
            if not any(s <= q <= e for s, e in cuts):
                cuts.append((max(0.0, q - 0.6), min(duration, q + 0.6)))
        cuts = merge(cuts)
        for q in sorted(pk[:4]):
            edl["takeovers"].append({"start": round(max(0.0, q - 0.3), 2), "end": round(min(duration, q + 1.2), 2)})
            zoom(q - 0.05, 0.5, 1.4, "snap")
            if env_at(env, q) >= 0.8:
                fx(q, q + 0.25, "trauma_shake")
                fx(q, q + 0.07, "rgb_split")
        sfx(p, "vine_boom", -6.0)
        sfx(p, "bass_boost", -9.0)
        if len(pk) > 1:
            sfx(pk[1], "metal_pipe", -8.0)
    elif scenario == "horror_jumpscare":  # dark creep in the quiet -> flash + shake + takeover on the scare, cut after it
        end = min(duration, p + SCREAM_CUT)
        cuts = speech_segments(words, start, end, max_gap=1.5)
        cuts[-1] = (cuts[-1][0], end)
        fx(cuts[0][0], p, "vignette")
        zoom(cuts[0][0], max(1.0, p - cuts[0][0]), 1.15, "creep")
        fx(p, p + 0.07, "flash")  # ~2 frames
        fx(p, p + 0.35, "trauma_shake")
        edl["takeovers"].append({"start": round(max(0.0, p - 0.1), 2), "end": round(end, 2)})
        pivot = p
    elif scenario == "confident_fail":  # sped-up confident setup -> scratch at the fail -> grayscale + sad violin
        f = p
        end = min(duration, f + 2.2)  # keep the aftermath
        cuts = speech_segments(words, start, f, max_gap=0.6) + [(f, end)]
        cuts = merge(cuts)
        edl["speed"].append({"start": round(cuts[0][0], 2), "end": round(f, 2), "factor": 1.15})
        sfx(f, "record_scratch", -5.0)
        edl["mutes"].append({"start": round(f + 0.05, 2), "end": round(f + 0.4, 2)})
        sfx(f + 0.3, "sad_violin", -9.0)
        fx(f, end, "grayscale")
        zoom(f, end - f, 1.12, "creep")
        pivot = f
    else:  # awkward_irl: long unhurried holds, slow drift, dry SFX only
        cuts = speech_segments(words, start, min(duration, last_word_end + 0.6), max_gap=1.5)
        cuts = min_holds(cuts, 3.0)
        zoom(cuts[0][0], max(1.0, cuts[-1][1] - cuts[0][0]), 1.06, "drift")
        sfx(p, "vine_boom" if env_at(env, p) >= 0.8 else "pop", -9.0)

    horror = scenario == "horror_jumpscare"
    cuts = merge(cuts) if horror else tail_guard(merge(cuts), env, duration, hop)  # horror's hard cut is the point
    # MIN_LEN is for the rendered short: a 1.15x speed ramp plays 8.5 s of source in 7.4 s, so pad by the fastest ramp
    fitted = fit_length(cuts, duration, MIN_LEN * max([sp["factor"] for sp in edl["speed"]] or [1.0]))
    edl["cuts"] = [{"start": s, "end": e} for s, e in word_safe([(c["start"], c["end"]) for c in fitted], words, duration, horror)]
    # flash-forward hook on long clips whose climax comes late (the payoff is far away): 2.2 s of the climax first
    if scenario in ("chat_roast", "confident_fail") and pivot is not None and duration > TEASER_MIN_CLIP \
            and pivot >= 0.65 * duration and total([(c["start"], c["end"]) for c in edl["cuts"]]) + TEASER_LEN <= MAX_LEN:
        a = max(0.0, min(pivot - 1.0, duration - TEASER_LEN))
        edl["hook_teaser"] = {"enabled": True, "start": round(a, 2), "end": round(a + TEASER_LEN, 2), "banner_text": "WAIT FOR IT..."}
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


def direct(words, audio, duration, sr=16000, model=None, meta=None, has_facecam=False, music=False, opener=None):
    """Full EDL for one clip. The scenario is always the rule-based classify(): on the 40 shorts of the 2026-11-14 run
    the 3B model, however it was asked (name one / judge-aware wording / 0-10 fit per scenario), filed 27-34 of 40
    clips under a single scenario, a different one per wording; classify() used all five (25/8/3/2/2) and its rules
    are code that rating batches can tune. Ollama writes the title / hashtags / banner (model=None or offline ->
    heuristic ones) and its fit scores are kept as scenario_fit for that calibration."""
    env = energy(audio, sr)
    scenario = classify(words, env, duration)
    ai = ollama_creative(words, duration, model, opener=opener, context=clip_context(meta, words, env, duration)) \
        if model and words else None
    edl = build_edl(scenario, words, env, duration)
    creative = heuristic_creative(words, scenario, meta)
    fallback_tags = creative["hashtags"]
    creative.update({k: v for k, v in (ai or {}).items() if v and k != "fit"})
    seen = {t.lower() for t in creative["hashtags"]}  # always 4: top up the model's tags with the fallback ones
    creative["hashtags"] = (creative["hashtags"] + [t for t in fallback_tags if t.lower() not in seen])[:4]
    if edl["hook_teaser"]["enabled"]:
        edl["hook_teaser"]["banner_text"] = creative["banner_text"]
    # awkward IRL keeps clean blur_fill framing; everything else stacks the facecam when there is one
    edl["layout"] = "split_stack" if has_facecam and scenario != "awkward_irl" else "blur_fill"
    if edl["layout"] != "split_stack":
        edl["takeovers"] = []  # no webcam to take over with
    edl.update(archetype=scenario, title=creative["social_title"], hashtags=creative["hashtags"][:4],
               social_title=f"{creative['social_title']} {' '.join(creative['hashtags'][:4])}".strip(),
               directed_by="ollama" if ai else "heuristic", scenario_fit=(ai or {}).get("fit"))
    edl["bg_music"]["enabled"] = bool(music)
    return edl

"""Highlight scoring. Pure functions (plus optional local Ollama calls) so they are easy to test.

Every signal becomes a per-second 0..1 curve (index = second of the video); peaks of the blended curve
become candidate windows that are sized around the spike instead of using a fixed length.
"""
import json
import math
import os
import re
import subprocess
import sys
import threading
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

LAUGH_RE = re.compile(r"\[(laughter|laughs|laughing)\]|\((laughter|laughs|laughing)\)|\b(ha){2,}\b|\bhehe+\b|\blol\b|\blmao\b", re.I)
REACTION_RE = re.compile(r"\b(oh my god|omg|no way|what the|wtf|holy|bruh|dude|wait what|are you serious)\b", re.I)
CALLBACK_RE = re.compile(r"\b(remember (that|this|when|him|her|the)|(he|she|they)('s| is| are) back|"
                         r"same (dude|guy|girl|car|person|one)|told you|like last time|(this|that) guy again)\b", re.I)
WORD_RE = re.compile(r"[a-z][a-z']{3,}")
STOP = frozenset("""that this with have what just like know yeah your from they them there then were been will would
could should about really because going gonna wanna want okay right well think here when where which their into over
some only also back come came make made take time good look said says dont didnt doesnt cant wont thats it's what's
i'm you're he's she's they're we're that's there's let's i'll you'll can't don't didn't doesn't won't isn't aren't
wasn't other more much very even still thing things something nothing anything everything people guys stuff shit
fuck fucking literally actually maybe sure need does doing done went getting give gave thank thanks hello please
these those tell told never always every first last little long great before after again away""".split())
# Final score = 50% comedy (transcript cues + Ollama judge) + 35% signal (replay heatmap / audio energy, averaged
# with vocal laughter bursts) + 15% visual action (a tie-breaker, so menu flashes and driving don't outrank dialogue).
W_COMEDY, W_SIGNAL, W_VISUAL = 0.5, 0.35, 0.15
# Whisper vocabulary hint + deterministic fixes for what it still mishears (one table for lines and word lists).
# scope: the fix only applies when the clip's context (streamer login, title, game) mentions it ("queso" is a real
# word everywhere else). workspace/custom_vocab.json adds your own: {"prompt_terms": [...], "fixes": {"wrong": "Right"}}.
GAMING_PROMPT = "Fortnite, Twitch, Jynxzi, CaseOh, Kai Cenat, IShowSpeed, Valorant, Minecraft, GTA, bro, chat, clip it."
GAMING_FIXES = [(("fortnight",), "Fortnite", None), (("jinxie",), "Jynxzi", None), (("jinx", "see"), "Jynxzi", None),
                (("case", "oh"), "CaseOh", None), (("queso",), "CaseOh", "caseoh")]
VOCAB_FILE = Path(__file__).resolve().parent.parent / "workspace" / "custom_vocab.json"


def custom_vocab(path=VOCAB_FILE):
    """(prompt terms, {wrong: right}) from custom_vocab.json; ([], {}) when missing or malformed (never fatal)."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return [], {}
    d = d if isinstance(d, dict) else {}
    terms = d.get("prompt_terms") if isinstance(d.get("prompt_terms"), list) else []
    fixes = d.get("fixes") if isinstance(d.get("fixes"), dict) else {}
    return [str(t).strip() for t in terms if str(t).strip()], {str(k): str(v) for k, v in fixes.items() if str(v).strip()}


def whisper_prompt(path=VOCAB_FILE):
    terms = custom_vocab(path)[0]
    return GAMING_PROMPT[:-1] + "".join(", " + t for t in terms) + "." if terms else GAMING_PROMPT


def gaming_fixes(context="", path=VOCAB_FILE):
    """[(tokens, replacement)] active for one clip: your custom_vocab.json fixes first (they win), then the built-ins
    whose scope (if any) the context mentions."""
    ctx = context.lower()
    custom = [(tuple(re.sub(r"[^a-z0-9 ]", "", wrong.lower()).split()), right) for wrong, right in custom_vocab(path)[1].items()]
    return [(t, r) for t, r in custom if t] + [(t, r) for t, r, scope in GAMING_FIXES if not scope or scope in ctx]


def fix_gaming_text(text, context="", path=VOCAB_FILE):
    for toks, rep in gaming_fixes(context, path):
        text = re.sub(r"\b" + r"[\s,]+".join(map(re.escape, toks)) + r"\b", lambda m, r=rep: r, text, flags=re.I)
    return text


def fix_gaming_words(words, context="", path=VOCAB_FILE):
    """[{word, start, end}] with misheard names replaced; a two-word miss ("case oh") becomes one word that
    spans both timestamps. Trailing punctuation of the last replaced word is kept."""
    fixes = gaming_fixes(context, path)
    out, i = [], 0
    norm = lambda w: re.sub(r"[^a-z0-9]", "", w["word"].lower())
    while i < len(words):
        for toks, rep in fixes:
            span = words[i:i + len(toks)]
            if len(span) == len(toks) and all(norm(w) == t for w, t in zip(span, toks)):
                tail = re.search(r"[^\w]*$", span[-1]["word"]).group()
                out.append({**span[0], "word": rep + tail, "end": span[-1]["end"]})
                i += len(toks)
                break
        else:
            out.append(words[i])
            i += 1
    return out
# Calibrated live on the 3B judge (2026-11-13). Small models copy any number they see near a field, so the prompt has
# no example values and no "score it 3 or lower" phrasing (that made self_contained=3 on nearly every clip); just one
# shared 1-3 / 4-6 / 7-10 scale. Payoff is asked first: it is what makes a short.
# Rubric = calibration_studio/clip_rate/TASTE_PROFILE.md (Core Comedy + Hard Reject sections). Batch 1 (2026-11-14):
# r vs the human 0.08 for this wording vs -0.03 for the old one on a fresh run (stored run 0.14): run-to-run noise is
# about +-0.15, so the 3B judge reading only the transcript cannot rank by this taste yet; kept because it states it.
JUDGE_PROMPT = """You are an editor picking viral vertical shorts (TikTok / YouTube Shorts) of streamers.
The viewer watches for the STREAMER'S PERSONALITY: their reactions, voice and goofy commentary. A classic punchline is
not required. Score every candidate clip on four pillars, each a whole number from 0 to 10:
- comedy_payoff: does the streamer deliver? A scream, rage or panic outburst, laughing at something dumb, an instant-karma
  fail, or committed goofy commentary (a ridiculous theory, story or rant told with personality) all pay off.
  Low: nothing happens, long silence, the streamer gives no real reaction, repeats the same line, or someone else talks
  while the streamer stays quiet.
- hook_strength: does the opening (the "opening" text = the first 2.5 seconds) grab attention: a yell, a weird
  statement, a strange situation? Low: greetings, small talk, menus.
- self_contained: would a stranger scrolling with zero stream context enjoy it? A clip can make little sense and still
  work when the reaction is big. Low: an award, a name or an event that only matters with context, and no reaction.
- quotability: are the spoken lines funny, punchy or absurd on their own?
One scale for every pillar: 1-3 = dead air, nothing happens, pointless talk; 4-6 = average banter; 7-10 = a big reaction,
hilarious goofy commentary or a clear punchline.
Judge each clip on its own: different clips deserve different scores, and different pillars of one clip can differ.
Also give "reason": one short sentence saying why it works or why it doesn't.
replay/vocal/visual are 0-1 signal strengths (audience replays, laughter/yelling, on-screen chaos).
Reply with JSON only: an object with key "clips", a list with one object per candidate, each having the keys
index, comedy_payoff, hook_strength, self_contained, quotability, reason.

Candidates:
"""
CALLBACK_PROMPT = """These are candidate clips from ONE video, in time order. A callback is when a later clip clearly refers back
to something set up in an earlier clip: the same person, object, running joke or phrase.
Reply with JSON only: an object with key "callbacks", a list of objects with the keys later and setup (clip indexes).
Use an empty list when there are none.

Clips:
"""
PILLARS = {"self_contained": 0.35, "comedy_payoff": 0.35, "hook_strength": 0.20, "quotability": 0.10}
# hard reject only pure dead air: no payoff AND no hook. Out-of-context clips are not vetoed (a 9/10 rage outburst is
# viral mid-conversation; the editor's teaser/trim add context); a low self_contained just lowers the weighted score,
# which the Min Comedy Score gate then judges.
MIN_PAYOFF, MIN_HOOK = 5, 5
# Twitch clips the community already proved (views) with a big vocal reaction get a small nudge, so a 3B model's one
# bad pillar can't throw away an obvious banger. +0.05 = half a point on the 0-10 gate.
VIRAL_VIEWS, VIRAL_VOCAL, VIRAL_BOOST = 3000, 0.5, 0.05
# Taste-profile streamer prior (Twitch login -> + on the 0-1 judge score). Batch 1: CaseOh rated 8.2 vs 6.3 for the
# rest; +0.1 raised human-vs-AI r by +0.11..0.13 on all three judge runs (stored, old prompt, new prompt).
STREAMER_BOOST = {"caseoh_": 0.1}

# the calibration studio's tuner saves overrides for these to backend/weights.json (tracked, so both machines get them)
TUNABLE = ("PILLARS", "MIN_PAYOFF", "MIN_HOOK", "VIRAL_VIEWS", "VIRAL_VOCAL", "VIRAL_BOOST", "W_COMEDY", "W_SIGNAL", "W_VISUAL")
DEFAULTS = {k: dict(v) if isinstance(v, dict) else v for k, v in globals().items() if k in TUNABLE}
WEIGHTS_FILE = Path(__file__).resolve().parent / "weights.json"


def _weight_ok(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0


def load_weights(path=None):
    """Apply weights.json over the defaults (module globals) -> the overrides used. Unknown keys, a dict with other
    keys, or a non-number / negative value are ignored, so a bad file can never break scoring."""
    try:
        data = json.loads(Path(path or WEIGHTS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    g, used = globals(), {}
    for k in TUNABLE:
        g[k] = dict(DEFAULTS[k]) if isinstance(DEFAULTS[k], dict) else DEFAULTS[k]
        v = data.get(k) if isinstance(data, dict) else None
        if isinstance(DEFAULTS[k], dict):
            ok = isinstance(v, dict) and set(v) == set(DEFAULTS[k]) and all(_weight_ok(x) for x in v.values())
        else:
            ok = _weight_ok(v)
        if ok:
            g[k] = dict(v) if isinstance(v, dict) else v
            used[k] = v
    return used


load_weights()


from history import jaccard  # noqa: E402,F401  (one copy, stdlib-only for the publisher; clipper uses scorer.jaccard)


def clock(sec):
    s = int(sec)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def local_median(x, k):
    x = np.asarray(x, dtype=np.float64)
    return np.array([np.median(x[max(0, i - k):i + k + 1]) for i in range(len(x))])


def heatmap_curve(heatmap, duration, baseline_sec=40):
    """yt-dlp 'heatmap' -> (per-second 0..1 prominence curve, per-second reason).
    Prominence = how far a segment spikes above the median of the surrounding ~80 s, so a sharp spike beats a
    high flat stretch. The first 5% of segments (everyone watches the intro) is zeroed."""
    vals = np.array([float(h["value"]) for h in heatmap])
    seg = max(1e-6, heatmap[0]["end_time"] - heatmap[0]["start_time"])
    base = local_median(vals, max(2, round(baseline_sec / seg)))
    prom = np.maximum(0.0, vals - base)
    prom[:max(1, len(vals) // 20)] = 0.0
    prom /= prom.max() if prom.max() > 0 else 1.0
    mids = [(h["start_time"] + h["end_time"]) / 2 for h in heatmap]
    secs = np.arange(int(np.ceil(duration))) + 0.5
    curve = np.interp(secs, mids, prom)  # linear between segment midpoints, so peaks sit at segment centres
    why = [f"Most Replayed {v / max(b, 0.01):.1f}x spike @ {clock(m)} | Peak {v:.2f}" for v, b, m in zip(vals, base, mids)]
    reasons = [why[min(len(why) - 1, int(s // seg))] for s in secs]
    return curve, reasons


def energy_curve(rms):
    """Per-second audio RMS -> (0..1 curve from loudness z-score plus sudden jumps, per-second reason)."""
    rms = np.asarray(rms, dtype=np.float64)
    if rms.size == 0:
        return np.zeros(0), []
    std = rms.std() or 1.0
    z = (rms - rms.mean()) / std
    jump = np.maximum(0.0, np.diff(rms, prepend=rms[0])) / std
    raw = np.maximum(0.0, 0.6 * z + 0.4 * jump)
    return raw / (raw.max() or 1.0), [f"Audio spike z={v:.1f} @ {clock(i)}" for i, v in enumerate(z)]


def visual_curve(frames, fps=2, baseline_sec=20):
    """Low-res grayscale frames (n, h, w) uint8 at `fps` -> per-second 0..1 visual action curve.
    Motion = mean absolute frame difference (chaos, crashes, hard cuts); flash = jump in mean brightness
    (explosions, Wasted screens). Scored as prominence over the local ~40 s median, because gameplay is
    always moving and only *sudden* action should count."""
    if len(frames) < 2:
        return np.zeros(0)
    # shortcut: int16 copy of all frames (~60 MB per hour at 160x90); chunk it if multi-hour VODs matter.
    f = frames.astype(np.int16)
    motion = np.abs(f[1:] - f[:-1]).mean(axis=(1, 2))
    flash = np.abs(np.diff(f.mean(axis=(1, 2))))
    per_frame = np.concatenate([[0.0], motion + 2 * flash])
    per_sec = per_frame[:len(per_frame) // fps * fps].reshape(-1, fps).max(axis=1)
    prom = np.maximum(0.0, per_sec - local_median(per_sec, baseline_sec))
    return prom / (prom.max() if prom.max() > 0 else 1.0)


def vocal_curve(audio, sr=16000, baseline_sec=20):
    """16 kHz mono float32 audio -> per-second 0..1 vocal excitement curve (laughing, wheezing, yelling).
    Two cues in the 300-3400 Hz voice band, both measured against the local ~40 s median:
    a sudden burst of voice-band energy, and rapid frame-to-frame flutter of that energy (the staccato
    "ha-ha-ha" / yell envelope). Multiplied by how voice-heavy the sound is, so steady engine rumble, bass and
    sustained music score ~0. It can't tell a laugh from loud excited speech; both count as comedy signal here."""
    win, hop = 512, sr // 50  # 32 ms frames every 20 ms -> 50 frames per second
    n = (len(audio) - win) // hop + 1
    secs = max(0, n // 50)
    if secs < 2:
        return np.zeros(len(audio) // sr)
    band = slice(int(300 * win / sr), int(3400 * win / sr) + 1)
    w = np.hanning(win).astype(np.float32)
    vocal, total = np.empty(n, np.float32), np.empty(n, np.float32)
    for i in range(0, n, 3000):  # 60 s of frames at a time keeps memory flat on long videos
        idx = np.arange(i, min(n, i + 3000))[:, None] * hop + np.arange(win)
        spec = np.abs(np.fft.rfft(audio[idx] * w, axis=1)) ** 2
        vocal[i:i + len(spec)] = spec[:, band].sum(axis=1)
        total[i:i + len(spec)] = spec.sum(axis=1) + 1e-9
    env = np.log10(vocal[:secs * 50] + 1e-9).reshape(secs, 50)
    voiced = (vocal / total)[:secs * 50].reshape(secs, 50).mean(axis=1)
    level = env.mean(axis=1)
    flutter = np.abs(np.diff(env, axis=1)).mean(axis=1)
    norm = lambda x: x / (x.max() if x.max() > 0 else 1.0)
    burst = norm(np.maximum(0.0, level - local_median(level, baseline_sec)))
    flut = norm(np.maximum(0.0, flutter - local_median(flutter, baseline_sec)))
    # flutter multiplies the burst: a loud but steady sound (music swell) keeps only a quarter of its burst
    return norm(burst * (0.25 + flut) * voiced)


def fit(curve, n):
    """Pad/trim an optional per-second curve to n seconds (zeros when missing)."""
    out = np.zeros(n)
    if curve is not None:
        out[:min(n, len(curve))] = curve[:n]
    return out


def signal_curves(primary, reasons, vocal=None, visual=None):
    """-> (signal, visual, ranking curve, reasons). signal = replay heatmap (or audio energy) averaged with vocal
    laughter bursts; the ranking curve weighs signal vs visual 35:15 like the final score."""
    n = len(primary)
    voc, vis = fit(vocal, n), fit(visual, n)
    signal = (primary + voc) / 2 if vocal is not None else primary
    rank = (W_SIGNAL * signal + W_VISUAL * vis) / (W_SIGNAL + W_VISUAL)
    tags = [r + (f" + vocal burst {a:.2f}" if a >= 0.5 else "") + (f" + visual action {b:.2f}" if b >= 0.5 else "")
            for r, a, b in zip(reasons, voc, vis)]
    return signal, vis, rank, tags


def final_score(cue, signal, visual, llm=None):
    """50% comedy (Ollama judge when available, nudged by transcript cues) + 35% signal + 15% visual.
    visual None = never measured (whole Twitch clips and uploads skip the visual scan): comedy and signal are
    scaled up to the full weight instead of counting a missing scan as zero action."""
    comedy = 0.7 * llm + 0.3 * cue if llm is not None else cue
    if visual is None:
        return round((W_COMEDY * comedy + W_SIGNAL * signal) * (W_COMEDY + W_SIGNAL + W_VISUAL) / (W_COMEDY + W_SIGNAL), 3)
    return round(W_COMEDY * comedy + W_SIGNAL * signal + W_VISUAL * visual, 3)


def apply_verdict(c, v):
    """Copy one ollama_judge() verdict onto a candidate (the YouTube and Twitch paths share this)."""
    c.update(llm=v["llm"], setup=v["setup"], judge_reject=v["reject"], judge_reason=v["reason"], pillars=v["pillars"])
    c["reason"] += f" + LLM {v['llm'] * 10:.1f}/10" + (" (rejected)" if v["reject"] else "")


def rejected(cands):
    """Manifest entries for the candidates the judge hard-rejected: where, and why."""
    return [{"start_sec": c["start_sec"], "end_sec": c["end_sec"], "llm": c["llm"], "pillars": c["pillars"],
             "judge_reason": c["judge_reason"]} for c in cands if c.get("judge_reject")]


def judge_summary(video_id, cands, min_score):
    """One line for when no candidate of a video passed: what the BEST one scored and which rule stopped it.
    None when something passed or nothing was judged (Ollama off: unjudged clips always pass)."""
    judged = [c for c in cands if c.get("llm") is not None]
    if not judged or any(passes_gate(c, min_score) for c in cands):
        return None
    b = max(judged, key=lambda c: c["llm"])
    p = b["pillars"]
    why = (f"payoff={p['comedy_payoff']:g}, hook={p['hook_strength']:g}: dead air (needs payoff >= {MIN_PAYOFF} "
           f"or hook >= {MIN_HOOK})" if b.get("judge_reject")
           else f"score {b['llm'] * 10:.1f}/10 < Min Comedy Score {min_score:g}")
    return (f"[JUDGE] Rejected all {len(cands)} candidate(s) for {video_id} (best: {why})"
            + (f": {b['judge_reason']}" if b.get("judge_reason") else ""))


def viral_boost(c, views):
    """Proven Twitch banger (>= VIRAL_VIEWS views and a vocal burst >= VIRAL_VOCAL): +VIRAL_BOOST on the judge score.
    -> True when applied. Never undoes a dead-air reject."""
    if c.get("llm") is None or (views or 0) < VIRAL_VIEWS or c.get("vocal", 0) < VIRAL_VOCAL:
        return False
    c["llm"] = round(min(1.0, c["llm"] + VIRAL_BOOST), 3)
    c["reason"] += f" + viral boost ({views} views)"
    return True


def streamer_boost(c, login):
    """TASTE_PROFILE streamer prior: +STREAMER_BOOST[login] on the judge score. -> True when applied. Like the viral
    boost it never undoes a dead-air reject."""
    b = STREAMER_BOOST.get(str(login or "").strip().lstrip("@").lower())
    if c.get("llm") is None or not b:
        return False
    c["llm"] = round(min(1.0, c["llm"] + b), 3)
    c["reason"] += f" + streamer prior ({login})"
    return True


def passes_gate(c, min_score):
    """Banger gate: never a clip the judge hard-rejected (out of context / no payoff), and the judge's weighted
    rating (0..10) must reach min_score. Unjudged clips (Ollama off) pass."""
    return not c.get("judge_reject") and (c.get("llm") is None or c["llm"] * 10 >= min_score)


def rank_candidates(cands, n, min_score=0.0):
    """Score every candidate on comedy signals, then keep the best n (this is where banter loses).
    Line snapping can push neighbouring windows into each other, so overlaps are re-checked here."""
    for c in cands:
        c["score"] = final_score(c["cue"], c["signal"], c["visual"], c.get("llm"))
    best = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        if passes_gate(c, min_score) and all(c["end_sec"] <= b["start_sec"] or c["start_sec"] >= b["end_sec"] for b in best):
            best.append(c)
            if len(best) == n:
                break
    return best


def size_window(curve, t, min_len, max_len, pre=5, post=3):
    """Window around peak second t sized by the spike itself: grow while the curve stays above 35% of the peak
    (buildup before, decay after), add `pre` s of setup and `post` s of reaction, then clamp into
    [min_len, max_len]. A clamped window is re-centred with 2/3 before the peak, because viewers rewind
    *after* the punchline, so replay/laugh peaks trail it by a few seconds."""
    n = len(curve)
    floor = curve[t] * 0.35
    lo = hi = t
    while lo > 0 and curve[lo - 1] >= floor and t - lo < max_len:
        lo -= 1
    while hi < n - 1 and curve[hi + 1] >= floor and hi - t < max_len:
        hi += 1
    start, end = lo - pre, hi + 1 + post
    length = min(max(end - start, min_len), max_len, n)
    if end - start != length:
        start = t + 0.5 - length * 2 / 3
    start = round(float(min(max(0.0, start), n - length)), 2)
    return start, round(start + length, 2)


def pick_windows(curve, reasons, n, min_len, max_len):
    """Top-n non-overlapping (start, end, score, reason) windows, best first, each sized by size_window."""
    chosen = []
    for t in np.argsort(-curve, kind="stable"):
        s, e = size_window(curve, int(t), min_len, max_len)
        if all(e <= cs or s >= ce for cs, ce, _, _ in chosen):
            chosen.append((s, e, float(curve[t]), reasons[t]))
            if len(chosen) == n:
                break
    return chosen


def transcript_between(cues, start, end):
    return " ".join(t for s, e, t in cues if e > start and s < end)


def snap_window(start, end, cues, max_shift=3.0, min_len=0.0, max_len=float("inf")):
    """Nudge start to the nearest line start and end to the nearest line end within ±max_shift s, so clips
    open on a spoken line instead of mid-word. Keeps the original window if snapping breaks the length bounds."""
    near = lambda pts, x: min((p for p in pts if abs(p - x) <= max_shift), key=lambda p: abs(p - x), default=x)
    s, e = near([c[0] for c in cues], start), near([c[1] for c in cues], end)
    return (round(s, 2), round(e, 2)) if max(1.0, min_len) <= e - s <= max_len else (start, end)


def find_callback(cues, payoff_start, payoff_text, min_gap=45, need=None):
    """Earlier transcript line (>= min_gap s before the payoff) that the payoff calls back to -> (start, end, shared
    words), or None. Strong matches only: >= 2 shared rare words (<= 3 uses in the whole video, not stopwords),
    or a callback phrase ("remember that guy", "he's back", ...) plus >= 1 shared rare word. `need` overrides
    the word count (1 when the Ollama judge already flagged the pair)."""
    words = lambda text: WORD_RE.findall(text.lower())
    counts = Counter(w for *_, t in cues for w in words(t))
    pay = {w for w in words(payoff_text) if w not in STOP and counts[w] <= 3}
    need = need or (1 if CALLBACK_RE.search(payoff_text) else 2)
    best = None
    for s, e, t in cues:
        if s > payoff_start - min_gap:
            break
        shared = pay & set(words(t))
        if len(shared) >= need and (best is None or len(shared) > len(best[2])):
            best = (s, e, sorted(shared))
    return best


def stitch_plan(keyframes, mention, payoff, max_len):
    """Plan [setup] + [payoff] cut points for stream copy, which can only start on a keyframe.
    Setup starts on the keyframe before the mention and runs 6-10 s; the payoff's pre-roll is trimmed (to a
    later keyframe) so the total stays <= max_len. Returns ((setup_s, setup_e), (payoff_s, payoff_e)) or None."""
    ms, me = mention
    ks = max((k for k in keyframes if k <= ms), default=0.0)
    if ms - ks > 6:
        return None  # the mention would land too deep in the setup for a tight 6-10 s part
    se = min(max(me + 1, ks + 6), ks + 10)
    ps, pe = payoff
    budget = max_len - (se - ks)
    lo = max(ps, pe - budget)
    kp = max((k for k in keyframes if k <= lo), default=0.0)
    if pe - kp > budget:
        kp = min((k for k in keyframes if k >= lo), default=pe)
    if pe - kp < 5:
        return None  # no room left for the payoff itself
    return (round(ks, 2), round(se, 2)), (round(kp, 2), round(pe, 2))


def cue_score(text):
    """Comedy/reaction cues in a transcript -> (0..1 score, short tag for the reason string)."""
    laughs = len(LAUGH_RE.findall(text))
    reactions = len(REACTION_RE.findall(text))
    words = max(1, len(text.split()))
    bangs = text.count("!") + text.count("?!")
    score = min(1.0, 0.25 * laughs + 0.15 * reactions + 3.0 * bangs / words)
    tags = [f"{laughs} laugh" if laughs else "", f"{reactions} reaction" if reactions else "", f"{bangs}!" if bangs else ""]
    return score, " ".join(t for t in tags if t)


def opening(c, seconds=2.5):
    """What is said in a candidate's first `seconds`: its timed Whisper lines, else ~3 words per second."""
    if c.get("lines"):
        return " ".join(t for s, e, t in c["lines"] if c["start_sec"] - 0.5 <= s < c["start_sec"] + seconds)
    return " ".join(c["transcript"].split()[:int(3 * seconds)])


# TOC_LOW_SPEC=1 (set by calibration_studio/clip_rate_worker.py): Whisper and Ollama never share VRAM (on the 4 GB
# GTX 1050 Ti both at once can hard-lock the display driver). Whisper runs on the CPU (int8, 4 threads: ~18 s + ~6 s
# load for a 30 s clip on the i7-860, 0 MB VRAM) so Ollama can stay warm in VRAM (keep_alive 5m: 0.45 s per call
# instead of a 12-32 s reload each time; measured 2026-11-14, 2.3 GB). The worker unloads it when it exits or is
# cancelled; the editor unloads it before a render when VRAM_RENDER_MIB isn't free. ffmpeg gets 4 threads.

import twitch_api  # noqa: E402  (stdlib-only .env reader)

twitch_api.load_env()  # .env may set the TOC_* overrides below; real environment variables still win


def total_ram_gib():
    """Physical RAM in GiB (POSIX sysconf), None where it isn't available (Windows)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (AttributeError, ValueError, OSError):
        return None


# A Linux box with <= 8 GB RAM (the Scala mini PC: Ryzen V120B 2C/4T, 6 GB, Vega 3, no NVIDIA) is low-spec without any
# env var: CPU Whisper, Ollama kept warm, 4 ffmpeg threads, a longer Ollama timeout (CPU inference). TOC_LOW_SPEC=0/1
# overrides either way. The flag is exported so the clipper / editor child processes see the same profile.
# TOC_CLOUD=1 is set on the RunPod Pod (backend/runpod_manager.py): 20 GB VRAM, so the big models and no low-spec rules.
CLOUD = os.environ.get("TOC_CLOUD") == "1"
MINI_PC = not CLOUD and sys.platform.startswith("linux") and (total_ram_gib() or 99) <= 8.5
LOW_SPEC = os.environ.get("TOC_LOW_SPEC", "1" if MINI_PC else "0") == "1"
if LOW_SPEC:
    os.environ["TOC_LOW_SPEC"] = "1"
KEEP_WARM = "10m" if CLOUD else "5m"
# the judge / director model: the 3B the taste profile was calibrated with; setup_linux.sh sets TOC_OLLAMA_MODEL=
# qwen2.5:1.5b in the mini PC's .env (2x faster on its CPU)
DEFAULT_MODEL = os.environ.get("TOC_OLLAMA_MODEL") or (
    "qwen2.5:14b-instruct" if CLOUD else "richardyoung/qwen2.5-3b-instruct-abliterated")
# Whisper: large-v3-turbo in float16 on the Pod; small.en here (the GTX 1050 Ti can't do float16)
DEFAULT_WHISPER = "large-v3-turbo" if CLOUD else "small.en"
WHISPER_GPU_COMPUTE = "float16" if CLOUD else "auto"
# s of wall clock per call, then one retry, then the caller's fallback; 60 s on the mini PC's CPU-only inference
OLLAMA_TIMEOUT = float(os.environ.get("TOC_OLLAMA_TIMEOUT") or (60 if MINI_PC else 45))
FALLBACK_MODELS = ("qwen2.5:1.5b",)  # setup_linux.sh pulls it next to the 3B the calibration was done with


def resolve_model(model, url="http://localhost:11434"):
    """`model` when Ollama has it pulled; else the first FALLBACK_MODELS one it has; else any model it has, so a machine
    that never pulled the configured 3B still gets a judge. Unchanged when Ollama is offline (heuristic fallback)."""
    try:
        with urllib.request.urlopen(url + "/api/tags", timeout=3) as r:
            tags = [m["name"] for m in json.loads(r.read()).get("models") or []]
    except (OSError, ValueError, KeyError, TypeError):
        return model
    have = set(tags) | {t.removesuffix(":latest") for t in tags}
    if not tags or model in have:
        return model
    pick = next((m for m in FALLBACK_MODELS if m in have), tags[0])
    print(json.dumps({"event": "log", "msg": f"[OLLAMA] {model} is not pulled on this machine: using {pick}"}), flush=True)
    return pick


def ollama_generate(prompt, model, url="http://localhost:11434", timeout=OLLAMA_TIMEOUT, retries=1, temperature=0,
                    opener=None):
    """Ollama /api/generate (JSON format) -> the parsed answer. The request runs in a daemon thread that is waited on
    for at most `timeout` s of wall clock (a socket timeout alone only bounds each read), so a stalled model can never
    freeze a batch. A timeout is logged and retried `retries` times, then raises TimeoutError (an OSError, so every
    caller's existing offline fallback applies). Any other error (offline, bad JSON) raises at once."""
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json",
                       "options": {"temperature": temperature}, **({"keep_alive": KEEP_WARM} if LOW_SPEC or CLOUD else {})}).encode()
    for attempt in range(retries + 1):
        box = {}

        def call():
            try:
                req = urllib.request.Request(url + "/api/generate", body, {"Content-Type": "application/json"})
                with (opener or urllib.request.urlopen)(req, timeout=timeout) as r:
                    box["raw"] = r.read()
            except Exception as e:  # handed to the waiting thread
                box["err"] = e

        t = threading.Thread(target=call, daemon=True)
        t.start()
        t.join(timeout)
        if "raw" in box:
            return json.loads(json.loads(box["raw"])["response"])
        if "err" in box and not isinstance(box["err"], TimeoutError):
            raise box["err"]
        print(json.dumps({"event": "log", "msg": f"[OLLAMA] {model}: no answer in {timeout:g} s, "
                          + ("retrying" if attempt < retries else "skipping (fallback)")}), flush=True)
    raise TimeoutError(f"Ollama gave no answer in {timeout:g} s ({retries + 1} tries)")


def _ask_json(prompt, model, url, timeout):
    return ollama_generate(prompt, model, url, timeout)


def _verdict(e):
    """One candidate's reply -> verdict, or None when a pillar is missing or not a number. Pillars become integers
    clamped to 0..10."""
    if not isinstance(e, dict):
        return None
    p = {k: e.get(k) for k in PILLARS}
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in p.values()):
        return None
    p = {k: int(min(10, max(0, round(float(v))))) for k, v in p.items()}
    return {"llm": round(sum(PILLARS[k] * v for k, v in p.items()) / 10, 3), "pillars": p,
            "reject": p["comedy_payoff"] < MIN_PAYOFF and p["hook_strength"] < MIN_HOOK,
            "reason": str(e.get("reason") or "").strip()[:200], "setup": None}


def ollama_judge(cands, model, url="http://localhost:11434", timeout=OLLAMA_TIMEOUT):
    """4-pillar verdict per candidate -> {index: {"llm": 0..1 weighted score, "reject": bool, "reason": str,
    "pillars": {...}, "setup": index | None}}, or None when Ollama is offline / nothing usable came back.
    ONE CLIP PER CALL: measured live (2026-11-13) the 3B model copied one number into every pillar and across clips
    when it saw ~12 at once (4/4/4/4), but scored each clip individually when judged alone. Story callbacks need
    all candidates together, so they get one small extra call (ollama_callbacks). A clip whose reply is unusable
    counts as unjudged. (A cold model load alone takes ~40 s on the GTX 1050 Ti: ollama_warmup starts it early, and
    ollama_generate's retry covers a load that still runs past the 45 s limit.)
    Serial on purpose: 3 concurrent calls with OLLAMA_NUM_PARALLEL=3 measured only ~7% faster on the GTX 1050 Ti
    (68.8 s -> 62.5 s for 7 clips) while tripling Ollama's KV cache (864 MB) next to Whisper in 4 GB of VRAM."""
    out = {}
    for i, c in enumerate(cands):
        item = {"index": 0, "replay": round(c["signal"], 2), "vocal": round(c.get("vocal", 0.0), 2),
                **({"visual": round(c["visual"], 2)} if c.get("visual") is not None else {}), "opening": opening(c)[:300], "transcript": c["transcript"][:1500]}
        try:
            data = _ask_json(JUDGE_PROMPT + json.dumps([item], ensure_ascii=False), model, url, timeout)
        except OSError:
            if i == 0:
                return None  # offline: don't wait out every candidate
            continue
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        entries = data.get("clips") if isinstance(data, dict) else None
        v = _verdict(entries[0] if isinstance(entries, list) and entries else data)  # tolerate a missing wrapper
        if v:
            out[i] = v
    if len(cands) > 1 and out:
        for later, setup in ollama_callbacks(cands, model, url, timeout).items():
            if later in out:
                out[later]["setup"] = setup
    return out or None


def ollama_callbacks(cands, model, url="http://localhost:11434", timeout=OLLAMA_TIMEOUT):
    """One call over all candidates (in time order): which later clip calls back an earlier one -> {later: setup}.
    {} on any error. Indexes must exist and differ; the clipper still checks the timing and a shared rare word."""
    order = sorted(range(len(cands)), key=lambda i: cands[i].get("start_sec", 0))
    items = [{"index": i, "transcript": cands[i]["transcript"][:600]} for i in order]
    try:
        data = _ask_json(CALLBACK_PROMPT + json.dumps(items, ensure_ascii=False), model, url, timeout)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}
    out = {}
    for e in data.get("callbacks", []) if isinstance(data, dict) and isinstance(data.get("callbacks"), list) else []:
        later, setup = (e.get("later"), e.get("setup")) if isinstance(e, dict) else (None, None)
        if all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x < len(cands) for x in (later, setup)) and later != setup:
            out[later] = setup
    return out


METADATA_PROMPT = """You write the post for an approved viral Short of a streamer clip. From the transcript and the
judge's note, reply with JSON only, keys:
- "title_angles": 4 different punchy titles, EACH 42 to 52 characters long, each a different angle (the reaction,
  the absurd line, the situation, a question). No hashtags or emojis in titles. Never invent events.
- "hashtags": exactly 5 hashtags (the streamer, the game or topic, the joke, #shorts, one broad one like #funny).
- "pinned_comment": one short line (under 100 characters) to pin under the video that makes people reply.
"""
TITLE_LEN = (38, 56)  # the prompt asks 42-52; a title a few characters off still reads fine, far off does not


def ollama_metadata(c, model, streamer=None, url="http://localhost:11434", timeout=OLLAMA_TIMEOUT):
    """-> {"title_angles": 1-4 titles, "hashtags": 5 or fewer, "pinned_comment"} for an approved clip, or {} (offline,
    junk). Only what passes validation is kept: nothing is made up to fill the gaps."""
    ctx = {"streamer": streamer, "transcript": str(c.get("transcript") or "")[:1200], "judge": c.get("judge_reason")}
    try:
        d = _ask_json(METADATA_PROMPT + json.dumps(ctx, ensure_ascii=False), model, url, timeout)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}
    if not isinstance(d, dict):
        return {}
    titles = [re.sub(r"\s+", " ", t).strip().strip('"') for t in d.get("title_angles") or [] if isinstance(t, str)]
    titles = list(dict.fromkeys(t for t in titles if TITLE_LEN[0] <= len(t) <= TITLE_LEN[1] and "#" not in t))[:4]
    tags = ["#" + re.sub(r"[^a-z0-9_]", "", t.lower().lstrip("#")) for t in d.get("hashtags") or [] if isinstance(t, str)]
    tags = list(dict.fromkeys(t for t in tags if len(t) > 1))[:5]
    pin = re.sub(r"\s+", " ", str(d.get("pinned_comment") or "")).strip()[:150]
    out = {k: v for k, v in (("title_angles", titles), ("hashtags", tags), ("pinned_comment", pin)) if v}
    return out


def ollama_warmup(model, url="http://localhost:11434"):
    """Load the model into memory without generating (an empty prompt), so the judge call later skips the
    ~40 s cold load. Fire-and-forget: errors just mean the judge pays the load itself. (LOW_SPEC runs Whisper on the
    CPU, so loading the model while it transcribes is safe there too.)"""
    body = json.dumps({"model": model, "keep_alive": KEEP_WARM if LOW_SPEC or CLOUD else "10m"}).encode()
    try:
        req = urllib.request.Request(url + "/api/generate", body, {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=180).close()
    except OSError:
        pass


def ollama_unload(url="http://localhost:11434"):
    """Unload every model Ollama holds (frees its VRAM now). -> names unloaded; [] when offline."""
    try:
        with urllib.request.urlopen(url + "/api/ps", timeout=3) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models") or []]
        for n in names:
            req = urllib.request.Request(url + "/api/generate", json.dumps({"model": n, "keep_alive": 0}).encode(),
                                         {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).close()
        return names
    except (OSError, ValueError, KeyError, TypeError):
        return []


def vram_free_mib():
    """Free VRAM of the first NVIDIA GPU in MiB (nvidia-smi), None without one / without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.split()[0]) if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def ollama_online(url="http://localhost:11434"):
    try:
        urllib.request.urlopen(url + "/api/tags", timeout=1).close()
        return True
    except OSError:
        return False


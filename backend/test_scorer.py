"""Self-check for scorer.py: python backend/test_scorer.py"""
import re

import numpy as np

import scorer


def bump(n, at, width, height=1.0):
    c = np.zeros(n)
    c[at - width // 2:at + width // 2 + 1] = height
    return c


# --- dynamic window sizing: grows with the spike, clamped to [min, max] -------------------------------------
curve = np.zeros(300)
curve[100:111] = [0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]  # buildup + decay around peak 104
s, e = scorer.size_window(curve, 104, 15, 55)
assert (s, e) == (96, 111), (s, e)  # 101-107 are above 35% of the peak, +5 s setup, +3 s reaction
s, e = scorer.size_window(bump(300, 150, 1), 150, 15, 55)  # 1 s blip: grown to min, 2/3 before the peak
assert e - s == 15 and s == 140.5, (s, e)
s, e = scorer.size_window(bump(300, 150, 101), 150, 15, 55)  # 101 s plateau: capped at max
assert round(e - s, 2) == 55, (s, e)
assert scorer.size_window(bump(300, 2, 1), 2, 15, 55)[0] == 0  # clamped to the video start
assert scorer.size_window(bump(300, 298, 1), 298, 15, 55)[1] == 300  # ... and end
assert scorer.size_window(bump(10, 5, 1), 5, 15, 55) == (0, 10)  # video shorter than min

# pick_windows: best first, non-overlapping, sized per peak
c = bump(300, 50, 5) + bump(300, 200, 1, 0.8) + bump(300, 53, 1, 0.9)
w = scorer.pick_windows(c, [f"r{i}" for i in range(300)], 3, 15, 55)
assert [x[3] for x in w][:2] == ["r48", "r200"], w  # 53 overlaps the first window, so 200 comes next
assert all(15 <= e - s <= 55 for s, e, *_ in w)
assert all(a[1] <= b[0] or b[1] <= a[0] for i, a in enumerate(w) for b in w[i + 1:])

# --- heatmap: prominence over the local ~80 s median, intro dropped, reason format --------------------------
hm = [{"start_time": i * 10, "end_time": i * 10 + 10, "value": v} for i, v in enumerate([1.0, .4, .3, .25, .2, .5, .2, .15, .1])]
curve, reasons = scorer.heatmap_curve(hm, 90)
assert int(np.argmax(curve)) in (54, 55) and curve.max() > 0.9, np.argmax(curve)  # peak at segment midpoint 55.0
assert reasons[55] == "Most Replayed 2.2x spike @ 0:55 | Peak 0.50", reasons[55]
assert curve[:5].max() == 0  # intro segment zeroed (up to its midpoint)
vals = [0.2] * 5 + [0.8] * 11 + [0.2] * 6 + [0.6] + [0.2] * 7  # sharp 0.6 spike vs flat 0.8 plateau
curve, _ = scorer.heatmap_curve([{"start_time": i * 10, "end_time": i * 10 + 10, "value": v} for i, v in enumerate(vals)], 300)
assert int(np.argmax(curve)) in (224, 225), np.argmax(curve)
assert scorer.clock(862) == "14:22" and scorer.clock(3725) == "1:02:05"

# --- audio energy: a loud jump at second 5 wins ------------------------------------------------------------
curve, reasons = scorer.energy_curve([0.1] * 5 + [0.9] + [0.1] * 5)
assert int(np.argmax(curve)) == 5 and reasons[5].startswith("Audio spike")

# --- visual: steady motion everywhere, one hard flash at second 30 -> the flash is the action peak -------------
rng = np.random.default_rng(0)
frames = rng.integers(90, 110, size=(120, 90, 160), dtype=np.uint8)  # 60 s of noisy "gameplay" at 2 fps
frames[60:62] = 250  # explosion / Wasted flash
v = scorer.visual_curve(frames)
assert len(v) == 60 and int(np.argmax(v)) == 30 and v.max() == 1.0, (len(v), np.argmax(v))
assert np.median(v) < 0.1  # constant motion alone is not "action"
sig, vis, rank, tags = scorer.signal_curves(np.zeros(60), ["x"] * 60, visual=v)
assert int(np.argmax(rank)) == 30 and tags[30] == "x + visual action 1.00" and tags[0] == "x"
assert rank.max() == 0.3  # visual alone only reaches its 15/50 share of the ranking curve
assert scorer.signal_curves(np.ones(3), ["y"] * 3)[3] == ["y"] * 3

# --- transcripts, snapping, transcript cues --------------------------------------------------------------------
cues = [(1, 3, "hello world"), (3, 5, "[Laughter] oh my god!")]
assert scorer.transcript_between(cues, 4, 10) == "[Laughter] oh my god!"

lines = [(18.5, 21, "a"), (21, 25, "b"), (38, 41.2, "c")]
assert scorer.snap_window(20, 40, lines) == (21, 41.2)
assert scorer.snap_window(20, 50, lines) == (21, 50)  # no line end near 50: end untouched
assert scorer.snap_window(20, 40, []) == (20, 40)
assert scorer.snap_window(20, 40, lines, max_len=20) == (20, 40)  # snapped 20.2 s would break max: keep original

s, tag = scorer.cue_score("[Laughter] oh my god! hahaha")
assert s > 0.5 and "laugh" in tag, (s, tag)
assert scorer.cue_score("the weather is mild today")[0] == 0
assert scorer.final_score(1, 1, 1) == 1 and scorer.final_score(1, 0, 0) == 0.5
assert scorer.final_score(0, 1, 0, llm=1) == 0.7  # 0.5 * (0.7 LLM) + 0.35 signal
assert scorer.final_score(1, 1, None) == 1 and scorer.final_score(0, 1, None, llm=1) == round(0.7 / 0.85, 3)  # no scan: no penalty

# --- story callbacks ----------------------------------------------------------------------------------------
story = [(10, 14, "okay we're going to the bank now"),
         (72, 76, "this guy in a purple dinosaur costume just stole my car"),
         (90, 94, "anyway let's go to the bank"),
         (300, 304, "wait remember that dinosaur dude"),
         (304, 308, "he's back in the purple dinosaur costume")]
hit = scorer.find_callback(story, 298, "wait remember that dinosaur dude he's back in the purple dinosaur costume")
assert hit and hit[:2] == (72, 76) and {"purple", "dinosaur", "costume"} <= set(hit[2]), hit
assert scorer.find_callback(story, 100, "purple dinosaur costume") is None  # earlier mention < 45 s before
assert scorer.find_callback(story, 298, "wait what is going on here") is None  # nothing shared: standalone clip
assert scorer.find_callback(story, 298, "let's go to the bank") is None  # 'bank' is common (3+ uses) and 1 word

# stitch plan: setup starts on a keyframe, 6-10 s long; payoff pre-roll trimmed so the total fits max
kfs = [float(k) for k in range(0, 400, 4)]  # keyframe every 4 s
(ss, se), (ps, pe) = scorer.stitch_plan(kfs, (72, 76), (290, 335), 55)
assert ss == 72 and 6 <= se - ss <= 10 and se >= 76, (ss, se)
assert ps in kfs and pe == 335 and (se - ss) + (pe - ps) <= 55, (ps, pe)
(ss, se), (ps, pe) = scorer.stitch_plan(kfs, (72, 76), (300, 320), 55)
assert ps == 300 and pe == 320  # short payoff fits as-is (300 is a keyframe)
assert scorer.stitch_plan([0.0], (72, 76), (300, 320), 55) is None  # mention 72 s into the setup part
assert scorer.stitch_plan(kfs, (72, 76), (300, 320), 12) is None  # no room left for the payoff

# --- vocal laughter / yell detection -------------------------------------------------------------------------
sr = 16000
t = np.arange(60 * sr) / sr
rng = np.random.default_rng(1)
audio = (0.3 * np.sin(2 * np.pi * 90 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)  # steady engine
music = (t >= 10) & (t < 20)
audio[music] += 0.4 * np.sin(2 * np.pi * 1000 * t[music])  # sustained voice-band tone: loud but steady (music)
laugh = (t >= 40) & (t < 42)
gate = (np.sin(2 * np.pi * 5 * t[laugh]) > 0).astype(np.float32)  # 5 Hz "ha-ha-ha" bursts
audio[laugh] += 0.4 * gate * (np.sin(2 * np.pi * 700 * t[laugh]) + np.sin(2 * np.pi * 1400 * t[laugh]))
voc = scorer.vocal_curve(audio, sr)
assert len(voc) == 59 and int(np.argmax(voc)) in (40, 41), (len(voc), np.argmax(voc))
assert voc[12:18].max() < 0.6 * voc.max(), voc[10:20]  # steady tone: burst but no flutter
assert voc[25:35].max() < 0.1 and voc[50:].max() < 0.1  # engine alone: nothing
assert len(scorer.vocal_curve(np.zeros(sr, np.float32))) == 1  # too short: zeros, no crash

# --- funnel: 12 candidates scored on comedy before picking N --------------------------------------------------
peaks = np.zeros(1000)
for i in range(12):
    peaks[40 + i * 80] = 1.0 - i * 0.05  # 12 separate spikes, the first one strongest
pool = scorer.pick_windows(peaks, [f"p{i}" for i in range(1000)], max(4 * 3, 12), 15, 55)
assert len(pool) == 12
cands = [{"start_sec": s, "end_sec": e, "cue": 0.0, "signal": sc, "visual": 0.0, "llm": 0.4} for s, e, sc, _ in pool]
cands[0]["llm"] = 0.1  # strongest replay spike, but the judge calls it banter
cands[9].update(cue=0.6, llm=0.9)  # 10th by signal, actually hilarious
best = scorer.rank_candidates(cands, 4)
assert best[0] is cands[9] and cands[0] not in best, [c["start_sec"] for c in best]
assert len(best) == 4 and all("score" in c for c in best)
cands[1].update(start_sec=cands[9]["start_sec"] + 5, end_sec=cands[9]["end_sec"] + 5, llm=0.85)  # snapped into #9
r = scorer.rank_candidates(cands, 4)
assert len(r) == 4 and not (cands[1] in r and cands[9] in r)  # overlapping pair never both kept

# --- Ollama judge: ONE clip per call (a batch made the 3B model copy scores), + one call for story callbacks ---------
import io, json, urllib.request
sent = []
class FakeResponse(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False
pill = lambda sc, hs, cp, q, **kw: dict(self_contained=sc, hook_strength=hs, comedy_payoff=cp, quotability=q, **kw)
def fake_urlopen(req, timeout=None):
    body = json.loads(req.data)
    sent.append(body)
    prompt = body["prompt"]
    if prompt.startswith(scorer.CALLBACK_PROMPT):  # bad pairs: self-callback, unknown index, a bool
        reply = {"callbacks": [{"later": 1, "setup": 0}, {"later": 3, "setup": 3}, {"later": 9, "setup": 0}, {"later": True, "setup": 0}]}
    elif "driving to the shop" in prompt:
        reply = {"clips": [dict(index=0, reason="Just driving to the shop.", **pill(2, 3, 1, 2))]}
    elif "purple dinosaur" in prompt:  # no "clips" wrapper, a float and an out-of-range score
        reply = dict(index=0, reason="He is back in the dinosaur suit.", **pill(14, 7, 9, 8.6))
    elif "trick shot" in prompt:
        reply = {"clips": [dict(index=0, reason="Great setup, but the joke never lands.", **pill(9, 9, 4, 9))]}
    else:
        reply = {"clips": [dict(index=0, reason="no numbers", **pill("x", 5, 5, 5))]}
    return FakeResponse(json.dumps({"response": json.dumps(reply)}).encode())
real_urlopen, urllib.request.urlopen = urllib.request.urlopen, fake_urlopen
judge_in = [{"signal": 0.9, "visual": 0.1, "vocal": 0.2, "transcript": "so yeah we're just driving to the shop"},
            {"signal": 0.5, "visual": 0.8, "vocal": 0.9, "transcript": "he's back in the purple dinosaur suit",
             "start_sec": 100.0, "lines": [(99.8, 101.5, "WAIT he's back"), (104.0, 106.0, "in the purple dinosaur suit")]},
            {"signal": 0.4, "visual": 0.0, "transcript": "menu"},
            {"signal": 0.6, "visual": 0.2, "transcript": "watch this trick shot it's going to be insane"}]
scorer.MIN_PAYOFF, scorer.MIN_HOOK = 5, 5  # the dead-air floor as the Weight Tuner can set it (default 0, 0 = off)
verdict = scorer.ollama_judge(judge_in, "m")
judge_calls = [b for b in sent if b["prompt"].startswith(scorer.JUDGE_PROMPT)]
tone_calls = [b for b in sent if b["prompt"].startswith(scorer.TONE_PROMPT)]
assert len(judge_calls) == 4 and len(tone_calls) == 3 and len(sent) == 8, len(sent)  # per clip + a tone call per judged clip + callbacks
assert all(b["format"] == "json" and b["stream"] is False for b in sent)
for b, c in zip(judge_calls, judge_in):  # each call sees exactly its own clip: nothing to copy from
    items = json.loads(b["prompt"].split("Candidates:", 1)[1])
    assert len(items) == 1 and items[0]["transcript"] == c["transcript"], items
assert json.loads(judge_calls[1]["prompt"].split("Candidates:", 1)[1])[0]["opening"] == "WAIT he's back"
assert json.loads(judge_calls[3]["prompt"].split("Candidates:", 1)[1])[0]["opening"] == "watch this trick shot it's going to"
assert "self_contained" in scorer.JUDGE_PROMPT and "comedy_payoff" in scorer.JUDGE_PROMPT and "setup" not in scorer.JUDGE_PROMPT
assert set(verdict) == {0, 1, 3}, verdict  # the non-numeric clip counts as unjudged
assert verdict[0]["reject"] and abs(verdict[0]["llm"] - 0.185) < 1e-9 and verdict[0]["reason"] == "Just driving to the shop."
assert verdict[1]["pillars"] == {"self_contained": 10, "comedy_payoff": 9, "hook_strength": 7, "quotability": 9}  # ints, clamped
assert all(type(v) is int for v in verdict[1]["pillars"].values()) and abs(verdict[1]["llm"] - 0.895) < 1e-9
assert verdict[1]["reason"] == "He is back in the dinosaur suit." and verdict[1]["setup"] == 0  # callback merged; bad pairs dropped
assert verdict[0]["setup"] is None and verdict[3]["setup"] is None
assert not verdict[3]["reject"] and verdict[3]["llm"] > 0.6, verdict[3]  # weak payoff but a 9 hook: weighted score decides
sent.clear()
def offline_first(req, timeout=None):
    sent.append(req)
    raise OSError("connection refused")
urllib.request.urlopen = offline_first
assert scorer.ollama_judge(judge_in, "m") is None and len(sent) == 1, "offline: gives up after the first call"
urllib.request.urlopen = fake_urlopen
c_rej = {"cue": 0.9, "signal": 0.9, "visual": 0.9, "start_sec": 0, "end_sec": 20, "reason": "Heatmap"}
scorer.apply_verdict(c_rej, verdict[0])  # payoff 1, hook 3: dead air
assert not scorer.passes_gate(c_rej, 0) and c_rej["judge_reason"].startswith("Just driving") and "(rejected)" in c_rej["reason"]
assert scorer.rejected([c_rej])[0]["judge_reason"] == "Just driving to the shop."
assert scorer.rank_candidates([c_rej], 4) == [], "hard-rejected even with the gate off and loud signals"

# no copy bait: the prompt shows no example numbers next to any pillar, and no "score it N" phrasing
prompt = scorer.JUDGE_PROMPT
assert not re.search(r'"?(self_contained|comedy_payoff|hook_strength|quotability)"?\s*[:=]\s*\d', prompt), "example number in the prompt"
assert "3 or lower" not in prompt and "1-3 = dead air" in prompt and "7-10 = a big reaction" in prompt

# the gate: payoff carries a clip even out of context; only dead air (no payoff AND no hook) is vetoed
def verdict_for(sc, cp, hs=7, q=6):
    reply = {"clips": [dict(index=0, self_contained=sc, comedy_payoff=cp, hook_strength=hs, quotability=q, reason="r")]}
    urllib.request.urlopen = lambda req, timeout=None: FakeResponse(json.dumps({"response": json.dumps(reply)}).encode())
    return scorer.ollama_judge(judge_in[:1], "m")[0]
banger = verdict_for(3, 9)  # a 9/10 outburst that starts mid-conversation
assert not banger["reject"] and banger["llm"] * 10 >= 6.0, banger  # passes the default Min Comedy Score 6 cleanly
dead = verdict_for(5, 2, hs=2)
assert dead["reject"], dead  # menus / dead air
assert not verdict_for(9, 4)["reject"] and not verdict_for(3, 3, hs=6)["reject"]  # a hook alone is not dead air
assert verdict_for(9, 4, hs=4)["reject"]
# the default: no floor, so a jokeless gameplay clip (3/3/3/3 = 4.5 with boosts) is judged by Min Comedy Score alone
scorer.MIN_PAYOFF, scorer.MIN_HOOK = scorer.DEFAULTS["MIN_PAYOFF"], scorer.DEFAULTS["MIN_HOOK"]
assert (scorer.MIN_PAYOFF, scorer.MIN_HOOK) == (0, 0)
gameplay = verdict_for(3, 3, hs=3, q=3)
assert not gameplay["reject"] and not verdict_for(0, 0, hs=0)["reject"], gameplay
g = dict(cue=0, signal=0.5, visual=0, start_sec=0, end_sec=45, reason="Twitch clip")
scorer.apply_verdict(g, gameplay)
assert scorer.passes_gate(g, 3.0) and not scorer.passes_gate(g, 5.0) and "(rejected)" not in g["reason"], g
scorer.MIN_PAYOFF, scorer.MIN_HOOK = 5, 5
# tone: its own call; an emotional_story needs strength >= EMOTION_PASS, then passes the gate on that alone
assert all(v["tone"] == "hype_chaos" and v["resonance"] == 0 for v in verdict.values()), verdict  # no tone keys -> hype
def tone_reply(reply):
    urllib.request.urlopen = lambda req, timeout=None: FakeResponse(json.dumps({"response": json.dumps(reply)}).encode())
    return scorer.ollama_tone({"transcript": "my dad passed last year, this one is for him"}, "m")
assert tone_reply({"tone": "emotional", "emotion": "Sad!", "strength": 9}) == {"tone": "emotional_story", "emotion": "sad", "resonance": 9}
assert tone_reply({"tone": "emotional", "emotion": "confused", "strength": 7})["tone"] == "hype_chaos", "7: the 3B's false-positive zone"
assert tone_reply({"tone": "funny", "emotion": "none", "strength": 9})["tone"] == "hype_chaos"
assert tone_reply({"tone": "emotional", "strength": "high"})["tone"] == "hype_chaos" and tone_reply("garbage")["tone"] == "hype_chaos"
story = dict(g, llm=0.21, tone="emotional_story", emotion="nostalgic", resonance=8)
assert scorer.passes_gate(story, 6.0) and not scorer.passes_gate(dict(story, tone="hype_chaos"), 6.0), "moving, not funny: still passes"
c_tone = {"cue": 0, "signal": 0.5, "visual": 0, "start_sec": 0, "end_sec": 30, "reason": "x"}
scorer.apply_verdict(c_tone, dict(gameplay, tone="emotional_story", emotion="proud", resonance=9))
assert (c_tone["tone"], c_tone["emotion"], c_tone["resonance"]) == ("emotional_story", "proud", 9)
urllib.request.urlopen = real_urlopen

# viral boost: a proven Twitch banger (>= 3000 views + a vocal burst) gets half a point; a quiet or small one doesn't
spoke = dict(cue=0, signal=0.9, visual=0, vocal=0.8, start_sec=0, end_sec=30, reason="Twitch clip")
scorer.apply_verdict(spoke, dict(banger, llm=0.57))  # "Spoke too soon": 5.7 from the 3B judge
assert not scorer.passes_gate(spoke, 6.0) and scorer.viral_boost(spoke, 3009)
assert spoke["llm"] == 0.62 and scorer.passes_gate(spoke, 6.0) and "viral boost (3009 views)" in spoke["reason"]
assert not scorer.viral_boost(dict(spoke, llm=0.5), 2999) and not scorer.viral_boost(dict(spoke, llm=0.5, vocal=0.2), 9000)
assert not scorer.viral_boost(dict(spoke, llm=None), 9000)  # unjudged: nothing to boost
# streamer prior (taste profile): CaseOh +1 point, nobody else, never on an unjudged clip
cs = dict(spoke, llm=0.6, reason="")
assert scorer.streamer_boost(cs, "CaseOh_") and cs["llm"] == 0.7 and "streamer prior" in cs["reason"]
assert not scorer.streamer_boost(dict(spoke, llm=0.6), "Jynxzi") and not scorer.streamer_boost(dict(spoke, llm=None), "caseoh_")
assert "STREAMER'S PERSONALITY" in scorer.JUDGE_PROMPT and "nothing happens" in scorer.JUDGE_PROMPT
assert not scorer.passes_gate(dict(c_rej, llm=0.99), 0), "a boost never undoes a dead-air reject"

# [JUDGE] summary: only when nothing passed; names the best candidate and the rule that stopped it
mk = lambda sc, cp, hs, llm, reason: dict(cue=0, signal=0.5, visual=0, start_sec=0, end_sec=20, reason="r", llm=llm,
                                          judge_reject=cp < 5 and hs < 5, judge_reason=reason,
                                          pillars={"self_contained": sc, "comedy_payoff": cp, "hook_strength": hs, "quotability": 5})
floor = [mk(5, 2, 3, 0.35, "A menu, nothing happens."), mk(4, 1, 2, 0.20, "Silence.")]
assert scorer.judge_summary("vid1", floor, 6) == ("[JUDGE] Rejected all 2 candidate(s) for vid1 (best: payoff=2, hook=3: dead air "
                                                   "(needs payoff >= 5 or hook >= 5)): A menu, nothing happens."), scorer.judge_summary("vid1", floor, 6)
assert scorer.judge_summary("vid2", [mk(6, 6, 6, 0.58, "")], 6) == "[JUDGE] Rejected all 1 candidate(s) for vid2 (best: score 5.8/10 < Min Comedy Score 6)"
assert scorer.judge_summary("vid3", [mk(6, 6, 6, 0.58, ""), mk(7, 8, 7, 0.75, "")], 6) is None  # one passed
assert scorer.judge_summary("vid4", [dict(cue=0, signal=0.5, visual=0)], 6) is None  # Ollama off: nothing judged
scorer.MIN_PAYOFF, scorer.MIN_HOOK = scorer.DEFAULTS["MIN_PAYOFF"], scorer.DEFAULTS["MIN_HOOK"]
urllib.request.urlopen = lambda req, timeout=None: FakeResponse(b'{"response": "not json"}')
assert scorer.ollama_judge(judge_in, "m") is None
def offline(req, timeout=None): raise OSError("connection refused")
urllib.request.urlopen = offline
assert scorer.ollama_judge(judge_in, "m") is None and not scorer.ollama_online()
urllib.request.urlopen = real_urlopen
# banger gate: Ollama-rated candidates under the gate never make the cut; unjudged ones (Ollama off) still can
gate = [dict(cue=0.9, signal=0.9, visual=0.9, llm=0.5, start_sec=0, end_sec=20),
        dict(cue=0.1, signal=0.2, visual=0.0, llm=0.7, start_sec=30, end_sec=50),
        dict(cue=0.5, signal=0.5, visual=0.0, start_sec=60, end_sec=80)]
assert [c["start_sec"] for c in scorer.rank_candidates(gate, 4, 6)] == [60, 30]
assert len(scorer.rank_candidates(gate, 4)) == 3  # default 0 = off

# gaming vocabulary: line text fixes (case-insensitive, two-word misses, punctuation between them)
import pathlib, tempfile
vdir = pathlib.Path(tempfile.mkdtemp())
none_file = vdir / "missing.json"  # never the real workspace/custom_vocab.json
assert scorer.fix_gaming_text("Fortnight with jinx see and Case, oh. Queso!", "caseoh_ chat roast", none_file) \
    == "Fortnite with Jynxzi and CaseOh. CaseOh!"
assert scorer.fix_gaming_text("Queso dip, fortnight", "xqc cooking stream", none_file) == "Queso dip, Fortnite"  # scoped
assert scorer.fix_gaming_words([{"word": "queso", "start": 1, "end": 2}], "xqc", none_file)[0]["word"] == "queso"
# workspace/custom_vocab.json: extra Whisper terms + your own fixes (they win), never fatal when broken
vf = vdir / "custom_vocab.json"
assert scorer.whisper_prompt(none_file) == scorer.GAMING_PROMPT
vf.write_text(json.dumps({"prompt_terms": ["Ohnepixel", "Marbles on Stream"], "fixes": {"oh nay pixel": "Ohnepixel", "fortnight": "FORTNITE"}}))
assert scorer.whisper_prompt(vf).endswith("clip it, Ohnepixel, Marbles on Stream."), scorer.whisper_prompt(vf)
assert scorer.fix_gaming_text("oh nay pixel plays fortnight", path=vf) == "Ohnepixel plays FORTNITE"
w = scorer.fix_gaming_words([{"word": "oh", "start": 1.0, "end": 1.2}, {"word": "nay", "start": 1.2, "end": 1.4},
                             {"word": "pixel!", "start": 1.4, "end": 1.9}], path=vf)
assert w == [{"word": "Ohnepixel!", "start": 1.0, "end": 1.9}], w
vf.write_text("{broken")
assert scorer.fix_gaming_text("fortnight", path=vf) == "Fortnite" and scorer.whisper_prompt(vf) == scorer.GAMING_PROMPT
assert scorer.fix_gaming_text("a fortnightly showcase ohm") == "a fortnightly showcase ohm"  # whole words only
# --- hard Ollama timeout: a stalled model (or one that never answers at all) costs timeout x 2, then a clean fallback
import contextlib, threading, time
stall_calls = []


def stalled(req, timeout=None):
    stall_calls.append(timeout)
    threading.Event().wait()  # hangs forever, like the stuck "Directing with ..." run


t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()) as log:
    try:
        scorer.ollama_generate("p", "m", timeout=0.2, opener=stalled)
        raise AssertionError("should time out")
    except TimeoutError:
        pass
assert len(stall_calls) == 2 and time.time() - t0 < 1.5, (stall_calls, time.time() - t0)  # one retry, wall clock bounded
assert "no answer in 0.2 s, retrying" in log.getvalue() and "skipping (fallback)" in log.getvalue()


class Answer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


tries = []
flaky = lambda req, timeout=None: (tries.append(1), threading.Event().wait(0.5) if len(tries) == 1 else None,
                                   Answer(json.dumps({"response": '{"ok": 1}'}).encode()))[2]
with contextlib.redirect_stdout(io.StringIO()):
    assert scorer.ollama_generate("p", "m", timeout=0.2, opener=flaky) == {"ok": 1} and len(tries) == 2  # retry answers


def refused(req, timeout=None):
    raise urllib.error.URLError(ConnectionRefusedError())


refused_calls = []
try:
    scorer.ollama_generate("p", "m", timeout=5, opener=lambda r, timeout=None: (refused_calls.append(1), refused(r))[1])
except OSError:
    pass
assert refused_calls == [1], "offline is not retried: fall back at once"

# --- weights.json (calibration tuner): overrides apply, junk is ignored, a missing file restores the defaults
wdir = pathlib.Path(tempfile.mkdtemp())
(wdir / "w.json").write_text(json.dumps({"MIN_PAYOFF": 6, "VIRAL_BOOST": 0.1, "W_COMEDY": "x", "NOPE": 1, "MIN_HOOK": -1,
                                         "PILLARS": {"self_contained": 0.25, "comedy_payoff": 0.45, "hook_strength": 0.2, "quotability": 0.1}}))
used = scorer.load_weights(wdir / "w.json")
assert set(used) == {"MIN_PAYOFF", "VIRAL_BOOST", "PILLARS"} and scorer.MIN_PAYOFF == 6 and scorer.W_COMEDY == 0.5, used
assert scorer._verdict({"self_contained": 10, "comedy_payoff": 0, "hook_strength": 0, "quotability": 0})["llm"] == 0.25
(wdir / "w.json").write_text(json.dumps({"PILLARS": {"comedy_payoff": 1.0}}))  # missing pillars: ignored
assert scorer.load_weights(wdir / "w.json") == {} and scorer.PILLARS == scorer.DEFAULTS["PILLARS"]
assert scorer.load_weights(wdir / "missing.json") == {} and scorer.MIN_PAYOFF == 0 and scorer.VIRAL_BOOST == 0.05
scorer.load_weights()  # back to the repo's file

# hardware profile + model fallback (the Linux mini PC): this Windows box is not a mini PC
assert not scorer.MINI_PC and scorer.OLLAMA_TIMEOUT == 45 and (scorer.total_ram_gib() is None or scorer.total_ram_gib() > 1)
tags = {"models": [{"name": "qwen2.5:1.5b"}, {"name": "llama3.2:latest"}]}
class Tags(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False
urllib.request.urlopen = lambda url, timeout=None: Tags(json.dumps(tags).encode())
with contextlib.redirect_stdout(io.StringIO()) as said:
    assert scorer.resolve_model("richardyoung/qwen2.5-3b-instruct-abliterated") == "qwen2.5:1.5b"  # 3B not pulled
assert "not pulled on this machine: using qwen2.5:1.5b" in said.getvalue()
assert scorer.resolve_model("llama3.2") == "llama3.2"  # ":latest" tag counts as pulled
tags["models"] = [{"name": "mistral:7b"}]
with contextlib.redirect_stdout(io.StringIO()):
    assert scorer.resolve_model("qwen-x") == "mistral:7b"  # no fallback pulled: anything beats no judge
def offline(url, timeout=None):
    raise OSError("offline")
urllib.request.urlopen = offline
assert scorer.resolve_model("qwen-x") == "qwen-x"  # offline: unchanged, the heuristic fallback applies
urllib.request.urlopen = real_urlopen


# approved clip -> title angles (42-52 asked, 38-56 kept), 5 hashtags, pinned line; junk or offline -> {} (nothing invented)
real_ask = scorer._ask_json
answers = []
scorer._ask_json = lambda prompt, model, url, timeout: (answers.append(prompt), {
    "title_angles": ["CaseOh found out what is inside the fridge at 3am", "short", "CaseOh found out what is inside the fridge at 3am",
                     "Chat could not stop laughing at this CaseOh moment", "#tag in a title that is long enough to keep it ok ok",
                     "Why would anyone hide THAT in the fridge, CaseOh?", 7],
    "hashtags": ["caseoh", "#Fridge Moment", "#funny", "#shorts", "#twitch", "#extra"], "pinned_comment": "  What would YOU do?  "})[1]
md = scorer.ollama_metadata({"transcript": "what is in the fridge bro", "judge_reason": "big reaction"}, "m", "caseoh_")
assert md["title_angles"] == ["CaseOh found out what is inside the fridge at 3am", "Chat could not stop laughing at this CaseOh moment",
                              "Why would anyone hide THAT in the fridge, CaseOh?"], md["title_angles"]  # short / dup / hashtag / non-str out
assert md["hashtags"] == ["#caseoh", "#fridgemoment", "#funny", "#shorts", "#twitch"] and md["pinned_comment"] == "What would YOU do?", md
assert '"streamer": "caseoh_"' in answers[0] and "42 to 52 characters" in answers[0]
scorer._ask_json = lambda *a: ["not", "a", "dict"]
assert scorer.ollama_metadata({"transcript": "x"}, "m") == {}
scorer._ask_json = lambda *a: (_ for _ in ()).throw(TimeoutError("slow"))
assert scorer.ollama_metadata({"transcript": "x"}, "m") == {}, "offline: no card, the run goes on"
scorer._ask_json = real_ask

print("scorer ok")

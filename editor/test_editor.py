"""Self-check for the shorts editor: python editor/test_editor.py  (real ffmpeg renders, no network / Ollama)"""
import contextlib
import io
import json
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

import director
import framing
import pipeline
import qc
import renderer

tmp = Path(tempfile.mkdtemp())
quiet = lambda: contextlib.redirect_stdout(io.StringIO())
SR = 16000


def words_at(pairs, dur=0.3):
    return [{"word": w, "start": t, "end": t + dur} for w, t in pairs]


def tone_audio(dur, bursts):
    """Quiet bed + loud 600 Hz bursts [(centre s, half width s, amplitude)] -> 16 kHz float32."""
    t = np.arange(int(dur * SR)) / SR
    a = 0.02 * np.sin(2 * np.pi * 220 * t)
    for c, hw, amp in bursts:
        m = np.abs(t - c) < hw
        a[m] += amp * np.sin(2 * np.pi * 600 * t[m]) * (1 - np.abs(t[m] - c) / hw)
    return a.astype(np.float32)


# --- SFX library: 7 procedural sounds; a user-dropped file always wins ------------------------------------------
sfx = tmp / "sfx"
sfx.mkdir()
(sfx / "vine_boom.mp3").write_bytes(b"user file")  # the user's own vine boom
renderer.ensure_sfx(sfx)
for name in renderer.SFX_NAMES:
    if name == "vine_boom":
        continue
    with wave.open(str(sfx / f"{name}.wav")) as f:
        x = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        assert f.getnchannels() == 1 and f.getframerate() == 48000, name
        assert 0.1 * 32767 < np.abs(x).max() <= 0.91 * 32767, (name, np.abs(x).max())  # audible, never clipping
assert not (sfx / "vine_boom.wav").exists() and renderer.sfx_file("vine_boom", sfx).name == "vine_boom.mp3"
(sfx / "pop.mp3").write_bytes(b"user")
assert renderer.sfx_file("pop", sfx).suffix == ".mp3"  # an .mp3 dropped next to the generated .wav wins too

# --- 5 scenarios: rule-based classification + tailored EDLs -----------------------------------------------------
dur = 20.0
calm = [(w, 0.5 + i * 0.55) for i, w in enumerate("so chat says i should build the house here okay".split())]

# chat_roast: calm talk, a pause, then an outburst
roast_w = words_at(calm + [("WHAT", 13.0), ("ARE", 13.3), ("YOU", 13.6), ("DOING", 13.9)])
roast_a = tone_audio(dur, [(13.5, 0.6, 0.9)])
env = director.energy(roast_a)
assert director.classify(roast_w, env, dur) == "chat_roast"
e = director.build_edl("chat_roast", roast_w, env, dur)


def still_max(edl):
    """Longest output stretch with no zoom moving / held and no cut join (the retention rule: <= 3 s)."""
    segs, _ = renderer.timeline(dict(edl, hook_teaser={"enabled": False}))
    busy = sorted((renderer.to_out(z["at"], segs), renderer.to_out(z["at"], segs) + z["duration"]) for z in edl["zooms"]
                  if not z.get("out") and renderer.to_out(z["at"], segs) is not None)
    marks, a, longest = [o for _, _, o, _, _ in segs[1:]], 0.0, 0.0
    for s, e_ in busy + [(renderer.main_length(segs), renderer.main_length(segs))]:
        for m in [x for x in marks if a < x < s] + [s]:
            longest, a = max(longest, m - a), m
        a = max(a, e_)
    return longest


snaps = [z for z in e["zooms"] if z["style"] == "snap"]
climax_snap = next(z for z in snaps if abs(z["at"] - 13.5) < 0.3)
assert climax_snap["scale"] == director.SNAP_CLIMAX and all(1.22 <= z["scale"] <= 1.35 for z in snaps), e["zooms"]
assert all(z["scale"] == director.RAMP_SCALE and 3.0 <= z["duration"] + 0.6 for z in e["zooms"] if z["style"] == "creep")
assert still_max(e) <= director.STILL_MAX + 0.01, (still_max(e), e["zooms"])  # never still for more than 3 s
segs_e, _ = renderer.timeline(dict(e, hook_teaser={"enabled": False}))
spans_ = sorted((renderer.to_out(z["at"], segs_e), renderer.to_out(z["at"], segs_e) + z["duration"]) for z in e["zooms"] if not z.get("out"))
assert all(b1 <= a2 + 1e-6 for (_, b1), (a2, _) in zip(spans_, spans_[1:])), "zooms never overlap"
assert {f["type"] for f in e["filters"] if not f.get("out")} == {"reaction_boost", "trauma_shake"} and e["hook_teaser"]["enabled"]
shake = next(f for f in e["filters"] if f["type"] == "trauma_shake")
assert abs(shake["end"] - shake["start"] - 0.25) < 0.01 and abs(shake["start"] - 13.5) < 0.3  # synced to the yell
names = [s["name"] for s in e["sfx"]]
assert names[:2] == ["record_scratch", "thud"] and e["mutes"] and set(names[2:]) == {"whoosh"} and len(names) - 2 == len(snaps)
gains = {s["name"]: s["gain_db"] for s in e["sfx"]}
assert gains == {"record_scratch": -14.0, "thud": -14.0, "whoosh": -16.0}, gains  # under the voice
assert abs(next(s for s in e["sfx"] if s["name"] == "thud")["at"] - 13.5) < 0.3, "the thud lands on the punchline"
# effects go on the LOUDEST moment after the setup, not the last mild peak; no scratch without a real yell
two_w = words_at(calm + [("NO", 9.0), ("WAY", 9.3), ("okay", 17.0), ("anyway", 17.4)])
two_a = tone_audio(dur, [(9.2, 0.6, 0.9), (17.2, 0.5, 0.5)])
t2 = director.build_edl("chat_roast", two_w, director.energy(two_a), dur)
assert abs(next(s for s in t2["sfx"] if s["name"] == "thud")["at"] - 9.2) < 0.3, t2["sfx"]
assert t2["cuts"][-1]["end"] >= 17.4, "the clip still runs to its last peak"
early_a = tone_audio(dur, [(3.0, 0.6, 0.9), (13.5, 0.6, 0.6)])  # the yell is in the setup; after it only a mild peak (~0.67)
assert "record_scratch" not in [s["name"] for s in director.build_edl("chat_roast", roast_w, director.energy(early_a), dur)["sfx"]]
assert e["font"] == "Arial Black" and set(e["word_tones"]) <= {"punch", "roast"}
assert e["word_tones"][roast_w.index(next(w for w in roast_w if w["word"] == "WHAT"))] == "roast"  # the yell: cyan
assert e["word_tones"][roast_w.index(next(w for w in roast_w if w["word"] == "house"))] == "punch"  # calm talk: yellow

# unfiltered_chaos: yelling everywhere
chaos_w = words_at([(w, 0.3 + i * 0.35) for i, w in enumerate(("yo " * 50).split())])
chaos_a = tone_audio(dur, [(2, 0.4, 0.9), (6, 0.4, 0.95), (10, 0.4, 0.9), (14, 0.4, 0.92), (18, 0.4, 0.9)])
env = director.energy(chaos_a)
assert director.classify(chaos_w, env, dur) == "unfiltered_chaos"
e = director.build_edl("unfiltered_chaos", chaos_w, env, dur)
assert len(e["takeovers"]) == 4 and {"trauma_shake", "rgb_split"} <= {f["type"] for f in e["filters"]}
assert {s["name"] for s in e["sfx"]} == {"thud", "whoosh"} and e["font"] == "Impact"  # no meme SFX pile-up
assert sum(f["type"] == "trauma_shake" for f in e["filters"]) == 5 and still_max(e) <= director.STILL_MAX + 0.01
gaps = [b["start"] - a["end"] for a, b in zip(e["cuts"], e["cuts"][1:])]
assert all(g > 0 for g in gaps)  # jump cuts actually remove time

# horror_jumpscare: quiet tension, one scream, cut right after it
horror_w = words_at([(w, 1 + i * 1.2) for i, w in enumerate("did you hear that".split())] + [("AHHH", 15.0)])
horror_a = tone_audio(dur, [(15.1, 0.3, 0.95)])
env = director.energy(horror_a)
assert director.classify(horror_w, env, dur) == "horror_jumpscare"
e = director.build_edl("horror_jumpscare", horror_w, env, dur)
p = director.peaks(env)[0]
assert abs(e["cuts"][-1]["end"] - (p + director.SCREAM_CUT)) < 0.01, e["cuts"]  # sharp cut ~0.4 s after the scream
assert {"vignette", "flash", "trauma_shake"} <= {f["type"] for f in e["filters"]} and e["takeovers"]
assert not e["hook_teaser"]["enabled"], "never show the scare first"
flash = next(f for f in e["filters"] if f["type"] == "flash")
assert flash["end"] - flash["start"] < 0.1  # ~2 frames

# confident_fail: bragging, then "i'm dead"
fail_w = words_at(calm[:8] + [("watch", 9.0), ("this", 9.3), ("oh", 11.8), ("no", 12.0), ("i'm", 12.3), ("dead", 12.5)])
fail_a = tone_audio(dur, [(12.0, 0.4, 0.9)])
env = director.energy(fail_a)
assert director.classify(fail_w, env, dur) == "confident_fail"
e = director.build_edl("confident_fail", fail_w, env, dur)
assert e["speed"] and e["speed"][0]["factor"] == 1.15 and e["speed"][0]["end"] <= director.peaks(env)[0] + 0.01
assert [s["name"] for s in e["sfx"]][:2] == ["record_scratch", "thud"] and "sad_violin" not in str(e["sfx"])
gray = next(f for f in e["filters"] if f["type"] == "grayscale")
assert abs(gray["start"] - director.peaks(env)[0]) < 0.01
assert fail_w[-1]["end"] <= e["cuts"][-1]["end"] <= fail_w[-1]["end"] + 0.3, e["cuts"]  # the spoken aftermath, not silence
tones = e["word_tones"]
assert tones[fail_w.index(fail_w[2])] == "punch" and tones[fail_w.index(fail_w[-3])] == "roast"  # "no" at the fail: cyan

# awkward_irl: slow talk, nothing loud -> push-ins keep it moving, one thud
irl_w = words_at([(w, 0.5 + i * 1.6) for i, w in enumerate("so um i guess we just wait here then".split())])
irl_a = (0.2 + 0.02 * np.sin(np.arange(int(dur * SR)) / SR)).astype(np.float32)
env = director.energy(irl_a)
assert director.classify(irl_w, env, dur) == "awkward_irl"
e = director.build_edl("awkward_irl", irl_w, env, dur)
assert len(e["cuts"]) > 1 and still_max(e) <= director.STILL_MAX + 0.01  # silences cut; the camera still moves
assert [s["name"] for s in e["sfx"]] == ["thud"] and e["font"] == "Arial Black"
CAM = {"layout": "split_stack", "cam_box": (0.0, 0.1, 0.3, 0.4), "track": [], "source_captions": False}
assert director.direct(irl_w, irl_a, dur, framing=CAM)["layout"] == "blur_fill"  # IRL stays clean

for s in director.SCENARIOS:  # every scenario stays inside QC's length window
    e = director.build_edl(s, roast_w, director.energy(roast_a), dur)
    assert director.MIN_LEN - 0.01 <= renderer.timeline(e)[1] <= director.MAX_LEN + 1, s
d = director.direct(roast_w, roast_a, dur, framing=CAM)
assert d["layout"] == "split_stack" and d["directed_by"] == "heuristic" and len(d["hashtags"]) == 4
assert d["zooms"] and d["zoom_anchor"] == [0.5, 0.5, 1.35] and not d["source_captions"]
track = [(1.0, 0.4), (10.0, 0.6)]
FC = {"layout": "face_crop", "track": track, "source_captions": False, "zoom": [0.5, 0.2, 1.3], "zoom_full": [0.5, 0.3, 1.08]}
fcrop = director.direct(roast_w, roast_a, dur, framing=FC)
assert fcrop["layout"] == "face_crop" and fcrop["crop_track"] == [[1.0, 0.4], [10.0, 0.6]], fcrop["layout"]
assert fcrop["zooms"] and fcrop["takeovers"] == [] and fcrop["zoom_anchor"] == [0.5, 0.2, 1.3], "head-safe zooms on the column"
theirs = director.direct(roast_w, roast_a, dur, framing=dict(FC, source_captions=True))
assert theirs["layout"] == "blur_fill" and theirs["source_captions"] and theirs["crop_track"] == [], "their captions stay whole"
assert theirs["zoom_anchor"] == [0.5, 0.3, 1.08], "blur_fill instead of the column: the whole frame's head-safe zoom"
twice = director.direct(roast_w, roast_a, dur, framing=dict(CAM, source_captions=True))
assert twice["layout"] == "blur_fill", "split_stack would show the stream's captions in both panels"
assert director.direct(roast_w, roast_a, dur)["layout"] == "blur_fill"  # no framing info: the safe full frame

# --- word timing: acoustic onset snap (Whisper starts words early, in the silence before them) -----------------
onset_env = np.full(40, 0.02)
onset_env[10:] = 1.0  # the voice starts at 0.50 s
snapped = pipeline.snap_words_to_audio([{"word": "a", "start": 0.35, "end": 1.0}, {"word": "b", "start": 0.1, "end": 1.0},
                                        {"word": "c", "start": 0.45, "end": 0.53}, {"word": "d", "start": 0.6, "end": 0.9}], onset_env)
assert [w["start"] for w in snapped] == [0.5, 0.1, 0.45, 0.6], snapped  # snapped; >0.25 s away: kept; guard; already voiced
assert [w["end"] for w in snapped] == [1.0, 1.0, 0.53, 0.9]  # voiced ends untouched
stretch_env = np.full(80, 0.02)
stretch_env[20:26] = 1.0  # "my" is heard 1.0-1.3 s; Whisper says 1.0-3.0 (stretched over the silence after it)
stretched = pipeline.snap_words_to_audio([{"word": "my", "start": 1.0, "end": 3.0}, {"word": "uh", "start": 3.2, "end": 3.25}], stretch_env)
assert stretched[0]["end"] == 1.3 and stretched[1] == {"word": "uh", "start": 3.2, "end": 3.25}, stretched  # >= min_len kept

# --- cut boundaries: never inside a word; reactions ring out (not for horror's hard cut) ------------------------
ws_words = words_at([("setup", 0.95), ("punch", 1.9)], 0.3)  # 0.95-1.25, 1.9-2.2
assert director.word_safe([(1.0, 2.05)], ws_words, 10) == [(0.87, 2.34)]  # 0.08 before, 0.14 after
assert director.word_safe([(1.0, 2.05)], ws_words, 10, keep_last_end=True) == [(0.87, 2.05)]
tail_env = np.zeros(200)
tail_env[:80] = 0.9  # loud (laughing) until 4.0 s
assert director.tail_guard([(0.0, 2.0)], tail_env, 10) == [(0.0, 3.2)]  # +1.2 s cap
tail_env[:50] = 0.9
tail_env[50:] = 0.1
assert director.tail_guard([(0.0, 2.0)], tail_env, 10) == [(0.0, 2.5)]  # stops once the laugh decays
# regression (froze a calibration batch): a cut ending within 0.005 s of a clip that is loud to its last sample
assert director.tail_guard([(0.0, 26.773)], np.ones(536), 26.7733125) == [(0.0, 26.773)]
assert director.fit_length([(0.0, director.MAX_LEN + 1e-12)], 99) == [{"start": 0.0, "end": round(director.MAX_LEN, 2)}]
long_dur = 32.0  # chat_roast on a long clip with a late outburst: flash-forward teaser
long_w = words_at([(w, 0.5 + i * 0.9) for i, w in enumerate(("so chat i was thinking " * 6).split())] + [("WHAT", 27.0), ("BRO", 27.3)])
long_a = tone_audio(long_dur, [(27.2, 0.5, 0.9)])
e = director.build_edl("chat_roast", long_w, director.energy(long_a), long_dur)
hook = e["hook_teaser"]
assert hook["enabled"] and director.TEASER_MIN - 0.02 <= hook["end"] - hook["start"] <= director.TEASER_LEN + 0.01 and hook["start"] <= 27.2 <= hook["end"], hook
assert any(c["start"] - 0.11 <= hook["start"] and hook["end"] <= c["end"] + 0.11 for c in e["cuts"]), "the teaser is the climax's own sound (+ <= 0.1 s)"
assert all(any(c["start"] <= w["start"] and w["end"] <= c["end"] for c in e["cuts"]) or
           all(w["end"] <= c["start"] or w["start"] >= c["end"] for c in e["cuts"]) for w in long_w), "a word was cut in half"
assert director.build_edl("chat_roast", roast_w, director.energy(roast_a), dur)["hook_teaser"]["enabled"]  # 20 s, late yell
fe = director.build_edl("confident_fail", fail_w, director.energy(fail_a), dur)
assert fe["freeze_end"] == director.FREEZE_LEN and renderer.timeline(fe)[1] >= director.MIN_LEN

# --- streamer key (Ollama context / gaming fixes) ---------------------------------------------------------------
twitch_m = {"twitch": {"broadcaster_name": "CaseOh_"}}
assert pipeline.streamer_key(twitch_m) == "caseoh_" and pipeline.streamer_key({"channel": "@SomeChannel"}) == "somechannel"
assert pipeline.streamer_key({"video_id": "x"}) is None
assert not hasattr(pipeline, "pick_facecam") and not hasattr(renderer, "detect_facecam_box"), "the old webcam guesser is gone"

# --- Ollama creative: structured JSON applied; nonsense falls back --------------------------------------------
class Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False

good = {"scenario": "horror_jumpscare", "social_title": "He was NOT ready", "hashtags": ["#minecraft", "#scary"]}
opener = lambda req, timeout=None: Resp(json.dumps({"response": json.dumps(good)}).encode())
d = director.direct(roast_w, roast_a, dur, model="m", opener=opener)
rule = director.classify(roast_w, director.energy(roast_a), dur)
assert d["scenario"] == rule != "horror_jumpscare"  # the model never picks the scenario (it collapsed to one per wording)
assert d["directed_by"] == "ollama" and d["hashtags"][:2] == ["#minecraft", "#scary"]
assert len(d["hashtags"]) == 4 and d["title"] == "He was NOT ready"
echoed = {"scenario": "chat_roast", "social_title": "x", "hashtags": ["#a", "#b", "#tag3", "#caseoh"]}
d = director.direct(roast_w, roast_a, dur, model="m", opener=lambda req, timeout=None: Resp(json.dumps({"response": json.dumps(echoed)}).encode()))
assert d["hashtags"][0] == "#caseoh" and "#a" not in d["hashtags"] and len(d["hashtags"]) == 4, d["hashtags"]  # placeholders dropped
sent = {}
def capture(req, timeout=None):
    sent.update(json.loads(req.data))
    return Resp(json.dumps({"response": json.dumps(good)}).encode())
director.direct(roast_w, roast_a, dur, model="m", opener=capture, meta={"broadcaster_name": "CaseOh_", "title": "he lost it",
                "pillars": {"hook_strength": 8, "comedy_payoff": 9}, "judge_reason": "huge payoff"})
assert "Streamer: CaseOh_" in sent["prompt"] and "hook_strength 8, comedy_payoff 9" in sent["prompt"], sent["prompt"]
assert "Judge said: huge payoff" in sent["prompt"] and "loud peak" in sent["prompt"] and '"fit"' in sent["prompt"]
fit = {"fit": {"chat_roast": 3, "unfiltered_chaos": 9, "horror_jumpscare": 2, "confident_fail": 9, "awkward_irl": 1}, "social_title": "x"}
d = director.direct(roast_w, roast_a, dur, model="m", opener=lambda req, timeout=None: Resp(json.dumps({"response": json.dumps(fit)}).encode()))
assert d["scenario"] == rule and d["scenario_fit"]["confident_fail"] == 9 and d["title"] == "x", d  # fit is advisory only
junk = {"fit": {"chat_roast": "0-10", "awkward_irl": True}, "social_title": "y"}
assert director.direct(roast_w, roast_a, dur, model="m", opener=lambda req, timeout=None: Resp(json.dumps({"response": json.dumps(junk)}).encode()))["scenario_fit"] is None
bad = lambda req, timeout=None: Resp(b'{"response": "{\\"scenario\\": \\"dance_battle\\"}"}')
assert director.direct(roast_w, roast_a, dur, model="m", opener=bad)["directed_by"] == "heuristic"

# --- kinetic captions: 1-3 word bursts, bounce, tone colours, font, placement ---------------------------------
cut_edl = {"cuts": [{"start": 0, "end": 4}, {"start": 6, "end": 10}], "font": "Impact",
           "hook_teaser": {"enabled": False, "start": 0, "end": 0, "banner_text": ""}}
ws = words_at([("let's", 1.0), ("go", 1.3), ("chat", 1.6), ("gone", 4.5), ("NO!", 7.0), ("why", 7.4), ("bro", 7.7), ("okay", 8.0)], 0.25)
cut_edl["word_tones"] = ["hype", "hype", "roast", "punch", "rage", "punch", "roast", "punch"]
renderer.write_ass(cut_edl, ws, tmp / "a.ass")
ass = (tmp / "a.ass").read_text(encoding="utf-8")
style = next(l for l in ass.splitlines() if l.startswith("Style: Pop"))
assert style.startswith("Style: Pop,Impact,") and ",4,3,2,140,140,660," in style  # outline 4, shadow 3, side margins clear of the app buttons, MarginV 660
events = [l for l in ass.splitlines() if l.startswith("Dialogue: 0,")]
assert "GONE" not in ass  # spoken in the removed gap
assert any(r"\c&H0044FF00&" + renderer.POP + "}LET'S" in l for l in events)  # a SETTINGS theme tone still works
assert any(r"\c&H00FFFF00&" + renderer.POP + "}CHAT" in l for l in events)  # cyan &H00FFFF00, popping 118% -> 100%
assert renderer.TONE_COLORS["punch"] == "&H0000FFFF&" and renderer.POP == r"\fscx118\fscy118\t(0,70,\fscx100\fscy100)"
no = next(l for l in events if r"}NO!" in l)
assert r"\c&H002A2AFF&\fnImpact" + renderer.POP + "}NO!" in no and no.startswith("Dialogue: 0,0:00:05.00")  # remapped 7.0 -> 5.0
assert all(l.count(renderer.POP) == 1 for l in events), "exactly the active word pops, in every event"
for l in events:  # never more than 3 words on screen (2-3 word lines)
    assert len(re.sub(r"\{[^}]*\}", "", l.split(",,", 1)[1]).split()) <= 3, l
plain = [dict(w, ostart=w["start"], oend=w["end"]) for w in words_at([(f"w{i}", i * 0.2) for i in range(9)], 0.15)]
assert all(len(g) <= 3 and g[-1]["oend"] - g[0]["ostart"] <= 1.3 for g in renderer.chunk(plain))

# never two captions on screen at once (incl. across a jump cut and a sped-up segment); no {\r} scale resets
dense = words_at([(f"w{i}", 0.4 + i * 0.27) for i in range(30)], 0.26)  # tails would overlap the next burst
dense_edl = {"cuts": [{"start": 0, "end": 4.2}, {"start": 4.25, "end": 9}], "speed": [{"start": 4.25, "end": 9, "factor": 1.15}],
             "font": "Arial Black", "hook_teaser": {"enabled": False}, "word_tones": ["rage" if i % 4 == 0 else "punch" for i in range(30)]}
renderer.write_ass(dense_edl, dense, tmp / "d.ass")
ev = [l for l in (tmp / "d.ass").read_text(encoding="utf-8").splitlines() if l.startswith("Dialogue: 0,")]
secs = lambda s: int(s.split(":")[0]) * 3600 + int(s.split(":")[1]) * 60 + float(s.split(":")[2])
times = [(secs(l.split(",")[1]), secs(l.split(",")[2])) for l in ev]
assert len(times) > 20 and all(b > a for a, b in times), times
assert all(b1 <= a2 for (_, b1), (a2, _) in zip(times, times[1:])), "overlapping captions"
assert all(",Pop,,0,0,0,," in l for l in ev)  # one style: Alignment 2 + MarginV 660 everywhere, bounce scales in place
assert not any(r"{\r}" in l for l in ev)
rage_lines = [l for l in ev if r"\c&H002A2AFF&\fnImpact" + renderer.POP in l]
assert rage_lines and all(r"{\c&H00FFFFFF&\fscx100\fscy100\fnArial Black}" in l for l in rage_lines)  # base font + 100% after it

# speed ramps remap captions and effects
sp = {"cuts": [{"start": 0, "end": 10}], "speed": [{"start": 0, "end": 5, "factor": 1.25}],
      "hook_teaser": {"enabled": False}}
segs, length = renderer.timeline(sp)
assert abs(length - 9.0) < 1e-9 and renderer.to_out(5.0, segs) == 4.0 and renderer.to_out(7.0, segs) == 6.0
assert renderer.to_src(2.0, segs) == (0, 2.5)

# --- framing: real face detections -> split_stack / face_crop / blur_fill (framing.decide is pure) ---------------
A = 16 / 9
times = [i * 0.5 for i in range(20)]
cam_face = (0.02, 0.24, 0.07, 0.15, 0.9)  # CaseOh-style webcam, top-left (measured on a real clip)
r = framing.decide(times, [[cam_face]] * 20, A)
assert r["layout"] == "split_stack", r
x, y, w, h = r["cam_box"]
assert x <= 0.02 and x + w >= 0.09 and y <= 0.24 - 0.6 * 0.15 and y + h >= 0.39, ("whole head (hair, headset) + chin inside the crop", r["cam_box"])
assert abs((w * A) / h - 1080 / 690) < 0.01 and 0 <= x and 0 <= y and x + w <= 1 and y + h <= 1, r["cam_box"]
edge = framing.cam_box((0.0, 0.0, 0.1, 0.2), A, 1080 / 690)
assert edge[0] == 0 and edge[1] == 0, "clamped into the frame at a corner, never off it"
lean = [[(0.1, 0.6, 0.11, 0.2, 0.9)]] * 14 + [[(0.1, 0.48, 0.11, 0.2, 0.9)]] * 2 + [[(0.1, 0.66, 0.11, 0.2, 0.9)]] * 4
r = framing.decide(times, lean, A)  # a corner webcam whose streamer leans up (2 samples) and down (4)
x, y, w, h = r["cam_box"]
assert r["layout"] == "split_stack" and y <= 0.48 - 0.6 * 0.2 and y + h >= 0.86, ("leaning head and chin stay inside", r["cam_box"])
crowd = [[(0.12, 0.35, 0.09, 0.19, 0.9), (0.37, 0.45, 0.06, 0.14, 0.9), (0.77, 0.5, 0.12, 0.27, 0.9)]] * 20  # 1249093115
assert framing.decide(times, crowd, A)["layout"] == "blur_fill", "people spread across the room: no split, no column"


def head_safe(anchor, faces, keep_bottom=None):
    """At the max zoom, the window (anchor fixed) still holds every face with hair / chin, and the caption line."""
    ax, ay, z = anchor
    x0, x1, y0, y1 = ax - ax / z, ax + (1 - ax) / z, ay - ay / z, ay + (1 - ay) / z
    ok = all(x0 <= f[0] - 0.2 * f[2] + 1e-3 and f[0] + 1.2 * f[2] <= x1 + 1e-3 and y0 <= f[1] - 0.6 * f[3] + 1e-3
             and f[1] + 1.15 * f[3] <= y1 + 1e-3 for f in faces)
    return ok and (keep_bottom is None or keep_bottom <= y1 + 1e-3)


crowd_zoom = framing.decide(times, crowd, A)["zoom"]
assert crowd_zoom[2] < 1.25 and head_safe(crowd_zoom, crowd[0]), ("people wide apart: only as much zoom as keeps them all", crowd_zoom)
solo_face = [(0.4, 0.3, 0.12, 0.25, 0.9)]
z1 = framing.zoom_room(solo_face)
assert z1[2] == framing.MAX_ZOOM and head_safe(z1, solo_face), z1  # full punch, the whole head kept
z2 = framing.zoom_room(solo_face, keep_bottom=0.97)
assert z2[2] < z1[2] and head_safe(z2, solo_face, 0.97), ("the stream's caption line stays in frame", z2)
assert framing.zoom_room([]) == [0.5, 0.5, framing.MAX_ZOOM], "gameplay only: centre, full punch"
corner = framing.zoom_room([(0.0, 0.05, 0.3, 0.3, 0.9)])  # a head hugging the top-left corner
assert corner[:2] == [0.0, 0.0] and corner[2] == framing.MAX_ZOOM, ("zooms from that corner: nothing more is lost", corner)
assert framing.zoom_room([(0.0, 0.1, 0.95, 0.3, 0.9)])[2] == 1.0, "a face wider than the panel: no zoom"
cam_r = framing.decide(times, [[cam_face]] * 20, A)
cam_panel = [((cam_face[0] - cam_r["cam_box"][0]) / cam_r["cam_box"][2], (cam_face[1] - cam_r["cam_box"][1]) / cam_r["cam_box"][3],
              cam_face[2] / cam_r["cam_box"][2], cam_face[3] / cam_r["cam_box"][3])]
assert cam_r["zoom"][2] >= 1.22 and head_safe(cam_r["zoom"], cam_panel), ("webcam panel: real punch-ins, head kept", cam_r["zoom"])
assert framing.decide(times, [[]] * 20, A)["layout"] == "blur_fill"
assert framing.decide(times, [[cam_face]] * 6 + [[]] * 14, A)["layout"] == "blur_fill", "a face in < half the samples"
wander = [[(0.05 + 0.04 * i, 0.2, 0.07, 0.15, 0.9)] for i in range(20)]  # a moving game character, not a webcam
assert framing.decide(times, wander, A)["layout"] != "split_stack"
host = [[(0.30 + 0.01 * (i % 5), 0.4, 0.2, 0.4, 0.9)] for i in range(20)]  # a big face-cam host (Just Chatting)
r = framing.decide(times, host, A)
col = (9 / 16) / A
assert r["layout"] == "face_crop" and len(r["track"]) == 10, r
assert all(col / 2 <= x <= 1 - col / 2 and abs(x - 0.42) < 0.05 for _, x in r["track"]), r["track"]  # on the face, in frame
assert framing.smooth([None, 1.0, None, None, 3.0], k=1) == [1.0, 1.0, 1.0, 3.0, 3.0]
# burned-in captions: a wide centred line low in the frame whose width keeps changing; a fixed HUD line is not one
cap = [[(0.5, 0.9, wd, 0.05)] for wd in (0.17, 0.21, 0.23, 0.15, 0.19, 0.29)]
hud = [[(0.5, 0.9, 0.2, 0.05), (0.1, 0.5, 0.3, 0.05)]] * 6
chat = [[(0.1, 0.8, 0.25, 0.04)]] * 6
assert framing.has_source_captions(cap) and not framing.has_source_captions(hud) and not framing.has_source_captions(chat)
assert not framing.has_source_captions(cap[:3]), "too few samples to call it"
pattern = tmp / "pattern.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=3", "-c:v", "libx264",
                "-preset", "ultrafast", str(pattern)], check=True)
ts, fr, aspect = framing.sample_frames(pattern)
assert len(fr) == 6 and fr[0].shape == (360, 640, 3) and abs(aspect - A) < 0.01 and ts[0] == 0.25, (len(fr), ts)
assert framing.detect_faces(fr) == [[]] * 6, "real YuNet: no face in a test pattern"
assert len(framing.caption_boxes(fr[:2])) == 2  # the PP-OCR text detector loads and runs
res = framing.analyze(pattern, words_at([("hi", 0.2)]))
assert res["layout"] == "blur_fill" and res["source_captions"] is False, res
assert framing.analyze(pattern, [], (0.0, 0.0, 0.3, 0.4))["layout"] == "split_stack", "a hand-set box forces split_stack"
# the face-crop track: piecewise-linear over OUTPUT time, held before the first / after the last keyframe
segs, _ = renderer.timeline({"cuts": [{"start": 0, "end": 4}, {"start": 6, "end": 10}], "hook_teaser": {"enabled": False}})
ex = renderer.track_expr([(1.0, 0.4), (3.0, 0.6), (5.0, 0.9), (8.0, 0.5)], segs)  # 5.0 is cut out: dropped
f = lambda tt: eval(ex, {"__builtins__": {}, "gte": lambda a, b: float(a >= b), "lt": lambda a, b: float(a < b), "t": tt})
assert abs(f(0.0) - 0.4) < 1e-6 and abs(f(2.0) - 0.5) < 1e-6 and abs(f(3.0) - 0.6) < 1e-6 and abs(f(9.0) - 0.5) < 1e-6
assert abs(f(4.0) - 0.5667) < 1e-3, f(4.0)  # 3.0 -> 6.0 output s interpolated: 0.6 + (0.5-0.6)/3

# --- loudness repair: a too_quiet render gets audio-only -14 LUFS passes before anything else ---------------------
quiet_mp4 = tmp / "quiet.mp4"  # a whisper-quiet mic with one sharp peak (the 912079449_1 shape)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=30:d=6", "-f", "lavfi", "-i",
                "aevalsrc='0.01*sin(2*PI*220*t)+0.6*sin(2*PI*900*t)*between(t,3,3.03)':s=48000:d=6", "-c:v", "libx264",
                "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(quiet_mp4)], check=True)
level = lambda p: float(re.search(r"mean_volume:\s*(-?[\d.]+)", subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(p), "-af",
                    "volumedetect", "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace").stderr).group(1))
before, frames_before = level(quiet_mp4), renderer.probe(quiet_mp4)
pipeline.boost_loudness(quiet_mp4)
after = level(quiet_mp4)
assert after > before + 10 and abs(renderer.probe(quiet_mp4)["duration"] - frames_before["duration"]) < 0.05, (before, after)
assert not quiet_mp4.with_name("quiet.loud.mp4").exists(), "repaired in place"
# the loop: QC says too_quiet twice, then passes -> two passes, no re-render; still quiet after 2 -> stops (fail)
calls = {"render": 0, "boost": 0}
qc_seq = iter([["too_quiet"], ["too_quiet"], []])
real = (renderer.render, pipeline.boost_loudness, qc.check)
renderer.render = lambda *a, **k: calls.__setitem__("render", calls["render"] + 1)
pipeline.boost_loudness = lambda p: calls.__setitem__("boost", calls["boost"] + 1)
qc.check = lambda p, **k: (lambda probs: {"ok": not probs, "problems": probs, "duration": 9.0, "mean_db": -21.0, "black": []})(next(qc_seq))
try:
    with quiet() as qlog:
        _, rep = pipeline.render_until_pass(tmp / "x.mp4", {"cuts": [{"start": 0, "end": 9}], "hook_teaser": {"enabled": False}}, [], tmp / "qwork")
    assert rep["ok"] and calls == {"render": 1, "boost": 2}, (calls, rep)
    assert qlog.getvalue().count("loudness repair") == 2, qlog.getvalue()
    qc_seq = iter([["too_quiet"]] * 3)
    calls.update(render=0, boost=0)
    with quiet():
        _, rep = pipeline.render_until_pass(tmp / "x.mp4", {"cuts": [{"start": 0, "end": 9}], "hook_teaser": {"enabled": False}}, [],
                                            tmp / "qwork", max_retries=0)
    assert not rep["ok"] and rep["problems"] == ["too_quiet"] and calls == {"render": 1, "boost": 2}, (calls, rep)
finally:
    renderer.render, pipeline.boost_loudness, qc.check = real

# --- post card: the approved clip's 14B title angles / 5 hashtags / pinned line beat the director's title ------------
base = {"title": "director title", "hashtags": ["#a", "#b", "#c", "#d"], "social_title": "director title #a #b #c #d"}
card = pipeline.post_card({"title_angles": ["Angle one that is long enough to post right now", "Angle two"],
                           "hashtags": ["#caseoh", "#funny", "#gaming", "#shorts", "#twitch"], "pinned_comment": "Who else?"}, base)
assert card["title"] == "Angle one that is long enough to post right now" and len(card["hashtags"]) == 5, card
assert card["social_title"] == "Angle one that is long enough to post right now #caseoh #funny #gaming #shorts #twitch", card
assert pipeline.post_card({"title_angles": None, "hashtags": None, "pinned_comment": None}, base) == {}, "Ollama was off: director's stays"
txt = pipeline.card_text(dict(base, **card))
assert f"1. {card['title']}  ({len(card['title'])} chars)" in txt and "PINNED COMMENT: Who else?" in txt, txt

# --- QC repair logic -------------------------------------------------------------------------------------------
two = {"cuts": [{"start": 0, "end": 5}, {"start": 10, "end": 15}], "hook_teaser": {"enabled": False}}
fixed = pipeline.repair(two, {"problems": ["black"], "black": [(6.0, 6.6)]})
assert fixed["cuts"] == [{"start": 0, "end": 5}, {"start": 10, "end": 10.95}, {"start": 11.65, "end": 15}], fixed["cuts"]
long_edl = {"cuts": [{"start": 0, "end": 40}, {"start": 50, "end": 80}], "hook_teaser": {"enabled": True, "start": 60, "end": 63.5}}
fixed = pipeline.repair(long_edl, {"problems": ["too_long"], "duration": 73.5})
assert not fixed["hook_teaser"]["enabled"] and renderer.timeline(fixed)[1] <= 57.5 and fixed["cuts"][-1] == {"start": 50, "end": 80}
assert pipeline.repair(two, {"problems": ["too_quiet"], "black": []}) is None
# too_short: the real stuck case (4.3 s over 5 cuts of a 55 s Twitch clip) grows back toward the source to 9 s
stuck = {"cuts": [{"start": 13.15, "end": 14.23}, {"start": 40.23, "end": 40.88}, {"start": 49.24, "end": 49.97},
                  {"start": 50.68, "end": 52.13}, {"start": 54.65, "end": 55.08}], "hook_teaser": {"enabled": False}}
grown = pipeline.repair(stuck, {"problems": ["too_short"], "black": []}, src_duration=55.08)
glen = director.total([(c["start"], c["end"]) for c in grown["cuts"]])
assert abs(glen - 9.0) < 0.05 and grown["cuts"][-1]["end"] == 55.08 and grown["cuts"][0]["start"] >= 0, grown["cuts"]
assert pipeline.repair(stuck, {"problems": ["too_short"], "black": []}) is None  # no source length: nothing to grow into
# black that would leave < 8.5 s: the footage itself is dark, so it stays (QC then skips black) instead of shredding
dark = {"cuts": [{"start": 0, "end": 10}], "hook_teaser": {"enabled": False}}
kept = pipeline.repair(dark, {"problems": ["black"], "black": [(1.0, 9.0)]}, src_duration=20)
assert kept["keep_dark"] and kept["cuts"] == dark["cuts"], kept
assert director.fit_length([(1.0, 2.0)], 3.0) == [{"start": 0.0, "end": 3.0}]  # a 3 s source keeps all of itself
assert director.fit_length([(10.0, 12.0)], 30.0) == [{"start": 3.5, "end": 12.0}]  # 2 s -> 8.5 s, payoff end kept

# --- pacing: words + VAD speech + loud reactions kept, every silence > 0.35 s cut, 0.1 s pad ---------------------
gw = [{"word": "a", "start": 1.0, "end": 1.5}, {"word": "b", "start": 1.8, "end": 2.0}, {"word": "c", "start": 5.0, "end": 5.5}]
assert director.keep_spans(0, 10, gw) == [(0.9, 2.1), (4.9, 5.6)], director.keep_spans(0, 10, gw)  # 0.3 s gap kept, 3 s cut
assert director.keep_spans(0, 10, gw, speech=[(2.0, 3.0)]) == [(0.9, 3.1), (4.9, 5.6)], "VAD speech (a laugh) is kept"
renv = np.full(200, 0.05)
renv[int(3.1 / director.HOP):int(4.8 / director.HOP)] = 0.9  # a wordless scream from 3.1 to 4.8 s
assert director.keep_spans(0, 10, gw, env=renv) == [(0.9, 2.1), (3.0, 5.6)], director.keep_spans(0, 10, gw, env=renv)
assert director.loud_runs(np.full(200, 0.6)) == [], "a constant game / music bed is not a reaction"
assert director.keep_spans(2.5, 4.0, gw) == [(2.5, 4.0)]  # nothing audible in range: keep it (scenario logic decides)
assert director.vad_speech(np.zeros(16000, np.float32)) == [], "Silero VAD: silence has no speech"
roast_cuts = [(c["start"], c["end"]) for c in director.build_edl("chat_roast", roast_w, director.energy(roast_a), dur)["cuts"]]
assert 0.5 - director.PAD - 0.09 <= roast_cuts[0][0] <= 0.5, ("opens on the first word", roast_cuts)

# cold open: a loud climax in the second half of a >= 16 s clip plays first (1.2-1.5 s, punched in, whoosh + flash)
cold_w = words_at([(w, 0.5 + i * 0.6) for i, w in enumerate(("so chat this guy said " * 3).split())] + [("NO", 15.0), ("WAY!", 15.3)])
cold_a = tone_audio(18.0, [(15.2, 0.6, 0.95)])
ce = director.build_edl("chat_roast", cold_w, director.energy(cold_a), 18.0)
hook = ce["hook_teaser"]
T = hook["end"] - hook["start"]
assert hook["enabled"] and director.TEASER_MIN - 0.01 <= T <= director.TEASER_LEN + 0.01 and hook["start"] <= 15.2 <= hook["end"], hook
outs = [x for k in ("zooms", "filters", "sfx") for x in ce[k] if x.get("out")]
assert {x.get("style") or x.get("type") or x.get("name") for x in outs} == {"snap", "flash", "whoosh"}, outs
assert next(z for z in ce["zooms"] if z.get("out"))["scale"] == director.TEASER_ZOOM
assert abs(next(f for f in ce["filters"] if f.get("out"))["start"] - (T - 0.04)) < 0.02, "the flash sits on the hard cut"
segs_on, _ = renderer.timeline(ce)
assert renderer.zoom_expr([z for z in ce["zooms"] if z.get("out")], segs_on)[0].startswith("between(it,0.000,")
segs_off, _ = renderer.timeline(dict(ce, hook_teaser=dict(hook, enabled=False)))
assert renderer.zoom_expr([z for z in ce["zooms"] if z.get("out")], segs_off) == [], "teaser dropped (QC repair): its effects go"
assert not director.build_edl("chat_roast", cold_w, director.energy(tone_audio(18.0, [(4.0, 0.6, 0.95)])), 18.0)["hook_teaser"]["enabled"]
assert not director.build_edl("chat_roast", roast_w, director.energy(roast_a), 15.0)["hook_teaser"]["enabled"], "< 16 s: no teaser"

# --- real render: split_stack + takeover + every effect passes QC (with an over-limit first try) ----------------
src = tmp / "src.mp4"
expr = "0.3*sin(2*PI*220*t)*(0.6+0.4*sin(2*PI*0.5*t))+0.5*sin(2*PI*700*t)*between(t,9,9.6)"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=16", "-f", "lavfi",
                "-i", f"aevalsrc='{expr}':s=48000:d=16", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                "-shortest", str(src)], check=True)
# pre-render sanity: fragments with no action moment are skipped before any render; a loud spike keeps them
frag_w = words_at([("uh", 1.0), ("m", 3.0), ("so", 6.0)])
e = director.build_edl("awkward_irl", frag_w, director.energy(tone_audio(dur, [])), dur)
assert e["skip_render"] and "clear word" in e["skip_reason"], e.get("skip_reason")
assert not director.build_edl("awkward_irl", frag_w, director.energy(tone_audio(dur, [(6.2, 0.4, 0.95)])), dur).get("skip_render")
assert not director.build_edl("chat_roast", roast_w, director.energy(roast_a), dur).get("skip_render")
skip_job = tmp / "skip_job"
skip_job.mkdir()
(skip_job / "clip_01_5s.mp4").write_bytes(src.read_bytes())
(skip_job / "clips_manifest.json").write_text(json.dumps({"video_id": "skipme", "clips": [{"clip_id": 1, "file": "clip_01_5s.mp4"}]}))
with quiet() as log:
    r = pipeline.edit_clip(skip_job / "clips_manifest.json", 1, tmp / "skip_out", model=None,
                           encoder=renderer.detect_encoder(), transcribe=lambda audio, m, d: words_at([("uh", 1.0), ("m", 2.0)]))
assert r["skipped"] and not r["ok"] and "Skipped skipme_1:" in log.getvalue(), log.getvalue()
assert not (tmp / "skip_out").exists(), "no render, no .work folder"

real = {"scenario": "chat_roast", "layout": "split_stack", "font": "Arial Black",
        "hook_teaser": {"enabled": True, "start": 9.0, "end": 11.0, "banner_text": "WAIT FOR IT..."},
        "cuts": [{"start": 0.5, "end": 6.0}, {"start": 7.0, "end": 15.5}],
        "speed": [{"start": 0.5, "end": 4.0, "factor": 1.15}],
        "zooms": [{"at": 2.0, "duration": 2.0, "scale": 1.45, "style": "ramp"}, {"at": 9.0, "duration": 0.8, "scale": 1.65, "style": "snap"}],
        "filters": [{"start": 9.0, "end": 9.3, "type": "trauma_shake"}, {"start": 9.0, "end": 9.9, "type": "reaction_boost"},
                    {"start": 9.1, "end": 9.24, "type": "rgb_split"}, {"start": 9.0, "end": 9.07, "type": "flash"},
                    {"start": 12.0, "end": 14.0, "type": "grayscale"}, {"start": 1.0, "end": 3.0, "type": "vignette"}],
        "takeovers": [{"start": 8.8, "end": 10.5}], "mutes": [{"start": 8.5, "end": 8.8}],
        "sfx": [{"at": 8.5, "name": "record_scratch", "gain_db": -6}, {"at": 9.0, "name": "vine_boom", "gain_db": -6},
                {"at": 1.0, "name": "nonexistent", "gain_db": -6}],
        "word_tones": ["punch", "punch", "roast", "punch", "rage"], "bg_music": {"enabled": False, "gain_db": -22}}
spoken = words_at([("this", 1.0), ("is", 1.4), ("totally", 1.8), ("fine", 2.4), ("WHAT!", 9.1)])
limits = {"max_dur": 10.0, "min_dur": 5.0}  # tight cap so the first render must fail
cam = (0.0, 0.0, 0.3, 0.4)
with quiet() as log:
    final, report = pipeline.render_until_pass(src, real, spoken, tmp / "work", limits, renderer.detect_encoder(), cam)
assert "FAIL too_long" in log.getvalue() and "QC PASS" in log.getvalue(), log.getvalue()
assert report["ok"] and report["width"] == 1080 and report["height"] == 1920 and report["duration"] <= 10.0, report
assert -20 <= report["mean_db"] <= -10 and report["bytes"] > 500_000, report
assert not final["hook_teaser"]["enabled"] and final["cuts"][-1]["end"] == 15.5  # teaser dropped, payoff end kept
cmd = json.loads((tmp / "work" / "ffmpeg_cmd.json").read_text())
graph = cmd[cmd.index("-filter_complex") + 1]
for needle in ("vstack", "pad=1080:694", "overlay=0:0:enable=", "zoompan", "rgbashift=rh=-3:bh=3", "unsharp=5:5:1.2",
               "eq=contrast=1.28:saturation=1.35:brightness=0.03", "hue=s=0", "volume=0:enable="):
    assert needle in graph, needle
assert graph.count("subtitles=") == 1 and "drawtext" not in graph, "exactly ONE caption track, no burned-in text/filenames"
fc_edl = dict(real, layout="face_crop", crop_track=[[1.0, 0.3], [9.0, 0.7]], zooms=[], takeovers=[])
fc_cmd = renderer.build_command(src, fc_edl, spoken, tmp / "fc.mp4", tmp / "work", renderer.probe(src))
fc_graph = fc_cmd[fc_cmd.index("-filter_complex") + 1]
assert "crop=w=trunc(ih*9/32)*2:h=ih:x='clip((" in fc_graph and "vstack" not in fc_graph and "boxblur" not in fc_graph
assert fc_graph.count("subtitles=") == 1
own = renderer.build_command(src, dict(fc_edl, source_captions=True), spoken, tmp / "fc.mp4", tmp / "work", renderer.probe(src))
assert "subtitles=" not in own[own.index("-filter_complex") + 1], "the source has captions: ours are not burned on top"

# --- real render: mostly-dark footage used to be stripped to < 7 s and stuck in .work as too_short -----------------
dark_src = tmp / "dark.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=60:d=20", "-f", "lavfi",
                "-i", "aevalsrc='0.3*sin(2*PI*220*t)':s=48000:d=20", "-vf",
                "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='lt(t,12)'", "-c:v", "libx264", "-preset",
                "ultrafast", "-c:a", "aac", "-shortest", str(dark_src)], check=True)
dark_edl = {"scenario": "awkward_irl", "layout": "blur_fill", "font": "Arial Black", "hook_teaser": {"enabled": False},
            "cuts": [{"start": 4.0, "end": 18.0}], "speed": [], "zooms": [], "filters": [], "takeovers": [], "mutes": [],
            "sfx": [], "word_tones": [], "bg_music": {"enabled": False, "gain_db": -22}}
with quiet() as log:
    final, report = pipeline.render_until_pass(dark_src, dark_edl, [], tmp / "dark_work", None, renderer.detect_encoder())
assert "FAIL black" in log.getvalue() and report["ok"] and final.get("keep_dark"), (log.getvalue(), report)
assert final["cuts"] == dark_edl["cuts"] and report["duration"] >= 13.5, final["cuts"]  # nothing shredded
assert renderer.probe(tmp / "dark_work" / "short.mp4")["fps"] == 30, "60 fps source -> 30 fps short (half the filter work)"
assert "colorbalance" not in graph and "rh=-6" not in graph and "contrast=1.55" not in graph  # no muddy red crush
legacy = dict(real, filters=[{"start": 9.0, "end": 9.9, "type": "deep_fry"}])  # older EDLs get the clean boost too
lg = renderer.build_command(src, legacy, spoken, tmp / "y.mp4", tmp / "work", renderer.probe(src), facecam=cam)
assert "contrast=1.28" in " ".join(lg) and "colorbalance" not in " ".join(lg)
# the repair trimmed the sped-up front away; the original EDL's command carries the speed ramp
first = renderer.build_command(src, real, spoken, tmp / "x.mp4", tmp / "work", renderer.probe(src), facecam=cam)
assert "setpts=(PTS-STARTPTS)/1.1500" in first[first.index("-filter_complex") + 1] and "atempo=1.1500" in " ".join(first)
# the 4 px seam between facecam and gameplay is dark in the rendered frame
raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "1", "-i", str(tmp / "work" / "short.mp4"), "-frames:v", "1",
                      "-vf", "crop=1080:4:0:690,format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
assert len(raw) == 1080 * 4 and np.frombuffer(raw, np.uint8).mean() < 40, np.frombuffer(raw, np.uint8).mean()
assert qc.check(tmp / "nope.mp4")["problems"] == ["missing"]
# gaming vocabulary: misheard names fixed in word lists; merged words span both timestamps, punctuation kept
fixed_words = pipeline.scorer.fix_gaming_words(words_at([("bro", 0.0), ("Fortnight", 0.5), ("case", 1.0), ("oh!", 1.3),
                                                       ("jinx", 2.0), ("see", 2.2), ("queso", 3.0)], 0.2),
                                               "caseoh_ late night chat", tmp / "no_vocab.json")  # a CaseOh clip
assert fixed_words == [{"word": "bro", "start": 0.0, "end": 0.2}, {"word": "Fortnite", "start": 0.5, "end": 0.7},
                       {"word": "CaseOh!", "start": 1.0, "end": 1.5}, {"word": "Jynxzi", "start": 2.0, "end": 2.2 + 0.2},
                       {"word": "CaseOh", "start": 3.0, "end": 3.2}], fixed_words
assert pipeline.scorer.fix_gaming_words(words_at([("queso", 3.0)]), "xqc cooking", tmp / "no_vocab.json")[0]["word"] == "queso"

# hook teaser + jump cuts + speed ramp on one caption track: never two caption events on screen at once
teased = dict(dense_edl, hook_teaser={"enabled": True, "start": 6.0, "end": 8.0, "banner_text": "WAIT FOR IT"})
renderer.write_ass(teased, dense, tmp / "t.ass")
ev = [l for l in (tmp / "t.ass").read_text(encoding="utf-8").splitlines() if l.startswith("Dialogue: 0,")]
times = [(secs(l.split(",")[1]), secs(l.split(",")[2])) for l in ev]
assert len(times) > 25 and all(b - a >= 0.05 for a, b in times), times  # ass_time rounds to 10 ms
assert all(b1 <= a2 for (_, b1), (a2, _) in zip(times, times[1:])), "overlapping captions with a teaser"
assert sum(a < 2.0 for a, _ in times) >= 3, "teaser words captioned at the start"

# captions from Whisper on the cut audio: its times are used as they are, the tone comes from the source word under it
cut_wav = renderer.cut_audio(src, real, tmp / "cut.wav")
cut_len = len(pipeline.clipper.decode_audio(cut_wav)) / 16000
assert abs(cut_len - renderer.main_length(renderer.timeline(real)[0])) < 0.15, (cut_len, renderer.timeline(real))
two = {"cuts": [{"start": 0, "end": 2}, {"start": 5, "end": 7}], "hook_teaser": {"enabled": False}, "word_tones": ["punch", "rage"]}
src_words = [{"word": "hey", "start": 0.5, "end": 0.8}, {"word": "NO", "start": 5.5, "end": 5.8}]
renderer.write_ass(two, src_words, tmp / "o.ass", out_words=[{"word": "hey", "start": 0.52, "end": 0.8}, {"word": "no", "start": 2.51, "end": 2.8}])
ev = [l for l in (tmp / "o.ass").read_text(encoding="utf-8").splitlines() if l.startswith("Dialogue: 0,")]
assert [round(secs(l.split(",")[1]), 2) for l in ev] == [0.52, 2.51], ev
assert "Impact" in ev[1] and "Impact" not in ev[0], "the rage tone of the source word at 5.5 s"

# SETTINGS caption theme: every highlighted word gets the one colour; editor defaults read from settings.json
renderer.write_ass(dict(dense_edl, caption_tone="hype"), dense, tmp / "c.ass")
colours = set(re.findall(r"\\c(&H[0-9A-F]{8}&)", (tmp / "c.ass").read_text(encoding="utf-8"))) - {"&H00FFFFFF&"}
assert colours == {renderer.TONE_COLORS["hype"]}, colours
(tmp / "settings.json").write_text(json.dumps({"caption_theme": "Cyan", "music_gain_db": -24, "facecam_box": "auto"}))
assert pipeline.load_settings(tmp / "settings.json")["music_gain_db"] == -24 and pipeline.THEMES["Cyan"] == "roast"
assert pipeline.load_settings(tmp / "missing.json") == {}
print("editor ok")

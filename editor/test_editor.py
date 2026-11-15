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
kinds = {z["style"] for z in e["zooms"]}
ramp = next(z for z in e["zooms"] if z["style"] == "ramp")
snap = next(z for z in e["zooms"] if z["style"] == "snap")
assert kinds == {"ramp", "snap"} and ramp["scale"] == 1.45 and snap["scale"] == 1.65 and ramp["at"] < snap["at"], e["zooms"]
assert {f["type"] for f in e["filters"]} == {"reaction_boost", "trauma_shake"}
assert [s["name"] for s in e["sfx"]] == ["record_scratch", "vine_boom"] and e["mutes"]  # the abrupt pause
# effects go on the LOUDEST moment after the setup, not the last mild peak; no SFX without a real yell
two_w = words_at(calm + [("NO", 9.0), ("WAY", 9.3), ("okay", 17.0), ("anyway", 17.4)])
two_a = tone_audio(dur, [(9.2, 0.6, 0.9), (17.2, 0.5, 0.5)])
t2 = director.build_edl("chat_roast", two_w, director.energy(two_a), dur)
snap2 = next(z for z in t2["zooms"] if z["style"] == "snap")
assert abs(snap2["at"] - 9.2) < 0.3 and [s["name"] for s in t2["sfx"]] == ["record_scratch", "vine_boom"], (snap2, t2["sfx"])
assert t2["cuts"][-1]["end"] >= 17.4, "the clip still runs to its last peak"
early_a = tone_audio(dur, [(3.0, 0.6, 0.9), (13.5, 0.6, 0.6)])  # the yell is in the setup; after it only a mild peak (~0.67)
assert director.build_edl("chat_roast", roast_w, director.energy(early_a), dur)["sfx"] == [], "no yell after the setup, no SFX"
assert e["font"] == "Arial Black" and e["word_tones"][roast_w.index(next(w for w in roast_w if w["word"] == "chat"))] == "roast"

# unfiltered_chaos: yelling everywhere
chaos_w = words_at([(w, 0.3 + i * 0.35) for i, w in enumerate(("yo " * 50).split())])
chaos_a = tone_audio(dur, [(2, 0.4, 0.9), (6, 0.4, 0.95), (10, 0.4, 0.9), (14, 0.4, 0.92), (18, 0.4, 0.9)])
env = director.energy(chaos_a)
assert director.classify(chaos_w, env, dur) == "unfiltered_chaos"
e = director.build_edl("unfiltered_chaos", chaos_w, env, dur)
assert len(e["takeovers"]) == 4 and {"trauma_shake", "rgb_split"} <= {f["type"] for f in e["filters"]}
assert {"vine_boom", "bass_boost", "metal_pipe"} <= {s["name"] for s in e["sfx"]} and e["font"] == "Impact"
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
assert next(z for z in e["zooms"])["style"] == "creep" and next(z for z in e["zooms"])["scale"] == 1.15
flash = next(f for f in e["filters"] if f["type"] == "flash")
assert flash["end"] - flash["start"] < 0.1  # ~2 frames

# confident_fail: bragging, then "i'm dead"
fail_w = words_at(calm[:8] + [("watch", 9.0), ("this", 9.3), ("oh", 11.8), ("no", 12.0), ("i'm", 12.3), ("dead", 12.5)])
fail_a = tone_audio(dur, [(12.0, 0.4, 0.9)])
env = director.energy(fail_a)
assert director.classify(fail_w, env, dur) == "confident_fail"
e = director.build_edl("confident_fail", fail_w, env, dur)
assert e["speed"] and e["speed"][0]["factor"] == 1.15 and e["speed"][0]["end"] <= director.peaks(env)[0] + 0.01
assert [s["name"] for s in e["sfx"]] == ["record_scratch", "sad_violin"]
gray = next(f for f in e["filters"] if f["type"] == "grayscale")
assert abs(gray["start"] - director.peaks(env)[0]) < 0.01 and e["cuts"][-1]["end"] > gray["start"] + 1.5  # aftermath kept
tones = e["word_tones"]
assert tones[fail_w.index(fail_w[2])] == "hype" and tones[-1] in ("rage", "roast")  # green setup, darker aftermath

# awkward_irl: slow talk, nothing loud -> long holds, drift, dry SFX
irl_w = words_at([(w, 0.5 + i * 1.6) for i, w in enumerate("so um i guess we just wait here then".split())])
irl_a = (0.2 + 0.02 * np.sin(np.arange(int(dur * SR)) / SR)).astype(np.float32)
env = director.energy(irl_a)
assert director.classify(irl_w, env, dur) == "awkward_irl"
e = director.build_edl("awkward_irl", irl_w, env, dur)
assert all(c["end"] - c["start"] >= 3.0 for c in e["cuts"]) and e["zooms"][0]["style"] == "drift"
assert {s["name"] for s in e["sfx"]} <= {"vine_boom", "pop"} and e["font"] == "Trebuchet MS"
assert director.direct(irl_w, irl_a, dur, has_facecam=True)["layout"] == "blur_fill"  # IRL stays clean

for s in director.SCENARIOS:  # every scenario stays inside QC's length window
    e = director.build_edl(s, roast_w, director.energy(roast_a), dur)
    assert director.MIN_LEN - 0.01 <= renderer.timeline(e)[1] <= director.MAX_LEN + 1, s
d = director.direct(roast_w, roast_a, dur, has_facecam=True)
assert d["layout"] == "split_stack" and d["directed_by"] == "heuristic" and len(d["hashtags"]) == 4

# --- word timing: acoustic onset snap (Whisper starts words early, in the silence before them) -----------------
onset_env = np.full(40, 0.02)
onset_env[10:] = 1.0  # the voice starts at 0.50 s
snapped = pipeline.snap_words_to_audio([{"word": "a", "start": 0.35, "end": 1.0}, {"word": "b", "start": 0.1, "end": 1.0},
                                        {"word": "c", "start": 0.45, "end": 0.53}, {"word": "d", "start": 0.6, "end": 0.9}], onset_env)
assert [w["start"] for w in snapped] == [0.5, 0.1, 0.45, 0.6], snapped  # snapped; >0.25 s away: kept; guard; already voiced
assert [w["end"] for w in snapped] == [1.0, 1.0, 0.53, 0.9]  # ends untouched

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
assert hook["enabled"] and abs(hook["end"] - hook["start"] - director.TEASER_LEN) < 0.01 and hook["start"] <= 27.2 <= hook["end"], hook
assert all(any(c["start"] <= w["start"] and w["end"] <= c["end"] for c in e["cuts"]) or
           all(w["end"] <= c["start"] or w["start"] >= c["end"] for c in e["cuts"]) for w in long_w), "a word was cut in half"
assert not director.build_edl("chat_roast", roast_w, director.energy(roast_a), dur)["hook_teaser"]["enabled"]  # 20 s clip
fe = director.build_edl("confident_fail", fail_w, director.energy(fail_a), dur)
assert fe["freeze_end"] == director.FREEZE_LEN and renderer.timeline(fe)[1] >= director.MIN_LEN

# --- per-streamer facecam presets: a hand-set box is kept per streamer, a preset skips detection, detections are saved --
pfile = tmp / "facecam_presets.json"
detected = []
detect = lambda src: (detected.append(src), (0.0, 0.13, 0.21, 0.28))[1]
twitch_m = {"twitch": {"broadcaster_name": "CaseOh_"}}
assert pipeline.streamer_key(twitch_m) == "caseoh_" and pipeline.streamer_key({"channel": "@SomeChannel"}) == "somechannel"
assert pipeline.streamer_key({"video_id": "x"}) is None
box, how = pipeline.pick_facecam(twitch_m, None, "clip.mp4", pfile, detect)
assert box == (0.0, 0.13, 0.21, 0.28) and detected == ["clip.mp4"] and "saved for caseoh_" in how, how
assert json.loads(pfile.read_text()) == {"caseoh_": [0.0, 0.13, 0.21, 0.28]}
box, how = pipeline.pick_facecam(twitch_m, None, "clip2.mp4", pfile, detect)
assert box == (0.0, 0.13, 0.21, 0.28) and detected[-1] == "clip2.mp4" and "caseoh_'s saved box" in how, "detection agrees: preset"
assert pipeline.pick_facecam(twitch_m, None, "clip2b.mp4", pfile, lambda src: None)[0] == (0.0, 0.13, 0.21, 0.28), "nothing found: preset"
box, how = pipeline.pick_facecam(twitch_m, (0.7, 0.0, 0.3, 0.3), "clip3.mp4", pfile, detect)
assert box == (0.7, 0.0, 0.3, 0.3) and json.loads(pfile.read_text())["caseoh_"] == [0.7, 0.0, 0.3, 0.3], "a hand-set box wins and is kept"
box, how = pipeline.pick_facecam(twitch_m, None, "clip4.mp4", pfile, detect)  # webcam elsewhere in this clip
assert box == (0.0, 0.13, 0.21, 0.28) and "doesn't match" in how and json.loads(pfile.read_text())["caseoh_"] == [0.7, 0.0, 0.3, 0.3], how
lp = {"twitch": {"broadcaster_name": "LosPollosTV"}}
assert pipeline.pick_facecam(lp, None, "lp1.mp4", pfile, detect)[0] == (0.0, 0.13, 0.21, 0.28)
assert pipeline.pick_facecam(lp, None, "lp2.mp4", pfile, lambda src: None)[0] is None, "IRL streamer, no webcam: blur_fill"
assert "lospollostv" not in json.loads(pfile.read_text()), "never a saved box for an IRL streamer"
box, how = pipeline.pick_facecam({"video_id": "anon"}, None, "clip5.mp4", pfile, lambda src: None)
assert box is None and how == "none found (blur_fill)" and set(json.loads(pfile.read_text())) == {"caseoh_"}, "unknown streamer: nothing saved"
pipeline.pick_facecam({"channel": "@yt_guy"}, None, "clip6.mp4", pfile, detect)
assert json.loads(pfile.read_text())["yt_guy"] == [0.0, 0.13, 0.21, 0.28], "YouTube channels get presets too"

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
assert style.startswith("Style: Pop,Impact,") and ",8,4,2,50,50,660," in style  # 8 px outline, 4 px shadow, MarginV 660
events = [l for l in ass.splitlines() if l.startswith("Dialogue: 0,")]
assert "GONE" not in ass  # spoken in the removed gap
assert any(r"\c&H0044FF00&}LET'S" in l for l in events) and any(r"\c&H00FFE500&}CHAT" in l for l in events)  # green, cyan
no = next(l for l in events if r"}NO!" in l)
assert r"\c&H002A2AFF&\fnImpact}NO!" in no and no.startswith("Dialogue: 0,0:00:05.00")  # red + Impact, remapped 7.0 -> 5.0
assert renderer.BOUNCE in no  # a burst starts with the pop-bounce
assert sum(renderer.BOUNCE in l for l in events) == len(renderer.chunk(
    [dict(w, ostart=w["start"], oend=w["end"]) for w in ws if w["word"] != "gone"]))  # one bounce per burst
for l in events:  # never more than 3 words on screen
    assert len(re.sub(r"\{[^}]*\}", "", l.split(",,", 1)[1]).split()) <= 3, l
plain = [dict(w, ostart=w["start"], oend=w["end"]) for w in words_at([(f"w{i}", i * 0.2) for i in range(9)], 0.15)]
assert all(len(g) <= 3 and g[-1]["oend"] - g[0]["ostart"] <= 0.9 for g in renderer.chunk(plain))

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
rage_lines = [l for l in ev if r"\c&H002A2AFF&\fnImpact}" in l]
assert rage_lines and all(r"{\c&H00FFFFFF&\fnArial Black}" in l for l in rage_lines)  # base font restored after a red word

# speed ramps remap captions and effects
sp = {"cuts": [{"start": 0, "end": 10}], "speed": [{"start": 0, "end": 5, "factor": 1.25}],
      "hook_teaser": {"enabled": False}}
segs, length = renderer.timeline(sp)
assert abs(length - 9.0) < 1e-9 and renderer.to_out(5.0, segs) == 4.0 and renderer.to_out(7.0, segs) == 6.0
assert renderer.to_src(2.0, segs) == (0, 2.5)

# --- facecam: corner box from a static face; moving / centred faces rejected ---------------------------------
probe_src = tmp / "probe.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=6", "-f", "lavfi",
                "-i", "sine=f=300:d=6", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(probe_src)], check=True)
corner = lambda g: [(30, 40, 50, 50)]  # same top-left spot in every frame (640 px wide samples)
box = renderer.detect_facecam_box(probe_src, detect=corner)
assert box and box[0] == 0.0 and box[1] < 0.2 and all(isinstance(v, float) for v in box), box
counter = iter(range(100))
moving = lambda g: [(30 + 120 * next(counter), 40, 50, 50)]  # wanders: a game character, not a webcam
assert renderer.detect_facecam_box(probe_src, detect=moving) is None
assert renderer.detect_facecam_box(probe_src, detect=lambda g: [(300, 160, 50, 50)]) is None  # middle of the gameplay
assert renderer.detect_facecam_box(probe_src) is None  # real Haar cascade: no face in a test pattern
# border refinement: a bright overlay rectangle with a "face" inside snaps to the overlay's edges
frames = []
for i in range(5):
    g = np.random.default_rng(i).integers(0, 60, (360, 640)).astype(np.uint8)  # changing gameplay noise
    g[40:200, 0:180] = 200  # static webcam overlay
    frames.append(g)
x0, y0, x1, y1 = renderer.refine_cam_box(frames, (60, 90, 50, 50))
assert x0 == 0 and abs(y0 - 40) <= 1 and abs(x1 - 180) <= 1 and abs(y1 - 200) <= 1, (x0, y0, x1, y1)

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

# SETTINGS caption theme: every highlighted word gets the one colour; editor defaults read from settings.json
renderer.write_ass(dict(dense_edl, caption_tone="hype"), dense, tmp / "c.ass")
colours = set(re.findall(r"\\c(&H[0-9A-F]{8}&)", (tmp / "c.ass").read_text(encoding="utf-8"))) - {"&H00FFFFFF&"}
assert colours == {renderer.TONE_COLORS["hype"]}, colours
(tmp / "settings.json").write_text(json.dumps({"caption_theme": "Cyan", "music_gain_db": -24, "facecam_box": "auto"}))
assert pipeline.load_settings(tmp / "settings.json")["music_gain_db"] == -24 and pipeline.THEMES["Cyan"] == "roast"
assert pipeline.load_settings(tmp / "missing.json") == {}
print("editor ok")

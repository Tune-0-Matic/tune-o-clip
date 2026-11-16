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


# --- 5 scenarios: rule-based classification + tailored EDLs -----------------------------------------------------
dur = 20.0
calm = [(w, 0.5 + i * 0.55) for i, w in enumerate("so chat says i should build the house here okay".split())]

# chat_roast: calm talk, a pause, then an outburst
roast_w = words_at(calm + [("WHAT", 13.0), ("ARE", 13.3), ("YOU", 13.6), ("DOING", 13.9)])
roast_a = tone_audio(dur, [(13.5, 0.6, 0.9)])
env = director.energy(roast_a)
assert director.classify(roast_w, env, dur) == "chat_roast"
e = director.build_edl("chat_roast", roast_w, env, dur)

NO_FX = ("filters", "takeovers", "mutes", "word_tones", "font", "freeze_end", "crop_track", "zoom_anchor")  # the old effect set
assert not any(k in e for k in NO_FX) and e["hook_teaser"]["enabled"], sorted(e)  # pycaps owns captions + cues
# retention effects: a 1.35x facecam punch-in + whoosh on the climax yell, 0.2 s shake + thud on it, pitch drop on "DOING"
assert [(z["scale"], z["duration"]) for z in e["zooms"]] == [(director.SNAP_CLIMAX, director.SNAP_HOLD)] and abs(e["zooms"][0]["at"] - 13.5) < 0.3, e["zooms"]
assert e["shakes"] and all(round(x["end"] - x["start"], 2) == director.SHAKE_LEN and abs(x["start"] - 13.5) < 0.3 for x in e["shakes"]), e["shakes"]
assert sorted((x["name"], x["gain_db"]) for x in e["sfx"]) == [("thud", -14.0), ("whoosh", -16.0)], e["sfx"]
assert e["pitch"]["factor"] == director.PITCH_DROP and abs(e["pitch"]["start"] - 13.3) < 0.01, e["pitch"]  # no "!": the word at the climax
bang = director.build_edl("chat_roast", roast_w[:-1] + [dict(roast_w[-1], word="DOING!")], director.energy(roast_a), dur)
assert abs(bang["pitch"]["start"] - 13.9) < 0.01, bang["pitch"]  # the last "!" word is the punchline
lines_w = words_at([(w, 0.5 + i * 0.5) for i, w in enumerate("no way! what? stop it! why? no! go! yes?".split())])
z = director.build_edl("chat_roast", lines_w, director.energy(roast_a), dur)["zooms"]
assert len(z) >= 2 and all(1.25 <= q["scale"] <= 1.35 for q in z), z
assert all(b["at"] - a["at"] >= director.SNAP_GAP and a["at"] + a["duration"] <= b["at"] for a, b in zip(z, z[1:])), "never overlapping"
# the clip runs to its LAST peak, even when the loudest outburst came earlier
two_w = words_at(calm + [("NO", 9.0), ("WAY", 9.3), ("okay", 17.0), ("anyway", 17.4)])
two_a = tone_audio(dur, [(9.2, 0.6, 0.9), (17.2, 0.5, 0.5)])
t2 = director.build_edl("chat_roast", two_w, director.energy(two_a), dur)
assert t2["cuts"][-1]["end"] >= 17.4, "the clip still runs to its last peak"

# unfiltered_chaos: yelling everywhere
chaos_w = words_at([(w, 0.3 + i * 0.35) for i, w in enumerate(("yo " * 50).split())])
chaos_a = tone_audio(dur, [(2, 0.4, 0.9), (6, 0.4, 0.95), (10, 0.4, 0.9), (14, 0.4, 0.92), (18, 0.4, 0.9)])
env = director.energy(chaos_a)
assert director.classify(chaos_w, env, dur) == "unfiltered_chaos"
e = director.build_edl("unfiltered_chaos", chaos_w, env, dur)
assert not any(k in e for k in NO_FX), sorted(e)
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
assert not e["hook_teaser"]["enabled"] and not any(k in e for k in NO_FX), "never show the scare first"

# confident_fail: bragging, then "i'm dead"
fail_w = words_at(calm[:8] + [("watch", 9.0), ("this", 9.3), ("oh", 11.8), ("no", 12.0), ("i'm", 12.3), ("dead", 12.5)])
fail_a = tone_audio(dur, [(12.0, 0.4, 0.9)])
env = director.energy(fail_a)
assert director.classify(fail_w, env, dur) == "confident_fail"
e = director.build_edl("confident_fail", fail_w, env, dur)
assert e["speed"] and e["speed"][0]["factor"] == 1.15 and e["speed"][0]["end"] <= director.peaks(env)[0] + 0.01
assert not any(k in e for k in NO_FX), sorted(e)
assert fail_w[-1]["end"] <= e["cuts"][-1]["end"] <= fail_w[-1]["end"] + 0.3, e["cuts"]  # the spoken aftermath, not silence

# awkward_irl: slow talk, nothing loud
irl_w = words_at([(w, 0.5 + i * 1.6) for i, w in enumerate("so um i guess we just wait here then".split())])
irl_a = (0.2 + 0.02 * np.sin(np.arange(int(dur * SR)) / SR)).astype(np.float32)
env = director.energy(irl_a)
assert director.classify(irl_w, env, dur) == "awkward_irl"
e = director.build_edl("awkward_irl", irl_w, env, dur)
assert len(e["cuts"]) > 1 and not any(k in e for k in NO_FX)  # silences cut
CAM = {"layout": "split_stack", "cam_box": (0.0, 0.1, 0.3, 0.4), "source_captions": False}
assert director.direct(irl_w, irl_a, dur, framing=CAM)["layout"] == "blur_fill"  # IRL stays clean

for s in director.SCENARIOS:  # every scenario stays inside QC's length window
    e = director.build_edl(s, roast_w, director.energy(roast_a), dur)
    assert director.MIN_LEN - 0.01 <= renderer.timeline(e)[1] <= director.MAX_LEN + 1, s
d = director.direct(roast_w, roast_a, dur, framing=CAM)
assert d["layout"] == "split_stack" and d["directed_by"] == "heuristic" and len(d["hashtags"]) == 4
assert not d["source_captions"] and not any(k in d for k in NO_FX) and d["zooms"], sorted(d)
bf = director.direct(roast_w, roast_a, dur)  # no webcam: no punch-ins / shake (the gameplay never moves), thud + pitch stay
assert bf["zooms"] == [] and bf["shakes"] == [] and [x["name"] for x in bf["sfx"]] == ["thud"] and bf["pitch"], bf["sfx"]
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
assert "freeze_end" not in fe and renderer.timeline(fe)[1] >= director.MIN_LEN

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

# --- captions: caption_doc -> pycaps whisper_json in OUTPUT seconds, 1-3 word uppercase lines -------------------
cut_edl = {"cuts": [{"start": 0, "end": 4}, {"start": 6, "end": 10}], "hook_teaser": {"enabled": False}}
ws = words_at([("let's", 1.0), ("go", 1.3), ("chat", 1.6), ("gone", 4.5), ("NO!", 7.0), ("why", 7.4), ("bro", 7.7), ("okay", 8.0)], 0.25)
doc = renderer.caption_doc(cut_edl, ws)
flat = [w for sg in doc["segments"] for w in sg["words"]]
assert [w["word"] for w in flat] == ["LET'S", "GO", "CHAT", "NO!", "WHY", "BRO", "OKAY"], flat  # GONE: spoken in the removed gap
assert next(w for w in flat if w["word"] == "NO!")["start"] == 5.0, "remapped 7.0 -> 5.0"
assert all(1 <= len(sg["words"]) <= 3 and sg["start"] == sg["words"][0]["start"] and sg["end"] == sg["words"][-1]["end"]
           for sg in doc["segments"]), doc
plain = [dict(w, ostart=w["start"], oend=w["end"]) for w in words_at([(f"w{i}", i * 0.2) for i in range(9)], 0.15)]
assert all(len(g) <= 3 and g[-1]["oend"] - g[0]["ostart"] <= 1.3 for g in renderer.chunk(plain))
# never two caption lines on screen at once (incl. across a jump cut and a sped-up segment)
dense = words_at([(f"w{i}", 0.4 + i * 0.27) for i in range(30)], 0.26)  # tails would overlap the next line
dense_edl = {"cuts": [{"start": 0, "end": 4.2}, {"start": 4.25, "end": 9}], "speed": [{"start": 4.25, "end": 9, "factor": 1.15}],
             "hook_teaser": {"enabled": False}}
lines = [(sg["start"], sg["end"]) for sg in renderer.caption_doc(dense_edl, dense)["segments"]]
assert len(lines) >= 10 and all(b > a for a, b in lines), lines
assert all(b1 <= a2 for (_, b1), (a2, _) in zip(lines, lines[1:])), "overlapping captions"
assert set(renderer.CAPTION_COLORS) == set(pipeline.THEMES.values()), "every SETTINGS theme has a colour"

# speed ramps remap captions and effects
sp = {"cuts": [{"start": 0, "end": 10}], "speed": [{"start": 0, "end": 5, "factor": 1.25}],
      "hook_teaser": {"enabled": False}}
segs, length = renderer.timeline(sp)
assert abs(length - 9.0) < 1e-9 and renderer.to_out(5.0, segs) == 4.0 and renderer.to_out(7.0, segs) == 6.0
assert renderer.to_src(2.0, segs) == (0, 2.5)

# --- framing: real face detections -> split_stack / blur_fill (framing.decide is pure) ---------------
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
assert framing.decide(times, crowd, A)["layout"] == "blur_fill", "people spread across the room: no split"

assert framing.decide(times, [[]] * 20, A)["layout"] == "blur_fill"
assert framing.decide(times, [[cam_face]] * 6 + [[]] * 14, A)["layout"] == "blur_fill", "a face in < half the samples"
wander = [[(0.05 + 0.04 * i, 0.2, 0.07, 0.15, 0.9)] for i in range(20)]  # a moving game character, not a webcam
assert framing.decide(times, wander, A)["layout"] != "split_stack"
host = [[(0.30 + 0.01 * (i % 5), 0.4, 0.2, 0.4, 0.9)] for i in range(20)]  # a big face-cam host (Just Chatting)
assert framing.decide(times, host, A)["layout"] == "blur_fill", "no corner webcam: the whole frame, no face-following crop"
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

# cold open: a loud climax in the second half of a >= 16 s clip plays first (1.2-1.5 s)
cold_w = words_at([(w, 0.5 + i * 0.6) for i, w in enumerate(("so chat this guy said " * 3).split())] + [("NO", 15.0), ("WAY!", 15.3)])
cold_a = tone_audio(18.0, [(15.2, 0.6, 0.95)])
ce = director.build_edl("chat_roast", cold_w, director.energy(cold_a), 18.0)
hook = ce["hook_teaser"]
T = hook["end"] - hook["start"]
assert hook["enabled"] and director.TEASER_MIN - 0.01 <= T <= director.TEASER_LEN + 0.01 and hook["start"] <= 15.2 <= hook["end"], hook
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

real = {"scenario": "chat_roast", "layout": "split_stack",
        "hook_teaser": {"enabled": True, "start": 9.0, "end": 11.0, "banner_text": "WAIT FOR IT..."},
        "cuts": [{"start": 0.5, "end": 6.0}, {"start": 7.0, "end": 15.5}],
        "speed": [{"start": 0.5, "end": 4.0, "factor": 1.15}], "bg_music": {"enabled": False, "gain_db": -22},
        "zooms": [{"at": 9.0, "duration": 0.8, "scale": 1.3}], "shakes": [{"start": 9.1, "end": 9.3}],
        "sfx": [{"at": 8.9, "name": "whoosh", "gain_db": -16}, {"at": 9.1, "name": "thud", "gain_db": -14}],
        "pitch": {"start": 9.1, "end": 9.4, "factor": 0.86},
        # an older EDL's effects: ignored now
        "filters": [{"start": 9.0, "end": 9.9, "type": "reaction_boost"}], "takeovers": [{"start": 8.8, "end": 10.5}]}
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
assert "vstack" in graph and "pad=1080:694" in graph
for needle in ("overlay=0:0:enable=", "eq=", "rgbashift", "hue=", "subtitles=", "drawtext"):
    assert needle not in graph, needle  # no old effect filters / ffmpeg captions
cam_line = next(l for l in graph.split(";") if l.startswith("[cs]"))
game_line = next(l for l in graph.split(";") if l.startswith("[gs]"))
assert "zoompan=z='min(1.35,1+" in cam_line and "crop=w=iw-12:h=ih-12" in cam_line, cam_line  # facecam: punch-in + shake
assert "zoompan" not in game_line and "crop=w=iw-12" not in game_line, "the gameplay panel never moves"
assert "asetrate=48000*0.86,aresample=48000,atempo=1.1628" in graph and graph.count("adelay=") == 2, graph
assert cmd[-1].endswith("base.mp4") and not (tmp / "work" / "base.mp4").exists(), "pycaps wrote the short; the base is gone"
burned = json.loads((tmp / "work" / "captions.json").read_text(encoding="utf-8"))
assert [w["word"] for sg in burned["segments"] for w in sg["words"]] == ["WHAT!"], burned  # the repair trimmed the setup away
# the captions really are on the picture: the same edit without them (source_captions) differs in the caption band
plain_out = renderer.render(src, dict(final, source_captions=True), spoken, tmp / "plain.mp4", tmp / "plain_work", renderer.detect_encoder(), cam)
assert json.loads((tmp / "plain_work" / "ffmpeg_cmd.json").read_text())[-1] == str(plain_out), "no captions: ffmpeg writes the short itself"
t_word = burned["segments"][0]["words"][0]["start"] + 0.2  # past the 0.14 s pop-in
band = lambda pth: np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t_word:.2f}", "-i", str(pth), "-frames:v", "1", "-vf",
                                                 "crop=1080:200:0:783,format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout, np.uint8).astype(int)
diff = np.abs(band(tmp / "work" / "short.mp4") - band(plain_out)).mean()
assert diff > 3, ("no caption pixels in the caption band", diff)
bf_cmd = renderer.build_command(src, dict(real, layout="blur_fill"), tmp / "bf.mp4", renderer.probe(src))
bf_graph = bf_cmd[bf_cmd.index("-filter_complex") + 1]
assert "boxblur" in bf_graph and "vstack" not in bf_graph and "zoompan" not in bf_graph

# --- real render: mostly-dark footage used to be stripped to < 7 s and stuck in .work as too_short -----------------
dark_src = tmp / "dark.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=60:d=20", "-f", "lavfi",
                "-i", "aevalsrc='0.3*sin(2*PI*220*t)':s=48000:d=20", "-vf",
                "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='lt(t,12)'", "-c:v", "libx264", "-preset",
                "ultrafast", "-c:a", "aac", "-shortest", str(dark_src)], check=True)
dark_edl = {"scenario": "awkward_irl", "layout": "blur_fill", "hook_teaser": {"enabled": False},
            "cuts": [{"start": 4.0, "end": 18.0}], "speed": [], "bg_music": {"enabled": False, "gain_db": -22}}
with quiet() as log:
    final, report = pipeline.render_until_pass(dark_src, dark_edl, [], tmp / "dark_work", None, renderer.detect_encoder())
assert "FAIL black" in log.getvalue() and report["ok"] and final.get("keep_dark"), (log.getvalue(), report)
assert final["cuts"] == dark_edl["cuts"] and report["duration"] >= 13.5, final["cuts"]  # nothing shredded
assert renderer.probe(tmp / "dark_work" / "short.mp4")["fps"] == 30, "60 fps source -> 30 fps short (half the filter work)"
# the repair trimmed the sped-up front away; the original EDL's command carries the speed ramp
first = renderer.build_command(src, real, tmp / "x.mp4", renderer.probe(src), facecam=cam)
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

# hook teaser + jump cuts + speed ramp on one caption track: never two lines on screen at once
teased = dict(dense_edl, hook_teaser={"enabled": True, "start": 6.0, "end": 8.0, "banner_text": "WAIT FOR IT"})
lines = [(sg["start"], sg["end"]) for sg in renderer.caption_doc(teased, dense)["segments"]]
assert len(lines) >= 12 and all(b - a >= 0.05 for a, b in lines), lines
assert all(b1 <= a2 for (_, b1), (a2, _) in zip(lines, lines[1:])), "overlapping captions with a teaser"
assert sum(a < 2.0 for a, _ in lines) >= 2, "teaser words captioned at the start"

# captions from Whisper on the cut audio: its times are used as they are
cut_wav = renderer.cut_audio(src, real, tmp / "cut.wav")
cut_len = len(pipeline.clipper.decode_audio(cut_wav)) / 16000
assert abs(cut_len - renderer.main_length(renderer.timeline(real)[0])) < 0.15, (cut_len, renderer.timeline(real))
two = {"cuts": [{"start": 0, "end": 2}, {"start": 5, "end": 7}], "hook_teaser": {"enabled": False}}
src_words = [{"word": "hey", "start": 0.5, "end": 0.8}, {"word": "NO", "start": 5.5, "end": 5.8}]
doc = renderer.caption_doc(two, src_words, out_words=[{"word": "hey", "start": 0.52, "end": 0.8}, {"word": "no", "start": 2.51, "end": 2.8}])
assert [sg["start"] for sg in doc["segments"]] == [0.52, 2.51], doc

# --- emotional_story profile (reference-matched); the hype_chaos edit stays exactly as it was ------------------------
hype = director.direct(roast_w, roast_a, dur, framing=CAM)
story_edl = director.direct(roast_w, roast_a, dur, framing=CAM, meta={"tone": "emotional_story", "emotion": "nostalgic"})
assert hype.get("profile") is None and hype["zooms"] and hype["sfx"] and hype.get("pitch"), "hype untouched"
assert story_edl["profile"] == "emotional_story" and story_edl["emotion"] == "nostalgic"
assert story_edl["zooms"] == [] and story_edl["shakes"] == [] and story_edl["sfx"] == [] and "pitch" not in story_edl
assert story_edl["speed"] == [] and not story_edl["hook_teaser"]["enabled"] and story_edl["bg_music"]["enabled"]
assert story_edl["cuts"] == hype["cuts"] and "montage" not in story_edl, "same tight hard cuts; a 20 s clip's 13 s edit: no montage"
assert director.direct(roast_w, roast_a, dur, meta={"tone": "hype_chaos"}).get("profile") is None
# flash montage: a >= 20 s story with >= 2 strong moments gets 8 x 0.45 s flashes (voice muted) before its final line
m_dur = 40.0
m_words = words_at([(w, 0.5 + i * 0.7) for i, w in enumerate(("i used to think about this every single night honestly " * 4).split())]
                   + [("and", 33.0), ("i", 33.3), ("never", 33.6), ("told", 34.0), ("you", 34.4), ("that.", 34.7),
                      ("thank", 36.5), ("you.", 36.9)])
m_audio = tone_audio(m_dur, [(5.0, 0.4, 0.8), (14.0, 0.4, 0.9), (24.0, 0.4, 0.85), (34.0, 0.4, 0.95)])
m_env = director.energy(m_audio)
m_cuts = [{"start": 0.3, "end": 37.4}]
mt = director.flash_montage(m_cuts, m_words, m_env)
assert mt and mt["at"] == 36.5, mt  # the final line "thank you." starts after the last sentence end
assert len(mt["clips"]) == director.MONTAGE_N and all(round(e - s, 2) == director.MONTAGE_FLASH for s, e in mt["clips"]), mt
assert [s for s, _ in mt["clips"]] == sorted(s for s, _ in mt["clips"]) and all(e <= mt["at"] for _, e in mt["clips"])
assert any(abs(s + 0.2 - 14.0) < 0.2 for s, _ in mt["clips"]) and any(abs(s + 0.2 - 24.0) < 0.2 for s, _ in mt["clips"]), "strong moments first"
assert director.flash_montage([{"start": 0.3, "end": 15.0}], m_words, m_env) is None, "< 20 s: no montage"
assert director.flash_montage(m_cuts, m_words, director.energy(tone_audio(m_dur, [(34.0, 0.4, 0.95)]))) is None, "one moment"
ms = director.story_profile({"cuts": [dict(c) for c in m_cuts], "hook_teaser": {"enabled": False}, "bg_music": {"enabled": False}},
                            m_dur, "grateful", m_words, m_env)
assert ms["cuts"] == [{"start": 0.3, "end": 36.5}, {"start": 36.5, "end": 37.4}], ms["cuts"]  # split where the montage goes in
segs_m, len_m = renderer.timeline(ms)
assert [k for _, _, _, k, _ in segs_m] == [False] + ["montage"] * 8 + [False] and abs(len_m - (37.1 + 8 * 0.45)) < 1e-6, segs_m
assert abs(renderer.to_out(37.0, segs_m) - (36.2 + 3.6 + 0.5)) < 1e-6, "the final line plays after the montage"
mg = ";".join(renderer.cut_graph(segs_m))
assert mg.count(",volume=0[a") == 8 and "xfade" not in mg and "concat=n=10" in mg, "flash pieces: picture only, hard cuts"
assert all(w["word"] != "THANK" or w["start"] > 36.2 for sg in renderer.caption_doc(ms, m_words)["segments"] for w in sg["words"])
# captions: bold uppercase lines of up to 6 words (pycaps sets them as 2 lines), the accent by emotion
sdoc = renderer.caption_doc(dict(story_edl), words_at([("Hey", 1.0), ("chat,", 1.3), ("you", 1.6), ("ready", 1.9), ("to", 2.2), ("go?", 2.5)], 0.25))
assert [w["word"] for w in sdoc["segments"][0]["words"]] == ["HEY", "CHAT,", "YOU", "READY", "TO", "GO?"], sdoc
assert {renderer.story_tint({"emotion": e}) for e in ("proud", "sad", "nostalgic")} == {"warm", "sad", "cool"}
assert renderer.STORY_ACCENT == {"warm": "#FFD60A", "sad": "#FF3B3B", "cool": "#3DE8FF"} and renderer.CAPTION_Y["story"] == 0.52
import captions  # noqa: E402
assert "rgba(255,214,10,0.38)" in captions.accent_css("#FFD60A") and ".last-line-in-segment" in captions.accent_css("#FFD60A")
# the graph: punchy grade + push-in, the bed after the voice's loudnorm at -19 dBFS (gain from its own RMS), ducked
music_tmp = tmp / "music"
bed = renderer.ensure_story_bed("sad_ambient", music_tmp)
with wave.open(str(bed)) as f:
    assert f.getframerate() == 48000 and abs(f.getnframes() / 48000 - 16.0) < 0.01
assert renderer.user_music(music_tmp) == [], "the story beds never become the hype edit's music"
(music_tmp / "mine.mp3").write_bytes(b"x")
assert [m.name for m in renderer.user_music(music_tmp)] == ["mine.mp3"] and renderer.rms_dbfs(music_tmp / "mine.mp3") is None
(music_tmp / "mine.mp3").unlink()
story_real = dict(real, profile="emotional_story", emotion="nostalgic", layout="blur_fill", zooms=[], shakes=[], sfx=[], speed=[],
                  hook_teaser={"enabled": False}, bg_music={"enabled": True, "gain_db": -22})
story_real.pop("pitch")
sc = renderer.build_command(src, story_real, tmp / "story.mp4", renderer.probe(src), music_dir=music_tmp)
sg = sc[sc.index("-filter_complex") + 1]
gain = renderer.STORY_BED_DBFS - renderer.rms_dbfs(bed)
for needle in ("eq=contrast=1.10:saturation=1.18", "colorbalance=rs=-0.05:gs=0.02:bs=0.07", "vignette=PI/6",
               "scale=w='2*trunc(540*(1+0.07*t/", ":eval=frame,crop=1080:1920", "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,asplit=2[vn][side]",
               f"volume={gain:.1f}dB,afade=t=in", "[mu][side]sidechaincompress", "[vn][duck]amix=inputs=2:normalize=0:duration=first,alimiter"):
    assert needle in sg, needle
for needle in ("zoompan", "asetrate", "adelay", "crop=w=iw-12", "xfade", "saturation=0.78"):
    assert needle not in sg, needle  # zero snap zooms / pitch drops / SFX / shakes / fades / washed-out grade
assert str(bed) in sc
warm = renderer.build_command(src, dict(story_real, emotion="proud"), tmp / "w.mp4", renderer.probe(src), music_dir=music_tmp)
assert str(music_tmp / "warm_ambient.wav") in warm, "proud: the major bed"
hype_graph = renderer.build_command(src, real, tmp / "x.mp4", renderer.probe(src), facecam=cam)
hg = hype_graph[hype_graph.index("-filter_complex") + 1]
assert "eq=contrast=1.10" not in hg and "eval=frame" not in hg and "alimiter" not in hg and "volume=0[a" not in hg, "hype graph unchanged"
# for real: renders, the bed sits at ~-19 dBFS where nobody talks, and pycaps burns the 2-line accent style
quiet_src = tmp / "quiet_mid.mp4"  # speech-like tone with a 3 s gap in the middle: the bed alone there
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=12", "-f", "lavfi", "-i",
                "aevalsrc='0.3*sin(2*PI*220*t)*(lt(t,4.5)+gt(t,7.5))':s=48000:d=12", "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", "-shortest", str(quiet_src)], check=True)
gap_edl = dict(story_real, cuts=[{"start": 0.0, "end": 12.0}])
with quiet():
    final_s, rep_s = pipeline.render_until_pass(quiet_src, gap_edl, words_at([("so", 1.0), ("i", 2.0), ("remember", 3.0), ("it", 9.0)]),
                                                tmp / "story_work", {"max_dur": 30.0, "min_dur": 5.0}, renderer.detect_encoder())
assert rep_s["ok"], rep_s
pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(tmp / "story_work" / "short.mp4"), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                     capture_output=True).stdout
x = np.frombuffer(pcm, np.int16).astype(np.float64) / 32768
gap_db = 20 * np.log10(np.sqrt(np.mean(x[int(5.3 * 16000):int(7.0 * 16000)] ** 2)) + 1e-9)
assert -22.5 <= gap_db <= -16.5, ("the bed in the speech gap", gap_db)
burned_s = json.loads((tmp / "story_work" / "captions.json").read_text())
assert burned_s["segments"][0]["words"][0]["word"] == "SO", burned_s

# editor defaults read from settings.json
(tmp / "settings.json").write_text(json.dumps({"caption_theme": "Cyan", "music_gain_db": -24, "facecam_box": "auto"}))
assert pipeline.load_settings(tmp / "settings.json")["music_gain_db"] == -24 and pipeline.THEMES["Cyan"] == "roast"
assert pipeline.load_settings(tmp / "missing.json") == {}
print("editor ok")

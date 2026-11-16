"""spike_clipper.py: chat / heatmap spikes -> 45-60 s slices, without the network (downloads and clipper runs faked).
Run: python backend/test_spikes.py
"""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spike_clipper as sc  # noqa: E402

tmp = Path(tempfile.mkdtemp())
DUR = 3600.0

# a quiet chat (2 regulars / 10 s) with two bursts: 1500 s (KEKW + "clip it") and 1530 s (smaller, too close),
# one at 2400 s, and a huge one in the first 5 minutes (stream intro: ignored)
chat = [(float(t), f"reg{t % 2}", "hello") for t in range(0, int(DUR), 10)]
for base, n, text in ((1500, 60, "KEKW KEKW clip it"), (1530, 25, "LUL"), (2400, 40, "OMEGALUL W"), (120, 200, "W W W")):
    chat += [(base + i * 0.05, f"viewer{base}_{i}", text) for i in range(n)]
curve = sc.chat_curve(chat, DUR)
assert len(curve) == int(DUR) + 1
assert abs(int(np.argmax(curve[300:3300])) + 300 - (1500 - sc.SHIFT)) <= 3, "the peak sits 8 s before chat reacts"
moments = sc.pick_moments(curve, DUR, 4)
assert moments[0] in range(1489, 1496) and any(2385 <= m <= 2396 for m in moments), moments
assert all(300 <= m <= DUR - 300 for m in moments), "intro / outro skipped"
assert all(abs(a - b) >= sc.GAP for i, a in enumerate(moments) for b in moments[i + 1:]), "spaced >= 90 s"
assert not any(1500 <= m <= 1530 + 8 for m in moments[1:]), "the 1530 s echo is the same moment"
assert sc.pick_moments(np.zeros(600), 600.0, 3) == [], "no spikes, no slices"
short = np.zeros(601)
short[100] = 1
assert sc.pick_moments(short, 600.0, 3) == [100], "a 10-min video keeps its first minutes"
s, e = sc.slice_of(1492, DUR)
assert 45 <= e - s <= 60 and s < 1492 < e and 1492 - s > e - 1492, "more setup than aftermath"
assert sc.slice_of(10, DUR)[0] == 0 and sc.slice_of(DUR - 5, DUR)[1] == DUR, "kept inside the video"
assert sc.link_at("https://www.twitch.tv/videos/123", 3725) == "https://www.twitch.tv/videos/123?t=1h2m5s"
assert sc.link_at("https://www.youtube.com/watch?v=x", 90) == "https://www.youtube.com/watch?v=x&t=90s"
assert sc.EMOTE_RE.findall("KEKW lol W wwww clip LUL") and not sc.EMOTE_RE.findall("hello world whatsoever")

# chat files: a TwitchDownloader export and the plain list
td = tmp / "td.json"
td.write_text(json.dumps({"video": {"id": "123"}, "comments": [
    {"content_offset_seconds": 12.5, "commenter": {"name": "alice"}, "message": {"body": "KEKW"}}]}), encoding="utf-8")
assert sc.load_chat(td) == [(12.5, "alice", "KEKW")]
plain = tmp / "plain.json"
plain.write_text(json.dumps([{"t": 3, "user": "bob", "text": "clip it"}]), encoding="utf-8")
assert sc.load_chat(plain) == [(3.0, "bob", "clip it")]

# the whole flow, offline: metadata faked, only the picked slices "download", each goes to clipper.py with its meta
chat_file = tmp / "chat.json"
chat_file.write_text(json.dumps([{"t": t, "user": u, "text": x} for t, u, x in chat]), encoding="utf-8")
job = tmp / "job"
job.mkdir()
sc.clipper.fetch_metadata = lambda url, ws: ({"id": "v123", "title": "big stream", "duration": DUR, "uploader": "CaseOh_",
                                              "webpage_url": "https://www.twitch.tv/videos/123"}, job)
calls = []


class Done:  # subprocess.CompletedProcess stand-in
    def __init__(self, stdout=""):
        self.returncode, self.stdout, self.stderr = 0, stdout, ""


def fake_run(cmd, **kw):
    calls.append(cmd)
    if "-g" in cmd:  # yt-dlp -g: Twitch = one muxed HLS URL, YouTube = video + audio
        return Done("https://h/v.m3u8\n" if "twitch" in cmd[-1] else "https://h/video\nhttps://h/audio\n")
    if cmd[0] == "ffmpeg":
        Path(cmd[-1]).write_bytes(b"slice")
    return Done()


def seek(cmd):
    """-> (start, length) of an ffmpeg slice command"""
    return float(cmd[cmd.index("-ss") + 1]), float(cmd[cmd.index("-t") + 1])


out = io.StringIO()
with contextlib.redirect_stdout(out):
    assert sc.main(["https://www.twitch.tv/videos/123", "--chat", str(chat_file), "--clips", "2", "--auto-edit"], run=fake_run) == 0
downloads = [c for c in calls if c[0] == "ffmpeg"]
runs = [c for c in calls if any(str(x).endswith("clipper.py") for x in c)]
assert len(downloads) == 2 and len(runs) == 2 and sum("-g" in c for c in calls) == 1, calls
a, n = seek(downloads[0])
assert 45 <= n <= 60 and downloads[0].index("-ss") < downloads[0].index("-i"), downloads[0]
meta = json.loads(runs[0][runs[0].index("--twitch-meta") + 1])
assert meta["broadcaster_name"] == "CaseOh_" and meta["clip_id"].startswith("v123_") and "?t=" in meta["url"], meta
assert runs[0][-1] == "--auto-edit", "clipper options pass through"
assert not any("--chat" in c for c in runs[0]), "spike-only options stay here"
assert "moment(s) from chat" in out.getvalue(), out.getvalue()
# a Twitch VOD without a chat log: a clear how-to, nothing downloaded
calls.clear()
try:
    sc.main(["https://www.twitch.tv/videos/123", "--chat", str(tmp / "missing.json")], run=fake_run)
    raise AssertionError("must explain the missing chat log")
except RuntimeError as err:
    assert "TwitchDownloaderCLI chatdownload --id 123" in str(err) and not calls, err
# YouTube: the Most Replayed heatmap instead of chat
heat = [{"start_time": i * 30.0, "end_time": (i + 1) * 30.0, "value": 1.0 if i == 60 else 0.1} for i in range(120)]
sc.clipper.fetch_metadata = lambda url, ws: ({"id": "yt1", "title": "vid", "duration": DUR, "uploader": "Kai",
                                              "webpage_url": "https://www.youtube.com/watch?v=yt1", "heatmap": heat}, job)
with contextlib.redirect_stdout(io.StringIO()) as yt_out:
    assert sc.main(["https://www.youtube.com/watch?v=yt1", "--clips", "1"], run=fake_run) == 0
ff = next(c for c in calls if c[0] == "ffmpeg")
a, n = seek(ff)
assert a < 1815 < a + n, (a, n)  # the 1800-1830 s heatmap segment
assert ff.count("-ss") == 2 and ["-map", "0:v:0", "-map", "1:a:0"] == ff[ff.index("-map"):ff.index("-map") + 4], ff
assert "Most Replayed heatmap" in yt_out.getvalue()
print("spikes ok")

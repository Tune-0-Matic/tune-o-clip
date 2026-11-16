"""spike_clipper.py automatic Twitch VOD chat fetch, offline: a fake GraphQL opener (pagination, cache reuse, fallback).
Run: python backend/test_spike_clipper.py
"""
import contextlib
import io
import json
import sys
import tempfile
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spike_clipper as sc  # noqa: E402

tmp = Path(tempfile.mkdtemp())
sc.CHAT_DIR = tmp / "chat"  # never the real workspace/chat
DUR = 3600.0


def node(t, user, text):
    return {"cursor": f"c{t}", "node": {"contentOffsetSeconds": t, "commenter": user and {"login": user},
                                        "message": {"fragments": [{"text": w} for w in text.split("|")]}}}


def page(edges, more):
    return [{"data": {"video": {"comments": {"edges": edges, "pageInfo": {"hasNextPage": more}}}}}]


# 3 pages: a quiet start, a big KEKW burst at 1500 s, a deleted user (commenter null) at the end
PAGES = [page([node(t, f"reg{t}", "hi") for t in range(0, 1490, 10)], True),
         page([node(1500, f"v{i}", "KEKW| |clip it") for i in range(60)][:-1] + [node(1500, "v59", "KEKW")], True),
         page([node(2000, None, "gone")], False)]


class FakeGql:
    def __init__(self, pages=PAGES, fail=None):
        self.pages, self.fail, self.requests, self.served = pages, fail or {}, [], 0

    def __call__(self, req, timeout):
        self.requests.append(req)
        i = len(self.requests) - 1  # fail = {request number: error}; pages are served in order to the successful ones
        if i in self.fail:
            err = self.fail[i]
            raise err() if callable(err) else err
        self.served += 1
        return io.BytesIO(json.dumps(self.pages[min(self.served - 1, len(self.pages) - 1)]).encode())


def http(code):
    return lambda: urllib.error.HTTPError(sc.GQL_URL, code, "x", {}, None)


sleeps = []
nosleep = sleeps.append

# pagination: offset 0 first, then each page's last cursor; Twitch's public Client-Id only, no auth / cookies
g = FakeGql()
out = tmp / "pg.json"
assert sc.fetch_chat("123", out, g, nosleep) == 149 + 60 + 1
vars_ = [json.loads(r.data)[0]["variables"] for r in g.requests]
assert vars_ == [{"videoID": "123", "contentOffsetSeconds": 0}, {"videoID": "123", "cursor": "c1480"},
                 {"videoID": "123", "cursor": "c1500"}], vars_
hdrs = {k.lower(): v for k, v in g.requests[0].header_items()}
assert hdrs["client-id"] == sc.GQL_CLIENT_ID and "cookie" not in hdrs and "authorization" not in hdrs, hdrs
assert g.requests[0].full_url == "https://gql.twitch.tv/gql" and g.requests[0].get_method() == "POST"
chat = sc.load_chat(out)  # the cache is a TwitchDownloader-shaped file
assert chat[0] == (0.0, "reg0", "hi") and (1500.0, "v0", "KEKW clip it") in chat and chat[-1] == (2000.0, "", "gone"), chat[-3:]
assert not list(tmp.glob("*.part")), "temp file renamed into place"

# a 429 and a dropped connection mid-fetch are retried with backoff; the fetch still completes
g = FakeGql(fail={1: http(429), 2: lambda: urllib.error.URLError("reset")})
assert sc.fetch_chat("123", tmp / "retry.json", g, nosleep) == 210 and sleeps[-2:] == [1.0, 2.0], sleeps

# the whole spike run, offline: no cache -> fetched + cached -> moments come from it
job = tmp / "job"
job.mkdir()
sc.clipper.fetch_metadata = lambda url, ws: ({"id": "v123", "title": "big stream", "duration": DUR, "uploader": "CaseOh_",
                                              "webpage_url": "https://www.twitch.tv/videos/123"}, job)
calls = []


class Done:  # subprocess.CompletedProcess stand-in
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def fake_run(cmd, **kw):
    calls.append(cmd)
    if "-g" in cmd:
        return Done(stdout="https://h/index-dvr.m3u8\n")
    if cmd[0] == "ffmpeg":
        Path(cmd[-1]).write_bytes(b"slice")
    return Done()


g = FakeGql()
with contextlib.redirect_stdout(io.StringIO()) as o:
    assert sc.main(["https://www.twitch.tv/videos/123", "--clips", "1"], run=fake_run, opener=g, sleep=nosleep) == 0
cache = sc.CHAT_DIR / "123.json"
assert cache.is_file() and len(g.requests) == 3, o.getvalue()
assert "Saved 210 chat messages" in o.getvalue() and "moment(s) from chat (210 messages)" in o.getvalue(), o.getvalue()
ff = next(c for c in calls if c[0] == "ffmpeg")
a, n = float(ff[ff.index("-ss") + 1]), float(ff[ff.index("-t") + 1])
assert a < 1500 - sc.SHIFT < a + n, (a, n)  # the KEKW burst

# --- slice download: yt-dlp -g once, then ffmpeg input-seeks straight into the HLS playlist ---
calls.clear()
assert sc.stream_urls("https://www.twitch.tv/videos/123", fake_run) == ["https://h/index-dvr.m3u8"]
g_cmd = calls[0]
assert g_cmd[g_cmd.index("-f") + 1] == "best[height<=1080]/best" and g_cmd[-1].endswith("/videos/123"), g_cmd
try:
    sc.stream_urls("https://www.twitch.tv/videos/123", lambda cmd, **kw: Done(1, "", "ERROR: [twitch:vod] 123: Video does not exist"))
    raise AssertionError("an unresolvable VOD must stop the run")
except RuntimeError as err:
    assert "Video does not exist" in str(err), err

cmd = sc.slice_cmd(["https://h/v.m3u8"], 12798.0, 12848.0, tmp / "s.mp4")
assert cmd == ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "warning", "-stats", "-ss", "12798", "-i",
               "https://h/v.m3u8", "-t", "50", "-c", "copy", str(tmp / "s.mp4")], cmd

slice_out = tmp / "slice.mp4"
tried, kwargs = [], []


def flaky(results):
    """A run() whose ffmpeg attempts play out `results` in order: 0 = ok (writes the file), int = exit code,
    "empty" = exit 0 but no bytes, "hang" = timeout."""
    def run(cmd, **kw):
        tried.append(cmd)
        kwargs.append(kw)
        r = results[len(tried) - 1]
        if r == "hang":
            Path(cmd[-1]).write_bytes(b"partial")
            raise sc.subprocess.TimeoutExpired(cmd, kw["timeout"])
        Path(cmd[-1]).write_bytes(b"" if r == "empty" else b"partial" if r else b"slice")
        return Done(0 if r in (0, "empty") else r, "", "frame=  10 fps=0\r[hls] Failed to open segment 1599.ts\n")
    return run


assert sc.download_slice(["u"], 0, 50, slice_out, flaky([0])) is None and len(tried) == 1, "stream copy is enough"
assert "-c" in tried[0] and kwargs[0]["timeout"] == sc.SLICE_TIMEOUT and kwargs[0]["capture_output"], kwargs[0]
tried.clear()
assert sc.download_slice(["u"], 0, 50, slice_out, flaky([1, 0])) is None and slice_out.read_bytes() == b"slice"
assert "libx264" in tried[1] and "aac" in tried[1] and "copy" not in tried[1], "copy failed -> re-encode"
tried.clear()
why = sc.download_slice(["u"], 0, 50, slice_out, flaky([1, "empty"]))
assert why == "[hls] Failed to open segment 1599.ts", why
assert not slice_out.exists() and len(tried) == 2, "a failed slice leaves no half file"
tried.clear()
why = sc.download_slice(["u"], 0, 50, slice_out, flaky(["hang"]))
assert why == f"ffmpeg timed out after {sc.SLICE_TIMEOUT} s" and len(tried) == 1 and not slice_out.exists(), why
assert sc.last_line("frame=  1\rsize=2\r\n[hls] 403 Forbidden\nframe= 3\r") == "[hls] 403 Forbidden"

# one bad slice is skipped with a [SLICE] warning; the others still go to clipper.py, the URL is resolved once
calls.clear()
bad = {"n": 0}


def one_bad(cmd, **kw):
    if cmd[0] == "ffmpeg":
        bad["n"] += 1
        if bad["n"] <= 2:  # moment 1: copy and re-encode both fail
            calls.append(cmd)
            return Done(1, "", "[https] HTTP error 403 Forbidden\n")
    return fake_run(cmd, **kw)


with contextlib.redirect_stdout(io.StringIO()) as o:
    assert sc.main(["https://www.twitch.tv/videos/123", "--chat", str(cache), "--clips", "2"], run=one_bad, opener=g, sleep=nosleep) == 0
assert "[SLICE] Moment 1" in o.getvalue() and "403 Forbidden" in o.getvalue(), o.getvalue()
assert sum("-g" in c for c in calls) == 1 and sum(str(c[-1]).endswith(".mp4") and c[0] == "ffmpeg" for c in calls) == 3, calls
assert sum(any(str(x).endswith("clipper.py") for x in c) for c in calls) == 1, "only the good slice is clipped"

# cache reuse: a second run never touches the network
g = FakeGql(fail={0: AssertionError("network used despite the cache")})
with contextlib.redirect_stdout(io.StringIO()) as o:
    assert sc.main(["https://www.twitch.tv/videos/123", "--clips", "1"], run=fake_run, opener=g, sleep=nosleep) == 0
assert not g.requests and "Using the cached chat replay 123.json" in o.getvalue(), o.getvalue()

# graceful fallback: rate-limited every time / blocked / GraphQL error / failure mid-pagination
for name, g, why in (("429", FakeGql(fail={i: http(429) for i in range(4)}), "HTTP 429 (rate-limited)"),
                     ("403", FakeGql(fail={0: http(403)}), "HTTP 403"),
                     ("gqlerr", FakeGql(pages=[[{"errors": [{"message": "failed integrity check"}]}]]), "failed integrity check"),
                     ("midway", FakeGql(fail={1: http(403)}), "HTTP 403"),
                     ("novod", FakeGql(pages=[[{"data": {"video": None}}]]), "no chat for this VOD")):
    calls.clear()
    sleeps.clear()
    url = f"https://www.twitch.tv/videos/{name}9"
    sc.clipper.fetch_metadata = lambda u, ws, url=url: ({"id": "v", "title": "t", "duration": DUR, "webpage_url": url}, job)
    sc.VOD_RE = __import__("re").compile(r"twitch\.tv/videos/(\w+)")  # word ids so each case has its own cache path
    with contextlib.redirect_stdout(io.StringIO()) as o:
        try:
            sc.main([url], run=fake_run, opener=g, sleep=nosleep)
            raise AssertionError(f"{name}: must stop with the how-to")
        except RuntimeError as err:
            assert "TwitchDownloaderCLI chatdownload" in str(err), err
    assert f"[CHAT] Couldn't fetch the chat automatically: " in o.getvalue() and why in o.getvalue(), (name, o.getvalue())
    assert not calls and not (sc.CHAT_DIR / f"{name}9.json").exists(), f"{name}: nothing downloaded, nothing cached"
    assert len(g.requests) == (4 if name == "429" else 2 if name == "midway" else 1), (name, len(g.requests))
print("spike chat fetch ok")

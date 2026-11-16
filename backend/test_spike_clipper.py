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
fake_run = lambda cmd, check=True: calls.append(cmd)
g = FakeGql()
with contextlib.redirect_stdout(io.StringIO()) as o:
    assert sc.main(["https://www.twitch.tv/videos/123", "--clips", "1"], run=fake_run, opener=g, sleep=nosleep) == 0
cache = sc.CHAT_DIR / "123.json"
assert cache.is_file() and len(g.requests) == 3, o.getvalue()
assert "Saved 210 chat messages" in o.getvalue() and "moment(s) from chat (210 messages)" in o.getvalue(), o.getvalue()
sec = next(c for c in calls if "--download-sections" in c)
a, b = (float(x) for x in sec[sec.index("--download-sections") + 1].lstrip("*").split("-"))
assert a < 1500 - sc.SHIFT < b, (a, b)  # the KEKW burst

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

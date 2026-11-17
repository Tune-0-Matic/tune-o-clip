"""Self-check for the TikTok + YouTube publishers: python publisher/test_publisher.py  (no network: fake APIs)"""
import base64
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import scheduler
import tiktok_poster as tp
import youtube_poster as yp

tmp = Path(tempfile.mkdtemp())
quiet = lambda: contextlib.redirect_stdout(io.StringIO())


def events(text):
    return [json.loads(l) for l in text.splitlines() if l.startswith("{")]


# --- a ready_to_post/ with three shorts (+ one without its video, one already posted) ---------------------------------
ready, posted, hist = tmp / "ready", tmp / "posted", tmp / "hist.json"
ready.mkdir()
man = tmp / "out" / "twitch_X"
man.mkdir(parents=True)
(man / "clips_manifest.json").write_text(json.dumps({"video_id": "111", "clips": [{"clip_id": 1, "score": 0.81}]}))
yt = tmp / "out" / "ytvid"
yt.mkdir(parents=True)
(yt / "clips_manifest.json").write_text(json.dumps({"video_id": "ytvid", "channel": "@SomeYouTuber",
                                                   "clips": [{"clip_id": 2, "score": 0.62}]}))


def short(sid, post, size=2048):
    (ready / f"{sid}_short.mp4").write_bytes(b"\0" * size)
    (ready / f"{sid}_post.json").write_text(json.dumps(post))


short("111_1", {"title": '"Spoke too soon"\nlol', "social_title": "Spoke too soon #a", "hashtags": ["#GTAV", "#CaseOh_", "#shorts", "#fyp"],
                "transcript": "Watch this, I literally never miss this jump. NOOO! I'm dead chat.", "duration": 31.0,
                "credit": {"broadcaster_name": "caseoh_", "view_count": 3009}, "qc": {"mean_db": -14.2, "duration": 31.0},
                "source": {"manifest": str(man / "clips_manifest.json"), "clip_id": 1}})
short("ytvid_2", {"title": "A very long title that keeps going and going well past the fifty five character limit for sure",
                  "hashtags": ["#minecraft"], "transcript": "So I built a house. It fell down.", "duration": 20.0,
                  "qc": {"mean_db": -13.8}, "source": {"manifest": str(yt / "clips_manifest.json"), "clip_id": 2}})
short("lost_1", {"title": "manifest gone", "transcript": "x", "duration": 70.0, "qc": {"mean_db": -25},
                 "source": {"manifest": str(tmp / "nope.json"), "clip_id": 1}})
(ready / "novideo_1_post.json").write_text(json.dumps({"title": "no mp4"}))  # its _short.mp4 is missing
short("old_1", {"title": "posted last week", "source": {}})
scheduler.save_history({"posts": [{"short_id": "old_1", "posted_at": time.time() - 7 * 86400, "status": "PUBLISH_COMPLETE"}]}, hist)

# --- best-short ranking: clip score (manifest) + loudness + length health; posted / video-less ones never --------------
pool = scheduler.candidates(ready, scheduler.load_history(hist))
assert [e["id"] for e in pool] == ["111_1", "ytvid_2", "lost_1"], [e["id"] for e in pool]
assert pool[0]["quality"] > pool[1]["quality"] > pool[2]["quality"], [e["quality"] for e in pool]
assert abs(pool[2]["quality"] - (0.8 * 0.5 + 0.1 * 0.0 + 0.1 * 0.5)) < 1e-9  # no manifest -> 0.5; -25 dB; 70 s

# --- exactly 5 hashtags, lowercase, deduplicated, streamer credited ----------------------------------------------------------
p1, p2 = pool[0]["post"], pool[1]["post"]
assert scheduler.hashtags(p1) == ["#fyp", "#gaming", "#twitchclips", "#caseoh_", "#gtav"], scheduler.hashtags(p1)
assert scheduler.hashtags(p2) == ["#fyp", "#gaming", "#clips", "#someyoutuber", "#minecraft"], scheduler.hashtags(p2)
assert scheduler.hashtags({"hashtags": []}) == ["#fyp", "#gaming", "#clips", "#funny", "#viral"]  # padded to 5
for p in (p1, p2, {"hashtags": ["#A", "#a", "#a!", "#b", "#c", "#d", "#e"]}):
    tags = scheduler.hashtags(p)
    assert len(tags) == 5 and len(set(tags)) == 5 and all(t == t.lower() and t.startswith("#") for t in tags), tags

# --- caption: clean title (< 55), a one-line hook quote, the credit, tags on line 2 ----------------------------------------
title, text = scheduler.caption(p1)
assert title == "Spoke too soon lol", title
assert text == 'Spoke too soon lol — "Watch this, I literally never miss this jump." (via @caseoh_ on Twitch)\n' \
               "#fyp #gaming #twitchclips #caseoh_ #gtav", text
title2, text2 = scheduler.caption(p2)
assert len(title2) <= 55 and title2.endswith("…") and "(via @SomeYouTuber on YouTube)" in text2, (title2, text2)
assert len(text) < 2200 and len(text.splitlines()) == 2

# --- dry run: picks + formats, touches nothing ------------------------------------------------------------------------------
with quiet() as out:
    assert scheduler.run(dry_run=True, platforms=("tiktok",), ready=ready, posted=posted, history_path=hist) == 0
ev = [e for e in events(out.getvalue()) if e["event"] == "publish"][0]
assert ev["dry_run"] and ev["short_id"] == "111_1" and ev["caption"] == text and len(ev["hashtags"]) == 5
assert (ready / "111_1_short.mp4").exists() and not posted.exists() and len(scheduler.load_history(hist)["posts"]) == 1


class FakePoster:
    def __init__(self):
        self.calls = []

    def post(self, video, caption, privacy):
        self.calls.append((Path(video).name, caption, privacy))
        return {"publish_id": f"pub{len(self.calls)}", "status": "PUBLISH_COMPLETE", "post_id": "7300", "privacy": privacy,
                "creator": "me"}


# --- post: history recorded, files moved to posted/, never twice --------------------------------------------------------------
fp, t0 = FakePoster(), time.time()
with quiet() as out:
    assert scheduler.run(platforms=("tiktok",), posters={"tiktok": fp}, ready=ready, posted=posted, history_path=hist, now=t0) == 0
assert fp.calls == [("111_1_short.mp4", text, "PUBLIC_TO_EVERYONE")]
assert (posted / "111_1_short.mp4").exists() and (posted / "111_1_post.json").exists() and not (ready / "111_1_short.mp4").exists()
h = scheduler.load_history(hist)["posts"][-1]
assert h["platform"] == "tiktok" and h["short_id"] == "111_1" and h["publish_id"] == "pub1" and h["posted_at"] == t0 and h["status"] == "PUBLISH_COMPLETE"
assert "[PUBLISH] Posted 111_1 to TikTok" in out.getvalue()

# --- 3.5 h pacing: not again 1 h later; --force overrides; allowed again after 3.5 h --------------------------------------------
with quiet() as out:
    scheduler.run(platforms=("tiktok",), posters={"tiktok": fp}, ready=ready, posted=posted, history_path=hist, now=t0 + 3600)
assert len(fp.calls) == 1 and "TikTok: next post allowed in 150 min" in out.getvalue(), out.getvalue()
with quiet():
    scheduler.run(platforms=("tiktok",), posters={"tiktok": fp}, ready=ready, posted=posted, history_path=hist, now=t0 + 3600, force=True)
assert [c[0] for c in fp.calls] == ["111_1_short.mp4", "ytvid_2_short.mp4"], "forced: the next best"
with quiet():
    scheduler.run(platforms=("tiktok",), posters={"tiktok": fp}, ready=ready, posted=posted, history_path=hist, now=t0 + 3600 + 3.4 * 3600)
assert len(fp.calls) == 2, "3.4 h after the last post: still waiting"
with quiet():
    scheduler.run(platforms=("tiktok",), posters={"tiktok": fp}, ready=ready, posted=posted, history_path=hist, now=t0 + 3600 + 3.5 * 3600, interval_h=3.5)
assert len(fp.calls) == 3 and fp.calls[-1][0] == "lost_1_short.mp4"
with quiet() as out:
    scheduler.run(platforms=("tiktok",), posters={"tiktok": fp}, ready=ready, posted=posted, history_path=hist, force=True)
assert len(fp.calls) == 3 and "Nothing to post" in out.getvalue()
assert scheduler.wait_left({"posts": [{"posted_at": 0, "status": "FAILED"}]}, 3.5, 100) == 0, "a failed post doesn't pace"


# --- the TikTok API client against a fake TikTok -------------------------------------------------------------------------------
class Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


class FakeTikTok:
    def __init__(self, privacy=("PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS", "SELF_ONLY"), statuses=("PROCESSING_UPLOAD", "PUBLISH_COMPLETE")):
        self.privacy, self.statuses, self.reqs, self.chunks, self.token_calls = list(privacy), list(statuses), [], [], []

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.reqs.append(req)
        if url == tp.TOKEN_URL:
            form = dict(x.split("=", 1) for x in req.data.decode().split("&"))
            self.token_calls.append(form)
            if form.get("code") == "BADCODE":
                raise urllib.error.HTTPError(url, 400, "bad", {}, io.BytesIO(b'{"error": "invalid_client", "error_description": "Client key or secret is incorrect."}'))
            if form.get("refresh_token") == "dead":
                raise urllib.error.HTTPError(url, 400, "bad", {}, io.BytesIO(b'{"error": "invalid_grant", "error_description": "refresh token expired"}'))
            return Resp(json.dumps({"access_token": "AT" + str(len(self.token_calls)), "refresh_token": "RT" + str(len(self.token_calls)),
                                    "expires_in": 7200, "refresh_expires_in": 15552000, "open_id": "oid", "scope": tp.SCOPES}).encode())
        if url.startswith("https://open-upload.tiktokapis.com/"):  # the signed upload_url: no bearer token
            assert req.get_method() == "PUT" and "Authorization" not in req.headers
            self.chunks.append((req.headers["Content-range"], int(req.headers["Content-length"]), len(req.data)))
            return Resp(b"")
        assert req.headers.get("Authorization", "").startswith("Bearer AT"), req.headers
        if url.endswith("/creator_info/query/"):
            return Resp(json.dumps({"data": {"creator_username": "me", "privacy_level_options": self.privacy,
                                             "comment_disabled": False, "duet_disabled": True, "stitch_disabled": False},
                                    "error": {"code": "ok"}}).encode())
        if url.endswith("/video/init/"):
            self.init = json.loads(req.data)
            return Resp(json.dumps({"data": {"publish_id": "v_pub~1", "upload_url": "https://open-upload.tiktokapis.com/upload/?id=1"},
                                    "error": {"code": "ok"}}).encode())
        if url.endswith("/status/fetch/"):
            assert json.loads(req.data) == {"publish_id": "v_pub~1"}
            st = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return Resp(json.dumps({"data": {"status": st, "publicaly_available_post_id": [7311] if st == "PUBLISH_COMPLETE" else [],
                                             "fail_reason": "spam_risk_too_many_posts" if st == "FAILED" else ""},
                                    "error": {"code": "ok"}}).encode())
        raise AssertionError(url)


# PKCE: TikTok desktop wants hex(SHA-256(verifier)), not base64url; chunk plan per the media-transfer rules
v, ch = tp.pkce()
assert 43 <= len(v) <= 128 and ch == hashlib.sha256(v.encode()).hexdigest() and len(ch) == 64
assert tp.plan_chunks(4_194_304) == (4_194_304, 1) and tp.plan_chunks(30_000_000) == (30_000_000, 1)
assert tp.plan_chunks(150_000_123) == (10_000_000, 15)  # floor; the last chunk takes the 123 extra bytes
assert tp.choose_privacy(["PUBLIC_TO_EVERYONE", "SELF_ONLY"], "PUBLIC_TO_EVERYONE") == "PUBLIC_TO_EVERYONE"
assert tp.choose_privacy(["FOLLOWER_OF_CREATOR", "MUTUAL_FOLLOW_FRIENDS", "SELF_ONLY"], "PUBLIC_TO_EVERYONE") == "SELF_ONLY"
try:
    tp.TikTok(client_key="", client_secret="")
    raise AssertionError("missing keys must fail")
except tp.TikTokError as e:
    assert "TIKTOK_CLIENT_KEY" in str(e)

# one-time login: browser opened with PKCE + state, loopback redirect caught, code traded, token saved
fake = FakeTikTok()
tok_file = tmp / "tiktok_token.json"
tt = tp.TikTok("ck", "cs", tok_file, opener=fake, sleep=lambda s: None)
opened = []


def browser(url):  # plays the user approving in the browser: TikTok redirects to our loopback port
    opened.append(url)
    q = dict(x.split("=", 1) for x in url.split("?", 1)[1].split("&"))
    target = f"http://127.0.0.1:{port}/callback/?code=THECODE&scopes=video.publish&state={q['state']}"
    threading.Thread(target=lambda: urllib.request.urlopen(target, timeout=10).read(), daemon=True).start()


port = 34555
tok = tt.login(open_browser=browser, timeout=20, port=port)
q = dict(x.split("=", 1) for x in opened[0].split("?", 1)[1].split("&"))
assert opened[0].startswith(tp.AUTH_URL) and q["client_key"] == "ck" and q["code_challenge_method"] == "S256" and len(q["code_challenge"]) == 64
assert fake.token_calls[0]["code"] == "THECODE" and fake.token_calls[0]["grant_type"] == "authorization_code"
assert len(fake.token_calls[0]["code_verifier"]) >= 43 and hashlib.sha256(fake.token_calls[0]["code_verifier"].encode()).hexdigest() == q["code_challenge"]
assert tok["access_token"] == "AT1" and json.loads(tok_file.read_text())["refresh_token"] == "RT1"
assert tp.TOKEN_URL == "https://open.tiktokapis.com/v2/oauth/token/" and tp.AUTH_URL == "https://www.tiktok.com/v2/auth/authorize/"
assert tp.API + "/creator_info/query/" == "https://open.tiktokapis.com/v2/post/publish/creator_info/query/"
assert fake.reqs[0].headers["Content-type"] == "application/x-www-form-urlencoded"
assert set(fake.token_calls[0]) == {"client_key", "client_secret", "code", "grant_type", "redirect_uri", "code_verifier"}
assert urllib.parse.unquote(fake.token_calls[0]["redirect_uri"]) == tp.REDIRECT_URI == "http://127.0.0.1:3455/callback/"
assert urllib.parse.unquote(q["redirect_uri"]) == tp.REDIRECT_URI, "the authorize and token calls send the same redirect URI"
# a failed FIRST login names TikTok's reason (not "session expired")
try:
    tt._token_request({"code": "BADCODE", "grant_type": "authorization_code", "redirect_uri": tp.REDIRECT_URI, "code_verifier": v})
    raise AssertionError("bad login must raise")
except tp.SessionExpired:
    raise AssertionError("a failed first login is not an expired session")
except tp.TikTokError as e:
    assert "TikTok login failed: Client key or secret is incorrect." in str(e) and tp.REDIRECT_URI in str(e), e

# post: creator info -> init (caption, privacy, the creator's interaction settings) -> one PUT -> status until complete
vid = tmp / "v.mp4"
vid.write_bytes(b"\1" * 3000)
res = tt.post(vid, "Title — hook #fyp #a #b #c #d", "PUBLIC_TO_EVERYONE")
assert res == {"publish_id": "v_pub~1", "status": "PUBLISH_COMPLETE", "privacy": "PUBLIC_TO_EVERYONE", "post_id": "7311", "creator": "me"}, res
assert fake.init["post_info"] == {"title": "Title — hook #fyp #a #b #c #d", "privacy_level": "PUBLIC_TO_EVERYONE",
                                  "disable_comment": False, "disable_duet": True, "disable_stitch": False}, fake.init
assert fake.init["source_info"] == {"source": "FILE_UPLOAD", "video_size": 3000, "chunk_size": 3000, "total_chunk_count": 1}
first_chunks = list(fake.chunks)
fake.statuses = ["PUBLISH_COMPLETE"]
tt.post(vid, "x", "SELF_ONLY", allow_comment=False, allow_stitch=False)  # our toggles can only turn things off
assert {k: fake.init["post_info"][k] for k in ("disable_comment", "disable_duet", "disable_stitch")} ==     {"disable_comment": True, "disable_duet": True, "disable_stitch": True}, fake.init
fake.chunks[:] = first_chunks
assert fake.chunks == [("bytes 0-2999/3000", 3000, 3000)], fake.chunks

# chunked upload (> 64 MB): floor(size / 10 MB) chunks, the last one takes the remainder, sequential ranges
big = tmp / "big.mp4"
with open(big, "wb") as f:
    f.truncate(70_000_005)
fake2 = FakeTikTok(privacy=("FOLLOWER_OF_CREATOR", "SELF_ONLY"))  # a private account: public isn't offered
tok_file.write_text(json.dumps({"access_token": "AT9", "refresh_token": "RT9", "expires_at": time.time() + 3600, "refresh_expires_at": time.time() + 9e6}))
res = tp.TikTok("ck", "cs", tok_file, opener=fake2, sleep=lambda s: None).post(big, "x", "PUBLIC_TO_EVERYONE")
assert res["privacy"] == "SELF_ONLY" and fake2.init["source_info"]["total_chunk_count"] == 7
assert [c[0] for c in fake2.chunks] == [f"bytes {i * 10_000_000}-{i * 10_000_000 + 9_999_999}/70000005" for i in range(6)] + \
    ["bytes 60000000-70000004/70000005"] and fake2.chunks[-1][1] == 10_000_005

# expired access token -> refreshed with the rotating refresh token (the new one is stored)
fake3 = FakeTikTok()
tok_file.write_text(json.dumps({"access_token": "OLD", "refresh_token": "RT_OLD", "expires_at": time.time() - 10, "refresh_expires_at": time.time() + 9e6}))
tp.TikTok("ck", "cs", tok_file, opener=fake3, sleep=lambda s: None).creator_info()
assert fake3.token_calls == [{"client_key": "ck", "client_secret": "cs", "grant_type": "refresh_token", "refresh_token": "RT_OLD"}]
assert json.loads(tok_file.read_text())["refresh_token"] == "RT1", "rotated refresh token saved"

# refresh refused / never logged in -> SessionExpired with the banner's wording
for bad in ({"access_token": "OLD", "refresh_token": "dead", "expires_at": 0, "refresh_expires_at": time.time() + 9e6}, None):
    if bad:
        tok_file.write_text(json.dumps(bad))
    else:
        tok_file.unlink()
    try:
        tp.TikTok("ck", "cs", tok_file, opener=FakeTikTok()).creator_info()
        raise AssertionError("expired session must raise")
    except tp.SessionExpired as e:
        assert str(e).startswith("[PUBLISH] TikTok session expired: click Login TikTok"), e

# TikTok rejecting the post (FAILED) and API errors surface clearly
tok_file.write_text(json.dumps({"access_token": "AT9", "refresh_token": "RT9", "expires_at": time.time() + 3600, "refresh_expires_at": time.time() + 9e6}))
try:
    tp.TikTok("ck", "cs", tok_file, opener=FakeTikTok(statuses=("FAILED",)), sleep=lambda s: None).post(vid, "x")
    raise AssertionError("FAILED must raise")
except tp.TikTokError as e:
    assert "spam_risk_too_many_posts" in str(e)


def unaudited(req, timeout=None):
    if req.full_url.endswith("/creator_info/query/"):
        return Resp(json.dumps({"data": {"privacy_level_options": ["PUBLIC_TO_EVERYONE", "SELF_ONLY"]}, "error": {"code": "ok"}}).encode())
    raise urllib.error.HTTPError(req.full_url, 403, "no", {}, io.BytesIO(json.dumps({"error": {
        "code": "unaudited_client_can_only_post_to_private_accounts", "message": "set the account to private"}}).encode()))


try:
    tp.TikTok("ck", "cs", tok_file, opener=unaudited).post(vid, "x")
    raise AssertionError
except tp.TikTokError as e:
    assert "unaudited_client_can_only_post_to_private_accounts" in str(e), e

# --- YouTube Shorts: metadata ---------------------------------------------------------------------------------------------------
yt_title, yt_desc, yt_tags = scheduler.youtube_meta(p1)
assert yt_title == "Spoke too soon lol #Shorts", yt_title
assert yt_desc == ("Spoke too soon lol\n\nWatch this, I literally never miss this jump. (Credit: @caseoh_ on Twitch)\n\n"
                   "#fyp #gaming #twitchclips #caseoh_ #gtav"), yt_desc
assert yt_tags == ["fyp", "gaming", "twitchclips", "caseoh_", "gtav", "shorts"], yt_tags  # "gaming" not repeated
long_t, _, _ = scheduler.youtube_meta({"title": "word " * 40 + "<script>"})
assert len(long_t) <= 100 and long_t.endswith(" #Shorts") and "<" not in long_t, long_t
assert scheduler.youtube_meta({"title": ""})[0] == "Funny gaming moment #Shorts"


# --- the YouTube API client against a fake Google ---------------------------------------------------------------------------------
class HResp(Resp):
    def __init__(self, body=b"", headers=None):
        super().__init__(body)
        self.headers = headers or {}


SESSION = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&upload_id=XYZ"


class FakeGoogle:
    """Token endpoint + resumable videos.insert. `plan`: per chunk PUT, "ok" | "short" (keeps 256 KiB less) | "drop"
    (stores it, then the connection fails with a 503)."""
    def __init__(self, plan=(), stale=(), quota=False):
        self.plan, self.stale, self.quota = list(plan), set(stale), quota
        self.token_calls, self.puts, self.meta, self.init_headers, self.have, self.checks = [], [], None, None, 0, 0

    def __call__(self, req, timeout=None):
        url, h = req.full_url, {k.lower(): v for k, v in req.header_items()}
        if url == yp.TOKEN_URL:
            form = {k: v[0] for k, v in urllib.parse.parse_qs(req.data.decode()).items()}
            self.token_calls.append(form)
            if form.get("refresh_token") == "dead" or form.get("code") == "BADCODE":
                raise urllib.error.HTTPError(url, 400, "bad", {}, io.BytesIO(b'{"error": "invalid_grant", "error_description": "Token has been expired or revoked."}'))
            out = {"access_token": "YA" + str(len(self.token_calls)), "expires_in": 3599, "scope": yp.SCOPE, "token_type": "Bearer"}
            if form["grant_type"] == "authorization_code":
                out["refresh_token"] = "1//RT"
            return Resp(json.dumps(out).encode())
        tok = h.get("authorization", "")[len("Bearer "):]
        if tok in self.stale:
            raise urllib.error.HTTPError(url, 401, "unauthorized", {}, io.BytesIO(b'{"error": {"code": 401, "errors": [{"reason": "authError"}]}}'))
        if url == yp.UPLOAD_URL:
            assert req.get_method() == "POST"
            if self.quota:
                raise urllib.error.HTTPError(url, 403, "forbidden", {}, io.BytesIO(json.dumps({"error": {"code": 403,
                    "message": "The request cannot be completed because you have exceeded your quota.",
                    "errors": [{"reason": "quotaExceeded", "domain": "youtube.quota"}]}}).encode()))
            self.meta, self.init_headers, self.size = json.loads(req.data), h, int(h["x-upload-content-length"])
            return HResp(b"", {"Location": SESSION})
        assert url == SESSION and req.get_method() == "PUT", url
        rng = h["content-range"]
        if rng == f"bytes */{self.size}":  # status check
            self.checks += 1
            return self._answer()
        self.puts.append((rng, len(req.data)))
        a, b = map(int, rng.split(" ")[1].split("/")[0].split("-"))
        assert a == self.have and b - a + 1 == len(req.data), (rng, self.have)
        act = self.plan.pop(0) if self.plan else "ok"
        self.have = b + 1 - (256 * 1024 if act == "short" else 0)
        if act == "drop":
            raise urllib.error.HTTPError(url, 503, "backend error", {}, io.BytesIO(b"{}"))
        return self._answer()

    def _answer(self):
        if self.have == self.size:
            return HResp(json.dumps({"id": "dQw4w9WgXcQ", "snippet": self.meta["snippet"],
                                     "status": {"privacyStatus": self.meta["status"]["privacyStatus"]}}).encode())
        raise urllib.error.HTTPError(SESSION, 308, "Resume Incomplete", {"Range": f"bytes=0-{self.have - 1}"} if self.have else {}, io.BytesIO(b""))


yp_tok = tmp / "youtube_token.json"
# keys from workspace/youtube_client_secret.json when .env has none; neither -> a clear error
cf = tmp / "client.json"
cf.write_text(json.dumps({"installed": {"client_id": "cid.apps.googleusercontent.com", "client_secret": "GOCSPX-x"}}))
with contextlib.ExitStack() as st:
    import os
    saved = {k: os.environ.pop(k, None) for k in ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET")}
    st.callback(lambda: os.environ.update({k: v for k, v in saved.items() if v}))
    yp.twitch_api.load_env = lambda *a, **k: None  # this test must not read the real .env
    assert yp.YouTube(token_path=yp_tok, client_file=cf).cid == "cid.apps.googleusercontent.com"
    try:
        yp.YouTube(token_path=yp_tok, client_file=tmp / "none.json")
        raise AssertionError("missing YouTube keys must fail")
    except yp.YouTubeError as e:
        assert "YOUTUBE_CLIENT_ID" in str(e)

# PKCE S256 the standard way: base64url(SHA-256), no padding
v, ch = yp.pkce()
assert ch == base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).rstrip(b"=").decode() and "=" not in ch and len(ch) == 43

# one-time login: consent URL (offline + consent, upload scope, PKCE), redirect on 127.0.0.1:<port>/callback, code traded
g = FakeGoogle()
yt_ = yp.YouTube("cid", "csec", yp_tok, opener=g, sleep=lambda s: None)
opened = []


def g_browser(url):
    opened.append(url)
    st = urllib.parse.parse_qs(url.split("?", 1)[1])["state"][0]
    threading.Thread(target=lambda: urllib.request.urlopen(f"http://127.0.0.1:{port + 1}/callback?code=4/GCODE&state={st}",
                                                           timeout=10).read(), daemon=True).start()


tok = yt_.login(open_browser=g_browser, timeout=20, port=port + 1)
q = {k: v[0] for k, v in urllib.parse.parse_qs(opened[0].split("?", 1)[1]).items()}
assert opened[0].startswith("https://accounts.google.com/o/oauth2/v2/auth?")
assert q["scope"] == "https://www.googleapis.com/auth/youtube.upload" and q["access_type"] == "offline" and q["prompt"] == "consent"
assert q["code_challenge_method"] == "S256" and q["response_type"] == "code" and q["redirect_uri"] == "http://127.0.0.1:3456/callback"
tc = g.token_calls[0]
assert set(tc) == {"client_id", "client_secret", "code", "code_verifier", "grant_type", "redirect_uri"} and tc["code"] == "4/GCODE"
assert base64.urlsafe_b64encode(hashlib.sha256(tc["code_verifier"].encode()).digest()).rstrip(b"=").decode() == q["code_challenge"]
assert tc["redirect_uri"] == q["redirect_uri"] and yp.TOKEN_URL == "https://oauth2.googleapis.com/token"
assert json.loads(yp_tok.read_text())["refresh_token"] == "1//RT" and tok["access_token"] == "YA1"

# upload: init (metadata, sizes) -> 2 MiB chunks; a 308 that kept less, a dropped chunk (503 -> backoff -> status check)
yv = tmp / "yt.mp4"
yv.write_bytes(bytes(range(256)) * (2 * yp.CHUNK // 256) + b"\7" * 1000)
size, C, K = yv.stat().st_size, yp.CHUNK, 256 * 1024
g = FakeGoogle(plan=["short", "drop", "ok"])
naps = []
res = yp.YouTube("cid", "csec", yp_tok, opener=g, sleep=naps.append).upload(yv, yt_title, yt_desc, yt_tags, "unlisted")
assert res == {"video_id": "dQw4w9WgXcQ", "url": "https://youtube.com/shorts/dQw4w9WgXcQ", "privacy": "unlisted"}, res
assert g.meta == {"snippet": {"title": "Spoke too soon lol #Shorts", "description": yt_desc, "tags": yt_tags, "categoryId": "20"},
                  "status": {"privacyStatus": "unlisted", "selfDeclaredMadeForKids": False}}, g.meta
assert g.init_headers["x-upload-content-length"] == str(size) and g.init_headers["x-upload-content-type"] == "video/*"
assert yp.UPLOAD_URL.startswith("https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet")
assert g.puts == [(f"bytes 0-{C - 1}/{size}", C), (f"bytes {C - K}-{2 * C - K - 1}/{size}", C),
                  (f"bytes {2 * C - K}-{size - 1}/{size}", size - (2 * C - K))], g.puts
assert all(n % K == 0 for _, n in g.puts[:-1]) and g.checks == 1 and naps == [2]
# scheduled publishing: publishAt goes in status, the upload is private until then (the API's rule)
g = FakeGoogle()
res = yp.YouTube("cid", "csec", yp_tok, opener=g).upload(vid, "t", "d", [], "public", publish_at="2026-11-17T14:00:00Z")
assert g.meta["status"] == {"privacyStatus": "private", "selfDeclaredMadeForKids": False, "publishAt": "2026-11-17T14:00:00Z"}, g.meta
assert res["publish_at"] == "2026-11-17T14:00:00Z", res
g = FakeGoogle()
yp.YouTube("cid", "csec", yp_tok, opener=g).upload(vid, "t", "d", [], "unlisted", category="24", made_for_kids=True)
assert g.meta["snippet"]["categoryId"] == "24" and g.meta["status"] == {"privacyStatus": "unlisted", "selfDeclaredMadeForKids": True}, g.meta
# channels.list(mine=true): which channel a login is; the extra read scope only when asked for
chan_reqs = []
def chan_opener(req, timeout=None):
    chan_reqs.append(req)
    return Resp(json.dumps({"items": [{"id": "UCabc", "snippet": {"title": "GTA Clips"}}]}).encode())
ro = yp.YouTube("cid", "csec", yp_tok, opener=chan_opener, scope=yp.SCOPE + " " + yp.READ_SCOPE)
assert ro.channel() == {"id": "UCabc", "title": "GTA Clips"} and chan_reqs[0].full_url == yp.CHANNELS_URL
assert "youtube.readonly" in urllib.parse.parse_qs(urllib.parse.urlparse(ro.auth_url("s", "c")).query)["scope"][0]
try:
    yp.YouTube("cid", "csec", yp_tok, opener=lambda req, timeout=None: Resp(b'{"items": []}')).channel()
    raise AssertionError("no channel must raise")
except yp.YouTubeError as e:
    assert "no YouTube channel" in str(e)

# expired access token mid-way: one 401 -> refresh (Google sends no new refresh token: the old one is kept) -> retried
yp_tok.write_text(json.dumps({"access_token": "OLD", "refresh_token": "1//RT", "expires_at": time.time() + 3600}))
g = FakeGoogle(stale=["OLD"])
assert yp.YouTube("cid", "csec", yp_tok, opener=g).upload(vid, "t #Shorts", "d", [], "public")["video_id"] == "dQw4w9WgXcQ"
assert g.token_calls == [{"client_id": "cid", "client_secret": "csec", "grant_type": "refresh_token", "refresh_token": "1//RT"}]
assert json.loads(yp_tok.read_text()) | {"expires_at": 0} == {"access_token": "YA1", "refresh_token": "1//RT", "scope": yp.SCOPE, "expires_at": 0}

# quotaExceeded (403) -> the banner's wording; a dead refresh token -> "token expired"
try:
    yp.YouTube("cid", "csec", yp_tok, opener=FakeGoogle(quota=True)).upload(vid, "t", "d", [], "public")
    raise AssertionError("quota must raise")
except yp.YouTubeError as e:
    assert str(e).startswith("[PUBLISH] YouTube API quota exceeded"), e
yp_tok.write_text(json.dumps({"access_token": "OLD", "refresh_token": "dead", "expires_at": 0}))
try:
    yp.YouTube("cid", "csec", yp_tok, opener=FakeGoogle()).upload(vid, "t", "d", [], "public")
    raise AssertionError("dead refresh token must raise")
except yp.YouTubeExpired as e:
    assert str(e).startswith("[PUBLISH] YouTube OAuth token expired: click Login YouTube (1-Time)"), e


# --- both platforms: independent 3.5 h cooldowns, posted/ only once both have it, one failing spares the other ---------------
class FakeYT:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def upload(self, video, title, description, tags, privacy):
        if self.fail:
            raise yp.YouTubeError(self.fail)
        self.calls.append((Path(video).name, title, privacy))
        return {"video_id": f"v{len(self.calls)}", "url": f"https://youtube.com/shorts/v{len(self.calls)}", "privacy": privacy}


ready2, posted2, hist2 = tmp / "ready2", tmp / "posted2", tmp / "hist2.json"
ready2.mkdir()
for sid, score in (("a_1", -13), ("b_1", -20)):
    (ready2 / f"{sid}_short.mp4").write_bytes(b"x")
    (ready2 / f"{sid}_post.json").write_text(json.dumps({"title": sid, "qc": {"mean_db": score}, "duration": 30}))
both = ("youtube", "tiktok")
fy, ft = FakeYT(), FakePoster()
kw = dict(platforms=both, posters={"youtube": fy, "tiktok": ft}, ready=ready2, posted=posted2, history_path=hist2)
with quiet() as out:
    scheduler.run(now=t0, youtube_privacy="private", **kw)
assert fy.calls == [("a_1_short.mp4", "a_1 #Shorts", "private")] and [c[0] for c in ft.calls] == ["a_1_short.mp4"]
assert "[PUBLISH] Posted to YouTube Shorts: https://youtube.com/shorts/v1" in out.getvalue() and (posted2 / "a_1_short.mp4").exists()
hh = scheduler.load_history(hist2)["posts"]
assert [(p["short_id"], p["platform"], p["status"]) for p in hh] == [("a_1", "youtube", "UPLOADED"), ("a_1", "tiktok", "PUBLISH_COMPLETE")]
# t0+2 h: neither platform is due
with quiet() as out:
    scheduler.run(now=t0 + 2 * 3600, **kw)
assert len(ft.calls) == 1 and out.getvalue().count("next post allowed in 90 min") == 2, out.getvalue()
# t0+3.5 h: YouTube fails (quota), TikTok still posts b_1; b_1 stays in ready2/ for YouTube's next try
with quiet() as out:
    rc = scheduler.run(now=t0 + 3.5 * 3600, platforms=both, posters={"youtube": FakeYT(fail=yp.QUOTA_MSG), "tiktok": ft},
                       ready=ready2, posted=posted2, history_path=hist2)
errs = [e["msg"] for e in events(out.getvalue()) if e["event"] == "error"]
assert rc == 1 and errs == [yp.QUOTA_MSG] and ft.calls[-1][0] == "b_1_short.mp4" and (ready2 / "b_1_short.mp4").exists()
# t0+4 h: YouTube's own cooldown (since t0) is over, TikTok's (since t0+3.5 h) is not -> YouTube alone posts b_1
with quiet() as out:
    scheduler.run(now=t0 + 4 * 3600, **kw)
assert len(ft.calls) == 2 and [c[0] for c in fy.calls] == ["a_1_short.mp4", "b_1_short.mp4"], (ft.calls, fy.calls)
assert "TikTok: every short is already posted there" in out.getvalue() and (posted2 / "b_1_short.mp4").exists(), out.getvalue()
h2 = scheduler.load_history(hist2)
assert scheduler.wait_left(h2, 3.5, t0 + 4 * 3600, "tiktok") == 3 * 3600 and scheduler.wait_left(h2, 3.5, t0 + 4 * 3600, "youtube") == 3.5 * 3600
# single platform: posted/ right away
(ready2 / "c_1_short.mp4").write_bytes(b"x")
(ready2 / "c_1_post.json").write_text(json.dumps({"title": "c"}))
with quiet():
    scheduler.run(now=t0 + 8 * 3600, platforms=("tiktok",), posters={"tiktok": ft}, ready=ready2, posted=posted2, history_path=hist2)
assert ft.calls[-1][0] == "c_1_short.mp4" and (posted2 / "c_1_short.mp4").exists()
# switching Both -> TikTok only: a short TikTok already has leaves ready/
(ready2 / "d_1_short.mp4").write_bytes(b"x")
(ready2 / "d_1_post.json").write_text(json.dumps({"title": "d"}))
with quiet():
    scheduler.run(now=t0 + 12 * 3600, platforms=both, posters={"youtube": FakeYT(fail="[PUBLISH] down"), "tiktok": ft},
                  ready=ready2, posted=posted2, history_path=hist2)
    assert (ready2 / "d_1_short.mp4").exists()
    scheduler.run(now=t0 + 12 * 3600, platforms=("tiktok",), posters={"tiktok": ft}, ready=ready2, posted=posted2, history_path=hist2)
assert (posted2 / "d_1_short.mp4").exists() and not list(ready2.iterdir())
# one chosen short, now (the panel's per-clip buttons): even from posted/, interval ignored, nothing moved
fy2 = FakeYT()
with quiet():
    assert scheduler.run(now=t0 + 12 * 3600 + 60, platforms=("youtube",), posters={"youtube": fy2}, ready=ready2,
                         posted=posted2, history_path=hist2, short_id="c_1") == 0
assert fy2.calls == [("c_1_short.mp4", "c #Shorts", "public")] and (posted2 / "c_1_short.mp4").exists()
(ready2 / "e_1_short.mp4").write_bytes(b"x")
(ready2 / "e_1_post.json").write_text(json.dumps({"title": "e"}))
(ready2 / "f_1_short.mp4").write_bytes(b"x")
(ready2 / "f_1_post.json").write_text(json.dumps({"title": "f", "qc": {"mean_db": -14}}))
with quiet():
    scheduler.run(now=t0 + 12 * 3600 + 120, platforms=("youtube",), posters={"youtube": fy2}, ready=ready2, posted=posted2,
                  history_path=hist2, short_id="e_1")
assert fy2.calls[-1][0] == "e_1_short.mp4" and (ready2 / "e_1_short.mp4").exists(), "not the best one, and not moved"
with quiet() as out:
    scheduler.run(now=t0 + 12 * 3600 + 180, platforms=("youtube",), posters={"youtube": fy2}, ready=ready2, posted=posted2,
                  history_path=hist2, short_id="e_1")
assert len(fy2.calls) == 2 and "e_1: already on YouTube Shorts" in out.getvalue(), out.getvalue()
for f in ready2.iterdir():
    f.unlink()
# old history entries (no "platform") count as TikTok
assert scheduler.wait_left({"posts": [{"posted_at": 100, "status": "PUBLISH_COMPLETE"}]}, 1, 100, "youtube") == 0
assert scheduler.wait_left({"posts": [{"posted_at": 100, "status": "PUBLISH_COMPLETE"}]}, 1, 100, "tiktok") == 3600

# --- the CLI end to end: --dry-run per platform ---------------------------------------------------
for pf in ("youtube", "tiktok", "both"):  # no API keys needed, nothing moved
    cli = subprocess.run([sys.executable, str(Path(scheduler.__file__)), "--dry-run", "--platform", pf],
                         capture_output=True, text=True, timeout=60)
    assert cli.returncode == 0 and any(e["event"] == "log" for e in events(cli.stdout)), cli.stdout + cli.stderr

# 30-day guard: a short of a moment posted in the last 30 days is left out; the same short still goes to another platform
g = tmp / "guard"
g.mkdir()
same = "he walked straight into the wall and then blamed chat for it again"
for sid, words in (("a_1", same), ("b_1", same + " lol"), ("c_1", "a totally different clip about cooking pasta at midnight")):
    (g / f"{sid}_post.json").write_text(json.dumps({"title": sid, "transcript": words, "duration": 20, "hashtags": []}))
    (g / f"{sid}_short.mp4").write_bytes(b"x")
gh = {"posts": [{"short_id": "a_1", "platform": "youtube", "posted_at": time.time() - 86400, "status": "UPLOADED", "transcript": same}]}
with contextlib.redirect_stdout(io.StringIO()) as glog:
    ids = {c["id"] for c in scheduler.candidates(g, gh, ("tiktok",))}
assert ids == {"a_1", "c_1"}, ids  # a_1 still owed to TikTok; b_1 is the same moment as the posted a_1
assert "b_1: the same moment as a_1" in glog.getvalue()
gh["posts"][0]["posted_at"] = time.time() - 31 * 86400
with contextlib.redirect_stdout(io.StringIO()):
    assert "b_1" in {c["id"] for c in scheduler.candidates(g, gh, ("tiktok",))}, "after 30 days it may go out"
# save_history keeps the clipper's "clips" entries written meanwhile (one file, two writers)
hp = tmp / "merge.json"
hp.write_text(json.dumps({"posts": [], "clips": [{"clip_id": "x", "streamer": "s", "at": time.time(), "transcript": "t"}]}))
scheduler.save_history({"posts": [{"short_id": "p1", "posted_at": 1}]}, hp)
saved = json.loads(hp.read_text())
assert saved["clips"][0]["clip_id"] == "x" and saved["posts"][0]["short_id"] == "p1", saved

print("publisher ok")

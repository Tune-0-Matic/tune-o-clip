"""YouTube Shorts publishing through the official YouTube Data API v3 (resumable videos.insert) + a one-time Google
desktop login. Stdlib only, zero cost.

  python publisher/youtube_poster.py --login     # once: opens Google in your browser, you approve, token saved

Setup: a Google Cloud project with the YouTube Data API v3 enabled, an OAuth client of type "Desktop app", and
YOUTUBE_CLIENT_ID / YOUTUBE_CLIENT_SECRET in the repo-root .env (or its JSON saved as
workspace/youtube_client_secret.json). Set the OAuth consent screen to "In production": while it is "Testing", Google
expires the refresh token after 7 days. Quota: 100 uploads per day per project, reset at midnight Pacific.
Checked against developers.google.com (Nov 2026); categoryId "20" = Gaming was not on the pages read.
"""
import argparse
import base64
import hashlib
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import twitch_api  # noqa: E402  (load_env: the shared .env reader)

import loopback  # noqa: E402

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet%2Cstatus"
SCOPE = "https://www.googleapis.com/auth/youtube.upload"
READ_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"  # channels.list(mine=true): which channel a login is
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true"
REDIRECT_PORT = 3456
REDIRECT_URI = f"http://127.0.0.1:{REDIRECT_PORT}/callback"
TOKEN_FILE = ROOT / "workspace" / "youtube_token.json"
CLIENT_FILE = ROOT / "workspace" / "youtube_client_secret.json"
CHUNK = 8 * 256 * 1024  # 2 MiB: chunks must be multiples of 256 KiB (except the last)
GAMING = "20"
EXPIRED_MSG = "[PUBLISH] YouTube OAuth token expired: click Login YouTube (1-Time) once"
QUOTA_MSG = ("[PUBLISH] YouTube API quota exceeded (100 uploads per day per project): waiting for the daily reset at "
             "midnight Pacific")


def emit(event, **data):
    print(json.dumps({"event": event, **data}), flush=True)


class YouTubeExpired(RuntimeError):
    pass


class YouTubeError(RuntimeError):
    pass


def pkce():
    """(code_verifier, code_challenge): standard PKCE, base64url(SHA-256) without padding (Google)."""
    verifier = secrets.token_urlsafe(48)
    return verifier, base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode()


def _google_error(http_error):
    """(reason, message) from a Google API error body; reason "" when there is none."""
    try:
        err = json.loads(http_error.read()).get("error") or {}
    except (ValueError, OSError, AttributeError):
        return "", str(http_error)
    if isinstance(err, str):  # the OAuth token endpoint answers {"error": "invalid_grant", ...}
        return err, err
    reasons = [e.get("reason", "") for e in err.get("errors") or [] if isinstance(e, dict)]
    return (reasons[0] if reasons else str(err.get("status") or "")), str(err.get("message") or "")


def _friendly(reason, message, code):
    if reason in ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"):
        return QUOTA_MSG
    if reason == "uploadLimitExceeded":
        return "[PUBLISH] YouTube upload limit reached for this channel (YouTube's own daily cap): try again tomorrow"
    if reason == "youtubeSignupRequired":
        return "[PUBLISH] YouTube Auth Failed: this Google account has no YouTube channel yet; create one, then Login YouTube"
    return f"[PUBLISH] YouTube upload failed (HTTP {code} {reason}): {message}".strip()


class YouTube:
    def __init__(self, client_id=None, client_secret=None, token_path=TOKEN_FILE, client_file=CLIENT_FILE,
                 opener=urllib.request.urlopen, clock=time.time, sleep=time.sleep, scope=SCOPE):
        self.scope = scope
        twitch_api.load_env()
        cid, secret = client_id or os.environ.get("YOUTUBE_CLIENT_ID"), client_secret or os.environ.get("YOUTUBE_CLIENT_SECRET")
        if not (cid and secret):  # the JSON Google Cloud Console offers for download ("installed" app)
            try:
                d = json.loads(Path(client_file).read_text(encoding="utf-8"))
                d = d.get("installed") or d.get("web") or {}
                cid, secret = cid or d.get("client_id"), secret or d.get("client_secret")
            except (OSError, ValueError, AttributeError):
                pass
        if not (cid and secret):
            raise YouTubeError("[PUBLISH] YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET must be set in .env "
                               "(Google Cloud > Credentials > OAuth client 'Desktop app')")
        self.cid, self.secret = cid, secret
        self.token_path, self.opener, self.clock, self.sleep = Path(token_path), opener, clock, sleep

    # --- login (once) ---------------------------------------------------------------------------------------------

    def auth_url(self, state, challenge):
        return AUTH_URL + "?" + urllib.parse.urlencode({
            "client_id": self.cid, "redirect_uri": REDIRECT_URI, "response_type": "code", "scope": self.scope,
            "access_type": "offline", "prompt": "consent", "code_challenge": challenge,
            "code_challenge_method": "S256", "state": state})

    def login(self, open_browser=webbrowser.open, timeout=180, port=REDIRECT_PORT):
        verifier, challenge = pkce()
        state = secrets.token_urlsafe(16)
        got = loopback.catch_redirect(self.auth_url(state, challenge), port, timeout, open_browser, self.clock)
        if "error" in got:
            raise YouTubeError(f"[PUBLISH] YouTube Auth Failed: Google said {got['error']}")
        if "code" not in got:
            raise YouTubeError(f"[PUBLISH] YouTube Auth Failed: login timed out after {timeout} s")
        if got.get("state") != state:
            raise YouTubeError("[PUBLISH] YouTube Auth Failed: the answer had the wrong state (possible forgery); try again")
        return self._token_request({"code": got["code"], "grant_type": "authorization_code",
                                    "redirect_uri": REDIRECT_URI, "code_verifier": verifier})

    # --- tokens ---------------------------------------------------------------------------------------------------

    def _token_request(self, fields):
        body = urllib.parse.urlencode({"client_id": self.cid, "client_secret": self.secret, **fields}).encode()
        req = urllib.request.Request(TOKEN_URL, body, {"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with self.opener(req, timeout=20) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            reason, msg = _google_error(e)
            if fields.get("grant_type") == "refresh_token":
                raise YouTubeExpired(f"{EXPIRED_MSG} ({reason or msg})")
            raise YouTubeError(f"[PUBLISH] YouTube Auth Failed: {reason or msg}. Check YOUTUBE_CLIENT_ID / "
                               f"YOUTUBE_CLIENT_SECRET (a 'Desktop app' OAuth client)")
        try:
            old = json.loads(self.token_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        tok = {"access_token": data["access_token"], "expires_at": self.clock() + float(data.get("expires_in", 3600)),
               # a refresh answer carries no refresh_token: keep the one we have
               "refresh_token": data.get("refresh_token") or old.get("refresh_token"), "scope": data.get("scope")}
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(json.dumps(tok, indent=2), encoding="utf-8")
        return tok

    def token(self, force_refresh=False):
        try:
            tok = json.loads(self.token_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise YouTubeExpired(EXPIRED_MSG + " (not logged in yet)")
        if not force_refresh and tok.get("expires_at", 0) - 60 > self.clock():
            return tok["access_token"]
        if not tok.get("refresh_token"):
            raise YouTubeExpired(EXPIRED_MSG)
        return self._token_request({"grant_type": "refresh_token", "refresh_token": tok["refresh_token"]})["access_token"]

    # --- resumable upload ---------------------------------------------------------------------------------------------

    def _send(self, make_req, retry_401=True):
        """Open a request built by make_req(token); one refresh-and-retry on 401 (an access token lasts ~1 h)."""
        try:
            return self.opener(make_req(self.token()), timeout=300)
        except urllib.error.HTTPError as e:
            if e.code == 401 and retry_401:
                return self.opener(make_req(self.token(force_refresh=True)), timeout=300)
            raise

    def channel(self):
        """channels.list(mine=true) -> {"id", "title"} of the channel this login posts to (needs READ_SCOPE)."""
        try:
            with self._send(lambda tok: urllib.request.Request(CHANNELS_URL, headers={"Authorization": f"Bearer {tok}"})) as r:
                items = json.loads(r.read()).get("items") or []
        except urllib.error.HTTPError as e:
            raise YouTubeError(_friendly(*_google_error(e), e.code))
        if not items:
            raise YouTubeError("[PUBLISH] YouTube Auth Failed: this Google account has no YouTube channel yet")
        return {"id": items[0]["id"], "title": (items[0].get("snippet") or {}).get("title", "")}

    def upload(self, video, title, description, tags, privacy="public", retries=5, publish_at=None, category=GAMING,
               made_for_kids=False):
        """videos.insert, resumable. -> {"video_id", "url", "privacy", "privacy_note"?, "publish_at"?}.
        publish_at (RFC 3339 UTC, e.g. "2026-11-17T14:00:00Z"): YouTube's own scheduled publishing; the video is
        uploaded private (the API requires that for publishAt) and YouTube makes it public at that time."""
        size = Path(video).stat().st_size
        if publish_at:
            privacy = "private"
        status = {"privacyStatus": privacy, "selfDeclaredMadeForKids": bool(made_for_kids), **({"publishAt": publish_at} if publish_at else {})}
        meta = json.dumps({"snippet": {"title": title, "description": description, "tags": tags, "categoryId": str(category)},
                           "status": status}).encode()
        try:
            with self._send(lambda tok: urllib.request.Request(UPLOAD_URL, meta, method="POST", headers={
                    "Authorization": f"Bearer {tok}", "Content-Type": "application/json; charset=UTF-8",
                    "X-Upload-Content-Length": str(size), "X-Upload-Content-Type": "video/*"})) as r:
                session = r.headers.get("Location")
        except urllib.error.HTTPError as e:
            raise YouTubeError(_friendly(*_google_error(e), e.code))
        if not session:
            raise YouTubeError("[PUBLISH] YouTube upload failed: no upload session (Location header) in Google's answer")
        offset, failures = 0, 0
        with open(video, "rb") as f:
            while True:
                f.seek(offset)
                data = f.read(min(CHUNK, size - offset))
                end = offset + len(data)

                def put(tok, data=data, start=offset, end=end):
                    return urllib.request.Request(session, data, method="PUT", headers={
                        "Authorization": f"Bearer {tok}", "Content-Type": "video/*", "Content-Length": str(len(data)),
                        "Content-Range": f"bytes {start}-{end - 1}/{size}"})
                try:
                    with self._send(put) as r:
                        res = json.loads(r.read() or b"{}")
                    break  # 200/201: the whole video is in
                except urllib.error.HTTPError as e:
                    if e.code == 308:  # Resume Incomplete: Range says what Google has
                        offset, failures = _have(e), 0
                        continue
                    if e.code == 404:
                        raise YouTubeError("[PUBLISH] YouTube upload session expired mid-upload: it will start over next time")
                    if e.code not in (500, 502, 503, 504):
                        raise YouTubeError(_friendly(*_google_error(e), e.code))
                    err = f"HTTP {e.code}"
                except urllib.error.URLError as e:  # connection dropped
                    err = str(e.reason)
                if failures >= retries:
                    raise YouTubeError(f"[PUBLISH] YouTube upload failed after {retries} retries: {err}")
                failures += 1
                self.sleep(min(60, 2 ** failures))  # exponential backoff, then ask Google where to resume
                offset, res = self._uploaded(session, size)
                if res is not None:
                    break
        vid = res.get("id")
        if not vid:
            raise YouTubeError("[PUBLISH] YouTube upload finished without a video id")
        got = (res.get("status") or {}).get("privacyStatus") or privacy
        out = {"video_id": vid, "url": f"https://youtube.com/shorts/{vid}", "privacy": got}
        if got != privacy:  # e.g. the channel's own settings forced private
            out["privacy_note"] = f"asked for {privacy}, YouTube made it {got}"
        if publish_at:
            out["publish_at"] = (res.get("status") or {}).get("publishAt") or publish_at
        return out

    def _uploaded(self, session, size):
        """Status check after an interruption (empty PUT, Content-Range bytes */size)
        -> (bytes Google has, the video resource if it already has them all)."""
        req = lambda tok: urllib.request.Request(session, b"", method="PUT", headers={
            "Authorization": f"Bearer {tok}", "Content-Length": "0", "Content-Range": f"bytes */{size}"})
        try:
            with self._send(req) as r:
                return size, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code == 308:
                return _have(e), None
            if e.code == 404:
                raise YouTubeError("[PUBLISH] YouTube upload session expired mid-upload: it will start over next time")
            raise YouTubeError(_friendly(*_google_error(e), e.code))


def _have(resp_308):
    """Bytes Google has stored, from a 308's "Range: bytes=0-N" (no Range = nothing yet)."""
    rng = (resp_308.headers or {}).get("Range")
    return int(rng.rsplit("-", 1)[1]) + 1 if rng else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--login", action="store_true", help="one-time Google login in your browser")
    a = ap.parse_args()
    try:
        yt = YouTube()
        if a.login:
            emit("stage", stage="publish", msg="YouTube login: approve Tune-O-Clip in the browser window (3 min)")
            yt.login()
            emit("log", msg="[PUBLISH] Logged in to YouTube (upload access saved; it renews itself)")
        else:
            yt.token()
            emit("log", msg="[PUBLISH] YouTube login OK")
        return 0
    except (YouTubeExpired, YouTubeError, OSError) as e:
        emit("error", msg=str(e) if str(e).startswith("[PUBLISH]") else f"[PUBLISH] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

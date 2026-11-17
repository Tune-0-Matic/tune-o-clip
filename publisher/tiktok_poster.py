"""TikTok publishing through TikTok's official Content Posting API (Direct Post) + a one-time desktop login.

  python publisher/tiktok_poster.py --login     # once: opens TikTok in your browser, you approve, token saved
  python publisher/tiktok_poster.py --status    # who is logged in, which privacy levels TikTok offers

No browser automation: the official API, stdlib HTTP only, zero cost. Needs your own (free) TikTok developer app
with Login Kit (Desktop) + Content Posting API (Direct Post), redirect URI http://127.0.0.1:3455/callback/, and
TIKTOK_CLIENT_KEY / TIKTOK_CLIENT_SECRET in the repo-root .env. Until TikTok audits the app, it can only post to a
PRIVATE TikTok account (TikTok's rule, error unaudited_client_can_only_post_to_private_accounts).
Endpoints and rules checked against developers.tiktok.com (Nov 2026), except TOKEN_URL (that page refused automated
reads; it is TikTok's standard v2 OAuth token endpoint).
"""
import argparse
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

import loopback  # noqa: E402  (same folder: the shared one-time browser login)

AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
API = "https://open.tiktokapis.com/v2/post/publish"
SCOPES = "user.info.basic,video.publish"
REDIRECT_PORT = 3455
REDIRECT_URI = f"http://127.0.0.1:{REDIRECT_PORT}/callback/"  # loopback + port: what TikTok allows for desktop apps
TOKEN_FILE = ROOT / "workspace" / "tiktok_token.json"
MAX_SINGLE = 64_000_000   # up to this, one chunk (TikTok: chunks 5-64 MB, files over 64 MB must be chunked)
CHUNK = 10_000_000        # chunk size above that; the last chunk absorbs the remainder (allowed up to 128 MB)
EXPIRED_MSG = "[PUBLISH] TikTok session expired: click Login TikTok (1-Time) once"


def emit(event, **data):
    print(json.dumps({"event": event, **data}), flush=True)


class SessionExpired(RuntimeError):
    pass


class TikTokError(RuntimeError):
    pass


def pkce():
    """(code_verifier, code_challenge). TikTok desktop wants the HEX SHA-256 of the verifier, not base64url."""
    verifier = secrets.token_urlsafe(48)  # 64 chars of A-Z a-z 0-9 - _
    return verifier, hashlib.sha256(verifier.encode("ascii")).hexdigest()


def plan_chunks(size):
    """(chunk_size, total_chunk_count) per TikTok's media-transfer rules."""
    if size <= MAX_SINGLE:
        return size, 1
    return CHUNK, size // CHUNK


def choose_privacy(options, wanted):
    """The creator's chosen level if TikTok offers it, else the most private one offered."""
    for level in [wanted, "SELF_ONLY", "MUTUAL_FOLLOW_FRIENDS", "FOLLOWER_OF_CREATOR"]:
        if level in options:
            return level
    if not options:
        raise TikTokError("TikTok offered no privacy level for this account")
    return options[0]


class TikTok:
    def __init__(self, client_key=None, client_secret=None, token_path=TOKEN_FILE, opener=urllib.request.urlopen,
                 clock=time.time, sleep=time.sleep):
        twitch_api.load_env()
        self.key = client_key or os.environ.get("TIKTOK_CLIENT_KEY")
        self.secret = client_secret or os.environ.get("TIKTOK_CLIENT_SECRET")
        if not (self.key and self.secret):
            raise TikTokError("TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET must be set in .env (your TikTok developer app)")
        self.token_path, self.opener, self.clock, self.sleep = Path(token_path), opener, clock, sleep

    # --- login (once) ---------------------------------------------------------------------------------------------

    def auth_url(self, state, challenge):
        return AUTH_URL + "?" + urllib.parse.urlencode({
            "client_key": self.key, "scope": SCOPES, "response_type": "code", "redirect_uri": REDIRECT_URI,
            "state": state, "code_challenge": challenge, "code_challenge_method": "S256"})

    def login(self, open_browser=webbrowser.open, timeout=180, port=REDIRECT_PORT):
        """Open TikTok's consent page, catch the redirect on the loopback port, trade the code for tokens."""
        verifier, challenge = pkce()
        state = secrets.token_urlsafe(16)
        got = loopback.catch_redirect(self.auth_url(state, challenge), port, timeout, open_browser, self.clock)
        if "error" in got:
            raise TikTokError(f"TikTok login refused: {got.get('error_description') or got['error']}")
        if "code" not in got:
            raise TikTokError(f"TikTok login timed out after {timeout} s (nothing came back to {REDIRECT_URI})")
        if got.get("state") != state:
            raise TikTokError("TikTok login answer had the wrong state (possible forgery); try again")
        return self._token_request({"code": got["code"], "grant_type": "authorization_code",
                                    "redirect_uri": REDIRECT_URI, "code_verifier": verifier})

    # --- tokens ---------------------------------------------------------------------------------------------------

    def _token_request(self, fields):
        body = urllib.parse.urlencode({"client_key": self.key, "client_secret": self.secret, **fields}).encode()
        req = urllib.request.Request(TOKEN_URL, body, {"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with self.opener(req, timeout=20) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            data = _json_or_empty(e)
        data = data.get("data", data)  # tolerate an enveloped answer
        if not data.get("access_token"):
            why = data.get("error_description") or data.get("error") or "no token in TikTok's answer"
            if fields.get("grant_type") == "refresh_token":
                raise SessionExpired(f"{EXPIRED_MSG} ({why})")
            # the first login itself failed: name TikTok's reason (wrong key/secret, redirect URI mismatch, ...)
            raise TikTokError(f"[PUBLISH] TikTok login failed: {why}. Check TIKTOK_CLIENT_KEY / TIKTOK_CLIENT_SECRET in .env "
                              f"and that {REDIRECT_URI} is registered under Login Kit > Desktop")
        now = self.clock()
        tok = {"access_token": data["access_token"], "refresh_token": data.get("refresh_token"),
               "open_id": data.get("open_id"), "scope": data.get("scope"),
               "expires_at": now + float(data.get("expires_in", 7200)),
               "refresh_expires_at": now + float(data.get("refresh_expires_in", 180 * 86400))}
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(json.dumps(tok, indent=2), encoding="utf-8")
        return tok

    def token(self):
        """A valid access token; refreshes it (2 h lifetime) with the rotating refresh token (180 days)."""
        try:
            tok = json.loads(self.token_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise SessionExpired(EXPIRED_MSG + " (not logged in yet)")
        now = self.clock()
        if tok.get("expires_at", 0) - 60 > now:
            return tok["access_token"]
        if not tok.get("refresh_token") or tok.get("refresh_expires_at", 0) <= now:
            raise SessionExpired(EXPIRED_MSG)
        return self._token_request({"grant_type": "refresh_token", "refresh_token": tok["refresh_token"]})["access_token"]

    # --- Direct Post -------------------------------------------------------------------------------------------------

    def _api(self, path, body=None):
        req = urllib.request.Request(API + path, json.dumps(body or {}).encode(), {
            "Authorization": f"Bearer {self.token()}", "Content-Type": "application/json; charset=UTF-8"})
        try:
            with self.opener(req, timeout=30) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            data = _json_or_empty(e) or {"error": {"code": f"http_{e.code}", "message": str(e)}}
        err = data.get("error") or {}
        if err.get("code", "ok") != "ok":
            if err["code"] in ("access_token_invalid", "auth_removed"):
                raise SessionExpired(EXPIRED_MSG)
            raise TikTokError(f"TikTok {path.strip('/')}: {err['code']}: {err.get('message', '')}".strip())
        return data.get("data") or {}

    def creator_info(self):
        return self._api("/creator_info/query/")

    def post(self, video, caption, privacy="PUBLIC_TO_EVERYONE", poll_every=5, timeout=900, allow_comment=True,
             allow_duet=True, allow_stitch=True):
        """Upload + publish one video. -> {"publish_id", "status", "post_id", "privacy", "creator"}.
        PUBLISH_COMPLETE = posted; a public post's id arrives only after TikTok's moderation (often ~1 min)."""
        info = self.creator_info()
        level = choose_privacy(info.get("privacy_level_options") or [], privacy)
        size = Path(video).stat().st_size
        chunk, count = plan_chunks(size)
        init = self._api("/video/init/", {
            "post_info": {"title": caption, "privacy_level": level,
                          # off when the creator turned it off in TikTok's app, or we did
                          "disable_comment": bool(info.get("comment_disabled")) or not allow_comment,
                          "disable_duet": bool(info.get("duet_disabled")) or not allow_duet,
                          "disable_stitch": bool(info.get("stitch_disabled")) or not allow_stitch},
            "source_info": {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": chunk, "total_chunk_count": count}})
        with open(video, "rb") as f:
            for i in range(count):
                start = i * chunk
                end = size if i == count - 1 else start + chunk  # the last chunk carries the remainder
                data = f.read(end - start)
                req = urllib.request.Request(init["upload_url"], data, method="PUT", headers={
                    "Content-Type": "video/mp4", "Content-Length": str(len(data)),
                    "Content-Range": f"bytes {start}-{end - 1}/{size}"})
                try:
                    with self.opener(req, timeout=300) as r:
                        r.read()
                except urllib.error.HTTPError as e:
                    raise TikTokError(f"TikTok upload chunk {i + 1}/{count} failed: HTTP {e.code}")
        deadline = self.clock() + timeout
        while True:
            st = self._api("/status/fetch/", {"publish_id": init["publish_id"]})
            status = st.get("status", "")
            if status == "FAILED":
                raise TikTokError(f"TikTok rejected the post: {st.get('fail_reason') or 'unknown reason'}")
            if status == "PUBLISH_COMPLETE" or self.clock() >= deadline:
                ids = st.get("publicaly_available_post_id") or []  # (sic) TikTok's spelling
                return {"publish_id": init["publish_id"], "status": status or "PROCESSING", "privacy": level,
                        "post_id": str(ids[0]) if ids else None, "creator": info.get("creator_username")}
            self.sleep(poll_every)


def _json_or_empty(http_error):
    try:
        return json.loads(http_error.read())
    except (ValueError, OSError):
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--login", action="store_true", help="one-time TikTok login in your browser")
    ap.add_argument("--status", action="store_true", help="check the saved login and the account's privacy options")
    a = ap.parse_args()
    try:
        tt = TikTok()
        if a.login:
            emit("stage", stage="publish", msg="TikTok login: approve Tune-O-Clip in the browser window (3 min)")
            tt.login()
            info = tt.creator_info()
            emit("log", msg=f"[PUBLISH] Logged in to TikTok as @{info.get('creator_username')} "
                            f"(privacy options: {', '.join(info.get('privacy_level_options') or [])})")
        else:
            info = tt.creator_info()
            emit("log", msg=f"[PUBLISH] TikTok login OK: @{info.get('creator_username')}, "
                            f"privacy options {', '.join(info.get('privacy_level_options') or [])}")
        return 0
    except (SessionExpired, TikTokError, OSError) as e:
        emit("error", msg=str(e) if str(e).startswith("[PUBLISH]") else f"[PUBLISH] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

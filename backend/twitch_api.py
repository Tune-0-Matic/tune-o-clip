"""Twitch Helix client (stdlib only): app access token with caching/refresh, games, users, top clips.

Credentials: TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET from the environment or the repo-root .env file
(create an app at https://dev.twitch.tv/console/apps; Client Credentials flow, no user login needed).
"""
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.twitch.tv/helix"
TOKEN_URL = "https://id.twitch.tv/oauth2/token"


def load_env(path=ROOT / ".env"):
    """KEY=VALUE lines from .env into os.environ (real environment variables win)."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():  # -sig: Notepad's BOM would hide line 1's key
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Twitch:
    def __init__(self, client_id=None, client_secret=None, token_path=ROOT / "workspace" / "twitch_token.json",
                 opener=urllib.request.urlopen, sleep=time.sleep, clock=time.time):
        if not (client_id and client_secret):
            load_env()
        self.client_id = client_id or os.environ.get("TWITCH_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("TWITCH_CLIENT_SECRET")
        if not self.client_id or not self.client_secret:
            raise RuntimeError("TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET must be set (environment or .env in the repo root)")
        self.token_path, self.opener, self.sleep, self.clock = Path(token_path), opener, sleep, clock

    def token(self, force=False):
        """Cached app access token; fetched again when missing, expiring within 60 s, or forced (after a 401)."""
        if not force and self.token_path.exists():
            cached = json.loads(self.token_path.read_text(encoding="utf-8"))
            if cached.get("client_id") == self.client_id and cached.get("expires_at", 0) > self.clock() + 60:
                return cached["access_token"]
        body = urllib.parse.urlencode({"client_id": self.client_id, "client_secret": self.client_secret,
                                       "grant_type": "client_credentials"}).encode()
        try:
            with self.opener(urllib.request.Request(TOKEN_URL, body), timeout=20) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:  # Twitch's JSON body says why (e.g. "invalid client"); never echo the secret
            reason = e.read().decode(errors="replace").strip()[:200] if e.fp else e.reason
            raise RuntimeError(f"Twitch token request failed ({e.code}): {reason}. Check TWITCH_CLIENT_ID / "
                               "TWITCH_CLIENT_SECRET in .env against https://dev.twitch.tv/console/apps") from None
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(json.dumps({"client_id": self.client_id, "access_token": data["access_token"],
                                               "expires_at": self.clock() + int(data.get("expires_in", 3600))}), encoding="utf-8")
        return data["access_token"]

    def get(self, path, params, tries=5):
        """GET a Helix endpoint -> parsed JSON. Refreshes the token once on 401; waits out 429s using
        Ratelimit-Reset (capped at 60 s, exponential when the header is missing)."""
        url = f"{API}/{path}?{urllib.parse.urlencode(params, doseq=True)}"
        refreshed = False
        for attempt in range(tries):
            req = urllib.request.Request(url, headers={"Client-Id": self.client_id,
                                                       "Authorization": f"Bearer {self.token(force=refreshed)}"})
            try:
                with self.opener(req, timeout=20) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code == 401 and not refreshed:
                    refreshed = True
                    continue
                if e.code == 429 and attempt < tries - 1:
                    reset = (e.headers or {}).get("Ratelimit-Reset")
                    self.sleep(min(60.0, max(1.0, float(reset) - self.clock())) if reset else min(60.0, 2.0 ** attempt))
                    continue
                raise
        raise RuntimeError(f"Twitch API {path}: still rate-limited after {tries} tries")

    def resolve_game_id(self, game_name="Grand Theft Auto V"):
        data = self.get("games", {"name": game_name}).get("data", [])
        return data[0]["id"] if data else None

    def user_ids(self, logins):
        """{login: broadcaster id} (Helix takes up to 100 logins per call)."""
        out = {}
        for i in range(0, len(logins), 100):
            for u in self.get("users", {"login": logins[i:i + 100]}).get("data", []):
                out[u["login"].lower()] = u["id"]
        return out

    def fetch_top_clips(self, game_id=None, broadcaster_id=None, hours_back=24, min_views=500, limit=25, language="en",
                        pages=1, skip=None, stats=None, sort="views"):
        """Most-viewed clips created in the last `hours_back` hours (None = all time) for a broadcaster and/or a
        game, filtered by view count and language (None = any), best first. Helix takes only one of broadcaster_id /
        game_id, so with both the streamer's clips are fetched and filtered to that game here.
        Follows pagination.cursor for up to `pages` pages while fewer than `limit` clips are left after the filters
        and `skip(clip)` (e.g. already processed). stats: {"returned", "filtered", "skipped"} sets of clip ids."""
        if not game_id and not broadcaster_id:
            raise ValueError("need game_id or broadcaster_id")
        params = {"first": 100}
        if hours_back is not None:
            now = datetime.fromtimestamp(self.clock(), timezone.utc)
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            params.update(started_at=(now - timedelta(hours=hours_back)).strftime(fmt), ended_at=now.strftime(fmt))
        params.update({"broadcaster_id": broadcaster_id} if broadcaster_id else {"game_id": game_id})
        stats = stats if stats is not None else {"returned": set(), "filtered": set(), "skipped": set()}
        clips = []
        for _ in range(pages):
            resp = self.get("clips", params)
            for c in resp.get("data", []):
                stats["returned"].add(c["id"])
                if (c.get("view_count", 0) < min_views or (language and c.get("language") != language)
                        or (broadcaster_id and game_id and c.get("game_id") != game_id)):
                    stats["filtered"].add(c["id"])
                elif skip and skip(c):
                    stats["skipped"].add(c["id"])
                else:
                    clips.append(c)
            cursor = (resp.get("pagination") or {}).get("cursor")
            if len(clips) >= limit or not cursor or not resp.get("data"):
                break
            params["after"] = cursor
        rank = (lambda c: -velocity(c, self.clock())) if sort == "velocity" else (lambda c: -c.get("view_count", 0))
        return sorted(clips, key=rank)[:limit]


LOGIN_RE = re.compile(r"[a-z0-9_]{1,25}")


def parse_logins(text):
    """'xqc, @Kai_Cenat\\nhttps://www.twitch.tv/caseoh_/clips' -> ['xqc', 'kai_cenat', 'caseoh_']: comma/space/newline
    separated, @ and twitch.tv URLs stripped, lowercased, invalid names dropped, duplicates removed (order kept)."""
    out = []
    for raw in re.split(r"[\s,;]+", text or ""):
        s = raw.strip().lower()
        if "twitch.tv/" in s:
            s = s.split("twitch.tv/", 1)[1]
        s = s.lstrip("@").split("/")[0].split("?")[0]
        if LOGIN_RE.fullmatch(s) and s not in out:
            out.append(s)
    return out


def clip_meta(clip):
    """The Helix fields kept with the job and written into clips_manifest.json."""
    return {"clip_id": clip["id"], "title": clip.get("title"), "broadcaster_name": clip.get("broadcaster_name"),
            "creator_name": clip.get("creator_name"), "view_count": clip.get("view_count"),
            "duration": clip.get("duration"), "game_id": clip.get("game_id"), "url": clip.get("url"),
            "created_at": clip.get("created_at"), "broadcaster_id": clip.get("broadcaster_id"),
            "video_id": clip.get("video_id") or None, "vod_offset": clip.get("vod_offset")}  # VOD + second: same-moment check


def velocity(clip, now):
    """Views per minute since the clip was made (at least 15 min, so a 2-minute-old clip can't win on noise):
    finds breakout moments that are still climbing. A clip without created_at falls back to its plain views."""
    try:
        made = datetime.strptime(clip["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except (KeyError, TypeError, ValueError):
        return float(clip.get("view_count", 0))
    return clip.get("view_count", 0) / max((now - made) / 60, 15)

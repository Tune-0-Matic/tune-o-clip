"""Pick the best unposted short in ready_to_post/, format it per platform (TikTok caption / YouTube Shorts title,
description and tags, each with exactly 5 hashtags), respect each platform's own posting interval, post it through
the official APIs (tiktok_poster.py, youtube_poster.py), record it and move it to posted/ once every enabled
platform has it.

  python publisher/scheduler.py --dry-run --platform both   # what would be posted next, and why; touches nothing
  python publisher/scheduler.py --platform youtube          # post the best short if YouTube's 3.5 h have passed
  python publisher/scheduler.py --force                     # post the best short now, interval or not
stdout is JSON Lines like the rest of Tune-O-Clip (log / publish / error).
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
READY, POSTED = ROOT / "ready_to_post", ROOT / "posted"
HISTORY = ROOT / "workspace" / "posted_history.json"
INTERVAL_H = 3.5  # a small account posting every 3.5 h stays clear of spam throttling
TITLE_MAX, HOOK_MAX, CAPTION_MAX = 55, 90, 2200
YT_TITLE_MAX = 100
# PROCESSING = uploaded, TikTok still finishing; UPLOADED = YouTube has it: all count as posted
DONE = ("PUBLISH_COMPLETE", "PROCESSING", "UPLOADED")
PLATFORMS = {"tiktok": ["tiktok"], "youtube": ["youtube"], "both": ["youtube", "tiktok"]}
LABEL = {"tiktok": "TikTok", "youtube": "YouTube Shorts"}


def emit(event, **data):
    print(json.dumps({"event": event, **data}), flush=True)


# --- history + pacing -------------------------------------------------------------------------------------------------

def load_history(path=HISTORY):
    try:
        h = json.loads(Path(path).read_text(encoding="utf-8"))
        return h if isinstance(h, dict) and isinstance(h.get("posts"), list) else {"posts": []}
    except (OSError, ValueError):
        return {"posts": []}


def save_history(history, path=HISTORY):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(history, indent=2), encoding="utf-8")


def _done(history, platform):
    """Posts that count on `platform` (entries from before YouTube support have no platform: TikTok)."""
    return [p for p in history["posts"] if p.get("status") in DONE and p.get("platform", "tiktok") == platform]


def wait_left(history, interval_h, now, platform="tiktok"):
    """Seconds until `platform` may post again (0 = now); each platform keeps its own interval."""
    times = [p["posted_at"] for p in _done(history, platform)]
    return max(0.0, max(times) + interval_h * 3600 - now) if times else 0.0


# --- picking ----------------------------------------------------------------------------------------------------------

def _clip_score(post):
    """The clip's score from its manifest (0..1); 0.5 when the manifest is gone."""
    src = post.get("source") or {}
    m = Path(str(src.get("manifest") or ""))
    m = m if m.is_absolute() else ROOT / m
    try:
        clips = json.loads(m.read_text(encoding="utf-8")).get("clips") or []
    except (OSError, ValueError):
        return 0.5
    return next((float(c.get("score") or 0.5) for c in clips if str(c.get("clip_id")) == str(src.get("clip_id"))), 0.5)


def quality(post):
    """80% the clip's own score + 10% loudness health (near -14 dB) + 10% length health (12-50 s)."""
    qc = post.get("qc") or {}
    loud = 1 - min(1.0, abs(float(qc.get("mean_db", -14)) + 14) / 6)
    dur = float(post.get("duration") or qc.get("duration") or 0)
    return round(0.8 * _clip_score(post) + 0.1 * loud + 0.1 * (1.0 if 12 <= dur <= 50 else 0.5), 4)


def candidates(ready=READY, history=None, platforms=("tiktok",)):
    """Shorts in ready/ still missing on some enabled platform, best first; each with "todo" = those platforms."""
    history = history or {"posts": []}
    done = {pf: {p["short_id"] for p in _done(history, pf)} for pf in platforms}
    out = []
    for pj in sorted(Path(ready).glob("*_post.json")):
        sid = pj.name[:-len("_post.json")]
        short = pj.with_name(sid + "_short.mp4")
        todo = [pf for pf in platforms if sid not in done[pf]]
        if not todo or not short.exists():
            continue
        try:
            post = json.loads(pj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"id": sid, "short": short, "post_json": pj, "post": post, "quality": quality(post), "todo": todo,
                    "mtime": short.stat().st_mtime})
    return sorted(out, key=lambda e: (-e["quality"], -e["mtime"]))


# --- formatting -------------------------------------------------------------------------------------------------------

def _tag(text):
    t = re.sub(r"[^a-z0-9_]", "", str(text).lower().lstrip("#"))
    return "#" + t if t else ""


def _creator(post):
    """(handle, platform) to credit: the Twitch streamer, else the YouTube channel from the manifest."""
    credit = post.get("credit") or {}
    if credit.get("broadcaster_name"):
        return str(credit["broadcaster_name"]), "Twitch"
    m = Path(str((post.get("source") or {}).get("manifest") or ""))
    try:
        ch = json.loads((m if m.is_absolute() else ROOT / m).read_text(encoding="utf-8")).get("channel")
    except (OSError, ValueError):
        ch = None
    return (str(ch).lstrip("@"), "YouTube") if ch else (None, None)


def hashtags(post):
    """Exactly 5: #fyp #gaming #twitchclips (#clips off Twitch) #<streamer> #<game or joke>, lowercase, no repeats;
    padded with the clip's own tags, then #funny / #viral / #clips."""
    who, platform = _creator(post)
    own = [_tag(t) for t in post.get("hashtags") or []]
    own = [t for t in own if t and t not in ("#shorts", "#youtubeshorts", "#fyp")]
    wanted = ["#fyp", "#gaming", "#twitchclips" if platform == "Twitch" else "#clips", _tag(who) if who else ""]
    out = []
    for t in wanted + own + ["#funny", "#viral", "#clips", "#streamer"]:
        if t and t not in out:
            out.append(t)
    return out[:5]


def _clean(text, limit):
    t = re.sub(r"\s+", " ", re.sub(r"[\"“”<>]", "", str(text or ""))).strip()  # YouTube rejects < and >
    if len(t) <= limit:
        return t
    cut = t[:limit - 1].rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:-") + "…"


def _name(post, limit):
    return _clean(re.sub(r"\s#\S+", "", post.get("title") or post.get("social_title") or ""), limit)


def _hook(post):
    """The transcript's first sentence: the line that sells the clip."""
    return _clean(re.split(r"(?<=[.!?])\s", _clean(post.get("transcript"), 400), maxsplit=1)[0], HOOK_MAX)


def caption(post):
    """(title, caption). Caption = Title — "hook quote" (via @streamer on Twitch) + the 5 hashtags on line 2."""
    title, hook = _name(post, TITLE_MAX), _hook(post)
    who, platform = _creator(post)
    line = title + (f' — "{hook}"' if hook else "") + (f" (via @{who} on {platform})" if who else "")
    return title, (line + "\n" + " ".join(hashtags(post)))[:CAPTION_MAX]


def youtube_meta(post):
    """(title, description, tags) for YouTube Shorts: '<Video Name> #Shorts' (<= 100 chars); name, hook + credit and
    the same 5 hashtags as TikTok; tags = those 5 without '#' + shorts, gaming."""
    name = _name(post, YT_TITLE_MAX - len(" #Shorts")) or _clean(_hook(post), YT_TITLE_MAX - 8) or "Funny gaming moment"
    who, platform = _creator(post)
    five = hashtags(post)
    said = _hook(post) + (f" (Credit: @{who} on {platform})" if who else "")
    desc = "\n\n".join(x for x in (name, said.strip(), " ".join(five)) if x)
    return name + " #Shorts", desc, list(dict.fromkeys([t.lstrip("#") for t in five] + ["shorts", "gaming"]))


# --- one run ------------------------------------------------------------------------------------------------------------

def _move(sid, ready, posted):
    Path(posted).mkdir(parents=True, exist_ok=True)
    for name in (sid + "_short.mp4", sid + "_post.json"):
        if (Path(ready) / name).exists():
            shutil.move(str(Path(ready) / name), Path(posted) / name)


def _poster(platform, posters):
    if platform not in posters:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        if platform == "youtube":
            from youtube_poster import YouTube
            posters[platform] = YouTube()
        else:
            from tiktok_poster import TikTok
            posters[platform] = TikTok()
    return posters[platform]


def run(dry_run=False, force=False, interval_h=INTERVAL_H, platforms=("youtube",), tiktok_privacy="PUBLIC_TO_EVERYONE",
        youtube_privacy="public", posters=None, ready=READY, posted=POSTED, history_path=HISTORY, now=None, short_id=None):
    """-> exit code (1 if a platform failed; the others still post). dry_run: pick + format + pacing check only.
    short_id: post exactly that short now (the panel's Post to YouTube / TikTok buttons), from ready_to_post/ or
    posted/; it moves nothing (the next normal run tidies ready_to_post/ by the configured platforms)."""
    now = time.time() if now is None else now
    posters = {} if posters is None else posters
    history = load_history(history_path)
    if not dry_run and not short_id:  # shorts every enabled platform already has (e.g. after switching Both -> one) go to posted/
        have = [{p["short_id"] for p in _done(history, pf)} for pf in platforms]
        for pj in Path(ready).glob("*_post.json"):
            sid = pj.name[:-len("_post.json")]
            if all(sid in h for h in have):
                _move(sid, ready, posted)
    pool = candidates(ready, history, platforms)
    if short_id:
        force = True
        pool = [c for c in pool + candidates(posted, history, platforms) if c["id"] == short_id][:1]
        if not pool:
            emit("log", msg=f"[PUBLISH] {short_id}: already on {' + '.join(LABEL[pf] for pf in platforms)}, or its short is gone")
            return 0
    if not pool:
        emit("log", msg="[PUBLISH] Nothing to post: no unposted short in ready_to_post/")
        return 0
    rc = 0
    for pf in platforms:
        mine = [c for c in pool if pf in c["todo"]]
        if not mine:
            emit("log", msg=f"[PUBLISH] {LABEL[pf]}: every short is already posted there")
            continue
        best, left = mine[0], wait_left(history, interval_h, now, pf)
        if pf == "youtube":
            title, text, tags = youtube_meta(best["post"])
        else:
            (title, text), tags = caption(best["post"]), hashtags(best["post"])
        emit("log", msg=f"[PUBLISH] {LABEL[pf]}: best of {len(mine)}: {best['id']} (quality {best['quality']:.2f}): {title}")
        if dry_run:
            emit("publish", dry_run=True, platform=pf, short_id=best["id"], title=title, caption=text, hashtags=tags,
                 quality=best["quality"], due_in_s=round(0 if force else left))
            continue
        if left > 0 and not force:
            emit("log", msg=f"[PUBLISH] {LABEL[pf]}: next post allowed in {left / 60:.0f} min (every {interval_h:g} h)")
            continue
        emit("stage", stage="publish", msg=f"Posting {best['id']} to {LABEL[pf]}")
        try:
            poster = _poster(pf, posters)
            if pf == "youtube":
                res = {**poster.upload(best["short"], title, text, tags, youtube_privacy), "status": "UPLOADED"}
            else:
                res = poster.post(best["short"], text, tiktok_privacy)
        except (RuntimeError, OSError) as e:  # TikTok/YouTube errors are RuntimeErrors; one platform failing spares the other
            emit("error", msg=str(e) if str(e).startswith("[PUBLISH]") else f"[PUBLISH] {LABEL[pf]}: {e}")
            rc = 1
            continue
        history["posts"].append({"short_id": best["id"], "platform": pf, "posted_at": now, "title": title, **res})
        save_history(history, history_path)
        best["todo"].remove(pf)
        if not best["todo"] and not short_id:
            _move(best["id"], ready, posted)
        emit("publish", platform=pf, short_id=best["id"], title=title, **res)
        if pf == "youtube":
            emit("log", msg=f"[PUBLISH] Posted to YouTube Shorts: {res['url']}"
                            + (f" ({res['privacy_note']})" if res.get("privacy_note") else ""))
        else:
            emit("log", msg=f"[PUBLISH] Posted {best['id']} to TikTok ({res.get('status')}, {res.get('privacy')})")
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="pick and format only; no upload, no files moved")
    ap.add_argument("--force", action="store_true", help="post now even if the interval hasn't passed")
    ap.add_argument("--platform", default="youtube", choices=list(PLATFORMS), help="where to post (default youtube)")
    ap.add_argument("--interval", type=float, default=INTERVAL_H, help="hours between posts, per platform (default 3.5)")
    ap.add_argument("--youtube-privacy", default="public", choices=["public", "unlisted", "private"])
    ap.add_argument("--tiktok-privacy", default="PUBLIC_TO_EVERYONE",
                    choices=["PUBLIC_TO_EVERYONE", "FOLLOWER_OF_CREATOR", "MUTUAL_FOLLOW_FRIENDS", "SELF_ONLY"])
    ap.add_argument("--short", metavar="ID", help="post this short (e.g. twitch_123_1) now instead of the best one")
    a = ap.parse_args()
    try:
        return run(a.dry_run, a.force, a.interval, PLATFORMS[a.platform], a.tiktok_privacy, a.youtube_privacy,
                   short_id=a.short)
    except OSError as e:
        emit("error", msg=f"[PUBLISH] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

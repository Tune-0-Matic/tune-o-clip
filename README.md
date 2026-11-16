# Tune-O-Clip

YouTube URL → top-N funny 30 s clips, cut with FFmpeg stream copy (no re-encode).
`backend/` is a standalone Python CLI; `godot_app/` is only a GUI that runs it and reviews the results.

## Setup

```sh
python -m venv .venv                      # Windows: .venv\Scripts\python   Linux: .venv/bin/python
.venv/Scripts/python -m pip install -r backend/requirements.txt
```
`requirements.txt` also installs Deno + `yt-dlp-ejs` into the venv: YouTube's "n challenge" needs a JS runtime,
otherwise logged-in downloads fail with "Only images are available". The backend points yt-dlp at the venv's Deno.

FFmpeg must be on PATH — Windows: `winget install Gyan.FFmpeg`, Linux Mint: `sudo apt install ffmpeg`.
yt-dlp runs as `python -m yt_dlp` from the venv (installed by requirements), so it doesn't need to be on PATH.
The Godot panel uses `.venv` automatically when it exists, otherwise `python` (Windows) / `python3` (Linux).
Open the panel with `Launch Tune-O-Clip.bat` (or the Desktop shortcut). On start it logs a `[BOOT]` preflight
(.venv, ffmpeg on PATH, Twitch keys in `.env`, saved streamers), starts `ollama serve` with
`OLLAMA_LLM_LIBRARY=vulkan` if Ollama is off (and stops it on exit only if it started it), and preloads the judge
model (`keep_alive` 30m). Headless runs (tests) skip starting services.

**Whisper on the GPU (Windows dev box only):** `.venv/Scripts/python -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"`.
The backend adds their DLL folders itself; no PATH editing. Without them (or on the Linux box, which has no NVIDIA GPU)
Whisper falls back to CPU int8 automatically. The GTX 1050 Ti has no float16 support in CTranslate2, so the
compute type is `auto` (int8 / float32), not float16.

**Ollama (optional):** `ollama pull richardyoung/qwen2.5-3b-instruct-abliterated`. On GPUs whose driver is older than
Ollama's CUDA build (error `the provided PTX was compiled with an unsupported toolchain`), set the user env var
`OLLAMA_LLM_LIBRARY=vulkan` and restart the Ollama tray app: the Vulkan backend runs fully on the GPU (~2 s per rating
on the GTX 1050 Ti vs ~15 s on CPU).

## How it picks clips

1. **Metadata pass:** info JSON + YouTube's "Most Replayed" heatmap. Captions are never requested: YouTube's
   caption endpoint 429s easily and the block hits the whole IP. All yt-dlp calls pace themselves
   (`--sleep-requests 2 --retries 5 --fragment-retries 5`) and solve YouTube's JS challenge with the venv's Deno
   (or Node on PATH).
2. **Download** the video once (≤1080p, H.264 preferred). In parallel, a tiny 144p H.264 video-only stream is
   fetched for the visual scan (decoding the 1080p source at 2 fps takes ~2 min; 144p takes ~2 s).
3. **Per-second curves.**
   - Replay: heatmap *prominence* (how far each segment spikes above the median of the surrounding ~80 s; the
     first 5% is zeroed because everyone watches the intro). No heatmap (or `--ignore-heatmap`): audio RMS energy.
   - Vocal excitement (laughing, wheezing, yelling): from the decoded 16 kHz audio, 300–3400 Hz voice-band
     energy bursts multiplied by rapid frame-to-frame flutter, scaled by how voice-heavy the sound is, so engines,
     bass and sustained music score ~0. It can't tell a laugh from loud excited speech; both count.
   - Visual action (`fps=2,scale=160:90,format=gray` → numpy): motion + brightness flashes as prominence over the
     local ~40 s median. Only a 15% tie-breaker, so menus and driving don't outrank dialogue.
4. **Wide candidate pool.** `max(3×N, 12)` non-overlapping windows from the ranking curve
   (35 signal : 15 visual). Each grows while its spike stays above 35% of the peak, +5 s setup and +3 s reaction,
   clamped to `--min-length`..`--max-length` (a clamped window keeps 2/3 before the peak: viewers rewind *after*
   a punchline).
5. **Transcripts.** ONE batched faster-whisper (`small.en`, GPU) pass over all candidate windows (±3 s padding).
   Start/end snap to the nearest spoken-line boundary within ±3 s. Whisper gets a gaming-vocabulary prompt and
   misheard names are fixed (`scorer.GAMING_FIXES`; scoped ones like `queso → CaseOh` only fire when the streamer /
   title mentions CaseOh). Add your own in `workspace/custom_vocab.json`:
   `{"prompt_terms": ["Ohnepixel"], "fixes": {"oh nay pixel": "Ohnepixel"}}` (your fixes win; a broken file is ignored).
6. **Comedy judge + final pick.** If Ollama is running, each candidate is judged **in its own** `format: "json"`
   call on four 0–10 integer pillars (comedy_payoff, self_contained, hook_strength, quotability; weights 35/35/20/10)
   with a one-sentence `judge_reason`. Measured live: in one ~12-candidate call the 3B model copied one score across
   pillars and clips; alone it scores individually (~10–13 s per candidate on the GTX 1050 Ti). One extra small call
   over all candidates flags story callbacks. Hard reject: comedy_payoff < 5 **and** hook_strength < 5 (dead air);
   `--min-score` gates the weighted score; Twitch clips with ≥3000 views and a vocal burst get +0.5.
   Final score = **50% comedy** (judge, nudged by laugh/reaction/exclamation cues) + **35% signal** (replay averaged
   with vocal bursts) + **15% visual**. Only then are the best N non-overlapping candidates kept; when none passes,
   a `warning` event says why (`[JUDGE] Rejected all N candidate(s) ...`).
7. **Story callbacks.** Pairs the Ollama judge flags (an earlier candidate setting up a later one ≥45 s on) are
   confirmed against the setup candidate's Whisper lines (a shared rare word, or a callback phrase like
   "remember that guy" / "he's back"), then `[setup] + [payoff]` are cut on keyframes and joined with the concat
   demuxer (no re-encode); the setup runs 6–10 s and the payoff's pre-roll is trimmed to stay ≤ max length.
8. **Cut** each window with `-c copy`. Stream copy starts on the keyframe before the window, so a clip can start
   up to one GOP (2–7 s on some videos) early.

`reason` reads like `Most Replayed 3.4x spike @ 14:22 | Peak 0.96 + vocal burst 0.87 + 2 laugh + LLM 8/10`, or for a
stitched clip `Story Callback: [1:12-1:20] -> [14:05-14:32] (dinosaur, purple) | ...`.

Output: `output/<video id>/clip_NN_<start>s.mp4` + `clips_manifest.json` (every run, single or queued; the download
and other temp files stay in `workspace/<video id>/`, which only holds temp files, `queue.db` and config JSONs)
(`clip_id, file, start_sec, end_sec, duration, setup_start_sec, setup_end_sec, score, reason, transcript`;
the setup fields are `null` unless the clip is a stitched callback).
Re-running the same video deletes its old `clip_*.mp4` files (after the new download succeeds), so each folder holds one batch.

## CLI

```
python backend/clipper.py URL [options]
  --clips N             number of clips (default 4)
  --min-length SEC      shortest clip (default 15)
  --max-length SEC      longest clip, incl. a stitched setup (default 55)
  --output DIR          finished clips + manifest go to DIR/<id>/ (default ./output)
  --workspace DIR       temp dir for downloads (default ./workspace)
  --cleanup-source      delete the full source video after cutting (use on the 64 GB SSD)
  --whisper-model NAME  small.en (default; use base.en on the CPU-only Linux box for ~3x speed)
  --device auto|cuda|cpu
  --ollama-model NAME   default richardyoung/qwen2.5-3b-instruct-abliterated; only used if Ollama answers
  --no-ollama           heuristic scoring only
  --ignore-heatmap      force the audio-energy pass
  --no-visual           skip the 144p visual action scan
  --no-callbacks        never stitch [setup] + [payoff] story callbacks
  --cookies-from-browser BROWSER  send edge/brave/firefox/chrome's YouTube login (fixes "Sign in to confirm
                        you're not a bot"; close that browser first, Chromium locks its cookie file while open)
  --cookies FILE        Netscape cookies.txt instead (export it with a browser extension; works with the browser open).
                        A cookies.txt in the project folder is used automatically. Keep it private: it is a live login.
```
stdout is JSON Lines: `{"event": "stage"|"log"|"clip"|"done"|"error", ...}`; exit code 0 = ok, 1 = failed, 2 = missing tools.

## Queue worker (batch / 24-7)

`backend/queue_worker.py` keeps a persistent SQLite queue in `workspace/queue.db` and runs `clipper.py` on one
video at a time. Options it doesn't know (`--clips`, `--max-length`, `--cookies`, ...) pass through to the clipper.

```
python backend/queue_worker.py --add-url URL            # video, playlist or channel (repeatable)
python backend/queue_worker.py --add-file urls.txt      # one URL per line, # comments ok
python backend/queue_worker.py --poll-rss feeds.txt     # https://www.youtube.com/feeds/videos.xml?channel_id=...
python backend/queue_worker.py --drain                  # process until empty, then exit
python backend/queue_worker.py --daemon --poll-rss feeds.txt --cooldown 20-45   # forever
python backend/queue_worker.py --status
python backend/queue_worker.py --daemon --auto-edit --prune-raw-days 7 ...   # 64 GB box: drop old raw clips
```
- **`--prune-raw-days N`** (default 0 = off): deletes raw `.mp4`s in `output/<id>/` older than N days whose
  `<video id>_<clip id>_short.mp4` is in `ready_to_post/` (shorts only land there after passing QC). Manifests stay.
  Runs at start-up and every daemon poll.
- **Never twice:** `seen_videos` remembers every finished video id; duplicates and re-adds are skipped.
- **Cooldown:** `--cooldown` (20-45 s) only follows YouTube jobs; Twitch clip jobs wait `--twitch-cooldown` (1.5 s),
  since they come from the official API, not scraping.
- **Disk:** every job runs with `--cleanup-source`; finished clips + `clips_manifest.json` move to
  `output/<video id>/` and the job's workspace folder (source, info JSON, analysis stream, partials) is deleted,
  also on failure.
- **Errors** retry up to 3 times, then the job is `failed`. A worker killed mid-job resumes that job next start.
- **YouTube blocks** (bot check / 403 / 429): the job goes back to `pending` without using an attempt, an alert
  is POSTed to `DISCORD_WEBHOOK_URL` if set, and the worker stops (`--drain`) or backs off `--block-backoff`
  seconds (default 3600, `--daemon`). Refresh `cookies.txt` if it keeps happening.
- In the Godot panel, source mode **Batch URLs** shows a paste box: one link per line → **START** drains them live.
- Every tunable lives in the panel's **SETTINGS** drawer and auto-saves to `workspace/settings.json`. The clipper
  gets them as flags; `editor/pipeline.py` reads its defaults (facecam box, caption theme, music gain, models)
  from that file, so queue/cron runs use them too (CLI flags win). `--min-score N` (banger gate) skips clips the
  Ollama judge rates below N/10.

### Twitch clips (official Helix API, no cookies)

Create an app at https://dev.twitch.tv/console/apps (category: any, client type **Confidential**), then put
`TWITCH_CLIENT_ID=...` and `TWITCH_CLIENT_SECRET=...` in `.env` in the repo root (gitignored). The app token is
fetched with the Client Credentials flow, cached in `workspace/twitch_token.json` and refreshed when it expires
or on a 401; 429s wait for `Ratelimit-Reset`.

```
python backend/queue_worker.py --poll-twitch-game "Grand Theft Auto V" --drain
python backend/queue_worker.py --poll-twitch-channels workspace/twitch_channels.txt --drain        # streamers, any game
python backend/queue_worker.py --poll-twitch-channels workspace/twitch_channels.txt --twitch-game "Grand Theft Auto V" --drain
python backend/queue_worker.py --daemon --poll-rss feeds.txt --poll-twitch-game "Grand Theft Auto V" --poll-every 900
```
`--twitch-limit` (20), `--twitch-per-streamer` (0 = even split), `--twitch-min-views` (500), `--twitch-hours` (24),
`--twitch-lang` (`en`, `""` = any), `--retry-rounds` (10: with `--drain`, while no clip of the run has passed the
AI judge, poll the next unseen clips and drain again; stops once one passes or Twitch has nothing new; 0 = off). Inside a time window clips are ranked by **view velocity** (views per minute since
creation, counting at least 15 min) so breakouts come first; the all-time window ranks by views. **Same moment, once:**
a clip from the same VOD within ±30 s (`video_id` + `vod_offset`) of one already queued/processed, or of a better one
in the same poll, is skipped; clips without a VOD are caught after Whisper instead (same streamer, last 24 h,
>60% of the transcript words shared → `duplicate_of` in the manifest, never judged or edited).
Clip ids are deduplicated forever like YouTube ids (`twitch_<clip id>`). Several streamers share the batch
round-robin (`ceil(limit / streamers)` each, interleaved), and a streamer whose newest clips are all done is
re-polled over 72 h, then 7 days, so repeated polls keep finding new clips. Twitch clips are already viewer-cut
highlights, so `clipper.py` skips windowing: it downloads the clip, Whisper-transcribes and Ollama-judges it whole,
and archives it to `output/twitch_<clip id>/` with the streamer credit (`broadcaster_name`, `view_count`, ...)
under `"twitch"` in `clips_manifest.json`. Channel files take logins comma/newline separated (`@name` and
`twitch.tv/name` links are normalised). In the panel: SETTINGS → TWITCH POLLING streamers (synced to
`workspace/twitch_channels.txt`, so a `--daemon` can use the same list) + optional game filter, then source mode
**Twitch Streamers** → **START**;
with no streamers it polls the game across all of Twitch. If nothing new clears `--twitch-min-views`, the poll
retries at 500 and then 100 views before giving up.

## Shorts editor (`editor/`)

Turns clips into finished 1080x1920 shorts in `ready_to_post/` (`<video id>_<clip id>_short.mp4` + `_post.json`).
```
python editor/pipeline.py --manifest output/<id>/clips_manifest.json [--clip 2] [--facecam-box 0,0.13,0.21,0.28]
python backend/queue_worker.py ... --drain --auto-edit      # every finished video flows straight through
python backend/clipper.py URL --auto-edit                    # single runs too
```
1. **Words:** faster-whisper `word_timestamps` on the clip *file*, cached as `words` in `clips_manifest.json`.
2. **Framing** (`framing.py`, every clip, 2 fps samples): faces from OpenCV YuNet (`editor/models/`, 230 KB) decide
   the layout, and it stays static (no crop tracking, no zooms). One small, steady face near a frame edge with no
   other comparable face = a verified webcam over gameplay -> `split_stack`, top panel centred on that face with
   headroom. Anything else (no webcam, a big IRL face, several people) -> `blur_fill`. A PP-OCRv3 text detector
   (2.4 MB) spots captions already burned into the stream (a wide, centred, changing line low in the frame while
   someone talks): then ours are off and the layout is `blur_fill` so theirs show once, whole.
   `--facecam-box x,y,w,h` (0..1) / SETTINGS forces `split_stack` with that box.
3. **Director** (`director.py`): local Ollama writes the title + 4 hashtags (rule-based fallback when offline); the
   scenario (rule-based), cuts, speed ramp and cold-open teaser are placed deterministically. Pacing: words + Silero
   VAD speech (bundled with faster-whisper) + loud reactions are kept, every silence > 0.35 s is cut (0.25 s in
   `unfiltered_chaos`), 0.1 s pad. `edl.json` times are source seconds. A 1.2-1.5 s cold open of a loud second-half
   climax on clips >= 16 s (never for horror); `confident_fail` plays its setup at 1.15x; horror cuts 0.4 s after
   the scare. Retention effects (`retention_fx`): facecam snap zooms 1.35x climax / 1.3x yells / 1.25x "!" "?" lines
   (>= 2 s apart, held <= 1.2 s) each with a -16 dB whoosh; a 0.2 s +-6 px facecam shake + -14 dB thud on every yell
   (>= 85% of the peak and 2.5x the median); the punchline (last kept "!" word, else the word at the climax) pitched
   down 0.86 (`asetrate` + `atempo`, so the length and sync stay). Zooms / shake only on split_stack's webcam panel.
4. **Renderer** (`renderer.py`):
   - ffmpeg pass: trims + concat, `split_stack` (facecam 1080x690 on top, 4 px seam, gameplay 1080x1226) or
     `blur_fill` (sharp 16:9 over a quarter-res blurred fill), the facecam effects; audio = original + speed ramps
     (`atempo`) + punchline pitch + whoosh / thud (procedural, `assets/sfx/`, your own file wins), music from
     `assets/music/` ducked under speech, `loudnorm=I=-14:TP=-1.5:LRA=11`. Encoder: h264_nvenc / h264_vaapi when a
     probe encode works, else libx264 superfast crf 20 (NVENC: p3, cq 22). Always 30 fps.
   - Captions + sound cues: [pycaps](https://github.com/francozanardi/pycaps) (MIT, `editor/captions.py`, its own
     process; style in `editor/pycaps_style/`): 1-3 word uppercase Komika Axis lines (~66 px), centred at 46% of
     the height on split_stack (just under the webcam seam) / 72% on blur_fill; the active word zooms in and
     turns yellow, hype words (regex: "no", "what", "bro", "let's go", "gta", "cops"...) green / red with a soft pop
     cue. Word times = Whisper on each render's own cut audio (`renderer.cut_audio`), so captions match the cuts;
     none when the source has its own captions. Browser-free renderer (html2pic), no AI service. Its encode is
     libx264 veryfast crf 21. SETTINGS caption theme = one active-word colour.
   - `emotional_story` profile (the judge's own short tone call, `scorer.ollama_tone`: any strong feeling, sad,
     nostalgic, wholesome, grateful, proud, inspiring..., strength >= 8; such clips pass Min Comedy Score on that
     alone), matched to a measured reference Short: no snap zooms / shakes / pitch drop / whoosh / thud / speed-up /
     teaser; the same tight VAD hard cuts; a muted 8 x 0.45 s flash montage of the strong moments before the final
     line (stories >= 20 s with >= 2 strong moments); a punchy grade (`eq=contrast=1.10:saturation=1.18`, teal
     shadows, light vignette) + a slow 1.0 -> 1.07x push-in; a generated ambient piano / pad loop
     (`assets/music/sad_ambient.wav` / `warm_ambient.wav`, your own file of that name wins) at -19 dBFS in the speech
     gaps (mixed after the voice's loudnorm, ducked under speech); bold 2-line uppercase captions at 52% whose second
     line is italic with a glowing accent (red sad / yellow warm / cyan otherwise; Archivo Black + Black Italic
     bundled in `editor/pycaps_style_story/`, SIL OFL). The panel shows [HYPE] / [STORY] on every clip card. The
     hype_chaos edit is unchanged.
5. **QC** (`qc.py`): > 500 KB, 1080x1920, 7.0-58.5 s, mean volume -20..-10 dBFS, no black > 0.3 s. On failure the
   EDL is repaired (black spans stripped, teaser dropped / front trimmed when too long) and re-rendered, up to 2 times.

In the panel: **Auto-Edit to Shorts** (SETTINGS, on by default), and **Preview Raw** / **PLAY EDITED SHORT** / **MAKE SHORT** on every clip.
Previews play in a built-in player (ffmpeg raw frames over a pipe + a temp `workspace/_preview_audio.wav` as the
clock; Space pause, Left/Right seek 3 s, Esc close, Delete Reject trashes the clip and its short after a 5 s UNDO window; Shift+click = at once). **Pop Out (ffplay)**
opens the same file in ffplay, which is also the fallback when the pipe can't start.
The CLIPS list shows every clip in `output/*/` and `workspace/*/` (newest first; filter ALL CLIPS / EDITED SHORTS
ONLY / LAST RUN) with **Open Ready-to-Post Folder** in its header.

### Headless on Linux Mint (cron)

```cron
# every 30 min: pick up new uploads from the feeds and drain the queue (flock stops overlapping runs)
*/30 * * * * cd /home/USER/tune-o-clip && flock -n workspace/.lock .venv/bin/python backend/queue_worker.py --poll-rss feeds.txt --drain --device cpu --whisper-model base.en >> worker.log 2>&1
```
Or run `--daemon` under a systemd user service instead of cron.

The panel saves its learned step times for the ETA countdowns to `workspace/eta_stats.json` at the end of every run.

## Publisher: YouTube Shorts + TikTok (`publisher/`, official APIs)

**YouTube Shorts** (`youtube_poster.py`, YouTube Data API v3, stdlib only): a Google Cloud project with the YouTube
Data API v3 enabled and an OAuth client of type *Desktop app*; `YOUTUBE_CLIENT_ID` / `YOUTUBE_CLIENT_SECRET` in
`.env` (or the downloaded JSON as `workspace/youtube_client_secret.json`); set the OAuth consent screen to
**In production** (while *Testing*, Google expires the refresh token after 7 days). Then
`python publisher/youtube_poster.py --login` once (PKCE S256, `access_type=offline`, loopback
`http://127.0.0.1:3456/callback`; token in `workspace/youtube_token.json`, refreshed on expiry and on a 401).
Upload = resumable `videos.insert` in 2 MiB chunks (multiples of 256 KiB), resuming from the `Range` of each 308 and
after 5xx/dropped connections (backoff + status check). Title `<name> #Shorts` (≤100), description
`name / hook (Credit: @streamer on Twitch) / 5 hashtags`, tags = the 5 hashtags + `shorts`, `gaming`, category 20,
`selfDeclaredMadeForKids=false`, privacy public|unlisted|private. Quota: 100 uploads/day per project (midnight PT).

**TikTok** (`tiktok_poster.py`):

Posts the best unposted short through TikTok's **official** Direct Post API (no browser automation, stdlib HTTP,
no cost). Setup: a free app at developers.tiktok.com with Login Kit (Desktop) + Content Posting API (Direct Post),
scopes `user.info.basic,video.publish`, redirect URI `http://127.0.0.1:3455/callback/`, and
`TIKTOK_CLIENT_KEY` / `TIKTOK_CLIENT_SECRET` in `.env`; then `python publisher/tiktok_poster.py --login` once (PKCE
with TikTok's hex SHA-256 challenge; access token 2 h, rotating refresh token 180 days, in
`workspace/tiktok_token.json`). **Until TikTok audits the app it can only post to a private account**
(`unaudited_client_can_only_post_to_private_accounts`).

```
python publisher/scheduler.py --dry-run --platform both   # pick + format per platform, touches nothing
python publisher/scheduler.py [--platform youtube|tiktok|both] [--force] [--interval 3.5] \
    [--youtube-privacy public|unlisted|private] [--tiktok-privacy PUBLIC_TO_EVERYONE|FOLLOWER_OF_CREATOR|MUTUAL_FOLLOW_FRIENDS|SELF_ONLY]
python backend/queue_worker.py --daemon --auto-post --post-platform both --post-interval 3.5 ...
```
- **Pick:** unposted `ready_to_post/*_short.mp4` + `_post.json`, ranked 0.8 × clip score (manifest) + 0.1 loudness
  health + 0.1 length health.
- **Caption:** `Title (<=55) — "hook quote" (via @streamer on Twitch)` + exactly 5 lowercase hashtags
  (`#fyp #gaming #twitchclips #<streamer> #<game/joke>`; `#clips` off Twitch).
- **Pacing:** one post per `--interval` hours **per platform** (`--force` overrides); history (with `platform`) in
  `workspace/posted_history.json`; files move to `posted/` once every enabled platform has the short. One platform
  failing (quota, expired login) never stops the other. TikTok Upload: creator_info → video/init (FILE_UPLOAD, ≤64 MB one chunk, else 10 MB
  chunks, last one takes the remainder) → PUT → status/fetch until `PUBLISH_COMPLETE`.
- Panel: the **AUTO POST** drawer (next to SETTINGS) holds the export settings (Auto-Edit, Facecam Box, Caption
  Theme, Music Gain), PUBLISHER (Publish Platform, Auto-Post, interval, YouTube / TikTok privacy), Login YouTube /
  TikTok and **POST BEST NOW**; clip rows have Post to YouTube / Post to TikTok and show POSTED TO YOUTUBE / TIKTOK.

## Calibration studio (`calibration_studio/`)

Rate a forced batch of clips by hand so the judge (`backend/scorer.py`) and director (`editor/director.py`) can be tuned.
Fully isolated: its own `runs/<batch>/queue.db`, `output/` and `shorts/`; nothing reaches `ready_to_post/`, `posted/`
or the live `seen_videos`.

**Double-click `Launch_TuneOClip_Studio.pyw`** (or `.bat`): an Edge / Chrome `--app` window, no console (log:
`calibration_studio/studio.log`); it quits a minute after the window closes unless a harvest is running.
`python calibration_studio/review_server.py` runs the same app in the default browser.

- **Harvest & Render**: channels, clips per channel, time window, min views; Start New Run / Resume Interrupted Run /
  Stop (kills the whole process tree; Resume carries on). Live harvested / rendered bars + ETA, per-streamer pills, the
  clip being worked on with its stage, finished shorts playable at once, log tail. Runs `harvest_batch.py` (also usable
  alone: `--streamers --per-streamer --hours --min-views --batch`); a run's settings are kept in `run.json`.
- Twitch windows widen (start -> 7 d -> 30 d -> all time) and min views drop (-> 100 -> 0, then any language) until each
  streamer has its count; the clipper runs with `--no-gate`, so judge-rejected / low clips are kept and edited too.
  Resuming skips processed clips and finished edits (Whisper words are cached), so only what is missing is rendered.
- **Review**: any run, finished or interrupted (`decision_log.json` + `DECISION_REPORT.md` are rebuilt from the files
  on disk when a run opens). Native `<video>` (HTTP Range), loop, Raw / Edited (R), Space, Left/Right +-2 s, Up/Down or
  N/P, 1-9, 0 = 10, labels K Keep / X Cut-Boring / W Wrong Archetype / F Bad Edit-Framing / D Audio-Caption Drift,
  tags, notes; transcript, judge pillars, archetype, cuts, QC. Everything saves to `reviews.json` at once.
- **Weight Tuner**: sliders for `scorer.TUNABLE` (pillar weights, dead-air rule, view boost, final-score weights) + Min
  Comedy Score, re-gating the run's judged clips live with scorer.py's own functions, against your Keep / Cut labels.
  **Save Weights** writes `backend/weights.json` (only changes from the defaults; loaded by `scorer.load_weights()` at
  import, junk ignored) + `min_score` in `workspace/settings.json`. Director rules are code: label "Wrong Archetype" and
  **Export Tuning Report** (`CLAUDE_TUNING_REPORT.md` + `.json`) for Claude.
- Ollama calls (judge + director) have a hard 45 s wall-clock limit (`scorer.ollama_generate`), one retry, then the
  normal fallback, so a stalled model can never freeze a batch.

## Tests

```sh
python backend/test_scorer.py
python backend/test_queue.py
python backend/test_twitch.py
python editor/test_editor.py
python publisher/test_publisher.py
python calibration_studio/test_calibration.py
godot --headless --path godot_app -s res://tests/test_main.gd
```

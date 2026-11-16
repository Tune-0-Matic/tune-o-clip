# Tune-O-Clip taste profile

My comedy and editing taste, learned from Clip Rate batches. Claude reads this FIRST on every "Send to Claude",
merges the new batch into it, and keeps older rules unless a newer batch contradicts them (then marks them
superseded with the date instead of deleting them). backend/scorer.py and editor/director.py must follow
the whole profile, not just the latest batch.

Evidence tags: (B1 xN) = batch 1, N separate notes or clips back the rule. Code references show where a rule is
enforced; rules without one are judged by me while rating and have no code yet.

## Core Comedy & Hook Preferences
What makes a clip an instant keep vs. a skip.

- **The streamer's personality IS the payoff.** Their reaction, voice and goofy commentary carry a clip; a classic
  punchline is not required. "caseoh is so loved that him making goofy commentary is all that's needed", "it didn't
  need to even say anything. hilarious". (B1 x6) -> scorer.JUDGE_PROMPT
- **Big vocal reactions are keeps:** screams, rage, panic, a jumpscare + a deadpan "welp". "I LOVE THE SCREAM, the whole
  thing is a rage" = 10/10. (B1 x5) -> JUDGE_PROMPT
- **Committed goofy stories, theories and rants work even when long:** a 54 s CaseOh theory "felt like 20 seconds";
  "the story is phenomenal". Length is fine while the story keeps going. (B1 x4)
- **Absurd or taboo talk and laughing at something dumb land** (fart jokes, a taboo line that "really kills it"). (B1 x3)
- **The hook = the first second or two:** open on a yell, a weird statement or a strange situation ("great start",
  "good hook" on most keeps). (B1 x12) -> JUDGE_PROMPT hook_strength
- **A clip may make little sense if the reaction and on-screen action are big enough** ("didn't make sense but there
  was enough clipping, movement and action"). (B1 x2) -> JUDGE_PROMPT self_contained
- The streamer looking cool / above it (Jynxzi singing "One Piece" instead of taking a call) plays well. (B1 x1)

## Hard Reject / Penalty Rules
Dead air, reading chat with no payoff, setup that needs earlier context, etc.

- **Nothing happens = skip:** dead air, long silence, waiting. "literally nothing happened", "too much silence",
  "nothing funny happened". Every 2/10 and 4/10 had this note. (B1 x7) -> JUDGE_PROMPT (low payoff)
- **No real reaction from the streamer:** IShowSpeed silent while a kid asks to be friends; Jynxzi just accepting
  second place. (B1 x2) -> JUDGE_PROMPT
- **Repeating the same line with nothing happening** ("said the same word over and over"). (B1 x1) -> JUDGE_PROMPT
- **Context-only events:** an award or name only stream regulars care about, with no reaction ("some random guy
  winning some random award speed pretended to care about" = 2/10). (B1 x2) -> JUDGE_PROMPT self_contained
- **Someone else talking pointlessly while the streamer is quiet** ("this guy talking is terrible and pointless",
  "she didn't say anything funny"). (B1 x2) -> JUDGE_PROMPT
- Watch: the judge's dead-air reject (payoff < 5 AND hook < 5) did not separate my ratings in B1 (rejected clips
  averaged 6.4 vs 6.9 kept; 4 rejected clips got 8-10). Kept unchanged; revisit if it keeps losing bangers.

## Editing, Pacing & Framing Rules
Cut timing, zoom behaviour, archetype choice (chat_roast, unfiltered_chaos, horror_jumpscare, confident_fail,
awkward_irl), captions, SFX.

- **Always see the streamer's whole face, on top.** Never crop the webcam panel onto an empty spot or cut off a head;
  no webcam in the clip -> full-frame blur_fill, not a fake split. (B1 x7; blur_fill shorts avg 8.5 vs split_stack
  6.6) -> pipeline.pick_facecam (a saved box must match this clip's detection; DETECT_EVERY_CLIP)
- **Tight jump cuts are a big part of why keeps work** ("love these cuts", "good cuts" on most keeps). (B1 x14)
- **Zoom-ins on the streamer's reaction are liked**, never a zoom on nothing. (B1 x4)
- **Effects go on the real moment:** the scratch / vine boom / snap zoom belong on the biggest yell, and SFX with
  nothing happening "looked artificial". (B1 x2) -> director.build_edl chat_roast (loudest peak after the setup;
  SFX only when it is >= LOUD)
- **Hook teaser = the biggest yell or it's skipped.** A teaser showing a meaningless moment was called out 4 times
  ("the yell with a cussword should've been the intro teaser"); teaser shorts still averaged 7.1 vs 6.4 without,
  so keep teasers, just on the right moment. (B1 x4) -> director (teaser pivot = the loudest peak)
- **No yellow banner text on top** of the teaser. (B1 x2) -> renderer.write_ass (Banner removed)
- **End soon after the payoff:** don't hold the tail of a scream or a dead last few seconds ("went on half a second
  longer", "end scream could've been cut sooner"); but never cut a story short. (B1 x5 vs x2 "great end cut")
  -> director.tail_guard limit 1.8 -> 1.2 s
- **Captions:** the kinetic style is liked; place them higher, clear of the platform title / subscribe UI (B1 x1)
  -> renderer MarginV 540 -> 660. Missing captions are disliked (B1 x1). Double captions when the source already has
  burned-in subtitles are disliked (B1 x1; no fix yet: the v10 detector fired only on game UI). A different
  colour / position for a second speaker is wanted (B1 x1; the old opt-in feature was removed in v9: revisit if
  this recurs).
- Grey screen on a scary moment was liked. (B1 x1)
- **Loud wordless action is not dead air:** a gap between words is kept when its audio is >= 0.2 of the clip's
  loudest (up to 5 s); quiet pauses stay jump-cut. (BM: 10/10 shorts keep 1.9 s max gaps, ours 0.8 s; on our raw
  clips 13 of 93 trimmed gaps were loud) -> director.loud_gap / speech_segments(env=)
- **No short under 15 s** when the source has 15 s (trimmed pauses / earlier setup come back first). (BM: no 10/10
  short under 15.3 s, 5 of our 20 at 8.5-12 s; B1 duration r = +0.17) -> director.TARGET_MIN
- **A picture change at least every 2.7 s, the first by 1.1 s** (1.2x snap punch on a word start; chat_roast,
  unfiltered_chaos, confident_fail only). (BM: 10/10 every 2.7 s vs ours 4.4 s) -> director.pace_punches
- Benchmark rules NOT adopted because my ratings disagree: a long aftermath after the climax (10/10s keep ~12 s,
  B1 says end soon after the payoff) and "more loud beats" (r = -0.11 / -0.21 with my B1 ratings).

Evidence tag (BM) = the 2026-11-15 benchmark of 20 reference 10/10 Shorts vs 20 of ours (workspace/UPGRADE_PLAN.md).
My own ratings outrank (BM): a later batch that contradicts a (BM) rule supersedes it.
- **Archetype matters less than framing and timing so far:** 1 "Wrong Archetype" tag in 35 (31 were chat_roast).
  The scenario is picked by director.classify() rules; the Ollama model's per-scenario fit is only recorded
  (scenario_fit), because asked to pick it collapsed to one scenario in all three prompt wordings tried.

## Per-Streamer Calibration Notes

### caseoh_
- Highest rated: 8.2 avg over 9 clips (the rest 6.3). Goofy commentary alone is enough; stories and theories can
  run long. (B1) -> scorer.STREAMER_BOOST +0.1
- Fixed top-left webcam: the saved box matched detection on every clip (overlap 0.54-0.85).

### ishowspeed
- 6.0 avg over 8. Great when he reacts big or absurd (a wordless reaction scored 10); weak when he is silent,
  repeats himself, or reacts to an award nobody knows.
- IRL streams have no fixed webcam: the saved box cut his head off / zoomed on nothing. -> per-clip detection check

### jynxzi
- 6.75 avg over 8. Rage and screams are 10s; low when nothing happens or another person does the talking.
- Webcam box reliable (overlap 0.49-1.0).

### lospollostv
- 6.2 avg over 10. Conversations, goofy topics and taboo jokes work; "we need to see the streamer".
- IRL / calls: camera layout changes every stream -> pipeline.DETECT_EVERY_CLIP (detect each clip, blur_fill when
  no webcam). Its blur_fill shorts got 10, 8, 8.

## Batch Change Log
One entry per batch: date, batch size, human vs AI correlation before and after, and the exact code changes.

| date | clips | correlation before | correlation after | code changes |
|---|---|---|---|---|
| 2026-11-14 | 35 | 0.14 | 0.23 | judge rubric, CaseOh prior, outburst/teaser/SFX placement, tail 1.2 s, no banner, captions higher, facecam per-clip check (details below) |
| 2026-11-15 | benchmark 20 vs 20 (no new ratings) | - | - | loud gaps kept, 15 s floor, pacing punches; scorer unchanged (details below) |

### 2026-11-14 (batch 1, 35 clips from calibration run 2026-11-14)
- **Correlation:** before r = 0.14 (stored judge). After r = 0.23 = a fresh Ollama run of the new judge prompt
  (0.135 alone) + viral + streamer boosts, scored exactly like clipper.py. Caveats: re-running the OLD prompt gave
  -0.03 and an earlier draft of the new one 0.08, so the judge alone moves by about +-0.15 between runs; the
  3B judge reading only the transcript cannot rank by this taste yet. The CaseOh prior is the reliable gain (+0.11..
  0.13 on all three judge runs) but was fitted on these same 35 clips. Editing changes (framing, teaser, SFX, tail,
  captions) can't be scored offline: they need new renders + the next batch's ratings.
- **What predicted my ratings:** CaseOh +0.38, blur_fill layout +0.28, quotability +0.29 (stored run only; 0.06-0.22
  on re-runs), every other judge pillar 0.11-0.18, views and words/second none.
- **Code changes:**
  - backend/scorer.py: JUDGE_PROMPT rewritten from this profile (personality payoff, "nothing happens" = low,
    context-only events, someone else talking); STREAMER_BOOST = {"caseoh_": 0.1} + streamer_boost(), applied in
    clipper.twitch_clip after viral_boost and in the studio tuner (review_server.gate).
  - editor/director.py: chat_roast outburst = loudest peak after the first 30% (was the latest peak); the clip still
    runs to the latest peak; the outburst is forced into the cuts; record scratch + vine boom + mute only when the
    outburst is >= LOUD (0.8); the hook teaser follows the outburst; tail_guard limit 1.8 -> 1.2 s.
  - editor/renderer.py: yellow teaser Banner removed; caption MarginV 540 -> 660.
  - editor/pipeline.py: pick_facecam runs detection on every clip; a detection overlapping the saved box < 0.3
    wins for that clip (preset kept); DETECT_EVERY_CLIP = {"lospollostv"} never saves/uses a preset.
  - Tests updated: test_scorer (prior, prompt wording), test_editor (outburst placement, SFX gate, tail cap,
    facecam rules, MarginV).

### 2026-11-15 (benchmark: 20 reference 10/10 Shorts vs our 20 newest; no new ratings)
- **Correlation:** not re-scored. Editing changes need new renders + the next batch's ratings. A/B renders of 3 clips
  are in output/ab_test/ (old vs new).
- **Code changes:**
  - editor/director.py: `loud_gap()` + `speech_segments(env=)` keep loud wordless gaps (every scenario but
    unfiltered_chaos); `TARGET_MIN = 15.0` floor for the edit (MIN_LEN 8.5 stays the QC/repair floor);
    `pace_punches()` adds 1.2x snap punch-ins so no still stretch exceeds 2.7 s (first by 1.1 s), on word starts.
  - backend/scorer.py: unchanged. The one selection signal (loud-beat count) anti-correlates with my B1 ratings.
  - Tests: test_editor (loud vs quiet gaps, 15 s floor, punch cadence, awkward_irl stays calm).

### 2026-11-15 (editor rebuild on my request: framing, captions, pacing; no new ratings)
- **Supersedes (my explicit request):** the (BM) "loud wordless gaps up to 5 s" and "no short under 15 s" rules, the
  per-streamer facecam presets, awkward_irl's 3 s holds. Dead silence > 0.35 s is now always cut.
- **Code changes:**
  - editor/framing.py (new): YuNet face detection on 2 fps samples -> split_stack only for one small, steady corner
    face (crop centred on it, head + headset inside), face_crop (9:16 column following the faces) or blur_fill; a
    PP-OCRv3 text detector spots captions burned into the stream -> ours off and the frame shown once (blur_fill).
    The Haar/overlay-edge webcam guesser and workspace/facecam_presets.json are gone.
  - editor/director.py: keep_spans() = words + Silero VAD + loud reactions, silences > 0.35 s cut, 0.1 s pad; no
    forced chat_roast end / outburst window or confident_fail silent aftermath; teaser = the climax's own sound;
    zooms / takeovers only in split_stack.
  - editor/pipeline.py: word ends snap back to the last audible frame (Whisper stretched words over silence).
  - editor/renderer.py: face_crop layout, one caption track (none when the source has its own), 1-4 word chunks,
    side margins 140 px.

### 2026-11-15 (retention stack on my request; no new ratings)
- editor/director.py `choreograph()`: snap punch-ins (1.35x climax / 1.3x yells / 1.22x "!" "?" lines, held to the
  next sentence, -16 dB whoosh each), 1.12x push-ins so nothing sits still > 3 s, 0.25 s +-8 px shake on yells, -14 dB
  thud (new clean `thud` SFX) on the climax, 1.2-1.5 s cold open of a loud second-half climax on clips >= 16 s (1.25x,
  white flash + whoosh, not horror). Supersedes the per-scenario 1.45x / 1.65x zooms and the vine boom / metal pipe /
  sad violin SFX; the record scratch stays (-14 dB).
- editor/framing.py `zoom_room()`: every zoom grows about a head-safe anchor and stops before a head (or the stream's
  own caption line) leaves the panel; face_crop and blur_fill zoom again (capped), so the earlier "no zooms on IRL
  frames" rule is superseded by the cap.
- editor/renderer.py: 1-3 word lines, active word pops 118% -> 100% in 70 ms, yellow / cyan on hype + loud words,
  outline 4, shadow 3.

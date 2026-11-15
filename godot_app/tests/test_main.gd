extends SceneTree
## godot --headless --path godot_app -s res://tests/test_main.gd
## Checks the min/max length controls, feeds backend JSON lines into the real panel (incl. a stitched
## story-callback clip), then rejects a clip and checks the manifest on disk.


func _initialize() -> void:
	var main: Control = (load("res://scenes/main.tscn") as PackedScene).instantiate()
	root.add_child(main)
	await process_frame  # let _ready build the UI

	# repo root: the folder above godot_app/ (where backend/ writes output/, workspace/, ready_to_post/). Note
	# globalize_path("res://").get_base_dir() would be godot_app itself (the path ends in "/").
	var repo: String = main._project_root()
	assert(repo == ProjectSettings.globalize_path("res://").path_join("..").simplify_path() and FileAccess.file_exists(repo.path_join("backend/clipper.py")), repo)
	assert(main._ready_dir() == repo.path_join("ready_to_post"), main._ready_dir())
	var real_clips := 0  # counted independently from every real manifest (read-only)
	for sub: String in ["output", "workspace"]:
		for d: String in DirAccess.get_directories_at(repo.path_join(sub)):
			var mj := JSON.new()
			if mj.parse(FileAccess.get_file_as_string(repo.path_join(sub).path_join(d).path_join("clips_manifest.json"))) == OK and mj.data is Dictionary:
				real_clips += ((mj.data as Dictionary).get("clips", []) as Array).size()
	assert(main._entries.size() == real_clips, "startup found %d of %d clips" % [main._entries.size(), real_clips])
	print("repo clips found at startup: ", real_clips)
	var real_shorts := Array(DirAccess.get_files_at(main._ready_dir())).filter(func(f: String) -> bool: return f.ends_with("_short.mp4")).size()
	main.refresh_stats()
	assert(main.stats_label.text.contains("Shorts ready: %d" % real_shorts), main.stats_label.text)

	# SETTINGS: never touch the real workspace files; defaults, then persistence through workspace/settings.json
	main.settings_path = OS.get_user_data_dir().path_join("test_settings/settings.json")
	main.eta_path = OS.get_user_data_dir().path_join("test_settings/eta_stats.json")  # never the real one
	DirAccess.remove_absolute(main.eta_path)
	main.twitch_channels_path = OS.get_user_data_dir().path_join("test_twitch/twitch_channels.txt")
	DirAccess.remove_absolute(main.settings_path)
	DirAccess.remove_absolute(main.twitch_channels_path)
	main.load_settings()
	assert(main.settings == main.DEFAULT_SETTINGS, str(main.settings))
	assert(not main.settings_panel.visible, "SETTINGS drawer closed by default")
	main.settings_button.button_pressed = true
	assert(main.settings_panel.visible, "SETTINGS toggle opens the drawer")
	main.settings_button.button_pressed = false
	# AUTO POST drawer: export + publisher settings, logins, POST BEST NOW; one drawer open at a time
	assert(not main.auto_post_panel.visible, "AUTO POST drawer closed by default")
	main.settings_button.button_pressed = true
	main.auto_post_button.button_pressed = true
	assert(main.auto_post_panel.visible and not main.settings_panel.visible and not main.settings_button.button_pressed, "opening one closes the other")
	main.auto_post_button.button_pressed = false
	assert(not main.auto_post_panel.visible and not main.settings_panel.visible, "a drawer button closes its drawer")
	for key: String in ["auto_edit", "facecam_box", "caption_theme", "music_gain_db", "post_platform", "auto_post", "post_interval_h", "youtube_privacy", "post_privacy"]:
		assert(main.auto_post_panel.is_ancestor_of(main._widgets[key]) and not main.settings_panel.is_ancestor_of(main._widgets[key]), key)
	for key: String in ["clips", "min_score", "twitch_streamers", "cookies", "ollama_model"]:
		assert(main.settings_panel.is_ancestor_of(main._widgets[key]), key)
	for b: Button in [main.yt_login_button, main.login_button, main.post_button]:
		assert(main.auto_post_panel.is_ancestor_of(b), b.text)
	main.set_setting("clips", 6.0)
	main.set_setting("caption_theme", "Cyan")
	main.set_setting("facecam_box", "0,0.13,0.21,0.28")
	var on_disk: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(main.settings_path))
	assert(int(on_disk.clips) == 6 and on_disk.caption_theme == "Cyan" and on_disk.facecam_box == "0,0.13,0.21,0.28", str(on_disk))
	main.settings.clips = 1
	main.load_settings()  # startup: the saved file wins over defaults, and the widgets show it
	assert(main.settings.clips == 6 and typeof(main.settings.clips) == TYPE_INT and main._widgets.clips.value == 6.0, str(main.settings))
	assert((main._widgets.caption_theme as OptionButton).get_item_text((main._widgets.caption_theme as OptionButton).selected) == "Cyan")
	main.set_setting("clips", 4)
	assert(not main.settings.has("speaker_captions") and not main._widgets.has("speaker_captions"), "2nd Speaker Captions removed")

	# publisher: settings (auto-post OFF by default), both login buttons, POST BEST NOW, arguments, banner, auto-post pacing
	assert(main.settings.auto_post == false and main.settings.post_interval_h == 3.5 and main.settings.post_privacy == "Public")
	assert(main.settings.post_platform == "YouTube Shorts" and main.settings.youtube_privacy == "public")
	assert(main._widgets.auto_post is CheckBox and main.login_button.text == "Login TikTok (1-Time)" and main.post_button.text == "POST BEST NOW")
	assert(main.yt_login_button.text == "Login YouTube (1-Time)")
	assert((main._widgets.post_platform as OptionButton).item_count == 3 and (main._widgets.youtube_privacy as OptionButton).item_count == 3)
	assert((main._widgets.post_platform.get_parent() as GridContainer).get_child_count() == 10, "5 publisher settings")
	var pargs: PackedStringArray = main.publish_args(true)
	assert(pargs[1].ends_with("publisher/scheduler.py") and pargs.has("--force") and pargs[pargs.find("--interval") + 1] == "3.5", str(pargs))
	assert(pargs[pargs.find("--platform") + 1] == "youtube" and pargs[pargs.find("--youtube-privacy") + 1] == "public", str(pargs))
	assert(pargs[pargs.find("--tiktok-privacy") + 1] == "PUBLIC_TO_EVERYONE" and not main.publish_args(false).has("--force"))
	main.set_setting("post_privacy", "Only Me")
	main.set_setting("youtube_privacy", "unlisted")
	main.set_setting("post_platform", "Both")
	pargs = main.publish_args(false)
	assert(pargs[pargs.find("--tiktok-privacy") + 1] == "SELF_ONLY" and pargs[pargs.find("--youtube-privacy") + 1] == "unlisted")
	assert(pargs[pargs.find("--platform") + 1] == "both", str(pargs))
	main.set_setting("post_platform", "TikTok")
	assert(main.publish_args(false)[main.publish_args(false).find("--platform") + 1] == "tiktok")
	main.set_setting("post_privacy", "Public")
	main.set_setting("youtube_privacy", "public")
	main.set_setting("post_platform", "YouTube Shorts")
	assert(main.explain_error("[PUBLISH] TikTok session expired: click Login TikTok (1-Time) once").begins_with(
			"TikTok Session Expired: Click Login TikTok (1-Time) in AUTO POST"))
	assert(main.explain_error("[PUBLISH] YouTube API quota exceeded (100 uploads per day per project): waiting").begins_with("YouTube Quota Exceeded"))
	assert(main.explain_error("[PUBLISH] YouTube OAuth token expired: click Login YouTube (1-Time) once").begins_with("YouTube Token Expired"))
	assert(main.explain_error("[PUBLISH] YouTube Auth Failed: invalid_client. Check YOUTUBE_CLIENT_ID").begins_with("YouTube Auth Failed"))
	assert(main.explain_error("[PUBLISH] YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET must be set in .env").begins_with("YouTube Auth Failed"))
	assert(main.explain_error("[PUBLISH] TikTok video/init: unaudited_client_can_only_post_to_private_accounts: x").begins_with("TikTok App Not Audited Yet"))
	assert(main.explain_error("[PUBLISH] TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET must be set").begins_with("TikTok Keys Missing"))
	var pub_ready := OS.get_user_data_dir().path_join("test_pub/ready_to_post")
	DirAccess.make_dir_recursive_absolute(pub_ready)
	DirAccess.make_dir_recursive_absolute(pub_ready.get_base_dir().path_join("posted"))
	main.ready_dir = pub_ready
	main.history_path = pub_ready.get_base_dir().path_join("posted_history.json")
	DirAccess.remove_absolute(main.history_path)
	for f: String in DirAccess.get_files_at(pub_ready):
		DirAccess.remove_absolute(pub_ready.path_join(f))
	var now_u := 1_800_000_000.0
	main._auto_post_try = -1e12
	assert(not main.auto_post_due(now_u), "auto-post off")
	main.set_setting("auto_post", true)
	assert(not main.auto_post_due(now_u), "nothing waiting in ready_to_post")
	FileAccess.open(pub_ready.path_join("v_1_short.mp4"), FileAccess.WRITE).store_string("x")
	assert(main.auto_post_due(now_u), "a short and no post yet")
	FileAccess.open(main.history_path, FileAccess.WRITE).store_string(JSON.stringify({"posts": [
			{"short_id": "a", "platform": "youtube", "posted_at": now_u - 3600, "status": "UPLOADED"},
			{"short_id": "a", "posted_at": now_u - 3 * 3600, "status": "PUBLISH_COMPLETE"}]}))  # no platform = TikTok
	assert(not main.auto_post_due(now_u), "YouTube posted 1 h ago: wait for 3.5 h")
	main.set_setting("post_platform", "Both")
	assert(not main.auto_post_due(now_u), "TikTok posted 3 h ago: not due either")
	assert(main.auto_post_due(now_u + 0.5 * 3600), "TikTok's own 3.5 h are over")
	main.set_setting("post_platform", "YouTube Shorts")
	assert(not main.auto_post_due(now_u + 0.5 * 3600), "YouTube only: still 2 h to go")
	assert(main.auto_post_due(now_u + 2.5 * 3600), "3.5 h after the last post: due")
	main._auto_post_try = now_u + 2.5 * 3600
	assert(not main.auto_post_due(now_u + 2.5 * 3600 + 60), "a try a minute ago: back off 10 min")
	main.set_setting("auto_post", false)
	main._auto_post_try = -1e12
	# a posted short (moved to posted/) still counts as the clip's short
	var posted_e := {"data": {"video_id": "pv"}, "clip": {"clip_id": 1}, "path": "x"}
	FileAccess.open(pub_ready.get_base_dir().path_join("posted/pv_1_short.mp4"), FileAccess.WRITE).store_string("x")
	FileAccess.open(pub_ready.get_base_dir().path_join("posted/pv_1_post.json"), FileAccess.WRITE).store_string(JSON.stringify({"title": "Spoke too soon", "duration": 30.5}))
	assert(main.short_path(posted_e) == pub_ready.get_base_dir().path_join("posted/pv_1_short.mp4"), main.short_path(posted_e))
	assert(main._short_summary(posted_e).begins_with("POSTED: Spoke too soon  |  30.5s"), main._short_summary(posted_e))
	FileAccess.open(main.history_path, FileAccess.WRITE).store_string(JSON.stringify({"posts": [
			{"short_id": "pv_1", "platform": "youtube", "posted_at": now_u, "status": "UPLOADED"},
			{"short_id": "pv_1", "platform": "tiktok", "posted_at": now_u, "status": "FAILED"}]}))
	main._posted_by = main.posted_by()
	assert(main._short_summary(posted_e).begins_with("POSTED TO YOUTUBE: Spoke too soon"), main._short_summary(posted_e))
	FileAccess.open(main.history_path, FileAccess.WRITE).store_string(JSON.stringify({"posts": [
			{"short_id": "pv_1", "platform": "youtube", "posted_at": now_u, "status": "UPLOADED"},
			{"short_id": "pv_1", "posted_at": now_u, "status": "PUBLISH_COMPLETE"}]}))
	main._posted_by = main.posted_by()
	assert(main._short_summary(posted_e).begins_with("POSTED TO YOUTUBE + TIKTOK: Spoke too soon"), main._short_summary(posted_e))
	main._handle_line(JSON.stringify({"event": "log", "msg": "[PUBLISH] Posted to YouTube Shorts: https://youtube.com/shorts/abc"}))
	main._handle_line(JSON.stringify({"event": "publish", "platform": "youtube", "short_id": "pv_1", "status": "UPLOADED"}))
	assert(main.log_view.get_parsed_text().contains("[PUBLISH] Posted to YouTube Shorts: https://youtube.com/shorts/abc"))
	main.ready_dir = ""
	# the widgets drive the same setting (and save it)
	(main._widgets.min_score as SpinBox).value = 7.5
	assert(main.settings.min_score == 7.5 and float(JSON.parse_string(FileAccess.get_file_as_string(main.settings_path)).min_score) == 7.5)
	main.set_setting("min_score", 6.0)

	# dynamic length bounds: defaults 15/55 reach the backend; min <= max is kept
	var args: PackedStringArray = main.build_args("https://youtu.be/x")
	assert(args.has("--min-length") and args[args.find("--min-length") + 1] == "15", "min default")
	assert(args.has("--max-length") and args[args.find("--max-length") + 1] == "55", "max default")
	assert(args[args.find("--min-score") + 1] == "6.0" and args[args.find("--whisper-model") + 1] == "small.en", str(args))
	assert(args[args.find("--ollama-model") + 1] == main.OLLAMA_MODEL, "ollama model passed")
	assert(not args.has("--length"), "fixed length flag gone")
	assert(not args.has("--cookies-from-browser") and not args.has("--cookies"), "no cookies by default")
	main.set_setting("cookies", "brave")
	args = main.build_args("https://youtu.be/x")
	assert(args[args.find("--cookies-from-browser") + 1] == "brave", "browser cookies passed through")
	main.set_setting("cookies", "cookies.txt")
	args = main.build_args("https://youtu.be/x")
	assert(args[args.find("--cookies") + 1].ends_with("/cookies.txt") and not args.has("--cookies-from-browser"), str(args))
	main.set_setting("cookies", "OFF")

	# source modes: single URL field / batch paste box / Twitch hint
	assert(main.url_edit.visible and not main.batch_box.visible and not main.twitch_hint.visible, "single URL by default")
	main.set_mode(2)
	assert(main.twitch_hint.visible and not main.url_edit.visible and not main.batch_box.visible, "Twitch mode")
	main.set_mode(1)
	assert(main.batch_box.visible and not main.url_edit.visible, "Batch mode shows the paste box")
	var bargs: PackedStringArray = main.build_batch_args("https://youtu.be/AAAAAAAAAAA
https://www.youtube.com/playlist?list=PL1
")
	assert(bargs[1].ends_with("backend/queue_worker.py") and bargs.has("--drain") and bargs.has("--add-file"), str(bargs))
	assert(bargs[bargs.find("--max-length") + 1] == "55", "clipper options pass through")
	var pasted := FileAccess.get_file_as_string(bargs[bargs.find("--add-file") + 1])
	assert(pasted.contains("AAAAAAAAAAA") and pasted.contains("list=PL1"), pasted)
	main._handle_line('{"event": "queue", "msg": "Cooldown 31s", "pending": 7, "done": 2, "failed": 1}')
	assert(main.log_view.get_parsed_text().contains("[QUEUE] Cooldown 31s  (pending 7, done 2, failed 1)"), "queue status logged")
	# Twitch: streamer logins normalised, saved for the daemon, optional game filter
	assert(main.settings.twitch_game == "", "game filter blank by default")
	assert(Array(main.parse_logins("xqc, @Kai_Cenat\nhttps://www.twitch.tv/caseoh_/clips?x=1  twitch.tv/Jynxzi;xqc, bad-name!, "))
			== ["xqc", "kai_cenat", "caseoh_", "jynxzi"], "logins normalised")
	var targs: PackedStringArray = main.build_twitch_args("XQC, @kai_cenat", "")
	assert(targs[targs.find("--poll-twitch-channels") + 1] == main.twitch_channels_path and not targs.has("--twitch-game"), str(targs))
	assert(FileAccess.get_file_as_string(main.twitch_channels_path) == "xqc\nkai_cenat\n", "streamers persisted")
	assert(targs.has("--drain") and targs[targs.find("--twitch-limit") + 1] == "20" and targs[1].ends_with("backend/queue_worker.py"), str(targs))
	assert(targs[targs.find("--twitch-min-views") + 1] == "1500" and targs[targs.find("--twitch-hours") + 1] == "24" and targs.has("--twitch-lang=en"), str(targs))
	main.set_setting("twitch_streamers", "@CaseOh_, jynxzi")  # typing in SETTINGS syncs the daemon's file
	assert(FileAccess.get_file_as_string(main.twitch_channels_path) == "caseoh_\njynxzi\n", "settings streamers synced")
	assert(targs[targs.find("--twitch-per-streamer") + 1] == "0", "Clips per Streamer: 0 = even split by default")
	(main._widgets.twitch_per_streamer as SpinBox).value = 5
	assert(main.settings.twitch_per_streamer == 5 and int(JSON.parse_string(FileAccess.get_file_as_string(main.settings_path)).twitch_per_streamer) == 5)
	targs = main.build_twitch_args("xqc", "")
	assert(targs[targs.find("--twitch-per-streamer") + 1] == "5", str(targs))
	main.set_setting("twitch_per_streamer", 0)
	main.set_setting("twitch_lang", "")
	assert(main.build_twitch_args("xqc", "").has("--twitch-lang="), "blank language = any")
	main.set_setting("twitch_lang", "en")
	targs = main.build_twitch_args("xqc", " Grand Theft Auto V ")
	assert(targs[targs.find("--twitch-game") + 1] == "Grand Theft Auto V", "streamers + game filter")
	targs = main.build_twitch_args("", "Grand Theft Auto VI")
	assert(targs[targs.find("--poll-twitch-game") + 1] == "Grand Theft Auto VI" and not targs.has("--poll-twitch-channels"), "game only")
	assert(main.build_twitch_args(" , ", "").is_empty(), "nothing to poll")
	assert(main.parse_logins(FileAccess.get_file_as_string(main.twitch_channels_path)) == PackedStringArray(["xqc"]), "reload")
	# auto-edit: on by default for queue runs, off -> flag dropped
	assert(main.settings.auto_edit and main.build_batch_args("https://youtu.be/AAAAAAAAAAA").has("--auto-edit"), "auto-edit default on")
	assert(main.build_twitch_args("xqc", "").has("--auto-edit"), "twitch runs auto-edit too")
	(main._widgets.auto_edit as CheckBox).button_pressed = false
	assert(not main.build_batch_args("https://youtu.be/AAAAAAAAAAA").has("--auto-edit"), "unticked -> no flag")
	(main._widgets.auto_edit as CheckBox).button_pressed = true
	main.set_mode(0)
	main.set_setting("min_length", 70)
	assert(main.settings.max_length == 70 and main._widgets.max_length.value == 70.0, "raising min past max drags max up")
	main.set_setting("max_length", 20)
	assert(main.settings.min_length == 20 and main._widgets.min_length.value == 20.0, "lowering max below min drags min down")
	main.set_setting("min_length", 15)
	main.set_setting("max_length", 55)

	# Ollama badge: explicit states, then a real poll must settle on ONLINE or OFFLINE within a few seconds
	main.set_ollama_status(true)
	assert(main.ollama_badge.text == "OLLAMA: ONLINE" and main.ollama_badge.get_theme_color("font_color") == main.GOOD, "online badge")
	assert(not main.banner.visible, "no error banner at a healthy startup")
	main.set_ollama_status(false)
	assert(main.ollama_badge.text == "OLLAMA: OFFLINE" and main.ollama_badge.get_theme_color("font_color") == main.AMBER, "offline badge")
	assert(main.banner.visible and main.banner_label.text.begins_with("Ollama Offline: Start Ollama from the Start Menu"), main.banner_label.text)
	(main.banner.get_child(0).get_child(1) as Button).pressed.emit()
	assert(not main.banner.visible, "x dismisses the banner")
	main.set_ollama_status(false)
	assert(not main.banner.visible, "staying offline does not re-open it every poll")
	assert(main.explain_error("Twitch token request failed (400): invalid client secret")
			== "Twitch Auth Failed (400): Check TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET in .env and ensure the Twitch app is Confidential.")
	assert(main.explain_error("clip 2: t_2 failed QC (silent)").begins_with("Short Failed QC: "))
	assert(main.explain_error("weird thing").begins_with("Run Failed: Read the CONSOLE"))
	# warnings (run finished but did less than asked): amber banner with the exact reason, LED not ERROR
	main.begin_run()
	main._handle_line(JSON.stringify({"event": "warning", "msg": "Twitch user not found: 'kai_cenatt'", "login": "kai_cenatt"}))
	assert(main.banner.visible and main.banner_label.text.begins_with("Twitch User Not Found: Fix that login in SETTINGS > TWITCH POLLING"), main.banner_label.text)
	main._handle_line(JSON.stringify({"event": "warning", "msg": "0 new clips: Twitch returned 100 clips for kai_cenat: 100 already processed, 0 filtered by views/lang (0 new)"}))
	assert(main.banner_label.text.begins_with("No New Twitch Clips: Every clip Twitch has for these streamers is already processed"), main.banner_label.text)
	assert(main.banner_label.text.contains("100 clips for kai_cenat: 100 already processed"), "the exact breakdown is shown")
	assert((main.banner.get_theme_stylebox("panel") as StyleBoxFlat).border_color == main.AMBER, "warning = amber")
	assert(main.log_view.get_parsed_text().contains("WARNING: 0 new clips: Twitch returned 100 clips"), "warning logged")
	main.on_run_ended(0)
	assert(main.status_state == "DONE" and main.banner.visible, "a warning keeps the banner but the run is DONE")
	main.show_banner("ffmpeg render failed")
	assert((main.banner.get_theme_stylebox("panel") as StyleBoxFlat).border_color == main.BAD, "errors stay red")
	main.banner.visible = false
	main._last_run_unix = 0  # begin_run above set the LAST RUN marker; the clip filter checks below expect none
	# ERRORS panel: every problem of the session stays listed (newest first) after the banner moves on; CLEAR empties it
	assert(main._errors.size() >= 3, "the banner calls above were all recorded")
	main.clear_errors()
	await process_frame
	assert(main.error_count.text == "ERRORS: 0" and not main.banner.visible and main.error_list.get_child_count() == 1)
	assert((main.error_list.get_child(0) as Label).text == "No problems this session.")
	main.show_banner("ffmpeg render failed")
	main._handle_line(JSON.stringify({"event": "warning", "msg": "0 new clips: Twitch returned 20 clips for xqc: 20 already processed, 0 filtered by views/lang (0 new)"}))
	await process_frame
	var err_lines: Array = main.error_list.get_children().map(func(l: Node) -> String: return (l as Label).text)
	assert(main.error_count.text == "ERRORS: 1   WARNINGS: 1", main.error_count.text)
	assert(err_lines.size() == 2 and err_lines[0].contains("No New Twitch Clips") and err_lines[1].contains("ffmpeg Problem: Make sure ffmpeg"), str(err_lines))
	assert(err_lines[1].begins_with("[") and err_lines[1].contains("ffmpeg render failed"), "time + cause + raw message")
	assert((main.error_list.get_child(1) as Label).tooltip_text == "ffmpeg render failed", "full message on hover")
	main.banner.visible = false
	assert(main.error_list.get_child_count() == 2, "closing the banner keeps the list")
	main.clear_errors()
	# [JUDGE]: a video where nothing passed explains itself (amber banner + ERRORS list), never a silent empty list
	main._handle_line(JSON.stringify({"event": "warning", "msg": "[JUDGE] Rejected all 12 candidate(s) for abc123 (best: self_contained=3 (needs >= 4), payoff=7 (needs >= 5)): Starts mid-argument."}))
	assert(main.banner_label.text.begins_with("No Clip Passed the AI Judge: Nothing in this video was funny or self-contained enough."), main.banner_label.text)
	assert(main.banner_label.text.contains("self_contained=3 (needs >= 4)") and main.error_count.text.contains("WARNINGS: 1"), main.error_count.text)
	main.clear_errors()
	main.ollama_badge.text = "OLLAMA: ..."
	main.poll_ollama()
	var waited := 0.0
	while main.ollama_badge.text == "OLLAMA: ..." and waited < 4.0:
		await create_timer(0.1).timeout
		waited += 0.1
	assert(main.ollama_badge.text in ["OLLAMA: ONLINE", "OLLAMA: OFFLINE"], "real poll settled: " + main.ollama_badge.text)
	print("live poll: ", main.ollama_badge.text)
	main._poll_timer.stop()  # from here on the test decides what is online

	# START gate: locked until every system is online, says what it waits for, unlocks by itself
	assert(main._systems.python and main._systems.ffmpeg, "boot preflight fed the gate: " + str(main._systems))
	main.set_mode(0)
	main._systems.twitch_keys = true
	main.set_ollama_status(false)
	assert(main.run_button.disabled and main.run_button.text == "WAITING: OLLAMA", main.run_button.text)
	assert(main.run_button.tooltip_text.contains("START unlocks by itself once these are online: Ollama"), main.run_button.tooltip_text)
	main.url_edit.text = "https://youtu.be/x"
	main.start()  # Enter in the URL box: same gate
	assert(main._thread == null and main.log_view.get_parsed_text().contains("START: waiting for Ollama to come online."), "blocked")
	main._systems.ffmpeg = false
	main.update_start_button()
	assert(main.run_button.text == "WAITING: FFMPEG, OLLAMA", main.run_button.text)
	main._systems.ffmpeg = true
	main.set_ollama_status(true)
	assert(not main.run_button.disabled and main.run_button.text == "START", "everything online: unlocked")
	main._systems.twitch_keys = false
	main.set_mode(2)
	assert(main.run_button.disabled and main.run_button.text == "WAITING: TWITCH KEYS", "Twitch mode also needs the keys")
	main.set_mode(0)
	assert(not main.run_button.disabled, "YouTube mode doesn't")
	main._systems.twitch_keys = true
	main.url_edit.text = ""
	# --- clips: every manifest under the clip roots (output-like + workspace-like), newest manifest first ------------
	var out_root := OS.get_user_data_dir().path_join("test_clips_out")
	var ws_root := OS.get_user_data_dir().path_join("test_clips_ws")
	var old_dir := out_root.path_join("old_job")
	var dir := ws_root.path_join("test_job")
	for d: String in [old_dir, dir, out_root.path_join("broken_job"), out_root.path_join("no_manifest")]:
		DirAccess.make_dir_recursive_absolute(d)
	FileAccess.open(out_root.path_join("broken_job/clips_manifest.json"), FileAccess.WRITE).store_string("{not json")
	FileAccess.open(old_dir.path_join("clip_01_10s.mp4"), FileAccess.WRITE).store_string("x")
	FileAccess.open(old_dir.path_join("clips_manifest.json"), FileAccess.WRITE).store_string(JSON.stringify({"video_id": "old",
			"title": "Older Video", "clips": [{"clip_id": 1, "file": "clip_01_10s.mp4", "start_sec": 10, "end_sec": 30, "score": 0.6}]}))
	await create_timer(1.1).timeout  # manifest mtimes have 1 s resolution
	for n: String in ["clip_01_90s.mp4", "clip_02_190s.mp4"]:
		FileAccess.open(dir.path_join(n), FileAccess.WRITE).store_string("x")
	var manifest := dir.path_join("clips_manifest.json")
	FileAccess.open(manifest, FileAccess.WRITE).store_string(JSON.stringify({"video_id": "t", "title": "Newest Video", "clips": [
		{"clip_id": 1, "file": "clip_01_90s.mp4", "start_sec": 90, "end_sec": 120, "score": 0.9, "reason": "Heatmap peak 0.90", "transcript": "haha", "judge_reason": "Clean setup, big payoff."},
		{"clip_id": 2, "file": "clip_02_190s.mp4", "start_sec": 845, "end_sec": 872, "setup_start_sec": 72, "setup_end_sec": 80,
			"duration": 35, "score": 0.5, "reason": "Story Callback: [1:12-1:20] -> [14:05-14:32] (dinosaur, purple) | Most Replayed 3.4x spike @ 14:22 | Peak 0.96",
			"transcript": "purple dinosaur ... he's back"}]}))
	main.clip_roots = PackedStringArray([out_root, ws_root, OS.get_user_data_dir().path_join("no_such_root")])
	main.ready_dir = OS.get_user_data_dir().path_join("test_ready")  # never the real ready_to_post
	DirAccess.make_dir_recursive_absolute(main.ready_dir)
	for f: String in DirAccess.get_files_at(main.ready_dir):
		DirAccess.remove_absolute(main.ready_dir.path_join(f))

	main._handle_line('{"event": "stage", "stage": "cutting", "msg": "Cutting clip 1/2"}')
	main._handle_line("not json")
	main._handle_line(JSON.stringify({"event": "done", "manifest": manifest, "count": 2}))
	await process_frame
	var rows: Array[Node] = main.clip_list.get_children()
	var head_of := func(row: Node) -> String: return (row.get_child(0).get_child(0) as Label).text
	assert(rows.size() == 3 and main.clip_count.text == "CLIPS: 3", "all manifests aggregated: %d %s" % [rows.size(), main.clip_count.text])
	assert(head_of.call(rows[0]).begins_with("#1   Newest Video") and head_of.call(rows[2]).begins_with("#1   Older Video"), "newest first")
	assert(main.log_view.get_parsed_text().contains("[CUTTING] Cutting clip 1/2"), "stage logged")
	assert(head_of.call(rows[1]).contains("00:14:05 - 00:14:32") and head_of.call(rows[1]).contains("Story Callback: [1:12-1:20] -> [14:05-14:32]"), str(head_of.call(rows[1])))

	# preview: short path from video id + clip id
	var e_new: Dictionary = main._entries[0]
	var saved_ready: String = main.ready_dir
	main.ready_dir = ""
	assert(main.short_path(e_new).ends_with("ready_to_post/t_1_short.mp4"), "default ready_to_post")
	main.ready_dir = saved_ready
	var row_text := func(row: Node) -> Array:
		return row.get_child(0).get_children().filter(func(n: Node) -> bool: return n is Label).map(func(l: Label) -> String: return l.text)
	var row_buttons := func(row: Node) -> Array:
		return row.get_child(0).get_child(row.get_child(0).get_child_count() - 1).get_children().map(func(b: Node) -> String: return (b as Button).text)
	assert(row_buttons.call(rows[0]).has("MAKE SHORT") and not row_buttons.call(rows[0]).has("Open Ready-to-Post Folder"), str(row_buttons.call(rows[0])))
	assert(not str(row_text.call(rows[0])).contains("EDITED SHORT"), "no short yet")
	assert(str(row_text.call(rows[0])).contains("AI judge: Clean setup, big payoff."), "why the judge kept it")
	assert(not str(row_text.call(rows[1])).contains("AI judge:"), "no reason, no line")
	# an edited short exists -> its row shows the title/length and PLAY EDITED SHORT
	var spath: String = main.short_path(e_new)
	FileAccess.open(spath, FileAccess.WRITE).store_string("x")
	FileAccess.open(spath.trim_suffix("_short.mp4") + "_post.json", FileAccess.WRITE).store_string(JSON.stringify(
			{"social_title": "Daddy Sausage #shorts", "duration": 46.4, "archetype": "setup_punchline", "source": {"manifest": manifest}}))
	main._handle_line(JSON.stringify({"event": "short", "path": spath, "video_id": "t", "duration": 46.4, "archetype": "setup_punchline", "title": "x"}))
	await process_frame
	rows = main.clip_list.get_children()
	assert(row_buttons.call(rows[0]).has("PLAY EDITED SHORT"), str(row_buttons.call(rows[0])))
	assert(str(row_text.call(rows[0])).contains("EDITED SHORT: Daddy Sausage #shorts  |  46.4s  setup punchline"), str(row_text.call(rows[0])))
	assert(row_buttons.call(rows[1]).has("MAKE SHORT"), "other clip still has no short")
	# a finished short gets Post to YouTube / Post to TikTok; a platform that has it shows a disabled tick
	assert(row_buttons.call(rows[0]).has("Post to YouTube") and row_buttons.call(rows[0]).has("Post to TikTok"), str(row_buttons.call(rows[0])))
	assert(not row_buttons.call(rows[1]).has("Post to YouTube"), "no short, no post buttons")
	var one: PackedStringArray = main.publish_args(true, "tiktok", "t_1")
	assert(one[one.find("--platform") + 1] == "tiktok" and one[one.find("--short") + 1] == "t_1" and one.has("--force"), str(one))
	assert(not main.publish_args(true).has("--short"))
	FileAccess.open(main.history_path, FileAccess.WRITE).store_string(JSON.stringify({"posts": [
			{"short_id": "t_1", "platform": "youtube", "posted_at": 1, "status": "UPLOADED"}]}))
	main.refresh_clips()
	await process_frame
	rows = main.clip_list.get_children()
	var yt_btn: Button = rows[0].get_child(0).get_child(rows[0].get_child(0).get_child_count() - 1).get_children().filter(
			func(b: Node) -> bool: return (b as Button).text.begins_with("On YouTube"))[0]
	assert(yt_btn.disabled and yt_btn.text == "On YouTube ✓" and row_buttons.call(rows[0]).has("Post to TikTok"), str(row_buttons.call(rows[0])))
	assert(str(row_text.call(rows[0])).contains("POSTED TO YOUTUBE: Daddy Sausage"), str(row_text.call(rows[0])))
	DirAccess.remove_absolute(main.history_path)
	main.refresh_clips()
	await process_frame
	rows = main.clip_list.get_children()

	# filters: EDITED SHORTS ONLY, LAST RUN (no run yet: the newest manifest; after a START: manifests since then)
	main.clip_filter.select(1)
	main.show_clips()
	assert(main.filtered_entries().size() == 1 and main.clip_count.text == "CLIPS: 1 of 3", main.clip_count.text)
	main.clip_filter.select(2)
	assert(main.filtered_entries().map(func(e: Dictionary) -> String: return e.data.video_id) == ["t", "t"], "last run before any START")
	main._last_run_unix = int(main._entries[2].mtime)
	assert(main.filtered_entries().size() == 3, "a run that started before both manifests")
	main._last_run_unix = int(main._entries[0].mtime) + 1
	assert(main.filtered_entries().is_empty(), "nothing written since the last START")
	# an empty LAST RUN falls back to ALL CLIPS (with a console note) instead of showing an empty list
	main.show_clips()
	assert(main.clip_filter.selected == 0 and main.filtered_entries().size() == 3, "fell back to ALL CLIPS")
	assert(main.log_view.get_parsed_text().contains("[FILTER] LAST RUN has no clips"), "fallback logged")
	main._last_run_unix = 0
	main.clip_filter.select(0)
	main.refresh_clips()

	# CLEAR hides what the list shows (saved; files untouched), later clips still show up, CLEARED + RESTORE undo it
	assert(main.clear_button.text == "CLEAR" and not main.clear_button.disabled)
	main.clear_clips()
	assert(main.filtered_entries().is_empty() and main.clip_count.text == "CLIPS: 0  ·  3 cleared" and main.clear_button.disabled, main.clip_count.text)
	assert(FileAccess.file_exists(dir.path_join("clip_01_90s.mp4")) and FileAccess.file_exists(old_dir.path_join("clip_01_10s.mp4")), "nothing deleted")
	assert((JSON.parse_string(FileAccess.get_file_as_string(manifest)).clips as Array).size() == 2, "manifests untouched")
	var saved_keys: Array = JSON.parse_string(FileAccess.get_file_as_string(main.settings_path)).cleared_clips
	# a run ending on an all-cleared list says why it is empty
	main._start(PackedStringArray(["-u", "-c", "print(1)"]))
	var cw := 0
	while main._thread != null and cw < 100:
		await create_timer(0.1).timeout
		cw += 1
	assert(main.log_view.get_parsed_text().contains("[FILTER] The list is empty because all 3 clip(s) are cleared"), "cleared hint")
	assert(saved_keys.size() == 3 and saved_keys.has("t#1") and saved_keys.has("t#2") and saved_keys.has("old#1"), str(saved_keys))
	var new_root := OS.get_user_data_dir().path_join("test_clips_new")
	DirAccess.make_dir_recursive_absolute(new_root.path_join("fresh_job"))
	FileAccess.open(new_root.path_join("fresh_job/clips_manifest.json"), FileAccess.WRITE).store_string(
			JSON.stringify({"video_id": "fresh", "clips": [{"clip_id": 1, "file": "clip_01_5s.mp4"}]}))
	main.clip_roots = PackedStringArray([out_root, ws_root, new_root])
	main.load_settings()  # restart: the cleared list comes back from settings.json
	main.refresh_clips()
	assert(main.filtered_entries().map(func(e: Dictionary) -> String: return e.data.video_id) == ["fresh"], "a clip after the clear shows")
	assert(main.clip_count.text == "CLIPS: 1  ·  3 cleared", main.clip_count.text)
	main.clip_filter.select(3)
	main.show_clips()
	assert(main.filtered_entries().size() == 3 and main.clear_button.text == "RESTORE" and main.clip_count.text == "CLIPS: 3 cleared", main.clip_count.text)
	main.clear_clips()  # RESTORE
	assert((main.settings.cleared_clips as Array).is_empty() and main.filtered_entries().is_empty(), "restored")
	main.clip_filter.select(0)
	DirAccess.remove_absolute(new_root.path_join("fresh_job/clips_manifest.json"))
	main.clip_roots = PackedStringArray([out_root, ws_root])
	main.refresh_clips()
	assert(main.clip_count.text == "CLIPS: 3", main.clip_count.text)
	main._preview_path = "C:/x/ready_to_post/t_1_short.mp4"
	main._handle_line(JSON.stringify({"event": "short", "path": "C:" + char(92) + "other" + char(92) + "t_2_short.mp4",
			"duration": 21.5, "archetype": "instant_fail", "title": "lol"}))  # Windows path with backslashes
	assert(main._preview_path != "" and main.log_view.get_parsed_text().contains("SHORT READY (21.5s, instant_fail): lol"), "other short logged, preview still waiting")
	main._preview_path = ""
	await process_frame
	rows = main.clip_list.get_children()  # the reloads above replaced the row nodes

	# Delete Reject: a 5 s grace period with UNDO DELETE on the dimmed row; Shift+click (immediate) skips it
	var e0: Dictionary = main._entries[0]
	var t_del := Time.get_ticks_msec()
	main.request_delete(e0, false, false, t_del)
	await process_frame
	var pending_row: Node = main.clip_list.get_child(0)
	assert(row_buttons.call(pending_row).has("UNDO DELETE") and (pending_row as Control).modulate.a < 1.0, str(row_buttons.call(pending_row)))
	assert(FileAccess.file_exists(dir.path_join("clip_01_90s.mp4")) and main.log_view.get_parsed_text().contains("Deleting clip_01_90s.mp4 in 5 s"))
	main.flush_deletes(t_del + 4_999)
	assert(FileAccess.file_exists(dir.path_join("clip_01_90s.mp4")), "still inside the grace period")
	main.undo_delete(e0)
	main.flush_deletes(t_del + 99_999)
	assert(FileAccess.file_exists(dir.path_join("clip_01_90s.mp4")) and main._pending_deletes.is_empty(), "UNDO kept it")
	await process_frame
	assert(row_buttons.call(main.clip_list.get_child(0)).has("Delete Reject"), "row back to normal")
	main.request_delete(e0, true)  # Shift+click
	await process_frame
	assert(main.clip_list.get_child_count() == 2 and main.clip_count.text == "CLIPS: 2", "row removed")
	assert(not FileAccess.file_exists(dir.path_join("clip_01_90s.mp4")), "clip trashed")
	var saved: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(manifest))
	assert((saved.clips as Array).size() == 1 and int(saved.clips[0].clip_id) == 2, "manifest updated")
	assert(main._clock(3725.0) == "01:02:05")

	# --- built-in preview player: a fake decoder (python writing raw RGB frames) stands in for ffmpeg ----------------
	var frames_py := "import sys; f = bytes([200, 30, 10]) * %d; [sys.stdout.buffer.write(f) for _ in range(%d)]"
	var py_exe: String = main._python()
	main.pv_command = func(_path: String, _start: float, size: Vector2i) -> Array:
		return [py_exe, ["-c", frames_py % [size.x * size.y, 45]]]  # 45 frames = 1.5 s at 30 fps
	var gone := func(pid: int) -> bool:  # a killed process can take a moment to disappear
		for i in 40:
			if not OS.is_process_running(pid):
				return true
			await create_timer(0.05).timeout
		return false
	main.open_preview("C:/fake/clip_01_10s.mp4")
	assert(main.preview.visible and main.pv_state == "playing" and main._pv_pid > 0 and main.pv_size == Vector2i(568, 320), main.pv_state)
	assert(main._pv_audio_pid == -1, "headless: no ffmpeg audio extraction")
	var first_pid: int = main._pv_pid
	var fw := 0
	while not main._pv_image.get_pixel(0, 0).is_equal_approx(Color8(200, 30, 10)) and fw < 100:
		await create_timer(0.05).timeout
		fw += 1
	assert(main._pv_image.get_pixel(0, 0).is_equal_approx(Color8(200, 30, 10)), "frames from the pipe reach the texture")
	assert(main.pv_view.texture == main._pv_texture, "texture shown")
	main.toggle_preview()
	var p1: float = main.preview_position()
	await create_timer(0.2).timeout
	assert(main.pv_state == "paused" and main.preview_position() == p1 and main.pv_play.text == "PLAY", "pause freezes the clock")
	main.seek_preview(1.0)
	assert(await gone.call(first_pid), "seek kills the old decoder")
	assert(main._pv_pid > 0 and main._pv_pid != first_pid and main._pv_pos == 1.0 and main.pv_state == "paused", "seek restarts paused at 1.0")
	assert(main.pv_time.text == "0:01 / 0:00", main.pv_time.text)
	main.toggle_preview()
	var ew := 0
	while main.pv_state != "ended" and ew < 100:
		await create_timer(0.05).timeout
		ew += 1
	assert(main.pv_state == "ended" and main.preview_position() >= 2.4, "plays to the end: %s %.2f" % [main.pv_state, main.preview_position()])
	main.toggle_preview()  # Space at the end: from the top
	assert(main.pv_state == "playing" and main._pv_pos == 0.0, "replay from 0")
	var esc := InputEventKey.new()
	esc.keycode = KEY_ESCAPE
	esc.pressed = true
	var last_pid: int = main._pv_pid
	main._input(esc)
	assert(main.pv_state == "closed" and not main.preview.visible and main._pv_pid == -1 and main._pv_thread == null, "Esc closes")
	assert(main._pv_image == null and main.pv_view.texture == null and main._pv_frames.is_empty(), "frame buffers freed")
	assert(await gone.call(last_pid) and not FileAccess.file_exists(main._pv_wav()), "decoder killed, no temp WAV left")
	# keys: Space pauses, Right seeks +3 s; shorts get the 9:16 frame size
	main.open_preview("C:/fake/t_1_short.mp4", {}, 30.0)
	assert(main.pv_size == Vector2i(320, 568), "short = 9:16")
	var key := func(code: Key) -> void:
		var k := InputEventKey.new()
		k.keycode = code
		k.pressed = true
		main._input(k)
	key.call(KEY_SPACE)
	assert(main.pv_state == "paused")
	var before: float = main.preview_position()
	key.call(KEY_RIGHT)
	assert(is_equal_approx(main._pv_pos, before + 3.0), "Right = +3 s")
	key.call(KEY_LEFT)
	key.call(KEY_LEFT)
	assert(main._pv_pos == 0.0, "Left clamps at 0")
	# 20 clips in a row: every decoder is gone afterwards (memory stays flat)
	var pids: Array[int] = []
	for i in 20:
		main.open_preview("C:/fake/clip_%02d.mp4" % i)
		pids.append(main._pv_pid)
	main.close_preview()
	for pid: int in pids:
		assert(await gone.call(pid), "decoder %d left running" % pid)
	# Delete Reject inside the player: the clip, its manifest entry and its edited short go
	var old_e: Dictionary = main._entries.filter(func(e: Dictionary) -> bool: return e.data.video_id == "old")[0]
	var old_short: String = main.short_path(old_e)
	FileAccess.open(old_short, FileAccess.WRITE).store_string("x")
	FileAccess.open(old_short.trim_suffix("_short.mp4") + "_post.json", FileAccess.WRITE).store_string("{}")
	var n_before: int = main._entries.size()
	main.open_preview(main._clip_file(old_e), old_e)
	main.reject_previewed()
	assert(main.pv_state == "closed" and main._pending_deletes.has(main.clip_key(old_e)) and FileAccess.file_exists(old_short), "pending, undoable")
	main.flush_deletes(Time.get_ticks_msec() + 5_001)
	assert(main._entries.size() == n_before - 1, "rejected from the player once the grace period is over")
	assert(not FileAccess.file_exists(main._clip_file(old_e)) and not FileAccess.file_exists(old_short), "clip + short trashed")
	assert(not FileAccess.file_exists(old_short.trim_suffix("_short.mp4") + "_post.json"), "post json trashed")
	assert((JSON.parse_string(FileAccess.get_file_as_string(old_e.path)).clips as Array).is_empty(), "old manifest updated")
	# isolation: a stalled or broken preview decoder during a real worker run never touches the run
	main._start(PackedStringArray(["-u", "-c", "import time, json; print(json.dumps({'event': 'stage', 'stage': 'download', 'msg': 'x'}), flush=True); time.sleep(3)"]))
	var worker_pid: int = main._pid
	var worker_thread: Thread = main._thread
	await create_timer(0.3).timeout
	var led_before: String = main.status_label.text
	var banner_before: String = main.banner_label.text if main.banner.visible else ""  # the Ollama poll may own it
	main.pv_command = func(_path: String, _start: float, _size: Vector2i) -> Array:
		return [py_exe, ["-c", "import time; time.sleep(30)"]]  # alive, never a picture
	main.open_preview("C:/fake/stalled.mp4")
	var stalled: int = main._pv_pid
	main.preview_tick(main._pv_start_ms + 4_000)
	assert(main.pv_state == "playing", "4 s without a picture: still waiting")
	main.preview_tick(main._pv_start_ms + 5_001)
	assert(main.pv_state == "closed" and not main.preview.visible and main._pv_pid == -1, "the watchdog closed the stalled preview")
	assert(main.log_view.get_parsed_text().contains("[PREVIEW] Preview unavailable: no picture within 5 s"), "non-fatal console line")
	assert(await gone.call(stalled), "stalled decoder killed")
	main.pv_command = func(_path: String, _start: float, _size: Vector2i) -> Array:
		return [py_exe, ["-c", "import sys; sys.exit(1)"]]  # a broken file: exits without a frame
	main.open_preview("C:/fake/broken.mp4")
	var bw := 0
	while main.pv_state != "closed" and bw < 60:
		await create_timer(0.05).timeout
		bw += 1
	assert(main.pv_state == "closed" and main.log_view.get_parsed_text().contains("the decoder stopped without a picture"), main.pv_state)
	assert(main._pid == worker_pid and main._thread == worker_thread and OS.is_process_running(worker_pid), "the worker never noticed")
	assert(main.status_state == "RUNNING" and main.status_label.text == led_before, "LED untouched: " + main.status_label.text)
	assert((main.banner_label.text if main.banner.visible else "") == banner_before and not main.banner_label.text.contains("PREVIEW"), "banner untouched")
	assert(main.run_button.disabled and main.run_button.text == "RUNNING...", "the run still owns START")
	var ww := 0
	while main._thread != null and ww < 100:
		await create_timer(0.1).timeout
		ww += 1
	assert(main._thread == null and main.status_state == "DONE", "the worker finished normally: " + main.status_label.text)
	main.pv_command = func(_path: String, _start: float, size: Vector2i) -> Array:
		return [py_exe, ["-c", frames_py % [size.x * size.y, 45]]]

	# --- status LED: IDLE -> RUNNING -> STUCK (watchdog) -> RUNNING -> COOLDOWN -> ERROR / BOT BLOCKED -> DONE -> IDLE ---
	main.on_run_ended(0)  # earlier steps fed "stage"/"error" lines with no run active
	main._reset_led()
	assert(main.status_state == "IDLE" and main.status_label.text == "IDLE", main.status_label.text)
	var led_color: Color = (main.status_dot.get_theme_stylebox("panel") as StyleBoxFlat).bg_color
	assert(led_color == Color("5c636a"), "idle grey dot")
	main._pid = OS.get_process_id()  # a live pid stands in for the backend child process
	main.begin_run()
	assert(main.status_label.text == "RUNNING: STARTING")
	main._handle_line('{"event": "stage", "stage": "download", "msg": "Downloading video (<=1080p)"}')
	assert(main.status_state == "RUNNING" and main.status_label.text == "RUNNING: DOWNLOAD", main.status_label.text)
	main._handle_line('{"event": "stage", "stage": "scoring", "msg": "Whisper small.en on 12 candidate windows"}')
	assert(main.status_label.text == "RUNNING: WHISPER", main.status_label.text)
	main._handle_line('{"event": "stage", "stage": "render", "msg": "Rendering 123_1"}')
	assert(main.status_label.text == "RUNNING: EDITING", main.status_label.text)
	var t0: int = main._last_heartbeat_ms
	main.watchdog(t0 + 89_000)
	assert(main.status_state == "RUNNING", "89 s of silence is fine")
	main.watchdog(t0 + 91_000)
	assert(main.status_state == "STUCK" and main.status_label.text == "STUCK (>90S NO OUTPUT)", main.status_label.text)
	assert(main.log_view.get_parsed_text().contains("[WATCHDOG] No backend output for 90s"), "watchdog logged")
	assert((main.status_dot.get_theme_stylebox("panel") as StyleBoxFlat).bg_color == Color("f59e0b"), "amber")
	main._handle_line('{"event": "log", "msg": "Whisper batch: 24.0s"}')
	assert(main.status_state == "RUNNING", "output resumed -> RUNNING again")
	main._handle_line('{"event": "queue", "msg": "Cooldown 1.5s", "pending": 3, "done": 1, "failed": 0}')  # Twitch job
	assert(main.status_label.text == "COOLDOWN 1.5S" and main._quiet_until_ms == main._last_heartbeat_ms + 1500, main.status_label.text)
	main._handle_line('{"event": "queue", "msg": "Cooldown 120s", "pending": 3, "done": 1, "failed": 0}')
	assert(main.status_label.text == "COOLDOWN 120S", main.status_label.text)
	t0 = main._last_heartbeat_ms
	main.watchdog(t0 + 150_000)
	assert(main.status_state == "RUNNING", "silence inside an announced cooldown is not STUCK")
	main.watchdog(t0 + 120_000 + 91_000)
	assert(main.status_state == "STUCK", "90 s past the cooldown end is")
	main._handle_line('{"event": "error", "msg": "ffmpeg render failed"}')
	assert(main.status_state == "ERROR" and main.status_label.text == "ERROR", main.status_label.text)
	main._handle_line(JSON.stringify({"event": "error", "msg": "YouTube's bot check blocked this IP (too many automated downloads)."}))
	assert(main.status_label.text == "BOT BLOCKED", main.status_label.text)
	assert(main.banner.visible and main.banner_label.text.begins_with(
			"YouTube Bot Block: Export a fresh cookies.txt from Brave into the project root and click START.\nYouTube's bot check"), main.banner_label.text)
	main._pid = -1
	main.watchdog(t0 + 999_999)
	assert(main.status_state == "ERROR", "no process -> no watchdog")
	main.on_run_ended(0)
	assert(main.status_state == "ERROR", "an error earlier in the run keeps ERROR at the end")
	main.begin_run()
	main._handle_line('{"event": "queue", "msg": "YouTube is blocking requests: backing off 3600s. Refresh cookies.txt or wait."}')
	assert(main.status_label.text == "BOT BLOCKED", "queue back-off -> BOT BLOCKED")
	main.begin_run()
	assert(not main.banner.visible, "a new run clears the old banner")
	main.on_run_ended(1)
	assert(main.status_label.text == "ERROR (EXIT 1)", main.status_label.text)
	assert(main.banner.visible and main.banner_label.text.contains("Backend exited with code 1"), "a silent crash still explains itself")
	main.begin_run()
	main._handle_line('{"event": "stage", "stage": "cutting", "msg": "Cutting clip 1/4"}')
	main.on_run_ended(0)
	assert(main.status_state == "DONE" and main.status_label.text == "DONE", main.status_label.text)
	assert((main.status_dot.get_theme_stylebox("panel") as StyleBoxFlat).bg_color == Color("38bdf8"), "cyan")
	main.url_edit.text = "https://youtu.be/new"
	main.url_edit.text_changed.emit(main.url_edit.text)
	assert(main.status_state == "IDLE", "editing an input resets DONE to IDLE")

	# --- self-start: preflight on a fake project folder, no services in headless, clean exit --------------------------
	var fake_root := OS.get_user_data_dir().path_join("test_boot")
	var win := OS.get_name() == "Windows"
	DirAccess.make_dir_recursive_absolute(fake_root.path_join(".venv/Scripts" if win else ".venv/bin"))
	DirAccess.make_dir_recursive_absolute(fake_root.path_join("workspace"))
	FileAccess.open(fake_root.path_join(".venv/Scripts/python.exe" if win else ".venv/bin/python"), FileAccess.WRITE).store_string("x")
	var envf := FileAccess.open(fake_root.path_join(".env"), FileAccess.WRITE)
	envf.store_buffer(PackedByteArray([0xEF, 0xBB, 0xBF]))  # Notepad's BOM in front of the first key
	envf.store_string("TWITCH_CLIENT_ID=abc123\n# comment\nTWITCH_CLIENT_SECRET = s3cr3t\n")
	envf.close()
	FileAccess.open(fake_root.path_join("workspace/twitch_channels.txt"), FileAccess.WRITE).store_string("xqc\ncaseoh_\njynxzi\n")
	var pf: Dictionary = main.preflight(fake_root)
	assert(pf.python and pf.twitch_keys and pf.streamers == 3, str(pf))
	assert(pf.ffmpeg, "ffmpeg found on PATH")  # this machine has it (C:/Users/Gavin/AppData/Local/ffmpeg/bin)
	assert(main.preflight_line(pf) == "[BOOT] Python .venv OK | ffmpeg OK | Twitch keys OK | 3 streamer(s) saved", main.preflight_line(pf))
	var empty_root := OS.get_user_data_dir().path_join("test_boot_empty")
	DirAccess.make_dir_recursive_absolute(empty_root)
	var pe: Dictionary = main.preflight(empty_root)
	assert(not pe.python and not pe.twitch_keys and pe.streamers == 0, str(pe))
	assert(main.preflight_line(pe).contains("Python .venv MISSING") and main.preflight_line(pe).contains("Twitch keys MISSING"))
	FileAccess.open(fake_root.path_join(".env"), FileAccess.WRITE).store_string("TWITCH_CLIENT_ID=abc\nTWITCH_CLIENT_SECRET=\n")
	assert(not main.preflight(fake_root).twitch_keys, "an empty secret is not a key")
	assert(main.log_view.get_parsed_text().contains("[BOOT] Python .venv"), "boot preflight logged at startup")
	assert(main._started_ollama_pid == -1, "headless runs never start Ollama")
	# exit only stops what the panel started: a real stand-in process plays the started Ollama
	var stand_in := OS.create_process(main._python(), ["-c", "import time; time.sleep(60)"])
	assert(stand_in > 0 and OS.is_process_running(stand_in), "stand-in started")
	main._started_ollama_pid = stand_in
	main.stop_started_services()
	var kwait := 0.0
	while OS.is_process_running(stand_in) and kwait < 5.0:
		await create_timer(0.1).timeout
		kwait += 0.1
	assert(not OS.is_process_running(stand_in) and main._started_ollama_pid == -1, "panel-started service stopped on exit")
	main.stop_started_services()  # nothing started: a no-op, never touches a user-run Ollama

	# --- telemetry: timers, task resets, step history, progress ----------------------------------------------------
	assert(main.format_timers(42_000, 140_000) == "Task: 42s   Overall: 2:20", main.format_timers(42_000, 140_000))
	assert(main.format_timers(14_000, 140_000, 34_000, 255_000) == "Task: 14s (ETA ~34s)   Overall: 2:20 (ETA ~4:15)")
	assert(main.fmt_dur(3_725_000) == "1:02:05" and main.fmt_dur(999) == "0s")
	assert(main.eta_key("Download video") == "Download" and main.eta_key("Whisper Transcribe") == "Whisper"
			and main.eta_key("Rendering short") == "Render" and main.eta_key("Fetch metadata") == "")
	# learned step times persist: saved when a run ends, loaded at startup; a broken file falls back to the seeds
	main.eta_ms = main.ETA_SEED.duplicate()
	main.eta_ms["Render"] = 31_000
	main.eta_ms["AI Judge"] = 74_000
	main.save_eta()
	main.eta_ms = main.ETA_SEED.duplicate()
	main.load_eta()
	assert(main.eta_ms["Render"] == 31_000 and main.eta_ms["AI Judge"] == 74_000 and main.eta_ms["Download"] == 25_000, str(main.eta_ms))
	FileAccess.open(main.eta_path, FileAccess.WRITE).store_string('{"Render": -5, "Whisper": "x", "Download": 41000}')
	main.load_eta()
	assert(main.eta_ms["Render"] == 48_000 and main.eta_ms["Whisper"] == 8_000 and main.eta_ms["Download"] == 41_000, str(main.eta_ms))
	FileAccess.open(main.eta_path, FileAccess.WRITE).store_string("{broken")
	main.load_eta()
	assert(main.eta_ms == main.ETA_SEED, "broken file: seeds")
	main.eta_ms = main.ETA_SEED.duplicate()  # the LED tests above fed real-time steps into the averages
	var tt0 := 1_000_000
	main.telemetry_begin(tt0)
	assert(main.task_label.text == "Starting" and main.progress_bar.value == 0.0)
	main.track_task({"event": "queue", "msg": "Twitch: 4 new clip(s) queued", "pending": 4, "done": 10, "failed": 1}, tt0 + 1_400)
	main.track_task({"event": "queue", "msg": "Processing twitch_x (attempt 1)", "pending": 3, "done": 10, "failed": 1}, tt0 + 1_500)
	assert(main.task_label.text == "Video 1/4", main.task_label.text)
	main.track_task({"event": "stage", "stage": "metadata", "msg": "Fetching metadata"}, tt0 + 1_600)
	main.track_task({"event": "stage", "stage": "scoring", "msg": "Whisper small.en on the clip"}, tt0 + 5_000)
	main.track_task({"event": "stage", "stage": "scoring", "msg": "Whisper small.en on the clip"}, tt0 + 6_000)
	assert(main._task_start_ms == tt0 + 5_000, "same step again: task timer not reset")
	main.track_task({"event": "stage", "stage": "scoring", "msg": "Ollama comedy judge on 1 candidates"}, tt0 + 11_200)
	assert(main.task_label.text == "AI Judge" and main._task_start_ms == tt0 + 11_200, "new step: task timer reset")
	assert(main.eta_ms.Whisper == 7_460, "Whisper average: 0.7 * 8s seed + 0.3 * 6.2s measured")
	main.telemetry_tick(tt0 + 14_200)
	# AI Judge 3s in: 2s of its 5s left; then 4 shorts * 48s; then 3 pending videos * (25 + 7.46 + 5 + 192 + 32 cooldown)
	assert(main.timer_label.text == "Task: 3s (ETA ~2s)   Overall: 14s (ETA ~16:18)", main.timer_label.text)
	main.telemetry_tick(tt0 + 53_200)
	assert(main.timer_label.text == "Task: 42s (ETA ~0s)   Overall: 53s (ETA ~16:16)", main.timer_label.text)
	main._twitch_run = true  # Twitch items: 1 short each, 1.5 s cooldown -> 2s + 48s + 3 * (25 + 7.46 + 5 + 48 + 1.5)
	main.telemetry_tick(tt0 + 14_200)
	assert(main.timer_label.text == "Task: 3s (ETA ~2s)   Overall: 14s (ETA ~5:10)", main.timer_label.text)
	main._twitch_run = false
	main.telemetry_tick(tt0 + 53_200)  # back to the moment the history checks below expect
	var hist: Array = main.history_box.get_children().map(func(l: Node) -> String: return (l as Label).text)
	assert(hist.has("✓ Twitch Poll (4 clips) — 1.4s") and hist.has("✓ Whisper Transcribe — 6.2s"), str(hist))
	assert(hist.back() == "▶ AI Judge — 42s", str(hist))
	assert(is_equal_approx(main.progress_bar.value, 12.5), str(main.progress_bar.value))  # video 1 of 4, half way through it
	main.track_task({"event": "stage", "stage": "edit", "msg": "Editing short 1/2"}, tt0 + 60_000)
	main.track_task({"event": "log", "msg": "Facecam: detected at (0.0, 0.13, 0.21, 0.28)"}, tt0 + 61_000)
	assert(main.task_label.text == "Editing short 1/2 [split_stack]", main.task_label.text)
	assert(is_equal_approx(main.progress_bar.value, 18.75), str(main.progress_bar.value))
	main.track_task({"event": "queue", "msg": "Processing twitch_y (attempt 1)", "pending": 2}, tt0 + 80_000)
	assert(main.task_label.text == "Video 2/4" and main.progress_bar.value >= 25.0, str(main.progress_bar.value))
	assert(main._queue_counts.done == 10 and main._queue_counts.failed == 1, str(main._queue_counts))
	main.telemetry_end(true, tt0 + 90_000)
	var saved_eta: Variant = JSON.parse_string(FileAccess.get_file_as_string(main.eta_path))
	assert(saved_eta is Dictionary and int(saved_eta.Whisper) == main.eta_ms.Whisper and int(saved_eta.Whisper) != 8_000, "run end saved the learned times")
	assert(main.progress_bar.value == 100.0 and main.task_label.text == "Done")
	main.telemetry_tick(tt0 + 500_000)
	assert(main.timer_label.text == "Task: 0s (ETA ~0s)   Overall: 1:30 (ETA ~0s)", "timers + ETAs freeze when the run ends: " + main.timer_label.text)

	# --- telemetry: hardware gauges + session stats ---------------------------------------------------------------
	assert(main.gauge_color(69.9) == Color("5ad06a") and main.gauge_color(70) == Color("e0a030")
			and main.gauge_color(88) == Color("e0a030") and main.gauge_color(88.1) == Color("ef4444"), "thresholds")
	var gpu: Dictionary = main.parse_nvidia_smi("37, 2345, 4096, 61")
	assert(gpu.util == 37.0 and gpu.used == 2345.0 and gpu.total == 4096.0 and gpu.temp == 61.0, str(gpu))
	assert(main.parse_wmic_load("\r\n\r\nLoadPercentage=12\r\n\r\n") == 12.0)
	assert(main.parse_wmic_load("LoadPercentage=10\r\nLoadPercentage=30\r\n") == 20.0, "two sockets averaged")
	var prev_cpu: Array = []
	assert(main.cpu_from_proc_stat("cpu  100 0 100 800 0 0 0 0 0 0", prev_cpu) == 0.0)
	assert(absf(main.cpu_from_proc_stat("cpu  200 0 200 900 0 0 0 0 0 0", prev_cpu) - 66.667) < 0.01, "busy 200 of 300 ticks")
	main.apply_telemetry({"gpu": gpu, "cpu": 95.0, "queue": {"pending": 2, "done": 12}})
	var gbar: ProgressBar = main.gauges["GPU"].bar
	var cbar: ProgressBar = main.gauges["CPU"].bar
	var rbar: ProgressBar = main.gauges["RAM"].bar
	assert(gbar.value == 37.0 and (gbar.get_theme_stylebox("fill") as StyleBoxFlat).bg_color == Color("5ad06a"), "GPU 37% green")
	assert((cbar.get_theme_stylebox("fill") as StyleBoxFlat).bg_color == Color("ef4444"), "CPU 95% red")
	assert((main.gauges["VRAM"].label as Label).text == "VRAM 2345 / 4096 MB" and (main.gauges["GPU"].label as Label).text == "GPU 37%   61°C")
	assert(rbar.value > 0 and (main.gauges["RAM"].label as Label).text.ends_with("GB"), "RAM from OS.get_memory_info")
	assert(main.disk_label.text.begins_with("Disk free: ") and main.disk_label.text.ends_with("GB"), main.disk_label.text)
	assert(main.stats_label.text.contains("Queue: 2 pending · 12 done · 0 failed") and main.stats_label.text.contains("($0.00)"), main.stats_label.text)
	assert(main.stats_label.text.contains("Encoder: NVENC (H.264)"), "NVIDIA GPU seen -> NVENC")
	main.track_task({"event": "log", "msg": "Encoder: libx264"})
	main.refresh_stats()
	assert(main.stats_label.text.contains("Encoder: CPU libx264"), "the editor's own report wins")
	main.apply_telemetry({})
	assert((main.gauges["GPU"].label as Label).text == "GPU: N/A", "no nvidia-smi, no VAAPI")
	var tok: String = main.twitch_token_status()
	assert(tok.begins_with("Valid — ") or tok.begins_with("Expired") or tok == "none yet", tok)

	# --- repeatable runs: a real backend process twice in a row, everything reset after each -------------------------
	assert(main.build_args("https://youtu.be/x").has("--auto-edit"), "RUN CLIPPER auto-edits when the box is ticked")
	var py := "import json; print(json.dumps({'event': 'stage', 'stage': 'download', 'msg': 'x'})); print(json.dumps({'event': 'log', 'msg': 'ok'}))"
	for attempt in 2:
		main._start(PackedStringArray(["-u", "-c", py]))
		assert(main._thread != null and main.run_button.disabled, "START disabled during run %d" % attempt)
		var rwait := 0.0
		while main._thread != null and rwait < 20.0:
			await create_timer(0.1).timeout
			rwait += 0.1
		assert(main._thread == null and main._pid == -1, "run %d finished and joined" % attempt)
		assert(not main.run_button.disabled and main.run_button.text == "START", "START re-enabled after run %d" % attempt)
		assert(main.status_state == "DONE", "LED DONE after run %d: %s" % [attempt, main.status_label.text])
	main._start(PackedStringArray(["-u", "-c", "import time; time.sleep(1.0)"]))
	main.set_mode(2)
	main.start()  # clicked while busy: reported, never silently dropped
	main.set_mode(0)
	assert(main.log_view.get_parsed_text().contains("START: a run is still in progress"), "busy START logged")
	var bwait := 0.0
	while main._thread != null and bwait < 20.0:
		await create_timer(0.1).timeout
		bwait += 0.1
	assert(main._thread == null and not main.run_button.disabled, "free again after the busy run")
	print("test_main ok")
	quit(0)

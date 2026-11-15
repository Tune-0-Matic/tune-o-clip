extends SceneTree
## Headless check: godot --headless --path . -s res://tests/test_main.gd
## Feed + rating flow, Rated-tab edit/remove, Send-to-Claude bookkeeping, worker event handling, safe ratings file.

var failures := 0


func check(ok: bool, what: String) -> void:
	if not ok:
		failures += 1
		printerr("FAIL: " + what)


func card(n: int) -> Dictionary:
	return {"key": "b/%d_1" % n, "batch": "b", "id": "%d_1" % n, "streamer": "caseoh_", "title": "clip %d" % n,
			"short": "C:/nowhere/%d_1_short.mp4" % n, "thumb": "", "transcript": "he lost it",
			"clipper": {"pillars": {"hook_strength": 6, "comedy_payoff": 8}, "judge_10": 7.0, "final_score": 0.6, "duration": 20.0},
			"editor": {"scenario": "chat_roast", "title": "Big Fail #shorts #lol", "qc": {"duration": 18.5}}}


func _initialize() -> void:
	var dir := OS.get_user_data_dir().path_join("test_ratings")
	DirAccess.make_dir_recursive_absolute(dir)
	var path := dir.path_join("ratings_queue.json")
	DirAccess.remove_absolute(path)
	var main: Control = load("res://scripts/main.gd").new()
	main.autostart = false
	main.ratings_path = path
	root.add_child(main)
	await process_frame  # let _ready build the UI
	main.player.command = func(_p: String, _s: float, _z: Vector2i) -> Array: return ["cmd", ["/c", "exit"]]  # starts, prints no frames

	for n in 3:
		main.handle_line(JSON.stringify(dict_with(card(n), "event", "short_ready")))
	check(main.feed.size() == 3 and main.current_key() == "b/0_1", "3 shorts in the feed, first shown")
	check(main.info_title.text == "Big Fail" and main.info_badge.text == "chat_roast", "AI title without hashtags + badge")
	check(main.pillar_bars["comedy_payoff"].value == 8, "pillar bars")
	main.handle_line(JSON.stringify(dict_with(card(1), "event", "short_ready")))  # a duplicate key replaces, never adds
	check(main.feed.size() == 3 and main.current_key() == "b/0_1", "duplicate short_ready ignored")

	check(not main.save_rating(), "nothing picked: save refused")
	main.go(1)
	check(main.current_key() == "b/1_1", "down = next")
	main.set_stars(4)
	main.set_like(true)
	main.toggle_tag("Banger / Keep")
	main.toggle_tag("Bad Cut / Pacing")
	main.toggle_tag("Bad Cut / Pacing")
	main.liked_box.text = "  the scream  "
	main.disliked_box.text = "cut earlier"
	main.extra_box.text = "more caseoh"
	check(main.save_rating(), "rating saved")
	check(main.feed.size() == 2 and main.current_key() == "b/2_1", "rated short left the feed, the next one slid in")
	var saved: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(path))
	var r: Dictionary = saved.get("b/1_1", {})
	check(r.get("stars") == 4.0 and r.get("like") == true and r.get("tags") == ["Banger / Keep"] and r.get("liked") == "the scream" and r.get("disliked") == "cut earlier"
			and r.get("extra_notes") == "more caseoh"
			and r.get("status") == "rated", "ratings_queue.json entry: %s" % r)
	check(main.rated_keys() == ["b/1_1"] and main.rated_list.get_child_count() == 1, "Rated tab lists it")
	check(main.cur_stars == 0 and main.cur_like == null and main.liked_box.text == "" and main.disliked_box.text == "" and main.extra_box.text == "", "form cleared for the next short")

	main.edit_rating("b/1_1")
	check(main.current_key() == "b/1_1" and main.cur_stars == 4 and main.cur_like == true and main.cur_tags == ["Banger / Keep"]
			and main.liked_box.text == "the scream" and main.disliked_box.text == "cut earlier" and main.extra_box.text == "more caseoh",
			"Edit: back in the scroller with the rating pre-filled")
	main.set_stars(5)
	main.save_rating()
	check(int(main.ratings["b/1_1"].stars) == 5, "edited rating saved")
	main.save_rating()  # current is now b/2_1 with an empty form
	main.set_like(false)
	main.save_rating()
	main.remove_rating("b/2_1")
	main.drafts["b/0_1"] = {"stars": 2, "notes": "old style note"}  # a rating saved before the three boxes
	main.refresh_feed("b/0_1")
	main.show_card()  # re-render the panel with the draft
	check(main.extra_box.text == "old style note" and main.liked_box.text == "", "old 'notes' loads into Extra Notes")
	main.extra_box.grab_focus()
	var esc := InputEventKey.new()
	esc.keycode = KEY_ESCAPE
	esc.pressed = true
	main._input(esc)
	check(main.get_viewport().gui_get_focus_owner() == null, "Esc leaves the text box")
	var five := InputEventKey.new()
	five.keycode = KEY_5
	five.pressed = true
	main._input(five)
	check(main.cur_stars == 5, "hotkeys work right after Esc")
	check(main.feed.any(func(c: Dictionary) -> bool: return c.key == "b/2_1") and not main.ratings.has("b/2_1"), "Remove: unrated again")

	main.on_sent({"event": "sent", "keys": ["b/1_1"], "md": "x.md", "prompt": "Read x.md"})
	check(main.ratings["b/1_1"].status == "trained" and main.rated_keys().is_empty(), "sent ratings archived as trained")
	check(main.feed.size() == 2, "trained clips stay out of the feed")

	check(main.stage_of({"event": "stage", "stage": "download", "msg": "Downloading Twitch clip"}) == 0, "stage: download")
	check(main.stage_of({"event": "stage", "stage": "scoring", "msg": "Whisper small.en on the clip"}) == 1, "stage: whisper")
	check(main.stage_of({"event": "stage", "stage": "scoring", "msg": "Ollama comedy judge"}) == 2, "stage: judge")
	check(main.stage_of({"event": "stage", "stage": "edit", "msg": "Directing with qwen"}) == 2, "stage: director")
	check(main.stage_of({"event": "stage", "stage": "render", "msg": "Rendering 1/1"}) == 3, "stage: render")
	main.handle_event({"event": "cr_progress", "stage": "Short ready", "completed": 1, "total": 4, "current": "x"})
	check(is_equal_approx(main.progress.value, 25.0), "progress bar")
	main.handle_line("not json at all")  # logged, no crash

	main.custom_streamers.text = "twitch.tv/Kai_Cenat/, @xQc, bad name!, caseoh_"
	var logins: PackedStringArray = main.selected_streamers()
	check(logins == PackedStringArray(["caseoh_", "ishowspeed", "jynxzi", "lospollostv", "kai_cenat", "xqc"]), "streamers: %s" % logins)
	main.url_edit.text = "--evil"
	main.upload_url()
	check(main._uploads.is_empty() and main._pid < 0, "non-URL refused before reaching argv")

	# an unreadable ratings file is kept aside, never overwritten
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_string("{broken")
	f.close()
	main.load_ratings()
	check(main.ratings.is_empty() and not FileAccess.file_exists(path), "broken file moved aside")
	var kept := Array(DirAccess.get_files_at(dir)).filter(func(n: String) -> bool: return n.contains(".unreadable-"))
	check(kept.size() >= 1, "unreadable copy kept")
	for n: String in DirAccess.get_files_at(dir):
		DirAccess.remove_absolute(dir.path_join(n))

	# the app lives in tune-o-clip/app: the worker is found one level up (else a tune-o-clip folder beside it)
	var here := ProjectSettings.globalize_path("res://")
	var toc: String = main.find_toc_root(here)
	if FileAccess.file_exists(here.path_join("../calibration_studio/clip_rate_worker.py").simplify_path()):
		check(toc == here.path_join("..").simplify_path(), "toc_root = the repo: " + toc)
	else:
		check(toc.ends_with("/tune-o-clip"), "toc_root = sibling tune-o-clip: " + toc)
	check(main.counts_label.text.contains("unrated") and main.counts_label.text.contains("GB free"), main.counts_label.text)
	check(main.bot_label.text.begins_with("Bot:"), main.bot_label.text)
	main.queue_free()
	print("clip rate: %s" % ("all tests passed" if failures == 0 else "%d FAILED" % failures))
	quit(1 if failures else 0)


func dict_with(d: Dictionary, k: String, v: Variant) -> Dictionary:
	d[k] = v
	return d

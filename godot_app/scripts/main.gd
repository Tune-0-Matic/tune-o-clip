extends Control
## Tune-O-Clip control panel: runs backend/clipper.py, streams its JSON lines into a log, lists the clips.

const AMBER := Color("e0a030")
const DIM := Color("8a8f94")
const BAD := Color("e05a4a")
const GOOD := Color("5ad06a")
const OLLAMA_URL := "http://127.0.0.1:11434/api/tags"
const LED_COLORS := {"IDLE": Color("5c636a"), "RUNNING": Color("3ddc84"), "STUCK": Color("f59e0b"),
		"ERROR": Color("ef4444"), "DONE": Color("38bdf8")}
const STUCK_MS := 90_000  # no stdout line for this long (outside a known sleep) = probably hung

const SOURCE_MODES := ["Single YouTube URL", "Batch URLs", "Twitch Streamers"]
const OLLAMA_MODEL := "richardyoung/qwen2.5-3b-instruct-abliterated"
const DEFAULT_SETTINGS := {
	"clips": 4, "min_length": 15, "max_length": 55, "min_score": 6.0, "cleanup_source": false,
	"twitch_streamers": "", "twitch_game": "", "twitch_min_views": 1500, "twitch_hours": 24, "twitch_limit": 20, "twitch_per_streamer": 0, "twitch_lang": "en",
	"auto_edit": true, "facecam_box": "auto", "caption_theme": "Auto by Scenario", "music_gain_db": -24.0,
	"cookies": "OFF", "whisper_model": "small.en", "ollama_model": OLLAMA_MODEL,
	"post_platform": "YouTube Shorts", "auto_post": false, "post_interval_h": 3.5, "youtube_privacy": "public",
	"post_privacy": "Public",  # TikTok's (key kept from before YouTube support so saved settings still load)
	"cleared_clips": [],  # clip_key()s hidden by CLEAR (no widget; files are never touched)
}
## [category, [[key, label, extra, tooltip]]]: extra = [lo, hi, step] for numbers, an Array of options for choices,
## a placeholder for text, null for checkboxes (the widget type follows DEFAULT_SETTINGS[key]).
const SETTINGS_UI := [
	["CLIPPER & QUALITY GATE", [
		["clips", "Clips per Video", [1, 20, 1], ""],
		["min_length", "Min Length (s)", [5, 180, 1], ""],
		["max_length", "Max Length (s)", [5, 180, 1], ""],
		["min_score", "Min Comedy Score", [0, 10, 0.5], "Banger gate: clips the Ollama judge rates below this /10 are skipped before cutting/rendering (0 = off; no effect while Ollama is offline)"],
		["cleanup_source", "Cleanup Source", null, "Single runs: delete the full downloaded video once the clips are cut (the queue always does)"]]],
	["TWITCH POLLING", [
		["twitch_streamers", "Streamer Logins", "xqc, kai_cenat, caseoh_, jynxzi", "Comma separated; @name or twitch.tv/name links work too. Synced to workspace/twitch_channels.txt."],
		["twitch_game", "Game Filter", "blank = any game", "e.g. Grand Theft Auto V. With no streamers, polls the game's top clips across Twitch."],
		["twitch_min_views", "Min Views", [0, 1000000, 100], ""],
		["twitch_hours", "Lookback Hours", [1, 720, 1], ""],
		["twitch_limit", "Max Clips / Poll", [1, 100, 1], "Total clips one poll may queue"],
		["twitch_per_streamer", "Clips per Streamer", [0, 50, 1],
				"How many clips each streamer gets per poll. 0 = Max Clips / Poll split evenly. The total stays capped by Max Clips / Poll."],
		["twitch_lang", "Language", "en (blank = any)", ""]]],
	["AI & SYSTEM", [
		["cookies", "YouTube Cookies", ["OFF", "cookies.txt", "brave", "chrome", "edge", "firefox"],
				"OFF still auto-uses a cookies.txt in the project folder. A browser must be closed first (it locks its cookie file)."],
		["whisper_model", "Whisper Model", ["small.en", "base.en"], "base.en is faster on CPU, small.en is more accurate"],
		["ollama_model", "Ollama Model", OLLAMA_MODEL, ""]]],
]
## The AUTO POST drawer (button next to SETTINGS): everything about finished shorts, same format as SETTINGS_UI.
const AUTO_POST_UI := [
	["EXPORT (EDITOR & CAPTIONS)", [
		["auto_edit", "Auto-Edit Shorts", null, "Every run: turn the clips into finished 9:16 shorts in ready_to_post/"],
		["facecam_box", "Facecam Box", "auto or x,y,w,h", "auto = detect the webcam overlay; or fractions 0..1, e.g. 0,0.13,0.21,0.28"],
		["caption_theme", "Caption Theme", ["Auto by Scenario", "Yellow", "Neon Green", "Red", "Cyan"], ""],
		["music_gain_db", "Music Gain (dB)", [-40, 0, 1], "Background music level (only when assets/music has a track)"]]],
	["PUBLISHER", [
		["post_platform", "Publish Platform", ["YouTube Shorts", "TikTok", "Both"],
				"Where POST BEST NOW and Auto-Post publish (official APIs). Both: each platform keeps its own interval."],
		["auto_post", "Auto-Post Best Short", null,
				"Post the best unposted short whenever Post Interval has passed. Off by default."],
		["post_interval_h", "Post Interval (h)", [1, 24, 0.5], "Hours between posts, per platform (3.5 keeps a small account clear of spam limits)"],
		["youtube_privacy", "YouTube Privacy", ["public", "unlisted", "private"], "Who sees the YouTube Shorts"],
		["post_privacy", "TikTok Privacy", ["Public", "Followers", "Friends", "Only Me"],
				"Who sees the TikTok posts. Until TikTok audits your developer app it can only post to a PRIVATE account."]]],
]

var mode_opt: OptionButton
var url_edit: LineEdit
var twitch_hint: Label
var run_button: Button
var settings_button: Button
var settings_panel: PanelContainer
var auto_post_button: Button
var auto_post_panel: PanelContainer
var settings := {}
var settings_path := ""  # "" = <project>/workspace/settings.json (tests point it elsewhere)
var _widgets := {}  # settings key -> its control
var log_view: RichTextLabel
var clip_list: VBoxContainer
var ollama_badge: Label
var batch_box: HBoxContainer
var batch_edit: TextEdit
var twitch_channels_path := ""
var banner: PanelContainer
var banner_label: Label
var _preview_path := ""
var ready_dir := ""  # "" = <project>/ready_to_post (tests point it elsewhere)

var status_dot: Panel
var status_label: Label
var status_state := "IDLE"
var _last_heartbeat_ms := 0
var _quiet_until_ms := 0  # end of an announced cooldown/back-off: silence until then is expected
var _run_error := false

var _thread: Thread
var _pid := -1
var _ollama_http: HTTPRequest


func _ready() -> void:
	theme = _make_theme()
	var bg := Panel.new()
	bg.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(bg)
	var margin := MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	for side: String in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 12)
	add_child(margin)
	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 6)
	margin.add_child(root)

	# launch bar: source mode + input + START, SETTINGS drawer toggle, Ollama badge, status light
	var bar := HFlowContainer.new()  # wraps onto a second line instead of pushing buttons off a narrow window
	bar.add_theme_constant_override("h_separation", 8)
	bar.add_theme_constant_override("v_separation", 6)
	root.add_child(bar)
	mode_opt = OptionButton.new()
	for m: String in SOURCE_MODES:
		mode_opt.add_item(m)
	mode_opt.item_selected.connect(set_mode)
	bar.add_child(mode_opt)
	url_edit = LineEdit.new()
	url_edit.placeholder_text = "https://www.youtube.com/watch?v=..."
	url_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	url_edit.custom_minimum_size.x = 150  # never squeezed to nothing by the status widgets
	url_edit.text_submitted.connect(func(_t: String) -> void: start())
	bar.add_child(url_edit)
	twitch_hint = _label("Polls the streamers / game set in SETTINGS > TWITCH POLLING")
	twitch_hint.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	twitch_hint.add_theme_color_override("font_color", DIM)
	bar.add_child(twitch_hint)
	run_button = Button.new()
	run_button.text = "START"
	run_button.pressed.connect(start)
	bar.add_child(run_button)
	settings_button = Button.new()
	settings_button.text = "SETTINGS"
	settings_button.toggle_mode = true
	settings_button.toggled.connect(func(on: bool) -> void: settings_panel.visible = on)
	bar.add_child(settings_button)
	auto_post_button = Button.new()
	auto_post_button.text = "AUTO POST"
	auto_post_button.toggle_mode = true
	auto_post_button.tooltip_text = "Export (editing) settings, YouTube / TikTok posting, logins and POST BEST NOW"
	auto_post_button.toggled.connect(func(on: bool) -> void: auto_post_panel.visible = on)
	bar.add_child(auto_post_button)
	var drawers := ButtonGroup.new()  # one drawer at a time: CLIPS keeps its height
	drawers.allow_unpress = true
	settings_button.button_group = drawers
	auto_post_button.button_group = drawers
	ollama_badge = _label("OLLAMA: ...")
	ollama_badge.add_theme_color_override("font_color", DIM)
	ollama_badge.tooltip_text = "Local Ollama comedy judge (polled every 5 s)"
	ollama_badge.mouse_filter = Control.MOUSE_FILTER_PASS
	bar.add_child(ollama_badge)
	bar.add_child(_make_led())

	twitch_channels_path = _project_root().path_join("workspace/twitch_channels.txt")
	load_settings()
	load_eta()
	settings_panel = _make_settings(SETTINGS_UI)
	settings_panel.visible = false
	root.add_child(settings_panel)
	auto_post_panel = _make_settings(AUTO_POST_UI, _make_post_actions())
	auto_post_panel.visible = false
	root.add_child(auto_post_panel)

	# batch mode: one link per line -> backend/queue_worker.py --add-file ... --drain (dedup, cooldowns, cleanup)
	batch_box = HBoxContainer.new()
	root.add_child(batch_box)
	batch_edit = TextEdit.new()
	batch_edit.placeholder_text = "One YouTube link per line: videos, playlists or channels. Already-processed videos are skipped."
	batch_edit.custom_minimum_size.y = 56  # 2 lines; it scrolls
	batch_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	batch_box.add_child(batch_edit)
	set_mode(0)

	# workspace: console + clips on the left, the 320 px telemetry sidebar on the right
	var main_row := HBoxContainer.new()
	main_row.size_flags_vertical = Control.SIZE_EXPAND_FILL
	main_row.add_theme_constant_override("separation", 10)
	root.add_child(main_row)
	# console above, clips below (draggable split); the clips keep >= 220 px however much is open above them
	var left := VSplitContainer.new()
	left.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	main_row.add_child(left)
	var side_scroll := ScrollContainer.new()  # the sidebar scrolls instead of forcing the window taller
	side_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	side_scroll.add_child(_make_telemetry())
	main_row.add_child(side_scroll)

	var console := VBoxContainer.new()
	console.size_flags_vertical = Control.SIZE_EXPAND_FILL
	console.size_flags_stretch_ratio = 0.6
	left.add_child(console)
	console.add_child(_label("CONSOLE"))
	log_view = RichTextLabel.new()
	log_view.bbcode_enabled = true
	log_view.scroll_following = true
	log_view.selection_enabled = true
	log_view.custom_minimum_size.y = 44
	log_view.size_flags_vertical = Control.SIZE_EXPAND_FILL
	console.add_child(log_view)

	var clips_box := VBoxContainer.new()
	clips_box.size_flags_vertical = Control.SIZE_EXPAND_FILL
	left.add_child(clips_box)
	clips_box.add_child(_make_clips_header())
	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size.y = 220
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	clips_box.add_child(scroll)
	clip_list = VBoxContainer.new()
	clip_list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	clip_list.add_theme_constant_override("separation", 6)
	scroll.add_child(clip_list)
	root.add_child(_make_banner())
	root.add_child(_make_error_panel())  # the very bottom: always there, lists every problem
	_log("Ready. Backend: %s" % _clipper_path(), DIM)
	for edit: Control in [url_edit, batch_edit]:
		edit.connect("text_changed", _reset_led)
	refresh_clips()

	_ollama_http = HTTPRequest.new()
	_ollama_http.timeout = 2.0
	_ollama_http.request_completed.connect(
			func(result: int, code: int, _h: PackedStringArray, _b: PackedByteArray) -> void:
				set_ollama_status(result == HTTPRequest.RESULT_SUCCESS and code == 200))
	add_child(_ollama_http)
	_poll_timer = Timer.new()
	_poll_timer.wait_time = 5.0
	_poll_timer.autostart = true
	_poll_timer.timeout.connect(func() -> void:
		poll_ollama()
		refresh_systems()  # .venv / ffmpeg / .env can be fixed while the panel is open
		if auto_post_due(Time.get_unix_time_from_system()):
			_auto_post_try = Time.get_unix_time_from_system()
			_start(publish_args(false)))
	add_child(_poll_timer)
	poll_ollama()
	boot()
	start_telemetry()
	add_child(_make_preview())  # last child: the overlay draws over everything


# --- self-starting: preflight + Ollama auto-start + model preload ------------------------------------------------

var _started_ollama_pid := -1  # only an Ollama this panel started is stopped on exit


## What the backend needs, checked without starting anything. root: the project folder (tests pass a fake one).
func preflight(root: String = "") -> Dictionary:
	root = _project_root() if root.is_empty() else root
	var win := OS.get_name() == "Windows"
	var python := FileAccess.file_exists(root.path_join(".venv/Scripts/python.exe" if win else ".venv/bin/python"))
	var ffmpeg := false
	for dir: String in OS.get_environment("PATH").split(";" if win else ":", false):
		if FileAccess.file_exists(dir.path_join("ffmpeg.exe" if win else "ffmpeg")):
			ffmpeg = true
			break
	var env := FileAccess.get_file_as_string(root.path_join(".env")).trim_prefix(char(0xFEFF))  # Notepad BOM
	var keys := 0
	for line: String in env.split("\n"):
		var kv := line.strip_edges().split("=", true, 1)
		if kv.size() == 2 and kv[0].strip_edges() in ["TWITCH_CLIENT_ID", "TWITCH_CLIENT_SECRET"] and not kv[1].strip_edges().is_empty():
			keys += 1
	var streamers := parse_logins(FileAccess.get_file_as_string(root.path_join("workspace/twitch_channels.txt"))).size()
	return {"python": python, "ffmpeg": ffmpeg, "twitch_keys": keys == 2, "streamers": streamers}


func preflight_line(p: Dictionary) -> String:
	var mark := func(ok: bool) -> String: return "OK" if ok else "MISSING"
	return "[BOOT] Python .venv %s | ffmpeg %s | Twitch keys %s | %d streamer(s) saved" % [
			mark.call(p.python), mark.call(p.ffmpeg), mark.call(p.twitch_keys), int(p.streamers)]


## Startup: log the preflight, start Ollama (Vulkan backend) if it's off, then preload the judge model into VRAM.
## Headless runs (tests) never start external services.
func boot() -> void:
	var p := refresh_systems()
	_log(preflight_line(p), GOOD if p.python and p.ffmpeg else BAD)
	if not (p.python and p.ffmpeg):
		show_banner("Python .venv missing" if not p.python else "ffmpeg not found on PATH")
	if DisplayServer.get_name() == "headless":
		return
	var online := await _ollama_reachable()
	if not online:
		start_ollama()
		for i in 40:  # first start can take ~30 s (GPU discovery)
			await get_tree().create_timer(1.0).timeout
			online = await _ollama_reachable()
			if online:
				break
	if not online:
		_log("[BOOT] Ollama did not come up: clips are scored without the AI judge.", AMBER)
		show_banner("Ollama did not start: clips are scored without the AI judge")
		return
	set_ollama_status(true)
	_log("[BOOT] Ollama online%s; preloading %s..." % [" (started by the panel)" if _started_ollama_pid > 0 else "", OLLAMA_MODEL], DIM)
	var http := HTTPRequest.new()
	http.timeout = 300.0
	add_child(http)
	var t0 := Time.get_ticks_msec()
	http.request_completed.connect(func(result: int, code: int, _h: PackedStringArray, _b: PackedByteArray) -> void:
		_log("[BOOT] AI judge model ready in VRAM (%.0fs)" % ((Time.get_ticks_msec() - t0) / 1000.0) if result == HTTPRequest.RESULT_SUCCESS and code == 200
				else "[BOOT] Model preload failed (result %d, HTTP %d); it will load on first use." % [result, code], DIM)
		http.queue_free())
	http.request("http://127.0.0.1:11434/api/generate", PackedStringArray(["Content-Type: application/json"]),
			HTTPClient.METHOD_POST, JSON.stringify({"model": OLLAMA_MODEL, "prompt": "", "keep_alive": "30m"}))


func _ollama_reachable() -> bool:
	var http := HTTPRequest.new()
	http.timeout = 2.0
	add_child(http)
	if http.request(OLLAMA_URL) != OK:
		http.queue_free()
		return false
	var res: Array = await http.request_completed
	http.queue_free()
	return res[0] == HTTPRequest.RESULT_SUCCESS and res[1] == 200


func start_ollama() -> void:
	if OS.get_name() == "Linux" and FileAccess.file_exists("/etc/systemd/system/ollama.service"):
		# setup_linux.sh installs Ollama as a system service and its models live with it: start that service through
		# the desktop's password prompt (pkexec), never a user `ollama serve` (it would have no models)
		OS.create_process("pkexec", ["systemctl", "start", "ollama"])
		_log("[BOOT] Ollama was off: starting the ollama service (enter your password in the prompt).", DIM)
		return
	if OS.get_name() == "Windows":
		# this PC's i7-860 (no AVX) + older NVIDIA driver only run Ollama on the GPU through Vulkan
		OS.set_environment("OLLAMA_LLM_LIBRARY", "vulkan")
	var exe := OS.get_environment("LOCALAPPDATA").path_join("Programs/Ollama/ollama.exe")
	if not FileAccess.file_exists(exe):
		exe = "ollama"  # on PATH (Linux, or a custom install)
	_started_ollama_pid = OS.create_process(exe, ["serve"])
	_log(("[BOOT] Ollama was off: started it" + (" (Vulkan)." if OS.get_name() == "Windows" else ".")) if _started_ollama_pid > 0
			else "[BOOT] Could not start Ollama (%s)." % exe,
			DIM if _started_ollama_pid > 0 else AMBER)


## Exit: stop only what this panel started (an Ollama you run yourself keeps running).
func stop_started_services() -> void:
	if _started_ollama_pid > 0 and OS.is_process_running(_started_ollama_pid):
		OS.kill(_started_ollama_pid)
	_started_ollama_pid = -1


## Pings the local Ollama server; the badge updates when the request completes (or fails).
func poll_ollama() -> void:
	if _ollama_http.get_http_client_status() == HTTPClient.STATUS_DISCONNECTED:
		if _ollama_http.request(OLLAMA_URL) != OK:
			set_ollama_status(false)


func set_ollama_status(online: bool) -> void:
	if not online and ollama_badge.text.ends_with("ONLINE"):  # only the drop, not every 5 s poll
		show_banner("Ollama went offline (http://127.0.0.1:11434 not answering)")
	ollama_badge.text = "OLLAMA: ONLINE" if online else "OLLAMA: OFFLINE"
	ollama_badge.add_theme_color_override("font_color", GOOD if online else AMBER)
	_systems.ollama = online
	update_start_button()


# --- START gate: nothing runs until every system it needs is online ------------------------------------------------

var _systems := {"python": false, "ffmpeg": false, "ollama": false, "twitch_keys": false}
var _poll_timer: Timer


## Re-checks .venv / ffmpeg / Twitch keys (cheap: file + PATH lookups) and updates START. -> the preflight dict.
func refresh_systems() -> Dictionary:
	var p := preflight()
	for k: String in ["python", "ffmpeg", "twitch_keys"]:
		_systems[k] = p[k]
	update_start_button()
	return p


## What START still waits for (Twitch keys only matter in Twitch Streamers mode). Empty = ready.
func missing_systems() -> PackedStringArray:
	var out := PackedStringArray()
	for k: String in ["python", "ffmpeg", "ollama"]:
		if not _systems[k]:
			out.append({"python": "Python .venv", "ffmpeg": "ffmpeg", "ollama": "Ollama"}[k])
	if mode_opt.selected == 2 and not _systems.twitch_keys:
		out.append("Twitch keys")
	return out


func update_start_button() -> void:
	if run_button == null or _thread != null:
		return  # a running job owns the button until it ends
	var missing := missing_systems()
	run_button.disabled = not missing.is_empty()
	run_button.text = "START" if missing.is_empty() else "WAITING: " + ", ".join(missing).to_upper()
	run_button.tooltip_text = ("Every system is online." if missing.is_empty() else
			"START unlocks by itself once these are online: %s. The [BOOT] line in the console and the ERRORS panel say what's wrong." % ", ".join(missing))


# --- YouTube Shorts + TikTok publisher (official APIs; publisher/) ------------------------------------------------

const PLATFORM := {"YouTube Shorts": "youtube", "TikTok": "tiktok", "Both": "both"}
const PLATFORM_LABEL := {"youtube": "YOUTUBE", "tiktok": "TIKTOK"}
const PRIVACY := {"Public": "PUBLIC_TO_EVERYONE", "Followers": "FOLLOWER_OF_CREATOR", "Friends": "MUTUAL_FOLLOW_FRIENDS", "Only Me": "SELF_ONLY"}
const POSTED_STATUSES := ["PUBLISH_COMPLETE", "PROCESSING", "UPLOADED"]  # as publisher/scheduler.py DONE
const AUTO_POST_RETRY_S := 600.0  # after a failed / skipped auto-post, wait this long before trying again
var login_button: Button  # TikTok
var yt_login_button: Button
var post_button: Button
var history_path := ""  # "" = <project>/workspace/posted_history.json (tests point it elsewhere)
var _auto_post_try := -1e12
var _posted_by := {}  # short id -> ["youtube", "tiktok"], rebuilt by refresh_clips


func _platforms() -> Array:
	var p: String = PLATFORM.get(settings.post_platform, "youtube")
	return ["youtube", "tiktok"] if p == "both" else [p]


## platform "" = AUTO POST > Publish Platform; short_id "" = the scheduler picks the best short.
func publish_args(force: bool, platform := "", short_id := "") -> PackedStringArray:
	var args := PackedStringArray(["-u", _project_root().path_join("publisher/scheduler.py"),
			"--platform", platform if not platform.is_empty() else PLATFORM.get(settings.post_platform, "youtube"),
			"--interval", str(float(settings.post_interval_h)),
			"--youtube-privacy", str(settings.youtube_privacy), "--tiktok-privacy", PRIVACY.get(settings.post_privacy, "PUBLIC_TO_EVERYONE")])
	if force:
		args.append("--force")
	if not short_id.is_empty():
		args.append_array(["--short", short_id])
	return args


func publish_now() -> void:
	if not _busy("POST BEST NOW"):
		_start(publish_args(true))


## A clip row's Post to YouTube / Post to TikTok button.
func post_short(short_id: String, platform: String) -> void:
	if not _busy("Post to " + platform):
		_start(publish_args(true, platform, short_id))


func login_tiktok() -> void:
	if not _busy("Login TikTok"):
		_start(PackedStringArray(["-u", _project_root().path_join("publisher/tiktok_poster.py"), "--login"]))


func login_youtube() -> void:
	if not _busy("Login YouTube"):
		_start(PackedStringArray(["-u", _project_root().path_join("publisher/youtube_poster.py"), "--login"]))


## Successful posts in posted_history.json; entries from before YouTube support (no "platform") are TikTok's.
func history_posts() -> Array:
	var path := history_path if not history_path.is_empty() else _project_root().path_join("workspace/posted_history.json")
	var json := JSON.new()
	if not FileAccess.file_exists(path) or json.parse(FileAccess.get_file_as_string(path)) != OK or not json.data is Dictionary:
		return []
	var posts: Variant = (json.data as Dictionary).get("posts", [])
	if not posts is Array:
		return []
	return (posts as Array).filter(func(p: Variant) -> bool: return p is Dictionary and str(p.get("status", "")) in POSTED_STATUSES)


func posted_by() -> Dictionary:
	var out := {}
	for p: Dictionary in history_posts():
		var pf := str(p.get("platform", "tiktok"))
		var have: Array = out.get(str(p.get("short_id", "")), [])
		if not pf in have:
			have.append(pf)
		out[str(p.get("short_id", ""))] = have
	return out


## Auto-Post: on, nothing running, a short waiting in ready_to_post/, some enabled platform's interval since its last
## post over, and no try in the last 10 min (so a failing post doesn't loop). The scheduler re-checks everything itself.
func auto_post_due(now_unix: float) -> bool:
	if not settings.auto_post or _thread != null or now_unix - _auto_post_try < AUTO_POST_RETRY_S:
		return false
	if not Array(DirAccess.get_files_at(_ready_dir())).any(func(f: String) -> bool: return f.ends_with("_short.mp4")):
		return false
	var posts := history_posts()
	for pf: String in _platforms():  # each platform keeps its own interval
		var last := 0.0
		for p: Dictionary in posts:
			if str(p.get("platform", "tiktok")) == pf:
				last = maxf(last, float(p.get("posted_at", 0)))
		if now_unix - last >= float(settings.post_interval_h) * 3600.0:
			return true
	return false


# --- running the backend -------------------------------------------------------------------------

## START: runs whatever the source mode says, once every system it needs is online.
func start() -> void:
	if _busy("START"):
		return
	var missing := missing_systems()
	if not missing.is_empty():  # Enter in the URL box bypasses the disabled button
		_log("START: waiting for %s to come online." % ", ".join(missing), AMBER)
		return
	[_run, run_batch, poll_twitch][mode_opt.selected].call()


## Shows the input the source mode needs: URL field, batch paste box, or a hint (Twitch reads SETTINGS).
func set_mode(i: int) -> void:
	mode_opt.select(i)
	url_edit.visible = i == 0
	batch_box.visible = i == 1
	twitch_hint.visible = i == 2
	_reset_led()
	update_start_button()  # Twitch mode also needs the Twitch keys


func _run() -> void:
	var url := url_edit.text.strip_edges()
	if url.is_empty():
		_log("START: paste a YouTube link first.", BAD)
	elif not _busy("START"):
		_twitch_run = false
		_start(build_args(url))


## Writes the pasted links to a file and drains them through the queue worker (same live console).
func run_batch() -> void:
	if batch_edit.text.strip_edges().is_empty():
		_log("START: paste some links first.", BAD)
	elif not _busy("START"):
		_twitch_run = false
		_start(build_batch_args(batch_edit.text))


func poll_twitch() -> void:
	if _busy("START"):
		return
	var args := build_twitch_args(str(settings.twitch_streamers), str(settings.twitch_game))
	if args.is_empty():
		_log("START: set streamer logins and/or a game in SETTINGS > TWITCH POLLING.", BAD)
	else:
		_twitch_run = true
		_start(args)


## True (and says so) while a backend run is still going: a click is never silently dropped.
func _busy(what: String) -> bool:
	if _thread != null:
		_log("%s: a run is still in progress (see the status light); try again when it shows DONE." % what, AMBER)
	return _thread != null


func _set_busy(on: bool) -> void:
	run_button.disabled = on
	run_button.text = "RUNNING..." if on else "START"
	if not on:
		update_start_button()  # back to START only if everything is still online


func _start(args: PackedStringArray) -> void:
	# blocking pipes: non-blocking ones read EOF instantly. Linux: setsid gives the run its own session (same pid,
	# setsid execs in place), so closing the panel can stop its clipper / ffmpeg / yt-dlp children too.
	var proc := OS.execute_with_pipe("setsid", PackedStringArray([_python()]) + args) if OS.get_name() == "Linux" \
			else OS.execute_with_pipe(_python(), args)
	if proc.is_empty():
		_log("Could not start Python (%s)." % _python(), BAD)
		show_banner("Could not start Python (%s)" % _python())
		return
	_pid = int(proc.pid)
	begin_run()
	_set_busy(true)
	_log("> %s %s" % [args[1].get_file(), " ".join(args.slice(2))], AMBER)
	_thread = Thread.new()
	_thread.start(_read_output.bind(proc))


func build_args(url: String) -> PackedStringArray:
	var args := PackedStringArray(["-u", _clipper_path(), url])
	args.append_array(_clipper_options())
	if settings.cleanup_source:
		args.append("--cleanup-source")
	args.append_array(_queue_options())  # --auto-edit: single runs become shorts too
	return args


func build_batch_args(links: String) -> PackedStringArray:
	var path := OS.get_user_data_dir().path_join("batch_urls.txt")
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_string(links.strip_edges() + "
")
	f.close()
	# the worker always cleans up sources and archives to output/<video id>/; clipper options pass through
	var args := PackedStringArray(["-u", _project_root().path_join("backend/queue_worker.py"), "--add-file", path, "--drain"])
	args.append_array(_queue_options())
	args.append_array(_clipper_options())
	return args


## Streamers given: save them (normalised) to twitch_channels_path and poll those channels, filtered to `game`
## if set. No streamers: poll the game across all of Twitch. Neither: empty args (nothing to do).
func build_twitch_args(streamers: String, game: String) -> PackedStringArray:
	var logins := parse_logins(streamers)
	game = game.strip_edges()
	var args := PackedStringArray(["-u", _project_root().path_join("backend/queue_worker.py")])
	if not logins.is_empty():
		_save_channels(logins)
		args.append_array(["--poll-twitch-channels", twitch_channels_path])
		if not game.is_empty():
			args.append_array(["--twitch-game", game])
	elif not game.is_empty():
		args.append_array(["--poll-twitch-game", game])
	else:
		return PackedStringArray()
	args.append_array(["--twitch-limit", str(int(settings.twitch_limit)), "--twitch-per-streamer", str(int(settings.twitch_per_streamer)),
			"--twitch-hours", str(int(settings.twitch_hours)),
			"--twitch-min-views", str(int(settings.twitch_min_views)), "--twitch-lang=" + str(settings.twitch_lang).strip_edges(), "--drain"])  # "=" form: blank (any language) survives as an argument
	args.append_array(_queue_options())
	args.append_array(_clipper_options())
	return args


func _save_channels(logins: PackedStringArray) -> void:
	DirAccess.make_dir_recursive_absolute(twitch_channels_path.get_base_dir())
	var f := FileAccess.open(twitch_channels_path, FileAccess.WRITE)
	if f:
		f.store_string("\n".join(logins) + "\n")


## "xqc, @Kai_Cenat\ntwitch.tv/caseoh_/clips" -> ["xqc", "kai_cenat", "caseoh_"] (same rules as twitch_api.parse_logins).
static func parse_logins(text: String) -> PackedStringArray:
	var out := PackedStringArray()
	var valid := RegEx.create_from_string("^[a-z0-9_]{1,25}$")
	for raw: String in text.replace(",", " ").replace(";", " ").replace("\n", " ").replace("\t", " ").split(" ", false):
		var s := raw.strip_edges().to_lower()
		var at := s.find("twitch.tv/")
		if at >= 0:
			s = s.substr(at + 10)
		s = s.trim_prefix("@").get_slice("/", 0).get_slice("?", 0)
		if valid.search(s) != null and not out.has(s):
			out.append(s)
	return out


func _queue_options() -> PackedStringArray:
	return PackedStringArray(["--auto-edit"]) if settings.auto_edit else PackedStringArray()


## Editor options (facecam box, caption theme, music gain, models) reach editor/pipeline.py via settings.json.
func _clipper_options() -> PackedStringArray:
	var opts := PackedStringArray(["--clips", str(int(settings.clips)), "--min-length", str(int(settings.min_length)),
			"--max-length", str(int(settings.max_length)), "--min-score", str(float(settings.min_score)),
			"--whisper-model", str(settings.whisper_model), "--ollama-model", str(settings.ollama_model).strip_edges()])
	match str(settings.cookies):
		"OFF":
			pass
		"cookies.txt":
			opts.append_array(["--cookies", _project_root().path_join("cookies.txt")])
		var browser:
			opts.append_array(["--cookies-from-browser", browser])
	return opts


# --- SETTINGS: one dict, auto-saved to workspace/settings.json on every change ---------------------------------------

func _settings_file() -> String:
	return settings_path if not settings_path.is_empty() else _project_root().path_join("workspace/settings.json")


## Defaults, overlaid with the saved file (known keys, coerced to the default's type); streamers come from
## twitch_channels.txt, which the daemon reads too.
func load_settings() -> void:
	settings = DEFAULT_SETTINGS.duplicate()
	var saved: Variant = JSON.parse_string(FileAccess.get_file_as_string(_settings_file())) if FileAccess.file_exists(_settings_file()) else null
	if saved is Dictionary:
		for k: String in settings:
			if (saved as Dictionary).has(k):
				settings[k] = type_convert(saved[k], typeof(DEFAULT_SETTINGS[k]))
	var logins := parse_logins(FileAccess.get_file_as_string(twitch_channels_path))
	if not logins.is_empty():
		settings.twitch_streamers = ", ".join(logins)
	for k: String in _widgets:
		_show_setting(k)


func save_settings() -> void:
	DirAccess.make_dir_recursive_absolute(_settings_file().get_base_dir())
	var f := FileAccess.open(_settings_file(), FileAccess.WRITE)
	if f == null:
		_log("Could not save %s" % _settings_file(), BAD)
		return
	f.store_string(JSON.stringify(settings, "  "))


## A change from a widget (or a test): store, keep min <= max, sync streamers, save.
func set_setting(key: String, value: Variant) -> void:
	settings[key] = type_convert(value, typeof(DEFAULT_SETTINGS[key]))
	if key == "min_length" and settings.max_length < settings.min_length:  # dragging one past the other pushes it along
		settings.max_length = settings.min_length
		_show_setting("max_length")
	elif key == "max_length" and settings.min_length > settings.max_length:
		settings.min_length = settings.max_length
		_show_setting("min_length")
	elif key == "twitch_streamers":
		var logins := parse_logins(str(value))
		if not logins.is_empty():
			_save_channels(logins)
	save_settings()
	_reset_led()


func _show_setting(key: String) -> void:
	var w: Control = _widgets.get(key)
	var v: Variant = settings[key]
	if w is SpinBox:
		(w as SpinBox).set_value_no_signal(v)
	elif w is CheckBox:
		(w as CheckBox).set_pressed_no_signal(v)
	elif w is OptionButton:
		var opt := w as OptionButton
		for i in opt.item_count:
			if opt.get_item_text(i) == str(v):
				opt.select(i)
	elif w is LineEdit and (w as LineEdit).text != str(v):
		(w as LineEdit).text = str(v)


func _make_settings(categories: Array, extra_column: Control = null) -> PanelContainer:
	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", _box(Color("111315"), AMBER.darkened(0.4), 1))
	panel.theme = Theme.new()
	panel.theme.default_font_size = 12  # compact: 4 columns fit 1280 px (font itself comes from the root theme)
	var scroll := ScrollContainer.new()  # capped height: never pushes CLIPS off the window
	scroll.custom_minimum_size.y = 220
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	panel.add_child(scroll)
	var cols := HBoxContainer.new()
	cols.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	cols.add_theme_constant_override("separation", 16)
	scroll.add_child(cols)
	for cat: Array in categories:
		var col := VBoxContainer.new()
		col.size_flags_horizontal = Control.SIZE_EXPAND_FILL  # equal shares; a column only grows past it for the model name
		col.add_child(_header(cat[0]))
		var grid := GridContainer.new()
		grid.columns = 2
		grid.add_theme_constant_override("v_separation", 3)
		col.add_child(grid)
		for item: Array in cat[1]:
			var key: String = item[0]
			var label := _label(item[1])
			label.tooltip_text = item[3]
			label.mouse_filter = Control.MOUSE_FILTER_PASS
			grid.add_child(label)
			var w := _setting_widget(key, item[2])
			w.tooltip_text = item[3]
			# a stretched checkbox draws its pressed style as a full-width bar
			w.size_flags_horizontal = Control.SIZE_SHRINK_BEGIN if w is CheckBox else Control.SIZE_EXPAND_FILL
			if key == "ollama_model":  # wide enough for the whole default model name
				w.custom_minimum_size.x = theme.default_font.get_string_size(OLLAMA_MODEL, HORIZONTAL_ALIGNMENT_LEFT, -1, 12).x + 20
			grid.add_child(w)
			_widgets[key] = w
			_show_setting(key)
		cols.add_child(col)
	if extra_column:
		cols.add_child(extra_column)
	return panel


## AUTO POST's last column: the one-time logins and POST BEST NOW.
func _make_post_actions() -> VBoxContainer:
	var col := VBoxContainer.new()
	col.add_child(_header("ACCOUNTS & ACTIONS"))
	yt_login_button = _login_button("Login YouTube (1-Time)", login_youtube,
			"Opens Google in your browser: approve Tune-O-Clip once. Needs YOUTUBE_CLIENT_ID / SECRET in .env.")
	col.add_child(yt_login_button)
	login_button = _login_button("Login TikTok (1-Time)", login_tiktok,
			"Opens TikTok in your browser: approve Tune-O-Clip once. Needs TIKTOK_CLIENT_KEY / SECRET in .env.")
	col.add_child(login_button)
	post_button = _login_button("POST BEST NOW", publish_now,
			"Post the best unposted short to the Publish Platform now (ignores the interval)")
	post_button.add_theme_color_override("font_color", AMBER)
	col.add_child(post_button)
	return col


func _login_button(text: String, action: Callable, tip: String) -> Button:
	var b := Button.new()
	b.text = text
	b.add_theme_font_size_override("font_size", 12)
	b.tooltip_text = tip
	b.pressed.connect(action)
	return b


func _setting_widget(key: String, extra: Variant) -> Control:
	var default: Variant = DEFAULT_SETTINGS[key]
	if default is bool:
		var c := CheckBox.new()
		c.toggled.connect(func(on: bool) -> void: set_setting(key, on))
		return c
	if default is int or default is float:
		var s := _spin(extra[0], extra[1], default, "")
		s.step = extra[2]
		s.value_changed.connect(func(v: float) -> void: set_setting(key, v))
		return s
	if extra is Array:
		var o := OptionButton.new()
		for opt: String in extra:
			o.add_item(opt)
		o.item_selected.connect(func(i: int) -> void: set_setting(key, o.get_item_text(i)))
		return o
	var e := LineEdit.new()
	e.placeholder_text = extra
	e.text_changed.connect(func(t: String) -> void: set_setting(key, t))
	return e


# Worker thread: blocking reads, handed to the main thread with call_deferred.
# stderr is read after stdout closes; the backend captures all tool output, so stderr stays tiny (tracebacks only).
func _read_output(proc: Dictionary) -> void:
	var stdio: FileAccess = proc.stdio
	var stderr: FileAccess = proc.stderr
	while true:
		var line := stdio.get_line()
		if stdio.get_error() != OK:
			break  # pipe closed: the process exited
		_handle_line.call_deferred(line)
	while true:
		var line := stderr.get_line()
		if stderr.get_error() != OK:
			break
		if not line.strip_edges().is_empty():
			_log.call_deferred(line, DIM)
	_finish.call_deferred()


func _handle_line(line: String) -> void:
	var data: Variant = JSON.parse_string(line) if line.begins_with("{") else null
	if not data is Dictionary:
		if not line.strip_edges().is_empty():
			_log(line, DIM)
		return
	var ev: Dictionary = data
	_heartbeat(ev)
	track_task(ev)
	match str(ev.get("event", "")):
		"stage":
			_log("[%s] %s" % [str(ev.get("stage", "")).to_upper(), str(ev.get("msg", ""))], AMBER)
		"log":
			_log(str(ev.get("msg", "")), DIM)
		"clip":
			_log("Clip %d: %s" % [int(ev.get("clip_id", 0)), str(ev.get("reason", ""))], Color.WHITE)
		"error":
			_log("ERROR: " + str(ev.get("msg", "")), BAD)
			show_banner(str(ev.get("msg", "")))
		"warning":  # nothing failed, but the run did less than asked (0 new clips, unknown streamer, ...)
			_log("WARNING: " + str(ev.get("msg", "")), AMBER)
			show_banner(str(ev.get("msg", "")), true)
		"queue":
			_log("[QUEUE] %s  (pending %d, done %d, failed %d)" % [str(ev.get("msg", "")), int(ev.get("pending", 0)),
					int(ev.get("done", 0)), int(ev.get("failed", 0))], GOOD)
		"done":
			_log("Done -> %s" % str(ev.get("manifest", "")), AMBER)
			refresh_clips()
		"publish":
			if ev.get("dry_run", false):
				_log("[PUBLISH] Would post %s to %s: %s" % [str(ev.get("short_id", "")), str(ev.get("platform", "tiktok")),
						str(ev.get("caption", "")).replace("\n", "  ")], DIM)
			else:
				refresh_clips()  # POSTED TO ... label, maybe moved to posted/ (the scheduler logs the post itself)
		"short":
			var spath := str(ev.get("path", "")).replace("\\", "/")
			_log("SHORT READY (%.1fs, %s): %s" % [float(ev.get("duration", 0)), str(ev.get("archetype", "")),
					str(ev.get("title", ""))], GOOD)
			refresh_clips()  # show the new short on its clip row right away
			if not _preview_path.is_empty() and spath.get_file() == _preview_path.get_file():
				_preview_path = ""
				var hit := _entries.filter(func(e: Dictionary) -> bool: return short_path(e).get_file() == spath.get_file())
				open_preview(spath, hit[0] if not hit.is_empty() else {}, float(ev.get("duration", 0)))


func _finish() -> void:
	_thread.wait_to_finish()
	_thread = null
	var code := OS.get_process_exit_code(_pid)  # -1 = unknown (not an error by itself)
	_pid = -1
	_set_busy(false)
	on_run_ended(code)
	refresh_clips()  # queue runs archive manifests to output/ after their "done" line
	var cleared := _entries.filter(_is_cleared).size()
	if clip_list.get_child_count() == 0 and cleared > 0 and clip_filter.selected != 3:
		_log("[FILTER] The list is empty because all %d clip(s) are cleared: pick CLEARED in the filter to see them, RESTORE to bring them back." % cleared, AMBER)


# --- error & action banner -----------------------------------------------------------------------------

## [regex on the error text, what it is, what to do next]; first match wins, the last one catches everything.
const ERROR_HELP := [
	["(?i)not a bot|bot.?check|blocking|sign in to confirm", "YouTube Bot Block",
			"Export a fresh cookies.txt from Brave into the project root and click START."],
	["^\\[JUDGE\\]", "No Clip Passed the AI Judge",
			"Nothing in this video was funny or self-contained enough. Lower Min Comedy Score in SETTINGS, or try another video / streamer."],
	["(?i)youtube api quota exceeded|quotaExceeded", "YouTube Quota Exceeded",
			"Google allows 100 uploads a day per Cloud project. Nothing to fix: posting resumes after midnight Pacific."],
	["(?i)youtube oauth token expired", "YouTube Token Expired",
			"Click Login YouTube (1-Time) in AUTO POST. If this comes back every 7 days, set your Google OAuth consent screen to In production."],
	["(?i)youtube auth failed|YOUTUBE_CLIENT_ID", "YouTube Auth Failed",
			"Put YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET (Google Cloud > Credentials > OAuth client, Desktop app) in .env, then click Login YouTube (1-Time)."],
	["(?i)tiktok session expired", "TikTok Session Expired",
			"Click Login TikTok (1-Time) in AUTO POST and approve once in the browser."],
	["(?i)unaudited_client", "TikTok App Not Audited Yet",
			"Until TikTok approves your developer app it can only post to a private account: set your TikTok account to private, or submit the app for audit."],
	["(?i)TIKTOK_CLIENT_KEY", "TikTok Keys Missing",
			"Put TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET (from your TikTok developer app) in .env, then click Login TikTok."],
	["(?i)twitch user not found", "Twitch User Not Found",
			"Fix that login in SETTINGS > TWITCH POLLING (it's the name in twitch.tv/<name>), then click START."],
	["(?i)twitch game not found", "Twitch Game Not Found",
			"Use the exact Twitch category name in SETTINGS > TWITCH POLLING > Game Filter (e.g. Grand Theft Auto V)."],
	["(?i)^0 new clips", "No New Twitch Clips",
			"Every clip Twitch has for these streamers is already processed or filtered. Add more streamers or a game in SETTINGS > TWITCH POLLING, or lower Min Views."],
	["(?i)twitch.*(4\\d\\d|auth|token|client|credential)", "Twitch Auth Failed",
			"Check TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET in .env and ensure the Twitch app is Confidential."],
	["(?i)failed qc", "Short Failed QC",
			"The render is kept in ready_to_post/.work. Click MAKE SHORT on the clip to retry, or set a Facecam Box in SETTINGS."],
	["(?i)ollama", "Ollama Offline", "Start Ollama from the Start Menu for AI comedy judging."],
	["(?i)ffmpeg", "ffmpeg Problem", "Make sure ffmpeg is installed and on PATH, then click START."],
	["(?i)python", "Python Not Found", "Create the .venv in the project folder (see README), then click START."],
	[".", "Run Failed", "Read the CONSOLE for details, fix the cause and click START again."],
]


## "<what> (<HTTP code>): <what to do>" for a raw backend error message.
static func explain_error(msg: String) -> String:
	for h: Array in ERROR_HELP:
		if RegEx.create_from_string(h[0]).search(msg):
			var code := RegEx.create_from_string("\\b([45]\\d\\d)\\b").search(msg)
			return "%s%s: %s" % [h[1], " (%s)" % code.get_string(1) if code else "", h[2]]
	return msg


func _make_banner() -> Control:
	banner = PanelContainer.new()
	banner.add_theme_stylebox_override("panel", _box(Color("3a1414"), BAD, 2))
	banner.visible = false
	var row := HBoxContainer.new()
	banner.add_child(row)
	banner_label = _label("")
	banner_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	banner_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	banner_label.add_theme_color_override("font_color", Color("ffd0c8"))
	row.add_child(banner_label)
	var close := Button.new()
	close.text = "×"
	close.tooltip_text = "Dismiss"
	close.pressed.connect(func() -> void: banner.visible = false)
	row.add_child(close)
	return banner


## warn: amber "did less than asked" instead of red "failed".
func show_banner(msg: String, warn: bool = false) -> void:
	var c := AMBER if warn else BAD
	banner.add_theme_stylebox_override("panel", _box(c.darkened(0.75), c, 2))
	banner_label.text = "%s\n%s" % [explain_error(msg), msg.strip_edges().get_slice("\n", 0).left(200)]
	banner.visible = true
	add_error(msg, warn)


# --- ERRORS panel: every problem of this session (the banner only shows the newest) --------------------------------

const MAX_ERRORS := 100
var error_list: VBoxContainer
var error_count: Label
var _errors: Array[Dictionary] = []  # newest first: {"time", "msg", "warn"}


func _make_error_panel() -> Control:
	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", _box(Color("111315"), Color("3a3f44"), 1))
	var col := VBoxContainer.new()
	col.add_theme_constant_override("separation", 2)
	panel.add_child(col)
	var head := HBoxContainer.new()
	col.add_child(head)
	error_count = _header("ERRORS: 0")
	error_count.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	head.add_child(error_count)
	var clear := Button.new()
	clear.text = "CLEAR"
	clear.tooltip_text = "Empty this list (and close the banner)"
	clear.add_theme_font_size_override("font_size", 12)  # compact: this panel lives on every screen
	clear.pressed.connect(clear_errors)
	head.add_child(clear)
	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size.y = 38  # 2 lines; scroll for more
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	col.add_child(scroll)
	error_list = VBoxContainer.new()
	error_list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	error_list.add_theme_constant_override("separation", 0)
	scroll.add_child(error_list)
	_show_errors()
	return panel


func add_error(msg: String, warn: bool = false) -> void:
	_errors.push_front({"time": Time.get_time_string_from_system(), "msg": msg.strip_edges(), "warn": warn})
	if _errors.size() > MAX_ERRORS:
		_errors.pop_back()
	_show_errors()


func clear_errors() -> void:
	_errors.clear()
	banner.visible = false
	_show_errors()


func _show_errors() -> void:
	for c: Node in error_list.get_children():
		c.queue_free()
	var bad := _errors.filter(func(e: Dictionary) -> bool: return not e.warn).size()
	error_count.text = "ERRORS: %d" % bad + ("   WARNINGS: %d" % (_errors.size() - bad) if _errors.size() > bad else "")
	error_count.add_theme_color_override("font_color", BAD if bad else (AMBER if _errors else DIM))
	if _errors.is_empty():
		error_list.add_child(_history_line("No problems this session.", DIM))
	for e: Dictionary in _errors:
		var line := _history_line("[%s]  %s   ·   %s" % [e.time, explain_error(e.msg), str(e.msg).get_slice("\n", 0)],
				AMBER if e.warn else BAD.lightened(0.2))
		line.tooltip_text = e.msg  # the whole message (tracebacks, yt-dlp output) on hover
		line.mouse_filter = Control.MOUSE_FILTER_PASS
		error_list.add_child(line)


# --- status LED + watchdog ---------------------------------------------------------------------------

func _make_led() -> Control:
	var pill := PanelContainer.new()
	pill.add_theme_stylebox_override("panel", _round_box(Color("0e0f10"), Color("3a3f44"), 12, 10))
	pill.tooltip_text = "Backend status. STUCK = the process is alive but silent for 90 s outside a cooldown."
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 7)
	pill.add_child(row)
	status_dot = Panel.new()
	status_dot.custom_minimum_size = Vector2(12, 12)
	status_dot.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	row.add_child(status_dot)
	status_label = _label("")
	row.add_child(status_label)
	set_status("IDLE", "IDLE")
	return pill


func _round_box(fill: Color, border: Color, radius: int, pad: int) -> StyleBoxFlat:
	var b := StyleBoxFlat.new()
	b.bg_color = fill
	b.border_color = border
	b.set_border_width_all(1)
	b.set_corner_radius_all(radius)
	b.content_margin_left = pad
	b.content_margin_right = pad
	b.content_margin_top = 3
	b.content_margin_bottom = 3
	return b


## state: IDLE | RUNNING | STUCK | ERROR | DONE (the colour); text: what the pill says.
func set_status(state: String, text: String) -> void:
	status_state = state
	var c: Color = LED_COLORS[state]
	status_dot.add_theme_stylebox_override("panel", _round_box(c, c.lightened(0.35), 6, 0))  # full circle + rim
	status_dot.modulate.a = 1.0
	status_label.text = text.to_upper()
	status_label.add_theme_color_override("font_color", c.lightened(0.25) if state != "IDLE" else DIM)


func begin_run() -> void:
	_run_error = false
	_quiet_until_ms = 0
	_last_heartbeat_ms = Time.get_ticks_msec()
	banner.visible = false  # a new run: the old problem is stale
	_last_run_unix = int(Time.get_unix_time_from_system())
	set_status("RUNNING", "RUNNING: STARTING")
	telemetry_begin()


## Every stdout JSON line is a heartbeat; it also says what the backend is doing.
func _heartbeat(ev: Dictionary) -> void:
	_last_heartbeat_ms = Time.get_ticks_msec()
	var msg := str(ev.get("msg", ""))
	match str(ev.get("event", "")):
		"error":
			_run_error = true
			var blocked := msg.containsn("not a bot") or msg.containsn("bot check") or msg.containsn("blocked")
			set_status("ERROR", "BOT BLOCKED" if blocked else "ERROR")
		"queue":
			var secs := RegEx.create_from_string("(\\d+(?:\\.\\d+)?)s\\b").search(msg)  # "Cooldown 31s", "Cooldown 1.5s"
			if secs:  # cooldown / idle sleep / back-off: silence is expected until it ends
				_quiet_until_ms = _last_heartbeat_ms + int(float(secs.get_string(1)) * 1000)
			if msg.containsn("blocking"):
				_run_error = true
				set_status("ERROR", "BOT BLOCKED")
				show_banner(msg)
			elif msg.begins_with("Cooldown") and secs:
				set_status("RUNNING", "COOLDOWN %sS" % secs.get_string(1))
			elif msg.begins_with("Queue empty") and secs:
				set_status("RUNNING", "WAITING FOR NEW VIDEOS")
			else:
				set_status("RUNNING", "RUNNING: QUEUE")
		"stage":
			var stage := str(ev.get("stage", "")).to_upper()
			if msg.contains("Whisper"):
				stage = "WHISPER"
			elif msg.contains("Ollama"):
				stage = "AI JUDGE"
			elif stage in ["EDIT", "RENDER"]:
				stage = "EDITING"
			set_status("RUNNING", "RUNNING: " + stage)
		_:
			if status_state == "STUCK":  # output resumed
				set_status("RUNNING", "RUNNING")


func on_run_ended(exit_code: int) -> void:
	telemetry_end(exit_code <= 0 and not _run_error)
	if exit_code > 0 or _run_error:
		if status_state != "ERROR":
			set_status("ERROR", "ERROR (EXIT %d)" % exit_code if exit_code > 0 else "ERROR")
		if not banner.visible:
			show_banner("Backend exited with code %d" % exit_code)
	else:
		set_status("DONE", "DONE")


## DONE/ERROR fall back to IDLE once the inputs change.
func _reset_led(_v: Variant = null) -> void:
	if _thread == null and status_state in ["DONE", "ERROR"]:
		set_status("IDLE", "IDLE")


## Called every frame with the clock; separate from _process so tests can feed it times.
func watchdog(now_ms: int) -> void:
	if _pid == -1 or not OS.is_process_running(_pid):
		return
	if status_state != "STUCK" and now_ms - maxi(_last_heartbeat_ms, _quiet_until_ms) > STUCK_MS:
		set_status("STUCK", "STUCK (>90S NO OUTPUT)")
		_log("[WATCHDOG] No backend output for 90s — process may be stalled", AMBER)


func _process(delta: float) -> void:
	var now := Time.get_ticks_msec()
	watchdog(now)
	if status_state == "RUNNING":
		status_dot.modulate.a = 0.6 + 0.4 * sin(now / 250.0)  # pulse
	elif status_state == "STUCK":
		status_dot.modulate.a = 1.0 if (now / 500) % 2 == 0 else 0.35  # flash
	_tick_accum += delta
	if _tick_accum >= 0.25:
		_tick_accum = 0.0
		telemetry_tick(now)
	preview_tick(now)
	if not _pending_deletes.is_empty():
		flush_deletes(now)


func _exit_tree() -> void:
	flush_deletes(0, true)
	close_preview()
	if _thread != null:  # killing it closes the pipes, which unblocks the reader thread
		if OS.get_name() == "Linux":
			OS.execute("kill", ["-TERM", "--", "-%d" % _pid])  # the run's whole session (see _start)
		else:
			OS.kill(_pid)
		_thread.wait_to_finish()
	stop_telemetry()
	stop_started_services()


# --- telemetry sidebar: active task + timers, step history, hardware gauges, session stats ------------------------

var task_label: Label
var timer_label: Label
var progress_bar: ProgressBar
var history_box: VBoxContainer
var gauges := {}  # name -> {"bar": ProgressBar, "label": Label}
var disk_label: Label
var stats_label: Label
var _overall_start_ms := 0
var _overall_end_ms := -1
var _task_start_ms := 0
var _task_name := ""
var _steps: Array[Dictionary] = []  # finished steps: {"name", "ms"}
var _progress := 0.0  # 0..1 for the whole run
var _batch_index := 0  # queue runs: video n of _batch_total
var _batch_total := 0
var _layout_tag := ""
var _encoder_name := ""
var _whisper_dev := ""
var _gpu_kind := ""
var _queue_counts := {"pending": 0, "done": 0, "failed": 0}
var _tick_accum := 0.0
var _telemetry_thread: Thread
var _telemetry_on := false


func _make_telemetry() -> Control:
	var panel := PanelContainer.new()
	panel.custom_minimum_size.x = 320
	panel.add_theme_stylebox_override("panel", _box(Color("111315"), Color("3a3f44"), 1))
	var col := VBoxContainer.new()
	col.add_theme_constant_override("separation", 6)
	panel.add_child(col)
	col.add_child(_header("ACTIVE TASK"))
	task_label = _label("Idle")
	task_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	task_label.add_theme_color_override("font_color", Color.WHITE)
	col.add_child(task_label)
	timer_label = _label(format_timers(0, 0))
	timer_label.add_theme_color_override("font_color", AMBER)
	col.add_child(timer_label)
	progress_bar = ProgressBar.new()
	progress_bar.custom_minimum_size.y = 18
	progress_bar.add_theme_stylebox_override("background", _box(Color("0e0f10"), Color("3a3f44"), 1))
	progress_bar.add_theme_stylebox_override("fill", _box(AMBER, AMBER, 0))
	col.add_child(progress_bar)
	col.add_child(_header("STEP HISTORY"))
	history_box = VBoxContainer.new()
	history_box.size_flags_vertical = Control.SIZE_EXPAND_FILL
	history_box.add_theme_constant_override("separation", 1)
	col.add_child(history_box)
	col.add_child(_header("HARDWARE"))
	for g: String in ["GPU", "VRAM", "CPU", "RAM"]:
		col.add_child(_gauge(g))
	disk_label = _label("Disk: --")
	col.add_child(disk_label)
	col.add_child(_header("SESSION STATS"))
	stats_label = _label("")
	stats_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	stats_label.add_theme_color_override("font_color", DIM)
	col.add_child(stats_label)
	refresh_stats()
	return panel


func _header(text: String) -> Label:
	var l := _label(text)
	l.add_theme_color_override("font_color", AMBER)
	l.add_theme_font_size_override("font_size", 12)
	return l


func _gauge(name: String) -> Control:
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 1)
	var label := _label("%s: --" % name)
	label.add_theme_font_size_override("font_size", 12)
	var bar := ProgressBar.new()
	bar.show_percentage = false
	bar.custom_minimum_size.y = 8
	bar.add_theme_stylebox_override("background", _box(Color("0e0f10"), Color("2a2d30"), 1))
	box.add_child(label)
	box.add_child(bar)
	gauges[name] = {"bar": bar, "label": label}
	return box


static func gauge_color(pct: float) -> Color:
	return Color("5ad06a") if pct < 70 else (Color("e0a030") if pct <= 88 else Color("ef4444"))


func set_gauge(name: String, pct: float, text: String) -> void:
	var g: Dictionary = gauges[name]
	var bar: ProgressBar = g.bar
	bar.value = clampf(pct, 0, 100)
	var c := gauge_color(pct)
	bar.add_theme_stylebox_override("fill", _box(c, c, 0))
	(g.label as Label).text = text


## 42 s -> "42s", 140 s -> "2:20", 3725 s -> "1:02:05".
static func fmt_dur(ms: int) -> String:
	var s := ms / 1000
	if s < 60:
		return "%ds" % s
	if s < 3600:
		return "%d:%02d" % [s / 60, s % 60]
	return "%d:%02d:%02d" % [s / 3600, (s / 60) % 60, s % 60]


## ETAs < 0 are left out (unknown).
static func format_timers(task_ms: int, overall_ms: int, task_eta: int = -1, overall_eta: int = -1) -> String:
	var eta := func(ms: int) -> String: return " (ETA ~%s)" % fmt_dur(ms) if ms >= 0 else ""
	return "Task: %s%s   Overall: %s%s" % [fmt_dur(task_ms), eta.call(task_eta), fmt_dur(overall_ms), eta.call(overall_eta)]


# --- ETA: moving average per stage (seeded for the GTX 1050 Ti box), updated as steps finish ------------------------

const ETA_ORDER := ["Download", "Whisper", "AI Judge", "Render"]  # one queue item; Render repeats per short
const ETA_SEED := {"Download": 25_000, "Whisper": 8_000, "AI Judge": 5_000, "Render": 48_000, "Cooldown": 32_000}
var eta_ms := ETA_SEED.duplicate()
var eta_path := ""  # "" = <project>/workspace/eta_stats.json (tests point it elsewhere)
var _eta_pos := -1  # furthest ETA_ORDER stage reached in the current item
const TWITCH_COOLDOWN_MS := 1500  # queue_worker --twitch-cooldown default
var _twitch_run := false  # Twitch queue items are whole clips: one short each
var _shorts_left := -1  # from "Editing short k/N": shorts after the current one (-1 = editing not started)


static func eta_key(task: String) -> String:
	for k: String in ["Download", "Whisper", "AI Judge", "Cooldown"]:
		if task.begins_with(k):
			return k
	return "Render" if task.begins_with("Rendering") or task.begins_with("Re-render") else ""


## Estimated ms left: current step + rest of this item + (pending items + their cooldowns).
## shortcut: YouTube items assume `clips` shorts each (gated runs make fewer); Twitch items make 1.
func eta_remaining(now_ms: int) -> int:
	var key := eta_key(_task_name)
	var left := maxi(0, int(eta_ms.get(key, 0)) - (now_ms - _task_start_ms))
	var shorts := (1 if _twitch_run else int(settings.clips)) if settings.auto_edit else 0
	if key != "Cooldown":
		for i in range(_eta_pos + 1, 3):  # Download / Whisper / AI Judge still to come
			left += int(eta_ms[ETA_ORDER[i]])
		var renders := shorts if _shorts_left < 0 else _shorts_left + (0 if key == "Render" else 1)
		left += renders * int(eta_ms.Render)
	if _batch_total > 0:
		var item := int(eta_ms.Download) + int(eta_ms.Whisper) + int(eta_ms["AI Judge"]) + shorts * int(eta_ms.Render)
		left += int(_queue_counts.pending) * (item + (TWITCH_COOLDOWN_MS if _twitch_run else int(eta_ms.Cooldown)))
	return left


func telemetry_begin(now_ms: int = -1) -> void:
	now_ms = Time.get_ticks_msec() if now_ms < 0 else now_ms
	_overall_start_ms = now_ms
	_overall_end_ms = -1
	_task_start_ms = now_ms
	_task_name = "Starting"
	_steps.clear()
	_progress = 0.0
	_batch_index = 0
	_batch_total = 0
	_layout_tag = ""
	_eta_pos = -1
	_shorts_left = -1
	progress_bar.value = 0
	telemetry_tick(now_ms)


func telemetry_end(ok: bool, now_ms: int = -1) -> void:
	now_ms = Time.get_ticks_msec() if now_ms < 0 else now_ms
	if _overall_end_ms >= 0:
		return
	_finish_step(now_ms)
	_overall_end_ms = now_ms
	_task_name = "Done" if ok else "Ended with errors"
	if ok:
		_set_progress(1.0, true)
	telemetry_tick(now_ms)
	refresh_stats()
	save_eta()


# the learned step times survive a restart, so the first ETAs of a session are already this PC's real ones

func _eta_file() -> String:
	return eta_path if not eta_path.is_empty() else _project_root().path_join("workspace/eta_stats.json")


func save_eta() -> void:
	DirAccess.make_dir_recursive_absolute(_eta_file().get_base_dir())
	var f := FileAccess.open(_eta_file(), FileAccess.WRITE)
	if f:
		f.store_string(JSON.stringify(eta_ms, "  "))


## Known step names with a positive number only; anything else keeps its seed (a missing/broken file = seeds).
func load_eta() -> void:
	eta_ms = ETA_SEED.duplicate()
	var json := JSON.new()
	if not FileAccess.file_exists(_eta_file()) or json.parse(FileAccess.get_file_as_string(_eta_file())) != OK or not json.data is Dictionary:
		return
	for k: String in ETA_SEED:
		var v: Variant = (json.data as Dictionary).get(k)
		if (v is float or v is int) and v > 0:
			eta_ms[k] = int(v)


func _finish_step(now_ms: int) -> void:
	var key := eta_key(_task_name)
	if not key.is_empty():
		if not (key == "Cooldown" and _twitch_run):  # the average is for the YouTube pause; Twitch's is fixed
			eta_ms[key] = int(0.7 * eta_ms[key] + 0.3 * (now_ms - _task_start_ms))
	if not _task_name.is_empty():
		_steps.append({"name": _task_name + _layout_tag, "ms": now_ms - _task_start_ms})
		if _steps.size() > 12:
			_steps.pop_front()


func _start_task(title: String, now_ms: int) -> void:
	_finish_step(now_ms)
	_layout_tag = ""  # a layout belongs to the edit it was detected in, not to whatever comes next
	_task_name = title
	_task_start_ms = now_ms
	var nm := RegEx.create_from_string("^Editing short (\\d+)/(\\d+)").search(title)
	if nm:
		_eta_pos = 3
		_shorts_left = int(nm.get_string(2)) - int(nm.get_string(1))
	elif title.begins_with("Video "):  # next queue item
		_eta_pos = -1
		_shorts_left = -1
	else:
		_eta_pos = maxi(_eta_pos, ETA_ORDER.find(eta_key(title)))


## Turns backend events into the active task, step history and progress (heartbeats feed the status light).
func track_task(ev: Dictionary, now_ms: int = -1) -> void:
	now_ms = Time.get_ticks_msec() if now_ms < 0 else now_ms
	var msg := str(ev.get("msg", ""))
	var title := ""
	var frac := -1.0
	match str(ev.get("event", "")):
		"stage":
			var stage := str(ev.get("stage", "")).to_upper()
			title = _task_title(stage, msg)
			frac = _stage_fraction(stage, msg)
		"queue":
			for k: String in ["pending", "done", "failed"]:
				if ev.has(k):
					_queue_counts[k] = int(ev[k])
			if msg.begins_with("Twitch:") or msg.begins_with("RSS:"):  # the poll that just finished gets its result
				var n := RegEx.create_from_string("(\\d+) new").search(msg)
				_task_name = "%s Poll (%s clips)" % ["Twitch" if msg.begins_with("Twitch") else "RSS", n.get_string(1) if n else "0"]
				title = "Queue"
			elif msg.begins_with("Processing"):
				_batch_index += 1
				_batch_total = maxi(_batch_total, _batch_index + int(ev.get("pending", 0)))
				title = "Video %d/%d" % [_batch_index, _batch_total]
				frac = 0.0
			elif msg.begins_with("Cooldown"):
				title = "Cooldown"
			elif msg.begins_with("Auto-editing"):
				title = "Auto-edit"
		"log":
			if msg.begins_with("Facecam:"):
				_layout_tag = " [split_stack]" if msg.contains("detected") else " [blur_fill]"
			elif msg.begins_with("Encoder: "):
				_encoder_name = msg.trim_prefix("Encoder: ")
			elif msg.contains("loaded on cuda"):
				_whisper_dev = "CUDA"
			elif msg.contains("loaded on cpu"):
				_whisper_dev = "CPU"
	if frac >= 0:
		_set_progress(frac)
	if not title.is_empty() and title != _task_name:
		_start_task(title, now_ms)
	telemetry_tick(now_ms)


func _task_title(stage: String, msg: String) -> String:
	if msg.contains("Whisper"):
		return "Whisper Transcribe"
	if msg.contains("Ollama"):
		return "AI Judge"
	match stage:
		"METADATA":
			return "Fetch metadata"
		"DOWNLOAD":
			return "Download video" if not msg.contains("Twitch") else "Download Twitch clip"
		"CUTTING":
			return msg.replace("Cutting clip", "Making clip")
		"EDIT":
			return msg if msg.begins_with("Editing short") else msg.get_slice(" with", 0).get_slice(" (", 0).left(32)
		"RENDER":
			return "Re-render (QC repair)" if msg.contains("repair") else "Rendering short"
	return msg.left(32)


## Progress within one video, from the stage (clips: metadata -> download -> scoring -> cutting; then shorts).
func _stage_fraction(stage: String, msg: String) -> float:
	var nm := RegEx.create_from_string("(\\d+)/(\\d+)").search(msg)
	var i := float(nm.get_string(1)) if nm else 0.0
	var n := maxf(1.0, float(nm.get_string(2))) if nm else 1.0
	match stage:
		"METADATA":
			return 0.03
		"DOWNLOAD":
			return 0.10
		"SCORING":
			return 0.35 if msg.contains("Whisper") else (0.5 if msg.contains("Ollama") else 0.25)
		"CUTTING":
			return 0.6 + 0.15 * i / n
		"EDIT":
			return 0.75 + 0.25 * (i - 1) / n if nm else -1.0
	return -1.0


func _set_progress(frac: float, force: bool = false) -> void:
	var overall := frac
	if _batch_total > 0 and not force:
		overall = (float(_batch_index - 1) + frac) / _batch_total
	_progress = clampf(overall, 0.0, 1.0) if force else maxf(_progress, clampf(overall, 0.0, 1.0))
	progress_bar.value = _progress * 100.0


func telemetry_tick(now_ms: int) -> void:
	var end := now_ms if _overall_end_ms < 0 else _overall_end_ms
	var running := _overall_end_ms < 0 and not _task_name.is_empty()
	task_label.text = (_task_name + _layout_tag) if not _task_name.is_empty() else "Idle"
	var task_eta := -1
	var all_eta := -1
	if running:
		var key := eta_key(_task_name)
		task_eta = maxi(0, int(eta_ms[key]) - (now_ms - _task_start_ms)) if not key.is_empty() else -1
		all_eta = eta_remaining(now_ms)
	elif _overall_end_ms >= 0:
		task_eta = 0  # finished: frozen at 0s
		all_eta = 0
	timer_label.text = format_timers(end - _task_start_ms if running else 0, end - _overall_start_ms if not _task_name.is_empty() else 0,
			task_eta, all_eta)
	for c: Node in history_box.get_children():
		c.queue_free()
	for s: Dictionary in _steps:
		history_box.add_child(_history_line("✓ %s — %s" % [s.name, _fmt_secs(int(s.ms))], DIM))
	if running:
		history_box.add_child(_history_line("▶ %s — %s" % [_task_name + _layout_tag, fmt_dur(now_ms - _task_start_ms)], Color.WHITE))


static func _fmt_secs(ms: int) -> String:
	return "%.1fs" % (ms / 1000.0) if ms < 60_000 else fmt_dur(ms)


func _history_line(text: String, color: Color) -> Label:
	var l := _label(text)
	l.add_theme_font_size_override("font_size", 12)
	l.add_theme_color_override("font_color", color)
	l.clip_text = true
	return l


# hardware sampling runs on a thread every 2 s (nvidia-smi / wmic can take ~100 ms); results land via call_deferred

func start_telemetry() -> void:
	if DisplayServer.get_name() == "headless":
		return  # tests feed apply_telemetry() directly
	_telemetry_on = true
	_telemetry_thread = Thread.new()
	_telemetry_thread.start(_telemetry_loop)


func stop_telemetry() -> void:
	_telemetry_on = false
	if _telemetry_thread != null:
		_telemetry_thread.wait_to_finish()
		_telemetry_thread = null


func _telemetry_loop() -> void:
	var prev_cpu: Array = []
	var i := 0
	while _telemetry_on:
		apply_telemetry.call_deferred(sample_system(prev_cpu, i % 5 == 0))
		i += 1
		for k in 20:
			if not _telemetry_on:
				return
			OS.delay_msec(100)


## One sample of GPU / CPU (+ queue totals every ~10 s). Runs on the telemetry thread: no UI access here.
func sample_system(prev_cpu: Array, with_queue: bool) -> Dictionary:
	var t := {}
	var out := []
	if OS.execute("nvidia-smi", ["--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
			"--format=csv,noheader,nounits"], out) == 0 and not out.is_empty():
		t["gpu"] = parse_nvidia_smi(str(out[0]))
	elif OS.get_name() != "Windows" and FileAccess.file_exists("/dev/dri/renderD128"):
		t["gpu_kind"] = "AMD Vega / VAAPI"
	out = []
	if OS.get_name() == "Windows":
		if OS.execute("wmic", ["cpu", "get", "loadpercentage", "/value"], out) == 0 and not out.is_empty():
			t["cpu"] = parse_wmic_load(str(out[0]))
	elif OS.execute("head", ["-n1", "/proc/stat"], out) == 0 and not out.is_empty():
		t["cpu"] = cpu_from_proc_stat(str(out[0]), prev_cpu)
	if with_queue:
		var db := _project_root().path_join("workspace/queue.db")
		out = []
		if FileAccess.file_exists(db) and OS.execute(_python(), ["-c", "import sqlite3,sys,json; c=sqlite3.connect(sys.argv[1]); "
				+ "print(json.dumps(dict(c.execute('select status,count(*) from jobs group by status').fetchall())))", db], out) == 0:
			var q: Variant = JSON.parse_string(str(out[0]).strip_edges()) if not out.is_empty() else null
			if q is Dictionary:
				t["queue"] = q
	return t


static func parse_nvidia_smi(line: String) -> Dictionary:
	var p := line.strip_edges().split("\n")[0].split(",")
	if p.size() < 4:
		return {}
	return {"util": float(p[0]), "used": float(p[1]), "total": maxf(1.0, float(p[2])), "temp": float(p[3])}


static func parse_wmic_load(text: String) -> float:
	var vals := []
	for line: String in text.split("\n"):
		if line.strip_edges().begins_with("LoadPercentage="):
			vals.append(float(line.strip_edges().get_slice("=", 1)))
	return vals.reduce(func(a: float, b: float) -> float: return a + b, 0.0) / maxf(1.0, vals.size())


## CPU % from two /proc/stat "cpu" lines (prev_cpu keeps [busy, total] between calls).
static func cpu_from_proc_stat(line: String, prev_cpu: Array) -> float:
	var nums := Array(line.split(" ", false)).slice(1).map(func(s: String) -> float: return float(s))
	if nums.size() < 4:
		return 0.0
	var idle: float = nums[3] + (nums[4] if nums.size() > 4 else 0.0)
	var total: float = nums.reduce(func(a: float, b: float) -> float: return a + b, 0.0)
	var pct := 0.0
	if prev_cpu.size() == 2 and total > prev_cpu[1]:
		pct = 100.0 * (1.0 - (idle - prev_cpu[0]) / (total - prev_cpu[1]))
	prev_cpu.clear()
	prev_cpu.append_array([idle, total])
	return pct


func apply_telemetry(t: Dictionary) -> void:
	if t.has("gpu") and not (t.gpu as Dictionary).is_empty():
		var g: Dictionary = t.gpu
		_gpu_kind = "NVIDIA"
		set_gauge("GPU", g.util, "GPU %d%%   %d°C" % [int(g.util), int(g.temp)])
		set_gauge("VRAM", 100.0 * g.used / g.total, "VRAM %d / %d MB" % [int(g.used), int(g.total)])
	else:
		_gpu_kind = str(t.get("gpu_kind", ""))
		(gauges["GPU"].label as Label).text = "GPU: " + (_gpu_kind if not _gpu_kind.is_empty() else "N/A")
		(gauges["VRAM"].label as Label).text = "VRAM: N/A"
	if t.has("cpu"):
		set_gauge("CPU", t.cpu, "CPU %d%%" % int(t.cpu))
	var mem := OS.get_memory_info()
	var total := float(mem.get("physical", 0))
	if total > 0:
		# Linux "available" = MemAvailable (right); Windows "available" counts the page file (can exceed RAM)
		var avail := float(mem.get("available", -1))
		var used := total - (avail if 0 < avail and avail <= total else float(mem.get("free", 0)))
		set_gauge("RAM", 100.0 * used / total, "RAM %.1f / %.1f GB" % [used / 1e9, total / 1e9])
	var free_gb := DirAccess.open(_project_root()).get_space_left() / 1e9 if DirAccess.open(_project_root()) else -1.0
	disk_label.text = "Disk free: %.1f GB" % free_gb if free_gb >= 0 else "Disk free: --"
	disk_label.add_theme_color_override("font_color", Color("5ad06a") if free_gb > 15 else (Color("e0a030") if free_gb > 5 else Color("ef4444")))
	if t.has("queue"):
		for k: String in ["pending", "done", "failed"]:
			_queue_counts[k] = int((t.queue as Dictionary).get(k, 0))
	refresh_stats()


func refresh_stats() -> void:
	var enc: String = {"h264_nvenc": "NVENC (H.264)", "h264_vaapi": "VAAPI (H.264)", "libx264": "CPU libx264"}.get(_encoder_name,
			"NVENC (H.264)" if _gpu_kind == "NVIDIA" else ("VAAPI (H.264)" if _gpu_kind.contains("VAAPI") else "CPU libx264"))
	var whisper := _whisper_dev if not _whisper_dev.is_empty() else ("CUDA" if _gpu_kind == "NVIDIA" else "CPU")
	var ollama := "off"
	if ollama_badge and ollama_badge.text.ends_with("ONLINE"):
		ollama = "Vulkan" if OS.get_name() == "Windows" else "on"
	var shorts := 0
	if DirAccess.dir_exists_absolute(_ready_dir()):
		shorts = Array(DirAccess.get_files_at(_ready_dir())).filter(func(f: String) -> bool: return f.ends_with("_short.mp4")).size()
	stats_label.text = "Encoder: %s\nAI: Whisper %s + Ollama %s ($0.00)\nQueue: %d pending · %d done · %d failed\nShorts ready: %d\nTwitch token: %s" % [
			enc, whisper, ollama, int(_queue_counts.pending), int(_queue_counts.done), int(_queue_counts.failed), shorts, twitch_token_status()]


func twitch_token_status(now_unix: float = -1.0) -> String:
	now_unix = Time.get_unix_time_from_system() if now_unix < 0 else now_unix
	var tok: Variant = JSON.parse_string(FileAccess.get_file_as_string(_project_root().path_join("workspace/twitch_token.json")))
	if not tok is Dictionary or not (tok as Dictionary).has("expires_at"):
		return "none yet"
	var left := (float(tok.expires_at) - now_unix) / 86400.0
	return "Valid — %dd left" % int(left) if left > 0 else "Expired (renews on next poll)"


# --- results: every clip of every clips_manifest.json in output/*/ and workspace/*/, newest first ------------------

const CLIP_FILTERS := ["ALL CLIPS", "EDITED SHORTS ONLY", "LAST RUN", "CLEARED"]
var clear_button: Button
const MAX_ROWS := 200  # shortcut: rows beyond this aren't built; page the list if the archive outgrows it
var clip_roots := PackedStringArray()  # empty = <project>/output + <project>/workspace (tests point it elsewhere)
var clip_filter: OptionButton
var clip_count: Label
var _entries: Array[Dictionary] = []  # {"path": manifest, "data": manifest dict, "clip": clip dict, "mtime": unix s}
var _last_run_unix := 0


func _make_clips_header() -> Control:
	var head := HBoxContainer.new()
	head.add_theme_constant_override("separation", 8)
	clip_count = _label("CLIPS")
	clip_count.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	head.add_child(clip_count)
	clip_filter = OptionButton.new()
	for f: String in CLIP_FILTERS:
		clip_filter.add_item(f)
	clip_filter.item_selected.connect(func(_i: int) -> void: show_clips())
	head.add_child(clip_filter)
	clear_button = Button.new()
	clear_button.text = "CLEAR"
	clear_button.pressed.connect(clear_clips)
	head.add_child(clear_button)
	var ready := Button.new()
	ready.text = "Open Ready-to-Post Folder"
	ready.pressed.connect(func() -> void:
		DirAccess.make_dir_recursive_absolute(_ready_dir())
		OS.shell_open(_ready_dir()))
	head.add_child(ready)
	return head


## Re-reads every manifest (startup, each finished video, each new short, end of every run) and redraws.
func refresh_clips() -> void:
	_entries.clear()
	_posted_by = posted_by()
	var roots := clip_roots if not clip_roots.is_empty() else PackedStringArray(
			[_project_root().path_join("output"), _project_root().path_join("workspace")])
	for root: String in roots:
		if not DirAccess.dir_exists_absolute(root):
			continue
		for d: String in DirAccess.get_directories_at(root):
			var path := root.path_join(d).path_join("clips_manifest.json")
			var json := JSON.new()  # .parse(): a half-written manifest is skipped quietly, no engine error spam
			if not FileAccess.file_exists(path) or json.parse(FileAccess.get_file_as_string(path)) != OK:
				continue
			var data: Variant = json.data
			if not data is Dictionary or not (data as Dictionary).get("clips") is Array:
				continue
			var mtime := FileAccess.get_modified_time(path)
			for clip: Variant in data.clips:
				if clip is Dictionary:
					_entries.append({"path": path, "data": data, "clip": clip, "mtime": mtime})
	_entries.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		if a.mtime != b.mtime:
			return a.mtime > b.mtime
		if a.path != b.path:
			return a.path < b.path
		return int(a.clip.get("clip_id", 0)) < int(b.clip.get("clip_id", 0)))
	show_clips()


## Survives the queue moving a manifest from workspace/ to output/ (the path changes, the video id doesn't).
static func clip_key(e: Dictionary) -> String:
	return "%s#%d" % [str(e.data.get("video_id", e.path)), int(e.clip.get("clip_id", 0))]


func _is_cleared(e: Dictionary) -> bool:
	return (settings.cleared_clips as Array).has(clip_key(e))


## The entries the filter keeps, cleared ones aside (CLEARED shows only those). LAST RUN = manifests written since
## this session's last START (before any run: the newest manifest).
func filtered_entries() -> Array[Dictionary]:
	if clip_filter.selected == 3:
		return _entries.filter(_is_cleared)
	var kept := _entries.filter(func(e: Dictionary) -> bool: return not _is_cleared(e))
	match clip_filter.selected:
		1:
			return kept.filter(func(e: Dictionary) -> bool: return FileAccess.file_exists(short_path(e)))
		2:
			var since: int = _last_run_unix if _last_run_unix > 0 else (kept[0].mtime if not kept.is_empty() else 0)
			return kept.filter(func(e: Dictionary) -> bool: return e.mtime >= since)
	return kept


func show_clips() -> void:
	for child: Node in clip_list.get_children():
		child.queue_free()
	var shown := filtered_entries()
	if shown.is_empty() and clip_filter.selected == 2 and _entries.any(func(e: Dictionary) -> bool: return not _is_cleared(e)):
		clip_filter.select(0)  # an empty LAST RUN must never hide the clips that do exist
		_log("[FILTER] LAST RUN has no clips (nothing passed the judge, or nothing new): showing ALL CLIPS.", AMBER)
		shown = filtered_entries()
	var cleared := _entries.filter(_is_cleared).size()
	var listed := _entries.size() - cleared
	if clip_filter.selected == 3:
		clip_count.text = "CLIPS: %d cleared" % shown.size()
	else:
		clip_count.text = ("CLIPS: %d" % listed if shown.size() == listed else "CLIPS: %d of %d" % [shown.size(), listed]) \
				+ ("  ·  %d cleared" % cleared if cleared else "")
	clear_button.text = "RESTORE" if clip_filter.selected == 3 else "CLEAR"
	clear_button.tooltip_text = ("Put these clips back in the list" if clip_filter.selected == 3
			else "Hide the clips listed now. Nothing is deleted; filter CLEARED shows them again.")
	clear_button.disabled = shown.is_empty()
	for e: Dictionary in shown.slice(0, MAX_ROWS):
		clip_list.add_child(_clip_row(e))


## CLEAR: hide every clip the list shows right now (saved in settings.json; files and manifests untouched).
## In the CLEARED view the same button restores them.
func clear_clips() -> void:
	var keys := filtered_entries().map(clip_key)
	var cleared: Array = settings.cleared_clips
	if clip_filter.selected == 3:
		set_setting("cleared_clips", cleared.filter(func(k: String) -> bool: return not keys.has(k)))
	else:
		set_setting("cleared_clips", cleared + keys.filter(func(k: String) -> bool: return not cleared.has(k)))
	show_clips()


func _clip_file(e: Dictionary) -> String:
	return str(e.path).get_base_dir().path_join(str(e.clip.get("file", "")))


func _clip_row(e: Dictionary) -> Control:
	var clip: Dictionary = e.clip
	var file := _clip_file(e)
	var panel := PanelContainer.new()
	panel.theme_type_variation = &"ClipRow"
	var box := VBoxContainer.new()
	panel.add_child(box)
	var head := _label("#%d   %s   %s - %s   SCORE %.2f   %s" % [int(clip.get("clip_id", 0)),
			str(e.data.get("title", e.data.get("video_id", ""))).left(40),
			_clock(float(clip.get("start_sec", 0))), _clock(float(clip.get("end_sec", 0))),
			float(clip.get("score", 0)), str(clip.get("reason", ""))])
	head.add_theme_color_override("font_color", AMBER)
	head.clip_text = true
	box.add_child(head)
	var text := _label("\"%s\"" % str(clip.get("transcript", "")).left(300))
	text.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	text.add_theme_color_override("font_color", DIM)
	box.add_child(text)
	if str(clip.get("judge_reason", "")) not in ["", "<null>"]:  # why the AI judge kept it
		var why := _label("AI judge: " + str(clip.judge_reason))
		why.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		why.add_theme_color_override("font_color", AMBER.lightened(0.3))
		box.add_child(why)
	var short_info := _short_summary(e)
	if not short_info.is_empty():
		var edited := _label(short_info)
		edited.add_theme_color_override("font_color", GOOD)
		edited.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		box.add_child(edited)
	var buttons := HFlowContainer.new()  # wraps onto a 2nd line when the row is narrow (up to 6 buttons)
	box.add_child(buttons)
	var play := Button.new()
	play.text = "Preview Raw"
	play.tooltip_text = "Instant built-in preview (Space pause, arrows seek, Esc close)"
	var dur: float = clip.duration if clip.get("duration") != null else float(clip.get("end_sec", 0)) - float(clip.get("start_sec", 0))
	play.pressed.connect(func() -> void: open_preview(file, e, dur))
	buttons.add_child(play)
	var folder := Button.new()
	folder.text = "Open Output Folder"
	folder.pressed.connect(func() -> void: OS.shell_open(file.get_base_dir()))
	buttons.add_child(folder)
	var reject := Button.new()
	if _pending_deletes.has(clip_key(e)):  # in its 5 s grace period: dimmed, one click keeps it
		panel.modulate.a = 0.45
		reject.text = "UNDO DELETE"
		reject.add_theme_color_override("font_color", AMBER)
		reject.tooltip_text = "Keep this clip (it goes to the Recycle Bin in a few seconds otherwise)"
		reject.pressed.connect(undo_delete.bind(e))
	else:
		reject.text = "Delete Reject"
		reject.tooltip_text = "Trash this clip in %d s (UNDO DELETE keeps it). Shift+click: at once." % (UNDO_MS / 1000)
		reject.pressed.connect(func() -> void: request_delete(e, Input.is_key_pressed(KEY_SHIFT)))
	buttons.add_child(reject)
	var has_short := FileAccess.file_exists(short_path(e))
	var short_button := Button.new()
	short_button.text = "PLAY EDITED SHORT" if has_short else "MAKE SHORT"
	short_button.tooltip_text = ("Play the edited 9:16 short from ready_to_post/" if has_short
			else "Edit this clip into a 9:16 short (ready_to_post/), then play it")
	if has_short:
		short_button.add_theme_color_override("font_color", GOOD)
	short_button.pressed.connect(preview_short.bind(e))
	buttons.add_child(short_button)
	if has_short:  # post this very short now; a platform that already has it shows a tick instead
		var sid := short_path(e).get_file().trim_suffix("_short.mp4")
		for pf: String in ["youtube", "tiktok"]:
			var site: String = {"youtube": "YouTube", "tiktok": "TikTok"}[pf]
			var post := Button.new()
			var done: bool = pf in _posted_by.get(sid, [])
			post.text = ("On %s ✓" if done else "Post to %s") % site
			post.disabled = done
			post.tooltip_text = "Upload this short to %s now (official API; ignores Post Interval and Publish Platform)" % site
			post.pressed.connect(post_short.bind(sid, pf))
			buttons.add_child(post)
	return panel


## "EDITED SHORT: <title + hashtags>  |  46.4s  setup_punchline" from the short's _post.json ("" if no short yet).
func _short_summary(e: Dictionary) -> String:
	var path := short_path(e)
	if not FileAccess.file_exists(path):
		return ""
	var post: Variant = JSON.parse_string(FileAccess.get_file_as_string(path.trim_suffix("_short.mp4") + "_post.json"))
	var on: Array = _posted_by.get(path.get_file().trim_suffix("_short.mp4"), [])
	var what := "EDITED SHORT"
	if not on.is_empty():
		what = "POSTED TO " + " + ".join(on.map(func(pf: String) -> String: return PLATFORM_LABEL.get(pf, pf.to_upper())))
	elif path.get_base_dir() == _posted_dir():
		what = "POSTED"
	if not post is Dictionary:
		return "%s: %s" % [what, path.get_file()]
	var p: Dictionary = post
	return what + ": %s  |  %.1fs  %s" % [str(p.get("social_title", p.get("title", ""))), float(p.get("duration", 0)),
			str(p.get("archetype", "")).replace("_", " ")]


## The clip's short: in ready_to_post/, or in posted/ once the publisher has posted it (then it is still "made").
func short_path(e: Dictionary) -> String:
	var name := "%s_%d_short.mp4" % [str(e.data.get("video_id", "")), int(e.clip.get("clip_id", 0))]
	var ready := _ready_dir().path_join(name)
	var posted := _posted_dir().path_join(name)
	return posted if not FileAccess.file_exists(ready) and FileAccess.file_exists(posted) else ready


func _posted_dir() -> String:
	return _ready_dir().get_base_dir().path_join("posted")


## Plays the finished short, or renders it with editor/pipeline.py and plays it when the "short" event arrives.
func preview_short(e: Dictionary) -> void:
	var path := short_path(e)
	if FileAccess.file_exists(path):
		var post: Variant = JSON.parse_string(FileAccess.get_file_as_string(path.trim_suffix("_short.mp4") + "_post.json"))
		open_preview(path, e, float((post as Dictionary).get("duration", 0)) if post is Dictionary else 0.0)
	elif _thread == null:
		_preview_path = path
		_start(PackedStringArray(["-u", _project_root().path_join("editor/pipeline.py"), "--manifest", str(e.path),
				"--clip", str(int(e.clip.get("clip_id", 0)))]))
	else:
		_log("Busy: preview after the current run finishes.", AMBER)


func _ready_dir() -> String:
	return ready_dir if not ready_dir.is_empty() else _project_root().path_join("ready_to_post")


## Sends the clip to the recycle bin and drops it from its manifest so a reload stays consistent.
func delete_clip(e: Dictionary) -> void:
	var file := _clip_file(e)
	if FileAccess.file_exists(file) and OS.move_to_trash(file) != OK:
		_log("Could not delete %s" % file, BAD)
		return
	(e.data.clips as Array).erase(e.clip)
	var f := FileAccess.open(str(e.path), FileAccess.WRITE)
	if f == null:
		_log("Could not update manifest %s" % e.path, BAD)
		return
	f.store_string(JSON.stringify(e.data, "  "))
	f.close()
	_entries.erase(e)
	show_clips()
	_log("Rejected %s" % file.get_file(), DIM)


# --- instant preview player: ffmpeg raw RGB frames over a pipe + a WAV for the audio clock (no VLC, no .ogv) -------

const PV_FPS := 24.0  # light on purpose: a preview must never compete with a render (NVENC) or Whisper (CUDA)
const PV_AHEAD := 8  # decoded frames buffered ahead of the clock (~4 MB at 568x320): ffmpeg blocks beyond this
const PV_TIMEOUT_MS := 5000  # no first picture by then: the preview gives up (never the run)
var preview: Control  # full-window overlay
var pv_view: TextureRect
var pv_title: Label
var pv_time: Label
var pv_slider: HSlider
var pv_play: Button
var pv_audio: AudioStreamPlayer
var pv_state := "closed"  # closed | playing | paused | ended
var pv_path := ""
var pv_entry: Dictionary = {}  # the clip being previewed ({} = none: no Delete Reject)
var pv_duration := 0.0
var pv_size := Vector2i(640, 360)
## (path, start_sec, size) -> [exe, args] for the frame pipe; tests swap in a fake decoder.
var pv_command: Callable = _pv_ffmpeg_command
var _pv_pid := -1
var _pv_thread: Thread
var _pv_mutex := Mutex.new()
var _pv_frames: Array[PackedByteArray] = []
var _pv_eof := false
var _pv_reading := false
var _pv_next_pts := 0.0  # source time of the next queued frame
var _pv_pos := 0.0  # clock: position at _pv_anchor_ms (wall clock until the audio is loaded)
var _pv_anchor_ms := 0
var _pv_audio_pid := -1
var _pv_start_ms := 0
var _pv_got_frame := false
var _pv_wav_dirty := false  # temp WAV still to delete (it was locked at close)
var _pv_image: Image
var _pv_texture: ImageTexture


func _make_preview() -> Control:
	preview = Control.new()
	preview.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	preview.visible = false
	var shade := ColorRect.new()
	shade.color = Color(0, 0, 0, 0.8)
	shade.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	shade.gui_input.connect(func(ev: InputEvent) -> void:
		if ev is InputEventMouseButton and ev.pressed:
			close_preview())
	preview.add_child(shade)
	var center := CenterContainer.new()
	center.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	center.mouse_filter = Control.MOUSE_FILTER_IGNORE
	preview.add_child(center)
	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", _box(Color("16181a"), AMBER, 1))
	center.add_child(panel)
	var col := VBoxContainer.new()
	panel.add_child(col)
	var head := HBoxContainer.new()
	col.add_child(head)
	pv_title = _label("")
	pv_title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	pv_title.clip_text = true
	pv_title.add_theme_color_override("font_color", AMBER)
	head.add_child(pv_title)
	head.add_child(_pv_button("Pop Out (ffplay)", "Open in ffplay (Esc closes it)", func() -> void: pop_out(pv_path)))
	head.add_child(_pv_button("×", "Close (Esc)", close_preview))
	pv_view = TextureRect.new()
	pv_view.custom_minimum_size = Vector2(720, 560)
	pv_view.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	pv_view.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	col.add_child(pv_view)
	var bar := HBoxContainer.new()
	col.add_child(bar)
	pv_play = _pv_button("PAUSE", "Space", toggle_preview)
	pv_play.custom_minimum_size.x = 70
	bar.add_child(pv_play)
	pv_slider = HSlider.new()
	pv_slider.step = 0.01
	pv_slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	pv_slider.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	pv_slider.focus_mode = Control.FOCUS_NONE  # arrows seek ±3 s instead of nudging the slider
	pv_slider.drag_ended.connect(func(_changed: bool) -> void: seek_preview(pv_slider.value))
	bar.add_child(pv_slider)
	pv_time = _label("0:00 / 0:00")
	bar.add_child(pv_time)
	var reject := _pv_button("Delete Reject", "Trash this clip (and its edited short) and close", reject_previewed)
	reject.add_theme_color_override("font_color", BAD)
	bar.add_child(reject)
	pv_audio = AudioStreamPlayer.new()
	preview.add_child(pv_audio)
	return preview


func _pv_button(text: String, tip: String, action: Callable) -> Button:
	var b := Button.new()
	b.text = text
	b.tooltip_text = tip
	b.focus_mode = Control.FOCUS_NONE  # Space must toggle play, not re-press the last clicked button
	b.pressed.connect(action)
	return b


func _pv_ffmpeg_command(path: String, start: float, size: Vector2i) -> Array:
	var vf := "fps=%d,scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2" % [
			int(PV_FPS), size.x, size.y, size.x, size.y]
	# low_delay/nobuffer: first frame ~170 ms instead of ~270 (output checked byte-identical on HEVC + H.264 B-frames)
	# software decode on 2 threads (ffmpeg never touches the GPU unless asked): the render keeps NVENC and the CPU
	return ["ffmpeg", ["-v", "error", "-nostdin", "-threads", "2", "-flags", "low_delay", "-fflags", "nobuffer",
			"-ss", "%.3f" % start, "-i", path, "-an", "-threads", "2", "-vf", vf,
			"-f", "rawvideo", "-pix_fmt", "rgb24", "-"]]


func _pv_wav() -> String:
	return _project_root().path_join("workspace/_preview_audio.wav")


## Opens `path` in the overlay. entry: the clip (enables Delete Reject); duration: a hint until the audio says.
func open_preview(path: String, entry: Dictionary = {}, duration: float = 0.0) -> void:
	close_preview()
	pv_path = path
	pv_entry = entry
	pv_duration = duration
	pv_size = Vector2i(320, 568) if path.ends_with("_short.mp4") else Vector2i(568, 320)
	_pv_image = Image.create_empty(pv_size.x, pv_size.y, false, Image.FORMAT_RGB8)
	_pv_texture = ImageTexture.create_from_image(_pv_image)
	pv_view.texture = _pv_texture
	pv_title.text = path.get_file()
	preview.visible = true
	_pv_pos = 0.0
	if not _pv_start_frames(0.0):
		close_preview()
		pop_out(path)  # no pipe: the barebones external player
		return
	if DisplayServer.get_name() != "headless":  # tests never touch ffmpeg / audio hardware
		DirAccess.make_dir_recursive_absolute(_pv_wav().get_base_dir())
		_pv_audio_pid = OS.create_process("ffmpeg", ["-v", "error", "-nostdin", "-y", "-i", path, "-vn", "-ac", "2",
				"-ar", "44100", "-f", "wav", _pv_wav()])
	_pv_set_state("playing")


## (Re)starts the frame pipe at `start` seconds. -> false if the decoder could not be started.
func _pv_start_frames(start: float) -> bool:
	_pv_stop_frames()
	var cmd: Array = pv_command.call(pv_path, start, pv_size)
	var proc := OS.execute_with_pipe(cmd[0], cmd[1])
	if proc.is_empty():
		return false
	_pv_pid = int(proc.pid)
	_pv_frames.clear()
	_pv_eof = false
	_pv_reading = true
	_pv_next_pts = start
	_pv_start_ms = Time.get_ticks_msec()  # the watchdog's clock: a picture must arrive within PV_TIMEOUT_MS
	_pv_got_frame = false
	_pv_thread = Thread.new()
	_pv_thread.start(_pv_read.bind(proc, pv_size.x * pv_size.y * 3))
	return true


func _pv_stop_frames() -> void:
	_pv_reading = false
	if _pv_pid > 0:
		OS.kill(_pv_pid)  # closes the pipe, which ends the blocking read
	if _pv_thread != null:
		_pv_thread.wait_to_finish()
		_pv_thread = null
	_pv_pid = -1
	_pv_mutex.lock()
	_pv_frames.clear()
	_pv_mutex.unlock()


## Reader thread: whole frames off the pipe into a short queue; waits while PV_AHEAD frames are unshown.
func _pv_read(proc: Dictionary, frame_bytes: int) -> void:
	var pipe: FileAccess = proc.stdio
	while _pv_reading:
		var buf := PackedByteArray()
		while buf.size() < frame_bytes:  # pipe reads can come back short
			var part := pipe.get_buffer(frame_bytes - buf.size())
			if part.is_empty():
				break
			buf.append_array(part)
		if buf.size() < frame_bytes:
			break  # EOF, or killed
		_pv_mutex.lock()
		_pv_frames.append(buf)
		var full := _pv_frames.size() >= PV_AHEAD
		_pv_mutex.unlock()
		while full and _pv_reading:
			OS.delay_msec(4)
			_pv_mutex.lock()
			full = _pv_frames.size() >= PV_AHEAD
			_pv_mutex.unlock()
	_pv_mutex.lock()
	_pv_eof = true
	_pv_mutex.unlock()


## Seconds into the clip: the audio clock once the WAV is playing, the wall clock before that.
func preview_position(now_ms: int = -1) -> float:
	if pv_state != "playing":
		return _pv_pos
	if pv_audio.playing:
		return pv_audio.get_playback_position() + AudioServer.get_time_since_last_mix()
	now_ms = Time.get_ticks_msec() if now_ms < 0 else now_ms
	return _pv_pos + (now_ms - _pv_anchor_ms) / 1000.0


func _pv_set_state(state: String) -> void:
	if pv_state == "playing" and state != "playing":
		_pv_pos = preview_position()
	pv_state = state
	_pv_anchor_ms = Time.get_ticks_msec()
	pv_play.text = "PAUSE" if state == "playing" else "PLAY"
	if pv_audio.stream != null:  # restart from the saved position on resume: also re-locks audio to the frames
		if state == "playing":
			pv_audio.play(_pv_pos)
		else:
			pv_audio.stop()


func toggle_preview() -> void:
	match pv_state:
		"playing":
			_pv_set_state("paused")
		"paused":
			_pv_set_state("playing")
		"ended":
			seek_preview(0.0)
			_pv_set_state("playing")


## Jump to `t` seconds: a fresh decoder from there (keeps play/pause), audio follows.
func seek_preview(t: float) -> void:
	if pv_state == "closed":
		return
	t = clampf(t, 0.0, maxf(0.0, pv_duration - 0.1)) if pv_duration > 0 else maxf(0.0, t)
	_pv_pos = t
	_pv_anchor_ms = Time.get_ticks_msec()
	if pv_state == "ended":
		pv_state = "paused"
	_pv_start_frames(t)
	if pv_audio.stream != null and pv_state == "playing":
		pv_audio.play(t)
	_pv_show_time(t)


## Kill the decoders, drop the audio and the frame buffers, delete the temp WAV, hide.
func close_preview() -> void:
	_pv_stop_frames()
	if _pv_audio_pid > 0 and OS.is_process_running(_pv_audio_pid):
		OS.kill(_pv_audio_pid)
	_pv_audio_pid = -1
	pv_audio.stop()
	pv_audio.stream = null
	_pv_image = null
	_pv_texture = null
	pv_view.texture = null
	# a just-killed ffmpeg keeps the WAV locked on Windows for a moment: preview_tick retries the delete
	_pv_wav_dirty = FileAccess.file_exists(_pv_wav()) and DirAccess.remove_absolute(_pv_wav()) != OK
	pv_state = "closed"
	preview.visible = false


## Barebones external fallback: ffplay opens in ~0.1 s; the OS player only if ffplay is missing.
func pop_out(path: String) -> void:
	if path.is_empty():
		return
	close_preview()
	var short := path.ends_with("_short.mp4")
	var pid := OS.create_process("ffplay", ["-autoexit", "-loglevel", "quiet", "-x", "450" if short else "800",
			"-y", "800" if short else "450", "-window_title", path.get_file(), path])
	if pid <= 0:
		OS.shell_open(path)


## Delete Reject from inside the player: the clip plus its edited short, with the same 5 s UNDO on its row.
func reject_previewed() -> void:
	var e := pv_entry
	close_preview()
	if not e.is_empty():
		request_delete(e, Input.is_key_pressed(KEY_SHIFT), true)


# --- Delete Reject safety: 5 s to change your mind (Shift+click skips it) ---------------------------------------------

const UNDO_MS := 5000
var _pending_deletes := {}  # clip_key -> {"e": entry, "due": ticks ms, "short": also trash its short}


func request_delete(e: Dictionary, immediate: bool = false, with_short: bool = false, now_ms: int = -1) -> void:
	if immediate:
		_delete_now(e, with_short)
		return
	now_ms = Time.get_ticks_msec() if now_ms < 0 else now_ms
	_pending_deletes[clip_key(e)] = {"e": e, "due": now_ms + UNDO_MS, "short": with_short}
	_log("Deleting %s in %d s: UNDO DELETE on its row keeps it." % [_clip_file(e).get_file(), UNDO_MS / 1000], AMBER)
	show_clips()


func undo_delete(e: Dictionary) -> void:
	if _pending_deletes.erase(clip_key(e)):
		_log("Kept %s." % _clip_file(e).get_file(), DIM)
		show_clips()


## Carries out the deletes whose grace period is over (every frame), or all of them (closing the panel: you did ask).
func flush_deletes(now_ms: int, all: bool = false) -> void:
	for key: String in _pending_deletes.keys():
		var d: Dictionary = _pending_deletes[key]
		if all or now_ms >= int(d.due):
			_pending_deletes.erase(key)
			_delete_now(d.e, d.short)


func _delete_now(e: Dictionary, with_short: bool) -> void:
	if with_short:
		var short := short_path(e)
		for f: String in [short, short.trim_suffix("_short.mp4") + "_post.json"]:
			if FileAccess.file_exists(f):
				OS.move_to_trash(f)
	delete_clip(e)


## Give up on this preview without touching anything else: no status light, no error banner, no worker. Just a
## console line and the overlay closes (its decoders are killed and the temp WAV removed by close_preview).
func preview_abort(why: String) -> void:
	_log("[PREVIEW] Preview unavailable: %s. Try Pop Out (ffplay); any running job is not affected." % why, AMBER)
	close_preview()


## Per frame (from _process): show the newest frame due at the clock, load the WAV when ffmpeg is done with it.
func preview_tick(now_ms: int = -1) -> void:
	if _pv_wav_dirty and pv_state == "closed":
		_pv_wav_dirty = FileAccess.file_exists(_pv_wav()) and DirAccess.remove_absolute(_pv_wav()) != OK
	if pv_state == "closed":
		return
	if _pv_audio_pid > 0 and not OS.is_process_running(_pv_audio_pid):
		_pv_audio_pid = -1
		# no WAV (no audio track / ffmpeg failed): stay on the wall clock, silently
		var wav := AudioStreamWAV.load_from_file(_pv_wav()) if FileAccess.file_exists(_pv_wav()) else null
		if wav != null:
			_pv_pos = preview_position(now_ms)  # hand the clock over to the audio at the current moment
			_pv_anchor_ms = Time.get_ticks_msec()
			pv_duration = wav.get_length()
			pv_audio.stream = wav
			if pv_state == "playing":
				pv_audio.play(_pv_pos)
	var t := preview_position(now_ms)
	var frame := PackedByteArray()
	_pv_mutex.lock()
	while not _pv_frames.is_empty() and _pv_next_pts <= t + 0.5 / PV_FPS:
		frame = _pv_frames.pop_front()
		_pv_next_pts += 1.0 / PV_FPS
	var drained := _pv_eof and _pv_frames.is_empty()
	_pv_mutex.unlock()
	if not frame.is_empty():
		_pv_got_frame = true
		_pv_image.set_data(pv_size.x, pv_size.y, false, Image.FORMAT_RGB8, frame)
		_pv_texture.update(_pv_image)
	elif not _pv_got_frame:  # watchdog: a decoder that stalls or dies before its first picture
		now_ms = Time.get_ticks_msec() if now_ms < 0 else now_ms
		if drained:
			preview_abort("the decoder stopped without a picture: broken or unreadable file")
			return
		if now_ms - _pv_start_ms > PV_TIMEOUT_MS:
			preview_abort("no picture within %d s (decoder stalled or the system is busy)" % (PV_TIMEOUT_MS / 1000))
			return
	if pv_state == "playing" and drained and t >= _pv_next_pts:  # the clock is past the last frame
		_pv_set_state("ended")
	_pv_show_time(preview_position(now_ms))


func _pv_show_time(t: float) -> void:
	if pv_duration > 0:
		pv_slider.max_value = pv_duration
	if not pv_slider.has_focus():
		pv_slider.set_value_no_signal(t)
	pv_time.text = "%s / %s" % [_mmss(t), _mmss(pv_duration)]


static func _mmss(s: float) -> String:
	return "%d:%02d" % [int(s) / 60, int(s) % 60]


func _input(ev: InputEvent) -> void:
	if pv_state == "closed" or not ev is InputEventKey or not ev.pressed:
		return
	match (ev as InputEventKey).keycode:
		KEY_ESCAPE:
			close_preview()
		KEY_SPACE:
			toggle_preview()
		KEY_LEFT:
			seek_preview(preview_position() - 3.0)
		KEY_RIGHT:
			seek_preview(preview_position() + 3.0)
		_:
			return
	get_viewport().set_input_as_handled()


# --- helpers ------------------------------------------------------------------------------------

func _project_root() -> String:
	# shortcut: backend is found next to the Godot project folder; export builds need it shipped beside the exe.
	return ProjectSettings.globalize_path("res://").path_join("..").simplify_path()


func _clipper_path() -> String:
	return _project_root().path_join("backend/clipper.py")


func _python() -> String:
	var win := OS.get_name() == "Windows"
	var venv := _project_root().path_join(".venv/Scripts/python.exe" if win else ".venv/bin/python")
	if FileAccess.file_exists(venv):
		return venv
	return "python" if win else "python3"


func _log(text: String, color: Color) -> void:
	log_view.push_color(color)
	log_view.add_text(text + "\n")
	log_view.pop()


func _clock(sec: float) -> String:
	var s := int(sec)
	return "%02d:%02d:%02d" % [s / 3600, (s / 60) % 60, s % 60]


func _label(text: String) -> Label:
	var l := Label.new()
	l.text = text
	return l


func _spin(lo: float, hi: float, value: float, suffix: String) -> SpinBox:
	var s := SpinBox.new()
	s.min_value = lo
	s.max_value = hi
	s.value = value
	s.suffix = suffix
	return s


func _make_theme() -> Theme:
	var t := Theme.new()
	var font := SystemFont.new()
	font.font_names = PackedStringArray(["Consolas", "DejaVu Sans Mono", "Liberation Mono", "monospace"])
	t.default_font = font
	t.default_font_size = 14
	t.set_stylebox("panel", "Panel", _box(Color("16181a"), Color("16181a"), 0))
	t.set_stylebox("panel", "ClipRow", _box(Color("202326"), Color("3a3f44"), 1))
	t.set_type_variation(&"ClipRow", &"PanelContainer")
	t.set_stylebox("normal", "LineEdit", _box(Color("0e0f10"), Color("3a3f44"), 1))
	t.set_stylebox("focus", "LineEdit", _box(Color("0e0f10"), AMBER, 1))
	t.set_stylebox("normal", "RichTextLabel", _box(Color("0e0f10"), Color("3a3f44"), 1))
	t.set_stylebox("normal", "Button", _box(Color("2a2d30"), Color("4a5056"), 1))
	t.set_stylebox("hover", "Button", _box(Color("33373b"), AMBER, 1))
	t.set_stylebox("pressed", "Button", _box(Color("4a3a18"), AMBER, 1))
	t.set_stylebox("disabled", "Button", _box(Color("1c1e20"), Color("2a2d30"), 1))
	t.set_stylebox("slider", "HSlider", _box(Color("2a2d30"), Color("4a5056"), 1))
	t.set_stylebox("grabber_area", "HSlider", _box(AMBER.darkened(0.3), AMBER, 1))
	t.set_stylebox("grabber_area_highlight", "HSlider", _box(AMBER, AMBER, 1))
	t.set_color("font_color", "Label", Color("c8ccd0"))
	t.set_color("font_color", "Button", Color("e8e8e8"))
	t.set_color("font_hover_color", "Button", AMBER)
	t.set_color("default_color", "RichTextLabel", Color("c8ccd0"))
	return t


func _box(fill: Color, border: Color, width: int) -> StyleBoxFlat:
	var b := StyleBoxFlat.new()
	b.bg_color = fill
	b.border_color = border
	b.set_border_width_all(width)
	b.set_content_margin_all(6)
	return b

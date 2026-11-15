extends Control
## Clip Rate: scroll tune-o-clip's rendered shorts in a 9:16 feed, rate them, and send the ratings to Claude so it can
## retune the bot. UI built in code (like tune-o-clip's panel). Backend: tune-o-clip/calibration_studio/
## clip_rate_worker.py, one process at a time, JSON lines over a pipe.

const STREAMERS := ["caseoh_", "ishowspeed", "jynxzi", "lospollostv"]
const TAGS := ["Boring / No Hook", "Bad Cut / Pacing", "Wrong Archetype", "Caption / Audio Drift", "Banger / Keep"]
const PILLARS := [["hook_strength", "Hook"], ["comedy_payoff", "Payoff"], ["self_contained", "Self-contained"],
		["quotability", "Quotable"]]
const STAGES := ["Downloading", "Whisper [VRAM safe]", "Ollama Director", "FFmpeg Render"]
const ACCENT := Color("ff4f7b")
const GOOD := Color("4ade80")
const BAD := Color("f87171")
const DIM := Color("8a8f98")
const BG := Color("0f1012")
const LOG_LINES := 400

## tune-o-clip sits next to this project (Documents/clip-rate + Documents/tune-o-clip)
var toc_root := find_toc_root(ProjectSettings.globalize_path("res://"))
var ratings_path := toc_root.path_join("calibration_studio/clip_rate/ratings_queue.json")
var autostart := true  # false in tests: no scan worker on _ready

var cards: Array[Dictionary] = []  # every short we know (scan + short_ready), in arrival order
var ratings := {}  # card key -> {batch, id, title, streamer, short, thumb, scenario, stars, like, tags, liked, disliked, extra_notes, status}
var feed: Array[Dictionary] = []  # cards without a rating: the scroller
var index := 0
var drafts := {}  # key -> a rating taken back from the Rated tab to edit
var cur_like: Variant = null  # true / false / null
var cur_stars := 0
var cur_tags: Array[String] = []

var tabs: TabContainer
var player: ClipPlayer
var counter: Label
var feed_status: Label
var info_streamer: Label
var info_title: Label
var info_twitch: Label
var info_badge: Label
var info_stats: Label
var pillar_bars := {}
var info_reason: Label
var info_transcript: Label
var like_button: Button
var dislike_button: Button
var star_buttons: Array[Button] = []
var tag_buttons := {}
var liked_box: TextEdit
var disliked_box: TextEdit
var extra_box: TextEdit
var save_hint: Label
var rated_list: VBoxContainer
var rated_count: Label
var send_button: Button
var send_dialog: AcceptDialog
var send_text: TextEdit
var streamer_checks := {}
var custom_streamers: LineEdit
var per_spin: SpinBox
var fetch_button: Button
var url_edit: LineEdit
var file_dialog: FileDialog
var stage_labels: Array[Label] = []
var progress: ProgressBar
var progress_label: Label
var cancel_button: Button
var log_view: RichTextLabel

var _pid := -1
var _thread: Thread
var _mode := ""
var _run_start_ms := 0
var _cancelled := false
var _uploads: PackedStringArray = []
var _last_md := ""
var _wheel_ms := 0
var _shown_key := ""  # the short the side panel + player show
var _had_error := false
var _login_re := RegEx.create_from_string("^[a-z0-9_]{2,25}$")

const BOT_SERVICE := "tune-o-clip"  # the 24/7 bot: a systemd user unit written by setup_linux.sh
const BOT_COMMANDS := {"start": ["enable", "--now"], "stop": ["disable", "--now"], "restart": ["restart"]}
const DASHBOARD := "http://127.0.0.1:8787"
var bot_label: Label
var counts_label: Label


## The tune-o-clip root: the repo this app lives in (tune-o-clip/app/), else a tune-o-clip folder beside the app
## (the older Documents/clip-rate + Documents/tune-o-clip layout).
static func find_toc_root(app_dir: String) -> String:
	var parent := app_dir.path_join("..").simplify_path()
	if FileAccess.file_exists(parent.path_join("calibration_studio/clip_rate_worker.py")):
		return parent
	return parent.path_join("tune-o-clip")


func _ready() -> void:
	_build_ui()
	load_ratings()
	get_window().files_dropped.connect(_on_files_dropped)
	refresh_feed()
	rebuild_rated()
	refresh_bar()
	var timer := Timer.new()  # bot status + free disk, every 5 s
	timer.wait_time = 5.0
	timer.timeout.connect(refresh_bar)
	add_child(timer)
	timer.start()
	if autostart:
		run_worker(["--mode", "scan"])


# --- UI ------------------------------------------------------------------------------------------------------------

func _build_ui() -> void:
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	var bg := ColorRect.new()
	bg.color = BG
	bg.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(bg)
	var page := VBoxContainer.new()  # control bar on top, the three tabs below
	page.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(page)
	page.add_child(_build_bar())
	tabs = TabContainer.new()
	tabs.size_flags_vertical = Control.SIZE_EXPAND_FILL
	tabs.tab_changed.connect(_on_tab_changed)
	page.add_child(tabs)
	for tab: Control in [_build_feed_tab(), _build_rated_tab(), _build_request_tab()]:
		var pad := MarginContainer.new()
		pad.name = tab.name  # the tab's title
		for side in ["left", "right", "top", "bottom"]:
			pad.add_theme_constant_override("margin_" + side, 14)
		pad.add_child(tab)
		tabs.add_child(pad)
	send_dialog = AcceptDialog.new()
	send_dialog.title = "Prompt copied: paste it into Claude Code (in the tune-o-clip folder)"
	send_dialog.ok_button_text = "Close"
	send_dialog.add_button("Open report", true, "open")
	send_dialog.custom_action.connect(func(_a: StringName) -> void: OS.shell_open(_last_md))
	send_text = TextEdit.new()
	send_text.editable = false
	send_text.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	send_text.custom_minimum_size = Vector2(640, 260)
	send_dialog.add_child(send_text)
	add_child(send_dialog)
	file_dialog = FileDialog.new()
	file_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILES
	file_dialog.access = FileDialog.ACCESS_FILESYSTEM
	file_dialog.use_native_dialog = true
	file_dialog.filters = PackedStringArray(["*.mp4,*.mov,*.mkv,*.webm ; Video clips"])
	file_dialog.files_selected.connect(func(paths: PackedStringArray) -> void: upload(paths))
	add_child(file_dialog)


func _build_bar() -> Control:
	var pad := MarginContainer.new()
	for side in ["left", "right"]:
		pad.add_theme_constant_override("margin_" + side, 14)
	pad.add_theme_constant_override("margin_top", 6)
	var bar := HBoxContainer.new()
	bar.add_theme_constant_override("separation", 8)
	pad.add_child(bar)
	bot_label = _label("", DIM)
	bot_label.custom_minimum_size.x = 150
	bar.add_child(bot_label)
	if OS.get_name() == "Linux":
		bar.add_child(_button("Start Bot", "Run the 24/7 clip bot now and after every reboot", bot_action.bind("start")))
		bar.add_child(_button("Stop Bot", "Stop it (and don't start it on boot)", bot_action.bind("stop")))
		bar.add_child(_button("Restart Bot", "Restart it (e.g. after editing .env or the streamer list)", bot_action.bind("restart")))
	bar.add_child(_button("Open Web Dashboard", "The calibration studio in your browser (%s)" % DASHBOARD, open_dashboard))
	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	bar.add_child(spacer)
	counts_label = _label("", DIM)
	bar.add_child(counts_label)
	return pad


## Bot service state (Linux) + unrated shorts + free disk; every 5 s and after each bot button.
func refresh_bar() -> void:
	if OS.get_name() == "Linux":
		var out := []
		OS.execute("systemctl", PackedStringArray(["--user", "is-active", BOT_SERVICE]), out)
		var state := str(out[0]).strip_edges() if not out.is_empty() else ""
		var shown: Array = {"active": ["Bot: Running", GOOD], "activating": ["Bot: Starting…", ACCENT],
				"failed": ["Bot: Crashed (Restart)", BAD]}.get(state, ["Bot: Stopped", DIM])
		bot_label.text = shown[0]
		bot_label.add_theme_color_override("font_color", shown[1])
	else:
		bot_label.text = "Bot: runs on the Linux mini PC"
	_update_counts()


func _update_counts() -> void:
	var dir := DirAccess.open(toc_root)
	var free := "%.1f GB free" % (dir.get_space_left() / 1e9) if dir != null else "disk: ?"
	counts_label.text = "%d unrated · %s" % [feed.size(), free]


func bot_action(action: String) -> void:
	var out := []
	var args := PackedStringArray(["--user"])
	args.append_array(PackedStringArray(BOT_COMMANDS[action]))
	args.append(BOT_SERVICE)
	var code := OS.execute("systemctl", args, out, true)
	var said := "".join(PackedStringArray(out)).strip_edges()
	_log("Bot %s: %s" % [action, "done" if code == 0 else said], GOOD if code == 0 else BAD)
	if code != 0:
		_status("Bot %s failed: %s" % [action, said], BAD)
	refresh_bar()


## Starts the studio server if it isn't up (a second copy can't bind the port and just exits), then opens it.
func open_dashboard() -> void:
	OS.create_process(python_path(), ["-u", toc_root.path_join("calibration_studio/review_server.py"), "--no-browser"])
	get_tree().create_timer(1.5).timeout.connect(func() -> void: OS.shell_open(DASHBOARD))


func _build_feed_tab() -> Control:
	var tab := HBoxContainer.new()
	tab.name = "Scroller"
	tab.add_theme_constant_override("separation", 24)
	var left := VBoxContainer.new()
	left.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	left.alignment = BoxContainer.ALIGNMENT_CENTER
	tab.add_child(left)
	var head := HBoxContainer.new()
	left.add_child(head)
	counter = _label("", ACCENT)
	counter.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	head.add_child(counter)
	head.add_child(_button("▲", "Previous (Up / wheel)", func() -> void: go(-1)))
	head.add_child(_button("▼", "Next (Down / wheel)", func() -> void: go(1)))
	head.add_child(_button("+ Fetch 5 More", "5 more shorts per checked streamer (Request tab)", func() -> void: fetch(5)))
	var center := HBoxContainer.new()
	center.alignment = BoxContainer.ALIGNMENT_CENTER
	center.size_flags_vertical = Control.SIZE_EXPAND_FILL
	left.add_child(center)
	player = ClipPlayer.new()
	player.custom_minimum_size = Vector2(380, 0)
	player.clicked.connect(func() -> void: player.toggle())
	center.add_child(player)
	player.view.gui_input.connect(_on_feed_wheel)
	feed_status = _label("", DIM)
	feed_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	left.add_child(feed_status)

	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	var right := VBoxContainer.new()  # clip info scrolls; the rating form below it never leaves the screen
	right.custom_minimum_size.x = 440
	tab.add_child(right)
	right.add_child(scroll)
	var side := VBoxContainer.new()
	side.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	side.add_theme_constant_override("separation", 6)
	scroll.add_child(side)
	info_streamer = _label("", ACCENT)
	info_streamer.add_theme_font_size_override("font_size", 20)
	side.add_child(info_streamer)
	info_title = _wrap("")
	info_title.add_theme_font_size_override("font_size", 17)
	side.add_child(info_title)
	info_twitch = _wrap("", DIM)
	side.add_child(info_twitch)
	var row := HBoxContainer.new()
	side.add_child(row)
	info_badge = _label("")
	var badge := StyleBoxFlat.new()
	badge.bg_color = ACCENT.darkened(0.55)
	badge.set_corner_radius_all(9)
	badge.content_margin_left = 8
	badge.content_margin_right = 8
	info_badge.add_theme_stylebox_override("normal", badge)
	row.add_child(info_badge)
	info_stats = _label("", DIM)
	row.add_child(info_stats)
	var grid := GridContainer.new()
	grid.columns = 2
	side.add_child(grid)
	for p: Array in PILLARS:
		grid.add_child(_label(p[1], DIM))
		var bar := ProgressBar.new()
		bar.max_value = 10
		bar.custom_minimum_size = Vector2(240, 18)
		bar.show_percentage = false
		bar.size_flags_vertical = Control.SIZE_SHRINK_CENTER
		grid.add_child(bar)
		var num := _label("", ACCENT)
		bar.add_child(num)
		num.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
		bar.value_changed.connect(func(v: float) -> void: num.text = "%d/10" % v if v > 0 else "n/a")
		pillar_bars[p[0]] = bar
	info_reason = _wrap("", DIM)
	side.add_child(info_reason)
	side.add_child(_label("Transcript", DIM))
	info_transcript = _wrap("")
	side.add_child(info_transcript)
	right.add_child(HSeparator.new())
	var form := VBoxContainer.new()
	right.add_child(form)

	var thumbs := HBoxContainer.new()
	form.add_child(thumbs)
	like_button = _button("👍 Like", "L", func() -> void: set_like(true))
	dislike_button = _button("👎 Dislike", "D", func() -> void: set_like(false))
	for b in [like_button, dislike_button]:
		b.toggle_mode = true
		b.custom_minimum_size.x = 110
		thumbs.add_child(b)
	var stars := HBoxContainer.new()
	form.add_child(stars)
	for n in range(1, 6):
		var s := _button("☆", "%d star%s (key %d)" % [n, "" if n == 1 else "s", n], set_stars.bind(n))
		s.add_theme_font_size_override("font_size", 26)
		s.flat = true
		stars.add_child(s)
		star_buttons.append(s)
	var flow := HFlowContainer.new()
	form.add_child(flow)
	for t: String in TAGS:
		var b := _button(t, "Quick tag (click again to remove)", toggle_tag.bind(t))
		b.toggle_mode = true
		flow.add_child(b)
		tag_buttons[t] = b
	liked_box = _note_box(form, "What I Liked", "What worked? (e.g. tight hook, funny scream, good zoom)", GOOD)
	disliked_box = _note_box(form, "What I Didn't Like", "What missed? (e.g. cut 3s too late, dead air, wrong framing)", BAD)
	extra_box = _note_box(form, "Extra Notes", "Any extra context or rules for Claude...", DIM)
	var save := _button("Save Rating  (Enter)", "Moves the clip to the Rated tab", save_rating)
	save.add_theme_color_override("font_color", GOOD)
	form.add_child(save)
	save_hint = _wrap("Keys: ↑/↓ or wheel scroll · Space play/pause · ←/→ seek · 1-5 stars · L like · D dislike · "
			+ "Enter save (Ctrl+Enter while typing) · Esc leaves a text box", DIM)
	save_hint.add_theme_font_size_override("font_size", 12)
	form.add_child(save_hint)
	return tab


func _build_rated_tab() -> Control:
	var tab := VBoxContainer.new()
	tab.name = "Rated"
	var head := HBoxContainer.new()
	tab.add_child(head)
	rated_count = _label("", ACCENT)
	rated_count.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	head.add_child(rated_count)
	send_button = _button("Send to Claude", "Writes CLAUDE_TRAINING_BATCH.md/.json and copies the prompt", send_to_claude)
	send_button.add_theme_font_size_override("font_size", 20)
	send_button.custom_minimum_size = Vector2(220, 44)
	head.add_child(send_button)
	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	tab.add_child(scroll)
	rated_list = VBoxContainer.new()
	rated_list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(rated_list)
	return tab


func _build_request_tab() -> Control:
	var tab := VBoxContainer.new()
	tab.name = "Request & Upload"
	tab.add_theme_constant_override("separation", 10)
	tab.add_child(_label("Fetch & edit Twitch clips", ACCENT))
	var flow := HFlowContainer.new()
	tab.add_child(flow)
	for s: String in STREAMERS:
		var c := CheckBox.new()
		c.text = s
		c.button_pressed = true
		flow.add_child(c)
		streamer_checks[s] = c
	custom_streamers = LineEdit.new()
	custom_streamers.placeholder_text = "more streamers: login, twitch.tv/name, …"
	custom_streamers.custom_minimum_size.x = 320
	flow.add_child(custom_streamers)
	var row := HBoxContainer.new()
	tab.add_child(row)
	row.add_child(_label("Clips per streamer"))
	per_spin = SpinBox.new()
	per_spin.min_value = 1
	per_spin.max_value = 20
	per_spin.value = 5
	row.add_child(per_spin)
	fetch_button = _button("Fetch & Edit Clips", "Downloads, judges and renders one clip at a time",
			func() -> void: fetch(int(per_spin.value)))
	row.add_child(fetch_button)
	tab.add_child(HSeparator.new())
	tab.add_child(_label("Upload your own clip", ACCENT))
	var up := HBoxContainer.new()
	tab.add_child(up)
	up.add_child(_button("Choose video file…", "Or drag .mp4 files onto this window", func() -> void:
		file_dialog.popup_centered_ratio(0.6)))
	url_edit = LineEdit.new()
	url_edit.placeholder_text = "…or paste a Twitch clip URL"
	url_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	url_edit.text_submitted.connect(func(_t: String) -> void: upload_url())
	up.add_child(url_edit)
	up.add_child(_button("Process URL", "", upload_url))
	tab.add_child(_label("Drop video files anywhere on this window to upload them.", DIM))
	tab.add_child(HSeparator.new())
	var stages := HBoxContainer.new()
	stages.add_theme_constant_override("separation", 18)
	tab.add_child(stages)
	for s: String in STAGES:
		var l := _label(s, DIM)
		stages.add_child(l)
		stage_labels.append(l)
	var prow := HBoxContainer.new()
	tab.add_child(prow)
	progress = ProgressBar.new()
	progress.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	progress.custom_minimum_size.y = 22
	prow.add_child(progress)
	cancel_button = _button("Cancel", "Stops the worker and everything it started", cancel_worker)
	cancel_button.disabled = true
	prow.add_child(cancel_button)
	progress_label = _label("Idle", DIM)
	tab.add_child(progress_label)
	log_view = RichTextLabel.new()
	log_view.size_flags_vertical = Control.SIZE_EXPAND_FILL
	log_view.scroll_following = true
	log_view.selection_enabled = true
	tab.add_child(log_view)
	return tab


## A small labelled, word-wrapping box (3 lines, scrolls past that) for the rating panel.
func _note_box(parent: Control, title: String, placeholder: String, color: Color) -> TextEdit:
	parent.add_child(_label(title, color))
	var box := TextEdit.new()
	box.placeholder_text = placeholder
	box.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	box.custom_minimum_size = Vector2(0, 56)  # two wrapped lines, scrolls past that
	parent.add_child(box)
	return box


func _label(text: String, color: Color = Color.WHITE) -> Label:
	var l := Label.new()
	l.text = text
	l.add_theme_color_override("font_color", color)
	return l


func _wrap(text: String, color: Color = Color.WHITE) -> Label:
	var l := _label(text, color)
	l.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	l.custom_minimum_size.x = 400
	return l


func _button(text: String, tip: String, action: Callable) -> Button:
	var b := Button.new()
	b.text = text
	b.tooltip_text = tip
	b.focus_mode = Control.FOCUS_NONE  # keys drive the scroller, never a focused button
	b.pressed.connect(action)
	return b


func _log(text: String, color: Color = DIM) -> void:
	log_view.push_color(color)
	log_view.add_text(text + "\n")
	log_view.pop()
	while log_view.get_paragraph_count() > LOG_LINES:
		log_view.remove_paragraph(0)


# --- feed ----------------------------------------------------------------------------------------------------------

func add_card(c: Dictionary) -> void:
	var key := str(c.get("key", ""))
	if key.is_empty():
		return
	for i in cards.size():
		if cards[i].key == key:
			cards[i] = c
			refresh_feed()
			return
	cards.append(c)
	refresh_feed()


## Rebuilds the unrated feed, staying on the same short when it is still there.
func refresh_feed(show_key: String = "") -> void:
	var keep := show_key if not show_key.is_empty() else current_key()
	feed = cards.filter(func(c: Dictionary) -> bool: return not ratings.has(c.key))
	var at := -1
	for i in feed.size():
		if feed[i].key == keep:
			at = i
	index = at if at >= 0 else clampi(index, 0, maxi(0, feed.size() - 1))
	if current_key() != _shown_key or feed.is_empty():
		show_card()
	else:
		_update_counter()  # a new short arrived behind the one playing: don't restart it


func current_key() -> String:
	return str(feed[index].key) if index < feed.size() else ""


func current() -> Dictionary:
	return feed[index] if index < feed.size() else {}


func go(delta: int) -> void:
	if feed.is_empty():
		return
	var to := clampi(index + delta, 0, feed.size() - 1)
	if to != index:
		index = to
		show_card()


func _update_counter() -> void:
	counter.text = "%d / %d unrated" % [index + 1, feed.size()] if not feed.is_empty() else "No unrated shorts"
	_update_counts()


func show_card() -> void:
	_update_counter()
	var c := current()
	_shown_key = str(c.get("key", ""))
	_load_draft(drafts.get(_shown_key, {}))
	if c.is_empty():
		player.close()
		player.message.text = "Nothing to rate.\n\nFetch new clips with “+ Fetch 5 More” or upload one in Request & Upload." \
				if _mode != "scan" else "Loading shorts…"
		for l: Label in [info_streamer, info_title, info_twitch, info_badge, info_stats, info_reason, info_transcript]:
			l.text = ""
		for bar: ProgressBar in pillar_bars.values():
			bar.value = 0
		return
	var cl: Dictionary = c.get("clipper", {})
	var ed: Dictionary = c.get("editor", {})
	info_streamer.text = str(c.get("streamer", "")) + "  ·  " + str(c.get("batch", ""))
	info_title.text = ai_title(c)
	info_twitch.text = "Twitch: " + str(c.get("title", ""))
	info_badge.text = str(ed.get("scenario", "?"))
	var qc: Dictionary = ed.get("qc", {}) if ed.get("qc") is Dictionary else {}
	var dur: float = float(qc.get("duration") if qc.get("duration") != null else cl.get("duration", 0))
	var judge: Variant = cl.get("judge_10")
	info_stats.text = "  %.1f s · score %.2f · judge %s · %s" % [dur, float(cl.get("final_score", 0) if cl.get("final_score") != null else 0),
			"%s/10" % judge if judge != null else "n/a", str(ed.get("layout", ""))]
	var pillars: Dictionary = cl.get("pillars") if cl.get("pillars") is Dictionary else {}
	for k: String in pillar_bars:
		pillar_bars[k].value = float(pillars.get(k, 0) if pillars.get(k) != null else 0)
		pillar_bars[k].tooltip_text = "%s/10" % pillars.get(k, "?")
	info_reason.text = "Judge: " + str(cl.get("judge_reason", "")) if cl.get("judge_reason") else ""
	info_transcript.text = str(c.get("transcript", "")) if c.get("transcript") else "(no transcript)"
	if tabs.current_tab == 0:
		player.open(str(c.short), dur)


## The director's social title without its hashtags (the Twitch title when the edit has none).
static func ai_title(c: Dictionary) -> String:
	var ed: Dictionary = c.get("editor", {}) if c.get("editor") is Dictionary else {}
	var t := str(ed.get("title", "")) if ed.get("title") else str(c.get("title", ""))
	var cut := t.find(" #")
	return t.substr(0, cut) if cut > 0 else t


func _on_tab_changed(tab: int) -> void:
	if tab == 0:
		if not current().is_empty() and (player.state == "closed" or player.path != str(current().short)):
			show_card()
	else:
		player.pause()


func _on_feed_wheel(ev: InputEvent) -> void:
	if ev is InputEventMouseButton and ev.pressed:
		var now := Time.get_ticks_msec()
		if ev.button_index in [MOUSE_BUTTON_WHEEL_UP, MOUSE_BUTTON_WHEEL_DOWN] and now - _wheel_ms > 300:
			_wheel_ms = now  # one notch = one short, however many events a smooth wheel sends
			go(-1 if ev.button_index == MOUSE_BUTTON_WHEEL_UP else 1)


func _input(ev: InputEvent) -> void:
	if tabs.current_tab != 0 or not ev is InputEventKey or not ev.pressed or ev.echo:
		return
	var key := (ev as InputEventKey).keycode
	var focus := get_viewport().gui_get_focus_owner()
	if focus is LineEdit or focus is TextEdit or focus is SpinBox:
		if key == KEY_ESCAPE:
			focus.release_focus()
			get_viewport().set_input_as_handled()
		elif (key == KEY_ENTER or key == KEY_KP_ENTER) and ev.is_command_or_control_pressed() and focus is TextEdit:
			save_rating()  # Enter alone is a new line in the feedback boxes
			get_viewport().set_input_as_handled()
		return  # typing: the box gets every other key
	match key:
		KEY_UP:
			go(-1)
		KEY_DOWN:
			go(1)
		KEY_SPACE:
			player.toggle()
		KEY_LEFT:
			player.seek(player.clock() - 3.0)
		KEY_RIGHT:
			player.seek(player.clock() + 3.0)
		KEY_1, KEY_2, KEY_3, KEY_4, KEY_5:
			set_stars(key - KEY_0)
		KEY_L:
			set_like(true)
		KEY_D:
			set_like(false)
		KEY_ENTER, KEY_KP_ENTER:
			save_rating()
		_:
			return
	get_viewport().set_input_as_handled()


# --- rating --------------------------------------------------------------------------------------------------------

func set_like(v: Variant) -> void:
	cur_like = null if cur_like == v else v  # pressing the active thumb again clears it
	like_button.set_pressed_no_signal(cur_like == true)
	dislike_button.set_pressed_no_signal(cur_like == false)


func set_stars(n: int) -> void:
	cur_stars = 0 if cur_stars == n else n
	for i in star_buttons.size():
		star_buttons[i].text = "★" if i < cur_stars else "☆"
		star_buttons[i].add_theme_color_override("font_color", Color("facc15") if i < cur_stars else DIM)


func toggle_tag(t: String) -> void:
	if t in cur_tags:
		cur_tags.erase(t)
	else:
		cur_tags.append(t)
	tag_buttons[t].set_pressed_no_signal(t in cur_tags)


func _load_draft(d: Dictionary) -> void:
	cur_like = null
	set_like(d.get("like"))
	cur_stars = 0
	set_stars(int(d.get("stars", 0) if d.get("stars") != null else 0))
	cur_tags.clear()
	for t: String in TAGS:
		tag_buttons[t].set_pressed_no_signal(false)
	for t: Variant in d.get("tags", []):
		if str(t) in TAGS:
			toggle_tag(str(t))
	liked_box.text = str(d.get("liked", ""))
	disliked_box.text = str(d.get("disliked", ""))
	extra_box.text = str(d.get("extra_notes", d.get("notes", "")))  # ratings saved before the three boxes


## Saves the current short's rating and moves on. -> false (with a hint) when there is nothing to save.
func save_rating() -> bool:
	var c := current()
	if c.is_empty():
		return false
	if cur_stars == 0 and cur_like == null:
		save_hint.text = "Pick 1-5 stars or a thumb first."
		save_hint.add_theme_color_override("font_color", BAD)
		return false
	save_hint.add_theme_color_override("font_color", DIM)
	var key := str(c.key)
	ratings[key] = {"batch": c.get("batch"), "id": c.get("id"), "title": ai_title(c), "streamer": c.get("streamer"),
			"scenario": c.get("editor", {}).get("scenario"), "short": c.get("short"), "thumb": c.get("thumb"),
			"stars": cur_stars if cur_stars > 0 else null, "like": cur_like, "tags": cur_tags.duplicate(),
			"liked": liked_box.text.strip_edges(), "disliked": disliked_box.text.strip_edges(),
			"extra_notes": extra_box.text.strip_edges(), "status": "rated", "rated_at": Time.get_datetime_string_from_system()}
	drafts.erase(key)
	for box: TextEdit in [liked_box, disliked_box, extra_box]:
		box.release_focus()
	if not save_ratings():
		ratings.erase(key)
		return false
	save_hint.text = "Saved. %d rated, waiting for Send to Claude." % rated_keys().size()
	refresh_feed(str(feed[index + 1].key) if index + 1 < feed.size() else "")  # the next short slides in
	rebuild_rated()
	return true


func rated_keys() -> Array:
	return ratings.keys().filter(func(k: String) -> bool: return ratings[k].get("status") == "rated")


## Back into the scroller with its rating pre-filled.
func edit_rating(key: String) -> void:
	if not ratings.has(key):
		return
	drafts[key] = ratings[key]
	ratings.erase(key)
	save_ratings()
	tabs.current_tab = 0
	refresh_feed(key)
	rebuild_rated()


func remove_rating(key: String) -> void:
	ratings.erase(key)
	save_ratings()
	refresh_feed()
	rebuild_rated()


func rebuild_rated() -> void:
	for n in rated_list.get_children():
		n.queue_free()
	var keys := rated_keys()
	var trained := ratings.size() - keys.size()
	rated_count.text = "%d rated, ready to send%s" % [keys.size(), " · %d trained (archived)" % trained if trained else ""]
	send_button.disabled = keys.is_empty() or _pid > 0
	for key: String in keys:
		rated_list.add_child(_rated_row(key, ratings[key]))


func _rated_row(key: String, r: Dictionary) -> Control:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 12)
	var thumb := TextureRect.new()
	thumb.custom_minimum_size = Vector2(54, 96)
	thumb.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	thumb.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	var jpg := str(r.get("thumb", "")) if r.get("thumb") else ""
	if not jpg.is_empty() and FileAccess.file_exists(jpg):
		var img := Image.load_from_file(jpg)
		if img != null:
			thumb.texture = ImageTexture.create_from_image(img)
	row.add_child(thumb)
	var col := VBoxContainer.new()
	col.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(col)
	col.add_child(_label("%s  ·  %s  ·  %s" % [r.get("streamer", ""), r.get("title", ""), r.get("scenario", "")]))
	var stars := int(r.stars) if r.get("stars") != null else 0
	var thumb_txt := {true: "👍", false: "👎"}.get(r.get("like"), "") as String
	col.add_child(_label("%s %s  %s" % ["★".repeat(stars) + "☆".repeat(5 - stars) if stars else "no stars", thumb_txt,
			", ".join(PackedStringArray(r.get("tags", [])))], Color("facc15")))
	for f: Array in [["liked", "Liked", GOOD], ["disliked", "Didn't like", BAD], ["extra_notes", "Notes", DIM], ["notes", "Notes", DIM]]:
		if str(r.get(f[0], "")).strip_edges() != "":
			col.add_child(_wrap("%s: %s" % [f[1], r[f[0]]], f[2]))
	var edit := _button("Edit", "Back to the scroller with this rating", edit_rating.bind(key))
	var rm := _button("Remove", "Drop the rating (the short goes back to unrated)", remove_rating.bind(key))
	rm.add_theme_color_override("font_color", BAD)
	for b in [edit, rm]:
		b.size_flags_vertical = Control.SIZE_SHRINK_CENTER
		row.add_child(b)
	return row


func load_ratings() -> void:
	ratings = {}
	if not FileAccess.file_exists(ratings_path):
		return
	var json := JSON.new()
	if json.parse(FileAccess.get_file_as_string(ratings_path)) != OK or not json.data is Dictionary:
		# never overwrite ratings we could not read: keep the file aside and start a fresh queue
		var aside := ratings_path + ".unreadable-%d" % Time.get_unix_time_from_system()
		DirAccess.rename_absolute(ratings_path, aside)
		_log("ratings_queue.json was unreadable; kept it as %s" % aside.get_file(), BAD)
		return
	ratings = json.data


## Temp file + rename: a crash mid-write never leaves half a ratings file. -> false (logged) on failure.
func save_ratings() -> bool:
	DirAccess.make_dir_recursive_absolute(ratings_path.get_base_dir())
	var tmp := ratings_path + ".tmp"
	var f := FileAccess.open(tmp, FileAccess.WRITE)
	if f == null:
		_log("Could not write %s (%s)" % [tmp, error_string(FileAccess.get_open_error())], BAD)
		return false
	f.store_string(JSON.stringify(ratings, "\t"))
	f.close()
	if DirAccess.rename_absolute(tmp, ratings_path) != OK:
		_log("Could not replace %s" % ratings_path, BAD)
		return false
	return true


# --- Send to Claude ------------------------------------------------------------------------------------------------

func send_to_claude() -> void:
	if not rated_keys().is_empty():
		run_worker(["--mode", "send", "--ratings", ratings_path])


func on_sent(ev: Dictionary) -> void:
	var now := Time.get_datetime_string_from_system()
	for key: Variant in ev.get("keys", []):
		if ratings.has(key):
			ratings[key].status = "trained"
			ratings[key].trained_at = now
	save_ratings()
	rebuild_rated()
	_last_md = str(ev.get("md", ""))
	var prompt := str(ev.get("prompt", ""))
	DisplayServer.clipboard_set(prompt)
	send_text.text = prompt
	if DisplayServer.get_name() != "headless":
		send_dialog.popup_centered()
	_log("Training batch written: %s (prompt copied)" % _last_md, GOOD)


# --- requests + uploads --------------------------------------------------------------------------------------------

func selected_streamers() -> PackedStringArray:
	var out := PackedStringArray()
	for s: String in streamer_checks:
		if streamer_checks[s].button_pressed:
			out.append(s)
	for raw in custom_streamers.text.replace(";", ",").split(",", false):
		var login := raw.strip_edges().trim_suffix("/").get_file().trim_prefix("@").to_lower()  # twitch.tv/name, @name
		if _login_re.search(login) and login not in out:
			out.append(login)
	return out


func fetch(per: int) -> void:
	var logins := selected_streamers()
	if logins.is_empty():
		_status("Pick at least one streamer in Request & Upload.", BAD)
		return
	run_worker(["--mode", "fetch", "--channels", ",".join(logins), "--per-channel", str(per)])


func upload(paths: PackedStringArray) -> void:
	for p in paths:
		_uploads.append(p)
	if _pid <= 0:
		_next_upload()
	else:
		_status("%d upload(s) queued: they start when the current job ends." % _uploads.size(), ACCENT)


func _next_upload() -> void:
	if not _uploads.is_empty():
		var p := _uploads[0]
		_uploads.remove_at(0)
		run_worker(["--mode", "upload", "--file", p])


func upload_url() -> void:
	var url := url_edit.text.strip_edges()
	if not (url.begins_with("https://") or url.begins_with("http://")):  # also keeps "-flags" out of argv
		_status("Paste a full clip URL (https://clips.twitch.tv/… or twitch.tv/<name>/clip/…).", BAD)
		return
	url_edit.text = ""
	upload(PackedStringArray([url]))


func _on_files_dropped(files: PackedStringArray) -> void:
	var vids := Array(files).filter(func(f: String) -> bool: return f.get_extension().to_lower() in ["mp4", "mov", "mkv", "webm"])
	if vids.is_empty():
		_status("Only video files (.mp4, .mov, .mkv, .webm) can be uploaded.", BAD)
		return
	upload(PackedStringArray(vids))


# --- worker process ------------------------------------------------------------------------------------------------

func python_path() -> String:
	var venv := toc_root.path_join(".venv/Scripts/python.exe" if OS.get_name() == "Windows" else ".venv/bin/python")
	return venv if FileAccess.file_exists(venv) else ("python" if OS.get_name() == "Windows" else "python3")


## Starts clip_rate_worker.py with `args` (one at a time). -> false when busy or Python would not start.
func run_worker(args: PackedStringArray) -> bool:
	if _pid > 0:
		_status("Busy with %s: wait for it or Cancel it in Request & Upload." % _mode, ACCENT)
		return false
	var argv := PackedStringArray(["-u", toc_root.path_join("calibration_studio/clip_rate_worker.py")])
	argv.append_array(args)
	var proc := OS.execute_with_pipe(python_path(), argv)  # blocking pipes, no console window
	if proc.is_empty():
		_status("Could not start Python (%s)." % python_path(), BAD)
		return false
	_pid = int(proc.pid)
	_mode = args[1]
	_cancelled = false
	_had_error = false
	_run_start_ms = Time.get_ticks_msec()
	progress.value = 0
	_set_stage(-1)
	cancel_button.disabled = false
	fetch_button.disabled = true
	send_button.disabled = true
	_status({"scan": "Loading shorts…", "fetch": "Fetching clips…", "upload": "Processing upload…",
			"send": "Writing the training batch…"}.get(_mode, _mode), ACCENT)
	_log("> clip_rate_worker " + " ".join(args), ACCENT)
	_thread = Thread.new()
	_thread.start(_read_output.bind(proc))
	return true


func _read_output(proc: Dictionary) -> void:
	var out: FileAccess = proc.stdio
	while true:
		var line := out.get_line()
		if out.get_error() != OK:
			break  # pipe closed: the worker exited
		handle_line.call_deferred(line)
	var err: FileAccess = proc.stderr
	while true:
		var line := err.get_line()
		if err.get_error() != OK:
			break
		if not line.strip_edges().is_empty():
			_log.call_deferred(line, DIM)
	_finish.call_deferred()


func handle_line(line: String) -> void:
	var json := JSON.new()
	if not line.begins_with("{") or json.parse(line) != OK or not json.data is Dictionary:
		if not line.strip_edges().is_empty():
			_log(line)
		return
	handle_event(json.data)


func handle_event(ev: Dictionary) -> void:
	var msg := str(ev.get("msg", ""))
	match str(ev.get("event", "")):
		"short_ready":
			add_card(ev)
		"scan_done":
			_log("%d shorts found." % int(ev.get("count", 0)), GOOD)
		"cr_progress":
			var total := maxi(1, int(ev.get("total", 1)))
			var done := int(ev.get("completed", 0))
			progress.value = 100.0 * done / total
			var eta := ""
			if done > 0 and done < total:
				var per := (Time.get_ticks_msec() - _run_start_ms) / 1000.0 / done
				eta = " · ETA %s" % ClipPlayer.mmss(per * (total - done))
			_status("%s: %d / %d shorts%s · %s" % [ev.get("stage", ""), done, total, eta, ev.get("current", "")], ACCENT)
		"stage", "progress":
			var s := stage_of(ev)
			if s >= 0:
				_set_stage(s)
			_log("[%s] %s" % [str(ev.get("stage", ev.get("phase", ""))).to_upper(), msg if msg else str(ev.get("title", ""))], ACCENT)
		"sent":
			on_sent(ev)
		"error":
			_had_error = true
			_status(msg, BAD)
			_log(msg, BAD)
		"warning":
			_log(msg, ACCENT)
		"log", "queue":
			_log(msg)


## Which of STAGES a clipper / editor event belongs to (-1 = none).
static func stage_of(ev: Dictionary) -> int:
	var text := (str(ev.get("stage", ev.get("phase", ""))) + " " + str(ev.get("msg", ""))).to_lower()
	if "render" in text:
		return 3
	if "ollama" in text or "directing" in text or "judge" in text:
		return 2
	if "whisper" in text or "scoring" in text:
		return 1
	if "download" in text or "copying" in text or "metadata" in text:
		return 0
	return -1


func _set_stage(s: int) -> void:
	for i in stage_labels.size():
		stage_labels[i].add_theme_color_override("font_color", ACCENT if i == s else (GOOD if i < s else DIM))
		stage_labels[i].text = ("▶ " if i == s else "") + STAGES[i]


func _finish() -> void:
	if _thread != null:
		_thread.wait_to_finish()
		_thread = null
	var was := _mode
	_pid = -1
	_mode = ""
	cancel_button.disabled = true
	fetch_button.disabled = false
	_set_stage(-1)
	if _cancelled:
		_status("Cancelled. Fetching again resumes where it stopped.", ACCENT)
	elif not _had_error:
		progress.value = 100
		_status("Done (%s): %d unrated shorts." % [was, feed.size()], GOOD)
	rebuild_rated()
	if feed.is_empty():
		show_card()  # replaces "Loading shorts…"
	if not _cancelled:
		_next_upload()


## Kills the worker AND its children (clipper, ffmpeg, yt-dlp): no orphans keep the GPU busy.
func cancel_worker() -> void:
	if _pid <= 0:
		return
	_cancelled = true
	_uploads.clear()
	if OS.get_name() == "Windows":
		OS.execute("taskkill", ["/T", "/F", "/PID", str(_pid)])
	else:  # the worker leads its own process group (os.setpgrp): SIGTERM the whole group, never Godot's
		OS.execute("kill", ["-TERM", "--", "-%d" % _pid])
	# the killed worker kept Ollama warm in VRAM and can't unload it any more: a tiny detached run does
	OS.create_process(python_path(), ["-u", toc_root.path_join("calibration_studio/clip_rate_worker.py"), "--mode", "unload"])
	_log("Cancelled.", ACCENT)


func _status(text: String, color: Color) -> void:
	progress_label.text = text
	progress_label.add_theme_color_override("font_color", color)
	feed_status.text = text
	feed_status.add_theme_color_override("font_color", color)


func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST and _pid > 0:
		cancel_worker()  # closing the window never leaves a render running

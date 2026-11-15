class_name ClipPlayer
extends VBoxContainer
## 9:16 MP4 player without a GDExtension or .ogv proxies: ffmpeg decodes raw RGB frames over a pipe (reader thread,
## PV_AHEAD frames of backpressure) while a second ffmpeg extracts a WAV that becomes the clock once loaded.
## Ported from tune-o-clip's preview player (godot_app/scripts/main.gd). Only one clip is ever open: open() kills the
## previous decoders and drops their buffers, so RAM stays at ~8 frames (~6 MB).

signal clicked

const FPS := 30.0  # the shorts are rendered at 30 fps: no dropped or doubled frames
const AHEAD := 8
const TIMEOUT_MS := 6000  # no first picture by then: show why instead of a black box
const SIZE := Vector2i(360, 640)

var state := "closed"  # closed | playing | paused | ended
var path := ""
var duration := 0.0
var loop := true
## (path, start_sec, size) -> [exe, args]; tests swap in something that never starts ffmpeg.
var command: Callable = ffmpeg_command

var view: TextureRect
var message: Label
var play_button: Button
var slider: HSlider
var time_label: Label
var audio: AudioStreamPlayer

var _pid := -1
var _thread: Thread
var _mutex := Mutex.new()
var _frames: Array[PackedByteArray] = []
var _eof := false
var _reading := false
var _next_pts := 0.0
var _pos := 0.0
var _anchor_ms := 0
var _audio_pid := -1
var _wav := ""
var _stale_wavs: PackedStringArray = []  # a killed ffmpeg keeps its WAV locked on Windows for a moment
var _start_ms := 0
var _got_frame := false
var _image: Image
var _texture: ImageTexture
var _opens := 0


func _init() -> void:  # UI exists as soon as .new() returns: main wires view.gui_input before adding us
	view = TextureRect.new()
	view.custom_minimum_size = Vector2(SIZE) * 0.95
	view.size_flags_vertical = Control.SIZE_EXPAND_FILL
	view.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	view.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	view.mouse_filter = Control.MOUSE_FILTER_STOP  # clicks + wheel (main connects view.gui_input)
	view.gui_input.connect(func(ev: InputEvent) -> void:
		if ev is InputEventMouseButton and ev.pressed and ev.button_index == MOUSE_BUTTON_LEFT:
			clicked.emit())
	add_child(view)
	message = Label.new()
	message.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	message.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	message.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	message.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	message.mouse_filter = Control.MOUSE_FILTER_IGNORE
	view.add_child(message)
	var bar := HBoxContainer.new()
	add_child(bar)
	play_button = Button.new()
	play_button.text = "PLAY"
	play_button.tooltip_text = "Space"
	play_button.focus_mode = Control.FOCUS_NONE  # Space toggles playback, never re-presses a button
	play_button.custom_minimum_size.x = 64
	play_button.pressed.connect(toggle)
	bar.add_child(play_button)
	slider = HSlider.new()
	slider.step = 0.01
	slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	slider.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	slider.focus_mode = Control.FOCUS_NONE
	slider.drag_ended.connect(func(_changed: bool) -> void: seek(slider.value))
	bar.add_child(slider)
	time_label = Label.new()
	time_label.text = "0:00 / 0:00"
	bar.add_child(time_label)
	audio = AudioStreamPlayer.new()
	add_child(audio)


func ffmpeg_command(p: String, start: float, size: Vector2i) -> Array:
	var vf := "fps=%d,scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2" % [
			int(FPS), size.x, size.y, size.x, size.y]
	# Windows: software decode on 2 threads (NVENC renders and the i7-860 keep the rest; d3d11va/cuda measured slower).
	# Linux: let ffmpeg use VAAPI (the mini PC's Vega 3) when it works, else it quietly decodes in software.
	var hw: Array = ["-hwaccel", "auto"] if OS.get_name() == "Linux" else []
	return ["ffmpeg", ["-v", "error", "-nostdin", "-threads", "2", "-flags", "low_delay", "-fflags", "nobuffer"] + hw +
			["-ss", "%.3f" % start, "-i", p, "-an", "-threads", "2", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]]


## Plays `p` from the start (closing whatever was open). duration: a hint until the audio says.
func open(p: String, duration_hint: float = 0.0) -> void:
	close()
	path = p
	duration = duration_hint
	_image = Image.create_empty(SIZE.x, SIZE.y, false, Image.FORMAT_RGB8)
	_texture = ImageTexture.create_from_image(_image)
	view.texture = _texture
	message.text = ""
	_pos = 0.0
	if not _start_frames(0.0):
		abort("ffmpeg could not be started (is it on PATH?)")
		return
	if DisplayServer.get_name() != "headless":  # tests never touch audio hardware
		_opens += 1
		_wav = OS.get_user_data_dir().path_join("clip_audio_%d.wav" % _opens)
		_audio_pid = OS.create_process("ffmpeg", ["-v", "error", "-nostdin", "-y", "-i", p, "-vn", "-ac", "2",
				"-ar", "44100", "-f", "wav", _wav])
	_set_state("playing")


func _start_frames(start: float) -> bool:
	_stop_frames()
	var cmd: Array = command.call(path, start, SIZE)
	var proc := OS.execute_with_pipe(cmd[0], cmd[1])
	if proc.is_empty():
		return false
	_pid = int(proc.pid)
	_frames.clear()
	_eof = false
	_reading = true
	_next_pts = start
	_start_ms = Time.get_ticks_msec()
	_got_frame = false
	_thread = Thread.new()
	_thread.start(_read.bind(proc, SIZE.x * SIZE.y * 3))
	return true


func _stop_frames() -> void:
	_reading = false
	if _pid > 0:
		OS.kill(_pid)  # closes the pipe, which ends the blocking read
	if _thread != null:
		_thread.wait_to_finish()
		_thread = null
	_pid = -1
	_mutex.lock()
	_frames.clear()
	_mutex.unlock()


func _read(proc: Dictionary, frame_bytes: int) -> void:
	var pipe: FileAccess = proc.stdio
	while _reading:
		var buf := PackedByteArray()
		while buf.size() < frame_bytes:  # pipe reads can come back short
			var part := pipe.get_buffer(frame_bytes - buf.size())
			if part.is_empty():
				break
			buf.append_array(part)
		if buf.size() < frame_bytes:
			break  # EOF, or killed
		_mutex.lock()
		_frames.append(buf)
		var full := _frames.size() >= AHEAD
		_mutex.unlock()
		while full and _reading:
			OS.delay_msec(4)
			_mutex.lock()
			full = _frames.size() >= AHEAD
			_mutex.unlock()
	_mutex.lock()
	_eof = true
	_mutex.unlock()


## Seconds into the clip: the audio clock once the WAV plays, the wall clock before that.
func clock(now_ms: int = -1) -> float:
	if state != "playing":
		return _pos
	if audio.playing:
		return audio.get_playback_position() + AudioServer.get_time_since_last_mix()
	now_ms = Time.get_ticks_msec() if now_ms < 0 else now_ms
	return _pos + (now_ms - _anchor_ms) / 1000.0


func _set_state(s: String) -> void:
	if state == "playing" and s != "playing":
		_pos = clock()
	state = s
	_anchor_ms = Time.get_ticks_msec()
	play_button.text = "PAUSE" if s == "playing" else "PLAY"
	if audio.stream != null:
		if s == "playing":
			audio.play(_pos)
		else:
			audio.stop()


func toggle() -> void:
	match state:
		"playing":
			_set_state("paused")
		"paused":
			_set_state("playing")
		"ended":
			seek(0.0)
			_set_state("playing")


func pause() -> void:
	if state == "playing":
		_set_state("paused")


func seek(t: float) -> void:
	if state == "closed":
		return
	t = clampf(t, 0.0, maxf(0.0, duration - 0.1)) if duration > 0 else maxf(0.0, t)
	_pos = t
	_anchor_ms = Time.get_ticks_msec()
	if state == "ended":
		state = "paused"
	_start_frames(t)
	if audio.stream != null and state == "playing":
		audio.play(t)
	_show_time(t)


## Kills both decoders, drops the frame buffers and the audio, deletes the temp WAV (or retries later).
func close() -> void:
	_stop_frames()
	if _audio_pid > 0 and OS.is_process_running(_audio_pid):
		OS.kill(_audio_pid)
	_audio_pid = -1
	audio.stop()
	audio.stream = null
	_image = null
	_texture = null
	view.texture = null
	if not _wav.is_empty():
		_stale_wavs.append(_wav)
		_wav = ""
	_clean_wavs()
	state = "closed"
	play_button.text = "PLAY"


func _clean_wavs() -> void:
	var left: PackedStringArray = []
	for w in _stale_wavs:
		if FileAccess.file_exists(w) and DirAccess.remove_absolute(w) != OK:
			left.append(w)
	_stale_wavs = left


func abort(why: String) -> void:
	close()
	message.text = "Can't play this short:\n" + why


func _exit_tree() -> void:
	close()


func _process(_delta: float) -> void:
	tick()


## Per frame: show the newest frame due at the clock, hand the clock to the WAV once ffmpeg has written it.
func tick(now_ms: int = -1) -> void:
	if not _stale_wavs.is_empty():
		_clean_wavs()
	if state == "closed":
		return
	if _audio_pid > 0 and not OS.is_process_running(_audio_pid):
		_audio_pid = -1
		var wav := AudioStreamWAV.load_from_file(_wav) if FileAccess.file_exists(_wav) else null
		if wav != null:  # no audio track: stays on the wall clock
			_pos = clock(now_ms)
			_anchor_ms = Time.get_ticks_msec()
			duration = wav.get_length()
			audio.stream = wav
			if state == "playing":
				audio.play(_pos)
	var t := clock(now_ms)
	var frame := PackedByteArray()
	_mutex.lock()
	while not _frames.is_empty() and _next_pts <= t + 0.5 / FPS:
		frame = _frames.pop_front()
		_next_pts += 1.0 / FPS
	var drained := _eof and _frames.is_empty()
	_mutex.unlock()
	if not frame.is_empty():
		_got_frame = true
		_image.set_data(SIZE.x, SIZE.y, false, Image.FORMAT_RGB8, frame)
		_texture.update(_image)
	elif not _got_frame:
		now_ms = Time.get_ticks_msec() if now_ms < 0 else now_ms
		if drained:
			abort("the decoder stopped without a picture (missing or broken file)")
			return
		if now_ms - _start_ms > TIMEOUT_MS:
			abort("no picture within %d s (system busy?)" % (TIMEOUT_MS / 1000))
			return
	if state == "playing" and drained and t >= _next_pts:
		if loop:  # scroller style: the short repeats until you move on
			seek(0.0)
		else:
			_set_state("ended")
	_show_time(clock(now_ms))


func _show_time(t: float) -> void:
	if duration > 0:
		slider.max_value = duration
	slider.set_value_no_signal(t)
	time_label.text = "%s / %s" % [mmss(t), mmss(duration)]


static func mmss(s: float) -> String:
	return "%d:%02d" % [int(s) / 60, int(s) % 60]

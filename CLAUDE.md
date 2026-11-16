# Tune-O-Clip project rules

- Keep `backend/` 100% portable between Windows 10 and Linux Mint XFCE: no OS-specific paths, shells or APIs; it must run headless from cron with zero changes.
- Zero paid API tokens at runtime: everything per video stays local (yt-dlp, faster-whisper, local Ollama); no cloud LLM calls. HTTP is stdlib urllib, no extra dependencies.
- `editor/` (director, renderer, qc, pipeline) is a standalone package sharing `.venv`; it must stay runnable headless via CLI. Encoder: h264_nvenc / h264_vaapi when a probe encode works, else libx264; never AV1.
- Keep H.264 preferred over AV1 in the yt-dlp format string (neither the GTX 1050 Ti nor the Ryzen V1202B hardware-decodes AV1).

- Linux deploy target: the Scala mini PC (Mint 22.3, Ryzen V120B 2C/4T, 6 GB RAM, Vega 3, no NVIDIA, 64 GB). `scorer.MINI_PC` (Linux + <= 8 GB RAM) turns on LOW_SPEC by itself; keep new code CPU-safe and disk-frugal there. Install/run: `setup_linux.sh`, `start_linux.sh`, `LINUX_DEPLOY.md` (shell scripts stay LF).

- RunPod (backend/runpod_manager.py): never more than one `tune-o-clip` Pod, gpuCount 1 / volumeInGb 0 / containerDisk <= 25, $RUNPOD_MAX_SESSION_USD spend cap + idle limit enforced on the Pod (watchdog) and in `status`. Tests must stay mock-only (test_runpod.py blocks the network). No paid LLM SDK / endpoint ever (test_no_paid_llm_calls.py).

Tests: `python backend/test_scorer.py`, `python backend/test_queue.py`, `python backend/test_twitch.py`, `python editor/test_editor.py`, `python calibration_studio/test_calibration.py`, `python calibration_studio/test_clip_rate.py`, `python backend/test_runpod.py`, `python backend/test_no_paid_llm_calls.py`, `python backend/test_benchmark.py`, `python backend/test_spikes.py`, `python backend/test_spike_clipper.py`, `python publisher/test_publisher.py` and `godot --headless --path godot_app -s res://tests/test_main.gd`.

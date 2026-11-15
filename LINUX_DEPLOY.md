# Deploying Tune-O-Clip to the Linux Mint mini PC

Target: Scala Media Player B396-H02 (Ryzen Embedded V120B 2C/4T, 6 GB RAM, Vega 3, 64 GB), Linux Mint 22.3 Xfce.
Nothing needs configuring for the hardware: on Linux with <= 8 GB RAM the code switches itself to CPU Whisper, a warm
Ollama, 4 ffmpeg threads, a 60 s Ollama timeout and VAAPI (Vega 3) or libx264 encoding.

**Once, on the mini PC** (Mint has no SSH server by default): `sudo apt install -y openssh-server`, and note its
user name + IP (`hostname -I`). Below, replace `you@192.168.1.50` with them.

## The 3 commands (Git Bash on the Windows PC)

```bash
tar -czf ~/tune-o-clip.tgz -C ~/Documents --exclude=.venv --exclude=output --exclude=ready_to_post --exclude=workspace --exclude=calibration_studio/runs --exclude=calibration_studio/.app_profile --exclude='tune-o-clip-(copy)' --exclude=__pycache__ tune-o-clip
scp ~/tune-o-clip.tgz you@192.168.1.50:~/
ssh -t you@192.168.1.50 'tar -xzf tune-o-clip.tgz && cd tune-o-clip && chmod +x setup_linux.sh start_linux.sh && ./setup_linux.sh'
```

The archive includes your `.env` (Twitch keys) and `cookies.txt`; it leaves out the Windows `.venv`, all videos and
the calibration runs (the taste profile and ratings in `calibration_studio/clip_rate/` do go along).
`setup_linux.sh` asks for your sudo password, then sets up swap, packages, Ollama + models, the Python venv and the
Whisper model, and prints a checklist. Running it again is safe.

No network between the PCs? Copy `tune-o-clip.tgz` with a USB stick and run the third command's part in quotes in
a terminal on the mini PC.

## Running it

```bash
./start_linux.sh                                  # in the foreground (Ctrl+C stops)
systemctl --user enable --now tune-o-clip         # 24/7, also after a reboot
journalctl --user -u tune-o-clip -f               # watch it
./start_linux.sh studio                           # calibration studio on http://127.0.0.1:8787 (mini PC only)
```

Streamers: `workspace/twitch_channels.txt`. Auto-posting is off; add it with e.g.
`./start_linux.sh --auto-post --post-platform youtube` (log in once first, at the mini PC's desktop since it opens a
browser: `.venv/bin/python publisher/youtube_poster.py --login`; or copy `workspace/youtube_token.json` over).
Updating later: repeat the 3 commands (your `.env`, `workspace/` and videos on the mini PC are kept).

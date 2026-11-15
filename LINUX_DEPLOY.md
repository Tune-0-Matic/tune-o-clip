# Deploying Tune-O-Clip to the Linux Mint mini PC

Target: Scala Media Player B396-H02 (Ryzen Embedded V120B 2C/4T, 6 GB RAM, Vega 3, 64 GB), Linux Mint 22.3 Xfce.
Nothing needs configuring for the hardware: on Linux with <= 8 GB RAM the code switches itself to CPU Whisper, a warm
Ollama, 4 ffmpeg threads, a 60 s Ollama timeout and VAAPI (Vega 3) or libx264 encoding.

The code lives in the private repo https://github.com/Tune-0-Matic/tune-o-clip. Secrets are NOT in git
(`.env`, `cookies.txt`, `workspace/` are ignored): they go straight from the Windows PC to the mini PC with `scp`.

## First install

On the mini PC (`remi@remi-desktop`):

```bash
sudo apt install -y git gh openssh-server
gh auth login            # GitHub.com -> HTTPS -> "Login with a web browser" (account Tune-0-Matic)
gh repo clone Tune-0-Matic/tune-o-clip ~/tune-o-clip
```

On the Windows PC (Git Bash): copy the keys + cookies over SSH (use the mini PC's IP if the name doesn't resolve):

```bash
scp ~/Documents/tune-o-clip/.env ~/Documents/tune-o-clip/cookies.txt remi@remi-desktop.local:~/tune-o-clip/
```

Back on the mini PC:

```bash
cd ~/tune-o-clip && ./setup_linux.sh
```

`setup_linux.sh` asks for your sudo password, then sets up swap, packages, Ollama + models, the Python venv and the
Whisper model, and prints a checklist. Running it again is safe.

## Or: from the Google Drive bundle (no SSH needed)

`tune-o-clip-bundle.zip` (built next to the repo on the Windows PC) holds the code, its git history (remote already
set to this repo), `.env` + a visible copy `env.txt`, and `cookies.txt`. Download it to `~/Downloads`, then:

```bash
unzip -oq ~/Downloads/tune-o-clip-bundle.zip -d ~
cd ~/tune-o-clip && ./setup_linux.sh
sudo apt install -y gh && gh auth login && git pull
```

The third line logs this machine into GitHub once (`gh auth login`: GitHub.com, HTTPS, "Login with a web browser",
yes to "Authenticate Git"), after which `git pull` updates it.

The bundle contains your keys and YouTube login cookies: keep it private on Drive (never "anyone with the link") and
delete it from Downloads once installed.

## Updating

Windows: commit + `git push`. Mini PC: `cd ~/tune-o-clip && git pull && ./setup_linux.sh` (then
`systemctl --user restart tune-o-clip` if it runs as a service). `.env`, `workspace/` and videos are never touched.

## The desktop app (no terminal needed)

`setup_linux.sh` downloads the official Godot 4.7.2 binary once into `bin/godot/` (~80 MB, not in git) and adds a
**Tune-O-Clip** launcher to the desktop and to Applications > Multimedia. It opens the tune-o-clip panel
(`godot_app/`, the same app as on Windows): Single / Batch / Twitch runs, SETTINGS, AUTO POST, the clip list with
the built-in preview player, and the telemetry sidebar. On this machine it runs the low-spec profile by itself (CPU
Whisper, warm Ollama, VAAPI or libx264). If Ollama is off at start-up it asks for your password once to start the
ollama service. Use either the panel or the 24/7 service at a time: both share workspace/queue.db and the 6 GB RAM.
If Xfce asks about an "untrusted launcher" on the first double-click, choose *Mark Executable* (only when setup ran
over SSH).

## Running it (terminal alternative)

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

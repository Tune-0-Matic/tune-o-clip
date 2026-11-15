#!/usr/bin/env bash
# Tune-O-Clip one-command setup for Linux Mint 22.x / Ubuntu 24.04, tuned for the Scala mini PC
# (Ryzen Embedded V120B 2C/4T, 6 GB RAM, Vega 3 iGPU, no NVIDIA, 64 GB drive).
# Idempotent: safe to run again any time, e.g. after copying a newer tune-o-clip over.
#
#   chmod +x setup_linux.sh start_linux.sh && ./setup_linux.sh
#
# Run it as your normal user (it asks for sudo where it needs it). The code then detects the low-RAM Linux box by
# itself (backend/scorer.py MINI_PC): Whisper on the CPU, Ollama kept warm, 4 ffmpeg threads, 60 s Ollama timeout,
# VAAPI (Vega 3) or libx264 encoding.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT="$PWD"
# Linux default judge / director model (~1 GB, about 2x faster than the 3B on the V120B's CPU). Written to .env as
# TOC_OLLAMA_MODEL. The taste profile was calibrated on richardyoung/qwen2.5-3b-instruct-abliterated: to use that
# instead, set TOC_OLLAMA_MODEL to it in .env and run this script again (it pulls whatever .env names).
OLLAMA_MODEL="qwen2.5:1.5b"
WHISPER="small.en"
SWAP_GB=4
say() { printf '\n\033[1;33m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;31m!! %s\033[0m\n' "$*"; }

if [ "$(id -u)" -eq 0 ]; then
  echo "Run this as your normal user, not root (it uses sudo where it needs it)."
  exit 1
fi
sudo -v

# --- 1. swap: 6 GB RAM + Ollama + Whisper + ffmpeg must never meet the OOM killer -----------------------------------
say "Swap: at least ${SWAP_GB} GB"
need_kb=$((SWAP_GB * 1024 * 1024 - 102400))  # a little under 4 GiB counts (swap headers)
total_kb=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
if [ "$total_kb" -lt "$need_kb" ]; then
  if [ -f /swapfile ] && [ "$(stat -c %s /swapfile)" -lt $((SWAP_GB * 1024 * 1024 * 1024)) ]; then
    echo "Growing the existing /swapfile to ${SWAP_GB} GB"  # Mint ships a 2 GB one
    sudo swapoff /swapfile 2>/dev/null || true
    sudo rm -f /swapfile
  fi
  if [ ! -f /swapfile ]; then
    sudo fallocate -l "${SWAP_GB}G" /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1M count=$((SWAP_GB * 1024)) status=none
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile >/dev/null
  fi
  sudo swapon /swapfile 2>/dev/null || true
  grep -q '^/swapfile[[:space:]]' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-tune-o-clip.conf >/dev/null
sudo sysctl -q -p /etc/sysctl.d/99-tune-o-clip.conf
swapon --show

# --- 2. system packages ---------------------------------------------------------------------------------------------
say "System packages"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip ffmpeg curl git ca-certificates \
  unzip fonts-dejavu-core fonts-liberation fontconfig mesa-va-drivers vainfo
# The caption fonts are Arial Black / Impact (editor/director.py FONTS). Microsoft's core fonts package has both; it
# needs its EULA accepted. Without it libass falls back to DejaVu Sans Bold: captions still render, just look different.
echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | sudo debconf-set-selections
if ! sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ttf-mscorefonts-installer; then
  warn "ttf-mscorefonts-installer failed (no multiverse / offline): captions will use DejaVu Sans Bold instead of Arial Black / Impact"
fi
fc-list | grep -qi "arial black" && echo "Caption fonts: Arial Black + Impact installed" || warn "Arial Black not found: DejaVu fallback"

# --- 3. Ollama: one model at a time, kept warm 5 min ----------------------------------------------------------------
say "Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=5m"
EOF
sudo systemctl daemon-reload
sudo systemctl enable ollama >/dev/null 2>&1 || true
sudo systemctl restart ollama
for _ in $(seq 1 60); do
  curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
  sleep 1
done
curl -fs http://127.0.0.1:11434/api/tags >/dev/null || { warn "Ollama did not come up on 127.0.0.1:11434 (see: journalctl -u ollama)"; exit 1; }
[ -f .env ] || { [ -f env.txt ] && mv env.txt .env; } || true  # the Drive bundle's visible copy of .env
[ -f .env ] || cp .env.example .env
sed -i '1s/^\xEF\xBB\xBF//; s/\r$//' .env  # a .env saved by Windows Notepad: BOM + CRLF
grep -q '^TOC_OLLAMA_MODEL=' .env || printf '\nTOC_OLLAMA_MODEL=%s\n' "$OLLAMA_MODEL" >> .env
OLLAMA_MODEL=$(sed -n 's/^TOC_OLLAMA_MODEL=//p' .env | tail -1 | tr -d '"\r')
ollama pull "$OLLAMA_MODEL" || warn "Could not pull $OLLAMA_MODEL: the bot uses any model Ollama has, or heuristics"

# --- 4. Python venv (CPU only: faster-whisper uses CTranslate2, no PyTorch / CUDA wheels) --------------------------
say "Python environment (.venv)"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --upgrade -r backend/requirements.txt  # --upgrade: always the latest yt-dlp
echo "Downloading the Whisper model ($WHISPER, ~480 MB) once..."
.venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('$WHISPER', device='cpu', compute_type='int8')"

# --- 5. config + launcher -------------------------------------------------------------------------------------------
say "Configuration"
mkdir -p workspace output ready_to_post
[ -s workspace/twitch_channels.txt ] || printf 'caseoh_\nishowspeed\njynxzi\nlospollostv\n' > workspace/twitch_channels.txt
chmod +x start_linux.sh
mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/tune-o-clip.service" <<EOF
[Unit]
Description=Tune-O-Clip 24/7 clip bot
After=network-online.target

[Service]
ExecStart=$ROOT/start_linux.sh
Restart=on-failure
RestartSec=60

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload 2>/dev/null || true
sudo loginctl enable-linger "$USER"  # the user service keeps running without anyone logged in

# --- 6. desktop app: the official Godot binary runs the tune-o-clip panel (godot_app/) directly: no export templates --
GODOT_VER="4.7.2-stable"  # the version the panel is built with (godot_app/project.godot)
say "Desktop app (Godot $GODOT_VER)"
GODOT="$ROOT/bin/godot/Godot_v${GODOT_VER}_linux.x86_64"
if [ ! -x "$GODOT" ]; then
  mkdir -p bin/godot
  curl -fL --retry 3 -o bin/godot/godot.zip \
    "https://github.com/godotengine/godot/releases/download/${GODOT_VER}/Godot_v${GODOT_VER}_linux.x86_64.zip"
  unzip -oq bin/godot/godot.zip -d bin/godot
  rm -f bin/godot/godot.zip
  chmod +x "$GODOT"
fi
"$GODOT" --headless --path godot_app --import >/dev/null 2>&1 || warn "Godot's first import failed: the app will import on its first launch"
launcher() {  # $1 = .desktop path; OpenGL 3.3 (Compatibility) suits the Vega 3 iGPU
  cat > "$1" <<EOF
[Desktop Entry]
Type=Application
Name=Tune-O-Clip
Comment=Clip YouTube videos and Twitch streamers into vertical shorts
Exec="$GODOT" --path "$ROOT/godot_app" --rendering-driver opengl3
Icon=$ROOT/godot_app/icon.png
Terminal=false
Categories=AudioVideo;Video;
EOF
  chmod +x "$1"
}
mkdir -p "$HOME/.local/share/applications"
launcher "$HOME/.local/share/applications/tune-o-clip.desktop"
DESK=$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")
mkdir -p "$DESK"
launcher "$DESK/Tune-O-Clip.desktop"
# Xfce 4.18 only runs a desktop launcher it trusts: GNOME-style flag + Xfce's checksum (needs the desktop session's
# D-Bus; over SSH this quietly fails and the first double-click asks "Mark executable" once instead)
gio set "$DESK/Tune-O-Clip.desktop" metadata::trusted true 2>/dev/null || true
gio set -t string "$DESK/Tune-O-Clip.desktop" metadata::xfce-exe-checksum \
  "$(sha256sum "$DESK/Tune-O-Clip.desktop" | cut -d' ' -f1)" 2>/dev/null || true
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
echo "Launchers: Applications menu > Multimedia > Tune-O-Clip, and Tune-O-Clip on the desktop"

say "Checklist"
missing=0
for key in TWITCH_CLIENT_ID TWITCH_CLIENT_SECRET; do
  if ! grep -Eq "^${key}=.+" .env; then
    warn "$key is empty in $ROOT/.env (required: https://dev.twitch.tv/console/apps)"
    missing=1
  fi
done
for key in DISCORD_WEBHOOK_URL YOUTUBE_CLIENT_ID YOUTUBE_CLIENT_SECRET TIKTOK_CLIENT_KEY TIKTOK_CLIENT_SECRET; do
  grep -Eq "^${key}=.+" .env || echo "   optional, empty: $key"
done
[ -f cookies.txt ] || echo "   optional: cookies.txt (only for YouTube links; Twitch needs none)"
echo "   streamers to poll: $(paste -sd, workspace/twitch_channels.txt)  (edit workspace/twitch_channels.txt)"
echo "   free disk: $(df -h --output=avail "$ROOT" | tail -1 | tr -d ' ')  (harvesting pauses under 5 GB)"
echo
if [ "$missing" -eq 0 ]; then
  echo "Ready. Double-click Tune-O-Clip on the desktop: Start Bot runs the 24/7 bot (and after every reboot)."
  echo "Terminal alternative: ./start_linux.sh   or   systemctl --user enable --now tune-o-clip (logs: journalctl --user -u tune-o-clip -f)"
else
  echo "Fill in .env, then double-click Tune-O-Clip on the desktop and press Start Bot."
fi
echo "Calibration studio web UI: ./start_linux.sh studio   (http://127.0.0.1:8787 on this machine)"

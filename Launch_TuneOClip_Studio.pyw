"""Double-click: Tune-O-Clip Calibration Studio in its own window, no console.
Errors go to calibration_studio/studio.log. Closing the window quits it a minute later (unless a harvest is running)."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv" / ("Scripts/pythonw.exe" if os.name == "nt" else "bin/python")
if VENV.exists() and Path(sys.executable).resolve() != VENV.resolve():  # double-clicked with the system Python
    subprocess.Popen([str(VENV), str(Path(__file__).resolve())], cwd=str(HERE))
    sys.exit(0)

log = open(HERE / "calibration_studio" / "studio.log", "a", encoding="utf-8", buffering=1)
sys.stdout = sys.stderr = log  # pythonw has no console to print to
sys.path.insert(0, str(HERE / "calibration_studio"))
import review_server  # noqa: E402

sys.exit(review_server.main(["--app"]))

"""Quality control for rendered shorts: ffprobe + one ffmpeg pass (volumedetect + blackdetect)."""
import re
import subprocess
from pathlib import Path

import renderer

LIMITS = {"min_bytes": 500_000, "width": 1080, "height": 1920, "min_dur": 7.0, "max_dur": 58.5,
          "min_db": -20.0, "max_db": -10.0, "black_max": 0.3}


def check(path, **limits):
    """-> {"ok", "problems": [codes], "duration", "mean_db", "black": [(start, end)], ...}.
    Codes: missing, too_small, resolution, too_short, too_long, too_quiet, too_loud, black."""
    lim = {**LIMITS, **limits}
    path = Path(path)
    report = {"ok": False, "problems": [], "path": str(path), "duration": 0.0, "mean_db": None, "black": []}
    if not path.exists():
        report["problems"].append("missing")
        return report
    report["bytes"] = path.stat().st_size
    if report["bytes"] < lim["min_bytes"]:
        report["problems"].append("too_small")
    info = renderer.probe(path)
    report.update(width=info["width"], height=info["height"], duration=round(info["duration"], 2))
    if (info["width"], info["height"]) != (lim["width"], lim["height"]):
        report["problems"].append("resolution")
    if info["duration"] < lim["min_dur"]:
        report["problems"].append("too_short")
    if info["duration"] > lim["max_dur"]:
        report["problems"].append("too_long")
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vf",
                          f"blackdetect=d={lim['black_max']}:pix_th=0.10", "-af", "volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    m = re.search(r"mean_volume:\s*(-?[\d.]+|-inf) dB", err)
    mean = float(m.group(1)) if m and m.group(1) != "-inf" else -99.0
    report["mean_db"] = mean
    if mean < lim["min_db"]:
        report["problems"].append("too_quiet")
    if mean > lim["max_db"]:
        report["problems"].append("too_loud")
    report["black"] = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", err)]
    if report["black"]:
        report["problems"].append("black")
    report["ok"] = not report["problems"]
    return report

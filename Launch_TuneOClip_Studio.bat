@echo off
rem Tune-O-Clip Calibration Studio: opens its own window, no console stays open.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" "Launch_TuneOClip_Studio.pyw"

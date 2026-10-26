@echo off
rem PyMC_Bot updater for Windows - runs updater.py with the best Python available.
rem
rem   updater              update PyMC_Bot (+ dependencies when they changed)
rem   updater --check      only show what would change
rem   updater --mod        also update the Fabric mod in %APPDATA%\.minecraft\mods
rem   updater --help       all options
rem
rem The update may replace this very file. cmd.exe reads batch files line by line, so every
rem line that starts Python also ends the script ("& exit /b") - nothing after it is read again.
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" updater.py %* & exit /b
)
where py >nul 2>nul
if not errorlevel 1 (
  py -3 updater.py %* & exit /b
)
where python >nul 2>nul
if not errorlevel 1 (
  python updater.py %* & exit /b
)
echo [update] ERROR: Python 3.10+ was not found - install it from https://www.python.org/downloads/
echo [update] (tick "Add python.exe to PATH" in the installer)
if "%~1"=="" pause
exit /b 1

@echo off
rem PyMC Bot - collect the Minecraftism dataset on this PC (scrapes pages for Kaggle; no LLM runs on this PC).
rem Double-click it. Extra options go to scripts\scrape_dataset_pc.py, e.g.:  collect_dataset_pc.bat --topics 10
rem Made with love by codero-sus - https://the-codero.web.app/
setlocal
title PyMC Bot - Minecraftism dataset
cd /d "%~dp0.."
set "VENV=.venv-dataset"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
if not exist "%VENV%\Scripts\python.exe" (
  echo Making a Python environment in %VENV% - once ...
  %PY% -m venv "%VENV%" || goto :nopython
)
echo Installing / updating ddgs, huggingface_hub, psutil ...
"%VENV%\Scripts\python.exe" -m pip install -q -U pip ddgs "huggingface_hub>=0.34" psutil || goto :failed
if not "%HF_TOKEN%"=="" goto :run
"%VENV%\Scripts\python.exe" -c "import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)" && goto :run
echo.
echo Log in to Hugging Face once: paste a WRITE token from https://huggingface.co/settings/tokens
echo It is saved on this PC only - the same account as the HF_TOKEN secret on Kaggle.
"%VENV%\Scripts\hf.exe" auth login || goto :failed
:run
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
"%VENV%\Scripts\python.exe" scripts\scrape_dataset_pc.py %* || goto :failed
echo.
echo Finished - scraped pages are on the Hub and in %USERPROFILE%\Minecraftism-scrape
pause
exit /b 0
:nopython
echo Python 3.10 or newer is needed: https://www.python.org/downloads/ - tick "Add python.exe to PATH".
pause
exit /b 1
:failed
echo Something above failed - see the message.
pause
exit /b 1

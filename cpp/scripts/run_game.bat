@echo off
REM =============================================================================
REM DouMH Botzone — Run Game Script
REM =============================================================================
REM Judge:  ..\..\..\judge\judge_botzone_ver.py
REM Bot:    ..\build\bot.exe
REM Runner: ..\tools\runner.py
REM Logs:   ..\build\game_state.log  ..\build\json_io.log  ..\build\bot_stderr.log
REM =============================================================================
REM Usage:
REM   run_game.bat                  Run 1 game (with logging, verbose)
REM   run_game.bat 50               Run 50 games (batch mode, no per-turn logs)
REM   run_game.bat --quiet          Run 1 game without verbose bot I/O
REM   run_game.bat 100 --quiet      Run 100 games quietly
REM =============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PYTHON=python"

REM Defaults
set "GAMES=1"
set "VERBOSE=--verbose"

REM Parse args
for %%a in (%*) do (
    echo %%a | findstr /r "^[0-9][0-9]*$" >nul && set "GAMES=%%a"
    if "%%a"=="--quiet" set "VERBOSE="
)

set "RUNNER=%SCRIPT_DIR%..\..\..\botzone\runner.py"
set "BOT=%SCRIPT_DIR%..\build\bot.exe"

if %GAMES% gtr 1 (
    REM Batch mode: no verbose, no per-turn logs by default
    echo Running %GAMES% games...
    "%PYTHON%" "%RUNNER%" --bot "%BOT%" --games %GAMES% --timeout 30
) else (
    REM Single game: verbose + logging
    "%PYTHON%" "%RUNNER%" --bot "%BOT%" %VERBOSE% --timeout 30
)

endlocal

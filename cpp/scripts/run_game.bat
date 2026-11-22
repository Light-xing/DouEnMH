@echo off
REM =============================================================================
REM run_game.bat — 用仓库内 runner 跑本地对局
REM =============================================================================
REM Runner: <repo>\tools\runner\runner.py  （内嵌裁判 + 子进程 bot）
REM Bot:    cpp\build\bot.exe
REM 日志:   cpp\build\bot_stderr_p*.log 等（runner 以 bot.exe 所在目录为工作目录）
REM
REM 用法:
REM   run_game.bat                 跑 1 局（verbose，带逐步日志）
REM   run_game.bat 50              跑 50 局（批量模式）
REM   run_game.bat --quiet         跑 1 局，不打印逐步 I/O
REM   run_game.bat 100 --quiet     跑 100 局，安静模式
REM =============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."
set "PYTHON=python"

REM Defaults
set "GAMES=1"
set "VERBOSE=--verbose"

REM Parse args
for %%a in (%*) do (
    echo %%a | findstr /r "^[0-9][0-9]*$" >nul && set "GAMES=%%a"
    if "%%a"=="--quiet" set "VERBOSE="
)

set "RUNNER=%REPO_ROOT%\tools\runner\runner.py"
set "BOT=%SCRIPT_DIR%..\build\bot.exe"

if not exist "%RUNNER%" (
    echo [ERROR] runner not found: %RUNNER%
    exit /b 1
)
if not exist "%BOT%" (
    echo [ERROR] bot not found: %BOT%
    echo   Build it first with build_local.bat, or run build.bat.
    exit /b 1
)

if %GAMES% gtr 1 (
    REM Batch mode: no verbose, no per-turn logs by default
    echo Running %GAMES% games...
    "%PYTHON%" "%RUNNER%" --bot "%BOT%" --games %GAMES% --timeout 30
) else (
    REM Single game: verbose + logging
    "%PYTHON%" "%RUNNER%" --bot "%BOT%" %VERBOSE% --timeout 30
)

endlocal

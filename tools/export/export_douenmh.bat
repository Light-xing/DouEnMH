@echo off
setlocal enabledelayedexpansion
set "EXIT_CODE=0"

REM ===========================================================================
REM export_douenmh.bat -- Export DouEnMH checkpoints to TorchScript .pt
REM ===========================================================================
REM Drives export_douenmh.py with a Python 3.8 + PyTorch 1.5.0 interpreter, so
REM that the resulting TorchScript can be loaded by Botzone's LibTorch 1.5.0.
REM See docs/model-export.md for the full export pipeline.
REM
REM The interpreter is taken from the BRIDGE_PYTHON environment variable when
REM set; otherwise the usual sibling environments\bridge_python layout is
REM probed. Set it explicitly if the environment lives elsewhere:
REM     set BRIDGE_PYTHON=<ENV>\python38_full\python.exe
REM
REM Features:
REM   - Auto-detects model architecture from checkpoint (d_model, layers, dim_ff, x_dim)
REM   - Auto-detects latest suffix if not specified
REM   - Exports 6 models: first/second/third + landlord/landlord_up/landlord_down
REM
REM Usage (run from anywhere; paths below are relative to this script):
REM     export_douenmh.bat
REM     export_douenmh.bat 620000000
REM     export_douenmh.bat 620000000 --fp16
REM     export_douenmh.bat 620000000 --fp16 --copy
REM
REM   First argument (optional): checkpoint suffix (frame number)
REM   --fp16:          Also export FP16 quantized versions
REM   --copy:          Copy .pt files to the C++ build model dir after export
REM   --checkpoint:    Override checkpoint directory
REM   --output:        Override output directory
REM ===========================================================================

set "SCRIPT_DIR=%~dp0"
set "EXPORT_PY=%SCRIPT_DIR%export_douenmh.py"

REM ---- Locate bridge Python (Python 3.8 + PyTorch 1.5.0) ----
if defined BRIDGE_PYTHON if exist "%BRIDGE_PYTHON%" goto :python_ready
if defined BRIDGE_PYTHON (
    echo [WARN] BRIDGE_PYTHON is set but does not exist: %BRIDGE_PYTHON%
    set "BRIDGE_PYTHON="
)
if defined BRIDGE_PYTHON_DIR if exist "%BRIDGE_PYTHON_DIR%\python38_full\python.exe" (
    for %%I in ("%BRIDGE_PYTHON_DIR%\python38_full\python.exe") do set "BRIDGE_PYTHON=%%~fI"
)
if not defined BRIDGE_PYTHON if exist "%SCRIPT_DIR%..\..\..\environments\bridge_python\python38_full\python.exe" (
    for %%I in ("%SCRIPT_DIR%..\..\..\environments\bridge_python\python38_full\python.exe") do set "BRIDGE_PYTHON=%%~fI"
)
if not defined BRIDGE_PYTHON (
    echo [ERROR] Cannot find a Python 3.8 + PyTorch 1.5.0 interpreter.
    echo         Set BRIDGE_PYTHON to its python.exe, for example:
    echo             set BRIDGE_PYTHON=^<ENV^>\python38_full\python.exe
    echo         or set BRIDGE_PYTHON_DIR to the environment directory that
    echo         contains python38_full\python.exe
    set "EXIT_CODE=1"
    goto :done
)

:python_ready
if not exist "%BRIDGE_PYTHON%" (
    echo [ERROR] Bridge Python not found: %BRIDGE_PYTHON%
    set "EXIT_CODE=1"
    goto :done
)
if not exist "%EXPORT_PY%" (
    echo [ERROR] export_douenmh.py not found: %EXPORT_PY%
    set "EXIT_CODE=1"
    goto :done
)

REM ---- Verify torch is importable and report its version ----
set "TORCH_VERSION="
for /f "delims=" %%v in ('^""%BRIDGE_PYTHON%" -c "import torch; print(torch.__version__)" 2^>nul^"') do set "TORCH_VERSION=%%v"
if not defined TORCH_VERSION (
    echo [ERROR] Cannot import torch with: %BRIDGE_PYTHON%
    echo         Install the Python 3.8 + PyTorch 1.5.0 bridge environment first.
    set "EXIT_CODE=1"
    goto :done
)
echo %TORCH_VERSION%|findstr /b "1.5." >nul
if errorlevel 1 (
    echo [WARN] PyTorch %TORCH_VERSION% is not 1.5.x; exported TorchScript may not
    echo        load under LibTorch 1.5.0. Use the bridge environment for real exports.
)

REM ---- Default paths (relative to this script; no absolute paths) ----
set "CHECKPOINT_DIR=%SCRIPT_DIR%..\..\python\douzero_checkpoints\DouEnMH_v1.2"
set "OUTPUT_DIR=%SCRIPT_DIR%export_model"

REM ---- Parse args ----
set "SUFFIX="
set "FP16_FLAG="
set "COPY_FLAG="

:parse_args
if "%~1"=="" goto :done_parsing
if /i "%~1"=="--fp16" (
    set "FP16_FLAG=--fp16"
    shift
    goto :parse_args
)
if /i "%~1"=="--copy" (
    set "COPY_FLAG=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--checkpoint" (
    set "CHECKPOINT_DIR=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--output" (
    set "OUTPUT_DIR=%~2"
    shift
    shift
    goto :parse_args
)
REM Assume any other non-flag is the suffix
echo %~1|findstr /r "^[0-9][0-9]*$" >nul
if %errorlevel% equ 0 (
    set "SUFFIX=%~1"
) else (
    echo [WARN] Unknown argument: %~1
)
shift
goto :parse_args
:done_parsing

if not exist "%CHECKPOINT_DIR%\" (
    echo [ERROR] Checkpoint directory not found: %CHECKPOINT_DIR%
    echo         Pass --checkpoint ^<dir^> to override.
    set "EXIT_CODE=1"
    goto :done
)

REM ---- Build suffix arg ----
set "SUFFIX_ARG="
if not "%SUFFIX%"=="" (
    set "SUFFIX_ARG=--suffix %SUFFIX%"
    echo Using specified suffix: %SUFFIX%
) else (
    echo Auto-detecting latest suffix...
)

REM ---- Run export ----
echo.
echo ============================================================
echo   DouEnMH TorchScript Export
echo ============================================================
echo   Python:     %BRIDGE_PYTHON%
echo   PyTorch:    %TORCH_VERSION%
echo   Checkpoint: %CHECKPOINT_DIR%
if not "%SUFFIX%"=="" echo   Suffix:     %SUFFIX%
echo   Output:     %OUTPUT_DIR%
echo ============================================================
echo.

"%BRIDGE_PYTHON%" "%EXPORT_PY%" ^
    --checkpoint-dir "%CHECKPOINT_DIR%" ^
    --output-dir "%OUTPUT_DIR%" ^
    %SUFFIX_ARG% ^
    %FP16_FLAG%

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Export failed with code %ERRORLEVEL%
    set "EXIT_CODE=%ERRORLEVEL%"
    goto :done
)

REM ---- Copy .pt to the C++ build model dir (only with --copy) ----
if not defined COPY_FLAG goto :skip_copy

set "ACTUAL_SUFFIX=%SUFFIX%"
if not defined ACTUAL_SUFFIX (
    REM Newest exported subdirectory first (dir /o-d), then pick the first numeric name
    for /f "tokens=*" %%i in ('dir /b /ad /o-d "%OUTPUT_DIR%" 2^>nul ^| findstr /r "^[0-9]"') do (
        if not defined ACTUAL_SUFFIX set "ACTUAL_SUFFIX=%%i"
    )
)
if not defined ACTUAL_SUFFIX (
    echo [WARN] Could not determine output suffix for --copy
    goto :skip_copy
)

set "EXPORT_SUBDIR=%OUTPUT_DIR%\%ACTUAL_SUFFIX%"
if not exist "%EXPORT_SUBDIR%\" (
    echo [WARN] Export directory not found for --copy: %EXPORT_SUBDIR%
    goto :skip_copy
)

set "CPP_MODEL_DIR=%SCRIPT_DIR%..\..\cpp\build\data\models"
if not exist "%CPP_MODEL_DIR%\" mkdir "%CPP_MODEL_DIR%"

echo.
echo --- Copying .pt files to C++ model directory ---
for %%f in (first second third landlord landlord_up landlord_down) do (
    if exist "%EXPORT_SUBDIR%\%%f_jit.pt" (
        copy /y "%EXPORT_SUBDIR%\%%f_jit.pt" "%CPP_MODEL_DIR%\" >nul
    )
)
echo   Copied to: %CPP_MODEL_DIR%

:skip_copy
echo.
echo ============================================================
echo   Export complete!
echo   Models: %OUTPUT_DIR%
echo ============================================================

:done
endlocal & set "SAVED_CODE=%EXIT_CODE%"
exit /b %SAVED_CODE%

@echo off
setlocal enabledelayedexpansion
set "EXIT_CODE=0"

REM ===========================================================================
REM export_douenmh.bat -- Export DouEnMH checkpoints to TorchScript .pt
REM ===========================================================================
REM Uses bridge_python (Python 3.8 + PyTorch 1.5.0) to export DouEnMH models
REM in TorchScript format for Botzone LibTorch 1.5.0.
REM
REM Features:
REM   - Auto-detects model architecture from checkpoint (d_model, layers, dim_ff, x_dim)
REM   - Auto-detects latest suffix if not specified
REM   - Exports 6 models: first/second/third + landlord/landlord_up/landlord_down
REM
REM Usage:
REM     export_douenmh.bat
REM     export_douenmh.bat 4093600
REM     export_douenmh.bat 4093600 --fp16
REM     export_douenmh.bat 4093600 --fp16 --copy
REM
REM   First argument (optional): checkpoint suffix (frame number)
REM   --fp16:          Also export FP16 quantized versions
REM   --copy:          Copy .pt files to C++ model dirs after export
REM   --checkpoint:    Override checkpoint directory
REM   --output:        Override output directory
REM ===========================================================================

set "SCRIPT_DIR=%~dp0"
set "BRIDGE_PYTHON=h:/bot/environments/bridge_python/python38_full/python.exe"
set "EXPORT_PY=%SCRIPT_DIR%export_douenmh.py"

REM Default paths
set "CHECKPOINT_DIR=h:/bot/DouEnMH-design/DouEnMH/DouEnMH-new/douzero_checkpoints/DouEnMH_v1.2"
set "OUTPUT_DIR=%SCRIPT_DIR%export_model"

REM ---- Determine suffix ----
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
echo %~1| findstr /r "^[0-9][0-9]*$" >nul
if %errorlevel% equ 0 (
    set "SUFFIX=%~1"
) else (
    echo [WARN] Unknown argument: %~1
)
shift
goto :parse_args
:done_parsing

REM ---- Validate prerequisites ----
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
if not exist "%CHECKPOINT_DIR%\" (
    echo [ERROR] Checkpoint directory not found: %CHECKPOINT_DIR%
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

REM ---- Copy .pt to C++ model dirs (only with --copy) ----
if defined COPY_FLAG (
    REM Determine the actual suffix used by reading the output directory
    for /f "tokens=*" %%i in ('dir /b /ad /o-n "%OUTPUT_DIR%" 2^>nul ^| findstr /r "^[0-9]"') do (
        set "ACTUAL_SUFFIX=%%i"
        goto :got_suffix
    )
    :got_suffix

    if defined ACTUAL_SUFFIX (
        set "EXPORT_SUBDIR=%OUTPUT_DIR%\%ACTUAL_SUFFIX%"
        echo.
        echo --- Copying .pt files to C++ model directories ---

        for %%d in (
            "%SCRIPT_DIR%../DouEnMHcpp/DouEnMHcpp-new/data/models"
            "%SCRIPT_DIR%../DouEnMHcpp/DouEnMHcpp-new/build/data/models"
        ) do (
            if not exist "%%d" mkdir "%%d"
            for %%f in (first second third landlord landlord_up landlord_down) do (
                if exist "!EXPORT_SUBDIR!\%%f_jit.pt" (
                    copy /y "!EXPORT_SUBDIR!\%%f_jit.pt" "%%d\" > nul
                )
            )
            echo   Copied to: %%d
        )
        echo   Done.
    ) else (
        echo [WARN] Could not determine output suffix for --copy
    )
)

echo.
echo ============================================================
echo   Export complete!
echo   Models: %OUTPUT_DIR%
echo ============================================================

:done
endlocal & set "SAVED_CODE=%EXIT_CODE%"
pause
exit /b %SAVED_CODE%

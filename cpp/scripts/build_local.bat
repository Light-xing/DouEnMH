@echo off

 setlocal enabledelayedexpansion

REM SCRIPT_DIR = scripts/, source files in ../src/, build output in ../build/
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\..\..\.."
pushd "%SCRIPT_DIR%\.."

REM ---- Parse args ----
set "MODE=release"
set "DO_RUN=0"
set "GAMES=1"
for %%a in (%*) do (
    if "%%a"=="debug"   set "MODE=debug"
    if "%%a"=="run"     set "DO_RUN=1"
    if "%%a"=="clean"   goto :clean
)
REM Check if last arg is a number (for run mode)
for %%a in (%*) do set "LAST=%%a"
echo %LAST%|findstr /r "^[0-9][0-9]*$" >nul && set "GAMES=%LAST%"

REM ---- Paths ----
set "MSVC_DIR=%PROJECT_ROOT%\environments\msvc"
set "LIBTORCH_DIR=%PROJECT_ROOT%\environments\libtorch\libtorch"
set "MODEL_DIR=%PROJECT_ROOT%\environments\model_exports"
set "BUILD_DIR=build"

REM ---- Verify MSVC exists ----
if not exist "%MSVC_DIR%\setup_x64.bat" (
    echo [ERROR] MSVC portable toolchain not found at:
    echo   %MSVC_DIR%
    echo Please run environments\portable-msvc.py first.
    exit /b 1
)

REM ---- Verify LibTorch exists ----
if not exist "%LIBTORCH_DIR%\include\torch\script.h" (
    echo [ERROR] LibTorch 1.5.0 not found at:
    echo   %LIBTORCH_DIR%
    exit /b 1
)

REM ---- Step 1: Setup MSVC environment ----
echo [1/5] Setting up MSVC portable toolchain...
REM 先显式清零 ERRORLEVEL: 前面 "echo %LAST%|findstr ..." 在无参数时会因找不到匹配而置 1,
REM 而 set / call 都不会清零它 (setup_x64.bat 全是 set 语句), 会导致下面误判为 "MSVC setup failed".
ver >nul
call "%MSVC_DIR%\setup_x64.bat"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] MSVC setup failed
    exit /b 1
)

REM ---- Step 2: Add LibTorch paths ----
echo [2/5] Adding LibTorch 1.5.0 paths...
set "INCLUDE=%LIBTORCH_DIR%\include;%LIBTORCH_DIR%\include\torch\csrc\api\include;%INCLUDE%"
set "LIB=%LIBTORCH_DIR%\lib;%LIB%"

REM ---- Step 3: Compile ----
echo [3/5] Compiling DouMH bot...

set "CL_FLAGS=/std:c++17 /EHsc /MD /W3 /utf-8 /D_CRT_SECURE_NO_WARNINGS"
set "CL_LIBS=torch.lib torch_cpu.lib c10.lib"
set "CL_OUT=/Fe%BUILD_DIR%\bot.exe"

if "%MODE%"=="debug" (
    set "CL_FLAGS=%CL_FLAGS% /Od /Zi /DDEBUG"
    echo   Mode: debug
) else (
    set "CL_FLAGS=%CL_FLAGS% /O2 /DNDEBUG"
    echo   Mode: release
)

REM Create build directory
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM Compile — source files in src/, jsoncpp in src/
REM 注意: 必须包含 src\oracle.cpp —— 神谕 DP 的实现是它, 缺了会在链接期报
REM       unresolved external symbol (compute_oracle_raw / dp_clear_cache / estimate_opponent_n_active)
cl %CL_FLAGS% ^
   /I"%CD%\src" ^
   %CL_OUT% ^
   src\main.cpp ^
   src\jsoncpp.cpp ^
   src\oracle.cpp ^
   /link %CL_LIBS%

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Compilation failed.
    popd
    exit /b 1
)
echo   Done. Output: %BUILD_DIR%\bot.exe

REM ---- Step 4: Copy runtime DLLs ----
echo [4/5] Copying runtime DLLs...

REM LibTorch DLLs (minimum set for CPU inference)
for %%d in (torch.dll torch_cpu.dll c10.dll asmjit.dll fbgemm.dll libiomp5md.dll) do (
    if exist "%LIBTORCH_DIR%\lib\%%d" (
        copy /y "%LIBTORCH_DIR%\lib\%%d" "%BUILD_DIR%\" >nul
        echo   %%d
    )
)

REM MSVC CRT DLLs
for %%d in (vcruntime140.dll vcruntime140_1.dll msvcp140.dll msvcp140_1.dll msvcp140_2.dll msvcp140_atomic_wait.dll msvcp140_codecvt_ids.dll) do (
    if exist "%MSVC_DIR%\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\%%d" (
        copy /y "%MSVC_DIR%\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\%%d" "%BUILD_DIR%\" >nul
        echo   %%d
    )
)

REM ---- Step 5: Models ----
echo [5/5] Setting up model files...

REM Create models symlink or copy
if not exist "%BUILD_DIR%\models" (
    if exist "%MODEL_DIR%" (
        REM Try symlink first, fall back to copy
        mklink /d "%BUILD_DIR%\models" "%MODEL_DIR%" >nul 2>&1
        REM 块内不能用 %ERRORLEVEL% (解析期展开, 取到的是 mklink 之前的值), 必须用运行时判定
        if errorlevel 1 (
            echo   Symlink failed, copying models...
            xcopy /e /i /q "%MODEL_DIR%" "%BUILD_DIR%\models\"
        ) else (
            echo   models/ -^> %MODEL_DIR%  [symlink]
        )
    ) else (
        echo   [WARN] Model directory not found: %MODEL_DIR%
        echo   Place .pt files in %BUILD_DIR%\ or set MODEL_DIR in source.
    )
)

echo.
echo ============================================================
REM 注意: 本脚本启用了 delayedexpansion, 消息文本里不要出现 '!', 否则会被当成延迟展开标记并吞掉后续文字
echo   Build complete.  %CD%\%BUILD_DIR%\bot.exe
echo ============================================================

REM ---- Step 6: Optional run ----
if "%DO_RUN%"=="1" (
    echo.
    echo Running %GAMES% games via runner.py...
    echo   python ..\..\botzone\runner.py --bot %BUILD_DIR%\bot.exe --games %GAMES%
    python ..\..\botzone\runner.py --bot %BUILD_DIR%\bot.exe --games %GAMES%
)

popd
endlocal
exit /b 0

:clean
echo Cleaning build artifacts...
if exist "build" (
    rmdir /s /q "build"
    echo   Removed build/
) else (
    echo   Nothing to clean.
)
exit /b 0

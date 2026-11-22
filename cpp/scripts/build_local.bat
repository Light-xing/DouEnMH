@echo off
REM =============================================================================
REM build_local.bat — 本地 Windows 构建 (MSVC + LibTorch 1.5.0)
REM =============================================================================
REM 环境变量 (必须由调用者提供，脚本内不写死任何本机路径):
REM   MSVC_DIR      便携版 MSVC 根目录 (含 setup_x64.bat)
REM   LIBTORCH_DIR  LibTorch 1.5.0 根目录 (含 include\torch\script.h)
REM   MODEL_DIR     可选。导出的 6 个 .pt 所在目录；未设置或缺失只警告
REM
REM 用法:
REM   build_local.bat                Release 构建
REM   build_local.bat debug          Debug 构建
REM   build_local.bat run [N]        构建后经 tools/runner/runner.py 跑 N 局
REM   build_local.bat clean          删除 cpp\build
REM
REM 输出: cpp\build\bot.exe (以及运行所需的 DLL 与 data\models)
REM =============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "CPP_DIR=%SCRIPT_DIR%.."
set "REPO_ROOT=%SCRIPT_DIR%..\.."

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

REM ---- Toolchain paths from environment ----
if not defined MSVC_DIR (
    where cl >nul 2>nul
    if not errorlevel 1 (
        set "MSVC_READY=1"
        echo [INFO] MSVC_DIR not set - using the cl.exe already on PATH.
    ) else (
        echo [ERROR] MSVC_DIR is not set and cl.exe is not on PATH.
        echo   Either run this script from an "x64 Native Tools Command Prompt for VS",
        echo   or point MSVC_DIR at a toolchain directory that contains setup_x64.bat:
        echo     set "MSVC_DIR=path\to\msvc"
        exit /b 1
    )
)
if not defined LIBTORCH_DIR (
    echo [ERROR] LIBTORCH_DIR is not set.
    echo   Point it at a LibTorch 1.5.0 directory that contains include\torch\script.h.
    echo   Example:
    echo     set "LIBTORCH_DIR=path\to\libtorch"
    exit /b 1
)

REM Resolve relative paths against the caller's current directory
if defined MSVC_DIR (
    for %%I in ("%MSVC_DIR%") do set "MSVC_DIR=%%~fI"
)
for %%I in ("%LIBTORCH_DIR%") do set "LIBTORCH_DIR=%%~fI"
if defined MODEL_DIR (
    for %%I in ("%MODEL_DIR%") do set "MODEL_DIR=%%~fI"
)

set "BUILD_DIR=build"

pushd "%CPP_DIR%"

REM ---- Verify MSVC exists ----
if defined MSVC_DIR (
if not exist "%MSVC_DIR%\setup_x64.bat" (
    echo [ERROR] setup_x64.bat not found under MSVC_DIR:
    echo   %MSVC_DIR%
    echo   Expected: %%MSVC_DIR%%\setup_x64.bat
    popd
    exit /b 1
)
)

REM ---- Verify LibTorch exists ----
if not exist "%LIBTORCH_DIR%\include\torch\script.h" (
    echo [ERROR] LibTorch 1.5.0 not found under LIBTORCH_DIR:
    echo   %LIBTORCH_DIR%
    echo   Expected: %%LIBTORCH_DIR%%\include\torch\script.h
    popd
    exit /b 1
)

REM ---- Step 1: Setup MSVC environment ----
if defined MSVC_READY (
    echo [1/5] cl.exe already on PATH - skipping MSVC setup.
) else (
echo [1/5] Setting up MSVC toolchain...
REM 先显式清零 ERRORLEVEL: 前面 "echo %LAST%|findstr ..." 在无参数时会因找不到匹配而置 1,
REM 而 set / call 都不会清零它 (setup_x64.bat 全是 set 语句), 会导致下面误判为 "MSVC setup failed".
ver >nul
call "%MSVC_DIR%\setup_x64.bat"
if errorlevel 1 (
    echo [ERROR] MSVC environment setup failed: %MSVC_DIR%\setup_x64.bat
    popd
    exit /b 1
)
)

REM ---- Step 2: Add LibTorch paths ----
echo [2/5] Adding LibTorch 1.5.0 paths...
set "INCLUDE=%LIBTORCH_DIR%\include;%LIBTORCH_DIR%\include\torch\csrc\api\include;%INCLUDE%"
set "LIB=%LIBTORCH_DIR%\lib;%LIB%"

REM ---- Step 3: Compile ----
echo [3/5] Compiling bot...

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

REM MSVC CRT DLLs — 从 setup_x64.bat 导出的 VCToolsInstallDir 定位, 不写死工具集版本
set "CRT_DIR="
if defined VCToolsInstallDir set "CRT_DIR=%VCToolsInstallDir%bin\Hostx64\x64"
if not defined CRT_DIR (
if defined MSVC_DIR (
    for /d %%v in ("%MSVC_DIR%\VC\Tools\MSVC\*") do set "CRT_DIR=%%~fv\bin\Hostx64\x64"
)
)
if defined CRT_DIR (
    for %%d in (vcruntime140.dll vcruntime140_1.dll msvcp140.dll msvcp140_1.dll msvcp140_2.dll msvcp140_atomic_wait.dll msvcp140_codecvt_ids.dll) do (
        if exist "%CRT_DIR%\%%d" (
            copy /y "%CRT_DIR%\%%d" "%BUILD_DIR%\" >nul
            echo   %%d
        )
    )
) else (
    echo   [WARN] MSVC CRT directory not found, runtime DLLs not copied.
)

REM ---- Step 5: Models ----
echo [5/5] Setting up model files...
REM main.cpp 从工作目录下的 data\models 读取模型; runner 以 bot.exe 所在目录为 cwd.
if exist "%BUILD_DIR%\data\models" goto :models_ready
if not defined MODEL_DIR goto :models_missing
if not exist "%MODEL_DIR%" goto :models_bad

if not exist "%BUILD_DIR%\data" mkdir "%BUILD_DIR%\data"
mklink /d "%BUILD_DIR%\data\models" "%MODEL_DIR%" >nul 2>&1
REM 块内不能用 %ERRORLEVEL% (解析期展开), 必须用运行时判定
if errorlevel 1 (
    echo   Symlink failed, copying models...
    xcopy /e /i /q "%MODEL_DIR%" "%BUILD_DIR%\data\models\"
) else (
    echo   data\models -^> %MODEL_DIR%  [symlink]
)
goto :models_done

:models_missing
echo   [WARN] MODEL_DIR is not set; bot.exe will start without models.
echo          Copy the 6 .pt files to %BUILD_DIR%\data\models (see models/README.md),
echo          or set MODEL_DIR before building.
goto :models_done

:models_bad
echo   [WARN] MODEL_DIR does not exist: %MODEL_DIR%
echo          Copy the 6 .pt files to %BUILD_DIR%\data\models (see models/README.md).
goto :models_done

:models_ready
echo   %BUILD_DIR%\data\models already exists, skipped.

:models_done
echo.
echo ============================================================
REM 注意: 本脚本启用了 delayedexpansion, 消息文本里不要出现 '!', 否则会被当成延迟展开标记并吞掉后续文字
echo   Build complete.  %CD%\%BUILD_DIR%\bot.exe
echo ============================================================

REM ---- Step 6: Optional run ----
if "%DO_RUN%"=="1" (
    echo.
    echo Running %GAMES% games via tools\runner\runner.py...
    python "%REPO_ROOT%\tools\runner\runner.py" --bot "%BUILD_DIR%\bot.exe" --games %GAMES%
)

popd
endlocal
exit /b 0

:clean
echo Cleaning build artifacts...
if exist "%CPP_DIR%\build" (
    rmdir /s /q "%CPP_DIR%\build"
    echo   Removed cpp\build
) else (
    echo   Nothing to clean.
)
exit /b 0

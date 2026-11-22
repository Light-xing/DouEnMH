@echo off
REM =============================================================================
REM build.bat — 最小构建 (MSVC + LibTorch 1.5.0)
REM =============================================================================
REM 环境变量 (必须由调用者提供，脚本内不写死任何本机路径):
REM   MSVC_DIR      便携版 MSVC 根目录 (含 setup_x64.bat)
REM   LIBTORCH_DIR  LibTorch 1.5.0 根目录 (含 include\torch\script.h)
REM   MODEL_DIR     可选。设置后把模型目录链接/复制到 cpp\build\data\models
REM
REM 输出: cpp\build\bot.exe
REM 完整流程 (含模式选择/清理/跑局) 用 build_local.bat。
REM =============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "CPP_DIR=%SCRIPT_DIR%.."

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

set "BUILD_DIR=%SCRIPT_DIR%..\build"

if defined MSVC_DIR (
if not exist "%MSVC_DIR%\setup_x64.bat" (
    echo [ERROR] setup_x64.bat not found under MSVC_DIR:
    echo   %MSVC_DIR%
    exit /b 1
)
)
if not exist "%LIBTORCH_DIR%\include\torch\script.h" (
    echo [ERROR] LibTorch 1.5.0 not found under LIBTORCH_DIR:
    echo   %LIBTORCH_DIR%
    exit /b 1
)

REM 切到项目目录: 否则 cl 会把 .obj 中间产物写到调用者的当前目录
pushd "%CPP_DIR%"

REM 先显式清零 ERRORLEVEL: 控制台残留的非零值不会被 set / call 清掉,
REM 会使下面的检查误判为 MSVC 环境初始化失败.
if defined MSVC_READY (
    echo Using cl.exe already on PATH.
) else (
ver >nul
call "%MSVC_DIR%\setup_x64.bat" > nul
if errorlevel 1 (
    echo [ERROR] MSVC environment setup failed.
    popd
    exit /b 1
)
)

set "INCLUDE=%LIBTORCH_DIR%\include;%LIBTORCH_DIR%\include\torch\csrc\api\include;%INCLUDE%"
set "LIB=%LIBTORCH_DIR%\lib;%LIB%"

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

cl /std:c++17 /EHsc /MD /O2 /W3 /utf-8 /D_CRT_SECURE_NO_WARNINGS /DNDEBUG /I"%SCRIPT_DIR%..\src" /Fe"%BUILD_DIR%\bot.exe" "%SCRIPT_DIR%..\src\main.cpp" "%SCRIPT_DIR%..\src\jsoncpp.cpp" "%SCRIPT_DIR%..\src\oracle.cpp" /link torch.lib torch_cpu.lib c10.lib

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Compilation failed.
    popd
    exit /b 1
)

REM Copy DLLs if not present
for %%d in (torch.dll torch_cpu.dll c10.dll asmjit.dll fbgemm.dll libiomp5md.dll) do (
    if exist "%LIBTORCH_DIR%\lib\%%d" (
        if not exist "%BUILD_DIR%\%%d" copy /y "%LIBTORCH_DIR%\lib\%%d" "%BUILD_DIR%\" > nul
    )
)

REM Optional model directory: %BUILD_DIR%\data\models is what main.cpp reads
if not exist "%BUILD_DIR%\data\models" if defined MODEL_DIR (
    if exist "%MODEL_DIR%" (
        if not exist "%BUILD_DIR%\data" mkdir "%BUILD_DIR%\data"
        mklink /d "%BUILD_DIR%\data\models" "%MODEL_DIR%" >nul 2>&1
        if errorlevel 1 xcopy /e /i /q "%MODEL_DIR%" "%BUILD_DIR%\data\models\"
        echo Models: %BUILD_DIR%\data\models
    ) else (
        echo [WARN] MODEL_DIR does not exist: %MODEL_DIR%
    )
)
if not exist "%BUILD_DIR%\data\models" (
    echo [WARN] No models under %BUILD_DIR%\data\models; see models/README.md
)

echo Done: %BUILD_DIR%\bot.exe

popd

@echo off
REM =============================================================================
REM Build DouMH Pass Bot (always-pass baseline, no LibTorch dependency)
REM =============================================================================
REM Source: ../tools/main_pass.cpp
REM Jsoncpp: ../src/jsoncpp.cpp
REM
REM Usage:
REM   build_pass.bat msvc       Build with MSVC cl.exe
REM   build_pass.bat mingw      Build with MinGW g++
REM
REM Botzone compiler: G++ 7.2.0 -O2 -std=c++17 with jsoncpp
REM =============================================================================

setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\..\..\.."
set "MODE=%1"

pushd "%SCRIPT_DIR%\.."

if not exist "tools\main_pass.cpp" (
    echo [ERROR] tools\main_pass.cpp not found - the pass-bot source is not in this repo.
    echo         This script is kept as a placeholder; add that file to build the baseline.
    popd
    exit /b 1
)

REM 注意: 块内的 %ERRORLEVEL% 是解析期展开 (进入块时就取值, 早于块内命令执行),
REM 因此这里统一改用运行时判定的 "if errorlevel N"。
if "%MODE%"=="mingw" (
    echo Building Pass Bot with MinGW...
    g++ -std=c++17 -O2 -Wall -Isrc -o build\pass_bot.exe tools\main_pass.cpp src\jsoncpp.cpp
    if errorlevel 1 (
        echo Build failed.
        popd
        exit /b 1
    ) else (
        echo Done: build\pass_bot.exe
    )
    goto :eof
)

if "%MODE%"=="msvc" (
    echo Building Pass Bot with MSVC...
    ver >nul
    call "%PROJECT_ROOT%\environments\msvc\setup_x64.bat" > nul 2>&1
    if errorlevel 1 (
        echo [ERROR] MSVC setup failed
        popd
        exit /b 1
    )
    cl /EHsc /std:c++17 /utf-8 /O2 /W3 /Isrc /Fe:build\pass_bot.exe tools\main_pass.cpp src\jsoncpp.cpp
    if errorlevel 1 (
        echo Build failed.
        popd
        exit /b 1
    ) else (
        echo Done: build\pass_bot.exe
    )
    goto :eof
)

echo Usage: build_pass.bat [mingw|msvc]
echo   mingw - Use MinGW g++ (matches Botzone G++ 7.2.0)
echo   msvc  - Use MSVC cl.exe
popd & goto :eof

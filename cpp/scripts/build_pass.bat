@echo off
REM =============================================================================
REM build_pass.bat — Pass Bot 基线构建 (不依赖 LibTorch)
REM =============================================================================
REM Source: ../tools/main_pass.cpp
REM Jsoncpp: ../src/jsoncpp.cpp
REM
REM 用法:
REM   build_pass.bat msvc      用 MSVC cl.exe 构建 (自动在 PATH 中查找 cl,
REM                            找不到时使用环境变量 MSVC_DIR 指向的便携版工具链)
REM   build_pass.bat mingw     用 MinGW g++ 构建
REM
REM Botzone 参考编译器: G++ 7.2.0 -O2 -std=c++17
REM 输出: cpp/build/pass_bot.exe
REM
REM 错误处理用 goto 而不是块内 exit /b: cmd 在嵌套括号块里执行 exit /b 后,
REM 若块内后面还有语句, 返回码可能被丢弃 (实测返回 0), 部署脚本不能依赖它.
REM =============================================================================

setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
set "CPP_DIR=%SCRIPT_DIR%.."
set "MODE=%1"

if defined MSVC_DIR (
    for %%I in ("%MSVC_DIR%") do set "MSVC_DIR=%%~fI"
)

pushd "%CPP_DIR%"

if not exist "tools\main_pass.cpp" (
    echo [ERROR] tools\main_pass.cpp not found - the pass-bot source is missing.
    goto :fail
)

if not exist "build" mkdir "build"

if "%MODE%"=="mingw" goto :mingw
if "%MODE%"=="msvc" goto :msvc

echo Usage: build_pass.bat [mingw^|msvc]
echo   mingw - Use MinGW g++ (matches Botzone G++ 7.2.0)
echo   msvc  - Use MSVC cl.exe
popd
exit /b 1

:mingw
echo Building Pass Bot with MinGW...
g++ -std=c++17 -O2 -Wall -Isrc -o build\pass_bot.exe tools\main_pass.cpp src\jsoncpp.cpp
if errorlevel 1 (
    echo Build failed.
    goto :fail
)
echo Done: build\pass_bot.exe
popd
exit /b 0

:msvc
echo Building Pass Bot with MSVC...

REM cl 已在 PATH 中则直接用; 否则从 MSVC_DIR 初始化环境
set "NEED_MSVC_SETUP=1"
where cl >nul 2>&1
if not errorlevel 1 set "NEED_MSVC_SETUP=0"

if "!NEED_MSVC_SETUP!"=="1" (
    if not defined MSVC_DIR (
        echo [ERROR] cl.exe is not on PATH and MSVC_DIR is not set.
        echo   Set MSVC_DIR to a portable MSVC directory containing setup_x64.bat.
        echo   Example:
        echo     set "MSVC_DIR=path\to\msvc"
        goto :fail
    )
    if not exist "%MSVC_DIR%\setup_x64.bat" (
        echo [ERROR] setup_x64.bat not found under MSVC_DIR:
        echo   %MSVC_DIR%
        goto :fail
    )
    ver >nul
    call "%MSVC_DIR%\setup_x64.bat" > nul 2>&1
    if errorlevel 1 (
        echo [ERROR] MSVC environment setup failed.
        goto :fail
    )
)

cl /EHsc /std:c++17 /utf-8 /O2 /W3 /Isrc /Fe:build\pass_bot.exe tools\main_pass.cpp src\jsoncpp.cpp
if errorlevel 1 (
    echo Build failed.
    goto :fail
)
echo Done: build\pass_bot.exe
popd
exit /b 0

:fail
popd
exit /b 1

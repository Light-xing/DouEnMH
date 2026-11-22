@echo off
setlocal
REM SCRIPT_DIR = scripts/, source in ../src/, build output in ../build/
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\..\..\.."
set "MSVC_DIR=%PROJECT_ROOT%\environments\msvc"
set "LIBTORCH_DIR=%PROJECT_ROOT%\environments\libtorch\libtorch"
set "BUILD_DIR=%SCRIPT_DIR%..\build"

REM 切到项目目录: 否则 cl 会把 .obj 中间产物写到调用者的当前目录
pushd "%SCRIPT_DIR%.."

REM 先显式清零 ERRORLEVEL: 控制台残留的非零值不会被 set / call 清掉,
REM 会使下面的检查误判为 MSVC 环境初始化失败.
ver >nul
call "%MSVC_DIR%\setup_x64.bat" > nul
if %ERRORLEVEL% neq 0 exit /b 1

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

echo Done: %BUILD_DIR%\bot.exe

popd

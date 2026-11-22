@echo off
REM =============================================================================
REM _compile.bat -- 用 MSVC 编译 oracle_core.c，输出 oracle_core.dll
REM =============================================================================
REM 产物: 本目录下的 oracle_core.dll（oracle.py 用 ctypes 按固定文件名加载；
REM       找不到 DLL 时 oracle.py 自动回退纯 Python DP，只是首帧慢很多）。
REM
REM MSVC 环境定位顺序:
REM   1. 环境变量 VCVARS 已设置且文件存在 -> 直接 call "%VCVARS%"
REM   2. 否则用 vswhere.exe 查找已安装 Visual Studio 的安装根目录，再调用
REM      其 VC\Auxiliary\Build\vcvars64.bat
REM   3. 都找不到 -> 打印安装/设置提示并退出（退出码 1）
REM
REM vswhere.exe 的位置: %ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
REM 需要安装 “使用 C++ 的桌面开发” 工作负载。
REM
REM 用法: 双击或在 cmd 中执行本脚本；脚本会自行切到自身所在目录。
REM =============================================================================

setlocal

set "SCRIPT_DIR=%~dp0"
set "VCVARS_FILE=%VCVARS%"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSINSTALL="

pushd "%SCRIPT_DIR%"

if defined VCVARS_FILE goto :use_vcvars

REM ---- 未设置 VCVARS: 用 vswhere 查找 Visual Studio ----
if not exist "%VSWHERE%" goto :no_vswhere

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL goto :no_vs

set "VCVARS_FILE=%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS_FILE%" goto :no_vcvars64

:use_vcvars
if not exist "%VCVARS_FILE%" (
    echo [ERROR] VCVARS points to a missing file:
    echo         %VCVARS_FILE%
    echo         Set VCVARS to the full path of vcvars64.bat, e.g.
    echo         set "VCVARS=path\to\VC\Auxiliary\Build\vcvars64.bat"
    goto :fail
)

echo [1/2] Initializing MSVC environment...
echo       %VCVARS_FILE%
REM 先清零 ERRORLEVEL: 控制台残留的非零值不会被 set 清掉, 会让下面的检查误判
ver >nul
call "%VCVARS_FILE%"
if errorlevel 1 (
    echo [ERROR] Failed to initialize MSVC environment: %VCVARS_FILE%
    goto :fail
)

where cl >nul 2>nul
if errorlevel 1 (
    echo [ERROR] cl.exe is still not on PATH after vcvars64.bat.
    echo         Run this script from a normal cmd window, or set VCVARS to a working vcvars64.bat.
    goto :fail
)

echo [2/2] Compiling oracle_core.c -^> oracle_core.dll ...
cl /O2 /LD /std:c11 oracle_core.c /Fe:oracle_core.dll
if errorlevel 1 (
    echo [ERROR] Compilation failed.
    goto :fail
)

echo [OK] Built %SCRIPT_DIR%oracle_core.dll
popd
endlocal
exit /b 0

REM ---- 失败分支 ----
:no_vswhere
echo [ERROR] vswhere.exe not found:
echo         %VSWHERE%
echo         Install Visual Studio 2019/2022 with the "Desktop development with C++"
echo         workload, or set VCVARS to the full path of vcvars64.bat and re-run.
goto :fail

:no_vs
echo [ERROR] No Visual Studio installation with the C++ toolchain was found by vswhere.
echo         Install the "Desktop development with C++" workload, or set VCVARS to
echo         the full path of vcvars64.bat and re-run.
goto :fail

:no_vcvars64
echo [ERROR] vcvars64.bat not found under the detected Visual Studio installation:
echo         %VSINSTALL%
echo         Expected: %VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat
goto :fail

:fail
popd
endlocal
exit /b 1

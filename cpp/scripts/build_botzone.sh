#!/bin/bash
# =============================================================================
# build_botzone.sh — Linux/GCC 参考构建 (GCC 7.2.0 兼容性验证)
# =============================================================================
# 对应 Botzone 的 “带很多库” 编译器环境（Ubuntu 16.04 / g++ 7.2.0）：
#   - 编译参数: -O2 -std=c++17
#   - 平台自带 jsoncpp（#include "jsoncpp/json.h"）
#   - LibTorch 1.5.0（默认装在 /usr/local/libtorch）
#   - 资源限制: 每步 1 秒、进程内存 256 MB
#
# 用法 (本地 Linux/WSL 验证):
#   chmod +x build_botzone.sh
#   ./build_botzone.sh                    # Release
#   ./build_botzone.sh debug              # Debug
#   ./build_botzone.sh clean              # 清理
#
# 环境变量:
#   LIBTORCH_DIR   LibTorch 1.5.0 根目录（默认 /usr/local/libtorch）
#   CXX            C++ 编译器（默认 g++）
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../src" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build/botzone"

# ---- LibTorch 路径 ----
# Botzone: /usr/local/libtorch/；本地 WSL/Ubuntu 可用 LIBTORCH_DIR 覆盖
LIBTORCH_DIR="${LIBTORCH_DIR:-/usr/local/libtorch}"
CXX="${CXX:-g++}"

# ---- Parse args ----
MODE="${1:-release}"

case "$MODE" in
    clean)
        echo "Cleaning Botzone build artifacts..."
        rm -rf "$BUILD_DIR"
        echo "Done."
        exit 0
        ;;
    debug)
        CXXFLAGS="-std=c++17 -O0 -g -Wall -DDEBUG"
        ;;
    release)
        CXXFLAGS="-std=c++17 -O2 -Wall -DNDEBUG"
        ;;
    *)
        echo "Usage: $0 [release|debug|clean]"
        exit 1
        ;;
esac

# ---- Check paths ----
if [ ! -f "$LIBTORCH_DIR/include/torch/script.h" ]; then
    echo "ERROR: LibTorch 1.5.0 not found under LIBTORCH_DIR=$LIBTORCH_DIR" >&2
    echo "       Expected: $LIBTORCH_DIR/include/torch/script.h" >&2
    echo "       Set LIBTORCH_DIR to the LibTorch root, or install it at /usr/local/libtorch." >&2
    exit 1
fi
if [ ! -f "$SRC_DIR/main.cpp" ]; then
    echo "ERROR: main.cpp not found: $SRC_DIR/main.cpp" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

# ---- Compile ----
echo "============================================================"
echo " DouMH Botzone Reference Build (GCC 7.2.0)"
echo "============================================================"
echo " Mode:      $MODE"
echo " Compiler:  $CXX"
echo " LibTorch:  $LIBTORCH_DIR"
echo " Source:    $SRC_DIR"
echo " Output:    $BUILD_DIR/bot"
echo "============================================================"
echo ""

set -x

"$CXX" $CXXFLAGS \
    -I"$SRC_DIR" \
    -I"$LIBTORCH_DIR/include" \
    -I"$LIBTORCH_DIR/include/torch/csrc/api/include" \
    -o "$BUILD_DIR/bot" \
    "$SRC_DIR/main.cpp" \
    "$SRC_DIR/jsoncpp.cpp" \
    "$SRC_DIR/oracle.cpp" \
    -L"$LIBTORCH_DIR/lib" \
    -ltorch -ltorch_cpu -lc10 \
    -lpthread -lstdc++fs

set +x

echo ""
echo "============================================================"
echo " Build successful: $BUILD_DIR/bot"
echo ""
echo " Run locally (stdin JSON test):"
echo "   echo '{\"requests\":[{\"own\":[...],\"bid\":[]}]}' | $BUILD_DIR/bot"
echo ""
echo " Botzone submission:"
echo "   1. Select compiler: '带很多库'"
echo "   2. Upload .pt models to User Storage"
echo "   3. Submit source files (see cpp/tools/prepare_botzone.py)"
echo "============================================================"

#!/bin/bash
# =============================================================================
# build_pass.sh — Pass Bot 基线构建 (不依赖 LibTorch, GCC)
# =============================================================================
# Source: ../tools/main_pass.cpp
# Jsoncpp include: ../src/ (needed for jsoncpp/json.h)
# Output: ../build/pass_bot
#
# Botzone 参考编译器: G++ 7.2.0 -O2 -std=c++17
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$CPP_DIR/src"
TOOLS_DIR="$CPP_DIR/tools"
BUILD_DIR="$CPP_DIR/build"
CXX="${CXX:-g++}"

if [ ! -f "$TOOLS_DIR/main_pass.cpp" ]; then
    echo "ERROR: tools/main_pass.cpp not found: $TOOLS_DIR/main_pass.cpp" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

echo "Building Pass Bot (GCC)..."
"$CXX" -std=c++17 -O2 -Wall \
    -I"$SRC_DIR" \
    -o "$BUILD_DIR/pass_bot" \
    "$TOOLS_DIR/main_pass.cpp" \
    "$SRC_DIR/jsoncpp.cpp"

echo "Done: $BUILD_DIR/pass_bot"
echo ""
echo "Test: echo '{\"requests\":[{\"own\":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],\"bid\":[]}]}' | $BUILD_DIR/pass_bot"

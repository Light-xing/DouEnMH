#!/bin/bash
# =============================================================================
# Build DouMH Pass Bot (always-pass baseline, no LibTorch dependency)
# =============================================================================
# Source: ../tools/main_pass.cpp
# Jsoncpp include: ../src/ (needed for jsoncpp/json.h)
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"
TOOLS_DIR="$SCRIPT_DIR/../tools"

echo "Building Pass Bot (GCC)..."
g++ -std=c++17 -O2 -Wall \
    -I"$SRC_DIR" \
    -o "$SCRIPT_DIR/../pass_bot" \
    "$TOOLS_DIR/main_pass.cpp" \
    "$SRC_DIR/jsoncpp.cpp"

echo "Done: $SCRIPT_DIR/../pass_bot"
echo ""
echo "Test: echo '{\"requests\":[{\"own\":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],\"bid\":[]}]}' | $SCRIPT_DIR/../pass_bot"

#!/bin/bash
# =============================================================================
# DouMH Botzone Build Reference (GCC 7.2.0 / Ubuntu)
# =============================================================================
# This is what Botzone's "带很多库" compiler effectively runs.
# Use this for local verification that the code is GCC-compatible.
#
# Botzone environment:
#   - Ubuntu 16.04, g++ 7.2.0, -O2 -std=c++17
#   - jsoncpp, nlohmann/json, Eigen pre-installed
#   - LibTorch 1.5.0 at standard paths
#   - 256 MB memory, 1 second per move
#
# Usage (local Ubuntu/WSL test):
#   chmod +x build_botzone.sh
#   ./build_botzone.sh              # Release build
#   ./build_botzone.sh debug        # Debug build
#   ./build_botzone.sh clean        # Clean
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"
BUILD_DIR="$SCRIPT_DIR/../build_botzone"

# ---- Botzone LibTorch paths (adjust for local testing) ----
# On Botzone: /usr/local/libtorch/
# Local WSL:  adjust to wherever LibTorch 1.5.0 GCC is installed
LIBTORCH_DIR="${LIBTORCH_DIR:-/usr/local/libtorch}"

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

mkdir -p "$BUILD_DIR"

# ---- Compile ----
echo "============================================================"
echo " DouMH Botzone Reference Build (GCC 7.2.0)"
echo "============================================================"
echo " Mode:      $MODE"
echo " LibTorch:  $LIBTORCH_DIR"
echo " Source:    $SRC_DIR"
echo " Output:    $BUILD_DIR/bot"
echo "============================================================"
echo ""

set -x

g++ $CXXFLAGS \
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
echo "   3. Submit source files (see prepare_botzone.py)"
echo "============================================================"

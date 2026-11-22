#!/bin/bash
# =============================================================================
# DouMH Amalgamate Script — 将多文件 C++ 项目合成为单文件
# =============================================================================
# 用于 Botzone 提交: 将 main.cpp + 所有项目 .h 内联为单一 .cpp 文件,
# 同时保留 jsoncpp/json.h 等 Botzone 平台自带库的 #include 不被内联.
#
# 原理: 临时移走 jsoncpp/ 目录 → Amalgamate 找不到 → 原样保留 #include → 恢复
#
# 依赖: Amalgamate.exe (已编译在 Amalgamate-master 目录下)
#
# 用法:
#   chmod +x amalgamate.sh
#   ./amalgamate.sh              # 合成到默认输出路径
#   ./amalgamate.sh clean        # 清理合成产物
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../src" && pwd)"
AMALGAMATE="$(cd "$SCRIPT_DIR/../../../../Amalgamate/Amalgamate-master" && pwd)/Amalgamate.exe"
OUTPUT="$SRC_DIR/amalgamated_bot.cpp"
JSON_DIR="$SRC_DIR/jsoncpp"
JSON_BACKUP="$SRC_DIR/jsoncpp.bak"

# ---- Parse args ----
MODE="${1:-build}"

case "$MODE" in
    clean)
        echo "Cleaning amalgamated output..."
        rm -f "$OUTPUT"
        echo "Done."
        exit 0
        ;;
    build)
        ;;
    *)
        echo "Usage: $0 [build|clean]"
        exit 1
        ;;
esac

# ---- Check prerequisites ----
if [ ! -f "$AMALGAMATE" ]; then
    echo "ERROR: Amalgamate.exe not found at: $AMALGAMATE"
    echo "Build it first with MSVC:"
    echo "  cd Amalgamate-master && cl /EHsc /O2 /Fe:Amalgamate.exe Amalgamate.cpp juce_core_amalgam.cpp ole32.lib shell32.lib user32.lib advapi32.lib ws2_32.lib"
    exit 1
fi

if [ ! -f "$SRC_DIR/main.cpp" ]; then
    echo "ERROR: main.cpp not found at: $SRC_DIR"
    exit 1
fi

echo "============================================================"
echo " DouMH C++ Amalgamation"
echo "============================================================"
echo " Source:    $SRC_DIR"
echo " Output:    $OUTPUT"
echo " Amalgamate: $AMALGAMATE"
echo "============================================================"
echo ""

# ---- Step 1: 临时移走 jsoncpp/ ----
if [ -d "$JSON_DIR" ]; then
    echo "[1/4] Moving jsoncpp/ out of the way..."
    mv "$JSON_DIR" "$JSON_BACKUP"
    JSON_MOVED=true
else
    echo "[1/4] jsoncpp/ already moved (skipping)"
    JSON_MOVED=false
fi

# ---- Step 2: 运行 Amalgamate ----
echo "[2/4] Running Amalgamate..."
export MSYS_NO_PATHCONV=1

# Amalgamate 是 Windows 原生程序, 需要 Windows 格式路径
AMALGAMATE_WIN="$(cygpath -w "$AMALGAMATE" 2>/dev/null || echo "$AMALGAMATE")"
SRC_DIR_WIN="$(cygpath -w "$SRC_DIR" 2>/dev/null || echo "$SRC_DIR")"
OUTPUT_WIN="$(cygpath -w "$OUTPUT" 2>/dev/null || echo "$OUTPUT")"

"$AMALGAMATE" \
    -i "$SRC_DIR_WIN" \
    -v \
    "$SRC_DIR_WIN/main.cpp" \
    "$OUTPUT_WIN"

echo ""

# ---- Step 3: 恢复 jsoncpp/ ----
echo "[3/4] Restoring jsoncpp/..."
if [ "$JSON_MOVED" = true ]; then
    mv "$JSON_BACKUP" "$JSON_DIR"
else
    echo "  (jsoncpp/ was not moved, nothing to restore)"
fi

# ---- Step 4: 验证 ----
echo "[4/4] Verifying output..."
INLINE_COUNT=$(grep -c "Start of inlined file:" "$OUTPUT" 2>/dev/null || echo 0)
LINE_COUNT=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
JSON_KEPT=$(grep -c '#include "jsoncpp/json.h"' "$OUTPUT" 2>/dev/null || echo 0)

echo ""
echo "============================================================"
echo " Amalgamation Complete"
echo "============================================================"
echo " Output:       $OUTPUT"
echo " Lines:        $LINE_COUNT"
echo " Files inlined: $INLINE_COUNT"
echo " jsoncpp/json.h kept:  $JSON_KEPT #include(s)"
echo ""
echo " Botzone submission:"
echo "   1. Select compiler: '带很多库'"
echo "   2. Upload .pt models to User Storage"
echo "   3. Submit: $OUTPUT"
echo "============================================================"

#!/bin/bash
# =============================================================================
# amalgamate.sh — 将 cpp/src 的多文件工程合成为单个 .cpp（Botzone 单文件提交用）
# =============================================================================
# 原理:
#   1. 临时移走 src/jsoncpp/ → Amalgamate 找不到该头文件 → 原样保留
#      #include "jsoncpp/json.h"（Botzone 平台自带 jsoncpp）
#   2. 用 Amalgamate 从 main.cpp 出发内联所有本地头文件
#   3. 追加 oracle.cpp 的实现（Amalgamate 只跟随 #include，不会自动带上 .cpp）
#   4. 恢复 src/jsoncpp/（脚本中断时由 trap 恢复）
#
# 依赖: Amalgamate（Windows 原生可执行文件）。
#   查找顺序: 环境变量 AMALGAMATE → PATH 中的 Amalgamate / Amalgamate.exe。
#   未找到时脚本会给出下载与自建提示后退出。
#
# 用法:
#   chmod +x amalgamate.sh
#   AMALGAMATE=/path/to/Amalgamate ./amalgamate.sh
#   ./amalgamate.sh clean        # 清理合成产物
#
# 输出: cpp/build/amalgamated_bot.cpp（cpp/build/ 已在 .gitignore 中）
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../src" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
OUTPUT="$BUILD_DIR/amalgamated_bot.cpp"
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

# ---- Locate Amalgamate ----
AMALGAMATE_BIN=""
if [ -n "${AMALGAMATE:-}" ]; then
    if command -v "$AMALGAMATE" >/dev/null 2>&1; then
        AMALGAMATE_BIN="$(command -v "$AMALGAMATE")"
    elif [ -f "$AMALGAMATE" ]; then
        AMALGAMATE_BIN="$AMALGAMATE"
    fi
else
    for name in Amalgamate Amalgamate.exe; do
        if command -v "$name" >/dev/null 2>&1; then
            AMALGAMATE_BIN="$(command -v "$name")"
            break
        fi
    done
fi

if [ -z "$AMALGAMATE_BIN" ]; then
    echo "ERROR: Amalgamate not found." >&2
    echo "  Set AMALGAMATE to the executable, or put Amalgamate on PATH:" >&2
    echo "    export AMALGAMATE=/path/to/Amalgamate" >&2
    echo "" >&2
    echo "  To get it: download the Amalgamate source (Vinnie Falco's Amalgamate)" >&2
    echo "  and build Amalgamate.cpp with MSVC, e.g.:" >&2
    echo "    cl /EHsc /O2 /Fe:Amalgamate.exe Amalgamate.cpp juce_core_amalgam.cpp \\" >&2
    echo "       ole32.lib shell32.lib user32.lib advapi32.lib ws2_32.lib" >&2
    exit 1
fi

if [ ! -f "$SRC_DIR/main.cpp" ]; then
    echo "ERROR: main.cpp not found: $SRC_DIR/main.cpp" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

# ---- jsoncpp/ 临时移走 + 退出时恢复 ----
JSON_MOVED=0
restore_jsoncpp() {
    if [ "$JSON_MOVED" = 1 ] && [ -d "$JSON_BACKUP" ]; then
        mv "$JSON_BACKUP" "$JSON_DIR"
        JSON_MOVED=0
    fi
}
trap restore_jsoncpp EXIT

echo "============================================================"
echo " DouMH C++ Amalgamation"
echo "============================================================"
echo " Source:     $SRC_DIR"
echo " Output:     $OUTPUT"
echo " Amalgamate: $AMALGAMATE_BIN"
echo "============================================================"
echo ""

# ---- Step 1: 临时移走 jsoncpp/ ----
if [ -d "$JSON_DIR" ]; then
    echo "[1/5] Moving jsoncpp/ out of the way..."
    mv "$JSON_DIR" "$JSON_BACKUP"
    JSON_MOVED=1
else
    echo "[1/5] jsoncpp/ already moved (skipping)"
fi

# ---- Step 2: 运行 Amalgamate ----
echo "[2/5] Running Amalgamate..."
export MSYS_NO_PATHCONV=1

# Amalgamate 可能是 Windows 原生程序, 需要 Windows 格式路径
AMALGAMATE_WIN="$(cygpath -w "$AMALGAMATE_BIN" 2>/dev/null || echo "$AMALGAMATE_BIN")"
SRC_DIR_WIN="$(cygpath -w "$SRC_DIR" 2>/dev/null || echo "$SRC_DIR")"
OUTPUT_WIN="$(cygpath -w "$OUTPUT" 2>/dev/null || echo "$OUTPUT")"

"$AMALGAMATE_BIN" \
    -i "$SRC_DIR_WIN" \
    -v \
    "$SRC_DIR_WIN/main.cpp" \
    "$OUTPUT_WIN"

echo ""

# ---- Step 3: 追加 oracle.cpp 实现 ----
echo "[3/5] Appending oracle.cpp implementation..."
# Amalgamate 只内联 #include 到的头文件，不会带上独立 .cpp；去掉其本地 include 后追加。
grep -v -E '^[[:space:]]*#include[[:space:]]+"(oracle\.h|card_defs\.h)"' \
    "$SRC_DIR/oracle.cpp" >> "$OUTPUT"

# ---- Step 4: 恢复 jsoncpp/ ----
echo "[4/5] Restoring jsoncpp/..."
restore_jsoncpp

# ---- Step 5: 验证 ----
echo "[5/5] Verifying output..."
INLINE_COUNT=$(grep -c "Start of inlined file:" "$OUTPUT" 2>/dev/null || echo 0)
LINE_COUNT=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
JSON_KEPT=$(grep -c '#include "jsoncpp/json.h"' "$OUTPUT" 2>/dev/null || echo 0)
ORACLE_DEF=$(grep -c 'compute_oracle_raw' "$OUTPUT" 2>/dev/null || echo 0)

echo ""
echo "============================================================"
echo " Amalgamation Complete"
echo "============================================================"
echo " Output:               $OUTPUT"
echo " Lines:                $LINE_COUNT"
echo " Files inlined:        $INLINE_COUNT"
echo " jsoncpp/json.h kept:  $JSON_KEPT #include(s)"
echo " oracle definitions:   $ORACLE_DEF reference(s)"
echo ""
echo " Botzone submission:"
echo "   1. Select compiler: '带很多库'"
echo "   2. Upload .pt models to User Storage"
echo "   3. Submit: $OUTPUT"
echo "============================================================"

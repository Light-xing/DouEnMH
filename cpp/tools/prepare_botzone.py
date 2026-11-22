"""
Package DouEnMH C++ source files for Botzone submission.

Usage:
    python prepare_botzone.py              # Create staging dir + zip
    python prepare_botzone.py --amalgamate # Create single-file amalgamation
    python prepare_botzone.py --check      # Check file completeness only
    python prepare_botzone.py --model-dir <dir>   # Where the .pt files are

Output (default):
    cpp/botzone_submit/
        main.cpp           # Main entry
        game_state.h       # State management
        card_encoding.h    # Card encoding + cards2array
        card_defs.h        # Card/Level types
        combo.h            # Combo detection + comparison
        move_generator.h   # 17 combo types enumeration
        legal_moves.h      # Legal move filtering
        feature_encoder.h  # Feature encoding bridge
        model_inference.h  # LibTorch inference
        oracle.h           # Oracle declarations
        oracle.cpp         # Oracle DP implementation
        opponent_modeling.h
        jsoncpp/           # JSON library
            json.h
            json-forwards.h
        jsoncpp.cpp
    cpp/botzone_submit.zip

Note:
    decision_router.h is NOT part of the submission. main.cpp does not include
    it, and it depends on MCTS sources (mcts_*.h) that are not in cpp/src.

Botzone submission steps:
    1. Select compiler: "带很多库" (LibTorch 1.5.0)
    2. Upload botzone_submit.zip
    3. Upload .pt models to User Storage -> data/models/
    4. main.cpp reads MODEL_DIR = "data/models"
"""
import os
import re
import sys
import shutil
import zipfile

try:  # Windows 控制台默认代码页可能不是 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(CPP_DIR)
SRC_DIR = os.path.join(CPP_DIR, "src")
OUTPUT_DIR = os.path.join(CPP_DIR, "botzone_submit")
BUILD_DIR = os.path.join(CPP_DIR, "build")

# Files needed for compilation (in dependency order, though it doesn't matter)
SOURCES = [
    "main.cpp",
    "game_state.h",
    "card_encoding.h",
    "card_defs.h",
    "combo.h",
    "move_generator.h",
    "legal_moves.h",
    "feature_encoder.h",
    "model_inference.h",
    "oracle.h",
    "oracle.cpp",
    "opponent_modeling.h",
]

JSONCPP = [
    os.path.join("jsoncpp", "json.h"),
    os.path.join("jsoncpp", "json-forwards.h"),
    "jsoncpp.cpp",
]

MODEL_FILES = [
    "first_jit.pt",
    "second_jit.pt",
    "third_jit.pt",
    "landlord_jit.pt",
    "landlord_up_jit.pt",
    "landlord_down_jit.pt",
]

AMALGAMATE_ORDER = [
    "card_defs.h",
    "combo.h",
    "move_generator.h",
    "legal_moves.h",
    "card_encoding.h",
    "game_state.h",
    "oracle.h",
    "opponent_modeling.h",
    "feature_encoder.h",
    "model_inference.h",
]

AMALGAMATE_CPP = [
    "oracle.cpp",
]

LOCAL_INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"')


def _local_includes(path):
    """Return quoted #include targets of a C++ file (local headers only)."""
    includes = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = LOCAL_INCLUDE_RE.match(line)
            if not m:
                continue
            inc = m.group(1)
            # Platform-provided jsoncpp is not shipped as a local include target
            if inc == "jsoncpp/json.h":
                continue
            includes.append(inc)
    return includes


def check_files():
    """Verify that the submission list exists and matches main.cpp's local includes."""
    missing = []
    for f in SOURCES + JSONCPP:
        path = os.path.join(SRC_DIR, f)
        if not os.path.exists(path):
            missing.append(f)
    if missing:
        print("ERROR: Missing files:")
        for f in missing:
            print(f"  - {f}")
        return False

    # Closure check: every local header reachable from the shipped sources
    # must itself be shipped.
    shipped = set(SOURCES) | set(JSONCPP)
    unresolved = []
    for f in SOURCES:
        for inc in _local_includes(os.path.join(SRC_DIR, f)):
            if inc not in shipped:
                unresolved.append((f, inc))
    if unresolved:
        print("ERROR: Local includes not covered by SOURCES:")
        for src, inc in unresolved:
            print(f"  - {src} includes {inc}")
        return False

    # main.cpp is the translation unit; make sure it does not pull in the
    # MCTS-based decision_router.h (whose dependencies are not in cpp/src).
    main_includes = _local_includes(os.path.join(SRC_DIR, "main.cpp"))
    if "decision_router.h" in main_includes:
        print("ERROR: main.cpp includes decision_router.h, which needs MCTS sources")
        print("       not shipped in cpp/src. Update SOURCES or main.cpp first.")
        return False

    print("All source files present; local include closure OK.")
    return True


def create_submit_dir():
    """Copy source files to staging directory."""
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    for f in SOURCES:
        src = os.path.join(SRC_DIR, f)
        dst = os.path.join(OUTPUT_DIR, f)
        shutil.copy2(src, dst)
        print(f"  {f}")

    for f in JSONCPP:
        src = os.path.join(SRC_DIR, f)
        dst = os.path.join(OUTPUT_DIR, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {f}")

    return True


def create_zip():
    """Create botzone_submit.zip from staging directory."""
    zip_path = os.path.join(CPP_DIR, "botzone_submit.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                full_path = os.path.join(root, file)
                arc_name = os.path.relpath(full_path, OUTPUT_DIR)
                zf.write(full_path, arc_name)
    size_kb = os.path.getsize(zip_path) / 1024
    print(f"\n  Zip created: cpp/botzone_submit.zip ({size_kb:.0f} KB)")
    return zip_path


def create_amalgamation():
    """Concatenate all sources into a single .cpp for pasting into Botzone."""
    os.makedirs(BUILD_DIR, exist_ok=True)
    output_path = os.path.join(BUILD_DIR, "bot_amalgamated.cpp")
    lines = []

    lines.append("// ============================================================")
    lines.append("// DouEnMH Amalgamated Bot — Paste into Botzone editor")
    lines.append("// Generated by cpp/tools/prepare_botzone.py --amalgamate")
    lines.append("// ============================================================")
    lines.append("")

    # All local headers that will be inlined (strip their #includes)
    ALL_INLINED = AMALGAMATE_ORDER + AMALGAMATE_CPP

    # First: jsoncpp (must come before any usage)
    jsoncpp_cpp = os.path.join(SRC_DIR, "jsoncpp.cpp")
    with open(jsoncpp_cpp, 'r', encoding='utf-8') as f:
        content = f.read()
    lines.append("// ============================================================")
    lines.append("// jsoncpp.cpp (amalgamated)")
    lines.append("// ============================================================")
    lines.append(content)
    lines.append("")
    # jsoncpp.cpp 在 MSVC 的 __cplusplus=199711L 兼容分支里会定义 isnan/isfinite
    # 宏；单文件里这些宏会泄漏进后续 LibTorch 头文件，必须在这里撤销。
    lines.append("// jsoncpp compatibility macros must not leak into LibTorch headers")
    lines.append("#ifdef isnan")
    lines.append("#undef isnan")
    lines.append("#endif")
    lines.append("#ifdef isfinite")
    lines.append("#undef isfinite")
    lines.append("#endif")
    lines.append("#ifdef sscanf")
    lines.append("#undef sscanf")
    lines.append("#endif")
    lines.append("")

    # Then all headers in dependency order
    for f in AMALGAMATE_ORDER:
        path = os.path.join(SRC_DIR, f)
        with open(path, 'r', encoding='utf-8') as fh:
            content = fh.read()
        lines.append("// ============================================================")
        lines.append(f"// {f}")
        lines.append("// ============================================================")
        filtered = []
        for line in content.split('\n'):
            stripped = line.strip()
            # Strip jsoncpp includes (already above)
            if stripped.startswith('#include "jsoncpp/json.h"'):
                continue
            # Strip includes to other inlined files
            if any(stripped.startswith(f'#include "{h}"') for h in ALL_INLINED):
                continue
            filtered.append(line)
        lines.extend(filtered)
        lines.append("")

    # Then: .cpp implementation files (oracle.cpp)
    for f in AMALGAMATE_CPP:
        path = os.path.join(SRC_DIR, f)
        with open(path, 'r', encoding='utf-8') as fh:
            content = fh.read()
        lines.append("// ============================================================")
        lines.append(f"// {f} (amalgamated)")
        lines.append("// ============================================================")
        filtered = []
        for line in content.split('\n'):
            stripped = line.strip()
            # Strip includes to inlined files
            if any(stripped.startswith(f'#include "{h}"') for h in ALL_INLINED):
                continue
            filtered.append(line)
        lines.extend(filtered)
        lines.append("")

    # Finally: main.cpp (without its local #includes)
    main_path = os.path.join(SRC_DIR, "main.cpp")
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    lines.append("// ============================================================")
    lines.append("// main.cpp")
    lines.append("// ============================================================")
    filtered = []
    for line in content.split('\n'):
        stripped = line.strip()
        if any(stripped.startswith(f'#include "{h}"') for h in ALL_INLINED + ["jsoncpp/json.h"]):
            continue
        filtered.append(line)
    lines.extend(filtered)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n  Amalgamation created: cpp/build/bot_amalgamated.cpp ({size_kb:.0f} KB)")
    print(f"  Paste contents into Botzone editor (single file).")
    return output_path


def print_instructions(model_dir):
    print(f"""
 {'='*60}
 Botzone Submission Instructions
 {'='*60}

 1. Compiler Option:
    When submitting on Botzone, select the "带很多库" compiler.
    This provides LibTorch 1.5.0 + jsoncpp + Eigen.

 2. Model Files (upload to Botzone User Storage -> data/models/):
""")
    total_mb = 0.0
    found = 0
    for m in MODEL_FILES:
        model_path = os.path.join(model_dir, m)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            total_mb += size_mb
            found += 1
            print(f"    - {m}  ({size_mb:.1f} MB)")
        else:
            print(f"    - {m}  (not found in {model_dir})")
    if found:
        print(f"\n    Found {found}/{len(MODEL_FILES)} models, total {total_mb:.0f} MB.")
    else:
        print("\n    No models found. See models/README.md for how to export them.")

    print(f"""
 3. Submit:
    - Upload cpp/botzone_submit.zip as source code
    - Select "带很多库" compiler
    - If using the amalgamated single file: paste cpp/build/bot_amalgamated.cpp

 4. Source Code Configuration:
    - MODEL_DIR = "data/models"  (Botzone User Storage path)

 {'='*60}
 """)


def main():
    amalgamate = "--amalgamate" in sys.argv
    check_only = "--check" in sys.argv

    model_dir = os.path.join(BUILD_DIR, "data", "models")
    if "--model-dir" in sys.argv:
        idx = sys.argv.index("--model-dir")
        if idx + 1 < len(sys.argv):
            model_dir = sys.argv[idx + 1]
    model_dir = os.environ.get("MODEL_DIR", model_dir)

    print("=" * 60)
    print("DouEnMH Botzone Submission Preparation")
    print("=" * 60)
    print()

    if not check_files():
        sys.exit(1)

    if check_only:
        return

    if amalgamate:
        print("\nCreating amalgamated source...")
        create_amalgamation()
    else:
        print("\nCreating submission directory...")
        create_submit_dir()
        create_zip()

    print_instructions(model_dir)


if __name__ == "__main__":
    main()

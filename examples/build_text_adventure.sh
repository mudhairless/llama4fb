#!/bin/sh
# Build text_adventure.bas against a forked llama.cpp tree.
#
# Usage:
#   ./build_text_adventure.sh /path/to/llama.cpp
#   LLAMA_CPP_DIR=/path/to/llama.cpp ./build_text_adventure.sh
#
# Prefers a CUDA build (build-cuda/bin) if present, otherwise build/bin.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INC_DIR="$PROJECT_DIR/inc"

LLAMA_CPP="${1:-${LLAMA_CPP_DIR:-}}"
if [ -z "$LLAMA_CPP" ]; then
    LLAMA_CPP="$PROJECT_DIR/vendor/llama.cpp"
    USING_SUBMODULE=1
    if [ ! -f "$LLAMA_CPP/CMakeLists.txt" ]; then
        echo "llama.cpp submodule not populated at $LLAMA_CPP" >&2
        echo "Run: git submodule update --init" >&2
        exit 1
    fi
fi
LLAMA_CPP="$(cd "$LLAMA_CPP" && pwd)"

if [ -n "${2:-}" ] && [ -d "$2" ]; then
    BUILD="$2"
elif [ -n "${LLAMA_BUILD_DIR:-}" ]; then
    BUILD="$LLAMA_BUILD_DIR"
elif [ -f "$LLAMA_CPP/build-cuda/bin/libggml-cuda.so" ]; then
    BUILD="$LLAMA_CPP/build-cuda/bin"
elif [ -f "$LLAMA_CPP/build-vulkan/bin/libggml-vulkan.so" ]; then
    BUILD="$LLAMA_CPP/build-vulkan/bin"
elif [ -n "${USING_SUBMODULE:-}" ] && [ ! -d "$LLAMA_CPP/build-cuda/bin" ] && [ ! -d "$LLAMA_CPP/build-vulkan/bin" ] && [ ! -d "$LLAMA_CPP/build/bin" ]; then
    echo "Vendored llama.cpp has no build tree. From vendor/llama.cpp, run:" >&2
    echo "  cmake -B build-cuda -DGGML_CUDA=ON" >&2
    echo "  cmake --build build-cuda --config Release -j" >&2
    exit 1
else
    BUILD="$LLAMA_CPP/build/bin"
fi

EXTRALIBS=""
if [ -f "$BUILD/libggml-cuda.so" ]; then
    EXTRALIBS="$EXTRALIBS -l ggml-cuda"
fi
if [ -f "$BUILD/libggml-vulkan.so" ]; then
    EXTRALIBS="$EXTRALIBS -l ggml-vulkan"
fi

# Submodule check
if [ ! -f "$PROJECT_DIR/vendor/fbejson/inc/json.bi" ]; then
    echo "fbejson submodule not populated." >&2
    echo "Run: git submodule update --init" >&2
    exit 1
fi

# Build fbejson libs if missing
if [ ! -f "$PROJECT_DIR/vendor/fbejson/lib/libfbejson.a" ] || [ ! -f "$PROJECT_DIR/vendor/fbejson/lib/libfbebase64.a" ]; then
    make -C "$PROJECT_DIR/vendor/fbejson" lib/libfbejson.a lib/libfbebase64.a
fi

fbc -i "$INC_DIR" \
    -i "$PROJECT_DIR/vendor/fbejson/inc" \
    -p "$PROJECT_DIR/vendor/fbejson/lib" \
    "$SCRIPT_DIR/text_adventure.bas" \
    -p "$BUILD" \
    -l llama -l ggml -l ggml-base -l ggml-cpu $EXTRALIBS \
    -l fbejson -l fbebase64 \
    -x "$SCRIPT_DIR/text_adventure"
echo "Build exit: $?"
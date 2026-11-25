#!/bin/sh
# Launcher for text_adventure. Ensures the forked llama.cpp shared libs are found.
#
# Usage:
#   ./run_text_adventure.sh /path/to/llama.cpp
#   LLAMA_CPP_DIR=/path/to/llama.cpp ./run_text_adventure.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_CPP="${1:-${LLAMA_CPP_DIR:-}}"
if [ -z "$LLAMA_CPP" ]; then
    echo "Usage: $0 /path/to/llama.cpp" >&2
    echo "       (or set LLAMA_CPP_DIR)" >&2
    exit 2
fi
LLAMA_CPP="$(cd "$LLAMA_CPP" && pwd)"

if [ -n "${2:-}" ] && [ -d "$2" ]; then
    LLAMA_BUILD="$2"
elif [ -n "${LLAMA_BUILD_DIR:-}" ]; then
    LLAMA_BUILD="$LLAMA_BUILD_DIR"
elif [ -f "$LLAMA_CPP/build-cuda/bin/libggml-cuda.so" ]; then
    LLAMA_BUILD="$LLAMA_CPP/build-cuda/bin"
elif [ -f "$LLAMA_CPP/build-vulkan/bin/libggml-vulkan.so" ]; then
    LLAMA_BUILD="$LLAMA_CPP/build-vulkan/bin"
else
    LLAMA_BUILD="$LLAMA_CPP/build/bin"
fi

EXE="$SCRIPT_DIR/text_adventure"

REQUIRED="libllama.so libggml.so libggml-base.so libggml-cpu.so"
MISSING=0
for lib in $REQUIRED; do
    if [ ! -f "$LLAMA_BUILD/$lib" ] && [ ! -f "$LLAMA_BUILD/${lib}.0" ]; then
        echo "MISSING: $lib (looked in $LLAMA_BUILD)" >&2
        MISSING=1
    fi
done
if [ "$MISSING" -ne 0 ]; then
    echo "Build llama.cpp first." >&2
    exit 1
fi

export LD_LIBRARY_PATH="$LLAMA_BUILD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ ! -x "$EXE" ]; then
    echo "Binary missing, building..."
    sh "$SCRIPT_DIR/build_text_adventure.sh" "$LLAMA_CPP"
fi

cd "$SCRIPT_DIR"
exec "$EXE" "$@"
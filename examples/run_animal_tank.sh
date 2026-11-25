#!/bin/sh
# Launcher for animal_tank. Ensures the forked llama.cpp shared libs are found.
#
# Usage:
#   ./run_animal_tank.sh /path/to/llama.cpp
#   LLAMA_CPP_DIR=/path/to/llama.cpp ./run_animal_tank.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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
    LLAMA_BUILD="$2"
elif [ -n "${LLAMA_BUILD_DIR:-}" ]; then
    LLAMA_BUILD="$LLAMA_BUILD_DIR"
elif [ -f "$LLAMA_CPP/build-cuda/bin/libggml-cuda.so" ]; then
    LLAMA_BUILD="$LLAMA_CPP/build-cuda/bin"
elif [ -f "$LLAMA_CPP/build-vulkan/bin/libggml-vulkan.so" ]; then
    LLAMA_BUILD="$LLAMA_CPP/build-vulkan/bin"
elif [ -n "${USING_SUBMODULE:-}" ] && [ ! -d "$LLAMA_CPP/build-cuda/bin" ] && [ ! -d "$LLAMA_CPP/build-vulkan/bin" ] && [ ! -d "$LLAMA_CPP/build/bin" ]; then
    echo "Vendored llama.cpp has no build tree. From vendor/llama.cpp, run:" >&2
    echo "  cmake -B build-cuda -DGGML_CUDA=ON" >&2
    echo "  cmake --build build-cuda --config Release -j" >&2
    exit 1
else
    LLAMA_BUILD="$LLAMA_CPP/build/bin"
fi

EXE="$SCRIPT_DIR/animal_tank"

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
    sh "$SCRIPT_DIR/build_animal_tank.sh" "$LLAMA_CPP"
fi

cd "$SCRIPT_DIR"
exec "$EXE" "$@"

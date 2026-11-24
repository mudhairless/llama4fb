#!/bin/sh
# Build hello_llama.bas against a forked llama.cpp tree.
#
# Usage:
#   ./build_hello_llama.sh /path/to/llama.cpp
#   LLAMA_CPP_DIR=/path/to/llama.cpp ./build_hello_llama.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INC_DIR="$PROJECT_DIR/inc"

LLAMA_CPP="${1:-${LLAMA_CPP_DIR:-}}"
if [ -z "$LLAMA_CPP" ]; then
    echo "Usage: $0 /path/to/llama.cpp" >&2
    echo "       (or set LLAMA_CPP_DIR)" >&2
    exit 2
fi
LLAMA_CPP="$(cd "$LLAMA_CPP" && pwd)"

if [ -n "${2:-}" ]; then
    BUILD="$2"
elif [ -f "$LLAMA_CPP/build-cuda/bin/libggml-cuda.so" ]; then
    BUILD="$LLAMA_CPP/build-cuda/bin"
else
    BUILD="$LLAMA_CPP/build/bin"
fi

EXTRALIBS=""
if [ -f "$BUILD/libggml-cuda.so" ]; then
    EXTRALIBS="-l ggml-cuda"
fi

fbc -i "$INC_DIR" \
    "$SCRIPT_DIR/hello_llama.bas" \
    -p "$BUILD" \
    -l llama -l ggml -l ggml-base -l ggml-cpu $EXTRALIBS \
    -x "$SCRIPT_DIR/hello_llama"
echo "Build exit: $?"
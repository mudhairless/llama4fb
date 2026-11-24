# License split in this repository

This repository contains material under three different licenses. The
root LICENSE file (MIT, Copyright (c) 2026 Ebben Feagan) is not the whole
story; read the split before reusing anything.

## The examples and original code: MIT

Everything written for this repository, including all programs under
`examples/`, is MIT licensed under the root LICENSE file.

## The translated headers under `inc/`: MIT, original authors

The `.bi` files under `inc/` are machine-generated FreeBASIC translations
of the public C headers of llama.cpp and ggml, from the raon-1-nano
branch of KRAFTON AI's llama.cpp fork
(https://github.com/krafton-ai/llama.cpp). They are derivative works of
those headers and remain under the upstream license:

MIT License, Copyright (c) 2023-2026 The ggml authors

The full upstream license text ships with llama.cpp itself in that
repository's LICENSE file. Translation does not transfer or change the
copyright in the header declarations.

## The model: Apache 2.0, not included in this repository

The examples download and run KRAFTON AI's Raon-1-Nano model in GGUF
form (https://huggingface.co/KRAFTON/Raon-1-nano-gguf). The model is
licensed under the Apache License 2.0 by KRAFTON AI. No model weights
are distributed in this repository; the examples fetch the model on
first run, and your use of the model is governed by its own license.

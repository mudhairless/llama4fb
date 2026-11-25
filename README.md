# llama4fb

FreeBASIC (`.bi`) translations of the C headers from
[krafton-ai/llama.cpp](https://github.com/krafton-ai/llama.cpp) (branch `raon-1-nano`).
Use them to call llama.cpp from FreeBASIC programs.

## What you need

- **FreeBASIC** with `fbc` on your `PATH` (tested with 1.10.x).
- The vendored **krafton-ai/llama.cpp** fork at `vendor/llama.cpp` (branch
  `raon-1-nano`), built as shared libraries (`libllama`, `libggml`,
  `libggml-base`, `libggml-cpu`, and `libggml-cuda` for GPU offload). This is
  the scripts' default location. Either `build/bin` or `build-cuda/bin` works;
  the scripts prefer `build-cuda` when present.
- A GGUF model. The examples target `raon-1-nano-Q4_K_M.gguf` and download it
  automatically on first run.
- (Optional but recommended) CUDA toolkit for the vendored CUDA build.

## Set up llama.cpp

After cloning, initialize the submodule and build its CUDA tree once:

```sh
git submodule update --init
cd vendor/llama.cpp
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda --config Release -j
```

With no path given, the scripts use `vendor/llama.cpp`, choosing
`build-cuda/bin` when `libggml-cuda.so` is present, then `build-vulkan/bin`,
then `build/bin`. If the submodule is empty, they tell you to run
`git submodule update --init`; if it has no build tree, they print the cmake
commands above.

To use a separate checkout instead, pass its path as the first argument or set
`LLAMA_CPP_DIR` (and optionally `LLAMA_BUILD_DIR`); an explicit path always
wins over the submodule.

## Layout

- `inc/llama.bi` — top-level header (includes `inc/ggml/*.bi`).
- `inc/ggml/*.bi` — ggml headers.
- `inc/stdbool.bi` — `bool` shim (`type bool as boolean`).
- `examples/` — small demo programs, including `hello_llama.bas`.

## Build & run an example

From `examples/`, the scripts default to the vendored tree:

```sh
cd examples
./build_rpg.sh
./run_rpg.sh
```

The same works for `animal_tank` and `hello_llama`. To use a separate
checkout, pass its path as the first argument (or set `LLAMA_CPP_DIR`):

```sh
./build_rpg.sh /path/to/llama.cpp
./run_rpg.sh /path/to/llama.cpp
```

## Super basic example

`examples/hello_llama.bas` sends one prompt to the model and prints its reply.
The core generation loop looks like this:

```basic
dim usrPrompt as string = "Hello! Who are you?"

' ... load model, init context + sampler, tokenize prompt, decode prompt ...

print "Assistant: ";
dim nCur as long = nTok
dim nGen as integer = 0
while nGen < 120
    dim tok as llama_token = llama_sampler_sample(smp, ctx, -1)
    llama_sampler_accept(smp, tok)
    if llama_token_is_eog(vocab, tok) then exit while
    dim pbuf as zstring ptr = allocate(512)
    dim plen as long = llama_token_to_piece(vocab, tok, pbuf, 512, 0, true)
    if plen > 0 then print left(*pbuf, plen);
    deallocate(pbuf)

    ' feed the sampled token back into the context
    dim b2 as llama_batch = llama_batch_init(1, 0, 1)
    *(b2.token) = tok
    if b2.pos <> 0 then *(b2.pos) = nCur
    *(b2.n_seq_id) = 1
    if b2.seq_id <> 0 then **(b2.seq_id) = 0
    *(b2.logits) = 1
    b2.n_tokens = 1
    llama_decode(ctx, b2)
    llama_batch_free(b2)
    nCur += 1
    nGen += 1
wend
print
```

Build and run it (from the `examples/` directory):

```sh
./build_hello_llama.sh
./hello_llama
```

Or compile directly against the vendored build (from the project root):

```sh
fbc -i ./inc examples/hello_llama.bas -p vendor/llama.cpp/build-cuda/bin \
    -l llama -l ggml -l ggml-base -l ggml-cpu -l ggml-cuda \
    -x examples/hello_llama
cd examples && LD_LIBRARY_PATH=../vendor/llama.cpp/build-cuda/bin ./hello_llama
```

A separate checkout still works: pass it as the first argument to the build
script, or replace the `-p` / `LD_LIBRARY_PATH` paths above.

## Verifying the headers

```sh
printf '#include once "llama.bi"\n' > /tmp/check.bas
fbc -i ./inc -c /tmp/check.bas -o /tmp/check.o
```

# llama4fb

FreeBASIC `.bi` translations of the C headers from krafton-ai/llama.cpp (branch `raon-1-nano`). Source of truth: a checkout of that fork (`include/llama.h`, `ggml/include/*.h`). Translations live under `inc/`.

## Layout

- `inc/llama.bi` (top-level) includes `inc/ggml/*.bi` with the `ggml/` prefix.
- `inc/ggml/*.bi` mirror `ggml/include/*.h`.
- `#inclib`: `inc/ggml/ggml.bi` -> `"ggml"`, `inc/llama.bi` -> `"llama"`.

## Tooling

- Initial translation was generated with fbfrog. Further corrections are done by hand.
- Do not regenerate over hand edits. If you must regenerate, run fbfrog against the fork headers, then re-apply the path prefix (`ggml/`), `#inclib`, and manual fixes below.

## Verification

Syntax-check a header with fbc (no linker needed). `stdbool.bi` now ships in `inc/`, so no shim is needed:

```sh
printf '#include once "llama.bi"\n' > /tmp/check.bas
fbc -i ./inc -c /tmp/check.bas -o /tmp/check.o
```

Replace `llama.bi` with a specific `ggml/*.bi` to check that header. Run from the project root.

## Known pitfalls (FB vs C)

- FreeBASIC is case-insensitive. A macro and a function/enum/type with the same name (e.g. `GGML_PAD` vs `ggml_pad`, `llama_pooling_type` enum vs function) conflict. Rename the FB-side identifier and use `alias "..."` to keep the C symbol.
- Opaque C structs need forward declarations: `type X as X_`. Affected: `ggml_context`, `ggml_cgraph`, `ggml_object`, `ggml_backend_buffer`, `ggml_threadpool`, `gguf_context`, `llama_vocab`, `llama_model`, `llama_context`, `llama_sampler`, `llama_adapter_lora`, `llama_batch_ext`.
- C `bool` is not a FreeBASIC type; use `boolean`.
- Reserved parameter names (`as`) must be renamed.

## Source mapping

When updating after a fork change, diff the corresponding `.h` -> `.bi` and keep the include paths and `#inclib` lines intact.

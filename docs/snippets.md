# llama4fb snippets — the plumbing every example copies

The examples all repeat the same boilerplate: download the model, load it,
build a sampler, tokenize, fill a batch by hand, decode, generate. This
document collects those pieces as copy-paste functions so a new program
starts from working code instead of from dissecting an example. The
snippets are taken from the examples themselves (hello_llama, rpg_village,
animal_tank), lightly reshaped into functions. They target the llama.bi in
this repo (the KRAFTON llama.cpp fork); where this fork differs from
upstream llama.cpp, the difference is called out.

Conventions: globals are named g_model, g_ctx, g_vocab as in the examples.
All snippets assume `#include once "llama.bi"`.

## 1. Make sure the model file exists

Downloads the ~2 GB GGUF on first run. Call before loading. Run programs
from the examples/ directory (or wherever models/ should live); the path
is relative to the current directory.

```freebasic
sub ensure_model()
    dim modelPath as string = "models/raon-1-nano-Q4_K_M.gguf"
    if dir("models") = "" then mkdir "models"
    if dir(modelPath) = "" then
        print "Model not found. Downloading (~2GB). This may take a while..."
        shell "curl -L -o " & modelPath & " https://huggingface.co/KRAFTON/Raon-1-nano-gguf/resolve/main/raon-1-nano-Q4_K_M.gguf"
        if dir(modelPath) = "" then
            print "Download failed."
            sleep
            end
        end if
    end if
end sub
```

## 2. Load the model and create a context

```freebasic
dim shared g_model as llama_model ptr
dim shared g_ctx as llama_context ptr
dim shared g_vocab as const llama_vocab ptr

sub llm_load()
    ensure_model()
    dim modelPath as string = "models/raon-1-nano-Q4_K_M.gguf"
    dim mparams as llama_model_params = llama_model_default_params()
    mparams.n_gpu_layers = -1   ' offload everything if the llama.cpp build has CUDA
    g_model = llama_model_load_from_file(strptr(modelPath), mparams)
    if g_model = 0 then print "Failed to load model": sleep: end
    dim cparams as llama_context_params = llama_context_default_params()
    cparams.n_ctx = 2048
    cparams.n_batch = 512
    g_ctx = llama_init_from_model(g_model, cparams)
    if g_ctx = 0 then print "Failed to create context": sleep: end
    g_vocab = llama_model_get_vocab(g_model)
end sub
```

Cleanup at exit, in this order:

```freebasic
llama_free(g_ctx)
llama_free_model(g_model)
```

## 3. Build a sampler chain (per the model card)

For Raon-1-Nano general use the official card recommends temperature 0.1
and presence_penalty 1.5. The penalties sampler goes first in the chain.

FORK DIFFERENCE: in this fork, llama_sampler_init_penalties takes the
vocab size as its first argument (upstream does not). Check llama.h in
your checkout if a call fails to compile.

```freebasic
function make_sampler(temp as single) as llama_sampler ptr
    dim sparams as llama_sampler_chain_params = llama_sampler_chain_default_params()
    dim smp as llama_sampler ptr = llama_sampler_chain_init(sparams)
    ' (n_vocab, repeat_last_n, repeat_penalty, frequency_penalty, presence_penalty)
    llama_sampler_chain_add(smp, llama_sampler_init_penalties(llama_vocab_n_tokens(g_vocab), 64, 1.0, 0.0, 1.5))
    llama_sampler_chain_add(smp, llama_sampler_init_temp(temp))
    llama_sampler_chain_add(smp, llama_sampler_init_dist(42))
    return smp
end function
```

Notes: free it with llama_sampler_free when done. animal_tank rebuilds the
chain per decision to vary temperature per species; for most programs one
chain built once is right. Raise temp above 0.1 only where variety is the
point (dialogue, descriptions), and keep the penalties sampler in either
case or long outputs start repeating themselves.

Seed warning (learned Nov 24, 2026 from grammar_forge): a fixed dist seed
such as 42 is only an initial seed while the sampler keeps running. If the
program calls `llama_sampler_reset` before each generation, the reset
re-seeds back to that fixed seed, and an identical prompt replays an
identical token stream, however high the temperature is. Use
`llama_sampler_init_dist(LLAMA_DEFAULT_SEED)` (&hFFFFFFFF in llama.bi)
when each generation must vary; keep the reset itself, since it is what
clears penalties and grammar state between generations. rpg_village and
animal_tank do not reset per generation, so their 42 is initial-only and
not the same bug.

## 4. Build a ChatML prompt with thinking disabled

Raon-1-Nano is designed for thinking mode, but on this model disabling it
costs nothing in output quality and is drastically faster. The technique:
prefill the assistant turn with an empty think block.

```freebasic
function build_prompt(sysPrompt as string, usrPrompt as string) as string
    dim p as string
    p = "<|im_start|>system" & chr(10) & sysPrompt & chr(10) & "<|im_end|>" & chr(10)
    p &= "<|im_start|>user" & chr(10) & usrPrompt & chr(10) & "<|im_end|>" & chr(10)
    p &= "<|im_start|>assistant" & chr(10) & "<think></think>" & chr(10) & chr(10)
    return p
end function
```

For multi-turn conversations, repeat the user/assistant block pair per
turn instead of the prefilled assistant opener. (llama_model_chat_template
can generate the framing instead; the hand-built form above is what the
examples use and is easier to debug.)

## 5. Tokenize a string

llama_tokenize returns a negative count when the buffer is too small;
the absolute value is the count needed. This helper handles that.

```freebasic
' Caller owns the returned buffer and must deallocate it.
function tokenize_text(text as string, nOut as long ptr) as llama_token ptr
    dim cap as long = 4096
    dim toks as llama_token ptr = allocate(cap * sizeof(llama_token))
    dim n as long = llama_tokenize(g_vocab, strptr(text), len(text), toks, cap, true, true)
    if n < 0 then
        n = -n
        if n > cap then
            deallocate(toks)
            toks = allocate(n * sizeof(llama_token))
            n = llama_tokenize(g_vocab, strptr(text), len(text), toks, n, true, true)
        end if
    end if
    *nOut = n
    return toks
end function
```

## 6. Decode a token array into the context

The examples fill llama_batch fields by hand (llama_batch_get_one is not
used) because the batch struct in the translated header is easiest to
drive this way. Only the last token gets logits, since only it is sampled
from. Returns the llama_decode result; check it.

```freebasic
function decode_tokens(toks as llama_token ptr, nTok as long, startPos as long) as long
    dim batch as llama_batch = llama_batch_init(nTok, 0, 1)
    dim tokArr as llama_token ptr = batch.token
    dim posArr as llama_pos ptr = batch.pos
    dim nseqArr as long ptr = batch.n_seq_id
    dim seqArr as llama_seq_id ptr ptr = batch.seq_id
    dim logArr as byte ptr = batch.logits
    for i as integer = 0 to nTok - 1
        *(tokArr + i) = *(toks + i)
        if posArr <> 0 then *(posArr + i) = startPos + i
        *(nseqArr + i) = 1
        if seqArr <> 0 then *(*(seqArr + i)) = 0
        *(logArr + i) = 0
    next
    *(logArr + (nTok - 1)) = 1
    batch.n_tokens = nTok
    dim rc as long = llama_decode(g_ctx, batch)
    llama_batch_free(batch)
    return rc
end function
```

## 7. Generate tokens after a decoded prompt

Assumes the prompt ending at position startPos - 1 has been decoded
(snippet 6). Samples up to maxGen tokens, stops at end-of-generation.

```freebasic
function generate(smp as llama_sampler ptr, startPos as long, maxGen as integer) as string
    dim result as string = ""
    dim nCur as long = startPos
    for nGen as integer = 1 to maxGen
        dim tok as llama_token = llama_sampler_sample(smp, g_ctx, -1)
        llama_sampler_accept(smp, tok)
        if llama_token_is_eog(g_vocab, tok) then exit for
        dim pbuf as zstring ptr = allocate(512)
        dim plen as long = llama_token_to_piece(g_vocab, tok, pbuf, 512, 0, true)
        if plen > 0 then result &= left(*pbuf, plen)
        deallocate(pbuf)
        if decode_tokens(@tok, 1, nCur) <> 0 then exit for
        nCur += 1
    next
    return result
end function
```

To stream instead of accumulate, print each piece inside the loop as it
arrives; the token pieces are already produced one at a time.

## 8. One-shot ask (the pattern most programs actually want)

Stateless question and answer: clear the memory, build the prompt,
tokenize, decode, generate. This is rpg_village's generate_reply reduced
to its skeleton. Use it whenever each call is independent (agent
decisions, NPC replies, classification, parsing). For a conversation that
must remember earlier turns, do NOT clear the memory; append turns to one
sequence instead (see persistent_companion.md).

```freebasic
function ask_once(smp as llama_sampler ptr, sysPrompt as string, usrPrompt as string, maxGen as integer) as string
    llama_memory_clear(llama_get_memory(g_ctx), true)
    dim prompt as string = build_prompt(sysPrompt, usrPrompt)
    dim nTok as long
    dim toks as llama_token ptr = tokenize_text(prompt, @nTok)
    if decode_tokens(toks, nTok, 0) <> 0 then
        deallocate(toks)
        return ""
    end if
    deallocate(toks)
    return generate(smp, nTok, maxGen)
end function
```

## 9. Keep a reply ASCII-only

The examples ask for ASCII in the prompt and still filter afterwards,
because a small model occasionally emits a stray non-ASCII character that
breaks simple text rendering. Newlines are kept; tabs become spaces.

```freebasic
function ascii_only(s as string) as string
    dim out as string = ""
    for i as integer = 1 to len(s)
        dim c as integer = asc(mid(s, i, 1))
        if (c >= 32 and c <= 126) or c = 10 or c = 13 then
            out &= mid(s, i, 1)
        elseif c = 9 then
            out &= " "
        end if
    next
    return out
end function
```

## 10. A note on context size (n_ctx is a budget, not a spec sheet)

Every example sets n_ctx = 2048, while the model natively supports
262,144 tokens. That gap is deliberate, and the reasoning is worth
copying even when the number changes:

- The cost of context is paid twice. Prefill: the model must process
  every prompt token before the first output token, so prompt length is
  response latency. Residence: the KV cache grows with tokens actually
  held, so a long conversation costs memory and slows each decode step.
- What matters is the largest prompt (plus generated reply) the program
  will ever hold at once, not the largest the model allows. The current
  examples hold a few hundred tokens per call by design (stateless asks,
  short personas), so 2048 is headroom, not a limit being hit.
- Raise n_ctx when the design holds more: persistent_companion is the
  example built around this, and its token counter exists so the budget
  is visible instead of discovered by a crash. When you raise it, raise
  n_batch with an eye on prefill speed for long prompts.
- One model-specific mercy: Raon's hybrid architecture (mostly linear
  attention layers with fixed-size state, a minority of full-attention
  layers) makes long contexts cheaper than a pure transformer of its
  size would. Cheaper is not free; prefill time still scales with what
  you feed it.

Rule of thumb for this repo: size the context to the program's actual
working set, print or log usage where the budget can bite, and treat
"the model supports 262K" as permission, not instruction.

## 11. Grammar-constrained generation (from grammar_forge)

Add a grammar sampler FIRST in the chain; it masks every token that
would make the output impossible to complete under the GBNF grammar, so
what reaches the later samplers is already shape-valid. Temperature can
then run hot for variety, because correctness no longer depends on it.

```freebasic
' The grammar is one string; keep it next to the type it describes.
dim g as string
g = "root ::= " & q & "{" & q & " ..." & chr(10)   ' rules, one per line

dim sparams as llama_sampler_chain_params = llama_sampler_chain_default_params()
g_smp = llama_sampler_chain_init(sparams)
llama_sampler_chain_add(g_smp, llama_sampler_init_grammar(g_vocab, strptr(g), strptr("root")))
llama_sampler_chain_add(g_smp, llama_sampler_init_penalties(llama_vocab_n_tokens(g_vocab), 64, 1.0, 0.0, 1.5))
llama_sampler_chain_add(g_smp, llama_sampler_init_top_k(40))
llama_sampler_chain_add(g_smp, llama_sampler_init_top_p(0.9, 1))
llama_sampler_chain_add(g_smp, llama_sampler_init_temp(0.8))
llama_sampler_chain_add(g_smp, llama_sampler_init_dist(LLAMA_DEFAULT_SEED))
```

The seed choice in the last line is load-bearing. `llama_sampler_reset`
re-seeds the dist sampler, so a chain reset before each generation with
a FIXED seed replays the identical random stream: same prompt, same
output, forever. grammar_forge shipped that bug (seed 42 plus a reset in
the generation function) and its 100-item demo passed a parse-count gate
while producing only ~20 distinct items. Use LLAMA_DEFAULT_SEED when
variety matters; use a fixed seed when reproducibility matters (tests),
and if you verify generated output, count DISTINCT outputs, not just
well-formed ones. A shape counter cannot see sameness.

Extraction: a scanner would suffice while the grammar guarantees the
shape, and grammar_forge shipped that way at first. It now parses with
fbejson (the vendored library at vendor/fbejson), checking each field's
presence and type, because the payoff showed up in the failure data:
with the grammar removed, the model still emits valid JSON, and only a
real parser separates "invalid json" from "missing field" from "right
field, wrong type." If a field is missing from grammar-shaped output,
the grammar was not actually applied; treat that as a bug, not data.

## 12. Embeddings: text in, ranked entries out (from ask_the_codex)

A separate context in embedding mode; no sampler, no generation. Create
it with embeddings on and the pooling the MODEL specifies (Qwen3-
Embedding is a decoder: LAST-token pooling; a BERT-style model would
want MEAN). Read the vector size from the model, never hardcode it.

```freebasic
cparams = llama_context_default_params()
cparams.n_ctx = 1024 : cparams.n_batch = 1024
cparams.pooling_type = LLAMA_POOLING_TYPE_LAST
cparams.embeddings = true
g_ctx = llama_init_from_model(g_model, cparams)
nEmbd = llama_model_n_embd(g_model)
```

Embedding one text: tokenize (add_special false), append the model's
EOS token (last-token pooling is defined over that final position),
clear the memory, decode the whole text as sequence 0 with the output
flag set on the LAST token only, copy the floats out of
llama_get_embeddings_seq immediately (the pointer dies at the next
decode), and normalize to unit length.

```freebasic
*(toks + nTok) = llama_vocab_eos(g_vocab) : nTok += 1
llama_memory_clear(llama_get_memory(g_ctx), true)
batch = llama_batch_init(nTok, 0, 1)
' fill token/pos/seq_id 0 for all positions; logits flag 1 on last only
llama_decode(g_ctx, batch) : llama_batch_free(batch)
src = llama_get_embeddings_seq(g_ctx, 0)   ' copy nEmbd singles NOW
' divide each component by the vector's L2 norm
```

With every vector unit-normalized, ranking is a dot product: score each
stored vector against the query vector, sort descending, take the top
few. Two habits carry the quality:

- Query/document asymmetry where the model asks for it. Qwen3-Embedding
  wants queries wrapped as "Instruct: <task>" + newline + "Query: <q>"
  and documents embedded raw. Forgetting the prefix is the most common
  way to make a good embedding model look mediocre.
- Model-specifics (file, pooling, prefixes) as named constants at the
  top of the file, so swapping embedding models is a constants change.
  On the codex corpus this setup runs ~7 ms per query and ~12 ms per
  entry at index time; the whole 24-entry index builds in ~0.3 s.

## 13. Tool calling without a templating engine (from text_adventure)

Raon's GGUF chat template (dump it with llama_model_chat_template;
never design from another model's habits) frames tool use like this:
the system prompt carries a "# Tools" section with the definitions as
JSON inside <tools>...</tools>; a call comes back as
<tool_call><function=name><parameter=key>value</parameter></function>
</tool_call> with NO other text; a tool result goes back as a user turn
containing <tool_response>...</tool_response>. Three small helpers
replicate exactly that slice:

- build_prompt(messages, tools): renders system/user/assistant/tool
  turns in the template's framing, tools rendered into the system turn,
  thinking off via the usual prefill. Build the definition JSON with
  fbejson (vendored at vendor/fbejson) instead of concatenating strings.
- parse_tool_call(reply): find the <tool_call> block, read the function
  name and each <parameter=key>value</parameter> pair. Keep an fbejson
  path beside it for JSON-shaped calls ({"name": ..., "arguments":
  {...}}): loadString, child("name"), child("arguments"), values via
  getString, or getNumber formatted back to a digit string. No block in
  either format is a distinct failure result, not an exception.
- append_tool_result(messages, name, result): appends the result as the
  template's user-turn <tool_response> framing.

Two habits carry the reliability: list the canonical argument values
(object names, directions) in the prompt's context block, so the model
copies "rubber chicken" instead of inventing "chicken"; and validate
every parsed call against world state in code before acting on it. The
model interprets; the code decides. Parse generation uses the card's
Agentic row (temp 1.0, presence_penalty 0.0).

## 14. Session save/load and the token budget (persistent_companion)

One context for the whole session; turns append, nothing clears. Save
with llama_state_save_file(ctx, path, tokens, n_tokens) after every
exchange and on exit; restore with llama_state_load_file into a token
buffer (the companion uses 16384) and continue decoding from there.
The token array is saved alongside the state because the caller owns
the sequence. Budget honestly: track tokens used, require
used + reserve (256) < n_ctx before generating, and when the wall
approaches, stop gracefully with a fixed explanation instead of
truncating or summarizing silently. One hard-won caveat: a loaded
session carries its ORIGINAL system prompt in the KV cache, so edits
to the system prompt do not retro-apply to a resumed conversation;
they take effect from the next fresh save.

## What is deliberately not here (yet)

- Grammar-constrained generation lives in section 11, embeddings in
  section 12, tool calling in section 13. Still absent: audio,
  batching multiple sequences, vision.

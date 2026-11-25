# hello_llama — the floor

Status: exists (examples/hello_llama.bas).

## What it showcases
The smallest complete path from "I have the translated headers" to "the
model answered me": model download and load, context creation, a sampler
chain, ChatML prompt assembly with thinking disabled, tokenization, prompt
decode, and a token-by-token generation loop. It is the reference for the
plumbing every other example copies, and the first thing a new user runs to
prove their build works.

## How it works
One hardcoded system and user prompt, one generation of up to ~150 tokens,
reply filtered to ASCII and printed. Nothing else. That emptiness is the
point: any line a reader does not understand is a line they need, because
there is no game logic to hide behind.

## Limitations
- The prompt is hardcoded; there is no user input at all.
- The sampler (temp 0.7, top_k 40, top_p 0.9, no penalties) does not follow
  the model card's general-use recommendation (temp 0.1, presence_penalty
  1.5). For a one-shot greeting this is harmless; as copied boilerplate it
  teaches the wrong default.
- The ChatML template is hand-assembled with string concatenation,
  including the prefilled empty think block. It will silently rot if the
  model's template changes.
- Batch construction is fully manual (writing token/pos/seq/logits arrays
  by hand), which is instructive once and error-prone forever after.
- No failure handling beyond load failure: decode results, sampler state,
  and tokenization edge cases are unchecked.
- The ASCII filter discards non-ASCII output instead of handling UTF-8.

## Improving on it in your own implementation
- Extract the init/tokenize/decode/generate plumbing into a shared module
  the first time you write a second program; do not copy this file's
  inline version repeatedly.
- Use llama_model_chat_template to build prompts instead of hand-rolled
  ChatML strings, and keep the thinking-off technique as one deliberate,
  commented step.
- Set the sampler from the model card for your use case (low temperature
  plus presence penalty for factual replies; raise temperature only where
  variety is the goal), and say so in a comment.
- Stream tokens to the screen as they are generated instead of printing
  the finished reply; the generation loop already produces them one at a
  time.
- Check decode return values and handle the context filling up; a real
  program needs a policy for both.

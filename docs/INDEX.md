# llama4fb example showcase docs

One document per example, describing what the example is made to showcase,
what it deliberately limits or simplifies, and how a real implementation
should improve on it. The doc is the brief for building (or rebuilding) the
example: if a behavior is not serving the showcase purpose stated here, it
does not belong in the example.

Existing examples:
- hello_llama.md — the floor: one prompt, one reply, minimal plumbing.
- animal_tank.md — autonomous agents: a continuous multi-agent decision loop.
- rpg_village.md — NPC dialogue with structured output the game can parse.

Planned examples (agreed Nov 23, 2026, in this build order):
1. grammar_forge.md — grammar-constrained generation: guaranteed-parseable JSON.
2. ask_the_codex.md — embeddings: semantic search over game lore.
3. text_adventure_parser.md — intent classification: freeform text to game commands.
4. persistent_companion.md — context lifecycle: a real multi-turn conversation.
5. procedural_describer.md — state-grounded generation: descriptions from world state.

Companion document:
- snippets.md — the repeated plumbing (model load, sampler chain, prompt
  building, tokenizing, batch decode, generation) collected as copy-paste
  functions, so new programs start from working code. Grows as new
  examples land (embeddings, grammar, session state).

Hardware expectations: the examples are developed and tuned on a desktop
with an RTX 4070. They offload to the GPU automatically when the
llama.cpp build provides a backend (n_gpu_layers = -1 at load): CUDA on
NVIDIA, Vulkan on AMD, and the build/run scripts detect a Vulkan build
tree automatically when no CUDA build is present. Verified Nov 24, 2026
on a Radeon iGPU with a GGML_VULKAN build of the fork: the examples ran
with GPU offload confirmed in the logs. CPU-only works too, but not at
the same feel.
One-shot and turn-based examples (hello_llama, grammar_forge,
ask_the_codex, text_adventure_parser, persistent_companion, rpg_village)
are fine on any reasonably modern CPU: each generation is a short wait,
not a throughput problem. The continuous examples are the sensitive ones:
animal_tank's decision loop is the first thing to tune on weaker hardware
(raise DECIDE_INTERVAL, fewer animals; the round-robin decider degrades
gracefully, animals simply think less often). Every doc and README should
state this plainly rather than letting a reader on integrated graphics
conclude the library is slow.

Shared conventions for every example:
- Runs from the examples/ directory; the model (Raon-1-Nano Q4_K_M) is
  auto-downloaded to examples/models/ on first run.
- Thinking mode stays off (prefilled empty think block or equivalent);
  it is drastically slower with no output difference on this model.
- Sampling follows the model's official card unless the doc says otherwise:
  general use is temperature 0.1 with presence_penalty 1.5; variety is a
  deliberate, documented choice per example, not an accident.
- Every example must build with a build_<name>.sh script next to it,
  taking the llama.cpp checkout path as an argument.
- Every example except hello_llama (the basic one) supports two standard
  command-line flags, following animal_tank's implementation:
  `--log` appends a structured session log to `<example>.log` in the
  working directory (decisions, prompts' outcomes, state changes; the
  file a verification run quotes as evidence), and `--demo` runs the
  example as a self-driving showcase without interactive input (the tank
  auto-cycles its camera across species; dialogue and turn-based
  examples play a scripted session end to end). Combined, `--demo --log`
  is the standard headless verification run for an example.
- Demo pacing standard (Ebben, Nov 24, 2026, from rpg_village): a demo
  is paced to be WATCHED, because it doubles as a promo video. Turn-based
  examples type their scripted commands in one letter at a time and
  pause after each result so the text can be read, landing the whole
  run at roughly 90 seconds before exiting on their own. During a demo,
  user input is ignored except Escape, which exits cleanly. A demo that
  blasts through its script as fast as the model answers is a failed
  demo even if its log is perfect.

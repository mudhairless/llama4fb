# grammar_forge — guaranteed-parseable generation

Status: built Nov 24, 2026 (llama4fb a8a36274, seed fix dd79cca).
File: examples/grammar_forge.bas. Demo gate: 100 generations, 100 parsed,
0 failures, 100 distinct outputs; the --no-grammar counter-run failed
100 of 100. Owner play-test confirmed variety after the seed fix.

## What it showcases
The grammar sampler (llama_sampler_init_grammar with a GBNF grammar): the
model's output is constrained token by token so it is guaranteed to match
a JSON schema shape. This is the technique that turns a small model from a
chatty text source into a reliable data source. The example generates game
items (weapons) as JSON, parses the output into a real user-defined type
with no error path for malformed output (there cannot be any), and puts
the item into a tiny inventory display with its generated name, stats, and
flavor line. It exists to teach the single most valuable habit for small
models: never parse freeform text when you can constrain the generation.

## How it should work
A keypress asks the model to invent a weapon for a given slot and rarity
(supplied in the prompt). The grammar forces exactly the fields the game
needs: name (string), damage (integer in range), speed (integer), element
(one of an enum), flavor (short string). The program parses the JSON with a
small hand-written extractor, constructs the item, and shows it. A counter
of generated items and parse failures is displayed; parse failures must
stay at zero across any number of generations, and that counter is the
example's proof.

## Limitations (deliberate, and to state in the example's README blurb)
- The grammar constrains shape, not sense. The model can still emit a
  damage of 999 if the grammar allows the digits, or flavor text that
  contradicts the element. Grammar is not validation.
- One fixed schema for one item type; no nesting beyond one level.
- The GBNF grammar is hand-written for this schema and must be maintained
  alongside the type definition; the two can drift.
- Generation is one item at a time, blocking on a keypress.
- JSON extraction uses fbejson, the repo's vendored JSON library
  (submodule at vendor/fbejson), with per-field presence and type
  checks. It began as a minimal string scanner, adequate because the
  grammar guarantees the shape; the retrofit (Nov 24, 2026, llama4fb
  1067ee41) traded that minimalism for a real parser and a failure
  taxonomy: in the --no-grammar counter-run, all 100 unconstrained
  outputs were still valid JSON, and the failures classified as
  wrong-typed fields (72), missing fields (28), invalid JSON (0),
  which a key-hunting scanner could not have distinguished.

## Improving on it in your own implementation
- Mind the sampler seed. A fixed seed plus a per-generation sampler
  reset makes identical prompts produce identical items (this example
  shipped that bug; the owner's play-test caught what the parse counter
  could not). Use LLAMA_DEFAULT_SEED for variety, a fixed seed when you
  want reproducible runs, and count distinct outputs when you verify.
- Always pair grammar with a semantic validation pass: clamp or reroll
  out-of-range stats, check enums against game rules, and reject flavor
  that violates content constraints. Grammar gets you parseable; your code
  must still decide acceptable.
- Generate the GBNF from your schema definition (or generate both from one
  source table) so the grammar and the type cannot drift apart.
- Compose grammars for nested structures (an NPC with an inventory of
  items) once flat objects are comfortable; the technique scales, the
  hand-writing is the cost.
- Reuse the pattern anywhere the model feeds game state: action selection
  (replacing substring parsing, as animal_tank should), quest parameters,
  dialogue unlock markers (replacing rpg_village's fuzzy matching).
- Cache or pre-generate content at load time if generation latency would
  interrupt play; constraint does not make generation instant.

## Done when
- Builds and runs like the other examples; 100 consecutive generations
  produce 100 parsed items and a parse-failure count of zero.
- Variety is proven, not assumed: the same slot and rarity forged five
  times in a row must produce five different items, and a 100-item demo
  run must show at least 90 distinct raw outputs. A parse counter cannot
  detect identical outputs (learned Nov 24, 2026: the demo passed
  100/100 on shape while every repeat of a slot/rarity was byte-identical,
  because `llama_sampler_reset` re-seeded the fixed-seed dist sampler
  each generation; the fix is `llama_sampler_init_dist(LLAMA_DEFAULT_SEED)`).
- The grammar demonstrably bites: removing it (a debug toggle) produces
  visibly malformed or chatty output within a few generations, showing the
  constraint is doing the work.

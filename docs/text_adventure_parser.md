# text_adventure_parser — freeform text to game commands

Status: built Nov 24, 2026 (llama4fb, initial build plus the owner's
art-polish rounds; HEAD 6deeec00 at verification). File:
examples/text_adventure.bas. Demo gate passed with room to spare:
20/20 intents matched (both misspellings included), the nonsense line
took the clarification path, the game was won through the parser, and
the paced demo ran about 81 seconds. The owner play-tested interactively
and directed several drawing upgrades (3D furniture, checkerboard
floor, sprites, clamped portrait pupils).

Two findings from the build that override this brief's assumptions:
- Raon's tool calls are NOT JSON. The GGUF template declares tools as
  JSON inside <tools> in the system prompt, but calls come back as
  <tool_call><function=name><parameter=key>value</parameter></function>
  </tool_call>, and results are framed as <tool_response> inside a user
  turn. The helpers are built around that real format (this is why the
  build began with a template dump, not an assumption).
- fbejson therefore works both ends: it builds and serializes the tool
  definitions, and it parses a JSON-shaped call when one appears
  (name/arguments via child/getString/getNumber, numbers normalized to
  digit strings); the tag format is parsed by the tag scanner. The
  library is vendored as a git submodule at vendor/fbejson.
Verification caught one report-versus-code mismatch (the penalties
presence term was 1.5 where the design and the run's own report said
the agentic row's 0.0); fixed the same evening (llama4fb d8d4f74b),
demo still 20/20. The chain as shipped is the agentic row throughout.

## What it showcases
Intent classification: using the model as a natural-language front end for
a deterministic game engine. The player types anything ("grab the lantern
and light it", "head north", "ask the keeper about the sword"); the model
maps it to a structured intent from a small fixed set (TAKE/DROP/GO/LOOK/
ASK/USE with arguments), the engine executes intents it understands, and
anything unparseable gets an in-character clarification rather than the
classic "I don't know the word 'xyzzy'". It demonstrates the division of
labor the whole repo teaches: the model interprets, the code decides.
A tiny hand-built world (a handful of rooms, objects, and one NPC) exists
only to give the parser something real to act on.

## How it should work
A text prompt at the bottom of the screen. Each input line goes to the
model with the intent list, the visible objects, and the room context;
the model replies with one intent line (ideally grammar-constrained, if
grammar_forge has landed, its technique applies directly). The engine
validates the intent against the world (you cannot take what is not here)
and narrates the result with ordinary code. A visible "intent:" echo shows
the player (and the reader) exactly what the model extracted.

## Tool calling integration (added Nov 23, 2026)
Raon-1-Nano is explicitly trained for tool calling (it is a headline
feature on the model card, with a BFCL v4 function-calling score of 56.3
and a dedicated Agentic sampling row: temperature 1.0, presence_penalty
0.0). This example is the natural home for that capability: the intent
vocabulary IS a tool set. Define the engine's verbs as tools with typed
arguments (take(object), drop(object), go(direction), ask(npc, topic),
use(item, target)) and let the model's native tool-call output produce
them, instead of an invented intent-line format. The division of labor is
unchanged (model interprets, code validates and executes), but the
contract between them is now the model's trained interface rather than a
prompt convention. Implementation notes for the build:
- The exact tool-call delimiters and JSON shape come from the model's own
  chat template inside the GGUF (dump it via llama_model_chat_template at
  build time and design around what it actually emits). Do not invent a
  format from memory of other models.
- The raw llama.h API has no tool-call parser (that lives in llama.cpp's
  common library, a C++ support library outside the translated C headers),
  and nobody needs a Jinja engine for one model's template. Instead the
  example implements three small helpers that replicate just the slice of
  behavior this template needs, with the limitations stated up front:
  (1) build a prompt from a message list plus a tool-definition list,
  following the template's fixed framing for system/user/assistant/tool
  turns; (2) parse a tool-call block out of a reply into a name and an
  argument list; (3) append a tool result as the template's result turn
  and continue. Limitations: the helpers encode THIS model's template as
  shipped in the GGUF, so a future model revision that changes its
  template means updating the helpers (dump the template again and diff);
  they render only the paths these examples use (tools present or absent,
  thinking off), not arbitrary template logic; and they will not carry
  over to a different model's template unchanged. That trade, a page of
  string code instead of a templating engine, is the point of the
  example, and the helpers are shared with persistent_companion.
- Use the card's Agentic sampling (temp 1.0, presence 0.0) for the parse
  generation, and say so in a comment; it contrasts deliberately with the
  General row used elsewhere in the repo.
- Keep the grammar-constrained intent format as the documented fallback
  for models without tool training; the engine side does not care which
  produced the intent.

## Limitations (deliberate)
- A small fixed intent vocabulary and a small world; this is a parser
  demo, not a game.
- One intent per input line in the base version, even when the player
  types two actions.
- No pronoun or reference resolution across turns ("take it" after looking
  at the key) beyond what the immediate context supplies.
- The model will occasionally map odd phrasings to the wrong intent;
  the engine's validation and the clarification fallback are part of the
  demo, not a failure of it.
- Latency per command is one generation; fast typists will feel it.

## Improving on it in your own implementation
- Constrain the intent output with a grammar (see grammar_forge) so the
  reply is always a well-formed intent or an explicit UNKNOWN token.
- Split multi-action input into a sequence of intents and execute them in
  order, stopping at the first failure with a specific message.
- Keep a small explicit focus state (last mentioned object, last room
  contents) in code and pass it in the prompt to resolve "it", "that",
  and "the other one" deterministically.
- Log misclassified inputs during playtesting and grow the prompt's few
  examples from real failures; intent prompts improve from data, not from
  adjectives.
- Cache the parse for repeated identical commands in the same context.

## Done when
- A scripted list of varied phrasings for each intent (including
  misspellings, synonyms, and wordy forms) executes the intended action,
  and deliberately nonsensical input produces the clarification path, not
  a wrong action or a crash.

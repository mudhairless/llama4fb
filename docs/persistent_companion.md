# persistent_companion — a conversation that actually continues

Status: LANDED Nov 25, 2026 (llama4fb acd378c; built by the hy4 flash
free-pool model, verified against the pushed code,
penalties line included). examples/persistent_companion.bas. The demo
gate passed in full: run 1 planted both facts, exercised all four
tools, and saved at 1371 tokens; run 2 loaded at exactly 1371 and
recalled all three facts (captain, mission, sensor note); the
--ctx 1024 run hit the graceful stop and exited cleanly. Two recall
misses were fixed in the system prompt only, as designed (an explicit
write_note trigger rule; "what's my name" means the captain). One
subtlety the run surfaced, worth knowing: a loaded session carries
its ORIGINAL system prompt in KV, so prompt changes only take effect
from a fresh save. Interactive play remains the owner's acceptance.

The console UI was then converted to a graphics window the same day
(llama4fb 293f07f..9751c08, owner-steered follow-ups included,
verified against the pushed code; owner acceptance Nov 25, 2026:
"my testing shows it all working"). An 800x600 fbgfx window in three
bands: a header with a line-drawn Cartographer readout (blinking
lights, fake telemetry) and Pip's face; the adventure's 64-line log
ring, taller (26 visible lines) and CONVERSATION-ONLY in the window
(inputs and finished replies; tool calls, results, token status and
verdicts stay in the --log file, which is unchanged); and a 3-line
wrapping input box with a hard capacity cap and Backspace editing.
Two owner-directed refinements worth recording: the token budget is
also an 18-chunk LED gauge between the panels (green at the bottom,
red at the top, topmost LED only at 100%), and Pip's face settled as
two amber eyes and a mouth on a comms panel after iterations through
a single HAL-style eye and a survey-drone design; the eyes-and-mouth
version is the shipped one. The engine, sampler, tools and lifecycle
were untouched throughout; the demo gates re-passed under the window
(run 1 SAVE 1396 = run 2 LOAD 1396, all recalls OK, --ctx 1024
graceful stop clean).

## What it showcases
Context lifecycle management: the one major API discipline none of the
current examples demonstrate. Every existing example clears the model's
memory and re-prompts from scratch for each generation. This one keeps a
real multi-turn conversation alive in one sequence: turns are appended,
the KV cache grows, and the companion (a ship's computer, a familiar, a
rival adventurer) remembers what was said earlier in the session because
it is genuinely still in context. It also demonstrates the two things a
long conversation forces on you: watching the context budget, and saving
and restoring a session (llama_state_save_file / llama_state_load_file)
so a conversation survives quitting the program.

## How it should work
A console-style chat with one strongly written persona. The player chats
freely; the program appends each exchange to the running context without
clearing it. A small status line shows tokens used versus the context
size. On exit the session state is saved; on start, if a save exists, the
player can resume the same conversation and the companion demonstrably
remembers it.

## Tool calling integration (added Nov 23, 2026)
A companion that can only talk is a chatbot; a companion with tools is an
agent, and Raon is trained for exactly this. Give the companion a small
tool set it can invoke mid-conversation: roll_dice(sides, count) for
games within the fiction, look_in_bag(item) against a code-side inventory,
write_note(text) / read_notes() against the durable-facts notebook this
doc already specifies as an improvement, and (if ask_the_codex has
landed) search_codex(query) so the companion can look things up instead
of inventing them. The loop is the canonical one: model emits a tool
call, code executes it and appends the result as a tool turn, the model
continues with the result in context. This also deepens the example's
core subject: tool and result turns consume context budget and live in
the saved session state, which the token counter makes visible. Use the
model card's Agentic sampling row (temperature 1.0, presence_penalty 0.0)
for turns where tools are offered, and parse the tool-call block emitted
per the model's own chat template (the raw API ships no parser; see
text_adventure_parser.md for the shared approach, which should be factored
so both examples use the same parsing code).

## Limitations (deliberate)
- When the context fills, the base version stops gracefully and says so.
  There is no summarization or sliding-window strategy; the wall is shown
  honestly rather than hidden.
- One companion, one sequence, no parallelism.
- Memory is verbatim context only: anything that scrolls out (in a fuller
  implementation) or predates the session is unknown to the companion, and
  the persona prompt is the only long-term truth.
- Persona consistency still rests on the system prompt and the sampler;
  a long session will test both.

## Improving on it in your own implementation
- Implement context compaction: when the budget runs low, summarize the
  oldest turns with the model itself, replace them with the summary, and
  keep recent turns verbatim. This is the standard production answer and
  the natural next step from this example.
- Extract durable facts from the conversation (names, promises, events)
  into a code-side notebook injected at session start, so memory survives
  even compaction. animal_tank's event memory and rpg_village's history
  are primitive forms of this.
- Save per-companion state files and metadata (turn count, last played)
  rather than one global save.
- Budget deliberately: reserve output space, show or log the token count,
  and choose the context size from the session length you intend to
  support rather than accepting defaults.

## Done when
- The companion correctly answers a question about something mentioned
  many turns earlier in the same session; after quitting and relaunching
  with the saved state, it still can.
- Filling the context produces the graceful stop, not a crash or silent
  truncation.

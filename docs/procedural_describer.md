# procedural_describer — descriptions generated from world state

Status: LANDED Nov 25, 2026 (llama4fb 15b5ccd; verified against
the pushed code, penalties line and contamination check included). examples/procedural_describer.bas (the shipped name
follows the example name, not this doc's original room_describer
proposal). The demo gate passed in full: 7 fact-list entries matching
the specified floor (walls respected), CHEST-LOOTED-OK,
TRAP-SPRUNG-OK, REDESCRIBE-DIFFERS-OK, DOOR-OPEN-OK, and
NO-CROSS-CONTAMINATION (the obelisk stayed in the Shrine). The run
also demonstrated this doc's stated limitation honestly: several
descriptions invented "empty shelves" no fact list contains, and the
report disclosed it rather than hiding it; the Tab debug panel is
what makes that visible. Demo ran 38 s (the ~90 s pacing guidance
assumed slower generation; on CUDA each description takes under a
second). Owner's interactive walk is the remaining acceptance.

Two follow-ups landed the same day (81215d4). First, input: the
owner found the player could not move. Two stacked bugs: the loop
read inkey twice so the first read consumed each key, and after
switching to multikey the edge checks tested = 1 where FreeBASIC
returns -1 for a held key. Both fixed; Escape stays on inkey.
Second, a design evolution worth more than the fix: each room's fact
list gained three DESIGNER sensory notes (a smell, the air, a sound
marked nearby or far), code data shown in the debug panel, and the
prompt now asks the model to rewrite those notes into atmosphere
instead of inventing its own. The "empty shelves" inventions happened
because the prose wanted sensory detail and the facts offered none;
feeding sanctioned sensory facts cut the invention down (residual
slips remain: occasional first person, exits called "paths", one
invented "stacked containers"). Descriptions went to four or five
sentences with a 200-token cap. The lesson generalizes: when a
grounded generator keeps inventing the same KIND of detail, the fix
is often to add that kind to the fact list, not to forbid harder.

## What it showcases
State-grounded generation: prose produced from structured facts the code
knows, so the text can be evocative without being allowed to invent the
world. A tiny grid of rooms (a roguelike floor or a small dungeon) has
real state per room: contents, exits, features, and a history of what has
happened there (a fight, a looted chest, a trap sprung). On entering a
room, the program builds a prompt from that state alone and the model
writes the room's description. Return after the fight and the description
reflects the aftermath, because the state changed. The discipline being
taught is the prompt-as-report: the model narrates the database, it does
not replace it.

## How it should work
Arrow keys move between a dozen connected rooms on one screen. Each room's
state is a small record; the description prompt lists its facts in words
(location type, notable contents, exits, last event) and asks for two or
three sentences in a consistent tone set by the system prompt. A debug
toggle shows the fact list beside the generated text so readers can see
exactly what the model was given, and verify it invented nothing.

## Limitations (deliberate)
- Descriptions are regenerated per visit from current state; without a
  cache, the same unchanged room can be described differently on two
  visits, which breaks continuity.
- The model can still embellish beyond the facts (a smell, a sound, a
  carving that does not exist). Grounding reduces invention; it does not
  eliminate it, and the example should not pretend otherwise.
- One floor, fixed room graph, minimal game (no monsters or combat of its
  own; events are scripted or triggered by simple actions like opening a
  chest).
- Tone consistency across rooms rests entirely on the system prompt and
  sampler settings.

## Improving on it in your own implementation
- Cache each description keyed by a hash of the room's state; regenerate
  only when the state changes. This fixes continuity and removes most of
  the generation latency from ordinary movement.
- Keep a per-room consistency note (one line of established descriptive
  facts: "torch sconces on the north wall") written once and fed back into
  later prompts for that room, so necessary embellishments become canon
  instead of contradictions.
- Separate immutable facts (exits, contents) from color in the prompt, and
  instruct the model that the fact list is exhaustive for anything the
  player can interact with; anything else it adds is atmosphere only and
  must never be actionable.
- Pre-generate descriptions for the whole floor at load time once the
  cache exists, so play is never blocked on generation.

## Done when
- Entering a room, changing its state (loot the chest, spring the trap),
  leaving, and returning produces a new description that reflects the
  change and contradicts nothing in the fact list, with the debug view
  making that checkable at a glance.

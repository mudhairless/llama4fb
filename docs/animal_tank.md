# animal_tank — autonomous agents in a living world

Status: exists (examples/animal_tank.bas). The flagship example.

## What it showcases
A continuous multi-agent decision loop: many independent agents share one
small model, each perceiving a local slice of a simulated world, choosing
an action from a fixed vocabulary, and living with the consequences. It
demonstrates the techniques that make a 1B-class model usable as a game
brain: perception built in code (vision lists ordered threats-first, needs
expressed as words with numbers in parentheses), a tiny action grammar
(MOVE/EAT/DRINK/MATE/REST), short memory carried in the prompt (last
result, last three events, a sticky current goal), per-species sampling
temperatures, a penalties sampler matched to the model card, and code-side
reflexes and keepers (flee reflex, population floor, the crocodile's
water pull-back) for anything too urgent or too structural to leave to the
model. The crocodile additionally showcases a system-maintained unique
agent: exactly one, invisible in its habitat, with habitat rules enforced
in code rather than requested in the prompt.

## How it works
Each animal re-decides every 2 seconds (one shared context, memory cleared
per decision, prompt rebuilt from current state). Needs decay continuously;
starvation, combat, mating, corpses, and population floors are ordinary
game code. The model only ever answers one question: what do I do next.

## Limitations
- Decisions are sequential and blocking: one animal thinks at a time, so
  decision rate caps how many agents the tank can support.
- Agent memory is three short event strings and one goal; there is no
  longer-term memory, learning, or relationship tracking.
- Action parsing is substring matching on a freeform reply. It works
  because the vocabulary is tiny and the temperature is low; it is not a
  robust contract.
- Species behavior, speeds, damage, and balance live in select-case tables
  scattered through one large file; tuning means editing code.
- The simulation's interesting failures are emergent and seed-dependent;
  there is no replay or seed control for reproducing a specific incident.

## Improving on it in your own implementation
- Constrain the action reply with a grammar sampler (see grammar_forge)
  so parsing can never fail, instead of substring matching.
- Use multiple sequences (n_seq_id > 1) or batched decode to let several
  agents think per pass when the agent count grows.
- Externalize species tables into data (a config file or data statements)
  so balance tuning does not require recompiling.
- Give agents a compact long-term memory (a rolling one-line summary per
  agent, updated at decision time) once behavior needs history longer than
  three events.
- Add a fixed-seed mode and an event log with timestamps when you need to
  debug emergent behavior; watching and guessing stops scaling quickly.
- Keep the example's core discipline: anything that must always be true
  (exactly one crocodile, prey flee in time, no species goes extinct) is
  enforced in code. The prompt is for judgment calls, not guarantees.

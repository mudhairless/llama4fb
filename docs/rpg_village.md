# rpg_village — NPC dialogue and a distributed mystery

Status: exists (examples/rpg_village.bas). The improvement pass landed
Nov 24, 2026: marker-only unlocks with per-topic visibility and lenient
matching, real NPC memory, sampler aligned to the model card, an ending
(journal, accusation, resolution), and tone of address. Remaining known
deficiencies are flagged under Limitations.

## What it showcases
Dialogue as a game system rather than a chat window. Five NPCs hold a
distributed mystery: each persona prompt carries what that character knows,
what they do not know, and what they must not speculate about, so the
player assembles the truth by triangulation and no single conversation
hands it over. It also demonstrates structured side-channel output: an NPC
reply may end with an [UNLOCK: subject] marker that the code parses to
expand that NPC's topic list, so the model grows the dialogue tree as the
fiction warrants. Repeat questions are a mechanic, not a bug: asking again
pressures the NPC to reveal more or get annoyed (ask count passed into the
prompt).

## How it works
Walk up, press Space, pick a topic from a menu, then pick a tone. Each
reply is a fresh stateless generation: persona + topic and tone + a
memory string (the last five exchanges with that NPC, replies condensed)
+ ask count + the NPC's disposition toward the player. Unlock markers are
stripped from the displayed text and applied to per-topic visibility
flags; matching is a lenient ladder (exact name, the topic's last word,
or the name minus its last word, after normalization), because the model
reliably shortens names and strict matching left topics locked forever.
Asking the three clue topics (the armory, the secret passage, the inn
cellar) fills the journal (J key) and opens an Accuse option at the guard
captain, which generates the confrontation and closes the story.

## Limitations (flagged deficiencies)
- No shared canon block. Each persona fences its own facts (what it
  knows, does not know, must not speculate about), but shared world facts
  such as the village's founding and the theft's timing are restated per
  persona, and an NPC can still improvise a contradiction on that shared
  ground. A fixed canon block injected into every prompt is the known
  remedy; it is deliberately not in yet, and earns its place only if play
  shows actual drift.
- NPCs are static sprites; the village does not live between
  conversations. Deliberate for now: a feel pass, not a correctness gap.
- The accusation ending is implemented and its code path is verified,
  but as of this write-up it has not yet had a full play-test acceptance
  from the owner; the --demo tour asks one topic per NPC and never
  assembles the clue set, so the ending is the least-exercised path in
  the example.
- NPC memory is bounded and lossy by design: the last five exchanges,
  each reply condensed to 160 characters. An NPC eventually forgets the
  start of a long interrogation. That is the context budget made visible,
  not an oversight, but a reader should know it before copying the
  pattern.
- The lenient unlock ladder can be generous: a marker naming a topic's
  last word ("Kael") unlocks the full topic ("the stranger Kael"). The
  owner chose that trade after strict matching demonstrably failed
  against real model output; the ladder's uniqueness rule (a rule fires
  only when it matches exactly one locked topic) is what keeps it from
  becoming the old substring free-for-all.

## Feature: tone of address (Ebben's idea, Nov 23, 2026; landed Nov 24)
After picking a topic, the player also picks a tone: polite, neutral,
angry/demanding, flattering, joking. The tone goes into the prompt ("The
player asks about the lost sword, angrily, like a man picking a fight")
and the NPC answers the manner as much as the question, in character:
Tor gives as good as he gets, Lia is quietly hurt, Ren treats it as
disorderly conduct, Bora is thrilled by the drama and will repeat it,
Maren has buried ruder men than you and is amused. Tone feeds a per-NPC
disposition value that persists across conversations (and belongs in the
NPC memory work above): rudeness is remembered, standing affects
willingness to unlock topics, and the right approach differs per NPC,
intimidation might work on one and permanently close another. This is the
cheapest big upgrade to the dialogue's feel: one extra menu, one line in
the prompt, one integer per NPC, and every conversation in the game gets
a second axis.

## Improving on it in your own implementation
- Add the shared canon block this example flags as missing: one fixed
  string of world facts and dates, injected into every persona prompt, so
  improvisation cannot contradict established facts.
- Give memory more depth than a five-entry window: keep the last one or
  two exchanges verbatim and summarize older ones into a line each, or
  persist memory across sessions with the context state save/load shown
  in persistent_companion.
- Let the village live between conversations: NPCs on small schedules or
  idle wander change who is where and what they have seen, which gives
  the dialogue system new facts to work with.
- Gate the resolution on the player's own words, not only on topics
  asked: an accusation the player must phrase, judged against the clue
  set, fails more interestingly than a menu entry that appears when the
  set is complete.

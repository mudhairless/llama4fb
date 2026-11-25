# ask_the_codex — semantic search with embeddings

Status: built Nov 24, 2026 (llama4fb e1ef2717, verified against the
pushed code). File: examples/ask_the_codex.bas. Demo gate passed:
top-1 9/10, top-3 10/10 on the fixed question set; the out-of-scope
question topped at 0.405, below every in-scope winner (lowest 0.462).
Index build ~284 ms for 24 entries; queries ~7 ms. The single top-1
miss is instructive: "Which weapon was taken from the village?" ranked
The Armory (0.617) a hair above The Ashenbrand (0.606); the two entries
overlap heavily (the armory entry names the sword), which is the
near-duplicate behavior the limitations below predict.

## What it showcases
The embeddings half of the llama.cpp API, which none of the current
examples touch: running text through the model to get embedding vectors
(llama_get_embeddings and the embedding-mode context settings), then
ranking by cosine similarity. The use case is a game codex: a body of lore
entries (monsters, places, history, items) that the player queries in
plain language ("what is weak to fire", "who founded the village") and
gets the relevant entries back by meaning, including questions whose words
appear in no entry. It demonstrates that a small local model is useful for
retrieval even where its generation would be weak, and it needs no
generation at all.

## The embedding model (decided Nov 23, 2026)
This example uses a dedicated embedding model, not the generative Raon:
**Qwen3-Embedding-0.6B, Q8_0 quant (639 MB)**, from Qwen's official GGUF
repo (Qwen/Qwen3-Embedding-0.6B-GGUF, file Qwen3-Embedding-0.6B-Q8_0.gguf),
auto-downloaded on first run like the other examples' models. Apache 2.0,
1024-dimensional vectors, 32K context. It is the accuracy leader in its
size class by a wide margin (MTEB English v2 retrieval 61.8, against
~53.9 for bge-small-en-v1.5 and a 62.3 overall MTEB average for
nomic-embed-text-v1.5), and at 0.6B parameters a single forward pass per
text is still effectively instant on consumer CPUs. The runner-up was
bge-small-en-v1.5 (MIT, 33M parameters, 36 MB at Q8_0): dramatically
smaller and faster, but roughly 8 retrieval points weaker and capped at
512 tokens per text. For a teaching example whose acceptance test is
paraphrase retrieval, accuracy headroom wins; a reader who needs the tiny
footprint can swap models without changing the ranking code.

Two model-specific techniques are part of the showcase, not details:
- Pooling is LAST-token (this is a decoder model, not a BERT-style
  encoder; mean pooling is the wrong default here). The context is
  created with embeddings enabled and pooling_type LLAMA_POOLING_TYPE_LAST.
- Queries are instruction-formatted, documents are not. The model was
  trained for an "Instruct: <task>\nQuery: <text>" prefix on the query
  side only, worth roughly 1-5% retrieval quality. The example uses one
  fixed task instruction ("Given a game lore question, retrieve relevant
  codex entries that answer the question") and shows the asymmetry
  explicitly, because forgetting it is the most common way to make this
  model look bad.

## How it should work
At startup, embed a fixed corpus of a few dozen short lore entries and
keep the vectors in memory. The player types a question; the program
embeds it, ranks all entries by cosine similarity, and displays the top
three with their scores. Entries and queries are plain strings in a data
table so readers can swap in their own corpus.

## Limitations (deliberate)
- A second model download (~640 MB) alongside Raon; the repo's other
  examples need only the one file. Accepted for the accuracy, and the
  download is automatic.
- Brute-force cosine over an in-memory corpus: correct and clear at this
  scale, not a search engine.
- Embeddings are recomputed at every startup; startup cost grows with the
  corpus.
- Entries are short and pre-chunked by hand. Real documents need a
  chunking strategy, which this example avoids entirely.
- Ranking quality is whatever the small model gives; there is no keyword
  fallback, no score thresholding nuance beyond a simple cutoff, and
  near-duplicate entries will rank near-identically.
- One query at a time; embedding calls are not batched.

## Improving on it in your own implementation
- Precompute embeddings once and cache them to disk keyed by a hash of the
  entry text and model name; recompute only what changed.
- Chunk long documents deliberately (by paragraph or fixed token windows
  with overlap) and rank by best chunk, keeping a pointer back to the
  source document and position.
- Batch embedding calls (multiple sequences per decode) when indexing a
  corpus of any real size.
- Combine semantic ranking with a keyword pass (hybrid search) and a
  minimum-score threshold; pure cosine on a small model will sometimes
  return confident nonsense for out-of-scope questions, and "no good
  match" is an answer your UI should be able to give.
- Normalize vectors once at index time so query-time ranking is a dot
  product.

## Done when
- Builds and runs like the other examples; a set of test questions with
  known right answers (including paraphrases using none of the entry's
  keywords) returns the intended entry at rank 1, and an unrelated
  question scores visibly lower than any in-scope question.

## Later composition: the librarian front-end
Once this example and one tool-calling example (text_adventure_parser or
persistent_companion) exist, the codex gains an optional second mode: the
player no longer queries the index directly but asks Raon, in character
as a librarian, who decides to call search_codex(query) as a tool, reads
the returned entries, and answers from them, citing which entries it
used. That is the standard agent-plus-retrieval loop, assembled entirely
from this repo's own examples, and it is the strongest single
demonstration the repo can offer of the pieces composing. It is a
follow-up mode, not the base example: the base example must stay a pure,
inspectable retrieval demo with no generation in the path.

## The embedding bake-off (run Nov 24, 2026)
Both models ship in the example behind a --model flag (qwen3 default,
nomic the alternative), each handled its own correct way (nomic:
search_document:/search_query: prefixes, mean pooling, llama_encode,
768 dims; Qwen3: instruction-formatted queries, last-token pooling,
1024 dims). Same corpus, same fixed test set, Ebben's hardware:

- Top-1: 9/10 each. Top-3: 10/10 each. Both models missed the SAME
  question (The Armory over The Ashenbrand, by 0.010 and 0.006), which
  says that item tests corpus overlap, not model quality.
- Mean margin (expected entry vs best wrong entry): Qwen3 0.105,
  nomic 0.072. The pre-agreed tiebreak (top-1, then margin) keeps
  Qwen3 as the default.
- The decisive practical difference is score behavior on the
  out-of-scope question. Qwen3 put it at 0.405, clearly below its
  lowest in-scope winner (0.462), so a "no good match" threshold
  exists. Nomic put it at 0.608 against a lowest in-scope winner of
  0.614: nomic scores everything higher and compresses the range, so
  no usable threshold exists. For a codex that must sometimes answer
  "nothing matches," Qwen3's distribution is better behaved.
- Nomic's genuine wins: speed (~2.7 ms vs ~5.5 ms per query, index
  build ~4.3 ms vs ~7.9 ms per entry) and size (140 MB vs 610 MB). At
  this corpus scale both are effectively instant, so speed did not
  move the decision; size would matter in a shipped game download.

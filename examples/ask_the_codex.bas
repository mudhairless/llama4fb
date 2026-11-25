' ask_the_codex.bas - semantic search over a fixed codex of game lore.
' Showcases: embeddings-based retrieval with a dedicated embedding model
' (Qwen3-Embedding-0.6B), last-token pooling, query/document asymmetry via
' an instruct prefix, unit-normalized vectors ranked by dot product.
' Deliberate limitations: brute-force cosine over an in-memory corpus;
' embeddings recomputed at every startup; entries short and pre-chunked by
' hand; no keyword fallback; one query at a time.
' No generation; the Raon model is never loaded.
' Build: see build_ask_the_codex.sh / run_ask_the_codex.sh

#include once "llama.bi"
#include once "vbcompat.bi"

' C-compatible context params (FB enums are 64-bit; C enums are 32-bit, so use Long)
type MY_CTX_PARAMS
    n_ctx as ulong
    n_batch as ulong
    n_ubatch as ulong
    n_ubatch_auto_mib as ulong
    n_seq_max as ulong
    n_rs_seq as ulong
    n_outputs_max as ulong
    n_outputs_max_per_seq as ulong
    n_threads as long
    n_threads_batch as long
    ctx_type as long
    rope_scaling_type as long
    pooling_type as long
    attention_type as long
    flash_attn_type as long
    rope_freq_base as single
    rope_freq_scale as single
    yarn_ext_factor as single
    yarn_attn_factor as single
    yarn_beta_fast as single
    yarn_beta_slow as single
    yarn_orig_ctx as ulong
    defrag_thold as single
    cb_eval as any ptr
    cb_eval_user_data as any ptr
    type_k as long
    type_v as long
    moe_cache_size as uinteger
    abort_callback as any ptr
    abort_callback_data as any ptr
    embeddings as boolean
    offload_kqv as boolean
    no_perf as boolean
    op_offload as boolean
    swa_full as boolean
    kv_unified as boolean
    samplers as any ptr
    n_samplers as uinteger
    ctx_other as any ptr
end type

declare function my_ctx_default_params alias "llama_context_default_params"() as MY_CTX_PARAMS
declare function my_init_from_model alias "llama_init_from_model"(byval model as llama_model ptr, byval params as MY_CTX_PARAMS) as llama_context ptr

' ---------- qwen3 profile constants ----------
const QWEN_MODEL_NAME   = "Qwen3-Embedding-0.6B-Q8_0.gguf"
const QWEN_MODEL_URL    = "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf"
const QWEN_POOLING_TYPE = LLAMA_POOLING_TYPE_LAST
const QWEN_N_CTX        = 1024
const QWEN_DOC_PREFIX   = ""
const QWEN_QUERY_PREFIX = "Instruct: Given a game lore question, retrieve relevant codex entries that answer the question" & chr(10) & "Query: "
const QWEN_ADD_SPECIAL  = 0
const QWEN_APPEND_EOS   = 1

' ---------- nomic profile constants ----------
const NOMIC_MODEL_NAME   = "nomic-embed-text-v1.5.Q8_0.gguf"
const NOMIC_MODEL_URL    = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf"
const NOMIC_POOLING_TYPE = LLAMA_POOLING_TYPE_MEAN
const NOMIC_N_CTX        = 1024
const NOMIC_DOC_PREFIX   = "search_document: "
const NOMIC_QUERY_PREFIX = "search_query: "
const NOMIC_ADD_SPECIAL  = 1
const NOMIC_APPEND_EOS   = 0

' ---------- active profile (selected by --model) ----------
dim shared MODEL_NAME     as string
dim shared MODEL_URL      as string
dim shared MODEL_PATH     as string
dim shared POOLING_TYPE   as long
dim shared N_CTX          as long
dim shared N_BATCH        as long = 1024
dim shared N_GPU_LAYERS   as long = -1
dim shared DOC_PREFIX     as string
dim shared QUERY_PREFIX   as string
dim shared ADD_SPECIAL    as integer
dim shared APPEND_EOS     as integer
const N_ENTRIES = 24

' ---------- globals ----------
dim shared g_model as llama_model ptr
dim shared g_ctx   as llama_context ptr
dim shared g_vocab as const llama_vocab ptr
dim shared nEmbd   as long
dim shared corpusEmb as single ptr   ' N_ENTRIES * nEmbd floats, unit-normalized

dim shared titles(N_ENTRIES-1) as string
dim shared texts(N_ENTRIES-1)  as string

dim shared logging  as integer = 0
dim shared demoMode as integer = 0

' ---------- helpers ----------
sub log_msg(msg as string)
    if logging = 0 then return
    dim f as integer = freefile
    open "ask_the_codex.log" for append as #f
    print #f, msg
    close #f
end sub

sub ensure_model()
    if dir("models") = "" then mkdir "models"
    if dir(MODEL_PATH) = "" then
        print "Model not found. Downloading (~639 MB). This may take a while..."
        shell "curl -L -o " & MODEL_PATH & " " & MODEL_URL
        if dir(MODEL_PATH) = "" then
            print "Download failed."
            sleep
            end
        end if
    end if
end sub

sub init_corpus()
    titles(0)  = "The Ashenbrand"       : texts(0)  = "The Ashenbrand is a longsword with a grey, smoke-darkened blade, forged a generation ago and kept in Ren's armory as the village's ceremonial weapon. It was stolen on the night of the autumn fair."
    titles(1)  = "Ren the Smith"        : texts(1)  = "Ren inherited the forge from his mother and has run it for thirty years. He is broad, quiet, and keeps the armory keys on an iron ring at his belt."
    titles(2)  = "The Inn Cellar"       : texts(2)  = "Below the Willow Inn, a stone cellar stores salt pork, ale, and winter apples. A loose flagstone near the back wall hides the mouth of an old smugglers' passage."
    titles(3)  = "The Secret Passage"   : texts(3)  = "A narrow tunnel runs from the inn cellar to a hollow beneath the armory floor. Smugglers used it during the toll wars; almost nobody alive remembers it exists."
    titles(4)  = "Kael the Stranger"    : texts(4)  = "Kael arrived in Willowbrook a week before the fair, claiming to be a traveling knife-sharpener. He asked many questions about the armory and was seen near the forge after dark."
    titles(5)  = "Maren the Herbalist"  : texts(5)  = "Maren gathers herbs in the marsh and treats fevers, cuts, and bad dreams. She hears more village news at her workbench than anyone hears at the inn."
    titles(6)  = "Tor the Innkeeper"    : texts(6)  = "Tor has kept the Willow Inn for twenty years and remembers every guest who ever shorted him. He prides himself on hearing nothing through his own doors, and on hearing everything anyway."
    titles(7)  = "Finn the Fisher"      : texts(7)  = "Finn fishes the river at dawn and mends nets by the old mill. He notices boats, strangers, and anything that moves on the water before the village wakes."
    titles(8)  = "Bora the Baker"       : texts(8)  = "Bora's ovens fire before sunrise, and her cellar stairs are the busiest shortcut in the village. She trades bread for gossip and keeps both fresh."
    titles(9)  = "The Willow Inn"       : texts(9)  = "The Willow Inn is the village's only inn, a timber building with a sagging porch where travelers, merchants, and gossips gather every evening."
    titles(10) = "Willowbrook Village"  : texts(10) = "Willowbrook is a small farming village in a river bend, named for the old willow at its center. Its people grow barley, keep goats, and mind their own business."
    titles(11) = "The River Sarn"       : texts(11) = "The River Sarn loops around Willowbrook in a slow green bend. It floods in spring, carries timber rafts in summer, and freezes hard enough to walk on in deep winter."
    titles(12) = "Ember Imps"           : texts(12) = "Ember imps are fist-sized pests that nest in chimneys and feed on sparks. They fear cold iron and drown quickly; a bucket of well water ends most infestations."
    titles(13) = "Marsh Wraiths"        : texts(13) = "Marsh wraiths drift over the bog after rain, drawn to lantern light. They cannot cross running water, and the toll of a silver bell scatters them for a night."
    titles(14) = "The Toll Wars"        : texts(14) = "The toll wars were a generation of smuggling and ambushes over river tolls, ended when the baron burned the smugglers' barges. Old tunnels and grudges survive from those years."
    titles(15) = "Baron Halloran"       : texts(15) = "Baron Halloran rules the valley from a stone keep two days upriver. His tolls paid for the roads, and his temper paid for the toll wars."
    titles(16) = "Goats of Willowbrook" : texts(16) = "The village goats are famously stubborn and escape their pens weekly. The bell goat, Old Nettle, leads every breakout and has never once been caught in the act."
    titles(17) = "The Autumn Fair"     : texts(17) = "Every autumn the village holds a fair with wrestling, cider, and a bonfire on the green. The fair night is the one night nobody watches the armory."
    titles(18) = "Silver Bells"         : texts(18) = "Silver bells are hung at marsh-edge farms to warn of wraiths. Their metal must be true silver; plated bells ring just as sweet and protect not at all."
    titles(19) = "Cold Iron"            : texts(19) = "Cold iron, worked without flame by hammer and patience, is anathema to imps and other hearth spirits. Few smiths still know the slow method."
    titles(20) = "The Old Willow"       : texts(20) = "The old willow at the village center is older than the village itself. Sweethearts carve initials in its bark, and its roots are said to drink from a buried spring."
    titles(21) = "Dream Tea"            : texts(21) = "Maren's dream tea, brewed from marsh mint and valerian, brings deep sleep and vivid dreams. Taken too strong, it makes the sleeper speak aloud what they dream."
    titles(22) = "The Armory"           : texts(22) = "The village armory is a stone room behind the forge holding the Ashenbrand, militia spears, and a dozen hunting bows. Its door is oak bound in iron, locked with Ren's heaviest key."
    titles(23) = "Timber Rafts"         : texts(23) = "In summer, raftsmen pole great rafts of pine logs down the Sarn to the baron's sawmills. A raft crew can carry news faster than any rider on the muddy valley road."
end sub

' ---------- core embedding ----------
function embed_one(s as string) as single ptr
    ' tokenize
    dim maxT as long = 4096
    dim toks as llama_token ptr = allocate(maxT * sizeof(llama_token))
    dim nTok as long = llama_tokenize(g_vocab, strptr(s), len(s), toks, maxT, ADD_SPECIAL <> 0, false)
    if nTok < 0 then
        nTok = -nTok
        if nTok > maxT then
            deallocate(toks)
            toks = allocate(nTok * sizeof(llama_token))
            nTok = llama_tokenize(g_vocab, strptr(s), len(s), toks, nTok, ADD_SPECIAL <> 0, false)
        end if
    end if
    ' append EOS token only for last-token pooling profile
    if APPEND_EOS <> 0 then
        if nTok + 1 > maxT then
            deallocate(toks)
            toks = allocate((nTok + 1) * sizeof(llama_token))
        end if
        *(toks + nTok) = llama_vocab_eos(g_vocab)
        nTok += 1
    end if

    ' clear memory
    llama_memory_clear(llama_get_memory(g_ctx), true)

    ' build batch: whole text as seq 0, positions 0..n-1
    dim batch as llama_batch = llama_batch_init(nTok, 0, 1)
    dim tokArr as llama_token ptr = batch.token
    dim posArr as llama_pos ptr = batch.pos
    dim nseqArr as long ptr = batch.n_seq_id
    dim seqArr as llama_seq_id ptr ptr = batch.seq_id
    dim logArr as byte ptr = batch.logits
    for i as integer = 0 to nTok-1
        *(tokArr + i) = *(toks + i)
        if posArr <> 0 then *(posArr + i) = i
        *(nseqArr + i) = 1
        if seqArr <> 0 then *(*(seqArr + i)) = 0
        if POOLING_TYPE = LLAMA_POOLING_TYPE_MEAN then
            *(logArr + i) = 1
        else
            if i = nTok-1 then
                *(logArr + i) = 1
            else
                *(logArr + i) = 0
            end if
        end if
    next
    batch.n_tokens = nTok

    if llama_model_has_encoder(g_model) orelse (llama_get_memory(g_ctx) = 0) then
        llama_encode(g_ctx, batch)
    else
        llama_decode(g_ctx, batch)
    end if
    llama_batch_free(batch)

    ' copy nEmbd floats out immediately (pointer dies at next decode)
    dim vec as single ptr = allocate(nEmbd * sizeof(single))
    dim src as single ptr = llama_get_embeddings_seq(g_ctx, 0)
    for i as integer = 0 to nEmbd-1
        *(vec + i) = *(src + i)
    next
    deallocate(toks)

    ' normalize to unit length
    dim norm as double = 0
    for i as integer = 0 to nEmbd-1
        norm += cdbl(*(vec + i)) * cdbl(*(vec + i))
    next
    norm = sqr(norm)
    if norm > 0 then
        for i as integer = 0 to nEmbd-1
            *(vec + i) = *(vec + i) / norm
        next
    end if
    return vec
end function

' documents
function embed_document(text as string) as single ptr
    return embed_one(DOC_PREFIX & text)
end function

' queries
function embed_query(text as string) as single ptr
    return embed_one(QUERY_PREFIX & text)
end function

function dot(a as single ptr, b as single ptr) as single
    dim s as single = 0
    for i as integer = 0 to nEmbd-1
        s += *(a + i) * *(b + i)
    next
    return s
end function

' ---------- ranking ----------
sub top_indices(qv as single ptr, idxs() as integer, scores() as single)
    dim sc(N_ENTRIES-1) as single
    for i as integer = 0 to N_ENTRIES-1
        sc(i) = dot(qv, corpusEmb + i * nEmbd)
    next
    ' selection of top 3
    for r as integer = 0 to 2
        dim best as integer = -1
        dim bestSc as single = -1e9
        for i as integer = 0 to N_ENTRIES-1
            dim used as integer = 0
            for k as integer = 0 to r-1
                if idxs(k) = i then used = 1
            next
            if used = 0 andalso sc(i) > bestSc then
                bestSc = sc(i)
                best = i
            end if
        next
        idxs(r) = best
        scores(r) = bestSc
    next
end sub

sub print_result(q as string, idxs() as integer, scores() as single, qMs as double)
    for r as integer = 0 to 2
        print str(r+1) & ". [" & format(scores(r), "0.000") & "] " & titles(idxs(r))
        print "    " & texts(idxs(r))
    next
    print "Query embedded in " & format(qMs, "0.0") & " ms"
end sub

sub answer_question(q as string)
    dim t0 as double = timer
    dim qv as single ptr = embed_query(q)
    dim qMs as double = (timer - t0) * 1000.0
    dim idxs(0 to 2) as integer
    dim scores(0 to 2) as single
    top_indices(qv, idxs(), scores())
    print_result(q, idxs(), scores(), qMs)
    if logging = 1 then
        log_msg("Q=" & q)
        for r as integer = 0 to 2
            log_msg(str(r+1) & " " & titles(idxs(r)) & " " & format(scores(r), "0.000"))
        next
        log_msg("q_ms=" & format(qMs, "0.0"))
    end if
    deallocate(qv)
end sub

' ---------- main ----------
' parse flags first
dim modelChoice as string = "qwen3"
for ai as integer = 1 to 10
    if command(ai) = "--log" or command(ai) = "-log" then logging = 1
    if command(ai) = "--demo" then demoMode = 1
    if command(ai) = "--model" then
        modelChoice = command(ai + 1)
        if modelChoice <> "qwen3" and modelChoice <> "nomic" then
            print "Usage: ask_the_codex [--demo] [--log] [--model {qwen3|nomic}]"
            end 1
        end if
    end if
next

' select profile
if modelChoice = "qwen3" then
    MODEL_NAME = QWEN_MODEL_NAME
    MODEL_URL = QWEN_MODEL_URL
    POOLING_TYPE = QWEN_POOLING_TYPE
    N_CTX = QWEN_N_CTX
    DOC_PREFIX = QWEN_DOC_PREFIX
    QUERY_PREFIX = QWEN_QUERY_PREFIX
    ADD_SPECIAL = QWEN_ADD_SPECIAL
    APPEND_EOS = QWEN_APPEND_EOS
else
    MODEL_NAME = NOMIC_MODEL_NAME
    MODEL_URL = NOMIC_MODEL_URL
    POOLING_TYPE = NOMIC_POOLING_TYPE
    N_CTX = NOMIC_N_CTX
    DOC_PREFIX = NOMIC_DOC_PREFIX
    QUERY_PREFIX = NOMIC_QUERY_PREFIX
    ADD_SPECIAL = NOMIC_ADD_SPECIAL
    APPEND_EOS = NOMIC_APPEND_EOS
end if
MODEL_PATH = "models/" & MODEL_NAME

ensure_model()

dim mparams as llama_model_params = llama_model_default_params()
mparams.n_gpu_layers = N_GPU_LAYERS
g_model = llama_model_load_from_file(strptr(MODEL_PATH), mparams)
if g_model = 0 then
    print "Failed to load model"
    sleep
    end
end if

dim cparams as MY_CTX_PARAMS = my_ctx_default_params()
cparams.n_ctx = N_CTX
cparams.n_batch = N_BATCH
cparams.n_ubatch = N_BATCH
cparams.n_seq_max = 1
cparams.pooling_type = POOLING_TYPE
cparams.embeddings = true
g_ctx = my_init_from_model(g_model, cparams)
g_vocab = llama_model_get_vocab(g_model)

nEmbd = llama_model_n_embd(g_model)

if logging = 1 then
    dim f0 as integer = freefile
    open "ask_the_codex.log" for output as #f0
    close #f0
end if

init_corpus()

' build index
dim t0 as double = timer
corpusEmb = allocate(N_ENTRIES * nEmbd * sizeof(single))
for i as integer = 0 to N_ENTRIES-1
    dim v as single ptr = embed_document(texts(i))
    for j as integer = 0 to nEmbd-1
        *(corpusEmb + (i * nEmbd + j)) = *(v + j)
    next
    deallocate(v)
next
dim buildMs as double = (timer - t0) * 1000.0

print "Model: " & MODEL_NAME
print "n_embd: " & str(nEmbd)
print "Entries: " & str(N_ENTRIES)
print "Index built in " & format(buildMs, "0.0") & " ms"
log_msg("Model=" & MODEL_NAME & " n_embd=" & str(nEmbd) & " entries=" & str(N_ENTRIES) & " build_ms=" & format(buildMs, "0.0"))

if demoMode = 1 then
    dim testQ(0 to 9) as string
    dim expected(0 to 9) as string
    testQ(0) = "Which weapon was taken from the village?"       : expected(0) = "The Ashenbrand"
    testQ(1) = "Which blade disappeared during the festival?"   : expected(1) = "The Ashenbrand"
    testQ(2) = "Who rules the valley?"                          : expected(2) = "Baron Halloran"
    testQ(3) = "How do you deal with pests in the fireplace?"   : expected(3) = "Ember Imps"
    testQ(4) = "Where does the tunnel from the inn lead?"       : expected(4) = "The Secret Passage"
    testQ(5) = "What happens if you drink the herbalist's tea too strong?" : expected(5) = "Dream Tea"
    testQ(6) = "Why won't a wraith follow you across a stream?" : expected(6) = "Marsh Wraiths"
    testQ(7) = "Who keeps escaping from the pen?"               : expected(7) = "Goats of Willowbrook"
    testQ(8) = "What is the fastest way to send news down the valley?" : expected(8) = "Timber Rafts"
    testQ(9) = "What does the smith wear on his belt?"          : expected(9) = "Ren the Smith"

    dim top1count as integer = 0
    dim top3count as integer = 0
    dim minInScopeTop1 as single = 2.0
    dim marginSum as double = 0
    for ti as integer = 0 to 9
        print "Q: " & testQ(ti)
        print "Expected: " & expected(ti)
        dim t0q as double = timer
        dim qv as single ptr = embed_query(testQ(ti))
        dim qMs as double = (timer - t0q) * 1000.0
        dim idxs(0 to 2) as integer
        dim scores(0 to 2) as single
        top_indices(qv, idxs(), scores())
        for r as integer = 0 to 2
            print str(r+1) & ". [" & format(scores(r), "0.000") & "] " & titles(idxs(r))
            print "    " & texts(idxs(r))
        next
        ' all scores for margin and rank
        dim allsc(N_ENTRIES-1) as single
        for i as integer = 0 to N_ENTRIES-1
            allsc(i) = dot(qv, corpusEmb + i * nEmbd)
        next
        ' rank of expected
        dim expRank as integer = 0
        for r as integer = 0 to 2
            if titles(idxs(r)) = expected(ti) then expRank = r + 1
        next
        if expRank = 0 then
            for i as integer = 0 to N_ENTRIES-1
                if titles(i) = expected(ti) then
                    dim rk as integer = 1
                    for j as integer = 0 to N_ENTRIES-1
                        if allsc(j) > allsc(i) then rk += 1
                    next
                    expRank = rk
                end if
            next
        end if
        print "rank of expected: " & str(expRank)
        ' margin
        dim expIdx as integer = -1
        for i as integer = 0 to N_ENTRIES-1
            if titles(i) = expected(ti) then expIdx = i
        next
        dim expScore as single = allsc(expIdx)
        dim bestOther as single = -1e9
        for i as integer = 0 to N_ENTRIES-1
            if i <> expIdx andalso allsc(i) > bestOther then bestOther = allsc(i)
        next
        dim margin as single = expScore - bestOther
        marginSum += margin
        print "margin=" & format(margin, "0.000")
        if expRank = 1 then top1count += 1
        if expRank >= 1 and expRank <= 3 then top3count += 1
        if scores(0) < minInScopeTop1 then minInScopeTop1 = scores(0)
        if logging = 1 then
            log_msg("Q=" & testQ(ti))
            log_msg("Expected=" & expected(ti))
            for r as integer = 0 to 2
                log_msg(str(r+1) & " " & titles(idxs(r)) & " " & format(scores(r), "0.000"))
            next
            log_msg("rank=" & str(expRank) & " q_ms=" & format(qMs, "0.0"))
            log_msg("margin=" & format(margin, "0.000"))
        end if
        deallocate(qv)
        if demoMode = 1 then sleep 2400, 1
    next

    ' out of scope
    dim oosQ as string = "How do I bake sourdough bread?"
    print "Q: " & oosQ
    print "Expected: (none)"
    dim qv2 as single ptr = embed_query(oosQ)
    dim oosIdxs(0 to 2) as integer
    dim oosScores(0 to 2) as single
    top_indices(qv2, oosIdxs(), oosScores())
    for r as integer = 0 to 2
        print str(r+1) & ". [" & format(oosScores(r), "0.000") & "] " & titles(oosIdxs(r))
        print "    " & texts(oosIdxs(r))
    next
    dim oosTop as single = oosScores(0)
    print "out-of-scope top score: " & format(oosTop, "0.000")
    if logging = 1 then
        log_msg("Q=" & oosQ)
        log_msg("Expected=(none)")
        for r as integer = 0 to 2
            log_msg(str(r+1) & " " & titles(oosIdxs(r)) & " " & format(oosScores(r), "0.000"))
        next
        log_msg("oos_top=" & format(oosTop, "0.000"))
    end if
    deallocate(qv2)
    sleep 3200, 1

    dim meanMargin as double = marginSum / 10.0
    print "top1=" & str(top1count) & "/10 top3=" & str(top3count) & "/10"
    print "mean_margin=" & format(meanMargin, "0.000")
    print "out-of-scope: " & format(oosTop, "0.000") & " vs lowest in-scope top-1: " & format(minInScopeTop1, "0.000")
    if logging = 1 then
        log_msg("top1=" & str(top1count) & "/10 top3=" & str(top3count) & "/10")
        log_msg("mean_margin=" & format(meanMargin, "0.000"))
        log_msg("out-of-scope: " & format(oosTop, "0.000") & " vs lowest in-scope top-1: " & format(minInScopeTop1, "0.000"))
    end if

    ' hold the final scores long enough to read them in the demo video
    if demoMode = 1 then sleep 7200, 1

    llama_sampler_free(0)
    llama_free(g_ctx)
    llama_free_model(g_model)
    end
end if

' interactive loop
do
    print
    print "Ask the codex (empty line quits): ";
    dim q as string
    line input q
    q = trim(q)
    if q = "" then exit do
    answer_question(q)
loop

llama_free(g_ctx)
llama_free_model(g_model)

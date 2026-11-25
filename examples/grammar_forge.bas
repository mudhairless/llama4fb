' grammar_forge.bas - GBNF grammar-constrained weapon generation demo
' The model invents fantasy weapons as JSON; a GBNF grammar constrains
' generation token by token so the output always parses. The parse-failure
' counter on screen is the proof and must stay at zero.
'
' Showcases: llama_sampler_init_grammar (GBNF), sampler-chain ordering
' (grammar first), and the decoupling of correctness (grammar) from variety
' (temperature 0.8). Deliberate limitations: the grammar constrains shape,
' not sense (a damage of 99 on a common stick is possible; grammar is not
' validation); one fixed flat schema. Extraction uses fbejson (vendored
' at vendor/fbejson); a generation fails on invalid JSON, a missing
' field, or a wrong-typed field.
'
' Build: see build_grammar_forge.sh / run_grammar_forge.sh

#include once "fbgfx.bi"
#include once "llama.bi"
#include once "json.bi"

const SC_W = 800
const SC_H = 600

' ---------- item data ----------
type Item
    name as string
    damage as integer
    speed as integer
    element as string
    flavor as string
end type

declare sub draw_screen()

' ---------- globals ----------
dim shared logging as integer = 0
dim shared demoMode as integer = 0
dim shared noGrammar as integer = 0
dim shared demoStarted as integer = 0

dim shared slotIndex as integer = 0
dim shared rarityIndex as integer = 0
dim shared slotNames(0 to 4) as string
dim shared rarityNames(0 to 3) as string
dim shared rarityGuidance(0 to 3) as string

sub init_data()
    slotNames(0) = "Sword": slotNames(1) = "Axe": slotNames(2) = "Bow": slotNames(3) = "Staff": slotNames(4) = "Dagger"
    rarityNames(0) = "Common": rarityNames(1) = "Rare": rarityNames(2) = "Epic": rarityNames(3) = "Legendary"
    rarityGuidance(0) = "It is ordinary, low damage."
    rarityGuidance(1) = "It is fine work, moderate damage."
    rarityGuidance(2) = "It is a mighty weapon, high damage."
    rarityGuidance(3) = "It is a weapon of legend, very high damage."
end sub

dim shared inventory(0 to 999) as Item
dim shared nInventory as integer = 0
dim shared forgedCount as integer = 0
dim shared parseFailures as integer = 0
dim shared latest as Item
dim shared hasLatest as integer = 0
dim shared forging as integer = 0

' ---------- LLM globals ----------
dim shared g_model as llama_model ptr
dim shared g_ctx as llama_context ptr
dim shared g_smp as llama_sampler ptr
dim shared g_vocab as const llama_vocab ptr
dim shared g_grammar_str as string

' ---------- logging ----------
sub log_msg(msg as string)
    if logging = 0 then return
    dim f as integer = freefile
    open "grammar_forge.log" for append as #f
    print #f, msg
    close #f
end sub

' ---------- model ----------
sub ensure_model()
    dim modelPath as string = "models/raon-1-nano-Q4_K_M.gguf"
    if dir("models") = "" then mkdir "models"
    if dir(modelPath) = "" then
        print "Model not found. Downloading (~2GB). This may take a while..."
        shell "curl -L -o " & modelPath & " https://huggingface.co/KRAFTON/Raon-1-nano-gguf/resolve/main/raon-1-nano-Q4_K_M.gguf"
        if dir(modelPath) = "" then
            print "Download failed."
            sleep
            end
        end if
    end if
end sub

sub llm_init()
    ensure_model()
    dim modelPath as string = "models/raon-1-nano-Q4_K_M.gguf"
    dim mparams as llama_model_params = llama_model_default_params()
    mparams.n_gpu_layers = -1
    g_model = llama_model_load_from_file(strptr(modelPath), mparams)
    if g_model = 0 then
        print "Failed to load model"
        sleep
        end
    end if
    dim cparams as llama_context_params = llama_context_default_params()
    cparams.n_ctx = 2048
    cparams.n_batch = 512
    g_ctx = llama_init_from_model(g_model, cparams)
    g_vocab = llama_model_get_vocab(g_model)

    ' build grammar string
    dim q as string = chr(34)
    dim bs as string = chr(92)
    dim g as string
    g = "root    ::= " & q & "{" & q & " ws " & q & bs & q & "name" & bs & q & q & " ws " & q & ":" & q & " ws string " & q & "," & q & " ws " & q & bs & q & "damage" & bs & q & q & " ws " & q & ":" & q & " ws damage " & q & "," & q & " ws " & q & bs & q & "speed" & bs & q & q & " ws " & q & ":" & q & " ws speed " & q & "," & q & " ws " & q & bs & q & "element" & bs & q & q & " ws " & q & ":" & q & " ws element " & q & "," & q & " ws " & q & bs & q & "flavor" & bs & q & q & " ws " & q & ":" & q & " ws string ws " & q & "}" & q & chr(10)
    g = g & "damage  ::= [1-9] [0-9]?" & chr(10)
    g = g & "speed   ::= " & q & "10" & q & " | [1-9]" & chr(10)
    g = g & "element ::= " & q & bs & q & "fire" & bs & q & q & " | " & q & bs & q & "ice" & bs & q & q & " | " & q & bs & q & "storm" & bs & q & q & " | " & q & bs & q & "earth" & bs & q & q & " | " & q & bs & q & "shadow" & bs & q & q & chr(10)
    g = g & "string  ::= " & q & bs & q & q & " char* " & q & bs & q & q & chr(10)
    g = g & "char    ::= [^" & q & bs & bs & "]" & " | " & q & bs & bs & q & " (" & "[" & q & bs & bs & "/" & "bfnrt]" & " | " & q & "u" & q & " [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])" & chr(10)
    g = g & "ws      ::= [ ]*" & chr(10)
    g_grammar_str = g


    dim sparams as llama_sampler_chain_params = llama_sampler_chain_default_params()
    g_smp = llama_sampler_chain_init(sparams)
    if noGrammar = 0 then
        dim gs as llama_sampler ptr = llama_sampler_init_grammar(g_vocab, strptr(g_grammar_str), strptr("root"))
        if gs <> 0 then llama_sampler_chain_add(g_smp, gs)
    end if
    llama_sampler_chain_add(g_smp, llama_sampler_init_penalties(llama_vocab_n_tokens(g_vocab), 64, 1.0, 0.0, 1.5))
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_k(40))
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_p(0.9, 1))
    llama_sampler_chain_add(g_smp, llama_sampler_init_temp(0.8))
    llama_sampler_chain_add(g_smp, llama_sampler_init_dist(LLAMA_DEFAULT_SEED))
end sub

' ---------- generation ----------
function gen_raw(sysPrompt as string, usrPrompt as string) as string
    llama_memory_clear(llama_get_memory(g_ctx), true)
    llama_sampler_reset(g_smp)

    dim prompt as string
    prompt = "<|im_start|>system" & chr(10) & sysPrompt & chr(10) & "<|im_end|>" & chr(10)
    prompt = prompt & "<|im_start|>user" & chr(10) & usrPrompt & chr(10) & "<|im_end|>" & chr(10)
    prompt = prompt & "<|im_start|>assistant" & chr(10) & "<think></think>" & chr(10) & chr(10)

    dim maxT as long = 4096
    dim toks as llama_token ptr = allocate(maxT * sizeof(llama_token))
    dim nTok as long = llama_tokenize(g_vocab, strptr(prompt), len(prompt), toks, maxT, true, true)
    if nTok < 0 then
        nTok = -nTok
        if nTok > maxT then
            deallocate(toks)
            toks = allocate(nTok * sizeof(llama_token))
            nTok = llama_tokenize(g_vocab, strptr(prompt), len(prompt), toks, nTok, true, true)
        end if
    end if

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
        if seqArr <> 0 then
            *(*(seqArr + i)) = 0
        end if
        *(logArr + i) = 0
    next
    *(logArr + (nTok-1)) = 1
    batch.n_tokens = nTok

    llama_decode(g_ctx, batch)
    llama_batch_free(batch)

    dim result as string = ""
    dim nCur as long = nTok
    dim nGen as integer = 0
    while nGen < 220
        dim tok as llama_token = llama_sampler_sample(g_smp, g_ctx, -1)
        if llama_token_is_eog(g_vocab, tok) then exit while
        dim pbuf as zstring ptr = allocate(512)
        dim plen as long = llama_token_to_piece(g_vocab, tok, pbuf, 512, 0, true)
        if plen > 0 then result += left(*pbuf, plen)
        deallocate(pbuf)

        dim b2 as llama_batch = llama_batch_init(1, 0, 1)
        dim t2 as llama_token ptr = b2.token
        dim p2 as llama_pos ptr = b2.pos
        dim n2 as long ptr = b2.n_seq_id
        dim s2 as llama_seq_id ptr ptr = b2.seq_id
        dim l2 as byte ptr = b2.logits
        *t2 = tok
        if p2 <> 0 then *p2 = nCur
        *n2 = 1
        if s2 <> 0 then **s2 = 0
        *l2 = 1
        b2.n_tokens = 1
        llama_decode(g_ctx, b2)
        llama_batch_free(b2)
        nCur += 1
        nGen += 1
    wend

    deallocate(toks)

    dim filtered as string = ""
    for i as integer = 1 to len(result)
        dim c as integer = asc(mid(result, i, 1))
        if (c >= 32 and c <= 126) or c = 10 or c = 13 then
            filtered += mid(result, i, 1)
        elseif c = 9 then
            filtered += " "
        end if
    next
    return filtered
end function

 ' ---------- extraction ----------
function try_parse(raw as string, byref it as Item, byref reason as string) as integer
    reason = ""
    dim jobj as json.JSONobject
    dim p as json.JSONobject ptr = jobj.loadString(raw)
    if p = 0 then
        reason = "invalid json"
        return 0
    end if

    ' name (string)
    if jobj.hasChild("name") = 0 then
        reason = "missing name"
        return 0
    end if
    dim nv as json.JSONvalue ptr = jobj.child("name")
    if nv = 0 then
        reason = "missing name"
        return 0
    end if
    if nv->valueType() <> json.jvalue_type.jstring then
        reason = "wrong type name"
        return 0
    end if
    it.name = nv->getString()

    ' damage (number)
    if jobj.hasChild("damage") = 0 then
        reason = "missing damage"
        return 0
    end if
    dim dv as json.JSONvalue ptr = jobj.child("damage")
    if dv = 0 then
        reason = "missing damage"
        return 0
    end if
    if dv->valueType() <> json.jvalue_type.jnumber then
        reason = "wrong type damage"
        return 0
    end if
    it.damage = CInt(dv->getNumber())

    ' speed (number)
    if jobj.hasChild("speed") = 0 then
        reason = "missing speed"
        return 0
    end if
    dim sv as json.JSONvalue ptr = jobj.child("speed")
    if sv = 0 then
        reason = "missing speed"
        return 0
    end if
    if sv->valueType() <> json.jvalue_type.jnumber then
        reason = "wrong type speed"
        return 0
    end if
    it.speed = CInt(sv->getNumber())

    ' element (string)
    if jobj.hasChild("element") = 0 then
        reason = "missing element"
        return 0
    end if
    dim ev as json.JSONvalue ptr = jobj.child("element")
    if ev = 0 then
        reason = "missing element"
        return 0
    end if
    if ev->valueType() <> json.jvalue_type.jstring then
        reason = "wrong type element"
        return 0
    end if
    it.element = ev->getString()

    ' flavor (string)
    if jobj.hasChild("flavor") = 0 then
        reason = "missing flavor"
        return 0
    end if
    dim fv as json.JSONvalue ptr = jobj.child("flavor")
    if fv = 0 then
        reason = "missing flavor"
        return 0
    end if
    if fv->valueType() <> json.jvalue_type.jstring then
        reason = "wrong type flavor"
        return 0
    end if
    it.flavor = fv->getString()

    return 1
end function

' ---------- forge ----------
sub do_forge()
    forging = 1
    draw_screen()
    dim usrPrompt as string
    usrPrompt = "Invent a " & rarityNames(rarityIndex) & " " & slotNames(slotIndex) & " for a fantasy adventurer. Reply with the JSON object only."
    usrPrompt = usrPrompt & " " & rarityGuidance(rarityIndex)
    dim sysPrompt as string
    sysPrompt = "You are the spirit of the Grammar Forge, an ancient smithy that dreams up weapons. You answer with a single JSON object and nothing else."

    dim raw as string = gen_raw(sysPrompt, usrPrompt)
    forging = 0

    dim it as Item
    dim reason as string = ""
    dim ok as integer = try_parse(raw, it, reason)
    if ok then
        inventory(nInventory) = it
        nInventory += 1
        latest = it
        hasLatest = 1
    else
        parseFailures += 1
    end if
    forgedCount += 1

    if logging = 1 then
        log_msg("RAW=" & raw)
        if ok then
            log_msg("PARSED name=" & it.name & " damage=" & str(it.damage) & " speed=" & str(it.speed) & " element=" & it.element & " flavor=" & it.flavor)
        else
            log_msg("PARSE FAILURE " & reason)
        end if
    end if
    draw_screen()
end sub

' ---------- rendering ----------
sub draw_screen()
    screenlock
    cls
    line (0, 0)-(SC_W-1, SC_H-1), rgb(20, 18, 28), bf
    draw string (20, 10), "The Grammar Forge"
    draw string (20, 32), "Slot: " & slotNames(slotIndex) & "   Rarity: " & rarityNames(rarityIndex)

    dim y as integer = 60
    if hasLatest = 1 then
        draw string (20, y), "Latest:"
        y += 16
        line (20, y)-(420, y + 110), rgb(50, 40, 70), bf
        line (20, y)-(420, y + 110), rgb(180, 180, 200), b
        draw string (30, y + 8), latest.name
        draw string (30, y + 26), "DMG " & str(latest.damage) & "  SPD " & str(latest.speed) & "  " & latest.element
        dim fl as string = chr(34) & latest.flavor & chr(34)
        dim fpos as integer = 1
        dim fline as integer = 0
        while fpos <= len(fl) and fline < 5
            dim chunk as string = mid(fl, fpos, 46)
            draw string (30, y + 44 + fline * 14), chunk
            fpos += 46
            fline += 1
        wend
        y += 120
    else
        draw string (20, y), "No item forged yet."
        y += 24
    end if
    if forging = 1 then
        draw string (20, y), "Forging..."
        y += 20
    end if
    draw string (20, y), "Inventory:"
    y += 16
    dim startIdx as integer = 0
    if nInventory > 12 then startIdx = nInventory - 12
    dim shown as integer = 0
    for i as integer = startIdx to nInventory - 1
        if shown >= 12 then exit for
        dim invname as string = inventory(i).name
        if len(invname) > 44 then invname = left(invname, 41) & "..."
        draw string (20, y), invname & "  (DMG " & str(inventory(i).damage) & ")"
        y += 14
        shown += 1
    next
    draw string (20, SC_H - 50), "Forged: " & str(forgedCount) & "  Parse failures: " & str(parseFailures)
    draw string (20, SC_H - 30), "Keys 1-5 slot, R rarity, Space forge, Q quit"
    screenunlock
end sub

' ---------- main ----------
randomize timer

' parse flags
for ai as integer = 1 to 10
    if command(ai) = "--log" or command(ai) = "-log" then logging = 1
    if command(ai) = "--demo" then demoMode = 1
    if command(ai) = "--no-grammar" then noGrammar = 1
next

init_data()

screenres SC_W, SC_H, 32
windowtitle "The Grammar Forge"

screenlock
cls
draw string (40, 40), "The Grammar Forge"
draw string (40, 70), "Loading model..."
screenunlock

llm_init()

if logging = 1 then
    dim f0 as integer = freefile
    open "grammar_forge.log" for output as #f0
    close #f0
end if

if demoMode = 1 then
    for di as integer = 0 to 99
        slotIndex = di mod 5
        rarityIndex = di mod 4
        do_forge()
        draw_screen()
        sleep 16
    next
    dim summary as string
    summary = "generated=100 parsed=" & str(nInventory) & " failures=" & str(parseFailures)
    print summary
    dim cf as integer = freefile
    open cons for output as #cf
    print #cf, summary
    close #cf
    log_msg(summary)
    if logging = 1 then
        ' summary line at exit already logged
    end if
    llama_sampler_free(g_smp)
    llama_free(g_ctx)
    llama_free_model(g_model)
    end
end if

dim lastSpace as integer = 0
dim lastR as integer = 0
dim lastNum(1 to 5) as integer
for ni as integer = 1 to 5
    lastNum(ni) = 0
next

do
    for ni as integer = 1 to 5
        dim sc as integer
        select case ni
            case 1: sc = FB.SC_1
            case 2: sc = FB.SC_2
            case 3: sc = FB.SC_3
            case 4: sc = FB.SC_4
            case 5: sc = FB.SC_5
        end select
        dim nowk as integer = multikey(sc)
        if nowk and not lastNum(ni) then
            slotIndex = ni - 1
        end if
        lastNum(ni) = nowk
    next

    dim spaceNow as integer = multikey(FB.SC_SPACE)
    if spaceNow and not lastSpace then
        do_forge()
    end if
    lastSpace = spaceNow

    dim rNow as integer = multikey(FB.SC_R)
    if rNow and not lastR then
        rarityIndex = (rarityIndex + 1) mod 4
    end if
    lastR = rNow

    draw_screen()
    sleep 16
loop until multikey(FB.SC_Q)

if logging = 1 then
    log_msg("forged=" & str(forgedCount) & " failures=" & str(parseFailures))
end if

llama_sampler_free(g_smp)
llama_free(g_ctx)
llama_free_model(g_model)

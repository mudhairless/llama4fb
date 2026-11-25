' text_adventure.bas - "Locked In at Fumble Labs" text adventure
' The model NEVER runs the game. It only translates each typed sentence
' into one structured tool call from a fixed tool set; a deterministic
' engine validates the call against world state and narrates the result
' with hand-written text.
'
' Showcases: native tool calling through three small chat-template
' helpers (build_prompt, parse_tool_call, append_tool_result). The helpers
' encode THIS model's template as shipped in the GGUF (raon-1-nano).
'
' Deliberate limitations: helpers encode this model's template as shipped
' in the GGUF; one intent per input line; no pronoun resolution across
' turns; a grammar-constrained intent format is the documented fallback
' for models without tool training.
'
' Build: see build_text_adventure.sh / run_text_adventure.sh

#include once "fbgfx.bi"
#include once "llama.bi"
#include once "json.bi"

const SC_W = 800
const SC_H = 600

' ---------- chat message struct ----------
type ChatMsg
    role as string
    content as string
    hasTool as integer
    toolName as string
    toolArgs as string
end type

' ---------- parsed tool call ----------
type ParsedCall
    ok as integer
    name as string
    argKeys(0 to 9) as string
    argVals(0 to 9) as string
    nArgs as integer
    norm as string
end type

' ---------- game state ----------
dim shared curRoom as integer = 0  ' 0 Lobby,1 Lab,2 Office,3 Break
dim shared inventory(0 to 31) as string
dim shared nInv as integer = 0
dim shared won as integer = 0
dim shared safeUnlocked as integer = 0
dim shared fridgeOpened as integer = 0
dim shared redHerringTaken as integer = 0
dim shared chickenTaken as integer = 0
dim shared handTaken as integer = 0
dim shared geraldTurn as integer = 0
dim shared microwaveOpened as integer = 0
dim shared microwaveHasHerring as integer = 0
dim shared microwaveRuined as integer = 0

' object location: room index or -1 inventory, -2 gone/taken
' we track taken via flags; presence checks use room + flags

' ---------- LLM globals ----------
dim shared g_model as llama_model ptr
dim shared g_ctx as llama_context ptr
dim shared g_smp as llama_sampler ptr
dim shared g_vocab as const llama_vocab ptr

' ---------- UI / log ----------
dim shared logLines(0 to 63) as string
dim shared nLog as integer = 0
dim shared inputBuf as string
dim shared logging as integer = 0
dim shared demoMode as integer = 0

' ---------- tool defs ----------
dim shared toolJson(0 to 10) as string
dim shared nTools as integer = 0

' ---------- helper: log to file ----------
sub file_log(msg as string)
    if logging = 0 then return
    dim f as integer = freefile
    open "text_adventure.log" for append as #f
    print #f, msg
    close #f
end sub

sub ui_log_append_one(msg as string)
    dim rest as string = msg
    do while len(rest) > 96
        dim chunk as string = left(rest, 96)
        dim sp as integer = instrrev(chunk, " ")
        if sp < 1 then sp = 96
        dim part as string = left(rest, sp)
        if nLog < 64 then
            logLines(nLog) = part
            nLog += 1
        else
            for i as integer = 0 to 62
                logLines(i) = logLines(i+1)
            next
            logLines(63) = part
        end if
        rest = trim(mid(rest, sp + 1))
    loop
    if rest <> "" then
        if nLog < 64 then
            logLines(nLog) = rest
            nLog += 1
        else
            for i as integer = 0 to 62
                logLines(i) = logLines(i+1)
            next
            logLines(63) = rest
        end if
    end if
end sub

sub ui_log(msg as string)
    dim cur as string = ""
    for i as integer = 1 to len(msg)
        dim ch as string = mid(msg, i, 1)
        if ch = chr(10) then
            ui_log_append_one(cur)
            cur = ""
        else
            cur += ch
        end if
    next
    if cur <> "" then ui_log_append_one(cur)
end sub

' ---------- helper 1: build_prompt ----------
function build_prompt(msgs() as ChatMsg, nMsg as integer, tools() as string, nT as integer) as string
    dim s as string = ""
    if nT > 0 then
        s += "<|fim_prefix|>system" & chr(10)
        s += "# Tools" & chr(10) & chr(10) & "You have access to the following functions:" & chr(10) & chr(10) & "<tools>"
        for ti as integer = 0 to nT-1
            s += chr(10) & tools(ti)
        next
        s += chr(10) & "</tools>"
        s += chr(10) & chr(10) & "If you choose to call a function ONLY reply in the following format with NO suffix:" & chr(10) & chr(10)
        s += "<tool_call>" & chr(10) & "<function=example_function_name>" & chr(10) & "<parameter=example_parameter_1>" & chr(10) & "value_1" & chr(10) & "</parameter>" & chr(10) & "<parameter=example_parameter_2>" & chr(10) & "This is the value for the second parameter" & chr(10) & "that can span" & chr(10) & "multiple lines" & chr(10) & "</parameter>" & chr(10) & "</function>" & chr(10) & "</tool_call>" & chr(10) & chr(10)
        s += "<IMPORTANT>" & chr(10) & "Reminder:" & chr(10)
        s += "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags" & chr(10)
        s += "- Required parameters MUST be specified" & chr(10)
        s += "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after" & chr(10)
        s += "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls" & chr(10)
        s += "</IMPORTANT>"
        if nMsg > 0 andalso msgs(0).role = "system" then
            dim content as string = trim(msgs(0).content)
            if content <> "" then
                s += chr(10) & chr(10) & content
            end if
        end if
        s += "<|fim_prefix|>" & chr(10)
    elseif nMsg > 0 andalso msgs(0).role = "system" then
        dim content as string = trim(msgs(0).content)
        s += "<|fim_prefix|>system" & chr(10) & content & "<|fim_prefix|>" & chr(10)
    end if

    ' iterate messages
    for i as integer = 0 to nMsg-1
        dim content as string = trim(msgs(i).content)
        if msgs(i).role = "system" then
            if i <> 0 then
                s += "<|fim_prefix|>system" & chr(10) & content & "<|fim_prefix|>" & chr(10)
            end if
        elseif msgs(i).role = "user" then
            s += "<|fim_prefix|>user" & chr(10) & content & "<|fim_prefix|>" & chr(10)
        elseif msgs(i).role = "assistant" then
            s += "<|fim_prefix|>assistant" & chr(10)
            dim reasoning as string = ""
            if left(content, len("<think>")) = "<think>" and instr(content, "</think>") > 0 then
                dim p as integer = instr(content, "</think>")
                reasoning = mid(content, len("<think>")+1, p - len("<think>") - 1)
                content = trim(mid(content, p + len("</think>")))
            end if
            if trim(reasoning) <> "" then
                s += "<think>" & chr(10) & trim(reasoning) & chr(10) & "</think>" & chr(10) & chr(10)
            else
                s += "<think></think>" & chr(10) & chr(10)
            end if
            s += content
            if msgs(i).hasTool = 1 then
                dim block as string
                if trim(content) <> "" then
                    block = chr(10) & chr(10) & "<tool_call>" & chr(10) & "<function=" & msgs(i).toolName & ">" & chr(10) & msgs(i).toolArgs & "</function>" & chr(10) & "</tool_call>"
                else
                    block = "<tool_call>" & chr(10) & "<function=" & msgs(i).toolName & ">" & chr(10) & msgs(i).toolArgs & "</function>" & chr(10) & "</tool_call>"
                end if
                s += block
            end if
            s += "<|fim_prefix|>" & chr(10)
            s += chr(10)
        elseif msgs(i).role = "tool" then
            if i > 0 andalso msgs(i-1).role <> "tool" then
                s += "<|fim_prefix|>user"
            end if
            s += chr(10) & "<tool_response>" & chr(10) & content & chr(10) & "</tool_response>"
            if i = nMsg-1 or (i < nMsg-1 andalso msgs(i+1).role <> "tool") then
                s += "<|fim_prefix|>" & chr(10)
            end if
        end if
    next

    s += "<|fim_prefix|>assistant" & chr(10)
    s += "<think></think>" & chr(10) & chr(10)
    return s
end function

' ---------- helper: format number to plain digit string ----------
function fmtNum(d as double) as string
    if d = int(d) then
        return str(clng(d))
    else
        return str(d)
    end if
end function

' ---------- helper 2: parse_tool_call ----------
function isAllDigits(s as string) as integer
    dim t as string = trim(s)
    if t = "" then return 0
    for i as integer = 1 to len(t)
        dim c as string = mid(t, i, 1)
        if c < "0" or c > "9" then return 0
    next
    return 1
end function

function escapeJsonStr(s as string) as string
    dim r as string = ""
    for i as integer = 1 to len(s)
        dim c as string = mid(s, i, 1)
        if c = chr(34) then
            r += "\" & chr(34)
        elseif c = "\" then
            r += "\\"
        elseif c = chr(10) then
            r += "\n"
        elseif c = chr(13) then
            r += "\r"
        elseif c = chr(9) then
            r += "\t"
        else
            r += c
        end if
    next
    return r
end function


function trimWS(s as string) as string
    dim t as string = s
    do while len(t) > 0
        dim c as string = left(t, 1)
        if c = " " or c = chr(9) or c = chr(10) or c = chr(13) then
            t = mid(t, 2)
        else
            exit do
        end if
    loop
    do while len(t) > 0
        dim c as string = right(t, 1)
        if c = " " or c = chr(9) or c = chr(10) or c = chr(13) then
            t = left(t, len(t)-1)
        else
            exit do
        end if
    loop
    return t
end function

function parse_tool_call(reply as string) as ParsedCall
    dim res as ParsedCall
    res.ok = 0
    res.name = ""
    res.nArgs = 0
    res.norm = ""

    dim startTag as string = "<tool_call>"
    dim endTag as string = "</tool_call>"
    dim i as integer = instr(reply, startTag)
    if i = 0 then return res
    dim j as integer = instr(i, reply, endTag)
    if j = 0 then return res
    dim block as string = mid(reply, i + len(startTag), j - i - len(startTag))

    dim fnTag as string = "<function="
    dim fnStart as integer = instr(block, fnTag)
    if fnStart = 0 then return res
    dim nameStart as integer = fnStart + len(fnTag)
    dim gt as integer = instr(nameStart, block, ">")
    if gt = 0 then return res
    dim fname as string = trimWS(mid(block, nameStart, gt - nameStart))
    if fname = "" then return res

    ' extract parameters
    dim keys(0 to 9) as string
    dim vals(0 to 9) as string
    dim n as integer = 0
    dim scanPos as integer = 1
    do
        dim pStart as integer = instr(scanPos, block, "<parameter=")
        if pStart = 0 then exit do
        dim keyStart as integer = pStart + len("<parameter=")
        dim keyEnd as integer = instr(keyStart, block, ">")
        if keyEnd = 0 then return res
        dim key as string = trimWS(mid(block, keyStart, keyEnd - keyStart))
        dim valStart as integer = keyEnd + 1
        dim closeTag as string = "</parameter>"
        dim valEnd as integer = instr(valStart, block, closeTag)
        if valEnd = 0 then return res
        dim value as string = trimWS(mid(block, valStart, valEnd - valStart))
        if n < 10 then
            keys(n) = key
            vals(n) = value
            n += 1
        end if
        scanPos = valEnd + len(closeTag)
    loop

    ' build JSON string and parse with fbejson
    dim jsonStr as string = ""
    jsonStr = "{ ""name"" : """ & escapeJsonStr(fname) & """, ""arguments"" : { "
    for k as integer = 0 to n-1
        if k > 0 then jsonStr += ", "
        jsonStr += """" & escapeJsonStr(keys(k)) & """ : "
        if isAllDigits(vals(k)) then
            jsonStr += vals(k)
        else
            jsonStr += """" & escapeJsonStr(vals(k)) & """"
        end if
    next
    jsonStr += " } }"

    dim jobj as json.JSONobject
    dim p as json.JSONobject ptr = jobj.loadString(jsonStr)
    if p = 0 then return res
    if jobj.hasChild("name") = 0 then return res
    dim nv as json.JSONvalue ptr = jobj.child("name")
    if nv = 0 then return res
    dim parsedName as string = nv->getString()
    if parsedName = "" then return res
    if jobj.hasChild("arguments") = 0 then return res
    dim av as json.JSONvalue ptr = jobj.child("arguments")
    if av = 0 then return res
    dim aobj as json.JSONobject ptr = av->getObject()
    if aobj = 0 then return res

    ' walk arguments
    res.name = parsedName
    res.nArgs = 0
    for k as integer = 0 to n-1
        if aobj->hasChild(keys(k)) then
            dim vv as json.JSONvalue ptr = aobj->child(keys(k))
            dim vstr as string = ""
            if vv <> 0 then
                if vv->valueType() = json.jvalue_type.jstring then
                    vstr = trimWS(vv->getString())
                elseif vv->valueType() = json.jvalue_type.jnumber then
                    vstr = fmtNum(vv->getNumber())
                else
                    vstr = str(vv)
                end if
            end if
            res.argKeys(res.nArgs) = keys(k)
            res.argVals(res.nArgs) = vstr
            res.nArgs += 1
        end if
    next


    ' reorder arguments into canonical tool order
    dim canon(0 to 9) as string
    dim ncanon as integer = 0
    if res.name = "go" then
        canon(0)="direction": ncanon=1
    elseif res.name = "examine" or res.name = "take" or res.name = "drop" or res.name = "open" or res.name = "talk_to" then
        canon(0)="object": ncanon=1
    elseif res.name = "unlock" then
        canon(0)="object": canon(1)="code": ncanon=2
    elseif res.name = "use" then
        canon(0)="item": canon(1)="target": ncanon=2
    end if
    if ncanon > 0 then
        dim newKeys(0 to 9) as string
        dim newVals(0 to 9) as string
        dim nn as integer = 0
        for ci as integer = 0 to ncanon-1
            for k as integer = 0 to res.nArgs-1
                if res.argKeys(k) = canon(ci) then
                    newKeys(nn) = res.argKeys(k)
                    newVals(nn) = res.argVals(k)
                    nn += 1
                end if
            next
        next
        ' append any extras
        for k as integer = 0 to res.nArgs-1
            dim found as integer = 0
            for ci as integer = 0 to nn-1
                if newKeys(ci) = res.argKeys(k) then found = 1
            next
            if found = 0 then
                newKeys(nn) = res.argKeys(k)
                newVals(nn) = res.argVals(k)
                nn += 1
            end if
        next
        for k as integer = 0 to nn-1
            res.argKeys(k) = newKeys(k)
            res.argVals(k) = newVals(k)
        next
        res.nArgs = nn
    end if

    ' build norm string
    if res.nArgs = 0 then
        res.norm = res.name
    else
        dim ns as string = res.name & "("
        for k as integer = 0 to res.nArgs-1
            if k > 0 then ns += ", "
            ns += res.argVals(k)
        next
        ns += ")"
        res.norm = ns
    end if

    res.ok = 1
    return res
end function

' ---------- helper 3: append_tool_result ----------
sub append_tool_result(msgs() as ChatMsg, byref nMsg as integer, rname as string, result as string)
    if nMsg < 64 then
        msgs(nMsg).role = "tool"
        msgs(nMsg).content = result
        msgs(nMsg).hasTool = 0
        msgs(nMsg).toolName = ""
        msgs(nMsg).toolArgs = ""
        nMsg += 1
    end if
end sub

' ---------- model init ----------
sub ensure_model()
    dim modelPath as string = "models/raon-1-nano-Q4_K_M.gguf"
    if dir("models") = "" then mkdir "models"
    if dir(modelPath) = "" then
        print "Model not found. Downloading (~2GB)..."
        shell "curl -L -o " & modelPath & " https://huggingface.co/KRAFTON/Raon-1-nano-gguf/resolve/main/raon-1-nano-Q4_K_M.gguf"
    end if
end sub

sub llm_init()
    ensure_model()
    dim modelPath as string = "models/raon-1-nano-Q4_K_M.gguf"
    dim mparams as llama_model_params = llama_model_default_params()
    mparams.n_gpu_layers = -1
    g_model = llama_model_load_from_file(strptr(modelPath), mparams)
    if g_model = 0 then print "Failed to load model": sleep: end
    dim cparams as llama_context_params = llama_context_default_params()
    cparams.n_ctx = 4096
    cparams.n_batch = 4096
    g_ctx = llama_init_from_model(g_model, cparams)
    g_vocab = llama_model_get_vocab(g_model)

    dim sparams as llama_sampler_chain_params = llama_sampler_chain_default_params()
    g_smp = llama_sampler_chain_init(sparams)
    ' AGENTIC row: penalties (presence 0.0), top_k 40, top_p 0.9, temp 1.0, dist seed
    llama_sampler_chain_add(g_smp, llama_sampler_init_penalties(llama_vocab_n_tokens(g_vocab), 64, 1.0, 0.0, 0.0))
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_k(40))
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_p(0.9, 1))
    llama_sampler_chain_add(g_smp, llama_sampler_init_temp(1.0))
    llama_sampler_chain_add(g_smp, llama_sampler_init_dist(LLAMA_DEFAULT_SEED))
end sub

function generate_reply(sysContent as string, usrInput as string) as string
    llama_memory_clear(llama_get_memory(g_ctx), true)

    dim msgs(0 to 3) as ChatMsg
    msgs(0).role = "system"
    msgs(0).content = sysContent
    msgs(0).hasTool = 0
    msgs(1).role = "user"
    msgs(1).content = usrInput
    msgs(1).hasTool = 0

    dim prompt as string = build_prompt(msgs(), 2, toolJson(), nTools)

    ' tokenize
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
        if seqArr <> 0 then *(*(seqArr + i)) = 0
        *(logArr + i) = 0
    next
    *(logArr + (nTok-1)) = 1
    batch.n_tokens = nTok
    llama_decode(g_ctx, batch)
    llama_batch_free(batch)

    dim result as string = ""
    dim nCur as long = nTok
    dim nGen as integer = 0
    while nGen < 128
        dim tok as llama_token = llama_sampler_sample(g_smp, g_ctx, -1)
        llama_sampler_accept(g_smp, tok)
        if llama_token_is_eog(g_vocab, tok) then exit while
        dim pbuf as zstring ptr = allocate(256)
        dim plen as long = llama_token_to_piece(g_vocab, tok, pbuf, 256, 0, true)
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

' ---------- room helpers ----------
function roomName(r as integer) as string
    select case r
        case 0: return "Lobby"
        case 1: return "The Lab"
        case 2: return "Director's Office"
        case 3: return "Break Room"
    end select
    return "Unknown"
end function

function roomExits(r as integer) as string
    select case r
        case 0: return "north, east, west"
        case 1: return "west"
        case 2: return "south"
        case 3: return "east"
    end select
    return ""
end function

function roomDesc(r as integer) as string
    dim desc as string
    select case r
        case 0: desc = "Fumble Labs' lobby, after hours. The exit door regards you with bureaucratic indifference. A palm scanner glows red beside it."
        case 1: desc = "The lab hums and bubbles. Something in a tank is awake."
        case 2: desc = "Dr. Fumble's office. The portrait's eyes follow you with the ease of long practice."
        case 3: desc = "The break room smells of old coffee and older grudges."
    end select
    return desc & " Exits: " & roomExits(r) & "."
end function

function hasItem(id as string) as integer
    for i as integer = 0 to nInv-1
        if inventory(i) = id then return 1
    next
    return 0
end function

sub addItem(id as string)
    if nInv < 32 then
        inventory(nInv) = id
        nInv += 1
    end if
end sub

sub removeItem(id as string)
    for i as integer = 0 to nInv-1
        if inventory(i) = id then
            for j as integer = i to nInv-2
                inventory(j) = inventory(j+1)
            next
            nInv -= 1
            return
        end if
    next
end sub

function roomObjects(r as integer) as string
    select case r
        case 0: return "poster, rubber plant, reception desk, exit door, palm scanner"
        case 1: return "fish tank, Gerald, vats, corkboard, rubber chicken"
        case 2: return "portrait, office desk, trophy, safe, prosthetic hand"
        case 3: return "whiteboard, fridge, microwave, red herring"
    end select
    return ""
end function

function objPresent(obj as string, r as integer) as integer
    ' is object in this room (considering state)
    if r = 0 then
        if obj = "poster" or obj = "rubber plant" or obj = "reception desk" or obj = "exit door" or obj = "palm scanner" then return 1
    elseif r = 1 then
        if obj = "fish tank" or obj = "Gerald" or obj = "vats" or obj = "corkboard" then return 1
        if obj = "rubber chicken" and chickenTaken = 0 then return 1
    elseif r = 2 then
        if obj = "portrait" or obj = "office desk" or obj = "trophy" or obj = "safe" then return 1
        if obj = "prosthetic hand" and safeUnlocked = 1 and handTaken = 0 then return 1
    elseif r = 3 then
        if obj = "whiteboard" or obj = "fridge" or obj = "microwave" then return 1
        if obj = "red herring" and fridgeOpened = 1 and redHerringTaken = 0 then return 1
    end if
    return 0
end function

function takeable(obj as string) as integer
    if obj = "rubber chicken" or obj = "red herring" or obj = "prosthetic hand" then return 1
    return 0
end function

function examineText(obj as string) as string
    select case obj
        case "poster": return "HANG IN THERE, a cat in a lab coat: The cat has tenure. You do not."
        case "rubber plant": return "It's thriving. Somehow this feels personal."
        case "reception desk": return "A sign-in sheet. The last entry reads: 'Dr. Fumble - OUT - do not wait up.' You waited up. All the drawers are stuck closed, as if the desk itself is refusing to cooperate."
        case "exit door": return "Locked. A small plaque: 'AUTHORIZED PALMS ONLY.'"
        case "palm scanner": return "It wants a palm with a career. Yours is still in beta."
        case "fish tank": return "Gerald the goldfish drifts inside, radiating the calm of a creature with no performance review."
        case "Gerald": return "Gerald the goldfish drifts inside, radiating the calm of a creature with no performance review."
        case "vats": return "Vat A is labeled 'DO NOT SHAKE'. Vat B is labeled 'SHAKE WELL'. The lab runs on trust."
        case "corkboard": return "A rota, three safety warnings, and a photo of last year's Science Fair trophy. Regional champions. Least explosions."
        case "rubber chicken": return "A rubber chicken. No label, no explanation, no apology."
        case "portrait": return "The plaque reads 'DR. A. FUMBLE, DIRECTOR'. The eyes read you."
        case "office desk": return "Locked drawers (all of them stuck closed, naturally), a phone with no dial tone, and a family photo facing the wall. The family knows what they did."
        case "trophy": return "REGIONAL SCIENCE FAIR CHAMPIONS, 1987. Category: Least Explosions. Someone has scratched out 'Least' and written 'Fewest'. Pedants."
        case "safe": 
            if safeUnlocked = 0 then return "A squat grey safe. Four-digit code, no hints, no mercy."
            return "The safe is open. Inside was a prosthetic hand."
        case "prosthetic hand": return "Dr. Fumble's spare prosthetic hand, in a velvet-lined box labeled 'FOR EMERGENCIES / DO NOT LEND TO GERALD'. The hand has a parking validation sticker."
        case "whiteboard": return "Among the grievances ('WHO KEEPS STEALING MY YOGURT', 'DO NOT MICROWAVE FISH') one line is underlined twice: 'Safe code changed AGAIN. It's the year we won the Science Fair. You're welcome. - F.'"
        case "fridge": return "It hums the song of its people."
        case "microwave": return "Inside is a fossil. It may once have been a burrito. Science will never know."
        case "red herring": return "A red herring. It will not help. They never do."
    end select
    return ""
end function

function geraldLine() as string
    if geraldTurn = 0 then
        geraldTurn = 1
        return "Gerald blows a bubble. It rises like your hopes and pops the same way."
    else
        geraldTurn = 0
        return "Gerald stares. He has seen interns come and go. Mostly go."
    end if
end function

function handleIntent(pc as ParsedCall) as string
    if won = 1 then return "You already escaped. (Type quit to exit.)"

    if pc.name = "go" then
        dim direction as string
        direction = ""
        if pc.nArgs >= 1 then direction = lcase(trim(pc.argVals(0)))
        ' exits
        dim dest as integer = -1
        select case curRoom
            case 0 ' Lobby
                if direction = "north" then dest = 2
                if direction = "east" then dest = 1
                if direction = "west" then dest = 3
            case 1 ' Lab
                if direction = "west" then dest = 0
            case 2 ' Office
                if direction = "south" then dest = 0
            case 3 ' Break
                if direction = "east" then dest = 0
        end select
        if dest < 0 then return "No exit that way."
        curRoom = dest
        return roomDesc(curRoom)
    elseif pc.name = "look" then
        return roomDesc(curRoom)
    elseif pc.name = "examine" then
        dim obj as string = ""
        if pc.nArgs >= 1 then obj = pc.argVals(0)
        if obj = "desk" then
            if curRoom = 0 then obj = "reception desk"
            if curRoom = 2 then obj = "office desk"
        end if
        if curRoom = 3 and obj = "fish" then obj = "red herring"
        if objPresent(obj, curRoom) = 0 and hasItem(obj) = 0 then
            return "You don't see that here."
        end if
        dim t as string = examineText(obj)
        if t = "" then return "Nothing remarkable."
        return t
    elseif pc.name = "take" then
        dim obj as string = ""
        if pc.nArgs >= 1 then obj = pc.argVals(0)
        if obj = "desk" then
            if curRoom = 0 then obj = "reception desk"
            if curRoom = 2 then obj = "office desk"
        end if
        if curRoom = 3 and obj = "fish" then obj = "red herring"
        if obj = "burrito" then return "You can't get it free, the burrito is now load bearing."
        if takeable(obj) = 0 then return "You can't take that."
        if objPresent(obj, curRoom) = 0 then return "That isn't here to take."
        if hasItem(obj) = 1 then return "You already have that."
        if obj = "red herring" then
            redHerringTaken = 1
            addItem(obj)
            return "You take the red herring. It will not help. They never do."
        elseif obj = "rubber chicken" then
            chickenTaken = 1
            addItem(obj)
            return "You take the rubber chicken."
        elseif obj = "prosthetic hand" then
            handTaken = 1
            addItem(obj)
            return "You take the prosthetic hand."
        end if
        addItem(obj)
        return "You take the " & obj & "."
    elseif pc.name = "drop" then
        dim obj as string = ""
        if pc.nArgs >= 1 then obj = pc.argVals(0)
        if hasItem(obj) = 0 then return "You aren't carrying that."
        removeItem(obj)
        return "You drop the " & obj & "."
    elseif pc.name = "open" then
        dim obj as string = ""
        if pc.nArgs >= 1 then obj = pc.argVals(0)
        if obj = "desk" then
            if curRoom = 0 then obj = "reception desk"
            if curRoom = 2 then obj = "office desk"
        end if
        if curRoom = 3 and obj = "fish" then obj = "red herring"
        if obj = "fridge" then
            if fridgeOpened = 1 then
                fridgeOpened = 0
                return "You close the fridge. It hums in relief."
            end if
            fridgeOpened = 1
            return "You open the fridge. Inside: a red herring. Of course."
        elseif obj = "safe" then
            return "It wants the code first. Safes are traditionalists."
        elseif obj = "microwave" then
            if microwaveOpened = 1 then
                if microwaveHasHerring = 1 and microwaveRuined = 0 then
                    microwaveRuined = 1
                    microwaveHasHerring = 0
                    microwaveOpened = 0
                    return "You close the microwave and hit start. A smell happens. Well, that microwave is ruined."
                end if
                microwaveOpened = 0
                return "You close the microwave."
            end if
            microwaveOpened = 1
            if microwaveRuined = 1 then
                return "You open the microwave. It is ruined. The fossil remains, somehow worse."
            end if
            return "You open the microwave. The fossil remains. Science still won't know."
        else
            return "There's nothing to open there."
        end if
    elseif pc.name = "unlock" then
        dim obj as string = ""
        if pc.nArgs >= 1 then obj = pc.argVals(0)
        dim code as string = ""
        if pc.nArgs >= 2 then code = pc.argVals(1)
        if obj = "safe" then
            if code = "1987" then
                safeUnlocked = 1
                return "The safe clicks open. Inside: a prosthetic hand in a velvet-lined box."
            else
                return "The safe buzzes. Somewhere, a tiny internal voice says 'nope'."
            end if
        end if
        return "You can't unlock that."
    elseif pc.name = "use" then
        dim item as string = ""
        if pc.nArgs >= 1 then item = pc.argVals(0)
        dim target as string = ""
        if pc.nArgs >= 2 then target = pc.argVals(1)
        if curRoom = 3 and item = "fish" then item = "red herring"
        if curRoom = 3 and target = "fish" then target = "red herring"
        if item = "red herring" and target = "microwave" and curRoom = 3 then
            if hasItem("red herring") = 0 then return "You aren't holding that."
            removeItem("red herring")
            microwaveHasHerring = 1
            return "You put the red herring in the microwave. It fits. Barely."
        end if
        if hasItem(item) = 0 then return "You aren't holding that."
        if item = "rubber chicken" then
            dim t as string = target
            if t = "" then t = "that"
            return "The rubber chicken makes a funny sound but " & t & " doesn't seem as amused as you are."
        end if
        if (item = "fish" or item = "fish tank") and target = "Gerald" then item = "red herring"
        if item = "red herring" and (target = "Gerald" or target = "fish tank") then
            if hasItem("red herring") = 0 then return "You aren't holding that."
            removeItem("red herring")
            return "Wow, who would have thought that little old Gerald was that hungry."
        end if
        if item = "red herring" and target = "vats" then
            if hasItem("red herring") = 0 then return "You aren't holding that."
            removeItem("red herring")
            return "Into Vat A (DO NOT SHAKE) it goes: a herring that refuses to be stirred. Into Vat B (SHAKE WELL) it goes: a herring that's well and truly shaken. Either way, it's a vat-astrophe."
        end if
        if item = "red herring" and target = "palm scanner" then
            return "Well, it's a little oily and smells of fish. What did you expect to happen?"
        end if
        if item = "red herring" and target = "exit door" then
            return "Fish are notoriously anti-door and yet it resists your attempts."
        end if
        if item = "red herring" and target = "rubber plant" then
            return "That's not its natural environment."
        end if
        if item = "prosthetic hand" and target = "palm scanner" and curRoom = 0 then
            won = 1
            return "The scanner considers the hand. The hand, it decides, has a future here. The door slides open." & chr(10) & "You escape into the night. The hand gets a parking validation. You do not."
        end if
        return "Nothing happens."
    elseif pc.name = "talk_to" then
        dim obj as string = ""
        if pc.nArgs >= 1 then obj = pc.argVals(0)
        if curRoom <> 1 then return "There's no one to talk to here."
        if obj = "Gerald" or obj = "fish tank" then
            return geraldLine()
        end if
        return "They don't respond."
    elseif pc.name = "inventory" then
        if nInv = 0 then return "You're carrying nothing."
        dim s as string = "You're carrying: "
        for i as integer = 0 to nInv-1
            if i > 0 then s += ", "
            s += inventory(i)
        next
        return s
    end if
    return "You can't do that."
end function

' ---------- drawing ----------
sub draw_room_bg(byref room_name as const string = "Lobby")
	dim wall_color as long = rgb(46,42,52)
	select case room_name
		case "Director's Office":
			wall_color = rgb(200, 200, 150)
		case "The Lab":
			wall_color = rgb(175, 175, 175)
		case "Break Room":
			wall_color = rgb(80, 80, 200)
	end select
    ' back wall
    line (0, 40)-(SC_W-1, 300), wall_color, bf
    ' checkerboard floor in perspective
    const TCOLS = 8
    const TROWS = 6
    dim topXs(0 to TCOLS) as integer
    dim botXs(0 to TCOLS) as integer
    for c as integer = 0 to TCOLS
        topXs(c) = 0 + c * 100
        botXs(c) = -100 + c * 125
    next
    for row as integer = 0 to TROWS-1
        dim yTop as integer = 300 + row * (60 \ TROWS)
        dim yBot as integer = 300 + (row + 1) * (60 \ TROWS)
        for col as integer = 0 to TCOLS-1
            dim light as integer
            if ((col + row) mod 2) = 0 then light = 1 else light = 0
            dim tc as uinteger
            if light = 1 then tc = rgb(95, 82, 68) else tc = rgb(52, 46, 38)
            for yy as integer = yTop to yBot - 1
                dim t as double = (yy - 300) / 60.0
                dim xl as integer = int(topXs(col) + (botXs(col) - topXs(col)) * t)
                dim xr as integer = int(topXs(col+1) + (botXs(col+1) - topXs(col+1)) * t)
                line (xl, yy)-(xr, yy), tc
            next
        next
    next
    ' baseboard
    line (0, 298)-(SC_W-1, 302), rgb(30, 26, 24), bf
end sub

sub draw_room()
    screenlock
    ' title bar
    line (0, 0)-(SC_W-1, 40), rgb(40, 40, 60), bf
    draw string (10, 12), roomName(curRoom)
    ' semi-3D room background
    draw_room_bg(roomName(curRoom))

    if curRoom = 0 then
        ' walls/floor
        ' door
        line (650, 100)-(760, 300), rgb(90, 60, 30), bf
        line (650, 100)-(760, 300), rgb(200, 200, 200), b
        if won = 1 then
            ' open door: draw gap
            line (650, 100)-(760, 300), rgb(90, 60, 30), b
            line (650, 100)-(750, 300), rgb(20, 20, 30), bf
        end if
        ' scanner plate
        dim scol as uinteger
        if won = 1 then scol = rgb(0, 200, 0) else scol = rgb(200, 0, 0)
        circle (600, 210), 24, scol, , , , f
        circle(600, 210), 10, rgb(200, 200, 100), , , 1.67, f
        line (576, 186)-(624, 234), rgb(200, 200, 200), b
        ' poster
        view screen(80, 120)-(160,200)
		line (80, 120)-(160, 200), rgb(240, 240, 200), bf
        circle (150, 120), 50, rgb(0, 100, 0), , , , f
        line (150, 135)-(160, 200), rgb(150, 110, 70), bf
        line (90, 160)-(150, 165), rgb(150, 110, 70), bf
        line (130, 160)-(135, 175), rgb(200, 50, 50), bf
        line (110, 160)-(115, 175), rgb(200, 50, 50), bf
        line (110, 175)-(135, 200), rgb(200, 200, 200), bf
        circle (122, 180), 25, rgb(200, 50, 50), , , 0.2, f
        draw string (85, 130), "HANG IN", rgb(20, 20, 20)
        draw string (85, 145), "THERE", rgb(20, 20, 20)
        view
        ' rubber plant (3D pot + foliage)
        'pot
        circle (260, 270), 15, rgb(110, 70, 40), , , 0.2, f
        line (245, 270)-(275, 270), rgb(110, 70, 40), bf
        line (245, 270)-(247, 300), rgb(110, 70, 40)
        line (275, 270)-(273, 300), rgb(110, 70, 40)
        'line (245, 300)-(275, 300), rgb(110, 70, 40)
        circle (260, 300), 13, rgb(110, 70, 40), , , 0.2, f
        paint (260, 280), rgb(110, 70, 40), rgb(110, 70, 40)
        circle (260, 270), 15, rgb(40, 70, 40), , , 0.2, f
        'foliage
        line (257, 205)-(263, 270), rgb(150, 110, 70), bf
        circle (260, 215), 26, rgb(40, 120, 40), , , , f
        circle (248, 205), 16, rgb(50, 140, 50), , , , f
        circle (272, 208), 14, rgb(30, 110, 40), , , , f
        ' reception desk (3D cuboid with thin legs)
        line (360, 240)-(520, 240), rgb(150, 110, 70)
        line (520, 240)-(540, 260), rgb(150, 110, 70)
        line (540, 260)-(380, 260), rgb(150, 110, 70)
        '
        line (380, 260)-(360, 240), rgb(150, 110, 70)
        paint (440, 250), rgb(150, 110, 70), rgb(150, 110, 70)
        line (360, 240)-(360, 280), rgb(120, 90, 60)
		line (380, 260)-(380, 300), rgb(120, 90, 60) 'left front edge
		line (360, 280)-(380, 300), rgb(120, 90, 60)
        line (380, 300)-(520, 300), rgb(120, 90, 60)
        line (520, 240)-(540, 260), rgb(100, 75, 50)
        line (540, 260)-(540, 300), rgb(100, 75, 50)
        line (520, 300)-(540, 300), rgb(100, 75, 50)
        'paint (350, 270), rgb(120, 90, 60), rgb(120, 90, 60)

        ' thin legs
        line (365, 285)-(365, 305), rgb(90, 65, 40)
        line (515, 300)-(515, 305), rgb(90, 65, 40)
        line (385, 300)-(385, 325), rgb(90, 65, 40)
        line (535, 300)-(535, 325), rgb(90, 65, 40)
        draw string (420, 270), "DESK"
    elseif curRoom = 1 then
        ' table for tank
        line (100, 285)-(240, 285), rgb(100, 70, 40)
        line (100, 285)-(90, 300), rgb(100, 70, 40)
        line (240, 285)-(250, 300), rgb(100, 70, 40)
        line (90, 300)-(250, 300), rgb(100, 70, 40)
        ' tank as cuboid
        line (120, 170)-(220, 170), rgb(100, 150, 200)
        line (220, 170)-(235, 185), rgb(100, 150, 200)
        line (235, 185)-(135, 185), rgb(100, 150, 200)
        line (135, 185)-(120, 170), rgb(100, 150, 200)
        paint (170, 178), rgb(100, 150, 200), rgb(100, 150, 200)
        line (120, 170)-(120, 275), rgb(80, 120, 180)
        line (220, 170)-(220, 275), rgb(80, 120, 180)
        line (120, 275)-(220, 275), rgb(80, 120, 180)
        line (220, 170)-(235, 185), rgb(70, 100, 150)
        line (235, 185)-(235, 290), rgb(70, 100, 150)
        line (220, 275)-(235, 290), rgb(70, 100, 150)
        ' Gerald the goldfish
        circle (170, 225), 16, rgb(255, 160, 40), , , 0.7, f
        line (154, 225)-(138, 213), rgb(255, 160, 40)
        line (154, 225)-(138, 237), rgb(255, 160, 40)
        line (138, 213)-(138, 237), rgb(255, 160, 40)
        paint (145, 225), rgb(255, 160, 40), rgb(255, 160, 40)
        circle (180, 221), 3, rgb(255, 255, 255), , , , f
        circle (181, 221), 1.5, rgb(0, 0, 0), , , , f
        circle (178, 227), 5, rgb(0, 0, 0), 4.9, 5.9
        draw string (130, 160), "TANK"
        ' vats as tall cylinders
        circle (360, 250), 28, rgb(100, 100, 120), , , 0.45, f
        line (332, 250)-(332, 300), rgb(100, 100, 120)
        line (388, 250)-(388, 300), rgb(100, 100, 120)
        circle (360, 300), 28, rgb(90, 90, 110), , , 0.45, f
        circle (470, 250), 28, rgb(100, 100, 120), , , 0.45, f
        line (442, 250)-(442, 300), rgb(100, 100, 120)
        line (498, 250)-(498, 300), rgb(100, 100, 120)
        circle (470, 300), 28, rgb(90, 90, 110), , , 0.45, f
        draw string (345, 205), "A"
        draw string (455, 205), "B"
        draw string (330, 210), "VATS"
        ' corkboard
        line (560, 140)-(680, 240), rgb(180, 140, 80), bf
        draw string (570, 150), "BOARD"
        ' rubber chicken
        if chickenTaken = 0 then
            ' stereotypical rubber chicken
            circle (525, 285), 14, rgb(230, 200, 60), , , 1.3, f
            circle (545, 272), 9, rgb(230, 200, 60), , , , f
            ' beak
            line (553, 270)-(566, 266), rgb(230, 120, 40)
            line (553, 274)-(566, 278), rgb(230, 120, 40)
            line (566, 266)-(566, 278), rgb(230, 120, 40)
            ' comb
            circle (544, 262), 4, rgb(220, 40, 40), , , , f
            circle (540, 260), 3, rgb(220, 40, 40), , , , f
            circle (548, 260), 3, rgb(220, 40, 40), , , , f
            ' wattle
            circle (548, 280), 3, rgb(220, 40, 40), , , , f
            ' wing
            circle (520, 282), 8, rgb(200, 170, 40), , , 0.6, f
            ' legs
            line (522, 295)-(520, 312), rgb(230, 180, 60)
            line (532, 295)-(534, 312), rgb(230, 180, 60)
            draw string (505, 260), "CHICKEN"
        end if
    elseif curRoom = 2 then
        ' large window behind desk, extending to the right
        line (290, 55)-(770, 205), rgb(120, 160, 200), bf
        line (290, 55)-(770, 205), rgb(60, 90, 130), b
        line (292, 57)-(768, 120), rgb(150, 190, 230), bf
        line (292, 120)-(768, 203), rgb(70, 70, 75), bf
        for i as integer = 0 to 6
            line (292 + i*68, 120)-(292 + i*68, 203), rgb(200, 200, 200)
        next
        ' one car in the lot
        line (520, 150)-(575, 175), rgb(200, 40, 40), bf
        line (525, 142)-(570, 150), rgb(200, 40, 40), bf
        circle (530, 178), 4, rgb(30, 30, 30), , , , f
        circle (565, 178), 4, rgb(30, 30, 30), , , , f
        ' portrait frame
        line (100, 100)-(220, 260), rgb(60, 40, 80), bf
        line (100, 100)-(220, 260), rgb(200, 200, 200), b
        ' face
        circle (160, 165), 42, rgb(230, 200, 170), , , 1.1, f
        ' shoulders / suit (trapezoid)
        line (108, 210)-(212, 210), rgb(120, 200, 120)
        line (108, 210)-(100, 260), rgb(120, 200, 120)
        line (212, 210)-(220, 260), rgb(120, 200, 120)
        line (100, 260)-(220, 260), rgb(120, 200, 120)
        paint (160, 235), rgb(120, 200, 120), rgb(120, 200, 120)
        ' eyes (white sclera fixed)
        circle (145, 170), 7, rgb(255, 255, 255), , , , f
        circle (175, 170), 7, rgb(255, 255, 255), , , , f
        ' pupils follow cursor, clamped inside sclera
        dim mx as integer, my as integer
        getmouse mx, my
        if mx < 0 then mx = 160
        if my < 0 then my = 180
        dim dx as integer = (mx - 400) \ 25
        dim dy as integer = (my - 300) \ 25
        if dx < -4 then dx = -4
        if dx > 4 then dx = 4
        if dy < -4 then dy = -4
        if dy > 4 then dy = 4
        circle (145 + dx, 170 + dy), 3, rgb(0, 0, 0), , , , f
        circle (175 + dx, 170 + dy), 3, rgb(0, 0, 0), , , , f
        ' nose and mouth
        line (160, 178)-(160, 190), rgb(180, 140, 120)
        circle (160, 198), 8, rgb(180, 100, 100), 4.7, 5.9
        draw string (110, 80), "PORTRAIT"
        ' office desk (3D cuboid with thin legs)
        line (300, 240)-(460, 240), rgb(130, 95, 60)
        line (460, 240)-(480, 260), rgb(130, 95, 60)
        line (480, 260)-(320, 260), rgb(130, 95, 60)
        line (320, 260)-(300, 240), rgb(130, 95, 60)
        paint (380, 250), rgb(130, 95, 60), rgb(130, 95, 60)
        line (300, 240)-(300, 300), rgb(100, 70, 40)
        line (460, 240)-(460, 300), rgb(100, 70, 40)
        line (300, 300)-(460, 300), rgb(100, 70, 40)
        line (460, 240)-(480, 260), rgb(85, 60, 35)
        line (480, 260)-(480, 300), rgb(85, 60, 35)
        line (460, 300)-(480, 300), rgb(85, 60, 35)
        line (305, 300)-(305, 325), rgb(80, 55, 30)
        line (455, 300)-(455, 325), rgb(80, 55, 30)
        line (325, 300)-(325, 325), rgb(80, 55, 30)
        line (475, 300)-(475, 325), rgb(80, 55, 30)
        draw string (310, 270), "DESK"
        ' table for trophy and safe
        line (500, 285)-(740, 285), rgb(100, 70, 40)
        line (500, 285)-(490, 300), rgb(100, 70, 40)
        line (740, 285)-(750, 300), rgb(100, 70, 40)
        line (490, 300)-(750, 300), rgb(100, 70, 40)
        ' trophy (cheap, says 1nd)
        ' base
        line (535, 278)-(565, 288), rgb(180, 140, 40), bf
        ' stem
        line (546, 260)-(554, 278), rgb(200, 170, 50), bf
        ' cup
        circle (550, 245), 16, rgb(220, 190, 60), , , 0.9, f
        line (532, 232)-(532, 260), rgb(220, 190, 60)
        line (568, 232)-(568, 260), rgb(220, 190, 60)
        line (532, 260)-(568, 260), rgb(220, 190, 60)
        circle (550, 232), 18, rgb(220, 190, 60), 3.14, 6.28
        ' handles
        circle (528, 245), 6, rgb(200, 170, 50)
        circle (572, 245), 6, rgb(200, 170, 50)
        draw string (540, 238), "1nd"
        draw string (520, 210), "TROPHY"
        ' safe
        line (620, 230)-(720, 285), rgb(120, 120, 120), bf
        line (620, 230)-(720, 300), rgb(200, 200, 200), b
        draw string (635, 210), "SAFE"
        if safeUnlocked = 1 and handTaken = 0 then
            line (635, 245)-(705, 285), rgb(240, 200, 200), bf
            draw string (640, 255), "HAND"
        end if
    elseif curRoom = 3 then
        ' whiteboard
        line (80, 120)-(300, 240), rgb(240, 240, 240), bf
        line (80, 120)-(300, 240), rgb(100, 100, 100), b
        draw string (90, 130), "WHITEBOARD"
        ' fridge as tall cuboid
        line (350, 150)-(430, 150), rgb(210, 220, 230)
        line (430, 150)-(445, 165), rgb(210, 220, 230)
        line (445, 165)-(365, 165), rgb(210, 220, 230)
        line (365, 165)-(350, 150), rgb(210, 220, 230)
        paint (395, 158), rgb(210, 220, 230), rgb(210, 220, 230)
        line (350, 150)-(350, 300), rgb(200, 210, 220)
        line (430, 150)-(430, 300), rgb(200, 210, 220)
        line (350, 300)-(430, 300), rgb(200, 210, 220)
        line (430, 150)-(445, 165), rgb(180, 190, 200)
        line (445, 165)-(445, 315), rgb(180, 190, 200)
        line (430, 300)-(445, 315), rgb(180, 190, 200)
        draw string (355, 130), "FRIDGE"
        if fridgeOpened = 1 and redHerringTaken = 0 then
            draw string (355, 315), "red herring inside"
        end if
        ' table for microwave
        line (490, 275)-(590, 275), rgb(100, 70, 40)
        line (485, 290)-(595, 290), rgb(100, 70, 40)
        line (490, 275)-(485, 290), rgb(100, 70, 40)
        line (590, 275)-(595, 290), rgb(100, 70, 40)
        ' microwave as smaller cuboid
        line (505, 245)-(575, 245), rgb(160, 160, 160)
        line (575, 245)-(585, 255), rgb(160, 160, 160)
        line (585, 255)-(515, 255), rgb(160, 160, 160)
        line (515, 255)-(505, 245), rgb(160, 160, 160)
        paint (540, 250), rgb(160, 160, 160), rgb(160, 160, 160)
        line (505, 245)-(505, 275), rgb(140, 140, 140)
        line (575, 245)-(575, 275), rgb(140, 140, 140)
        line (505, 275)-(575, 275), rgb(140, 140, 140)
        line (575, 245)-(585, 255), rgb(120, 120, 120)
        line (585, 255)-(585, 285), rgb(120, 120, 120)
        line (575, 275)-(585, 285), rgb(120, 120, 120)
        draw string (505, 225), "MICROWAVE"
    end if

    ' log area
    line (0, 360)-(SC_W-1, 540), rgb(10, 10, 20), bf
    dim y as integer = 365
    dim start as integer = 0
    if nLog > 14 then start = nLog - 14
    for i as integer = start to nLog-1
        draw string (8, y), logLines(i)
        y += 12
        if y > 530 then exit for
    next

    ' input line
    line (0, 540)-(SC_W-1, SC_H-1), rgb(30, 30, 40), bf
    draw string (8, 555), "> " & inputBuf & "_"

    screenunlock
end sub

' ---------- tool json init ----------
sub init_tools()
    ' go
    dim func as json.JSONobject ptr = new json.JSONobject
    func->addChild("name", new json.JSONvalue("go"))
    func->addChild("description", new json.JSONvalue("move to an adjacent room"))
    dim params as json.JSONobject ptr = new json.JSONobject
    params->addChild("type", new json.JSONvalue("object"))
    dim props as json.JSONobject ptr = new json.JSONobject
    dim dp as json.JSONobject ptr = new json.JSONobject
    dp->addChild("type", new json.JSONvalue("string"))
    dim en as json.JSONvalue ptr ptr = new json.JSONvalue ptr[4]
    en[0] = new json.JSONvalue("north"): en[1] = new json.JSONvalue("south"): en[2] = new json.JSONvalue("east"): en[3] = new json.JSONvalue("west")
    dp->addChild("enum", new json.JSONvalue(new json.JSONarray(en, 4)))
    props->addChild("direction", new json.JSONvalue(dp))
    params->addChild("properties", new json.JSONvalue(props))
    dim req as json.JSONvalue ptr ptr = new json.JSONvalue ptr[1]
    req[0] = new json.JSONvalue("direction")
    params->addChild("required", new json.JSONvalue(new json.JSONarray(req, 1)))
    func->addChild("parameters", new json.JSONvalue(params))
    dim tool as json.JSONobject ptr = new json.JSONobject
    tool->addChild("type", new json.JSONvalue("function"))
    tool->addChild("function", new json.JSONvalue(func))
    toolJson(nTools) = *tool: nTools += 1

    ' look
    func = new json.JSONobject
    func->addChild("name", new json.JSONvalue("look"))
    func->addChild("description", new json.JSONvalue("look around the current room"))
    params = new json.JSONobject
    params->addChild("type", new json.JSONvalue("object"))
    params->addChild("properties", new json.JSONvalue(new json.JSONobject))
    func->addChild("parameters", new json.JSONvalue(params))
    tool = new json.JSONobject
    tool->addChild("type", new json.JSONvalue("function"))
    tool->addChild("function", new json.JSONvalue(func))
    toolJson(nTools) = *tool: nTools += 1

    ' examine
    func = new json.JSONobject
    func->addChild("name", new json.JSONvalue("examine"))
    func->addChild("description", new json.JSONvalue("examine an object"))
    params = new json.JSONobject
    params->addChild("type", new json.JSONvalue("object"))
    props = new json.JSONobject
    dp = new json.JSONobject
    dp->addChild("type", new json.JSONvalue("string"))
    props->addChild("object", new json.JSONvalue(dp))
    params->addChild("properties", new json.JSONvalue(props))
    req = new json.JSONvalue ptr[1]
    req[0] = new json.JSONvalue("object")
    params->addChild("required", new json.JSONvalue(new json.JSONarray(req, 1)))
    func->addChild("parameters", new json.JSONvalue(params))
    tool = new json.JSONobject
    tool->addChild("type", new json.JSONvalue("function"))
    tool->addChild("function", new json.JSONvalue(func))
    toolJson(nTools) = *tool: nTools += 1

    ' take
    func = new json.JSONobject
    func->addChild("name", new json.JSONvalue("take"))
    func->addChild("description", new json.JSONvalue("take an object"))
    params = new json.JSONobject
    params->addChild("type", new json.JSONvalue("object"))
    props = new json.JSONobject
    dp = new json.JSONobject
    dp->addChild("type", new json.JSONvalue("string"))
    props->addChild("object", new json.JSONvalue(dp))
    params->addChild("properties", new json.JSONvalue(props))
    req = new json.JSONvalue ptr[1]
    req[0] = new json.JSONvalue("object")
    params->addChild("required", new json.JSONvalue(new json.JSONarray(req, 1)))
    func->addChild("parameters", new json.JSONvalue(params))
    tool = new json.JSONobject
    tool->addChild("type", new json.JSONvalue("function"))
    tool->addChild("function", new json.JSONvalue(func))
    toolJson(nTools) = *tool: nTools += 1

    ' drop
    func = new json.JSONobject
    func->addChild("name", new json.JSONvalue("drop"))
    func->addChild("description", new json.JSONvalue("drop a carried object"))
    params = new json.JSONobject
    params->addChild("type", new json.JSONvalue("object"))
    props = new json.JSONobject
    dp = new json.JSONobject
    dp->addChild("type", new json.JSONvalue("string"))
    props->addChild("object", new json.JSONvalue(dp))
    params->addChild("properties", new json.JSONvalue(props))
    req = new json.JSONvalue ptr[1]
    req[0] = new json.JSONvalue("object")
    params->addChild("required", new json.JSONvalue(new json.JSONarray(req, 1)))
    func->addChild("parameters", new json.JSONvalue(params))
    tool = new json.JSONobject
    tool->addChild("type", new json.JSONvalue("function"))
    tool->addChild("function", new json.JSONvalue(func))
    toolJson(nTools) = *tool: nTools += 1

    ' open
    func = new json.JSONobject
    func->addChild("name", new json.JSONvalue("open"))
    func->addChild("description", new json.JSONvalue("open an object"))
    params = new json.JSONobject
    params->addChild("type", new json.JSONvalue("object"))
    props = new json.JSONobject
    dp = new json.JSONobject
    dp->addChild("type", new json.JSONvalue("string"))
    props->addChild("object", new json.JSONvalue(dp))
    params->addChild("properties", new json.JSONvalue(props))
    req = new json.JSONvalue ptr[1]
    req[0] = new json.JSONvalue("object")
    params->addChild("required", new json.JSONvalue(new json.JSONarray(req, 1)))
    func->addChild("parameters", new json.JSONvalue(params))
    tool = new json.JSONobject
    tool->addChild("type", new json.JSONvalue("function"))
    tool->addChild("function", new json.JSONvalue(func))
    toolJson(nTools) = *tool: nTools += 1

    ' unlock
    func = new json.JSONobject
    func->addChild("name", new json.JSONvalue("unlock"))
    func->addChild("description", new json.JSONvalue("unlock an object with a code"))
    params = new json.JSONobject
    params->addChild("type", new json.JSONvalue("object"))
    props = new json.JSONobject
    dp = new json.JSONobject
    dp->addChild("type", new json.JSONvalue("string"))
    props->addChild("object", new json.JSONvalue(dp))
    dp = new json.JSONobject
    dp->addChild("type", new json.JSONvalue("string"))
    props->addChild("code", new json.JSONvalue(dp))
    params->addChild("properties", new json.JSONvalue(props))
    req = new json.JSONvalue ptr[2]
    req[0] = new json.JSONvalue("object")
    req[1] = new json.JSONvalue("code")
    params->addChild("required", new json.JSONvalue(new json.JSONarray(req, 2)))
    func->addChild("parameters", new json.JSONvalue(params))
    tool = new json.JSONobject
    tool->addChild("type", new json.JSONvalue("function"))
    tool->addChild("function", new json.JSONvalue(func))
    toolJson(nTools) = *tool: nTools += 1

    ' use
    func = new json.JSONobject
    func->addChild("name", new json.JSONvalue("use"))
    func->addChild("description", new json.JSONvalue("use an item on a target"))
    params = new json.JSONobject
    params->addChild("type", new json.JSONvalue("object"))
    props = new json.JSONobject
    dp = new json.JSONobject
    dp->addChild("type", new json.JSONvalue("string"))
    props->addChild("item", new json.JSONvalue(dp))
    dp = new json.JSONobject
    dp->addChild("type", new json.JSONvalue("string"))
    props->addChild("target", new json.JSONvalue(dp))
    params->addChild("properties", new json.JSONvalue(props))
    req = new json.JSONvalue ptr[2]
    req[0] = new json.JSONvalue("item")
    req[1] = new json.JSONvalue("target")
    params->addChild("required", new json.JSONvalue(new json.JSONarray(req, 2)))
    func->addChild("parameters", new json.JSONvalue(params))
    tool = new json.JSONobject
    tool->addChild("type", new json.JSONvalue("function"))
    tool->addChild("function", new json.JSONvalue(func))
    toolJson(nTools) = *tool: nTools += 1

    ' talk_to
    func = new json.JSONobject
    func->addChild("name", new json.JSONvalue("talk_to"))
    func->addChild("description", new json.JSONvalue("talk to an object or person"))
    params = new json.JSONobject
    params->addChild("type", new json.JSONvalue("object"))
    props = new json.JSONobject
    dp = new json.JSONobject
    dp->addChild("type", new json.JSONvalue("string"))
    props->addChild("object", new json.JSONvalue(dp))
    params->addChild("properties", new json.JSONvalue(props))
    req = new json.JSONvalue ptr[1]
    req[0] = new json.JSONvalue("object")
    params->addChild("required", new json.JSONvalue(new json.JSONarray(req, 1)))
    func->addChild("parameters", new json.JSONvalue(params))
    tool = new json.JSONobject
    tool->addChild("type", new json.JSONvalue("function"))
    tool->addChild("function", new json.JSONvalue(func))
    toolJson(nTools) = *tool: nTools += 1

    ' inventory
    func = new json.JSONobject
    func->addChild("name", new json.JSONvalue("inventory"))
    func->addChild("description", new json.JSONvalue("list carried items"))
    params = new json.JSONobject
    params->addChild("type", new json.JSONvalue("object"))
    params->addChild("properties", new json.JSONvalue(new json.JSONobject))
    func->addChild("parameters", new json.JSONvalue(params))
    tool = new json.JSONobject
    tool->addChild("type", new json.JSONvalue("function"))
    tool->addChild("function", new json.JSONvalue(func))
    toolJson(nTools) = *tool: nTools += 1
end sub

' ---------- sys content builder ----------
function buildSysContent() as string
    dim s as string
    s = "You are a parser for a text adventure. Translate the player's sentence into exactly ONE tool call from the available tools. "
    s += "Map synonyms and misspellings to the canonical object ids listed for the current room. Output only the tool call in the required format, no extra text." & chr(10)
    s += "If the sentence does not correspond to any tool, reply with a short clarification sentence and NO tool call." & chr(10)
    s += "IMPORTANT: For nonsense or unknown input, do not call any tool; write a short clarification sentence only." & chr(10)
    s += "Current room: " & roomName(curRoom) & ". Room objects: " & roomObjects(curRoom) & "." & chr(10)
    dim inv as string
    inv = ""
    if nInv = 0 then
        inv = "(empty)"
    else
        for i as integer = 0 to nInv-1
            if i > 0 then inv += ", "
            inv += inventory(i)
        next
    end if
    s += "Inventory: " & inv & "." & chr(10)
    s += "Example: input 'read the whiteboard' -> " & chr(10)
    s += "<tool_call>" & chr(10) & "<function=examine>" & chr(10) & "<parameter=object>" & chr(10) & "whiteboard" & chr(10) & "</parameter>" & chr(10) & "</function>" & chr(10) & "</tool_call>" & chr(10)
    s += "Example: input 'put the fish in the microwave' -> " & chr(10)
    s += "<tool_call>" & chr(10) & "<function=use>" & chr(10) & "<parameter=item>" & chr(10) & "red herring" & chr(10) & "</parameter>" & chr(10) & "<parameter=target>" & chr(10) & "microwave" & chr(10) & "</parameter>" & chr(10) & "</function>" & chr(10) & "</tool_call>" & chr(10)
    s += "Example: input 'use the fish on Gerald' -> " & chr(10)
    s += "<tool_call>" & chr(10) & "<function=use>" & chr(10) & "<parameter=item>" & chr(10) & "red herring" & chr(10) & "</parameter>" & chr(10) & "<parameter=target>" & chr(10) & "Gerald" & chr(10) & "</parameter>" & chr(10) & "</function>" & chr(10) & "</tool_call>" & chr(10)
    s += "Example: input 'xyzzy the plugh' -> The lab's parser pretends not to hear that."
    return s
end function

' ---------- main ----------
randomize timer
screenres SC_W, SC_H, 32
windowtitle "Locked In at Fumble Labs"

init_tools()
llm_init()

for ai as integer = 1 to 10
    if command(ai) = "--log" or command(ai) = "-log" then logging = 1
    if command(ai) = "--demo" then demoMode = 1
next
if logging = 1 then
    dim f0 as integer = freefile
    open "text_adventure.log" for output as #f0
    close #f0
end if

' intro
ui_log "Locked In at Fumble Labs"
ui_log roomDesc(0)
file_log "=== SESSION START ==="

dim demoStarted as integer = 0
dim matched as integer = 0

' demo script
dim demoCmds(0 to 19) as string
demoCmds(0) = "look"
demoCmds(1) = "examine the poster"
demoCmds(2) = "head west"
demoCmds(3) = "read the whiteboard"
demoCmds(4) = "open the fridge"
demoCmds(5) = "take the red herring"
demoCmds(6) = "examine the microwave"
demoCmds(7) = "go east"
demoCmds(8) = "go east"
demoCmds(9) = "grab the chiken"
demoCmds(10) = "talk to the fish"
demoCmds(11) = "go west"
demoCmds(12) = "go north"
demoCmds(13) = "examine the trofy"
demoCmds(14) = "unlock the safe with 1987"
demoCmds(15) = "take the hand"
demoCmds(16) = "inventory"
demoCmds(17) = "xyzzy the plugh"
demoCmds(18) = "go south"
demoCmds(19) = "put the hand on the scanner"

dim demoExp(0 to 19) as string
demoExp(0) = "look"
demoExp(1) = "examine(poster)"
demoExp(2) = "go(west)"
demoExp(3) = "examine(whiteboard)"
demoExp(4) = "open(fridge)"
demoExp(5) = "take(red herring)"
demoExp(6) = "examine(microwave)"
demoExp(7) = "go(east)"
demoExp(8) = "go(east)"
demoExp(9) = "take(rubber chicken)"
demoExp(10) = "talk_to(fish tank)"
demoExp(11) = "go(west)"
demoExp(12) = "go(north)"
demoExp(13) = "examine(trophy)"
demoExp(14) = "unlock(safe, 1987)"
demoExp(15) = "take(prosthetic hand)"
demoExp(16) = "inventory"
demoExp(17) = "clarification"
demoExp(18) = "go(south)"
demoExp(19) = "use(prosthetic hand, palm scanner)"

do
    ' escape handling
    dim k as string = inkey
    if k = chr(27) then
        exit do
    end if

    if demoMode = 1 and demoStarted = 0 then
        demoStarted = 1
        ' helper 3 sample conversation rendered into log
        dim hmsgs(0 to 4) as ChatMsg
        hmsgs(0).role = "system": hmsgs(0).content = "You are a parser.": hmsgs(0).hasTool = 0
        hmsgs(1).role = "user": hmsgs(1).content = "take the chicken": hmsgs(1).hasTool = 0
        hmsgs(2).role = "assistant": hmsgs(2).content = "": hmsgs(2).hasTool = 1: hmsgs(2).toolName = "take": hmsgs(2).toolArgs = "<parameter=object>" & chr(10) & "rubber chicken" & chr(10) & "</parameter>" & chr(10)
        hmsgs(3).role = "tool": hmsgs(3).content = "You take the rubber chicken.": hmsgs(3).hasTool = 0
        hmsgs(4).role = "user": hmsgs(4).content = "what now?": hmsgs(4).hasTool = 0
        dim hprompt as string = build_prompt(hmsgs(), 5, toolJson(), nTools)
        ui_log "--- helper3 sample render ---"
        ' split into lines
        dim cur as string = ""
        for i as integer = 1 to len(hprompt)
            dim ch as string = mid(hprompt, i, 1)
            if ch = chr(10) then
                ui_log cur
                cur = ""
            else
                cur += ch
            end if
        next
        if cur <> "" then ui_log cur
        ui_log "--- end helper3 ---"
        file_log("HELPER3_RENDER=" & hprompt)

        ' play script
        for di as integer = 0 to 19
            ' check escape
            if inkey = chr(27) then exit for
            ' type command
            inputBuf = ""
            dim cmd as string = demoCmds(di)
            for ci as integer = 1 to len(cmd)
                if inkey = chr(27) then exit for
                inputBuf += mid(cmd, ci, 1)
                draw_room()
                sleep 50
            next
            ' submit
            dim userInput as string = inputBuf
            inputBuf = ""
            ui_log "> " & userInput
            file_log("INPUT=" & userInput)

            ' generate
            dim sysC as string = buildSysContent()
            dim raw as string = generate_reply(sysC, userInput)
            file_log("RAW=" & raw)

            ' parse
            dim pc as ParsedCall = parse_tool_call(raw)
            dim parsedStr as string
            if pc.ok = 1 then
                parsedStr = pc.norm
            else
                parsedStr = "clarification"
            end if

            ' engine response
            dim resp as string
            if pc.ok = 1 then
                resp = handleIntent(pc)
            else
                resp = "The lab's parser pretends not to hear that."
            end if
            file_log("PARSED=" & parsedStr)
            ' print response lines
            dim rcur as string = ""
            for i as integer = 1 to len(resp)
                dim rch as string = mid(resp, i, 1)
                if rch = chr(10) then
                    ui_log rcur
                    rcur = ""
                else
                    rcur += rch
                end if
            next
            if rcur <> "" then ui_log rcur
            file_log("ENGINE=" & resp)

            ' demo match
            dim expected as string = demoExp(di)
            dim status as string
            if expected = "clarification" then
                if pc.ok = 0 then status = "MATCH" else status = "MISS"
            else
                if pc.ok = 1 and pc.norm = expected then status = "MATCH" else status = "MISS"
            end if
            if status = "MATCH" then matched += 1
            file_log("EXPECTED=" & expected & " PARSED=" & parsedStr & " " & status)
            ui_log "  (expected " & expected & ") " & status

            draw_room()
            ' pause ~2.5s
            for pi as integer = 0 to 50
                if inkey = chr(27) then exit for
                sleep 50
            next
        next

        ' ending card hold
        ui_log "intents=" & str(matched) & "/20"
        if won = 1 then ui_log "won=yes" else ui_log "won=no"
        file_log("intents=" & str(matched) & "/20")
        if won = 1 then file_log("won=yes") else file_log("won=no")
        draw_room()
        sleep 4000
        exit do
    end if

    ' interactive mode
    if k = chr(13) then
        if inputBuf <> "" then
            dim userInput as string = inputBuf
            inputBuf = ""
            ui_log "> " & userInput
            file_log("INPUT=" & userInput)

            if lcase(trim(userInput)) = "quit" then
                exit do
            end if

            dim sysC as string = buildSysContent()
            dim raw as string = generate_reply(sysC, userInput)
            file_log("RAW=" & raw)

            dim pc as ParsedCall = parse_tool_call(raw)
            dim parsedStr as string
            if pc.ok = 1 then parsedStr = pc.norm else parsedStr = "clarification"
            file_log("PARSED=" & parsedStr)

            dim resp as string
            if pc.ok = 1 then
                resp = handleIntent(pc)
            else
                resp = "The lab's parser pretends not to hear that."
            end if
            dim rcur as string = ""
            for i as integer = 1 to len(resp)
                dim rch as string = mid(resp, i, 1)
                if rch = chr(10) then
                    ui_log rcur
                    rcur = ""
                else
                    rcur += rch
                end if
            next
            if rcur <> "" then ui_log rcur
            file_log("ENGINE=" & resp)
        end if
    elseif k = chr(8) then
        if len(inputBuf) > 0 then inputBuf = left(inputBuf, len(inputBuf)-1)
    elseif len(k) = 1 and asc(k) >= 32 and asc(k) <= 126 then
        inputBuf += k
    end if

    draw_room()
    sleep 16
loop

file_log "=== SESSION END ==="

llama_sampler_free(g_smp)
llama_free(g_ctx)
llama_model_free(g_model)
end

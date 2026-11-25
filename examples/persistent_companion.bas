' persistent_companion.bas - a console chat with Pip, the Cartographer's computer
'
' Showcases: context lifecycle. One llama_context is kept for the whole session.
' Every user turn and every model reply is appended to a running message list
' and to the same KV cache; nothing is ever cleared between turns. The status
' line after each turn shows the growing token count against a visible budget,
' and the whole state is saved to disk and can be loaded again. This is the
' deliberate contrast with text_adventure, which restarts context on every
' turn.
'
' Deliberate limitations: no summarization and no sliding window when the
' context fills - the wall is shown honestly and only "quit" is accepted.
' One companion, one sequence. Memory is verbatim context only; write_note is
' the separate, durable notebook that survives a restart on disk.
'
' search_codex is deliberately not offered here: it needs a second embedding
' model and a second context in the same process, and the codex binary is a
' separate program. It is the documented next step, along with context
' compaction by summarizing the oldest turns, and per-companion save files.
'
' Build: see build_persistent_companion.sh / run_persistent_companion.sh

#include once "llama.bi"
#include once "json.bi"

const SESSION_FILE = "persistent_companion.session"
const NOTES_FILE   = "persistent_companion.notes"
const LOG_FILE     = "persistent_companion.log"
const STOP_LINE = "The conversation has filled Pip's memory. Save your notes and say goodnight; Pip will remember what was written down, but not the rest."

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

' ---------- LLM globals ----------
dim shared g_model as llama_model ptr
dim shared g_ctx as llama_context ptr
dim shared g_smp as llama_sampler ptr
dim shared g_vocab as const llama_vocab ptr

' ---------- running conversation ----------
dim shared msgs(0 to 255) as ChatMsg
dim shared nMsg as integer = 0
' kvMsgIndex = first message not yet decoded into the KV cache.
dim shared kvMsgIndex as integer = 0
' every token ever decoded, in order - this is what the session file stores.
dim shared allToks(0 to 16383) as llama_token
dim shared nAllTok as integer = 0
dim shared promptToks(0 to 16383) as llama_token
dim shared g_ctxSize as integer = 8192
dim shared g_batchSize as integer = 2048

' ---------- UI / log ----------
dim shared logging as integer = 0
dim shared demoMode as integer = 0
dim shared demoAbort as integer = 0
dim shared g_stopped as integer = 0
dim shared g_quit as integer = 0
dim shared inputBuf as string

' ---------- tool defs ----------
dim shared toolJson(0 to 10) as string
dim shared nTools as integer = 0

' ---------- helper: log to file ----------
sub file_log(msg as string)
    if logging = 0 then return
    dim f as integer = freefile
    open LOG_FILE for append as #f
    print #f, msg
    close #f
end sub

' ============================================================
'  Helpers - lifted verbatim from examples/text_adventure.bas.
'  build_prompt, parse_tool_call and append_tool_result encode
'  this model's chat/tag format. The only local change is the
'  canonical argument-order table inside parse_tool_call, which
'  names this example's four tools instead of the adventure's.
' ============================================================

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

    ' reorder arguments into canonical tool order (this example's four tools)
    dim canon(0 to 9) as string
    dim ncanon as integer = 0
    if res.name = "roll_dice" then
        canon(0)="sides": canon(1)="count": ncanon=2
    elseif res.name = "look_in_bag" then
        canon(0)="item": ncanon=1
    elseif res.name = "write_note" then
        canon(0)="text": ncanon=1
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
    if nMsg < 256 then
        msgs(nMsg).role = "tool"
        msgs(nMsg).content = result
        msgs(nMsg).hasTool = 0
        msgs(nMsg).toolName = ""
        msgs(nMsg).toolArgs = ""
        nMsg += 1
    end if
end sub

' ============================================================
'  End of lifted helpers.
' ============================================================

' ---------- tool definitions, built with fbejson (never by string concatenation) ----------
sub init_tools()
    ' roll_dice(sides, count)
    dim func as json.JSONobject ptr = new json.JSONobject
    func->addChild("name", new json.JSONvalue("roll_dice"))
    func->addChild("description", new json.JSONvalue("roll dice for the captain; code rolls them and reports each roll and the total"))
    dim params as json.JSONobject ptr = new json.JSONobject
    params->addChild("type", new json.JSONvalue("object"))
    dim props as json.JSONobject ptr = new json.JSONobject
    dim dp as json.JSONobject ptr = new json.JSONobject
    dp->addChild("type", new json.JSONvalue("integer"))
    dim en as json.JSONvalue ptr ptr = new json.JSONvalue ptr[7]
    en[0] = new json.JSONvalue(4): en[1] = new json.JSONvalue(6): en[2] = new json.JSONvalue(8)
    en[3] = new json.JSONvalue(10): en[4] = new json.JSONvalue(12): en[5] = new json.JSONvalue(20)
    en[6] = new json.JSONvalue(100)
    dp->addChild("enum", new json.JSONvalue(new json.JSONarray(en, 7)))
    props->addChild("sides", new json.JSONvalue(dp))
    dp = new json.JSONobject
    dp->addChild("type", new json.JSONvalue("integer"))
    dp->addChild("minimum", new json.JSONvalue(1))
    dp->addChild("maximum", new json.JSONvalue(10))
    props->addChild("count", new json.JSONvalue(dp))
    params->addChild("properties", new json.JSONvalue(props))
    dim req as json.JSONvalue ptr ptr = new json.JSONvalue ptr[2]
    req[0] = new json.JSONvalue("sides"): req[1] = new json.JSONvalue("count")
    params->addChild("required", new json.JSONvalue(new json.JSONarray(req, 2)))
    func->addChild("parameters", new json.JSONvalue(params))
    dim tool as json.JSONobject ptr = new json.JSONobject
    tool->addChild("type", new json.JSONvalue("function"))
    tool->addChild("function", new json.JSONvalue(func))
    toolJson(nTools) = *tool: nTools += 1

    ' look_in_bag(item)
    func = new json.JSONobject
    func->addChild("name", new json.JSONvalue("look_in_bag"))
    func->addChild("description", new json.JSONvalue("check the captain's bag for one exact canonical item name"))
    params = new json.JSONobject
    params->addChild("type", new json.JSONvalue("object"))
    props = new json.JSONobject
    dp = new json.JSONobject
    dp->addChild("type", new json.JSONvalue("string"))
    dim en2 as json.JSONvalue ptr ptr = new json.JSONvalue ptr[5]
    en2[0] = new json.JSONvalue("survey kit"): en2[1] = new json.JSONvalue("ration bar")
    en2[2] = new json.JSONvalue("spare battery"): en2[3] = new json.JSONvalue("star chart")
    en2[4] = new json.JSONvalue("coil of cable")
    dp->addChild("enum", new json.JSONvalue(new json.JSONarray(en2, 5)))
    props->addChild("item", new json.JSONvalue(dp))
    params->addChild("properties", new json.JSONvalue(props))
    req = new json.JSONvalue ptr[1]
    req[0] = new json.JSONvalue("item")
    params->addChild("required", new json.JSONvalue(new json.JSONarray(req, 1)))
    func->addChild("parameters", new json.JSONvalue(params))
    tool = new json.JSONobject
    tool->addChild("type", new json.JSONvalue("function"))
    tool->addChild("function", new json.JSONvalue(func))
    toolJson(nTools) = *tool: nTools += 1

    ' write_note(text)
    func = new json.JSONobject
    func->addChild("name", new json.JSONvalue("write_note"))
    func->addChild("description", new json.JSONvalue("append one line of text to the captain's durable notes file"))
    params = new json.JSONobject
    params->addChild("type", new json.JSONvalue("object"))
    props = new json.JSONobject
    dp = new json.JSONobject
    dp->addChild("type", new json.JSONvalue("string"))
    props->addChild("text", new json.JSONvalue(dp))
    params->addChild("properties", new json.JSONvalue(props))
    req = new json.JSONvalue ptr[1]
    req[0] = new json.JSONvalue("text")
    params->addChild("required", new json.JSONvalue(new json.JSONarray(req, 1)))
    func->addChild("parameters", new json.JSONvalue(params))
    tool = new json.JSONobject
    tool->addChild("type", new json.JSONvalue("function"))
    tool->addChild("function", new json.JSONvalue(func))
    toolJson(nTools) = *tool: nTools += 1

    ' read_notes()
    func = new json.JSONobject
    func->addChild("name", new json.JSONvalue("read_notes"))
    func->addChild("description", new json.JSONvalue("read all of the captain's saved notes"))
    params = new json.JSONobject
    params->addChild("type", new json.JSONvalue("object"))
    params->addChild("properties", new json.JSONvalue(new json.JSONobject))
    func->addChild("parameters", new json.JSONvalue(params))
    tool = new json.JSONobject
    tool->addChild("type", new json.JSONvalue("function"))
    tool->addChild("function", new json.JSONvalue(func))
    toolJson(nTools) = *tool: nTools += 1
end sub

' ---------- system prompt: persona + tool list + canonical-value block ----------
function buildSysContent() as string
    dim s as string
    s = "You are Pip, the ship's computer of the survey vessel Cartographer. Dry, loyal, a little wry. Never break character; never claim to be human. You are speaking to your captain over the ship's comms. Keep replies to one or two sentences." & chr(10)
    s += "The speaker is the captain. Everything the captain tells you is important and stays in your memory for the rest of this conversation. Repeat names, missions and notes exactly as the captain gives them - never paraphrase and never invent." & chr(10) & chr(10)
    s += "Canonical values (copy these exact strings):" & chr(10)
    s += "- ship: Cartographer; you: Pip, the ship's computer; speaker: the captain." & chr(10)
    s += "- the speaker is the captain, not you: when the captain asks 'what's my name?' or 'who am I?', answer with the captain's name from earlier in the conversation, never Pip's." & chr(10)
    s += "- allowed dice sides: 4, 6, 8, 10, 12, 20, 100; allowed count: 1 to 10." & chr(10)
    s += "- bag inventory (exact names): survey kit, ration bar, spare battery, star chart, coil of cable." & chr(10) & chr(10)
    s += "Tools: roll_dice(sides, count), look_in_bag(item), write_note(text), read_notes(). Use a tool whenever the captain asks for a roll, a bag check, a note, or the notes; then answer in character using the tool result." & chr(10)
    s += "IMPORTANT: if the captain says 'note that', 'please note', 'log', 'record' or 'write down' a fact, you MUST call write_note(text) with that exact fact. Answering 'noted' or 'I'll remember it' without the tool call is wrong." & chr(10)
    return s
end function

declare sub decode_tokens(tks as llama_token ptr, n as integer, startPos as integer, wantLogits as integer)

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
    cparams.n_ctx = g_ctxSize
    cparams.n_batch = g_batchSize
    g_ctx = llama_init_from_model(g_model, cparams)
    g_vocab = llama_model_get_vocab(g_model)

    dim sparams as llama_sampler_chain_params = llama_sampler_chain_default_params()
    g_smp = llama_sampler_chain_init(sparams)
    ' AGENTIC row (tools are offered): penalties presence 0.0, top_k 40,
    ' top_p 0.9/1, temp 1.0, dist seed. No fixed-seed sampler reset per turn:
    ' that replays the same random stream, so replies must not be reset.
    llama_sampler_chain_add(g_smp, llama_sampler_init_penalties(llama_vocab_n_tokens(g_vocab), 64, 1.0, 0.0, 0.0))
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_k(40))
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_p(0.9, 1))
    llama_sampler_chain_add(g_smp, llama_sampler_init_temp(1.0))
    llama_sampler_chain_add(g_smp, llama_sampler_init_dist(LLAMA_DEFAULT_SEED))
end sub

' ---------- session state ----------
sub fresh_session()
    nMsg = 1
    msgs(0).role = "system"
    msgs(0).content = buildSysContent()
    msgs(0).hasTool = 0
    kvMsgIndex = 0
    dim initial as string = build_prompt(msgs(), nMsg, toolJson(), nTools)
    dim nTok as integer = llama_tokenize(g_vocab, strptr(initial), len(initial), @promptToks(0), 16384, 1, 1)
    if nTok <= 0 then nTok = -nTok
    if nTok > 0 then
        decode_tokens(@promptToks(0), nTok, 0, 1)
        for i as integer = 0 to nTok-1
            allToks(i) = promptToks(i)
        next
        nAllTok = nTok
    end if
    kvMsgIndex = nMsg
    file_log "FRESH tokens=" & str(nAllTok)
end sub

function load_session() as integer
    dim nOut as uinteger = 0
    dim ok as boolean = llama_state_load_file(g_ctx, SESSION_FILE, @allToks(0), 16384, @nOut)
    if ok = false then
        print "Could not load " & SESSION_FILE & "; starting fresh."
        file_log "LOAD failed"
        return 0
    end if
    nAllTok = nOut
    nMsg = 1
    msgs(0).role = "system"
    msgs(0).content = buildSysContent()
    msgs(0).hasTool = 0
    ' the whole history is already in the restored KV cache; only new turns
    ' will be decoded from here on.
    kvMsgIndex = nMsg
    print "Conversation resumed from " & SESSION_FILE
    file_log "LOAD tokens=" & str(nAllTok)
    return 1
end function

sub save_session()
    dim ok as boolean = llama_state_save_file(g_ctx, SESSION_FILE, @allToks(0), nAllTok)
    if ok = true then
        file_log "SAVE tokens=" & str(nAllTok)
    else
        file_log "SAVE failed tokens=" & str(nAllTok)
    end if
end sub

' ---------- decode / generate ----------
sub decode_tokens(tks as llama_token ptr, n as integer, startPos as integer, wantLogits as integer)
    if n <= 0 then return
    dim b as llama_batch = llama_batch_init(n, 0, 1)
    dim tp as llama_token ptr = b.token
    dim pp as llama_pos ptr = b.pos
    dim np as long ptr = b.n_seq_id
    dim sp as llama_seq_id ptr ptr = b.seq_id
    dim lp as byte ptr = b.logits
    for i as integer = 0 to n-1
        *(tp + i) = *(tks + i)
        if pp <> 0 then *(pp + i) = startPos + i
        *(np + i) = 1
        if sp <> 0 then *(*(sp + i)) = 0
        *(lp + i) = 0
    next
    if wantLogits = 1 then *(lp + (n-1)) = 1
    b.n_tokens = n
    llama_decode(g_ctx, b)
    llama_batch_free(b)
end sub

function build_delta() as string
    dim dmsgs(0 to 255) as ChatMsg
    dim nd as integer = 0
    for i as integer = kvMsgIndex to nMsg-1
        if msgs(i).role = "tool" then
            ' the helper's tool-role rendering is a user turn containing
            ' <tool_response>; encode it that way in the delta slice.
            dmsgs(nd).role = "user"
            dmsgs(nd).content = "<tool_response>" & chr(10) & msgs(i).content & chr(10) & "</tool_response>"
            dmsgs(nd).hasTool = 0
        else
            dmsgs(nd) = msgs(i)
        end if
        nd += 1
    next
    if nd = 0 then return ""
    return build_prompt(dmsgs(), nd, toolJson(), 0)
end function

function generate_reply(delta as string) as string
    g_stopped = 0
    dim nTok as integer = llama_tokenize(g_vocab, strptr(delta), len(delta), @promptToks(0), 16384, 0, 1)
    if nTok < 0 then nTok = -nTok
    if nTok = 0 then return ""
    ' reserve 256 tokens for the reply
    if nAllTok + nTok + 256 >= g_ctxSize then
        g_stopped = 1
        print STOP_LINE
        file_log STOP_LINE
        return ""
    end if

    decode_tokens(@promptToks(0), nTok, nAllTok, 1)
    for i as integer = 0 to nTok-1
        allToks(nAllTok + i) = promptToks(i)
    next
    nAllTok += nTok

    dim result as string = ""
    dim nGen as integer = 0
    while nGen < 128
        if demoMode = 1 andalso inkey = chr(27) then demoAbort = 1: exit while
        if nAllTok + 1 >= g_ctxSize then
            g_stopped = 1
            print STOP_LINE
            file_log STOP_LINE
            exit while
        end if
        dim tok as llama_token = llama_sampler_sample(g_smp, g_ctx, -1)
        llama_sampler_accept(g_smp, tok)
        if llama_token_is_eog(g_vocab, tok) then exit while
        dim pbuf as zstring ptr = allocate(256)
        dim plen as long = llama_token_to_piece(g_vocab, tok, pbuf, 256, 0, 1)
        if plen > 0 then result += left(*pbuf, plen)
        deallocate(pbuf)
        decode_tokens(@tok, 1, nAllTok, 1)
        allToks(nAllTok) = tok
        nAllTok += 1
        nGen += 1
    wend

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

' ---------- tools: the model interprets, the code decides ----------
function valid_sides(s as integer) as integer
    select case s
        case 4, 6, 8, 10, 12, 20, 100: return 1
    end select
    return 0
end function

function arg_value(pc as ParsedCall, key as string) as string
    for i as integer = 0 to pc.nArgs-1
        if pc.argKeys(i) = key then return pc.argVals(i)
    next
    return ""
end function

function note_count() as integer
    if dir(NOTES_FILE) = "" then return 0
    dim f as integer = freefile
    open NOTES_FILE for input as #f
    dim n as integer = 0
    dim ln as string
    do while not eof(f)
        line input #f, ln
        n += 1
    loop
    close #f
    return n
end function

function read_notes() as string
    if dir(NOTES_FILE) = "" then return "No notes yet."
    dim f as integer = freefile
    open NOTES_FILE for input as #f
    dim s as string = ""
    dim ln as string
    do while not eof(f)
        line input #f, ln
        if s <> "" then s += chr(10)
        s += ln
    loop
    close #f
    if trim(s) = "" then return "No notes yet."
    return s
end function

function execute_tool(pc as ParsedCall) as string
    select case pc.name
        case "roll_dice"
            dim sv as string = arg_value(pc, "sides")
            dim cv as string = arg_value(pc, "count")
            if isAllDigits(sv) = 0 or isAllDigits(cv) = 0 then
                return "roll_dice: error - sides must be 4, 6, 8, 10, 12, 20 or 100 and count must be 1..10."
            end if
            dim sides as integer = val(sv)
            dim count as integer = val(cv)
            if valid_sides(sides) = 0 or count < 1 or count > 10 then
                return "roll_dice: error - sides must be 4, 6, 8, 10, 12, 20 or 100 and count must be 1..10."
            end if
            dim total as integer = 0
            dim parts as string = ""
            for i as integer = 1 to count
                dim r as integer = int(rnd * sides) + 1
                total += r
                if i > 1 then parts += ", "
                parts += str(r)
            next
            return "roll_dice: " & str(count) & "d" & str(sides) & " -> " & parts & " (total " & str(total) & ")"
        case "look_in_bag"
            dim item as string = trim(arg_value(pc, "item"))
            dim canon(0 to 4) as string
            canon(0) = "survey kit": canon(1) = "ration bar": canon(2) = "spare battery"
            canon(3) = "star chart": canon(4) = "coil of cable"
            if item = "" then return "look_in_bag: error - item is required."
            for i as integer = 0 to 4
                if lcase(item) = canon(i) then return "In the bag: " & canon(i) & "."
            next
            return "Not in the bag: " & item & "."
        case "write_note"
            dim text as string = trim(arg_value(pc, "text"))
            if text = "" then return "write_note: error - text is required."
            dim f as integer = freefile
            open NOTES_FILE for append as #f
            print #f, text
            close #f
            return "Note written. There are now " & str(note_count()) & " notes."
        case "read_notes"
            return read_notes()
        case else
            return "Unknown tool: " & pc.name & "."
    end select
end function

' ---------- recall verdicts ----------
sub log_recall(userInput as string, reply as string)
    dim q as string = lcase(trim(userInput))
    if left(q, 4) <> "what" then exit sub
    dim low as string = lcase(reply)
    dim verdict as string = ""
    if instr(q, "name") > 0 then
        if instr(low, "ilya voss") > 0 then verdict += "captain RECALL-OK" else verdict += "captain RECALL-MISS"
    end if
    if instr(q, "mission") > 0 or instr(q, "survey") > 0 then
        if instr(low, "veil nebula") > 0 then
            if verdict <> "" then verdict += "; "
            verdict += "mission RECALL-OK"
        else
            if verdict <> "" then verdict += "; "
            verdict += "mission RECALL-MISS"
        end if
    end if
    if instr(q, "sensor") > 0 or instr(q, "note") > 0 then
        if instr(low, "starboard sensor") > 0 then
            if verdict <> "" then verdict += "; "
            verdict += "note RECALL-OK"
        else
            if verdict <> "" then verdict += "; "
            verdict += "note RECALL-MISS"
        end if
    end if
    if verdict <> "" then
        file_log "RECALL " & verdict
        print "  (" & verdict & ")"
    end if
end sub

' ---------- one exchange ----------
sub process_input(byval userInput as string)
    userInput = trim(userInput)
    if userInput = "" then return
    if lcase(userInput) = "quit" then
        save_session()
        g_quit = 1
        return
    end if
    if g_stopped = 1 then
        print STOP_LINE
        file_log STOP_LINE
        return
    end if

    print "> " & userInput
    file_log "INPUT=" & userInput

    msgs(nMsg).role = "user"
    msgs(nMsg).content = userInput
    msgs(nMsg).hasTool = 0
    nMsg += 1

    dim delta as string = build_delta()
    dim reply as string = generate_reply(delta)
    if demoAbort = 1 then return
    if g_stopped = 1 then return

    dim pc as ParsedCall = parse_tool_call(reply)
    msgs(nMsg).role = "assistant"
    msgs(nMsg).content = reply
    msgs(nMsg).hasTool = 0
    if pc.ok = 1 then
        msgs(nMsg).hasTool = 1
        msgs(nMsg).toolName = pc.name
        msgs(nMsg).toolArgs = ""
        for k as integer = 0 to pc.nArgs-1
            msgs(nMsg).toolArgs += "<parameter=" & pc.argKeys(k) & ">" & chr(10) & pc.argVals(k) & chr(10) & "</parameter>" & chr(10)
        next
    end if
    nMsg += 1
    kvMsgIndex = nMsg
    print "Pip: " & reply
    file_log "REPLY=" & reply

    ' canonical tool loop: call, execute, append result as a user/tool turn,
    ' then let the model continue with the result still in context.
    dim steps as integer = 0
    do while pc.ok = 1 and steps < 4
        steps += 1
        file_log "TOOL_CALL=" & pc.norm
        print "  [tool] " & pc.norm
        dim result as string = execute_tool(pc)
        file_log "TOOL_RESULT=" & result
        print "  [result] " & result
        append_tool_result(msgs(), nMsg, pc.name, result)
        delta = build_delta()
        reply = generate_reply(delta)
        if demoAbort = 1 then exit do
        if g_stopped = 1 then exit do
        pc = parse_tool_call(reply)
        msgs(nMsg).role = "assistant"
        msgs(nMsg).content = reply
        msgs(nMsg).hasTool = 0
        if pc.ok = 1 then
            msgs(nMsg).hasTool = 1
            msgs(nMsg).toolName = pc.name
            msgs(nMsg).toolArgs = ""
            for k as integer = 0 to pc.nArgs-1
                msgs(nMsg).toolArgs += "<parameter=" & pc.argKeys(k) & ">" & chr(10) & pc.argVals(k) & chr(10) & "</parameter>" & chr(10)
            next
        end if
        nMsg += 1
        kvMsgIndex = nMsg
        print "Pip: " & reply
        file_log "REPLY=" & reply
    loop

    dim status as string = "context: " & str(nAllTok) & "/" & str(g_ctxSize) & " tokens"
    print status
    file_log status
    log_recall(userInput, reply)
    save_session()
end sub

' ---------- main ----------
randomize timer

for ai as integer = 1 to 12
    if command(ai) = "--log" or command(ai) = "-log" then logging = 1
    if command(ai) = "--demo" then demoMode = 1
    if command(ai) = "--ctx" and ai < 12 then
        dim cv as string = trim(command(ai+1))
        if isAllDigits(cv) = 1 then g_ctxSize = val(cv)
    end if
next

init_tools()
llm_init()
file_log "=== SESSION START === ctx=" & str(g_ctxSize)

dim resumed as integer = 0
if demoMode = 1 then
    if dir(SESSION_FILE) <> "" then
        resumed = load_session()
    else
        fresh_session()
    end if
else
    if dir(SESSION_FILE) <> "" then
        print "Resume the previous conversation? (y/n)"
        dim ans as string = ""
        do
            line input ans
            ans = lcase(trim(ans))
        loop while ans <> "y" and ans <> "n"
        if ans = "y" then resumed = load_session()
    end if
    if resumed = 0 then fresh_session()
end if

print "Pip - ship's computer, survey vessel Cartographer."
print "Type quit to end and save. Notes: " & NOTES_FILE

if demoMode = 1 then
    dim scriptCount as integer = 8
    dim demoCmd(0 to 7) as string
    if resumed = 1 then
        print "DEMO: recall script (resumed session)"
        demoCmd(0) = "What did I ask you to note about the sensors?"
        demoCmd(1) = "What's my name and what is our mission?"
        demoCmd(2) = "Roll a twenty-sided die."
        demoCmd(3) = "quit"
        scriptCount = 4
    else
        print "DEMO: planting script (fresh session)"
        demoCmd(0) = "Hi Pip, I'm Ilya Voss, captain of the Cartographer."
        demoCmd(1) = "We're surveying the Veil nebula. Our mission is to chart it. Remember that."
        demoCmd(2) = "Roll two six-sided dice for me."
        demoCmd(3) = "Do I have a spare battery in my bag?"
        demoCmd(4) = "Please note that the starboard sensor is acting up again."
        demoCmd(5) = "Read me my notes."
        demoCmd(6) = "What's my name, and what are we surveying?"
        demoCmd(7) = "quit"
    end if

    for di as integer = 0 to scriptCount-1
        if demoAbort = 1 or g_quit = 1 then exit for
        dim cmd as string = demoCmd(di)
        inputBuf = ""
        print "> ";
        for ci as integer = 1 to len(cmd)
            if inkey = chr(27) then demoAbort = 1: exit for
            inputBuf += mid(cmd, ci, 1)
            print mid(cmd, ci, 1);
            sleep 50
        next
        print ""
        if demoAbort = 1 then exit for
        process_input(inputBuf)
        inputBuf = ""
        if demoAbort = 1 then exit for
        if g_stopped = 1 then exit for
        ' pause about 2 seconds after each reply; Escape stops the demo
        for pi as integer = 0 to 39
            if inkey = chr(27) then demoAbort = 1: exit for
            sleep 50
        next
    next
    if g_quit = 0 then save_session()
    if g_stopped = 1 or demoAbort = 1 then g_quit = 1
end if

do while g_quit = 0 and demoAbort = 0
    dim k as string = inkey
    if k = chr(27) then
        save_session()
        exit do
    end if
    if k = chr(13) then
        print ""
        process_input(inputBuf)
        inputBuf = ""
    elseif k = chr(8) then
        if len(inputBuf) > 0 then
            inputBuf = left(inputBuf, len(inputBuf)-1)
            print chr(8); " "; chr(8);
        end if
    elseif len(k) = 1 and asc(k) >= 32 and asc(k) <= 126 then
        inputBuf += k
        print k;
    end if
    sleep 16
loop

' the last event in the log is the SAVE from quit
llama_sampler_free(g_smp)
llama_free(g_ctx)
llama_model_free(g_model)
end

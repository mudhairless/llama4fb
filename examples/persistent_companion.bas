' persistent_companion.bas - Pip, the Cartographer's computer, in a graphics window
'
' Showcases: context lifecycle. One llama_context is kept for the whole session.
' Every user turn and every model reply is appended to a running message list
' and to the same KV cache; nothing is ever cleared between turns. The status
' line after each turn shows the growing token count against a visible budget,
' and the whole state is saved to disk and can be loaded again. This is the
' deliberate contrast with text_adventure, which restarts context on every
' turn. The conversation is now presented in an fbgfx window: a ship readout
' on the left, a context gauge between the panels, Pip's face on the right, a
' wrapped log in the middle and a three-line input box at the bottom - the
' presentation model of text_adventure, adapted for a chat.
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

#include once "fbgfx.bi"
#include once "llama.bi"
#include once "json.bi"

const SC_W = 800
const SC_H = 600
const LOG_WRAP = 96
const LOG_RING = 64
const LOG_VISIBLE = 26
const LOG_TOP = 176
const LOG_BOTTOM = 500
const INPUT_TOP = 506
const INPUT_WRAP = 96
const INPUT_LINES = 3
const CTX_BAR_X = 484
const CTX_BAR_W = 15
const CTX_BAR_TOP = 10
const CTX_LED_COUNT = 18
const CTX_LED_H = 6
const CTX_LED_GAP = 3

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
dim shared logLines(0 to 63) as string
dim shared nLog as integer = 0
dim shared logging as integer = 0
dim shared demoMode as integer = 0
dim shared demoAbort as integer = 0
dim shared g_stopped as integer = 0
dim shared g_quit as integer = 0
dim shared inputBuf as string
' 0 idle, 1 generating, 2 tool blink
dim shared g_pipState as integer = 0
dim shared g_toolBlinkUntil as double = 0
dim shared g_cursorOn as integer = 1

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

' ---------- window log: wrap on add, tail-anchored when painted ----------
sub ui_log_append_one(msg as string)
    dim rest as string = msg
    do while len(rest) > LOG_WRAP
        dim chunk as string = left(rest, LOG_WRAP)
        dim sp as integer = instrrev(chunk, " ")
        if sp < 1 then sp = LOG_WRAP
        dim part as string = left(rest, sp)
        if nLog < LOG_RING then
            logLines(nLog) = part
            nLog += 1
        else
            for i as integer = 0 to LOG_RING-2
                logLines(i) = logLines(i+1)
            next
            logLines(LOG_RING-1) = part
        end if
        rest = trim(mid(rest, sp + 1))
    loop
    if rest <> "" then
        if nLog < LOG_RING then
            logLines(nLog) = rest
            nLog += 1
        else
            for i as integer = 0 to LOG_RING-2
                logLines(i) = logLines(i+1)
            next
            logLines(LOG_RING-1) = rest
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

' Only finished, spoken turns reach the window log. Raw tool calls and
' reasoning stay in --log; the file_log calls are unchanged.
sub ui_pip_reply(reply as string)
    if instr(reply, "<tool_call>") > 0 then return
    dim shown as string = reply
    do while instr(shown, "<think>") > 0
        dim s as integer = instr(shown, "<think>")
        dim e as integer = instr(s, shown, "</think>")
        if e = 0 then exit do
        shown = left(shown, s-1) & mid(shown, e + len("</think>"))
    loop
    shown = trim(shown)
    if shown <> "" then ui_log shown
end sub

' ---------- input box: word-wrap with "> " on the first line ----------
' returns the line count; counts past 3 so the caller can reject a character.
function wrap_input(buf as string, outLines() as string) as integer
    dim rest as string = buf
    dim prefix as string = "> "
    dim count as integer = 0
    do
        dim room as integer = INPUT_WRAP - len(prefix)
        if room < 1 then room = 1
        dim chunk as string = left(rest, room)
        dim take as integer = len(chunk)
        if len(rest) > room then
            dim sp as integer = instrrev(chunk, " ")
            if sp > 1 then take = sp
        end if
        if count <= INPUT_LINES then outLines(count) = prefix & rtrim(mid(rest, 1, take))
        count += 1
        rest = trim(mid(rest, take + 1))
        if rest = "" then exit do
        if count > INPUT_LINES then exit do
        prefix = ""
    loop
    return count
end function

function input_fits(buf as string) as integer
    dim probe(0 to INPUT_LINES) as string
    if wrap_input(buf, probe()) <= INPUT_LINES then return 1
    return 0
end function

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

' ============================================================
'  Window presentation. Primitives only; no image assets.
' ============================================================

' upper half-circle from line segments, so the dome always reads as a dome
sub draw_dome(cx as integer, cy as integer, r as integer, col as uinteger)
    dim px as integer = cx + r
    dim py as integer = cy
    for a as double = 0.18 to 3.14159 step 0.18
        dim nx as integer = cx + r * cos(a)
        dim ny as integer = cy - r * sin(a)
        line (px, py)-(nx, ny), col
        px = nx
        py = ny
    next
end sub

' vertical context gauge: LED chunks green at the bottom, red at the top.
' The fill grows upward, so the topmost LED lights only at a full context.
sub draw_ctx_bar()
    dim frameBottom as integer = CTX_BAR_TOP + CTX_LED_COUNT * (CTX_LED_H + CTX_LED_GAP)
    line (CTX_BAR_X-2, CTX_BAR_TOP-2)-(CTX_BAR_X+CTX_BAR_W+1, frameBottom), rgb(18, 22, 34), bf
    line (CTX_BAR_X-2, CTX_BAR_TOP-2)-(CTX_BAR_X+CTX_BAR_W+1, frameBottom), rgb(70, 80, 100), b

    dim ratio as double = 0
    if g_ctxSize > 0 then ratio = nAllTok / g_ctxSize
    if ratio < 0 then ratio = 0
    if ratio > 1 then ratio = 1
    dim lit as integer = int(ratio * CTX_LED_COUNT + 0.9999)

    for i as integer = 0 to CTX_LED_COUNT-1
        dim f as double = i / (CTX_LED_COUNT-1)
        dim lr as integer = int(60 + 170*f)
        dim lg as integer = int(220 - 170*f)
        dim lb as integer = int(90 - 50*f)
        dim y as integer = CTX_BAR_TOP + (CTX_LED_COUNT-1-i) * (CTX_LED_H + CTX_LED_GAP)
        if i < lit then
            line (CTX_BAR_X, y)-(CTX_BAR_X+CTX_BAR_W-1, y+CTX_LED_H-1), rgb(lr, lg, lb), bf
        else
            line (CTX_BAR_X, y)-(CTX_BAR_X+CTX_BAR_W-1, y+CTX_LED_H-1), rgb(int(lr*0.22), int(lg*0.22), int(lb*0.22)), bf
        end if
    next
end sub

' Pip is a small survey drone: two amber eyes, a smile, a bobbing antenna and
' thinking arcs that spin while a reply streams. Not a red monolith.
sub draw_pip()
    line (502, 4)-(796, 171), rgb(8, 10, 16), bf
    line (502, 4)-(796, 171), rgb(80, 110, 120), b

    dim cx as integer = 650
    dim bob as integer = int(3 * sin(timer * 1.3))
    dim cy as integer = 70 + bob
    dim thinking as integer = 0
    dim flashing as integer = 0
    if g_pipState = 1 then thinking = 1
    if g_pipState = 2 andalso timer < g_toolBlinkUntil then flashing = 1

    dim bright as double = 0.32
    if thinking = 1 then
        bright = 0.72 + 0.28 * (0.5 + 0.5 * sin(timer * 3.2))
    elseif flashing = 1 then
        if int((g_toolBlinkUntil - timer) * 8) mod 2 = 0 then bright = 1.0 else bright = 0.22
    end if
    dim eyeC as uinteger = rgb(int(120 + 110*bright), int(90 + 115*bright), int(55 + 80*bright))
    dim gleamC as uinteger = rgb(int(170 + 80*bright), int(150 + 90*bright), int(120 + 100*bright))

    ' antenna bead bobs and brightens while thinking
    line (cx, cy-40)-(cx, cy-56), rgb(110, 150, 160)
    dim beadY as integer = cy - 58 + int(2 * sin(timer * (2.2 + thinking * 3)))
    circle (cx, beadY), 4, eyeC, , , 1, f

    ' side pods
    circle (cx-50, cy), 9, rgb(40, 70, 82), , , 1, f
    circle (cx+50, cy), 9, rgb(40, 70, 82), , , 1, f
    circle (cx-50, cy), 4, rgb(90, 140, 150), , , 1, f
    circle (cx+50, cy), 4, rgb(90, 140, 150), , , 1, f

    ' rounded shell, face plate, visor and a highlight on the crown
    circle (cx, cy), 52, rgb(58, 96, 110), , , 0.78, f
    circle (cx, cy-2), 46, rgb(112, 155, 165), , , 0.76, f
    circle (cx, cy), 38, rgb(16, 30, 40), , , 0.62, f
    circle (cx, cy-6), 40, rgb(150, 195, 200), 3.4, 5.6, 0.6

    ' eyes dart while processing; blink when idle or during a tool flash
    dim dx as integer = 0
    dim dy as integer = 0
    if thinking = 1 then
        dx = int(3 * sin(timer * 6.1))
        dy = int(2 * sin(timer * 4.3))
    end if
    dim blinking as integer = 0
    if flashing = 1 then blinking = 1
    if thinking = 0 and flashing = 0 and (timer mod 3.2) < 0.18 then blinking = 1
    for side as integer = -1 to 1 step 2
        dim ex as integer = cx + side*14 + dx
        dim ey as integer = cy - 2 + dy
        if blinking = 1 then
            line (ex-6, ey)-(ex+6, ey), eyeC
        else
            circle (ex, ey), 6, eyeC, , , 1, f
            circle (ex-2, ey-2), 2, gleamC, , , 1, f
        end if
    next

    ' smile at rest, small voice meter while thinking
    if thinking = 0 then
        circle (cx, cy+12), 11, rgb(180, 200, 170), 3.7, 5.7, 0.5
    else
        for i as integer = 0 to 4
            dim bh as integer = 3 + abs(int(8 * sin(timer * 7 + i * 0.9)))
            line (cx-16+i*8, cy+20-bh)-(cx-16+i*8, cy+20), eyeC
        next
    end if

    ' a small status bead keeps circling at rest
    if thinking = 0 and flashing = 0 then
        dim aa as double = timer * 0.9
        circle (int(cx + 56*cos(aa)), int(cy + 45*sin(aa))), 2, rgb(90, 160, 160), , , 1, f
    end if

    ' thinking arcs orbit the shell; a tool flash sends a ring outward
    if thinking = 1 then
        dim a as double = timer * 1.8
        circle (cx, cy), 62, rgb(120, 200, 190), a, a + 1.1, 0.8
        circle (cx, cy), 62, rgb(120, 200, 190), a + 3.14, a + 4.24, 0.8
        circle (cx, cy), 67, rgb(80, 150, 150), a + 1.6, a + 2.4, 0.8
    end if
    if flashing = 1 then
        dim ph as double = 1 - (g_toolBlinkUntil - timer) / 0.7
        if ph < 0 then ph = 0
        if ph > 1 then ph = 1
        circle (cx, cy), 54 + int(14*ph), rgb(220, 190, 110), , , 0.8
    end if

    draw string (638, 130), "PIP", rgb(200, 220, 190)
    draw string (602, 148), "CARTOGRAPHER", rgb(130, 160, 165)
end sub

sub draw_header_body()
    line (0, 0)-(SC_W-1, 175), rgb(12, 14, 24), bf
    line (499, 0)-(499, 175), rgb(80, 90, 110)

    ' ---------- left: the ship readout ----------
    for gy as integer = 16 to 160 step 32
        line (0, gy)-(480, gy), rgb(20, 26, 40)
    next
    for gx as integer = 40 to 480 step 40
        line (gx, 0)-(gx, 100), rgb(18, 22, 36)
    next

    ' the Cartographer: hull, dome, antenna; a ship silhouette, not a schematic
    dim hull(0 to 8, 0 to 1) as integer
    hull(0,0)=28:  hull(0,1)=70
    hull(1,0)=70:  hull(1,1)=52
    hull(2,0)=190: hull(2,1)=52
    hull(3,0)=246: hull(3,1)=64
    hull(4,0)=304: hull(4,1)=72
    hull(5,0)=246: hull(5,1)=84
    hull(6,0)=190: hull(6,1)=96
    hull(7,0)=70:  hull(7,1)=96
    hull(8,0)=28:  hull(8,1)=70
    for k as integer = 0 to 7
        line (hull(k,0), hull(k,1))-(hull(k+1,0), hull(k+1,1)), rgb(150, 180, 200)
    next
    line (70, 62)-(190, 62), rgb(90, 120, 145)
    for wx as integer = 84 to 180 step 18
        line (wx, 68)-(wx+8, 74), rgb(120, 190, 210), bf
    next
    draw_dome(150, 52, 26, rgb(130, 170, 195))
    line (124, 52)-(176, 52), rgb(90, 120, 145)
    line (110, 34)-(96, 10), rgb(150, 180, 200)
    circle (96, 10), 6, rgb(160, 190, 210)
    line (96, 10)-(96, 2), rgb(160, 190, 210)
    dim eglow as integer = 100 + (int(timer * 3) mod 3) * 40
    line (18, 64)-(28, 76), rgb(eglow, 70, 30), bf

    ' status lights: slow rotating pattern
    for i as integer = 0 to 3
        dim onoff as integer = (int(timer * 2) + i) mod 4
        dim lc as uinteger
        if onoff = 0 then lc = rgb(120, 230, 140) else lc = rgb(35, 60, 45)
        circle (24 + i*20, 108), 4, lc, , , , f
        circle (24 + i*20, 108), 4, rgb(70, 80, 90)
    next
    draw string (18, 116), "PWR NAV SVR COM", rgb(90, 110, 120)

    ' fake telemetry, refreshed every second or two
    dim tick as integer = int(timer)
    dim dtemp as integer = 46 + (tick mod 5)
    dim prog as integer = 61 + ((tick \ 2) mod 12)
    dim sstat as string = "nominal"
    select case (tick \ 3) mod 4
        case 1: sstat = "scanning"
        case 2: sstat = "nominal"
        case 3: sstat = "calibrating"
    end select
    draw string (18, 130), "drive temp: " & str(dtemp) & " C", rgb(120, 200, 200)
    draw string (18, 142), "veil survey: " & str(prog) & "%", rgb(120, 200, 200)
    draw string (18, 154), "sensors: " & sstat, rgb(120, 200, 200)
    ' the real budget lives in the LED bar between the readout and Pip
    draw_ctx_bar()

    ' ---------- right: Pip ----------
    draw_pip()
end sub

sub draw_header()
    screenlock
    draw_header_body()
    screenunlock
end sub

sub draw_window()
    g_cursorOn = (int(timer * 2) mod 2 = 0)
    screenlock
    draw_header_body()

    ' log band: last N wrapped lines, tail-anchored
    line (0, LOG_TOP)-(SC_W-1, LOG_BOTTOM), rgb(10, 10, 20), bf
    dim visible as integer = nLog
    if visible > LOG_VISIBLE then visible = LOG_VISIBLE
    dim first as integer = nLog - visible
    dim y as integer = LOG_BOTTOM - visible * 12
    for i as integer = first to nLog-1
        draw string (8, y), logLines(i), rgb(200, 212, 224)
        y += 12
    next

    ' input band: three wrapped lines with the prompt and a blinking cursor
    line (0, INPUT_TOP)-(SC_W-1, SC_H-1), rgb(30, 30, 40), bf
    line (6, INPUT_TOP+4)-(SC_W-7, SC_H-5), rgb(90, 100, 120), b
    dim wl(0 to INPUT_LINES) as string
    dim wc as integer = wrap_input(inputBuf, wl())
    y = INPUT_TOP + 9
    for i as integer = 0 to wc-1
        dim txt as string = wl(i)
        if i = wc-1 and g_cursorOn = 1 then txt += "_"
        draw string (12, y), txt, rgb(225, 232, 240)
        y += 12
    next
    screenunlock
end sub

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
        ui_log "Model not found. Downloading (~2GB)..."
        shell "curl -L -o " & modelPath & " https://huggingface.co/KRAFTON/Raon-1-nano-gguf/resolve/main/raon-1-nano-Q4_K_M.gguf"
    end if
end sub

sub llm_init()
    ensure_model()
    dim modelPath as string = "models/raon-1-nano-Q4_K_M.gguf"
    dim mparams as llama_model_params = llama_model_default_params()
    mparams.n_gpu_layers = -1
    g_model = llama_model_load_from_file(strptr(modelPath), mparams)
    if g_model = 0 then
        ui_log "Failed to load model"
        sleep
        end
    end if
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
        g_pipState = 1
        decode_tokens(@promptToks(0), nTok, 0, 1)
        g_pipState = 0
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
        ui_log "Could not load " & SESSION_FILE & "; starting fresh."
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
    if n > 32 then draw_header()
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
    if n > 32 then draw_header()
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
    g_pipState = 1
    dim nTok as integer = llama_tokenize(g_vocab, strptr(delta), len(delta), @promptToks(0), 16384, 0, 1)
    if nTok < 0 then nTok = -nTok
    if nTok = 0 then g_pipState = 0: return ""
    ' reserve 256 tokens for the reply
    if nAllTok + nTok + 256 >= g_ctxSize then
        g_stopped = 1
        g_pipState = 0
        ui_log STOP_LINE
        file_log STOP_LINE
        draw_window()
        return ""
    end if

    decode_tokens(@promptToks(0), nTok, nAllTok, 1)
    draw_header()
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
            ui_log STOP_LINE
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
        ' keep the face moving while tokens stream
        draw_header()
    wend
    g_pipState = 0

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
    if verdict <> "" then file_log "RECALL " & verdict
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
        ui_log STOP_LINE
        file_log STOP_LINE
        draw_window()
        return
    end if

    ui_log "> " & userInput
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
    ui_pip_reply(reply)
    file_log "REPLY=" & reply
    draw_window()

    ' canonical tool loop: call, execute, append result as a user/tool turn,
    ' then let the model continue with the result still in context.
    dim steps as integer = 0
    do while pc.ok = 1 and steps < 4
        steps += 1
        file_log "TOOL_CALL=" & pc.norm
        g_pipState = 2
        g_toolBlinkUntil = timer + 0.7
        draw_header()
        dim result as string = execute_tool(pc)
        file_log "TOOL_RESULT=" & result
        draw_window()
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
        ui_pip_reply(reply)
        file_log "REPLY=" & reply
        draw_window()
    loop

    dim status as string = "context: " & str(nAllTok) & "/" & str(g_ctxSize) & " tokens"
    file_log status
    draw_window()
    log_recall(userInput, reply)
    save_session()
end sub

' ---------- main ----------
randomize timer
screenres SC_W, SC_H, 32
windowtitle "Pip - survey vessel Cartographer"

for ai as integer = 1 to 12
    if command(ai) = "--log" or command(ai) = "-log" then logging = 1
    if command(ai) = "--demo" then demoMode = 1
    if command(ai) = "--ctx" and ai < 12 then
        dim cv as string = trim(command(ai+1))
        if isAllDigits(cv) = 1 then g_ctxSize = val(cv)
    end if
next

g_pipState = 1
draw_window()
init_tools()
llm_init()
g_pipState = 0
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
        ui_log "Resume the previous conversation? (y/n)"
        draw_window()
        dim ans as string = ""
        do
            dim rk as string = inkey
            if rk = "y" or rk = "Y" then ans = "y"
            if rk = "n" or rk = "N" then ans = "n"
            if rk = chr(27) then g_quit = 1: exit do
            if ans <> "" then exit do
            sleep 16
        loop
        if g_quit = 0 andalso ans = "y" then resumed = load_session()
    end if
    if g_quit = 0 andalso resumed = 0 then fresh_session()
end if

draw_window()

if demoMode = 1 then
    dim scriptCount as integer = 8
    dim demoCmd(0 to 7) as string
    if resumed = 1 then
        demoCmd(0) = "What did I ask you to note about the sensors?"
        demoCmd(1) = "What's my name and what is our mission?"
        demoCmd(2) = "Roll a twenty-sided die."
        demoCmd(3) = "quit"
        scriptCount = 4
    else
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
        draw_window()
        for ci as integer = 1 to len(cmd)
            if inkey = chr(27) then demoAbort = 1: exit for
            dim ch as string = mid(cmd, ci, 1)
            if input_fits(inputBuf & ch) = 1 then inputBuf += ch
            draw_window()
            sleep 50
        next
        if demoAbort = 1 then exit for
        process_input(inputBuf)
        inputBuf = ""
        draw_window()
        if demoAbort = 1 then exit for
        if g_stopped = 1 then exit for
        ' pause about 2 seconds after each reply; Escape stops the demo
        for pi as integer = 0 to 39
            if inkey = chr(27) then demoAbort = 1: exit for
            draw_window()
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
        process_input(inputBuf)
        inputBuf = ""
    elseif k = chr(8) then
        if len(inputBuf) > 0 then inputBuf = left(inputBuf, len(inputBuf)-1)
    elseif len(k) = 1 and asc(k) >= 32 and asc(k) <= 126 then
        if input_fits(inputBuf & k) = 1 then inputBuf += k
    end if
    draw_window()
    sleep 16
loop

' the last event in the log is the SAVE from quit
llama_sampler_free(g_smp)
llama_free(g_ctx)
llama_model_free(g_model)
end

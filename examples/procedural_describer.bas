' procedural_describer.bas - state-grounded dungeon narration via raon-1-nano
'
' What this showcases: prompt-as-report. A 12-room dungeon floor is held as
' REAL state in code (type, contents, features, exits, visited, last event,
' plus the chest / trap / door flags). On entering a room the program turns
' that state alone into a flat fact list and the model writes two or three
' sentences of prose from it. The model narrates the database; it does not
' replace it.
'
' Deliberate limitations:
'   * Descriptions are regenerated on EVERY entry, with NO cache. An unchanged
'     room can be described differently on two visits; continuity is not
'     guaranteed and is not attempted.
'   * Grounding reduces invention but does not eliminate it. The model may add
'     a smell or a sound to colour a listed fact. It must never add an object,
'     creature, person, or exit that the fact list does not contain; Tab (debug)
'     exposes the facts so a reader can check the prose against them.
'   * One floor, a fixed room graph, and no combat. The only state changes are
'     the chest, the dart trap, and the vault door.
'   * Tone consistency rests entirely on the system prompt and the sampler.
'
' Each entry is a FRESH generation: clear the context, send system + that
' room's fact prompt, generate. persistent_companion holds one context on
' purpose; this example restarts per description on purpose, so each
' description must stand on its own room's facts alone, with no bleed from
' the previous room's prose.
'
' Build: ./build_procedural_describer.sh /path/to/llama.cpp
' Run:   ./run_procedural_describer.sh /path/to/llama.cpp [--demo] [--log]

#include once "fbgfx.bi"
#include once "llama.bi"

const SC_W = 800
const SC_H = 600
const GW = 4
const GH = 3

const SYS_PROMPT = "You narrate a small dungeon floor to a player in terse, evocative, second-person prose. Write exactly two or three sentences, no more. Describe ONLY what is in the fact list you are given. Atmosphere may colour a listed fact (how the light falls on the spears, how the air feels), and you may add a smell or a sound, but you must never introduce an object, furniture, shelf, container, creature, person, or exit that is not in the list. When Contents says 'nothing of note', give the room no contents at all. Refer to exits only by the directions listed (north, east, south, west); never name a room the player has not entered. If a fact says something is empty, spent, open, or closed, say so. Use only ASCII characters. No markdown, no headings, no lists."

type RoomRec
    name as string
    contents as string
    feature as string
    lastEvent as string
    visited as integer
    looted as integer
    trapArmed as integer
    doorSealed as integer
end type

dim shared rooms(0 to GW-1, 0 to GH-1) as RoomRec

' ---------- UI / state ----------
dim shared px as integer = 0
dim shared py as integer = 0
dim shared g_markX as integer = 56
dim shared g_markY as integer = 118
dim shared curDesc as string
dim shared statusMsg as string
dim shared debugOn as integer = 0
dim shared logging as integer = 0
dim shared demoMode as integer = 0
dim shared g_abort as integer = 0
dim shared g_generating as integer = 0
dim shared hasKey as integer = 0

' one line per generated description, for the demo verdicts
dim shared descHist(0 to 15) as string
dim shared factHist(0 to 15) as string
dim shared descX(0 to 15) as integer
dim shared descY(0 to 15) as integer
dim shared nDesc as integer = 0

' ---------- LLM globals ----------
dim shared g_model as llama_model ptr
dim shared g_ctx as llama_context ptr
dim shared g_smp as llama_sampler ptr
dim shared g_vocab as const llama_vocab ptr

' ---------- grid helpers ----------
function cell_cx(byval x as integer) as integer
    return 24 + x * 68 + 32
end function

function cell_cy(byval y as integer) as integer
    return 64 + y * 116 + 54
end function

' Exits are the orthogonal grid neighbours except two walled pairs:
' (1,0)-(1,1) and (2,2)-(3,2). dir: 0=north 1=east 2=south 3=west.
function has_exit(byval x as integer, byval y as integer, byval dirIdx as integer) as integer
    select case dirIdx
        case 0
            if y <= 0 then return 0
            if x = 1 and y = 1 then return 0
        case 1
            if x >= GW-1 then return 0
            if x = 2 and y = 2 then return 0
        case 2
            if y >= GH-1 then return 0
            if x = 1 and y = 0 then return 0
        case 3
            if x <= 0 then return 0
            if x = 3 and y = 2 then return 0
    end select
    return 1
end function

function dir_from_delta(byval dx as integer, byval dy as integer) as integer
    if dy = -1 then return 0
    if dx = 1 then return 1
    if dy = 1 then return 2
    return 3
end function

function exits_text(byval x as integer, byval y as integer) as string
    dim names(0 to 3) as string = {"north", "east", "south", "west"}
    dim s as string = ""
    for d as integer = 0 to 3
        if has_exit(x, y, d) = 1 then
            if s <> "" then s += ", "
            s += names(d)
        end if
    next
    if s = "" then s = "none"
    return s
end function

' The fact list, built in code. oneLine=0 gives the multi-line debug form;
' oneLine=1 gives the same fields joined for the log. Both are verbatim.
function fact_list(byval x as integer, byval y as integer, byval oneLine as integer) as string
    dim c as string = rooms(x, y).contents
    if c = "" then c = "nothing of note"
    dim ft as string = rooms(x, y).feature
    if ft = "" then ft = "none"
    dim ev as string = rooms(x, y).lastEvent
    if ev = "" then ev = "nothing has happened here yet"

    dim f1 as string = "Room: " & rooms(x, y).name
    dim f2 as string = "Contents: " & c
    dim f3 as string = "Feature: " & ft
    dim f4 as string = "Exits: " & exits_text(x, y)
    dim f5 as string = "Last event: " & ev

    if oneLine = 1 then
        return f1 & " | " & f2 & " | " & f3 & " | " & f4 & " | " & f5
    end if
    return f1 & chr(10) & f2 & chr(10) & f3 & chr(10) & f4 & chr(10) & f5
end function

sub init_floor()
    ' (0,0)
    rooms(0,0).name = "Entrance Hall"
    rooms(0,0).feature = "an iron gate, now closed behind you"
    ' (1,0)
    rooms(1,0).name = "Dusty Corridor"
    ' (2,0)
    rooms(2,0).name = "Guard Post"
    rooms(2,0).contents = "a rack of rusted spears"
    ' (3,0)
    rooms(3,0).name = "Collapsed Stair"
    rooms(3,0).feature = "rubble blocking the stairs up"
    ' (0,1)
    rooms(0,1).name = "Store Room"
    rooms(0,1).contents = "crates and barrels"
    ' (1,1)
    rooms(1,1).name = "Chest Chamber"
    rooms(1,1).contents = "a wooden chest holding a silver key"
    rooms(1,1).looted = 0
    ' (2,1)
    rooms(2,1).name = "Trap Corridor"
    rooms(2,1).feature = "a dart trap, armed"
    rooms(2,1).trapArmed = 1
    ' (3,1)
    rooms(3,1).name = "Shrine"
    rooms(3,1).feature = "an obsidian obelisk"
    ' (0,2)
    rooms(0,2).name = "Flooded Cellar"
    rooms(0,2).feature = "ankle-deep black water"
    ' (1,2)
    rooms(1,2).name = "Library"
    rooms(1,2).contents = "moldering books"
    ' (2,2)
    rooms(2,2).name = "Armory"
    rooms(2,2).contents = "rows of empty weapon racks"
    ' (3,2)
    rooms(3,2).name = "Vault Antechamber"
    rooms(3,2).feature = "a heavy sealed door"
    rooms(3,2).doorSealed = 1
end sub

' ---------- logging ----------
sub file_log(msg as string)
    if logging = 0 then return
    dim f as integer = freefile
    open "procedural_describer.log" for append as #f
    print #f, msg
    close #f
end sub

' ---------- wrapping (same approach as text_adventure's log window) ----------
sub wrap_lines(lines() as string, byref nLines as integer, s as string, maxw as integer)
    dim rest as string = s
    do while len(rest) > maxw
        dim chunk as string = left(rest, maxw)
        dim sp as integer = instrrev(chunk, " ")
        if sp < 1 then sp = maxw
        if nLines < 64 then
            lines(nLines) = left(rest, sp)
            nLines += 1
        else
            for i as integer = 0 to 62
                lines(i) = lines(i+1)
            next
            lines(63) = left(rest, sp)
        end if
        rest = trim(mid(rest, sp + 1))
    loop
    if rest <> "" then
        if nLines < 64 then
            lines(nLines) = rest
            nLines += 1
        else
            for i as integer = 0 to 62
                lines(i) = lines(i+1)
            next
            lines(63) = rest
        end if
    end if
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

    dim sparams as llama_sampler_chain_params = llama_sampler_chain_default_params()
    g_smp = llama_sampler_chain_init(sparams)
    ' Sampler = rpg_village's proven prose row: the job is the same, short
    ' grounded prose in a set tone. penalties(64, 1.0, 0.0, 1.5), top_k(40),
    ' top_p(0.9, 1), temp(0.4), dist(LLAMA_DEFAULT_SEED). No per-generation
    ' sampler reset with a fixed seed (the grammar_forge seed lesson): the
    ' RNG keeps advancing, so a revisit can produce a different description.
    llama_sampler_chain_add(g_smp, llama_sampler_init_penalties(llama_vocab_n_tokens(g_vocab), 64, 1.0, 0.0, 1.5))
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_k(40))
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_p(0.9, 1))
    llama_sampler_chain_add(g_smp, llama_sampler_init_temp(0.4))
    llama_sampler_chain_add(g_smp, llama_sampler_init_dist(LLAMA_DEFAULT_SEED))
end sub

' ---------- drawing ----------
sub draw_all()
    screenlock
    line (0,0)-(SC_W-1, SC_H-1), rgb(16,16,22), bf
    draw string (16,14), "PROCEDURAL DESCRIBER - prose from room state"
    draw string (16,30), "Debug (Tab): " & iif(debugOn = 1, "ON", "OFF") & "   Demoscript: " & iif(demoMode = 1, "yes", "no")

    ' map panel
    line (16,50)-(306,432), rgb(28,28,40), bf
    line (16,50)-(306,432), rgb(120,120,150), b
    draw string (22,54), "MAP (4x3)"
    for ry as integer = 0 to GH-1
        for rx as integer = 0 to GW-1
            dim cx as integer = 24 + rx * 68
            dim cy as integer = 64 + ry * 116
            dim bgCol as uinteger = rgb(24,24,32)
            if rooms(rx,ry).visited = 1 then bgCol = rgb(58,64,86)
            if rx = px and ry = py then bgCol = rgb(92,126,74)
            line (cx,cy)-(cx+64,cy+108), bgCol, bf
            ' walls: a solid edge where the graph has no exit
            dim wall as uinteger = rgb(190,190,215)
            if has_exit(rx,ry,0) = 0 then line (cx,cy)-(cx+64,cy), wall
            if has_exit(rx,ry,1) = 0 then line (cx+64,cy)-(cx+64,cy+108), wall
            if has_exit(rx,ry,2) = 0 then line (cx,cy+108)-(cx+64,cy+108), wall
            if has_exit(rx,ry,3) = 0 then line (cx,cy)-(cx,cy+108), wall
            if rooms(rx,ry).visited = 1 then
                draw string (cx+5, cy+8), left(rooms(rx,ry).name, 1), rgb(235,235,220)
            else
                draw string (cx+5, cy+8), "?", rgb(80,80,95)
            end if
            draw string (cx+5, cy+94), "(" & str(rx) & "," & str(ry) & ")", rgb(130,130,150)
            if rx = px and ry = py then
                circle (g_markX, g_markY), 9, rgb(255,220,60), , , , f
                draw string (cx+22, cy+74), "You", rgb(255,240,150)
            end if
        next
    next

    ' description panel, facts panel when debugging
    dim dx0 as integer = 316
    dim dx1 as integer = 790
    if debugOn = 1 then dx1 = 556
    line (dx0,50)-(dx1,432), rgb(28,28,40), bf
    line (dx0,50)-(dx1,432), rgb(120,120,150), b
    draw string (dx0+8, 56), "DESCRIPTION - " & rooms(px,py).name, rgb(255,220,140)
    dim dl(0 to 63) as string
    dim ndl as integer = 0
    if g_generating = 1 then
        draw string (dx0+8, 78), "Describing...", rgb(150,200,150)
    else
        wrap_lines(dl(), ndl, curDesc, iif(debugOn = 1, 27, 57))
        for i as integer = 0 to ndl-1
            draw string (dx0+8, 78 + i*16), dl(i), rgb(220,225,235)
        next
    end if
    if debugOn = 1 then
        line (566,50)-(790,432), rgb(34,30,28), bf
        line (566,50)-(790,432), rgb(200,150,90), b
        draw string (574, 56), "FACTS (verbatim)", rgb(255,190,110)
        dim fl(0 to 63) as string
        dim nfl as integer = 0
        wrap_lines(fl(), nfl, fact_list(px, py, 0), 26)
        for i as integer = 0 to nfl-1
            draw string (574, 78 + i*15), fl(i), rgb(235,210,175)
        next
    end if

    ' status line
    line (16,444)-(790,502), rgb(26,26,36), bf
    line (16,444)-(790,502), rgb(110,110,135), b
    draw string (24, 452), "Position (" & str(px) & "," & str(py) & ") - " & rooms(px,py).name & "   silver key: " & iif(hasKey = 1, "yes", "no"), rgb(200,220,200)
    dim sl(0 to 63) as string
    dim nsl as integer = 0
    wrap_lines(sl(), nsl, statusMsg, 94)
    for i as integer = 0 to nsl-1
        draw string (24, 472 + i*14), sl(i), rgb(255,210,140)
    next
    draw string (16,520), "Arrows: move   Space: interact   Tab: debug   Esc: quit", rgb(150,150,170)
    screenunlock
end sub

' ---------- generation ----------
function describe_room(byval x as integer, byval y as integer) as string
    g_generating = 1
    draw_all()
    ' Fresh context per description: no previous room's prose can bleed in.
    llama_memory_clear(llama_get_memory(g_ctx), true)

    dim usr as string
    usr = "Describe the room you have just entered in two or three sentences, using only these facts:" & chr(10)
    usr = usr & fact_list(x, y, 0) & chr(10)
    usr = usr & "This list is exhaustive for anything the player can interact with. Do not add any object, creature, person, or exit."

    dim prompt as string
    prompt = "<|im_start|>system" & chr(10) & SYS_PROMPT & chr(10) & "<|im_end|>" & chr(10)
    prompt = prompt & "<|im_start|>user" & chr(10) & usr & chr(10) & "<|im_end|>" & chr(10)
    ' thinking mode OFF
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
    if nTok <= 0 then
        deallocate(toks)
        g_generating = 0
        return ""
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
    while nGen < 110
        ' Escape still exits cleanly while a description is generating.
        if g_abort = 1 then exit while
        if demoMode = 1 andalso inkey = chr(27) then g_abort = 1: exit while
        dim tok as llama_token = llama_sampler_sample(g_smp, g_ctx, -1)
        llama_sampler_accept(g_smp, tok)
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
    g_generating = 0
    if g_abort = 1 then return ""

    dim filtered as string = ""
    for i as integer = 1 to len(result)
        dim c as integer = asc(mid(result, i, 1))
        if c >= 32 and c <= 126 then
            filtered += mid(result, i, 1)
        elseif c = 10 or c = 13 or c = 9 then
            filtered += " "
        end if
    next
    ' collapse whitespace: one description is one line in the log
    dim flat as string = ""
    dim prevSpace as integer = 0
    for i as integer = 1 to len(filtered)
        if mid(filtered, i, 1) = " " then
            if prevSpace = 0 then flat += " "
            prevSpace = 1
        else
            flat += mid(filtered, i, 1)
            prevSpace = 0
        end if
    next
    return trim(flat)
end function

' ---------- room entry ----------
sub enter_room(byval x as integer, byval y as integer, byval entryNo as integer)
    px = x: py = y
    g_markX = cell_cx(px)
    g_markY = cell_cy(py)
    rooms(x,y).visited = 1

    ' The trap is part of the state BEFORE the description is built.
    if x = 2 and y = 1 and rooms(x,y).trapArmed = 1 then
        rooms(x,y).trapArmed = 0
        rooms(x,y).feature = "a spent dart trap"
        rooms(x,y).lastEvent = "a dart trap sprung as you entered; the darts missed"
        statusMsg = "A dart trap springs as you enter. The darts miss."
        file_log("STATE TRAP feature=" & rooms(x,y).feature & " event=" & rooms(x,y).lastEvent)
        file_log("TRAP-SPRUNG-OK feature=" & rooms(x,y).feature & " event=" & rooms(x,y).lastEvent)
    end if

    dim d as string = describe_room(x, y)
    if g_abort = 1 then exit sub
    if d = "" then d = "(no description returned)"
    curDesc = d

    file_log("FACTS entry=" & str(entryNo) & " " & fact_list(x, y, 1))
    file_log("DESC=" & d)
    if nDesc <= 15 then
        descHist(nDesc) = d
        factHist(nDesc) = fact_list(x, y, 1)
        descX(nDesc) = x
        descY(nDesc) = y
        nDesc += 1
    end if
    draw_all()
end sub

' ---------- interaction ----------
sub interact_press()
    if px = 1 and py = 1 then
        if rooms(px,py).looted = 0 then
            rooms(px,py).looted = 1
            rooms(px,py).contents = "an open, empty wooden chest"
            rooms(px,py).lastEvent = "the chest was looted and the silver key taken"
            hasKey = 1
            statusMsg = "You lift the lid and take the silver key."
            file_log("STATE CHEST contents=" & rooms(px,py).contents & " key=yes event=" & rooms(px,py).lastEvent)
            file_log("CHEST-LOOTED-OK contents=" & rooms(px,py).contents & " key=yes event=" & rooms(px,py).lastEvent)
        else
            statusMsg = "The chest is already empty."
        end if
    elseif px = 3 and py = 2 then
        if rooms(px,py).doorSealed = 1 then
            if hasKey = 1 then
                rooms(px,py).doorSealed = 0
                rooms(px,py).feature = "a heavy door, now open"
                rooms(px,py).contents = "a hoard of dull gold coins"
                rooms(px,py).lastEvent = "the sealed door was opened with the silver key"
                statusMsg = "The silver key turns. The door swings open."
                file_log("STATE DOOR feature=" & rooms(px,py).feature & " contents=" & rooms(px,py).contents & " event=" & rooms(px,py).lastEvent)
                file_log("DOOR-OPEN-OK feature=" & rooms(px,py).feature & " contents=" & rooms(px,py).contents & " event=" & rooms(px,py).lastEvent)
            else
                statusMsg = "The door is locked. It needs a key."
            end if
        else
            statusMsg = "The vault door stands open."
        end if
    else
        statusMsg = "Nothing here responds to you."
    end if
    draw_all()
end sub

sub try_move(byval dx as integer, byval dy as integer)
    dim nx as integer = px + dx
    dim ny as integer = py + dy
    if nx < 0 or nx > GW-1 or ny < 0 or ny > GH-1 or has_exit(px, py, dir_from_delta(dx, dy)) = 0 then
        statusMsg = "No exit that way."
        draw_all()
        exit sub
    end if
    enter_room(nx, ny, nDesc+1)
end sub

' ---------- demo pacing ----------
sub hold_ms(byval ms as integer)
    dim elapsed as integer = 0
    while elapsed < ms
        if g_abort = 1 then exit while
        if inkey = chr(27) then g_abort = 1: exit while
        sleep 50
        elapsed += 50
    wend
end sub

sub demo_step(byval dx as integer, byval dy as integer)
    ' about one second of walking per step
    dim ox as integer = g_markX
    dim oy as integer = g_markY
    dim tx as integer = cell_cx(px + dx)
    dim ty as integer = cell_cy(py + dy)
    for i as integer = 1 to 60
        if g_abort = 1 then exit for
        if inkey = chr(27) then g_abort = 1: exit for
        g_markX = ox + (tx - ox) * i \ 60
        g_markY = oy + (ty - oy) * i \ 60
        draw_all()
        sleep 16
    next
    if g_abort = 1 then exit sub
    enter_room(px + dx, py + dy, nDesc+1)
    if g_abort = 1 then exit sub
    ' leave each description on screen long enough to read it
    hold_ms(3000)
end sub

sub demo_interact()
    interact_press()
    hold_ms(1200)
end sub

' ---------- demo checks ----------
sub log_final_checks()
    dim firstI as integer = -1
    dim secondI as integer = -1
    for i as integer = 0 to nDesc-1
        if descX(i) = 1 and descY(i) = 1 then
            if firstI < 0 then firstI = i else secondI = i
        end if
    next
    if firstI >= 0 and secondI >= 0 then
        if descHist(secondI) <> descHist(firstI) and instr(factHist(secondI), "an open, empty wooden chest") > 0 and instr(factHist(secondI), "the chest was looted and the silver key taken") > 0 then
            file_log("REDESCRIBE-DIFFERS-OK first=" & descHist(firstI) & " second=" & descHist(secondI))
        else
            file_log("REDESCRIBE-DIFFERS-FAIL first=" & descHist(firstI) & " second=" & descHist(secondI))
        end if
    else
        file_log("REDESCRIBE-DIFFERS-FAIL missing Chest Chamber pair")
    end if

    dim leak as integer = 0
    for i as integer = 0 to nDesc-1
        if instr(lcase(descHist(i)), "obelisk") > 0 then
            if descX(i) <> 3 or descY(i) <> 1 then
                leak = 1
                file_log("NO-CROSS-CONTAMINATION FAIL room=(" & str(descX(i)) & "," & str(descY(i)) & ") description=" & descHist(i))
            end if
        end if
    next
    if leak = 0 then file_log("NO-CROSS-CONTAMINATION OK: obelisk appears in no description outside the Shrine")
end sub

' ---------- main ----------
randomize timer
screenres SC_W, SC_H, 32
windowtitle "Procedural Describer - Raon-1-nano"

for ai as integer = 1 to 12
    if command(ai) = "--log" or command(ai) = "-log" then logging = 1
    if command(ai) = "--demo" then demoMode = 1
next
if logging = 1 then
    dim f0 as integer = freefile
    open "procedural_describer.log" for output as #f0
    close #f0
    file_log("=== SESSION START === demo=" & str(demoMode))
end if

init_floor()

screenlock
cls
draw string (40, 40), "Procedural Describer"
draw string (40, 70), "Loading model..."
screenunlock

llm_init()

if demoMode = 1 then debugOn = 1
statusMsg = "Enter a room to have it described from its facts."
g_markX = cell_cx(px)
g_markY = cell_cy(py)

if demoMode = 0 then
    ' interactive starts with the Entrance Hall already described
    enter_room(px, py, nDesc+1)
end if

do
    if inkey = chr(27) then g_abort = 1
    if g_abort = 1 then exit do

    if demoMode = 1 then
        ' 1. south to the Store Room, east to the Chest Chamber
        demo_step(0, 1)
        if g_abort = 1 then exit do
        demo_step(1, 0)
        if g_abort = 1 then exit do
        ' 2. loot the chest
        demo_interact()
        if g_abort = 1 then exit do
        ' 3. east into the Trap Corridor (trap springs on entry)
        demo_step(1, 0)
        if g_abort = 1 then exit do
        ' 4. west back to the changed Chest Chamber: a NEW description
        demo_step(-1, 0)
        if g_abort = 1 then exit do
        ' 5. east, east to the Shrine
        demo_step(1, 0)
        if g_abort = 1 then exit do
        demo_step(1, 0)
        if g_abort = 1 then exit do
        ' 6. south to the Vault Antechamber, open the door with the key
        demo_step(0, 1)
        if g_abort = 1 then exit do
        demo_interact()
        if g_abort = 1 then exit do
        ' the final held description reflects the opened door and hoard
        dim finalDesc as string = describe_room(px, py)
        if g_abort = 1 then exit do
        if finalDesc <> "" then
            curDesc = finalDesc
            file_log("REFACT " & fact_list(px, py, 1))
            file_log("REDESC=" & finalDesc)
        end if
        draw_all()
        log_final_checks()
        file_log("=== DEMO END ===")
        hold_ms(3500)
        exit do
    end if

    dim k as string = inkey
    select case k
        case chr(255) + "H"
            try_move(0, -1)
        case chr(255) + "P"
            try_move(0, 1)
        case chr(255) + "K"
            try_move(-1, 0)
        case chr(255) + "M"
            try_move(1, 0)
        case " "
            interact_press()
        case chr(9)
            debugOn = 1 - debugOn
            draw_all()
        case chr(27)
            exit do
    end select
    sleep 16
loop

llama_sampler_free(g_smp)
llama_free(g_ctx)
llama_free_model(g_model)
end

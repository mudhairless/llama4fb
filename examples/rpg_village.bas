' rpg_village.bas - tiny RPG village demo using raon-1-nano via llama.cpp
' Sprites are drawn procedurally (no external assets). Drop BMPs into assets/ to replace.
' Build: see README in this folder

#include once "fbgfx.bi"
#include once "llama.bi"

extern "C"
    declare function llama_sampler_init_penalties(byval n_vocab as long, byval penalty_last_n as long, byval penalty_repeat as single, byval penalty_freq as single, byval penalty_present as single) as llama_sampler ptr
end extern

const SC_W = 800
const SC_H = 600

' ---------- NPC data ----------
type Npc
    nombre as string
    persona as string
    x as integer
    y as integer
    col as uinteger
    topics(0 to 9) as string
    nTopics as integer
    unlocked(0 to 9) as integer
    disposition as integer
    greeting as string
    hasGreeting as integer
    history as string
    asked(0 to 9) as integer
end type

dim shared npcs(0 to 4) as Npc
dim shared nNpc as integer = 5
dim shared endingDone as integer = 0
dim shared logging as integer = 0
dim shared demoMode as integer = 0
dim shared demoAskCount as integer = 0
dim shared demoStarted as integer = 0

' ---------- LLM globals ----------
dim shared g_model as llama_model ptr
dim shared g_ctx as llama_context ptr
dim shared g_smp as llama_sampler ptr
dim shared g_tmpl as const zstring ptr
dim shared g_vocab as const llama_vocab ptr

sub log_msg(msg as string)
    if logging = 0 then return
    dim f as integer = freefile
    open "rpg_village.log" for append as #f
    print #f, msg
    close #f
end sub

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
    mparams.n_gpu_layers = -1 ' offload to GPU if a CUDA build is used
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
    g_tmpl = llama_model_chat_template(g_model, 0)

    dim sparams as llama_sampler_chain_params = llama_sampler_chain_default_params()
    g_smp = llama_sampler_chain_init(sparams)
    llama_sampler_chain_add(g_smp, llama_sampler_init_penalties(llama_vocab_n_tokens(g_vocab), 64, 1.0, 0.0, 1.5))
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_k(40))
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_p(0.9, 1))
    llama_sampler_chain_add(g_smp, llama_sampler_init_temp(0.4))
    llama_sampler_chain_add(g_smp, llama_sampler_init_dist(42))
end sub

function generate_reply(sysPrompt as string, usrPrompt as string, history as string, askCount as integer, disp as integer) as string
    llama_memory_clear(llama_get_memory(g_ctx), true)

    dim fullSys as string = sysPrompt
    if disp > 0 then
        fullSys = fullSys & chr(10) & "You currently feel warmly toward this player because they have spoken kindly to you."
    elseif disp < 0 then
        fullSys = fullSys & chr(10) & "You are annoyed with this player because of how they have spoken to you; be guarded and share less."
    end if
    if history <> "" then
        fullSys = fullSys & chr(10) & "Your memory of earlier conversations with this player: " & history
    end if
    if askCount > 1 then
        fullSys = fullSys & chr(10) & "The player has asked this same topic " & str(askCount) & " times. If you were holding something back, reveal a little more now; otherwise you may become slightly annoyed at repeating yourself, but stay in character."
    end if
    fullSys = fullSys & chr(10) & "Use only ASCII characters in your reply."
    dim prompt as string
    prompt = "<|im_start|>system" & chr(10) & fullSys & chr(10) & "<|im_end|>" & chr(10)
    prompt = prompt & "<|im_start|>user" & chr(10) & usrPrompt & chr(10) & "<|im_end|>" & chr(10)
    prompt = prompt & "<|im_start|>assistant" & chr(10) & "<think></think>" & chr(10) & chr(10)

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
    while nGen < 150
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

    dim filtered as string = ""
    for i as integer = 1 to len(result)
        dim c as integer = asc(mid(result, i, 1))
        if (c >= 32 and c <= 126) or c = 10 or c = 13 then
            filtered += mid(result, i, 1)
        elseif c = 9 then
            filtered += " "
        end if
    next
    log_msg("SYS=" & fullSys)
    log_msg("USR=" & usrPrompt)
    log_msg("REPLY=" & filtered)
    return filtered
end function

' ---------- helpers ----------
sub show_reply(npcName as string, reply as string)
    dim lines(0 to 200) as string
    dim nLines as integer = 0
    dim cur as string = ""
    for i as integer = 1 to len(reply)
        dim ch as string = mid(reply, i, 1)
        if ch = chr(10) then
            lines(nLines) = cur: nLines += 1: cur = ""
        else
            cur += ch
            if len(cur) >= 80 then
                lines(nLines) = cur: nLines += 1: cur = ""
            end if
        end if
    next
    if cur <> "" then lines(nLines) = cur: nLines += 1

    dim pageSize as integer = 9
    dim page as integer = 0
    dim totalPages as integer = (nLines + pageSize - 1) \ pageSize
    if totalPages < 1 then totalPages = 1
    while page < totalPages
        screenlock
        line (0, 390)-(799, 599), rgb(20,20,30), bf
        line (60, 400)-(740, 580), rgb(30,30,50), bf
        line (60, 400)-(740, 580), rgb(200,200,200), b
        draw string (70, 410), npcName
        dim y as integer = 430
        for j as integer = 0 to pageSize-1
            dim idx as integer = page*pageSize + j
            if idx < nLines then
                draw string (70, y), lines(idx)
                y += 14
            end if
        next
        draw string (70, 565), "Page " & str(page+1) & "/" & str(totalPages) & "  Press any key..."
        screenunlock
        if demoMode = 1 then
            sleep 500
        else
            while inkey <> "": wend
            sleep
            while inkey <> "": wend
        end if
        page += 1
    wend
end sub

function dist(ax as integer, ay as integer, bx as integer, by as integer) as double
    return sqr((ax-bx)^2 + (ay-by)^2)
end function

function find_topic_index(byref n as Npc, topicName as string) as integer
    for t as integer = 0 to 9
        if lcase(trim(n.topics(t))) = lcase(trim(topicName)) then return t
    next
    return -1
end function

function all_clues_asked() as integer
    dim ri as integer = find_topic_index(npcs(4), "the armory")
    if ri < 0 or npcs(4).asked(ri) = 0 then return 0
    dim bi1 as integer = find_topic_index(npcs(1), "the secret passage")
    if bi1 < 0 or npcs(1).asked(bi1) = 0 then return 0
    dim bi2 as integer = find_topic_index(npcs(1), "the inn cellar")
    if bi2 < 0 or npcs(1).asked(bi2) = 0 then return 0
    return 1
end function

function norm_name(s as string) as string
    dim t as string = lcase(trim(s))
    if left(t, 4) = "the " then t = trim(mid(t, 5))
    return t
end function

function last_word(s as string) as string
    dim t as string = trim(s)
    dim p as integer = instrrev(t, " ")
    if p = 0 then return t
    return mid(t, p + 1)
end function

function remove_last_word(s as string) as string
    dim t as string = trim(s)
    dim p as integer = instrrev(t, " ")
    if p = 0 then return ""
    return trim(left(t, p - 1))
end function

sub show_journal()
    dim c1 as integer = 0, c2 as integer = 0, c3 as integer = 0
    dim ri as integer = find_topic_index(npcs(4), "the armory")
    if ri >= 0 andalso npcs(4).asked(ri) > 0 then c1 = 1
    dim bi1 as integer = find_topic_index(npcs(1), "the secret passage")
    if bi1 >= 0 andalso npcs(1).asked(bi1) > 0 then c2 = 1
    dim bi2 as integer = find_topic_index(npcs(1), "the inn cellar")
    if bi2 >= 0 andalso npcs(1).asked(bi2) > 0 then c3 = 1
    dim total as integer = c1 + c2 + c3
    screenlock
    line (0, 0)-(799, 599), rgb(20, 20, 30), bf
    line (60, 80)-(740, 520), rgb(30, 30, 50), bf
    line (60, 80)-(740, 520), rgb(200, 200, 200), b
    draw string (70, 90), "Journal"
    dim y as integer = 120
    if c1 = 1 then draw string (70, y), "Ren saw Kael at the armory" else draw string (70, y), "Someone saw something the night of the theft"
    y += 20
    if c2 = 1 then draw string (70, y), "Bora knows a secret passage" else draw string (70, y), "The inn keeps secrets"
    y += 20
    if c3 = 1 then draw string (70, y), "Something clanks in the inn cellar" else draw string (70, y), "Something is hidden below the village"
    y += 30
    draw string (70, y), "Clues: " & str(total) & "/3"
    draw string (70, 500), "Press any key to close..."
    screenunlock
    if demoMode = 1 then
        sleep 500
    else
        while inkey <> "": wend
        sleep
        while inkey <> "": wend
    end if
end sub

' ---------- init npcs ----------
sub init_npcs()
    npcs(0).nombre = "Elder Maren"
    npcs(0).persona = "You are Elder Maren, a wise old village elder. Speak slowly, warmly, and a little cryptically, like a grandparent telling a story. You know Willowbrook was founded 200 years ago by settlers fleeing the dragon Vaelgor. You know Ashenbrand was forged by the old smiths and stolen three nights ago, and you suspect the stranger Kael. You do not know where the Crown Jewels are hidden. Speak in natural sentences, not lists."
    npcs(0).x = 200: npcs(0).y = 150: npcs(0).col = rgb(200,180,120)
    npcs(0).topics(0) = "the village"
    npcs(0).topics(1) = "the old castle"
    npcs(0).topics(2) = "the dragon"
    npcs(0).topics(3) = "the lost sword"
    npcs(0).topics(4) = "King Aldric"
    npcs(0).topics(5) = "the stranger Kael"
    npcs(0).topics(6) = "prophecy"
    npcs(0).nTopics = 0
    for t as integer = 0 to 9
        if npcs(0).topics(t) <> "" then npcs(0).nTopics += 1
    next
    for t as integer = 0 to 3: npcs(0).unlocked(t) = 1: next
    npcs(0).disposition = 0
    npcs(0).hasGreeting = 0
    npcs(0).greeting = ""

    npcs(1).nombre = "Innkeeper Bora"
    npcs(1).persona = "You are Bora, the cheerful innkeeper of The Wandering Stag. You are gossipy, warm, and a little dramatic. You love sharing rumors. You know the stranger Kael stayed at your inn and asked about a secret passage. You heard clanking from the cellar last night. You do not know anything about Ashenbrand. Speak naturally, like a friendly neighbor chatting over a drink."
    npcs(1).x = 600: npcs(1).y = 200: npcs(1).col = rgb(220,120,120)
    npcs(1).topics(0) = "rumors"
    npcs(1).topics(1) = "the stranger"
    npcs(1).topics(2) = "the treasure"
    npcs(1).topics(3) = "the secret passage"
    npcs(1).topics(4) = "the lost sword"
    npcs(1).topics(5) = "the inn cellar"
    npcs(1).topics(6) = "Kael's visit"
    npcs(1).nTopics = 0
    for t as integer = 0 to 9
        if npcs(1).topics(t) <> "" then npcs(1).nTopics += 1
    next
    for t as integer = 0 to 3: npcs(1).unlocked(t) = 1: next
    npcs(1).disposition = 0
    npcs(1).hasGreeting = 0
    npcs(1).greeting = ""

    npcs(2).nombre = "Blacksmith Tor"
    npcs(2).persona = "You are Tor, a gruff blacksmith. You speak in short, blunt sentences and grumble. You care about weapons and metal and dislike magic. You forged Ashenbrand years ago and recognize its steel. You know whoever stole it must know metalwork. You do not know Kael and have never met him. Speak like a tired craftsman, not a robot."
    npcs(2).x = 150: npcs(2).y = 450: npcs(2).col = rgb(120,120,200)
    npcs(2).topics(0) = "weapons"
    npcs(2).topics(1) = "the lost sword"
    npcs(2).topics(2) = "the dragon"
    npcs(2).topics(3) = "armor"
    npcs(2).topics(4) = "Ashenbrand's history"
    npcs(2).topics(5) = "the thief's craft"
    npcs(2).nTopics = 0
    for t as integer = 0 to 9
        if npcs(2).topics(t) <> "" then npcs(2).nTopics += 1
    next
    for t as integer = 0 to 3: npcs(2).unlocked(t) = 1: next
    npcs(2).disposition = 0
    npcs(2).hasGreeting = 0
    npcs(2).greeting = ""

    npcs(3).nombre = "Priestess Lia"
    npcs(3).persona = "You are Priestess Lia, gentle and spiritual. You speak softly and kindly, with quiet conviction. You tend the sacred grove east of Willowbrook. You had a vision of Vaelgor stirring in his sleep. You believe Ashenbrand can banish the dragon. You know nothing about the Crown Jewels. Speak like a calm, caring priestess."
    npcs(3).x = 650: npcs(3).y = 460: npcs(3).col = rgb(180,220,180)
    npcs(3).topics(0) = "the sacred grove"
    npcs(3).topics(1) = "blessings"
    npcs(3).topics(2) = "the dragon"
    npcs(3).topics(3) = "prophecy"
    npcs(3).topics(4) = "King Aldric"
    npcs(3).topics(5) = "the old castle"
    npcs(3).nTopics = 0
    for t as integer = 0 to 9
        if npcs(3).topics(t) <> "" then npcs(3).nTopics += 1
    next
    for t as integer = 0 to 3: npcs(3).unlocked(t) = 1: next
    npcs(3).disposition = 0
    npcs(3).hasGreeting = 0
    npcs(3).greeting = ""

    npcs(4).nombre = "Guard Captain Ren"
    npcs(4).persona = "You are Captain Ren, a stern guard. You speak in a clipped, military tone. You patrol Highgarden Keep's gates daily. You saw Kael near the armory the night Ashenbrand was stolen and believe the thief fled toward the inn. You know nothing about any secret passage beneath the inn and have never heard of one. Do not speculate about it. Speak like a soldier giving a report."
    npcs(4).x = 400: npcs(4).y = 100: npcs(4).col = rgb(150,150,150)
    npcs(4).topics(0) = "patrols"
    npcs(4).topics(1) = "the castle gates"
    npcs(4).topics(2) = "the stranger"
    npcs(4).topics(3) = "the dragon"
    npcs(4).topics(4) = "the lost sword"
    npcs(4).topics(5) = "the armory"
    npcs(4).topics(6) = "Kael's whereabouts"
    npcs(4).nTopics = 0
    for t as integer = 0 to 9
        if npcs(4).topics(t) <> "" then npcs(4).nTopics += 1
    next
    for t as integer = 0 to 3: npcs(4).unlocked(t) = 1: next
    npcs(4).disposition = 0
    npcs(4).hasGreeting = 0
    npcs(4).greeting = ""
end sub

' ---------- dialogue ----------
sub talk_to(idx as integer)
    while multikey(FB.SC_SPACE)
        sleep 16
    wend
    while inkey <> "": wend
    ' greeting (cached)
    if npcs(idx).hasGreeting = 0 then
        npcs(idx).greeting = generate_reply(npcs(idx).persona, "Greet the player in character, speaking naturally as a real person would, one or two sentences. Avoid robotic or formulaic phrases. If you have a memory of this player, greet them like someone you have spoken with before.", npcs(idx).history, 0, npcs(idx).disposition)
        npcs(idx).hasGreeting = 1
    end if
    show_reply npcs(idx).nombre, npcs(idx).greeting
    dim sel as integer = 0
    dim done as integer = 0
    while done = 0
        ' build visible list
        dim visIdx(0 to 9) as integer
        dim nVis as integer = 0
        for t as integer = 0 to npcs(idx).nTopics - 1
            if npcs(idx).topics(t) <> "" and npcs(idx).unlocked(t) = 1 then
                visIdx(nVis) = t
                nVis += 1
            end if
        next
        dim hasAccuse as integer = 0
        if idx = 4 and all_clues_asked() = 1 and endingDone = 0 then hasAccuse = 1
        dim totalItems as integer = nVis + hasAccuse
        if totalItems = 0 then totalItems = 1
        if sel >= totalItems then sel = 0

        screenlock
        line (0, 0)-(799, 599), rgb(50,110,50), bf
        line (0, 300)-(799, 340), rgb(180,150,100), bf
        for i as integer = 0 to nNpc-1
            line (npcs(i).x-6, npcs(i).y)-(npcs(i).x+6, npcs(i).y+16), npcs(i).col, bf
            circle (npcs(i).x, npcs(i).y-8), 8, npcs(i).col, , , , f
            draw string (npcs(i).x-30, npcs(i).y-30), npcs(i).nombre
        next
        line (400-6, 300)-(400+6, 316), rgb(255,220,60), bf
        circle (400, 292), 8, rgb(255,220,60), , , , f
        line (80, 80)-(720, 360), rgb(30,30,50), bf
        line (80, 80)-(720, 360), rgb(200,200,200), b
        draw string (90, 90), "Talking to " & npcs(idx).nombre
        draw string (90, 110), "Choose a topic:"
        for i as integer = 0 to nVis - 1
            dim prefix as string = "  "
            if i = sel then prefix = "> "
            dim rt as integer = visIdx(i)
            draw string (90, 135 + i*18), prefix & str(i+1) & ". " & npcs(idx).topics(rt)
        next
        if hasAccuse = 1 then
            dim ai2 as integer = nVis
            dim prefix2 as string = "  "
            if sel = ai2 then prefix2 = "> "
            draw string (90, 135 + ai2*18), prefix2 & str(ai2+1) & ". Accuse Kael"
        end if
        draw string (90, 330), "Up/Down select, Enter ask, Esc leave"
        screenunlock

        dim k as string
        if demoMode = 1 then
            sleep 400
            if sel < totalItems then
                k = chr(13)
            else
                k = "q"
            end if
        else
            k = inkey
        end if
        if k = chr(255)+"H" then
            sel -= 1
            if sel < 0 then sel = totalItems - 1
        elseif k = chr(255)+"P" then
            sel += 1
            if sel >= totalItems then sel = 0
        elseif k = chr(13) then
            if sel < nVis then
                dim realTopic as integer = visIdx(sel)
                dim topic as string = npcs(idx).topics(realTopic)

                ' tone menu
                dim toneSel as integer = 0
                dim toneDone as integer = 0
                dim toneEsc as integer = 0
                while toneDone = 0
                    screenlock
                    line (0, 0)-(799, 599), rgb(50,110,50), bf
                    line (0, 300)-(799, 340), rgb(180,150,100), bf
                    line (80, 80)-(720, 360), rgb(30,30,50), bf
                    line (80, 80)-(720, 360), rgb(200,200,200), b
                    draw string (90, 90), "Talking to " & npcs(idx).nombre
                    draw string (90, 110), "Choose tone:"
                    dim toneLabels(0 to 4) as string = {"Polite","Neutral","Angry","Flattering","Joking"}
                    for i as integer = 0 to 4
                        dim tp as string = "  "
                        if i = toneSel then tp = "> "
                        draw string (90, 135 + i*18), tp & str(i+1) & ". " & toneLabels(i)
                    next
                    draw string (90, 330), "Up/Down select, Enter ask, Esc back"
                    screenunlock
                    dim tk as string
                    if demoMode = 1 then
                        sleep 200
                        toneSel = demoAskCount mod 5
                        tk = chr(13)
                    else
                        tk = inkey
                    end if
                    if tk = chr(255)+"H" then
                        toneSel -= 1
                        if toneSel < 0 then toneSel = 4
                    elseif tk = chr(255)+"P" then
                        toneSel += 1
                        if toneSel > 4 then toneSel = 0
                    elseif tk = chr(13) then
                        toneDone = 1
                    elseif tk = chr(27) then
                        toneEsc = 1
                        toneDone = 1
                    end if
                    sleep 16
                wend
                if toneEsc = 1 then
                    continue while
                end if

                ' map tone
                dim toneWord as string = ""
                dim toneSentence as string = ""
                dim toneDelta as integer = 0
                if toneSel = 0 then
                    toneWord = "politely"
                    toneSentence = "The player asks about " & topic & ", politely and respectfully."
                    toneDelta = 1
                elseif toneSel = 1 then
                    toneWord = ""
                    toneSentence = ""
                    toneDelta = 0
                elseif toneSel = 2 then
                    toneWord = "angrily"
                    toneSentence = "The player asks about " & topic & ", angrily and demanding, like someone picking a fight."
                    toneDelta = -2
                elseif toneSel = 3 then
                    toneWord = "flatteringly"
                    toneSentence = "The player asks about " & topic & ", laying on the flattery thick."
                    toneDelta = 1
                elseif toneSel = 4 then
                    toneWord = "jokingly"
                    toneSentence = "The player asks about " & topic & ", joking and teasing, not entirely serious."
                    toneDelta = 0
                end if

                npcs(idx).asked(realTopic) += 1
                demoAskCount += 1
                dim askCount as integer = npcs(idx).asked(realTopic)

                ' disposition update
                npcs(idx).disposition += toneDelta
                if npcs(idx).disposition > 5 then npcs(idx).disposition = 5
                if npcs(idx).disposition < -5 then npcs(idx).disposition = -5

                ' locked list
                dim lockedList as string = ""
                for t as integer = 0 to npcs(idx).nTopics - 1
                    if npcs(idx).topics(t) <> "" and npcs(idx).unlocked(t) = 0 then
                        if lockedList = "" then lockedList = npcs(idx).topics(t) else lockedList = lockedList & ", " & npcs(idx).topics(t)
                    end if
                next

                screenlock
                line (0, 390)-(799, 599), rgb(20,20,30), bf
                draw string (70, 450), npcs(idx).nombre & " is thinking..."
                screenunlock
                dim usrPrompt as string
                if toneSentence = "" then
                    usrPrompt = "The player asks about: " & topic & ". Reply in character, speaking naturally as a real person would. If your answer brings up a new subject the player could ask about, you may append [UNLOCK: subject] using exactly one of these names: " & lockedList & "."
                else
                    usrPrompt = toneSentence & " Reply in character, speaking naturally as a real person would. If your answer brings up a new subject the player could ask about, you may append [UNLOCK: subject] using exactly one of these names: " & lockedList & "."
                end if
                dim reply as string = generate_reply(npcs(idx).persona, usrPrompt, npcs(idx).history, askCount, npcs(idx).disposition)

                ' parse unlock markers with ladder
                dim unlockedNow as string = ""
                dim cleanReply as string = reply
                dim searchPos as integer = 1
                do
                    dim start as integer = instr(searchPos, cleanReply, "[UNLOCK:")
                    if start = 0 then exit do
                    dim finish as integer = instr(start, cleanReply, "]")
                    if finish = 0 then exit do
                    dim subject as string = trim(mid(cleanReply, start + 8, finish - start - 8))
                    dim matchedTopic as integer = -1
                    dim matchedRule as integer = 0
                    for rule as integer = 1 to 3
                        dim candidates as integer = 0
                        dim candIdx as integer = -1
                        for t as integer = 0 to npcs(idx).nTopics - 1
                            if npcs(idx).topics(t) = "" then continue for
                            if npcs(idx).unlocked(t) <> 0 then continue for
                            dim ntopic as string = norm_name(npcs(idx).topics(t))
                            dim nsubj as string = norm_name(subject)
                            dim ok as integer = 0
                            if rule = 1 then
                                if nsubj = ntopic then ok = 1
                            elseif rule = 2 then
                                if nsubj = last_word(ntopic) then ok = 1
                            elseif rule = 3 then
                                if nsubj = remove_last_word(ntopic) then ok = 1
                            end if
                            if ok = 1 then
                                candidates += 1
                                candIdx = t
                            end if
                        next
                        if candidates = 1 then
                            matchedTopic = candIdx
                            matchedRule = rule
                            exit for
                        end if
                    next
                    if matchedTopic >= 0 then
                        npcs(idx).unlocked(matchedTopic) = 1
                        if instr(unlockedNow, npcs(idx).topics(matchedTopic)) = 0 then
                            if unlockedNow = "" then unlockedNow = npcs(idx).topics(matchedTopic) else unlockedNow = unlockedNow & ", " & npcs(idx).topics(matchedTopic)
                        end if
                    end if
                    log_msg("NPC=" & npcs(idx).nombre & " TOPIC=" & topic & " TONE=" & toneWord & " RAW_SUBJECT=" & subject & " RULE=" & str(matchedRule) & " CHANGED=" & iif(matchedTopic>=0, npcs(idx).topics(matchedTopic), "(none)") & " DISPOSITION=" & str(npcs(idx).disposition))
                    cleanReply = left(cleanReply, start - 1) & mid(cleanReply, finish + 1)
                    searchPos = start
                loop

                dim condensed as string = cleanReply
                dim c2 as string = ""
                for ci as integer = 1 to len(condensed)
                    dim ch as string = mid(condensed, ci, 1)
                    if ch = chr(10) or ch = chr(13) then
                        c2 += " "
                    else
                        c2 += ch
                    end if
                next
                condensed = trim(c2)
                if len(condensed) > 160 then condensed = left(condensed, 160)
                if unlockedNow <> "" then
                    cleanReply = cleanReply & chr(10) & "[New topic unlocked: " & unlockedNow & "]"
                end if
                reply = cleanReply

                dim memEntry as string
                if toneWord = "" then
                    memEntry = "Asked about " & topic & " - you said: " & condensed
                else
                    memEntry = "Asked " & toneWord & " about " & topic & " - you said: " & condensed
                end if
                dim newHistory as string
                if npcs(idx).history = "" then
                    newHistory = memEntry
                else
                    newHistory = npcs(idx).history & " | " & memEntry
                end if
                dim ecount as integer = 1
                dim epos as integer = 1
                do
                    dim ep as integer = instr(epos, newHistory, " | ")
                    if ep = 0 then exit do
                    ecount += 1
                    epos = ep + 3
                loop
                do while ecount > 5
                    dim ep2 as integer = instr(newHistory, " | ")
                    if ep2 = 0 then exit do
                    newHistory = mid(newHistory, ep2 + 3)
                    ecount -= 1
                loop
                npcs(idx).history = newHistory
                if demoMode = 1 then sel += 1
                log_msg("NPC=" & npcs(idx).nombre & " TOPIC=" & topic & " TONE=" & toneWord & " DISPOSITION=" & str(npcs(idx).disposition) & " HISTORY=" & npcs(idx).history)

                screenlock
                line (0, 390)-(799, 599), rgb(20,20,30), bf
                screenunlock
                show_reply npcs(idx).nombre, reply
            else
                ' accuse
                dim usrPrompt as string = "The player presents these clues: Ren saw Kael at the armory; Bora knows a secret passage; something clanks in the inn cellar. The sword Ashenbrand is hidden in the inn cellar, reached by the secret passage, and Kael was seen at the armory. Confront Kael and give the outcome, in character."
                dim reply as string = generate_reply(npcs(idx).persona, usrPrompt, npcs(idx).history, 0, npcs(idx).disposition)
                show_reply npcs(idx).nombre, reply
                screenlock
                line (0, 0)-(799, 599), rgb(20, 20, 30), bf
                draw string (100, 200), "Ashenbrand is recovered from the inn cellar."
                draw string (100, 220), "Kael is taken."
                draw string (100, 240), "Willowbrook is safe."
                draw string (100, 300), "Press any key..."
                screenunlock
                if demoMode = 1 then
                    sleep 800
                else
                    while inkey <> "": wend
                    sleep
                    while inkey <> "": wend
                end if
                endingDone = 1
                done = 1
            end if
        elseif k = chr(27) or k = "q" or k = "Q" then
            done = 1
        end if
        sleep 16
    wend
    while inkey <> "": wend
end sub

' ---------- main ----------
randomize timer
screenres SC_W, SC_H, 32
windowtitle "RPG Village - Raon-1-nano"

init_npcs()

screenlock
cls
draw string (40, 40), "Willowbrook"
draw string (40, 70), "Loading model..."
screenunlock

llm_init()

for ai as integer = 1 to 10
    if command(ai) = "--log" or command(ai) = "-log" then logging = 1
    if command(ai) = "--demo" then demoMode = 1
next
if logging = 1 then
    dim f0 as integer = freefile
    open "rpg_village.log" for output as #f0
    close #f0
end if

dim introText as string
introText = "The kingdom's pride, the sword Ashenbrand, has vanished from Highgarden Keep." & chr(10)
introText = introText & "King Aldric sends you to the village of Willowbrook." & chr(10)
introText = introText & "The villagers whisper of a stranger, a dragon stirring, and a hidden treasure." & chr(10)
introText = introText & "Speak to the people. Uncover clues. Find the sword before Vaelgor wakes."

dim introShown as integer = 0

dim px as integer = 400
dim py as integer = 300
dim lastSpace as integer = 0
dim lastJ as integer = 0

do
    if multikey(FB.SC_LEFT)  then px -= 3
    if multikey(FB.SC_RIGHT) then px += 3
    if multikey(FB.SC_UP)    then py -= 3
    if multikey(FB.SC_DOWN)  then py += 3
    if px < 20 then px = 20
    if px > SC_W-20 then px = SC_W-20
    if py < 20 then py = 20
    if py > SC_H-20 then py = SC_H-20

    dim spaceNow as integer = multikey(FB.SC_SPACE)
    if spaceNow and not lastSpace then
        for i as integer = 0 to nNpc-1
            if dist(px, py, npcs(i).x, npcs(i).y) < 45 then
                talk_to(i)
                exit for
            end if
        next
    end if
    lastSpace = spaceNow

    dim jNow as integer = multikey(FB.SC_J)
    if jNow and not lastJ then
        show_journal()
    end if
    lastJ = jNow

    if demoMode = 1 and demoStarted = 0 then
        demoStarted = 1
        for di as integer = 0 to nNpc-1
            talk_to(di)
        next
        show_journal()
    end if

    screenlock
    line (0,0)-(SC_W-1, SC_H-1), rgb(50,110,50), bf
    line (0, 300)-(SC_W-1, 340), rgb(180,150,100), bf
    for hx as integer = 80 to 720 step 180
        line (hx, 80)-(hx+70, 140), rgb(130,90,60), bf
        line (hx-5, 70)-(hx+75, 85), rgb(160,60,60), bf
        line (hx+25, 100)-(hx+45, 140), rgb(60,40,20), bf
    next
    for tx as integer = 40 to SC_W-40 step 90
        circle (tx, 500), 18, rgb(30,90,30), , , , f
        line (tx-3, 500)-(tx+3, 535), rgb(90,60,30), bf
    next
    for i as integer = 0 to nNpc-1
        line (npcs(i).x-6, npcs(i).y)-(npcs(i).x+6, npcs(i).y+16), npcs(i).col, bf
        circle (npcs(i).x, npcs(i).y-8), 8, npcs(i).col, , , , f
        draw string (npcs(i).x-30, npcs(i).y-30), npcs(i).nombre
    next
    line (px-6, py)-(px+6, py+16), rgb(255,220,60), bf
    circle (px, py-8), 8, rgb(255,220,60), , , , f
    draw string (px-14, py-30), "You"
    draw string (10, 10), "Arrow keys move, Space talk, J journal, Q quit"
    screenunlock

    if introShown = 0 then
        show_reply "Willowbrook", introText
        introShown = 1
    end if

    sleep 16
loop until multikey(FB.SC_Q)

llama_sampler_free(g_smp)
llama_free(g_ctx)
llama_free_model(g_model)
end
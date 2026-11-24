' animal_tank.bas - autonomous wildlife tank demo using raon-1-nano via llama.cpp
' A "fish tank" where instead of fish there are land animals that decide their own
' next action by prompting the model. Arrow keys pan the viewport.
' Build: see build_animal_tank.sh / run_animal_tank.sh

#include once "fbgfx.bi"
#include once "llama.bi"

extern "C"
    declare function llama_sampler_init_penalties(byval n_vocab as long, byval penalty_last_n as long, byval penalty_repeat as single, byval penalty_freq as single, byval penalty_present as single) as llama_sampler ptr
end extern

const SC_W = 800
const SC_H = 600
const WORLD_W = 2600
const WORLD_H = 1900
const VISION = 260
const DECIDE_INTERVAL = 2.0
const MAX_ANIMALS = 80
const MAX_CORPSES = 100

' ---------- world data ----------
type Pond
    x as single
    y as single
    r as single
end type

type Food
    x as single
    y as single
    active as integer
    regen as double
end type

type Corpse
    active as integer
    x as single
    y as single
    food as integer
    expire as double
end type

type Animal
    active as integer
    species as string
    name as string
    trait as string
    x as single
    y as single
    vx as single
    vy as single
    hunger as single
    thirst as single
    state as string
    moveTimer as double
    lastDecide as double
    lastAction as string
    actionTime as double
    mateCooldown as double
    energy as single
    sleepTimer as double
    hp as single
    attackedTimer as double
    goal as string
    lastOutcome as string
    mem(0 to 2) as string
    attacker as integer
    size as single
    col as uinteger
end type

dim shared ponds(0 to 6) as Pond
dim shared nPonds as integer = 1
dim shared foods(0 to 119) as Food
dim shared nFoods as integer = 120
dim shared animals(0 to MAX_ANIMALS-1) as Animal
dim shared selectedAnimal as integer = -1
dim shared logging as integer = 0
dim shared demoMode as integer = 0
dim shared nAnimals as integer = 0
dim shared corpses(0 to MAX_CORPSES-1) as Corpse
dim shared nCorpses as integer = 0

' ---------- LLM globals ----------
dim shared g_model as llama_model ptr
dim shared g_ctx as llama_context ptr
dim shared g_smp as llama_sampler ptr
dim shared g_tmpl as const zstring ptr
dim shared g_vocab as const llama_vocab ptr

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
    g_tmpl = llama_model_chat_template(g_model, 0)

    dim sparams as llama_sampler_chain_params = llama_sampler_chain_default_params()
    g_smp = llama_sampler_chain_init(sparams)
    llama_sampler_chain_add(g_smp, llama_sampler_init_penalties(llama_vocab_n_tokens(g_vocab), 64, 1.0, 0.0, 1.5))
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_k(40))
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_p(0.9, 1))
    llama_sampler_chain_add(g_smp, llama_sampler_init_temp(0.12))
    llama_sampler_chain_add(g_smp, llama_sampler_init_dist(42))
end sub

sub rebuild_sampler(temp as single)
    if g_smp <> 0 then
        llama_sampler_free(g_smp)
        g_smp = 0
    end if
    dim sparams as llama_sampler_chain_params = llama_sampler_chain_default_params()
    g_smp = llama_sampler_chain_init(sparams)
    llama_sampler_chain_add(g_smp, llama_sampler_init_penalties(llama_vocab_n_tokens(g_vocab), 64, 1.0, 0.0, 1.5))
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_k(40))
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_p(0.9, 1))
    llama_sampler_chain_add(g_smp, llama_sampler_init_temp(temp))
    llama_sampler_chain_add(g_smp, llama_sampler_init_dist(42))
end sub

function generate_reply(sysPrompt as string, usrPrompt as string) as string
    llama_memory_clear(llama_get_memory(g_ctx), true)

    dim fullSys as string = sysPrompt
    fullSys = fullSys & chr(10) & "Use only ASCII characters in your reply. Reply with exactly one line, no explanation."
    dim prompt as string
    prompt = "<|im_start|>system" & chr(10) & fullSys & chr(10) & "<|im_end|>" & chr(10)
    prompt = prompt & "<|im_start|>user" & chr(10) & usrPrompt & chr(10) & "<|im_end|>" & chr(10)
    prompt = prompt & "ASSISTANT_TOKENassistant" & chr(10) & " <think> </think> " & chr(10) & chr(10)

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
    while nGen < 120
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
    ' strip any thinking block
    dim tstart as integer = instr(lcase(filtered), "<think>")
    if tstart > 0 then
        dim tend as integer = instr(tstart, lcase(filtered), "</think>")
        if tend > 0 then
            filtered = mid(filtered, tend + len("</think>"))
        else
            ' no closing tag; drop everything up to end
            filtered = ""
        end if
    end if
    dim nl as integer = instr(filtered, chr(10))
    if nl > 0 then filtered = left(filtered, nl-1)
    return trim(filtered)
end function

' ---------- helpers ----------
function dist(ax as single, ay as single, bx as single, by as single) as single
    return sqr((ax-bx)^2 + (ay-by)^2)
end function

declare sub assign_identity(slot as integer)
declare sub spawn_animal(sp as string, bx as single, by as single)

sub init_world()
    ' one large watering hole
    ponds(0).x=WORLD_W/2: ponds(0).y=WORLD_H/2: ponds(0).r=200

    randomize timer
    for i as integer = 0 to nFoods-1
        foods(i).x = 60 + rnd*(WORLD_W-120)
        foods(i).y = 60 + rnd*(WORLD_H-120)
        foods(i).active = 1
        foods(i).regen = 0
    next

    dim speciesList(0 to 7) as string = {"lion","cheetah","hyena","gazelle","zebra","wildebeest","elephant","giraffe"}
    dim sizes(0 to 7) as single = {18, 14, 15, 10, 13, 14, 26, 22}
    dim cols(0 to 7) as uinteger = {rgb(210,170,90), rgb(220,180,70), rgb(120,100,80), rgb(200,170,120), rgb(235,235,235), rgb(90,70,50), rgb(150,150,150), rgb(200,170,90)}
    ' species mix: more prey than predators
    dim initSpecies(0 to 13) as string = {"lion","lion","cheetah","hyena", _
        "gazelle","gazelle","gazelle","gazelle","zebra","zebra", _
        "wildebeest","wildebeest","elephant","giraffe"}
    ' herd centers for prey clustering
    dim herd1x as single = WORLD_W*0.35
    dim herd1y as single = WORLD_H*0.4
    dim herd2x as single = WORLD_W*0.65
    dim herd2y as single = WORLD_H*0.6
    for i as integer = 0 to 13
        dim spName as string = initSpecies(i)
        ' find index in speciesList
        dim sp as integer = 0
        for t as integer = 0 to 7
            if speciesList(t)=spName then sp=t
        next
        animals(i).active = 1
        animals(i).species = spName
        if spName="lion" or spName="cheetah" or spName="hyena" then
            animals(i).x = 100 + rnd*(WORLD_W-200)
            animals(i).y = 100 + rnd*(WORLD_H-200)
        else
            ' cluster prey around a herd center
            dim cx as single, cy as single
            if rnd < 0.5 then
                cx = herd1x: cy = herd1y
            else
                cx = herd2x: cy = herd2y
            end if
            animals(i).x = cx + (rnd*120 - 60)
            animals(i).y = cy + (rnd*120 - 60)
        end if
        if animals(i).x < 20 then animals(i).x = 20
        if animals(i).y < 20 then animals(i).y = 20
        if animals(i).x > WORLD_W-20 then animals(i).x = WORLD_W-20
        if animals(i).y > WORLD_H-20 then animals(i).y = WORLD_H-20
        animals(i).vx = 0
        animals(i).vy = 0
        animals(i).hunger = 20 + rnd*40
        animals(i).thirst = 20 + rnd*40
        animals(i).state = "idle"
        animals(i).moveTimer = 0
        animals(i).lastDecide = 0
        animals(i).lastAction = ""
        animals(i).actionTime = 0
        animals(i).mateCooldown = 0
        animals(i).energy = 80 + rnd*20
        animals(i).sleepTimer = 0
        animals(i).hp = 100
        animals(i).attackedTimer = 0
        animals(i).attacker = -1
        animals(i).goal = ""
        animals(i).lastOutcome = ""
        animals(i).mem(0) = ""
        animals(i).mem(1) = ""
        animals(i).mem(2) = ""
        animals(i).size = sizes(sp)
        animals(i).col = cols(sp)
        assign_identity(i)
    next
    nAnimals = 14
    spawn_animal("crocodile", ponds(0).x, ponds(0).y)
    if nAnimals < 15 then nAnimals = 15
    for i as integer = 0 to MAX_CORPSES-1
        corpses(i).active = 0
    next
end sub

function species_behavior(sp as string) as string
    select case sp
        case "lion": return "You are a lion, an apex predator. You hunt gazelles, zebras and wildebeest. You rest often and dominate the watering hole. You are bold. You target isolated prey when possible. You always drink when thirsty before hunting."
        case "cheetah": return "You are a cheetah, the fastest predator. You hunt gazelles. You avoid lions. You are alert and quick."
        case "hyena": return "You are a hyena. You scavenge and hunt in groups, targeting weak prey. You are opportunistic."
        case "gazelle": return "You are a gazelle. You graze on grass and drink only at water. You strongly prefer to stay within a few body-lengths of your herd mates; being alone is dangerous. You flee immediately from lions, cheetahs and hyenas. Grouping is your main defense."
        case "zebra": return "You are a zebra. You graze and drink at water. You stay tightly with your herd and are very wary of predators; you flee from lions, cheetahs and hyenas."
        case "wildebeest": return "You are a wildebeest. You graze in dense herds and move toward water together. You avoid predators and stay with the group."
        case "elephant": return "You are an elephant. You drink deeply at the watering hole and eat plants. You are large and mostly unthreatened; others give you space."
        case "giraffe": return "You are a giraffe. You eat leaves from tall trees and drink carefully at water. You are calm and gentle."
        case "crocodile": return "You are a crocodile, an ambush predator of the watering hole. You live in the water and almost never leave it. You float quietly near the edge and snap at animals that come close to drink. You do not chase prey across the land; you are slow on land and quick in the water. You always return to the water. You fear nothing in the tank."
    end select
    return "You are an animal."
end function

function trait_sentence(trait as string) as string
    select case lcase(trim(trait))
    case "bold": return "You are bold. You act fast, take risks, and rarely hesitate."
    case "cautious": return "You are cautious. You keep your distance from danger and think before you act."
    case "lazy": return "You are lazy. You prefer resting and staying put unless a need is pressing."
    case "curious": return "You are curious. You like wandering and investigating what you see."
    case "greedy": return "You are greedy. You eat whenever you get the chance, even when you are not very hungry."
    case "restless": return "You are restless. You dislike staying still and move often."
    case "calm": return "You are calm. You are slow to panic and steady in your habits."
    case "social": return "You are social. You prefer to stay close to others of your kind."
    end select
    return ""
end function

function species_name_pool_str(sp as string) as string
    select case sp
    case "lion": return "Simba|Nala|Mufasa|Zuri|Kito|Asha|Jengo|Sarabi|Kofi|Tamasha|Jabari|Amara|Nia|Sefu|Kesi|Zola|Dakarai|Eshe|Jelani|Kamaria|Odion|Safara|Taji|Zalika"
    case "cheetah": return "Dash|Zola|Kip|Sefu|Amara|Jengo|Nia|Tamba|Zane|Kesi|Asha|Jabari|Kofi|Zuri|Eshe|Jelani|Kamaria|Odion|Safara|Taji|Zalika|Dakarai|Bibi|Kito"
    case "hyena": return "Shenzi|Banzai|Ed|Kamaria|Jabari|Nia|Tamba|Zola|Kito|Asha|Sefu|Kesi|Zuri|Amara|Jengo|Kofi|Eshe|Jelani|Odion|Safara|Taji|Zalika|Dakarai|Bibi"
    case "gazelle": return "Zara|Kito|Nia|Sefu|Amara|Jengo|Kesi|Tamba|Zuri|Bibi|Asha|Zola|Jabari|Kofi|Eshe|Jelani|Kamaria|Odion|Safara|Taji|Zalika|Dakarai|Lulu|Penda"
    case "zebra": return "Zane|Zola|Kito|Nia|Stripes|Tamba|Asha|Jengo|Kesi|Sefu|Zuri|Amara|Jabari|Kofi|Bibi|Eshe|Jelani|Kamaria|Odion|Safara|Taji|Zalika|Dakarai|Penda"
    case "wildebeest": return "Gnu|Kofi|Tamba|Zuri|Jengo|Nia|Bibi|Sefu|Asha|Kito|Amara|Zola|Jabari|Kesi|Eshe|Jelani|Kamaria|Odion|Safara|Taji|Zalika|Dakarai|Penda|Bomani"
    case "elephant": return "Tembo|Nia|Jengo|Zuri|Kofi|Asha|Bibi|Tamba|Sefu|Kesi|Amara|Jabari|Zola|Kito|Eshe|Jelani|Kamaria|Odion|Safara|Taji|Zalika|Dakarai|Penda|Bomani"
    case "giraffe": return "Twiga|Zuri|Kito|Nia|Jengo|Asha|Tamba|Kesi|Sefu|Bibi|Amara|Zola|Jabari|Kofi|Eshe|Jelani|Kamaria|Odion|Safara|Taji|Zalika|Dakarai|Penda|Lulu"
    case "crocodile": return "Sobek|Snap|Kroko|Jabari|Zuri|Tamba|Kesi|Sefu|Nia|Kito|Asha|Jengo|Zola|Amara|Kofi|Bibi|Eshe|Jelani|Kamaria|Odion|Safara|Taji|Zalika|Dakarai"
    end select
    return ""
end function

sub assign_identity(slot as integer)
    dim sp as string = animals(slot).species
    ' trait
    dim traitKeys(0 to 7) as string = {"bold","cautious","lazy","curious","greedy","restless","calm","social"}
    animals(slot).trait = traitKeys(int(rnd*8))
    ' name pool
    dim poolStr as string = species_name_pool_str(sp)
    dim pool(0 to 31) as string
    dim nPool as integer = 0
    dim cur as string = ""
    for k as integer = 1 to len(poolStr)
        dim ch as string = mid(poolStr, k, 1)
        if ch = "|" then
            pool(nPool) = cur
            nPool += 1
            cur = ""
        else
            cur += ch
        end if
    next
    if cur <> "" then
        pool(nPool) = cur
        nPool += 1
    end if
    ' check used names among living same species (excluding slot)
    dim used(0 to MAX_ANIMALS-1) as string
    dim nUsed as integer = 0
    for j as integer = 0 to MAX_ANIMALS-1
        if j = slot then continue for
        if animals(j).active = 1 and animals(j).species = sp then
            used(nUsed) = animals(j).name
            nUsed += 1
        end if
    next
    ' collect unused pool indices, then pick uniformly at random
    dim chosen as string = ""
    dim unusedIdx(0 to 31) as integer
    dim nUnused as integer = 0
    for p as integer = 0 to nPool-1
        dim inUse as integer = 0
        for u as integer = 0 to nUsed-1
            if used(u) = pool(p) then inUse = 1
        next
        if inUse = 0 then
            unusedIdx(nUnused) = p
            nUnused += 1
        end if
    next
    if nUnused > 0 then
        chosen = pool(unusedIdx(int(rnd*nUnused)))
    end if
    if chosen = "" then
        ' suffix fallback
        dim baseName as string = pool(int(rnd*nPool))
        dim suffix as integer = 2
        do
            chosen = baseName & " " & str(suffix)
            dim inUse2 as integer = 0
            for u as integer = 0 to nUsed-1
                if used(u) = chosen then inUse2 = 1
            next
            ' also check against animals directly for safety
            for j as integer = 0 to MAX_ANIMALS-1
                if j = slot then continue for
                if animals(j).active = 1 and animals(j).species = sp and animals(j).name = chosen then inUse2 = 1
            next
            if inUse2 = 0 then exit do
            suffix += 1
        loop
    end if
    animals(slot).name = chosen
end sub

function species_cap(sp as string) as integer
    select case sp
        case "lion": return 4
        case "cheetah": return 3
        case "hyena": return 4
        case "gazelle": return 14
        case "zebra": return 10
        case "wildebeest": return 12
        case "elephant": return 5
        case "giraffe": return 5
        case "crocodile": return 1
    end select
    return 6
end function

function is_predator(sp as string) as integer
    if sp="lion" or sp="cheetah" or sp="hyena" or sp="crocodile" then return 1
    return 0
end function

function damage_mult(sp as string) as single
    select case sp
        case "elephant": return 0.25
        case "giraffe": return 0.75
    end select
    return 1.0
end function

function species_temp(sp as string) as single
    select case sp
        case "elephant": return 0.10
        case "giraffe": return 0.11
        case "lion": return 0.12
        case "wildebeest": return 0.12
        case "hyena": return 0.13
        case "zebra": return 0.14
        case "cheetah": return 0.14
        case "gazelle": return 0.14
        case "crocodile": return 0.13
    end select
    return 0.12
end function

function species_size(sp as string) as single
    select case sp
        case "lion": return 18
        case "cheetah": return 14
        case "hyena": return 15
        case "gazelle": return 10
        case "zebra": return 13
        case "wildebeest": return 14
        case "elephant": return 26
        case "giraffe": return 22
        case "crocodile": return 20
    end select
    return 12
end function

function species_color(sp as string) as uinteger
    select case sp
        case "lion": return rgb(210,170,90)
        case "cheetah": return rgb(220,180,70)
        case "hyena": return rgb(120,100,80)
        case "gazelle": return rgb(240,225,200)
        case "zebra": return rgb(235,235,235)
        case "wildebeest": return rgb(90,70,50)
        case "elephant": return rgb(150,150,150)
        case "giraffe": return rgb(200,170,90)
        case "crocodile": return rgb(70,95,55)
    end select
    return rgb(200,200,200)
end function

function count_species(sp as string) as integer
    dim c as integer = 0
    for i as integer = 0 to MAX_ANIMALS-1
        if animals(i).active=1 and animals(i).species=sp then c+=1
    next
    return c
end function

sub spawn_animal(sp as string, bx as single, by as single)
    dim slot as integer = -1
    for i as integer = 0 to MAX_ANIMALS-1
        if animals(i).active=0 then
            slot=i
            exit for
        end if
    next
    if slot=-1 then
        if nAnimals < MAX_ANIMALS then
            slot=nAnimals
            nAnimals+=1
        else
            return
        end if
    end if
    animals(slot).active=1
    animals(slot).species=sp
    animals(slot).x=bx + (rnd*40-20)
    animals(slot).y=by + (rnd*40-20)
    if animals(slot).x<20 then animals(slot).x=20
    if animals(slot).y<20 then animals(slot).y=20
    if animals(slot).x>WORLD_W-20 then animals(slot).x=WORLD_W-20
    if animals(slot).y>WORLD_H-20 then animals(slot).y=WORLD_H-20
    animals(slot).vx=0
    animals(slot).vy=0
    animals(slot).hunger=10
    animals(slot).thirst=10
    animals(slot).state="idle"
    animals(slot).moveTimer=0
    animals(slot).lastDecide=timer + rnd*2
    animals(slot).lastAction="born"
    animals(slot).actionTime=timer
    animals(slot).mateCooldown=timer+8
    animals(slot).energy=90
    animals(slot).sleepTimer=0
    animals(slot).hp=100
    animals(slot).attackedTimer=0
    animals(slot).attacker=-1
    animals(slot).goal=""
    animals(slot).lastOutcome=""
    animals(slot).mem(0)=""
    animals(slot).mem(1)=""
    animals(slot).mem(2)=""
    animals(slot).size=species_size(sp)
    animals(slot).col=species_color(sp)
    assign_identity(slot)
end sub

function dir_from_delta(dx as single, dy as single) as string
    dim s as string = ""
    if dy < -10 then s += "N"
    if dy > 10 then s += "S"
    if dx > 10 then s += "E"
    if dx < -10 then s += "W"
    if s = "" then s = "N"
    return s
end function

function dir_name(dx as single, dy as single) as string
    dim ang as single = atan2(dy, dx)  ' radians, 0=E, positive CCW? FB atan2(y,x)
    ' map to 8 compass directions
    dim deg as single = ang * 57.2958
    if deg < 0 then deg += 360
    if deg >= 337.5 or deg < 22.5 then return "E"
    if deg >= 22.5 and deg < 67.5 then return "SE"
    if deg >= 67.5 and deg < 112.5 then return "S"
    if deg >= 112.5 and deg < 157.5 then return "SW"
    if deg >= 157.5 and deg < 202.5 then return "W"
    if deg >= 202.5 and deg < 247.5 then return "NW"
    if deg >= 247.5 and deg < 292.5 then return "N"
    return "NE"
end function

function dist_desc(d as single) as string
    if d < 90 then return "near"
    if d < 180 then return "far"
    return "very far"
end function

function need_word(need as string, value as integer) as string
    dim band as integer
    if value >= 80 then
        band = 0
    elseif value >= 60 then
        band = 1
    elseif value >= 40 then
        band = 2
    elseif value >= 20 then
        band = 3
    elseif value >= 1 then
        band = 4
    else
        band = 5
    end if
    select case need
        case "fullness"
            select case band
                case 0: return "well fed"
                case 1: return "satisfied"
                case 2: return "peckish"
                case 3: return "hungry"
                case 4: return "very hungry"
                case else: return "starving"
            end select
        case "hydration"
            select case band
                case 0: return "well hydrated"
                case 1: return "hydrated"
                case 2: return "a bit thirsty"
                case 3: return "thirsty"
                case 4: return "very thirsty"
                case else: return "dying of thirst"
            end select
        case "energy"
            select case band
                case 0: return "energetic"
                case 1: return "steady"
                case 2: return "tiring"
                case 3: return "tired"
                case 4: return "exhausted"
                case else: return "exhausted"
            end select
        case "health"
            select case band
                case 0: return "healthy"
                case 1: return "a little hurt"
                case 2: return "hurt"
                case 3: return "badly hurt"
                case 4: return "near death"
                case else: return "near death"
            end select
    end select
    return ""
end function

function build_vision(idx as integer) as string
    dim viewerPred as integer = is_predator(animals(idx).species)
    dim isPreyViewer as integer = iif(viewerPred = 1, 0, 1)

    ' collect candidates within VISION
    dim maxC as integer = 256
    dim candD(0 to 255) as single
    dim candE(0 to 255) as string
    dim candT(0 to 255) as integer
    dim nCand as integer = 0

    for j as integer = 0 to nAnimals-1
        if j = idx then continue for
        if animals(j).active = 0 then continue for
        ' crocodile is invisible to others while in the water, unless it recently
        ' attacked this viewer and is very close
        if animals(j).species = "crocodile" then
            dim _inw as integer = 0
            for _p as integer = 0 to nPonds-1
                if dist(animals(j).x, animals(j).y, ponds(_p).x, ponds(_p).y) < ponds(_p).r then _inw = 1
            next
            if _inw = 1 then
                dim _visible as integer = 0
                if animals(idx).attackedTimer > 0 and animals(idx).attacker = j then
                    if dist(animals(idx).x, animals(idx).y, animals(j).x, animals(j).y) < 60 then _visible = 1
                end if
                if _visible = 0 then continue for
            end if
        end if
        dim d as single = dist(animals(idx).x, animals(idx).y, animals(j).x, animals(j).y)
        if d > VISION then continue for
        dim threat as integer = 0
        if isPreyViewer = 1 and is_predator(animals(j).species) = 1 then threat = 1
        dim dx as integer = int(animals(j).x - animals(idx).x)
        dim dy as integer = int(animals(j).y - animals(idx).y)
        dim entry as string
        if threat = 1 then
            entry = "THREAT " & animals(j).species & " " & dir_name(dx, dy) & " " & dist_desc(d)
        else
            entry = animals(j).species & " " & dir_name(dx, dy) & " " & dist_desc(d)
        end if
        candD(nCand) = d
        candE(nCand) = entry
        candT(nCand) = threat
        nCand += 1
    next

    if isPreyViewer = 1 then
        for f as integer = 0 to nFoods-1
            if foods(f).active = 0 then continue for
            dim d as single = dist(animals(idx).x, animals(idx).y, foods(f).x, foods(f).y)
            if d > VISION then continue for
            dim dx as integer = int(foods(f).x - animals(idx).x)
            dim dy as integer = int(foods(f).y - animals(idx).y)
            dim entry as string = "food " & dir_name(dx, dy) & " " & dist_desc(d)
            candD(nCand) = d
            candE(nCand) = entry
            candT(nCand) = 0
            nCand += 1
        next
    end if

    ' sort: threats first, then others; each group by distance asc
    for a as integer = 0 to nCand - 2
        for b as integer = a + 1 to nCand - 1
            dim doSwap as integer = 0
            if candT(b) > candT(a) then
                doSwap = 1
            elseif candT(b) = candT(a) and candD(b) < candD(a) then
                doSwap = 1
            end if
            if doSwap = 1 then
                dim tmpD as single = candD(a)
                candD(a) = candD(b)
                candD(b) = tmpD
                dim tmpE as string = candE(a)
                candE(a) = candE(b)
                candE(b) = tmpE
                dim tmpT as integer = candT(a)
                candT(a) = candT(b)
                candT(b) = tmpT
            end if
        next
    next

    ' cap at 6
    if nCand > 6 then nCand = 6

    dim s as string = ""
    for i as integer = 0 to nCand - 1
        if s <> "" then s += "; "
        s += candE(i)
    next

    ' water: always show nearest watering hole
    dim bestP as integer = 0
    dim bestPD as single = 99999
    for p as integer = 0 to nPonds-1
        dim d as single = dist(animals(idx).x, animals(idx).y, ponds(p).x, ponds(p).y)
        if d < bestPD then
            bestPD = d
            bestP = p
        end if
    next
    for p as integer = 0 to nPonds-1
        if p <> bestP then continue for
        dim d as single = dist(animals(idx).x, animals(idx).y, ponds(p).x, ponds(p).y)
        dim dx as integer = int(ponds(p).x - animals(idx).x)
        dim dy as integer = int(ponds(p).y - animals(idx).y)
        dim waterStr as string = "water " & dir_name(dx, dy) & " " & dist_desc(d)
        if s <> "" then
            s = s & "; " & waterStr
        else
            s = waterStr
        end if
    next

    ' herd for non-predator viewers
    if isPreyViewer = 1 then
        dim sumX as single = 0
        dim sumY as single = 0
        dim cnt as integer = 0
        for j as integer = 0 to nAnimals-1
            if j = idx then continue for
            if animals(j).active = 0 then continue for
            if animals(j).species <> animals(idx).species then continue for
            dim d as single = dist(animals(idx).x, animals(idx).y, animals(j).x, animals(j).y)
            if d <= VISION then
                sumX += animals(j).x
                sumY += animals(j).y
                cnt += 1
            end if
        next
        if cnt >= 1 then
            dim cx as single = sumX / cnt
            dim cy as single = sumY / cnt
            dim dx as single = cx - animals(idx).x
            dim dy as single = cy - animals(idx).y
            dim hdir as string = dir_name(dx, dy)
            s = s & " | herd " & hdir
        end if
    end if

    if s = "" then s = "nothing nearby"
    return s
end function

function remove_substr(s as string, needle as string) as string
    dim r as string = s
    dim p as integer = instr(r, needle)
    while p > 0
        r = left(r, p-1) + mid(r, p+len(sub))
        p = instr(r, needle)
    wend
    return r
end function

sub apply_direction(idx as integer, d as string)
    dim sp as single = 0
    select case animals(idx).species
        case "gazelle": sp = 120
        case "cheetah": sp = 160
        case "zebra": sp = 95
        case "wildebeest": sp = 80
        case "lion": sp = 70
        case "hyena": sp = 90
        case "elephant": sp = 40
        case "giraffe": sp = 50
        case "crocodile"
            if dist(animals(idx).x, animals(idx).y, ponds(0).x, ponds(0).y) < ponds(0).r then
                sp = 100
            else
                sp = 30
            end if
    end select
    dim vx as single = 0, vy as single = 0
    if instr(d,"N")>0 then vy = -sp
    if instr(d,"S")>0 then vy = sp
    if instr(d,"E")>0 then vx = sp
    if instr(d,"W")>0 then vx = -sp
    if vx <> 0 and vy <> 0 then
        vx = vx * 0.7071
        vy = vy * 0.7071
    end if
    animals(idx).vx = vx
    animals(idx).vy = vy
    animals(idx).moveTimer = 1.2 + rnd*0.8
    animals(idx).state = "move"
end sub

sub log_msg(msg as string)
    if logging = 0 then return
    dim f as integer = freefile
    open "animal_tank.log" for append as #f
    print #f, msg
    close #f
end sub

sub push_event(idx as integer, text as string)
    if animals(idx).mem(2) = text then return
    animals(idx).mem(0) = animals(idx).mem(1)
    animals(idx).mem(1) = animals(idx).mem(2)
    animals(idx).mem(2) = text
end sub

sub decide_animal(idx as integer)
    dim dstr as string
    dim sysPrompt as string
    dim traitSent as string = trait_sentence(animals(idx).trait)
    dim nameInsert as string = " Your name is " & animals(idx).name & "."
    if traitSent <> "" then nameInsert &= " " & traitSent
    sysPrompt = species_behavior(animals(idx).species) & nameInsert & " " & _
        "You must choose exactly one action. Allowed actions: " & _
        "MOVE <dir> where dir is N,S,E,W,NE,NW,SE,SW (move that direction), " & _
        "EAT (eat: predators kill nearby prey; herbivores eat grass), " & _
        "DRINK (drink nearby water), MATE (mate with nearby same species if population allows), " & _
        "REST (stay and rest). " & _
        "Reply with only the action, e.g. MOVE NE or EAT."
    dim vis as string = build_vision(idx)
    dim touchWater as integer = 0
    for p as integer = 0 to nPonds-1
        if dist(animals(idx).x, animals(idx).y, ponds(p).x, ponds(p).y) <= ponds(p).r + 2 then
            touchWater = 1
        end if
    next
    dim usrPrompt as string
    dim attackedStr as string = "no"
    if animals(idx).attackedTimer > 0 then attackedStr = "yes"
    dim hadGoal as integer = 0
    if animals(idx).goal <> "" then hadGoal = 1
    dim isAttacked as integer = 0
    if animals(idx).attackedTimer > 0 then isAttacked = 1
    dim curGoal as string = animals(idx).goal
    if curGoal = "" then curGoal = "none"
    dim inWater as integer = 0
    for p as integer = 0 to nPonds-1
        if dist(animals(idx).x, animals(idx).y, ponds(p).x, ponds(p).y) < ponds(p).r then inWater = 1
    next
    dim lastResultStr as string = ""
    if animals(idx).lastOutcome <> "" then
        lastResultStr = "Last result: " & animals(idx).lastOutcome & ". "
    end if
    dim memJoined as string = ""
    for m as integer = 0 to 2
        if animals(idx).mem(m) <> "" then
            if memJoined <> "" then memJoined += "; "
            memJoined += animals(idx).mem(m)
        end if
    next
    dim recentStr as string = ""
    if memJoined <> "" then
        recentStr = "Recent events: " & memJoined & ". "
    end if
    dim waterSentence as string
    if animals(idx).species = "crocodile" then
        waterSentence = "The water is your home; stay in it or right at its edge. "
    else
        waterSentence = "Do not enter the water; drink from the edge. "
    end if
    usrPrompt = "State: you are " & need_word("fullness", int(animals(idx).hunger)) & " and " & need_word("hydration", int(animals(idx).thirst)) & "; your energy is " & need_word("energy", int(animals(idx).energy)) & " and you are " & need_word("health", int(animals(idx).hp)) & ". (fullness=" & str(int(animals(idx).hunger)) & "/100, hydration=" & str(int(animals(idx).thirst)) & "/100, energy=" & str(int(animals(idx).energy)) & "/100, health=" & str(int(animals(idx).hp)) & "/100) attacked=" & attackedStr & ", in water=" & iif(inWater=1,"yes","no") & ". " & _
        lastResultStr & recentStr & _
        "Current goal: " & curGoal & ". Stick to your current goal until it is done; change goals only if a predator is threatening you, you were attacked, or a need becomes urgent. " & _
        "Touching water: " & iif(touchWater=1,"yes","no") & ". " & _
        waterSentence & _
        iif(animals(idx).species = "crocodile" and inWater = 1, "You are hidden in the water; prey cannot see you. ", "") & _
        "You can see: " & vis & ". " & _
        "Stay in the central area near the watering hole; avoid map edges. " & _
        "Urgency: if you are thirsty or very thirsty, move toward water before anything else; if you are hungry or very hungry, seek food/prey before anything else. if you are badly hurt or near death, get away from predators and stay safe; you recover slowly while you are fed, watered, and not being attacked. " & _
        "What do you do? DRINK only if touching water."

    rebuild_sampler(species_temp(animals(idx).species))
    dim reply as string = generate_reply(sysPrompt, usrPrompt)
    dim goalPart as string = ""
    dim actionPart as string = reply
    dim gi as integer = instr(ucase(reply), "GOAL:")
    if gi > 0 then
        dim rest as string = mid(reply, gi + 5)
        dim semi as integer = instr(rest, ";")
        if semi > 0 then
            goalPart = trim(left(rest, semi - 1))
            dim ai as integer = instr(ucase(rest), "ACTION:")
            if ai > 0 then
                actionPart = trim(mid(rest, ai + 7))
            else
                actionPart = trim(mid(rest, semi + 1))
            end if
        else
            goalPart = trim(rest)
        end if
    end if
    dim explicitGoal as integer = 0
    if goalPart <> "" then
        animals(idx).goal = lcase(goalPart)
        explicitGoal = 1
    end if
    ' infer goal from action so LLM can change goals by choosing actions
    dim ua as string = ucase(actionPart)
    if explicitGoal = 0 and (hadGoal = 0 or isAttacked = 1) then
        if instr(ua, "DRINK") > 0 then
            animals(idx).goal = "drink"
        elseif instr(ua, "EAT") > 0 then
            animals(idx).goal = "eat"
        elseif instr(ua, "REST") > 0 or instr(ua, "SLEEP") > 0 then
            animals(idx).goal = "rest"
        elseif instr(ua, "MATE") > 0 then
            animals(idx).goal = "mate"
        end if
    end if
    if animals(idx).goal = "" then
        if animals(idx).thirst < 40 then
            animals(idx).goal = "drink"
        elseif animals(idx).hunger < 40 then
            animals(idx).goal = "eat"
        else
            animals(idx).goal = "wander"
        end if
    end if
    if animals(idx).goal = "" then
        if animals(idx).thirst < 40 then
            animals(idx).goal = "drink"
        elseif animals(idx).hunger < 40 then
            animals(idx).goal = "eat"
        else
            animals(idx).goal = "wander"
        end if
    end if
    dim u as string = ucase(actionPart)
    animals(idx).lastAction = actionPart
    animals(idx).actionTime = timer

    if instr(u,"EAT") > 0 then
        dim ate as integer = 0
        if is_predator(animals(idx).species)=1 then
            ' try corpse first
            for c as integer = 0 to MAX_CORPSES-1
                if corpses(c).active=1 then
                    if dist(animals(idx).x, animals(idx).y, corpses(c).x, corpses(c).y) < 28 then
                        corpses(c).food -= 1
                        if corpses(c).food <= 0 then corpses(c).active = 0
                        ate = 1
                        animals(idx).lastOutcome = "you ate from a corpse"
                        push_event(idx, "you ate from a corpse")
                        exit for
                    end if
                end if
            next
            ' otherwise attack prey
            if ate=0 then
                for j as integer = 0 to MAX_ANIMALS-1
                    if j=idx then continue for
                    if animals(j).active=0 then continue for
                    if is_predator(animals(j).species)=1 then continue for
                    if dist(animals(idx).x, animals(idx).y, animals(j).x, animals(j).y) < 35 then
                        animals(j).hp -= 25 * damage_mult(animals(j).species)
                        push_event(j, "you were attacked by a " & animals(idx).species)
                        if animals(j).hp <= 0 then
                            for c as integer = 0 to MAX_CORPSES-1
                                if corpses(c).active=0 then
                                    corpses(c).active=1
                                    corpses(c).x=animals(j).x
                                    corpses(c).y=animals(j).y
                                    corpses(c).food=5
                                    corpses(c).expire=timer+15
                                    exit for
                                end if
                            next
                            animals(j).active=0
                            animals(idx).lastOutcome = "you killed a " & animals(j).species & " and ate"
                            push_event(idx, "you killed a " & animals(j).species & " and ate")
                        else
                            animals(idx).lastOutcome = "you attacked a " & animals(j).species
                            push_event(idx, "you attacked a " & animals(j).species)
                        end if
                        ate=1
                        exit for
                    end if
                next
            end if
            if ate then
                animals(idx).hunger = animals(idx).hunger + 35
                if animals(idx).hunger > 100 then animals(idx).hunger = 100
            end if
        else
            ' herbivore eats grass
            for f as integer = 0 to nFoods-1
                if foods(f).active = 0 then continue for
                if dist(animals(idx).x, animals(idx).y, foods(f).x, foods(f).y) < 28 then
                    foods(f).active = 0
                    foods(f).regen = timer + 8 + rnd*6
                    ate = 1
                    animals(idx).lastOutcome = "you ate grass"
                    push_event(idx, "you ate grass")
                    exit for
                end if
            next
        end if
        if ate then
            animals(idx).state = "eat"
            animals(idx).lastAction = "EAT"
            if animals(idx).goal = "eat" then animals(idx).goal = ""
        else
            animals(idx).state = "idle"
            animals(idx).lastAction = "IDLE"
            if is_predator(animals(idx).species)=1 then
                animals(idx).lastOutcome = "you tried to eat, but no prey or corpse was within reach"
                push_event(idx, "you tried to eat, but no prey or corpse was within reach")
            else
                animals(idx).lastOutcome = "you tried to eat, but no grass was within reach"
                push_event(idx, "you tried to eat, but no grass was within reach")
            end if
        end if
        animals(idx).vx = 0: animals(idx).vy = 0
        animals(idx).moveTimer = 0.6
    elseif instr(u,"DRINK") > 0 then
        dim drank as integer = 0
        for p as integer = 0 to nPonds-1
            if dist(animals(idx).x, animals(idx).y, ponds(p).x, ponds(p).y) <= ponds(p).r + 2 then
                drank = 1
                exit for
            end if
        next
        if drank then
            animals(idx).thirst = animals(idx).thirst + 40
            if animals(idx).thirst > 100 then animals(idx).thirst = 100
            animals(idx).state = "drink"
            animals(idx).lastAction = "DRINK"
            if animals(idx).goal = "drink" then animals(idx).goal = ""
            animals(idx).lastOutcome = "you drank"
            push_event(idx, "you drank")
        else
            animals(idx).state = "idle"
            animals(idx).lastAction = "IDLE"
            animals(idx).lastOutcome = "you tried to drink, but you were not at the water's edge"
            push_event(idx, "you tried to drink, but you were not at the water's edge")
        end if
        animals(idx).vx = 0: animals(idx).vy = 0
        animals(idx).moveTimer = 0.6
    elseif instr(u,"MATE") > 0 then
        dim mated as integer = 0
        if timer >= animals(idx).mateCooldown and count_species(animals(idx).species) < species_cap(animals(idx).species) then
            for j as integer = 0 to MAX_ANIMALS-1
                if j=idx then continue for
                if animals(j).active=0 then continue for
                if animals(j).species <> animals(idx).species then continue for
                if dist(animals(idx).x, animals(idx).y, animals(j).x, animals(j).y) < 40 then
                    spawn_animal animals(idx).species, (animals(idx).x+animals(j).x)/2, (animals(idx).y+animals(j).y)/2
                    animals(idx).mateCooldown = timer + 12
                    animals(j).mateCooldown = timer + 12
                    mated = 1
                    exit for
                end if
            next
        end if
        if mated then
            animals(idx).state = "mate"
            animals(idx).lastOutcome = "you mated and a baby " & animals(idx).species & " was born"
            push_event(idx, "you mated and a baby " & animals(idx).species & " was born")
            if animals(idx).goal = "mate" then animals(idx).goal = ""
        else
            animals(idx).state = "idle"
            animals(idx).lastOutcome = "you tried to mate, but it did not happen"
            push_event(idx, "you tried to mate, but it did not happen")
        end if
        animals(idx).vx = 0: animals(idx).vy = 0
        animals(idx).moveTimer = 0.6
    elseif instr(u,"REST") > 0 then
        animals(idx).state = "rest"
        animals(idx).lastOutcome = "you rested"
        animals(idx).vx = 0: animals(idx).vy = 0
        animals(idx).moveTimer = 1.5
    else
        dstr = ""
        dim mi as integer = instr(u, "MOVE")
        if mi > 0 then
            dim rest as string = trim(mid(u, mi + 4))
            dim sp as integer = instr(rest, " ")
            if sp > 0 then rest = left(rest, sp - 1)
            dstr = trim(rest)
        end if
        if dstr = "" then
            if instr(u, "NW") > 0 then
                dstr = "NW"
            elseif instr(u, "NE") > 0 then
                dstr = "NE"
            elseif instr(u, "SW") > 0 then
                dstr = "SW"
            elseif instr(u, "SE") > 0 then
                dstr = "SE"
            elseif instr(u, "N") > 0 then
                dstr = "N"
            elseif instr(u, "S") > 0 then
                dstr = "S"
            elseif instr(u, "W") > 0 then
                dstr = "W"
            elseif instr(u, "E") > 0 then
                dstr = "E"
            end if
        end if
        if dstr = "" then dstr = "N"
        if animals(idx).x > WORLD_W - 100 then dstr = remove_substr(dstr, "E")
        if animals(idx).x < 100 then dstr = remove_substr(dstr, "W")
        if animals(idx).y > WORLD_H - 100 then dstr = remove_substr(dstr, "S")
        if animals(idx).y < 100 then dstr = remove_substr(dstr, "N")
        if dstr = "" then dstr = "N"
        animals(idx).lastAction = "MOVE " & dstr
        apply_direction(idx, dstr)
    end if

    log_msg("DECIDE " & animals(idx).name & " (" & animals(idx).species & ") idx=" & str(idx) & " pos=" & str(int(animals(idx).x)) & "," & str(int(animals(idx).y)) & chr(10) & _
           "SYS: " & sysPrompt & chr(10) & _
           "USR: " & usrPrompt & chr(10) & _
           "GOAL: " & animals(idx).goal & chr(10) & _
           "REPLY: " & reply & chr(10) & _
           "STATE: " & animals(idx).state & " ACTION: " & animals(idx).lastAction)
    animals(idx).lastDecide = timer
end sub

' ---------- drawing ----------
sub world_to_screen(wx as single, wy as single, camX as single, camY as single, byref sx as integer, byref sy as integer)
    sx = int(wx - camX)
    sy = int(wy - camY)
end sub

sub draw_animal(a as Animal, camX as single, camY as single)
    if a.active=0 then return
    dim sx as integer, sy as integer
    world_to_screen a.x, a.y, camX, camY, sx, sy
    if sx < -40 or sx > SC_W+40 or sy < -40 or sy > SC_H+40 then return
    dim s as single = a.size
    select case a.species
        case "lion"
            ' body
            circle (sx, sy), s*0.95, a.col, , , 0.6, f
            ' legs
            line (sx-s*0.5, sy+s*0.4)-(sx-s*0.5, sy+s*1.2), a.col
            line (sx+s*0.1, sy+s*0.4)-(sx+s*0.1, sy+s*1.2), a.col
            line (sx+s*0.5, sy+s*0.4)-(sx+s*0.5, sy+s*1.2), a.col
            ' tail
            line (sx-s*0.8, sy)-(sx-s*1.5, sy-s*0.3), a.col
            circle (sx-s*1.5, sy-s*0.3), s*0.18, rgb(60,40,20), , , , f
            ' mane
            circle (sx+s*0.6, sy-s*0.3), s*1.05, rgb(120,70,20), , , , f
            ' head
            circle (sx+s*0.6, sy-s*0.3), s*0.55, a.col, , , , f
            ' face
            circle (sx+s*0.85, sy-s*0.25), s*0.08, rgb(40,30,20), , , , f
            circle (sx+s*0.85, sy-s*0.45), s*0.08, rgb(40,30,20), , , , f
        case "cheetah"
            circle (sx, sy), s*0.85, a.col, , , 0.55, f
            line (sx-s*0.4, sy+s*0.35)-(sx-s*0.4, sy+s*1.1), a.col
            line (sx+s*0.2, sy+s*0.35)-(sx+s*0.2, sy+s*1.1), a.col
            line (sx+s*0.6, sy+s*0.35)-(sx+s*0.6, sy+s*1.1), a.col
            circle (sx+s*0.75, sy-s*0.2), s*0.4, a.col, , , , f
            ' spots
            for t as integer = 0 to 6
                dim ax as single = sx + (t-3)*s*0.28
                dim ay as single = sy + (t mod 3 -1)*s*0.2
                circle (ax, ay), 2.2, rgb(40,30,20), , , , f
            next
            ' tail
            line (sx-s*0.7, sy)-(sx-s*1.4, sy-s*0.5), rgb(60,40,20)
            for t as integer = 0 to 3
                circle (sx-s*1.0-t*s*0.12, sy-s*0.35-t*s*0.05), 1.6, rgb(40,30,20), , , , f
            next
            ' ears
            line (sx+s*0.6, sy-s*0.45)-(sx+s*0.75, sy-s*0.85), a.col
            line (sx+s*0.9, sy-s*0.45)-(sx+s*0.75, sy-s*0.85), a.col
        case "hyena"
            circle (sx, sy), s*0.9, a.col, , , 0.5, f
            line (sx-s*0.5, sy+s*0.4)-(sx-s*0.5, sy+s*1.1), a.col
            line (sx, sy+s*0.45)-(sx, sy+s*1.1), a.col
            line (sx+s*0.5, sy+s*0.4)-(sx+s*0.5, sy+s*1.1), a.col
            circle (sx+s*0.7, sy-s*0.25), s*0.42, a.col, , , , f
            ' sloped back
            line (sx-s*0.8, sy-s*0.5)-(sx+s*0.7, sy-s*0.1), rgb(90,70,50)
            ' ears
            circle (sx+s*0.5, sy-s*0.8), s*0.18, a.col, , , , f
            circle (sx+s*0.85, sy-s*0.75), s*0.18, a.col, , , , f
            ' spots
            circle (sx-s*0.2, sy-0.1*s), 2.5, rgb(60,40,20), , , , f
            circle (sx+s*0.1, sy+0.2*s), 2.5, rgb(60,40,20), , , , f
        case "gazelle"
            circle (sx, sy), s*0.7, a.col, , , 0.5, f
            line (sx-s*0.25, sy+s*0.4)-(sx-s*0.25, sy+s*1.3), a.col
            line (sx+s*0.25, sy+s*0.4)-(sx+s*0.25, sy+s*1.3), a.col
            circle (sx+s*0.55, sy-s*0.35), s*0.32, a.col, , , , f
            ' horns
            line (sx+s*0.5, sy-s*0.6)-(sx+s*0.35, sy-s*1.5), rgb(70,50,30)
            line (sx+s*0.6, sy-s*0.6)-(sx+s*0.75, sy-s*1.5), rgb(70,50,30)
            ' white belly
            circle (sx, sy+s*0.15), s*0.45, rgb(240,230,210), , , 0.4, f
            ' tail
            line (sx-s*0.6, sy)-(sx-s*0.85, sy-s*0.1), a.col
        case "zebra"
            circle (sx, sy), s*0.85, rgb(240,240,240), , , 0.55, f
            line (sx-s*0.3, sy+s*0.4)-(sx-s*0.3, sy+s*1.2), rgb(240,240,240)
            line (sx+s*0.2, sy+s*0.4)-(sx+s*0.2, sy+s*1.2), rgb(240,240,240)
            line (sx+s*0.6, sy+s*0.4)-(sx+s*0.6, sy+s*1.2), rgb(240,240,240)
            circle (sx+s*0.7, sy-s*0.3), s*0.4, rgb(240,240,240), , , , f
            ' stripes
            for t as integer = -3 to 3
                line (sx+t*s*0.22, sy-s*0.9)-(sx+t*s*0.22, sy+s*0.9), rgb(30,30,30)
            next
            ' mane
            for t as integer = 0 to 4
                line (sx+s*0.4+t*s*0.08, sy-s*0.6-t*s*0.12)-(sx+s*0.4+t*s*0.08, sy-s*1.2-t*s*0.12), rgb(30,30,30)
            next
        case "wildebeest"
            circle (sx, sy), s*0.9, a.col, , , 0.5, f
            line (sx-s*0.35, sy+s*0.4)-(sx-s*0.35, sy+s*1.2), a.col
            line (sx+s*0.2, sy+s*0.45)-(sx+s*0.2, sy+s*1.2), a.col
            line (sx+s*0.55, sy+s*0.4)-(sx+s*0.55, sy+s*1.2), a.col
            circle (sx+s*0.7, sy-s*0.25), s*0.45, a.col, , , , f
            ' horns
            line (sx+s*0.4, sy-s*0.6)-(sx+s*0.15, sy-s*1.4), rgb(80,60,30)
            line (sx+s*0.9, sy-s*0.6)-(sx+s*1.2, sy-s*1.4), rgb(80,60,30)
            ' beard
            for t as integer = 0 to 3
                line (sx+s*0.6, sy+t*s*0.15)-(sx+s*0.6, sy+s*0.4+t*s*0.12), rgb(60,40,20)
            next
            ' shaggy front
            circle (sx+s*0.7, sy+s*0.1), s*0.25, rgb(70,50,30), , , , f
        case "elephant"
            circle (sx, sy), s*1.05, a.col, , , 0.65, f
            line (sx-s*0.3, sy+s*0.6)-(sx-s*0.3, sy+s*1.4), a.col
            line (sx+s*0.3, sy+s*0.6)-(sx+s*0.3, sy+s*1.4), a.col
            ' head
            circle (sx+s*0.85, sy-s*0.2), s*0.6, a.col, , , , f
            ' trunk
            line (sx+s*1.1, sy+s*0.1)-(sx+s*1.35, sy+s*0.9), a.col
            line (sx+s*1.15, sy+s*0.1)-(sx+s*1.4, sy+s*0.9), a.col
            circle (sx+s*1.38, sy+s*0.95), s*0.18, a.col, , , , f
            ' ears
            circle (sx+s*0.55, sy-s*0.7), s*0.55, a.col, , , 0.7, f
            circle (sx+s*1.1, sy-s*0.7), s*0.55, a.col, , , 0.7, f
            ' tusks
            line (sx+s*1.0, sy+s*0.2)-(sx+s*1.25, sy+s*0.45), rgb(245,245,230)
            line (sx+s*1.0, sy+s*0.3)-(sx+s*1.2, sy+s*0.55), rgb(245,245,230)
            ' tail
            line (sx-s*0.9, sy)-(sx-s*1.3, sy-s*0.2), a.col
        case "giraffe"
            circle (sx, sy), s*0.85, a.col, , , 0.55, f
            line (sx-s*0.25, sy+s*0.5)-(sx-s*0.25, sy+s*1.5), a.col
            line (sx+s*0.25, sy+s*0.5)-(sx+s*0.25, sy+s*1.5), a.col
            ' neck
            line (sx+s*0.35, sy-s*0.4)-(sx+s*0.3, sy-s*2.4), a.col
            line (sx+s*0.65, sy-s*0.4)-(sx+s*0.6, sy-s*2.4), a.col
            circle (sx+s*0.45, sy-s*2.6), s*0.4, a.col, , , , f
            ' spots
            circle (sx-s*0.1, sy), 3.5, rgb(120,80,40), , , , f
            circle (sx+s*0.2, sy+0.2*s), 3.5, rgb(120,80,40), , , , f
            circle (sx+s*0.45, sy-s*1.2), 3.5, rgb(120,80,40), , , , f
            circle (sx+s*0.5, sy-s*1.9), 3.5, rgb(120,80,40), , , , f
            ' horns (ossicones)
            line (sx+s*0.35, sy-s*2.9)-(sx+s*0.3, sy-s*3.2), rgb(90,60,30)
            circle (sx+s*0.3, sy-s*3.25), s*0.12, rgb(90,60,30), , , , f
            line (sx+s*0.55, sy-s*2.9)-(sx+s*0.6, sy-s*3.2), rgb(90,60,30)
            circle (sx+s*0.6, sy-s*3.25), s*0.12, rgb(90,60,30), , , , f
            ' tail
            line (sx-s*0.7, sy)-(sx-s*1.0, sy-s*0.5), a.col
            circle (sx-s*1.0, sy-s*0.5), s*0.15, rgb(60,40,20), , , , f
        case "crocodile"
            ' elongated low body lying horizontally
            dim _cx as single = sx
            dim _cy as single = sy
            ' body (ellipse-ish via circle stretched)
            circle (_cx, _cy), s*1.6, rgb(70,95,55), , , 0.35, f
            ' darker back stripe
            line (_cx-s*1.4, _cy-s*0.1)-(_cx+s*1.4, _cy-s*0.1), rgb(50,75,40)
            ' tail tapered at -x
            line (_cx-s*1.5, _cy)-(_cx-s*2.6, _cy), rgb(70,95,55)
            line (_cx-s*1.5, _cy+s*0.15)-(_cx-s*2.6, _cy), rgb(70,95,55)
            line (_cx-s*1.5, _cy-s*0.15)-(_cx-s*2.6, _cy), rgb(70,95,55)
            ' legs (small)
            line (_cx-s*0.6, _cy+s*0.3)-(_cx-s*0.6, _cy+s*0.7), rgb(60,85,45)
            line (_cx-s*0.1, _cy+s*0.3)-(_cx-s*0.1, _cy+s*0.7), rgb(60,85,45)
            line (_cx+s*0.4, _cy+s*0.3)-(_cx+s*0.4, _cy+s*0.7), rgb(60,85,45)
            line (_cx+s*0.9, _cy+s*0.3)-(_cx+s*0.9, _cy+s*0.7), rgb(60,85,45)
            ' head at +x end
            circle (_cx+s*1.7, _cy-s*0.05), s*0.55, rgb(70,95,55), , , 0.6, f
            ' long snout/jaw
            line (_cx+s*1.9, _cy)-(_cx+s*2.7, _cy), rgb(60,85,45)
            line (_cx+s*1.9, _cy+s*0.15)-(_cx+s*2.7, _cy+s*0.05), rgb(60,85,45)
            ' teeth
            for t as integer = 0 to 3
                line (_cx+s*2.0+t*s*0.18, _cy)-(_cx+s*2.0+t*s*0.18, _cy+s*0.12), rgb(230,230,210)
            next
            ' eye bumps raised on top of head
            circle (_cx+s*1.6, _cy-s*0.45), s*0.18, rgb(90,115,65), , , , f
            circle (_cx+s*1.6, _cy-s*0.45), s*0.1, rgb(30,40,20), , , , f
            circle (_cx+s*1.95, _cy-s*0.4), s*0.18, rgb(90,115,65), , , , f
            circle (_cx+s*1.95, _cy-s*0.4), s*0.1, rgb(30,40,20), , , , f
    end select
    circle (sx, sy), s*1.08, rgb(0,0,0), , , 0.6
    if a.attackedTimer > 0 then
        line (sx-s*1.2, sy-s*1.2)-(sx+s*1.2, sy+s*1.2), rgb(220,40,40)
        line (sx-s*1.2, sy+s*1.2)-(sx+s*1.2, sy-s*1.2), rgb(220,40,40)
    end if
    circle (sx, sy), s*1.08, rgb(0,0,0), , , 0.6
    if a.attackedTimer > 0 then
        line (sx-s*1.2, sy-s*1.2)-(sx+s*1.2, sy+s*1.2), rgb(220,40,40)
        line (sx-s*1.2, sy+s*1.2)-(sx+s*1.2, sy-s*1.2), rgb(220,40,40)
    end if
end sub

sub draw_world(camX as single, camY as single)
    line (0,0)-(SC_W-1,SC_H-1), rgb(190,170,110), bf
    ' savannah grass patches in world space
    for i as integer = 0 to 90
        dim wx as single = (i*997.0) mod WORLD_W
        dim wy as single = (i*1319.0) mod WORLD_H
        dim sx as integer, sy as integer
        world_to_screen wx, wy, camX, camY, sx, sy
        if sx > -40 and sx < SC_W+40 and sy > -40 and sy < SC_H+40 then
            circle (sx, sy), 18 + (i mod 6)*3, rgb(150,135,70), , , , f
            circle (sx+4, sy+2), 12 + (i mod 4)*2, rgb(170,150,85), , , , f
        end if
    next
    ' scattered acacia trees
    for i as integer = 0 to 18
        dim wx as single = (i*2137.0 + 300) mod (WORLD_W-200) + 100
        dim wy as single = (i*1733.0 + 200) mod (WORLD_H-200) + 100
        dim sx as integer, sy as integer
        world_to_screen wx, wy, camX, camY, sx, sy
        if sx > -30 and sx < SC_W+30 and sy > -40 and sy < SC_H+40 then
            line (sx, sy)-(sx, sy-28), rgb(110,80,40), bf
            circle (sx, sy-34), 16, rgb(90,120,60), , , , f
            circle (sx-8, sy-28), 12, rgb(70,100,50), , , , f
            circle (sx+8, sy-28), 12, rgb(70,100,50), , , , f
        end if
    next

    for p as integer = 0 to nPonds-1
        dim sx as integer, sy as integer
        world_to_screen ponds(p).x, ponds(p).y, camX, camY, sx, sy
        if sx > -120 and sx < SC_W+120 and sy > -120 and sy < SC_H+120 then
            circle (sx, sy), ponds(p).r, rgb(40,90,170), , , , f
            circle (sx, sy), ponds(p).r, rgb(80,130,200)
            circle (sx - ponds(p).r*0.3, sy - ponds(p).r*0.3), ponds(p).r*0.25, rgb(120,170,220), , , , f
        end if
    next

    for f as integer = 0 to nFoods-1
        if foods(f).active = 0 then continue for
        dim sx as integer, sy as integer
        world_to_screen foods(f).x, foods(f).y, camX, camY, sx, sy
        if sx >= 0 and sx <= SC_W and sy >= 0 and sy <= SC_H then
            circle (sx, sy), 3, rgb(40,160,60), , , , f
            circle (sx, sy-2), 2, rgb(80,200,90), , , , f
        end if
    next

    ' corpses
    for c as integer = 0 to MAX_CORPSES-1
        if corpses(c).active=0 then continue for
        dim sx as integer, sy as integer
        world_to_screen corpses(c).x, corpses(c).y, camX, camY, sx, sy
        if sx > -20 and sx < SC_W+20 and sy > -20 and sy < SC_H+20 then
            circle (sx, sy), 8, rgb(90,40,40), , , , f
            circle (sx, sy), 8, rgb(60,20,20)
        end if
    next
    for i as integer = 0 to MAX_ANIMALS-1
        if animals(i).active=0 then continue for
        draw_animal animals(i), camX, camY
    next
    if selectedAnimal >= 0 and animals(selectedAnimal).active=1 then
        dim sx as integer, sy as integer
        world_to_screen animals(selectedAnimal).x, animals(selectedAnimal).y, camX, camY, sx, sy
        circle (sx, sy), animals(selectedAnimal).size+8, rgb(255,220,60)
        circle (sx, sy), animals(selectedAnimal).size+9, rgb(255,220,60)
    end if
end sub

' ---------- main ----------
randomize timer
screenres SC_W, SC_H, 32
windowtitle "Animal Tank - Raon-1-nano"

init_world()

screenlock
cls
draw string (40, 40), "Animal Tank"
draw string (40, 70), "Loading model..."
screenunlock

llm_init()

for ai as integer = 1 to 10
    if command(ai) = "--log" or command(ai) = "-log" then logging = 1
    if command(ai) = "--demo" then demoMode = 1
next

dim speciesOrder(0 to 8) as string = {"lion","cheetah","hyena","gazelle","zebra","wildebeest","elephant","giraffe","crocodile"}
dim lastViewed(0 to 8) as integer
for ki as integer = 0 to 8
    lastViewed(ki) = -1
next
dim keyWas(1 to 9) as integer
dim prevMouseButtons as integer = 0
dim camX as single = WORLD_W/2 - SC_W/2
dim camY as single = WORLD_H/2 - SC_H/2
dim demoStart as double = timer
dim demoTimer as double = 0
dim demoIndex as integer = 0
dim lastTime as double = timer
dim decideIdx as integer = 0
dim lastAttempt as double = timer
dim lastFloorCheck as double = timer

do
    dim dt as single = timer - lastTime
    if dt > 0.1 then dt = 0.1
    lastTime = timer

    if multikey(FB.SC_LEFT) or multikey(FB.SC_RIGHT) or multikey(FB.SC_UP) or multikey(FB.SC_DOWN) then selectedAnimal = -1
    if multikey(FB.SC_LEFT) then camX -= 260*dt
    if multikey(FB.SC_RIGHT) then camX += 260*dt
    if multikey(FB.SC_UP) then camY -= 260*dt
    if multikey(FB.SC_DOWN) then camY += 260*dt
    if camX < 0 then camX = 0
    if camY < 0 then camY = 0
    if camX > WORLD_W - SC_W then camX = WORLD_W - SC_W
    if camY > WORLD_H - SC_H then camY = WORLD_H - SC_H

    ' mouse click selection
    dim mx as integer, my as integer, mbuttons as integer
    getmouse mx, my, , mbuttons
    if (mbuttons and 1) and (prevMouseButtons and 1)=0 then
        if mx >= 0 and mx < SC_W and my >= 0 and my < SC_H then
            dim wx as single = camX + mx
            dim wy as single = camY + my
            dim best as integer = -1
            dim bestD as single = 9999
            for j as integer = 0 to MAX_ANIMALS-1
                if animals(j).active=0 then continue for
                dim d as single = dist(wx, wy, animals(j).x, animals(j).y)
                if d < bestD and d < animals(j).size + 10 then
                    bestD = d
                    best = j
                end if
            next
            selectedAnimal = best
        end if
    end if
    prevMouseButtons = mbuttons

    ' demo mode auto-cycle
    if demoMode = 1 then
        if timer - demoStart < 2 then
            camX = WORLD_W/2 - SC_W/2
            camY = WORLD_H/2 - SC_H/2
        else
            if demoTimer = 0 then
                demoTimer = timer
                demoIndex = 0
            elseif timer - demoTimer >= 10 then
                demoTimer = timer
                demoIndex = (demoIndex + 1) mod 9
            end if
            ' select first active of species demoIndex
            dim spName as string = speciesOrder(demoIndex)
            for j as integer = 0 to MAX_ANIMALS-1
                if animals(j).active=1 and animals(j).species=spName then
                    selectedAnimal = j
                    exit for
                end if
            next
        end if
    end if

    ' follow selected
    if selectedAnimal >= 0 then
        if animals(selectedAnimal).active=0 then
            selectedAnimal = -1
        else
            camX = animals(selectedAnimal).x - SC_W/2
            camY = animals(selectedAnimal).y - SC_H/2
            if camX < 0 then camX = 0
            if camY < 0 then camY = 0
            if camX > WORLD_W - SC_W then camX = WORLD_W - SC_W
            if camY > WORLD_H - SC_H then camY = WORLD_H - SC_H
        end if
    end if

    ' number keys: center on species (via inkey for discrete presses)
    dim k as string = inkey
    if k >= "1" and k <= "9" then
        dim ki as integer = asc(k) - asc("1") + 1
        dim spName as string = speciesOrder(ki-1)
        dim best as integer = -1
        if lastViewed(ki-1) >= 0 then
            for j as integer = lastViewed(ki-1)+1 to MAX_ANIMALS-1
                if animals(j).active=1 and animals(j).species=spName then
                    best=j
                    exit for
                end if
            next
        end if
        if best=-1 then
            for j as integer = 0 to MAX_ANIMALS-1
                if animals(j).active=1 and animals(j).species=spName then
                    best=j
                    exit for
                end if
            next
        end if
        if best>=0 then
            lastViewed(ki-1)=best
            selectedAnimal = best
        end if
    end if

    for i as integer = 0 to MAX_ANIMALS-1
        if animals(i).active=0 then continue for
        ' starvation/thirst damage
        if animals(i).hunger <= 0 or animals(i).thirst <= 0 then
            animals(i).hp -= dt*6
        end if
        if animals(i).hp <= 0 then
            ' create corpse
            for c as integer = 0 to MAX_CORPSES-1
                if corpses(c).active=0 then
                    corpses(c).active=1
                    corpses(c).x=animals(i).x
                    corpses(c).y=animals(i).y
                    corpses(c).food=4
                    corpses(c).expire=timer+15
                    exit for
                end if
            next
            animals(i).active=0
            continue for
        end if
        ' HP regeneration when fed, watered, and not under attack
        if animals(i).hunger > 0 and animals(i).thirst > 0 and animals(i).attackedTimer <= 0 and animals(i).hp < 100 then
            animals(i).hp += dt*0.3
            if animals(i).hp > 100 then animals(i).hp = 100
        end if
        ' attacked behavior
        if animals(i).attackedTimer > 0 then
            if animals(i).attacker >= 0 and (animals(i).attacker >= MAX_ANIMALS or animals(animals(i).attacker).active=0) then
                animals(i).attackedTimer = 0
                animals(i).attacker = -1
            else
                animals(i).attackedTimer -= dt
            end if
            dim isFighter as integer = 0
            if animals(i).species="elephant" or animals(i).species="giraffe" or animals(i).species="wildebeest" then isFighter=1
            if isFighter=1 then
                ' fight back: damage attacker
                dim atk as integer = animals(i).attacker
                if atk>=0 and atk<MAX_ANIMALS and animals(atk).active=1 then
                    if dist(animals(i).x, animals(i).y, animals(atk).x, animals(atk).y) < 60 then
                        animals(atk).hp -= dt*18 * damage_mult(animals(atk).species)
                        if animals(atk).hp <= 0 then
                            for c as integer = 0 to MAX_CORPSES-1
                                if corpses(c).active=0 then
                                    corpses(c).active=1
                                    corpses(c).x=animals(atk).x
                                    corpses(c).y=animals(atk).y
                                    corpses(c).food=4
                                    corpses(c).expire=timer+15
                                    exit for
                                end if
                            next
                            animals(atk).active=0
                        end if
                    end if
                end if
            else
                ' flee from attacker
                dim atk as integer = animals(i).attacker
                if atk>=0 and atk<MAX_ANIMALS and animals(atk).active=1 and animals(i).moveTimer <= 0 then
                    dim dx as single = animals(atk).x - animals(i).x
                    dim dy as single = animals(atk).y - animals(i).y
                    apply_direction(i, dir_from_delta(-dx, -dy))
                    animals(i).moveTimer = 0.6
                end if
            end if
        end if
        if animals(i).goal = "drink" and animals(i).thirst > 80 then animals(i).goal = ""
        if animals(i).goal = "eat" and animals(i).hunger > 80 then animals(i).goal = ""
        if animals(i).goal = "rest" and animals(i).energy > 90 then animals(i).goal = ""
        if animals(i).state = "sleep" then
            animals(i).sleepTimer -= dt
            animals(i).energy += dt*8
            if animals(i).energy > 100 then animals(i).energy = 100
            if animals(i).sleepTimer <= 0 then
                animals(i).state = "idle"
                animals(i).energy = 100
                animals(i).lastAction = "WAKE"
                animals(i).actionTime = timer
            end if
            ' still update bounds (none) and continue
            continue for
        end if
        animals(i).energy -= dt*0.2
        if animals(i).energy < 0 then animals(i).energy = 0
        if animals(i).energy <= 0 then
            animals(i).state = "sleep"
            animals(i).sleepTimer = 5 + rnd*3
            animals(i).vx = 0
            animals(i).vy = 0
            animals(i).moveTimer = 0
            animals(i).lastAction = "SLEEPING"
            animals(i).actionTime = timer
            continue for
        end if
        if animals(i).species = "crocodile" then
            animals(i).hunger -= dt*0.03
            animals(i).thirst -= dt*0.04
        else
            animals(i).hunger -= dt*0.3
            animals(i).thirst -= dt*0.4
        end if
        if animals(i).hunger < 0 then animals(i).hunger = 0
        if animals(i).thirst < 0 then animals(i).thirst = 0
        if animals(i).hunger > 100 then animals(i).hunger = 100
        if animals(i).thirst > 100 then animals(i).thirst = 100

        if animals(i).moveTimer > 0 then
            animals(i).moveTimer -= dt
            animals(i).x += animals(i).vx * dt
            animals(i).y += animals(i).vy * dt
            ' gentle flocking for prey toward nearest same species (only if no predator nearby)
            dim predatorNear as integer = 0
            for j as integer = 0 to MAX_ANIMALS-1
                if animals(j).active=0 then continue for
                if not (animals(j).species="lion" or animals(j).species="cheetah" or animals(j).species="hyena" or animals(j).species="crocodile") then continue for
                if dist(animals(i).x, animals(i).y, animals(j).x, animals(j).y) < VISION*1.5 then
                    predatorNear = 1
                    exit for
                end if
            next
            if is_predator(animals(i).species)=0 and predatorNear=0 then
                dim bestD as single = 9999
                dim bx as single = 0, by as single = 0
                for j as integer = 0 to MAX_ANIMALS-1
                    if j=i or animals(j).active=0 then continue for
                    if animals(j).species <> animals(i).species then continue for
                    dim d as single = dist(animals(i).x, animals(i).y, animals(j).x, animals(j).y)
                    if d > 0 and d < bestD and d < 160 then
                        bestD = d
                        bx = animals(j).x
                        by = animals(j).y
                    end if
                next
                if bestD < 9999 then
                    animals(i).x += (bx - animals(i).x) * 0.0015
                    animals(i).y += (by - animals(i).y) * 0.0015
                end if
            end if
            ' emergency override: hungry predator stops when prey very close, so EAT can land
            if is_predator(animals(i).species)=1 and animals(i).hunger < 40 then
                for j as integer = 0 to MAX_ANIMALS-1
                    if animals(j).active=0 then continue for
                    if animals(j).species="lion" or animals(j).species="cheetah" or animals(j).species="hyena" or animals(j).species="crocodile" then continue for
                    if dist(animals(i).x, animals(i).y, animals(j).x, animals(j).y) < VISION*0.7 then
                        animals(i).moveTimer = 0
                        animals(i).vx = 0
                        animals(i).vy = 0
                        exit for
                    end if
                next
            end if
            if animals(i).moveTimer <= 0 then
                animals(i).vx = 0
                animals(i).vy = 0
                if animals(i).state = "move" then animals(i).state = "idle"
                if animals(i).state = "eat" or animals(i).state = "drink" or animals(i).state = "rest" then animals(i).state = "idle"
            end if
        else
            ' predator chase (only when hungry)
            if (animals(i).species="lion" or animals(i).species="cheetah" or animals(i).species="hyena" or animals(i).species="crocodile") and animals(i).hunger < 40 then
                dim bestD as single = 9999
                dim tx as single = 0, ty as single = 0
                for j as integer = 0 to MAX_ANIMALS-1
                    if animals(j).active=0 then continue for
                    if animals(j).species="lion" or animals(j).species="cheetah" or animals(j).species="hyena" or animals(j).species="crocodile" then continue for
                    dim d as single = dist(animals(i).x, animals(i).y, animals(j).x, animals(j).y)
                    if d < bestD and d < VISION*1.5 then
                        bestD = d
                        tx = animals(j).x
                        ty = animals(j).y
                    end if
                next
                if bestD < 9999 then
                    dim dx as single = tx - animals(i).x
                    dim dy as single = ty - animals(i).y
                    apply_direction(i, dir_from_delta(dx, dy))
                    animals(i).moveTimer = 0.9
                end if
            else
                ' prey flee from nearest predator
                dim bestD as single = 9999
                dim tx as single = 0, ty as single = 0
                for j as integer = 0 to MAX_ANIMALS-1
                    if animals(j).active=0 then continue for
                    if not (animals(j).species="lion" or animals(j).species="cheetah" or animals(j).species="hyena" or animals(j).species="crocodile") then continue for
                    dim d as single = dist(animals(i).x, animals(i).y, animals(j).x, animals(j).y)
                    if d < bestD and d < VISION*1.2 then
                        bestD = d
                        tx = animals(j).x
                        ty = animals(j).y
                    end if
                next
                if bestD < 9999 then
                    dim dx as single = tx - animals(i).x
                    dim dy as single = ty - animals(i).y
                    apply_direction(i, dir_from_delta(-dx, -dy))
                    animals(i).moveTimer = 0.9
                end if
            end if
            ' edge escape
            if animals(i).moveTimer <= 0 then
                if animals(i).x > WORLD_W - 200 then
                    apply_direction(i, "W")
                    animals(i).lastAction = "MOVE W"
                    animals(i).actionTime = timer
                    animals(i).moveTimer = 0.7
                elseif animals(i).x < 200 then
                    apply_direction(i, "E")
                    animals(i).lastAction = "MOVE E"
                    animals(i).actionTime = timer
                    animals(i).moveTimer = 0.7
                elseif animals(i).y > WORLD_H - 200 then
                    apply_direction(i, "N")
                    animals(i).lastAction = "MOVE N"
                    animals(i).actionTime = timer
                    animals(i).moveTimer = 0.7
                elseif animals(i).y < 200 then
                    apply_direction(i, "S")
                    animals(i).lastAction = "MOVE S"
                    animals(i).actionTime = timer
                    animals(i).moveTimer = 0.7
                end if
            end if
            ' goal-based steering
            if animals(i).moveTimer <= 0 and animals(i).goal <> "" then
                if animals(i).goal = "drink" then
                    ' move toward nearest water
                    dim bestD as single = 9999
                    dim tx as single = 0, ty as single = 0
                    for p as integer = 0 to nPonds-1
                        dim d as single = dist(animals(i).x, animals(i).y, ponds(p).x, ponds(p).y)
                        if d < bestD then
                            bestD = d
                            tx = ponds(p).x
                            ty = ponds(p).y
                        end if
                    next
                    if bestD > ponds(0).r + 5 then
                        dim dx as single = tx - animals(i).x
                        dim dy as single = ty - animals(i).y
                        apply_direction(i, dir_from_delta(dx, dy))
                        animals(i).moveTimer = 0.9
                    end if
                elseif animals(i).goal = "eat" then
                    if is_predator(animals(i).species)=1 then
                        ' move toward nearest prey
                        dim bestD as single = 9999
                        dim tx as single = 0, ty as single = 0
                        for j as integer = 0 to MAX_ANIMALS-1
                            if animals(j).active=0 then continue for
                            if animals(j).species="lion" or animals(j).species="cheetah" or animals(j).species="hyena" or animals(j).species="crocodile" then continue for
                            dim d as single = dist(animals(i).x, animals(i).y, animals(j).x, animals(j).y)
                            if d < bestD then
                                bestD = d
                                tx = animals(j).x
                                ty = animals(j).y
                            end if
                        next
                        if bestD < 9999 then
                            dim dx as single = tx - animals(i).x
                            dim dy as single = ty - animals(i).y
                            apply_direction(i, dir_from_delta(dx, dy))
                            animals(i).moveTimer = 0.9
                        end if
                    else
                        ' move toward nearest food
                        dim bestD as single = 9999
                        dim tx as single = 0, ty as single = 0
                        for f as integer = 0 to nFoods-1
                            if foods(f).active=0 then continue for
                            dim d as single = dist(animals(i).x, animals(i).y, foods(f).x, foods(f).y)
                            if d < bestD then
                                bestD = d
                                tx = foods(f).x
                                ty = foods(f).y
                            end if
                        next
                        if bestD < 9999 then
                            dim dx as single = tx - animals(i).x
                            dim dy as single = ty - animals(i).y
                            apply_direction(i, dir_from_delta(dx, dy))
                            animals(i).moveTimer = 0.9
                        end if
                    end if
                elseif animals(i).goal = "flee" then
                    ' move away from nearest predator
                    dim bestD as single = 9999
                    dim tx as single = 0, ty as single = 0
                    for j as integer = 0 to MAX_ANIMALS-1
                        if animals(j).active=0 then continue for
                        if not (animals(j).species="lion" or animals(j).species="cheetah" or animals(j).species="hyena" or animals(j).species="crocodile") then continue for
                        dim d as single = dist(animals(i).x, animals(i).y, animals(j).x, animals(j).y)
                        if d < bestD then
                            bestD = d
                            tx = animals(j).x
                            ty = animals(j).y
                        end if
                    next
                    if bestD < 9999 then
                        dim dx as single = tx - animals(i).x
                        dim dy as single = ty - animals(i).y
                        apply_direction(i, dir_from_delta(-dx, -dy))
                        animals(i).moveTimer = 0.9
                    end if
                end if
            end if
            ' idle wandering
            if animals(i).state = "idle" and animals(i).moveTimer <= 0 and rnd < 0.05 then
                dim dirs(0 to 7) as string = {"N","S","E","W","NE","NW","SE","SW"}
                dim wdir as string = dirs(int(rnd*8))
                apply_direction(i, wdir)
                animals(i).lastAction = "MOVE " & wdir
                animals(i).actionTime = timer
                animals(i).moveTimer = 0.6 + rnd*0.6
            end if
        end if

        ' auto-actions when very close
        if animals(i).state <> "sleep" then
            if is_predator(animals(i).species)=1 then
                if animals(i).hunger < 50 then
                    for j as integer = 0 to MAX_ANIMALS-1
                        if animals(j).active=0 then continue for
                        if animals(j).species="lion" or animals(j).species="cheetah" or animals(j).species="hyena" or animals(j).species="crocodile" then continue for
                        if dist(animals(i).x, animals(i).y, animals(j).x, animals(j).y) < 32 then
                            animals(j).hp -= dt*20 * damage_mult(animals(j).species)
                        animals(j).attackedTimer = 1.2
                        animals(j).attacker = i
                        push_event(j, "you were attacked by a " & animals(i).species)
                            if animals(j).hp <= 0 then
                                for c as integer = 0 to MAX_CORPSES-1
                                    if corpses(c).active=0 then
                                        corpses(c).active=1
                                        corpses(c).x=animals(j).x
                                        corpses(c).y=animals(j).y
                                        corpses(c).food=5
                                        corpses(c).expire=timer+15
                                        exit for
                                    end if
                                next
                                animals(j).active=0
                                animals(i).hunger = animals(i).hunger + 35
                                if animals(i).hunger > 100 then animals(i).hunger = 100
                                animals(i).lastAction = "EAT"
                                animals(i).actionTime = timer
                                log_msg("AUTO-EAT kill by " & animals(i).species & " prey hp 0")
                            end if
                            exit for
                        end if
                    next
                end if
            else
                ' prey auto-eat grass
                if animals(i).hunger < 50 then
                    for f as integer = 0 to nFoods-1
                        if foods(f).active=0 then continue for
                        if dist(animals(i).x, animals(i).y, foods(f).x, foods(f).y) < 20 then
                            foods(f).active=0
                            foods(f).regen=timer+8+rnd*6
                            animals(i).hunger = animals(i).hunger + 35
                            if animals(i).hunger > 100 then animals(i).hunger = 100
                            animals(i).lastAction = "EAT"
                            animals(i).actionTime = timer
                            animals(i).goal = ""
                            animals(i).state = "eat"
                            animals(i).moveTimer = 0.6
                            animals(i).vx = 0
                            animals(i).vy = 0
                            exit for
                        end if
                    next
                end if
            end if
            ' auto-drink
            if animals(i).thirst < 50 then
                for p as integer = 0 to nPonds-1
                    if dist(animals(i).x, animals(i).y, ponds(p).x, ponds(p).y) <= ponds(p).r + 2 then
                        animals(i).thirst = animals(i).thirst + 40
                        if animals(i).thirst > 100 then animals(i).thirst = 100
                        animals(i).lastAction = "DRINK"
                        animals(i).actionTime = timer
                        animals(i).goal = ""
                        animals(i).state = "drink"
                        animals(i).moveTimer = 0.6
                        animals(i).vx = 0
                        animals(i).vy = 0
                        exit for
                    end if
                next
            end if
        end if

        ' avoid entering water
        if animals(i).species <> "crocodile" and animals(i).state <> "drink" and animals(i).state <> "sleep" then
            for p as integer = 0 to nPonds-1
                if dist(animals(i).x, animals(i).y, ponds(p).x, ponds(p).y) < ponds(p).r then
                    dim dxw as single = animals(i).x - ponds(p).x
                    dim dyw as single = animals(i).y - ponds(p).y
                    apply_direction(i, dir_from_delta(dxw, dyw))
                    animals(i).moveTimer = 0.5
                    exit for
                end if
            next
        end if

        ' crocodile leash: return to pond if it wanders too far
        if animals(i).species = "crocodile" and animals(i).state <> "sleep" then
            if dist(animals(i).x, animals(i).y, ponds(0).x, ponds(0).y) > ponds(0).r + 60 then
                dim _ldx as single = ponds(0).x - animals(i).x
                dim _ldy as single = ponds(0).y - animals(i).y
                apply_direction(i, dir_from_delta(_ldx, _ldy))
                animals(i).moveTimer = 0.5
            end if
        end if

        if animals(i).x < 20 then
            animals(i).x = 20
            if animals(i).vx < 0 then animals(i).vx = 0
        end if
        if animals(i).y < 20 then
            animals(i).y = 20
            if animals(i).vy < 0 then animals(i).vy = 0
        end if
        if animals(i).x > WORLD_W-20 then
            animals(i).x = WORLD_W-20
            if animals(i).vx > 0 then animals(i).vx = 0
        end if
        if animals(i).y > WORLD_H-20 then
            animals(i).y = WORLD_H-20
            if animals(i).vy > 0 then animals(i).vy = 0
        end if

        ' hard-coded flee reflex (no LLM call): near-range predator triggers immediate flee
        if is_predator(animals(i).species)=0 and animals(i).species <> "elephant" and animals(i).state <> "sleep" then
            dim _nrBest as single = 99999
            dim _nrX as single = 0, _nrY as single = 0
            dim _nrSpecies as string = ""
            dim _nrFound as integer = 0
            for _j as integer = 0 to MAX_ANIMALS-1
                if animals(_j).active=0 then continue for
                if not (animals(_j).species="lion" or animals(_j).species="cheetah" or animals(_j).species="hyena") then continue for
                dim _d as single = dist(animals(i).x, animals(i).y, animals(_j).x, animals(_j).y)
                if _d < _nrBest then
                    _nrBest = _d
                    _nrX = animals(_j).x
                    _nrY = animals(_j).y
                    _nrSpecies = animals(_j).species
                    _nrFound = 1
                end if
            next
            if _nrFound=1 and _nrBest < 90 then
                dim _dx as single = _nrX - animals(i).x
                dim _dy as single = _nrY - animals(i).y
                apply_direction(i, dir_from_delta(-_dx, -_dy))
                animals(i).moveTimer = 0.9
                animals(i).lastAction = "FLEE"
                animals(i).actionTime = timer
                animals(i).goal = "flee"
                animals(i).lastDecide = timer
                push_event(i, "you fled from a " & _nrSpecies)
            end if
        end if
    next

    for f as integer = 0 to nFoods-1
        if foods(f).active = 0 and timer >= foods(f).regen then
            foods(f).active = 1
        end if
    next

    ' corpse sweep: despawn expired (15s) or eaten-out corpses
    for c as integer = 0 to MAX_CORPSES-1
        if corpses(c).active = 1 then
            if timer >= corpses(c).expire or corpses(c).food <= 0 then
                corpses(c).active = 0
            end if
        end if
    next

    ' population floor keeper: top up any species below 2
    if timer - lastFloorCheck >= 3.0 then
        lastFloorCheck = timer
        dim floorSpecies(0 to 7) as string = {"lion","cheetah","hyena","gazelle","zebra","wildebeest","elephant","giraffe"}
        for si as integer = 0 to 7
            dim sp as string = floorSpecies(si)
            dim c as integer = count_species(sp)
            if c < 2 then
                dim surv as integer = -1
                for j as integer = 0 to MAX_ANIMALS-1
                    if animals(j).active=1 and animals(j).species=sp then
                        surv = j
                        exit for
                    end if
                next
                dim bx as single, by as single
                if surv >= 0 then
                    bx = animals(surv).x
                    by = animals(surv).y
                else
                    bx = ponds(0).x + ponds(0).r + 30
                    by = ponds(0).y
                end if
                dim need as integer = 2 - c
                for k as integer = 1 to need
                    spawn_animal sp, bx, by
                    log_msg("floor: spawned " & sp & " (was " & str(c) & ")")
                next
            end if
        next
        ' crocodile floor: exactly one, spawns at pond center
        if count_species("crocodile") = 0 then
            spawn_animal "crocodile", ponds(0).x, ponds(0).y
            log_msg("floor: spawned crocodile (was 0)")
        end if
    end if

    if timer - lastAttempt > 0.05 then
        lastAttempt = timer
        for k as integer = 0 to nAnimals-1
            dim i as integer = (decideIdx + k) mod MAX_ANIMALS
            if animals(i).active=0 then continue for
            ' attacked behavior
        if animals(i).attackedTimer > 0 then
            if animals(i).attacker >= 0 and (animals(i).attacker >= MAX_ANIMALS or animals(animals(i).attacker).active=0) then
                animals(i).attackedTimer = 0
                animals(i).attacker = -1
            else
                animals(i).attackedTimer -= dt
            end if
            dim isFighter as integer = 0
            if animals(i).species="elephant" or animals(i).species="giraffe" or animals(i).species="wildebeest" then isFighter=1
            if isFighter=1 then
                ' fight back: damage attacker
                dim atk as integer = animals(i).attacker
                if atk>=0 and atk<MAX_ANIMALS and animals(atk).active=1 then
                    if dist(animals(i).x, animals(i).y, animals(atk).x, animals(atk).y) < 60 then
                        animals(atk).hp -= dt*18 * damage_mult(animals(atk).species)
                        if animals(atk).hp <= 0 then
                            for c as integer = 0 to MAX_CORPSES-1
                                if corpses(c).active=0 then
                                    corpses(c).active=1
                                    corpses(c).x=animals(atk).x
                                    corpses(c).y=animals(atk).y
                                    corpses(c).food=4
                                    corpses(c).expire=timer+15
                                    exit for
                                end if
                            next
                            animals(atk).active=0
                        end if
                    end if
                end if
            else
                ' flee from attacker
                dim atk as integer = animals(i).attacker
                if atk>=0 and atk<MAX_ANIMALS and animals(atk).active=1 and animals(i).moveTimer <= 0 then
                    dim dx as single = animals(atk).x - animals(i).x
                    dim dy as single = animals(atk).y - animals(i).y
                    apply_direction(i, dir_from_delta(-dx, -dy))
                    animals(i).moveTimer = 0.6
                end if
            end if
        end if
        if animals(i).state = "sleep" then continue for
            ' urgent conditions
            dim _urgent as integer = 0
            dim _bestPD as single = 99999
            if animals(i).attackedTimer > 0 then _urgent = 1
            if is_predator(animals(i).species)=0 then
                for _j2 as integer = 0 to MAX_ANIMALS-1
                    if animals(_j2).active=0 then continue for
                    if not (animals(_j2).species="lion" or animals(_j2).species="cheetah" or animals(_j2).species="hyena") then continue for
                    dim _dd as single = dist(animals(i).x, animals(i).y, animals(_j2).x, animals(_j2).y)
                    if _dd < _bestPD then _bestPD = _dd
                next
                if _bestPD < 90 then _urgent = 1
            end if
            ' skip mid-move unless urgent
            if animals(i).moveTimer > 0 and _urgent = 0 then continue for
            if animals(i).lastDecide = 0 or (timer - animals(i).lastDecide) >= DECIDE_INTERVAL then
                decide_animal(i)
                decideIdx = (i + 1) mod nAnimals
                exit for
            end if
        next
    end if

    screenlock
    draw_world camX, camY
    line (0,0)-(SC_W,22), rgb(0,0,0), bf
    draw string (8,4), "Arrows pan viewport | Q quit   Animals decide via LLM"
    draw string (8, SC_H-14), "1 lion 2 cheetah 3 hyena 4 gazelle 5 zebra 6 wildebeest 7 elephant 8 giraffe 9 crocodile"
    screenunlock

    ' overlay for selected animal
    if selectedAnimal >= 0 and animals(selectedAnimal).active=1 then
        dim a as Animal = animals(selectedAnimal)
        line (SC_W-230, 30)-(SC_W-10, 215), rgb(20,20,30), bf
        line (SC_W-230, 30)-(SC_W-10, 215), rgb(200,200,200), b
        draw string (SC_W-220, 40), "Selected: " & a.name & " the " & a.species
        draw string (SC_W-220, 58), "Fullness: " & str(int(a.hunger)) & "/100"
        draw string (SC_W-220, 76), "Hydration: " & str(int(a.thirst)) & "/100"
        draw string (SC_W-220, 94), "Energy: " & str(int(a.energy)) & "/100"
        draw string (SC_W-220, 112), "HP: " & str(int(a.hp)) & "/100"
        draw string (SC_W-220, 130), "Goal: " & a.goal
        draw string (SC_W-220, 148), "Trait: " & a.trait
        draw string (SC_W-220, 166), "State: " & a.state
        draw string (SC_W-220, 184), "Thought: " & left(a.lastAction, 24)
        draw string (SC_W-220, 202), "Click empty to deselect"
    end if
    sleep 16
loop until multikey(FB.SC_Q)

llama_sampler_free(g_smp)
llama_free(g_ctx)
llama_free_model(g_model)

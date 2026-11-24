' hello_llama.bas - minimal example: send one prompt to raon-1-nano and print the reply.
' Build: see README, or: ./build_hello_llama.sh /path/to/llama.cpp

#include once "llama.bi"

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

ensure_model()

dim modelPath as string = "models/raon-1-nano-Q4_K_M.gguf"

dim mparams as llama_model_params = llama_model_default_params()
mparams.n_gpu_layers = -1
dim model as llama_model ptr = llama_model_load_from_file(strptr(modelPath), mparams)
if model = 0 then
    print "Failed to load model"
    sleep
    end
end if

dim cparams as llama_context_params = llama_context_default_params()
cparams.n_ctx = 2048
cparams.n_batch = 512
dim ctx as llama_context ptr = llama_init_from_model(model, cparams)
dim vocab as const llama_vocab ptr = llama_model_get_vocab(model)

dim sparams as llama_sampler_chain_params = llama_sampler_chain_default_params()
dim smp as llama_sampler ptr = llama_sampler_chain_init(sparams)
llama_sampler_chain_add(smp, llama_sampler_init_top_k(40))
llama_sampler_chain_add(smp, llama_sampler_init_top_p(0.9, 1))
llama_sampler_chain_add(smp, llama_sampler_init_temp(0.7))
llama_sampler_chain_add(smp, llama_sampler_init_dist(42))

dim usrPrompt as string = "Hello! Who are you?"
dim fullSys as string = "You are a helpful assistant. Use only ASCII characters in your reply."

dim prompt as string
prompt = "<|im_start|>system" & chr(10) & fullSys & chr(10) & "<|im_end|>" & chr(10)
prompt = prompt & "<|im_start|>user" & chr(10) & usrPrompt & chr(10) & "<|im_end|>" & chr(10)
prompt = prompt & "<|im_start|>assistant" & chr(10) & "<think></think>" & chr(10) & chr(10)

' tokenize
dim maxT as long = 4096
dim toks as llama_token ptr = allocate(maxT * sizeof(llama_token))
dim nTok as long = llama_tokenize(vocab, strptr(prompt), len(prompt), toks, maxT, true, true)
if nTok < 0 then
    nTok = -nTok
    if nTok > maxT then
        deallocate(toks)
        toks = allocate(nTok * sizeof(llama_token))
        nTok = llama_tokenize(vocab, strptr(prompt), len(prompt), toks, nTok, true, true)
    end if
end if

' decode prompt
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

llama_decode(ctx, batch)
llama_batch_free(batch)

' generate
print "Assistant: ";
dim nCur as long = nTok
dim nGen as integer = 0
while nGen < 120
    dim tok as llama_token = llama_sampler_sample(smp, ctx, -1)
    llama_sampler_accept(smp, tok)
    if llama_token_is_eog(vocab, tok) then exit while
    dim pbuf as zstring ptr = allocate(512)
    dim plen as long = llama_token_to_piece(vocab, tok, pbuf, 512, 0, true)
    if plen > 0 then print left(*pbuf, plen);
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
    llama_decode(ctx, b2)
    llama_batch_free(b2)
    nCur += 1
    nGen += 1
wend
print
print

deallocate(toks)
llama_sampler_free(smp)
llama_free(ctx)
llama_model_free(model)
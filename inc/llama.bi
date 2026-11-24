#pragma once

#include once "ggml/ggml.bi"
#include once "ggml/ggml-cpu.bi"
#include once "ggml/ggml-backend.bi"
#include once "ggml/ggml-opt.bi"
#include once "ggml/gguf.bi"
#include once "crt/stddef.bi"
#include once "crt/stdint.bi"
#include once "crt/stdio.bi"

#inclib "llama"

' Forward declarations for opaque C structs
type llama_vocab as llama_vocab_
type llama_model as llama_model_
type llama_context as llama_context_
type llama_sampler as llama_sampler_
type llama_adapter_lora as llama_adapter_lora_
type llama_batch_ext as llama_batch_ext_

extern "C"

#define LLAMA_H
#define LLAMA_API
'' TODO: # define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
const LLAMA_DEFAULT_SEED = &hFFFFFFFF
const LLAMA_TOKEN_NULL = -1
const LLAMA_FILE_MAGIC_GGLA = &h67676c61u ' 'ggla'
const LLAMA_FILE_MAGIC_GGSN = &h6767736eu ' 'ggsn'
const LLAMA_FILE_MAGIC_GGSQ = &h67677371u ' 'ggsq'
const LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN
const LLAMA_SESSION_VERSION = 11
const LLAMA_STATE_SEQ_MAGIC = LLAMA_FILE_MAGIC_GGSQ
const LLAMA_STATE_SEQ_VERSION = 4
'
' C interface
'
' TODO: show sample usage
'

type llama_memory_t as llama_memory_i ptr
type llama_pos as long
type llama_token as long
type llama_seq_id as long

enum llama_vocab_type
	LLAMA_VOCAB_TYPE_NONE = 0 ' For models without vocab
	LLAMA_VOCAB_TYPE_SPM = 1 ' LLaMA tokenizer based on byte-level BPE with byte fallback
	LLAMA_VOCAB_TYPE_BPE = 2 ' GPT-2 tokenizer based on byte-level BPE
	LLAMA_VOCAB_TYPE_WPM = 3 ' BERT tokenizer based on WordPiece
	LLAMA_VOCAB_TYPE_UGM = 4 ' T5 tokenizer based on Unigram
	LLAMA_VOCAB_TYPE_RWKV = 5 ' RWKV tokenizer based on greedy tokenization
	LLAMA_VOCAB_TYPE_PLAMO2 = 6 ' PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
	LLAMA_VOCAB_TYPE_TEST = 7 ' Dummy tokenizer for testing: rolling hash of fixed-size chunks -> tokens, tokens -> hex
	LLAMA_VOCAB_TYPE_PLAMO3 = 8 ' PLaMo-3 tokenizer with pre-segmentation and dynamic programming
end enum

enum llama_rope_type
	LLAMA_ROPE_TYPE_NONE = -1
	LLAMA_ROPE_TYPE_NORM = 0
	LLAMA_ROPE_TYPE_NEOX = GGML_ROPE_TYPE_NEOX
	LLAMA_ROPE_TYPE_MROPE = GGML_ROPE_TYPE_MROPE
	LLAMA_ROPE_TYPE_IMROPE = GGML_ROPE_TYPE_IMROPE
	LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION
end enum

enum llama_token_type
	' TODO: remove, required until per token attributes are available from GGUF file
	LLAMA_TOKEN_TYPE_UNDEFINED = 0
	LLAMA_TOKEN_TYPE_NORMAL = 1
	LLAMA_TOKEN_TYPE_UNKNOWN = 2
	LLAMA_TOKEN_TYPE_CONTROL = 3
	LLAMA_TOKEN_TYPE_USER_DEFINED = 4
	LLAMA_TOKEN_TYPE_UNUSED = 5
	LLAMA_TOKEN_TYPE_BYTE = 6
end enum

enum llama_token_attr
	LLAMA_TOKEN_ATTR_UNDEFINED = 0
	LLAMA_TOKEN_ATTR_UNKNOWN = 1 shl 0
	LLAMA_TOKEN_ATTR_UNUSED = 1 shl 1
	LLAMA_TOKEN_ATTR_NORMAL = 1 shl 2
	LLAMA_TOKEN_ATTR_CONTROL = 1 shl 3 ' SPECIAL?
	LLAMA_TOKEN_ATTR_USER_DEFINED = 1 shl 4
	LLAMA_TOKEN_ATTR_BYTE = 1 shl 5
	LLAMA_TOKEN_ATTR_NORMALIZED = 1 shl 6
	LLAMA_TOKEN_ATTR_LSTRIP = 1 shl 7
	LLAMA_TOKEN_ATTR_RSTRIP = 1 shl 8
	LLAMA_TOKEN_ATTR_SINGLE_WORD = 1 shl 9
end enum

' model file types

enum llama_ftype
	LLAMA_FTYPE_ALL_F32 = 0
	LLAMA_FTYPE_MOSTLY_F16 = 1 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q4_0 = 2 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q4_1 = 3 ' except 1d tensors
	' LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
	' LLAMA_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
	' LLAMA_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
	LLAMA_FTYPE_MOSTLY_Q8_0 = 7 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q5_0 = 8 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q5_1 = 9 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q2_K = 10 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q3_K_S = 11 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q3_K_M = 12 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q3_K_L = 13 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q4_K_S = 14 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q4_K_M = 15 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q5_K_S = 16 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q5_K_M = 17 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q6_K = 18 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_IQ2_XXS = 19 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_IQ2_XS = 20 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q2_K_S = 21 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_IQ3_XS = 22 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_IQ3_XXS = 23 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_IQ1_S = 24 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_IQ4_NL = 25 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_IQ3_S = 26 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_IQ3_M = 27 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_IQ2_S = 28 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_IQ2_M = 29 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_IQ4_XS = 30 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_IQ1_M = 31 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_BF16 = 32 ' except 1d tensors
	' LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
	' LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
	' LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
	LLAMA_FTYPE_MOSTLY_TQ1_0 = 36 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_TQ2_0 = 37 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_MXFP4_MOE = 38 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_NVFP4 = 39 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q1_0 = 40 ' except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q2_0 = 41 ' except 1d tensors
	LLAMA_FTYPE_GUESSED = 1024 ' not specified in the model file
end enum

' Get the model file type (quantization) as a string, e.g. "Q8_0" or "Q4_K - Medium"
declare function llama_ftype_name(byval ftype as llama_ftype) as const zstring ptr

enum llama_rope_scaling_type
	LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
	LLAMA_ROPE_SCALING_TYPE_NONE = 0
	LLAMA_ROPE_SCALING_TYPE_LINEAR = 1
	LLAMA_ROPE_SCALING_TYPE_YARN = 2
	LLAMA_ROPE_SCALING_TYPE_LONGROPE = 3
	LLAMA_ROPE_SCALING_TYPE_MAX_VALUE = LLAMA_ROPE_SCALING_TYPE_LONGROPE
end enum

enum llama_pooling_type
	LLAMA_POOLING_TYPE_UNSPECIFIED = -1
	LLAMA_POOLING_TYPE_NONE = 0
	LLAMA_POOLING_TYPE_MEAN = 1
	LLAMA_POOLING_TYPE_CLS = 2
	LLAMA_POOLING_TYPE_LAST = 3
	LLAMA_POOLING_TYPE_RANK = 4 ' used by reranking models to attach the classification head to the graph
end enum

enum llama_attention_type
	LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
	LLAMA_ATTENTION_TYPE_CAUSAL = 0
	LLAMA_ATTENTION_TYPE_NON_CAUSAL = 1
end enum

enum llama_flash_attn_type
	LLAMA_FLASH_ATTN_TYPE_AUTO = -1
	LLAMA_FLASH_ATTN_TYPE_DISABLED = 0
	LLAMA_FLASH_ATTN_TYPE_ENABLED = 1
end enum

declare function llama_flash_attn_type_name(byval flash_attn_type as llama_flash_attn_type) as const zstring ptr

enum llama_split_mode
	LLAMA_SPLIT_MODE_NONE = 0 ' single GPU
	LLAMA_SPLIT_MODE_LAYER = 1 ' split layers and KV across GPUs
	LLAMA_SPLIT_MODE_ROW = 2 ' split layers and KV across GPUs, use tensor parallelism if supported
	LLAMA_SPLIT_MODE_TENSOR = 3
end enum

enum llama_load_mode
	LLAMA_LOAD_MODE_AUTO = -1 ' auto-detect based on device capabilities
	LLAMA_LOAD_MODE_NONE = 0 ' no special loading mode
	LLAMA_LOAD_MODE_MMAP = 1 ' memory map the model
	LLAMA_LOAD_MODE_MLOCK = 2 ' force system to keep model in RAM rather than swapping or compressing
	LLAMA_LOAD_MODE_MMAP_MLOCK = 3 ' mmap + force system to keep model in RAM rather than swapping or compressing
	LLAMA_LOAD_MODE_DIRECT_IO = 4 ' use direct I/O if available
end enum

declare function llama_load_mode_name(byval load_mode as llama_load_mode) as const zstring ptr
declare function llama_load_mode_from_str(byval str as const zstring ptr) as llama_load_mode

enum llama_lazy_mode
	LLAMA_LAZY_MODE_OFF = 0 ' always read the whole tensor up front
	LLAMA_LAZY_MODE_AUTO = 1 ' lazy only for marked tensors larger than 4 GiB (requires mmap)
	LLAMA_LAZY_MODE_ON = 2 ' read the rows of tensors marked by the arch on demand (requires mmap)
end enum

enum llama_context_type
	LLAMA_CONTEXT_TYPE_DEFAULT = 0
	LLAMA_CONTEXT_TYPE_MTP = 1
end enum

' TODO: simplify (https://github.com/ggml-org/llama.cpp/pull/9294#pullrequestreview-2286561979)

type llama_token_data
	id as llama_token ' token id
	logit as single ' log-odds of the token
	p as single ' probability of the token
end type

type llama_token_data_array
	' TODO: consider SoA
	' NOTE: this pointer can be modified by the samplers
	data as llama_token_data ptr
	size as uinteger
	selected as longint ' this is the index in the data array (i.e. not the token id)
	sorted as bool ' note: do not assume the data is sorted - always check this flag
end type

type llama_progress_callback as function(byval progress as single, byval user_data as any ptr) as bool
' Input data for llama_encode/llama_decode
' A llama_batch object can contain input about one or many sequences
' The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
'
' - token  : the token ids of the input (used when embd is NULL)
' - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
' - pos    : the positions of the respective token in the sequence
'            (if set to NULL, the token position will be tracked automatically by llama_encode/llama_decode)
' - seq_id : the sequence to which the respective token belongs
'            (if set to NULL, the sequence ID will be assumed to be 0)
' - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
'            (if set to NULL:
'               - if embeddings: all tokens are output
'               - if not:        only the last token is output
'            )
'

type llama_batch
	n_tokens as long
	token as llama_token ptr
	embd as single ptr
	pos as llama_pos ptr
	n_seq_id as long ptr
	seq_id as llama_seq_id ptr ptr
	logits as byte ptr ' TODO: rename this to "output"
end type

enum llama_model_kv_override_type
	LLAMA_KV_OVERRIDE_TYPE_INT
	LLAMA_KV_OVERRIDE_TYPE_FLOAT
	LLAMA_KV_OVERRIDE_TYPE_BOOL
	LLAMA_KV_OVERRIDE_TYPE_STR
end enum

enum llama_model_meta_key
	LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE
	LLAMA_MODEL_META_KEY_SAMPLING_TOP_K
	LLAMA_MODEL_META_KEY_SAMPLING_TOP_P
	LLAMA_MODEL_META_KEY_SAMPLING_MIN_P
	LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY
	LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD
	LLAMA_MODEL_META_KEY_SAMPLING_TEMP
	LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N
	LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT
	LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT
	LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU
	LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA
end enum

enum llama_process_type
	LLAMA_PROCESS_TYPE_ENCODE
	LLAMA_PROCESS_TYPE_DECODE
end enum

type llama_model_kv_override
	tag as llama_model_kv_override_type
	key as zstring * 128

	union
		val_i64 as longint
		val_f64 as double
		val_bool as bool
		val_str as zstring * 128
	end union
end type

type llama_model_tensor_buft_override
	pattern as const zstring ptr
	buft as ggml_backend_buffer_type_t
end type

type llama_model_params
	' NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
	devices as ggml_backend_dev_t ptr
	' NULL-terminated list of buffer types to use for tensors that match a pattern
	tensor_buft_overrides as const llama_model_tensor_buft_override ptr
	n_gpu_layers as long ' number of layers to store in VRAM, a negative value means all layers
	split_mode as llama_split_mode ' how to split the model across multiple GPUs
	load_mode as llama_load_mode ' how to load the model
	lazy_mode as llama_lazy_mode ' on-demand reading of tensors marked by the arch
	' the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
	main_gpu as long
	' proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
	tensor_split as const single ptr
	' Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
	' If the provided progress_callback returns true, model loading continues.
	' If it returns false, model loading is immediately aborted.
	progress_callback as llama_progress_callback
	' context pointer passed to the progress callback
	progress_callback_user_data as any ptr
	' override key-value pairs of the model meta data
	kv_overrides as const llama_model_kv_override ptr
	' Keep the booleans together to avoid misalignment during copy-by-value.
	vocab_only as bool ' only load the vocabulary, no weights
	check_tensors as bool ' validate model tensor data
	use_extra_bufts as bool ' use extra buffer types (used for weight repacking)
	no_host as bool ' bypass host buffer allowing extra buffers to be used
	no_alloc as bool ' only load metadata and simulate memory allocations
	load_mtp as bool ' whether to load MTP layers
end type

type llama_sampler as llama_sampler_

type llama_sampler_seq_config
	seq_id as llama_seq_id
	sampler as llama_sampler ptr
end type

' NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
'       https://github.com/ggml-org/llama.cpp/pull/7544

type llama_context_params
	n_ctx as ulong ' text context, 0 = from model
	n_batch as ulong ' logical maximum batch size that can be submitted to llama_decode
	n_ubatch as ulong ' physical maximum batch size
	n_ubatch_auto_mib as ulong ' > 0: the largest physical batch size up to n_batch (halving from n_batch, at least
	' n_ubatch) whose compute buffer on every non-CPU device stays within this many MiB
	n_seq_max as ulong ' max number of sequences (i.e. distinct states for recurrent models)
	n_rs_seq as ulong ' number of recurrent-state snapshots per seq for rollback (0 = no rollback) [EXPERIMENTAL]
	n_outputs_max as ulong ' max outputs in a ubatch (0 = n_batch)
	n_outputs_max_per_seq as ulong ' max outputs per sequence (0 = n_outputs_max)
	n_threads as long ' number of threads to use for generation
	n_threads_batch as long ' number of threads to use for batch processing
	ctx_type as llama_context_type ' set the context type (e.g. MTP)
	rope_scaling_type as llama_rope_scaling_type ' RoPE scaling type, from `enum llama_rope_scaling_type`
	pooling_type as llama_pooling_type ' whether to pool (sum) embedding results by sequence id
	attention_type as llama_attention_type ' attention type to use for embeddings
	flash_attn_type as llama_flash_attn_type ' when to enable Flash Attention
	' ref: https://github.com/ggml-org/llama.cpp/pull/2054
	rope_freq_base as single ' RoPE base frequency, 0 = from model
	rope_freq_scale as single ' RoPE frequency scaling factor, 0 = from model
	yarn_ext_factor as single ' YaRN extrapolation mix factor, negative = from model
	yarn_attn_factor as single ' YaRN magnitude scaling factor
	yarn_beta_fast as single ' YaRN low correction dim
	yarn_beta_slow as single ' YaRN high correction dim
	yarn_orig_ctx as ulong ' YaRN original context size
	defrag_thold as single ' [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)
	cb_eval as ggml_backend_sched_eval_callback
	cb_eval_user_data as any ptr
	type_k as ggml_type ' data type for K cache [EXPERIMENTAL]
	type_v as ggml_type ' data type for V cache [EXPERIMENTAL]
	moe_cache_size as uinteger ' device cache in bytes for the experts kept in host memory, split among the devices like the layers, 0 = disabled [EXPERIMENTAL]
	' Abort callback
	' if it returns true, execution of llama_decode() will be aborted
	' currently works only with CPU execution
	abort_callback as ggml_abort_callback
	abort_callback_data as any ptr
	' Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
	embeddings as bool ' if true, extract embeddings (together with logits, except for a pooling type other than NONE)
	offload_kqv as bool ' offload the KQV ops (including the KV cache) to GPU
	no_perf as bool ' measure performance timings
	op_offload as bool ' offload host tensor operations to device
	swa_full as bool ' use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
	' NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
	'       ref: https://github.com/ggml-org/llama.cpp/pull/13845#issuecomment-2924800573
	kv_unified as bool ' use a unified buffer across the input sequences when computing the attention
	' try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
	' ref: https://github.com/ggml-org/llama.cpp/pull/14363
	' [EXPERIMENTAL]
	' backend sampler chain configuration (make sure the caller keeps the sampler chains alive)
	' note: the samplers must be sampler chains (i.e. use llama_sampler_chain_init)
	samplers as llama_sampler_seq_config ptr
	n_samplers as uinteger
	' a source/target/parent context
	' can be utilized in various ways, for example by sharing results or llama_memory between 2 contexts
	ctx_other as llama_context ptr
end type

type llama_model_tensor_override
	pattern as const zstring ptr
	as ggml_type type
end type

type llama_model_imatrix_data
	name as const zstring ptr
	data as const single ptr
	size as uinteger
end type

' model quantization parameters

type llama_model_quantize_params
	nthread as long ' number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
	ftype as llama_ftype ' quantize to this llama_ftype
	output_tensor_type as ggml_type ' output tensor type
	token_embedding_type as ggml_type ' token embeddings tensor type
	allow_requantize as bool ' allow quantizing non-f32/f16 tensors
	quantize_output_tensor as bool ' quantize output.weight
	only_copy as bool ' only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
	pure as bool ' quantize all tensors to the default type
	keep_split as bool ' quantize to the same number of shards
	dry_run as bool ' calculate and show the final quantization size without performing quantization
	imatrix as const llama_model_imatrix_data ptr ' pointer to importance matrix data
	kv_overrides as const llama_model_kv_override ptr ' pointer to kv overrides
	tt_overrides as const llama_model_tensor_override ptr ' pointer to tensor overrides
	prune_layers as const long ptr ' pointer to layer indices to prune
	max_buf_size as uinteger ' max bytes of tensor rows kept in memory at once, 0 = default (8 GiB)
end type

type llama_logit_bias
	token as llama_token
	bias as single
end type

type llama_sampler_chain_params
	no_perf as bool ' whether to measure performance timings
end type

' used in chat template

type llama_chat_message
	role as const zstring ptr
	content as const zstring ptr
end type

' lora adapter
declare function llama_version() as const zstring ptr
' Helpers for getting default parameters
' TODO: update API to start accepting pointers to params structs (https://github.com/ggml-org/llama.cpp/discussions/9172)
declare function llama_model_default_params() as llama_model_params
declare function llama_context_default_params() as llama_context_params
declare function llama_sampler_chain_default_params() as llama_sampler_chain_params
declare function llama_model_quantize_default_params() as llama_model_quantize_params
' Initialize the llama + ggml backend
' Call once at the start of the program
declare sub llama_backend_init()
' Call once at the end of the program - currently only used for MPI
declare sub llama_backend_free()
' Optional: enable numa optimizations
' TODO: deprecate and make part of llama_backend_init()
declare sub llama_numa_init(byval numa as ggml_numa_strategy)
' Optional: an auto threadpool gets created in ggml if not passed explicitly
declare sub llama_attach_threadpool(byval ctx as llama_context ptr, byval threadpool as ggml_threadpool_t, byval threadpool_batch as ggml_threadpool_t)
declare sub llama_detach_threadpool(byval ctx as llama_context ptr)
type llama_model_set_tensor_data_t as sub(byval tensor as ggml_tensor ptr, byval userdata as any ptr)
' Create a new model from GGUF metadata as well as a function to set the tensor data
'   - tensors are created as GGML_TYPE_F32 by default,
'     override by adding a tensor with the same name but a different name to the context
'' TODO: struct llama_model * llama_model_init_from_user( struct gguf_context * metadata, llama_model_set_tensor_data_t set_tensor_data, [TK_FBCODE ' function to initialize tensor data with] void * set_tensor_data_ud, [TK_FBCODE ' userdata for function] struct llama_model_params params);
declare function llama_load_model_from_file(byval path_model as const zstring ptr, byval params as llama_model_params) as llama_model ptr
' Load a model from a file
' If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
' If the split file name does not follow this pattern, use llama_model_load_from_splits
declare function llama_model_load_from_file(byval path_model as const zstring ptr, byval params as llama_model_params) as llama_model ptr
' Load a model from an open FILE pointer
' The GGUF is read from the current position, so it can be embedded in a larger file
' mmap needs the GGUF data section at a file offset to be aligned to the CPU tensor alignment (32 bytes)
declare function llama_model_load_from_file_ptr(byval file as FILE ptr, byval params as llama_model_params) as llama_model ptr
' Load a model from multiple splits (support custom naming scheme)
' The paths must be in the correct order
declare function llama_model_load_from_splits(byval paths as const zstring ptr ptr, byval n_paths as uinteger, byval params as llama_model_params) as llama_model ptr
declare sub llama_model_save_to_file(byval model as const llama_model ptr, byval path_model as const zstring ptr)
declare sub llama_free_model(byval model as llama_model ptr)
declare sub llama_model_free(byval model as llama_model ptr)
declare function llama_init_from_model(byval model as llama_model ptr, byval params as llama_context_params) as llama_context ptr
declare function llama_new_context_with_model(byval model as llama_model ptr, byval params as llama_context_params) as llama_context ptr
' Frees all allocated memory
declare sub llama_free(byval ctx as llama_context ptr)
declare function llama_time_us() as longint
declare function llama_max_devices() as uinteger
declare function llama_max_parallel_sequences() as uinteger
declare function llama_max_tensor_buft_overrides() as uinteger
declare function llama_supports_mmap() as bool
declare function llama_supports_mlock() as bool
declare function llama_supports_gpu_offload() as bool
declare function llama_supports_rpc() as bool
' NOTE: After creating a llama_context, it is recommended to query the actual values using these functions
'       In some cases the requested values via llama_context_params may differ from the actual values used by the context
'       ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
declare function llama_n_ctx(byval ctx as const llama_context ptr) as ulong
declare function llama_n_ctx_seq(byval ctx as const llama_context ptr) as ulong
declare function llama_n_batch(byval ctx as const llama_context ptr) as ulong
declare function llama_n_ubatch(byval ctx as const llama_context ptr) as ulong
declare function llama_n_seq_max(byval ctx as const llama_context ptr) as ulong
declare function llama_n_rs_seq(byval ctx as const llama_context ptr) as ulong
declare function llama_n_ctx_train(byval model as const llama_model ptr) as long
declare function llama_n_embd(byval model as const llama_model ptr) as long
declare function llama_n_layer(byval model as const llama_model ptr) as long
declare function llama_n_head(byval model as const llama_model ptr) as long
declare function llama_n_vocab(byval vocab as const llama_vocab ptr) as long
declare function llama_get_model(byval ctx as const llama_context ptr) as const llama_model ptr
declare function llama_get_memory(byval ctx as const llama_context ptr) as llama_memory_t
declare function llama_pooling_type_ alias "llama_pooling_type"(byval ctx as const llama_context ptr) as llama_pooling_type ' TODO: rename to llama_get_pooling_type
declare function llama_model_get_vocab(byval model as const llama_model ptr) as const llama_vocab ptr
declare function llama_model_rope_type(byval model as const llama_model ptr) as llama_rope_type
declare function llama_model_n_ctx_train(byval model as const llama_model ptr) as long
declare function llama_model_n_embd(byval model as const llama_model ptr) as long
declare function llama_model_n_embd_inp(byval model as const llama_model ptr) as long
declare function llama_model_n_embd_out(byval model as const llama_model ptr) as long
declare function llama_model_n_layer(byval model as const llama_model ptr) as long
declare function llama_model_n_layer_nextn(byval model as const llama_model ptr) as long
declare function llama_model_n_head(byval model as const llama_model ptr) as long
declare function llama_model_n_head_kv(byval model as const llama_model ptr) as long
declare function llama_model_n_swa(byval model as const llama_model ptr) as long
' Get the model's RoPE frequency scaling factor
declare function llama_model_rope_freq_scale_train(byval model as const llama_model ptr) as single
' Returns the number of classifier outputs (only valid for classifier models)
' Undefined behavior for non-classifier models
declare function llama_model_n_cls_out(byval model as const llama_model ptr) as ulong
' Returns label of classifier output by index (<n_cls_out). Returns nullptr if no label provided
declare function llama_model_cls_label(byval model as const llama_model ptr, byval i as ulong) as const zstring ptr
declare function llama_vocab_type_ alias "llama_vocab_type"(byval vocab as const llama_vocab ptr) as llama_vocab_type
declare function llama_vocab_n_tokens(byval vocab as const llama_vocab ptr) as long
' Functions to access the model's GGUF metadata scalar values
' - The functions return the length of the string on success, or -1 on failure
' - The output string is always null-terminated and cleared on failure
' - When retrieving a string, an extra byte must be allocated to account for the null terminator
' - GGUF array values are not supported by these functions
' Get metadata value as a string by key name
declare function llama_model_meta_val_str(byval model as const llama_model ptr, byval key as const zstring ptr, byval buf as zstring ptr, byval buf_size as uinteger) as long
' Get the number of metadata key/value pairs
declare function llama_model_meta_count(byval model as const llama_model ptr) as long
' Get sampling metadata key name. Returns nullptr if the key is invalid
declare function llama_model_meta_key_str(byval key as llama_model_meta_key) as const zstring ptr
' Get metadata key name by index
declare function llama_model_meta_key_by_index(byval model as const llama_model ptr, byval i as long, byval buf as zstring ptr, byval buf_size as uinteger) as long
' Get metadata value as a string by index
declare function llama_model_meta_val_str_by_index(byval model as const llama_model ptr, byval i as long, byval buf as zstring ptr, byval buf_size as uinteger) as long
' Get a string describing the model type
declare function llama_model_desc(byval model as const llama_model ptr, byval buf as zstring ptr, byval buf_size as uinteger) as long
' Get the model file type (quantization), e.g. LLAMA_FTYPE_MOSTLY_Q8_0
declare function llama_model_ftype(byval model as const llama_model ptr) as llama_ftype
' Returns the total size of all the tensors in the model in bytes
declare function llama_model_size(byval model as const llama_model ptr) as ulongint
' Get the default chat template. Returns nullptr if not available
' If name is NULL, returns the default chat template
declare function llama_model_chat_template(byval model as const llama_model ptr, byval name as const zstring ptr) as const zstring ptr
' Returns the total number of parameters in the model
declare function llama_model_n_params(byval model as const llama_model ptr) as ulongint
' Returns true if the model contains an encoder that requires llama_encode() call
declare function llama_model_has_encoder(byval model as const llama_model ptr) as bool
' Returns true if the model contains a decoder that requires llama_decode() call
declare function llama_model_has_decoder(byval model as const llama_model ptr) as bool
' For encoder-decoder models, this function returns id of the token that must be provided
' to the decoder to start generating output sequence. For other models, it returns -1.
declare function llama_model_decoder_start_token(byval model as const llama_model ptr) as llama_token
' Returns true if the model is recurrent (like Mamba, RWKV, etc.)
declare function llama_model_is_recurrent(byval model as const llama_model ptr) as bool
' Returns true if the model is hybrid (like Jamba, Granite, etc.)
declare function llama_model_is_hybrid(byval model as const llama_model ptr) as bool
' Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
declare function llama_model_is_diffusion(byval model as const llama_model ptr) as bool
' Returns 0 on success
declare function llama_model_quantize(byval fname_inp as const zstring ptr, byval fname_out as const zstring ptr, byval params as const llama_model_quantize_params ptr) as ulong
'
' Adapters
'
' Load a LoRA adapter from file
' The adapter is valid as long as the associated model is not freed
declare function llama_adapter_lora_init(byval model as llama_model ptr, byval path_lora as const zstring ptr) as llama_adapter_lora ptr
' Load a LoRA adapter from an open FILE pointer, reading from its current position
declare function llama_adapter_lora_init_from_file_ptr(byval model as llama_model ptr, byval file as FILE ptr) as llama_adapter_lora ptr
' Functions to access the adapter's GGUF metadata scalar values
' - The functions return the length of the string on success, or -1 on failure
' - The output string is always null-terminated and cleared on failure
' - When retrieving a string, an extra byte must be allocated to account for the null terminator
' - GGUF array values are not supported by these functions
' Get metadata value as a string by key name
declare function llama_adapter_meta_val_str(byval adapter as const llama_adapter_lora ptr, byval key as const zstring ptr, byval buf as zstring ptr, byval buf_size as uinteger) as long
' Get the number of metadata key/value pairs
declare function llama_adapter_meta_count(byval adapter as const llama_adapter_lora ptr) as long
' Get metadata key name by index
declare function llama_adapter_meta_key_by_index(byval adapter as const llama_adapter_lora ptr, byval i as long, byval buf as zstring ptr, byval buf_size as uinteger) as long
' Get metadata value as a string by index
declare function llama_adapter_meta_val_str_by_index(byval adapter as const llama_adapter_lora ptr, byval i as long, byval buf as zstring ptr, byval buf_size as uinteger) as long
' Manually free a LoRA adapter
' NOTE: loaded adapters that are not manually freed will be freed when the associated model is deleted
declare sub llama_adapter_lora_free(byval adapter as llama_adapter_lora ptr)
' Get the invocation tokens if the current lora is an alora
declare function llama_adapter_get_alora_n_invocation_tokens(byval adapter as const llama_adapter_lora ptr) as ulongint
declare function llama_adapter_get_alora_invocation_tokens(byval adapter as const llama_adapter_lora ptr) as const llama_token ptr
' The following functions operate on a llama_context, hence the naming: llama_verb_...
' Set LoRa adapters on the context. Will only modify if the adapters currently in context are different.
declare function llama_set_adapters_lora(byval ctx as llama_context ptr, byval adapters as llama_adapter_lora ptr ptr, byval n_adapters as uinteger, byval scales as single ptr) as long
' Apply a loaded control vector to a llama_context, or if data is NULL, clear
' the currently loaded vector.
' n_embd should be the size of a single layer's control, and data should point
' to an n_embd x n_layers buffer starting from layer 1.
' il_start and il_end are the layer range the vector should apply to (both inclusive)
' See llama_control_vector_load in common to load a control vector.
declare function llama_set_adapter_cvec(byval ctx as llama_context ptr, byval data as const single ptr, byval len as uinteger, byval n_embd as long, byval il_start as long, byval il_end as long) as long
'
' Memory
'
' Clear the memory contents
' If data == true, the data buffers will also be cleared together with the metadata
declare sub llama_memory_clear(byval mem as llama_memory_t, byval data as bool)
' Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
' Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
' seq_id < 0 : match any sequence [TAG_LLAMA_SEQ_ID_NEG]
' p0 < 0     : [0,  p1]
' p1 < 0     : [p0, inf)
declare function llama_memory_seq_rm(byval mem as llama_memory_t, byval seq_id as llama_seq_id, byval p0 as llama_pos, byval p1 as llama_pos) as bool
' Copy all tokens that belong to the specified sequence to another sequence
' p0 < 0 : [0,  p1]
' p1 < 0 : [p0, inf)
declare sub llama_memory_seq_cp(byval mem as llama_memory_t, byval seq_id_src as llama_seq_id, byval seq_id_dst as llama_seq_id, byval p0 as llama_pos, byval p1 as llama_pos)
' Removes all tokens that do not belong to the specified sequence
declare sub llama_memory_seq_keep(byval mem as llama_memory_t, byval seq_id as llama_seq_id)
' Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
' p0 < 0 : [0,  p1]
' p1 < 0 : [p0, inf)
declare sub llama_memory_seq_add(byval mem as llama_memory_t, byval seq_id as llama_seq_id, byval p0 as llama_pos, byval p1 as llama_pos, byval delta as llama_pos)
' Integer division of the positions by factor of `d > 1`
' p0 < 0 : [0,  p1]
' p1 < 0 : [p0, inf)
declare sub llama_memory_seq_div(byval mem as llama_memory_t, byval seq_id as llama_seq_id, byval p0 as llama_pos, byval p1 as llama_pos, byval d as long)
' Returns the smallest position present in the memory for the specified sequence
' This is typically non-zero only for SWA caches
' Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
' Return -1 if the sequence is empty
declare function llama_memory_seq_pos_min(byval mem as llama_memory_t, byval seq_id as llama_seq_id) as llama_pos
' Returns the largest position present in the memory for the specified sequence
' Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
' Return -1 if the sequence is empty
declare function llama_memory_seq_pos_max(byval mem as llama_memory_t, byval seq_id as llama_seq_id) as llama_pos
' Check if the memory supports shifting
declare function llama_memory_can_shift(byval mem as llama_memory_t) as bool
'
' State / sessions
'
' Returns the *actual* size in bytes of the state
' (logits, embedding and memory)
' Only use when saving the state, not when restoring it, otherwise the size may be too small.
declare function llama_state_get_size(byval ctx as llama_context ptr) as uinteger
declare function llama_get_state_size(byval ctx as llama_context ptr) as uinteger
' Copies the state to the specified destination address.
' Destination needs to have allocated enough memory.
' Returns the number of bytes copied
declare function llama_state_get_data(byval ctx as llama_context ptr, byval dst as ubyte ptr, byval size as uinteger) as uinteger
declare function llama_copy_state_data(byval ctx as llama_context ptr, byval dst as ubyte ptr) as uinteger
' Set the state reading from the specified address
' Returns the number of bytes read
declare function llama_state_set_data(byval ctx as llama_context ptr, byval src as const ubyte ptr, byval size as uinteger) as uinteger
declare function llama_set_state_data(byval ctx as llama_context ptr, byval src as const ubyte ptr) as uinteger
' Save/load session file
declare function llama_state_load_file(byval ctx as llama_context ptr, byval path_session as const zstring ptr, byval tokens_out as llama_token ptr, byval n_token_capacity as uinteger, byval n_token_count_out as uinteger ptr) as bool
declare function llama_load_session_file(byval ctx as llama_context ptr, byval path_session as const zstring ptr, byval tokens_out as llama_token ptr, byval n_token_capacity as uinteger, byval n_token_count_out as uinteger ptr) as bool
declare function llama_state_save_file(byval ctx as llama_context ptr, byval path_session as const zstring ptr, byval tokens as const llama_token ptr, byval n_token_count as uinteger) as bool
declare function llama_save_session_file(byval ctx as llama_context ptr, byval path_session as const zstring ptr, byval tokens as const llama_token ptr, byval n_token_count as uinteger) as bool
' Get the exact size needed to copy the state of a single sequence
declare function llama_state_seq_get_size(byval ctx as llama_context ptr, byval seq_id as llama_seq_id) as uinteger
' Copy the state of a single sequence into the specified buffer
declare function llama_state_seq_get_data(byval ctx as llama_context ptr, byval dst as ubyte ptr, byval size as uinteger, byval seq_id as llama_seq_id) as uinteger
' Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
' Returns:
'  - Positive: Ok
'  - Zero: Failed to load
declare function llama_state_seq_set_data(byval ctx as llama_context ptr, byval src as const ubyte ptr, byval size as uinteger, byval dest_seq_id as llama_seq_id) as uinteger
declare function llama_state_seq_save_file(byval ctx as llama_context ptr, byval filepath as const zstring ptr, byval seq_id as llama_seq_id, byval tokens as const llama_token ptr, byval n_token_count as uinteger) as uinteger
' If tokens_out is NULL, only the token count is reported through n_token_count_out and no state is loaded
declare function llama_state_seq_load_file(byval ctx as llama_context ptr, byval filepath as const zstring ptr, byval dest_seq_id as llama_seq_id, byval tokens_out as llama_token ptr, byval n_token_capacity as uinteger, byval n_token_count_out as uinteger ptr) as uinteger
const LLAMA_STATE_SEQ_FLAGS_NONE = 0
' for backwards-compat
const LLAMA_STATE_SEQ_FLAGS_SWA_ONLY = 1
' work only with partial states, such as SWA KV cache or recurrent cache (e.g. Mamba)
const LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY = 1
' Keeps the tensor data on device buffers (i.e. not accessible in host memory, but faster save/load).
' Getting the state for a seq_id with this flag invalidates all prior states gotten for that seq_id with this flag.
const LLAMA_STATE_SEQ_FLAGS_ON_DEVICE = 2
type llama_state_seq_flags as ulong
declare function llama_state_seq_get_size_ext(byval ctx as llama_context ptr, byval seq_id as llama_seq_id, byval flags as llama_state_seq_flags) as uinteger
declare function llama_state_seq_get_data_ext(byval ctx as llama_context ptr, byval dst as ubyte ptr, byval size as uinteger, byval seq_id as llama_seq_id, byval flags as llama_state_seq_flags) as uinteger
declare function llama_state_seq_set_data_ext(byval ctx as llama_context ptr, byval src as const ubyte ptr, byval size as uinteger, byval dest_seq_id as llama_seq_id, byval flags as llama_state_seq_flags) as uinteger
'
' Decoding
'
' Return batch for single sequence of tokens
' The sequence ID will be fixed to 0
' The position of the tokens will be tracked automatically by llama_decode
'
' NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
'
declare function llama_batch_get_one(byval tokens as llama_token ptr, byval n_tokens as long) as llama_batch
' Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
' Each token can be assigned up to n_seq_max sequence ids
' The batch has to be freed with llama_batch_free()
' If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
' Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
' The rest of the llama_batch members are allocated with size n_tokens
' All members are left uninitialized
declare function llama_batch_init(byval n_tokens as long, byval embd as long, byval n_seq_max as long) as llama_batch
' Frees a batch of tokens allocated with llama_batch_init()
declare sub llama_batch_free(byval batch as llama_batch)
' Process a batch of tokens.
' In contrast to llama_decode() - this call does not use KV cache.
' For encode-decoder contexts, processes the batch using the encoder.
' Can store the encoder output internally for later use by the decoder's cross-attention layers.
'   0 - success
' < 0 - error. the memory state is restored to the state before this call
declare function llama_encode(byval ctx as llama_context ptr, byval batch as llama_batch) as long
' Process a batch of tokens.
' Requires the context to have a memory.
' For encode-decoder contexts, processes the batch using the decoder.
' Positive return values does not mean a fatal error, but rather a warning.
' Upon fatal-error or abort, the ubatches that managed to be been processed will remain in the memory state of the context
'   To handle this correctly, query the memory state using llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
' Upon other return values, the memory state is restored to the state before this call
'    0 - success
'    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
'    2 - aborted     (processed ubatches will remain in the context's memory)
'   -1 - invalid input batch
' < -1 - fatal error (processed ubatches will remain in the context's memory)
declare function llama_decode(byval ctx as llama_context ptr, byval batch as llama_batch) as long
'
' Extended batch API
'

type llama_embd
	data as const single ptr
	n_rows as uinteger ' number of embedding rows in data
	n_embd as uinteger ' size of one row
end type

declare function llama_batch_ext_init(byval ctx as llama_context ptr) as llama_batch_ext ptr
declare sub llama_batch_ext_free(byval batch as llama_batch_ext ptr)
declare sub llama_batch_ext_clear(byval batch as llama_batch_ext ptr)
' Add an input token to the batch, with default values:
'     id = LLAMA_TOKEN_NULL
'     embd = nullptr
'     pos = not set, the caller must set it with llama_batch_ext_set_pos()
' Returns the batch index (>= 0)
' On error:
'     -1: batch is full
'     -2: token is invalid (id == LLAMA_TOKEN_NULL or invalid embd)
'     -3: invalid sequence id
declare function llama_batch_ext_add(byval batch as llama_batch_ext ptr, byval seq_id as llama_seq_id) as long
' Add an input token to the batch, with a specified token ID or token embedding
declare function llama_batch_ext_add_token(byval batch as llama_batch_ext ptr, byval seq_id as llama_seq_id, byval id as llama_token) as long
declare function llama_batch_ext_add_embd(byval batch as llama_batch_ext ptr, byval seq_id as llama_seq_id, byval embd as llama_embd) as long
' Add the token at index idx in the batch to another sequence id. The position will stays the same.
' Note: this should be called before other _set() functions
declare function llama_batch_ext_add_seq(byval batch as llama_batch_ext ptr, byval idx as long, byval seq_id as llama_seq_id) as bool
' Set the token embedding for the token at index idx in the batch
' use it after llama_batch_ext_add_token() to have an entry with both a token id and an embedding
declare function llama_batch_ext_set_embd_token(byval batch as llama_batch_ext ptr, byval idx as long, byval embd as llama_embd) as bool
' Set the "state" embedding for the token at index idx in the batch
' "state" here means extra hidden state carried over from a previous stage, e.g.:
'   - MTP: state from N layers of the target model
'   - Qwen3 VL (deepstack): state from N layers of the vision encoder
'   - LiliCorr: context embeddings for the transformers
' Returns false if the context does not take a state embedding (MTP and LiliCorr contexts do)
declare function llama_batch_ext_set_embd_state(byval batch as llama_batch_ext ptr, byval idx as long, byval embd as llama_embd) as bool
' Set if output embeddings should be available for the token at index idx in the batch
' Note: for now, this is equivalent to setting the output logits
declare function llama_batch_ext_set_output_embd(byval batch as llama_batch_ext ptr, byval idx as long, byval value as bool) as bool
' Set output logits for the token at index idx in the batch
' Note: for now, this is equivalent to setting the output embd
declare function llama_batch_ext_set_output_logits(byval batch as llama_batch_ext ptr, byval idx as long, byval value as bool) as bool
' Set custom position for the token at index idx in the batch
' For M-RoPE models:
'     - Embedding tokens must have n_pos_per_embd positions per token, in order [t, y, x, z]; t is also the KV cache position
'     - Text token only requires one single position per token
declare function llama_batch_ext_set_pos(byval batch as llama_batch_ext ptr, byval idx as long, byval pos as const llama_pos ptr) as bool
' TODO: implement get_embeddings() and get_logits() for llama_batch_ext
' Return values are the same as llama_decode()
declare function llama_process(byval ctx as llama_context ptr, byval type as llama_process_type, byval batch as llama_batch_ext ptr) as long
' Set the number of threads used for decoding
' n_threads is the number of threads used for generation (single token)
' n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
declare sub llama_set_n_threads(byval ctx as llama_context ptr, byval n_threads as long, byval n_threads_batch as long)
' Get the number of threads used for generation of a single token.
declare function llama_n_threads(byval ctx as llama_context ptr) as long
' Get the number of threads used for prompt and batch processing (multiple token).
declare function llama_n_threads_batch(byval ctx as llama_context ptr) as long
' Set whether the context outputs embeddings or not
' TODO: rename to avoid confusion with llama_get_embeddings()
declare sub llama_set_embeddings(byval ctx as llama_context ptr, byval embeddings as bool)
' Set whether to use causal attention or not
' If set to true, the model will only attend to the past tokens
declare sub llama_set_causal_attn(byval ctx as llama_context ptr, byval causal_attn as bool)
' Returns whether the context is currently using causal attention
declare function llama_get_causal_attn(byval ctx as const llama_context ptr) as bool
' Set whether the model is in warmup mode or not
' If true, all model tensors are activated during llama_decode() to load and cache their weights.
'
' note: using this can cause extra graph reallocations because it changes the graph topology with MoE models,
'       so it is generally not recommended to use in practice. will be removed in the future
declare sub llama_set_warmup(byval ctx as llama_context ptr, byval warmup as bool)
' Set abort callback
declare sub llama_set_abort_callback(byval ctx as llama_context ptr, byval abort_callback as ggml_abort_callback, byval abort_callback_data as any ptr)
' Wait until all computations are finished
' This is automatically done when using one of the functions below to obtain the computation results
' and is not necessary to call it explicitly in most cases
declare sub llama_synchronize(byval ctx as llama_context ptr)
' Token logits obtained from the last call to llama_decode()
' The logits for which llama_batch.logits[i] != 0 are stored contiguously
' in the order they have appeared in the batch.
' Rows: number of tokens for which llama_batch.logits[i] != 0
' Cols: n_vocab
' TODO: deprecate in favor of llama_get_logits_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
declare function llama_get_logits(byval ctx as llama_context ptr) as single ptr
' Logits for the ith token. For positive indices, Equivalent to:
' llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
' Negative indices can be used to access logits in reverse order, -1 is the last logit.
' returns NULL for invalid ids.
declare function llama_get_logits_ith(byval ctx as llama_context ptr, byval i as long) as single ptr
' Get all output token embeddings.
' when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
' the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
' in the order they have appeared in the batch.
' shape: [n_outputs*n_embd]
' Otherwise, returns NULL.
' TODO: deprecate in favor of llama_get_embeddings_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
declare function llama_get_embeddings(byval ctx as llama_context ptr) as single ptr
' Get the embeddings for the ith token. For positive indices, Equivalent to:
' llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
' Negative indices can be used to access embeddings in reverse order, -1 is the last embedding.
' shape: [n_embd] (1-dimensional)
' returns NULL for invalid ids.
declare function llama_get_embeddings_ith(byval ctx as llama_context ptr, byval i as long) as single ptr
' Get the embeddings for a sequence id
' Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
' when pooling_type == LLAMA_POOLING_TYPE_RANK, returns float[n_cls_out] with the rank(s) of the sequence
' otherwise: float[n_embd] (1-dimensional)
declare function llama_get_embeddings_seq(byval ctx as llama_context ptr, byval seq_id as llama_seq_id) as single ptr
'
' backend sampling API [EXPERIMENTAL]
' note: use only if the llama_context was created with at least one llama_sampler_seq_config
'
' Get the backend sampled token for the ith token.
' With multiple outputs, sampler state advances when the token is accepted,
' not when it is read through this function.
' When accepting multiple outputs, accept a contiguous prefix in output order.
' Returns LLAMA_TOKEN_NULL if no token was sampled.
declare function llama_get_sampled_token_ith(byval ctx as llama_context ptr, byval i as long) as llama_token
' Get the backend sampled probabilities for the ith token
' The index matches llama_get_sampled_token_ith().
' Returns NULL if no probabilities were generated.
declare function llama_get_sampled_probs_ith(byval ctx as llama_context ptr, byval i as long) as single ptr
declare function llama_get_sampled_probs_count_ith(byval ctx as llama_context ptr, byval i as long) as ulong
' Get the backend sampled logits for the ith token
' Returns NULL if no logits were sampled.
declare function llama_get_sampled_logits_ith(byval ctx as llama_context ptr, byval i as long) as single ptr
declare function llama_get_sampled_logits_count_ith(byval ctx as llama_context ptr, byval i as long) as ulong
' Get the backend sampled candidates (token ids) for the ith token
' These are needed to map probability/logit indices to vocab token ids.
' Returns NULL if no candidates were sampled.
declare function llama_get_sampled_candidates_ith(byval ctx as llama_context ptr, byval i as long) as llama_token ptr
declare function llama_get_sampled_candidates_count_ith(byval ctx as llama_context ptr, byval i as long) as ulong
'
' Vocab
'
declare function llama_vocab_get_text(byval vocab as const llama_vocab ptr, byval token as llama_token) as const zstring ptr
declare function llama_vocab_get_score(byval vocab as const llama_vocab ptr, byval token as llama_token) as single
declare function llama_vocab_get_attr(byval vocab as const llama_vocab ptr, byval token as llama_token) as llama_token_attr
' Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
declare function llama_vocab_is_eog(byval vocab as const llama_vocab ptr, byval token as llama_token) as bool
' Identify if Token Id is a control token or a render-able token
declare function llama_vocab_is_control(byval vocab as const llama_vocab ptr, byval token as llama_token) as bool
' Special tokens
declare function llama_vocab_bos(byval vocab as const llama_vocab ptr) as llama_token ' beginning-of-sentence
declare function llama_vocab_eos(byval vocab as const llama_vocab ptr) as llama_token ' end-of-sentence
declare function llama_vocab_eot(byval vocab as const llama_vocab ptr) as llama_token ' end-of-turn
declare function llama_vocab_sep(byval vocab as const llama_vocab ptr) as llama_token ' sentence separator
declare function llama_vocab_nl(byval vocab as const llama_vocab ptr) as llama_token ' next-line
declare function llama_vocab_pad(byval vocab as const llama_vocab ptr) as llama_token ' padding
declare function llama_vocab_mask(byval vocab as const llama_vocab ptr) as llama_token ' mask
declare function llama_vocab_get_add_bos(byval vocab as const llama_vocab ptr) as bool
declare function llama_vocab_get_add_eos(byval vocab as const llama_vocab ptr) as bool
declare function llama_vocab_get_add_sep(byval vocab as const llama_vocab ptr) as bool
' model-specific suppress tokens (gguf key: tokenizer.ggml.suppress_tokens)
declare function llama_vocab_get_suppress_tokens(byval vocab as const llama_vocab ptr, byval n_suppress_tokens as long ptr) as const llama_token ptr
declare function llama_vocab_fim_pre(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_vocab_fim_suf(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_vocab_fim_mid(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_vocab_fim_pad(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_vocab_fim_rep(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_vocab_fim_sep(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_token_get_text(byval vocab as const llama_vocab ptr, byval token as llama_token) as const zstring ptr
declare function llama_token_get_score(byval vocab as const llama_vocab ptr, byval token as llama_token) as single
declare function llama_token_get_attr(byval vocab as const llama_vocab ptr, byval token as llama_token) as llama_token_attr
declare function llama_token_is_eog(byval vocab as const llama_vocab ptr, byval token as llama_token) as bool
declare function llama_token_is_control(byval vocab as const llama_vocab ptr, byval token as llama_token) as bool
declare function llama_token_bos(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_token_eos(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_token_eot(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_token_cls(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_token_sep(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_token_nl(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_token_pad(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_add_bos_token(byval vocab as const llama_vocab ptr) as bool
declare function llama_add_eos_token(byval vocab as const llama_vocab ptr) as bool
declare function llama_token_fim_pre(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_token_fim_suf(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_token_fim_mid(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_token_fim_pad(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_token_fim_rep(byval vocab as const llama_vocab ptr) as llama_token
declare function llama_token_fim_sep(byval vocab as const llama_vocab ptr) as llama_token
' CLS is equivalent to BOS
declare function llama_vocab_cls(byval vocab as const llama_vocab ptr) as llama_token
'
' Tokenization
'
' The API is thread-safe.
'
' / @details Convert the provided text into tokens.
' / @param tokens The tokens pointer must be large enough to hold the resulting tokens.
' / @return Returns the number of tokens on success, no more than n_tokens_max
' / @return Returns a negative number on failure - the number of tokens that would have been returned
' / @return Returns INT32_MIN on overflow (e.g., tokenization result size exceeds int32_t limit)
' / @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
' / @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
' /                      as plaintext. Does not insert a leading space.
declare function llama_tokenize(byval vocab as const llama_vocab ptr, byval text as const zstring ptr, byval text_len as long, byval tokens as llama_token ptr, byval n_tokens_max as long, byval add_special as bool, byval parse_special as bool) as long
' Token Id -> Piece.
' Uses the vocabulary in the provided context.
' Does not write null terminator to the buffer.
' User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
' @param special If true, special tokens are rendered in the output.
declare function llama_token_to_piece(byval vocab as const llama_vocab ptr, byval token as llama_token, byval buf as zstring ptr, byval length as long, byval lstrip as long, byval special as bool) as long
' / @details Convert the provided tokens into text (inverse of llama_tokenize()).
' / @param text The char pointer must be large enough to hold the resulting text.
' / @return Returns the number of chars/bytes on success, no more than text_len_max.
' / @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
' / @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
' / @param unparse_special If true, special tokens are rendered in the output.
declare function llama_detokenize(byval vocab as const llama_vocab ptr, byval tokens as const llama_token ptr, byval n_tokens as long, byval text as zstring ptr, byval text_len_max as long, byval remove_special as bool, byval unparse_special as bool) as long
'
' Chat templates
'
' / Apply chat template. Inspired by hf apply_chat_template() on python.
' /
' / NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
' / @param tmpl A Jinja template to use for this chat.
' / @param chat Pointer to a list of multiple llama_chat_message
' / @param n_msg Number of llama_chat_message in this chat
' / @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
' / @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
' / @param length The size of the allocated buffer
' / @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
declare function llama_chat_apply_template(byval tmpl as const zstring ptr, byval chat as const llama_chat_message ptr, byval n_msg as uinteger, byval add_ass as bool, byval buf as zstring ptr, byval length as long) as long
' Get list of built-in chat templates
declare function llama_chat_builtin_templates(byval output as const zstring ptr ptr, byval len as uinteger) as long
'
' Sampling API
'
' Sample usage:
'
'    // prepare the sampling chain at the start
'    auto sparams = llama_sampler_chain_default_params();
'
'    llama_sampler * smpl = llama_sampler_chain_init(sparams);
'
'    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
'    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
'    llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.8));
'
'    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
'    // this sampler will be responsible to select the actual token
'    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
'
'    ...
'
'    // decoding loop:
'    while (...) {
'        ...
'
'        llama_decode(ctx, batch);
'
'        // sample from the logits of the last token in the batch
'        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
'
'        ...
'    }
'
'    llama_sampler_free(smpl);
'
type llama_sampler_context_t as any ptr

type llama_sampler_data
	logits as ggml_tensor ptr
	probs as ggml_tensor ptr
	sampled as ggml_tensor ptr
	candidates as ggml_tensor ptr
end type

' user code can implement the interface below in order to create custom llama_sampler

type llama_sampler_i
	name as function(byval smpl as const llama_sampler ptr) as const zstring ptr ' can be NULL
	accept as sub(byval smpl as llama_sampler ptr, byval token as llama_token) ' can be NULL
	apply as sub(byval smpl as llama_sampler ptr, byval cur_p as llama_token_data_array ptr) ' required
	reset as sub(byval smpl as llama_sampler ptr) ' can be NULL
	clone as function(byval smpl as const llama_sampler ptr) as llama_sampler ptr ' can be NULL if ctx is NULL
	free as sub(byval smpl as llama_sampler ptr) ' can be NULL if ctx is NULL
	' [EXPERIMENTAL]
	' backend sampling interface:
	' return true if the backend supports all ops needed by the sampler and can handle up to n_outputs_max_per_seq outputs per sequence
	' note: call once per sampler
	backend_init as function(byval smpl as llama_sampler ptr, byval buft as ggml_backend_buffer_type_t, byval n_outputs_max_per_seq as ulong) as bool
	' call after .backend_apply()
	backend_accept as sub(byval smpl as llama_sampler ptr, byval ctx as ggml_context ptr, byval gf as ggml_cgraph ptr, byval selected_token as ggml_tensor ptr)
	' call after .backend_init()
	backend_apply as sub(byval smpl as llama_sampler ptr, byval ctx as ggml_context ptr, byval gf as ggml_cgraph ptr, byval data as llama_sampler_data ptr)
	' called before graph execution to set inputs for the current ubatch
	backend_set_input as sub(byval smpl as llama_sampler ptr)
	' called before rebuilding a sampling graph to clear any internal sampler state
	backend_reset as sub(byval smpl as llama_sampler ptr)
	' copy mutable state from src into dst while keeping dst's references to the current sampling graph
	' src and dst must have the same type and configuration
	copy_state as sub(byval src as const llama_sampler ptr, byval dst as llama_sampler ptr)
end type

type llama_sampler_
	iface as llama_sampler_i ptr
	ctx as llama_sampler_context_t
end type

' [EXPERIMENTAL]
' attach a sampler to the context
' note: prefer initializing the context with llama_context_params.samplers when possible
declare function llama_set_sampler(byval ctx as llama_context ptr, byval seq_id as llama_seq_id, byval smpl as llama_sampler ptr) as bool
' mirror of llama_sampler_i:
declare function llama_sampler_init(byval iface as llama_sampler_i ptr, byval ctx as llama_sampler_context_t) as llama_sampler ptr
declare function llama_sampler_name(byval smpl as const llama_sampler ptr) as const zstring ptr
declare sub llama_sampler_accept(byval smpl as llama_sampler ptr, byval token as llama_token)
declare sub llama_sampler_apply(byval smpl as llama_sampler ptr, byval cur_p as llama_token_data_array ptr)
declare sub llama_sampler_reset(byval smpl as llama_sampler ptr)
declare function llama_sampler_clone(byval smpl as const llama_sampler ptr) as llama_sampler ptr
declare sub llama_sampler_copy(byval src as const llama_sampler ptr, byval dst as llama_sampler ptr)
' important: do not free if the sampler has been added to a llama_sampler_chain (via llama_sampler_chain_add)
declare sub llama_sampler_free(byval smpl as llama_sampler ptr)
' llama_sampler_chain
' a type of llama_sampler that can chain multiple samplers one after another
declare function llama_sampler_chain_init(byval params as llama_sampler_chain_params) as llama_sampler ptr
' important: takes ownership of the sampler object and will free it when llama_sampler_free is called
declare sub llama_sampler_chain_add(byval chain as llama_sampler ptr, byval smpl as llama_sampler ptr)
' return NULL if:
'   - the sampler is NULL
'   - the sampler is not a llama_sampler_chain
'   - the index is out of bounds, unless i == -1
'   - if i == -1, returns the chain itself (can be used to check if the sampler is a chain)
declare function llama_sampler_chain_get(byval chain as llama_sampler ptr, byval i as long) as llama_sampler ptr
' the total number of samplers in the chain
declare function llama_sampler_chain_n(byval chain as const llama_sampler ptr) as long
' after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
declare function llama_sampler_chain_remove(byval chain as llama_sampler ptr, byval i as long) as llama_sampler ptr
' available samplers:
declare function llama_sampler_init_greedy() as llama_sampler ptr
' / seed == LLAMA_DEFAULT_SEED to use a random seed.
declare function llama_sampler_init_dist(byval seed as ulong) as llama_sampler ptr
' / @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
' / Setting k <= 0 makes this a noop
declare function llama_sampler_init_top_k(byval k as long) as llama_sampler ptr
' / @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
declare function llama_sampler_init_top_p(byval p as single, byval min_keep as uinteger) as llama_sampler ptr
' / @details Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841
declare function llama_sampler_init_min_p(byval p as single, byval min_keep as uinteger) as llama_sampler ptr
' / @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
declare function llama_sampler_init_typical(byval p as single, byval min_keep as uinteger) as llama_sampler ptr
' / #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
declare function llama_sampler_init_temp(byval t as single) as llama_sampler ptr
' / @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
declare function llama_sampler_init_temp_ext(byval t as single, byval delta as single, byval exponent as single) as llama_sampler ptr
' / @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
declare function llama_sampler_init_xtc(byval p as single, byval t as single, byval min_keep as uinteger, byval seed as ulong) as llama_sampler ptr
' / @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
declare function llama_sampler_init_top_n_sigma(byval n as single) as llama_sampler ptr
' / @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
' / @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
' / @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
' / @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
' / @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
' / @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
declare function llama_sampler_init_mirostat(byval n_vocab as long, byval seed as ulong, byval tau as single, byval eta as single, byval m as long) as llama_sampler ptr
' / @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
' / @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
' / @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
' / @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
' / @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
declare function llama_sampler_init_mirostat_v2(byval seed as ulong, byval tau as single, byval eta as single) as llama_sampler ptr
' / @details Initializes a GBNF grammar, see grammars/README.md for details.
' / @param vocab The vocabulary that this grammar will be used with.
' / @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
' / @param grammar_root The name of the start symbol for the grammar.
declare function llama_sampler_init_grammar(byval vocab as const llama_vocab ptr, byval grammar_str as const zstring ptr, byval grammar_root as const zstring ptr) as llama_sampler ptr
declare function llama_sampler_init_grammar_lazy(byval vocab as const llama_vocab ptr, byval grammar_str as const zstring ptr, byval grammar_root as const zstring ptr, byval trigger_words as const zstring ptr ptr, byval num_trigger_words as uinteger, byval trigger_tokens as const llama_token ptr, byval num_trigger_tokens as uinteger) as llama_sampler ptr
' / @details Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639
' / @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
' / @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
declare function llama_sampler_init_grammar_lazy_patterns(byval vocab as const llama_vocab ptr, byval grammar_str as const zstring ptr, byval grammar_root as const zstring ptr, byval trigger_patterns as const zstring ptr ptr, byval num_trigger_patterns as uinteger, byval trigger_tokens as const llama_token ptr, byval num_trigger_tokens as uinteger) as llama_sampler ptr
' / NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
' Manually added: fbfrog emits this declaration as a TODO comment.
' Signature is the fork's llama.h declaration. If the headers are
' regenerated with fbfrog this line is lost and the TODO returns;
' reapply it then.
declare function llama_sampler_init_penalties(byval n_vocab as long, byval penalty_last_n as long, byval penalty_repeat as single, byval penalty_freq as single, byval penalty_present as single) as llama_sampler ptr
' must be finite, 0.0 = disabled
' /  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
'' TODO: struct llama_sampler * llama_sampler_init_dry( const struct llama_vocab * vocab, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, [TK_FBCODE ' last n tokens to penalize (0 = disable penalty)] const char ** seq_breakers, size_t num_breakers);
' / adaptive-p: select tokens near a configurable target probability over time.
' /
' / the adaptive-p sampler transforms the token probability distribution to favor tokens
' / that fall near a user-configurable probability target.
' /
' / internally, the sampler maintains an exponential moving average of the *ORIGINAL*
' / probabilities of selected tokens at each sampling step. it uses this EMA to compute an
' / adapted target probability at each sampling step, thus maintaining the desired target
' / probability over time.
' /
' / adaptive-p selects a token ID rather than just mutating candidates, so it must be last
' / in the sampler chain (like mirostat, dist, greedy).
' /
' / only mild truncation before this sampler is recommended. we suggest applying min-p
' / before adaptive-p as the only other active sampler in the chain.
' /
' / @param target select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
' / @param decay  EMA decay for adaptation; history ≈ 1/(1-decay) tokens (valid range 0.0 - 0.99)
' / @param seed   RNG seed
' /
' / ref: https://github.com/ggml-org/llama.cpp/pull/17927
' /
declare function llama_sampler_init_adaptive_p(byval target as single, byval decay as single, byval seed as ulong) as llama_sampler ptr
declare function llama_sampler_init_logit_bias(byval n_vocab as long, byval n_logit_bias as long, byval logit_bias as const llama_logit_bias ptr) as llama_sampler ptr
' this sampler is meant to be used for fill-in-the-middle infilling
' it's supposed to be used after top_k + top_p sampling
'
' 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
' 2. combine probs of tokens that have the same prefix
'
' example:
'
' - before:
'   "hel":   0.5
'   "hell":  0.2
'   "hello": 0.1
'   "dummy": 0.1
'
' - after:
'   "hel":   0.8
'   "dummy": 0.1
'
' 3. discard non-EOG tokens with low prob
' 4. if no tokens are left -> pick EOT
'
declare function llama_sampler_init_infill(byval vocab as const llama_vocab ptr) as llama_sampler ptr
' Returns the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise
declare function llama_sampler_get_seed(byval smpl as const llama_sampler ptr) as ulong
' / @details Sample and accept a token from the idx-th output of the last evaluation
' For multiple outputs from one sampler, call this function in output order without gaps.
'
' Shorthand for:
'    const auto * logits = llama_get_logits_ith(ctx, idx);
'    llama_token_data_array cur_p = { ... init from logits ... };
'    llama_sampler_apply(smpl, &cur_p);
'    auto token = cur_p.data[cur_p.selected].id;
'    llama_sampler_accept(smpl, token);
'    return token;
' Returns the sampled token
declare function llama_sampler_sample(byval smpl as llama_sampler ptr, byval ctx as llama_context ptr, byval idx as long) as llama_token
' TODO: extend in the future
' LLAMA_API void llama_decode_with_sampler(struct llama_context * ctx, struct llama_sampler * smpl, struct llama_batch batch, ...);
'
' Model split
'
' / @details Build a split GGUF final path for this chunk.
' /          llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
'  Returns the split_path length.
declare function llama_split_path(byval split_path as zstring ptr, byval maxlen as uinteger, byval path_prefix as const zstring ptr, byval split_no as long, byval split_count as long) as long
' / @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
' /          llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
'  Returns the split_prefix length.
declare function llama_split_prefix(byval split_prefix as zstring ptr, byval maxlen as uinteger, byval split_path as const zstring ptr, byval split_no as long, byval split_count as long) as long
' Print system information
declare function llama_print_system_info() as const zstring ptr
' Set callback for all future logging events.
' If this is not called, or NULL is supplied, everything is output on stderr.
' The logger state is global so these functions are NOT thread safe.
declare sub llama_log_get(byval log_callback as ggml_log_callback ptr, byval user_data as any ptr ptr)
declare sub llama_log_set(byval log_callback as ggml_log_callback, byval user_data as any ptr)
'
' Performance utils
'
' NOTE: Used by llama.cpp examples/tools, avoid using in third-party apps. Instead, do your own performance measurements.
'

type llama_perf_context_data
	' ms == milliseconds
	t_start_ms as double ' absolute start time
	t_load_ms as double ' time needed for loading the model
	t_p_eval_ms as double ' time needed for processing the prompt
	t_eval_ms as double ' time needed for generating tokens
	n_p_eval as long ' number of prompt tokens
	n_eval as long ' number of generated tokens
	n_reused as long ' number of times a ggml compute graph had been reused
	n_reallocs_same as long ' number of graph reallocations during inference with an unchanged graph size
	n_reallocs_diff as long ' number of graph reallocations during inference with a changed graph size
end type

type llama_perf_sampler_data
	t_sample_ms as double ' time needed for sampling in ms
	n_sample as long ' number of sampled tokens
end type

declare function llama_perf_context(byval ctx as const llama_context ptr) as llama_perf_context_data
declare sub llama_perf_context_print(byval ctx as const llama_context ptr)
declare sub llama_perf_context_reset(byval ctx as llama_context ptr)
' NOTE: the following work only with samplers constructed via llama_sampler_chain_init
declare function llama_perf_sampler(byval chain as const llama_sampler ptr) as llama_perf_sampler_data
declare sub llama_perf_sampler_print(byval chain as const llama_sampler ptr)
declare sub llama_perf_sampler_reset(byval chain as llama_sampler ptr)
'
' training
'
' function that returns whether or not a given tensor contains trainable parameters
type llama_opt_param_filter as function(byval tensor as const ggml_tensor ptr, byval userdata as any ptr) as bool
' always returns true
declare function llama_opt_param_filter_all(byval tensor as const ggml_tensor ptr, byval userdata as any ptr) as bool

type llama_opt_params
	n_ctx_train as ulong ' assumed context size post training, use context size specified in llama_context if 0
	param_filter as llama_opt_param_filter ' callback for determining which tensors contain trainable parameters
	param_filter_ud as any ptr ' userdata for determining which tensors contain trainable parameters
	get_opt_pars as ggml_opt_get_optimizer_params ' callback for calculating optimizer parameters
	get_opt_pars_ud as any ptr ' userdata for calculating optimizer parameters
	optimizer_type as ggml_opt_optimizer_type
end type

declare sub llama_opt_init(byval lctx as llama_context ptr, byval model as llama_model ptr, byval lopt_params as llama_opt_params)
' declare sub llama_opt_epoch(...) ' omitted: ggml_opt_epoch_callback undefined

end extern

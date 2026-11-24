#pragma once

#include once "stdbool.bi"
#include once "crt/stddef.bi"
#include once "crt/stdint.bi"
#include once "crt/stdio.bi"

#inclib "ggml"

' Forward declarations for opaque C structs
type ggml_context as ggml_context_
type ggml_cgraph as ggml_cgraph_
type ggml_object as ggml_object_
type ggml_backend_buffer as ggml_backend_buffer_
type ggml_threadpool as ggml_threadpool_

extern "C"

'
' GGML Tensor Library
'
' This documentation is still a work in progress.
' If you wish some specific topics to be covered, feel free to drop a comment:
'
'   https://github.com/ggml-org/whisper.cpp/issues/40
'
' ## Overview
'
' This library implements:
'
'  - a set of tensor operations
'  - automatic differentiation
'  - basic optimization algorithms
'
' The aim of this library is to provide a minimalistic approach for various machine learning tasks. This includes,
' but is not limited to, the following:
'
'  - linear regression
'  - support vector machines
'  - neural networks
'
' The library allows the user to define a certain function using the available tensor operations. This function
' definition is represented internally via a computation graph. Each tensor operation in the function definition
' corresponds to a node in the graph. Having the computation graph defined, the user can choose to compute the
' function's value and/or its gradient with respect to the input variables. Optionally, the function can be optimized
' using one of the available optimization algorithms.
'
' For example, here we define the function: f(x) = a*x^2 + b
'
'   {
'       struct ggml_init_params params = {
'           .mem_size   = 16*1024*1024,
'           .mem_buffer = NULL,
'       };
'
'       // memory allocation happens here
'       struct ggml_context * ctx = ggml_init(params);
'
'       struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
'
'       ggml_set_param(ctx, x); // x is an input variable
'
'       struct ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
'       struct ggml_tensor * b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
'       struct ggml_tensor * x2 = ggml_mul(ctx, x, x);
'       struct ggml_tensor * f  = ggml_add(ctx, ggml_mul(ctx, a, x2), b);
'
'       ...
'   }
'
' Notice that the function definition above does not involve any actual computation. The computation is performed only
' when the user explicitly requests it. For example, to compute the function's value at x = 2.0:
'
'   {
'       ...
'
'       struct ggml_cgraph * gf = ggml_new_graph(ctx);
'       ggml_build_forward_expand(gf, f);
'
'       // set the input variable and parameter values
'       ggml_set_f32(x, 2.0f);
'       ggml_set_f32(a, 3.0f);
'       ggml_set_f32(b, 4.0f);
'
'       ggml_graph_compute_with_ctx(ctx, &gf, n_threads);
'
'       printf("f = %f\n", ggml_get_f32_1d(f, 0));
'
'       ...
'   }
'
' The actual computation is performed in the ggml_graph_compute() function.
'
' The ggml_new_tensor_...() functions create new tensors. They are allocated in the memory buffer provided to the
' ggml_init() function. You have to be careful not to exceed the memory buffer size. Therefore, you have to know
' in advance how much memory you need for your computation. Alternatively, you can allocate a large enough memory
' and after defining the computation graph, call the ggml_used_mem() function to find out how much memory was
' actually needed.
'
' The ggml_set_param() function marks a tensor as an input variable. This is used by the automatic
' differentiation and optimization algorithms.
'
' The described approach allows to define the function graph once and then compute its forward or backward graphs
' multiple times. All computations will use the same memory buffer allocated in the ggml_init() function. This way
' the user can avoid the memory allocation overhead at runtime.
'
' The library supports multi-dimensional tensors - up to 4 dimensions. The FP16 and FP32 data types are first class
' citizens, but in theory the library can be extended to support FP8 and integer data types.
'
' Each tensor operation produces a new tensor. Initially the library was envisioned to support only the use of unary
' and binary operations. Most of the available operations fall into one of these two categories. With time, it became
' clear that the library needs to support more complex operations. The way to support these operations is not clear
' yet, but a few examples are demonstrated in the following operations:
'
'   - ggml_permute()
'   - ggml_conv_1d_1s()
'   - ggml_conv_1d_2s()
'
' For each tensor operator, the library implements a forward and backward computation function. The forward function
' computes the output tensor value given the input tensor values. The backward function computes the adjoint of the
' input tensors given the adjoint of the output tensor. For a detailed explanation of what this means, take a
' calculus class, or watch the following video:
'
'   What is Automatic Differentiation?
'   https://www.youtube.com/watch?v=wG_nF1awSSY
'
'
' ## Tensor data (struct ggml_tensor)
'
' The tensors are stored in memory via the ggml_tensor struct. The structure provides information about the size of
' the tensor, the data type, and the memory buffer where the tensor data is stored. Additionally, it contains
' pointers to the "source" tensors - i.e. the tensors that were used to compute the current tensor. For example:
'
'   {
'       struct ggml_tensor * c = ggml_add(ctx, a, b);
'
'       assert(c->src[0] == a);
'       assert(c->src[1] == b);
'   }
'
' The multi-dimensional tensors are stored in row-major order. The ggml_tensor struct contains fields for the
' number of elements in each dimension ("ne") as well as the number of bytes ("nb", a.k.a. stride). This allows
' to store tensors that are not contiguous in memory, which is useful for operations such as transposition and
' permutation. All tensor operations have to take the stride into account and not assume that the tensor is
' contiguous in memory.
'
' The data of the tensor is accessed via the "data" pointer. For example:
'
'   {
'       const int nx = 2;
'       const int ny = 3;
'
'       struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nx, ny);
'
'       for (int y = 0; y < ny; y++) {
'           for (int x = 0; x < nx; x++) {
'               *(float *) ((char *) a->data + y*a->nb[1] + x*a->nb[0]) = x + y;
'           }
'       }
'
'       ...
'   }
'
' Alternatively, there are helper functions, such as ggml_get_f32_1d() and ggml_set_f32_1d() that can be used.
'
' ## The matrix multiplication operator (ggml_mul_mat)
'
' TODO
'
'
' ## Multi-threading
'
' TODO
'
'
' ## Overview of ggml.c
'
' TODO
'
'
' ## SIMD optimizations
'
' TODO
'
'
' ## Debugging ggml
'
' TODO
'
'
'' TODO: # define GGML_API extern
' TODO: support for clang
'' TODO: # define GGML_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))

#ifdef __FB_WIN32__
	const _WIN32_WINNT = &h0A00
#endif

const GGML_FILE_MAGIC = &h67676d6c ' "ggml"
const GGML_FILE_VERSION = 2
const GGML_QNT_VERSION = 2 ' bump this on quantization format changes
const GGML_QNT_VERSION_FACTOR = 1000 ' do not change this
const GGML_MAX_DIMS = 4
const GGML_MAX_PARAMS = 2048
const GGML_MAX_SRC = 10
const GGML_MAX_N_THREADS = 512
const GGML_MAX_OP_PARAMS = 64
const GGML_MAX_NAME = 64
const GGML_DEFAULT_N_THREADS = 4
const GGML_DEFAULT_GRAPH_SIZE = 2048
const GGML_MEM_ALIGN = 16
const GGML_EXIT_SUCCESS = 0
const GGML_EXIT_ABORTED = 1
' TODO: convert to enum https://github.com/ggml-org/llama.cpp/pull/16187#discussion_r2388538726
const GGML_ROPE_TYPE_NORMAL = 0
const GGML_ROPE_TYPE_NEOX = 2
const GGML_ROPE_TYPE_MROPE = 8
const GGML_ROPE_TYPE_VISION = 24
const GGML_ROPE_TYPE_IMROPE = 40 ' binary: 101000
const GGML_MROPE_SECTIONS = 4
#define GGML_UNUSED(x) (x)
#define GGML_UNUSED_VARS(__VA_ARGS__...) scope : /' TODO: (void)sizeof((__VA_ARGS__, 0)); '/ : end scope
#define GGML_PAD_(x, n) ((((x) + (n)) - 1) and (not ((n) - 1)))
#macro GGML_UNREACHABLE()
	scope
		fprintf(stderr, !"statement should be unreachable\n")
		abort()
	end scope
#endmacro
#define GGML_NORETURN _Noreturn
#define GGML_ABORT(__VA_ARGS__...) ggml_abort(__FILE__, __LINE__, __VA_ARGS__)

'' TODO: #define GGML_ASSERT(x) if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)
' used to copy the number of elements and stride in bytes of tensors into local variables.
' main purpose is to reduce code duplication and improve readability.
'
' example:
'
'    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
'    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb);
'
'' TODO: #define GGML_TENSOR_LOCALS_1(type, prefix, pointer, array) const type prefix##0 = (pointer) ? (pointer)->array[0] : 0; GGML_UNUSED(prefix##0);
'' TODO: #define GGML_TENSOR_LOCALS_2(type, prefix, pointer, array) GGML_TENSOR_LOCALS_1 (type, prefix, pointer, array) const type prefix##1 = (pointer) ? (pointer)->array[1] : 0; GGML_UNUSED(prefix##1);
'' TODO: #define GGML_TENSOR_LOCALS_3(type, prefix, pointer, array) GGML_TENSOR_LOCALS_2 (type, prefix, pointer, array) const type prefix##2 = (pointer) ? (pointer)->array[2] : 0; GGML_UNUSED(prefix##2);
'' TODO: #define GGML_TENSOR_LOCALS(type, prefix, pointer, array) GGML_TENSOR_LOCALS_3 (type, prefix, pointer, array) const type prefix##3 = (pointer) ? (pointer)->array[3] : 0; GGML_UNUSED(prefix##3);
'' TODO: #define GGML_TENSOR_UNARY_OP_LOCALS GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) GGML_TENSOR_LOCALS(size_t, nb0, src0, nb) GGML_TENSOR_LOCALS(int64_t, ne, dst, ne) GGML_TENSOR_LOCALS(size_t, nb, dst, nb)
'' TODO: #define GGML_TENSOR_BINARY_OP_LOCALS GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) GGML_TENSOR_LOCALS(size_t, nb0, src0, nb) GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) GGML_TENSOR_LOCALS(size_t, nb1, src1, nb) GGML_TENSOR_LOCALS(int64_t, ne, dst, ne) GGML_TENSOR_LOCALS(size_t, nb, dst, nb)
'' TODO: #define GGML_TENSOR_TERNARY_OP_LOCALS GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) GGML_TENSOR_LOCALS(size_t, nb0, src0, nb) GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) GGML_TENSOR_LOCALS(size_t, nb1, src1, nb) GGML_TENSOR_LOCALS(int64_t, ne2, src2, ne) GGML_TENSOR_LOCALS(size_t, nb2, src2, nb) GGML_TENSOR_LOCALS(int64_t, ne, dst, ne) GGML_TENSOR_LOCALS(size_t, nb, dst, nb)
'' TODO: #define GGML_TENSOR_BINARY_OP_LOCALS01 GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) GGML_TENSOR_LOCALS(size_t, nb0, src0, nb) GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) GGML_TENSOR_LOCALS(size_t, nb1, src1, nb)
' Function type used in fatal error callbacks

type ggml_abort_callback_t as sub(byval error_message as const zstring ptr)
' Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
' Returns the old callback for chaining
declare function ggml_set_abort_callback(byval callback as ggml_abort_callback_t) as ggml_abort_callback_t
'' TODO: _Noreturn __attribute__((format(printf, 3, 4))) extern void ggml_abort(const char * file, int line, const char * fmt, ...);

enum ggml_status
	GGML_STATUS_ALLOC_FAILED = -2
	GGML_STATUS_FAILED = -1
	GGML_STATUS_SUCCESS = 0
	GGML_STATUS_ABORTED = 1
end enum

' get ggml_status name string
declare function ggml_status_to_string(byval status as ggml_status) as const zstring ptr
' ieee 754-2008 half-precision float16
' todo: make this not an integral type
type ggml_fp16_t as ushort
declare function ggml_fp16_to_fp32(byval as ggml_fp16_t) as single
declare function ggml_fp32_to_fp16(byval as single) as ggml_fp16_t
declare sub ggml_fp16_to_fp32_row(byval as const ggml_fp16_t ptr, byval as single ptr, byval as longint)
declare sub ggml_fp32_to_fp16_row(byval as const single ptr, byval as ggml_fp16_t ptr, byval as longint)
' google brain half-precision bfloat16

type ggml_bf16_t
	bits as ushort
end type

declare function ggml_fp32_to_bf16(byval as single) as ggml_bf16_t
declare function ggml_bf16_to_fp32(byval as ggml_bf16_t) as single ' consider just doing << 16
declare sub ggml_bf16_to_fp32_row(byval as const ggml_bf16_t ptr, byval as single ptr, byval as longint)
declare sub ggml_fp32_to_bf16_row_ref(byval as const single ptr, byval as ggml_bf16_t ptr, byval as longint)
declare sub ggml_fp32_to_bf16_row(byval as const single ptr, byval as ggml_bf16_t ptr, byval as longint)
' NOTE: always add types at the end of the enum to keep backward compatibility

enum ggml_type
	GGML_TYPE_F32 = 0
	GGML_TYPE_F16 = 1
	GGML_TYPE_Q4_0 = 2
	GGML_TYPE_Q4_1 = 3
	' GGML_TYPE_Q4_2 = 4, support has been removed
	' GGML_TYPE_Q4_3 = 5, support has been removed
	GGML_TYPE_Q5_0 = 6
	GGML_TYPE_Q5_1 = 7
	GGML_TYPE_Q8_0 = 8
	GGML_TYPE_Q8_1 = 9
	GGML_TYPE_Q2_K = 10
	GGML_TYPE_Q3_K = 11
	GGML_TYPE_Q4_K = 12
	GGML_TYPE_Q5_K = 13
	GGML_TYPE_Q6_K = 14
	GGML_TYPE_Q8_K = 15
	GGML_TYPE_IQ2_XXS = 16
	GGML_TYPE_IQ2_XS = 17
	GGML_TYPE_IQ3_XXS = 18
	GGML_TYPE_IQ1_S = 19
	GGML_TYPE_IQ4_NL = 20
	GGML_TYPE_IQ3_S = 21
	GGML_TYPE_IQ2_S = 22
	GGML_TYPE_IQ4_XS = 23
	GGML_TYPE_I8 = 24
	GGML_TYPE_I16 = 25
	GGML_TYPE_I32 = 26
	GGML_TYPE_I64 = 27
	GGML_TYPE_F64 = 28
	GGML_TYPE_IQ1_M = 29
	GGML_TYPE_BF16 = 30
	' GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
	' GGML_TYPE_Q4_0_4_8 = 32,
	' GGML_TYPE_Q4_0_8_8 = 33,
	GGML_TYPE_TQ1_0 = 34
	GGML_TYPE_TQ2_0 = 35
	' GGML_TYPE_IQ4_NL_4_4 = 36,
	' GGML_TYPE_IQ4_NL_4_8 = 37,
	' GGML_TYPE_IQ4_NL_8_8 = 38,
	GGML_TYPE_MXFP4 = 39 ' MXFP4 (1 block)
	GGML_TYPE_NVFP4 = 40 ' NVFP4 (4 blocks, E4M3 scale)
	GGML_TYPE_Q1_0 = 41
	GGML_TYPE_Q2_0 = 42
	GGML_TYPE_COUNT = 43
end enum

' [TAG_GGML_PREC]
' this enum is used to declare the allowed numerical precision/data-types types that can be used during the compute of an op
' the declared types can be:
'  - result accumulation type
'  - source tensor data representation type
'  - etc.
' the precision parameters are stored as ggml_tensor.op_params to the respective ops

enum ggml_prec
	GGML_PREC_UNDEFINED = 0
	GGML_PREC_DEFAULT = 0 ' note: deprecated, use GGML_PREC_UNDEFINED
	GGML_PREC_F32 = 10
	GGML_PREC_BF16 = 15
	GGML_PREC_F16 = 20
	GGML_PREC_Q8 = 30
	GGML_PREC_Q4 = 40
end enum

' op hint

enum ggml_op_hint
	GGML_HINT_NONE = 0
	GGML_HINT_SRC0_IS_HADAMARD = 1
end enum

' model file types

enum ggml_ftype
	GGML_FTYPE_UNKNOWN = -1
	GGML_FTYPE_ALL_F32 = 0
	GGML_FTYPE_MOSTLY_F16 = 1 ' except 1d tensors
	GGML_FTYPE_MOSTLY_Q4_0 = 2 ' except 1d tensors
	GGML_FTYPE_MOSTLY_Q4_1 = 3 ' except 1d tensors
	GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4 ' tok_embeddings.weight and output.weight are F16
	GGML_FTYPE_MOSTLY_Q8_0 = 7 ' except 1d tensors
	GGML_FTYPE_MOSTLY_Q5_0 = 8 ' except 1d tensors
	GGML_FTYPE_MOSTLY_Q5_1 = 9 ' except 1d tensors
	GGML_FTYPE_MOSTLY_Q2_K = 10 ' except 1d tensors
	GGML_FTYPE_MOSTLY_Q3_K = 11 ' except 1d tensors
	GGML_FTYPE_MOSTLY_Q4_K = 12 ' except 1d tensors
	GGML_FTYPE_MOSTLY_Q5_K = 13 ' except 1d tensors
	GGML_FTYPE_MOSTLY_Q6_K = 14 ' except 1d tensors
	GGML_FTYPE_MOSTLY_IQ2_XXS = 15 ' except 1d tensors
	GGML_FTYPE_MOSTLY_IQ2_XS = 16 ' except 1d tensors
	GGML_FTYPE_MOSTLY_IQ3_XXS = 17 ' except 1d tensors
	GGML_FTYPE_MOSTLY_IQ1_S = 18 ' except 1d tensors
	GGML_FTYPE_MOSTLY_IQ4_NL = 19 ' except 1d tensors
	GGML_FTYPE_MOSTLY_IQ3_S = 20 ' except 1d tensors
	GGML_FTYPE_MOSTLY_IQ2_S = 21 ' except 1d tensors
	GGML_FTYPE_MOSTLY_IQ4_XS = 22 ' except 1d tensors
	GGML_FTYPE_MOSTLY_IQ1_M = 23 ' except 1d tensors
	GGML_FTYPE_MOSTLY_BF16 = 24 ' except 1d tensors
	GGML_FTYPE_MOSTLY_MXFP4 = 25 ' except 1d tensors
	GGML_FTYPE_MOSTLY_NVFP4 = 26 ' except 1d tensors
	GGML_FTYPE_MOSTLY_Q1_0 = 27 ' except 1d tensors
	GGML_FTYPE_MOSTLY_Q2_0 = 28 ' except 1d tensors
end enum

' available tensor operations:

enum ggml_op
	GGML_OP_NONE = 0
	GGML_OP_DUP
	GGML_OP_ADD
	GGML_OP_ADD_ID
	GGML_OP_ADD1
	GGML_OP_ACC
	GGML_OP_SUB
	GGML_OP_MUL
	GGML_OP_DIV
	GGML_OP_SQR
	GGML_OP_SQRT
	GGML_OP_LOG
	GGML_OP_SIN
	GGML_OP_COS
	GGML_OP_SUM
	GGML_OP_SUM_ROWS
	GGML_OP_CUMSUM
	GGML_OP_MEAN
	GGML_OP_ARGMAX
	GGML_OP_COUNT_EQUAL
	GGML_OP_REPEAT
	GGML_OP_REPEAT_BACK
	GGML_OP_CONCAT
	GGML_OP_SILU_BACK
	GGML_OP_NORM ' normalize
	GGML_OP_RMS_NORM
	GGML_OP_RMS_NORM_BACK
	GGML_OP_GROUP_NORM
	GGML_OP_L2_NORM
	GGML_OP_MUL_MAT
	GGML_OP_MUL_MAT_ID
	GGML_OP_OUT_PROD
	GGML_OP_SCALE
	GGML_OP_SET
	GGML_OP_CPY
	GGML_OP_CONT
	GGML_OP_RESHAPE
	GGML_OP_VIEW
	GGML_OP_PERMUTE
	GGML_OP_TRANSPOSE
	GGML_OP_GET_ROWS
	GGML_OP_GET_ROWS_BACK
	GGML_OP_SET_ROWS
	GGML_OP_DIAG
	GGML_OP_DIAG_MASK_INF
	GGML_OP_DIAG_MASK_ZERO
	GGML_OP_SOFT_MAX
	GGML_OP_SOFT_MAX_BACK
	GGML_OP_ROPE
	GGML_OP_ROPE_BACK
	GGML_OP_CLAMP
	GGML_OP_CONV_TRANSPOSE_1D
	GGML_OP_IM2COL
	GGML_OP_IM2COL_BACK
	GGML_OP_IM2COL_3D
	GGML_OP_COL2IM_1D
	GGML_OP_CONV_2D
	GGML_OP_CONV_3D
	GGML_OP_CONV_2D_DW
	GGML_OP_CONV_TRANSPOSE_2D
	GGML_OP_POOL_1D
	GGML_OP_POOL_2D
	GGML_OP_POOL_2D_BACK
	GGML_OP_UPSCALE
	GGML_OP_PAD
	GGML_OP_PAD_REFLECT_1D
	GGML_OP_ROLL
	GGML_OP_ARANGE
	GGML_OP_TIMESTEP_EMBEDDING
	GGML_OP_ARGSORT
	GGML_OP_TOP_K
	GGML_OP_LEAKY_RELU
	GGML_OP_TRI
	GGML_OP_FILL
	GGML_OP_FLASH_ATTN_EXT
	GGML_OP_FLASH_ATTN_BACK
	GGML_OP_SSM_CONV
	GGML_OP_SSM_SCAN
	GGML_OP_WIN_PART
	GGML_OP_WIN_UNPART
	GGML_OP_GET_REL_POS
	GGML_OP_ADD_REL_POS
	GGML_OP_RWKV_WKV6
	GGML_OP_GATED_LINEAR_ATTN
	GGML_OP_RWKV_WKV7
	GGML_OP_SOLVE_TRI
	GGML_OP_GATED_DELTA_NET
	GGML_OP_LIGHTNING_INDEXER
	GGML_OP_DSV4_HC_COMB
	GGML_OP_DSV4_HC_PRE
	GGML_OP_DSV4_HC_POST
	GGML_OP_MLA_EXPAND_KV
	GGML_OP_UNARY
	GGML_OP_MAP_CUSTOM1
	GGML_OP_MAP_CUSTOM2
	GGML_OP_MAP_CUSTOM3
	GGML_OP_CUSTOM
	GGML_OP_CROSS_ENTROPY_LOSS
	GGML_OP_CROSS_ENTROPY_LOSS_BACK
	GGML_OP_OPT_STEP_ADAMW
	GGML_OP_OPT_STEP_SGD
	GGML_OP_GLU
	GGML_OP_COUNT
end enum

enum ggml_unary_op
	GGML_UNARY_OP_ABS
	GGML_UNARY_OP_SGN
	GGML_UNARY_OP_NEG
	GGML_UNARY_OP_STEP
	GGML_UNARY_OP_TANH
	GGML_UNARY_OP_ELU
	GGML_UNARY_OP_RELU
	GGML_UNARY_OP_SIGMOID
	GGML_UNARY_OP_GELU
	GGML_UNARY_OP_GELU_QUICK
	GGML_UNARY_OP_SILU
	GGML_UNARY_OP_HARDSWISH
	GGML_UNARY_OP_HARDSIGMOID
	GGML_UNARY_OP_EXP
	GGML_UNARY_OP_EXPM1
	GGML_UNARY_OP_SOFTPLUS
	GGML_UNARY_OP_GELU_ERF
	GGML_UNARY_OP_XIELU
	GGML_UNARY_OP_FLOOR
	GGML_UNARY_OP_CEIL
	GGML_UNARY_OP_ROUND
	GGML_UNARY_OP_TRUNC
	GGML_UNARY_OP_COUNT
end enum

enum ggml_glu_op
	GGML_GLU_OP_REGLU
	GGML_GLU_OP_GEGLU
	GGML_GLU_OP_SWIGLU
	GGML_GLU_OP_SWIGLU_OAI
	GGML_GLU_OP_GEGLU_ERF
	GGML_GLU_OP_GEGLU_QUICK
	GGML_GLU_OP_SWIGLU_CLAMP
	GGML_GLU_OP_COUNT
end enum

enum ggml_object_type
	GGML_OBJECT_TYPE_TENSOR
	GGML_OBJECT_TYPE_GRAPH
	GGML_OBJECT_TYPE_WORK_BUFFER
end enum

enum ggml_log_level
	GGML_LOG_LEVEL_NONE = 0
	GGML_LOG_LEVEL_DEBUG = 1
	GGML_LOG_LEVEL_INFO = 2
	GGML_LOG_LEVEL_WARN = 3
	GGML_LOG_LEVEL_ERROR = 4
	GGML_LOG_LEVEL_CONT = 5 ' continue previous log
end enum

' this tensor...

enum ggml_tensor_flag
	GGML_TENSOR_FLAG_INPUT = 1 ' ...is an input for the GGML compute graph
	GGML_TENSOR_FLAG_OUTPUT = 2 ' ...is an output for the GGML compute graph
	GGML_TENSOR_FLAG_PARAM = 4 ' ...contains trainable parameters
	GGML_TENSOR_FLAG_LOSS = 8 ' ...defines loss for numerical optimization (multiple loss tensors add up)
	GGML_TENSOR_FLAG_COMPUTE = 16 ' ...must be computed
end enum

enum ggml_tri_type
	GGML_TRI_TYPE_UPPER_DIAG = 0
	GGML_TRI_TYPE_UPPER = 1
	GGML_TRI_TYPE_LOWER_DIAG = 2
	GGML_TRI_TYPE_LOWER = 3
end enum

type ggml_init_params
	' memory pool
	mem_size as uinteger ' bytes
	mem_buffer as any ptr ' if NULL, memory will be allocated internally
	no_alloc as bool ' don't allocate memory for the tensor data
end type

' n-dimensional tensor

type ggml_tensor
	as ggml_type type
	buffer as ggml_backend_buffer ptr
	ne(0 to 3) as longint ' number of elements
	nb(0 to 3) as uinteger ' stride in bytes:
	' nb[0] = ggml_type_size(type)
	' nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
	' nb[i] = nb[i-1] * ne[i-1]
	' compute data
	op as ggml_op
	' op params - allocated as int32_t for alignment
	op_params(0 to (64 / sizeof(long)) - 1) as long
	flags as long
	src(0 to 9) as ggml_tensor ptr
	' source tensor and offset for views
	view_src as ggml_tensor ptr
	view_offs as uinteger
	data as any ptr
	name as zstring * 64
	extra as any ptr ' extra things e.g. for ggml-cuda.cu
	padding as zstring * 8
end type

dim shared GGML_TENSOR_SIZE as const uinteger = sizeof(ggml_tensor)
' Abort callback
' If not NULL, called before ggml computation
' If it returns true, the computation is aborted

type ggml_abort_callback as function(byval data as any ptr) as bool
'
' GUID
'
' GUID types
type ggml_guid_t as ubyte ptr

declare function ggml_guid_matches(byval guid_a as ggml_guid_t, byval guid_b as ggml_guid_t) as bool
' misc
declare function ggml_version() as const zstring ptr
declare function ggml_commit() as const zstring ptr
declare sub ggml_time_init() ' call this once at the beginning of the program
declare function ggml_time_ms() as longint
declare function ggml_time_us() as longint
declare function ggml_cycles() as longint
declare function ggml_cycles_per_ms() as longint
' accepts a UTF-8 path, even on Windows
declare function ggml_fopen(byval fname as const zstring ptr, byval mode as const zstring ptr) as FILE ptr
declare sub ggml_print_object(byval obj as const ggml_object ptr)
declare sub ggml_print_objects(byval ctx as const ggml_context ptr)
declare function ggml_nelements(byval tensor as const ggml_tensor ptr) as longint
declare function ggml_nrows(byval tensor as const ggml_tensor ptr) as longint
declare function ggml_nbytes(byval tensor as const ggml_tensor ptr) as uinteger
declare function ggml_nbytes_pad(byval tensor as const ggml_tensor ptr) as uinteger ' same as ggml_nbytes() but padded to GGML_MEM_ALIGN
declare function ggml_blck_size(byval type as ggml_type) as longint
declare function ggml_type_size(byval type as ggml_type) as uinteger ' size in bytes for all elements in a block
declare function ggml_row_size(byval type as ggml_type, byval ne as longint) as uinteger ' size in bytes for all elements in a row
declare function ggml_type_sizef(byval type as ggml_type) as double
declare function ggml_type_name(byval type as ggml_type) as const zstring ptr
declare function ggml_op_name(byval op as ggml_op) as const zstring ptr
declare function ggml_op_symbol(byval op as ggml_op) as const zstring ptr
declare function ggml_unary_op_name(byval op as ggml_unary_op) as const zstring ptr
declare function ggml_glu_op_name(byval op as ggml_glu_op) as const zstring ptr
declare function ggml_op_desc(byval t as const ggml_tensor ptr) as const zstring ptr ' unary or op name
declare function ggml_element_size(byval tensor as const ggml_tensor ptr) as uinteger
declare function ggml_is_quantized(byval type as ggml_type) as bool
' TODO: temporary until model loading of ggml examples is refactored
declare function ggml_ftype_to_ggml_type(byval ftype as ggml_ftype) as ggml_type
declare function ggml_is_transposed(byval tensor as const ggml_tensor ptr) as bool
declare function ggml_is_permuted(byval tensor as const ggml_tensor ptr) as bool
declare function ggml_is_empty(byval tensor as const ggml_tensor ptr) as bool
declare function ggml_is_view(byval tensor as const ggml_tensor ptr) as bool
declare function ggml_is_scalar(byval tensor as const ggml_tensor ptr) as bool
declare function ggml_is_vector(byval tensor as const ggml_tensor ptr) as bool
declare function ggml_is_matrix(byval tensor as const ggml_tensor ptr) as bool
declare function ggml_is_3d(byval tensor as const ggml_tensor ptr) as bool
declare function ggml_n_dims(byval tensor as const ggml_tensor ptr) as long ' returns 1 for scalars
' returns whether the tensor elements can be iterated over with a flattened index (no gaps, no permutation)
declare function ggml_is_contiguous(byval tensor as const ggml_tensor ptr) as bool
declare function ggml_is_contiguous_0(byval tensor as const ggml_tensor ptr) as bool ' same as ggml_is_contiguous()
declare function ggml_is_contiguous_1(byval tensor as const ggml_tensor ptr) as bool ' contiguous for dims >= 1
declare function ggml_is_contiguous_2(byval tensor as const ggml_tensor ptr) as bool ' contiguous for dims >= 2
declare function ggml_is_contiguous_to_1(byval tensor as const ggml_tensor ptr) as bool ' contiguous for dims < 1
declare function ggml_is_contiguous_to_2(byval tensor as const ggml_tensor ptr) as bool ' contiguous for dims < 2
declare function ggml_is_contiguous_to_3(byval tensor as const ggml_tensor ptr) as bool ' contiguous for dims < 3
' returns whether the tensor elements are allocated as one contiguous block of memory (no gaps, but permutation ok)
declare function ggml_is_contiguously_allocated(byval tensor as const ggml_tensor ptr) as bool
' true for tensor that is stored in memory as CxWxHxN and has been permuted to WxHxCxN
declare function ggml_is_contiguous_channels(byval tensor as const ggml_tensor ptr) as bool
' true if the elements in dimension 0 are contiguous, or there is just 1 block of elements
declare function ggml_is_contiguous_rows(byval tensor as const ggml_tensor ptr) as bool
declare function ggml_are_same_shape(byval t0 as const ggml_tensor ptr, byval t1 as const ggml_tensor ptr) as bool
declare function ggml_are_same_stride(byval t0 as const ggml_tensor ptr, byval t1 as const ggml_tensor ptr) as bool
declare function ggml_can_repeat(byval t0 as const ggml_tensor ptr, byval t1 as const ggml_tensor ptr) as bool
' use this to compute the memory overhead of a tensor
declare function ggml_tensor_overhead() as uinteger
declare function ggml_validate_row_data(byval type as ggml_type, byval data as const any ptr, byval nbytes as uinteger) as bool
' main
declare function ggml_init(byval params as ggml_init_params) as ggml_context ptr
declare sub ggml_reset(byval ctx as ggml_context ptr)
declare sub ggml_free(byval ctx as ggml_context ptr)
declare function ggml_used_mem(byval ctx as const ggml_context ptr) as uinteger
declare function ggml_get_no_alloc(byval ctx as ggml_context ptr) as bool
declare sub ggml_set_no_alloc(byval ctx as ggml_context ptr, byval no_alloc as bool)
declare function ggml_get_mem_buffer(byval ctx as const ggml_context ptr) as any ptr
declare function ggml_get_mem_size(byval ctx as const ggml_context ptr) as uinteger
declare function ggml_get_max_tensor_size(byval ctx as const ggml_context ptr) as uinteger
declare function ggml_new_tensor(byval ctx as ggml_context ptr, byval type as ggml_type, byval n_dims as long, byval ne as const longint ptr) as ggml_tensor ptr
declare function ggml_new_tensor_1d(byval ctx as ggml_context ptr, byval type as ggml_type, byval ne0 as longint) as ggml_tensor ptr
declare function ggml_new_tensor_2d(byval ctx as ggml_context ptr, byval type as ggml_type, byval ne0 as longint, byval ne1 as longint) as ggml_tensor ptr
declare function ggml_new_tensor_3d(byval ctx as ggml_context ptr, byval type as ggml_type, byval ne0 as longint, byval ne1 as longint, byval ne2 as longint) as ggml_tensor ptr
declare function ggml_new_tensor_4d(byval ctx as ggml_context ptr, byval type as ggml_type, byval ne0 as longint, byval ne1 as longint, byval ne2 as longint, byval ne3 as longint) as ggml_tensor ptr
declare function ggml_new_buffer(byval ctx as ggml_context ptr, byval nbytes as uinteger) as any ptr
declare function ggml_dup_tensor(byval ctx as ggml_context ptr, byval src as const ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_view_tensor(byval ctx as ggml_context ptr, byval src as ggml_tensor ptr) as ggml_tensor ptr
' Context tensor enumeration and lookup
declare function ggml_get_first_tensor(byval ctx as const ggml_context ptr) as ggml_tensor ptr
declare function ggml_get_next_tensor(byval ctx as const ggml_context ptr, byval tensor as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_get_tensor(byval ctx as ggml_context ptr, byval name as const zstring ptr) as ggml_tensor ptr
' Converts a flat index into coordinates
declare sub ggml_unravel_index(byval tensor as const ggml_tensor ptr, byval i as longint, byval i0 as longint ptr, byval i1 as longint ptr, byval i2 as longint ptr, byval i3 as longint ptr)
declare function ggml_get_unary_op(byval tensor as const ggml_tensor ptr) as ggml_unary_op
declare function ggml_get_glu_op(byval tensor as const ggml_tensor ptr) as ggml_glu_op
declare function ggml_get_data(byval tensor as const ggml_tensor ptr) as any ptr
declare function ggml_get_data_f32(byval tensor as const ggml_tensor ptr) as single ptr
declare function ggml_get_name(byval tensor as const ggml_tensor ptr) as const zstring ptr
declare function ggml_set_name(byval tensor as ggml_tensor ptr, byval name as const zstring ptr) as ggml_tensor ptr
declare function ggml_format_name(byval tensor as ggml_tensor ptr, byval fmt as const zstring ptr, ...) as ggml_tensor ptr
' Tensor flags
declare sub ggml_set_input(byval tensor as ggml_tensor ptr)
declare sub ggml_set_output(byval tensor as ggml_tensor ptr)
declare sub ggml_set_param(byval tensor as ggml_tensor ptr)
declare sub ggml_set_loss(byval tensor as ggml_tensor ptr)
'
' operations on tensors with backpropagation
'
declare function ggml_dup(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' in-place, returns view(a)
declare function ggml_dup_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_add(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_add_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_add_cast(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval type as ggml_type) as ggml_tensor ptr
' dst[i0, i1, i2] = a[i0, i1, i2] + b[i0, ids[i1, i2]]
declare function ggml_add_id(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval ids as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_add1(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_add1_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
' dst = a
' view(dst, nb1, nb2, nb3, offset) += b
' return dst
declare function ggml_acc(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval nb1 as uinteger, byval nb2 as uinteger, byval nb3 as uinteger, byval offset as uinteger) as ggml_tensor ptr
declare function ggml_acc_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval nb1 as uinteger, byval nb2 as uinteger, byval nb3 as uinteger, byval offset as uinteger) as ggml_tensor ptr
declare function ggml_sub(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_sub_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_mul(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_mul_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_div(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_div_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_sqr(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_sqr_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_sqrt(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_sqrt_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_log(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_log_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_expm1(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_expm1_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_softplus(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_softplus_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_sin(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_sin_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_cos(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_cos_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' return scalar
declare function ggml_sum(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
declare function ggml_sum_rows(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_cumsum(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' mean along rows
declare function ggml_mean(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' argmax along rows
declare function ggml_argmax(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' count number of equal elements in a and b
declare function ggml_count_equal(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
' if a is the same shape as b, and a is not parameter, return a
' otherwise, return a new tensor: repeat(a) to fit in b
declare function ggml_repeat(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
' repeat a to the specified shape
declare function ggml_repeat_4d(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval ne0 as longint, byval ne1 as longint, byval ne2 as longint, byval ne3 as longint) as ggml_tensor ptr
' sums repetitions in a into shape of b
declare function ggml_repeat_back(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
' sum up values that are adjacent in dims > 0 instead of repeated with same stride
' concat a and b along dim
' used in stable-diffusion
declare function ggml_concat(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval dim as long) as ggml_tensor ptr
declare function ggml_abs(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_abs_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_sgn(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_sgn_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_neg(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_neg_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_step(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_step_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_tanh(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_tanh_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_elu(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_elu_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_relu(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_leaky_relu(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval negative_slope as single, byval inplace as bool) as ggml_tensor ptr
declare function ggml_relu_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_sigmoid(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_sigmoid_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_gelu(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_gelu_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' GELU using erf (error function) when possible
' some backends may fallback to approximation based on Abramowitz and Stegun formula
declare function ggml_gelu_erf(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_gelu_erf_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_gelu_quick(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_gelu_quick_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_silu(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_silu_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' a - dy
' b - x
declare function ggml_silu_back(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
' hardswish(x) = x * relu6(x + 3) / 6
declare function ggml_hardswish(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' hardsigmoid(x) = relu6(x + 3) / 6
declare function ggml_hardsigmoid(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_exp(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_exp_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_floor(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_floor_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_ceil(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_ceil_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_round(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_round_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
/'*
     * Truncates the fractional part of each element in the tensor (towards zero).
     * For example: trunc(3.7) = 3.0, trunc(-2.9) = -2.0
     * Similar to std::trunc in C/C++.
     '/
declare function ggml_trunc(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_trunc_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' xIELU activation function
' x = x * (c_a(alpha_n) + c_b(alpha_p, beta) * sigmoid(beta * x)) + eps * (x > 0)
' where c_a = softplus and c_b(a, b) = softplus(a) + b are constraining functions
' that constrain the positive and negative source alpha values respectively
declare function ggml_xielu(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval alpha_n as single, byval alpha_p as single, byval beta as single, byval eps as single) as ggml_tensor ptr
' gated linear unit ops
' A: n columns, r rows,
' result is n / 2 columns, r rows,
' expects gate in second half of row, unless swapped is true
declare function ggml_glu(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval op as ggml_glu_op, byval swapped as bool) as ggml_tensor ptr
declare function ggml_reglu(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_reglu_swapped(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_geglu(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_geglu_swapped(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_swiglu(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_swiglu_swapped(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_geglu_erf(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_geglu_erf_swapped(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_geglu_quick(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_geglu_quick_swapped(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' A: n columns, r rows,
' B: n columns, r rows,
declare function ggml_glu_split(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval op as ggml_glu_op) as ggml_tensor ptr
declare function ggml_reglu_split(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_geglu_split(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_swiglu_split(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_geglu_erf_split(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_geglu_quick_split(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_swiglu_oai(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval alpha as single, byval limit as single) as ggml_tensor ptr
declare function ggml_swiglu_clamp(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval limit as single) as ggml_tensor ptr
' normalize along rows
declare function ggml_norm(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval eps as single) as ggml_tensor ptr
declare function ggml_norm_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval eps as single) as ggml_tensor ptr
declare function ggml_rms_norm(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval eps as single) as ggml_tensor ptr
declare function ggml_rms_norm_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval eps as single) as ggml_tensor ptr
' group normalize along ne0*ne1*n_groups
' used in stable-diffusion
declare function ggml_group_norm(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval n_groups as long, byval eps as single) as ggml_tensor ptr
declare function ggml_group_norm_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval n_groups as long, byval eps as single) as ggml_tensor ptr
' l2 normalize along rows
' used in rwkv v7
declare function ggml_l2_norm(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval eps as single) as ggml_tensor ptr
declare function ggml_l2_norm_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval eps as single) as ggml_tensor ptr
' a - x
' b - dy
declare function ggml_rms_norm_back(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval eps as single) as ggml_tensor ptr
' [TAG_GGML_PREC]
' set the minimum required accumulator type for the implementation to use during the compute
' for example:
'  - GGML_PREC_F32  - requires accumulation of the results in F32
'  - GGML_PREC_BF16 - can accumulate the results in BF16, F32
'  - GGML_PREC_F16  - can accumulate the results in F16, F32
'  - GGML_PREC_Q8   - not allowed
'  - GGML_PREC_Q4   - not allowed
'
' return false on faliure
declare function ggml_prec_set_acc(byval a as ggml_tensor ptr, byval prec as ggml_prec) as bool
' [TAG_GGML_PREC]
' set the smallest rank that the implementation can use to internally convert the src[idx] data to
' ranks in decreasing order:
'  - GGML_PREC_F32  - GGML_TYPE_F32
'  - GGML_PREC_BF16 - GGML_TYPE_BF16
'  - GGML_PREC_F16  - GGML_TYPE_F16,
'  - GGML_PREC_Q8   - GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, GGML_TYPE_Q8_K, etc.
'  - GGML_PREC_Q4   - GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_NVFP4, GGML_TYPE_MXFP4, etc.
'
' for example:
'   - ggml_prec_set_src(a, GGML_PREC_Q8, 1):
'     - allows the implementation to quantize F32, BF16, F16 data of src[1] down to GGML_TYPE_Q8_0
'     - cannot quantize it down to GGML_TYPE_Q4_0 or GGML_TYPE_NVFP4
'   - ggml_prec_set_src(a, GGML_PREC_Q4, 1):
'     - allows the implementation to quantize F32, BF16, F16 data of src[1] down to 4-bit datatypes such as GGML_TYPE_Q4_K, GGML_TYPE_NVFP4 etc.
'
' return false on faliure
declare function ggml_prec_set_src(byval a as ggml_tensor ptr, byval prec as ggml_prec, byval idx as long) as bool
' A: k columns, n rows => [ne03, ne02, n, k]
' B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
' result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
declare function ggml_mul_mat(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
' change the precision of a matrix multiplication
' set to GGML_PREC_F32 for higher precision (useful for phi-2)
declare sub ggml_mul_mat_set_prec(byval a as ggml_tensor ptr, byval prec as ggml_prec)
' change the hint of a matrix multiplication
declare sub ggml_mul_mat_set_hint(byval a as ggml_tensor ptr, byval hint as ggml_op_hint)
' indirect matrix multiplication
'
' [TAG_MUL_MAT_ID_DUPLICATE_ID]
' note: this op has undefined behaviour if a given `a` is referenced more than once for a given `b`
'       (i.e. duplicate experts per token are not guaranteed to be supported in all backends)
' TODO: add backend support: https://github.com/ggml-org/llama.cpp/pull/29998#issuecomment-6153742533
declare function ggml_mul_mat_id(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval ids as ggml_tensor ptr) as ggml_tensor ptr
' A: m columns, n rows,
' B: p columns, n rows,
' result is m columns, p rows
declare function ggml_out_prod(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
'
' operations on tensors without backpropagation
'
declare function ggml_scale(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval s as single) as ggml_tensor ptr
' in-place, returns view(a)
declare function ggml_scale_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval s as single) as ggml_tensor ptr
' x = s * a + b
declare function ggml_scale_bias(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval s as single, byval b as single) as ggml_tensor ptr
declare function ggml_scale_bias_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval s as single, byval b as single) as ggml_tensor ptr
' b -> view(a,offset,nb1,nb2,3), return modified a
declare function ggml_set(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval nb1 as uinteger, byval nb2 as uinteger, byval nb3 as uinteger, byval offset as uinteger) as ggml_tensor ptr
' in bytes
' b -> view(a,offset,nb1,nb2,3), return view(a)
declare function ggml_set_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval nb1 as uinteger, byval nb2 as uinteger, byval nb3 as uinteger, byval offset as uinteger) as ggml_tensor ptr
' in bytes
declare function ggml_set_1d(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval offset as uinteger) as ggml_tensor ptr
' in bytes
declare function ggml_set_1d_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval offset as uinteger) as ggml_tensor ptr
' in bytes
' b -> view(a,offset,nb1,nb2,3), return modified a
declare function ggml_set_2d(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval nb1 as uinteger, byval offset as uinteger) as ggml_tensor ptr
' in bytes
' b -> view(a,offset,nb1,nb2,3), return view(a)
declare function ggml_set_2d_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval nb1 as uinteger, byval offset as uinteger) as ggml_tensor ptr
' in bytes
' a -> b, return view(b)
declare function ggml_cpy(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
' note: casting from f32 to i32 will discard the fractional part
declare function ggml_cast(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval type as ggml_type) as ggml_tensor ptr
' make contiguous
declare function ggml_cont(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' make contiguous, with new shape
declare function ggml_cont_1d(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval ne0 as longint) as ggml_tensor ptr
declare function ggml_cont_2d(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval ne0 as longint, byval ne1 as longint) as ggml_tensor ptr
declare function ggml_cont_3d(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval ne0 as longint, byval ne1 as longint, byval ne2 as longint) as ggml_tensor ptr
declare function ggml_cont_4d(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval ne0 as longint, byval ne1 as longint, byval ne2 as longint, byval ne3 as longint) as ggml_tensor ptr
' return view(a), b specifies the new shape
' TODO: when we start computing gradient, make a copy instead of view
declare function ggml_reshape(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
' return view(a)
' TODO: when we start computing gradient, make a copy instead of view
declare function ggml_reshape_1d(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval ne0 as longint) as ggml_tensor ptr
declare function ggml_reshape_2d(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval ne0 as longint, byval ne1 as longint) as ggml_tensor ptr
' return view(a)
' TODO: when we start computing gradient, make a copy instead of view
declare function ggml_reshape_3d(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval ne0 as longint, byval ne1 as longint, byval ne2 as longint) as ggml_tensor ptr
declare function ggml_reshape_4d(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval ne0 as longint, byval ne1 as longint, byval ne2 as longint, byval ne3 as longint) as ggml_tensor ptr
' offset in bytes
declare function ggml_view_1d(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval ne0 as longint, byval offset as uinteger) as ggml_tensor ptr

'' TODO: extern struct ggml_tensor * ggml_view_2d( struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, size_t nb1, [TK_FBCODE ' row stride in bytes] size_t offset);
'' TODO: extern struct ggml_tensor * ggml_view_3d( struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, [TK_FBCODE ' row   stride in bytes] size_t nb2, [TK_FBCODE ' slice stride in bytes] size_t offset);
'' TODO: extern struct ggml_tensor * ggml_view_4d( struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, [TK_FBCODE ' row   stride in bytes] size_t nb2, [TK_FBCODE ' slice stride in bytes] size_t nb3, size_t offset);

declare function ggml_permute(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval axis0 as long, byval axis1 as long, byval axis2 as long, byval axis3 as long) as ggml_tensor ptr
' alias for ggml_permute(ctx, a, 1, 0, 2, 3)
declare function ggml_transpose(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' supports 4D a:
' a     [n_embd, ne1, ne2, ne3]
' b I32 [n_rows, ne2, ne3, 1]
'
' return [n_embd, n_rows, ne2, ne3]

'' TODO: extern struct ggml_tensor * ggml_get_rows( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' data] struct ggml_tensor * b);
' row indices
'' TODO: extern struct ggml_tensor * ggml_get_rows_back( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' gradients of ggml_get_rows result] struct ggml_tensor * b, [TK_FBCODE ' row indices] struct ggml_tensor * c);
' data for ggml_get_rows, only used for its shape
' a TD  [n_embd, ne1,    ne2,    ne3]
' b TS  [n_embd, n_rows, ne02,   ne03] | ne02 == ne2, ne03 == ne3
' c I64 [n_rows, ne11,   ne12,   1]    | c[i] in [0, ne1)
'
' undefined behavior if destination rows overlap
'
' broadcast:
'   ne2 % ne11 == 0
'   ne3 % ne12 == 0
'
' return view(a)
'' TODO: extern struct ggml_tensor * ggml_set_rows( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' destination] struct ggml_tensor * b, [TK_FBCODE ' source] struct ggml_tensor * c);
' row indices

declare function ggml_diag(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' set elements above the diagonal to -INF
declare function ggml_diag_mask_inf(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval n_past as long) as ggml_tensor ptr
' in-place, returns view(a)
declare function ggml_diag_mask_inf_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval n_past as long) as ggml_tensor ptr
' set elements above the diagonal to 0
declare function ggml_diag_mask_zero(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval n_past as long) as ggml_tensor ptr
' in-place, returns view(a)
declare function ggml_diag_mask_zero_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval n_past as long) as ggml_tensor ptr
declare function ggml_clamp(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval min as single, byval max as single) as ggml_tensor ptr
' in-place, returns view(a)
declare function ggml_clamp_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval min as single, byval max as single) as ggml_tensor ptr
declare function ggml_soft_max(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' in-place, returns view(a)
declare function ggml_soft_max_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr) as ggml_tensor ptr
' a    [ne0, ne01, ne02, ne03]
' mask [ne0, ne11, ne12, ne13] | ne11 >= ne01, F16 or F32, optional
'
' broadcast:
'   ne02 % ne12 == 0
'   ne03 % ne13 == 0
'
' fused soft_max(a*scale + mask*(ALiBi slope))
' max_bias = 0.0f for no ALiBi
declare function ggml_soft_max_ext(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval mask as ggml_tensor ptr, byval scale as single, byval max_bias as single) as ggml_tensor ptr
declare function ggml_soft_max_ext_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval mask as ggml_tensor ptr, byval scale as single, byval max_bias as single) as ggml_tensor ptr
declare sub ggml_soft_max_add_sinks(byval a as ggml_tensor ptr, byval sinks as ggml_tensor ptr)
declare function ggml_soft_max_ext_back(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval scale as single, byval max_bias as single) as ggml_tensor ptr
' in-place, returns view(a)
declare function ggml_soft_max_ext_back_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval scale as single, byval max_bias as single) as ggml_tensor ptr
' rotary position embedding
' if (mode & 1) - skip n_past elements (NOT SUPPORTED)
' if (mode & GGML_ROPE_TYPE_NEOX) - GPT-NeoX style
'
' b is an int32 vector with size a->ne[2], it contains the positions
declare function ggml_rope(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval n_dims as long, byval mode as long) as ggml_tensor ptr
' in-place, returns view(a)
declare function ggml_rope_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval n_dims as long, byval mode as long) as ggml_tensor ptr
' RoPE operations with extended options
' a is the input tensor to apply RoPE to, shape [n_embd, n_head, n_token]
' b is an int32 vector with size n_token
' c is freq factors (e.g. phi3-128k), (optional)
' mode can be GGML_ROPE_TYPE_NORMAL or NEOX; for MROPE and VISION mode, use ggml_rope_multi
'
' pseudo-code for computing theta:
'   for i in [0, n_dims/2):
'     theta[i] = b[i] * powf(freq_base, -2.0 * i / n_dims);
'     theta[i] = theta[i] / c[i];  # if c is provided, divide theta by c
'     theta[i] = rope_yarn(theta[i], ...);  # note: theta = theta * freq_scale is applied here
'
' other params are used by YaRN RoPE scaling, these default values will disable YaRN:
'   freq_scale  = 1.0f
'   ext_factor  = 0.0f
'   attn_factor = 1.0f
'   beta_fast   = 0.0f
'   beta_slow   = 0.0f
'
' example:
'   (marking: c = cos, s = sin, 0 = unrotated)
'   given a single head with size = 8 --> [00000000]
'   GGML_ROPE_TYPE_NORMAL  n_dims = 4 --> [cscs0000]
'   GGML_ROPE_TYPE_NORMAL  n_dims = 8 --> [cscscscs]
'   GGML_ROPE_TYPE_NEOX    n_dims = 4 --> [ccss0000]
'   GGML_ROPE_TYPE_NEOX    n_dims = 8 --> [ccccssss]
declare function ggml_rope_ext(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval c as ggml_tensor ptr, byval n_dims as long, byval mode as long, byval n_ctx_orig as long, byval freq_base as single, byval freq_scale as single, byval ext_factor as single, byval attn_factor as single, byval beta_fast as single, byval beta_slow as single) as ggml_tensor ptr
' multi-dimensional RoPE, for Qwen-VL and similar vision models
' mode can be either VISION, MROPE, IMROPE, cannot be combined with NORMAL or NEOX
' sections specify how many dimensions to rotate in each section:
'   section length is equivalent to number of cos/sin pairs, NOT the number of dims
'   (i.e. sum of 4 sections are expected to be n_dims/2)
'   last sections can be 0, means ignored
' all other options are identical to ggml_rope_ext
'
' important note:
'   - NEOX ordering is automatically applied and cannot be disabled for MROPE and VISION
'     if you need normal ordering, there are 2 methods:
'     (1) split the tensor manually using ggml_view
'     (2) permute the weight upon conversion
'   - for VISION, n_dims must be head_size/2
'
' example M-RoPE:
'  given sections = [t=4, y=2, x=2, 0]
'  given a single head with size = 18 --> [000000000000000000]
'  GGML_ROPE_TYPE_MROPE   n_dims = 16 --> [ttttyyxxttttyyxx00] (cos/sin are applied in NEOX ordering)
'  GGML_ROPE_TYPE_IMROPE  n_dims = 16 --> [ttyxttyxttyxttyx00] (interleaved M-RoPE, still NEOX ordering)
'  note: the theta for each dim is computed the same way as ggml_rope_ext, no matter the section
'        in other words, idx used for theta: [0123456789... until n_dims/2], not reset for each section
'
' example vision RoPE:
'  given sections = [y=4, x=4, 0, 0] (last 2 sections are ignored)
'  given a single head with size = 8 --> [00000000]
'  GGML_ROPE_TYPE_VISION  n_dims = 4 --> [yyyyxxxx]
'  other values of n_dims are untested and is undefined behavior
'  note: unlike MROPE, the theta for each dim is computed differently for each section
'        in other words, idx used for theta: [0123] for y section, then [0123] for x section
declare function ggml_rope_multi(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval c as ggml_tensor ptr, byval n_dims as long, byval sections as long ptr, byval mode as long, byval n_ctx_orig as long, byval freq_base as single, byval freq_scale as single, byval ext_factor as single, byval attn_factor as single, byval beta_fast as single, byval beta_slow as single) as ggml_tensor ptr
' in-place, returns view(a)
declare function ggml_rope_ext_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval c as ggml_tensor ptr, byval n_dims as long, byval mode as long, byval n_ctx_orig as long, byval freq_base as single, byval freq_scale as single, byval ext_factor as single, byval attn_factor as single, byval beta_fast as single, byval beta_slow as single) as ggml_tensor ptr
declare function ggml_rope_multi_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval c as ggml_tensor ptr, byval n_dims as long, byval sections as long ptr, byval mode as long, byval n_ctx_orig as long, byval freq_base as single, byval freq_scale as single, byval ext_factor as single, byval attn_factor as single, byval beta_fast as single, byval beta_slow as single) as ggml_tensor ptr
declare function ggml_rope_custom(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval n_dims as long, byval mode as long, byval n_ctx_orig as long, byval freq_base as single, byval freq_scale as single, byval ext_factor as single, byval attn_factor as single, byval beta_fast as single, byval beta_slow as single) as ggml_tensor ptr
declare function ggml_rope_custom_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval n_dims as long, byval mode as long, byval n_ctx_orig as long, byval freq_base as single, byval freq_scale as single, byval ext_factor as single, byval attn_factor as single, byval beta_fast as single, byval beta_slow as single) as ggml_tensor ptr
' compute correction dims for YaRN RoPE scaling
declare sub ggml_rope_yarn_corr_dims(byval n_dims as long, byval n_ctx_orig as long, byval freq_base as single, byval beta_fast as single, byval beta_slow as single, byval dims as single ptr)
' rotary position embedding backward, i.e compute dx from dy
' a - dy
'' TODO: extern struct ggml_tensor * ggml_rope_ext_back( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' gradients of ggml_rope result] struct ggml_tensor * b, [TK_FBCODE ' positions] struct ggml_tensor * c, [TK_FBCODE ' freq factors] int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
declare function ggml_rope_multi_back(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval c as ggml_tensor ptr, byval n_dims as long, byval sections as long ptr, byval mode as long, byval n_ctx_orig as long, byval freq_base as single, byval freq_scale as single, byval ext_factor as single, byval attn_factor as single, byval beta_fast as single, byval beta_slow as single) as ggml_tensor ptr
' set the offset dims for RoPE
' a must be GGML_OP_ROPE or GGML_OP_ROPE_BACK
' vision RoPE is not supported
' example: (marking: x = rotated, 0 = unrotated)
'     n_embd = 10, n_dims = 4, offset = 2 --> [00xxxx0000]
declare function ggml_rope_set_offset(byval a as ggml_tensor ptr, byval n_offs as long) as ggml_tensor ptr
' im2col
' converts data into a format that effectively results in a convolution when combined with matrix multiplication

'' TODO: extern struct ggml_tensor * ggml_im2col( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' convolution kernel] struct ggml_tensor * b, [TK_FBCODE ' data] int s0, [TK_FBCODE ' stride dimension 0] int s1, [TK_FBCODE ' stride dimension 1] int p0, [TK_FBCODE ' padding dimension 0] int p1, [TK_FBCODE ' padding dimension 1] int d0, [TK_FBCODE ' dilation dimension 0] int d1, [TK_FBCODE ' dilation dimension 1] bool is_2D, enum ggml_type dst_type);
'' TODO: extern struct ggml_tensor * ggml_im2col_back( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' convolution kernel] struct ggml_tensor * b, [TK_FBCODE ' gradient of im2col output] int64_t * ne, [TK_FBCODE ' shape of im2col input] int s0, [TK_FBCODE ' stride dimension 0] int s1, [TK_FBCODE ' stride dimension 1] int p0, [TK_FBCODE ' padding dimension 0] int p1, [TK_FBCODE ' padding dimension 1] int d0, [TK_FBCODE ' dilation dimension 0] int d1, [TK_FBCODE ' dilation dimension 1] bool is_2D);
' col2im_1d: scatter-add GEMM columns back to 1D signal
' a: [K*OC, T_in]  (columns from matmul, K = a->ne[0]/OC)
' result: [T_out, OC]  where T_out = (T_in - 1)*s0 + K - 2*p0
'' TODO: extern struct ggml_tensor * ggml_col2im_1d( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' columns [K*OC, T_in]] int s0, [TK_FBCODE ' stride] int oc, [TK_FBCODE ' output channels] int p0);
' padding to crop from both sides
'' TODO: extern struct ggml_tensor * ggml_conv_1d( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' convolution kernel] struct ggml_tensor * b, [TK_FBCODE ' data] int s0, [TK_FBCODE ' stride] int p0, [TK_FBCODE ' padding] int d0);
' dilation
' conv_1d with padding = half
' alias for ggml_conv_1d(a, b, s, a->ne[0]/2, d)
'' TODO: extern struct ggml_tensor* ggml_conv_1d_ph( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' convolution kernel] struct ggml_tensor * b, [TK_FBCODE ' data] int s, [TK_FBCODE ' stride] int d);
' dilation
' depthwise
' TODO: this is very likely wrong for some cases! - needs more testing
'' TODO: extern struct ggml_tensor * ggml_conv_1d_dw( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' convolution kernel] struct ggml_tensor * b, [TK_FBCODE ' data] int s0, [TK_FBCODE ' stride] int p0, [TK_FBCODE ' padding] int d0);
' dilation
'' TODO: extern struct ggml_tensor * ggml_conv_1d_dw_ph( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' convolution kernel] struct ggml_tensor * b, [TK_FBCODE ' data] int s0, [TK_FBCODE ' stride] int d0);
' dilation
'' TODO: extern struct ggml_tensor * ggml_conv_transpose_1d( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' convolution kernel] struct ggml_tensor * b, [TK_FBCODE ' data] int s0, [TK_FBCODE ' stride] int p0, [TK_FBCODE ' padding] int d0);
' dilation
'' TODO: extern struct ggml_tensor * ggml_conv_2d( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' convolution kernel] struct ggml_tensor * b, [TK_FBCODE ' data] int s0, [TK_FBCODE ' stride dimension 0] int s1, [TK_FBCODE ' stride dimension 1] int p0, [TK_FBCODE ' padding dimension 0] int p1, [TK_FBCODE ' padding dimension 1] int d0, [TK_FBCODE ' dilation dimension 0] int d1);
' dilation dimension 1
'' TODO: extern struct ggml_tensor * ggml_im2col_3d( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int64_t IC, int s0, [TK_FBCODE ' stride width] int s1, [TK_FBCODE ' stride height] int s2, [TK_FBCODE ' stride depth] int p0, [TK_FBCODE ' padding width] int p1, [TK_FBCODE ' padding height] int p2, [TK_FBCODE ' padding depth] int d0, [TK_FBCODE ' dilation width] int d1, [TK_FBCODE ' dilation height] int d2, [TK_FBCODE ' dilation depth] enum ggml_type dst_type);
' a: [OC*IC, KD, KH, KW]
' b: [N*IC, ID, IH, IW]
' result: [N*OC, OD, OH, OW]
'' TODO: extern struct ggml_tensor * ggml_conv_3d( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int64_t IC, int s0, [TK_FBCODE ' stride width] int s1, [TK_FBCODE ' stride height] int s2, [TK_FBCODE ' stride depth] int p0, [TK_FBCODE ' padding width] int p1, [TK_FBCODE ' padding height] int p2, [TK_FBCODE ' padding depth] int d0, [TK_FBCODE ' dilation width] int d1, [TK_FBCODE ' dilation height] int d2 [TK_FBCODE ' dilation depth] );
' kernel size is a->ne[0] x a->ne[1]
' stride is equal to kernel size
' padding is zero
' example:
' a:     16   16    3  768
' b:   1024 1024    3    1
' res:   64   64  768    1
' used in sam

declare function ggml_conv_2d_sk_p0(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
' kernel size is a->ne[0] x a->ne[1]
' stride is 1
' padding is half
' example:
' a:      3    3    256  256
' b:     64   64    256    1
' res:   64   64    256    1
' used in sam
declare function ggml_conv_2d_s1_ph(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr) as ggml_tensor ptr
' depthwise (via im2col and mul_mat)

'' TODO: extern struct ggml_tensor * ggml_conv_2d_dw( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' convolution kernel] struct ggml_tensor * b, [TK_FBCODE ' data] int s0, [TK_FBCODE ' stride dimension 0] int s1, [TK_FBCODE ' stride dimension 1] int p0, [TK_FBCODE ' padding dimension 0] int p1, [TK_FBCODE ' padding dimension 1] int d0, [TK_FBCODE ' dilation dimension 0] int d1);
' dilation dimension 1
' Depthwise 2D convolution
' may be faster than ggml_conv_2d_dw, but not available in all backends
' a:   KW    KH    1    C    convolution kernel
' b:   W     H     C    N    input data
' res: W_out H_out C    N
declare function ggml_conv_2d_dw_direct(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval stride0 as long, byval stride1 as long, byval pad0 as long, byval pad1 as long, byval dilation0 as long, byval dilation1 as long) as ggml_tensor ptr
declare function ggml_conv_transpose_2d_p0(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval stride as long) as ggml_tensor ptr
'' TODO: extern struct ggml_tensor * ggml_conv_2d_direct( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' convolution kernel [KW, KH, IC, OC]] struct ggml_tensor * b, [TK_FBCODE ' input data [W, H, C, N]] int s0, [TK_FBCODE ' stride dimension 0] int s1, [TK_FBCODE ' stride dimension 1] int p0, [TK_FBCODE ' padding dimension 0] int p1, [TK_FBCODE ' padding dimension 1] int d0, [TK_FBCODE ' dilation dimension 0] int d1);
' dilation dimension 1
'' TODO: extern struct ggml_tensor * ggml_conv_3d_direct( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' kernel [KW, KH, KD, IC * OC]] struct ggml_tensor * b, [TK_FBCODE ' input  [W, H, D, C * N]] int s0, [TK_FBCODE ' stride] int s1, int s2, int p0, [TK_FBCODE ' padding] int p1, int p2, int d0, [TK_FBCODE ' dilation] int d1, int d2, int n_channels, int n_batch, int n_channels_out);

enum ggml_op_pool
	GGML_OP_POOL_MAX
	GGML_OP_POOL_AVG
	GGML_OP_POOL_COUNT
end enum

'' TODO: extern struct ggml_tensor * ggml_pool_1d( struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_op_pool op, int k0, [TK_FBCODE ' kernel size] int s0, [TK_FBCODE ' stride] int p0);
' padding
' the result will have 2*p0 padding for the first dimension
' and 2*p1 padding for the second dimension
declare function ggml_pool_2d(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval op as ggml_op_pool, byval k0 as long, byval k1 as long, byval s0 as long, byval s1 as long, byval p0 as single, byval p1 as single) as ggml_tensor ptr
'' TODO: extern struct ggml_tensor * ggml_pool_2d_back( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * af, [TK_FBCODE ' "a"/input used in forward pass] enum ggml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1);

enum ggml_scale_mode
	GGML_SCALE_MODE_NEAREST = 0
	GGML_SCALE_MODE_BILINEAR = 1
	GGML_SCALE_MODE_BICUBIC = 2
	GGML_SCALE_MODE_COUNT
end enum

enum ggml_scale_flag
	GGML_SCALE_FLAG_ALIGN_CORNERS = 1 shl 8
	GGML_SCALE_FLAG_ANTIALIAS = 1 shl 9
end enum

' interpolate
' multiplies ne0 and ne1 by scale factor
declare function ggml_upscale(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval scale_factor as long, byval mode as ggml_scale_mode) as ggml_tensor ptr
' interpolate
' interpolate scale to specified dimensions
declare function ggml_upscale_ext(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval ne0 as long, byval ne1 as long, byval ne2 as long, byval ne3 as long, byval mode as ggml_scale_mode) as ggml_tensor ptr
' Up- or downsamples the input to the specified size.
' 2D scale modes (eg. bilinear) are applied to the first two dimensions.
declare function ggml_interpolate(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval ne0 as longint, byval ne1 as longint, byval ne2 as longint, byval ne3 as longint, byval mode as ulong) as ggml_tensor ptr
' ggml_scale_mode [ | ggml_scale_flag...]
' pad each dimension with zeros: [x, ..., x] -> [x, ..., x, 0, ..., 0]
declare function ggml_pad(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval p0 as long, byval p1 as long, byval p2 as long, byval p3 as long) as ggml_tensor ptr
' pad each dimension with values on the other side of the torus (looping around)
declare function ggml_pad_circular(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval p0 as long, byval p1 as long, byval p2 as long, byval p3 as long) as ggml_tensor ptr
declare function ggml_pad_ext(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval lp0 as long, byval rp0 as long, byval lp1 as long, byval rp1 as long, byval lp2 as long, byval rp2 as long, byval lp3 as long, byval rp3 as long) as ggml_tensor ptr
' pad each dimension with values on the other side of the torus (looping around)
declare function ggml_pad_ext_circular(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval lp0 as long, byval rp0 as long, byval lp1 as long, byval rp1 as long, byval lp2 as long, byval rp2 as long, byval lp3 as long, byval rp3 as long) as ggml_tensor ptr
' pad each dimension with reflection: [a, b, c, d] -> [b, a, b, c, d, c]
declare function ggml_pad_reflect_1d(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval p0 as long, byval p1 as long) as ggml_tensor ptr
' Move tensor elements by an offset given for each dimension. Elements that
' are shifted beyond the last position are wrapped around to the beginning.
declare function ggml_roll(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval shift0 as long, byval shift1 as long, byval shift2 as long, byval shift3 as long) as ggml_tensor ptr
' Convert matrix into a triangular one (upper, strict upper, lower or strict lower) by writing
' zeroes everywhere outside the masked area
declare function ggml_tri(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval type as ggml_tri_type) as ggml_tensor ptr
' Fill tensor a with constant c
declare function ggml_fill(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval c as single) as ggml_tensor ptr
declare function ggml_fill_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval c as single) as ggml_tensor ptr
' Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
' timesteps: [N,]
' return: [N, dim]
declare function ggml_timestep_embedding(byval ctx as ggml_context ptr, byval timesteps as ggml_tensor ptr, byval dim as long, byval max_period as long) as ggml_tensor ptr
' sort rows

enum ggml_sort_order
	GGML_SORT_ORDER_ASC
	GGML_SORT_ORDER_DESC
end enum

declare function ggml_argsort(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval order as ggml_sort_order) as ggml_tensor ptr
' similar to ggml_top_k but implemented as `argsort` + `view`
declare function ggml_argsort_top_k(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval k as long) as ggml_tensor ptr
' top k elements per row
' note: the resulting top k indices are in no particular order
declare function ggml_top_k(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval k as long) as ggml_tensor ptr
declare function ggml_arange(byval ctx as ggml_context ptr, byval start as single, byval stop as single, byval step as single) as ggml_tensor ptr
' q:    [n_embd_k, n_batch, n_head,    ne3 ]
' k:    [n_embd_k, n_kv,    n_head_kv, ne3 ]
' v:    [n_embd_v, n_kv,    n_head_kv, ne3 ] !! not transposed !!
' mask: [n_kv,     n_batch, ne32,      ne33]
'   or per-query key limits (I32): [1, n_batch, ne32, ne33], KV row k is seen by query row j iff k <= limit[j]
'   (backends without support for limits report the op as unsupported)
' res:  [n_embd_v, n_head,  n_batch,   ne3 ] !! permuted !!
'
' broadcast:
'   n_head % n_head_kv == 0
'   n_head % ne32      == 0
'   ne3    % ne33      == 0
'
declare function ggml_flash_attn_ext(byval ctx as ggml_context ptr, byval q as ggml_tensor ptr, byval k as ggml_tensor ptr, byval v as ggml_tensor ptr, byval mask as ggml_tensor ptr, byval scale as single, byval max_bias as single, byval logit_softcap as single) as ggml_tensor ptr
declare sub ggml_flash_attn_ext_set_prec(byval a as ggml_tensor ptr, byval prec as ggml_prec)
declare function ggml_flash_attn_ext_get_prec(byval a as const ggml_tensor ptr) as ggml_prec
' Use finite mask entries as a sparse K/V set. Set 0 to disable.
' n_kv_max must bound the number of finite entries in every mask row.
declare sub ggml_flash_attn_ext_set_n_kv_max(byval a as ggml_tensor ptr, byval n_kv_max as long)
declare sub ggml_flash_attn_ext_add_sinks(byval a as ggml_tensor ptr, byval sinks as ggml_tensor ptr)
' TODO: needs to be adapted to ggml_flash_attn_ext
declare function ggml_flash_attn_back(byval ctx as ggml_context ptr, byval q as ggml_tensor ptr, byval k as ggml_tensor ptr, byval v as ggml_tensor ptr, byval d as ggml_tensor ptr, byval masked as bool) as ggml_tensor ptr
declare function ggml_ssm_conv(byval ctx as ggml_context ptr, byval sx as ggml_tensor ptr, byval c as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_ssm_scan(byval ctx as ggml_context ptr, byval s as ggml_tensor ptr, byval x as ggml_tensor ptr, byval dt as ggml_tensor ptr, byval A as ggml_tensor ptr, byval B as ggml_tensor ptr, byval C as ggml_tensor ptr, byval ids as ggml_tensor ptr, byval K as longint) as ggml_tensor ptr
' partition into non-overlapping windows with padding if needed
' example:
' a:   768   64   64    1
' w:    14
' res: 768   14   14    25
' used in sam
declare function ggml_win_part(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval w as long) as ggml_tensor ptr
' reverse of ggml_win_part
' used in sam
declare function ggml_win_unpart(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval w0 as long, byval h0 as long, byval w as long) as ggml_tensor ptr
declare function ggml_unary(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval op as ggml_unary_op) as ggml_tensor ptr
declare function ggml_unary_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval op as ggml_unary_op) as ggml_tensor ptr
' used in sam
declare function ggml_get_rel_pos(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval qh as long, byval kh as long) as ggml_tensor ptr
' used in sam
declare function ggml_add_rel_pos(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval pw as ggml_tensor ptr, byval ph as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_add_rel_pos_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval pw as ggml_tensor ptr, byval ph as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_rwkv_wkv6(byval ctx as ggml_context ptr, byval k as ggml_tensor ptr, byval v as ggml_tensor ptr, byval r as ggml_tensor ptr, byval tf as ggml_tensor ptr, byval td as ggml_tensor ptr, byval state as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_gated_linear_attn(byval ctx as ggml_context ptr, byval k as ggml_tensor ptr, byval v as ggml_tensor ptr, byval q as ggml_tensor ptr, byval g as ggml_tensor ptr, byval state as ggml_tensor ptr, byval scale as single) as ggml_tensor ptr
declare function ggml_rwkv_wkv7(byval ctx as ggml_context ptr, byval r as ggml_tensor ptr, byval w as ggml_tensor ptr, byval k as ggml_tensor ptr, byval v as ggml_tensor ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval state as ggml_tensor ptr) as ggml_tensor ptr
/' Solves a specific equation of the form Ax=B, where A is a triangular matrix
    *  without zeroes on the diagonal (i.e. invertible).
    *  B can have any number of columns, but must have the same number of rows as A
    *  If A is [n, n] and B is [n, m], then the result will be [n, m] as well
    *  Has O(n^3) complexity (unlike most matrix ops out there), so use on cases
    *  where n > 100 sparingly, pre-chunk if necessary.
    *
    *  If left = false, solves xA=B instead
    *  If lower = false, assumes upper triangular instead
    *  If uni = true, assumes diagonal of A to be all ones (will override actual values)
    *
    *  TODO: currently only lower, right, non-unitriangular variant is implemented
    '/
declare function ggml_solve_tri(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval left as bool, byval lower as bool, byval uni as bool) as ggml_tensor ptr
' TODO: add ggml_gated_delta_net_set_bcast() to be able to configure Q, K broadcast type: tiled vs interleaved [TAG_GGML_GDN_BCAST]
' ref: https://github.com/ggml-org/llama.cpp/pull/19468#discussion_r2786394306
'
' tensor shapes (S_k == S_v, H_v % H_k == 0):
'   q, k  : [S_k, H_k, n_tokens, n_seqs]
'   v     : [S_v, H_v, n_tokens, n_seqs]
'   g     : [1, H_v, n_tokens, n_seqs] (scalar gate) or [S_v, H_v, n_tokens, n_seqs] (KDA)
'   beta  : [1, H_v, n_tokens, n_seqs]
'   state : [S_v, S_v, H_v, n_seqs] -- initial recurrent state s0
'
' the output packs the attention scores [S_v, H_v, n_tokens, n_seqs] followed by K state
' snapshots, most-recent first (slot 0 = final state, slot s = state s tokens back). K == 1
' keeps only the final state; when n_tokens < K only slots 0..n_tokens-1 are written.
' when n_tokens > K only slot 0 is guaranteed; backends may leave slots 1..K-1 unwritten.
declare function ggml_gated_delta_net(byval ctx as ggml_context ptr, byval q as ggml_tensor ptr, byval k as ggml_tensor ptr, byval v as ggml_tensor ptr, byval g as ggml_tensor ptr, byval beta as ggml_tensor ptr, byval state as ggml_tensor ptr, byval K as longint) as ggml_tensor ptr
' DSA lightning indexer
'
' q:       [n_embd_idx, n_head_idx, n_batch, ne3 ]
' k:       [n_embd_idx, 1,          n_kv,    ne3 ]
' weights: [n_head_idx, n_batch,    1,       ne3 ] !! prescaled !!
' mask:    [n_kv,       n_batch,    1,       ne33] !! f16 !!
' res:     [n_kv,       n_batch,    1,       ne3 ]
'
' broadcast:
'   ne3 % ne33 == 0
'
declare function ggml_lightning_indexer(byval ctx as ggml_context ptr, byval q as ggml_tensor ptr, byval k as ggml_tensor ptr, byval weights as ggml_tensor ptr, byval mask as ggml_tensor ptr) as ggml_tensor ptr
' DeepSeek V4 hyper-connections (ref. https://arxiv.org/pdf/2512.24880)
' In short these operations are replacements for the original residual connection (x = transformer(x) + x)
' using a richer representation through streams.
'
' hc_comb: mixes [(2 + hc)*hc, n_tokens], scale [3], base [(2 + hc)*hc]
'          -> [dst_hc, src_hc, n_tokens]
' logits[dst, src, t] = mixes[2*hc + dst + hc*src, t]*scale[2]
'                         + base[2*hc + dst + hc*src]
' Softmax over dst, add eps, normalize over src, then repeat normalization
' over dst followed by src for iterations 1 through n_iter - 1.
declare function ggml_dsv4_hc_comb(byval ctx as ggml_context ptr, byval mixes as ggml_tensor ptr, byval scale as ggml_tensor ptr, byval base as ggml_tensor ptr, byval eps as single, byval n_iter as long) as ggml_tensor ptr
' hc_pre: x [n_embd, hc, n_tokens], weights [hc, n_tokens] -> [n_embd, n_tokens]
'   result[i, t] = sum_h x[i, h, t]*weights[h, t]
'
declare function ggml_dsv4_hc_pre(byval ctx as ggml_context ptr, byval x as ggml_tensor ptr, byval weights as ggml_tensor ptr) as ggml_tensor ptr
' hc_pre with a per-element gate (Qwen3.8-Flash-Next): gate [n_embd, hc, n_tokens]
'   result[i, t] = scale*sum_h x[i, h, t]*sigmoid(gate[i, h, t])
'
declare function ggml_dsv4_hc_pre_gated(byval ctx as ggml_context ptr, byval x as ggml_tensor ptr, byval gate as ggml_tensor ptr, byval scale as single) as ggml_tensor ptr
' hc_post: x [n_embd, n_tokens], residual [n_embd, hc, n_tokens],
'          post [hc, n_tokens], comb [dst_hc, src_hc, n_tokens]
'          -> [n_embd, hc, n_tokens]
'   result[i, dst, t] = x[i, t]*post[dst, t]
'                       + sum_src residual[i, src, t]*comb[dst, src, t]
'   comb == NULL uses the identity: result[i, dst, t] = x[i, t]*post[dst, t] + residual[i, dst, t]
'
declare function ggml_dsv4_hc_post(byval ctx as ggml_context ptr, byval x as ggml_tensor ptr, byval residual as ggml_tensor ptr, byval post as ggml_tensor ptr, byval comb as ggml_tensor ptr) as ggml_tensor ptr
' MLA: per-head K and V of a head group, expanded from the latent KV cache (prefill attention on expanded K/V)
'   kv     [rank + n_rope, n_kv] F16 rows of the latent cache (row stride kv->nb[1]): latent | shared RoPE part
'   k_b    [n_nope, rank, n_head] (W_UK per head, any type with a dequantization to F16)
'   v_b    [rank, n_v, n_head]    (W_UV per head, any type with a dequantization to F16)
'   result F16, 1-D: K [n_nope + n_rope, n_heads, n_kv] (W_UK latent | RoPE part) followed by V [n_v, n_heads, n_kv],
'          for heads head0 .. head0 + n_heads - 1; products accumulate in F32 and round to F16 once
declare function ggml_mla_expand_kv(byval ctx as ggml_context ptr, byval kv as ggml_tensor ptr, byval k_b as ggml_tensor ptr, byval v_b as ggml_tensor ptr, byval head0 as long, byval n_heads as long) as ggml_tensor ptr
' custom operators

type ggml_custom1_op_t as sub(byval dst as ggml_tensor ptr, byval a as const ggml_tensor ptr, byval ith as long, byval nth as long, byval userdata as any ptr)
type ggml_custom2_op_t as sub(byval dst as ggml_tensor ptr, byval a as const ggml_tensor ptr, byval b as const ggml_tensor ptr, byval ith as long, byval nth as long, byval userdata as any ptr)
type ggml_custom3_op_t as sub(byval dst as ggml_tensor ptr, byval a as const ggml_tensor ptr, byval b as const ggml_tensor ptr, byval c as const ggml_tensor ptr, byval ith as long, byval nth as long, byval userdata as any ptr)
const GGML_N_TASKS_MAX = -1
' n_tasks == GGML_N_TASKS_MAX means to use max number of tasks

declare function ggml_map_custom1(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval fun as ggml_custom1_op_t, byval n_tasks as long, byval userdata as any ptr) as ggml_tensor ptr
declare function ggml_map_custom1_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval fun as ggml_custom1_op_t, byval n_tasks as long, byval userdata as any ptr) as ggml_tensor ptr
declare function ggml_map_custom2(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval fun as ggml_custom2_op_t, byval n_tasks as long, byval userdata as any ptr) as ggml_tensor ptr
declare function ggml_map_custom2_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval fun as ggml_custom2_op_t, byval n_tasks as long, byval userdata as any ptr) as ggml_tensor ptr
declare function ggml_map_custom3(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval c as ggml_tensor ptr, byval fun as ggml_custom3_op_t, byval n_tasks as long, byval userdata as any ptr) as ggml_tensor ptr
declare function ggml_map_custom3_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval b as ggml_tensor ptr, byval c as ggml_tensor ptr, byval fun as ggml_custom3_op_t, byval n_tasks as long, byval userdata as any ptr) as ggml_tensor ptr
type ggml_custom_op_t as sub(byval dst as ggml_tensor ptr, byval ith as long, byval nth as long, byval userdata as any ptr)
declare function ggml_custom_4d(byval ctx as ggml_context ptr, byval type as ggml_type, byval ne0 as longint, byval ne1 as longint, byval ne2 as longint, byval ne3 as longint, byval args as ggml_tensor ptr ptr, byval n_args as long, byval fun as ggml_custom_op_t, byval n_tasks as long, byval userdata as any ptr) as ggml_tensor ptr
declare function ggml_custom_inplace(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval args as ggml_tensor ptr ptr, byval n_args as long, byval fun as ggml_custom_op_t, byval n_tasks as long, byval userdata as any ptr) as ggml_tensor ptr
' loss function

'' TODO: extern struct ggml_tensor * ggml_cross_entropy_loss( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' logits] struct ggml_tensor * b);
' labels
'' TODO: extern struct ggml_tensor * ggml_cross_entropy_loss_back( struct ggml_context * ctx, struct ggml_tensor * a, [TK_FBCODE ' logits] struct ggml_tensor * b, [TK_FBCODE ' labels] struct ggml_tensor * c);
' gradients of cross_entropy_loss result
' AdamW optimizer step
' Paper: https://arxiv.org/pdf/1711.05101v3.pdf
' PyTorch: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

declare function ggml_opt_step_adamw(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval grad as ggml_tensor ptr, byval m as ggml_tensor ptr, byval v as ggml_tensor ptr, byval adamw_params as ggml_tensor ptr) as ggml_tensor ptr
' parameters such as the learning rate
' stochastic gradient descent step (with weight decay)
declare function ggml_opt_step_sgd(byval ctx as ggml_context ptr, byval a as ggml_tensor ptr, byval grad as ggml_tensor ptr, byval sgd_params as ggml_tensor ptr) as ggml_tensor ptr
' alpha, weight decay
' build forward multiple tensors and select one of them for computing
' this is useful for creating graphs that have constant topology but compute different things based on the input
' ref: https://github.com/ggml-org/llama.cpp/pull/18550
'
' nodes:
'   | - build forward into the graph but do not compute
'   c - build forward into the graph and compute
'
'    |  |  ...  c  ...  |
'    |  |  ...  c  ...  |
'    |  |  ...  c  ...  |
'   [0  1  ... idx ...  n-1]        <-- ggml_build_forward_select(..., n, idx)
'               c
'               c
'
' example:
'   struct ggml_tensor * curs[3];
'
'   curs[0]  = compute0(...);
'   curs[1]  = compute1(...);
'   curs[2]  = compute2(...);
'
'   int idx = select_branch(some_input);
'
'   struct ggml_tensor * out = ggml_build_forward_select(cgraph, curs, 3, idx);
'
declare function ggml_build_forward_select(byval cgraph as ggml_cgraph ptr, byval tensors as ggml_tensor ptr ptr, byval n_tensors as long, byval idx as long) as ggml_tensor ptr
declare sub ggml_build_forward_expand(byval cgraph as ggml_cgraph ptr, byval tensor as ggml_tensor ptr)
' add the tensor and its parents to the graph without marking them for compute
' the flag is set later, when the tensor is reached from a node that computes
declare sub ggml_build_forward_order(byval cgraph as ggml_cgraph ptr, byval tensor as ggml_tensor ptr)
'' TODO: extern void ggml_build_backward_expand( struct ggml_context * ctx, [TK_FBCODE ' context for gradient computation] struct ggml_cgraph * cgraph, struct ggml_tensor ** grad_accs);
' graph allocation in a context
declare function ggml_new_graph(byval ctx as ggml_context ptr) as ggml_cgraph ptr ' size = GGML_DEFAULT_GRAPH_SIZE, grads = false
declare function ggml_new_graph_custom(byval ctx as ggml_context ptr, byval size as uinteger, byval grads as bool) as ggml_cgraph ptr
declare function ggml_graph_dup(byval ctx as ggml_context ptr, byval cgraph as ggml_cgraph ptr, byval force_grads as bool) as ggml_cgraph ptr
declare sub ggml_graph_cpy(byval src as ggml_cgraph ptr, byval dst as ggml_cgraph ptr)
declare sub ggml_graph_reset(byval cgraph as ggml_cgraph ptr) ' set regular grads + optimizer momenta to 0, set loss grad to 1
declare sub ggml_graph_clear(byval cgraph as ggml_cgraph ptr)
declare function ggml_graph_size(byval cgraph as ggml_cgraph ptr) as long
declare function ggml_graph_node(byval cgraph as ggml_cgraph ptr, byval i as long) as ggml_tensor ptr ' if i < 0, returns nodes[n_nodes + i]
declare function ggml_graph_nodes(byval cgraph as ggml_cgraph ptr) as ggml_tensor ptr ptr
declare function ggml_graph_n_nodes(byval cgraph as ggml_cgraph ptr) as long
declare sub ggml_graph_add_node(byval cgraph as ggml_cgraph ptr, byval tensor as ggml_tensor ptr)
declare function ggml_graph_overhead() as uinteger
declare function ggml_graph_overhead_custom(byval size as uinteger, byval grads as bool) as uinteger
declare function ggml_graph_get_tensor(byval cgraph as const ggml_cgraph ptr, byval name as const zstring ptr) as ggml_tensor ptr
declare function ggml_graph_get_grad(byval cgraph as const ggml_cgraph ptr, byval node as const ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_graph_get_grad_acc(byval cgraph as const ggml_cgraph ptr, byval node as const ggml_tensor ptr) as ggml_tensor ptr
' print info and performance information for the graph
declare sub ggml_graph_print(byval cgraph as const ggml_cgraph ptr)
' dump the graph into a file using the dot format
declare sub ggml_graph_dump_dot(byval gb as const ggml_cgraph ptr, byval cgraph as const ggml_cgraph ptr, byval filename as const zstring ptr)
' TODO these functions were sandwiched in the old optimization interface, is there a better place for them?

type ggml_log_callback as sub(byval level as ggml_log_level, byval text as const zstring ptr, byval user_data as any ptr)
' Set callback for all future logging events.
' If this is not called, or NULL is supplied, everything is output on stderr.

declare sub ggml_log_get(byval log_callback as ggml_log_callback ptr, byval user_data as any ptr ptr)
declare sub ggml_log_set(byval log_callback as ggml_log_callback, byval user_data as any ptr)
declare function ggml_set_zero(byval tensor as ggml_tensor ptr) as ggml_tensor ptr
'
' quantization
'
' - ggml_quantize_init can be called multiple times with the same type
'   it will only initialize the quantization tables for the first call or after ggml_quantize_free
'   automatically called by ggml_quantize_chunk for convenience
'
' - ggml_quantize_free will free any memory allocated by ggml_quantize_init
'   call this at the end of the program to avoid memory leaks
'
' note: these are thread-safe
'
declare sub ggml_quantize_init(byval type as ggml_type)
declare sub ggml_quantize_free()
' some quantization type cannot be used without an importance matrix
declare function ggml_quantize_requires_imatrix(byval type as ggml_type) as bool
' calls ggml_quantize_init internally (i.e. can allocate memory)
declare function ggml_quantize_chunk(byval type as ggml_type, byval src as const single ptr, byval dst as any ptr, byval start as longint, byval nrows as longint, byval n_per_row as longint, byval imatrix as const single ptr) as uinteger
'' TODO: # define GGML_RESTRICT restrict
type ggml_to_float_t as sub(byval x as const any ptr, byval y as single ptr, byval k as longint)
type ggml_from_float_t as sub(byval x as const single ptr, byval y as any ptr, byval k as longint)

type ggml_type_traits
	type_name as const zstring ptr
	blck_size as longint
	blck_size_interleave as longint ' interleave elements in blocks
	type_size as uinteger
	is_quantized as bool
	to_float as ggml_to_float_t
	from_float_ref as ggml_from_float_t
end type

declare function ggml_get_type_traits(byval type as ggml_type) as const ggml_type_traits ptr
' ggml threadpool
' TODO: currently, only a few functions are in the base ggml API, while the rest are in the CPU backend
' the goal should be to create an API that other backends can use move everything to the ggml base
' scheduling priorities

enum ggml_sched_priority
	GGML_SCHED_PRIO_LOW = -1
	GGML_SCHED_PRIO_NORMAL
	GGML_SCHED_PRIO_MEDIUM
	GGML_SCHED_PRIO_HIGH
	GGML_SCHED_PRIO_REALTIME
end enum

' threadpool params
' Use ggml_threadpool_params_default() or ggml_threadpool_params_init() to populate the defaults

type ggml_threadpool_params
	cpumask(0 to 511) as bool ' mask of cpu cores (all-zeros means use default affinity settings)
	n_threads as long ' number of threads
	prio as ggml_sched_priority ' thread priority
	poll as ulong ' polling level (0 - no polling, 100 - aggressive polling)
	strict_cpu as bool ' strict cpu placement
	paused as bool ' start in paused state
end type

' forward declaration, see ggml.c
type ggml_threadpool_t as ggml_threadpool ptr
declare function ggml_threadpool_params_default(byval n_threads as long) as ggml_threadpool_params
declare sub ggml_threadpool_params_init(byval p as ggml_threadpool_params ptr, byval n_threads as long)
declare function ggml_threadpool_params_match(byval p0 as const ggml_threadpool_params ptr, byval p1 as const ggml_threadpool_params ptr) as bool

end extern

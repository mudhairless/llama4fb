#pragma once

#include once "ggml.bi"
#include once "ggml-backend.bi"

extern "C"

' the compute plan that needs to be prepared for ggml_graph_compute()
' since https://github.com/ggml-org/ggml/issues/287

type ggml_cplan
	work_size as uinteger ' size of work buffer, calculated by `ggml_graph_plan()`
	work_data as ubyte ptr ' work buffer, to be allocated by caller before calling to `ggml_graph_compute()`
	n_threads as long
	threadpool as ggml_threadpool ptr
	' abort ggml_graph_compute when true
	abort_callback as ggml_abort_callback
	abort_callback_data as any ptr
	' use only reference implementations
	use_ref as bool
end type

' numa strategies

enum ggml_numa_strategy
	GGML_NUMA_STRATEGY_DISABLED = 0
	GGML_NUMA_STRATEGY_DISTRIBUTE = 1
	GGML_NUMA_STRATEGY_ISOLATE = 2
	GGML_NUMA_STRATEGY_NUMACTL = 3
	GGML_NUMA_STRATEGY_MIRROR = 4
	GGML_NUMA_STRATEGY_COUNT
end enum

declare sub ggml_numa_init(byval numa as ggml_numa_strategy) ' call once for better performance on NUMA systems
declare function ggml_is_numa() as bool ' true if init detected that system has >1 NUMA node
declare function ggml_new_i32(byval ctx as ggml_context ptr, byval value as long) as ggml_tensor ptr
declare function ggml_new_f32(byval ctx as ggml_context ptr, byval value as single) as ggml_tensor ptr
declare function ggml_set_i32(byval tensor as ggml_tensor ptr, byval value as long) as ggml_tensor ptr
declare function ggml_set_f32(byval tensor as ggml_tensor ptr, byval value as single) as ggml_tensor ptr
declare function ggml_get_i32_1d(byval tensor as const ggml_tensor ptr, byval i as long) as long
declare sub ggml_set_i32_1d(byval tensor as const ggml_tensor ptr, byval i as long, byval value as long)
declare function ggml_get_i32_nd(byval tensor as const ggml_tensor ptr, byval i0 as long, byval i1 as long, byval i2 as long, byval i3 as long) as long
declare sub ggml_set_i32_nd(byval tensor as const ggml_tensor ptr, byval i0 as long, byval i1 as long, byval i2 as long, byval i3 as long, byval value as long)
declare function ggml_get_f32_1d(byval tensor as const ggml_tensor ptr, byval i as long) as single
declare sub ggml_set_f32_1d(byval tensor as const ggml_tensor ptr, byval i as long, byval value as single)
declare function ggml_get_f32_nd(byval tensor as const ggml_tensor ptr, byval i0 as long, byval i1 as long, byval i2 as long, byval i3 as long) as single
declare sub ggml_set_f32_nd(byval tensor as const ggml_tensor ptr, byval i0 as long, byval i1 as long, byval i2 as long, byval i3 as long, byval value as single)
declare function ggml_threadpool_new(byval params as ggml_threadpool_params ptr) as ggml_threadpool ptr
declare sub ggml_threadpool_free(byval threadpool as ggml_threadpool ptr)
declare function ggml_threadpool_get_n_threads(byval threadpool as ggml_threadpool ptr) as long
declare sub ggml_threadpool_pause(byval threadpool as ggml_threadpool ptr)
declare sub ggml_threadpool_resume(byval threadpool as ggml_threadpool ptr)
' ggml_graph_plan() has to be called before ggml_graph_compute()
' when plan.work_size > 0, caller must allocate memory for plan.work_data
'' TODO: extern struct ggml_cplan ggml_graph_plan( const struct ggml_cgraph * cgraph, int n_threads, [TK_FBCODE /' = GGML_DEFAULT_N_THREADS '/] struct ggml_threadpool * threadpool [TK_FBCODE /' = NULL '/] );
declare function ggml_graph_compute(byval cgraph as ggml_cgraph ptr, byval cplan as ggml_cplan ptr) as ggml_status
' same as ggml_graph_compute() but the work data is allocated as a part of the context
' note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
declare function ggml_graph_compute_with_ctx(byval ctx as ggml_context ptr, byval cgraph as ggml_cgraph ptr, byval n_threads as long) as ggml_status
'
' system info
'
' x86
declare function ggml_cpu_has_sse3() as long
declare function ggml_cpu_has_ssse3() as long
declare function ggml_cpu_has_avx() as long
declare function ggml_cpu_has_avx_vnni() as long
declare function ggml_cpu_has_avx2() as long
declare function ggml_cpu_has_bmi2() as long
declare function ggml_cpu_has_f16c() as long
declare function ggml_cpu_has_fma() as long
declare function ggml_cpu_has_avx512() as long
declare function ggml_cpu_has_avx512_vbmi() as long
declare function ggml_cpu_has_avx512_vnni() as long
declare function ggml_cpu_has_avx512_bf16() as long
declare function ggml_cpu_has_amx_int8() as long
' ARM
declare function ggml_cpu_has_neon() as long
declare function ggml_cpu_has_arm_fma() as long
declare function ggml_cpu_has_fp16_va() as long
declare function ggml_cpu_has_dotprod() as long
declare function ggml_cpu_has_matmul_int8() as long
declare function ggml_cpu_has_sve() as long
declare function ggml_cpu_get_sve_cnt() as long ' sve vector length in bytes
declare function ggml_cpu_has_sme() as long
declare function ggml_cpu_has_sme2() as long
' other
declare function ggml_cpu_has_riscv_v() as long
declare function ggml_cpu_get_rvv_vlen() as long ' risc-v vector length in bytes
declare function ggml_cpu_has_vsx() as long
declare function ggml_cpu_has_vxe() as long
declare function ggml_cpu_has_wasm_simd() as long
declare function ggml_cpu_has_llamafile() as long
' Internal types and functions exposed for tests and benchmarks
type ggml_vec_dot_t as sub(byval n as long, byval s as single ptr, byval bs as uinteger, byval x as const any ptr, byval bx as uinteger, byval y as const any ptr, byval by as uinteger, byval nrc as long)

type ggml_type_traits_cpu
	from_float as ggml_from_float_t
	vec_dot as ggml_vec_dot_t
	vec_dot_type as ggml_type
	nrows as longint ' number of rows to process simultaneously
end type

declare function ggml_get_type_traits_cpu(byval type as ggml_type) as const ggml_type_traits_cpu ptr
declare sub ggml_cpu_init()
'
' CPU backend
'
declare function ggml_backend_cpu_init() as ggml_backend_t
declare function ggml_backend_is_cpu(byval backend as ggml_backend_t) as bool
declare sub ggml_backend_cpu_set_n_threads(byval backend_cpu as ggml_backend_t, byval n_threads as long)
declare sub ggml_backend_cpu_set_threadpool(byval backend_cpu as ggml_backend_t, byval threadpool as ggml_threadpool_t)
declare sub ggml_backend_cpu_set_abort_callback(byval backend_cpu as ggml_backend_t, byval abort_callback as ggml_abort_callback, byval abort_callback_data as any ptr)
declare sub ggml_backend_cpu_set_use_ref(byval backend_cpu as ggml_backend_t, byval use_ref as bool)
declare function ggml_backend_cpu_reg() as ggml_backend_reg_t
declare sub ggml_cpu_fp32_to_fp32(byval as const single ptr, byval as single ptr, byval as longint)
declare sub ggml_cpu_fp32_to_i32(byval as const single ptr, byval as long ptr, byval as longint)
declare sub ggml_cpu_fp32_to_fp16(byval as const single ptr, byval as ggml_fp16_t ptr, byval as longint)
declare sub ggml_cpu_fp16_to_fp32(byval as const ggml_fp16_t ptr, byval as single ptr, byval as longint)
declare sub ggml_cpu_fp32_to_bf16(byval as const single ptr, byval as ggml_bf16_t ptr, byval as longint)
declare sub ggml_cpu_bf16_to_fp32(byval as const ggml_bf16_t ptr, byval as single ptr, byval as longint)

end extern

#pragma once

#include once "ggml.bi"

extern "C"

type ggml_backend_buffer_type_t as ggml_backend_buffer_type ptr
type ggml_backend_buffer_t as ggml_backend_buffer ptr
type ggml_backend_t as ggml_backend ptr
' Tensor allocator

type ggml_tallocr
	buffer as ggml_backend_buffer_t
	base as any ptr
	alignment as uinteger
	offset as uinteger
end type

declare function ggml_tallocr_new(byval buffer as ggml_backend_buffer_t) as ggml_tallocr
declare function ggml_tallocr_alloc(byval talloc as ggml_tallocr ptr, byval tensor as ggml_tensor ptr) as ggml_status
' Graph allocator
/'
  Example usage:
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

    // optional: create a worst-case graph and reserve the buffers to avoid reallocations
    ggml_gallocr_reserve(galloc, build_graph(max_batch));

    // allocate the graph
    struct ggml_cgraph * graph = build_graph(batch);
    ggml_gallocr_alloc_graph(galloc, graph);

    printf("compute buffer size: %zu bytes\n", ggml_gallocr_get_buffer_size(galloc, 0));

    // evaluate the graph
    ggml_backend_graph_compute(backend, graph);
'/
' special tensor flags for use with the graph allocator:
'   ggml_set_input(): all input tensors are allocated at the beginning of the graph in non-overlapping addresses
'   ggml_set_output(): output tensors are never freed and never overwritten
type ggml_gallocr_t as ggml_gallocr ptr
declare function ggml_gallocr_new(byval buft as ggml_backend_buffer_type_t) as ggml_gallocr_t
declare function ggml_gallocr_new_n(byval bufts as ggml_backend_buffer_type_t ptr, byval n_bufs as long) as ggml_gallocr_t
declare sub ggml_gallocr_free(byval galloc as ggml_gallocr_t)
' pre-allocate buffers from a measure graph - does not allocate or modify the graph
' call with a worst-case graph to avoid buffer reallocations
' not strictly required for single buffer usage: ggml_gallocr_alloc_graph will reallocate the buffers automatically if needed
' returns false if the buffer allocation failed
' ggml_gallocr_resrve_n_size writes the buffer sizes per galloc buffer that would be allocated by ggml_gallocr_reserve_n to sizes
declare function ggml_gallocr_reserve(byval galloc as ggml_gallocr_t, byval graph as ggml_cgraph ptr) as bool
declare sub ggml_gallocr_reserve_n_size(byval galloc as ggml_gallocr_t, byval graph as ggml_cgraph ptr, byval node_buffer_ids as const long ptr, byval leaf_buffer_ids as const long ptr, byval sizes as uinteger ptr)
declare function ggml_gallocr_reserve_n(byval galloc as ggml_gallocr_t, byval graph as ggml_cgraph ptr, byval node_buffer_ids as const long ptr, byval leaf_buffer_ids as const long ptr) as bool
' automatic reallocation if the topology changes when using a single buffer
' returns false if using multiple buffers and a re-allocation is needed (call ggml_gallocr_reserve_n first to set the node buffers)
declare function ggml_gallocr_alloc_graph(byval galloc as ggml_gallocr_t, byval graph as ggml_cgraph ptr) as bool
declare function ggml_gallocr_get_buffer_size(byval galloc as ggml_gallocr_t, byval buffer_id as long) as uinteger
' Utils
' Create a buffer and allocate all the tensors in a ggml_context
' returns the size of the buffer that would be allocated by ggml_backend_alloc_ctx_tensors_from_buft. returns 0 on failure
declare function ggml_backend_alloc_ctx_tensors_from_buft_size(byval ctx as ggml_context ptr, byval buft as ggml_backend_buffer_type_t) as uinteger
' returns NULL on failure or if all tensors in ctx are already allocated or zero-sized
declare function ggml_backend_alloc_ctx_tensors_from_buft(byval ctx as ggml_context ptr, byval buft as ggml_backend_buffer_type_t) as ggml_backend_buffer ptr
declare function ggml_backend_alloc_ctx_tensors(byval ctx as ggml_context ptr, byval backend as ggml_backend_t) as ggml_backend_buffer ptr

end extern

#pragma once

#include once "ggml.bi"
#include once "ggml-alloc.bi"

extern "C"

'' TODO: # define GGML_BACKEND_API extern
type ggml_backend_buffer_type_t as ggml_backend_buffer_type ptr
type ggml_backend_buffer_t as ggml_backend_buffer ptr
type ggml_backend_event_t as ggml_backend_event ptr
type ggml_backend_t as ggml_backend ptr
type ggml_backend_graph_plan_t as any ptr
type ggml_backend_reg_t as ggml_backend_reg ptr
type ggml_backend_dev_t as ggml_backend_device ptr
'
' Backend buffer type
'

declare function ggml_backend_buft_name(byval buft as ggml_backend_buffer_type_t) as const zstring ptr
declare function ggml_backend_buft_alloc_buffer(byval buft as ggml_backend_buffer_type_t, byval size as uinteger) as ggml_backend_buffer_t
declare function ggml_backend_buft_alloc_buffer_n(byval buft as ggml_backend_buffer_type_t, byval tensors as ggml_tensor ptr ptr, byval n_tensors as long) as ggml_backend_buffer_t
declare function ggml_backend_buft_get_alignment(byval buft as ggml_backend_buffer_type_t) as uinteger
declare function ggml_backend_buft_get_max_size(byval buft as ggml_backend_buffer_type_t) as uinteger
declare function ggml_backend_buft_get_alloc_size(byval buft as ggml_backend_buffer_type_t, byval tensor as const ggml_tensor ptr) as uinteger
declare function ggml_backend_buft_get_alloc_size_n(byval buft as ggml_backend_buffer_type_t, byval tensors as ggml_tensor ptr ptr, byval n_tensors as long) as uinteger
declare function ggml_backend_buft_is_host(byval buft as ggml_backend_buffer_type_t) as bool
declare function ggml_backend_buft_get_device(byval buft as ggml_backend_buffer_type_t) as ggml_backend_dev_t
'
' Backend buffer
'

enum ggml_backend_buffer_usage
	GGML_BACKEND_BUFFER_USAGE_ANY = 0
	GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1
	GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2
end enum

declare function ggml_backend_buffer_name(byval buffer as ggml_backend_buffer_t) as const zstring ptr
declare sub ggml_backend_buffer_free(byval buffer as ggml_backend_buffer_t)
declare function ggml_backend_buffer_get_base(byval buffer as ggml_backend_buffer_t) as any ptr
declare function ggml_backend_buffer_get_size(byval buffer as ggml_backend_buffer_t) as uinteger
declare function ggml_backend_buffer_init_tensor(byval buffer as ggml_backend_buffer_t, byval tensor as ggml_tensor ptr) as ggml_status
declare function ggml_backend_buffer_get_alignment(byval buffer as ggml_backend_buffer_t) as uinteger
declare function ggml_backend_buffer_get_max_size(byval buffer as ggml_backend_buffer_t) as uinteger
declare function ggml_backend_buffer_get_alloc_size(byval buffer as ggml_backend_buffer_t, byval tensor as const ggml_tensor ptr) as uinteger
declare sub ggml_backend_buffer_clear(byval buffer as ggml_backend_buffer_t, byval value as ubyte)
declare function ggml_backend_buffer_is_host(byval buffer as ggml_backend_buffer_t) as bool
declare sub ggml_backend_buffer_set_usage(byval buffer as ggml_backend_buffer_t, byval usage as ggml_backend_buffer_usage)
declare function ggml_backend_buffer_get_usage(byval buffer as ggml_backend_buffer_t) as ggml_backend_buffer_usage
declare function ggml_backend_buffer_get_type(byval buffer as ggml_backend_buffer_t) as ggml_backend_buffer_type_t
declare sub ggml_backend_buffer_reset(byval buffer as ggml_backend_buffer_t)
' tensor copy between different backends
declare sub ggml_backend_tensor_copy(byval src as const ggml_tensor ptr, byval dst as ggml_tensor ptr)
'
' Backend (stream)
'
declare function ggml_backend_guid(byval backend as ggml_backend_t) as ggml_guid_t
declare function ggml_backend_name(byval backend as ggml_backend_t) as const zstring ptr
declare sub ggml_backend_free(byval backend as ggml_backend_t)
declare function ggml_backend_get_default_buffer_type(byval backend as ggml_backend_t) as ggml_backend_buffer_type_t
declare function ggml_backend_alloc_buffer(byval backend as ggml_backend_t, byval size as uinteger) as ggml_backend_buffer_t
declare function ggml_backend_get_alignment(byval backend as ggml_backend_t) as uinteger
declare function ggml_backend_get_max_size(byval backend as ggml_backend_t) as uinteger
declare sub ggml_backend_tensor_set_async(byval backend as ggml_backend_t, byval tensor as ggml_tensor ptr, byval data as const any ptr, byval offset as uinteger, byval size as uinteger)
declare sub ggml_backend_tensor_get_async(byval backend as ggml_backend_t, byval tensor as const ggml_tensor ptr, byval data as any ptr, byval offset as uinteger, byval size as uinteger)
declare sub ggml_backend_tensor_set_2d_async(byval backend as ggml_backend_t, byval tensor as ggml_tensor ptr, byval data as const any ptr, byval offset as uinteger, byval size as uinteger, byval n_copies as uinteger, byval stride_tensor as uinteger, byval stride_data as uinteger)
declare sub ggml_backend_tensor_get_2d_async(byval backend as ggml_backend_t, byval tensor as const ggml_tensor ptr, byval data as any ptr, byval offset as uinteger, byval size as uinteger, byval n_copies as uinteger, byval stride_tensor as uinteger, byval stride_data as uinteger)
' "offset" refers to the offset in tensor->data for setting/getting data
declare sub ggml_backend_tensor_set(byval tensor as ggml_tensor ptr, byval data as const any ptr, byval offset as uinteger, byval size as uinteger)
declare sub ggml_backend_tensor_get(byval tensor as const ggml_tensor ptr, byval data as any ptr, byval offset as uinteger, byval size as uinteger)
declare sub ggml_backend_tensor_set_2d(byval tensor as ggml_tensor ptr, byval data as const any ptr, byval offset as uinteger, byval size as uinteger, byval n_copies as uinteger, byval stride_tensor as uinteger, byval stride_data as uinteger)
declare sub ggml_backend_tensor_get_2d(byval tensor as const ggml_tensor ptr, byval data as any ptr, byval offset as uinteger, byval size as uinteger, byval n_copies as uinteger, byval stride_tensor as uinteger, byval stride_data as uinteger)
declare sub ggml_backend_tensor_memset(byval tensor as ggml_tensor ptr, byval value as ubyte, byval offset as uinteger, byval size as uinteger)
declare sub ggml_backend_synchronize(byval backend as ggml_backend_t)
declare function ggml_backend_graph_plan_create(byval backend as ggml_backend_t, byval cgraph as ggml_cgraph ptr) as ggml_backend_graph_plan_t
declare sub ggml_backend_graph_plan_free(byval backend as ggml_backend_t, byval plan as ggml_backend_graph_plan_t)
declare function ggml_backend_graph_plan_compute(byval backend as ggml_backend_t, byval plan as ggml_backend_graph_plan_t) as ggml_status
declare function ggml_backend_graph_compute(byval backend as ggml_backend_t, byval cgraph as ggml_cgraph ptr) as ggml_status
declare function ggml_backend_graph_compute_async(byval backend as ggml_backend_t, byval cgraph as ggml_cgraph ptr) as ggml_status
' NOTE: will be removed, use device version instead
declare function ggml_backend_supports_op(byval backend as ggml_backend_t, byval op as const ggml_tensor ptr) as bool
declare function ggml_backend_supports_buft(byval backend as ggml_backend_t, byval buft as ggml_backend_buffer_type_t) as bool
declare function ggml_backend_offload_op(byval backend as ggml_backend_t, byval op as const ggml_tensor ptr) as bool
' asynchronous copy
' the copy is performed after all the currently queued operations in backend_src
' backend_dst will wait for the copy to complete before performing other operations
' automatic fallback to sync copy if async is not supported
declare sub ggml_backend_tensor_copy_async(byval backend_src as ggml_backend_t, byval backend_dst as ggml_backend_t, byval src as const ggml_tensor ptr, byval dst as ggml_tensor ptr)
declare function ggml_backend_get_device(byval backend as ggml_backend_t) as ggml_backend_dev_t
'
' Events
'
declare function ggml_backend_event_new(byval device as ggml_backend_dev_t) as ggml_backend_event_t
declare sub ggml_backend_event_free(byval event as ggml_backend_event_t)
declare sub ggml_backend_event_record(byval event as ggml_backend_event_t, byval backend as ggml_backend_t)
declare sub ggml_backend_event_synchronize(byval event as ggml_backend_event_t)
declare sub ggml_backend_event_wait(byval backend as ggml_backend_t, byval event as ggml_backend_event_t)
'
' Backend device
'

enum ggml_backend_dev_type
	' CPU device using system memory
	GGML_BACKEND_DEVICE_TYPE_CPU
	' GPU device using dedicated memory
	GGML_BACKEND_DEVICE_TYPE_GPU
	' integrated GPU device using host memory
	GGML_BACKEND_DEVICE_TYPE_IGPU
	' accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
	GGML_BACKEND_DEVICE_TYPE_ACCEL
	' "meta" device wrapping multiple other devices for tensor parallelism
	GGML_BACKEND_DEVICE_TYPE_META
end enum

' functionality supported by the device

type ggml_backend_dev_caps
	' asynchronous operations
	async as bool
	' pinned host buffer
	host_buffer as bool
	' creating buffers from host ptr
	buffer_from_host_ptr as bool
	' event synchronization
	events as bool
	' mmap is supported for loading
	mmap_support as bool
end type

' all the device properties

type ggml_backend_dev_props
	' device name
	name as const zstring ptr
	' device description
	description as const zstring ptr
	' device free memory in bytes
	memory_free as uinteger
	' device total memory in bytes
	memory_total as uinteger
	' device type
	as ggml_backend_dev_type type
	' device id
	'   for PCI devices, this should be the lower-case PCI bus id formatted as "domain:bus:device.function" (e.g. "0000:c1:00.0")
	'   if the id is unknown, this should be NULL
	device_id as const zstring ptr
	' device capabilities
	caps as ggml_backend_dev_caps
end type

declare function ggml_backend_dev_name(byval device as ggml_backend_dev_t) as const zstring ptr
declare function ggml_backend_dev_description(byval device as ggml_backend_dev_t) as const zstring ptr
declare sub ggml_backend_dev_memory(byval device as ggml_backend_dev_t, byval free as uinteger ptr, byval total as uinteger ptr)
declare function ggml_backend_dev_type_ alias "ggml_backend_dev_type"(byval device as ggml_backend_dev_t) as ggml_backend_dev_type
declare sub ggml_backend_dev_get_props(byval device as ggml_backend_dev_t, byval props as ggml_backend_dev_props ptr)
declare function ggml_backend_dev_backend_reg(byval device as ggml_backend_dev_t) as ggml_backend_reg_t
declare function ggml_backend_dev_init(byval device as ggml_backend_dev_t, byval params as const zstring ptr) as ggml_backend_t
declare function ggml_backend_dev_buffer_type(byval device as ggml_backend_dev_t) as ggml_backend_buffer_type_t
declare function ggml_backend_dev_host_buffer_type(byval device as ggml_backend_dev_t) as ggml_backend_buffer_type_t
declare function ggml_backend_dev_buffer_from_host_ptr(byval device as ggml_backend_dev_t, byval ptr as any ptr, byval size as uinteger, byval max_tensor_size as uinteger) as ggml_backend_buffer_t
declare function ggml_backend_dev_supports_op(byval device as ggml_backend_dev_t, byval op as const ggml_tensor ptr) as bool
declare function ggml_backend_dev_supports_buft(byval device as ggml_backend_dev_t, byval buft as ggml_backend_buffer_type_t) as bool
declare function ggml_backend_dev_offload_op(byval device as ggml_backend_dev_t, byval op as const ggml_tensor ptr) as bool
'
' Backend (reg)
'
declare function ggml_backend_reg_name(byval reg as ggml_backend_reg_t) as const zstring ptr
declare function ggml_backend_reg_dev_count(byval reg as ggml_backend_reg_t) as uinteger
declare function ggml_backend_reg_dev_get(byval reg as ggml_backend_reg_t, byval index as uinteger) as ggml_backend_dev_t
declare function ggml_backend_reg_get_proc_address(byval reg as ggml_backend_reg_t, byval name as const zstring ptr) as any ptr
' Common functions that may be obtained using ggml_backend_reg_get_proc_address
' Context management and operations for faster communication between backends, used for tensor parallelism (meta backend)

type ggml_backend_comm_init_t as function(byval backends as ggml_backend_t ptr, byval n_backends as uinteger) as any ptr
type ggml_backend_comm_free_t as sub(byval comm_ctx as any ptr)
type ggml_backend_comm_allreduce_tensor_t as function(byval comm_ctx as any ptr, byval tensors as ggml_tensor ptr ptr) as bool
' Split buffer type for tensor parallelism (old)
type ggml_backend_split_buffer_type_t as function(byval main_device as long, byval tensor_split as const single ptr) as ggml_backend_buffer_type_t
' Set the number of threads for the backend
type ggml_backend_set_n_threads_t as sub(byval backend as ggml_backend_t, byval n_threads as long)
' Get additional buffer types provided by the device (returns a NULL-terminated array)
type ggml_backend_dev_get_extra_bufts_t as function(byval device as ggml_backend_dev_t) as ggml_backend_buffer_type_t ptr
' Set the abort callback for the backend
type ggml_backend_set_abort_callback_t as sub(byval backend as ggml_backend_t, byval abort_callback as ggml_abort_callback, byval abort_callback_data as any ptr)
' Get a list of feature flags supported by the backend (returns a NULL-terminated array)

type ggml_backend_feature
	name as const zstring ptr
	value as const zstring ptr
end type

type ggml_backend_get_features_t as function(byval reg as ggml_backend_reg_t) as ggml_backend_feature ptr
'
' Backend registry
'

declare sub ggml_backend_register(byval reg as ggml_backend_reg_t)
declare sub ggml_backend_device_register(byval device as ggml_backend_dev_t)
' Backend (reg) enumeration
declare function ggml_backend_reg_count() as uinteger
declare function ggml_backend_reg_get(byval index as uinteger) as ggml_backend_reg_t
declare function ggml_backend_reg_by_name(byval name as const zstring ptr) as ggml_backend_reg_t
' Device enumeration
declare function ggml_backend_dev_count() as uinteger
declare function ggml_backend_dev_get(byval index as uinteger) as ggml_backend_dev_t
declare function ggml_backend_dev_by_name(byval name as const zstring ptr) as ggml_backend_dev_t
declare function ggml_backend_dev_by_type(byval type as ggml_backend_dev_type) as ggml_backend_dev_t
' Direct backend (stream) initialization
' = ggml_backend_dev_init(ggml_backend_dev_by_name(name), params)
declare function ggml_backend_init_by_name(byval name as const zstring ptr, byval params as const zstring ptr) as ggml_backend_t
' = ggml_backend_dev_init(ggml_backend_dev_by_type(type), params)
declare function ggml_backend_init_by_type(byval type as ggml_backend_dev_type, byval params as const zstring ptr) as ggml_backend_t
' = ggml_backend_dev_init(ggml_backend_dev_by_type(GPU) OR ggml_backend_dev_by_type(CPU), NULL)
declare function ggml_backend_init_best() as ggml_backend_t
' Load a backend from a dynamic library and register it
declare function ggml_backend_load(byval path as const zstring ptr) as ggml_backend_reg_t
' Unload a backend if loaded dynamically and unregister it
declare sub ggml_backend_unload(byval reg as ggml_backend_reg_t)
' Load all known backends from dynamic libraries
declare sub ggml_backend_load_all()
declare sub ggml_backend_load_all_from_path(byval dir_path as const zstring ptr)
'
' Backend scheduler
'
' The backend scheduler allows for multiple backend devices to be used together
' Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
' The backends are selected based on:
' - the backend that supports the operation
' - the location of the pre-allocated tensors (e.g. the weights)
/'
      Example usage:

        // operations that use tensors allocated in a buffer with USAGE_WEIGHTS will be assigned
        // preferably to run on the same backend as the buffer
        ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        sched = ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends, GGML_DEFAULT_GRAPH_SIZE, false, true);

        // initialize buffers from a max size graph (optional)
        reserve_graph = build_graph(sched, max_batch_size);

        // manually assign nodes to a backend (optional, should not be needed in most cases)
        struct ggml_tensor * node = ggml_mul_mat(ctx, ...);
        ggml_backend_sched_set_tensor_backend(sched, node, backend_gpu);

        ggml_backend_sched_reserve(sched, reserve_graph);

        // compute
        graph = build_graph(sched); // the graph and its tensors are single-use in terms of allocation, multi-use in terms of computation
        for (int i = 0; i < 10; ++i) {
            ggml_backend_sched_graph_compute(sched, graph); // on the first iteration the graph is allocated automatically
        }

        // if there are graph inputs:
        graph = build_graph(sched); // get a new graph that is not allocated (the metadata for the old graph is freed once ggml_free is called)
        ggml_backend_sched_reset(sched); // clear the allocation of the previous graph
        ggml_backend_sched_alloc_graph(sched, graph); // explicitly allocate the new graph but do not execute it
        ggml_backend_tensor_set(input_tensor, ...); // copy data to the newly allocated graph tensors
        ggml_backend_sched_graph_compute(sched, graph); // execute the graph

        // as an alternative to the above it is also possible to assign the inputs to a dedicated context and
        // allocate them statically via ggml_backend_alloc_ctx_tensors
    }
    '/

type ggml_backend_sched_t as ggml_backend_sched ptr
' Evaluation callback for each node in the graph (set with ggml_backend_sched_set_eval_callback)
' when ask == true, the scheduler wants to know if the user wants to observe this node
' this allows the scheduler to batch nodes together in order to evaluate them in a single call
'
' when ask == false, the scheduler is passing the node tensor to the user for observation
' if the user returns false, the scheduler will cancel the graph compute
'
type ggml_backend_sched_eval_callback as function(byval t as ggml_tensor ptr, byval ask as bool, byval user_data as any ptr) as bool
' Callback while copying input weights of a split
' if the user returns false the scheduler simply copies the entire weight
' the callback is called only for input weights in host buffers
' the callback is called after all non-weight inputs of the split have been copied
' `src` is the tensor in the previous split
' `dst` is the copy of `src` in the split
' `graph` is the compute graph of the split
type ggml_backend_sched_copy_callback as function(byval backend as ggml_backend_t, byval src as const ggml_tensor ptr, byval dst as ggml_tensor ptr, byval graph as ggml_cgraph ptr, byval user_data as any ptr) as bool
' Callback when the buffers reserved for a graph are not sufficient and have to be reserved again
' it happens on the graph allocation path and costs a synchronization of the backends
' `sched` is the scheduler
' `graph` is the compute graph being allocated
' `backend_ids_changed` is true if the backend assignment of the graph nodes changed since the last allocation
' `graph_size_changed` is true if the graph size changed since the last allocation
type ggml_backend_sched_realloc_callback as sub(byval sched as ggml_backend_sched_t, byval graph as ggml_cgraph ptr, byval backend_ids_changed as bool, byval graph_size_changed as bool, byval user_data as any ptr)
' Initialize a backend scheduler, backends with low index are given priority over backends with high index

declare function ggml_backend_sched_new(byval backends as ggml_backend_t ptr, byval bufts as ggml_backend_buffer_type_t ptr, byval n_backends as long, byval graph_size as uinteger, byval parallel as bool, byval op_offload as bool) as ggml_backend_sched_t
declare sub ggml_backend_sched_free(byval sched as ggml_backend_sched_t)
' Initialize backend buffers from a measure graph
declare sub ggml_backend_sched_reserve_size(byval sched as ggml_backend_sched_t, byval measure_graph as ggml_cgraph ptr, byval sizes as uinteger ptr)
declare function ggml_backend_sched_reserve(byval sched as ggml_backend_sched_t, byval measure_graph as ggml_cgraph ptr) as bool ' returns success
declare function ggml_backend_sched_get_n_backends(byval sched as ggml_backend_sched_t) as long
declare function ggml_backend_sched_get_backend(byval sched as ggml_backend_sched_t, byval i as long) as ggml_backend_t
' Get the number of splits of the last graph
declare function ggml_backend_sched_get_n_splits(byval sched as ggml_backend_sched_t) as long
declare function ggml_backend_sched_get_n_copies(byval sched as ggml_backend_sched_t) as long
declare function ggml_backend_sched_get_buffer_type(byval sched as ggml_backend_sched_t, byval backend as ggml_backend_t) as ggml_backend_buffer_type_t
declare function ggml_backend_sched_get_buffer_size(byval sched as ggml_backend_sched_t, byval backend as ggml_backend_t) as uinteger
declare sub ggml_backend_sched_set_tensor_backend(byval sched as ggml_backend_sched_t, byval node as ggml_tensor ptr, byval backend as ggml_backend_t)
declare function ggml_backend_sched_get_tensor_backend(byval sched as ggml_backend_sched_t, byval node as ggml_tensor ptr) as ggml_backend_t
' Split graph without allocating it
declare sub ggml_backend_sched_split_graph(byval sched as ggml_backend_sched_t, byval graph as ggml_cgraph ptr)
' Allocate and compute graph on the backend scheduler
declare function ggml_backend_sched_alloc_graph(byval sched as ggml_backend_sched_t, byval graph as ggml_cgraph ptr) as bool ' returns success
declare function ggml_backend_sched_graph_compute(byval sched as ggml_backend_sched_t, byval graph as ggml_cgraph ptr) as ggml_status
declare function ggml_backend_sched_graph_compute_async(byval sched as ggml_backend_sched_t, byval graph as ggml_cgraph ptr) as ggml_status
declare sub ggml_backend_sched_synchronize(byval sched as ggml_backend_sched_t)
' Reset all assignments and allocators - must be called before changing the node backends or allocating a new graph.
' This in effect deallocates all tensors that were previously allocated and leaves them with dangling pointers.
' The correct way to use this API is to discard the deallocated tensors and create new ones.
declare sub ggml_backend_sched_reset(byval sched as ggml_backend_sched_t)
' Set a callback to be called for each resulting node during graph compute
declare sub ggml_backend_sched_set_eval_callback(byval sched as ggml_backend_sched_t, byval callback as ggml_backend_sched_eval_callback, byval user_data as any ptr)
' Set a callback to be called when the inputs weights of a split are being copied
declare sub ggml_backend_sched_set_copy_callback(byval sched as ggml_backend_sched_t, byval callback as ggml_backend_sched_copy_callback, byval user_data as any ptr)
' Set a callback to be called when the compute buffers of a graph are being reallocated
declare sub ggml_backend_sched_set_realloc_callback(byval sched as ggml_backend_sched_t, byval callback as ggml_backend_sched_realloc_callback, byval user_data as any ptr)
'
' Meta backend
'
const GGML_BACKEND_META_MAX_DEVICES = 16

enum ggml_backend_meta_split_axis
	' tensor split by tensor dimensions:
	GGML_BACKEND_SPLIT_AXIS_0 = 0
	GGML_BACKEND_SPLIT_AXIS_1 = 1
	GGML_BACKEND_SPLIT_AXIS_2 = 2
	GGML_BACKEND_SPLIT_AXIS_3 = 3
	GGML_BACKEND_SPLIT_AXIS_MIRRORED = 10 ' all values on all backends
	GGML_BACKEND_SPLIT_AXIS_PARTIAL = 11 ' each backend has a partial sum
	' for internal bookkeeping only:
	GGML_BACKEND_SPLIT_AXIS_NONE = 98
	GGML_BACKEND_SPLIT_AXIS_UNKNOWN = 99
end enum

declare function ggml_backend_meta_split_axis_name(byval split_axis as ggml_backend_meta_split_axis) as const zstring ptr

type ggml_backend_meta_split_state
	axis as ggml_backend_meta_split_axis
	' for tensors with axis >= 0 && axis < GGML_MAX_DIMS:
	'   - each device has a slice of the tensor along the split axis
	'   - most tensors have n_segments == 1 and a contiguous slice of the tensor data
	'   - some tensors have an inhomogenenous data layout along the split axis,
	'     those tensors are divided into segments which are each individually split across devices
	'     (this usually happens when multiple tensors are fused into a single one)
	'   - ne has one entry per segment and device and that segment repeats nr times,
	'     in total when accounting for repetitions the segments add up to ggml_tensor::ne for that axis,
	'     the outer/inner loops are over segments/devices like [seg0_dev0_r0, seg0_dev1_r0, seg0_dev0_r1, seg0_dev1_r1, seg1_dev0_r0, seg1_dev1_r0],
	'   - for example, a transformer may have a fused QKV matrix rather than 3 matrices, those would be 3 separate segments
	'     that each need to be split individually across devices so that each device gets a slice of Q, K, and V,
	'     the Q matrix can be larger than the K and V matrices so this can either be expressed as 3 segments or as 2 segments
	'     where the segment for K/V repeats twice
	ne(0 to (16 * 16) - 1) as longint
	nr(0 to 15) as ulong
	n_segments as ulong
end type

' function to assign split states for statically allocated tensors, compute tensor split states will be assigned to be compatible:
type ggml_backend_meta_get_split_state_t as function(byval tensor as const ggml_tensor ptr, byval userdata as any ptr) as ggml_backend_meta_split_state
' create a new meta device from "simple" devices, meta buffer type/buffer/backend is then derived from this:
' TODO: this looks a bit strange - a backend API creates a device. I think we should try
'       express this as a backend registry functionality instead
declare function ggml_backend_meta_device(byval devs as ggml_backend_dev_t ptr, byval n_devs as uinteger, byval get_split_state as ggml_backend_meta_get_split_state_t, byval get_split_state_ud as any ptr) as ggml_backend_dev_t
'
' Utils
'

type ggml_backend_graph_copy
	buffer as ggml_backend_buffer_t
	ctx_allocated as ggml_context ptr
	ctx_unallocated as ggml_context ptr
	graph as ggml_cgraph ptr
end type

' Copy a graph to a different backend
declare function ggml_backend_graph_copy(byval backend as ggml_backend_t, byval graph as ggml_cgraph ptr) as ggml_backend_graph_copy
declare sub ggml_backend_graph_copy_free(byval copy as ggml_backend_graph_copy)
type ggml_backend_eval_callback as function(byval node_index as long, byval t1 as ggml_tensor ptr, byval t2 as ggml_tensor ptr, byval user_data as any ptr) as bool
' Compare the output of two backends
declare function ggml_backend_compare_graph_backend(byval backend1 as ggml_backend_t, byval backend2 as ggml_backend_t, byval graph as ggml_cgraph ptr, byval callback as ggml_backend_eval_callback, byval user_data as any ptr, byval test_nodes as const ggml_tensor const ptr ptr, byval num_test_nodes as uinteger) as bool
' Tensor initialization
declare function ggml_backend_tensor_alloc(byval buffer as ggml_backend_buffer_t, byval tensor as ggml_tensor ptr, byval addr as any ptr) as ggml_status
declare function ggml_backend_view_init(byval tensor as ggml_tensor ptr) as ggml_status
' CPU buffer types are always available
declare function ggml_backend_cpu_buffer_from_ptr(byval ptr as any ptr, byval size as uinteger) as ggml_backend_buffer_t
declare function ggml_backend_cpu_buffer_type() as ggml_backend_buffer_type_t

end extern

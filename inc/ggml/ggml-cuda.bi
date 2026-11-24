#pragma once

#include once "ggml.bi"
#include once "ggml-backend.bi"

extern "C"

#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
const GGML_CUDA_MAX_DEVICES = 16
' backend API

declare function ggml_backend_cuda_init(byval device as long) as ggml_backend_t
declare function ggml_backend_is_cuda(byval backend as ggml_backend_t) as bool
' device buffer
declare function ggml_backend_cuda_buffer_type(byval device as long) as ggml_backend_buffer_type_t
' conduct allreduce operation between devices
declare function ggml_backend_cuda_allreduce_tensor(byval backends as ggml_backend_t ptr, byval tensors as ggml_tensor ptr ptr, byval n_backends as uinteger) as bool
' pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
declare function ggml_backend_cuda_host_buffer_type() as ggml_backend_buffer_type_t
declare function ggml_backend_cuda_get_device_count() as long
declare sub ggml_backend_cuda_get_device_description(byval device as long, byval description as zstring ptr, byval description_size as uinteger)
declare sub ggml_backend_cuda_get_device_memory(byval device as long, byval free as uinteger ptr, byval total as uinteger ptr)
declare function ggml_backend_cuda_register_host_buffer(byval buffer as any ptr, byval size as uinteger) as bool
declare sub ggml_backend_cuda_unregister_host_buffer(byval buffer as any ptr)
declare function ggml_backend_cuda_reg() as ggml_backend_reg_t

end extern

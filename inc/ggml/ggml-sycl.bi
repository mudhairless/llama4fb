#pragma once

#include once "ggml.bi"
#include once "ggml-backend.bi"

extern "C"

'
'  MIT license
'  Copyright (C) 2024 Intel Corporation
'  SPDX-License-Identifier: MIT
'
#define GGML_SYCL_NAME "SYCL"
const GGML_SYCL_MAX_DEVICES = 48
' backend API
declare function ggml_backend_sycl_init(byval device as long) as ggml_backend_t
declare function ggml_backend_is_sycl(byval backend as ggml_backend_t) as bool
' devide buffer
declare function ggml_backend_sycl_buffer_type(byval device as long) as ggml_backend_buffer_type_t
' split tensor buffer that splits matrices by rows across multiple devices
declare function ggml_backend_sycl_split_buffer_type(byval main_device as long, byval tensor_split as const single ptr) as ggml_backend_buffer_type_t
' Tensor parallelism (--split-mode tensor): comm_init/free/allreduce_tensor
' trio queried by the meta-backend via ggml_backend_reg_get_proc_address.
' See typedefs in ggml/include/ggml-backend.h. Mirrors the CUDA backend's
' pattern (ggml_backend_cuda_comm_*).
declare function ggml_backend_sycl_comm_init(byval backends as ggml_backend_t ptr, byval n_backends as uinteger) as any ptr
declare sub ggml_backend_sycl_comm_free(byval comm_ctx as any ptr)
declare function ggml_backend_sycl_comm_allreduce_tensor(byval comm_ctx as any ptr, byval tensors as ggml_tensor ptr ptr) as bool
' pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
' pins on device 0 - a copy between another device and this memory can fail,
' use ggml_backend_dev_host_buffer_type to pin on the device that does the copy
declare function ggml_backend_sycl_host_buffer_type() as ggml_backend_buffer_type_t
declare sub ggml_backend_sycl_print_sycl_devices()
declare sub ggml_backend_sycl_get_gpu_list(byval id_list as long ptr, byval max_len as long)
declare sub ggml_backend_sycl_get_device_description(byval device as long, byval description as zstring ptr, byval description_size as uinteger)
declare function ggml_backend_sycl_get_device_count() as long
declare sub ggml_backend_sycl_get_device_memory(byval device as long, byval free as uinteger ptr, byval total as uinteger ptr)
' SYCL doesn't support registering host memory, keep here for reference
' GGML_BACKEND_API bool ggml_backend_sycl_register_host_buffer(void * buffer, size_t size);
' GGML_BACKEND_API void ggml_backend_sycl_unregister_host_buffer(void * buffer);
declare function ggml_backend_sycl_reg() as ggml_backend_reg_t

end extern

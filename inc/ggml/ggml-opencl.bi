#pragma once

#include once "ggml.bi"
#include once "ggml-backend.bi"

extern "C"

#define GGML_OPENCL_H
'
' backend API
'

declare function ggml_backend_opencl_init() as ggml_backend_t
declare function ggml_backend_is_opencl(byval backend as ggml_backend_t) as bool
declare function ggml_backend_opencl_buffer_type() as ggml_backend_buffer_type_t
declare function ggml_backend_opencl_host_buffer_type() as ggml_backend_buffer_type_t
declare function ggml_backend_opencl_reg() as ggml_backend_reg_t

end extern

#pragma once

#include once "ggml-backend.bi"
#include once "cstring.bi"

extern "C"

#define GGML_OPENVINO_NAME "OPENVINO"
' backend API
declare function ggml_backend_openvino_init(byval device as long) as ggml_backend_t
declare function ggml_backend_is_openvino(byval backend as ggml_backend_t) as bool
declare function ggml_backend_buffer_is_openvino(byval buffer as ggml_backend_buffer_t) as bool
declare function ggml_backend_buft_is_openvino(byval buft as ggml_backend_buffer_type_t) as bool
declare function ggml_backend_buft_is_openvino_host(byval buft as ggml_backend_buffer_type_t) as bool
declare function ggml_backend_openvino_buffer_get_ctx_id(byval buffer as ggml_backend_buffer_t) as uinteger
' device buffer
declare function ggml_backend_openvino_buffer_type(byval device as long) as ggml_backend_buffer_type_t
declare function ggml_backend_openvino_host_buffer_type(byval device as long) as ggml_backend_buffer_type_t
declare function ggml_backend_openvino_get_device_count() as long
declare function ggml_backend_openvino_reg() as ggml_backend_reg_t

end extern

#pragma once

#include once "ggml.bi"
#include once "ggml-backend.bi"

extern "C"

' device buffer
declare function ggml_backend_zdnn_buffer_type() as ggml_backend_buffer_type_t
declare function ggml_backend_is_zdnn(byval backend as ggml_backend_t) as bool
declare function ggml_backend_zdnn_reg() as ggml_backend_reg_t

end extern

#pragma once

#include once "ggml-backend.bi"
#include once "ggml.bi"

extern "C"

' backend API
declare function ggml_backend_zendnn_init() as ggml_backend_t
declare function ggml_backend_is_zendnn(byval backend as ggml_backend_t) as bool
' number of threads used for zendnn operations
declare sub ggml_backend_zendnn_set_n_threads(byval backend_zendnn as ggml_backend_t, byval n_threads as long)
declare function ggml_backend_zendnn_reg() as ggml_backend_reg_t

end extern

#pragma once

#include once "ggml.bi"
#include once "ggml-backend.bi"

extern "C"

' backend API
declare function ggml_backend_hexagon_init() as ggml_backend_t
declare function ggml_backend_is_hexagon(byval backend as ggml_backend_t) as bool
declare function ggml_backend_hexagon_reg() as ggml_backend_reg_t

end extern

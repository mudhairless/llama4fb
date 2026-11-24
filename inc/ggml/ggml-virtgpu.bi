#pragma once

#include once "ggml.bi"
#include once "ggml-backend.bi"

extern "C"

declare function ggml_backend_virtgpu_reg() as ggml_backend_reg_t

end extern

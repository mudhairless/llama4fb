#pragma once

#include once "ggml.bi"
#include once "ggml-backend.bi"

extern "C"

#define GGML_WEBGPU_NAME "WebGPU"
' Needed for examples in ggml
declare function ggml_backend_webgpu_init() as ggml_backend_t
declare function ggml_backend_webgpu_reg() as ggml_backend_reg_t

end extern

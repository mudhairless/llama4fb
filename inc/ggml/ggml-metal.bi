#pragma once

#include once "ggml.bi"
#include once "ggml-backend.bi"
#include once "crt/stddef.bi"
#include once "stdbool.bi"

extern "C"

' Note: this description is outdated
'
' An interface allowing to compute ggml_cgraph with Metal
'
' This is a fully functional interface that extends ggml with GPU support for Apple devices.
' A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, etc.)
'
' How it works?
'
' As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
' interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
' use ggml_metal_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
'
' You only need to make sure that all memory buffers that you used during the graph creation
' are mapped to the device memory with the ggml_metal_add_buffer() function. This mapping is
' used during the graph evaluation to determine the arguments of the compute kernels.
'
' Synchronization between device and host memory (for example for input and output tensors)
' is done with the ggml_metal_set_tensor() and ggml_metal_get_tensor() functions.
'
'
' backend API
' user-code should use only these functions
'
' TODO: remove in the future
declare function ggml_backend_metal_init() as ggml_backend_t
declare function ggml_backend_is_metal(byval backend as ggml_backend_t) as bool
declare sub ggml_backend_metal_set_abort_callback(byval backend as ggml_backend_t, byval abort_callback as ggml_abort_callback, byval user_data as any ptr)
' helper to check if the device supports a specific family
' ideally, the user code should be doing these checks
' ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
declare function ggml_backend_metal_supports_family(byval backend as ggml_backend_t, byval family as long) as bool
' capture all command buffers committed the next time `ggml_backend_graph_compute` is called
declare sub ggml_backend_metal_capture_next_compute(byval backend as ggml_backend_t)
declare function ggml_backend_metal_reg() as ggml_backend_reg_t

end extern

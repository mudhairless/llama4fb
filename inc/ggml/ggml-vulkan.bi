#pragma once

#include once "ggml.bi"
#include once "ggml-backend.bi"

extern "C"

#define GGML_VK_NAME "Vulkan"
const GGML_VK_MAX_DEVICES = 16
' backend API

declare function ggml_backend_vk_init(byval dev_num as uinteger) as ggml_backend_t
declare function ggml_backend_is_vk(byval backend as ggml_backend_t) as bool
declare function ggml_backend_vk_get_device_count() as long
declare sub ggml_backend_vk_get_device_description(byval device as long, byval description as zstring ptr, byval description_size as uinteger)
declare sub ggml_backend_vk_get_device_memory(byval device as long, byval free as uinteger ptr, byval total as uinteger ptr)
declare function ggml_backend_vk_buffer_type(byval dev_num as uinteger) as ggml_backend_buffer_type_t
' pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
declare function ggml_backend_vk_host_buffer_type() as ggml_backend_buffer_type_t
declare function ggml_backend_vk_reg() as ggml_backend_reg_t

end extern

#pragma once

#include once "ggml.bi"
#include once "ggml-backend.bi"

extern "C"

#define GGML_ET_NAME "ET"
' backend API
declare function ggml_backend_et_guid() as ggml_guid_t
declare function ggml_backend_et_init(byval devidx as uinteger) as ggml_backend_t
declare function ggml_backend_is_et(byval backend as ggml_backend_t) as bool
declare function ggml_backend_et_get_device_count() as long
declare sub ggml_backend_et_get_device_description(byval devidx as long, byval description as zstring ptr, byval description_size as uinteger)
declare sub ggml_backend_et_get_device_memory(byval devidx as long, byval free as uinteger ptr, byval total as uinteger ptr)
declare function ggml_backend_et_buffer_type(byval dev_num as uinteger) as ggml_backend_buffer_type_t
declare function ggml_backend_et_host_buffer_type() as ggml_backend_buffer_type_t
declare function ggml_backend_et_reg() as ggml_backend_reg_t

end extern

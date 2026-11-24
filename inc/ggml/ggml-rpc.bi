#pragma once

#include once "ggml-backend.bi"

extern "C"

const RPC_PROTO_MAJOR_VERSION = 8
const RPC_PROTO_MINOR_VERSION = 0
const RPC_PROTO_PATCH_VERSION = 0
const GGML_RPC_MAX_SERVERS = 16
' backend API

declare function ggml_backend_rpc_init(byval endpoint as const zstring ptr, byval device as ulong) as ggml_backend_t
declare function ggml_backend_is_rpc(byval backend as ggml_backend_t) as bool
declare function ggml_backend_rpc_buffer_type(byval endpoint as const zstring ptr, byval device as ulong) as ggml_backend_buffer_type_t
declare sub ggml_backend_rpc_get_device_memory(byval endpoint as const zstring ptr, byval device as ulong, byval free as uinteger ptr, byval total as uinteger ptr)
declare sub ggml_backend_rpc_start_server(byval endpoint as const zstring ptr, byval cache_dir as const zstring ptr, byval n_threads as uinteger, byval n_devices as uinteger, byval devices as ggml_backend_dev_t ptr)
declare function ggml_backend_rpc_reg() as ggml_backend_reg_t
declare function ggml_backend_rpc_add_server(byval endpoint as const zstring ptr) as ggml_backend_reg_t

end extern

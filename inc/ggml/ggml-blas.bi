#pragma once

#include once "ggml.bi"
#include once "ggml-backend.bi"

extern "C"

' backend API
declare function ggml_backend_blas_init() as ggml_backend_t
declare function ggml_backend_is_blas(byval backend as ggml_backend_t) as bool
' number of threads used for conversion to float
' for openblas and blis, this will also set the number of threads used for blas operations
declare sub ggml_backend_blas_set_n_threads(byval backend_blas as ggml_backend_t, byval n_threads as long)
declare function ggml_backend_blas_reg() as ggml_backend_reg_t

end extern

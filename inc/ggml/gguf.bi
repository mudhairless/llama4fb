#pragma once

#include once "ggml.bi"
#include once "stdbool.bi"
#include once "crt/stdint.bi"

type gguf_context as gguf_context_

extern "C"

' This file contains functionality related to "GGUF" files, the binary file format used by ggml.
' GGUF files have the following structure:
'
' 1. File magic "GGUF" (4 bytes).
' 2. File version (uint32_t).
' 3. Number of ggml tensors in file (int64_t).
' 4. Number of key-value-pairs in file (int64_t).
' 5. For each KV pair:
'   1. The key (string).
'   2. The value type (gguf_type).
'   3a. If the value type is GGUF_TYPE_ARRAY:
'     1. The type of the array (gguf_type).
'     2. The number of elements in the array (uint64_t).
'     3. The binary representation of each element in the array.
'   3b. Otherwise:
'     1. The binary representation of the value.
' 6. For each ggml tensor:
'   1. The tensor name (string).
'   2. The number of dimensions of the tensor (uint32_t).
'   3. For each dimension:
'     1. The size of the tensor in the dimension (int64_t).
'   4. The tensor data type (ggml_type).
'   5. The tensor data offset in the tensor data binary blob (uint64_t).
' 7. The tensor data binary blob (optional, aligned).
'
' Strings are serialized as the string length (uint64_t) followed by the C string without the null terminator.
' All enums are stored as int32_t.
' All bool values are stored as int8_t.
' If the special key "general.alignment" (uint32_t) is defined it is used for alignment,
'   otherwise GGUF_DEFAULT_ALIGNMENT is used.
'
' Module maintainer: Johannes Gäßler (@JohannesGaessler, johannesg@5d6.de)
#define GGUF_MAGIC "GGUF"
const GGUF_VERSION = 3
#define GGUF_KEY_GENERAL_ALIGNMENT "general.alignment"
const GGUF_DEFAULT_ALIGNMENT = 32
' types that can be stored as GGUF KV data

enum gguf_type
	GGUF_TYPE_UINT8 = 0
	GGUF_TYPE_INT8 = 1
	GGUF_TYPE_UINT16 = 2
	GGUF_TYPE_INT16 = 3
	GGUF_TYPE_UINT32 = 4
	GGUF_TYPE_INT32 = 5
	GGUF_TYPE_FLOAT32 = 6
	GGUF_TYPE_BOOL = 7
	GGUF_TYPE_STRING = 8
	GGUF_TYPE_ARRAY = 9
	GGUF_TYPE_UINT64 = 10
	GGUF_TYPE_INT64 = 11
	GGUF_TYPE_FLOAT64 = 12
	GGUF_TYPE_COUNT ' marks the end of the enum
end enum

type gguf_init_params
	no_alloc as bool
	' if not NULL, create a ggml_context and allocate the tensor data in it
	ctx as ggml_context ptr ptr
end type

' callback to simulate or wrap a FILE pointer - read up to `len` bytes at `offset` into `output` and return the number of bytes read
type gguf_reader_callback_t as function(byval userdata as any ptr, byval output as any ptr, byval offset as ulongint, byval len as uinteger) as uinteger
declare function gguf_init_empty() as gguf_context ptr
declare function gguf_init_from_file_ptr(byval file as FILE ptr, byval params as gguf_init_params) as gguf_context ptr
declare function gguf_init_from_file(byval fname as const zstring ptr, byval params as gguf_init_params) as gguf_context ptr
declare function gguf_init_from_buffer(byval data as const any ptr, byval size as uinteger, byval params as gguf_init_params) as gguf_context ptr
' max_chunk_read is the maximum number of bytes that the GGUF code will read at once from the callback, a value of 0 means no limit
declare function gguf_init_from_callback(byval callback as gguf_reader_callback_t, byval userdata as any ptr, byval max_chunk_read as uinteger, byval max_expected_size as ulongint, byval params as gguf_init_params) as gguf_context ptr
declare sub gguf_free(byval ctx as gguf_context ptr)
declare function gguf_type_name(byval type as gguf_type) as const zstring ptr
declare function gguf_get_version(byval ctx as const gguf_context ptr) as ulong
declare function gguf_get_alignment(byval ctx as const gguf_context ptr) as uinteger
declare function gguf_get_data_offset(byval ctx as const gguf_context ptr) as uinteger ' padded to gguf_get_alignment if and only if the gguf_context contains at least one tensor
declare function gguf_get_n_kv(byval ctx as const gguf_context ptr) as longint
declare function gguf_find_key(byval ctx as const gguf_context ptr, byval key as const zstring ptr) as longint ' returns -1 if key is not found
declare function gguf_get_key(byval ctx as const gguf_context ptr, byval key_id as longint) as const zstring ptr
declare function gguf_get_kv_type(byval ctx as const gguf_context ptr, byval key_id as longint) as gguf_type
declare function gguf_get_arr_type(byval ctx as const gguf_context ptr, byval key_id as longint) as gguf_type
' will abort if the wrong type is used for the key
declare function gguf_get_val_u8(byval ctx as const gguf_context ptr, byval key_id as longint) as ubyte
declare function gguf_get_val_i8(byval ctx as const gguf_context ptr, byval key_id as longint) as byte
declare function gguf_get_val_u16(byval ctx as const gguf_context ptr, byval key_id as longint) as ushort
declare function gguf_get_val_i16(byval ctx as const gguf_context ptr, byval key_id as longint) as short
declare function gguf_get_val_u32(byval ctx as const gguf_context ptr, byval key_id as longint) as ulong
declare function gguf_get_val_i32(byval ctx as const gguf_context ptr, byval key_id as longint) as long
declare function gguf_get_val_f32(byval ctx as const gguf_context ptr, byval key_id as longint) as single
declare function gguf_get_val_u64(byval ctx as const gguf_context ptr, byval key_id as longint) as ulongint
declare function gguf_get_val_i64(byval ctx as const gguf_context ptr, byval key_id as longint) as longint
declare function gguf_get_val_f64(byval ctx as const gguf_context ptr, byval key_id as longint) as double
declare function gguf_get_val_bool(byval ctx as const gguf_context ptr, byval key_id as longint) as bool
declare function gguf_get_val_str(byval ctx as const gguf_context ptr, byval key_id as longint) as const zstring ptr
declare function gguf_get_val_data(byval ctx as const gguf_context ptr, byval key_id as longint) as const any ptr
declare function gguf_get_arr_n(byval ctx as const gguf_context ptr, byval key_id as longint) as uinteger
' get raw pointer to the first element of the array with the given key_id
' for bool arrays, note that they are always stored as int8 on all platforms (usually this makes no difference)
declare function gguf_get_arr_data(byval ctx as const gguf_context ptr, byval key_id as longint) as const any ptr
' get ith C string from array with given key_id
declare function gguf_get_arr_str(byval ctx as const gguf_context ptr, byval key_id as longint, byval i as uinteger) as const zstring ptr
declare function gguf_get_n_tensors(byval ctx as const gguf_context ptr) as longint
declare function gguf_find_tensor(byval ctx as const gguf_context ptr, byval name as const zstring ptr) as longint ' returns -1 if the tensor is not found
declare function gguf_get_tensor_offset(byval ctx as const gguf_context ptr, byval tensor_id as longint) as uinteger
declare function gguf_get_tensor_name(byval ctx as const gguf_context ptr, byval tensor_id as longint) as const zstring ptr
declare function gguf_get_tensor_ne(byval ctx as const gguf_context ptr, byval tensor_id as longint) as const longint ptr ' returns ne, an array of GGML_MAX_DIMS elements; ne[dim] is 1 for dim >= n_dims
declare function gguf_get_tensor_type(byval ctx as const gguf_context ptr, byval tensor_id as longint) as ggml_type
declare function gguf_get_tensor_size(byval ctx as const gguf_context ptr, byval tensor_id as longint) as uinteger
' removes key if it exists, returns id that the key had prior to removal (-1 if it didn't exist)
declare function gguf_remove_key(byval ctx as gguf_context ptr, byval key as const zstring ptr) as longint
' overrides an existing KV pair or adds a new one, the new KV pair is always at the back
declare sub gguf_set_val_u8(byval ctx as gguf_context ptr, byval key as const zstring ptr, byval val as ubyte)
declare sub gguf_set_val_i8(byval ctx as gguf_context ptr, byval key as const zstring ptr, byval val as byte)
declare sub gguf_set_val_u16(byval ctx as gguf_context ptr, byval key as const zstring ptr, byval val as ushort)
declare sub gguf_set_val_i16(byval ctx as gguf_context ptr, byval key as const zstring ptr, byval val as short)
declare sub gguf_set_val_u32(byval ctx as gguf_context ptr, byval key as const zstring ptr, byval val as ulong)
declare sub gguf_set_val_i32(byval ctx as gguf_context ptr, byval key as const zstring ptr, byval val as long)
declare sub gguf_set_val_f32(byval ctx as gguf_context ptr, byval key as const zstring ptr, byval val as single)
declare sub gguf_set_val_u64(byval ctx as gguf_context ptr, byval key as const zstring ptr, byval val as ulongint)
declare sub gguf_set_val_i64(byval ctx as gguf_context ptr, byval key as const zstring ptr, byval val as longint)
declare sub gguf_set_val_f64(byval ctx as gguf_context ptr, byval key as const zstring ptr, byval val as double)
declare sub gguf_set_val_bool(byval ctx as gguf_context ptr, byval key as const zstring ptr, byval val as bool)
declare sub gguf_set_val_str(byval ctx as gguf_context ptr, byval key as const zstring ptr, byval val as const zstring ptr)
' creates a new array with n elements of the given type and copies the corresponding number of bytes from data
declare sub gguf_set_arr_data(byval ctx as gguf_context ptr, byval key as const zstring ptr, byval type as gguf_type, byval data as const any ptr, byval n as uinteger)
' creates a new array with n strings and copies the corresponding strings from data
declare sub gguf_set_arr_str(byval ctx as gguf_context ptr, byval key as const zstring ptr, byval data as const zstring ptr ptr, byval n as uinteger)
' set or add KV pairs from another context
declare sub gguf_set_kv(byval ctx as gguf_context ptr, byval src as const gguf_context ptr)
' add tensor to GGUF context, tensor name must be unique
declare sub gguf_add_tensor(byval ctx as gguf_context ptr, byval tensor as const ggml_tensor ptr)
' after changing a tensor's type, the offsets of all tensors with higher indices are immediately recalculated
'   in such a way that the tensor data remains as one contiguous block (except for padding)
declare sub gguf_set_tensor_type(byval ctx as gguf_context ptr, byval name as const zstring ptr, byval type as ggml_type)
' assumes that at least gguf_get_tensor_size bytes can be read from data
declare sub gguf_set_tensor_data(byval ctx as gguf_context ptr, byval name as const zstring ptr, byval data as const any ptr)
' writing gguf files can be done in 3 ways:
'
' - write the entire gguf_context to a binary file in a single pass:
'
'   gguf_write_to_file(ctx, fname, /*only_meta =*/ false);
'
' - write only the meta data to a file, then re-open the file and append the tensor data:
'
'   gguf_write_to_file(ctx, fname, /*only_meta =*/ true);
'   FILE * f = fopen(fname, "ab");
'   fwrite(f, ...); // write tensor data
'   fclose(f);
'
' - first prepare a file with a placeholder for the meta data, write the tensor data, then write the meta data:
'
'   FILE * f = fopen(fname, "wb");
'   const size_t size_meta = gguf_get_meta_size(ctx);
'   fseek(f, size_meta, SEEK_SET);
'   fwrite(f, ...); // write tensor data
'   void * data = malloc(size_meta);
'   gguf_get_meta_data(ctx, data);
'   rewind(f);
'   fwrite(data, 1, data, f);
'   free(data);
'   fclose(f);
'
' write the entire context to a binary file
declare function gguf_write_to_file_ptr(byval ctx as const gguf_context ptr, byval file as FILE ptr, byval only_meta as bool) as bool
declare function gguf_write_to_file(byval ctx as const gguf_context ptr, byval fname as const zstring ptr, byval only_meta as bool) as bool
' get the size in bytes of the meta data (header, kv pairs, tensor info) including padding
declare function gguf_get_meta_size(byval ctx as const gguf_context ptr) as uinteger
' writes the meta data to pointer "data"
declare sub gguf_get_meta_data(byval ctx as const gguf_context ptr, byval data as any ptr)

end extern

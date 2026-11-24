#pragma once

#include once "ggml.bi"
#include once "ggml-backend.bi"
#include once "crt/stdint.bi"

extern "C"

' This file contains functionality for training models using GGML.
' It is not strictly needed vs. just vanilla GGML but it provides a more high-level interface for common needs such as datasets.
' At the bottom of this file especially there are relatively high-level functions that are suitable use or adaptation in user code.
'
' Module maintainer: Johannes Gäßler (@JohannesGaessler, johannesg@5d6.de)
type ggml_opt_dataset_t as ggml_opt_dataset ptr
type ggml_opt_context_t as ggml_opt_context ptr
type ggml_opt_result_t as ggml_opt_result ptr
' ====== Loss ======
' built-in loss types, i.e. the built-in quantities minimized by the optimizer
' custom loss types can be defined via mean or sum which simply reduce the outputs for all datapoints to a single value

enum ggml_opt_loss_type
	GGML_OPT_LOSS_TYPE_MEAN
	GGML_OPT_LOSS_TYPE_SUM
	GGML_OPT_LOSS_TYPE_CROSS_ENTROPY
	GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR
end enum

' ====== Dataset ======
'' TODO: extern ggml_opt_dataset_t ggml_opt_dataset_init( enum ggml_type type_data, [TK_FBCODE ' the type for the internal data tensor] enum ggml_type type_label, [TK_FBCODE ' the type for the internal labels tensor] int64_t ne_datapoint, [TK_FBCODE ' number of elements per datapoint] int64_t ne_label, [TK_FBCODE ' number of elements per label] int64_t ndata, [TK_FBCODE ' total number of datapoints/labels] int64_t ndata_shard);
' number of datapoints/labels per shard (unit at which the dataset is shuffled/copied)
declare sub ggml_opt_dataset_free(byval dataset as ggml_opt_dataset_t)
' get underlying tensors that store the data
declare function ggml_opt_dataset_ndata(byval dataset as ggml_opt_dataset_t) as longint
declare function ggml_opt_dataset_data(byval dataset as ggml_opt_dataset_t) as ggml_tensor ptr ' shape = [ne_datapoint, ndata]
declare function ggml_opt_dataset_labels(byval dataset as ggml_opt_dataset_t) as ggml_tensor ptr ' shape = [nd_label,     ndata]
' shuffle idata first datapoints from dataset with RNG from opt_ctx, shuffle all datapoints if idata is negative
declare sub ggml_opt_dataset_shuffle(byval opt_ctx as ggml_opt_context_t, byval dataset as ggml_opt_dataset_t, byval idata as longint)
' get batch at position ibatch from dataset and copy the data to data_batch and labels_batch
'' TODO: extern void ggml_opt_dataset_get_batch( ggml_opt_dataset_t dataset, struct ggml_tensor * data_batch, [TK_FBCODE ' shape = [ne_datapoint, ndata_batch]] struct ggml_tensor * labels_batch, [TK_FBCODE ' shape = [ne_label,     ndata_batch]] int64_t ibatch);
declare sub ggml_opt_dataset_get_batch_host(byval dataset as ggml_opt_dataset_t, byval data_batch as any ptr, byval nb_data_batch as uinteger, byval labels_batch as any ptr, byval ibatch as longint)
' ====== Model / Context ======

enum ggml_opt_build_type
	GGML_OPT_BUILD_TYPE_FORWARD = 10
	GGML_OPT_BUILD_TYPE_GRAD = 20
	GGML_OPT_BUILD_TYPE_OPT = 30
end enum

enum ggml_opt_optimizer_type
	GGML_OPT_OPTIMIZER_TYPE_ADAMW
	GGML_OPT_OPTIMIZER_TYPE_SGD
	GGML_OPT_OPTIMIZER_TYPE_COUNT
end enum

' parameters that control which optimizer is used and how said optimizer tries to find the minimal loss

type ggml_opt_optimizer_params_adamw
	alpha as single ' learning rate
	beta1 as single ' first AdamW momentum
	beta2 as single ' second AdamW momentum
	eps as single ' epsilon for numerical stability
	wd as single ' weight decay - 0.0f to disable
end type

type ggml_opt_optimizer_params_sgd
	alpha as single ' learning rate
	wd as single ' weight decay
end type

type ggml_opt_optimizer_params
	adamw as ggml_opt_optimizer_params_adamw
	sgd as ggml_opt_optimizer_params_sgd
end type

' callback to calculate optimizer parameters prior to a backward pass
' userdata can be used to pass arbitrary data
type ggml_opt_get_optimizer_params as function(byval userdata as any ptr) as ggml_opt_optimizer_params
' returns the default optimizer params (constant, hard-coded values)
' userdata is not used
declare function ggml_opt_get_default_optimizer_params(byval userdata as any ptr) as ggml_opt_optimizer_params
' casts userdata to ggml_opt_optimizer_params and returns it
declare function ggml_opt_get_constant_optimizer_params(byval userdata as any ptr) as ggml_opt_optimizer_params
' parameters for initializing a new optimization context

type ggml_opt_params
	backend_sched as ggml_backend_sched_t ' defines which backends are used to construct the compute graphs
	' by default the forward graph needs to be reconstructed for each eval
	' if ctx_compute, inputs, and outputs are set the graphs are instead allocated statically
	ctx_compute as ggml_context ptr
	inputs as ggml_tensor ptr
	outputs as ggml_tensor ptr
	loss_type as ggml_opt_loss_type
	build_type as ggml_opt_build_type
	opt_period as long ' after how many gradient accumulation steps an optimizer step should be done
	get_opt_pars as ggml_opt_get_optimizer_params ' callback for calculating optimizer parameters
	get_opt_pars_ud as any ptr ' userdata for calculating optimizer parameters
	' only GGML_OPT_OPTIMIZER_TYPE_ADAMW needs m, v momenta per parameter tensor
	optimizer as ggml_opt_optimizer_type
end type

' get parameters for an optimization context with defaults set where possible
' parameters for which no sensible defaults exist are supplied as arguments to this function
declare function ggml_opt_default_params(byval backend_sched as ggml_backend_sched_t, byval loss_type as ggml_opt_loss_type) as ggml_opt_params
declare function ggml_opt_init(byval params as ggml_opt_params) as ggml_opt_context_t
declare sub ggml_opt_free(byval opt_ctx as ggml_opt_context_t)
' set gradients to zero, initialize loss, and optionally reset the optimizer
declare sub ggml_opt_reset(byval opt_ctx as ggml_opt_context_t, byval optimizer as bool)
declare function ggml_opt_static_graphs(byval opt_ctx as ggml_opt_context_t) as bool ' whether the graphs are allocated_statically
' get underlying tensors that store data
' if not using static graphs these pointers become invalid with the next call to ggml_opt_alloc
declare function ggml_opt_inputs(byval opt_ctx as ggml_opt_context_t) as ggml_tensor ptr ' forward graph input tensor
declare function ggml_opt_outputs(byval opt_ctx as ggml_opt_context_t) as ggml_tensor ptr ' forward graph output tensor
declare function ggml_opt_labels(byval opt_ctx as ggml_opt_context_t) as ggml_tensor ptr ' labels to compare outputs against
declare function ggml_opt_loss(byval opt_ctx as ggml_opt_context_t) as ggml_tensor ptr ' scalar tensor that contains the loss
declare function ggml_opt_pred(byval opt_ctx as ggml_opt_context_t) as ggml_tensor ptr ' predictions made by outputs
declare function ggml_opt_ncorrect(byval opt_ctx as ggml_opt_context_t) as ggml_tensor ptr ' number of matching predictions between outputs and labels
' get the gradient accumulator for a node from the forward graph
declare function ggml_opt_grad_acc(byval opt_ctx as ggml_opt_context_t, byval node as ggml_tensor ptr) as ggml_tensor ptr
declare function ggml_opt_context_optimizer_type(byval as ggml_opt_context_t) as ggml_opt_optimizer_type ' TODO consistent naming scheme
declare function ggml_opt_optimizer_name(byval as ggml_opt_optimizer_type) as const zstring ptr
' ====== Optimization Result ======
declare function ggml_opt_result_init() as ggml_opt_result_t
declare sub ggml_opt_result_free(byval result as ggml_opt_result_t)
declare sub ggml_opt_result_reset(byval result as ggml_opt_result_t)
' get data from result, uncertainties are optional and can be ignored by passing NULL
declare sub ggml_opt_result_ndata(byval result as ggml_opt_result_t, byval ndata as longint ptr) ' writes 1 value, number of datapoints
declare sub ggml_opt_result_loss(byval result as ggml_opt_result_t, byval loss as double ptr, byval unc as double ptr) ' writes 1 value
declare sub ggml_opt_result_pred(byval result as ggml_opt_result_t, byval pred as long ptr) ' writes ndata values
declare sub ggml_opt_result_accuracy(byval result as ggml_opt_result_t, byval accuracy as double ptr, byval unc as double ptr) ' writes 1 value
' ====== Computation ======
' if not using static graphs, this function must be called prior to ggml_opt_alloc
declare sub ggml_opt_prepare_alloc(byval opt_ctx as ggml_opt_context_t, byval ctx_compute as ggml_context ptr, byval gf as ggml_cgraph ptr, byval inputs as ggml_tensor ptr, byval outputs as ggml_tensor ptr)
' allocate the next graph for evaluation, either forward or forward + backward
' must be called exactly once prior to calling ggml_opt_eval
declare sub ggml_opt_alloc(byval opt_ctx as ggml_opt_context_t, byval backward as bool)
' do forward pass, increment result if not NULL, do backward pass if allocated
declare sub ggml_opt_eval(byval opt_ctx as ggml_opt_context_t, byval result as ggml_opt_result_t)
' ############################################################################
' ## The high-level functions start here. They do not depend on any private ##
' ## functions or structs and can be copied to and adapted for user code.   ##
' ############################################################################
' ====== Intended Usage ======
'
' 1. Select the appropriate loss for your problem.
' 2. Create a dataset and set the data for the "data" tensor. Also set the "labels" tensor if your loss needs them.
'    Setting the shard size to 1 will be fine, it's the granularity with which data is shuffled/loaded (bigger values are faster).
' 3. Create a GGML graph for your model with no_alloc == true. Use two separate contexts for the tensors.
'    The first context should contain the model parameters and inputs and be allocated statically in user code.
'    The second context should contain all other tensors and will be (re)allocated automatically.
'    Due to this automated allocation the data of the second context is not defined when accessed in user code.
'    Note that the second dimension of the inputs/outputs are interpreted as the number of datapoints in those tensors.
' 4. Call ggml_opt_fit. If you need more control you can use ggml_opt_epoch instead.
' signature for a callback while evaluating opt_ctx on dataset, called after an evaluation
'' TODO: typedef void (*ggml_opt_epoch_callback)( bool train, [TK_FBCODE ' true after training evaluation, false after validation evaluation] ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, ggml_opt_result_t result, [TK_FBCODE ' result associated with the dataset subsection] int64_t ibatch, [TK_FBCODE ' number of batches that have been evaluated so far] int64_t ibatch_max, [TK_FBCODE ' total number of batches in this dataset subsection] int64_t t_start_us);
' time at which the evaluation on the dataset subsection was started
' do training on front of dataset, do evaluation only on back of dataset
'' TODO: extern void ggml_opt_epoch( ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, ggml_opt_result_t result_train, [TK_FBCODE ' result to increment during training, ignored if NULL] ggml_opt_result_t result_eval, [TK_FBCODE ' result to increment during evaluation, ignored if NULL] int64_t idata_split, [TK_FBCODE ' data index at which to split training and evaluation] ggml_opt_epoch_callback callback_train, ggml_opt_epoch_callback callback_eval);
' callback that prints a progress bar on stderr
declare sub ggml_opt_epoch_callback_progress_bar(byval train as bool, byval opt_ctx as ggml_opt_context_t, byval dataset as ggml_opt_dataset_t, byval result as ggml_opt_result_t, byval ibatch as longint, byval ibatch_max as longint, byval t_start_us as longint)
' fit model defined by inputs and outputs to dataset
'' TODO: extern void ggml_opt_fit( ggml_backend_sched_t backend_sched, [TK_FBCODE ' backend scheduler for constructing the compute graphs] struct ggml_context * ctx_compute, [TK_FBCODE ' context with temporarily allocated tensors to calculate the outputs] struct ggml_tensor * inputs, [TK_FBCODE ' input tensor with shape [ne_datapoint, ndata_batch]] struct ggml_tensor * outputs, [TK_FBCODE ' output tensor, must have shape [ne_label, ndata_batch] if labels are used] ggml_opt_dataset_t dataset, [TK_FBCODE ' dataset with data and optionally also labels] enum ggml_opt_loss_type loss_type, [TK_FBCODE ' loss to minimize] enum ggml_opt_optimizer_type optimizer, [TK_FBCODE ' sgd or adamw] ggml_opt_get_optimizer_params get_opt_pars, [TK_FBCODE ' callback to get optimizer params, userdata is pointer to epoch (of type int64_t)] int64_t nepoch, [TK_FBCODE ' how many times the dataset should be iterated over] int64_t nbatch_logical, [TK_FBCODE ' datapoints optimizer step, must be a multiple of ndata_batch in inputs/outputs] float val_split, [TK_FBCODE ' fraction of the dataset to use for validation, must be in [0.0f, 1.0f)] bool silent);
' whether or not info prints to stderr should be suppressed

end extern

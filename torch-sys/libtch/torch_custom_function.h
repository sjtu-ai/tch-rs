#ifndef __TORCH_CUSTOM_FUNCTION_INCLUDE_H__
#define __TORCH_CUSTOM_FUNCTION_INCLUDE_H__

#include <stdio.h>
#include <torch/torch.h>

#include <cstdarg>

#ifdef __cplusplus
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

struct VecArrayDef;

extern "C" {
typedef torch::Tensor* tensor;
typedef VecArrayDef* vec_array;
typedef at::cuda::CUDAStream* at_cuda_stream_t;
typedef at::Stream* at_stream_t;
typedef at::cuda::CUDAEvent* at_cuda_event_t;
typedef at::cuda::CUDAGraph* at_cuda_graph_t;
#else
typedef void* tensor;
typedef void* vec_array;
typedef void* at_cuda_stream_t;
typedef void* at_stream_t;
typedef void* at_cuda_event_t;
typedef void* at_cuda_graph_t;
#endif

using namespace torch::autograd;
using Tensor = torch::Tensor;

vec_array rust_custom_function_forward_callback(void*);
vec_array rust_custom_function_backward_callback(void*, tensor*, u_int32_t);
// VecArray rust_custom_function_backward_callback(void*, Tensor*, u_int32_t);
void rust_custom_function_destroy_vec_array(vec_array);

tensor* invoke_custom_function_from_rust(void* callback_target);

tensor* invoke_custom_function_from_rust_1(void* callback_target, tensor a);

tensor* invoke_custom_function_from_rust_2(void* callback_target, tensor a, tensor b);

tensor* invoke_custom_function_from_rust_3(void* callback_target, tensor a, tensor b, tensor c);

tensor* invoke_custom_function_from_rust_4(void* callback_target, tensor a, tensor b, tensor c, tensor d);

tensor* invoke_custom_function_from_rust_5(void* callback_target, tensor a, tensor b, tensor c, tensor d, tensor e);

void invoke_custom_function_from_rust_1_void(void* callback_target, tensor a);

void invoke_custom_function_from_rust_2_void(void* callback_target, tensor a, tensor b);

void invoke_custom_function_from_rust_3_void(void* callback_target, tensor a, tensor b, tensor c);

void invoke_custom_function_from_rust_4_void(void* callback_target, tensor a, tensor b, tensor c, tensor d);

void invoke_custom_function_from_rust_5_void(void* callback_target, tensor a, tensor b, tensor c, tensor d, tensor e);

bool is_grad_enabled();

void run_backward_batch(tensor* tensors, int n_tensors, tensor* grad_tensors, int n_grad_tensors, tensor* inputs,
                        int n_inputs, int keep_graph, int create_graph);

#ifdef __cplusplus
}; // extern "C"
#endif

#endif
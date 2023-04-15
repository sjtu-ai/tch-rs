#ifndef __TORCH_SYS_CUDA_STREAM_INCLUDE_H__
#define __TORCH_SYS_CUDA_STREAM_INCLUDE_H__
#include <torch/torch.h>

#ifdef __cplusplus
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

extern "C" {
typedef at::cuda::CUDAStream* at_cuda_stream_t;
typedef at::Stream* at_stream_t;
typedef at::cuda::CUDAEvent* at_cuda_event_t;
typedef at::cuda::CUDAGraph* at_cuda_graph_t;
#else
typedef void* at_cuda_stream_t;
typedef void* at_stream_t;
typedef void* at_cuda_event_t;
typedef void* at_cuda_graph_t;
#endif

typedef int64_t at_stream_id_t;
typedef int8_t at_device_index_t;
typedef int8_t at_device_type_t;
typedef void* at_device_t;

/* device stream */
at_stream_t at_device_stream_new(at_stream_id_t stream_id, at_device_index_t device_index,
                                 at_device_type_t device_type);

void at_device_stream_free(at_stream_t stream);

/* cuda stream */
at_cuda_stream_t at_cuda_stream_get_current(at_device_index_t device_index);

void at_cuda_stream_set_current(at_cuda_stream_t stream);

at_cuda_stream_t at_cuda_stream_get_default(at_device_index_t device_index);

at_cuda_stream_t at_cuda_stream_new(bool is_high_priority, at_device_index_t device_index);

void at_cuda_stream_free(at_cuda_stream_t stream);

at_device_index_t at_cuda_stream_get_device_index(at_cuda_stream_t stream);

at_stream_id_t at_cuda_stream_get_id(at_cuda_stream_t stream);

void at_cuda_stream_synchronize(at_cuda_stream_t stream);

/* allow tf32 , https://pytorch.org/docs/stable/notes/cuda.html*/
void at_cuda_matmul_set_allow_tf32(bool allow);
bool at_cuda_matmul_get_allow_tf32();
void at_cudnn_set_allow_tf32(bool allow);
bool at_cudnn_get_allow_tf32();

/* float 16 */
void at_set_allow_bf16_reduced_precision(bool allow);
bool at_get_allow_bf16_reduced_precision();
void at_set_allow_fp16_reduced_precision(bool allow);
bool at_get_allow_fp16_reduced_precision();

/* memory */
void at_cuda_memory_stats(at_device_index_t device_index, c10::cuda::CUDACachingAllocator::DeviceStats* stats);
void at_cuda_memory_empty_cache();

/* cuda event */
at_cuda_event_t at_cuda_event_new(int flag);
void at_cuda_event_free(at_cuda_event_t event);
void at_cuda_event_record(at_cuda_event_t event);
void at_cuda_event_record_stream(at_cuda_event_t event, at_cuda_stream_t stream);
void at_cuda_event_record_stream_once(at_cuda_event_t event, at_cuda_stream_t stream);
bool at_cuda_event_query(at_cuda_event_t event);
void at_cuda_event_synchronize(at_cuda_event_t event);
float at_cuda_event_elapsed_time(at_cuda_event_t event1, at_cuda_event_t event2);
void at_cuda_event_block(at_cuda_event_t event, at_cuda_stream_t stream);

/* cuda graph*/
struct MemPoolId {
    unsigned long long t1;
    unsigned long long t2;
};
at_cuda_graph_t at_cuda_graph_new();
void at_cuda_graph_free(at_cuda_graph_t graph);
void at_cuda_graph_capture_begin(at_cuda_graph_t graph, MemPoolId pool);
void at_cuda_graph_capture_end(at_cuda_graph_t graph);
void at_cuda_graph_replay(at_cuda_graph_t graph);
void at_cuda_graph_reset(at_cuda_graph_t graph);
MemPoolId at_cuda_graph_pool(at_cuda_graph_t graph);
void at_cuda_graph_enable_debug_mode(at_cuda_graph_t graph);
void at_cuda_graph_debug_dump(at_cuda_graph_t graph, char* debug_path);

#ifdef __cplusplus
}; // extern "C"
#endif

#endif
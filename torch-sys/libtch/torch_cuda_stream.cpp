#include "torch_cuda_stream.h"
extern thread_local char* torch_last_err;

#define PROTECT_DESC_BEGIN() try {
#define PROTECT_DESC_END(desc)                            \
    }                                                     \
    catch (std::exception & e) {                          \
        torch_last_err = strdup(e.what());                \
        std::cerr << desc << torch_last_err << std::endl; \
        throw e;                                          \
    }

at_stream_t at_device_stream_new(at_stream_id_t stream_id, at_device_index_t device_index,
                                 at_device_type_t device_type) {
    PROTECT_DESC_BEGIN();
    return new at::Stream(at::Stream::UNSAFE, torch::Device((torch::DeviceType)device_type, device_index), stream_id);
    PROTECT_DESC_END("at_device_stream_new: ");
}

void at_device_stream_free(at_stream_t stream) {
    PROTECT_DESC_BEGIN();
    delete (torch::Stream*)stream;
    PROTECT_DESC_END("at_device_stream_free: ");
}

at_cuda_stream_t at_cuda_stream_get_current(at_device_index_t device_index) {
    PROTECT_DESC_BEGIN();
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    return new at::cuda::CUDAStream(stream);
    PROTECT_DESC_END("at_cuda_stream_get_current: ");
}
void at_cuda_stream_set_current(at_cuda_stream_t stream) {
    PROTECT_DESC_BEGIN();
    at::cuda::setCurrentCUDAStream(*((at::cuda::CUDAStream*)stream));
    PROTECT_DESC_END("at_cuda_stream_set_current: ");
}
at_cuda_stream_t at_cuda_stream_get_default(at_device_index_t device_index) {
    PROTECT_DESC_BEGIN();
    auto stream = at::cuda::getDefaultCUDAStream(device_index);
    return new at::cuda::CUDAStream(stream);
    PROTECT_DESC_END("at_cuda_stream_get_default: ");
}
at_cuda_stream_t at_cuda_stream_new(bool is_high_priority, at_device_index_t device_index) {
    PROTECT_DESC_BEGIN();
    auto stream = at::cuda::getStreamFromPool(is_high_priority, (torch::DeviceIndex)device_index);
    return new at::cuda::CUDAStream(stream);
    PROTECT_DESC_END("at_cuda_stream_new: ");
}
void at_cuda_stream_free(at_cuda_stream_t stream) {
    PROTECT_DESC_BEGIN();
    delete (at::cuda::CUDAStream*)stream;
    PROTECT_DESC_END("at_cuda_stream_free: ");
}
at_device_index_t at_cuda_stream_get_device_index(at_cuda_stream_t stream) {
    PROTECT_DESC_BEGIN();
    return ((at::cuda::CUDAStream*)stream)->device_index();
    PROTECT_DESC_END("at_cuda_stream_get_device_index: ");
}

at_stream_id_t at_cuda_stream_get_id(at_cuda_stream_t stream) {
    PROTECT_DESC_BEGIN();
    return ((at::cuda::CUDAStream*)stream)->id();
    PROTECT_DESC_END("at_cuda_stream_get_id: ");
}
void at_cuda_stream_synchronize(at_cuda_stream_t stream) {
    PROTECT_DESC_BEGIN();
    // cudaStreamSynchronize(((at::cuda::CUDAStream*)stream)->stream());
    ((at::cuda::CUDAStream*)stream)->synchronize();
    PROTECT_DESC_END("at_cuda_stream_synchronize: ");
}

void at_cuda_matmul_set_allow_tf32(bool allow) {
    PROTECT_DESC_BEGIN();
    at::globalContext().setAllowTF32CuBLAS(allow);
    PROTECT_DESC_END("at_cuda_matmul_set_allow_tf32:");
}
bool at_cuda_matmul_get_allow_tf32() {
    PROTECT_DESC_BEGIN();
    return at::globalContext().allowTF32CuBLAS();
    PROTECT_DESC_END("at_cuda_matmul_get_allow_tf32:");
}
void at_cudnn_set_allow_tf32(bool allow) {
    PROTECT_DESC_BEGIN();
    at::globalContext().setAllowTF32CuDNN(allow);
    PROTECT_DESC_END("at_cudnn_set_allow_tf32:");
}
bool at_cudnn_get_allow_tf32() {
    PROTECT_DESC_BEGIN();
    return at::globalContext().allowTF32CuDNN();
    PROTECT_DESC_END("at_cudnn_get_allow_tf32:");
}

void at_set_allow_bf16_reduced_precision(bool allow) {
    PROTECT_DESC_BEGIN();
    at::globalContext().setAllowBF16ReductionCuBLAS(allow);
    PROTECT_DESC_END("at_set_allow_bf16_reduced_precision:");
}
bool at_get_allow_bf16_reduced_precision() {
    PROTECT_DESC_BEGIN();
    return at::globalContext().allowBF16ReductionCuBLAS();
    PROTECT_DESC_END("at_get_allow_bf16_reduced_precision:");
}
void at_set_allow_fp16_reduced_precision(bool allow) {
    PROTECT_DESC_BEGIN();
    at::globalContext().setAllowFP16ReductionCuBLAS(allow);
    PROTECT_DESC_END("at_set_allow_fp16_reduced_precision:");
}
bool at_get_allow_fp16_reduced_precision() {
    PROTECT_DESC_BEGIN();
    return at::globalContext().allowFP16ReductionCuBLAS();
    PROTECT_DESC_END("at_get_allow_fp16_reduced_precision:");
}

/* memory */
void at_cuda_memory_stats(at_device_index_t device, c10::cuda::CUDACachingAllocator::DeviceStats* stats) {
    PROTECT_DESC_BEGIN();
    *stats = c10::cuda::CUDACachingAllocator::getDeviceStats((int)device);
    PROTECT_DESC_END("at_cuda_memory_stats:");
}
void at_cuda_memory_empty_cache() {
    PROTECT_DESC_BEGIN();
    c10::cuda::CUDACachingAllocator::emptyCache();
    PROTECT_DESC_END("at_cuda_memory_empty_cache:");
}

/* cuda event */
at_cuda_event_t at_cuda_event_new(int flag) {
    PROTECT_DESC_BEGIN();
    return new at::cuda::CUDAEvent(flag);
    PROTECT_DESC_END("at_cuda_event_new:");
}
void at_cuda_event_free(at_cuda_event_t event) {
    PROTECT_DESC_BEGIN();
    delete (at::cuda::CUDAEvent*)event;
    PROTECT_DESC_END("at_cuda_event_free:");
}
void at_cuda_event_record(at_cuda_event_t event) {
    PROTECT_DESC_BEGIN();
    ((at::cuda::CUDAEvent*)event)->record();
    PROTECT_DESC_END("at_cuda_event_record:");
}
void at_cuda_event_record_stream(at_cuda_event_t event, at_cuda_stream_t stream) {
    PROTECT_DESC_BEGIN();
    ((at::cuda::CUDAEvent*)event)->record(*((at::cuda::CUDAStream*)stream));
    PROTECT_DESC_END("at_cuda_event_record_stream:");
}
void at_cuda_event_record_stream_once(at_cuda_event_t event, at_cuda_stream_t stream) {
    PROTECT_DESC_BEGIN();
    ((at::cuda::CUDAEvent*)event)->recordOnce(*((at::cuda::CUDAStream*)stream));
    PROTECT_DESC_END("at_cuda_event_record_stream_once:");
}
bool at_cuda_event_query(at_cuda_event_t event) {
    PROTECT_DESC_BEGIN();
    return ((at::cuda::CUDAEvent*)event)->query();
    PROTECT_DESC_END("at_cuda_event_query:");
}
void at_cuda_event_synchronize(at_cuda_event_t event) {
    PROTECT_DESC_BEGIN();
    ((at::cuda::CUDAEvent*)event)->synchronize();
    PROTECT_DESC_END("at_cuda_event_synchronize:");
}
float at_cuda_event_elapsed_time(at_cuda_event_t event1, at_cuda_event_t event2) {
    PROTECT_DESC_BEGIN();
    return ((at::cuda::CUDAEvent*)event1)->elapsed_time(*((at::cuda::CUDAEvent*)event2));
    PROTECT_DESC_END("at_cuda_event_elapsed_time:");
}
void at_cuda_event_block(at_cuda_event_t event, at_cuda_stream_t stream) {
    PROTECT_DESC_BEGIN();
    ((at::cuda::CUDAEvent*)event)->block(*((at::cuda::CUDAStream*)stream));
    PROTECT_DESC_END("at_cuda_event_block:");
}

/* cuda graph */
at_cuda_graph_t at_cuda_graph_new() {
    PROTECT_DESC_BEGIN();
    return new at::cuda::CUDAGraph();
    PROTECT_DESC_END("at_cuda_graph_new: ");
}
void at_cuda_graph_free(at_cuda_graph_t graph) {
    PROTECT_DESC_BEGIN();
    delete (at::cuda::CUDAGraph*)graph;
    PROTECT_DESC_END("at_cuda_graph_free: ");
}
void at_cuda_graph_capture_begin(at_cuda_graph_t graph, MemPoolId pool) {
    PROTECT_DESC_BEGIN();
    ((at::cuda::CUDAGraph*)graph)->capture_begin({pool.t1, pool.t2});
    PROTECT_DESC_END("at_cuda_graph_capture_begin: ");
}
void at_cuda_graph_capture_end(at_cuda_graph_t graph) {
    PROTECT_DESC_BEGIN();
    ((at::cuda::CUDAGraph*)graph)->capture_end();
    PROTECT_DESC_END("at_cuda_graph_capture_end: ");
}
void at_cuda_graph_replay(at_cuda_graph_t graph) {
    PROTECT_DESC_BEGIN();
    ((at::cuda::CUDAGraph*)graph)->replay();
    PROTECT_DESC_END("at_cuda_graph_replay: ");
}
void at_cuda_graph_reset(at_cuda_graph_t graph) {
    PROTECT_DESC_BEGIN();
    ((at::cuda::CUDAGraph*)graph)->reset();
    PROTECT_DESC_END("at_cuda_graph_reset: ");
}
MemPoolId at_cuda_graph_pool(at_cuda_graph_t graph) {
    PROTECT_DESC_BEGIN();
    auto pool = ((at::cuda::CUDAGraph*)graph)->pool();
    return {pool.first, pool.second};
    PROTECT_DESC_END("at_cuda_graph_pool: ");
}
void at_cuda_graph_enable_debug_mode(at_cuda_graph_t graph) {
    PROTECT_DESC_BEGIN();
    ((at::cuda::CUDAGraph*)graph)->enable_debug_mode();
    PROTECT_DESC_END("at_cuda_graph_enable_debug_mode: ");
}
void at_cuda_graph_debug_dump(at_cuda_graph_t graph, char* debug_path) {
    PROTECT_DESC_BEGIN();
    ((at::cuda::CUDAGraph*)graph)->debug_dump(debug_path);
    PROTECT_DESC_END("at_cuda_graph_debug_dump: ");
}
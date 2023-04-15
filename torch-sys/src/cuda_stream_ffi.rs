use std::ffi::{c_char, c_int};

#[repr(C)]
pub struct C_device_stream {
    _private: [u8; 0],
}

#[repr(C)]
pub struct C_cuda_stream {
    _private: [u8; 0],
}

#[repr(C)]
pub struct C_cuda_event {
    _private: [u8; 0],
}

#[repr(C)]
pub struct C_cuda_graph {
    _private: [u8; 0],
}

#[derive(Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemPoolId {
    first: u64,
    second: u64,
}

pub mod memory {
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
    #[repr(C)]
    pub struct DeviceStats {
        pub allocation: StatArray,
        pub segment: StatArray,
        pub active: StatArray,
        pub inactive_split: StatArray,
        pub allocated_bytes: StatArray,
        pub reserved_bytes: StatArray,
        pub active_bytes: StatArray,
        pub inactive_split_bytes: StatArray,
        pub requested_bytes: StatArray,
        pub num_alloc_retries: i64,
        pub num_ooms: i64,
        pub oversized_allocations: Stat,
        pub oversize_segments: Stat,
        pub max_split_size: i64,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
    #[repr(C)]
    pub struct StatArray {
        pub all: Stat,
        pub small_pool: Stat,
        pub large_pool: Stat,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
    #[repr(C)]
    pub struct Stat {
        pub current: i64,
        pub peak: i64,
        pub allocated: i64,
        pub freed: i64,
    }
}

extern "C" {
    pub fn at_device_stream_new(
        stream_id: i64,
        device_index: u8,
        device_type: u8,
    ) -> *mut C_device_stream;
    pub fn at_device_stream_free(c_cuda_stream: *mut C_device_stream);

    pub fn at_cuda_stream_new(is_high_priority: bool, device_index: u8) -> *mut C_cuda_stream;
    pub fn at_cuda_stream_free(c_cuda_stream: *mut C_cuda_stream);
    pub fn at_cuda_stream_get_current(device_index: u8) -> *mut C_cuda_stream;
    pub fn at_cuda_stream_set_current(c_cuda_stream: *mut C_cuda_stream);
    pub fn at_cuda_stream_get_default(device_index: u8) -> *mut C_cuda_stream;
    pub fn at_cuda_stream_get_device_index(c_cuda_stream: *mut C_cuda_stream) -> u8;
    pub fn at_cuda_stream_get_id(c_cuda_stream: *mut C_cuda_stream) -> i64;
    pub fn at_cuda_stream_synchronize(c_cuda_stream: *mut C_cuda_stream);

    /* TensorFloat32, enable flag */
    pub fn at_cuda_matmul_set_allow_tf32(allow_tf32: bool);
    pub fn at_cuda_matmul_get_allow_tf32() -> bool;
    pub fn at_cudnn_set_allow_tf32(allow_tf32: bool);
    pub fn at_cudnn_get_allow_tf32() -> bool;

    /* fp16, bf16 */
    pub fn at_set_allow_bf16_reduced_precision(allow: bool);
    pub fn at_get_allow_bf16_reduced_precision() -> bool;
    pub fn at_set_allow_fp16_reduced_precision(allow: bool);
    pub fn at_get_allow_fp16_reduced_precision() -> bool;

    /* memory */
    pub fn at_cuda_memory_stats(device_index: u8, stats: *mut memory::DeviceStats);
    pub fn at_cuda_memory_empty_cache();

    /* cuda event */
    pub fn at_cuda_event_new(flag: c_int) -> *mut C_cuda_event;
    pub fn at_cuda_event_free(event: *mut C_cuda_event);
    pub fn at_cuda_event_record(event: *mut C_cuda_event);
    pub fn at_cuda_event_record_stream(event: *mut C_cuda_event, stream: *mut C_cuda_stream);
    pub fn at_cuda_event_record_stream_once(event: *mut C_cuda_event, stream: *mut C_cuda_stream);
    pub fn at_cuda_event_query(event: *mut C_cuda_event) -> bool;
    pub fn at_cuda_event_synchronize(event: *mut C_cuda_event);
    pub fn at_cuda_event_elapsed_time(event: *mut C_cuda_event, other: *mut C_cuda_event) -> f64;
    pub fn at_cuda_event_block(event: *mut C_cuda_event, stream: *mut C_cuda_stream);

    /* cuda graph */
    pub fn at_cuda_graph_new() -> *mut C_cuda_graph;
    pub fn at_cuda_graph_free(graph: *mut C_cuda_graph);
    pub fn at_cuda_graph_capture_begin(graph: *mut C_cuda_graph, pool_id: MemPoolId);
    pub fn at_cuda_graph_capture_end(graph: *mut C_cuda_graph);
    pub fn at_cuda_graph_replay(graph: *mut C_cuda_graph);
    pub fn at_cuda_graph_reset(graph: *mut C_cuda_graph);
    pub fn at_cuda_graph_pool(graph: *mut C_cuda_graph) -> MemPoolId;
    pub fn at_cuda_graph_enable_debug_mode(graph: *mut C_cuda_graph);
    pub fn at_cuda_graph_debug_dump(graph: *mut C_cuda_graph, path: *const c_char);

}

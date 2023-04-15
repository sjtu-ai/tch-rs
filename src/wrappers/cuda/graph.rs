use std::ffi::CString;

pub use torch_sys::cuda_stream_ffi::MemPoolId;
use torch_sys::cuda_stream_ffi::{self, C_cuda_graph};

use crate::{empty_cuda_cache, CudaStream};

pub struct CudaGrpah {
    pub c_cuda_graph: *mut C_cuda_graph,
}
impl Drop for CudaGrpah {
    fn drop(&mut self) {
        unsafe {
            cuda_stream_ffi::at_cuda_graph_free(self.c_cuda_graph);
        }
    }
}
impl CudaGrpah {
    pub fn new() -> Self {
        let c_cuda_graph = unsafe { cuda_stream_ffi::at_cuda_graph_new() };
        Self { c_cuda_graph }
    }
    pub fn capture_begin(&self, pool_id: MemPoolId) {
        unsafe {
            cuda_stream_ffi::at_cuda_graph_capture_begin(self.c_cuda_graph, pool_id);
        }
    }
    pub fn capture_end(&self) {
        unsafe {
            cuda_stream_ffi::at_cuda_graph_capture_end(self.c_cuda_graph);
        }
    }
    pub fn replay(&self) {
        unsafe {
            cuda_stream_ffi::at_cuda_graph_replay(self.c_cuda_graph);
        }
    }
    pub fn reset(&self) {
        unsafe {
            cuda_stream_ffi::at_cuda_graph_reset(self.c_cuda_graph);
        }
    }
    pub fn get_pool_id(&self) -> MemPoolId {
        unsafe { cuda_stream_ffi::at_cuda_graph_pool(self.c_cuda_graph) }
    }
    pub fn enable_debug_mode(&self) {
        unsafe {
            cuda_stream_ffi::at_cuda_graph_enable_debug_mode(self.c_cuda_graph);
        }
    }
    pub fn debug_dump(&self, p: &str) {
        let c_path = CString::new(p).expect("dubug_dump CString::new failed");
        unsafe {
            cuda_stream_ffi::at_cuda_graph_debug_dump(self.c_cuda_graph, c_path.as_ptr());
        }
    }
    pub fn with_graph<CTX, T>(
        &self,
        mem_pool_id: MemPoolId,
        capture_stream: &CudaStream,
        ctx: CTX,
        f: impl FnOnce(CTX) -> T,
    ) -> T {
        capture_stream.synchronize();
        empty_cuda_cache();
        capture_stream.with_stream(ctx, |ctx| {
            self.capture_begin(mem_pool_id);
            let ret = f(ctx);
            self.capture_end();
            ret
        })
    }
}

#[cfg(test)]
mod tests {

    use crate::{CudaStream, Device, Kind, Tensor};

    use super::*;
    #[test]
    fn test_cuda_graph() {
        // Placeholder input used for capture
        let device = Device::Cuda(0);
        let mut static_input = Tensor::empty(&[5], (Kind::Float, device));
        let mut static_output = Tensor::empty(&[5], (Kind::Float, device));

        // Warmup before capture
        let s = CudaStream::new(device, false);
        let current_stream = CudaStream::get_current_cuda_stream(device).unwrap();

        s.wait_stream(&current_stream);
        s.with_stream((&static_input, &mut static_output), |(input, output)| {
            let v = input * 2;
            output.copy_(&v);
        });
        current_stream.wait_stream(&s);

        // Captures the graph
        // To allow capture, automatically sets a side stream as the current
        // stream in the context
        let graph = CudaGrpah::new();
        // graph.enable_debug_mode();
        graph.with_graph(
            Default::default(),
            &s,
            (&static_input, &mut static_output),
            move |(input, output)| {
                let v = input * 2;
                output.copy_(&v);
            },
        );
        current_stream.wait_stream(&s);

        let input = Tensor::full(&[5], 3, (Kind::Float, device));
        static_input.copy_(&input);
        graph.replay();
        assert_eq!(f32::from(static_output.sum(Kind::Float)), 30.);

        let input = Tensor::full(&[5], 4, (Kind::Float, device));
        static_input.copy_(&input);
        graph.replay();
        assert_eq!(f32::from(static_output.sum(Kind::Float)), 40.);

        // test othre functions
        let _pool_id = graph.get_pool_id();
        // graph.debug_dump("/tmp/_test_graph.dump");
    }
}

use torch_sys::cuda_stream_ffi::{self, C_cuda_event};

use crate::CudaStream;

pub struct CudaEvent {
    pub(crate) c_cuda_event: *mut C_cuda_event,
}
impl CudaEvent {
    pub fn new() -> Self {
        Self::new_with_config(&Default::default())
    }
    pub fn new_with_config(config: &CudaEventConfig) -> Self {
        let flag = config.to_c_flag();
        let c_cuda_event = unsafe { cuda_stream_ffi::at_cuda_event_new(flag) };
        Self { c_cuda_event }
    }
    pub fn record(&self) {
        unsafe {
            cuda_stream_ffi::at_cuda_event_record(self.c_cuda_event);
        }
    }
    pub fn record_stream(&self, stream: &CudaStream) {
        unsafe {
            cuda_stream_ffi::at_cuda_event_record_stream(self.c_cuda_event, stream.c_cuda_stream);
        }
    }
    pub fn record_stream_once(&self, stream: &CudaStream) {
        unsafe {
            cuda_stream_ffi::at_cuda_event_record_stream_once(
                self.c_cuda_event,
                stream.c_cuda_stream,
            );
        }
    }
    pub fn synchronize(&self) {
        unsafe {
            cuda_stream_ffi::at_cuda_event_synchronize(self.c_cuda_event);
        }
    }
    pub fn elapsed_time(&self, other: &Self) -> f64 {
        unsafe {
            cuda_stream_ffi::at_cuda_event_elapsed_time(self.c_cuda_event, other.c_cuda_event)
        }
    }
    pub fn query(&self) -> bool {
        unsafe { cuda_stream_ffi::at_cuda_event_query(self.c_cuda_event) }
    }
    pub fn wait(&self, stream: &CudaStream) {
        unsafe {
            cuda_stream_ffi::at_cuda_event_block(self.c_cuda_event, stream.c_cuda_stream);
        }
    }
}
impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            cuda_stream_ffi::at_cuda_event_free(self.c_cuda_event);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CudaEventConfig {
    pub blocking_sync: bool,
    pub enable_timing: bool,
    pub interprocess: bool,
}

impl Default for CudaEventConfig {
    fn default() -> Self {
        Self { blocking_sync: false, enable_timing: false, interprocess: false }
    }
}
impl CudaEventConfig {
    pub(crate) fn to_c_flag(&self) -> i32 {
        let mut flag = 0;
        if self.blocking_sync {
            flag |= CudaEventFlag::BlockingSync as i32;
        }
        if !self.enable_timing {
            flag |= CudaEventFlag::DisableTiming as i32;
        }
        if self.interprocess {
            flag |= CudaEventFlag::InterProcess as i32;
        }
        flag
    }
}

pub enum CudaEventFlag {
    BlockingSync = 1,
    DisableTiming = 2,
    InterProcess = 4,
}

#[cfg(test)]
mod tests {
    use crate::{Device, Kind, Tensor};

    use super::*;

    #[test]
    fn test_cuda_event() {
        let device = Device::Cuda(0);
        let start_event = CudaEvent::new_with_config(&CudaEventConfig {
            enable_timing: true,
            ..Default::default()
        });
        let end_event = CudaEvent::new_with_config(&CudaEventConfig {
            enable_timing: true,
            ..Default::default()
        });
        start_event.record();

        // todo something
        let _v = Tensor::arange(1000, (Kind::Float, device));

        end_event.record();
        let stream = CudaStream::get_default_stream(device);
        start_event.wait(&stream);
        start_event.synchronize();
        end_event.wait(&stream);
        end_event.synchronize();

        let elapsed_time = end_event.elapsed_time(&start_event);
        println!("elapsed_time: {}", elapsed_time);
    }
}

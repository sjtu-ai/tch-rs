use torch_sys::cuda_stream_ffi;
pub use torch_sys::cuda_stream_ffi::memory::*;

use crate::Device;

pub fn get_device_memoty_stats(device: Device) -> DeviceStats {
    let Device::Cuda(device_index) = device else {
        panic!("Cannot get memory stats on a CPU device");
    };
    let mut stats = DeviceStats::default();
    unsafe {
        cuda_stream_ffi::at_cuda_memory_stats(device_index as u8, &mut stats);
    }
    stats
}
pub fn empty_cuda_cache() {
    unsafe {
        cuda_stream_ffi::at_cuda_memory_empty_cache();
    }
}

#[cfg(test)]
mod tests {
    use crate::{Cuda, Kind, Tensor};

    use super::*;

    #[test]
    fn test_get_device_memory_stats() {
        if Cuda::device_count() == 0 {
            println!("No CUDA device found, skipping test");
            return;
        }
        let device = Device::Cuda(2);
        let _v = Tensor::arange(10, (Kind::Float, device));
        let stats = get_device_memoty_stats(device);
        println!("{:#?}", stats);
    }

    #[test]
    fn test_empty_cuda_cache() {
        if Cuda::device_count() == 0 {
            println!("No CUDA device found, skipping test");
            return;
        }
        empty_cuda_cache();
    }
}

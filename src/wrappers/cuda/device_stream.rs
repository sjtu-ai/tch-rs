use torch_sys::cuda_stream_ffi::{self, C_device_stream};

use crate::{wrappers::device::DeviceType, Device};

#[must_use]
pub struct DeviceStream {
    pub(super) c_device_stream: *mut C_device_stream,
}
// impl !Send for CudaStream {}
// impl !Sync for CudaStream {}
impl Drop for DeviceStream {
    fn drop(&mut self) {
        unsafe {
            cuda_stream_ffi::at_device_stream_free(self.c_device_stream);
        }
    }
}
impl DeviceStream {
    pub fn new<T: Into<Device>>(device: T, stream_id: Option<i64>) -> Self {
        let device: Device = device.into();
        let Device::Cuda(device_index)= device else{
            panic!("Cannot create a CUDA stream on a CPU device");
        };
        let stream_id = stream_id.unwrap_or_default();
        let c_cuda_stream = unsafe {
            cuda_stream_ffi::at_device_stream_new(
                stream_id,
                device_index as u8,
                DeviceType::Cuda as u8,
            )
        };
        Self { c_device_stream: c_cuda_stream }
    }
}

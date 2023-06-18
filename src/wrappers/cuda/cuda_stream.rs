use torch_sys::cuda_stream_ffi::{self, C_cuda_stream};

use crate::{CudaEvent, Device};

#[must_use]
pub struct CudaStream {
    pub(super) c_cuda_stream: *mut C_cuda_stream,
}
impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            cuda_stream_ffi::at_cuda_stream_free(self.c_cuda_stream);
        }
    }
}

impl CudaStream {
    pub fn new(device: Device, is_high_priority: bool) -> Self {
        let Device::Cuda(device_index) = device else {
            panic!("Cannot create a CUDA stream on a CPU device");
        };
        let c_cuda_stream =
            unsafe { cuda_stream_ffi::at_cuda_stream_new(is_high_priority, device_index as u8) };
        Self { c_cuda_stream }
    }

    pub fn id(&self) -> i64 {
        unsafe { cuda_stream_ffi::at_cuda_stream_get_id(self.c_cuda_stream) }
    }

    pub fn set_current(&self) {
        unsafe {
            cuda_stream_ffi::at_cuda_stream_set_current(self.c_cuda_stream);
        }
    }
    pub fn with_stream<CTX, T>(&self, ctx: CTX, f: impl FnOnce(CTX) -> T) -> T {
        let prev_stream = Self::get_current_cuda_stream(self.get_device());
        self.set_current();
        let result = f(ctx);
        if let Some(prev_stream) = prev_stream {
            prev_stream.set_current();
        }
        result
    }
    pub fn get_device(&self) -> Device {
        let index = unsafe { cuda_stream_ffi::at_cuda_stream_get_device_index(self.c_cuda_stream) };
        Device::Cuda(index as usize)
    }
    pub fn synchronize(&self) {
        unsafe {
            cuda_stream_ffi::at_cuda_stream_synchronize(self.c_cuda_stream);
        }
    }
    pub fn record_event(&self, event: &CudaEvent) {
        event.record_stream(&self)
    }
    pub fn wait_event(&self, event: &CudaEvent) {
        event.wait(&self)
    }
    pub fn wait_stream(&self, stream: &CudaStream) {
        let event = CudaEvent::new();
        stream.record_event(&event);
        self.wait_event(&event);
    }
}
impl CudaStream {
    pub fn get_current_cuda_stream(device: Device) -> Option<Self> {
        let Device::Cuda(device_index) = device else {
            return None;
        };
        let c_cuda_stream =
            unsafe { cuda_stream_ffi::at_cuda_stream_get_current(device_index as u8) };
        let stream = Self { c_cuda_stream };
        Some(stream)
    }
    pub fn get_default_stream(device: Device) -> Self {
        let Device::Cuda(device_index) = device else {
            panic!("Cannot create a CUDA stream on a CPU device");
        };
        let c_cuda_stream =
            unsafe { cuda_stream_ffi::at_cuda_stream_get_default(device_index as u8) };
        Self { c_cuda_stream }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cuda, Device, Kind, Tensor};

    #[test]
    fn test_cuda_stream() {
        if Cuda::device_count() == 0 {
            println!("No CUDA device found, skipping test");
            return;
        }
        let device = Device::Cuda(0);
        let stream = CudaStream::new(device, false);
        assert_eq!(stream.get_device(), device);
        assert_eq!(stream.id(), 1);
    }

    #[test]
    fn test_cuda_stream_current() {
        if Cuda::device_count() == 0 {
            println!("No CUDA device found, skipping test");
            return;
        }
        let device = Device::Cuda(1);
        let stream = CudaStream::get_current_cuda_stream(device);
        assert_eq!(stream.is_some(), true);
        let stream = stream.unwrap();
        assert_eq!(stream.get_device(), device);
    }
    #[test]
    fn test_cuda_stream_run_with() {
        if Cuda::device_count() == 0 {
            println!("No CUDA device found, skipping test");
            return;
        }
        let device = Device::Cuda(1);
        let stream = CudaStream::get_current_cuda_stream(device);
        assert_eq!(stream.is_some(), true);
        let stream = stream.unwrap();
        assert_eq!(stream.get_device(), device);
        let result = stream.with_stream((), |()| {
            let stream = CudaStream::get_current_cuda_stream(device);
            assert_eq!(stream.is_some(), true);
            let stream = stream.unwrap();
            assert_eq!(stream.get_device(), device);
            assert_eq!(stream.id(), 0);
            42
        });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_multi_cuda_stream_id() {
        if Cuda::device_count() == 0 {
            println!("No CUDA device found, skipping test");
            return;
        }
        let device = Device::Cuda(1);
        let stream1 = CudaStream::new(device, false);
        let stream2 = CudaStream::new(device, false);
        assert_ne!(stream1.id(), stream2.id());

        let default_stream = CudaStream::get_default_stream(device);
        assert_eq!(default_stream.id(), 0);
    }

    #[test]
    fn test_sum_in_stream() {
        if Cuda::device_count() == 0 {
            println!("No CUDA device found, skipping test");
            return;
        }
        let device = Device::Cuda(1);
        let a = Tensor::eye(100, (Kind::Float, device));
        let stream = CudaStream::new(device, false);

        let b = stream.with_stream((), |_| {
            let b = a.sum(Kind::Float);
            assert_eq!(f32::try_from(&b).unwrap(), 100.0);
            b
        });

        stream.synchronize();
        assert_eq!(f32::try_from(&b).unwrap(), 100.0);
    }

    #[test]
    fn test_wait_stream() {
        if Cuda::device_count() == 0 {
            println!("No CUDA device found, skipping test");
            return;
        }
        let device = Device::Cuda(1);
        let stream1 = CudaStream::new(device, false);
        let stream2 = CudaStream::new(device, false);
        let sum = stream2.with_stream((), |_| {
            let a = Tensor::eye(100, (Kind::Float, device));
            let b = Tensor::eye(100, (Kind::Float, device));
            let c = Tensor::eye(100, (Kind::Float, device));
            let d = Tensor::eye(100, (Kind::Float, device));
            let e = Tensor::eye(100, (Kind::Float, device));
            let f = Tensor::eye(100, (Kind::Float, device));
            let g = Tensor::eye(100, (Kind::Float, device));
            let h = Tensor::eye(100, (Kind::Float, device));
            let i = Tensor::eye(100, (Kind::Float, device));
            let j = Tensor::eye(100, (Kind::Float, device));
            let k = Tensor::eye(100, (Kind::Float, device));
            let l = Tensor::eye(100, (Kind::Float, device));
            let m = Tensor::eye(100, (Kind::Float, device));
            let n = Tensor::eye(100, (Kind::Float, device));
            let o = Tensor::eye(100, (Kind::Float, device));
            let p = Tensor::eye(100, (Kind::Float, device));
            let q = Tensor::eye(100, (Kind::Float, device));
            let r = Tensor::eye(100, (Kind::Float, device));
            let s = Tensor::eye(100, (Kind::Float, device));
            let t = Tensor::eye(100, (Kind::Float, device));
            let u = Tensor::eye(100, (Kind::Float, device));
            let v = Tensor::eye(100, (Kind::Float, device));
            let w = Tensor::eye(100, (Kind::Float, device));
            let x = Tensor::eye(100, (Kind::Float, device));
            let y = Tensor::eye(100, (Kind::Float, device));
            let z = Tensor::eye(100, (Kind::Float, device));
            let aa = Tensor::eye(100, (Kind::Float, device));

            #[rustfmt::skip]
            a + b + c + d + e + f + g //.
                + h + i + j + k + l + m + n // .
                + o + p + q + r + s + t + u //.
                + v + w + x + y + z + aa
        });
        stream1.wait_stream(&stream2);
        let sum = stream1.with_stream((), |_| sum.sum(Kind::Float));
        stream1.synchronize();
        assert_eq!(f32::try_from(&sum).unwrap(), 2700.0);
    }
}

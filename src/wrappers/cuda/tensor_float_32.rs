use torch_sys::cuda_stream_ffi;

// https://pytorch.org/docs/stable/notes/cuda.html
// 是否允许使用 TensorFloat32 (TF32) 张量核心，
// 自 Ampere 以来在新的 NVIDIA GPU 上可用，在内部计算
// matmul（矩阵乘法和批量矩阵乘法）和卷积。
pub struct TensorFloat32;
impl TensorFloat32 {
    pub fn cudnn_set_allow_tf32(allow_tf32: bool) {
        unsafe {
            cuda_stream_ffi::at_cudnn_set_allow_tf32(allow_tf32);
        }
    }
    pub fn cudnn_get_allow_tf32() -> bool {
        unsafe { cuda_stream_ffi::at_cudnn_get_allow_tf32() }
    }
    pub fn cuda_matmul_set_allow_tf32(allow_tf32: bool) {
        unsafe {
            cuda_stream_ffi::at_cuda_matmul_set_allow_tf32(allow_tf32);
        }
    }
    pub fn cuda_matmul_get_allow_tf32() -> bool {
        unsafe { cuda_stream_ffi::at_cuda_matmul_get_allow_tf32() }
    }
}

#[cfg(test)]
mod test_tensor_float_32 {
    use crate::Cuda;

    use super::*;
    #[test]
    fn test_tensor_float_32() {
        if !Cuda::is_available() {
            println!("No CUDA device found, skipping test");
            return;
        }

        TensorFloat32::cudnn_set_allow_tf32(true);
        assert_eq!(TensorFloat32::cudnn_get_allow_tf32(), true);

        TensorFloat32::cudnn_set_allow_tf32(false);
        assert_eq!(TensorFloat32::cudnn_get_allow_tf32(), false);

        TensorFloat32::cuda_matmul_set_allow_tf32(true);
        assert_eq!(TensorFloat32::cuda_matmul_get_allow_tf32(), true);

        TensorFloat32::cuda_matmul_set_allow_tf32(false);
        assert_eq!(TensorFloat32::cuda_matmul_get_allow_tf32(), false);
    }
}

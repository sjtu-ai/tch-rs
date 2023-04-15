use torch_sys::cuda_stream_ffi;

pub struct PrecisionReduction;
impl PrecisionReduction {
    pub fn new() -> Self {
        Self {}
    }
    pub fn set_bf16_reduced_precision(allow_bf16: bool) {
        unsafe {
            cuda_stream_ffi::at_set_allow_bf16_reduced_precision(allow_bf16);
        }
    }
    pub fn get_bf16_reduced_precision() -> bool {
        unsafe { cuda_stream_ffi::at_get_allow_bf16_reduced_precision() }
    }
    pub fn set_fp16_reduced_precision(allow_fp16: bool) {
        unsafe {
            cuda_stream_ffi::at_set_allow_fp16_reduced_precision(allow_fp16);
        }
    }
    pub fn get_fp16_reduced_precision() -> bool {
        unsafe { cuda_stream_ffi::at_get_allow_fp16_reduced_precision() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore]
    fn test_precision_reduction() {
        PrecisionReduction::set_bf16_reduced_precision(true);
        assert_eq!(PrecisionReduction::get_bf16_reduced_precision(), true);
        PrecisionReduction::set_fp16_reduced_precision(true);
        assert_eq!(PrecisionReduction::get_fp16_reduced_precision(), true);

        PrecisionReduction::set_bf16_reduced_precision(false);
        assert_eq!(PrecisionReduction::get_bf16_reduced_precision(), false);
        PrecisionReduction::set_fp16_reduced_precision(false);
        assert_eq!(PrecisionReduction::get_fp16_reduced_precision(), false);
    }
}

use std::{ffi::c_int, slice};

use libc::c_void;

use crate::{at_requires_grad, get_and_reset_last_err, C_tensor};

#[repr(C)]
struct CallbackTarget {
    forward: Option<Box<dyn FnOnce() -> Vec<*mut C_tensor>>>,
    backward: Option<Box<dyn FnOnce(&[*mut C_tensor]) -> Vec<*mut C_tensor>>>,
}

#[repr(C)]
struct VecArray {
    tensors: *mut *mut C_tensor,
    len: i64,
    tensor_raw: Box<Vec<*mut C_tensor>>,
}

fn get_tensors_from_ptr(
    c_tensors: *mut *mut C_tensor,
    free_pointer: bool,
    max_size: usize,
) -> Vec<*mut C_tensor> {
    let mut r__ = vec![];
    for i in 0..max_size {
        let c__ = unsafe { *c_tensors.add(i) };
        if c__.is_null() {
            break;
        }
        r__.push(c__);
    }
    if free_pointer {
        unsafe { libc::free(c_tensors as *mut libc::c_void) }
    }
    r__
}

fn box_tensor_array(tensors: Vec<*mut C_tensor>) -> *mut VecArray {
    let mut tensors = Box::new(tensors);
    let r =
        VecArray { tensors: tensors.as_mut_ptr(), len: tensors.len() as i64, tensor_raw: tensors };
    let r = Box::new(r);
    Box::into_raw(r)
}

#[no_mangle]
extern "C" fn rust_custom_function_forward_callback(target: *mut CallbackTarget) -> *mut VecArray {
    unsafe {
        let tensors = ((*target).forward.take().unwrap())();
        box_tensor_array(tensors)
    }
}

#[no_mangle]
extern "C" fn rust_custom_function_backward_callback(
    target: *mut c_void,
    grad_output: *mut *mut C_tensor,
    size: u32,
) -> *mut VecArray {
    unsafe {
        let tensors = slice::from_raw_parts(grad_output, size as usize);
        let mut target = Box::from_raw(target as *mut CallbackTarget);
        let callback = target.backward.take().unwrap();
        let tensors = callback(tensors);
        box_tensor_array(tensors)
    }
}

#[no_mangle]
extern "C" fn rust_custom_function_destroy_vec_array(struct_instance: *mut VecArray) {
    unsafe {
        let _ = Box::from_raw(struct_instance);
    }
}

extern "C" {
    #[allow(improper_ctypes)]
    fn invoke_custom_function_from_rust_1(
        target: *mut CallbackTarget,
        input: *mut C_tensor,
    ) -> *mut *mut C_tensor;
    #[allow(improper_ctypes)]
    fn invoke_custom_function_from_rust_2(
        target: *mut CallbackTarget,
        a: *mut C_tensor,
        b: *mut C_tensor,
    ) -> *mut *mut C_tensor;
    #[allow(improper_ctypes)]
    fn invoke_custom_function_from_rust_3(
        target: *mut CallbackTarget,
        a: *mut C_tensor,
        b: *mut C_tensor,
        c: *mut C_tensor,
    ) -> *mut *mut C_tensor;
    #[allow(improper_ctypes)]
    fn invoke_custom_function_from_rust_4(
        target: *mut CallbackTarget,
        a: *mut C_tensor,
        b: *mut C_tensor,
        c: *mut C_tensor,
        d: *mut C_tensor,
    ) -> *mut *mut C_tensor;
    #[allow(improper_ctypes)]
    fn invoke_custom_function_from_rust_5(
        target: *mut CallbackTarget,
        a: *mut C_tensor,
        b: *mut C_tensor,
        c: *mut C_tensor,
        d: *mut C_tensor,
        e: *mut C_tensor,
    ) -> *mut *mut C_tensor;

    #[allow(improper_ctypes)]
    fn invoke_custom_function_from_rust_1_void(
        target: *mut CallbackTarget,
        input: *mut C_tensor,
    ) -> *mut *mut C_tensor;
    #[allow(improper_ctypes)]
    fn invoke_custom_function_from_rust_2_void(
        target: *mut CallbackTarget,
        a: *mut C_tensor,
        b: *mut C_tensor,
    ) -> *mut *mut C_tensor;
    #[allow(improper_ctypes)]
    fn invoke_custom_function_from_rust_3_void(
        target: *mut CallbackTarget,
        a: *mut C_tensor,
        b: *mut C_tensor,
        c: *mut C_tensor,
    ) -> *mut *mut C_tensor;
    #[allow(improper_ctypes)]
    fn invoke_custom_function_from_rust_4_void(
        target: *mut CallbackTarget,
        a: *mut C_tensor,
        b: *mut C_tensor,
        c: *mut C_tensor,
        d: *mut C_tensor,
    ) -> *mut *mut C_tensor;
    #[allow(improper_ctypes)]
    fn invoke_custom_function_from_rust_5_void(
        target: *mut CallbackTarget,
        a: *mut C_tensor,
        b: *mut C_tensor,
        c: *mut C_tensor,
        d: *mut C_tensor,
        e: *mut C_tensor,
    ) -> *mut *mut C_tensor;

    pub fn is_grad_enabled() -> bool;

    pub fn run_backward_batch(
        tensors: *const *mut C_tensor,
        n_tensors: c_int,
        grad_tensors: *const *mut C_tensor,
        n_grad_tensors: c_int,
        inputs: *const *mut C_tensor,
        n_inputs: c_int,
        keep_graph: c_int,
        create_graph: c_int,
    );
}

pub fn custom_function(
    input: &[*mut C_tensor],
    forward: impl FnOnce() -> Vec<*mut C_tensor> + 'static,
    backward: impl FnOnce(&[*mut C_tensor]) -> Vec<*mut C_tensor> + 'static,
) -> Vec<*mut C_tensor> {
    unsafe {
        let all_no_grad = !is_grad_enabled() || input.iter().all(|t| at_requires_grad(*t) == 0);
        get_and_reset_last_err();
        if all_no_grad {
            return forward();
        }
    }

    let target =
        CallbackTarget { forward: Some(Box::new(forward)), backward: Some(Box::new(backward)) };
    let target = Box::new(target);
    let target = Box::into_raw(target);

    let result = unsafe {
        match input.len() {
            1 => invoke_custom_function_from_rust_1(target, input[0]),
            2 => invoke_custom_function_from_rust_2(target, input[0], input[1]),
            3 => invoke_custom_function_from_rust_3(target, input[0], input[1], input[2]),
            4 => invoke_custom_function_from_rust_4(target, input[0], input[1], input[2], input[3]),
            5 => invoke_custom_function_from_rust_5(
                target, input[0], input[1], input[2], input[3], input[4],
            ),
            _ => unimplemented!(),
        }
    };
    get_tensors_from_ptr(result, true, u16::MAX as usize)
}

pub fn custom_function_void(
    input: &[*mut C_tensor],
    forward: impl FnOnce() -> Vec<*mut C_tensor> + 'static,
    backward: impl FnOnce(&[*mut C_tensor]) -> Vec<*mut C_tensor> + 'static,
) {
    unsafe {
        let all_no_grad = !is_grad_enabled() || input.iter().all(|t| at_requires_grad(*t) == 0);
        get_and_reset_last_err();
        if all_no_grad {
            #[cfg(test)]
            println!("no grad, just forward");
            forward();
            return;
        }
    }

    let target = {
        let target =
            CallbackTarget { forward: Some(Box::new(forward)), backward: Some(Box::new(backward)) };
        let target = Box::new(target);
        Box::into_raw(target)
    };

    let _ = unsafe {
        match input.len() {
            1 => invoke_custom_function_from_rust_1_void(target, input[0]),
            2 => invoke_custom_function_from_rust_2_void(target, input[0], input[1]),
            3 => invoke_custom_function_from_rust_3_void(target, input[0], input[1], input[2]),
            4 => invoke_custom_function_from_rust_4_void(
                target, input[0], input[1], input[2], input[3],
            ),
            5 => invoke_custom_function_from_rust_5_void(
                target, input[0], input[1], input[2], input[3], input[4],
            ),
            _ => unimplemented!(),
        }
    };
}

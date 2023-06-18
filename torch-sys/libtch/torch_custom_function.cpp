#include "torch_custom_function.h"

extern thread_local char* torch_last_err;
extern std::vector<torch::Tensor> of_carray_tensor(torch::Tensor** vs, int len);

typedef struct VecArrayDef {
    tensor* tensors;
    int64_t len;
    void* raw;

    variable_list to_list() {
        variable_list out__;
        for (int i = 0; i < len; ++i) {
            auto tensor = tensors[i];
            out__.push_back(*tensor);
        }
        return out__;
    };
}* VecArray;

struct RustCallbackContext : torch::CustomClassHolder {
    void* callback_target;
    RustCallbackContext() : callback_target(nullptr) {}
    RustCallbackContext(void* callback_target) : callback_target(callback_target) {}
};
TORCH_LIBRARY(rust_custom_function, m) {
    m.class_<RustCallbackContext>("RustCallbackContext").def(torch::init<>());
}

class RustCallbackFunction : public Function<RustCallbackFunction> {
public:
    static variable_list forward(AutogradContext* ctx, void* callback_target, ...) {
        try {
            torch::IValue rust_context = torch::make_custom_class<RustCallbackContext>(callback_target);
            ctx->saved_data["context"] = rust_context.toCustomClass<RustCallbackContext>();

            VecArray fw = rust_custom_function_forward_callback(callback_target);

            auto output = fw->to_list();
            rust_custom_function_destroy_vec_array(fw);

            return output;
        } catch (std::exception& e) {
            torch_last_err = strdup(e.what());
            std::cerr << "error forward : " << torch_last_err << std::endl;
            throw e;
        }
    }

    static variable_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
        try {
            auto rust_context = ctx->saved_data["context"].toCustomClass<RustCallbackContext>();

            auto sz = grad_outputs.size();
            Tensor** out__ = (Tensor**)malloc((sz + 1) * sizeof(Tensor*));
            for (int i = 0; i < sz; ++i) {
                out__[i] = &grad_outputs.at(i);
            }
            out__[sz] = nullptr;

            VecArray grad = rust_custom_function_backward_callback(rust_context->callback_target, out__, sz);

            free(out__);

            auto output = grad->to_list();
            rust_custom_function_destroy_vec_array(grad);

            output.insert(output.begin(), Tensor());
            return output;
        } catch (std::exception& e) {
            torch_last_err = strdup(e.what());
            std::cerr << "error backward : " << torch_last_err << std::endl;
            throw e;
        }
    }
};

tensor* copy_result_to_pointer_list(variable_list list) {
    auto sz = list.size();
    tensor* out__ = (Tensor**)malloc((sz + 1) * sizeof(Tensor*));
    for (int i = 0; i < sz; ++i) {
        auto item = list[i];
        out__[i] = new Tensor(item);
    }
    out__[sz] = nullptr;
    return out__;
}

tensor* invoke_custom_function_from_rust(void* callback_target) {
    // auto result = RustCallbackFunction::apply(callback_target, args...);
    auto result = RustCallbackFunction::apply(callback_target);
    // std::va_list args;
    // va_start(args, callback_target);
    // auto result = RustCallbackFunction::apply(callback_target, va_arg(args, Tensor));
    // va_end(args);

    return copy_result_to_pointer_list(result);
};

tensor* invoke_custom_function_from_rust_1(void* callback_target, Tensor* a) {
    auto result = RustCallbackFunction::apply(callback_target, *a);
    return copy_result_to_pointer_list(result);
}

tensor* invoke_custom_function_from_rust_2(void* callback_target, Tensor* a, Tensor* b) {
    auto result = RustCallbackFunction::apply(callback_target, *a, *b);
    return copy_result_to_pointer_list(result);
}

tensor* invoke_custom_function_from_rust_3(void* callback_target, Tensor* a, Tensor* b, Tensor* c) {
    auto result = RustCallbackFunction::apply(callback_target, *a, *b, *c);
    return copy_result_to_pointer_list(result);
}

tensor* invoke_custom_function_from_rust_4(void* callback_target, Tensor* a, Tensor* b, Tensor* c, Tensor* d) {
    auto result = RustCallbackFunction::apply(callback_target, *a, *b, *c, *d);
    return copy_result_to_pointer_list(result);
}

tensor* invoke_custom_function_from_rust_5(void* callback_target, Tensor* a, Tensor* b, Tensor* c, Tensor* d,
                                           Tensor* e) {
    auto result = RustCallbackFunction::apply(callback_target, *a, *b, *c, *d, *e);
    return copy_result_to_pointer_list(result);
}

void invoke_custom_function_from_rust_1_void(void* callback_target, Tensor* a) {
    auto result = RustCallbackFunction::apply(callback_target, *a);
}

void invoke_custom_function_from_rust_2_void(void* callback_target, Tensor* a, Tensor* b) {
    auto result = RustCallbackFunction::apply(callback_target, *a, *b);
}

void invoke_custom_function_from_rust_3_void(void* callback_target, Tensor* a, Tensor* b, Tensor* c) {
    auto result = RustCallbackFunction::apply(callback_target, *a, *b, *c);
}

void invoke_custom_function_from_rust_4_void(void* callback_target, Tensor* a, Tensor* b, Tensor* c, Tensor* d) {
    auto result = RustCallbackFunction::apply(callback_target, *a, *b, *c, *d);
}

void invoke_custom_function_from_rust_5_void(void* callback_target, Tensor* a, Tensor* b, Tensor* c, Tensor* d,
                                             Tensor* e) {
    auto result = RustCallbackFunction::apply(callback_target, *a, *b, *c, *d, *e);
}

bool is_grad_enabled() {
    try {
        return torch::autograd::GradMode::is_enabled();
    } catch (std::exception& e) {
        torch_last_err = strdup(e.what());
        std::cerr << "is grad error: " << torch_last_err << std::endl;
        throw e;
    }
}


void run_backward_batch(tensor* tensors, int n_tensors, tensor* grad_tensors, int n_grad_tensors, tensor* inputs,
                        int n_inputs, int keep_graph, int create_graph) {
    variable_list tensors_list = of_carray_tensor(tensors, n_tensors);
    variable_list grad_tensors_list = of_carray_tensor(grad_tensors, n_grad_tensors);

    for (int i = n_grad_tensors; i < n_tensors; ++i) {
        grad_tensors_list.push_back(torch::ones_like(*tensors[i]));
    }

    variable_list inputs_list = of_carray_tensor(inputs, n_inputs);
    backward(tensors_list, grad_tensors_list, keep_graph, create_graph, inputs_list);
}

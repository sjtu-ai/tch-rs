#include "torch_custom_function.h"

vector<torch::Tensor> of_carray_tensor(torch::Tensor** vs, int len);

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

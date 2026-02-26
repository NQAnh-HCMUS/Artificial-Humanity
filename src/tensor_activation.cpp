#include "../include/activation_functions.h"
#include "../include/tensor.h"

Tensor sigmoidtensor(Tensor x)
{
    // Clone tensor shape
    Tensor result(x.shape());

    // Get raw data pointers
    const double *input_data = x.data();
    double *output_data = result.data();

    // Apply element-wise
    for (size_t i = 0; i < x.size(); ++i)
        output_data[i] = sigmoid(input_data[i]);
    return result;
}

Tensor sigmoid_derivative_tensor(Tensor x)
{
    Tensor result(x.shape());

    const double *src = x.data();
    double *dst = result.data();

    for (size_t i = 0; i < x.size(); ++i)
        dst[i] = sigmoidDerivative(src[i]);
    return result;
}

std::vector<Tensor> sigmoid_tensor_vector(const std::vector<Tensor> &tensors)
{
    std::vector<Tensor> result;
    result.reserve(tensors.size());
    for (const Tensor &t : tensors)
        result.push_back(sigmoidtensor(t));
    return result;
}

std::vector<Tensor> sigmoid_derivative_tensor_vector(const std::vector<Tensor> &tensors)
{
    std::vector<Tensor> result;
    result.reserve(tensors.size());
    for (const Tensor &t : tensors)
        result.push_back(sigmoid_derivative_tensor(t));
    return result;
}

Tensor relutensor(Tensor x)
{
    Tensor result(x.shape());
    const double *src = x.data();
    double *dst = result.data();
    for (size_t i = 0; i < x.size(); ++i)
        dst[i] = relu(src[i]);
    return result;
}
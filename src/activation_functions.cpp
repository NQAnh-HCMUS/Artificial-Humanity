#include "../include/activation_functions.h"
#include "../include/tensor.h"

double sigmoid(double x)
{
    double result = 1 / (1 + exp(-x));
    return result;
}

double sigmoidDerivative(double x)
{
    double result = exp(x) / pow((exp(x) + 1), 2);
    return result;
}

std::vector<double> vector_sigmoid(const std::vector<double> x)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(sigmoid(i));
    return result;
}

std::vector<double> vector_sigmoid_derivative(const std::vector<double> x)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(sigmoidDerivative(i));
    return result;
}

double relu(double x)
{
    double result = (x > 0) ? x : 0;
    return result;
}

std::vector<double> vector_relu(const std::vector<double> &x)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double val : x)
        result.push_back(relu(val));
    return result;
}

double reluDerivative(double x)
{
    double result = (x >= 0) ? 1 : 0;
    return result;
}

std::vector<double> vector_relu_derivative(const std::vector<double> &x)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double val : x)
        result.push_back(reluDerivative(val));
    return result;
}

double apl(double x, const std::vector<double> &a, const std::vector<double> &b)
{
    double result = relu(x);
    int S = a.size();
    for (size_t i = 0; i < S; ++i)
    {
        if (x < b[i]) // max(0, -x + b_i)
            result += a[i] * (b[i] - x);
    }
    return result;
}

double apl_derivative(double x, const std::vector<double> &a, const std::vector<double> &b)
{
    double dx = reluDerivative(x);
    int S = a.size();
    for (size_t i = 0; i < S; ++i)
    {
        if (x < b[i])
            dx -= a[i];
    }
    return dx;
}

std::vector<double> vector_apl(const std::vector<double> &x, const std::vector<double> &a, const std::vector<double> &b)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double val : x)
        result.push_back(apl(val, a, b));
    return result;
}

std::vector<double> vector_apl_derivative(const std::vector<double> &x, const std::vector<double> &a, const std::vector<double> &b)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double val : x)
        result.push_back(apl_derivative(val, a, b));
    return result;
}

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
#include "activation_functions.h"

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

double reluDerivative(double x)
{
    double result = (x >= 0) ? 1 : 0;
    return result;
}

std::vector<double> vector_reLu(const std::vector<double> x)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(relu(i));
    return result;
}

std::vector<double> vector_reLu_derivative(const std::vector<double> x)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(reluDerivative(i));
    return result;
}

double leakyRelu(double x, double alpha)
{
    double result = (x > 0) ? x : alpha * x;
    return result;
}

double leakyRelu_derivative(double x, double alpha)
{
    double result = (x >= 0) ? 1 : alpha;
    return result;
}

std::vector<double> vector_leakyRelu(const std::vector<double> x, double alpha)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(leakyRelu(i, alpha));
    return result;
}

std::vector<double> vector_leakyRelu_derivative(const std::vector<double> x, double alpha)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(leakyRelu_derivative(i, alpha));
    return result;
}

std::vector<double> softmax(std::vector<double> z)
{
    std::vector<double> output;
    double sum = 0.0;
    for (double i : z)
        sum += exp(i);
    for (int j = 0; j < z.size(); j++)
        output.push_back(exp(z[j]) / sum);
    return output;
}

double apl(double x, const std::vector<double> &a, const std::vector<double> &b)
{
    double result = relu(x);
    // int S = a.size();
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (x < b[i]) // max(0, -x + b_i)
            result += a[i] * (b[i] - x);
    }
    return result;
}

double apl_derivative(double x, const std::vector<double> &a, const std::vector<double> &b)
{
    double dx = reluDerivative(x);
    // int S = a.size();
    for (size_t i = 0; i < a.size(); ++i)
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
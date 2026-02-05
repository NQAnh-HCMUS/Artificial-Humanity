#include "activation_functions.h"

double sigmoid(double x)
{
    double result = 1 / (1 + exp(-x));
    // std::cout << "sigmoid(" << x << ") = " << result << std::endl;
    return result;
}

double sigmoidDerivative(double x)
{
    double result = exp(x) / pow((exp(x) + 1), 2);
    // std::cout << "derivative of sigmoid(" << x << ") = " << result << std::endl;
    return result;
}

std::vector<double> vector_sigmoid(const std::vector<double> x)
{
    /**
     * A vectorized version of the sigmoid function.
     * @param x the input vector
     * @return a vector where each element is the sigmoid of the corresponding element in x
     */
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(sigmoid(i));
    // std::cout << "sigmoid vector (size = " << x.size() << ") completed" << std::endl;
    return result;
}

std::vector<double> vector_sigmoid_derivative(const std::vector<double> x)
{
    /**
     * A vectorized version of the derivative of the sigmoid function.
     * @param x the input vector
     * @return a vector where each element is the derivative of the sigmoid of the corresponding element in x
     */
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(sigmoidDerivative(i));
    // std::cout << "derivative sigmoid vector (size = " << x.size() << ") completed" << std::endl;
    return result;
}

double relu(double x)
{
    double result = (x > 0) ? x : 0;
    // std::cout << "Rectified Linear Unit(" << x << ") = " << result << std::endl;
    return result;
}

double reluDerivative(double x)
{
    double result = (x >= 0) ? 1 : 0;
    // std::cout << "derivative of Rectified Linear Unit(" << x << ") = " << result << std::endl;
    return result;
}

std::vector<double> vector_reLu(const std::vector<double> x)
{ /**
   * A vectorized version of the Rectified Linear Unit (ReLU) activation function.
   * @param x the input vector
   * @return a vector where each element is the ReLU of the corresponding element in x
   */
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(relu(i));
    // std::cout << "vector of Rectified Linear Unit(size = " << x.size() << ") completed" << std::endl;
    return result;
}

std::vector<double> vector_reLu_derivative(const std::vector<double> x)
{ /**
   * A vectorized version of the derivative of the Rectified Linear Unit (ReLU) activation function.
   * @param x the input vector
   * @return a vector where each element is the derivative of the ReLU function of the corresponding element in x
   */
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(reluDerivative(i));
    // std::cout << "derivative vector of Rectified Linear Unit(size = " << x.size() << ") completed" << std::endl;
    return result;
}

double leakyRelu(double x, double alpha)
{
    double result = (x > 0) ? x : alpha * x;
    // std::cout << "Leaky Rectified Linear Unit(" << x << ", " << alpha << ") = " << result << std::endl;
    return result;
}

double leakyRelu_derivative(double x, double alpha)
{
    double result = (x >= 0) ? 1 : alpha;
    // std::cout << "derivative of Leaky Rectified Linear Unit(" << x << ", " << alpha << ") = " << result << std::endl;
    return result;
}

std::vector<double> vector_leakyRelu(const std::vector<double> x, double alpha)
{ /**
   * A vectorized version of the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
   * @param x the input vector
   * @param alpha the leak rate, defaults to 0.01
   * @return a vector where each element is the Leaky ReLU of the corresponding element in x
   */
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(leakyRelu(i, alpha));
    std::cout << "vector of Leaky Rectified Linear Unit(size=" << x.size() << ", alpha=" << alpha << ") completed" << std::endl;
    return result;
}

std::vector<double> vector_leakyRelu_derivative(const std::vector<double> x, double alpha)
{ /**
   * A vectorized version of the derivative of the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
   * @param x the input vector
   * @param alpha the leak rate, defaults to 0.01
   * @return a vector where each element is the derivative of the Leaky ReLU function of the corresponding element in x
   */
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(leakyRelu_derivative(i, alpha));
    // std::cout << "derivative vector of Leaky Rectified Linear Unit(size=" << x.size() << ", alpha=" << alpha << ") completed" << std::endl;
    return result;
}

double tanh(double x)
{
    double result = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    // std::cout << "tanh(" << x << ") = " << result << std::endl;
    return result;
}

double tanh_derivative(double x)
{
    double result = 1 - pow(tanh(x), 2);
    // std::cout << "derivative of tanh(" << x << ") = " << result << std::endl;
    return result;
}

std::vector<double> vector_tanh(const std::vector<double> x)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(tanh(i));
    // std::cout << "tanh vector (size=" << x.size() << ") completed" << std::endl;
    return result;
}

std::vector<double> vector_tanh_derivative(const std::vector<double> x)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(tanh_derivative(i));
    // std::cout << "derivative tanh vector (size=" << x.size() << ") completed" << std::endl;
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

double Adaptive_Piecewise_Linear(double x, const std::vector<double> &a, const std::vector<double> &b)
{
    double result = std::max(0.0, x);

    try
    {
        if (a.size() != b.size())
        {
            throw "Input vectors must have the same size!";
        }
        for (size_t s = 0; s < a.size(); ++s)
        {
            result += a[s] * std::max(0.0, -x + b[s]);
        }
        return result;
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }
}

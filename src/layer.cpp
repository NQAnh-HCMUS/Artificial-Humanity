#include "../include/layer.h"

std::vector<double> Sigmoid::forward(const std::vector<double> input_data)
{
    input = input_data;
    output = vector_sigmoid(input);
    return output;
}

std::vector<double> Sigmoid::backward(std::vector<double> error, double learning_rate)
{
    std::vector<double> derivative = vector_sigmoid_derivative(input);
    std::vector<double> gradient_input;
    for (int i = 0; i < derivative.size(); ++i)
    {
        gradient_input.push_back(derivative[i] * error[i]);
    }
    return gradient_input;
}

std::vector<double> Relu::forward(const std::vector<double> input_data)
{
    input = input_data;
    output = vector_relu(input);
    return output;
}

std::vector<double> Relu::backward(std::vector<double> error, double learning_rate)
{
    std::vector<double> derivative = vector_relu_derivative(input);
    std::vector<double> grad_input;
    for (int i = 0; i < derivative.size(); ++i)
    {
        grad_input.push_back(derivative[i] * error[i]);
    }
    return grad_input;
}

APL::APL(int num_hings) : S(num_hings)
{
    a.resize(S, 0.0); // start with zero contribution -> behaves like ReLU
    b.resize(S);
    if (S == 1)
        b[0] = 0.0;
    else
        for (int i = 0; i < S; ++i)
            b[i] = -1.0 + (2.0 * i) / (S - 1); // linearly spaced in [-1, 1]
}

std::vector<double> APL::forward(const std::vector<double> input_data)
{
    input = input_data;
    output = vector_apl(input, a, b);
    return output;
}

std::vector<double> APL::backward(std::vector<double> error, double learning_rate)
{
    // Derivative of activation w.r.t input
    std::vector<double> derivative = vector_apl_derivative(input, a, b);

    // Gradient w.r.t input (dL/dx)
    std::vector<double> gradient_input;
    int input_size = input.size();
    gradient_input.reserve(input_size);
    for (size_t j = 0; j < input_size; ++j)
        gradient_input.push_back(derivative[j] * error[j]);

    // Accumulate gradients for a_i and b_i over all input samples
    grad_a.assign(S, 0.0);
    grad_b.assign(S, 0.0);
    for (size_t j = 0; j < input_size; ++j)
    {
        double x = input[j];
        double err = error[j];
        for (int i = 0; i < S; ++i)
        {
            if (x < b[i]) // only when the hinge is active
            {
                grad_a[i] += err * (b[i] - x); // dL/da_i
                grad_b[i] += err * a[i];       // dL/db_i
            }
        }
    }

    // Update parameters (SGD)
    for (int i = 0; i < S; ++i)
    {
        a[i] -= learning_rate * grad_a[i];
        b[i] -= learning_rate * grad_b[i];
    }

    // Optional: enforce sorted knot positions (improves stability)
    std::sort(b.begin(), b.end());

    return gradient_input;
}

Linear::Linear(int num_in, int num_out)
{
    input_neuron = num_in;
    output_neuron = num_out;
    weights = uniformWeightInitializer(num_out, num_in);
    bias = biasInitailizer(num_out);
}

std::vector<double> Linear::forward(const std::vector<double> input_data)
{
    input = input_data;
    output.clear();
    // #pragma omp parallel for
    for (int i = 0; i < output_neuron; i++)
    {
        output.push_back(dotProduct(weights[i], input) + bias[i]);
    }
    return output;
}

std::vector<double> Linear::backward(std::vector<double> error, double learning_rate)
{
    std::vector<double> input_error;               // dE/dX
    std::vector<std::vector<double>> weight_error; // dE/dW
    std::vector<double> bias_error;                // dE/dB
    std::vector<std::vector<double>> weight_transpose;
    weight_error.clear();
    bias_error.clear();
    input_error.clear();
    weight_transpose.clear();

    weight_transpose = transpose(weights);
    bias_error = error;
    for (int i = 0; i < weight_transpose.size(); i++)
    {
        input_error.push_back(dotProduct(weight_transpose[i], error));
    }
    for (int j = 0; j < error.size(); j++)
    {
        std::vector<double> row;
        for (int i = 0; i < input.size(); i++)
        {
            row.push_back(error[j] * input[i]);
        }
        weight_error.push_back(row);
    }

    std::vector<double> delta_bias = scalarVectorMultiplication(bias_error, learning_rate);
    bias = subtract(bias, delta_bias);
    for (int i = 0; i < weight_error.size(); i++)
    {
        std::vector<double> delta_weight = scalarVectorMultiplication(weight_error[i], learning_rate);
        weights[i] = subtract(weights[i], delta_weight);
    }
    return input_error;
}

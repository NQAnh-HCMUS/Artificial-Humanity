// Layers Implementation
// Then we will move into the very core of constructing a neural network which is the layers. To implement these layer first we need to import the following dependencies.

#include <vector>
#include <iostream>
#include "activation_functions.cpp"
// #include "activation_functions.h"
#include "math_functions.cpp"
// #include "math_functions.h"
// Based class 'Layer'

// First, we shall define the based class for all layer. This class will consists of two public variables, input and output, and two public virtual method, forward and backward.

class Layer
{
public:
    std::vector<double> input;
    std::vector<double> output;
    virtual std::vector<double> forward(const std::vector<double> input_data) = 0;
    virtual std::vector<double> backward(std::vector<double> error, double learning_rate) = 0;
};
// Sigmoid layer

// The Sigmoid class is inherited from the Layer class with two override method for forward and backward, which allows the information to propagate through feed forward process and backpropagation process.

class Sigmoid : public Layer
{
public:
    std::vector<double> forward(const std::vector<double> input_data) override
    {
        input = input_data;
        output = vector_sigmoid(input);
        // std::cout << "Sigmoid::forward(size=" << input.size() << ") completed" << std::endl;
        return output;
    }
    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        std::vector<double> derivative = vector_sigmoid_derivative(input);
        std::vector<double> grad_input;
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        // std::cout << "Sigmoid::backward(size=" << grad_input.size() << ", lr=" << learning_rate << ") completed" << std::endl;
        return grad_input;
    }
};
// ReLU layer

// The Relu class is inherited from the Layer class with two override method for forward and backward, which allows the information to propagate through feed forward process and backpropagation process.

class Relu : public Layer
{
public:
    std::vector<double> forward(const std::vector<double> input_data) override
    {
        input = input_data;
        output = vector_reLu(input);
        return output;
    }
    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        std::vector<double> derivative = vector_reLu_derivative(input);
        std::vector<double> grad_input;
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        // std::cout << "Relu::backward(size=" << grad_input.size() << ", lr=" << learning_rate << ") completed" << std::endl;
        return grad_input;
    }
};
// Leaky ReLU layer

// The LeakyRelu class is inherited from the Layer class with two override method for forward and backward, which allows the information to propagate through feed forward process and backpropagation process.

class LeakyRelu : public Layer
{
public:
    double alpha = 0.01;
    std::vector<double> forward(const std::vector<double> input_data) override
    {
        input = input_data;
        output = vector_leakyRelu(input, alpha);
        // std::cout << "LeakyRelu::forward(size=" << input.size() << ", alpha=" << alpha << ") completed" << std::endl;
        return output;
    }
    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        std::vector<double> derivative = vector_leakyRelu_derivative(input, alpha);
        std::vector<double> grad_input;
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        // std::cout << "LeakyRelu::backward(size=" << grad_input.size() << ", lr=" << learning_rate << ") completed" << std::endl;
        return grad_input;
    }
};
// Tanh layer

// The Tanh class is inherited from the Layer class with two override method for forward and backward, which allows the information to propagate through feed-forward process and backpropagation process.

class Tanh : public Layer
{
public:
    std::vector<double> forward(const std::vector<double> input_data) override
    {
        input = input_data;
        output = vector_tanh(input);
        // std::cout << "Tanh::forward(size=" << input.size() << ") completed" << std::endl;
        return output;
    }
    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        std::vector<double> derivative = vector_tanh_derivative(input);
        std::vector<double> grad_input;
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        // std::cout << "Tanh::backward(size=" << grad_input.size() << ", lr=" << learning_rate << ") completed" << std::endl;
        return grad_input;
    }
};
// Linear layer

// The Linear layer or fully connected layer is also inherited from the Layer class. The Linear class constructor requires the number of input and output neurons to create its instance. Then according to these numbers, its weights and bias will be created.

class Linear : public Layer
{
public:
    int input_neuron;
    int output_neuron;
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;

    Linear(int num_in, int num_out)
    {
        input_neuron = num_in;
        output_neuron = num_out;
        weights = uniformWeightInitializer(num_out, num_in);
        bias = biasInitailizer(num_out);
    }

    std::vector<double> forward(const std::vector<double> input_data) override
    {
        input = input_data;
        output.clear();
        for (int i = 0; i < output_neuron; i++)
        {
            output.push_back(dotProduct(weights[i], input) + bias[i]);
        }
        // std::cout << "Linear::forward(in=" << input_neuron << ", out=" << output_neuron << ") completed" << std::endl;
        return output;
    }
    std::vector<double> backward(std::vector<double> error, double learning_rate) override
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
        // std::cout << "Linear::backward(size=" << input_error.size() << ", lr=" << learning_rate << ") completed" << std::endl;
        return input_error;
    }
};
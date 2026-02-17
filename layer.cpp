#include <vector>
// #include <iostream>
#include "activation_functions.cpp"
#include "math_functions.cpp"

// Base layer class
/**
 * @brief Abstract base class for neural network layers.
 */
class Layer
{
public:
    std::vector<double> input;
    std::vector<double> output;
    /**
     * @brief Compute the layer output for the given input.
     * @param input_data input vector for the layer
     * @return output vector produced by the layer
     */
    virtual std::vector<double> forward(const std::vector<double> input_data) = 0;
    /**
     * @brief Backpropagate error through the layer and apply parameter updates.
     * @param error gradient of the loss with respect to the layer output
     * @param learning_rate scalar learning rate used for parameter updates
     * @return gradient of the loss with respect to the layer input
     */
    virtual std::vector<double> backward(std::vector<double> error, double learning_rate) = 0;
};

/**
 * @brief Sigmoid activation layer.
 *
 * Applies sigmoid function during forward pass and its derivative during backpropagation.
 */
class Sigmoid : public Layer
{
public:
    /**
     * @brief Apply sigmoid to each input element.
     * @param input_data input vector
     * @return vector of sigmoid activations
     */
    std::vector<double> forward(const std::vector<double> input_data) override
    {
        input = input_data;
        output = vector_sigmoid(input);
        return output;
    }
    /**
     * @brief Backpropagate through the sigmoid nonlinearity.
     * @param error gradient at the layer output
     * @param learning_rate unused (no trainable parameters)
     * @return gradient with respect to the layer input
     */
    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        std::vector<double> derivative = vector_sigmoid_derivative(input);
        std::vector<double> gradient_input;
        for (int i = 0; i < derivative.size(); ++i)
        {
            gradient_input.push_back(derivative[i] * error[i]);
        }
        return gradient_input;
    }
};

#define DEFAULT_HINGES 3
/**
 * @brief Adaptive Piecewise Linear (APL) activation layer.
 *
 * Implements a learnable piecewise-linear activation with hinge positions (b_i) and slopes (a_i). Parameters are updated with SGD during backprop.
 */
class APL : public Layer
{
public:
    /** @brief Number of hinges (hyperparameter). */
    int s;
    /** @brief Learnable slopes for each hinge. */
    std::vector<double> a;
    /** @brief Learnable hinge positions (thresholds). */
    std::vector<double> b;

    /**
     * @brief Construct an APL layer and initialize hinge parameters.
     * @param num_hings requested number of hinges (currently unused)
     */
    APL(int num_hings = DEFAULT_HINGES) : s(num_hings)
    {
        a.resize(s, 0.0); // start with zero contribution -> behaves like ReLU
        b.resize(s);
        if (s == 1)
            b[0] = 0.0;
        else
            for (int i = 0; i < s; ++i)
                b[i] = -1.0 + (2.0 * i) / (s - 1); // linearly spaced in [-1, 1]
    }

    /**
     * @brief Apply APL(x) = ReLU(x) + Σ a_i * max(0, b_i - x) to each input element.
     * @param input_data input vector
     * @return vector of APL activations
     */
    std::vector<double> forward(const std::vector<double> input_data) override
    {
        input = input_data;
        output = vector_apl(input, a, b);
        return output;
    }

    /**
     * @brief Backpropagate through APL, compute dL/dx, dL/da_i, dL/db_i and update a_i, b_i and update hinge parameters.
     * @param error gradient at the layer output
     * @param learning_rate learning rate for SGD updates of a and b
     * @return gradient with respect to the layer input
     */
    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        // Derivative of activation w.r.t input
        std::vector<double> derivative = vector_apl_derivative(input, a, b);

        // Gradient w.r.t input (dL/dx)
        std::vector<double> gradient_input;
        gradient_input.reserve(input.size());
        for (size_t j = 0; j < input.size(); ++j)
            gradient_input.push_back(derivative[j] * error[j]);

        // Accumulate gradients for a_i and b_i over all input samples
        std::vector<double> grad_a(s, 0.0);
        std::vector<double> grad_b(s, 0.0);
        for (size_t j = 0; j < input.size(); ++j)
        {
            double x = input[j];
            double err = error[j];
            for (int i = 0; i < s; ++i)
            {
                if (x < b[i]) // only when the hinge is active
                {
                    grad_a[i] += err * (b[i] - x); // dL/da_i
                    grad_b[i] += err * a[i];       // dL/db_i
                }
            }
        }

        // Update parameters (SGD)
        for (int i = 0; i < s; ++i)
        {
            a[i] -= learning_rate * grad_a[i];
            b[i] -= learning_rate * grad_b[i];
        }

        // Optional: enforce sorted knot positions (improves stability)
        // std::sort(b.begin(), b.end());

        return gradient_input;
    }
};

/**
 * @brief Fully connected linear layer.
 *
 * Computes an affine transform: output = W * input + b.
 */
class Linear : public Layer
{
public:
    /** @brief Number of input neurons. */
    int input_neuron;
    /** @brief Number of output neurons. */
    int output_neuron;
    /** @brief Weight matrix with shape [output_neuron][input_neuron]. */
    std::vector<std::vector<double>> weights;
    /** @brief Bias vector with length output_neuron. */
    std::vector<double> bias;

    /**
     * @brief Construct a linear layer with initialized weights and biases.
     * @param num_in number of input neurons
     * @param num_out number of output neurons
     */
    Linear(int num_in, int num_out)
    {
        input_neuron = num_in;
        output_neuron = num_out;
        weights = uniformWeightInitializer(num_out, num_in);
        bias = biasInitailizer(num_out);
    }

    /**
     * @brief Compute the affine transform for the input.
     * @param input_data input vector
     * @return output vector after linear transform
     */
    std::vector<double> forward(const std::vector<double> input_data) override
    {
        input = input_data;
        output.clear();
        for (int i = 0; i < output_neuron; i++)
        {
            output.push_back(dotProduct(weights[i], input) + bias[i]);
        }
        return output;
    }
    /**
     * @brief Backpropagate through the linear layer and update parameters.
     * @param error gradient at the layer output
     * @param learning_rate learning rate for SGD updates
     * @return gradient with respect to the layer input
     */
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
        return input_error;
    }
};
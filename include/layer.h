#ifndef LAYER_H
#define LAYER_H

#include "../src/activation_functions.cpp"
// #include "../include/activation_functions.h"
#include "../src/math_functions.cpp"
#include "../include/tensor.h"

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
    std::vector<double> forward(const std::vector<double> input_data) override;
    /**
     * @brief Backpropagate through the sigmoid nonlinearity.
     * @param error gradient at the layer output
     * @param learning_rate unused (no trainable parameters)
     * @return gradient with respect to the layer input
     */
    std::vector<double> backward(std::vector<double> error, double learning_rate) override;
};

// Define ReLu class inherited from Layer
class ReLu : public Layer
{
public:
    std::vector<double> forward(const std::vector<double> input_data) override;

    std::vector<double> backward(std::vector<double> error, double learning_rate) override;
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
    int S;
    /** @brief Learnable slopes for each hinge. */
    std::vector<double> a;
    /** @brief Learnable hinge positions (thresholds). */
    std::vector<double> b;
    /** @brief Accumulated gradients for slopes (a_i) during backpropagation. */
    std::vector<double> grad_a, grad_b;

    /**
     * @brief Construct an APL layer and initialize hinge parameters.
     * @param num_hings requested number of hinges (currently unused)
     */
    APL(int num_hings = DEFAULT_HINGES);

    /**
     * @brief Apply APL(x) = ReLU(x) + Σ a_i * max(0, b_i - x) to each input element.
     * @param input_data input vector
     * @return vector of APL activations
     */
    std::vector<double> forward(const std::vector<double> input_data) override;

    /**
     * @brief Backpropagate through APL, compute dL/dx, dL/da_i, dL/db_i and update a_i, b_i and update hinge parameters.
     * @param error gradient at the layer output
     * @param learning_rate learning rate for SGD updates of a and b
     * @return gradient with respect to the layer input
     */
    std::vector<double> backward(std::vector<double> error, double learning_rate) override;
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
    Linear(int num_in, int num_out);

    /**
     * @brief Compute the affine transform for the input.
     * @param input_data input vector
     * @return output vector after linear transform
     */
    std::vector<double> forward(const std::vector<double> input_data) override;

    /**
     * @brief Backpropagate through the linear layer and update parameters.
     * @param error gradient at the layer output
     * @param learning_rate learning rate for SGD updates
     * @return gradient with respect to the layer input
     */
    std::vector<double> backward(std::vector<double> error, double learning_rate) override;
};

// Base layer class
/**
 * @brief Abstract base class for neural network layers.
 */
class TensorLayer
{
public:
    Tensor input;
    Tensor output;

    /**
     * @brief Compute the layer output for the given input.
     * @param input_data input tensor for the layer
     * @return output tensor produced by the layer
     */
    virtual Tensor forward(const Tensor &input_data) = 0;

    /**
     * @brief Backpropagate error through the layer and apply parameter updates.
     * @param error gradient of the loss with respect to the layer output
     * @param learning_rate scalar learning rate used for parameter updates
     * @return gradient of the loss with respect to the layer input
     */
    virtual Tensor backward(Tensor error, double learning_rate) = 0;
};

/**
 * @brief Sigmoid activation layer.
 *
 * Applies sigmoid function during forward pass and its derivative during backpropagation.
 */
class TensorSigmoid : public TensorLayer
{
public:
    /**
     * @brief Apply sigmoid to each input element.
     * @param input_data input vector
     * @return vector of sigmoid activations
     */
    Tensor forward(const Tensor &input_data) override;

    /**
     * @brief Backpropagate through the sigmoid nonlinearity.
     * @param error gradient at the layer output
     * @param learning_rate unused (no trainable parameters)
     * @return gradient with respect to the layer input
     */
    Tensor backward(Tensor error, double learning_rate) override;
};

class TensorReLu : public TensorLayer
{
public:
    Tensor forward(const Tensor &input_data) override;

    Tensor backward(Tensor error, double learning_rate) override;
};

#endif // LAYER_H
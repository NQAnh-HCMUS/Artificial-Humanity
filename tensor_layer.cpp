#include <vector>
// #include <iostream>
#include "tensor_activation.cpp"
#include "math_functions.h"
#include "tensor.h"

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
    virtual Tensor backward(Tensor error, Tensor learning_rate) = 0;
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
    Tensor forward(const Tensor &input_data) override
    {
        input = input_data;
        output = sigmoidtensor(input);
        return output;
    }
    /**
     * @brief Backpropagate through the sigmoid nonlinearity.
     * @param error gradient at the layer output
     * @param learning_rate unused (no trainable parameters)
     * @return gradient with respect to the layer input
     */
    Tensor backward(Tensor error, Tensor learning_rate) override
    {
        Tensor derivative = sigmoid_derivative_tensor(input);
        Tensor gradient_input(derivative.shape());

        const float *src1 = derivative.data();
        const float *src2 = error.data();
        float *dst = gradient_input.data();
        for (size_t i = 0; i < derivative.size(); ++i)
        {
            dst[i] = src1[i] * src2[i];
        }
        return gradient_input;
    }
};
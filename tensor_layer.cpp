#include <vector>
// #include <iostream>
#include "activation_functions.cpp"
#include "math_functions.cpp"
#include "tensor.cpp"

// Base layer class
/**
 * @brief Abstract base class for neural network layers.
 */
class Layer
{
public:
    /**
     * @brief Compute the layer output for the given input.
     * @param input input tensor for the layer
     * @return output tensor produced by the layer
     */
    virtual Tensor forward(const Tensor &input) = 0;

    /**
     * @brief Backpropagate error through the layer and apply parameter updates.
     * @param error gradient of the loss with respect to the layer output
     * @param learning_rate scalar learning rate used for parameter updates
     * @return gradient of the loss with respect to the layer input
     */
    virtual Tensor backward(const Tensor &gradient_output, double learning_rate) = 0;
};

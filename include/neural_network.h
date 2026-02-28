#include <vector>
#include <memory>
#include <iostream>
#include "../src/layer.cpp"
// #include "../include/layer.h"
#include "../src/loss_function.cpp"

class neural_network
{
private:
    // neural_network();

public:
    std::vector<std::unique_ptr<Layer>> layers;

    /**
     * @brief Add a layer to the network.
     * @param layer Layer to append.
     */
    void add(std::unique_ptr<Layer> layer);

    /**
     * @brief Run a forward pass to produce a prediction.
     * @param input Input vector.
     * @return Network output after forward propagation.
     */
    std::vector<double> predict(std::vector<double> input);

    /**
     * @brief Forward propagation through all layers.
     * @param input Input vector.
     * @return Output vector after all layers.
     */
    std::vector<double> forward_propagation(const std::vector<double> input);

    /**
     * @brief Backward propagation through all layers.
     * @param error Loss derivative with respect to the network output.
     * @param learning_rate Step size used to update weights.
     */
    void backward_propagation(const std::vector<double> &error, double learning_rate);

    /**
     * @brief Train the network for a fixed number of epochs.
     * @param X Input samples.
     * @param y Target outputs.
     * @param epochs Number of training epochs.
     * @param learning_rate Step size used to update weights.
     */
    void fit(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &y, int epochs, double learning_rate);
};
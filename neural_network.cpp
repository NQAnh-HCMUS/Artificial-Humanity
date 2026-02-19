#include <vector>
#include <memory>
#include <iostream>
#include "layer.cpp"
#include "loss_function.cpp"

/**
 * @brief Simple feed-forward neural network container.
 */
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
    void add(std::unique_ptr<Layer> layer)
    {
        layers.push_back(std::move(layer));
    }

    /**
     * @brief Run a forward pass to produce a prediction.
     * @param input Input vector.
     * @return Network output after forward propagation.
     */
    std::vector<double> predict(std::vector<double> input)
    {
        auto result = forward_propagation(input);
        return result;
    }

    /**
     * @brief Forward propagation through all layers.
     * @param input Input vector.
     * @return Output vector after all layers.
     */
    std::vector<double> forward_propagation(const std::vector<double> input)
    {
        std::vector<double> output = input;
        for (const auto &layer : layers)
        {
            output = layer->forward(output);
        }
        return output;
    }

    /**
     * @brief Backward propagation through all layers.
     * @param error Loss derivative with respect to the network output.
     * @param learning_rate Step size used to update weights.
     */
    void backward_propagation(const std::vector<double> &error, double learning_rate)
    {
        std::vector<double> grad = error;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            grad = (*it)->backward(grad, learning_rate);
        }
    }

    /**
     * @brief Train the network for a fixed number of epochs.
     * @param X Input samples.
     * @param y Target outputs.
     * @param epochs Number of training epochs.
     * @param learning_rate Step size used to update weights.
     */
    void fit(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &y, int epochs, double learning_rate)
    {
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double total_loss = 0.0;
            for (size_t i = 0; i < X.size(); ++i)
            {
                // Forward pass
                std::vector<double> output = forward_propagation(X[i]);

                // Compute loss
                double loss = BCELossFunction(y[i], output);
                total_loss += loss;

                std::vector<double> loss_derivative = BCELossDerivative(y[i], output);
                // Backward pass
                backward_propagation(loss_derivative, learning_rate);
            }

            // Print loss for monitoring
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << total_loss / X.size() << std::endl;
        }
        std::cout << "neural network::fit(epochs=" << epochs << ", lr=" << learning_rate << ") training completed" << std::endl;
    }
};
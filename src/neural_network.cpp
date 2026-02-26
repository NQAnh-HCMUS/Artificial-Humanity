// #include <vector>
// #include <memory>
// #include <iostream>
// #include "../src/layer.cpp"
// #include "../src/loss_function.cpp"
// #include "../src/tensor_layer.cpp"
#include "../include/neural_network.h"

void neural_network::add(std::unique_ptr<Layer> layer)
{
    layers.push_back(std::move(layer));
}

std::vector<double> neural_network::predict(std::vector<double> input)
{
    auto result = forward_propagation(input);
    return result;
}

std::vector<double> neural_network::forward_propagation(const std::vector<double> input)
{
    std::vector<double> output = input;
    for (const auto &layer : layers)
    {
        output = layer->forward(output);
    }
    return output;
}

void neural_network::backward_propagation(const std::vector<double> &error, double learning_rate)
{
    std::vector<double> grad = error;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it)
    {
        grad = (*it)->backward(grad, learning_rate);
    }
}

void neural_network::fit(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &y, int epochs, double learning_rate)
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

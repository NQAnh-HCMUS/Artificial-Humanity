// Neural Network Implementation
// Now, let's implement the most important part of this project, the neural network itself. There are five main method that we need to implement as follow:

// add() - To add each layer into our neural network.
// predict() - To perform feed forward operation which will bestow us with the predicted output.
// forward_propagation() - To perform feed forward operation as well, but this function will be use solely in the training process.
// backward_propagation() - To perform backpropagation in order to update the learnable parameter of our neural network, namely weights and bias.
// fit() - To train our neural network.
// The abstract structure of our neural network (NN class)

#include <vector>
#include <memory>
#include <iostream>
#include "layer.cpp"
#include "loss_function.cpp"

class neural_network
{
public:
    std::vector<std::unique_ptr<Layer>> layers;

    // Add layers dynamically
    void add(Layer *layer)
    {
        layers.emplace_back(layer);
        // std::cout << "NN::add - Layer added. Total layers: " << layers.size() << std::endl;
    }

    // Make prediction using feed forward process
    std::vector<double> predict(std::vector<double> input)
    {
        auto result = forward_propagation(input);
        // std::cout << "NN::predict(size=" << input.size() << ") output size: " << result.size() << std::endl;
        return result;
    }

    // Forward propagation
    std::vector<double> forward_propagation(const std::vector<double> input)
    {
        std::vector<double> output = input;
        for (const auto &layer : layers)
        {
            output = layer->forward(output);
        }
        // std::cout << "NN::forward_propagation completed with " << layers.size() << " layers" << std::endl;
        return output;
    }

    // Backward propagation
    void backward_propagation(const std::vector<double> &error, double learning_rate)
    {
        std::vector<double> grad = error;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            grad = (*it)->backward(grad, learning_rate);
        }
        // std::cout << "NN::backward_propagation completed with lr=" << learning_rate << std::endl;
    }

    // Training function
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
                double loss = BCELoss(y[i], output);
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
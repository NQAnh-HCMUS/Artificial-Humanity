// #include "../include/neural_network.h"
#include "neural_network.cpp"

#define number_of_hinges 3
#define epochs 5000
#define learning_rate 0.01

int main()
{
    // Initialize the neural network
    neural_network neural_network;

    // Add layers to the network
    // neural_network.add(std::make_unique<Linear>(2, number_of_hinges));
    // neural_network.add(std::make_unique<APL>(number_of_hinges));
    // neural_network.add(std::make_unique<Linear>(number_of_hinges, number_of_hinges));
    // neural_network.add(std::make_unique<APL>(number_of_hinges));
    // neural_network.add(std::make_unique<Linear>(number_of_hinges, 1));
    // neural_network.add(std::make_unique<Sigmoid>());
    neural_network.add(std::make_unique<Linear>(2, number_of_hinges));
    neural_network.add(std::make_unique<ReLu>());
    neural_network.add(std::make_unique<Linear>(number_of_hinges, number_of_hinges));
    neural_network.add(std::make_unique<ReLu>());
    neural_network.add(std::make_unique<Linear>(number_of_hinges, 1));
    neural_network.add(std::make_unique<Sigmoid>());

    // Example input data: XOR logic gate
    std::vector<std::vector<double>> X_train = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> y_train = {{0}, {0}, {0}, {1}};

    // Train network
    neural_network.fit(X_train, y_train, epochs, learning_rate);

    // Test network
    const std::vector<std::vector<double>> X_test = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<int> y_test = {0, 0, 0, 1};

    for (size_t i = 0; i < X_test.size(); ++i)
    {
        const std::vector<double> &input = X_test[i];
        const std::vector<double> output_prob = neural_network.predict(input);
        const int output = (output_prob[0] > 0.5) ? 1 : 0;

        std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
        std::cout << "Output Probability: " << output_prob[0] << std::endl;
        std::cout << "Output: " << output << std::endl;
        std::cout << "Expected Output: " << y_test[i] << std::endl;
        std::cout << "----------------------" << std::endl;
    }

    // // Initialize the neural network
    // neural_network neural_network;

    // // Add layers to the network
    // neural_network.add(std::make_unique<Linear>(2, number_of_hinges));
    // neural_network.add(std::make_unique<APL>(number_of_hinges));
    // neural_network.add(std::make_unique<Linear>(number_of_hinges, number_of_hinges));
    // neural_network.add(std::make_unique<APL>(number_of_hinges));
    // neural_network.add(std::make_unique<Linear>(number_of_hinges, 1));
    // neural_network.add(std::make_unique<Sigmoid>());

    // Example input data: XOR logic gate
    std::vector<std::vector<double>> X_train_data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> y_train_data = {{0}, {0}, {0}, {1}};

    std::vector<Tensor> X_train_tensor;
    std::vector<Tensor> y_train_tensor;

    for (size_t i = 0; i < X_train_data.size(); ++i)
    {
        X_train_tensor.emplace_back(std::vector<size_t>{2}, 0.0);
        y_train_tensor.emplace_back(std::vector<size_t>{1}, 0.0);
        for (size_t j = 0; j < 2; ++j)
        {
            X_train_tensor.back()(j) = X_train_data[i][j];
        }
        y_train_tensor.back()(0) = y_train_data[i][0];
    }

    for (size_t i = 0; i < X_train_tensor.size(); ++i)
    {
        std::cout << "X_train_tensor[" << i << "]: ";
        for (size_t j = 0; j < X_train_tensor[i].shape()[0]; ++j)
        {
            std::cout << X_train_tensor[i](j) << " ";
        }
        std::cout << " | y_train_tensor[" << i << "]: " << y_train_tensor[i](0) << std::endl;
    }

    // // Train network
    // neural_network.fit(X_train_tensor, y_train_tensor, epochs, learning_rate);

    // // Test network
    // const std::vector<std::vector<double>> X_test = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    // const std::vector<int> y_test = {0, 0, 0, 1};

    // for (size_t i = 0; i < X_test.size(); ++i)
    // {
    //     const std::vector<double> &input = X_test[i];
    //     const std::vector<double> output_prob = neural_network.predict(input);
    //     const int output = (output_prob[0] > 0.5) ? 1 : 0;

    //     std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    //     std::cout << "Output Probability: " << output_prob[0] << std::endl;
    //     std::cout << "Output: " << output << std::endl;
    //     std::cout << "Expected Output: " << y_test[i] << std::endl;
    //     std::cout << "----------------------" << std::endl;
    // }

    return 0;
}
#include <iostream>
#include <vector>
#include "neural_network.cpp"
// #include "knowledge_logic.cpp"

#define number_of_hinges 3
#define epochs 10000
#define learning_rate 0.01

int main()
{
    // Initialize the neural network
    neural_network neural_network;

    // Add layers to the network
    neural_network.add(new Linear(2, number_of_hinges));
    neural_network.add(new APL(number_of_hinges));
    neural_network.add(new Linear(number_of_hinges, number_of_hinges));
    neural_network.add(new APL(number_of_hinges));
    neural_network.add(new Linear(number_of_hinges, 1));
    neural_network.add(new Sigmoid());

    // Example input data: XOR logic gate
    std::vector<std::vector<double>> X_train = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> y_train = {{0}, {1}, {1}, {0}};

    // Train network
    neural_network.fit(X_train, y_train, epochs, learning_rate);

    // Test network
    const std::vector<std::vector<double>> X_test = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<int> y_test = {0, 1, 1, 0};

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

    return 0;
}
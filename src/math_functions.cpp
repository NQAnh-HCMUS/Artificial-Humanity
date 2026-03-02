#include "../include/math_functions.h"
#include "tensor.cpp"

double randomDouble(double min, double max)
{
    static std::random_device rd;
    static std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

double dotProduct(std::vector<double> &vector1, std::vector<double> &vector2)
{
    double result = 0;
    for (int i = 0; i < vector1.size(); i++)
    {
        result += vector1[i] * vector2[i];
    }
    return result;
}

std::vector<double> scalarVectorMultiplication(std::vector<double> &vector, double scalar)
{
    std::transform(vector.begin(), vector.end(), vector.begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, scalar));
    return vector;
}

std::vector<double> subtract(std::vector<double> &vector1, std::vector<double> &vector2)
{
    std::vector<double> out;
    std::transform(vector1.begin(), vector1.end(), vector2.begin(), std::back_inserter(out), std::minus<double>());
    return out;
}

std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> &matrix)
{
    std::vector<std::vector<double>> trans_vec(matrix[0].size(), std::vector<double>());

    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < matrix[i].size(); j++)
        {
            if (trans_vec[j].size() != matrix.size())
                trans_vec[j].resize(matrix.size());
            trans_vec[j][i] = matrix[i][j];
        }
    }
    return trans_vec;
}

std::vector<std::vector<double>> uniformWeightInitializer(int rows, int cols)
{
    // Initialize weights matrix
    std::vector<std::vector<double>> weights(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            weights[i][j] = randomDouble(-1.0, 1.0);
        }
    }

    return weights;
}

// Tensor uniformTensorInitializer(std::vector<double> rows, std::vector<double> cols)
// {
//     // Random number generation setup
//     std::random_device rd;
//     std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
//     std::uniform_real_distribution<> dis(-1.0, 1.0);

//     // Initialize weights matrix
//     Tensor weights(rows, cols);

//     // for (double i = 0; i < rows; ++i)
//     // {
//     //     for (double j = 0; j < cols; ++j)
//     //     {
//     //         weights[i][j] = dis(gen);
//     //     }
//     // }

//     return weights;
// }

std::vector<double> biasInitailizer(int size)
{
    std::random_device rd;
    std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<double> bias(size);

    for (int i = 0; i < size; ++i)
    {
        bias[i] = randomDouble(-1.0, 1.0);
    }
    return bias;
}

std::vector<std::vector<int>> getExtreme(std::vector<int> values, int deviation)
{
    std::vector<int> drops;
    std::vector<int> spikes;

    int average = 0;
    for (int val : values)
    {
        average += val;
    }
    average = average / values.size();

    for (int val : values)
    {
        if (val < average - deviation)
        {
            drops.push_back(val);
        }
        if (val > average + deviation)
        {
            spikes.push_back(val);
        }
    }

    return {drops, spikes};
}

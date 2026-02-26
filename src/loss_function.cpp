
#include "../include/math_functions.h"
#include "../include/tensor.h"

/**
 * @brief Binary Cross Entropy loss function
 *
 * It quantifies the difference between the predicted probability distribution and the actual binary labels (0 or 1). Here we demonstrate the implementation of the BCE loss function as well as its derivative.
 * @param true_label true labels of the data
 * @param pred_prob predicted probabilities
 * @return binary cross entropy loss
 */
double BCELossFunction(std::vector<double> true_label, std::vector<double> pred_prob)
{
    double sum = 0;
    for (int i = 0; i < pred_prob.size(); i++)
    {
        sum += true_label[i] * log(pred_prob[i]) + (1 - true_label[i]) * log((1 - pred_prob[i]));
    }
    int size = true_label.size();
    double loss = -(1.0 / size) * sum;
    return loss;
}

/**
 * @brief Derivative of Binary Cross Entropy loss function
 * @param true_label true labels of the data
 * @param pred_prob predicted probabilities
 * @return derivative of binary cross entropy loss
 */
std::vector<double> BCELossDerivative(const std::vector<double> &true_label,
                                      const std::vector<double> &pred_prob)
{
    std::vector<double> d(pred_prob.size());
    for (size_t i = 0; i < d.size(); ++i)
    {
        d[i] = (pred_prob[i] - true_label[i]) / (pred_prob[i] * (1 - pred_prob[i]) + 1e-8);
    }
    return d;
}

// double BCELossTensor(const Tensor &true_label, const Tensor &pred_prob)
// {
//     check_shape_match(true_label, pred_prob);

//     double sum = 0.0;
//     // Access the underlying flat data arrays
//     const std::vector<double> true_data = true_label.data();
//     std::vector<double> pred_data = pred_prob.data();

//     for (size_t i = 0; i < true_label.size(); ++i)
//     {
//         sum += true_data[i] * log(pred_data[i]) + (1 - true_data[i]) * log((1 - pred_data[i]));
//     }

//     // Mean over all elements
//     return -(sum / static_cast<double>(true_label.size()));
// }

// std::vector<double> BCETensorDerivative(const Tensor &true_label, const Tensor &pred_prob)
// {
//     std::vector<double> d(pred_prob.size());
//     for (size_t i = 0; i < d.size(); ++i)
//     {
//         d[i] = (pred_prob.data()[i] - true_label.data()[i]) / (pred_prob.data()[i] * (1 - pred_prob.data()[i]) + 1e-8);
//     }
//     return d;
// }

/**
 * @brief Mean Squared Error loss function
 *
 * The mean-squared error (MSE) loss function is a versatile metric commonly used in regression tasks to measure the average squared difference between the predicted values and the actual target values. The following code is the implementation of MSE loss function and its derivative.
 * @param true_label true labels of the data
 * @param pred predicted values
 * @return mean squared error loss
 */
double MSELoss(std::vector<double> true_label, std::vector<double> pred)
{
    double sum = 0;
    for (int i = 0; i < true_label.size(); i++)
    {
        sum += pow(true_label[i] - pred[i], 2.0);
    }
    int size = true_label.size();
    double loss = (1.0 / size) * sum;
    return loss;
}

std::vector<double> MSELossDerivative(std::vector<double> true_label, std::vector<double> pred)
{ /**
   * @brief Derivative of Mean Squared Error loss function
   * @param true_label true labels of the data
   * @param pred predicted values
   * @return derivative of mean squared error loss
   */
    std::vector<double> sub = subtract(pred, true_label);
    std::vector<double> dev = scalarVectorMultiplication(sub, 2);
    return dev;
}

// double MSELoss(Tensor true_label, Tensor pred)
// {
//     // Ensure both tensors have the same number of elements
//     if (true_label.size() != pred.size())
//     {
//         throw std::invalid_argument("MSELoss: true_label and pred must have the same number of elements");
//     }

//     double sum = 0.0;
//     // Access the underlying flat data arrays
//     std::vector<double> true_data = true_label.data();
//     std::vector<double> pred_data = pred.data();

//     for (size_t i = 0; i < true_label.size(); ++i)
//     {
//         double diff = true_data[i] - pred_data[i];
//         sum += diff * diff; // equivalent to pow(diff, 2.0)
//     }

//     // Mean over all elements
//     return sum / static_cast<double>(true_label.size());
// }
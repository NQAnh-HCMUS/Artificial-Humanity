
#include "math_functions.h"

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
    // std::cout << "Binary Cross Entropy Loss(size=" << size << ") = " << loss << std::endl;
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
    // std::cout << "MSELoss(size=" << size << ") = " << loss << std::endl;
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
    // std::cout << "MSELossDerivative(size=" << dev.size() << ") completed" << std::endl;
    return dev;
}
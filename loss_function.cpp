// Loss Function Implementation
// In this project, we implement two frequently used loss function, namely binary cross-entropy loss and mean-square error loss. To implement these loss functions you will need to import the following dependencies.

#include <vector>
#include <math.h>
#include <cmath>
#include <iostream>
#include "math_functions.h"
// Binary cross-entropy loss (BCE)

// This loss function is specifically designed for binary classification tasks, normally known as binary cross-entropy (BCE) or log loss. It quantifies the difference between the predicted probability distribution and the actual binary labels (0 or 1). Here we demonstrate the implementation of the BCE loss function as well as its derivative.

double BCELoss(std::vector<double> true_label, std::vector<double> pred_prob)
{ /**
   * Binary Cross Entropy Loss
   * @param true_label true labels of the data
   * @param pred_prob predicted probabilities
   * @return binary cross entropy loss
   */
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

std::vector<double> BCELossDerivative(std::vector<double> true_label, std::vector<double> pred_prob)
{ /**
   * Compute derivative of binary cross entropy loss
   * @param true_label true labels of the data
   * @param pred_prob predicted probabilities
   * @return derivative of binary cross entropy loss
   */
    std::vector<double> dev = {(pred_prob[0] - true_label[0]) / ((pred_prob[0]) * (1 - pred_prob[0]))};
    // std::cout << "Binary Cross Entropy Loss Derivative = " << dev[0] << std::endl;
    return dev;
}
// Mean-squared error loss (MSE)

// The mean-squared error (MSE) loss function is a versatile metric commonly used in regression tasks to measure the average squared difference between the predicted values and the actual target values. The following code is the implementation of MSE loss function and its derivative.

double MSELoss(std::vector<double> true_label, std::vector<double> pred)
{ /**
   * Mean Squared Error Loss
   * @param true_label true labels of the data
   * @param pred predicted values
   * @return mean squared error loss
   */
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
   * Compute derivative of mean squared error loss
   * @param true_label true labels of the data
   * @param pred predicted values
   * @return derivative of mean squared error loss
   */
    std::vector<double> sub = subtract(pred, true_label);
    std::vector<double> dev = scalarVectorMultiplication(sub, 2);
    // std::cout << "MSELossDerivative(size=" << dev.size() << ") completed" << std::endl;
    return dev;
}
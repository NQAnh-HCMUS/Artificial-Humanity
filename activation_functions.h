#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

// Activation Function Declarations
// This header file contains declarations for various activation functions
// used in neural networks. These functions allow neural networks to perform
// non-linear transformations on input data.

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

/**
 * The sigmoid function maps any real-valued number to a value between 0 and 1.
 * It is often used in the output layer of a neural network when the task is a
 * binary classification problem.
 * @param x the input value
 * @return the output value of the sigmoid function
 */
double sigmoid(double x);

/**
 * @brief The derivative of the sigmoid function.
 * @param x the input value
 * @return the output value of the derivative of the sigmoid function
 */
double sigmoidDerivative(double x);

/**
 * @brief A vectorized version of the sigmoid function.
 * @param x the input vector
 * @return a vector where each element is the sigmoid of the corresponding element in x
 */
std::vector<double> vector_sigmoid(const std::vector<double> x);

/**
 * @brief A vectorized version of the derivative of the sigmoid function.
 * @param x the input vector
 * @return a vector where each element is the derivative of the sigmoid of the corresponding element in x
 */
std::vector<double> vector_sigmoid_derivative(const std::vector<double> x);

/**
 * @brief The Rectified Linear Unit (ReLU) activation function.
 * @param x the input value
 * @return the output value of the ReLU function
 */
double relu(double x);

/**
 * @brief The derivative of the Rectified Linear Unit (ReLU) activation function.
 * @param x the input value
 * @return the output value of the derivative of the ReLU function
 */
double reluDerivative(double x);

// APL: Adaptive Piecewise Linear
/**
 * @brief The Adaptive Piecewise Linear (APL) activation function.
 * @param x the input value
 * @param a the learned slope parameters for each hinge
 * @param b the learned shift parameters for each hinge
 * @return the output value of the APL function
 */
double apl(double x, const std::vector<double> &a, const std::vector<double> &b);

/**
 * @brief The derivative of the Adaptive Piecewise Linear (APL) activation function.
 * @param x the input value
 * @param a the learned slope parameters for each hinge
 * @param b the learned shift parameters for each hinge
 * @return the output value of the derivative of the APL function
 */
double apl_derivative(double x, const std::vector<double> &a, const std::vector<double> &b);

/**
 * @brief A vectorized version of the Adaptive Piecewise Linear (APL) activation function.
 * @param x the input vector
 * @param a the learned slope parameters for each hinge
 * @param b the learned shift parameters for each hinge
 * @return a vector where each element is the APL of the corresponding element in x
 */
std::vector<double> vector_apl(const std::vector<double> &x, const std::vector<double> &a, const std::vector<double> &b);

/**
 * @brief A vectorized version of the derivative of the Adaptive Piecewise Linear (APL) activation function.
 * @param x the input vector
 * @param a the learned slope parameters for each hinge
 * @param b the learned shift parameters for each hinge
 * @return a vector where each element is the derivative of the APL function of the corresponding element in x
 */
std::vector<double> vector_apl_derivative(const std::vector<double> &x, const std::vector<double> &a, const std::vector<double> &b);

#endif // ACTIVATION_FUNCTIONS_H
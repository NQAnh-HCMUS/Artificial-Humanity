#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include "../include/tensor.h"

/**
 * Sigmoid function maps any real number to a 0 or 1, often used in the output layer of a neural network when the task is a binary classification problem.
 * @param x the input value
 * @return the output value of the sigmoid function
 */
double sigmoid(double x);

/**
 * @brief Derivative Sigmoid function
 * @param x input
 * @return derivative of sigmoid of input
 */
double sigmoidDerivative(double x);

/**
 * @brief Vectorized Sigmoid function.
 * @param x the input vector
 * @return a vector where each element is the sigmoid of the corresponding element in x
 */
std::vector<double> vector_sigmoid(const std::vector<double> x);

/**
 * @brief Vectorized Derivative Sigmoid function.
 * @param x the input vector
 * @return a vector where each element is the derivative of the sigmoid of the corresponding element in x
 */
std::vector<double> vector_sigmoid_derivative(const std::vector<double> x);

Tensor sigmoid_tensor(Tensor x);

Tensor sigmoid_derivative_tensor(Tensor x);

std::vector<Tensor> sigmoid_tensor_vector(const std::vector<Tensor> &tensors);

std::vector<Tensor> sigmoid_derivative_tensor_vector(const std::vector<Tensor> &tensors);

/**
 * @brief Rectified Linear Unit (ReLU) activation function
 * @param x the input value
 * @return the output value of the ReLU function
 */
double relu(double x);

std::vector<double> vector_relu(const std::vector<double> &x);

/**
 * @brief Derivative of Rectified Linear Unit (ReLU) activation function.
 * @param x the input value
 * @return the output value of the derivative of the ReLU function
 */
double reluDerivative(double x);

std::vector<double> vector_relu_derivative(const std::vector<double> &x);

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
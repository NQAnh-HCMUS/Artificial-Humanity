#ifndef math_functions_H
#define math_functions_H

// Standard Library Includes
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <functional>
#include <algorithm>
#include <math.h>
#include <cmath>
#include <chrono>
#include <numeric>
#include "../include/tensor.h"

/**
 * @brief Generate a random double between
 * @param min
 * @param max
 * @return random double
 */
double randomDouble(double min, double max);

/**
 * @brief Computes the dot product of two vectors
 * @param[in] vector1 The first vector
 * @param[in] vector2 The second vector
 * @return The dot product of the two vectors
 */
double dotProduct(std::vector<double> &vector1, std::vector<double> &vector2);

/**
 * @brief Computes the element-wise multiplication of a vector and a scalar
 * @param[in] vector The vector to multiply
 * @param[in] scalar The scalar to multiply the vector with
 * @return A new vector with the element-wise multiplication of vector and scalar
 */
std::vector<double> scalarVectorMultiplication(std::vector<double> &vector, double scalar);

/**
 * @brief Computes the element-wise subtraction of two vectors
 * @param[in] vector1 The first vector
 * @param[in] vector2 The second vector
 * @return A new vector with the elementwise subtraction of vector1 and vector2
 */
std::vector<double> subtract(std::vector<double> &vector1, std::vector<double> &vector2);

/**
 * @brief Computes the transpose of a matrix
 * @param[in] matrix The matrix to transpose
 * @return The transpose of the matrix
 */
std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> &matrix);

/**
 * @brief Initializes a matrix with uniform random weights between -1.0 and 1.0
 * @param[in] rows The number of rows in the matrix
 * @param[in] cols The number of columns in the matrix
 * @return A matrix with uniform random weights between -1.0 and 1.0
 */
std::vector<std::vector<double>> uniformWeightInitializer(int rows, int cols);

/**
 * @brief Initializes a vector of biases with uniform random weights between -1.0 and 1.0.
 * @param[in] size The size of the vector
 * @return A vector of biases with uniform random weights between -1.0 and 1.0
 */
std::vector<double> biasInitailizer(int size);

#endif // math_functions_H
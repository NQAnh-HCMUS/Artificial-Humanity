#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iomanip>
#include <limits>

/**
 * @brief Simple dense tensor with row-major storage.
 */
class Tensor
{
private:
    std::vector<double> data_;    ///< Raw data storage
    std::vector<size_t> shape_;   ///< Dimensions (e.g. shape_ = {32, 128, 768})
    std::vector<size_t> strides_; ///< Strides for each dimension (in elements)

    /**
     * @brief Calculates strides_ based on the current shape_
     */
    void compute_strides();

    /**
     * @brief Checks if another tensor has matching shape.
     * @param other The tensor to compare against.
     * @throws std::invalid_argument If shapes do not match.
     */
    void check_shape_match(const Tensor &other) const;

    /**
     * @brief Helper to enforce correct mapping
     *
     * Given a vector of logical indices, compute the flat index into data_ using the stored strides_
     * @param indices Index per dimension.
     * @return Offset into the flat data buffer.
     * @throws std::invalid_argument If rank does not match.
     */
    size_t offset(const std::vector<size_t> &indices) const;

public:
    //----------Constructors and Destructor----------

    /**
     * @brief Default constructor
     */
    Tensor();

    /**
     * @brief Construct a tensor with:
     * @param shape Tensor dimensions.
     * @param init_val Initial value for all elements.
     */
    Tensor(const std::vector<size_t> &shape, double init_val = 0.0);

    /**
     * @brief Copy constructor
     * @param other The tensor to copy from.
     */
    Tensor(const Tensor &other);

    /**
     * @brief Copy assignment operator. Assigns the contents of another tensor.
     * @param other The tensor to assign from.
     * @return Reference to this tensor.
     */
    Tensor &operator=(const Tensor &other);

    /**
     * @brief Move constructor
     * @param other The tensor to move from.
     */
    Tensor(Tensor &&other) noexcept;

    /**
     * @brief Move ownership of data from another tensor.
     * @param other The tensor to move from.
     * @return Reference to this tensor.
     */
    Tensor &operator=(Tensor &&other) noexcept;

    /**
     * @brief Destructor
     */
    ~Tensor() = default;

    //----------Tensor Element Queries----------

    /**
     * @brief Get tensor shape
     * @return Reference to the shape vector.
     */
    const std::vector<size_t> &shape() const;

    /**
     * @brief Get tensor strides
     * @return Reference to the strides vector.
     */
    const std::vector<size_t> &strides() const;

    /**
     * @brief Get number of dimensions
     * @return Number of dimensions.
     */
    size_t ndim() const;

    /**
     * @brief Get total number of elements
     * @return Total element count.
     */
    size_t size() const;

    /**
     * @brief Get the size of a specific dimension
     * @param i Dimension index.
     * @return Size of i-th dimension.
     */
    size_t dim(size_t i) const;

    //----------Element Access and Manipulation----------

    /**
     * @brief Access element by multidimensional indices
     * @param indices Index per dimension.
     * @return Reference to the element.
     */
    double &operator()(const std::vector<size_t> &indices);

    /**
     * @brief Access element by multidimensional indices (const).
     * @param indices Index per dimension.
     * @return Const reference to the element.
     */
    const double &operator()(const std::vector<size_t> &indices) const;

    /**
     * @brief Access element in a 1D tensor.
     * @param i Index for the only dimension.
     * @return Reference to the element.
     * @throws std::invalid_argument If rank is not 1.
     */
    double &operator()(size_t i);

    /**
     * @brief Access element in a 2D tensor.
     * @param i Row index.
     * @param j Column index.
     * @return Reference to the element.
     * @throws std::invalid_argument If rank is not 2.
     */
    double &operator()(size_t i, size_t j);

    /**
     * @brief Access element in a 3D tensor.
     * @param i First dimension index.
     * @param j Second dimension index.
     * @param k Third dimension index.
     * @return Reference to the element.
     * @throws std::invalid_argument If rank is not 3.
     */
    double &operator()(size_t i, size_t j, size_t k);

    //----------Reshape and View Operations----------

    /**
     * @brief Reshape tensor without changing total size.
     * @param new_shape Target shape.
     * @return Reference to this tensor.
     * @throws std::invalid_argument If total element count changes.
     */
    Tensor &reshape(const std::vector<size_t> &new_shape);

    //----------Raw Data Access----------

    /**
     * @brief Get a mutable pointer to the data buffer.
     * @return Pointer to data.
     */
    double *data();

    /**
     * @brief Get a const pointer to the data buffer.
     * @return Const pointer to data.
     */
    const double *data() const;

    /**
     * @brief Fill all elements of the tensor with a constant value.
     * @param value The constant value to fill with.
     */
    void fill(double value);

    //----------Arithmetic and Matrix Operations----------

    /**
     * @brief Element-wise addition assignment.
     * @param other Tensor to add
     * @return Reference to this tensor.
     * @throws std::invalid_argument If shapes do not match.
     */
    Tensor &operator+=(const Tensor &other);

    /**
     * @brief Element-wise subtraction assignment.
     * @param other Tensor to subtract from
     * @return Reference to this tensor.
     * @throws std::invalid_argument If shapes do not match.
     */
    Tensor &operator-=(const Tensor &other);

    /**
     * @brief Element-wise multiplication assignment.
     * @param other Tensor to multiply with
     * @return Reference to this tensor.
     * @throws std::invalid_argument If shapes do not match.
     */
    Tensor &operator*=(const Tensor &other);

    /**
     * @brief Element-wise division assignment.
     * @param other The tensor to divide this one by.
     * @return Reference to this tensor.
     * @throws std::invalid_argument If shapes do not match.
     */
    Tensor &operator/=(const Tensor &other);

    /**
     * @brief Scalar addition assignment.
     * @param scalar The scalar value to add.
     * @return Reference to this tensor.
     */
    Tensor &operator+=(double scalar);

    /**
     * @brief Scalar subtraction assignment.
     * @param scalar The scalar value to subtract.
     * @return Reference to this tensor.
     */
    Tensor &operator-=(double scalar);

    /**
     * @brief Scalar multiplication assignment.
     * @param scalar The scalar value to multiply by.
     * @return Reference to this tensor.
     */
    Tensor &operator*=(double scalar);

    /**
     * @brief Scalar division assignment.
     * @param scalar The scalar value to divide by.
     * @return Reference to this tensor.
     */
    Tensor &operator/=(double scalar);

    //----------Stream Output (for Debugging)----------

    /**
     * @brief Stream output operator for debugging.
     * @param os Output stream.
     * @param t The tensor to output.
     * @return Reference to the output stream.
     */
    friend std::ostream &operator<<(std::ostream &os, const Tensor &t);
};

//----------Free Functions----------

/**
 * @brief Helper function to check if two tensors have matching shapes.
 * @param a First tensor.
 * @param b Second tensor.
 * @throws std::invalid_argument If shapes do not match.
 */
void check_shape_match(const Tensor &a, const Tensor &b);

/**
 * @brief Element-wise addition of two tensors.
 * @param a First tensor.
 * @param b Second tensor.
 * @return Result tensor containing element-wise sum.
 * @throws std::invalid_argument If shapes do not match.
 */
Tensor operator+(const Tensor &a, const Tensor &b);

/**
 * @brief Element-wise subtraction of two tensors.
 * @param a First tensor.
 * @param b Second tensor.
 * @return Result tensor containing element-wise difference.
 * @throws std::invalid_argument If shapes do not match.
 */
Tensor operator-(const Tensor &a, const Tensor &b);

/**
 * @brief Element-wise multiplication of two tensors.
 * @param a First tensor.
 * @param b Second tensor.
 * @return Result tensor containing element-wise product.
 * @throws std::invalid_argument If shapes do not match.
 */
Tensor operator*(const Tensor &a, const Tensor &b);

/**
 * @brief Element-wise division of two tensors.
 * @param a First tensor (numerator).
 * @param b Second tensor (denominator).
 * @return Result tensor containing element-wise quotient.
 * @throws std::invalid_argument If shapes do not match.
 */
Tensor operator/(const Tensor &a, const Tensor &b);

/**
 * @brief Add scalar to tensor (element-wise).
 * @param t Input tensor.
 * @param scalar Scalar value to add.
 * @return Result tensor.
 */
Tensor operator+(const Tensor &t, double scalar);

/**
 * @brief Add scalar to tensor (element-wise, commutative).
 * @param scalar Scalar value to add.
 * @param t Input tensor.
 * @return Result tensor.
 */
Tensor operator+(double scalar, const Tensor &t);

/**
 * @brief Subtract scalar from tensor (element-wise).
 * @param t Input tensor.
 * @param scalar Scalar value to subtract.
 * @return Result tensor.
 */
Tensor operator-(const Tensor &t, double scalar);

/**
 * @brief Subtract tensor from scalar (element-wise).
 * @param scalar Scalar value.
 * @param t Input tensor.
 * @return Result tensor.
 */
Tensor operator-(double scalar, const Tensor &t);

/**
 * @brief Multiply tensor by scalar (element-wise).
 * @param t Input tensor.
 * @param scalar Scalar multiplier.
 * @return Result tensor.
 */
Tensor operator*(const Tensor &t, double scalar);

/**
 * @brief Multiply tensor by scalar (element-wise, commutative).
 * @param scalar Scalar multiplier.
 * @param t Input tensor.
 * @return Result tensor.
 */
Tensor operator*(double scalar, const Tensor &t);

/**
 * @brief Divide tensor by scalar (element-wise).
 * @param t Input tensor.
 * @param scalar Scalar divisor.
 * @return Result tensor.
 */
Tensor operator/(const Tensor &t, double scalar);

/**
 * @brief Divide scalar by tensor (element-wise).
 * @param scalar Scalar numerator.
 * @param t Input tensor (denominator).
 * @return Result tensor.
 */
Tensor operator/(double scalar, const Tensor &t);

/**
 * @brief Matrix multiply two 2D tensors.
 * @param A Left-hand matrix.
 * @param B Right-hand matrix.
 * @return Result matrix C = A * B.
 * @throws std::invalid_argument If inputs are not 2D or inner dims mismatch.
 */
Tensor matmul(const Tensor &A, const Tensor &B);

/**
 * @brief Transpose a 2D tensor.
 * @param A Input matrix.
 * @return Transposed tensor.
 * @throws std::invalid_argument If tensor is not 2D.
 */
Tensor transpose(const Tensor &A);

/**
 * @brief Sum all elements of a tensor.
 * @param t Input tensor.
 * @return Sum of all elements.
 */
double sum(const Tensor &t);

#endif // TENSOR_H

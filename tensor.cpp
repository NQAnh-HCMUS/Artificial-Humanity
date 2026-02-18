#include <vector>
#include <cstddef>
#include <algorithm>
#include <numeric>
#include <stdexcept>

/**
 * @brief Simple dense tensor with row-major storage.
 */
class Tensor
{
private:
    std::vector<float> data_;     // raw data
    std::vector<size_t> shape_;   // dimensions (e.g. shape_ = {32, 128, 768})
    std::vector<size_t> strides_; // strides for each dimension (in elements) (e.g. {1,4} & {3,7}, stride is 3-1=2)

    /**
     * @brief Calculates strides_ based on the current shape_
     */
    void compute_strides()
    {
        strides_.resize(shape_.size());
        size_t stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i)
        {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    void check_shape_match(const Tensor &other) const
    {
        if (shape_ != other.shape_)
            throw std::invalid_argument("Tensor shape mismatch");
    }

    /**
     * @brief Helper to enforce correct mapping
     *
     * Given a vector of logical indices, compute the flat index into data_ using the stored strides_
     * @param indices Index per dimension.
     * @return Offset into the flat data buffer.
     * @throws std::invalid_argument If rank does not match.
     */
    size_t offset(const std::vector<size_t> &indices) const
    {
        if (indices.size() != shape_.size())
            throw std::invalid_argument("Number of indices not matching tensor dimensions");
        size_t off = 0;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            off += indices[i] * strides_[i];
        }
        return off;
    }

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
    Tensor(const std::vector<size_t> &shape, float init_val = 0.0f) : shape_(shape)
    {
        size_t total = std::accumulate(shape.begin(), shape.end(), 1ull, std::multiplies<size_t>());
        data_.resize(total, init_val);
        compute_strides();
    }
    /**
     * @brief Copy constructor
     * @param other The tensor to copy from.
     */
    Tensor(const Tensor &other)
        : data_(other.data_), // vector copies its elements
          shape_(other.shape_),
          strides_(other.strides_)
    {
    }
    /**
     * @brief Copy assignment operator. Assigns the contents of another tensor.
     * @param other The tensor to assign from.
     * @return Reference to this tensor.
     */
    Tensor &operator=(const Tensor &other)
    {
        data_ = other.data_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        return *this;
    }
    /**
     * @brief Move constructor
     * @param other The tensor to move from.
     */
    // Move constructor & move assignment
    Tensor(Tensor &&other) noexcept
        : data_(std::move(other.data_)),
          shape_(std::move(other.shape_)),
          strides_(std::move(other.strides_))
    {
    }
    /**
     * @brief Move ownership of data from another tensor.
     * @param other The tensor to move from.
     * @return Reference to this tensor.
     */
    Tensor &operator=(Tensor &&other) noexcept
    {
        if (this != &other)
        {
            data_ = std::move(other.data_);
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
        }
        return *this;
    }
    /**
     * @brief Destructor
     */
    ~Tensor() = default;

    //----------Tensor Element Queries----------
    /**
     * @brief Get tensor shape
     */
    const std::vector<size_t> &shape() const
    {
        return shape_;
    }
    /**
     * @brief Get tensor strides
     */
    const std::vector<size_t> &strides() const
    {
        return strides_;
    }
    /**
     * @brief Get number of dimensions
     */
    size_t ndim() const
    {
        return shape_.size();
    }
    /**
     * @brief Get total number of elements
     */
    size_t size() const
    {
        return data_.size();
    }
    /**
     * @brief Get the size of a specific dimension
     * @param i Dimension index.
     * @return Size of i-th dimension.
     */
    size_t dim(size_t i) const
    {
        return shape_.at(i);
    }

    //----------Element Access and Manipulation----------
    /**
     * @brief Access element by multidimensional indices
     * @param indices Index per dimension.
     * @return Reference to the element.
     */
    float &operator()(const std::vector<size_t> &indices)
    {
        return data_[offset(indices)];
    }

    /**
     * @brief Access element by multidimensional indices (const).
     * @param indices Index per dimension.
     * @return Const reference to the element.
     */
    const float &operator()(const std::vector<size_t> &indices) const
    {
        return data_[offset(indices)];
    }

    /**
     * @brief Access element in a 1D tensor.
     * @param i Index for the only dimension.
     * @return Reference to the element.
     * @throws std::invalid_argument If rank is not 1.
     */
    float &operator()(size_t i)
    {
        if (shape_.size() != 1)
            throw std::invalid_argument("Index dimension mismatch");
        return data_[i];
    }
    /**
     * @brief Access element in a 2D tensor.
     * @param i Row index.
     * @param j Column index.
     * @return Reference to the element.
     * @throws std::invalid_argument If rank is not 2.
     */
    float &operator()(size_t i, size_t j)
    {
        if (shape_.size() != 2)
            throw std::invalid_argument("Index dimension mismatch");
        return data_[i * strides_[0] + j];
    }
    /**
     * @brief Access element in a 3D tensor.
     * @param i First dimension index.
     * @param j Second dimension index.
     * @param k Third dimension index.
     * @return Reference to the element.
     * @throws std::invalid_argument If rank is not 3.
     */
    float &operator()(size_t i, size_t j, size_t k)
    {
        if (shape_.size() != 3)
            throw std::invalid_argument("Index dimension mismatch");
        return data_[i * strides_[0] + j * strides_[1] + k];
    }

    //----------Reshape and View Operations----------
    /**
     * @brief Reshape tensor without changing total size.
     * @param new_shape Target shape.
     * @return Reference to this tensor.
     * @throws std::invalid_argument If total element count changes.
     */
    Tensor &reshape(const std::vector<size_t> &new_shape)
    {
        size_t new_total = std::accumulate(new_shape.begin(), new_shape.end(), 1ull, std::multiplies<size_t>());
        if (new_total != size())
            throw std::invalid_argument("Total size must remain unchanged for reshape");
        shape_ = new_shape;
        compute_strides();
        return *this;
    }

    // /**
    //  * @brief Create a view of this tensor with a different shape.
    //  * @param new_shape The new shape for the view.
    //  * @return A new tensor that shares data with this one but has the specified shape.
    //  * @throws std::invalid_argument If total element count does not match.
    //  */
    // Tensor view(const std::vector<size_t> &new_shape) const;

    // /**
    //  * @brief Create a slice (view) of this tensor.
    //  * @param start Starting indices for each dimension.
    //  * @param size Size of the slice for each dimension.
    //  * @return A new tensor containing the sliced view.
    //  * @throws std::invalid_argument If indices or sizes are out of bounds.
    //  */
    // Tensor slice(const std::vector<size_t> &start,
    //              const std::vector<size_t> &size) const;

    //----------Raw Data Access----------
    /**
     * @brief Get a mutable pointer to the data buffer.
     */
    float *data()
    {
        return data_.data();
    }
    /**
     * @brief Get a const pointer to the data buffer.
     */
    const float *data() const
    {
        return data_.data();
    }

    /**
     * @brief Fill all elements of the tensor with a constant value.
     * @param value The constant value to fill with.
     */
    void fill(float value)
    {
        std::fill(data_.begin(), data_.end(), value);
    }

    //----------Arithmetic and Matrix Operations----------
    /**
     * @brief Element-wise addition assignment.
     * @param other Tensor to add
     * @return Reference to this tensor.
     * @throws std::invalid_argument If shapes do not match.
     */
    Tensor &operator+=(const Tensor &other)
    {
        check_shape_match(other);
        for (size_t i = 0; i < data_.size(); ++i)
            data_[i] += other.data_[i];
        return *this;
    }
    /**
     * @brief Element-wise subtraction assignment.
     * @param other Tensor to subtract from
     * @return Reference to this tensor.
     * @throws std::invalid_argument If shapes do not match.
     */
    Tensor &operator-=(const Tensor &other)
    {
        check_shape_match(other);
        for (size_t i = 0; i < data_.size(); ++i)
            data_[i] -= other.data_[i];
        return *this;
    }
    /**
     * @brief Element-wise multiplication assignment.
     * @param other Tensor to multiply with
     * @return Reference to this tensor.
     * @throws std::invalid_argument If shapes do not match.
     */
    Tensor &operator*=(const Tensor &other)
    {
        check_shape_match(other);
        for (size_t i = 0; i < data_.size(); ++i)
            data_[i] *= other.data_[i];
        return *this;
    }
    /**
     * @brief Element-wise division assignment.
     * @param other The tensor to divide this one by.
     * @return Reference to this tensor.
     * @throws std::invalid_argument If shapes do not match.
     */
    Tensor &operator/=(const Tensor &other)
    {
        check_shape_match(other);
        for (size_t i = 0; i < data_.size(); ++i)
            data_[i] /= other.data_[i];
        return *this;
    }

    // Scalar versions
    /**
     * @brief Scalar addition assignment.
     * @param scalar The scalar value to add.
     * @return Reference to this tensor.
     */
    Tensor &operator+=(float scalar)
    {
        for (size_t i = 0; i < data_.size(); ++i)
            data_[i] += scalar;
        return *this;
    }
    /**
     * @brief Scalar subtraction assignment.
     * @param scalar The scalar value to subtract.
     * @return Reference to this tensor.
     */
    Tensor &operator-=(float scalar)
    {
        for (size_t i = 0; i < data_.size(); ++i)
            data_[i] -= scalar;
        return *this;
    }
    /**
     * @brief Scalar multiplication assignment.
     * @param scalar The scalar value to multiply by.
     * @return Reference to this tensor.
     */
    Tensor &operator*=(float scalar)
    {
        for (size_t i = 0; i < data_.size(); ++i)
            data_[i] *= scalar;
        return *this;
    }
    /**
     * @brief Scalar division assignment.
     * @param scalar The scalar value to divide by.
     * @return Reference to this tensor.
     */
    Tensor &operator/=(float scalar)
    {
        for (size_t i = 0; i < data_.size(); ++i)
            data_[i] /= scalar;
        return *this;
    }

    //----------Stream Output (for Debugging)----------
    /**
     * @brief Stream output operator for debugging.
     * @param os Output stream.
     * @param t The tensor to output.
     * @return Reference to the output stream.
     */
    friend std::ostream &operator<<(std::ostream &os, const Tensor &t);
};

//----------Arithmetic and Matrix Operations----------
/**
 * @brief Helper function to check if two tensors have matching shapes.
 * @param a First tensor.
 * @param b Second tensor.
 * @throws std::invalid_argument If shapes do not match.
 */
inline void check_shape_match(const Tensor &a, const Tensor &b)
{
    if (a.shape() != b.shape())
        throw std::invalid_argument("Tensor shape mismatch");
}

/**
 * @brief Element-wise addition of two tensors.
 * @param a First tensor.
 * @param b Second tensor.
 * @return Result tensor containing element-wise sum.
 * @throws std::invalid_argument If shapes do not match.
 */
Tensor operator+(const Tensor &a, const Tensor &b)
{
    check_shape_match(a, b);
    Tensor result(a.shape());
    const float *a_data = a.data();
    const float *b_data = b.data();
    float *res_data = result.data();
    for (size_t i = 0; i < a.size(); ++i)
        res_data[i] = a_data[i] + b_data[i];
    return result;
}
Tensor operator-(const Tensor &a, const Tensor &b)
{
    check_shape_match(a, b);
    Tensor result(a.shape());
    const float *a_data = a.data();
    const float *b_data = b.data();
    float *res_data = result.data();
    for (size_t i = 0; i < a.size(); ++i)
        res_data[i] = a_data[i] - b_data[i];
    return result;
}

Tensor operator*(const Tensor &a, const Tensor &b)
{
    check_shape_match(a, b);
    Tensor result(a.shape());
    const float *a_data = a.data();
    const float *b_data = b.data();
    float *res_data = result.data();
    for (size_t i = 0; i < a.size(); ++i)
        res_data[i] = a_data[i] * b_data[i];
    return result;
}

Tensor operator/(const Tensor &a, const Tensor &b)
{
    check_shape_match(a, b);
    Tensor result(a.shape());
    const float *a_data = a.data();
    const float *b_data = b.data();
    float *res_data = result.data();
    for (size_t i = 0; i < a.size(); ++i)
        res_data[i] = a_data[i] / b_data[i];
    return result;
}

// Scalar versions
Tensor operator+(const Tensor &t, float scalar)
{
    Tensor result(t.shape());
    const float *src = t.data();
    float *dst = result.data();
    for (size_t i = 0; i < t.size(); ++i)
        dst[i] = src[i] + scalar;
    return result;
}

Tensor operator+(float scalar, const Tensor &t)
{
    return t + scalar;
}

Tensor operator-(const Tensor &t, float scalar)
{
    Tensor result(t.shape());
    const float *src = t.data();
    float *dst = result.data();
    for (size_t i = 0; i < t.size(); ++i)
        dst[i] = src[i] - scalar;
    return result;
}

Tensor operator-(float scalar, const Tensor &t)
{
    Tensor result(t.shape());
    const float *src = t.data();
    float *dst = result.data();
    for (size_t i = 0; i < t.size(); ++i)
        dst[i] = scalar - src[i];
    return result;
}

Tensor operator*(const Tensor &t, float scalar)
{
    Tensor result(t.shape());
    const float *src = t.data();
    float *dst = result.data();
    for (size_t i = 0; i < t.size(); ++i)
        dst[i] = src[i] * scalar;
    return result;
}

Tensor operator*(float scalar, const Tensor &t)
{
    return t * scalar;
}

Tensor operator/(const Tensor &t, float scalar)
{
    Tensor result(t.shape());
    const float *src = t.data();
    float *dst = result.data();
    for (size_t i = 0; i < t.size(); ++i)
        dst[i] = src[i] / scalar;
    return result;
}

Tensor operator/(float scalar, const Tensor &t)
{
    Tensor result(t.shape());
    const float *src = t.data();
    float *dst = result.data();
    for (size_t i = 0; i < t.size(); ++i)
        dst[i] = scalar / src[i];
    return result;
}

/**
 * @brief Matrix multiply two 2D tensors.
 * @param A Left-hand matrix.
 * @param B Right-hand matrix.
 * @return Result matrix C = A * B.
 * @throws std::invalid_argument If inputs are not 2D or inner dims mismatch.
 */
Tensor matmul(const Tensor &A, const Tensor &B)
{
    if (A.ndim() != 2 || B.ndim() != 2)
        throw std::invalid_argument("matmul expects 2D tensors");

    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = B.shape()[1];

    if (K != B.shape()[0])
        throw std::invalid_argument("Inner dimensions must match");

    Tensor C({M, N}, 0.0f);
    const float *a = A.data();
    const float *b = B.data();
    float *c = C.data();

    // Naive triple loop
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k)
            {
                sum += a[i * K + k] * b[k * N + j]; // row-major access
            }
            c[i * N + j] = sum;
        }
    }
    return C;
}

/**
 * @brief Transpose a 2D tensor.
 * @return Transposed tensor.
 * @throws std::invalid_argument If tensor is not 2D.
 */
Tensor transpose(const Tensor &A)
{
    if (A.ndim() != 2)
        throw std::invalid_argument("transpose requires 2D tensor");

    size_t rows = A.shape()[0];
    size_t cols = A.shape()[1];

    // Create result tensor with swapped dimensions
    Tensor result({cols, rows});

    // Get raw pointers to the data
    const float *src = A.data();
    float *dst = result.data();

    // Perform the transpose using linear indexing
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            // src index: i * cols + j   (row-major)
            // dst index: j * rows + i   (row-major, because result shape is cols x rows)
            dst[j * rows + i] = src[i * cols + j];
        }
    }

    return result;
}

/**
 * @brief Sum all elements of a tensor.
 * @param t Input tensor.
 * @return Sum of all elements.
 */
float sum(const Tensor &t)
{
    return std::accumulate(t.data(), t.data() + t.size(), 0.0f);
}

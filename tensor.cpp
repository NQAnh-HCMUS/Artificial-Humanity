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
    // Default constructor
    Tensor();
    /**
     * @brief Construct a tensor with the given shape and initial value.
     * @param shape Tensor dimensions.
     * @param init_val Initial value for all elements.
     */
    Tensor(const std::vector<size_t> &shape, float init_val = 0.0f)
        : shape_(shape)
    {
        size_t total = std::accumulate(shape.begin(), shape.end(), 1ull, std::multiplies<size_t>());
        data_.resize(total, init_val);
        compute_strides();
    }
    // Copy constructor and assignment operator
    Tensor(const Tensor &other);
    Tensor &operator=(const Tensor &other);

    // Move constructor and assignment
    Tensor(Tensor &&other) noexcept;
    Tensor &operator=(Tensor &&other) noexcept;

    // Destructor
    ~Tensor() = default;

    //----------Tensor Element Queries----------
    /**
     * @brief Get tensor shape.
     */
    const std::vector<size_t> &shape() const { return shape_; }
    /**
     * @brief Get tensor strides.
     */
    const std::vector<size_t> &strides() const { return strides_; }
    /**
     * @brief Get number of dimensions.
     */
    size_t ndim() const { return shape_.size(); }
    /**
     * @brief Get total number of elements.
     */
    size_t size() const { return data_.size(); }
    /**
     * @brief Get the size of a specific dimension
     */
    size_t dim(size_t i) const { return shape_.at(i); }
    /**
     * @brief Get a mutable pointer to the data buffer.
     */
    float *data() { return data_.data(); }
    /**
     * @brief Get a const pointer to the data buffer.
     */
    const float *data() const { return data_.data(); }

    //----------Element Access and Manipulation----------
    /**
     * @brief Access element by multidimensional indices.
     * @param indices Index per dimension.
     * @return Reference to the element.
     * @throws std::invalid_argument If rank does not match.
     */
    float &operator()(const std::vector<size_t> &indices)
    {
        return data_[offset(indices)];
    }

    /**
     * @brief Access element by multidimensional indices (const).
     * @param indices Index per dimension.
     * @return Const reference to the element.
     * @throws std::invalid_argument If rank does not match.
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

    // Tensor operator+(const Tensor &a, const Tensor &b)
    // {
    //     if (a.shape() != b.shape())
    //         throw std::invalid_argument("Shapes must match for element-wise addition");
    //     Tensor result(a.shape());
    //     const float *a_data = a.data();
    //     const float *b_data = b.data();
    //     float *res_data = result.data();
    //     size_t n = a.size();
    //     for (size_t i = 0; i < n; ++i)
    //     {
    //         res_data[i] = a_data[i] + b_data[i];
    //     }
    //     return result;
    // }

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
     * @brief Sum all elements of a tensor.
     * @param t Input tensor.
     * @return Sum of all elements.
     */
    float sum(const Tensor &t)
    {
        return std::accumulate(t.data(), t.data() + t.size(), 0.0f);
    }

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

    /**
     * @brief Transpose a 2D tensor.
     * @return Transposed tensor.
     * @throws std::invalid_argument If tensor is not 2D.
     */
    Tensor transpose()
    {
        if (ndim() != 2)
            throw std::invalid_argument("transpose requires 2D tensor");
        Tensor result({shape_[1], shape_[0]});
        // copy elements manually or use indexing
        for (size_t i = 0; i < shape_[0]; ++i)
            for (size_t j = 0; j < shape_[1]; ++j)
                result(j, i) = (*this)(i, j);
        return result;
    }
};

/**
 * @brief Lightweight view into an existing tensor buffer.
 */
class TensorView
{
    float *data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_; // starting point in the original data

public:
    /**
     * @brief Construct a tensor view over existing storage.
     * @param base Pointer to the base data buffer.
     * @param shape View shape.
     * @param strides View strides (in elements).
     * @param offset Starting offset in the base buffer.
     */
    TensorView(float *base, const std::vector<size_t> &shape,
               const std::vector<size_t> &strides, size_t offset = 0)
        : data_(base), shape_(shape), strides_(strides), offset_(offset) {}

    /**
     * @brief Access an element within the view.
     * @param indices Index per dimension.
     * @return Reference to the element.
     */
    float &operator()(const std::vector<size_t> &indices)
    {
        size_t off = offset_;
        for (size_t i = 0; i < indices.size(); ++i)
            off += indices[i] * strides_[i];
        return data_[off];
    }
    // ... const version, etc.
};

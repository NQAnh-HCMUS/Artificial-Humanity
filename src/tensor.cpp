#include "../include/tensor.h"

void Tensor::compute_strides()
{
    strides_.resize(shape_.size());
    size_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i)
    {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

void Tensor::check_shape_match(const Tensor &other) const
{
    if (shape_ != other.shape_)
        throw std::invalid_argument("Tensor shape mismatch");
}

size_t Tensor::offset(const std::vector<size_t> &indices) const
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

//----------Constructors and Destructor----------

Tensor::Tensor() {}
// Tensor::Tensor() = default;

Tensor::Tensor(const std::vector<size_t> &shape, double init_val) : shape_(shape)
{
    size_t total = std::accumulate(shape.begin(), shape.end(), 1ull, std::multiplies<size_t>());
    data_.resize(total, init_val);
    compute_strides();
}

Tensor::Tensor(const Tensor &other)
    : data_(other.data_),
      shape_(other.shape_),
      strides_(other.strides_)
{
}

Tensor &Tensor::operator=(const Tensor &other)
{
    data_ = other.data_;
    shape_ = other.shape_;
    strides_ = other.strides_;
    return *this;
}

Tensor::Tensor(Tensor &&other) noexcept
    : data_(std::move(other.data_)),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_))
{
}

Tensor &Tensor::operator=(Tensor &&other) noexcept
{
    if (this != &other)
    {
        data_ = std::move(other.data_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
    }
    return *this;
}

//----------Tensor Element Queries----------

double *Tensor::data()
{
    return data_.data();
}

const double *Tensor::data() const
{
    return data_.data();
}

std::vector<double> Tensor::get_data() const
{
    return data_;
}

void Tensor::set_data(const std::vector<double> &data)
{
    if (data.size() != size())
        throw std::invalid_argument("Data size does not match tensor size");
    data_ = data;
}

const std::vector<size_t> &Tensor::shape() const
{
    return shape_;
}

void Tensor::set_shape(const std::vector<size_t> &shape)
{
    size_t new_total = std::accumulate(shape.begin(), shape.end(), 1ull, std::multiplies<size_t>());
    if (new_total != size())
        throw std::invalid_argument("Total size must remain unchanged for reshape");
    shape_ = shape;
    compute_strides();
}

const std::vector<size_t> &Tensor::strides() const
{
    return strides_;
}

void Tensor::set_strides(const std::vector<size_t> &strides)
{
    if (strides.size() != shape_.size())
        throw std::invalid_argument("Strides size must match shape size");
    strides_ = strides;
}

size_t Tensor::ndim() const
{
    return shape_.size();
}

size_t Tensor::size() const
{
    return data_.size();
}

size_t Tensor::dim(size_t i) const
{
    return shape_.at(i);
}

//----------Element Access and Manipulation----------

double &Tensor::operator()(const std::vector<size_t> &indices)
{
    return data_[offset(indices)];
}

const double &Tensor::operator()(const std::vector<size_t> &indices) const
{
    return data_[offset(indices)];
}

double &Tensor::operator()(size_t i)
{
    if (shape_.size() != 1)
        throw std::invalid_argument("Index dimension mismatch");
    return data_[i];
}

double &Tensor::operator()(size_t i, size_t j)
{
    if (shape_.size() != 2)
        throw std::invalid_argument("Index dimension mismatch");
    return data_[i * strides_[0] + j];
}

double &Tensor::operator()(size_t i, size_t j, size_t k)
{
    if (shape_.size() != 3)
        throw std::invalid_argument("Index dimension mismatch");
    return data_[i * strides_[0] + j * strides_[1] + k];
}

//----------Reshape and View Operations----------

Tensor &Tensor::reshape(const std::vector<size_t> &new_shape)
{
    size_t new_total = std::accumulate(new_shape.begin(), new_shape.end(), 1ull, std::multiplies<size_t>());
    if (new_total != size())
        throw std::invalid_argument("Total size must remain unchanged for reshape");
    shape_ = new_shape;
    compute_strides();
    return *this;
}

//----------Raw Data Access----------

void Tensor::fill(double value)
{
    std::fill(data_.begin(), data_.end(), value);
}

//----------Arithmetic and Matrix Operations----------

Tensor &Tensor::operator+=(const Tensor &other)
{
    check_shape_match(other);
    for (size_t i = 0; i < data_.size(); ++i)
        data_[i] += other.data_[i];
    return *this;
}

Tensor &Tensor::operator-=(const Tensor &other)
{
    check_shape_match(other);
    for (size_t i = 0; i < data_.size(); ++i)
        data_[i] -= other.data_[i];
    return *this;
}

Tensor &Tensor::operator*=(const Tensor &other)
{
    check_shape_match(other);
    for (size_t i = 0; i < data_.size(); ++i)
        data_[i] *= other.data_[i];
    return *this;
}

Tensor &Tensor::operator/=(const Tensor &other)
{
    check_shape_match(other);
    for (size_t i = 0; i < data_.size(); ++i)
        data_[i] /= other.data_[i];
    return *this;
}

Tensor &Tensor::operator+=(double scalar)
{
    for (size_t i = 0; i < data_.size(); ++i)
        data_[i] += scalar;
    return *this;
}

Tensor &Tensor::operator-=(double scalar)
{
    for (size_t i = 0; i < data_.size(); ++i)
        data_[i] -= scalar;
    return *this;
}

Tensor &Tensor::operator*=(double scalar)
{
    for (size_t i = 0; i < data_.size(); ++i)
        data_[i] *= scalar;
    return *this;
}

Tensor &Tensor::operator/=(double scalar)
{
    for (size_t i = 0; i < data_.size(); ++i)
        data_[i] /= scalar;
    return *this;
}

//----------Stream Output (for Debugging)----------

std::ostream &operator<<(std::ostream &os, const Tensor &t)
{
    os << "Tensor(";
    const auto &shape = t.shape();
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
            os << "x";
        os << shape[i];
    }
    os << ") [";
    for (size_t i = 0; i < t.size(); ++i)
    {
        if (i > 0)
            os << ", ";
        os << t.data()[i];
    }
    os << "]";
    return os;
}

//----------Free Functions----------

void check_shape_match(const Tensor &a, const Tensor &b)
{
    if (a.shape() != b.shape())
        throw std::invalid_argument("Tensor shape mismatch");
}

Tensor operator+(const Tensor &a, const Tensor &b)
{
    check_shape_match(a, b);
    Tensor result(a.shape());
    const double *a_data = a.data();
    const double *b_data = b.data();
    double *res_data = result.data();
    for (size_t i = 0; i < a.size(); ++i)
        res_data[i] = a_data[i] + b_data[i];
    return result;
}

Tensor operator-(const Tensor &a, const Tensor &b)
{
    check_shape_match(a, b);
    Tensor result(a.shape());
    const double *a_data = a.data();
    const double *b_data = b.data();
    double *res_data = result.data();
    for (size_t i = 0; i < a.size(); ++i)
        res_data[i] = a_data[i] - b_data[i];
    return result;
}

Tensor operator*(const Tensor &a, const Tensor &b)
{
    check_shape_match(a, b);
    Tensor result(a.shape());
    const double *a_data = a.data();
    const double *b_data = b.data();
    double *res_data = result.data();
    for (size_t i = 0; i < a.size(); ++i)
        res_data[i] = a_data[i] * b_data[i];
    return result;
}

Tensor operator/(const Tensor &a, const Tensor &b)
{
    check_shape_match(a, b);
    Tensor result(a.shape());
    const double *a_data = a.data();
    const double *b_data = b.data();
    double *res_data = result.data();
    for (size_t i = 0; i < a.size(); ++i)
        res_data[i] = a_data[i] / b_data[i];
    return result;
}

Tensor operator+(const Tensor &t, double scalar)
{
    Tensor result(t.shape());
    const double *src = t.data();
    double *dst = result.data();
    for (size_t i = 0; i < t.size(); ++i)
        dst[i] = src[i] + scalar;
    return result;
}

Tensor operator+(double scalar, const Tensor &t)
{
    return t + scalar;
}

Tensor operator-(const Tensor &t, double scalar)
{
    Tensor result(t.shape());
    const double *src = t.data();
    double *dst = result.data();
    for (size_t i = 0; i < t.size(); ++i)
        dst[i] = src[i] - scalar;
    return result;
}

Tensor operator-(double scalar, const Tensor &t)
{
    Tensor result(t.shape());
    const double *src = t.data();
    double *dst = result.data();
    for (size_t i = 0; i < t.size(); ++i)
        dst[i] = scalar - src[i];
    return result;
}

Tensor operator*(const Tensor &t, double scalar)
{
    Tensor result(t.shape());
    const double *src = t.data();
    double *dst = result.data();
    for (size_t i = 0; i < t.size(); ++i)
        dst[i] = src[i] * scalar;
    return result;
}

Tensor operator*(double scalar, const Tensor &t)
{
    return t * scalar;
}

Tensor operator/(const Tensor &t, double scalar)
{
    Tensor result(t.shape());
    const double *src = t.data();
    double *dst = result.data();
    for (size_t i = 0; i < t.size(); ++i)
        dst[i] = src[i] / scalar;
    return result;
}

Tensor operator/(double scalar, const Tensor &t)
{
    Tensor result(t.shape());
    const double *src = t.data();
    double *dst = result.data();
    for (size_t i = 0; i < t.size(); ++i)
        dst[i] = scalar / src[i];
    return result;
}

Tensor matmul(const Tensor &A, const Tensor &B)
{
    if (A.ndim() != 2 || B.ndim() != 2)
        throw std::invalid_argument("matmul expects 2D tensors");

    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = B.shape()[1];

    if (K != B.shape()[0])
        throw std::invalid_argument("Inner dimensions must match");

    Tensor C({M, N}, 0.0);
    const double *a = A.data();
    const double *b = B.data();
    double *c = C.data();

    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            double sum = 0.0;
            for (size_t k = 0; k < K; ++k)
            {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
    return C;
}

Tensor transpose(const Tensor &A)
{
    if (A.ndim() != 2)
        throw std::invalid_argument("transpose requires 2D tensor");

    size_t rows = A.shape()[0];
    size_t cols = A.shape()[1];

    Tensor result({cols, rows});

    const double *src = A.data();
    double *dst = result.data();

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            dst[j * rows + i] = src[i * cols + j];
        }
    }

    return result;
}

double sum(const Tensor &t)
{
    return std::accumulate(t.data(), t.data() + t.size(), 0.0);
}

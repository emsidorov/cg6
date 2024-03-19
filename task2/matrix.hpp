#include <vector>
#include <iostream>
#include <cassert>
#include <glm/glm.hpp>


class Matrix {
public:
    std::vector<float> data;
    size_t rows, cols;

    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data(rows * cols) {
        for (int i = 0; i < data.size(); ++i) { 
            data[i] = 0;
        }
    }

    Matrix() : rows(0), cols(0) {}

     Matrix(const glm::vec3& vec) : rows(1), cols(3), data({vec.x, vec.y, vec.z}) {}

    float& operator()(size_t row, size_t col) {
        return data[row * cols + col];
    }

    const float& operator()(size_t row, size_t col) const {
        return data[row * cols + col];
    }

    static Matrix transpose(const Matrix& m) {
        Matrix t(m.cols, m.rows);
        for (size_t i = 0; i < m.rows; ++i) {
            for (size_t j = 0; j < m.cols; ++j) {
                t(j, i) = m(i, j);
            }
        }
        return t;
    }

    static Matrix multiply(const Matrix& a, const Matrix& b) {
        assert(a.cols == b.rows);
        Matrix result(a.rows, b.cols);
        for (size_t i = 0; i < result.rows; ++i) {
            for (size_t j = 0; j < result.cols; ++j) {
                float sum = 0.0;
                for (size_t k = 0; k < a.cols; ++k) {
                    sum += a(i, k) * b(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    Matrix operator+(const Matrix& rhs) const {
        assert(rows == rhs.rows && cols == rhs.cols); // Убедитесь, что размеры матриц совпадают
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = (*this)(i, j) + rhs(i, j);
            }
        }
        return result;
    }

    Matrix operator-(const Matrix& rhs) const {
        assert(rows == rhs.rows && cols == rhs.cols);
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = (*this)(i, j) - rhs(i, j);
            }
        }
        return result;
    }

    Matrix operator*(const Matrix& rhs) const {
        assert(rows == rhs.rows && cols == rhs.cols);
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = (*this)(i, j) * rhs(i, j);
            }
        }
        return result;
    }

    Matrix operator/(const Matrix& rhs) const {
        assert(rows == rhs.rows && cols == rhs.cols);
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = (*this)(i, j) / rhs(i, j);
            }
        }
        return result;
    }

    Matrix operator/(const float& val) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = (*this)(i, j) / val;
            }
        }
        return result;
    }

    Matrix operator*(const float& val) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = (*this)(i, j) * val;
            }
        }
        return result;
    }

    Matrix operator+(const float& val) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = (*this)(i, j) + val;
            }
        }
        return result;
    }

    Matrix sum() const {
        Matrix sum(1, 1);
        for (size_t i = 0; i < rows * cols; ++i) {
            sum.data[0] += data[i];
        }
        return sum;
    }

    Matrix abs() const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; ++i) {
            result.data[i] = std::abs(data[i]);
        }
        return result;
    }

    Matrix sqrt() const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; ++i) {
            result.data[i] = std::sqrt(data[i]);
        }
        return result;
    }

    float max() const {
        float maxValue = data[0];
        for (size_t i = 1; i < rows * cols; ++i) {
            if (data[i] > maxValue) {
                maxValue = data[i];
            }
        }
        return maxValue;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        for (size_t i = 0; i < m.rows; ++i) {
            for (size_t j = 0; j < m.cols; ++j) {
                os << m(i, j) << " ";
            }
            os << std::endl;
        }
        return os;
    }
};
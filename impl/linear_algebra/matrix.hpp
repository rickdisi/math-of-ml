#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath> 

// complete Matrix object

class Matrix {
    private:
        int rows_, cols_;
        std::vector<double> data_;

    public:
        Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols) {}

        double& operator()(int row, int col) {
            return data_[row * cols_ + col];
        }
        const double& operator()(int row, int col) const {
            return data_[row * cols_ + col];
        }

        int rows() const {
            return rows_;
        }
        int cols() const {
            return cols_;
        }

        double& operator()(int i) {
            return data_[i];
        }
        const double& operator()(int i) const {
            return data_[i];
        }

        // Define matrix addition
        Matrix operator+(const Matrix& mat) const {
            if (rows_ != mat.rows_ || cols_ != mat.cols_) {
                throw std::invalid_argument("Matrix dimensions not equal");
            };

            Matrix result(rows_, cols_);
            for (int i = 0; i < rows_; i++) {
                for (int j = 0; j < cols_; j++) {
                    result.data_[i * cols_ + j] = data_[i * cols_ + j] + mat.data_[i * cols_ + j];
                }
            }
            return result;
        }

        // Define matrix multiplication
        Matrix operator*(const Matrix& mat) const {
            if (cols_ != mat.rows_) {
                throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
            }

            Matrix result(rows_, mat.cols_);
            for (int i = 0; i < rows_; i++) {
                for (int j = 0; j < mat.cols_; j++) {
                    double sum = 0;
                    for (int k = 0; k < cols_; k++) {
                        sum += data_[i * cols_ + k] * mat.data_[k * mat.cols_ + j];
                    }
                    result.data_[i * mat.cols_ + j] = sum;
                }
            }
            return result;
        }

        // Define transposition
        Matrix transpose() const {

            Matrix result(cols_, rows_);
            for (int i = 0; i < rows_; i++) {
                for (int j = 0; j < cols_; j++) {
                    result(j, i) = data_[i * cols_ + j];
                }
            }
            return result;
        }
};

// Other Operators

// dot(u, v): dot product of two n x 1 column vectors, stored as Matrix(n,1)
double dot(const Matrix& u, const Matrix& v) {
    double sum = 0.0;
    for (int i = 0; i < u.rows(); ++i) {
        sum += u(i, 0) * v(i, 0);
    }
    return sum;
}

// norm(v): sqrt(dot(v,v))
double norm(const Matrix& v) {
    return std::sqrt(dot(v, v));
}

// scale(v, c): returns c * v for an n x 1 vector v
Matrix scale(const Matrix& v, double c) {
    int n = v.rows();
    Matrix result(n, 1);
    for (int i = 0; i < n; ++i) {
        result(i, 0) = c * v(i, 0);
    }
    return result;
}

// subtractInPlace(u, w): u -= w, for n x 1 vectors
void subtractInPlace(Matrix& u, const Matrix& w) {
    for (int i = 0; i < u.rows(); ++i) {
        u(i, 0) -= w(i, 0);
    }
}

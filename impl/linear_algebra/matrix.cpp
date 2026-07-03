#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

class Matrix {
    private:
        int rows_, cols_;
        vector<double> data_;

    public:
        Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols) {}

        double& operator()(int row, int col) {
            return data_[row * cols_ + col];
        }
        const double& operator()(int row, int col) const {
            return data_[row * cols_ + col];
        }

        // Define matrix addition
        Matrix operator+(const Matrix& mat) const {
            if (rows_ != mat.rows_ || cols_ != mat.cols_) {
                throw invalid_argument("Matrix dimensions not equal");
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
                throw invalid_argument("Matrix dimensions incompatible for multiplication");
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

int main() {
    Matrix x = Matrix(5, 5);
    x(2, 3) = 2222;

    Matrix y = Matrix(5,5);
    y(2,4) = 6969;
    y(2,3) = 6767;

    Matrix z = x + y;
    Matrix w = x * z;
    Matrix v = y.transpose();

    cout << (4,2) << endl;
    return 0;
}
#include <iostream>
#include <cmath>
#include <utility>
#include "matrix.hpp"

std::vector<int> factorise(Matrix& A) {
    int n = A.rows();
    std::vector<int> perm(n);

    for (int i = 0; i < n; ++i) perm[i] = i;

    for (int k = 0; k < n - 1; ++k) {
        // Step 1: find pivot row
        int r = k;
        double maxVal = std::abs(A(k, k));

        for (int i = k + 1; i < n; ++i) {
            double val = std::abs(A(i, k));
            if (val > maxVal) {
                maxVal = val;
                r = i;
            }
        }

        // Step 2: swap if needed
        if (r != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(A(r, j), A(k, j));
            }
            std::swap(perm[r], perm[k]);
        }

        // Step 3: eliminate
        for (int i = k + 1; i < n; ++i) {
            double m = A(i, k) / A(k, k);
            A(i, k) = m;
            for (int j = k + 1; j < n; ++j) {
                A(i, j) = A(i, j) - m * A(k, j);
            }
        }
    }

    return perm;
};


Matrix solve(const Matrix& LU, const std::vector<int>& perm, const Matrix& b) {
    int n = LU.rows(); 

    // Construct bPrime
    Matrix bPrime(n, 1);
    for (int i = 0; i < n; ++i) {
        bPrime(i) = b(perm[i], 0);
    }

    // Forward substitution
    Matrix y(n, 1);
    for (int k = 0; k < n; ++k) {
        double sum = 0.0;
        for (int j = 0; j < k; ++j) {
            sum += LU(k, j) * y(j);
        }
        y(k) = bPrime(k) - sum;
    }

    // Backward substitution
    Matrix x(n, 1);
    for (int k = n - 1; k >= 0; --k) {
        double sum = 0.0;
        for (int j = k + 1; j < n; ++j) {
            sum += LU(k, j) * x(j);
        }
        x(k) = (y(k) - sum) / LU(k, k);
    }

    return x;
}

int main() {

    Matrix A(2, 2);
    A(0, 0) = 2; A(0, 1) = 3;
    A(1, 0) = 4; A(1, 1) = 7;

    Matrix b(2, 1);
    b(0, 0) = 5;
    b(1, 0) = 11;

    std::vector<int> perm = factorise(A); // A is overwritten in place with LU
    Matrix x = solve(A, perm, b);

    for (int i = 0; i < 2; ++i) {
        std::cout << "x[" << i << "] = " << x(i) << std::endl;
    }

    return 0;
}
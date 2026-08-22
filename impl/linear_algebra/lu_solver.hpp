#pragma once

#include <cmath>
#include <utility>
#include "matrix.hpp"

// Factorises A=LU and then solves a linear system via forward and back substitution

std::vector<int> factoriseLU(Matrix& A) {
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


Matrix solveLU(const Matrix& LU, const std::vector<int>& perm, const Matrix& b) {
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
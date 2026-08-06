#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include "matrix.hpp"
#include "jacobi_eigensolver.hpp"

// Full SVD via eigendecomposition

// Helpers

Matrix getColumn(const Matrix& A, int k) {
    int n = A.rows();
    Matrix col(n, 1);
    for (int i = 0; i < n; ++i) {
        col(i, 0) = A(i, k);
    }
    return col;
}

void setColumn(Matrix& M, int k, const Matrix& v) {
    int n = M.rows();
    for (int i = 0; i < n; ++i) {
        M(i, k) = v(i, 0);
    }
}

// Singular Value Decomposition

Matrix computeAtA(const Matrix& A, Matrix& eigenvalues) {
    
    int n = A.cols();
    Matrix AtA = A.transpose() * A;
    Matrix V(n, n);

    jacobiEigenDecomposition(AtA, V, eigenvalues);

    return V;
}

void sortEigenDecomposition(const Matrix& eigenvalues, const Matrix& Q, Matrix& sortedEigenvalues, Matrix& sortedQ) {
    int n = eigenvalues.rows(); // assumption: eigenvalues is n x 1, Q is n x n

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return eigenvalues(a, 0) > eigenvalues(b, 0); // descending
    });

    for (int newPos = 0; newPos < n; ++newPos) {
        int oldPos = idx[newPos];
        sortedEigenvalues(newPos, 0) = eigenvalues(oldPos, 0);
        for (int row = 0; row < n; ++row) {
            sortedQ(row, newPos) = Q(row, oldPos);
        }
    }
}

void computeSigmaAndU(const Matrix& A, const Matrix& sortedEigenvalues, const Matrix& sortedV, Matrix& sigma, Matrix& U) {
    int m = A.rows();
    int n = A.cols();
    const double tol = 1e-10;

    for (int i = 0; i < n; ++i) {
        double s = std::sqrt(std::max(sortedEigenvalues(i, 0), 0.0));
        sigma(i, 0) = s;

        if (s > tol) {
            Matrix vi = getColumn(sortedV, i);
            Matrix Avi = A * vi; // m x 1
            Avi = scale(Avi, 1.0 / s); 
            setColumn(U, i, Avi);
        }
        // else: leave column i of U as zero, which is already true by construction
    }
}

int main() {

    Matrix A(4, 6);
    A(0,0) = 2;  A(0,1) = -1; A(0,2) = 3;  A(0,3) = 0;  A(0,4) = 5; A(0,5) = 1;
    A(1,0) = 1;  A(1,1) = 4;  A(1,2) = -2; A(1,3) = 3;  A(1,4) = 0; A(1,5) = 2;
    A(2,0) = 0;  A(2,1) = 2;  A(2,2) = 1;  A(2,3) = -1; A(2,4) = 3; A(2,5) = 4;
    A(3,0) = 3;  A(3,1) = 3;  A(3,2) = 1;  A(3,3) = 3;  A(3,4) = 5; A(3,5) = 3;

    int n = A.cols();
    int m = A.rows();

    Matrix eigenvalues(n, 1);
    Matrix V = computeAtA(A, eigenvalues);

    Matrix sortedEigenvalues(n, 1);
    Matrix sortedV(n, n);
    sortEigenDecomposition(eigenvalues, V, sortedEigenvalues, sortedV);

    Matrix sigma(n, 1);
    Matrix U(m, n);
    computeSigmaAndU(A, sortedEigenvalues, sortedV, sigma, U);

    // --- Verifier ---

    // Build Sigma (n x n, zero except diagonal)
    Matrix Sigma(n, n);
    for (int i = 0; i < n; ++i) {
        Sigma(i, i) = sigma(i, 0);
    }

    // Reconstruct A from U, Sigma, V
    Matrix reconstructed = U * Sigma * sortedV.transpose();

    // Max absolute reconstruction error
    double maxDiff = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double diff = std::abs(A(i, j) - reconstructed(i, j));
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }
    }

    std::cout << "max reconstruction error: " << maxDiff << std::endl;

    return 0;
}
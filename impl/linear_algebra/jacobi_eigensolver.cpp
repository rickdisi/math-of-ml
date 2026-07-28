#include <iostream>
#include <cmath>
#include "matrix.hpp"

// Applies a Jacobi Eigen Decomposition

double findLargestOffDiagonal(const Matrix& A, int& p, int& q) {
    int n = A.rows();
    double maxVal = 0.0;
    p = -1;
    q = -1;

    // Iterate through matrix entries and find the biggest off diagonal value.
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) { // A is symmetric so only scan the upper triangle
            double val = std::abs(A(i, j));
            if (val > maxVal) {
                maxVal = val;
                p = i;
                q = j;
            }
        }
    }

    return maxVal;
}

void computeRotationAngle(double App, double Aqq, double Apq, double& c, double& s) {

    if (Apq == 0.0) {
        // Already zero at this position; rotation is identity.
        c = 1.0;
        s = 0.0;
        return;
    }

    double tau = (Aqq - App) / (2.0 * Apq);

    double t;
    if (tau >= 0.0) {
        t = 1.0 / (tau + std::sqrt(1.0 + tau * tau));
    } else {
        t = 1.0 / (tau - std::sqrt(1.0 + tau * tau));
    }

    c = 1.0 / std::sqrt(1.0 + t * t);
    s = t * c;
}

Matrix buildRotationMatrix(int n, int p, int q, double c, double s) {
    Matrix J(n,n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            J(i, j) = (i == j) ? 1.0 : 0.0; // Shorthand to write Identity
        }
    }
    
    J(p, p) = c;
    J(q, q) = c;
    J(p, q) = s;
    J(q, p) = -s;

    return J;
}

Matrix applyJacobiRotation(const Matrix& A, const Matrix& J) {
    return J.transpose() * A * J;
}

// A is the initial symmetric Matrix, Q is the Matrix of Orthonormal Eigenvectors
void jacobiEigenDecomposition(const Matrix& A, Matrix& Q, Matrix& eigenvalues) {
    
    int p, q;
    double c, s;

    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            Q(i, j) = (i == j) ? 1.0 : 0.0; // Shorthand to write Identity
        }
    }

    const double tol = 1e-10;
    int iter = 0;
    Matrix Awork = A;
    int n = Awork.rows();

    while (true) { // Iterate the Jacobi Eigen Decomposition until biggest value is within a tolerance
        double maxVal = findLargestOffDiagonal(Awork, p, q);
        if (maxVal < tol || iter > 50 * n * n) {
            break;
        }

        computeRotationAngle(Awork(p,p), Awork(q,q), Awork(p,q), c, s); // Computes theta
        Matrix J = buildRotationMatrix(n, p, q, c, s); // Constructs Matrix
        Awork = applyJacobiRotation(Awork, J); // Applies the rotation
        Q = Q * J; // One rotation at a time goes from the basis to the Eigenbasis

        ++iter;
    }

    for (int i = 0; i < Awork.rows(); ++i) {
        eigenvalues(i, 0) = Awork(i, i);
    }
}


int main() {
    Matrix A(5, 5);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0; A(0,3) = 0; A(0,4) = 0;
    A(1,0) = 1; A(1,1) = 3; A(1,2) = 1; A(1,3) = 0; A(1,4) = 0;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 2; A(2,3) = 1; A(2,4) = 0;
    A(3,0) = 0; A(3,1) = 0; A(3,2) = 1; A(3,3) = 3; A(3,4) = 1;
    A(4,0) = 0; A(4,1) = 0; A(4,2) = 0; A(4,3) = 1; A(4,4) = 4;

    int n = A.rows();
    Matrix Q(n,n), eigenvalues(n, 1);

    jacobiEigenDecomposition(A, Q, eigenvalues);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << "Q[" << i << j << "] = " << Q(i, j) << std::endl;
        }
    }
    for (int i = 0; i < n; ++i) {
        std::cout << "eigenvalue[" << i << "] = " << eigenvalues(i, 0) << std::endl;
    }
    return 0;
}


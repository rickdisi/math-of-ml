#include <cmath>
#include <iostream>
#include "../linear_algebra/matrix.hpp"
#include "../linear_algebra/jacobi_eigensolver.hpp"

// Implements the Gradient Descent algorithm

int dim = 6;
Matrix A(dim, dim);

double testFunction(const Matrix& x) {
    Matrix xtAx = x.transpose() * A * x;
    double result = xtAx(0,0);
    return result;
}

Matrix analyticGradient(const Matrix& x) { // Same gradient function in 'gradient_checker.cpp'
    Matrix sum = A + A.transpose();
    Matrix sumx = sum * x;
    return sumx;
}

double computeL(const Matrix& A) { // Computes Lipschitz Constant
    int n = A.rows();
    Matrix Q(n, n);
    Matrix eigenvalues(n, 1);
    jacobiEigenDecomposition(A, Q, eigenvalues);

    double lambdaMax = eigenvalues(0,0);
    for (int i = 1; i < n; ++i) {
        double lambda = eigenvalues(i,0);
        if (lambda > lambdaMax) {
            lambdaMax = lambda;
        }
    }
    double L = 2 * lambdaMax;
    
    return L;
}

Matrix gradientDescent(const Matrix& x0, double eta, double epsilon, int& iterations) { // Main algorithm: gradient descent sequence
    Matrix x = x0;
    iterations = 0;

    while (true) {
        Matrix grad = analyticGradient(x);
        if (norm(grad) < epsilon) {
            break;
        }
        subtractInPlace(x, scale(grad, eta));
        iterations += 1;
    }
    return x;
}


int main() { // Test case

    Matrix M(5, 6);
    M(0,0) = 4;  M(0,1) = -3; M(0,2) = 2;  M(0,3) = 1;  M(0,4) = -5; M(0,5) = 0;
    M(1,0) = -2; M(1,1) = 6;  M(1,2) = 0;  M(1,3) = -4; M(1,4) = 3;  M(1,5) = 5;
    M(2,0) = 1;  M(2,1) = 1;  M(2,2) = -6; M(2,3) = 2;  M(2,4) = 0;  M(2,5) = -3;
    M(3,0) = 5;  M(3,1) = 0;  M(3,2) = 3;  M(3,3) = -1; M(3,4) = 4;  M(3,5) = 2;
    M(4,0) = -3; M(4,1) = 2;  M(4,2) = -1; M(4,3) = 6;  M(4,4) = 1;  M(4,5) = -4;

    A = M.transpose() * M; // Guarantee a symmetric nxn matrix

    double L = computeL(A);
    double eta = 1.0 / L;

    Matrix x0(6, 1);
    x0(0,0) = 2;
    x0(1,0) = -3;
    x0(2,0) = 6;
    x0(3,0) = -8;
    x0(4,0) = 1;
    x0(5,0) = 4;

    const double epsilon = 10e-7;
    int iterations;
    Matrix descent = gradientDescent(x0, eta, epsilon, iterations);
    double fxk = testFunction(descent);
    
    double x0squared = dot(x0, x0); // norm squared
    double twoEtaK = 2 * eta * iterations;
    double ratio = x0squared / twoEtaK;

    std::cout << "f(x_K): " << fxk << std::endl;
    std::cout << "ratio: " << ratio << std::endl;
    std::cout << "iterations: " << iterations << std::endl;

    return 0;
}
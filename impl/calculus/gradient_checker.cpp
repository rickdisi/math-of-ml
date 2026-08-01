#include <iostream>
#include <cmath>
#include "../linear_algebra/matrix.hpp"

// Gradient Checker

int dim = 4;
Matrix A(dim,dim);

Matrix numericalGradient(double (*f)(const Matrix&), const Matrix& x, double h) {
    int n = x.rows();
    
    Matrix grad(n, 1);
    for (int i = 0; i < n; ++i) {
        Matrix xPlus = x;
        Matrix xMinus = x;

        xPlus(i,0) += h;
        xMinus(i,0) -= h;

        grad(i, 0) = (f(xPlus) - f(xMinus)) / (2 * h); // Calculates the finite difference Gradient
    }
    return grad;
}

double testFunction(const Matrix& x) {
    Matrix xtAx = x.transpose() * A * x;
    double result = xtAx(0,0);
    return result;
}

Matrix analyticGradient(const Matrix& x) {
    Matrix sum = A + A.transpose();
    Matrix sumx = sum * x;
    return sumx;
}


int main() {

    Matrix x(dim, 1);
    double h = 10e-5;

    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0; A(0,3) = 3;
    A(1,0) = 1; A(1,1) = 3; A(1,2) = 1; A(1,3) = 6;
    A(2,0) = 8; A(2,1) = 1; A(2,2) = 2; A(2,3) = 1;
    A(3,0) = 7; A(3,1) = 9; A(3,2) = 1; A(3,3) = 5;

    x(0,0) = 6; x(1,0) = 3; x(2,0) = 7; x(3,0) = 9;

    Matrix numerical = numericalGradient(testFunction, x, h);
    Matrix analytic = analyticGradient(x);

    double maxError = 0.0;
    for (int i = 0; i < dim; ++i) {
        std::cout << numerical(i, 0) << " ";
        std::cout << analytic(i, 0) << " " << std::endl;
        double diff = std::fabs(numerical(i,0) - analytic(i,0));
        if (diff > maxError) {
            maxError = diff;
        }
    }
    std::cout << "Max abs error: " << maxError << std::endl;
}
#include <iostream>
#include "jacobi_eigensolver.hpp"


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


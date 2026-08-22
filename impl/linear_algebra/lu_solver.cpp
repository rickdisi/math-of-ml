#include <iostream>
#include "lu_solver.hpp"


int main() {

    Matrix A(2, 2);
    A(0, 0) = 1; A(0, 1) = 1;
    A(1, 0) = 2; A(1, 1) = 3;

    Matrix b(2, 1);
    b(0, 0) = 2;
    b(1, 0) = 5;

    std::vector<int> perm = factoriseLU(A); // A is overwritten in place with LU
    Matrix x = solveLU(A, perm, b);

    for (int i = 0; i < 2; ++i) {
        std::cout << "x[" << i << "] = " << x(i) << std::endl;
    }

    return 0;
}
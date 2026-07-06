#include <iostream>
#include "matrix.hpp"


int main() {
    Matrix x = Matrix(5, 5);
    x(2, 3) = 2222;

    Matrix y = Matrix(5,5);
    y(2,4) = 6969;
    y(2,3) = 6767;

    Matrix z = x + y;
    Matrix w = x * z;
    Matrix v = y.transpose();

    std::cout << (4,2) << std::endl;
    return 0;
}
#include <iostream>
#include <cmath>
#include "matrix.hpp"

// Helpers

// dot(u, v): dot product of two n x 1 column vectors, stored as Matrix(n,1)
double dot(const Matrix& u, const Matrix& v) {
    double sum = 0.0;
    for (int i = 0; i < u.rows(); ++i) {
        sum += u(i, 0) * v(i, 0);
    }
    return sum;
}

// norm(v): sqrt(dot(v,v))
double norm(const Matrix& v) {
    return std::sqrt(dot(v, v));
}

// getColumn(A, k): extract column k of A as an n x 1 Matrix
Matrix getColumn(const Matrix& A, int k) {
    int n = A.rows();
    Matrix col(n, 1);
    for (int i = 0; i < n; ++i) {
        col(i, 0) = A(i, k);
    }
    return col;
}

// setColumn(M, k, v): write n x 1 vector v into column k of M
void setColumn(Matrix& M, int k, const Matrix& v) {
    int n = M.rows();
    for (int i = 0; i < n; ++i) {
        M(i, k) = v(i, 0);
    }
}

// scale(v, c): returns c * v for an n x 1 vector v
Matrix scale(const Matrix& v, double c) {
    int n = v.rows();
    Matrix result(n, 1);
    for (int i = 0; i < n; ++i) {
        result(i, 0) = c * v(i, 0);
    }
    return result;
}

// subtractInPlace(u, w): u -= w, for n x 1 vectors
void subtractInPlace(Matrix& u, const Matrix& w) {
    for (int i = 0; i < u.rows(); ++i) {
        u(i, 0) -= w(i, 0);
    }
}

// Loop

void gramSchmidtQR(const Matrix& A, Matrix& Q, Matrix& R) {
    int n = A.rows(); // assumption: A is square, n x n; not checked

    for (int k = 0; k < n; ++k) {
        Matrix uk = getColumn(A, k);

        for (int j = 0; j < k; ++j) {
            Matrix qj = getColumn(Q, j);
            double Rjk = dot(getColumn(A, k), qj); // R_{jk} := <a_k, q_j>
            R(j, k) = Rjk;
            Matrix proj = scale(qj, Rjk);
            subtractInPlace(uk, proj);
        }

        double Rkk = norm(uk);
        R(k, k) = Rkk;

        Matrix qk = scale(uk, 1.0 / Rkk);
        setColumn(Q, k, qk);
    }
}

// main() verification

int main() {
    int n = 3;
    Matrix A(n, n);

    A(0,0) = 12; A(0,1) = -51; A(0,2) = 4;
    A(1,0) = 6;  A(1,1) = 167; A(1,2) = -68;
    A(2,0) = -4; A(2,1) = 24;  A(2,2) = -41;

    Matrix Q(n, n);
    Matrix R(n, n);

    gramSchmidtQR(A, Q, R);

    std::cout << "Q:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << Q(i, j) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nR:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << R(i, j) << " ";
        }
        std::cout << "\n";
    }

    Matrix QtQ = Q.transpose() * Q;
    std::cout << "\nQ^T Q (should be approx identity):\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << QtQ(i, j) << " ";
        }
        std::cout << "\n";
    }

    Matrix QR = Q * R;
    std::cout << "\nQR (should be approx A):\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << QR(i, j) << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
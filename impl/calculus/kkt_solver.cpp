#include "../linear_algebra/matrix.hpp"
#include "../linear_algebra/lu_solver.hpp"


void assembleKKT(
    const Matrix& Q, const Matrix& A, const Matrix& c, 
    const Matrix& b, Matrix& K, Matrix& d) {
    
    int n = Q.rows();
    int m = A.rows();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            K(i, j) = Q(i, j);
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            K(i, n+j) = -A(j, i);
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            K(n+i, j) = A(i, j);
        }
    }
    for (int i = 0; i < n; ++i) {
        d(i, 0) = -c(i, 0);
    }
    for (int i = 0; i < m; ++i) {
        d(n+i, 0) = b(i, 0);
    }
}

void solveKKT(
    const Matrix& Q, const Matrix& A, const Matrix& c, 
    const Matrix& b, Matrix& x, Matrix& lambda) {
    
    int n = Q.rows();
    int m = A.rows();

    Matrix K(n+m, n+m);
    Matrix d(n+m, 1);

    assembleKKT(Q, A, c, b, K, d);

    std::vector<int> perm = factoriseLU(K);
    Matrix z = solveLU(K, perm, d);

    for (int i = 0; i < n; ++i) {
        x(i, 0) = z(i, 0);
    }
    for (int i = 0; i < m; ++i) {
        lambda(i, 0) = z(n+i, 0);
    }
}


int main() {
    // Test case: min (1/2)x^T Q x + c^T x  s.t.  Ax = b
    // Q built as M^T M for a non-symmetric M = [[2,1],[0,3]], giving a
    // non-identity SPD Q rather than the maximally symmetric I_2 case.
    // M^T M = [[4,2],[2,10]]
    // c = (1,-2) (nonzero, breaking the earlier c=0 triviality)
    // A = (1 1), b = (1)

    int n = 2;
    int m = 1;

    Matrix Q(n, n);
    Q(0,0) = 4; Q(0,1) = 2;
    Q(1,0) = 2; Q(1,1) = 10;

    Matrix A(m, n);
    A(0,0) = 1; A(0,1) = 1;

    Matrix c(n, 1);
    c(0,0) = 1;
    c(1,0) = -2;

    Matrix b(m, 1);
    b(0,0) = 1;

    Matrix x(n, 1);
    Matrix lambda(m, 1);

    solveKKT(Q, A, c, b, x, lambda);

    std::cout << "x:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "  x[" << i << "] = " << x(i, 0) << std::endl;
    }

    std::cout << "lambda:" << std::endl;
    for (int i = 0; i < m; ++i) {
        std::cout << "  lambda[" << i << "] = " << lambda(i, 0) << std::endl;
    }

    // Verification: residuals of the two KKT equations
    Matrix Atlambda(n, 1);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < m; ++j) {
            sum += A(j, i) * lambda(j, 0);
        }
        Atlambda(i, 0) = sum;
    }

    Matrix Qx(n, 1);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += Q(i, j) * x(j, 0);
        }
        Qx(i, 0) = sum;
    }

    std::cout << "residual Qx - A^T lambda + c:" << std::endl;
    for (int i = 0; i < n; ++i) {
        double r1 = Qx(i, 0) - Atlambda(i, 0) + c(i, 0);
        std::cout << "  [" << i << "] = " << r1 << std::endl;
    }

    std::cout << "residual Ax - b:" << std::endl;
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += A(i, j) * x(j, 0);
        }
        double r2 = sum - b(i, 0);
        std::cout << "  [" << i << "] = " << r2 << std::endl;
    }

    return 0;
}
#include <algorithm>
#include "../linear_algebra/matrix.hpp"

struct Layer {
    Matrix A;
    Matrix b;
    bool hasActivation;

    Layer(int n_in, int n_out, bool hasActivation): A(n_out, n_in), b(n_out, 1), hasActivation(hasActivation) {}
};

double reluScalar(double t) {
    return std::max(t, 0.0);
}

double identityScalar(double t) {
    return t;
}

Matrix applyActivation(const Matrix& v, double (*f)(double)) {
    Matrix result(v.rows(), 1);
    for (int i = 0; i < v.rows(); ++i) {
        result(i, 0) = f(v(i, 0));
    }
    return result;
}

struct Network {
    std::vector<Layer> layers;
    double (*activation)(double);

    Matrix forward(const Matrix& x0) {
        Matrix current = x0;
        for (size_t i = 0; i < layers.size(); ++i) {
            Matrix z = layers[i].A * current + layers[i].b;
            if (layers[i].hasActivation) {
                current = applyActivation(z, activation);
            } else {
                current = z;
            }
        }
        return current;
    }
};


Matrix softmax(const Matrix& z) {

    double m = z(0, 0);
    for (int i = 1; i < z.rows(); ++i) {
        m = std::max(m, z(i, 0));
    }

    Matrix result(z.rows(), 1);
    double sum = 0.0;
    for (int i = 0; i < z.rows(); ++i) {
        result(i, 0) = std::exp(z(i, 0) - m);
        sum += result(i, 0);
    }

    for (int i = 0; i < z.rows(); ++i) {
        result(i, 0) = result(i, 0) / sum;
    }

    return result;
}


int main() {

    Network net;
    net.layers.push_back(Layer(2, 2, true));
    net.layers.push_back(Layer(2, 2, false));
    net.activation = identityScalar;

    // Layer 0: A (2x2), b (2x1)
    net.layers[0].A(0,0) = 1; net.layers[0].A(0,1) = 2;
    net.layers[0].A(1,0) = 3; net.layers[0].A(1,1) = 4;
    net.layers[0].b(0,0) = 1;
    net.layers[0].b(1,0) = -1;

    // Layer 1: A (2x2), b (2x1)
    net.layers[1].A(0,0) = 3; net.layers[1].A(0,1) = 3;
    net.layers[1].A(1,0) = -5; net.layers[1].A(1,1) = -1;
    net.layers[1].b(0,0) = 2;
    net.layers[1].b(1,0) = 3;

    // Input
    Matrix x0(2, 1);
    x0(0,0) = 1;
    x0(1,0) = 1;

    // Expected results
    Matrix expected = net.layers[1].A * (net.layers[0].A * x0 + net.layers[0].b) + net.layers[1].b;

    Matrix output = net.forward(x0);

    std::cout << "net.forward(x0):" << std::endl;
    for (int i = 0; i < output.rows(); ++i) {
        std::cout << "  [" << i << "] = " << output(i, 0) << std::endl;
    }

    std::cout << "expected:" << std::endl;
    for (int i = 0; i < expected.rows(); ++i) {
        std::cout << "  [" << i << "] = " << expected(i, 0) << std::endl;
    }

    // Softmax check
    Matrix probs = softmax(output);

    std::cout << "softmax(net.forward(x0)):" << std::endl;
    double sum = 0.0;
    for (int i = 0; i < probs.rows(); ++i) {
        std::cout << "  [" << i << "] = " << probs(i, 0) << std::endl;
        sum += probs(i, 0);
    }

    std::cout << "sum of softmax entries: " << sum << std::endl;

    return 0;
}
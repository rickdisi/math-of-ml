#include "../neural_networks/network.hpp"


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
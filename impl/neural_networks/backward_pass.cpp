#include <iostream>
#include <cmath>
#include "../linear_algebra/matrix.hpp"
#include "../neural_networks/network.hpp"


int main() {
    // Main test function written by Claude
    // Step 1: build the network
    Network net;
    net.layers.push_back(Layer(2, 3, true));   // hidden layer, ReLU
    net.layers.push_back(Layer(3, 2, false));  // output layer, no activation

    net.layers[0].A(0,0) = 0.5;  net.layers[0].A(0,1) = -0.3;
    net.layers[0].A(1,0) = 0.2;  net.layers[0].A(1,1) = 0.8;
    net.layers[0].A(2,0) = -0.1; net.layers[0].A(2,1) = 0.4;
    net.layers[0].b(0,0) = 0.1;
    net.layers[0].b(1,0) = -0.2;
    net.layers[0].b(2,0) = 0.05;

    net.layers[1].A(0,0) = 1.0; net.layers[1].A(0,1) = -0.5; net.layers[1].A(0,2) = 0.3;
    net.layers[1].A(1,0) = -0.2; net.layers[1].A(1,1) = 0.7; net.layers[1].A(1,2) = 0.1;
    net.layers[1].b(0,0) = 0.0;
    net.layers[1].b(1,0) = 0.0;

    net.activation = reluScalar;
    net.activationDerivative = reluScalarDerivative;

    // Step 2: test case
    Matrix x0(2, 1);
    x0(0,0) = 1.0;
    x0(1,0) = -0.5;
    int y = 1; // valid class index, output layer has width 2

    // Step 3: scalar loss function
    auto totalLoss = [](Network& net, const Matrix& x0, int y) -> double {
        Matrix zL = net.forward(x0);
        Matrix probs = softmax(zL);
        return -std::log(probs(y, 0));
    };

    // Step 4: analytic gradients
    std::vector<LayerGradients> grads = net.backward(x0, y);

    double h = 1e-5;

    // Steps 5-7: spot-check net.layers[0].A(0,0)
    {
        double original = net.layers[0].A(0,0);

        net.layers[0].A(0,0) = original + h;
        double lossPlus = totalLoss(net, x0, y);

        net.layers[0].A(0,0) = original - h;
        double lossMinus = totalLoss(net, x0, y);

        net.layers[0].A(0,0) = original; // restore

        double numerical = (lossPlus - lossMinus) / (2*h);
        double analytic = grads[0].gradLA(0,0);

        std::cout << "layers[0].A(0,0): numerical = " << numerical
                   << ", analytic = " << analytic
                   << ", diff = " << std::abs(numerical - analytic) << std::endl;
    }

    // Step 8: additional spot-checks
    {
        // one entry in the other layer
        double original = net.layers[1].A(1,2);

        net.layers[1].A(1,2) = original + h;
        double lossPlus = totalLoss(net, x0, y);

        net.layers[1].A(1,2) = original - h;
        double lossMinus = totalLoss(net, x0, y);

        net.layers[1].A(1,2) = original;

        double numerical = (lossPlus - lossMinus) / (2*h);
        double analytic = grads[1].gradLA(1,2);

        std::cout << "layers[1].A(1,2): numerical = " << numerical
                   << ", analytic = " << analytic
                   << ", diff = " << std::abs(numerical - analytic) << std::endl;
    }

    {
        // one bias entry
        double original = net.layers[0].b(2,0);

        net.layers[0].b(2,0) = original + h;
        double lossPlus = totalLoss(net, x0, y);

        net.layers[0].b(2,0) = original - h;
        double lossMinus = totalLoss(net, x0, y);

        net.layers[0].b(2,0) = original;

        double numerical = (lossPlus - lossMinus) / (2*h);
        double analytic = grads[0].gradLb(2,0);

        std::cout << "layers[0].b(2,0): numerical = " << numerical
                   << ", analytic = " << analytic
                   << ", diff = " << std::abs(numerical - analytic) << std::endl;
    }

    return 0;
}
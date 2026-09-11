#include <iostream>
#include <vector>
#include "../linear_algebra/matrix.hpp"
#include "../neural_networks/network.hpp"
#include "../mnist/mnist_loader.hpp"

int main() {
    std::vector<Matrix> trainImages = loadImages("impl/mnist/data/mnist/train-images-idx3-ubyte");
    std::vector<int> trainLabels = loadLabels("impl/mnist/data/mnist/train-labels-idx1-ubyte");

    Network net;
    net.layers.push_back(Layer(784, 128, true));
    net.layers.push_back(Layer(128, 10, false));

    initializeWeights(net);

    int numEpochs = 3;
    net.activation = reluScalar;
    net.activationDerivative = reluScalarDerivative;

    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        for (size_t i = 0; i < trainImages.size(); ++i) {
            net.backward(trainImages[i], trainLabels[i]);
            // TODO: apply gradients (M15)
            std::cout << "epoch " << epoch << " example " << i << "\n";
        }
    }

    return 0;
}
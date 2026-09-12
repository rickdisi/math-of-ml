#include <iostream>
#include <vector>
#include "../linear_algebra/matrix.hpp"
#include "../neural_networks/network.hpp"
#include "../mnist/mnist_loader.hpp"

int main() {
    std::vector<Matrix> trainImages = loadImages("impl/mnist/data/mnist/train-images-idx3-ubyte");
    std::vector<int> trainLabels = loadLabels("impl/mnist/data/mnist/train-labels-idx1-ubyte");
    std::vector<Matrix> testImages = loadImages("impl/mnist/data/mnist/t10k-images-idx3-ubyte");
    std::vector<int> testLabels = loadLabels("impl/mnist/data/mnist/t10k-labels-idx1-ubyte");

    Network net;
    net.layers.push_back(Layer(784, 128, true));
    net.layers.push_back(Layer(128, 10, false));

    initializeWeights(net);

    net.activation = reluScalar;
    net.activationDerivative = reluScalarDerivative;

    int numEpochs = 3;
    size_t batchSize = 32;
    double learningRate = 0.1;

    size_t N = trainImages.size();
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(std::random_device{}());

    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t batchStart = 0; batchStart < N; batchStart += batchSize) {
            size_t batchEnd = std::min(batchStart + batchSize, N);
            size_t actualBatchSize = batchEnd - batchStart;

            std::vector<LayerGradients> accum;
            for (size_t i = 0; i < net.layers.size(); ++i) {
                accum.push_back(LayerGradients(
                    Matrix(net.layers[i].A.rows(), net.layers[i].A.cols()),
                    Matrix(net.layers[i].b.rows(), net.layers[i].b.cols())
                ));
            }

            for (size_t k = batchStart; k < batchEnd; ++k) {
                size_t idx = indices[k];
                std::vector<LayerGradients> exampleGrads = net.backward(trainImages[idx], trainLabels[idx]);

                for (size_t i = 0; i < net.layers.size(); ++i) {
                    accum[i].gradLA = accum[i].gradLA + exampleGrads[i].gradLA;
                    accum[i].gradLb = accum[i].gradLb + exampleGrads[i].gradLb;
                }
            }

            for (size_t i = 0; i < net.layers.size(); ++i) {
                accum[i].gradLA = scale(accum[i].gradLA, 1.0 / static_cast<double>(actualBatchSize));
                accum[i].gradLb = scale(accum[i].gradLb, 1.0 / static_cast<double>(actualBatchSize));
            }

            applyGradients(net, accum, learningRate);
        }

        std::cout << "epoch " << epoch << " complete\n";
        double accuracy = evaluateAccuracy(net, testImages, testLabels);
        std::cout << "epoch " << epoch << " test accuracy: " << accuracy << "\n";
    }

    return 0;
}
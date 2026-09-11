#include <fstream>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include "../linear_algebra/matrix.hpp"
#include "../neural_networks/network.hpp"


int32_t readInt32BigEndian(std::ifstream& stream) {
    unsigned char buf[4];
    stream.read(reinterpret_cast<char*>(buf), 4);
    uint32_t value = (static_cast<uint32_t>(buf[0]) << 24)
                    | (static_cast<uint32_t>(buf[1]) << 16)
                    | (static_cast<uint32_t>(buf[2]) << 8)
                    |  static_cast<uint32_t>(buf[3]);
    int32_t result;
    std::memcpy(&result, &value, sizeof(result));

    return result;
}

struct ImageHeader {
    int32_t magic;
    int32_t numImages;
    int32_t rows;
    int32_t cols;
};

ImageHeader readImageHeader(std::ifstream& file) {
    int32_t magic = readInt32BigEndian(file);
    int32_t numImages = readInt32BigEndian(file);
    int32_t rows = readInt32BigEndian(file);
    int32_t cols = readInt32BigEndian(file);

    if (magic != 2051) {
        throw std::runtime_error("Invalid MNIST image file: bad magic number");
    }

    return ImageHeader{magic, numImages, rows, cols};
}

std::vector<Matrix> loadImages(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open image file: " + path);
    }

    ImageHeader header = readImageHeader(file);

    size_t pixelCount = static_cast<size_t>(header.numImages)
                       * static_cast<size_t>(header.rows)
                       * static_cast<size_t>(header.cols);

    std::vector<unsigned char> buffer(pixelCount);
    file.read(reinterpret_cast<char*>(buffer.data()), pixelCount);

    std::vector<Matrix> images;
    images.reserve(header.numImages);

    size_t imageSize = static_cast<size_t>(header.rows) * static_cast<size_t>(header.cols);

    for (int32_t i = 0; i < header.numImages; ++i) {
        Matrix img(imageSize, 1);
        size_t offset = static_cast<size_t>(i) * imageSize;
        for (size_t p = 0; p < imageSize; ++p) {
            img(p, 0) = static_cast<double>(buffer[offset + p]) / 255.0;
        }
        images.push_back(img);
    }

    return images;
}

struct LabelHeader {
    int32_t magic;
    int32_t numLabels;
};

LabelHeader readLabelHeader(std::ifstream& file) {
    int32_t magic = readInt32BigEndian(file);
    int32_t numLabels = readInt32BigEndian(file);

    if (magic != 2049) {
        throw std::runtime_error("Invalid MNIST label file: bad magic number");
    }

    return LabelHeader{magic, numLabels};
}

std::vector<int> loadLabels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open label file: " + path);
    }

    LabelHeader header = readLabelHeader(file);

    size_t labelCount = static_cast<size_t>(header.numLabels);

    std::vector<unsigned char> buffer(labelCount);
    file.read(reinterpret_cast<char*>(buffer.data()), labelCount);

    std::vector<int> labels(labelCount);
    for (size_t i = 0; i < labelCount; ++i) {
        labels[i] = static_cast<int>(buffer[i]);
    }

    return labels;
}

void initializeWeights(Network& net) {
    std::mt19937 engine(std::random_device{}());

    for (Layer& layer : net.layers) {
        size_t n_out = layer.A.rows();
        size_t n_in = layer.A.cols();

        double limit = 1.0 / std::sqrt(static_cast<double>(n_in));
        std::uniform_real_distribution<double> dist(-limit, limit);

        for (size_t r = 0; r < n_out; ++r) {
            for (size_t c = 0; c < n_in; ++c) {
                layer.A(r, c) = dist(engine);
            }
        }
        // b stays at zero per convention.
    }
}
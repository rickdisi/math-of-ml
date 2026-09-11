#pragma once

#include <algorithm>
#include "../linear_algebra/matrix.hpp"


struct LayerGradients {
    Matrix gradLA;
    Matrix gradLb;

    LayerGradients(Matrix gradA, Matrix gradB): gradLA(gradA), gradLb(gradB) {}
};


double reluScalar(double t) {
    return std::max(t, 0.0);
}
double reluScalarDerivative(double t) {
    double tPrime;
    // ReLU is not differentiable at t=0; ReLU'(0):=0 is a convention 
    if (t <= 0) {
        tPrime = 0;
    }
    else {
        tPrime = 1;
    }

    return tPrime;
}

double identityScalar(double t) {
    return t;
}
double identityScalarDerivative() {
    return 1;
}

Matrix applyActivation(const Matrix& v, double (*f)(double)) {
    Matrix result(v.rows(), 1);
    for (int i = 0; i < v.rows(); ++i) {
        result(i, 0) = f(v(i, 0));
    }
    return result;
}

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

Matrix hadamard(const Matrix& u, const Matrix& v) {

    Matrix hadprod(u.rows(), 1);
    for (int j = 0; j < u.rows(); ++j) {
        hadprod(j, 0) = u(j, 0) * v(j, 0);
    }
    return hadprod;
}

Matrix outputLayerError(const Matrix& zL, int y) {
    
    int K = zL.rows();
    Matrix pHat(K, 1);
    pHat(y, 0) = 1;
    
    Matrix qThetaX = softmax(zL);
    subtractInPlace(qThetaX, pHat);

    Matrix deltaL = qThetaX;
    return deltaL;
}

Matrix backwardStep(const Matrix& aNext, const Matrix& deltaNext, const Matrix& z, double (*f)(double)) {
    Matrix sigmaPrime = applyActivation(z, f);
    Matrix delta = hadamard(aNext.transpose() * deltaNext, sigmaPrime);
    
    return delta;
}

LayerGradients parameterGradients(const Matrix& delta, const Matrix& xMinusOne) {
    Matrix gradLA = delta * xMinusOne.transpose();
    Matrix gradLb = delta;

    return LayerGradients(gradLA, gradLb);
}


struct Layer {
    Matrix A;
    Matrix b;
    bool hasActivation;

    Layer(int n_in, int n_out, bool hasActivation): A(n_out, n_in), b(n_out, 1), hasActivation(hasActivation) {}
};

struct ForwardCache {
    std::vector<Matrix> preActivations;
    std::vector<Matrix> postActivations;
};

struct Network {
    std::vector<Layer> layers;
    double (*activation)(double);
    double (*activationDerivative)(double);

    Matrix forward(const Matrix& x0) {
        ForwardCache cache = forwardTrace(x0);
        Matrix current = cache.postActivations.back();

        return current;
    }

    ForwardCache forwardTrace(const Matrix& x0) {
        ForwardCache cache;
        cache.postActivations.push_back(x0);
        
        Matrix current = x0;
        for (size_t i = 0; i < layers.size(); ++i) {
            Matrix z = layers[i].A * current + layers[i].b;
            cache.preActivations.push_back(z);

            if (layers[i].hasActivation) {
                current = applyActivation(z, activation);
            } 
            else {
                current = z;
            }
            cache.postActivations.push_back(current);
        }
        return cache;
    }

    std::vector<LayerGradients> backward(const Matrix& x0, int y) {
        ForwardCache cache = forwardTrace(x0);
        int N = layers.size();

        Matrix delta = outputLayerError(cache.preActivations.back(), y);

        std::vector<LayerGradients> grads;
        grads.push_back(parameterGradients(delta, cache.postActivations[N-1]));
        // loop index i (0-indexed) corresponds to math layer i+1; the weight matrix needed here is A_{i+2}, i.e. layers[i+1].A.
        for (int i = N - 2; i >= 0; --i) {
            delta = backwardStep(layers[i+1].A, delta, cache.preActivations[i], activationDerivative);
            grads.push_back(parameterGradients(delta, cache.postActivations[i]));
        }
        std::reverse(grads.begin(), grads.end());
        return grads;
    }
};
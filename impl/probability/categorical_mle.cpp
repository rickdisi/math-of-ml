#include <iostream>
#include <vector>
#include <random>
#include <cmath>

std::vector<int> sampleCategorical(const std::vector<double>& p, int n, std::mt19937& rng) {
    
    std::vector<double> cdf(p.size());
    double runningTotal = 0.0;
    for (size_t k = 0; k < p.size(); ++k) {
        runningTotal += p[k];
        cdf[k] = runningTotal; 
    }
    std::vector<int> samples(n);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        double u = dist(rng);

        samples[i] = static_cast<int>(cdf.size()) - 1;
        for (size_t k = 0; k < cdf.size(); ++k) {
            if (cdf[k] >= u) {
                samples[i] = static_cast<int>(k);
                break;
            }
        }
    }
    return samples;
}

std::vector<int> countOccurrences(const std::vector<int>& samples, int K) {
    std::vector<int> counts(K, 0);

    for (int s : samples) {
        counts[s]++;
    }
    return counts;
}

std::vector<double> computeMLE(const std::vector<int>& counts, int n) {
    std::vector<double> pHat(counts.size());

    for (size_t k = 0; k < counts.size(); ++k) {
        pHat[k] = static_cast<double>(counts[k]) / n;
    }
    return pHat;
}

double logLikelihood(const std::vector<int>& counts, const std::vector<double>& p) {
    double logLik = 0.0;

    for (size_t k = 0; k < counts.size(); ++k) {
        logLik += counts[k] * std::log(p[k]);
    }

    return logLik;
}

std::vector<double> perturb(const std::vector<double>& p, int i, int j, double epsilon) {
    std::vector<double> result = p;   // makes a copy of p

    result[i] = result[i] - epsilon;
    result[j] = result[j] + epsilon;

    return result;
}

int main() {
    std::mt19937 rng(42);

    // True distribution and sample size
    std::vector<double> pTrue = {0.5, 0.3, 0.2};
    int n = 1000;
    int K = static_cast<int>(pTrue.size());

    // Step 1: draw samples
    std::vector<int> samples = sampleCategorical(pTrue, n, rng);

    // Step 2: compute counts and the MLE
    std::vector<int> counts = countOccurrences(samples, K);
    std::vector<double> pHat = computeMLE(counts, n);

    // Piece 1: print the MLE against the true distribution
    for (size_t k = 0; k < pHat.size(); ++k) {
        std::cout << "p_hat[" << k << "] = " << pHat[k]
                   << "  (true p[" << k << "] = " << pTrue[k] << ")" << std::endl;
    }

    // Compute log-likelihood at the MLE
    double llAtMLE = logLikelihood(counts, pHat);

    // Piece 2: print it
    std::cout << "log-likelihood at MLE: " << llAtMLE << std::endl;

    // Piece 3: perturb away from the MLE in three directions and confirm
    // the log-likelihood only ever decreases (or stays equal), consistent
    // with prop:categorical-mle claiming pHat is the maximiser.
    double epsilon = 0.01;

    std::vector<double> pPerturbed01 = perturb(pHat, 0, 1, epsilon);
    double llPerturbed01 = logLikelihood(counts, pPerturbed01);
    std::cout << "perturbed (0->1) log-likelihood: " << llPerturbed01
               << "  (vs MLE: " << llAtMLE << ")" << std::endl;

    std::vector<double> pPerturbed12 = perturb(pHat, 1, 2, epsilon);
    double llPerturbed12 = logLikelihood(counts, pPerturbed12);
    std::cout << "perturbed (1->2) log-likelihood: " << llPerturbed12
               << "  (vs MLE: " << llAtMLE << ")" << std::endl;

    std::vector<double> pPerturbed20 = perturb(pHat, 2, 0, epsilon);
    double llPerturbed20 = logLikelihood(counts, pPerturbed20);
    std::cout << "perturbed (2->0) log-likelihood: " << llPerturbed20
               << "  (vs MLE: " << llAtMLE << ")" << std::endl;

    return 0;
}
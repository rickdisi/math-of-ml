# The Mathematics of Machine Learning

A self-directed study project building rigorously from linear algebra through to a working neural network, implemented in C++.

## End product

A single expository PDF (~40 pages) in which every result in the neural networks chapter traces back to something proved in the preceding chapters. Accompanied by from-scratch C++ implementations with no external libraries.

## Topics

1. Linear algebra — rank-nullity, LU decomposition, spectral theorem, SVD
2. Calculus and optimisation — chain rule, gradient descent convergence
3. Probability and statistics — MLE, cross-entropy derived from first principles
4. Neural networks — forward pass, backpropagation, Universal Approximation Theorem

## Structure

```
paper/      # the end product — LaTeX source for the expository paper
weekly/     # rough working notes per topic
impl/       # from-scratch C++ implementations, one file per algorithm
```

## Tooling

- C++17, no external libraries
- Build: `make run FILE=impl/<topic>/<file>.cpp`
- LaTeX for mathematical exposition

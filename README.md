# The Mathematics of Machine Learning

A self-directed study project building rigorously from linear algebra through to a working neural network — every result proved from first principles, every algorithm implemented from scratch in C++.

## Overview

The end product is a single expository paper (~40 pages) with a coherent narrative arc: each chapter earns the next, and every result in the neural networks chapter traces back to something proved earlier in the paper. No result is stated without proof or derivation, and no algorithm is used without having been implemented from scratch — no linear algebra, calculus, or ML libraries anywhere in the codebase, only the C++ standard library.

The paper stands on its own; the C++ implementations are a companion that demonstrates the mathematics actually works, and doubles as practice with systems-level C++ (manual memory management, RAII, cache-friendly data layout).

## Topics

1. **Linear algebra** — rank-nullity theorem, LU decomposition, Inner Product spaces, spectral theorem for symmetric matrices, SVD
2. **Calculus and optimisation** — multivariable chain rule, gradient descent convergence under L-smoothness and convexity, Lagrange multipliers
3. **Probability and statistics** — probability spaces, maximum likelihood estimation, cross-entropy loss derived from MLE, Jensen's inequality, KL divergence
4. **Neural networks** — forward pass, backpropagation derived from the chain rule, Universal Approximation Theorem, trained on MNIST

## Progress

16 milestones in total, each pairing a piece of mathematics with a from-scratch C++ implementation.

- [X] **M1** — Vector spaces and linear maps; rank-nullity theorem — `Matrix` class
- [X] **M2** — LU decomposition with partial pivoting — LU solver
- [X] **M3** — Inner product spaces, Gram-Schmidt, QR decomposition — QR solver
- [X] **M4** — Eigenvalues, spectral theorem for symmetric matrices — Jacobi eigensolver
- [X] **M5** — SVD derived from the eigendecomposition of $A^\top A$ — SVD via Jacobi on $A^\top A$
- [X] **M6** — Multivariate calculus, chain rule, gradient as steepest ascent — numerical gradient checker
- [X] **M7** — Gradient descent convergence under L-smoothness and convexity — gradient descent
- [ ] **M8** — Lagrange multipliers — KKT solver reusing the M2 LU solver
- [ ] **M9–M10** — Probability foundations, MLE and cross-entropy
- [ ] **M11–M13** — Network architecture, backpropagation, Universal Approximation Theorem
- [ ] **M14–M15** — MNIST data pipeline and training, from scratch
- [ ] **M16** — Final paper polish

## Structure

```
math-of-ml/
├── paper/                # the end product — LaTeX source for the expository paper
│   ├── main.tex
│   ├── preamble.tex
│   ├── chapters/
│   └── references.bib
└── impl/                 # from-scratch C++ implementations, one file per algorithm
    ├── linear_algebra/
    ├── calculus/
    ├── probability/
    └── neural_networks/
```

## Tooling

- **C++17**, one `.cpp` file per algorithm, no external libraries — only the standard library
- Build and run: `make run FILE=impl/<topic>/<file>.cpp`
- **LaTeX** for the mathematical exposition, with C++ snippets included via `listings`

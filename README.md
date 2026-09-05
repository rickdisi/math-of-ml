# The Mathematics of Machine Learning

A self-directed study project building rigorously from linear algebra through to a working neural network — every result proved from first principles, every algorithm implemented from scratch in C++.

## Overview

The end product is a single expository paper with a coherent narrative arc: each chapter earns the next, and every result in the neural networks chapter traces back to something proved earlier in the paper. No result is stated without proof or derivation, and no algorithm is used without having been implemented from scratch — no linear algebra, calculus, or ML libraries anywhere in the codebase, only the C++ standard library.

The paper stands on its own; the C++ implementations are a companion that demonstrates the mathematics actually works, and doubles as practice with systems-level C++ (manual memory management, RAII, cache-friendly data layout). Originally scoped at roughly 40 pages, the paper has grown to 81 as of Chapter 4's opening section — a consequence of writing the exposition at full rigour rather than trimming it to fit the original estimate.

## Topics

1. **Linear algebra** — vector spaces and linear operators, rank-nullity theorem, LU decomposition with partial pivoting, inner product spaces and Gram-Schmidt, QR decomposition, eigenvalues and the spectral theorem for symmetric matrices, SVD derived from eigendecomposition
2. **Calculus and optimisation** — the multivariable chain rule (the mathematical heart of backpropagation), the gradient as the direction of steepest ascent, convergence of gradient descent under L-smoothness and convexity, Lagrange multipliers and constrained optimisation
3. **Probability and statistics** — probability spaces and Kolmogorov's axioms, conditional probability and independence, Bernoulli and categorical distributions, maximum likelihood estimation as a principled framework, cross-entropy loss derived from categorical MLE, Jensen's inequality, KL divergence
4. **Neural networks** — affine maps and why composing them alone gains nothing, activation functions, the feedforward network architecture, softmax as a categorical distribution, backpropagation derived from the chain rule, the Universal Approximation Theorem via a ReLU staircase/bump construction, trained on MNIST

## Progress

16 milestones in total, each pairing a piece of mathematics with a from-scratch C++ implementation. 11 of 16 complete.

- [X] **M1** — Vector spaces and linear maps, rank-nullity theorem — `Matrix` class
- [X] **M2** — LU decomposition with partial pivoting — LU solver
- [X] **M3** — Inner product spaces, Gram-Schmidt, QR decomposition — QR solver
- [X] **M4** — Eigenvalues, spectral theorem for symmetric matrices — Jacobi eigensolver
- [X] **M5** — SVD derived from the eigendecomposition of $A^\top A$ — SVD via Jacobi on $A^\top A$
- [X] **M6** — Multivariate calculus, chain rule, gradient as steepest ascent — numerical gradient checker
- [X] **M7** — Gradient descent convergence under L-smoothness and convexity — gradient descent
- [X] **M8** — Lagrange multipliers, constrained optimisation — KKT solver, reusing the M2 LU solver
- [X] **M9** — Probability spaces, conditional probability, Bernoulli/categorical distributions, Jensen's inequality, KL divergence — no C++ deliverable
- [X] **M10** — Maximum likelihood estimation, cross-entropy derived from categorical MLE — categorical MLE demo
- [X] **M11** — Network architecture: affine maps, activations, feedforward networks, softmax output layer — forward pass
- [ ] **M12** — Backpropagation derived rigorously from the chain rule — backward pass
- [ ] **M13** — Universal Approximation Theorem via a ReLU staircase/bump construction — no C++ deliverable
- [ ] **M14** — MNIST data loading, weight initialisation, training scaffold — IDX binary loader, scaffold
- [ ] **M15** — MNIST training and evaluation — mini-batch SGD, end-to-end run
- [ ] **M16** — Final paper polish, figures, narrative coherence pass

## Structure

```
math-of-ml/
├── paper/                     # LaTeX source for the expository paper
│   ├── main.tex
│   ├── preamble.tex
│   ├── appendices.tex         # C++ source listings, one section per algorithm
│   ├── chapters/
│   │   ├── 00_preface.tex
│   │   ├── 01_linear_algebra.tex
│   │   ├── 02_calculus.tex
│   │   ├── 03_probability.tex
│   │   └── 04_neural_networks.tex
│   └── references.bib
└── impl/                      # C++ implementations, one file per algorithm
    ├── linear_algebra/        # Matrix class, LU solver, QR, Jacobi eigensolver, SVD
    ├── calculus/              # gradient checker, gradient descent, KKT solver
    ├── probability/           # categorical MLE
    └── neural_networks/       # forward pass
```

## Tooling

- **C++17**, one `.cpp` file per algorithm, no external libraries — only the standard library
- Build and run: `make run FILE=impl/<topic>/<file>.cpp`
- **LaTeX** for the mathematical exposition, with C++ snippets included via `listings`; every implementation referenced in the paper is included verbatim in the appendix, so the paper and the code never drift apart

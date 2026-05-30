# Preference-shaped expected hypervolume and R2 improvement acquisitions

This small reference implementation accompanies the JMLR/arXiv development of
preference-shaped expected indicator improvement.  It implements transparent
Python routines for the acquisition functions in the comparison table:

- canonical two-objective EHVI under independent Gaussian objectives;
- product-kernel / desirability-shaped EHVI using aspiration and reservation levels;
- cone-based EHVI via a simplicial cone transformation;
- discrete R2 improvement and ER2I in achievement space;
- exact/integral R2 ER2I in two objectives via quadrature under an independent objective-space Gaussian predictive distribution;
- Monte Carlo checks for the deterministic quadrature routines.

The implementation is intentionally explicit and theorem-oriented.  It is meant
as a reference artifact, not as an optimized Bayesian optimization package.

## Convention

All objective-space routines use **minimization**.  Smaller objective values are
better.

Hypervolume uses a worse/dystopian reference point `reference = r`.  In two
objectives, the dominated box of a point `y` is

```text
[y_1, r_1] x [y_2, r_2].
```

R2 uses a utopian point `z_plus` and the weighted Tchebycheff achievement
function

```text
g_lambda(y; z_plus) = max_i lambda_i * (y_i - z_plus_i).
```

For a set `A`, the current scalarization envelope is

```text
h_A(lambda) = min_{a in A} g_lambda(a; z_plus).
```

## Dependencies

```bash
pip install numpy scipy
```

No plotting package is required for the computational examples.

## Main file

```text
preference_shaped_acquisitions.py
```

Run the built-in example suite:

```bash
python preference_shaped_acquisitions.py
```

The examples use the same basic data as the TikZ figures in the report/slides:

```python
A = np.array([[1.0, 3.5], [2.0, 2.5], [3.0, 1.5]])
reference = np.array([5.0, 4.0])
mu = np.array([2.0, 1.5])
sigma = np.array([0.7, 0.6])
z_plus = np.array([0.0, 0.0])
```

The script prints deterministic quadrature values and Monte Carlo estimates.
The Monte Carlo checks use a fixed seed.  A typical output is of the form

```text
canonical_ehvi_2d                          deterministic=1.7823017  MC=1.8137479  SE=0.0242  |diff|/SE=1.30
product_desirability_weighted_ehvi_2d      deterministic=0.15334904 MC=0.15397905 SE=0.00148 |diff|/SE=0.42
cone_ehvi_2d                               deterministic=4.2784268  MC=4.3295596  SE=0.0632  |diff|/SE=0.81
integral_er2i_objective_gaussian_2d        deterministic=0.13234087 MC=0.13613621 SE=0.0019  |diff|/SE=2.00
```

As a rule of thumb, the deterministic value should lie within a few Monte Carlo
standard errors.  Monte Carlo is used only as an empirical check, not as the
primary definition of the acquisition.

## API overview

### Canonical EHVI

```python
ehvi_2d(mu, sigma, points, reference, n_gh=40)
mc_ehvi_2d(mu, sigma, points, reference, n_samples=100_000, seed=12345)
```

Computes

```text
E[HVI(Y, A; r)]
```

for independent objective-space Gaussian coordinates

```text
Y_i ~ N(mu_i, sigma_i^2).
```

The deterministic implementation uses tensor-product Gauss--Hermite quadrature.
The Monte Carlo routine samples objective vectors and evaluates deterministic
hypervolume improvement.

### Product-kernel / desirability-weighted EHVI

```python
AspirationReservationDesirability(aspiration, reservation, epsilon=0.05)
weighted_ehvi_product_2d(mu, sigma, points, desirability, n_gh=40)
mc_weighted_ehvi_product_2d(mu, sigma, points, desirability, n_samples=100_000)
```

For minimization, the desirability map satisfies

```text
D_i(aspiration_i) = 1,
D_i(reservation_i) = 0.
```

The default transition is a smooth cubic.  The parameter `epsilon=0.05` creates
small endpoint plateaus, following the aspiration--reservation spirit of the
D-PHI examples.  Product-kernel weighted hypervolume is computed as ordinary
maximization hypervolume in desirability space.

Example:

```python
des = AspirationReservationDesirability(
    aspiration=np.array([1.0, 1.0]),
    reservation=np.array([5.0, 4.0]),
    epsilon=0.05,
)
value = weighted_ehvi_product_2d(mu, sigma, A, des)
```

### Cone-based EHVI

```python
cone_ehvi_2d(mu, sigma, points, reference, cone_matrix, n_gh=40)
mc_cone_ehvi_2d(mu, sigma, points, reference, cone_matrix, n_samples=100_000)
```

The columns of `cone_matrix` are the cone generators.  With

```text
L = inverse(cone_matrix),
```

cone dominance is transformed into ordinary Pareto dominance in `L`-coordinates.
The deterministic expectation is evaluated by Gauss--Hermite quadrature over the
original independent objective-space Gaussian prediction.

Example:

```python
cone_matrix = np.array([[1.0, -0.35], [-0.35, 1.0]])
value = cone_ehvi_2d(mu, sigma, A, reference, cone_matrix)
```

### Discrete R2 and achievement-space ER2I

```python
make_weights_2d(n_weights, include_endpoints=True)
r2_envelope(points, z_plus, weights)
r2_improvement_discrete(candidate, points, z_plus, weights)
er2i_discrete_achievement(mu_w, sigma_w, h_A_w)
```

For discrete ER2I in achievement space, the inputs are already indexed by
weights:

```text
Z_k(x) ~ N(mu_w[k], sigma_w[k]^2),
h_A_w[k] = h_A(lambda_k).
```

The acquisition is

```text
ER2I_K(x) = (1/K) sum_k EI(h_A(lambda_k); mu_w[k], sigma_w[k]).
```

This implementation is the cleanest computational expression of the
achievement-space model: one scalar Gaussian expected improvement per weight.

### Exact/integral R2 ER2I in 2D under objective-space Gaussians

```python
er2i_integral_objective_gaussian_2d(mu, sigma, points, z_plus, n_lambda=64)
mc_er2i_integral_objective_gaussian_2d(mu, sigma, points, z_plus, n_lambda=128)
```

This implements the two-dimensional exact/integral R2 acquisition with an
independent objective-space Gaussian model.  For each lambda in `(0,1)`, the
inner layer-cake integral is

```text
int_{-infty}^{h_A(lambda)}
    Phi((z_1^+ + t/lambda - mu_1)/sigma_1)
    Phi((z_2^+ + t/(1-lambda) - mu_2)/sigma_2)
 dt.
```

The outer integral over `lambda` is Gauss--Legendre quadrature.  Boundary weights
`lambda=0` and `lambda=1` are avoided by the quadrature nodes.  This is the
objective-space Gaussian route.  It is exact for the model, but more expensive
than achievement-space Gaussian ER2I because each weight requires an inner
one-dimensional integral.

## Optional Yildiz--Suri backend

The file is standalone for all two-dimensional routines.  If you want to use the
attached Yildiz--Suri anchored hypervolume implementation for higher-dimensional
experiments, load it explicitly:

```python
from preference_shaped_acquisitions import load_yildiz_suri_backend

ys = load_yildiz_suri_backend("yildiz_suri_4d_anchor-1 (1).py")
value = ys.anchored_hypervolume(points_anchored, method="auto", prune=True)
```

The optional backend is not required for the examples in this file.  It is kept
separate because the present artifact focuses on two-dimensional acquisition
checks and on the R2 quadrature representation.

## Numerical comments

1. `ehvi_2d`, `weighted_ehvi_product_2d`, and `cone_ehvi_2d` use Gauss--Hermite
   quadrature over the independent Gaussian objective distribution.
2. `er2i_integral_objective_gaussian_2d` uses Gauss--Legendre quadrature over
   weights and adaptive SciPy quadrature for the inner layer-cake integral.
3. Monte Carlo checks are provided for validation, not for production use.
4. The routines are intentionally explicit and suitable for inspecting the
   formulas in the paper.  For large Bayesian optimization runs, caching,
   vectorization, sparse Gaussian processes, and specialized hypervolume
   backends should be added.

## Minimal usage example

```python
import numpy as np
from preference_shaped_acquisitions import (
    ehvi_2d,
    AspirationReservationDesirability,
    weighted_ehvi_product_2d,
    make_weights_2d,
    r2_envelope,
    er2i_discrete_achievement,
)

A = np.array([[1.0, 3.5], [2.0, 2.5], [3.0, 1.5]])
reference = np.array([5.0, 4.0])
mu = np.array([2.0, 1.5])
sigma = np.array([0.7, 0.6])
z_plus = np.array([0.0, 0.0])

print(ehvi_2d(mu, sigma, A, reference))

des = AspirationReservationDesirability(
    aspiration=np.array([1.0, 1.0]),
    reservation=reference,
    epsilon=0.05,
)
print(weighted_ehvi_product_2d(mu, sigma, A, des))

weights = make_weights_2d(21)
h_A_w = r2_envelope(A, z_plus, weights)
mu_w = np.array([max(w * (mu - z_plus)) for w in weights])
sigma_w = np.sqrt((weights[:, 0] * sigma[0])**2 + (weights[:, 1] * sigma[1])**2)
print(er2i_discrete_achievement(mu_w, sigma_w, h_A_w))
```

## Citation note

This artifact is intended to support the manuscript on preference-shaped
expected hypervolume and R2 improvement.  The formulas follow the notation of
that manuscript: EHVI is treated through hypervolume geometry, while exact R2
improvement is treated through scalarization-envelope or Tchebycheff-shadow
geometry.

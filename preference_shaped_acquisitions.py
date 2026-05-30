"""
Preference-shaped expected hypervolume and R2 improvement acquisitions.

This standalone module implements small, transparent reference routines for the
acquisition functions discussed in the report/presentation:

* canonical two-objective EHVI under independent Gaussian objectives;
* product-kernel / desirability-shaped EHVI using aspiration--reservation
  desirability functions;
* cone-based EHVI via a simplicial cone transformation;
* discrete R2 improvement and ER2I in achievement space;
* exact/integral R2 ER2I in two objectives using quadrature under an
  independent objective-space Gaussian predictive distribution;
* Monte Carlo checks for the deterministic quadrature routines.

Convention
----------
All objective-space routines use minimization.  Smaller objective values are
better.  Hypervolume uses a worse/dystopian reference point r.  R2 uses a
utopian point z_plus and weighted Tchebycheff scalarizations

    g_lambda(y; z_plus) = max_i lambda_i (y_i - z_plus_i).

The code is written for clarity rather than speed.  It is suitable for testing,
figures, examples, and JMLR-style artifact preparation.  More aggressive
hypervolume algorithms, such as the Yildiz--Suri 4D implementation, can be used
as optional backends for higher-dimensional deterministic hypervolume, but the
main acquisition examples here are two-dimensional.

Author: Michael T. M. Emmerich and ChatGPT, 2026.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple
import importlib.util
import math

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import ndtr

ArrayLike = Sequence[float] | np.ndarray
EPS = 1e-14


# ---------------------------------------------------------------------------
# Basic normal and expected-improvement utilities
# ---------------------------------------------------------------------------


def normal_pdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal density."""
    return np.exp(-0.5 * np.asarray(x) ** 2) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal distribution function."""
    return ndtr(x)


def scalar_ei_min(threshold: float, mean: float, sigma: float) -> float:
    """Expected improvement for minimization.

    For Z ~ N(mean, sigma^2), returns

        E[(threshold - Z)_+].

    This is the scalar EI that appears in achievement-space ER2I.
    """
    threshold = float(threshold)
    mean = float(mean)
    sigma = float(sigma)
    if sigma <= EPS:
        return max(threshold - mean, 0.0)
    u = (threshold - mean) / sigma
    return float((threshold - mean) * normal_cdf(u) + sigma * normal_pdf(u))


def _as_2d_points(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    arr = np.atleast_2d(arr)
    if arr.shape[1] != 2:
        raise ValueError(f"Expected points of shape (n, 2), got {arr.shape}.")
    return arr


def _as_vector2(x: ArrayLike, name: str = "vector") -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size != 2:
        raise ValueError(f"{name} must have length 2, got shape {arr.shape}.")
    return arr


# ---------------------------------------------------------------------------
# Two-dimensional hypervolume, minimization and maximization orientations
# ---------------------------------------------------------------------------


def pareto_front_min_2d(points: np.ndarray, *, strict: bool = True) -> np.ndarray:
    """Return nondominated points for two-objective minimization.

    Points are sorted by the first coordinate increasingly and filtered so that
    the second coordinate strictly improves.  Boundary duplicates are removed.
    """
    points = _as_2d_points(points)
    if len(points) == 0:
        return points
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        return points
    order = np.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[order]
    front = []
    best_y = math.inf
    for x, y in sorted_points:
        if strict:
            keep = y < best_y - EPS
        else:
            keep = y <= best_y + EPS
        if keep:
            front.append((x, y))
            best_y = min(best_y, y)
    return np.asarray(front, dtype=float)


def hv2d_min(points: np.ndarray | Sequence[Sequence[float]], reference: ArrayLike) -> float:
    """Exact two-dimensional hypervolume for minimization.

    Computes the area of the union of rectangles [p_1, r_1] x [p_2, r_2]
    for points p that are strictly better than the reference in both
    coordinates.  Points outside the reference box are ignored.
    """
    ref = _as_vector2(reference, "reference")
    points = _as_2d_points(points)
    if len(points) == 0:
        return 0.0
    mask = (points[:, 0] < ref[0]) & (points[:, 1] < ref[1])
    points = points[mask]
    if len(points) == 0:
        return 0.0
    front = pareto_front_min_2d(points)
    if len(front) == 0:
        return 0.0
    area = 0.0
    previous_y = ref[1]
    for x, y in front:
        if y < previous_y:
            area += max(ref[0] - x, 0.0) * max(previous_y - y, 0.0)
            previous_y = y
    return float(max(area, 0.0))


def hvi2d_min(candidate: ArrayLike, points: np.ndarray | Sequence[Sequence[float]], reference: ArrayLike) -> float:
    """Two-dimensional hypervolume improvement for minimization."""
    candidate = _as_vector2(candidate, "candidate")
    points = _as_2d_points(points)
    old = hv2d_min(points, reference)
    new = hv2d_min(np.vstack([points, candidate]), reference)
    return float(max(new - old, 0.0))


def hv2d_max(points: np.ndarray | Sequence[Sequence[float]], reference: ArrayLike) -> float:
    """Two-dimensional hypervolume for maximization with lower reference.

    This is implemented by sign reversal and the minimization routine.
    """
    points = _as_2d_points(points)
    ref = _as_vector2(reference, "reference")
    return hv2d_min(-points, -ref)


def hvi2d_max(candidate: ArrayLike, points: np.ndarray | Sequence[Sequence[float]], reference: ArrayLike) -> float:
    candidate = _as_vector2(candidate, "candidate")
    points = _as_2d_points(points)
    return float(max(hv2d_max(np.vstack([points, candidate]), reference) - hv2d_max(points, reference), 0.0))


# ---------------------------------------------------------------------------
# Deterministic Gaussian integration and Monte Carlo helpers
# ---------------------------------------------------------------------------


def gauss_hermite_expectation_2d(
    func: Callable[[np.ndarray], float],
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    n_gh: int = 40,
) -> float:
    """Approximate E[func(Y)] for independent two-dimensional Gaussian Y.

    Uses tensor-product Gauss--Hermite quadrature.  If X_i are Hermite nodes
    and W_i weights for exp(-x^2), then

        E[f(mu + sigma Z)] ~= (1/pi) sum_ij W_i W_j
                              f(mu + sqrt(2) sigma * X).
    """
    mu = _as_vector2(mu, "mu")
    sigma = _as_vector2(sigma, "sigma")
    if np.any(sigma < 0):
        raise ValueError("sigma entries must be nonnegative.")
    nodes, weights = hermgauss(int(n_gh))
    total = 0.0
    root2 = math.sqrt(2.0)
    for i, xi in enumerate(nodes):
        y0 = mu[0] + root2 * sigma[0] * xi
        wi = weights[i]
        for j, xj in enumerate(nodes):
            y = np.array([y0, mu[1] + root2 * sigma[1] * xj], dtype=float)
            total += wi * weights[j] * float(func(y))
    return float(total / math.pi)


def mc_expectation_2d(
    func: Callable[[np.ndarray], float],
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    n_samples: int = 100_000,
    seed: int = 12345,
) -> Tuple[float, float]:
    """Monte Carlo estimate and standard error for E[func(Y)]."""
    mu = _as_vector2(mu, "mu")
    sigma = _as_vector2(sigma, "sigma")
    rng = np.random.default_rng(seed)
    samples = rng.normal(loc=mu, scale=sigma, size=(int(n_samples), 2))
    values = np.fromiter((float(func(y)) for y in samples), dtype=float, count=int(n_samples))
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(n_samples))


# ---------------------------------------------------------------------------
# Canonical EHVI and preference-shaped hypervolume improvements
# ---------------------------------------------------------------------------


def ehvi_2d(
    mu: ArrayLike,
    sigma: ArrayLike,
    points: np.ndarray | Sequence[Sequence[float]],
    reference: ArrayLike,
    *,
    n_gh: int = 40,
) -> float:
    """Two-objective EHVI under independent Gaussian objectives.

    This is a deterministic Gauss--Hermite evaluation of

        E[HVI(Y, A; r)].
    """
    points = _as_2d_points(points)
    ref = _as_vector2(reference, "reference")
    return gauss_hermite_expectation_2d(lambda y: hvi2d_min(y, points, ref), mu, sigma, n_gh=n_gh)


def mc_ehvi_2d(
    mu: ArrayLike,
    sigma: ArrayLike,
    points: np.ndarray | Sequence[Sequence[float]],
    reference: ArrayLike,
    *,
    n_samples: int = 100_000,
    seed: int = 12345,
) -> Tuple[float, float]:
    """Monte Carlo check for two-objective EHVI."""
    points = _as_2d_points(points)
    ref = _as_vector2(reference, "reference")
    return mc_expectation_2d(lambda y: hvi2d_min(y, points, ref), mu, sigma, n_samples=n_samples, seed=seed)


@dataclass(frozen=True)
class AspirationReservationDesirability:
    """Coordinate-wise aspiration--reservation desirability for minimization.

    The map is clipped to [0, 1], equals 1 at and below the aspiration level,
    equals 0 at and above the reservation level, and uses a smooth cubic
    transition in between.  The optional epsilon creates small plateaus near
    the endpoints; epsilon=0.05 is the default used in the examples.

    For objective value y_i, with aspiration a_i and reservation r_i:

        s_i = normalized position of y_i between a_i and r_i,
        D_i(y_i) = 1 - (3 s_i^2 - 2 s_i^3).

    This is decreasing in y_i, as appropriate for minimization.
    """

    aspiration: ArrayLike
    reservation: ArrayLike
    epsilon: float = 0.05

    def __post_init__(self) -> None:
        a = _as_vector2(self.aspiration, "aspiration")
        r = _as_vector2(self.reservation, "reservation")
        if np.any(r <= a):
            raise ValueError("For minimization, reservation must be coordinatewise larger than aspiration.")
        if not (0.0 <= self.epsilon < 0.5):
            raise ValueError("epsilon must satisfy 0 <= epsilon < 0.5.")
        object.__setattr__(self, "aspiration", a)
        object.__setattr__(self, "reservation", r)

    def normalized(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        s = (y - self.aspiration) / (self.reservation - self.aspiration)
        s = np.clip(s, 0.0, 1.0)
        if self.epsilon > 0:
            s = np.clip((s - self.epsilon) / (1.0 - 2.0 * self.epsilon), 0.0, 1.0)
        return s

    def transform(self, y: np.ndarray | Sequence[Sequence[float]] | ArrayLike) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float)
        s = self.normalized(y_arr)
        smoothstep = 3.0 * s**2 - 2.0 * s**3
        return 1.0 - smoothstep

    @property
    def reference_desirability(self) -> np.ndarray:
        return self.transform(self.reservation)


def weighted_hvi_product_2d(
    candidate: ArrayLike,
    points: np.ndarray | Sequence[Sequence[float]],
    desirability: AspirationReservationDesirability,
) -> float:
    """Product-kernel weighted HVI via desirability coordinates.

    The coordinate-wise desirability map converts minimization objectives into
    maximization desirabilities.  Weighted HVI is then ordinary maximization HVI
    in desirability space.
    """
    points = _as_2d_points(points)
    cand = _as_vector2(candidate, "candidate")
    D_points = desirability.transform(points)
    D_cand = desirability.transform(cand)
    D_ref = desirability.reference_desirability
    return hvi2d_max(D_cand, D_points, D_ref)


def weighted_ehvi_product_2d(
    mu: ArrayLike,
    sigma: ArrayLike,
    points: np.ndarray | Sequence[Sequence[float]],
    desirability: AspirationReservationDesirability,
    *,
    n_gh: int = 40,
) -> float:
    """Expected weighted HVI under independent Gaussian objectives."""
    points = _as_2d_points(points)
    return gauss_hermite_expectation_2d(
        lambda y: weighted_hvi_product_2d(y, points, desirability), mu, sigma, n_gh=n_gh
    )


def mc_weighted_ehvi_product_2d(
    mu: ArrayLike,
    sigma: ArrayLike,
    points: np.ndarray | Sequence[Sequence[float]],
    desirability: AspirationReservationDesirability,
    *,
    n_samples: int = 100_000,
    seed: int = 12345,
) -> Tuple[float, float]:
    points = _as_2d_points(points)
    return mc_expectation_2d(
        lambda y: weighted_hvi_product_2d(y, points, desirability),
        mu,
        sigma,
        n_samples=n_samples,
        seed=seed,
    )


def cone_hvi_2d(
    candidate: ArrayLike,
    points: np.ndarray | Sequence[Sequence[float]],
    reference: ArrayLike,
    cone_matrix: np.ndarray,
) -> float:
    """Cone-based HVI for a two-dimensional simplicial cone.

    The columns of cone_matrix are the cone generators.  With L=C^{-1}, the
    cone order is transformed into the ordinary Pareto order.  This routine
    computes ordinary minimization HVI in L-coordinates.
    """
    C = np.asarray(cone_matrix, dtype=float)
    if C.shape != (2, 2):
        raise ValueError("cone_matrix must be 2x2.")
    L = np.linalg.inv(C)
    points = _as_2d_points(points)
    cand = _as_vector2(candidate, "candidate")
    ref = _as_vector2(reference, "reference")
    return hvi2d_min(L @ cand, (L @ points.T).T, L @ ref)


def cone_ehvi_2d(
    mu: ArrayLike,
    sigma: ArrayLike,
    points: np.ndarray | Sequence[Sequence[float]],
    reference: ArrayLike,
    cone_matrix: np.ndarray,
    *,
    n_gh: int = 40,
) -> float:
    """Expected cone-based HVI under independent objective-space Gaussians."""
    points = _as_2d_points(points)
    return gauss_hermite_expectation_2d(
        lambda y: cone_hvi_2d(y, points, reference, cone_matrix), mu, sigma, n_gh=n_gh
    )


def mc_cone_ehvi_2d(
    mu: ArrayLike,
    sigma: ArrayLike,
    points: np.ndarray | Sequence[Sequence[float]],
    reference: ArrayLike,
    cone_matrix: np.ndarray,
    *,
    n_samples: int = 100_000,
    seed: int = 12345,
) -> Tuple[float, float]:
    points = _as_2d_points(points)
    return mc_expectation_2d(
        lambda y: cone_hvi_2d(y, points, reference, cone_matrix),
        mu,
        sigma,
        n_samples=n_samples,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# R2, Tchebycheff envelopes, and ER2I
# ---------------------------------------------------------------------------


def make_weights_2d(n_weights: int, *, include_endpoints: bool = True) -> np.ndarray:
    """Equally spaced two-objective weights as rows [lambda, 1-lambda]."""
    if n_weights < 2:
        raise ValueError("n_weights must be at least 2.")
    if include_endpoints:
        lambdas = np.linspace(0.0, 1.0, int(n_weights))
    else:
        lambdas = (np.arange(int(n_weights)) + 0.5) / int(n_weights)
    return np.column_stack([lambdas, 1.0 - lambdas])


def tchebycheff_achievement(y: ArrayLike, z_plus: ArrayLike, weight: ArrayLike) -> float:
    """Weighted Tchebycheff achievement value for minimization."""
    y = np.asarray(y, dtype=float).reshape(-1)
    z = np.asarray(z_plus, dtype=float).reshape(-1)
    w = np.asarray(weight, dtype=float).reshape(-1)
    if not (len(y) == len(z) == len(w)):
        raise ValueError("y, z_plus, and weight must have equal length.")
    return float(np.max(w * (y - z)))


def r2_envelope(points: np.ndarray | Sequence[Sequence[float]], z_plus: ArrayLike, weights: np.ndarray) -> np.ndarray:
    """Current Tchebycheff envelope h_A(lambda_k) for a finite weight set."""
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    z = np.asarray(z_plus, dtype=float).reshape(-1)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array.")
    if weights.ndim != 2 or weights.shape[1] != points.shape[1]:
        raise ValueError("weights must have shape (K, m) matching points.")
    values = np.empty(len(weights), dtype=float)
    for k, w in enumerate(weights):
        values[k] = min(tchebycheff_achievement(p, z, w) for p in points)
    return values


def r2_improvement_discrete(
    candidate: ArrayLike,
    points: np.ndarray | Sequence[Sequence[float]],
    z_plus: ArrayLike,
    weights: np.ndarray,
    *,
    weight_probabilities: Optional[ArrayLike] = None,
) -> float:
    """Deterministic discrete R2 improvement of a candidate point."""
    weights = np.asarray(weights, dtype=float)
    h_A = r2_envelope(points, z_plus, weights)
    candidate_values = np.array([tchebycheff_achievement(candidate, z_plus, w) for w in weights])
    improvements = np.maximum(h_A - candidate_values, 0.0)
    if weight_probabilities is None:
        return float(np.mean(improvements))
    probs = np.asarray(weight_probabilities, dtype=float).reshape(-1)
    probs = probs / probs.sum()
    return float(np.dot(probs, improvements))


def er2i_discrete_achievement(
    mu_w: ArrayLike,
    sigma_w: ArrayLike,
    h_A_w: ArrayLike,
    *,
    weight_probabilities: Optional[ArrayLike] = None,
) -> float:
    """Discrete ER2I from Gaussian achievement-space predictions.

    Inputs are arrays indexed by weights:

        Z_k(x) ~ N(mu_w[k], sigma_w[k]^2),
        h_A_w[k] = current envelope value at that weight.
    """
    mu_w = np.asarray(mu_w, dtype=float).reshape(-1)
    sigma_w = np.asarray(sigma_w, dtype=float).reshape(-1)
    h_A_w = np.asarray(h_A_w, dtype=float).reshape(-1)
    if not (len(mu_w) == len(sigma_w) == len(h_A_w)):
        raise ValueError("mu_w, sigma_w, and h_A_w must have equal length.")
    eis = np.array([scalar_ei_min(h, m, s) for h, m, s in zip(h_A_w, mu_w, sigma_w)], dtype=float)
    if weight_probabilities is None:
        return float(np.mean(eis))
    probs = np.asarray(weight_probabilities, dtype=float).reshape(-1)
    probs = probs / probs.sum()
    return float(np.dot(probs, eis))


def _h_A_lambda_2d(lam: float, points: np.ndarray, z_plus: np.ndarray) -> float:
    w = np.array([lam, 1.0 - lam], dtype=float)
    return min(tchebycheff_achievement(p, z_plus, w) for p in points)


def er2i_integral_objective_gaussian_2d(
    mu: ArrayLike,
    sigma: ArrayLike,
    points: np.ndarray | Sequence[Sequence[float]],
    z_plus: ArrayLike,
    *,
    rho: Optional[Callable[[float], float]] = None,
    n_lambda: int = 64,
    inner_epsabs: float = 1e-9,
    inner_epsrel: float = 1e-9,
) -> float:
    """Exact/integral ER2I in 2D by quadrature under objective Gaussians.

    The predictive model is independent objective-space Gaussian:

        Y_i ~ N(mu_i, sigma_i^2).

    For each lambda in (0,1), the integrand is computed by the layer-cake
    formula

        int_{-inf}^{h_A(lambda)} P(g_lambda(Y) <= t) dt,

    where independence factorizes the probability into two Gaussian CDFs.
    The outer lambda integral uses Gauss--Legendre quadrature on (0,1).
    """
    mu = _as_vector2(mu, "mu")
    sigma = _as_vector2(sigma, "sigma")
    points = _as_2d_points(points)
    z_plus = _as_vector2(z_plus, "z_plus")
    if rho is None:
        rho = lambda lam: 1.0
    nodes, weights = leggauss(int(n_lambda))
    lambdas = 0.5 * (nodes + 1.0)
    lambda_weights = 0.5 * weights

    total = 0.0
    for lam, w_lam in zip(lambdas, lambda_weights):
        h = _h_A_lambda_2d(float(lam), points, z_plus)
        lam1 = float(lam)
        lam2 = 1.0 - lam1

        def cdf_achievement_leq_t(t: float) -> float:
            # Boundary weights are avoided by Gauss--Legendre nodes in (0, 1).
            b1 = z_plus[0] + t / lam1
            b2 = z_plus[1] + t / lam2
            return float(normal_cdf((b1 - mu[0]) / sigma[0]) * normal_cdf((b2 - mu[1]) / sigma[1]))

        inner, _ = quad(cdf_achievement_leq_t, -np.inf, h, epsabs=inner_epsabs, epsrel=inner_epsrel, limit=100)
        total += float(w_lam) * float(rho(lam1)) * inner
    return float(total)


def r2_improvement_integral_candidate_2d(
    candidate: ArrayLike,
    points: np.ndarray | Sequence[Sequence[float]],
    z_plus: ArrayLike,
    *,
    rho: Optional[Callable[[float], float]] = None,
    n_lambda: int = 128,
) -> float:
    """Deterministic exact/integral R2 improvement of a candidate point in 2D."""
    candidate = _as_vector2(candidate, "candidate")
    points = _as_2d_points(points)
    z_plus = _as_vector2(z_plus, "z_plus")
    if rho is None:
        rho = lambda lam: 1.0
    nodes, weights = leggauss(int(n_lambda))
    lambdas = 0.5 * (nodes + 1.0)
    lambda_weights = 0.5 * weights
    total = 0.0
    for lam, wl in zip(lambdas, lambda_weights):
        w = np.array([lam, 1.0 - lam], dtype=float)
        h = _h_A_lambda_2d(float(lam), points, z_plus)
        g = tchebycheff_achievement(candidate, z_plus, w)
        total += wl * float(rho(float(lam))) * max(h - g, 0.0)
    return float(total)


def mc_er2i_integral_objective_gaussian_2d(
    mu: ArrayLike,
    sigma: ArrayLike,
    points: np.ndarray | Sequence[Sequence[float]],
    z_plus: ArrayLike,
    *,
    rho: Optional[Callable[[float], float]] = None,
    n_lambda: int = 128,
    n_samples: int = 100_000,
    seed: int = 12345,
) -> Tuple[float, float]:
    """Monte Carlo check for exact/integral ER2I under objective Gaussians.

    This implementation vectorizes the weight integration over the Monte Carlo
    samples and is much faster than calling a quadrature routine per sample.
    """
    mu = _as_vector2(mu, "mu")
    sigma = _as_vector2(sigma, "sigma")
    points = _as_2d_points(points)
    z_plus = _as_vector2(z_plus, "z_plus")
    if rho is None:
        rho = lambda lam: 1.0

    nodes, weights = leggauss(int(n_lambda))
    lambdas = 0.5 * (nodes + 1.0)
    lambda_weights = 0.5 * weights
    rho_values = np.array([rho(float(lam)) for lam in lambdas], dtype=float)
    h_values = np.array([_h_A_lambda_2d(float(lam), points, z_plus) for lam in lambdas], dtype=float)

    rng = np.random.default_rng(seed)
    samples = rng.normal(loc=mu, scale=sigma, size=(int(n_samples), 2))
    centered = samples - z_plus
    # g_matrix[n, k] = max(lambda_k*y_1, (1-lambda_k)*y_2)
    g1 = centered[:, [0]] * lambdas[None, :]
    g2 = centered[:, [1]] * (1.0 - lambdas)[None, :]
    g = np.maximum(g1, g2)
    improvements = np.maximum(h_values[None, :] - g, 0.0)
    values = improvements @ (lambda_weights * rho_values)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(n_samples))


# ---------------------------------------------------------------------------
# Optional Yildiz--Suri backend loader
# ---------------------------------------------------------------------------


def load_yildiz_suri_backend(module_path: str | Path):
    """Load an optional Yildiz--Suri anchored hypervolume module.

    The external module is expected to expose a function named
    anchored_hypervolume(points, method="auto", prune=True/False).  This loader
    keeps the present file standalone while allowing the attached implementation
    to be used in higher-dimensional experiments.
    """
    module_path = Path(module_path)
    if not module_path.exists():
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location("yildiz_suri_backend", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, "anchored_hypervolume"):
        raise AttributeError("Backend module has no anchored_hypervolume function.")
    return module


# ---------------------------------------------------------------------------
# Examples from the TikZ figures and smoke tests
# ---------------------------------------------------------------------------


def tikz_example_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Default example inspired by the EHVI/Tchebycheff-shadow TikZ figures."""
    A = np.array([[1.0, 3.5], [2.0, 2.5], [3.0, 1.5]], dtype=float)
    reference = np.array([5.0, 4.0], dtype=float)
    mu = np.array([2.0, 1.5], dtype=float)
    sigma = np.array([0.7, 0.6], dtype=float)
    return A, reference, mu, sigma


def run_tikz_examples(*, n_mc: int = 50_000, seed: int = 2026) -> dict[str, tuple[float, float, float]]:
    """Run deterministic calculations and Monte Carlo checks.

    Returns a dictionary mapping test names to triples

        (deterministic_value, mc_value, mc_standard_error).
    """
    A, reference, mu, sigma = tikz_example_data()
    z_plus = np.array([0.0, 0.0])

    results: dict[str, tuple[float, float, float]] = {}

    det = ehvi_2d(mu, sigma, A, reference, n_gh=45)
    mc, se = mc_ehvi_2d(mu, sigma, A, reference, n_samples=n_mc, seed=seed)
    results["canonical_ehvi_2d"] = (det, mc, se)

    desirability = AspirationReservationDesirability(
        aspiration=np.array([1.0, 1.0]),
        reservation=reference,
        epsilon=0.05,
    )
    det = weighted_ehvi_product_2d(mu, sigma, A, desirability, n_gh=45)
    mc, se = mc_weighted_ehvi_product_2d(mu, sigma, A, desirability, n_samples=n_mc, seed=seed + 1)
    results["product_desirability_weighted_ehvi_2d"] = (det, mc, se)

    # Symmetric obtuse cone-like generator matrix.  The identity is the Pareto cone.
    cone_matrix = np.array([[1.0, -0.35], [-0.35, 1.0]], dtype=float)
    det = cone_ehvi_2d(mu, sigma, A, reference, cone_matrix, n_gh=45)
    mc, se = mc_cone_ehvi_2d(mu, sigma, A, reference, cone_matrix, n_samples=n_mc, seed=seed + 2)
    results["cone_ehvi_2d"] = (det, mc, se)

    weights = make_weights_2d(21, include_endpoints=True)
    h_w = r2_envelope(A, z_plus, weights)
    # Simple achievement-space prediction induced by evaluating g at the mean
    # and using a conservative scalar standard deviation proxy per weight.
    mu_w = np.array([tchebycheff_achievement(mu, z_plus, w) for w in weights])
    sigma_w = np.sqrt((weights[:, 0] * sigma[0]) ** 2 + (weights[:, 1] * sigma[1]) ** 2)
    det = er2i_discrete_achievement(mu_w, sigma_w, h_w)
    # No MC comparison here because this is a direct achievement-space model,
    # not the push-forward of the objective-space Gaussian.
    results["discrete_er2i_achievement"] = (det, float("nan"), float("nan"))

    det = er2i_integral_objective_gaussian_2d(mu, sigma, A, z_plus, n_lambda=32)
    mc, se = mc_er2i_integral_objective_gaussian_2d(
        mu, sigma, A, z_plus, n_lambda=64, n_samples=n_mc, seed=seed + 3
    )
    results["integral_er2i_objective_gaussian_2d"] = (det, mc, se)

    return results


def _format_check(name: str, det: float, mc: float, se: float) -> str:
    if math.isnan(mc):
        return f"{name:42s} deterministic={det:.8g}"
    z = abs(det - mc) / max(se, EPS)
    return f"{name:42s} deterministic={det:.8g}  MC={mc:.8g}  SE={se:.3g}  |diff|/SE={z:.2f}"


def main() -> None:
    print("Preference-shaped acquisition examples from the TikZ data")
    print("minimization convention; independent Gaussian objectives")
    print()
    results = run_tikz_examples(n_mc=5_000)
    for name, (det, mc, se) in results.items():
        print(_format_check(name, det, mc, se))
    print()
    print("Rule of thumb: deterministic quadrature values should usually lie within a few MC standard errors.")


if __name__ == "__main__":
    main()

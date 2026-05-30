#!/usr/bin/env python3
"""
3D and 4D examples for product-kernel weighted hypervolume via desirability maps.

This script is standalone: it contains the aspiration--reservation desirability
transformation, exact small-n anchored hypervolume by inclusion--exclusion,
Monte Carlo checks, and an optional import of the Yildiz--Suri backend from
``yildiz_suri_hv.py`` if that file is placed next to this script.

Mathematical convention
-----------------------
All objectives are minimized.  For each objective i we use an
aspiration--reservation desirability map

    D_i(y_i) = 1,                              y_i <= a_i,
             (r_i - y_i) / (r_i - a_i),        a_i < y_i < r_i,
             0,                              y_i >= r_i.

The product-kernel weighted hypervolume of A with respect to the reservation
point r is the ordinary anchored hypervolume of the transformed set D(A) in
desirability coordinates:

    union_{a in A} [0, D_1(a_1)] x ... x [0, D_d(a_d)].

For the examples below, aspiration levels are chosen as epsilon * reservation
with epsilon = 0.05, matching the aspiration/reservation style used in the
preference-shaping examples.

Optional backend
----------------
If ``yildiz_suri_hv.py`` is available, the script also calls its
``anchored_hypervolume`` function.  This provides an includable route to the
Yildiz--Suri-style implementation supplied at

    https://github.com/emmerichmtm/ImplementationOfYilidizAndSuriSubquadratic4DHypervolume

The script still runs without this backend by using its own exact
inclusion--exclusion routine.  This fallback is exponential in the number of
points and is intended only for small validation examples.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence
import importlib.util
import math
import sys

import numpy as np


ArrayLike = Sequence[float] | np.ndarray
EPS = 1e-14


@dataclass(frozen=True)
class AspirationReservationDesirability:
    """Coordinate-wise aspiration--reservation desirability for minimization."""

    aspiration: np.ndarray
    reservation: np.ndarray
    epsilon: float = 0.05

    @classmethod
    def from_reservation(
        cls,
        reservation: ArrayLike,
        *,
        epsilon: float = 0.05,
        aspiration: Optional[ArrayLike] = None,
    ) -> "AspirationReservationDesirability":
        reservation = np.asarray(reservation, dtype=float).reshape(-1)
        if aspiration is None:
            aspiration = epsilon * reservation
        aspiration = np.asarray(aspiration, dtype=float).reshape(-1)
        if aspiration.shape != reservation.shape:
            raise ValueError("aspiration and reservation must have the same dimension.")
        if np.any(reservation <= aspiration):
            raise ValueError("For minimization, each reservation level must exceed aspiration.")
        return cls(aspiration=aspiration, reservation=reservation, epsilon=float(epsilon))

    def transform(self, y: ArrayLike) -> np.ndarray:
        """Transform objective vector(s) to desirability coordinates in [0,1]."""
        arr = np.asarray(y, dtype=float)
        d = (self.reservation - arr) / (self.reservation - self.aspiration)
        return np.clip(d, 0.0, 1.0)

    def density(self, y: ArrayLike) -> np.ndarray:
        """Product-kernel density at objective vector(s)."""
        arr = np.asarray(y, dtype=float)
        inside = (arr >= self.aspiration) & (arr <= self.reservation)
        coord_density = np.where(inside, 1.0 / (self.reservation - self.aspiration), 0.0)
        return np.prod(coord_density, axis=-1)


def remove_dominated_max(points: np.ndarray, *, eps: float = EPS) -> np.ndarray:
    """Remove dominated desirability points under maximization."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.empty((0, 0))
    pts = np.atleast_2d(pts)
    keep = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            if np.all(q >= p - eps) and np.any(q > p + eps):
                dominated = True
                break
        if not dominated and np.all(p > eps):
            keep.append(p)
    if not keep:
        return np.empty((0, pts.shape[1]))
    arr = np.array(keep, dtype=float)
    _, idx = np.unique(np.round(arr, 14), axis=0, return_index=True)
    return arr[np.sort(idx)]


def anchored_hypervolume_inclusion_exclusion(points: np.ndarray) -> float:
    """Exact anchored HV of union of boxes [0,p] for small examples."""
    pts = remove_dominated_max(np.asarray(points, dtype=float))
    if pts.size == 0:
        return 0.0
    n, d = pts.shape
    total = 0.0
    for k in range(1, n + 1):
        sign = 1.0 if (k % 2 == 1) else -1.0
        for idx in combinations(range(n), k):
            corner = np.min(pts[list(idx)], axis=0)
            total += sign * float(np.prod(np.maximum(corner, 0.0)))
    return max(total, 0.0)


def load_yildiz_suri_backend(path: str | Path = "yildiz_suri_hv.py"):
    """Load optional Yildiz--Suri backend if available."""
    path = Path(path)
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("yildiz_suri_hv", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"Warning: could not load Yildiz--Suri backend: {exc}")
        return None
    if not hasattr(module, "anchored_hypervolume"):
        return None
    return module


def anchored_hypervolume(points: np.ndarray, *, backend=None) -> float:
    """Anchored HV, using Yildiz--Suri backend if provided, else fallback."""
    pts = remove_dominated_max(np.asarray(points, dtype=float))
    if pts.size == 0:
        return 0.0
    if backend is not None and hasattr(backend, "anchored_hypervolume"):
        try:
            return float(backend.anchored_hypervolume(pts.tolist()))
        except Exception as exc:
            print(f"Warning: backend failed; using fallback: {exc}")
    return anchored_hypervolume_inclusion_exclusion(pts)


def weighted_hv_product(
    points: np.ndarray,
    desirability: AspirationReservationDesirability,
    *,
    backend=None,
) -> float:
    """Product-kernel weighted HV by desirability-space anchored HV."""
    transformed = desirability.transform(points)
    return anchored_hypervolume(transformed, backend=backend)


def weighted_hvi_product(
    points: np.ndarray,
    candidate: ArrayLike,
    desirability: AspirationReservationDesirability,
    *,
    backend=None,
) -> float:
    """Product-kernel weighted hypervolume improvement of one candidate."""
    points = np.asarray(points, dtype=float)
    candidate = np.asarray(candidate, dtype=float).reshape(1, -1)
    before = weighted_hv_product(points, desirability, backend=backend)
    after = weighted_hv_product(np.vstack([points, candidate]), desirability, backend=backend)
    return max(after - before, 0.0)


def mc_weighted_hv_product(
    points: np.ndarray,
    desirability: AspirationReservationDesirability,
    *,
    n_samples: int = 200_000,
    seed: int = 12345,
) -> tuple[float, float]:
    """Monte Carlo check in desirability space."""
    rng = np.random.default_rng(seed)
    T = remove_dominated_max(desirability.transform(points))
    if T.size == 0:
        return 0.0, 0.0
    _, d = T.shape
    U = rng.random((n_samples, d))
    dominated = np.zeros(n_samples, dtype=bool)
    for p in T:
        dominated |= np.all(U <= p, axis=1)
    estimate = float(np.mean(dominated))
    se = math.sqrt(max(estimate * (1.0 - estimate), 0.0) / n_samples)
    return estimate, se


def mc_weighted_hvi_product(
    points: np.ndarray,
    candidate: ArrayLike,
    desirability: AspirationReservationDesirability,
    *,
    n_samples: int = 200_000,
    seed: int = 12345,
) -> tuple[float, float]:
    """Monte Carlo check for product-kernel weighted HVI."""
    points = np.asarray(points, dtype=float)
    candidate = np.asarray(candidate, dtype=float).reshape(1, -1)
    hv0, se0 = mc_weighted_hv_product(points, desirability, n_samples=n_samples, seed=seed)
    hv1, se1 = mc_weighted_hv_product(
        np.vstack([points, candidate]), desirability, n_samples=n_samples, seed=seed + 1
    )
    return max(hv1 - hv0, 0.0), math.sqrt(se0 * se0 + se1 * se1)


def mc_weighted_ehvi_product_gaussian(
    points: np.ndarray,
    mu: ArrayLike,
    sigma: ArrayLike,
    desirability: AspirationReservationDesirability,
    *,
    n_samples: int = 20_000,
    seed: int = 2026,
    backend=None,
) -> tuple[float, float]:
    """Monte Carlo product-kernel weighted EHVI under independent objective Gaussians."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    samples = rng.normal(loc=mu, scale=sigma, size=(n_samples, mu.size))
    vals = np.empty(n_samples, dtype=float)
    for i, y in enumerate(samples):
        vals[i] = weighted_hvi_product(points, y, desirability, backend=backend)
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(n_samples))


def tikz_inspired_3d_example():
    """3D example extending the 2D TikZ HVI points with a third coordinate."""
    points = np.array(
        [
            [1.0, 3.5, 2.4],
            [2.0, 2.5, 1.7],
            [3.0, 1.5, 1.2],
        ],
        dtype=float,
    )
    reservation = np.array([5.0, 4.0, 3.0], dtype=float)
    candidate = np.array([2.45, 1.05, 1.45], dtype=float)
    mu = candidate.copy()
    sigma = np.array([0.70, 0.60, 0.45], dtype=float)
    return points, reservation, candidate, mu, sigma


def tikz_inspired_4d_example():
    """4D example extending the 2D TikZ HVI points with two extra coordinates."""
    points = np.array(
        [
            [1.0, 3.5, 2.4, 3.2],
            [2.0, 2.5, 1.7, 2.6],
            [3.0, 1.5, 1.2, 2.0],
            [2.4, 2.2, 2.0, 1.2],
        ],
        dtype=float,
    )
    reservation = np.array([5.0, 4.0, 3.0, 4.0], dtype=float)
    candidate = np.array([2.45, 1.05, 1.45, 1.55], dtype=float)
    mu = candidate.copy()
    sigma = np.array([0.70, 0.60, 0.45, 0.50], dtype=float)
    return points, reservation, candidate, mu, sigma


def run_example(
    name: str,
    points: np.ndarray,
    reservation: np.ndarray,
    candidate: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    backend=None,
):
    desirability = AspirationReservationDesirability.from_reservation(
        reservation, epsilon=0.05
    )
    transformed = desirability.transform(points)
    transformed_candidate = desirability.transform(candidate)

    hv = weighted_hv_product(points, desirability, backend=backend)
    hvi = weighted_hvi_product(points, candidate, desirability, backend=backend)
    mc_hv, mc_hv_se = mc_weighted_hv_product(points, desirability, n_samples=100_000, seed=7)
    mc_hvi, mc_hvi_se = mc_weighted_hvi_product(points, candidate, desirability, n_samples=100_000, seed=11)
    mc_ehvi, mc_ehvi_se = mc_weighted_ehvi_product_gaussian(
        points, mu, sigma, desirability, n_samples=8_000, seed=13, backend=backend
    )

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print(f"dimension                : {points.shape[1]}")
    print(f"points A                 :\n{points}")
    print(f"reservation              : {reservation}")
    print(f"aspiration (epsilon=0.05): {desirability.aspiration}")
    print(f"candidate                : {candidate}")
    print(f"transformed A            :\n{np.round(transformed, 6)}")
    print(f"transformed candidate    : {np.round(transformed_candidate, 6)}")
    print(f"weighted HV              : {hv:.10f}")
    print(f"weighted HVI(candidate)  : {hvi:.10f}")
    print(f"MC weighted HV           : {mc_hv:.10f} +/- {2.0 * mc_hv_se:.10f} (approx. 95%)")
    print(f"MC weighted HVI          : {mc_hvi:.10f} +/- {2.0 * mc_hvi_se:.10f} (approx. 95%)")
    print(f"MC weighted EHVI         : {mc_ehvi:.10f} +/- {2.0 * mc_ehvi_se:.10f} (approx. 95%)")
    if backend is not None:
        fallback_hv = anchored_hypervolume_inclusion_exclusion(transformed)
        print(f"fallback inclusion HV    : {fallback_hv:.10f}")
        print(f"backend agreement        : {abs(hv - fallback_hv):.3e}")


def main():
    here = Path(__file__).resolve().parent
    backend = load_yildiz_suri_backend(here / "yildiz_suri_hv.py")
    print("Yildiz--Suri backend:", "loaded" if backend is not None else "not found; using fallback")
    print("Repository reference:", "https://github.com/emmerichmtm/ImplementationOfYilidizAndSuriSubquadratic4DHypervolume")

    run_example("3D product-kernel weighted HV example", *tikz_inspired_3d_example(), backend=backend)
    run_example("4D product-kernel weighted HV example", *tikz_inspired_4d_example(), backend=backend)


if __name__ == "__main__":
    main()

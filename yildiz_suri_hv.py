"""
Yildiz-Suri-style anchored hypervolume implementation.

This module computes the volume of a union of boxes anchored at the origin,

    [0,p_1] x ... x [0,p_d].

It contains:

* direct low-dimensional anchored-frontier solvers for d <= 3;
* the original specialized 4D Yildiz--Suri-style implementation;
* a general fixed-dimensional Yildiz--Suri-style implementation for d > 4.

The general implementation follows the sweep/weighting reduction of Yildiz and
Suri.  A d-dimensional anchored instance is swept in the last coordinate, and
the active cross-section is maintained as a weighted-volume problem in dimension
d-2.  The weighted structure uses an Overmars--Yap/Yildiz--Suri trellis
partition, stores full-cell fragments in a binary partition tree, and maintains
leaf boundary fragments as weighted axis-parallel halfspaces using gradient
treaps and an ordered-product segment tree.

Complexity, for fixed dimension d:

* d=1: O(n)
* d=2: O(n log n)
* d=3: expected O(n log n)
* d=4: expected amortized O(n^(3/2) log n), special implementation
* d>4: O(n^((d-1)/2) polylog n), general Yildiz--Suri-style implementation

The constants grow exponentially with d, as is typical for fixed-dimensional
computational-geometry data structures.

Author: Michael T. M. Emmerich and ChatGPT, 2026-05-08.

Repository reference
--------------------
This file is included as an optional anchored hypervolume backend for the
preference-shaped acquisition examples.  The code repository supplied by
Michael T. M. Emmerich is:

    https://github.com/emmerichmtm/ImplementationOfYilidizAndSuriSubquadratic4DHypervolume

The routines compute anchored hypervolume of boxes [0,p_1] x ... x [0,p_d].
For product-kernel weighted hypervolume in minimization, objective vectors are
first transformed by coordinate-wise desirability functions and then anchored
hypervolume is computed in desirability space.

Generated wrapper filename: yildiz_suri_hv.py
"""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from math import ceil, sqrt, inf
from itertools import product
from random import Random
from typing import Iterable, List, Optional, Sequence, Tuple

EPS = 1e-12


class Fenwick:
    """1-indexed Fenwick tree for floating point sums."""

    def __init__(self, n: int):
        self.n = max(1, n)
        self.bit = [0.0] * (self.n + 1)

    def add(self, idx0: int, delta: float) -> None:
        i = idx0 + 1
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def prefix(self, idx0: int) -> float:
        if idx0 < 0:
            return 0.0
        if idx0 >= self.n:
            idx0 = self.n - 1
        i = idx0 + 1
        s = 0.0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s

    def total(self) -> float:
        return self.prefix(self.n - 1)

    def suffix_after(self, idx0: int) -> float:
        """sum over indices > idx0."""
        return self.total() - self.prefix(idx0)


class SparseFenwick:
    """Sparse 1-indexed Fenwick tree for floating point sums.

    This has the same O(log n) operation bound as ``Fenwick`` but allocates
    entries only when they are touched.  It is important in the trellis data
    structures, where many small cell structures share a global rank universe.
    """

    def __init__(self, n: int):
        self.n = max(1, int(n))
        self.bit = {}

    def add(self, idx0: int, delta: float) -> None:
        if idx0 < 0 or idx0 >= self.n or abs(delta) <= EPS:
            return
        i = idx0 + 1
        while i <= self.n:
            self.bit[i] = self.bit.get(i, 0.0) + delta
            i += i & -i

    def prefix(self, idx0: int) -> float:
        if idx0 < 0:
            return 0.0
        if idx0 >= self.n:
            idx0 = self.n - 1
        i = idx0 + 1
        s = 0.0
        while i > 0:
            s += self.bit.get(i, 0.0)
            i -= i & -i
        return s

    def total(self) -> float:
        return self.prefix(self.n - 1)

    def suffix_after(self, idx0: int) -> float:
        return self.total() - self.prefix(idx0)



@dataclass
class Segment1D:
    start: float
    end: float
    rank: int
    weight: float

    @property
    def length(self) -> float:
        return max(0.0, self.end - self.start)


class _TreapSeg:
    __slots__ = (
        "start", "end", "rank", "weight", "prio", "left", "right",
        "min_start", "max_end", "min_weight", "max_weight",
    )

    _rng = Random(123456789)

    def __init__(self, start: float, end: float, rank: int, weight: float):
        self.start = float(start)
        self.end = float(end)
        self.rank = int(rank)
        self.weight = float(weight)
        self.prio = self._rng.random()
        self.left = None
        self.right = None
        self.min_start = self.start
        self.max_end = self.end
        self.min_weight = self.weight
        self.max_weight = self.weight

    @property
    def length(self) -> float:
        return max(0.0, self.end - self.start)


def _upd(t):
    if t is None:
        return None
    t.min_start = t.start
    t.max_end = t.end
    t.min_weight = t.weight
    t.max_weight = t.weight
    if t.left is not None:
        t.min_start = min(t.min_start, t.left.min_start)
        t.max_end = max(t.max_end, t.left.max_end)
        t.min_weight = min(t.min_weight, t.left.min_weight)
        t.max_weight = max(t.max_weight, t.left.max_weight)
    if t.right is not None:
        t.min_start = min(t.min_start, t.right.min_start)
        t.max_end = max(t.max_end, t.right.max_end)
        t.min_weight = min(t.min_weight, t.right.min_weight)
        t.max_weight = max(t.max_weight, t.right.max_weight)
    return t


def _merge(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a.prio < b.prio:
        a.right = _merge(a.right, b)
        return _upd(a)
    else:
        b.left = _merge(a, b.left)
        return _upd(b)


def _clone_segment(t, start: float, end: float):
    if end <= start + EPS:
        return None
    return _TreapSeg(start, end, t.rank, t.weight)


def _split_coord(t, x: float):
    """Split by coordinate: left has segments in [..,x), right in [x,..)."""
    if t is None:
        return None, None
    if t.end <= x + EPS:
        a, b = _split_coord(t.right, x)
        t.right = a
        return _upd(t), b
    if t.start >= x - EPS:
        a, b = _split_coord(t.left, x)
        t.left = b
        return a, _upd(t)

    # x lies strictly inside this segment; split the segment itself.
    left_child, right_child = t.left, t.right
    left_seg = _clone_segment(t, t.start, x)
    right_seg = _clone_segment(t, x, t.end)
    left_root = _merge(left_child, left_seg)
    right_root = _merge(right_seg, right_child)
    return left_root, right_root


def _split_first_weight_less(t, w: float):
    """
    Split a nonincreasing-weight treap into prefix with weights >= w and
    suffix with weights < w.
    """
    if t is None:
        return None, None
    if t.min_weight >= w - EPS:
        return t, None
    if t.left is not None and t.left.min_weight < w - EPS:
        ge, lt = _split_first_weight_less(t.left, w)
        t.left = lt
        return ge, _upd(t)
    if t.weight < w - EPS:
        ge = t.left
        t.left = None
        return ge, _upd(t)
    ge_r, lt = _split_first_weight_less(t.right, w)
    t.right = ge_r
    return _upd(t), lt


def _rightmost(t):
    if t is None:
        return None
    while t.right is not None:
        t = t.right
    return t


def _collect_delta(t, on_delta, sign: float):
    if t is None:
        return
    _collect_delta(t.left, on_delta, sign)
    if t.length > EPS:
        on_delta(t.rank, sign * t.length)
    _collect_delta(t.right, on_delta, sign)


def _inorder_segments(t, out):
    if t is None:
        return
    _inorder_segments(t.left, out)
    out.append(Segment1D(t.start, t.end, t.rank, t.weight))
    _inorder_segments(t.right, out)


class Gradient1D:
    """
    Upper envelope of anchored 1D intervals [0, extent] with weights.

    This is the asymptotic version: the step function is stored in a randomized
    treap of maximal segments.  An insertion performs a coordinate split and a
    split at the first segment whose weight is below the new weight, then replaces
    the affected suffix by one segment.  The expected amortized cost is
    O(log m + r), where r is the number of deleted gradient segments; every
    deleted segment is charged to the insertion that created it.  The same bound
    holds for eliminate_below.
    """

    def __init__(self, length: float, on_delta):
        self.length = float(length)
        self.root = None
        self.on_delta = on_delta

    @property
    def segments(self) -> List[Segment1D]:
        out: List[Segment1D] = []
        _inorder_segments(self.root, out)
        return out

    def insert(self, extent: float, rank: int, weight: float) -> bool:
        a = max(0.0, min(self.length, float(extent)))
        if a <= EPS or weight <= 0:
            return False

        left, right = _split_coord(self.root, a)
        covered_end = left.max_end if left is not None else 0.0
        rm = _rightmost(left)
        if covered_end >= a - EPS and rm is not None and rm.weight >= weight - EPS:
            self.root = _merge(left, right)
            return False

        ge, lt = _split_first_weight_less(left, weight)
        if lt is not None:
            start = lt.min_start
            _collect_delta(lt, self.on_delta, -1.0)
        else:
            start = ge.max_end if ge is not None else 0.0
            start = min(start, a)

        newseg = None
        changed = False
        if a > start + EPS:
            newseg = _TreapSeg(start, a, rank, weight)
            self.on_delta(rank, a - start)
            changed = True

        self.root = _merge(_merge(ge, newseg), right)
        return changed

    def eliminate_below(self, threshold: float) -> bool:
        if self.root is None or self.root.min_weight >= threshold - EPS:
            return False
        ge, lt = _split_first_weight_less(self.root, threshold)
        if lt is not None:
            _collect_delta(lt, self.on_delta, -1.0)
        self.root = ge
        return lt is not None

    def min_weight(self) -> float:
        return self.root.min_weight if self.root is not None else inf


class CellHalfplaneDS:
    """
    Weighted volume in one rectangular cell for lower-left anchored vertical
    and horizontal strips. Maintains exact max-weight integral over the cell.
    """

    def __init__(self, width: float, height: float, rank_weights: Sequence[float]):
        self.Lx = float(width)
        self.Ly = float(height)
        self.rank_weights = list(rank_weights)
        self.n = len(rank_weights)

        self.Ax = SparseFenwick(self.n)   # vertical gradient widths by rank
        self.Ay = SparseFenwick(self.n)   # horizontal gradient heights by rank
        self.wAx = SparseFenwick(self.n)
        self.wAy = SparseFenwick(self.n)
        self.weighted_volume = 0.0

        self.vertical = Gradient1D(self.Lx, self._delta_vertical)
        self.horizontal = Gradient1D(self.Ly, self._delta_horizontal)

    def _delta_vertical(self, rank: int, delta: float) -> None:
        if abs(delta) <= EPS:
            return
        w = self.rank_weights[rank]
        # Update V before Fenwick changes, using current horizontal arrays.
        self.weighted_volume += w * delta * (self.Ly - self.Ay.suffix_after(rank))
        self.weighted_volume += -delta * self.wAy.prefix(rank - 1)
        self.Ax.add(rank, delta)
        self.wAx.add(rank, w * delta)

    def _delta_horizontal(self, rank: int, delta: float) -> None:
        if abs(delta) <= EPS:
            return
        w = self.rank_weights[rank]
        self.weighted_volume += w * delta * (self.Lx - self.Ax.suffix_after(rank))
        self.weighted_volume += -delta * self.wAx.prefix(rank - 1)
        self.Ay.add(rank, delta)
        self.wAy.add(rank, w * delta)

    def insert_vertical(self, width: float, rank: int, weight: float) -> bool:
        return self.vertical.insert(width, rank, weight)

    def insert_horizontal(self, height: float, rank: int, weight: float) -> bool:
        return self.horizontal.insert(height, rank, weight)

    def eliminate_below(self, threshold: float) -> bool:
        a = self.vertical.eliminate_below(threshold)
        b = self.horizontal.eliminate_below(threshold)
        return a or b

    def ordinary_area(self) -> float:
        vx = max(0.0, min(self.Lx, self.Ax.total()))
        hy = max(0.0, min(self.Ly, self.Ay.total()))
        return vx * self.Ly + hy * self.Lx - vx * hy

    def min_weight(self) -> float:
        return min(self.vertical.min_weight(), self.horizontal.min_weight())


@dataclass
class Node:
    l: int
    r: int
    area: float
    left: Optional[int] = None
    right: Optional[int] = None
    entry_weight: Optional[float] = None
    A: float = 0.0
    contrib: float = 0.0
    min_subtree_weight: float = inf


class SlabTree:
    """Segment tree over the y-cells of one vertical slab."""

    def __init__(self, xlo: float, xhi: float, ycuts: Sequence[float], rank_weights: Sequence[float]):
        self.xlo, self.xhi = float(xlo), float(xhi)
        self.width = max(0.0, self.xhi - self.xlo)
        self.ycuts = list(ycuts)
        self.cells: List[CellHalfplaneDS] = []
        for a, b in zip(self.ycuts[:-1], self.ycuts[1:]):
            self.cells.append(CellHalfplaneDS(self.width, b - a, rank_weights))
        self.nodes: List[Node] = []
        self.leaf_node: List[int] = [-1] * len(self.cells)
        self.total_weighted = 0.0
        if self.cells:
            self.root = self._build(0, len(self.cells) - 1)
            self._pull(self.root)
        else:
            self.root = -1

    def _build(self, l: int, r: int) -> int:
        area = self.width * (self.ycuts[r + 1] - self.ycuts[l])
        idx = len(self.nodes)
        self.nodes.append(Node(l, r, area))
        if l == r:
            self.leaf_node[l] = idx
        else:
            m = (l + r) // 2
            self.nodes[idx].left = self._build(l, m)
            self.nodes[idx].right = self._build(m + 1, r)
        return idx

    def _node_min(self, idx: int) -> float:
        node = self.nodes[idx]
        vals = []
        if node.entry_weight is not None:
            vals.append(node.entry_weight)
        if node.l == node.r:
            vals.append(self.cells[node.l].min_weight())
        else:
            vals.append(self.nodes[node.left].min_subtree_weight)  # type: ignore[index]
            vals.append(self.nodes[node.right].min_subtree_weight)  # type: ignore[index]
        return min(vals) if vals else inf

    def _pull(self, idx: int) -> None:
        node = self.nodes[idx]
        old_contrib = node.contrib
        if node.l == node.r:
            node.A = self.cells[node.l].ordinary_area()
        else:
            a = 0.0
            for child_idx in (node.left, node.right):
                child = self.nodes[child_idx]  # type: ignore[index]
                a += child.area if child.entry_weight is not None else child.A
            node.A = a
        node.contrib = (node.entry_weight * max(0.0, node.area - node.A)) if node.entry_weight is not None else 0.0
        self.total_weighted += node.contrib - old_contrib
        node.min_subtree_weight = self._node_min(idx)

    def _refresh_path_to_leaf(self, leaf: int, idx: Optional[int] = None) -> bool:
        if idx is None:
            idx = self.root
        node = self.nodes[idx]
        if node.l == node.r:
            self._pull(idx)
            return node.l == leaf
        if leaf <= self.nodes[node.left].r:  # type: ignore[index]
            found = self._refresh_path_to_leaf(leaf, node.left)
        else:
            found = self._refresh_path_to_leaf(leaf, node.right)
        self._pull(idx)
        return found

    def _max_entry_on_path(self, leaf: int, idx: Optional[int] = None, cur: float = -inf) -> float:
        if idx is None:
            idx = self.root
        node = self.nodes[idx]
        if node.entry_weight is not None:
            cur = max(cur, node.entry_weight)
        if node.l == node.r:
            return cur
        if leaf <= self.nodes[node.left].r:  # type: ignore[index]
            return self._max_entry_on_path(leaf, node.left, cur)
        return self._max_entry_on_path(leaf, node.right, cur)

    def _eliminate_below(self, idx: int, threshold: float, include_self: bool = False) -> bool:
        node = self.nodes[idx]
        if node.min_subtree_weight + EPS >= threshold:
            return False
        changed = False
        if include_self and node.entry_weight is not None and node.entry_weight + EPS < threshold:
            old = node.contrib
            node.entry_weight = None
            node.contrib = 0.0
            self.total_weighted -= old
            changed = True
        if node.l == node.r:
            before = node.entry_weight
            if node.entry_weight is not None and node.entry_weight + EPS < threshold:
                old = node.contrib
                node.entry_weight = None
                node.contrib = 0.0
                self.total_weighted -= old
                changed = True
            old_wv = self.cells[node.l].weighted_volume
            if self.cells[node.l].eliminate_below(threshold):
                self.total_weighted += self.cells[node.l].weighted_volume - old_wv
                changed = True
            self._pull(idx)
            return changed
        # Remove lower entries at children and below.
        for child_idx in (node.left, node.right):
            child = self.nodes[child_idx]  # type: ignore[index]
            if child.entry_weight is not None and child.entry_weight + EPS < threshold:
                old = child.contrib
                child.entry_weight = None
                child.contrib = 0.0
                self.total_weighted -= old
                changed = True
            if child.min_subtree_weight + EPS < threshold:
                changed = self._eliminate_below(child_idx, threshold, include_self=False) or changed  # type: ignore[arg-type]
        self._pull(idx)
        return changed

    def _insert_range(self, idx: int, ql: int, qr: int, rank: int, weight: float, ancestor_max: float) -> bool:
        if idx < 0 or qr < self.nodes[idx].l or self.nodes[idx].r < ql:
            return False
        node = self.nodes[idx]
        if node.entry_weight is not None:
            # Existing entry at this node also dominates descendants.
            if node.entry_weight >= weight - EPS:
                return False
        if ancestor_max >= weight - EPS:
            return False

        changed = False
        if ql <= node.l and node.r <= qr:
            old_contrib = node.contrib
            if node.entry_weight is None or node.entry_weight + EPS < weight:
                node.entry_weight = weight
                self._pull(idx)
                changed = True
                # Delete lower-weight fragments strictly below this node.
                if node.l == node.r:
                    # lower strips in this cell
                    old_wv = self.cells[node.l].weighted_volume
                    if self.cells[node.l].eliminate_below(weight):
                        self.total_weighted += self.cells[node.l].weighted_volume - old_wv
                        changed = True
                else:
                    for child_idx in (node.left, node.right):
                        child = self.nodes[child_idx]  # type: ignore[index]
                        if child.entry_weight is not None and child.entry_weight + EPS < weight:
                            old = child.contrib
                            child.entry_weight = None
                            child.contrib = 0.0
                            self.total_weighted -= old
                            changed = True
                        if child.min_subtree_weight + EPS < weight:
                            changed = self._eliminate_below(child_idx, weight, include_self=False) or changed  # type: ignore[arg-type]
                self._pull(idx)
            return changed

        new_ancestor = max(ancestor_max, node.entry_weight if node.entry_weight is not None else -inf)
        if node.left is not None:
            changed = self._insert_range(node.left, ql, qr, rank, weight, new_ancestor) or changed
        if node.right is not None:
            changed = self._insert_range(node.right, ql, qr, rank, weight, new_ancestor) or changed
        self._pull(idx)
        return changed

    def insert_full_prefix(self, y: float, rank: int, weight: float) -> bool:
        """Insert full-cell fragment covering cells with y_hi <= y."""
        if self.root < 0:
            return False
        # last cell whose upper cut <= y
        r = bisect_right(self.ycuts, y + EPS) - 2
        if r < 0:
            return False
        r = min(r, len(self.cells) - 1)
        return self._insert_range(self.root, 0, r, rank, weight, -inf)

    def find_y_cell(self, y: float) -> Optional[int]:
        j = bisect_right(self.ycuts, y) - 1
        if j < 0 or j >= len(self.cells):
            return None
        if self.ycuts[j] + EPS < y < self.ycuts[j + 1] - EPS:
            return j
        return None

    def insert_horizontal_boundary(self, y: float, rank: int, weight: float) -> bool:
        j = self.find_y_cell(y)
        if j is None:
            return False
        if self._max_entry_on_path(j) >= weight - EPS:
            return False
        height = y - self.ycuts[j]
        old = self.cells[j].weighted_volume
        changed = self.cells[j].insert_horizontal(height, rank, weight)
        if changed:
            self.total_weighted += self.cells[j].weighted_volume - old
            self._refresh_path_to_leaf(j)
        return changed

    def insert_vertical_boundary_prefix(self, y: float, width: float, rank: int, weight: float) -> bool:
        """Insert vertical boundary strip into all cells fully below y."""
        if width <= EPS:
            return False
        r = bisect_right(self.ycuts, y + EPS) - 2
        if r < 0:
            return False
        r = min(r, len(self.cells) - 1)
        changed_any = False
        for j in range(r + 1):
            if self._max_entry_on_path(j) >= weight - EPS:
                continue
            old = self.cells[j].weighted_volume
            changed = self.cells[j].insert_vertical(width, rank, weight)
            if changed:
                self.total_weighted += self.cells[j].weighted_volume - old
                self._refresh_path_to_leaf(j)
                changed_any = True
        return changed_any


class WeightedAnchoredRect2D_YildizSuri:
    """
    Dynamic weighted area of anchored 2D rectangles [0,x]x[0,y] with weight z.

    Insert-only. Query current integral of maximum weight.
    """

    def __init__(self, rectangles: Sequence[Tuple[float, ...]], block_size: Optional[int] = None):
        # rectangles are either (x,y,weight) or (x,y,weight,rank).
        raw = []
        for idx, rec in enumerate(rectangles):
            x, y, w = float(rec[0]), float(rec[1]), float(rec[2])
            if x > 0 and y > 0 and w > 0:
                raw.append((x, y, w, int(rec[3]) if len(rec) > 3 else idx))
        self.rectangles = raw
        self.n = len(self.rectangles)
        if self.n == 0:
            self.rank_weights = []
            self.weight_to_ranks = {}
            self.xcuts = [0.0, 0.0]
            self.slabs = []
            return
        self.X = max(x for x, _, _, _ in self.rectangles)
        self.Y = max(y for _, y, _, _ in self.rectangles)
        self.s = block_size or max(1, int(ceil(sqrt(self.n))))

        # Rank weights; ties are okay because the value used in formulas is the actual weight.
        # Distinct ranks are important even for equal weights; the Yildiz-Suri
        # formulas assume a strict order. Equal weights are tie-broken by input rank.
        ranked = sorted((w, rid) for _, _, w, rid in self.rectangles)
        self.rank_of_id = {rid: i for i, (w, rid) in enumerate(ranked)}
        self.rank_weights = [w for w, rid in ranked]

        self.xcuts = self._make_xcuts()
        self.slabs: List[SlabTree] = []
        for a, b in zip(self.xcuts[:-1], self.xcuts[1:]):
            if b <= a + EPS:
                continue
            ycuts = self._make_ycuts_for_slab(a, b)
            self.slabs.append(SlabTree(a, b, ycuts, self.rank_weights))

    def _make_xcuts(self) -> List[float]:
        xs = sorted(set(x for x, _, _, _ in self.rectangles if 0 < x < self.X))
        cuts = [0.0]
        for i in range(self.s - 1, len(xs), self.s):
            if xs[i] > cuts[-1] + EPS:
                cuts.append(xs[i])
        if self.X > cuts[-1] + EPS:
            cuts.append(self.X)
        if len(cuts) == 1:
            cuts.append(self.X)
        return cuts

    def _make_ycuts_for_slab(self, xlo: float, xhi: float) -> List[float]:
        # Horizontal boundaries whose segment passes through the slab.
        passing = sorted(y for x, y, _, _ in self.rectangles if x > xlo + EPS and 0 < y < self.Y)
        cuts = [0.0]
        for i in range(self.s - 1, len(passing), self.s):
            y = passing[i]
            if y > cuts[-1] + EPS:
                cuts.append(y)
        # Box corners inside this slab: force their y-coordinate to be a cut.
        for x, y, _, _ in self.rectangles:
            if xlo + EPS < x < xhi - EPS and 0 < y < self.Y:
                cuts.append(y)
        cuts.append(self.Y)
        cuts = sorted(set(round(c, 15) for c in cuts))
        # remove non-increasing duplicates after rounding
        out = []
        for c in cuts:
            if not out or c > out[-1] + EPS:
                out.append(c)
        if len(out) < 2:
            out = [0.0, self.Y]
        return out

    def _rank(self, rank_id: int) -> int:
        return self.rank_of_id[rank_id]

    def insert(self, x: float, y: float, weight: float, rank_id: int) -> None:
        if self.n == 0 or x <= 0 or y <= 0 or weight <= 0:
            return
        x, y, weight = float(x), float(y), float(weight)
        rank = self._rank(rank_id)
        for slab in self.slabs:
            if slab.xlo >= x - EPS:
                break
            if slab.xhi <= x + EPS:
                # Full cells below y.
                slab.insert_full_prefix(y, rank, weight)
                # Top boundary if y cuts through a cell.
                slab.insert_horizontal_boundary(y, rank, weight)
            else:
                # This is the slab containing the right boundary x.
                slab.insert_vertical_boundary_prefix(y, x - slab.xlo, rank, weight)
                # Corner y was inserted as a cut in this slab, so no horizontal boundary needed.
                break

    def weighted_area(self) -> float:
        return sum(slab.total_weighted for slab in self.slabs)


# ----------------------- General anchored hypervolume API -----------------------

class _FrontierNode2D:
    """Treap node for the 2D anchored frontier sorted by x.

    The in-order sequence is a Pareto frontier with strictly increasing x and
    strictly decreasing y.  The subtree aggregate ``area0`` is the anchored
    area represented by that subsequence when the previous x-coordinate is 0.
    """

    __slots__ = ("x", "y", "prio", "left", "right", "first_y", "last_x", "min_y", "max_y", "area0")

    _rng = Random(987654321)

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)
        self.prio = self._rng.random()
        self.left = None
        self.right = None
        self.first_y = self.y
        self.last_x = self.x
        self.min_y = self.y
        self.max_y = self.y
        self.area0 = self.x * self.y


def _f_upd(t):
    if t is None:
        return None
    t.first_y = t.left.first_y if t.left is not None else t.y
    t.last_x = t.right.last_x if t.right is not None else t.x
    t.min_y = t.y
    t.max_y = t.y
    if t.left is not None:
        t.min_y = min(t.min_y, t.left.min_y)
        t.max_y = max(t.max_y, t.left.max_y)
    if t.right is not None:
        t.min_y = min(t.min_y, t.right.min_y)
        t.max_y = max(t.max_y, t.right.max_y)

    area = 0.0
    if t.left is not None:
        area += t.left.area0
        prev_x = t.left.last_x
    else:
        prev_x = 0.0
    area += (t.x - prev_x) * t.y
    if t.right is not None:
        area += t.right.area0 - t.x * t.right.first_y
    t.area0 = area
    return t


def _f_merge(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a.prio < b.prio:
        a.right = _f_merge(a.right, b)
        return _f_upd(a)
    b.left = _f_merge(a, b.left)
    return _f_upd(b)


def _f_split_le(t, x: float):
    """Split into keys <= x and keys > x."""
    if t is None:
        return None, None
    if t.x <= x + EPS:
        a, b = _f_split_le(t.right, x)
        t.right = a
        return _f_upd(t), b
    a, b = _f_split_le(t.left, x)
    t.left = b
    return a, _f_upd(t)


def _f_leftmost(t):
    while t is not None and t.left is not None:
        t = t.left
    return t


def _f_rightmost(t):
    while t is not None and t.right is not None:
        t = t.right
    return t


def _f_split_first_y_le(t, y: float):
    """Split decreasing-y frontier into prefix y>threshold and suffix y<=threshold."""
    if t is None:
        return None, None
    if t.min_y > y + EPS:
        return t, None
    if t.left is not None and t.left.min_y <= y + EPS:
        a, b = _f_split_first_y_le(t.left, y)
        t.left = b
        return a, _f_upd(t)
    if t.y <= y + EPS:
        left = t.left
        t.left = None
        return left, _f_upd(t)
    a, b = _f_split_first_y_le(t.right, y)
    t.right = a
    return _f_upd(t), b


class DynamicAnchoredArea2D:
    """Insert-only union area of 2D anchored rectangles.

    Maintains the Pareto frontier of inserted corners (x,y).  Insertions take
    expected O(log n) time in the treap model; dominated frontier suffixes are
    removed by aggregate treap splits rather than by scanning a Python list.
    """

    def __init__(self):
        self.root = None
        self.size = 0

    def insert(self, x: float, y: float) -> bool:
        x = float(x)
        y = float(y)
        if x <= 0 or y <= 0:
            return False
        le, gt = _f_split_le(self.root, x)
        # Dominance can come from the first point strictly to the right, or
        # from an existing point with the same x-coordinate in ``le``.
        succ = _f_leftmost(gt)
        same_or_left = _f_rightmost(le)
        if (succ is not None and succ.y >= y - EPS) or (same_or_left is not None and abs(same_or_left.x - x) <= EPS and same_or_left.y >= y - EPS):
            self.root = _f_merge(le, gt)
            return False
        keep, _drop = _f_split_first_y_le(le, y)
        node = _FrontierNode2D(x, y)
        self.root = _f_merge(_f_merge(keep, node), gt)
        self.size += 1
        return True

    def area(self) -> float:
        return 0.0 if self.root is None else self.root.area0


def anchored_hypervolume_1d(points: Sequence[Tuple[float, ...]]) -> float:
    vals = [float(p[0]) for p in points if len(p) >= 1 and p[0] > 0]
    return max(vals) if vals else 0.0


def anchored_hypervolume_2d(points: Sequence[Tuple[float, ...]]) -> float:
    """O(n log n) anchored rectangle union area."""
    pts = [(float(p[0]), float(p[1])) for p in points if len(p) >= 2 and p[0] > 0 and p[1] > 0]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: (-p[0], -p[1]))
    area = 0.0
    max_y = 0.0
    i = 0
    while i < len(pts):
        xcur = pts[i][0]
        while i < len(pts) and abs(pts[i][0] - xcur) <= EPS:
            if pts[i][1] > max_y:
                max_y = pts[i][1]
            i += 1
        xnext = pts[i][0] if i < len(pts) else 0.0
        area += (xcur - xnext) * max_y
    return area


def anchored_hypervolume_3d(points: Sequence[Tuple[float, ...]]) -> float:
    """O(n log n) sweep for 3D anchored boxes using a dynamic 2D frontier."""
    pts = [(float(p[0]), float(p[1]), float(p[2])) for p in points if len(p) >= 3 and all(c > 0 for c in p[:3])]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: -p[2])
    ds = DynamicAnchoredArea2D()
    volume = 0.0
    i = 0
    while i < len(pts):
        zcur = pts[i][2]
        while i < len(pts) and abs(pts[i][2] - zcur) <= EPS:
            ds.insert(pts[i][0], pts[i][1])
            i += 1
        znext = pts[i][2] if i < len(pts) else 0.0
        volume += ds.area() * (zcur - znext)
    return volume


def anchored_hypervolume_recursive_general(points: Sequence[Tuple[float, ...]]) -> float:
    """Exact fallback for arbitrary fixed dimension.

    This is a straightforward recursive sweep.  It is useful for testing and for
    dimensions not covered by the optimized special cases, but it is not the
    general Yildiz--Suri weighted-volume data structure.
    """
    pts = [tuple(map(float, p)) for p in points if p and all(c > 0 for c in p)]
    if not pts:
        return 0.0
    d = len(pts[0])
    if any(len(p) != d for p in pts):
        raise ValueError("all points must have the same dimension")
    if d == 1:
        return anchored_hypervolume_1d(pts)
    if d == 2:
        return anchored_hypervolume_2d(pts)
    if d == 3:
        return anchored_hypervolume_3d(pts)
    if d == 4:
        return anchored_hypervolume_4d_yildiz(pts)

    pts_sorted = sorted(pts, key=lambda p: -p[-1])
    vol = 0.0
    prefix: List[Tuple[float, ...]] = []
    i = 0
    while i < len(pts_sorted):
        cur = pts_sorted[i][-1]
        while i < len(pts_sorted) and abs(pts_sorted[i][-1] - cur) <= EPS:
            prefix.append(pts_sorted[i][:-1])
            i += 1
        nxt = pts_sorted[i][-1] if i < len(pts_sorted) else 0.0
        vol += anchored_hypervolume(prefix) * (cur - nxt)
    return vol


def anchored_hypervolume(points: Sequence[Tuple[float, ...]], *, method: str = "auto", prune: bool = False) -> float:
    """Exact anchored hypervolume in any fixed dimension.

    Parameters
    ----------
    points:
        Iterable of points p.  Each point represents the anchored box
        [0,p_1] x ... x [0,p_d].  Coordinates must be nonnegative.
    method:
        ``"auto"`` dispatches to optimized implementations for d <= 4 and to
        the recursive exact fallback for d > 4.  ``"recursive"`` forces the
        fallback.  ``"yildiz4d"`` requires d=4 and calls the specialized 4D
        Yildiz--Suri-style implementation.
    prune:
        Passed to the 4D special case when used directly.  Leave False for the
        stated asymptotic bound of the 4D solver.

    Complexity
    ----------
    d=1: O(n)
    d=2: O(n log n)
    d=3: expected O(n log n)
    d=4: expected amortized O(n^(3/2) log n) via the special implementation
    d>4: exact recursive fallback; not the full general Yildiz--Suri structure.
    """
    pts = [tuple(map(float, p)) for p in points if p]
    if not pts:
        return 0.0
    d = len(pts[0])
    if any(len(p) != d for p in pts):
        raise ValueError("all points must have the same dimension")
    if any(c < 0 for p in pts for c in p):
        raise ValueError("anchored_hypervolume expects nonnegative coordinates")
    pts = [p for p in pts if all(c > 0 for c in p)]
    if not pts:
        return 0.0

    if method == "yildiz4d":
        if d != 4:
            raise ValueError("method='yildiz4d' requires four-dimensional input")
        return anchored_hypervolume_4d_yildiz(pts, prune=prune)  # type: ignore[arg-type]
    if method == "recursive":
        return anchored_hypervolume_recursive_general(pts)
    if method != "auto":
        raise ValueError("method must be one of 'auto', 'recursive', or 'yildiz4d'")

    if d == 1:
        return anchored_hypervolume_1d(pts)
    if d == 2:
        return anchored_hypervolume_2d(pts)
    if d == 3:
        return anchored_hypervolume_3d(pts)
    if d == 4:
        return anchored_hypervolume_4d_yildiz(pts, prune=prune)  # type: ignore[arg-type]
    return anchored_hypervolume_recursive_general(pts)

# ----------------------- 4D anchored volume -----------------------


def remove_dominated(points: Sequence[Tuple[float, ...]]) -> List[Tuple[float, ...]]:
    """Remove points whose anchored box is contained in another anchored box."""
    pts = [tuple(map(float, p)) for p in points if all(c > 0 for c in p)]
    keep = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i != j and all(p[k] <= q[k] + EPS for k in range(len(p))) and any(p[k] < q[k] - EPS for k in range(len(p))):
                dominated = True
                break
        if not dominated:
            keep.append(p)
    # Remove exact duplicates.
    return sorted(set(keep))


def anchored_hypervolume_4d_yildiz(points: Sequence[Tuple[float, float, float, float]], *, prune: bool = False) -> float:
    """
    Exact 4D anchored union volume using the Yildiz-Suri-style algorithm.

    Leave prune=False for the stated subquadratic bound.  The optional
    dominance pruning routine is quadratic and is intended only for small tests.
    """
    pts = [tuple(map(float, p)) for p in points if all(c > 0 for c in p)]
    if prune:
        pts = remove_dominated(pts)
    if not pts:
        return 0.0

    # Rectangles to be inserted into the weighted 2D DS are (x,y,weight=z).
    rects = [(x, y, z, rid) for rid, (x, y, z, _w) in enumerate(pts)]
    ds = WeightedAnchoredRect2D_YildizSuri(rects)

    # Sweep descending w. Insert all points with the same w, then integrate down to next w.
    pts_with_id = [(x, y, z, w, rid) for rid, (x, y, z, w) in enumerate(pts)]
    pts_sorted = sorted(pts_with_id, key=lambda p: -p[3])
    volume = 0.0
    i = 0
    while i < len(pts_sorted):
        wcur = pts_sorted[i][3]
        while i < len(pts_sorted) and abs(pts_sorted[i][3] - wcur) <= EPS:
            x, y, z, _w, rid = pts_sorted[i]
            ds.insert(x, y, z, rid)
            i += 1
        wnext = pts_sorted[i][3] if i < len(pts_sorted) else 0.0
        volume += ds.weighted_area() * (wcur - wnext)
    return volume


# ----------------------- Baselines for testing -----------------------


def anchored_hypervolume_recursive(points: Sequence[Tuple[float, ...]]) -> float:
    """Simple exact recursive sweep baseline for small/medium instances."""
    pts = [tuple(map(float, p)) for p in points if all(c > 0 for c in p)]
    if not pts:
        return 0.0
    d = len(pts[0])
    pts = remove_dominated(pts)
    if not pts:
        return 0.0
    if d == 1:
        return max(p[0] for p in pts)
    pts_sorted = sorted(pts, key=lambda p: -p[-1])
    vol = 0.0
    prefix: List[Tuple[float, ...]] = []
    i = 0
    while i < len(pts_sorted):
        cur = pts_sorted[i][-1]
        while i < len(pts_sorted) and abs(pts_sorted[i][-1] - cur) <= EPS:
            prefix.append(pts_sorted[i][:-1])
            i += 1
        nxt = pts_sorted[i][-1] if i < len(pts_sorted) else 0.0
        vol += anchored_hypervolume_recursive(prefix) * (cur - nxt)
    return vol


def anchored_hypervolume_grid(points: Sequence[Tuple[float, ...]]) -> float:
    """Coordinate-compression brute force. Use only for small tests."""
    pts = [tuple(map(float, p)) for p in points if all(c > 0 for c in p)]
    if not pts:
        return 0.0
    d = len(pts[0])
    coords = []
    for k in range(d):
        vals = sorted(set([0.0] + [p[k] for p in pts]))
        coords.append(vals)
    vol = 0.0
    ranges = [range(len(coords[k]) - 1) for k in range(d)]
    for idxs in product(*ranges):
        hi = [coords[k][idxs[k] + 1] for k in range(d)]
        covered = any(all(hi[k] <= p[k] + EPS for k in range(d)) for p in pts)
        if covered:
            cell_vol = 1.0
            for k in range(d):
                cell_vol *= coords[k][idxs[k] + 1] - coords[k][idxs[k]]
            vol += cell_vol
    return vol


def _assert_close(a: float, b: float, label: str, tol: float = 1e-8) -> None:
    if abs(a - b) > tol * max(1.0, abs(a), abs(b)):
        raise AssertionError(f"{label}: {a} != {b}")


def run_tests(verbose: bool = True) -> None:
    examples = []
    examples.append(([(1, 1, 1, 1)], 1.0, "single unit 4-box"))
    examples.append(([(1, 1, 1, 1), (2, 2, 2, 2)], 16.0, "one dominated by larger"))
    examples.append(([(1, 1, 1, 1), (2, 1, 1, 1)], 2.0, "two boxes same yzw"))
    examples.append(([(1, 2, 1, 2), (2, 1, 2, 1)], None, "two crossing boxes"))
    examples.append(([(0.5, 0.5, 0.5, 0.5), (1, 0.25, 1, 0.25), (0.25, 1, 0.25, 1)], None, "fractional"))

    for pts, expected, label in examples:
        got = anchored_hypervolume_4d_yildiz(pts)
        base = expected if expected is not None else anchored_hypervolume_grid(pts)
        _assert_close(got, base, label)
        if verbose:
            print(f"{label:28s} got={got:.12g} expected={base:.12g}")

    rng = Random(7)
    for n in range(1, 18):
        for trial in range(80):
            pts = []
            for _ in range(n):
                # discrete coordinates create ties and degeneracies too
                pts.append(tuple(rng.randint(1, 9) / 9.0 for _ in range(4)))
            got = anchored_hypervolume_4d_yildiz(pts)
            base = anchored_hypervolume_recursive(pts)
            _assert_close(got, base, f"random n={n} trial={trial}", tol=1e-7)
    if verbose:
        print("Random tests n=1..17, 80 trials each: passed")

    # A few larger tests against the recursive baseline.
    for n in [25, 40, 60]:
        for trial in range(10):
            pts = [tuple(rng.random() for _ in range(4)) for _ in range(n)]
            got = anchored_hypervolume_4d_yildiz(pts)
            base = anchored_hypervolume_recursive(pts)
            _assert_close(got, base, f"larger n={n} trial={trial}", tol=1e-7)
    if verbose:
        print("Larger random tests n=25,40,60: passed")


if __name__ == "__main__":
    run_tests(verbose=True)

# ---------------------------------------------------------------------------
# General fixed-dimension Yildiz--Suri-style weighted structures
# ---------------------------------------------------------------------------

class OrderedProductTree:
    """Maintain ``sum_i w_i * prod_j X_j[i]`` under range additions.

    The dimension ``dim`` is assumed to be a fixed constant.  Each node stores
    all subset-product sums

        S[mask] = sum_i w_i * prod_{j in mask} X_j[i].

    Adding ``delta`` to coordinate ``a`` over a range transforms every subset
    containing ``a`` by ``S[mask] += delta * S[mask without a]``.  This is the
    ordered-product primitive used by Yildiz--Suri for weighted halfspaces.
    """

    def __init__(self, weights: Sequence[float], lengths: Sequence[float]):
        self.weights = list(map(float, weights))
        self.lengths = list(map(float, lengths))
        self.dim = len(self.lengths)
        self.n = len(self.weights)
        self.size_masks = 1 << self.dim
        self.sums: List[List[float]] = []
        self.lazy: List[List[float]] = []
        if self.n:
            self.sums = [[0.0] * self.size_masks for _ in range(4 * self.n + 5)]
            self.lazy = [[0.0] * self.dim for _ in range(4 * self.n + 5)]
            self._build(1, 0, self.n - 1)

    def _leaf_sums(self, weight: float) -> List[float]:
        out = [0.0] * self.size_masks
        for mask in range(self.size_masks):
            prodv = weight
            for j in range(self.dim):
                if mask & (1 << j):
                    prodv *= self.lengths[j]
            out[mask] = prodv
        return out

    def _build(self, node: int, l: int, r: int) -> None:
        if l == r:
            self.sums[node] = self._leaf_sums(self.weights[l])
            return
        m = (l + r) // 2
        self._build(node * 2, l, m)
        self._build(node * 2 + 1, m + 1, r)
        for mask in range(self.size_masks):
            self.sums[node][mask] = self.sums[node * 2][mask] + self.sums[node * 2 + 1][mask]

    def _apply(self, node: int, axis: int, delta: float) -> None:
        if abs(delta) <= EPS:
            return
        bit = 1 << axis
        old = self.sums[node]
        new = old[:]
        # Use old lower-subset values, so iterate over all masks based on old.
        for mask in range(self.size_masks):
            if mask & bit:
                new[mask] = old[mask] + delta * old[mask ^ bit]
        self.sums[node] = new
        self.lazy[node][axis] += delta

    def _push(self, node: int) -> None:
        for a, delta in enumerate(self.lazy[node]):
            if abs(delta) > EPS:
                self._apply(node * 2, a, delta)
                self._apply(node * 2 + 1, a, delta)
                self.lazy[node][a] = 0.0

    def range_add(self, axis: int, lq: int, rq: int, delta: float) -> None:
        if self.n == 0 or rq < lq or rq < 0 or lq >= self.n or abs(delta) <= EPS:
            return
        lq = max(0, lq)
        rq = min(self.n - 1, rq)
        self._range_add(1, 0, self.n - 1, axis, lq, rq, delta)

    def _range_add(self, node: int, l: int, r: int, axis: int, lq: int, rq: int, delta: float) -> None:
        if lq <= l and r <= rq:
            self._apply(node, axis, delta)
            return
        self._push(node)
        m = (l + r) // 2
        if lq <= m:
            self._range_add(node * 2, l, m, axis, lq, rq, delta)
        if rq > m:
            self._range_add(node * 2 + 1, m + 1, r, axis, lq, rq, delta)
        for mask in range(self.size_masks):
            self.sums[node][mask] = self.sums[node * 2][mask] + self.sums[node * 2 + 1][mask]

    def full_product_sum(self) -> float:
        if self.n == 0:
            return 0.0
        return self.sums[1][self.size_masks - 1]


class WeightedHalfspaceDS:
    """Weighted volume of anchored axis-parallel halfspaces in fixed dimension.

    The local universe consists of a fixed set of possible halfspaces, each with
    a unique global id and a weight.  Updates insert halfspaces of the form
    ``0 <= x_axis <= extent`` in a rectangular cell with side lengths
    ``lengths``.  The current value is the integral of the maximum weight of any
    inserted halfspace covering a point of the cell.
    """

    def __init__(self, lengths: Sequence[float], candidates: Sequence[Tuple[int, float]]):
        self.lengths = [float(x) for x in lengths]
        self.dim = len(self.lengths)
        ranked = sorted((float(w), int(gid)) for gid, w in candidates)
        self.rank_of_id = {gid: i for i, (w, gid) in enumerate(ranked)}
        self.rank_weights = [w for w, gid in ranked]
        self.n = len(self.rank_weights)
        self.after = OrderedProductTree(self.rank_weights, self.lengths)
        self.ge = OrderedProductTree(self.rank_weights, self.lengths)
        self.axis_totals = [0.0] * self.dim
        self.gradients = [Gradient1D(self.lengths[a], self._make_delta(a)) for a in range(self.dim)]

    def _make_delta(self, axis: int):
        def on_delta(rank: int, delta: float) -> None:
            if abs(delta) <= EPS:
                return
            self.axis_totals[axis] += delta
            # A_axis[rank] += delta.  Complement lengths decrease for all
            # rank thresholds for which this segment is included.
            if rank > 0:
                self.after.range_add(axis, 0, rank - 1, -delta)
            self.ge.range_add(axis, 0, rank, -delta)
        return on_delta

    def insert_halfspace(self, axis: int, extent: float, gid: int) -> bool:
        if gid not in self.rank_of_id:
            # This should not happen if the partition candidates are correct,
            # but silently ignoring makes the structure robust for degenerate
            # zero-volume fragments.
            return False
        rank = self.rank_of_id[gid]
        weight = self.rank_weights[rank]
        return self.gradients[axis].insert(extent, rank, weight)

    def eliminate_below(self, threshold: float) -> bool:
        changed = False
        for g in self.gradients:
            changed = g.eliminate_below(threshold) or changed
        return changed

    def weighted_volume(self) -> float:
        return self.after.full_product_sum() - self.ge.full_product_sum()

    def ordinary_volume(self) -> float:
        if self.dim == 0:
            return 0.0
        uncovered = 1.0
        total = 1.0
        for L, A in zip(self.lengths, self.axis_totals):
            total *= L
            uncovered *= max(0.0, L - A)
        return total - uncovered

    def min_weight(self) -> float:
        vals = [g.min_weight() for g in self.gradients]
        return min(vals) if vals else inf


@dataclass
class _WBox:
    coords: Tuple[float, ...]
    weight: float
    gid: int


class _YSPartitionNode:
    __slots__ = (
        "lows", "highs", "left", "right", "leaf_ds", "volume",
        "entry_weight", "A", "contrib", "min_subtree_weight",
    )

    def __init__(self, lows: Sequence[float], highs: Sequence[float]):
        self.lows = tuple(map(float, lows))
        self.highs = tuple(map(float, highs))
        self.left: Optional[_YSPartitionNode] = None
        self.right: Optional[_YSPartitionNode] = None
        self.leaf_ds: Optional[WeightedHalfspaceDS] = None
        vol = 1.0
        for a, b in zip(self.lows, self.highs):
            vol *= max(0.0, b - a)
        self.volume = vol
        self.entry_weight: Optional[float] = None
        self.A = 0.0
        self.contrib = 0.0
        self.min_subtree_weight = inf


class WeightedAnchoredBoxesYS:
    """Yildiz--Suri-style dynamic weighted volume for anchored boxes.

    This is the general fixed-dimensional version used after the sweep/weighting
    reduction.  It maintains weighted volume of boxes ``[0,p_1] x ... x
    [0,p_m]`` with weights under insert-only updates.  The construction follows
    the Overmars--Yap/Yildiz--Suri trellis partition: leaves contain only
    halfspace fragments, and full-cell fragments are stored in a binary partition
    tree with elimination of dominated lower-weight fragments.

    The implementation is intended for fixed small ``m``.  Constants grow roughly
    as ``2^m`` because of the ordered-product structure for halfspaces.
    """

    def __init__(self, boxes: Sequence[Tuple[Sequence[float], float, int]], block_size: Optional[int] = None):
        raw = []
        for coords, weight, gid in boxes:
            c = tuple(float(x) for x in coords)
            w = float(weight)
            if c and all(x > 0 for x in c) and w > 0:
                raw.append(_WBox(c, w, int(gid)))
        self.boxes = raw
        self.n = len(raw)
        self.dim = len(raw[0].coords) if raw else 0
        if any(len(b.coords) != self.dim for b in raw):
            raise ValueError("all weighted boxes must have the same dimension")
        self.s = block_size or max(1, int(ceil(sqrt(max(1, self.n)))))
        self.total_weighted = 0.0
        if not raw:
            self.root = None
            return
        highs = [max(b.coords[a] for b in raw) for a in range(self.dim)]
        lows = [0.0] * self.dim
        ids = list(range(self.n))
        self.root = self._build_region(lows, highs, ids, 0)
        self._pull(self.root)

    def _intersects(self, b: _WBox, lows: Sequence[float], highs: Sequence[float]) -> bool:
        return all(b.coords[a] > lows[a] + EPS and highs[a] > lows[a] + EPS for a in range(self.dim))

    def _contains_region(self, b: _WBox, lows: Sequence[float], highs: Sequence[float]) -> bool:
        return all(b.coords[a] >= highs[a] - EPS for a in range(self.dim))

    def _partial_axes(self, b: _WBox, lows: Sequence[float], highs: Sequence[float]) -> List[int]:
        out = []
        for a in range(self.dim):
            if lows[a] + EPS < b.coords[a] < highs[a] - EPS:
                out.append(a)
        return out

    def _build_region(self, lows: Sequence[float], highs: Sequence[float], ids: List[int], axis: int) -> _YSPartitionNode:
        if axis >= self.dim:
            node = _YSPartitionNode(lows, highs)
            lengths = [highs[a] - lows[a] for a in range(self.dim)]
            candidates = []
            for idx in ids:
                b = self.boxes[idx]
                if not self._intersects(b, lows, highs) or self._contains_region(b, lows, highs):
                    continue
                pa = self._partial_axes(b, lows, highs)
                if len(pa) == 1:
                    candidates.append((b.gid, b.weight))
                elif len(pa) > 1:
                    # The recursive cut rule is supposed to prevent this.  We
                    # keep no candidate; insertion will raise a clearer error.
                    pass
            # Deduplicate candidates by id.
            cand_dict = {gid: w for gid, w in candidates}
            node.leaf_ds = WeightedHalfspaceDS(lengths, list(cand_dict.items()))
            return node

        lo, hi = lows[axis], highs[axis]
        inside = []
        forced = []
        for idx in ids:
            b = self.boxes[idx]
            x = b.coords[axis]
            if lo + EPS < x < hi - EPS:
                inside.append(x)
                if any(lows[j] + EPS < b.coords[j] < highs[j] - EPS for j in range(axis)):
                    forced.append(x)
        inside.sort()
        cuts = [lo, hi]
        # Quantile cuts: each interval contains O(sqrt(n)) bounds in this axis.
        for k in range(self.s - 1, len(inside), self.s):
            cuts.append(inside[k])
        cuts.extend(forced)
        cuts = sorted(cuts)
        uniq = []
        for c in cuts:
            if not uniq or c > uniq[-1] + EPS:
                uniq.append(c)
        cuts = uniq
        if len(cuts) <= 2:
            return self._build_region(lows, highs, ids, axis + 1)

        def build_interval_tree(i: int, j: int) -> _YSPartitionNode:
            # Represents the slab [cuts[i], cuts[j]] along the current axis.
            nl = list(lows)
            nh = list(highs)
            nl[axis] = cuts[i]
            nh[axis] = cuts[j]
            relevant = [idx for idx in ids if self._intersects(self.boxes[idx], nl, nh)]
            if j == i + 1:
                return self._build_region(nl, nh, relevant, axis + 1)
            mid = (i + j) // 2
            node = _YSPartitionNode(nl, nh)
            node.left = build_interval_tree(i, mid)
            node.right = build_interval_tree(mid, j)
            self._pull(node)
            return node

        return build_interval_tree(0, len(cuts) - 1)

    def _node_min(self, node: _YSPartitionNode) -> float:
        vals = []
        if node.entry_weight is not None:
            vals.append(node.entry_weight)
        if node.leaf_ds is not None:
            vals.append(node.leaf_ds.min_weight())
        if node.left is not None:
            vals.append(node.left.min_subtree_weight)
        if node.right is not None:
            vals.append(node.right.min_subtree_weight)
        return min(vals) if vals else inf

    def _pull(self, node: Optional[_YSPartitionNode]) -> None:
        if node is None:
            return
        old = node.contrib
        if node.leaf_ds is not None:
            node.A = node.leaf_ds.ordinary_volume()
        else:
            a = 0.0
            for child in (node.left, node.right):
                if child is None:
                    continue
                a += child.volume if child.entry_weight is not None else child.A
            node.A = a
        node.contrib = (node.entry_weight * max(0.0, node.volume - node.A)) if node.entry_weight is not None else 0.0
        self.total_weighted += node.contrib - old
        node.min_subtree_weight = self._node_min(node)

    def _disjoint_coords(self, coords: Sequence[float], node: _YSPartitionNode) -> bool:
        return any(coords[a] <= node.lows[a] + EPS for a in range(self.dim))

    def _contains_coords(self, coords: Sequence[float], node: _YSPartitionNode) -> bool:
        return all(coords[a] >= node.highs[a] - EPS for a in range(self.dim))

    def _eliminate_below(self, node: _YSPartitionNode, threshold: float, include_self: bool = True) -> bool:
        if node.min_subtree_weight + EPS >= threshold:
            return False
        changed = False
        if include_self and node.entry_weight is not None and node.entry_weight + EPS < threshold:
            old = node.contrib
            node.entry_weight = None
            node.contrib = 0.0
            self.total_weighted -= old
            changed = True
        if node.leaf_ds is not None:
            oldwv = node.leaf_ds.weighted_volume()
            if node.leaf_ds.eliminate_below(threshold):
                self.total_weighted += node.leaf_ds.weighted_volume() - oldwv
                changed = True
            self._pull(node)
            return changed
        for child in (node.left, node.right):
            if child is None:
                continue
            if child.entry_weight is not None and child.entry_weight + EPS < threshold:
                old = child.contrib
                child.entry_weight = None
                child.contrib = 0.0
                self.total_weighted -= old
                changed = True
            if child.min_subtree_weight + EPS < threshold:
                changed = self._eliminate_below(child, threshold, include_self=False) or changed
        self._pull(node)
        return changed

    def insert(self, coords: Sequence[float], weight: float, gid: int) -> None:
        if self.root is None:
            return
        c = tuple(float(x) for x in coords)
        if len(c) != self.dim:
            raise ValueError("inserted box has wrong dimension")
        if any(x <= 0 for x in c) or weight <= 0:
            return
        self._insert_rec(self.root, c, float(weight), int(gid), -inf)

    def _insert_rec(self, node: _YSPartitionNode, coords: Tuple[float, ...], weight: float, gid: int, ancestor_max: float) -> bool:
        if self._disjoint_coords(coords, node):
            return False
        if ancestor_max >= weight - EPS:
            return False
        if self._contains_coords(coords, node):
            if node.entry_weight is not None and node.entry_weight >= weight - EPS:
                return False
            node.entry_weight = weight
            # The new full-cell fragment dominates lower-weight fragments below.
            if node.leaf_ds is not None:
                oldwv = node.leaf_ds.weighted_volume()
                if node.leaf_ds.eliminate_below(weight):
                    self.total_weighted += node.leaf_ds.weighted_volume() - oldwv
            else:
                for child in (node.left, node.right):
                    if child is not None and child.min_subtree_weight + EPS < weight:
                        self._eliminate_below(child, weight, include_self=True)
            self._pull(node)
            return True

        new_ancestor = max(ancestor_max, node.entry_weight if node.entry_weight is not None else -inf)
        if node.leaf_ds is not None:
            if node.entry_weight is not None and node.entry_weight >= weight - EPS:
                return False
            axes = [a for a in range(self.dim) if node.lows[a] + EPS < coords[a] < node.highs[a] - EPS]
            if len(axes) == 0:
                # Numerically on a boundary; either full or empty was handled.
                return False
            if len(axes) > 1:
                raise RuntimeError(
                    "partition invariant violated: a boundary fragment is not a halfspace"
                )
            a = axes[0]
            oldwv = node.leaf_ds.weighted_volume()
            changed = node.leaf_ds.insert_halfspace(a, coords[a] - node.lows[a], gid)
            if changed:
                self.total_weighted += node.leaf_ds.weighted_volume() - oldwv
                self._pull(node)
            return changed

        changed = False
        if node.left is not None:
            changed = self._insert_rec(node.left, coords, weight, gid, new_ancestor) or changed
        if node.right is not None:
            changed = self._insert_rec(node.right, coords, weight, gid, new_ancestor) or changed
        if changed:
            self._pull(node)
        return changed

    def weighted_volume(self) -> float:
        return self.total_weighted


def anchored_hypervolume_yildiz_general(points: Sequence[Tuple[float, ...]], *, block_size: Optional[int] = None) -> float:
    """Yildiz--Suri-style anchored hypervolume for arbitrary fixed dimension.

    For d <= 4 this dispatches to the existing specialized routines.  For d > 4
    it performs the Yildiz--Suri sweep/weighting reduction and maintains the
    resulting weighted (d-2)-dimensional anchored boxes with the general trellis
    data structure above.
    """
    pts = [tuple(map(float, p)) for p in points if p and all(float(c) > 0 for c in p)]
    if not pts:
        return 0.0
    d = len(pts[0])
    if any(len(p) != d for p in pts):
        raise ValueError("all points must have the same dimension")
    if d <= 4:
        return anchored_hypervolume(pts, method="auto")

    weighted_boxes = [(p[:-2], p[-2], i) for i, p in enumerate(pts)]
    ds = WeightedAnchoredBoxesYS(weighted_boxes, block_size=block_size)
    events = sorted([(p[-1], i, p) for i, p in enumerate(pts)], reverse=True)
    vol = 0.0
    i = 0
    while i < len(events):
        cur = events[i][0]
        while i < len(events) and abs(events[i][0] - cur) <= EPS:
            _wlast, gid, p = events[i]
            ds.insert(p[:-2], p[-2], gid)
            i += 1
        nxt = events[i][0] if i < len(events) else 0.0
        vol += ds.weighted_volume() * (cur - nxt)
    return vol


# Replace the public dispatcher with one that can call the general YS structure.
_anchored_hypervolume_old_dispatch = anchored_hypervolume

def anchored_hypervolume(points: Sequence[Tuple[float, ...]], *, method: str = "auto", prune: bool = False) -> float:  # type: ignore[no-redef]
    """Exact anchored hypervolume in any fixed dimension.

    Methods
    -------
    ``auto``
        Uses direct optimal/specialized routines for d<=4 and the general
        Yildiz--Suri-style trellis structure for d>4.
    ``yildiz-general``
        Forces the general Yildiz--Suri-style implementation for d>4 and
        dispatches to special routines for d<=4.
    ``yildiz4d``
        Forces the optimized 4D special case.
    ``recursive``
        Uses the exact recursive fallback, mainly for verification.
    """
    pts = [tuple(map(float, p)) for p in points if p]
    if not pts:
        return 0.0
    d = len(pts[0])
    if any(len(p) != d for p in pts):
        raise ValueError("all points must have the same dimension")
    if any(c < 0 for p in pts for c in p):
        raise ValueError("anchored_hypervolume expects nonnegative coordinates")
    pts = [p for p in pts if all(c > 0 for c in p)]
    if not pts:
        return 0.0
    if method in ("auto", "yildiz-general"):
        if d <= 4:
            return _anchored_hypervolume_old_dispatch(pts, method="auto", prune=prune)
        return anchored_hypervolume_yildiz_general(pts)
    return _anchored_hypervolume_old_dispatch(pts, method=method, prune=prune)

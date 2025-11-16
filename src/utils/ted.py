"""
Tree Edit Distance (TED) for SymPy expressions.

This module converts SymPy expressions into ordered trees and computes the
tree edit distance (unit costs) between two expressions. It also supports
optional canonicalization of commutative operators to reduce sensitivity
to argument order.

Typical uses in symbolic regression:
- Quantify structural difference between learned and ground-truth formulas
- Report raw TED and a normalized score (e.g., TED / max(|T1|, |T2|))

Dependencies: sympy
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Callable, Sequence, Dict
from functools import lru_cache
from sympy import sympify, symbols, Number, Basic, lambdify
from sympy.core.operations import AssocOp

# -------------------- Tree --------------------

@dataclass(frozen=True)
class Node:
    """Immutable ordered tree node (hashable)."""
    label: str                  # e.g., 'Add', 'Mul', 'Pow', 'sin', 'x', '2.0'
    children: Tuple['Node', ...]
    kind: str                   # 'op' | 'sym' | 'num'
    numval: float | None        # numeric value if kind == 'num' else None

    def size(self, weight_fn: Callable[['Node'], float]) -> float:
        return weight_fn(self) + sum(c.size(weight_fn) for c in self.children)


def _label_kind(expr: Basic) -> Tuple[str, str, float | None]:
    """Return (label, kind, numeric_value?)."""
    if not expr.args:
        # leaf: number or symbol
        if isinstance(expr, Number):
            v = float(expr.evalf())
            return (str(v), 'num', v)
        else:
            # symbol or other atomic
            return (str(expr), 'sym', None)
    # internal op
    return (expr.func.__name__, 'op', None)


def _canonicalize_children(expr: Basic) -> Tuple[Basic, Tuple[Basic, ...]]:
    """Sort commutative associative ops to reduce order sensitivity."""
    if isinstance(expr, AssocOp) and getattr(expr, "is_commutative", False):
        args = tuple(sorted(expr.args, key=lambda a: a.sort_key()))
        return expr, args
    if expr.func.__name__ in {'Add', 'Mul'}:
        args = tuple(sorted(expr.args, key=lambda a: a.sort_key()))
        return expr, args
    return expr, tuple(expr.args)


def sympy_to_tree(expr: Basic,
                  canonicalize_commutative: bool = True,
                  enforce_mul_for_add_terms: bool = True) -> Node:
    """
    Convert a SymPy expression to a Node tree.

    If enforce_mul_for_add_terms=True, every non-numeric direct child of an Add
    is represented as a Mul with an explicit numeric coefficient:
        term -> Mul(1, term)   (if term is not a number and not already a Mul)

    This aligns e.g. 2*sin(x) and sin(x) structurally, so coefficient changes
    are charged at the numeric-leaf level rather than as structural edits.
    """
    # Canonicalize child order at the SymPy level when desired
    if canonicalize_commutative:
        expr, args = _canonicalize_children(expr)
    else:
        args = tuple(expr.args)

    label, kind, numval = _label_kind(expr)

    # Special handling for Add: wrap its non-numeric, non-Mul children
    if label == 'Add' and enforce_mul_for_add_terms:
        new_children = []
        for a in args:
            if a.is_Number or a.func.__name__ == 'Mul':
                # keep as-is
                new_children.append(sympy_to_tree(a, canonicalize_commutative, enforce_mul_for_add_terms))
            else:
                # wrap as Mul(1, a) at the NODE level (no need to rebuild SymPy object)
                one_node = Node('1.0', tuple(), 'num', 1.0)
                base_node = sympy_to_tree(a, canonicalize_commutative, enforce_mul_for_add_terms)
                mul_node = Node('Mul', (one_node, base_node), 'op', None)
                new_children.append(mul_node)
        return Node('Add', tuple(new_children), 'op', None)

    # Leaves
    if not args:
        return Node(label, tuple(), kind, numval)

    # Internal nodes
    return Node(label,
                tuple(sympy_to_tree(a, canonicalize_commutative, enforce_mul_for_add_terms) for a in args),
                kind, numval)

# -------------------- Weighted TED with coefficient/bias awareness --------------------

def make_costs(
    *,
    # numeric replacement: tolerance and scaling
    num_tol_abs: float = 1e-6,
    num_tol_rel: float = 1e-3,
    num_replace_cap: float = 1.0,     # cap numeric replacement cost
    num_replace_scale: float = 1.0,   # multiply numeric replacement cost
    # structural rename cost
    op_rename_cost: float = 1.0,
    sym_rename_cost: float = 1.0,     
    # insertion/deletion weights
    weight_num_leaf: float = 1.0,    
    weight_sym_leaf: float = 1.0,
    weight_op_node: float  = 1.0,
):
    """
    Returns:
      - node_weight(n): per-node weight used to compute subtree insert/delete costs
      - rename_cost(x,y): cost to rename x->y (0 if same label), numeric-aware
    """
    def node_weight(n: Node) -> float:
        if n.kind == 'num':
            return weight_num_leaf
        if n.kind == 'sym':
            return weight_sym_leaf
        return weight_op_node

    def numeric_close(a: float, b: float) -> bool:
        # close if |a-b| <= max(num_tol_abs, num_tol_rel*max(|a|,|b|))
        diff = abs(a - b)
        scale = max(num_tol_abs, num_tol_rel * max(abs(a), abs(b), 1.0))
        return diff <= scale

    def rename_cost(x: Node, y: Node) -> float:
        if x.kind == 'num' and y.kind == 'num':
            if numeric_close(x.numval, y.numval):
                return 0.0
            # smoothly scale cost with relative difference, then cap
            diff = abs(x.numval - y.numval)
            denom = max(abs(x.numval), abs(y.numval), 1.0)
            c = num_replace_scale * min(num_replace_cap, diff / denom)
            return c
        # symbols/operators
        if x.label == y.label and x.kind == y.kind:
            return 0.0
        # different kinds or labels
        if x.kind == 'op' and y.kind == 'op':
            return op_rename_cost
        if x.kind == 'sym' and y.kind == 'sym':
            return sym_rename_cost
        # crossing kinds (e.g., num <-> op)
        return max(op_rename_cost, sym_rename_cost, num_replace_cap)

    return node_weight, rename_cost


def ted_weighted(a: Node, b: Node,
                 node_weight: Callable[[Node], float],
                 rename_cost: Callable[[Node, Node], float]) -> float:
    """Ordered TED with weighted insert/delete and numeric-aware rename."""

    @lru_cache(maxsize=None)
    def _size(n: Node) -> float:
        return n.size(node_weight)

    @lru_cache(maxsize=None)
    def _dist(x: Node, y: Node) -> float:
        A, B = x.children, y.children
        m, n = len(A), len(B)
        dp = [[0.0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            dp[i][0] = dp[i-1][0] + _size(A[i-1])
        for j in range(1, n+1):
            dp[0][j] = dp[0][j-1] + _size(B[j-1])
        for i in range(1, m+1):
            for j in range(1, n+1):
                del_cost = dp[i-1][j] + _size(A[i-1])
                ins_cost = dp[i][j-1] + _size(B[j-1])
                match_cost = dp[i-1][j-1] + _dist(A[i-1], B[j-1])
                dp[i][j] = min(del_cost, ins_cost, match_cost)
        return dp[m][n] + rename_cost(x, y)

    return _dist(a, b)


def ted_weighted_normalized(a: Node, b: Node,
                            node_weight: Callable[[Node], float],
                            rename_cost: Callable[[Node, Node], float],
                            mode: str = "max") -> float:
    denom = max(a.size(node_weight), b.size(node_weight)) if mode == "max" \
            else (a.size(node_weight) + b.size(node_weight))
    if denom == 0:
        return 0.0
    return ted_weighted(a, b, node_weight, rename_cost) / denom
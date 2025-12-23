"""
Expression Complexity Module

This module provides utilities for computing the complexity of symbolic expressions,
as used in symbolic regression. Complexity metrics help evaluate the simplicity
and interpretability of mathematical expressions.

The visitation_length metric is implemented based on the definition from:
"PARETO-FRONT EXPLOITATION IN SYMBOLIC REGRESSION" by Guido F. Smits & Mark Kotanchek, 2005.
link: https://link.springer.com/chapter/10.1007/0-387-23254-0_17
"""

import sympy as sp
from typing import Union, List


def compute_complexity(expr: Union[sp.Expr, List[sp.Expr]], 
                      metric: str = 'node_count') -> int:
    """
    Compute the complexity of a symbolic expression or list of expressions.
    
    In symbolic regression, complexity is a measure of how "complicated" an
    expression is. Simpler expressions are generally preferred as they are
    more interpretable and less prone to overfitting.
    
    Args:
        expr: SymPy expression or list of expressions
        metric: Complexity metric to use. Options:
            - 'node_count': Total number of nodes in the expression tree (default)
            - 'depth': Maximum depth of the expression tree
            - 'visitation_length': Sum of sizes of all subtrees (structural complexity)
    
    Returns:
        Complexity score (int)
        
    Examples:
        >>> import sympy as sp
        >>> x = sp.Symbol('x')
        >>> expr = x**2 + 2*x + 1
        >>> compute_complexity(expr, metric='node_count')
        7
        >>> compute_complexity(expr, metric='depth')
        3
        >>> compute_complexity(expr, metric='visitation_length')
        21
    """
    # Handle list of expressions
    if isinstance(expr, list):
        return sum(compute_complexity(e, metric) for e in expr)
    
    # Select metric
    if metric == 'node_count':
        return _node_count(expr)
    elif metric == 'depth':
        return _tree_depth(expr)
    elif metric == 'visitation_length':
        return _visitation_length(expr)
    else:
        raise ValueError(f"Unknown metric: {metric}. Choose from: 'node_count', 'depth', 'visitation_length'")


def _node_count(expr: sp.Expr) -> int:
    """
    Count the total number of nodes in the expression tree.
    
    This is the most common complexity metric in symbolic regression.
    Every symbol, constant, and operation is counted as a node.
    
    Args:
        expr: SymPy expression
        
    Returns:
        Number of nodes
    """
    if expr.is_Atom:
        return 1
    
    count = 1  # Count current node
    for arg in expr.args:
        count += _node_count(arg)
    
    return count


def _tree_depth(expr: sp.Expr) -> int:
    """
    Calculate the maximum depth of the expression tree.
    
    Depth represents the maximum nesting level of operations.
    Deeper trees are generally more complex to interpret.
    
    Args:
        expr: SymPy expression
        
    Returns:
        Maximum tree depth
    """
    if expr.is_Atom:
        return 1
    
    if not expr.args:
        return 1
    
    max_child_depth = max(_tree_depth(arg) for arg in expr.args)
    return 1 + max_child_depth


def _visitation_length(expr: sp.Expr) -> int:
    """
    Calculate the visitation length of the expression tree.
    
    Visitation length measures structural complexity as the sum of sizes
    of all subtrees. Each node contributes the size of its entire subtree
    to the total. Deeper, more nested trees yield larger visitation lengths.
    
    Formally: VisitationLength(T) = Σ size(n) for all nodes n in T
    where size(n) = 1 + Σ size(c) for all children c of n
    
    Args:
        expr: SymPy expression
        
    Returns:
        Visitation length (sum of all subtree sizes)
        
    Example:
        For expression (x + 1) * x:
        Tree:    
                  (*)
                 /   \\
               (+)    x
              /   \\
             x     1
        
        Subtree sizes: * has size 5, + has size 3, each leaf has size 1
        Visitation length = 5 + 3 + 1 + 1 + 1 = 11
    """
    # Base case: atomic expression (leaf node)
    if expr.is_Atom:
        return 1
    
    # Recursively compute for all children
    # Returns: (subtree_size, visitation_length)
    def compute(node):
        if node.is_Atom:
            return (1, 1)
        
        # Compute for all children
        child_results = [compute(arg) for arg in node.args]
        
        # Subtree size = 1 (this node) + sum of children sizes
        subtree_size = 1 + sum(size for size, _ in child_results)
        
        # Visitation length = this subtree's size + sum of children's visitation lengths
        visitation = subtree_size + sum(vl for _, vl in child_results)
        
        return (subtree_size, visitation)
    
    _, result = compute(expr)
    return result


def complexity_report(expr: Union[sp.Expr, List[sp.Expr]]) -> dict:
    """
    Generate a comprehensive complexity report for an expression.
    
    Returns all complexity metrics in a dictionary.
    
    Args:
        expr: SymPy expression or list of expressions
        
    Returns:
        Dictionary with all complexity metrics
        
    Example:
        >>> import sympy as sp
        >>> x = sp.Symbol('x')
        >>> expr = x**2 + 2*x + 1
        >>> report = complexity_report(expr)
        >>> print(report)
        {
            'node_count': 7,
            'depth': 3,
            'visitation_length': 21
        }
    """
    return {
        'node_count': compute_complexity(expr, 'node_count'),
        'depth': compute_complexity(expr, 'depth'),
        'visitation_length': compute_complexity(expr, 'visitation_length')
    }

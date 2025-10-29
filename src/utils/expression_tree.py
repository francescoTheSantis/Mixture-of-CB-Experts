"""
Expression Tree Module

This module provides utilities for converting SymPy expressions into
custom expression trees that can be used for symbolic computation,
visualization, and graph-based reasoning.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import sympy as sp


@dataclass
class ExpressionTreeNode:
    """
    Represents a node in an expression tree.
    
    Attributes:
        op (str): The operator or function name (e.g., '+', '*', 'ReLU', 'tanh').
        children (list): List of child nodes (operands or sub-expressions).
        value (Optional): For leaf nodes, stores the actual value (variable, constant).
    """
    op: str
    children: List["ExpressionTreeNode"] = field(default_factory=list)
    value: Optional[object] = None
    
    def __repr__(self):
        if not self.children:
            return f"Node(op={self.op}, value={self.value})"
        return f"Node(op={self.op}, children={len(self.children)})"
    
    def to_string(self, indent=0):
        """
        Returns a string representation of the tree for visualization.
        
        Args:
            indent (int): Current indentation level.
            
        Returns:
            str: Formatted string representation of the tree.
        """
        prefix = "  " * indent
        if not self.children:
            return f"{prefix}{self.op}: {self.value}\n"
        
        result = f"{prefix}{self.op}\n"
        for child in self.children:
            result += child.to_string(indent + 1)
        return result


def sympy_to_tree(expr):
    """
    Converts a SymPy expression into a custom expression tree.
    
    This function recursively traverses a SymPy expression and constructs
    an expression tree where:
    - Leaves are constants, variables, or symbols
    - Internal nodes are operators or functions
    
    Args:
        expr (sympy.Expr): SymPy symbolic expression.
        
    Returns:
        ExpressionTreeNode: Root node of the expression tree.
        
    Examples:
        >>> import sympy as sp
        >>> x, y = sp.symbols('x y')
        >>> expr = (x + y) * 2
        >>> tree = sympy_to_tree(expr)
        >>> print(tree.to_string())
    """
    # Handle atoms (leaf nodes: symbols, numbers, etc.)
    if expr.is_Atom:
        if expr.is_Symbol:
            return ExpressionTreeNode(op="symbol", value=str(expr))
        elif expr.is_Number:
            return ExpressionTreeNode(op="constant", value=float(expr))
        else:
            return ExpressionTreeNode(op="atom", value=str(expr))
    
    # Handle matrix symbols
    if isinstance(expr, sp.MatrixSymbol):
        return ExpressionTreeNode(op="matrix", value=str(expr))
    
    # Handle matrix expressions
    if isinstance(expr, sp.MatMul):
        children = [sympy_to_tree(arg) for arg in expr.args]
        return ExpressionTreeNode(op="matmul", children=children)
    
    if isinstance(expr, sp.MatAdd):
        children = [sympy_to_tree(arg) for arg in expr.args]
        return ExpressionTreeNode(op="matadd", children=children)
    
    # Handle basic arithmetic operations
    if isinstance(expr, sp.Add):
        children = [sympy_to_tree(arg) for arg in expr.args]
        return ExpressionTreeNode(op="add", children=children)
    
    if isinstance(expr, sp.Mul):
        children = [sympy_to_tree(arg) for arg in expr.args]
        return ExpressionTreeNode(op="mul", children=children)
    
    if isinstance(expr, sp.Pow):
        base = sympy_to_tree(expr.args[0])
        exponent = sympy_to_tree(expr.args[1])
        return ExpressionTreeNode(op="pow", children=[base, exponent])
    
    # Handle applied undefined functions (e.g., ReLU(expr), Sigmoid(expr))
    if isinstance(expr, sp.core.function.AppliedUndef):
        func_name = str(expr.func)
        children = [sympy_to_tree(arg) for arg in expr.args]
        return ExpressionTreeNode(op=func_name, children=children)
    
    # Handle functions (including activation functions)
    if isinstance(expr, sp.Function):
        func_name = expr.func.__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return ExpressionTreeNode(op=func_name, children=children)
    
    # Handle custom functions by name
    if hasattr(expr, 'func') and hasattr(expr.func, '__name__'):
        func_name = expr.func.__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return ExpressionTreeNode(op=func_name, children=children)
    
    # Fallback: treat as a generic operation
    if hasattr(expr, 'args') and expr.args:
        op_name = type(expr).__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return ExpressionTreeNode(op=op_name, children=children)
    
    # Last resort: convert to string
    return ExpressionTreeNode(op="unknown", value=str(expr))


def print_tree(tree: ExpressionTreeNode):
    """
    Pretty-prints an expression tree.
    
    Args:
        tree (ExpressionTreeNode): Root node of the tree to print.
    """
    print(tree.to_string())


def tree_to_latex(tree: ExpressionTreeNode) -> str:
    """
    Converts an expression tree back to LaTeX format.
    
    Args:
        tree (ExpressionTreeNode): Root node of the expression tree.
        
    Returns:
        str: LaTeX representation of the expression.
    """
    if not tree.children:
        if tree.op == "constant":
            return str(tree.value)
        elif tree.op in ["symbol", "matrix"]:
            return tree.value
        else:
            return str(tree.value)
    
    if tree.op == "add":
        terms = [tree_to_latex(child) for child in tree.children]
        return "(" + " + ".join(terms) + ")"
    
    if tree.op == "mul":
        factors = [tree_to_latex(child) for child in tree.children]
        return "(" + r" \cdot ".join(factors) + ")"
    
    if tree.op == "matmul":
        matrices = [tree_to_latex(child) for child in tree.children]
        return "(" + " ".join(matrices) + ")"
    
    if tree.op == "matadd":
        terms = [tree_to_latex(child) for child in tree.children]
        return "(" + " + ".join(terms) + ")"
    
    if tree.op == "pow":
        base = tree_to_latex(tree.children[0])
        exp = tree_to_latex(tree.children[1])
        return f"{base}^{{{exp}}}"
    
    # Handle functions
    args = [tree_to_latex(child) for child in tree.children]
    return f"{tree.op}({', '.join(args)})"

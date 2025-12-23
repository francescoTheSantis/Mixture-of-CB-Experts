"""
Expression Utilities Module

This module provides utility functions for generating symbolic expressions
that represent different model architectures and complexity bounds.
"""

import sympy as sp
from typing import Union, List

import os
import pickle
import dill

# Import complexity module if available
try:
    from src.utils.complexity import complexity_report
except ModuleNotFoundError:
    from complexity import complexity_report

# Import tree utilities for visualization
try:
    from src.utils.expression_tree import sympy_to_tree
    from src.utils.tree_visualizer import visualize_tree
except ModuleNotFoundError:
    from expression_tree import sympy_to_tree
    from tree_visualizer import visualize_tree

HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def linear_classifier_expression(n_vars: int, include_bias: bool = True) -> sp.Expr:
    """
    Generates the symbolic expression for a linear classifier predictor.
    
    This function creates an expression representing a linear combination
    of input variables with optional bias term, which is the form used by
    linear classifiers:
        w_0 * x_0 + w_1 * x_1 + ... + w_n * x_n + b  (if include_bias=True)
        w_0 * x_0 + w_1 * x_1 + ... + w_n * x_n      (if include_bias=False)
    
    Args:
        n_vars (int): Number of input variables.
        include_bias (bool): Whether to include the bias term. Default is True.
        
    Returns:
        sympy.Expr: Symbolic expression representing the linear classifier.
        
    Examples:
        >>> expr = linear_classifier_expression(3)
        >>> print(expr)
        b + w_0*x_0 + w_1*x_1 + w_2*x_2
        
        >>> expr_no_bias = linear_classifier_expression(3, include_bias=False)
        >>> print(expr_no_bias)
        w_0*x_0 + w_1*x_1 + w_2*x_2
        
    Raises:
        ValueError: If n_vars is not a positive integer.
    """
    if not isinstance(n_vars, int) or n_vars <= 0:
        raise ValueError("n_vars must be a positive integer")
    
    # Create weight symbols
    weights = sp.symbols(f'w_0:{n_vars}')
    
    # Create input variable symbols
    variables = sp.symbols(f'x_0:{n_vars}')
    
    # Build the linear expression: sum of weighted inputs
    expression = sum(w * x for w, x in zip(weights, variables))
    
    # Add bias term if requested
    if include_bias:
        bias = sp.symbols('b')
        expression = expression + bias
    
    return expression


def boolean_and_expression(n_vars: int) -> sp.Expr:
    """
    Generates the maximum complexity boolean AND expression with negations.
    
    This function creates a boolean expression representing the most complex
    form that can be instantiated by DCR and CMR models. The expression is
    an AND of all input variables, where each variable is negated to maximize
    the number of nodes in the expression tree:
        (NOT x_0) AND (NOT x_1) AND ... AND (NOT x_n)
    
    This represents the upper bound on complexity for boolean reasoning models
    that use conjunctive formulas with optional negations.
    
    Args:
        n_vars (int): Number of input variables.
        
    Returns:
        sympy.Expr: Symbolic boolean expression with AND and NOT operations.
        
    Examples:
        >>> expr = boolean_and_expression(3)
        >>> print(expr)
        ~x_0 & ~x_1 & ~x_2
        
    Raises:
        ValueError: If n_vars is not a positive integer.
    """
    if not isinstance(n_vars, int) or n_vars <= 0:
        raise ValueError("n_vars must be a positive integer")
    
    # Create input variable symbols
    variables = sp.symbols(f'x_0:{n_vars}')
    
    # Build the boolean AND expression with all variables negated
    # Start with the negation of the first variable
    expression = ~variables[0]
    
    # AND with the negation of each subsequent variable
    for var in variables[1:]:
        expression = expression & ~var
    
    return expression


def chain_expression(n: int) -> sp.Expr:
    """
    Generates a chain expression with n-1 edges: n_1 -> f_1 -> f_2 -> ... -> f_{N-1}.
    
    This function creates a chain starting with a single node n_1, followed by
    N-1 function symbols representing edges. The chain represents a sequential 
    computation: f_{N-1}(...f_2(f_1(n_1))...)
    
    Args:
        n (int): Number defining the chain length. Must be at least 1.
                The chain will have n-1 edges/functions.
    
    Returns:
        sympy.Expr: Symbolic expression representing the chain.
                    - For n=1: returns n_1
                    - For n=2: returns f_1(n_1)
                    - For n=3: returns f_2(f_1(n_1))
                    - For n=4: returns f_3(f_2(f_1(n_1)))
                    - For n=N: returns f_{N-1}(...f_2(f_1(n_1))...)
        
    Examples:
        >>> expr = chain_expression(1)
        >>> print(expr)
        n_1
        
        >>> expr = chain_expression(2)
        >>> print(expr)
        f_1(n_1)
        
        >>> expr = chain_expression(3)
        >>> print(expr)
        f_2(f_1(n_1))
        
        >>> expr = chain_expression(4)
        >>> print(expr)
        f_3(f_2(f_1(n_1)))
        
    Raises:
        ValueError: If n is not a positive integer.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    
    # Start with node n_1
    n_1 = sp.Symbol('n_1')
    
    # Handle case with no edges (n=1)
    if n == 1:
        return n_1
    
    # Build the chain: f_{N-1}(...f_2(f_1(n_1))...)
    expression = n_1
    for i in range(1, n):
        f_i = sp.Function(f'f_{i}')
        expression = f_i(expression)
    
    return expression


def kan_expression(w: List[List[int]], nonlinearity: Union[str, None] = None) -> sp.Expr:
    """
    Generates the symbolic expression for a Kolmogorov-Arnold Network (KAN).
    
    This function creates a nested expression representing a KAN with the given
    layer structure. Each edge in the network has four parameters (a, b, c, d)
    and applies the transformation: φ(x) = a * g(b*x + c) + d, where g is a
    non-linear function.
    
    The network supports two types of aggregation neurons:
    - Summation neurons: aggregate inputs via addition (Σ)
    - Multiplication neurons: aggregate inputs via multiplication (Π)
    
    Args:
        w (List[List[int]]): List of layer specifications [[n_sum_0, n_mult_0], [n_sum_1, n_mult_1], ...]
                            where each sublist [n_sum, n_mult] specifies:
                            - n_sum: number of summation neurons in that layer
                            - n_mult: number of multiplication neurons in that layer
                            The first layer specifies input dimension as [n_inputs, 0].
        nonlinearity (Union[str, None]): Name of the nonlinearity function to use.
                           If None (default), leaves the nonlinearity as an undefined
                           symbolic function g_{l,j,i}(...). Supported concrete values
                           include 'exp', 'sin', 'cos', 'tanh', etc.
    
    Returns:
        sympy.Expr: Symbolic expression representing the full KAN computation.
                    For scalar output (total neurons in last layer == 1), returns a single expression.
                    For vector output, returns a list-like expression.
    
    Examples:
        >>> # KAN with structure: 2 inputs -> [1 sum, 1 mult] -> 1 output (sum)
        >>> expr = kan_expression([[2, 0], [1, 1], [1, 0]])
        >>> # Layer 0->1: First neuron sums, second neuron multiplies
        >>> # Layer 1->2: Output neuron sums the two previous neurons
        
        >>> # KAN with specific nonlinearity
        >>> expr_sin = kan_expression([[2, 0], [1, 0]], nonlinearity='sin')
        >>> # Result: sum of transformed inputs using sin
    
    Raises:
        ValueError: If w has fewer than 2 layers, contains non-positive integers,
                   if the first layer has multiplication neurons,
                   or if the nonlinearity is not supported.
    
    Notes:
        - The first layer must be of the form [n_inputs, 0] (inputs cannot be multiplication neurons)
        - For summation neurons: x_{l+1,j} = Σ_i [a_{l,j,i} * g(b_{l,j,i} * x_{l,i} + c_{l,j,i}) + d_{l,j,i}]
        - For multiplication neurons: x_{l+1,j} = Π_i [a_{l,j,i} * g(b_{l,j,i} * x_{l,i} + c_{l,j,i}) + d_{l,j,i}]
        - The expression grows exponentially with depth and multiplicatively
    """
    if not isinstance(w, (list, tuple)) or len(w) < 2:
        raise ValueError("w must be a list or tuple with at least 2 elements")
    
    if any(not isinstance(layer, (list, tuple)) or len(layer) != 2 for layer in w):
        raise ValueError("Each element in w must be a list/tuple of exactly 2 integers [n_sum, n_mult]")
    
    if any(not isinstance(n, int) or n < 0 for layer in w for n in layer):
        raise ValueError("All elements in layer specifications must be non-negative integers")
    
    if any(sum(layer) == 0 for layer in w):
        raise ValueError("Each layer must have at least one neuron (n_sum + n_mult > 0)")
    
    if w[0][1] != 0:
        raise ValueError("First layer (input) must have n_mult = 0, i.e., be of the form [n_inputs, 0]")
    
    # Validate nonlinearity if specified
    if nonlinearity is not None:
        valid_nonlinearities = ['exp', 'sin', 'cos', 'tan', 'tanh', 'sinh', 'cosh',
                               'log', 'sqrt', 'abs', 'sign']
        if nonlinearity not in valid_nonlinearities:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}. "
                            f"Supported values are: {valid_nonlinearities}")
    
    # Number of layers (excluding input)
    L = len(w) - 1
    
    # Create input symbols x_0, x_1, ..., x_{n_0-1}
    n_inputs = w[0][0]
    input_vars = sp.symbols(f'x_0:{n_inputs}')
    
    # Initialize the current layer activations with the input
    current_layer = list(input_vars)
    
    # Process each layer
    for layer_idx in range(L):
        n_in = sum(w[layer_idx])      # total number of neurons in current layer
        n_sum = w[layer_idx + 1][0]   # number of summation neurons in next layer
        n_mult = w[layer_idx + 1][1]  # number of multiplication neurons in next layer
        n_out = n_sum + n_mult        # total neurons in next layer
        next_layer = []
        
        # Compute summation neurons (indices 0 to n_sum-1)
        for j in range(n_sum):
            neuron_sum = 0
            
            # Sum over all input neurons
            for i in range(n_in):
                # Create parameter symbols for this edge
                a = sp.Symbol(f'a_{layer_idx}_{j}_{i}')
                b = sp.Symbol(f'b_{layer_idx}_{j}_{i}')
                c = sp.Symbol(f'c_{layer_idx}_{j}_{i}')
                d = sp.Symbol(f'd_{layer_idx}_{j}_{i}')
                
                # Create the argument for the nonlinearity
                arg = b * current_layer[i] + c
                
                # Apply nonlinearity
                if nonlinearity is None:
                    # Leave as undefined symbolic function
                    g = sp.Function(f'g_{layer_idx}_{j}_{i}')
                    nonlin_expr = g(arg)
                else:
                    # Use specific nonlinearity
                    nonlin_expr = _apply_nonlinearity(nonlinearity, arg)
                
                # Build the edge expression: a * g(b * x_i + c) + d
                edge_expr = a * nonlin_expr + d
                
                # Add to the sum for this neuron
                neuron_sum += edge_expr
            
            next_layer.append(neuron_sum)
        
        # Compute multiplication neurons (indices n_sum to n_sum+n_mult-1)
        for j in range(n_sum, n_out):
            neuron_prod = 1
            
            # Multiply over all input neurons
            for i in range(n_in):
                # Create parameter symbols for this edge
                a = sp.Symbol(f'a_{layer_idx}_{j}_{i}')
                b = sp.Symbol(f'b_{layer_idx}_{j}_{i}')
                c = sp.Symbol(f'c_{layer_idx}_{j}_{i}')
                d = sp.Symbol(f'd_{layer_idx}_{j}_{i}')
                
                # Create the argument for the nonlinearity
                arg = b * current_layer[i] + c
                
                # Apply nonlinearity
                if nonlinearity is None:
                    # Leave as undefined symbolic function
                    g = sp.Function(f'g_{layer_idx}_{j}_{i}')
                    nonlin_expr = g(arg)
                else:
                    # Use specific nonlinearity
                    nonlin_expr = _apply_nonlinearity(nonlinearity, arg)
                
                # Build the edge expression: a * g(b * x_i + c) + d
                edge_expr = a * nonlin_expr + d
                
                # Multiply for this neuron
                neuron_prod *= edge_expr
            
            next_layer.append(neuron_prod)
        
        # Move to the next layer
        current_layer = next_layer
    
    # Return the final output
    # If output is scalar, return the single expression
    if sum(w[-1]) == 1:
        return current_layer[0]
    else:
        # For vector output, return as a list
        return current_layer


def _apply_nonlinearity(nonlinearity: str, arg: sp.Expr) -> sp.Expr:
    """Helper function to apply a specific nonlinearity to an argument."""
    if nonlinearity == 'exp':
        return sp.exp(arg)
    elif nonlinearity == 'sin':
        return sp.sin(arg)
    elif nonlinearity == 'cos':
        return sp.cos(arg)
    elif nonlinearity == 'tan':
        return sp.tan(arg)
    elif nonlinearity == 'tanh':
        return sp.tanh(arg)
    elif nonlinearity == 'sinh':
        return sp.sinh(arg)
    elif nonlinearity == 'cosh':
        return sp.cosh(arg)
    elif nonlinearity == 'log':
        return sp.log(arg)
    elif nonlinearity == 'sqrt':
        return sp.sqrt(arg)
    elif nonlinearity == 'abs':
        return sp.Abs(arg)
    elif nonlinearity == 'sign':
        return sp.sign(arg)

def store_eq(equation: sp.Expr, log_dir: Union[str, None], idx: int = None) -> None:
    """
    Stores the given equations as string representation to the specified log directory.
    Uses SymPy's srepr() for serialization to avoid recursion issues with complex expressions.
    To load: equation = sp.sympify(open(filename).read())
    """

    if log_dir is None:
        return

    os.makedirs(log_dir, exist_ok=True)
    if idx is None:
        filename = os.path.join(log_dir, "equation.txt")
    else:
        filename = os.path.join(log_dir, f"equation_{idx}.txt")
    
    # Store as string representation using srepr for full fidelity
    with open(filename, "w") as f:
        f.write(sp.srepr(equation))


if __name__ == "__main__":
    import os
    
    # Create output directory for visualizations
    output_dir = "results/figs/expression_graphs"
    output_dir = os.path.join(HOME, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Example usage and testing
    print("=" * 70)
    print("LINEAR CLASSIFIER EXPRESSIONS")
    print("=" * 70)
    
    print("\nLinear Classifier Expression (3 variables, with bias):")
    linear_expr = linear_classifier_expression(3)
    print(f"  Expression: {linear_expr}")
    report = complexity_report(linear_expr)
    print(f"  Complexity: {report}")
    
    # Visualize linear classifier
    print("  Generating graph visualization...")
    linear_tree = sympy_to_tree(linear_expr)
    visualize_tree(linear_tree, 
                   title="Linear Classifier (3 variables)", 
                   save_path=f"{output_dir}/linear_classifier_3vars.pdf")
    print()
    
    print("=" * 70)
    print("BOOLEAN EXPRESSIONS")
    print("=" * 70)
    
    print("\nBoolean AND Expression with Negations (3 variables):")
    bool_expr = boolean_and_expression(3)
    print(f"  Expression: {bool_expr}")
    report = complexity_report(bool_expr)
    print(f"  Complexity: {report}")
    
    # Visualize boolean expression
    print("  Generating graph visualization...")
    bool_tree = sympy_to_tree(bool_expr)
    visualize_tree(bool_tree, 
                   title="Boolean AND with Negations (3 variables)", 
                   save_path=f"{output_dir}/boolean_and_3vars.pdf")
    print()
    
    print("=" * 70)
    print("CHAIN EXPRESSIONS")
    print("=" * 70)
    
    print("Chain Expression (3 nodes):")
    chain_expr_3 = chain_expression(3)
    print(f"  Expression: {chain_expr_3}")
    report_3 = complexity_report(chain_expr_3)
    print(f"  Complexity: {report_3}")
    
    # Visualize smaller chain
    print("  Generating graph visualization...")
    chain_tree_3 = sympy_to_tree(chain_expr_3)
    visualize_tree(chain_tree_3, 
                   title="Chain Expression (3 nodes)", 
                   save_path=f"{output_dir}/chain_3nodes.pdf")
    print()

    print("=" * 70)
    print("KAN EXPRESSIONS")
    print("=" * 70)
    
    print("\nKAN with undefined nonlinearity: [[2, 0], [1, 0]] (2 inputs -> 1 sum neuron)")
    kan_expr_undef = kan_expression([[2, 0], [1, 0]])
    print(f"  Expression: {kan_expr_undef}")
    report = complexity_report(kan_expr_undef)
    print(f"  Complexity: {report}")
    print(f"  Parameters: 2 edges * 4 params/edge = 8")
    
    # Visualize KAN expression
    print("  Generating graph visualization...")
    kan_tree = sympy_to_tree(kan_expr_undef)
    visualize_tree(kan_tree, 
                   title="KAN [[2,0], [1,0]] with Undefined Nonlinearity", 
                   save_path=f"{output_dir}/kan_2_1_undefined.pdf")
    print()
    
    # Example with mixed aggregation: [[2, 0], [1, 1], [1, 0]]
    print("=" * 70)
    print("KAN WITH MIXED AGGREGATION: [[2, 0], [1, 1], [1, 0]]")
    print("=" * 70)
    print()
    print("Structure breakdown:")
    print("  Layer 0 (Input): x_0, x_1")
    print()
    print("  Layer 1 (Hidden):")
    print("    - Neuron 0 (SUM):  aggregates inputs using ADDITION")
    print("    - Neuron 1 (MULT): aggregates inputs using MULTIPLICATION")
    print()
    print("  Layer 2 (Output):")
    print("    - Neuron 0 (SUM):  aggregates layer 1 outputs using ADDITION")
    print()
    
    kan_expr_mixed = kan_expression([[2, 0], [1, 1], [1, 0]])
    print("Full Expression:")
    print(f"  {kan_expr_mixed}")
    print()
    
    print("Breaking it down:")
    print()
    print("  Layer 0 -> Layer 1:")
    print("    Hidden Neuron 0 (SUM):")
    print("      = [a_0_0_0 * g_0_0_0(b_0_0_0*x_0 + c_0_0_0) + d_0_0_0]")
    print("      + [a_0_0_1 * g_0_0_1(b_0_0_1*x_1 + c_0_0_1) + d_0_0_1]")
    print()
    print("    Hidden Neuron 1 (MULT):")
    print("      = [a_0_1_0 * g_0_1_0(b_0_1_0*x_0 + c_0_1_0) + d_0_1_0]")
    print("      * [a_0_1_1 * g_0_1_1(b_0_1_1*x_1 + c_0_1_1) + d_0_1_1]")
    print()
    print("  Layer 1 -> Layer 2:")
    print("    Output Neuron 0 (SUM):")
    print("      = [a_1_0_0 * g_1_0_0(b_1_0_0*h_0 + c_1_0_0) + d_1_0_0]")
    print("      + [a_1_0_1 * g_1_0_1(b_1_0_1*h_1 + c_1_0_1) + d_1_0_1]")
    print()
    print("    where h_0 = Hidden Neuron 0 (sum), h_1 = Hidden Neuron 1 (mult)")
    print()
    
    report_mixed = complexity_report(kan_expr_mixed)
    print(f"Complexity: {report_mixed}")
    print()
    print("Key Difference:")
    print("  - SUM neurons use + between edge outputs")
    print("  - MULT neurons use * between edge outputs")
    
    # Visualize mixed KAN expression
    print()
    print("Generating graph visualization...")
    kan_tree_mixed = sympy_to_tree(kan_expr_mixed)
    visualize_tree(kan_tree_mixed, 
                   title="KAN [[2,0], [1,1], [1,0]] Mixed Aggregation", 
                   save_path=f"{output_dir}/kan_mixed_aggregation.pdf")
    print()
    
    print("=" * 70)
    print("COMPLEXITY COMPARISON SUMMARY")
    print("=" * 70)
    print("\nComplexity scales with network size:")
    print("  Linear (3 vars):         node_count =", complexity_report(linear_classifier_expression(3))['node_count'])
    print("  Boolean AND (3 vars):    node_count =", complexity_report(boolean_and_expression(3))['node_count'])
    print("  Chain (3 nodes):         node_count =", complexity_report(chain_expression(3))['node_count'])
    print("  KAN 1 layer:             node_count =", complexity_report(kan_expression([[2, 0], [1, 0]]))['node_count'])
    print("  KAN mixed aggregation:   node_count =", complexity_report(kan_expression([[2, 0], [1, 1], [1, 0]]))['node_count'])
    print()
    
    print("=" * 70)
    print("COMPLEXITY COMPARISON SUMMARY")
    print("=" * 70)
    print("\nComplexity scales with network size:")
    print("  Linear (3 vars):         visitation_length =", complexity_report(linear_classifier_expression(3))['visitation_length'])
    print("  Boolean AND (3 vars):    visitation_length =", complexity_report(boolean_and_expression(3))['visitation_length'])
    print("  Chain (3 nodes):         visitation_length =", complexity_report(chain_expression(3))['visitation_length'])
    print("  KAN 1 layer:             visitation_length =", complexity_report(kan_expression([[2, 0], [1, 0]]))['visitation_length'])
    print("  KAN mixed aggregation:   visitation_length =", complexity_report(kan_expression([[2, 0], [1, 1], [1, 0]]))['visitation_length'])
    print()

    print("=" * 70)
    print(f"✅ All visualizations saved to: {output_dir}/")
    print("=" * 70)

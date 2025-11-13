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


def kan_expression(w: List[int], nonlinearity: Union[str, None] = None) -> sp.Expr:
    """
    Generates the symbolic expression for a Kolmogorov-Arnold Network (KAN).
    
    This function creates a nested expression representing a KAN with the given
    layer structure. Each edge in the network has four parameters (a, b, c, d)
    and applies the transformation: φ(x) = a * g(b*x + c) + d, where g is a
    non-linear function.
    
    The network computes:
        x_{l+1,j} = Σ_i [a_{l,j,i} * g_{l,j,i}(b_{l,j,i} * x_{l,i} + c_{l,j,i}) + d_{l,j,i}]
    
    for each layer l and output neuron j, where the sum is over all input neurons i.
    
    Args:
        w (List[int]): List of layer sizes [n_0, n_1, ..., n_L] where n_0 is the
                       input dimension and n_L is the output dimension (typically 1
                       for scalar output).
        nonlinearity (Union[str, None]): Name of the nonlinearity function to use.
                           If None (default), leaves the nonlinearity as an undefined
                           symbolic function g_{l,j,i}(...). Supported concrete values
                           include 'exp', 'sin', 'cos', 'tanh', etc.
    
    Returns:
        sympy.Expr: Symbolic expression representing the full KAN computation.
                    For scalar output (w[-1] == 1), returns a single expression.
                    For vector output, returns a list-like expression.
    
    Examples:
        >>> # KAN with undefined nonlinearity (most general form)
        >>> expr = kan_expression([2, 1])
        >>> # Result: a_0_0_0*g_0_0_0(b_0_0_0*x_0 + c_0_0_0) + ...
        
        >>> # KAN with specific nonlinearity
        >>> expr_sin = kan_expression([2, 1], nonlinearity='sin')
        >>> # Result: a_0_0_0*sin(b_0_0_0*x_0 + c_0_0_0) + ...
    
    Raises:
        ValueError: If w has fewer than 2 layers, contains non-positive integers,
                   or if the nonlinearity is not supported.
    
    Notes:
        - For a network with layers w = [n_0, n_1, ..., n_L], the total number
          of parameters is 4 * Σ_{l=0}^{L-1} (n_l * n_{l+1}).
        - The expression grows exponentially with depth, so it may become very
          large for deep networks.
        - Each edge has its own nonlinearity symbol g_{l,j,i}, allowing for
          heterogeneous activations in the general case.
    """
    if not isinstance(w, (list, tuple)) or len(w) < 2:
        raise ValueError("w must be a list or tuple with at least 2 elements")
    
    if any(not isinstance(n, int) or n <= 0 for n in w):
        raise ValueError("All elements in w must be positive integers")
    
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
    input_vars = sp.symbols(f'x_0:{w[0]}')
    
    # Initialize the current layer activations with the input
    current_layer = list(input_vars)
    
    # Process each layer
    for layer_idx in range(L):
        n_in = w[layer_idx]      # number of neurons in current layer
        n_out = w[layer_idx + 1]  # number of neurons in next layer
        next_layer = []
        
        # Compute each output neuron
        for j in range(n_out):
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
                    if nonlinearity == 'exp':
                        nonlin_expr = sp.exp(arg)
                    elif nonlinearity == 'sin':
                        nonlin_expr = sp.sin(arg)
                    elif nonlinearity == 'cos':
                        nonlin_expr = sp.cos(arg)
                    elif nonlinearity == 'tan':
                        nonlin_expr = sp.tan(arg)
                    elif nonlinearity == 'tanh':
                        nonlin_expr = sp.tanh(arg)
                    elif nonlinearity == 'sinh':
                        nonlin_expr = sp.sinh(arg)
                    elif nonlinearity == 'cosh':
                        nonlin_expr = sp.cosh(arg)
                    elif nonlinearity == 'log':
                        nonlin_expr = sp.log(arg)
                    elif nonlinearity == 'sqrt':
                        nonlin_expr = sp.sqrt(arg)
                    elif nonlinearity == 'abs':
                        nonlin_expr = sp.Abs(arg)
                    elif nonlinearity == 'sign':
                        nonlin_expr = sp.sign(arg)
                
                # Build the edge expression: a * g(b * x_i + c) + d
                edge_expr = a * nonlin_expr + d
                
                # Add to the sum for this neuron
                neuron_sum += edge_expr
            
            next_layer.append(neuron_sum)
        
        # Move to the next layer
        current_layer = next_layer
    
    # Return the final output
    # If output is scalar, return the single expression
    if w[-1] == 1:
        return current_layer[0]
    else:
        # For vector output, return as a list
        return current_layer

def store_eq(equation: sp.Expr, log_dir: Union[str, None], idx: int = None) -> None:
    """
    Stores the given equations in pickle format to the specified log directory.
    Uses dill for serialization to handle SymPy's dynamically created Function objects.
    """

    if log_dir is None:
        return

    os.makedirs(log_dir, exist_ok=True)
    if idx is None:
        filename = os.path.join(log_dir, "equation.pkl")
    else:
        filename = os.path.join(log_dir, f"equation_{idx}.pkl")
    with open(filename, "wb") as f:
        dill.dump(equation, f)


if __name__ == "__main__":
    import os
    
    # Create output directory for visualizations
    output_dir = "figs/expression_graphs"
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
    
    print("\nKAN with undefined nonlinearity: [2, 1]")
    kan_expr_undef = kan_expression([2, 1])
    print(f"  Expression: {kan_expr_undef}")
    report = complexity_report(kan_expr_undef)
    print(f"  Complexity: {report}")
    print(f"  Parameters: 2 edges * 4 params/edge = 8")
    
    # Visualize KAN expression
    print("  Generating graph visualization...")
    kan_tree = sympy_to_tree(kan_expr_undef)
    visualize_tree(kan_tree, 
                   title="KAN [2, 1] with Undefined Nonlinearity", 
                   save_path=f"{output_dir}/kan_2_1_undefined.pdf")
    print()
    
    # Additional KAN example with specific nonlinearity
    print("\nKAN with sin nonlinearity: [2, 1]")
    kan_expr_sin = kan_expression([2, 1], nonlinearity='sin')
    print(f"  Expression (first 100 chars): {str(kan_expr_sin)[:100]}...")
    report_sin = complexity_report(kan_expr_sin)
    print(f"  Complexity: {report_sin}")

    print()
    
    print("=" * 70)
    print("COMPLEXITY COMPARISON SUMMARY")
    print("=" * 70)
    print("\nComplexity scales with network size:")
    print("  Linear (3 vars):         node_count =", complexity_report(linear_classifier_expression(3))['node_count'])
    print("  Boolean AND (3 vars):    node_count =", complexity_report(boolean_and_expression(3))['node_count'])
    print("  Chain (3 nodes):         node_count =", complexity_report(chain_expression(3))['node_count'])
    print("  KAN 1 layer:               node_count =", complexity_report(kan_expression([2, 1]))['node_count'])
    print()
    
    print("=" * 70)
    print("COMPLEXITY COMPARISON SUMMARY")
    print("=" * 70)
    print("\nComplexity scales with network size:")
    print("  Linear (3 vars):         visitation_length =", complexity_report(linear_classifier_expression(3))['visitation_length'])
    print("  Boolean AND (3 vars):    visitation_length =", complexity_report(boolean_and_expression(3))['visitation_length'])
    print("  Chain (3 nodes):         visitation_length =", complexity_report(chain_expression(3))['visitation_length'])
    print("  KAN 1 layer:               visitation_length =", complexity_report(kan_expression([2, 1]))['visitation_length'])
    print()

    print("=" * 70)
    print(f"✅ All visualizations saved to: {output_dir}/")
    print("=" * 70)

import ast
from graphviz import Digraph

# Expression
expr = "x**2 + 3*x + 2"

# Parse expression into AST
tree = ast.parse(expr, mode='eval')

# Create a directed graph
dot = Digraph()
dot.attr('node', shape='circle')

# Recursive function to traverse the AST
def add_nodes_edges(node, parent=None):
    node_id = str(id(node))
    label = type(node).__name__

    # Custom labels for specific node types
    if isinstance(node, ast.BinOp):
        label = type(node.op).__name__
    elif isinstance(node, ast.Constant):
        label = str(node.value)
    elif isinstance(node, ast.Name):
        label = node.id

    dot.node(node_id, label)

    if parent:
        dot.edge(str(id(parent)), node_id)

    # Recursively process children
    for child_name, child in ast.iter_fields(node):
        if isinstance(child, ast.AST):
            add_nodes_edges(child, node)
        elif isinstance(child, list):
            for n in child:
                if isinstance(n, ast.AST):
                    add_nodes_edges(n, node)

# Build tree from root
add_nodes_edges(tree.body)

# Save or render the tree
dot.render("expression_tree", format="pdf", cleanup=True)

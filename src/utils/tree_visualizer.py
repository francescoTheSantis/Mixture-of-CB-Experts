"""
Expression Tree Visualizer

This module provides utilities for visualizing expression trees as graph figures.
Uses NetworkX and matplotlib for clean hierarchical tree visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

try:
    from src.utils.expression_tree import ExpressionTreeNode
except ModuleNotFoundError:
    from expression_tree import ExpressionTreeNode


class TreeVisualizer:
    """
    Visualizes expression trees as hierarchical graphs using NetworkX.
    """
    
    def __init__(self, figsize=(12, 8), dpi=100):
        """
        Initialize the visualizer.
        
        Args:
            figsize (tuple): Figure size (width, height)
            dpi (int): Figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Color scheme
        self.colors = {
            'symbol': '#8ecae6',      # Light blue for symbols
            'constant': '#ffb703',    # Orange for constants
            'add': '#06d6a0',         # Green for addition
            'mul': '#ef476f',         # Pink for multiplication
            'function': '#9d4edd',    # Purple for functions (ReLU, etc.)
            'matrix': '#118ab2',      # Dark blue for matrices
            'default': '#f4a261'      # Default tan color
        }
    
    def _get_node_color(self, node):
        """Get color for a node based on its operation type."""
        op_lower = node.op.lower()
        
        if op_lower == 'symbol':
            return self.colors['symbol']
        elif op_lower == 'constant':
            return self.colors['constant']
        elif op_lower in ['add', 'matadd']:
            return self.colors['add']
        elif op_lower in ['mul', 'matmul']:
            return self.colors['mul']
        elif op_lower == 'matrix':
            return self.colors['matrix']
        elif op_lower in ['relu', 'sigmoid', 'tanh', 'leakyrelu', 'elu', 'gelu', 'softmax']:
            return self.colors['function']
        else:
            return self.colors['default']
    
    def _get_node_label(self, node):
        """Get display label for a node."""
        if node.value is not None and node.op in ['symbol', 'constant', 'matrix']:
            return str(node.value)
        return node.op
    
    def _build_graph(self, tree):
        """
        Build a NetworkX graph from the expression tree.
        
        Args:
            tree: Root node of the expression tree
            
        Returns:
            tuple: (NetworkX Graph, dict of node attributes)
        """
        G = nx.Graph()
        node_attrs = {}
        node_counter = [0]  # Use list to make it mutable in nested function
        input_counter = [0]  # Counter for input variables
        output_counter = [0]  # Counter for output nodes
        
        def _add_nodes_edges(node, parent_id=None, is_root=False):
            """Recursively add nodes and edges to the graph."""
            # Create unique node ID
            node_id = node_counter[0]
            node_counter[0] += 1
            
            # Get node label and color
            label = self._get_node_label(node)
            
            # Rename input variables to x_1, x_2, etc.
            if node.op == 'symbol' and node.value == 'x':
                input_counter[0] += 1
                label = f'x_{input_counter[0]}'
            
            # Rename output if it's the root
            if is_root:
                output_counter[0] += 1
                # If root is a simple variable, rename it as output
                if node.op in ['symbol', 'matrix'] and not node.children:
                    label = f'y_{output_counter[0]}'
            
            color = self._get_node_color(node)
            
            # Add node to graph
            G.add_node(node_id, label=label, color=color)
            node_attrs[node_id] = {'label': label, 'color': color}
            
            # Add edge from child to parent (reversed direction)
            if parent_id is not None:
                G.add_edge(node_id, parent_id)
            
            # Recursively add children
            for child in node.children:
                _add_nodes_edges(child, node_id, is_root=False)
            
            return node_id
        
        _add_nodes_edges(tree, is_root=True)
        return G, node_attrs
    
    def visualize(self, tree, title="Expression Tree", save_path=None, show=False):
        """
        Create a visualization of the expression tree using NetworkX.
        
        Args:
            tree (ExpressionTreeNode): Root node of the tree
            title (str): Title for the figure
            save_path (str): Path to save the figure (optional)
            show (bool): Whether to display the figure (default: False)
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Build NetworkX graph
        G, node_attrs = self._build_graph(tree)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Try hierarchical layout first (requires pygraphviz)
        try:
            pos = graphviz_layout(G, prog='dot')
        except:
            # Fallback to hierarchical_layout if pygraphviz not available
            try:
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            except:
                # Final fallback to spring layout with custom positioning
                pos = self._hierarchical_layout(G)
        
        # Extract node colors and labels
        node_colors = [node_attrs[node]['color'] for node in G.nodes()]
        node_labels = {node: node_attrs[node]['label'] for node in G.nodes()}
        
        # Draw the graph
        nx.draw_networkx_edges(G, pos, ax=ax, 
                              edge_color='#2c3e50',
                              width=2.5,
                              alpha=0.7)
        
        # Draw nodes as grey hollow circles
        nx.draw_networkx_nodes(G, pos, ax=ax,
                              node_color='white',
                              node_size=3000,
                              node_shape='o',
                              edgecolors='#808080',
                              linewidths=2.5,
                              alpha=1.0)
        
        nx.draw_networkx_labels(G, pos, node_labels, ax=ax,
                               font_size=11,
                               font_weight='bold',
                               font_color='#000000')
        
        # Remove title and legend
        ax.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            # Determine format from extension, default to PDF
            if not save_path.endswith(('.png', '.pdf', '.svg', '.jpg', '.jpeg')):
                save_path = save_path + '.pdf'
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ Figure saved to: {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _hierarchical_layout(self, G):
        """
        Create a hierarchical layout for the tree (fallback when pygraphviz not available).
        
        Args:
            G: NetworkX Graph
            
        Returns:
            dict: Node positions
        """
        # Find root (node with only one neighbor in a tree structure)
        # In an undirected tree, we can use degree to find potential roots
        # or just pick the first node with degree 1 or the first node overall
        root = list(G.nodes())[0]
        
        # For undirected graph, find a node that could be a root (any node works for BFS)
        # We'll do BFS to assign levels
        levels = {}
        queue = [(root, 0)]
        visited = {root}
        
        while queue:
            node, level = queue.pop(0)
            levels[node] = level
            # Get neighbors (undirected)
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, level + 1))
        
        # Group nodes by level
        level_nodes = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)
        
        # Calculate positions
        pos = {}
        max_width = max(len(nodes) for nodes in level_nodes.values())
        height = len(level_nodes)
        
        for level, nodes in level_nodes.items():
            y = height - level  # Top to bottom
            width = len(nodes)
            for i, node in enumerate(nodes):
                x = (i - width / 2) * (max_width / max(width, 1))
                pos[node] = (x, y * 2)  # Scale y for better spacing
        
        return pos
    
    def _add_legend(self, ax):
        """Add a legend explaining node colors."""
        legend_elements = [
            mpatches.Patch(facecolor=self.colors['symbol'], edgecolor='#2c3e50', 
                          label='Variables (x, W, b)', linewidth=2),
            mpatches.Patch(facecolor=self.colors['constant'], edgecolor='#2c3e50',
                          label='Constants', linewidth=2),
            mpatches.Patch(facecolor=self.colors['add'], edgecolor='#2c3e50',
                          label='Addition (+)', linewidth=2),
            mpatches.Patch(facecolor=self.colors['mul'], edgecolor='#2c3e50',
                          label='Multiplication (*)', linewidth=2),
            mpatches.Patch(facecolor=self.colors['function'], edgecolor='#2c3e50',
                          label='Activation Functions', linewidth=2),
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 framealpha=0.95, fontsize=9, edgecolor='#2c3e50')


def visualize_tree(tree, title="Expression Tree", save_path=None, show=False, 
                   figsize=(12, 8), dpi=100):
    """
    Convenience function to visualize an expression tree.
    
    Args:
        tree (ExpressionTreeNode): Root node of the tree
        title (str): Title for the figure
        save_path (str): Path to save the figure (optional, defaults to PDF if no extension)
        show (bool): Whether to display the figure (default: False)
        figsize (tuple): Figure size (width, height)
        dpi (int): Figure resolution
        
    Returns:
        matplotlib.figure.Figure: The created figure
        
    Example:
        >>> from src.models.encoders.mlp import MLPEncoder
        >>> from src.utils.expression_tree import sympy_to_tree
        >>> from src.utils.tree_visualizer import visualize_tree
        >>> 
        >>> encoder = MLPEncoder(input_size=5, output_size=2, hidden_size=4, num_layers=2)
        >>> expr = encoder.to_symbolic()
        >>> tree = sympy_to_tree(expr)
        >>> visualize_tree(tree, title="2-Layer MLP", save_path="mlp_tree.pdf")
    """
    visualizer = TreeVisualizer(figsize=figsize, dpi=dpi)
    return visualizer.visualize(tree, title=title, save_path=save_path, show=show)

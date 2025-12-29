"""
Utility functions for extracting, parsing, and cleaning equations for storage.
Used by the Engine class during test time to collect per-sample equation data.
"""

import torch


def clean_equation(equation):
    """
    Clean equation string by:
    1. Removing task name prefix (e.g., "otter: " -> "")
    2. Fixing negative coefficients (e.g., "+ -0.5" -> "- 0.5")
    3. Stripping whitespace
    
    Args:
        equation (str): The raw equation string
        
    Returns:
        str: Cleaned equation string
    """
    # Remove task name prefix if present
    if ':' in equation:
        equation = equation.split(':', 1)[1]
    
    # Fix negative coefficients: replace "+ -" with "- "
    equation = equation.replace('+ -', '- ')
    
    # Strip and return
    return equation.strip()


def parse_memory_equations(equations_per_slot, y_names):
    """
    Pre-parse equation strings into a structured format for fast lookup.
    Returns a dictionary mapping (memory_idx, output_name) -> equation_string.
    This avoids repeated string parsing inside the per-sample loop.
    
    Args:
        equations_per_slot (dict): Dictionary mapping memory_slot_idx -> equation_string
        y_names (list): List of output names
        
    Returns:
        dict: Dictionary mapping (memory_idx, output_name) -> equation_string
    """
    parsed = {}
    
    for mem_idx, slot_equations in equations_per_slot.items():
        if ';' in slot_equations:
            # Multi-output format: "y0: eq0; y1: eq1; ..."
            eqs = slot_equations.split(';')
            for eq in eqs:
                eq = eq.strip()
                if ':' in eq:
                    # Extract output name and equation
                    output_name, equation = eq.split(':', 1)
                    parsed[(mem_idx, output_name.strip())] = equation.strip()
        else:
            # Single output format
            if ':' in slot_equations:
                output_name, equation = slot_equations.split(':', 1)
                parsed[(mem_idx, output_name.strip())] = equation.strip()
            else:
                # No prefix - assume it applies to all outputs
                for y_name in y_names:
                    parsed[(mem_idx, y_name)] = slot_equations.strip()
    
    return parsed


def extract_per_sample_equations(model_output, batch_size, predictions_np, c_names, y_names, task):
    """
    Extract equation strings for each sample (used by LinearConceptEmbeddingModel).
    Returns a list of equation strings, one per sample.
    Optimized to only extract equations for predicted outputs to avoid unnecessary computation.
    Equations are cleaned (no task name prefix, fixed negative coefficients).
    
    Args:
        model_output (dict): Model output containing 'weights' and optionally 'y_bias'
        batch_size (int): Number of samples in batch
        predictions_np (np.ndarray): Predictions for the batch
        c_names (list): List of concept names
        y_names (list): List of output names
        task (str): Task type ('classification' or 'regression')
        
    Returns:
        list: List of equation strings, one per sample
    """
    equations = []
    
    # Extract weights and bias from model output
    # weights shape: (batch_size, 1, n_concepts, n_outputs)
    weights = model_output['weights'].detach().cpu().numpy()
    
    # y_bias shape: (batch_size, 1, n_outputs) if present, else None
    y_bias = model_output.get('y_bias', None)
    if y_bias is not None:
        y_bias = y_bias.detach().cpu().numpy()
    
    n_outputs = weights.shape[-1]
    n_concepts = weights.shape[-2]
    
    # Build equation for each sample
    for sample_idx in range(batch_size):
        # Single-output or classification: only extract equation for predicted class
        prediction = predictions_np[sample_idx]
        if task == 'classification':
            # For classification, prediction is the predicted class index
            pred_class_idx = int(prediction) if prediction.ndim == 0 else int(prediction[0])
        else:
            # For single-output regression, use index 0
            pred_class_idx = 0
        
        # Only build equation for predicted output
        out_idx = pred_class_idx
        terms = []
        
        # Add weighted concept terms
        for c_idx in range(n_concepts):
            weight = weights[sample_idx, 0, c_idx, out_idx]
            if abs(weight) > 1e-6:  # Only include non-zero terms
                c_name = c_names[c_idx]
                # Use proper sign formatting
                if weight >= 0:
                    terms.append(f"{weight:.4f}*{c_name}")
                else:
                    terms.append(f"{weight:.4f}*{c_name}")  # Negative already included
        
        # Add bias if present
        if y_bias is not None:
            bias_value = y_bias[sample_idx, 0, out_idx]
            if abs(bias_value) > 1e-6:
                terms.append(f"{bias_value:.4f}")
        
        # Build equation string without task name prefix
        # Join terms with proper signs
        if terms:
            eq_str = terms[0]
            for term in terms[1:]:
                if term.startswith('-'):
                    eq_str += f" - {term[1:]}"
                else:
                    eq_str += f" + {term}"
        else:
            eq_str = "0"
        
        equations.append(eq_str.strip())

    return equations


def extract_memory_equations(model, model_name):
    """
    Extract equation strings for each memory slot based on model type.
    Returns a dictionary mapping memory_slot_idx -> equation_string.
    
    Args:
        model: The model instance
        model_name (str): Name of the model class
        
    Returns:
        dict: Dictionary mapping memory_slot_idx -> equation_string
    """
    equations = {}
    
    if model_name == 'KANSymbolicCBM':
        # Extract from KAN predictor or SymbolicPredictor
        if hasattr(model.predictor, 'trainable_equations'):
            # SymbolicPredictor with learned equations
            for mem_idx, set_name in enumerate(sorted(model.predictor.trainable_equations.keys())):
                eq_strs = []
                for eq_name in model.predictor.equation_names[set_name]:
                    eq_module = model.predictor.trainable_equations[set_name][eq_name]
                    eq_strs.append(f"{eq_name}: {eq_module.get_equation_string()}")
                equations[mem_idx] = "; ".join(eq_strs)
        elif hasattr(model.predictor, 'kans'):
            # KANPredictor - abstract representation
            for mem_idx in range(len(model.predictor.kans)):
                equations[mem_idx] = f"KAN{mem_idx}[{model.widths}]"
        else:
            equations[0] = "No equations available"
    
    elif model_name == 'LinearSymbolicCBM':
        # Extract linear equations from memory
        try:
            equation_weights = model.linear_memory_predictor.equation_decoder(
                model.linear_memory_predictor.equation_memory.weight
            )
            equation_weights = equation_weights.view(
                model.memory_size, 
                len(model.linear_memory_predictor.parameters), 
                len(model.y_names)
            )
            weights_np = equation_weights.detach().cpu().numpy()
            
            for mem_idx in range(model.memory_size):
                eq_strs = []
                for out_idx, y_name in enumerate(model.y_names):
                    # Build equation string
                    terms = []
                    n_concepts = len(model.c_names)
                    for c_idx in range(n_concepts):
                        weight = weights_np[mem_idx, c_idx, out_idx]
                        if abs(weight) > 1e-6:  # Only include non-zero terms
                            terms.append(f"{weight:.4f}*{model.c_names[c_idx]}")
                    
                    # Add bias
                    if model.bias == 'local':
                        bias_value = weights_np[mem_idx, -1, out_idx]
                        terms.append(f"{bias_value:.4f}")
                    elif model.bias == 'global':
                        bias_value = model.linear_memory_predictor.bias_params[out_idx].item()
                        terms.append(f"{bias_value:.4f}")
                    
                    eq_str = f"{y_name}: " + " + ".join(terms) if terms else f"{y_name}: 0"
                    eq_strs.append(eq_str)
                
                equations[mem_idx] = "; ".join(eq_strs)
        except Exception as e:
            for mem_idx in range(getattr(model, 'memory_size', 1)):
                equations[mem_idx] = f"Error extracting equation: {str(e)}"
    
    elif model_name == 'PriorSymbolicCBM':
        # Extract from prior_predictor
        if hasattr(model.prior_predictor, 'trainable_equations'):
            for mem_idx, set_name in enumerate(sorted(model.prior_predictor.trainable_equations.keys())):
                eq_strs = []
                for eq_name in model.prior_predictor.equation_names[set_name]:
                    eq_module = model.prior_predictor.trainable_equations[set_name][eq_name]
                    eq_strs.append(f"{eq_name}: {eq_module.get_equation_string()}")
                equations[mem_idx] = "; ".join(eq_strs)
        else:
            equations[0] = "No equations available"
    
    elif model_name == 'SymbolicRegressorCBM':
        # Extract from predictor
        if hasattr(model.predictor, 'trainable_equations'):
            # SymbolicPredictor with learned equations
            for mem_idx, set_name in enumerate(sorted(model.predictor.trainable_equations.keys())):
                eq_strs = []
                for eq_name in model.predictor.equation_names[set_name]:
                    eq_module = model.predictor.trainable_equations[set_name][eq_name]
                    # NOTE: get_equation_string() returns equations with current fine-tuned parameter values
                    eq_strs.append(f"{eq_name}: {eq_module.get_equation_string()}")
                equations[mem_idx] = "; ".join(eq_strs)
        else:
            # BlackBoxPredictor or not yet trained
            for mem_idx in range(getattr(model, 'memory_size', 1)):
                equations[mem_idx] = "No symbolic equations (BlackBoxPredictor)"
    elif model_name == 'MemoryCBM':
        # Get the symbolic equivalent of each blackbox predictor
        for mem_idx in range(model.memory_size):
            eq_result = model.get_symbolic_equivalent(memory_idx=mem_idx, return_equations=True)
            # For multi-output, eq_result is a list of equations
            try:
                # Try to iterate - if it's a list/tuple, this will work
                eq_strs = [f"{model.y_names[i]}: {str(eq)}" for i, eq in enumerate(eq_result)]
                equations[mem_idx] = "; ".join(eq_strs)
            except (TypeError, AttributeError):
                # Single output - not iterable
                y_name = model.y_names[0] if isinstance(model.y_names, list) else model.y_names
                equations[mem_idx] = f"{y_name}: {str(eq_result)}"
    
    elif model_name in ['BlackBox', 'ConceptEmbeddingModel']:
        # These should use cached equations set in on_test_start
        # This is a fallback for when caching is not available
        try:
            eq_result = model.get_symbolic_equivalent(return_equations=True)
            # For multi-output, eq_result is a list of equations
            try:
                # Try to iterate - if it's a list/tuple, this will work
                eq_strs = [f"{model.y_names[i]}: {str(eq)}" for i, eq in enumerate(eq_result)]
                equations[0] = "; ".join(eq_strs)
            except (TypeError, AttributeError):
                # Single output - not iterable
                y_name = model.y_names[0] if isinstance(model.y_names, list) else model.y_names
                equations[0] = f"{y_name}: {str(eq_result)}"
        except Exception as e:
            equations[0] = f"Error extracting equation: {str(e)}"
    
    else:
        # Other models - no memory-based equations
        equations[0] = f"Model {model_name} does not use memory-based equations"
    
    return equations

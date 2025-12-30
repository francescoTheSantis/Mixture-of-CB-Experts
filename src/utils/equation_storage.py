"""
Utility functions for extracting, parsing, and cleaning equations for storage.
Used by the Engine class during test time to collect per-sample equation data.
"""

import torch
from tqdm import tqdm


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
    parsed = dict()
    
    for mem_idx, slot_equations in tqdm(equations_per_slot.items(), desc="Parsing equations", leave=False):
        # Check for multi-output format first (most common case)
        semicolon_idx = slot_equations.find(';')
        if semicolon_idx != -1:
            # Multi-output format: "y0: eq0; y1: eq1; ..."
            # Split and process in one pass
            for eq in slot_equations.split(';'):
                colon_idx = eq.find(':')
                if colon_idx != -1:
                    # Strip only the necessary parts, not the whole string
                    output_name = eq[:colon_idx].strip()
                    equation = eq[colon_idx+1:].strip()
                    parsed[(mem_idx, output_name)] = equation
        else:
            # Single output format
            colon_idx = slot_equations.find(':')
            if colon_idx != -1:
                output_name = slot_equations[:colon_idx].strip()
                equation = slot_equations[colon_idx+1:].strip()
                parsed[(mem_idx, output_name)] = equation
            else:
                # No prefix - assume it applies to all outputs
                # Strip once and reuse
                eq_stripped = slot_equations.strip()
                for y_name in y_names:
                    parsed[(mem_idx, y_name)] = eq_stripped
    
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
        predictor = model.predictor
        if hasattr(predictor, 'trainable_equations'):
            # SymbolicPredictor with learned equations
            trainable_eqs = predictor.trainable_equations
            eq_names = predictor.equation_names
            sorted_keys = sorted(trainable_eqs.keys())
            
            for mem_idx, set_name in enumerate(tqdm(sorted_keys, desc="Extracting KAN equations", leave=False)):
                # Build list of equation strings more efficiently
                eq_strs = [f"{eq_name}: {trainable_eqs[set_name][eq_name].get_equation_string()}"
                          for eq_name in eq_names[set_name]]
                equations[mem_idx] = "; ".join(eq_strs)
        elif hasattr(predictor, 'kans'):
            # KANPredictor - abstract representation
            widths_str = str(model.widths)
            equations = {mem_idx: f"KAN{mem_idx}[{widths_str}]"
                        for mem_idx in range(len(predictor.kans))}
        else:
            equations[0] = "No equations available"
    
    elif model_name == 'LinearSymbolicCBM':
        # Extract linear equations from memory
        try:
            # Cache attribute lookups
            predictor = model.linear_memory_predictor
            memory_size = model.memory_size
            c_names = model.c_names
            y_names = model.y_names
            n_concepts = len(c_names)
            bias_mode = model.bias
            
            # Compute weights once
            equation_weights = predictor.equation_decoder(predictor.equation_memory.weight)
            equation_weights = equation_weights.view(
                memory_size, 
                len(predictor.parameters), 
                len(y_names)
            )
            # Add the Mask
            if predictor.stage == 'fine_tuning' and hasattr(predictor, 'mask'):
                reshaped_mask = predictor.mask.view(
                    memory_size, 
                    len(predictor.parameters), 
                    len(y_names)
                )
                equation_weights = equation_weights * reshaped_mask.to(equation_weights.device)
            weights_np = equation_weights.detach().cpu().numpy()
            
            # Pre-compute bias values if global
            global_bias = None
            if bias_mode == 'global':
                global_bias = [predictor.bias_params[i].item() for i in range(len(y_names))]
            
            # Build equations for all memory slots
            for mem_idx in tqdm(range(memory_size), desc="Extracting linear equations", leave=False):
                eq_strs = []
                for out_idx, y_name in enumerate(y_names):
                    # Build equation string using list comprehension for terms
                    terms = [f"{weights_np[mem_idx, c_idx, out_idx]:.4f}*{c_names[c_idx]}"
                            for c_idx in range(n_concepts)
                            if abs(weights_np[mem_idx, c_idx, out_idx]) > 1e-6]
                    
                    # Add bias only if non-zero
                    if bias_mode == 'local':
                        bias_value = weights_np[mem_idx, -1, out_idx]
                        if abs(bias_value) > 1e-6:
                            terms.append(f"{bias_value:.4f}")
                    elif bias_mode == 'global' and global_bias:
                        bias_value = global_bias[out_idx]
                        if abs(bias_value) > 1e-6:
                            terms.append(f"{bias_value:.4f}")
                    
                    eq_str = f"{y_name}: {' + '.join(terms)}" if terms else f"{y_name}: 0"
                    eq_strs.append(eq_str)
                
                equations[mem_idx] = "; ".join(eq_strs)
        except Exception as e:
            error_msg = f"Error extracting equation: {str(e)}"
            equations = {mem_idx: error_msg for mem_idx in range(getattr(model, 'memory_size', 1))}
    
    elif model_name == 'PriorSymbolicCBM':
        # Extract from prior_predictor
        predictor = model.prior_predictor
        if hasattr(predictor, 'trainable_equations'):
            trainable_eqs = predictor.trainable_equations
            eq_names = predictor.equation_names
            sorted_keys = sorted(trainable_eqs.keys())
            
            for mem_idx, set_name in enumerate(tqdm(sorted_keys, desc="Extracting prior equations", leave=False)):
                eq_strs = [f"{eq_name}: {trainable_eqs[set_name][eq_name].get_equation_string()}"
                          for eq_name in eq_names[set_name]]
                equations[mem_idx] = "; ".join(eq_strs)
        else:
            equations[0] = "No equations available"
    
    elif model_name == 'SymbolicRegressorCBM':
        # Extract from predictor
        predictor = model.predictor
        if hasattr(predictor, 'trainable_equations'):
            # SymbolicPredictor with learned equations
            trainable_eqs = predictor.trainable_equations
            eq_names = predictor.equation_names
            sorted_keys = sorted(trainable_eqs.keys())
            
            for mem_idx, set_name in enumerate(tqdm(sorted_keys, desc="Extracting SR equations", leave=False)):
                # NOTE: get_equation_string() returns equations with current fine-tuned parameter values
                eq_strs = [f"{eq_name}: {trainable_eqs[set_name][eq_name].get_equation_string()}"
                          for eq_name in eq_names[set_name]]
                equations[mem_idx] = "; ".join(eq_strs)
        else:
            # BlackBoxPredictor or not yet trained
            no_eq_msg = "No symbolic equations (BlackBoxPredictor)"
            equations = {mem_idx: no_eq_msg for mem_idx in range(getattr(model, 'memory_size', 1))}

    elif model_name == 'MemoryCBM':
        # Get the symbolic equivalent of each blackbox predictor
        memory_size = model.memory_size
        y_names = model.y_names
        
        for mem_idx in tqdm(range(memory_size), desc="Extracting memory equations", leave=False):
            eq_result = model.get_symbolic_equivalent(memory_idx=mem_idx, return_equations=True)
            # For multi-output, eq_result is a list of equations
            # Check if iterable (but not string)
            if hasattr(eq_result, '__iter__') and not isinstance(eq_result, str):
                # Multi-output: use list comprehension
                eq_strs = [f"{y_names[i]}: {eq}" for i, eq in enumerate(eq_result)]
                equations[mem_idx] = "; ".join(eq_strs)
            else:
                # Single output
                y_name = y_names[0] if isinstance(y_names, list) else y_names
                equations[mem_idx] = f"{y_name}: {eq_result}"
    else:
        # Other models - no memory-based equations
        equations[0] = f"Model {model_name} does not use memory-based equations"
    
    return equations

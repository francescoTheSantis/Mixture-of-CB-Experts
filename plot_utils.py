"""
Yes, we generated a lot of tables/plots :)
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import torch
import pandas as pd
import scienceplots
import warnings
import yaml
import signal
from contextlib import contextmanager
from tqdm import tqdm
from show_results import table_path, result_figs, regression_datasets, MEMORY_MODELS_LIST, COMPLEXITY_METRICS_LIST, models_order, model_styles, custom_order, NUMBER_OF_CLASSES_PER_DATASET


class TimeoutError(Exception):
    """Custom exception for timeout."""
    pass


@contextmanager
def timeout(seconds=30):
    """Context manager that raises TimeoutError if operation takes longer than specified seconds."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set up the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# I used scienceplots for the style of the plots, but you can use any other style you want.
plt.style.use(['science', 'ieee', 'no-latex'])

########################################
########## Name conversion #############
########################################

def get_df_name(df):
    if df=='xor':
        return 'XOR'
    elif df=='dot':
        return 'DOT'
    elif df=='trigonometry':
        return 'Trigonometry'
    elif df=='and':
        return 'AND'
    elif df=='or':
        return 'OR'
    elif df=='cebab':
        return 'CEBaB'
    elif df=='mnist_addition':
        return 'MNIST+'
    elif df=='mnist_addition_incomplete':
        return 'MNIST-Add-Inc.'
    elif df=='cub':
        return 'CUB200'
    elif df=='imdb':
        return 'IMDB'
    elif df=='celeba':
        return 'CelebA'
    elif df=='mnist_even_odd':
        return 'MNIST-E/O'
    elif df=='awa2':
        return 'AWA2'
    elif df=='checkmark':
        return 'Checkmark'
    elif df=='celeba':
        return 'CelebA'
    elif df=='cub_incomplete':
        return 'CUB200-Incomplete'
    elif df=='awa2_incomplete':
        return 'AWA2-Incomplete'
    elif df=='cifar10':
        return 'CIFAR10'
    elif df=='cifar100':
        return 'CIFAR100'
    elif df=='dsprites_simple':
        return 'dSprites-Exp.'
    elif df=='dsprites_complex':
        return 'dSprites-Composed'
    elif df=='mnist_arithmetic':
        return 'MNIST-Arith.'
    elif df=='pendulum':
        return 'Pendulum'
    elif df=='mawps':
        return 'MAWPS'
    elif df=='feynman_I_6_2':
        return 'Feynman I.6.20'
    elif df=='feynman_I_9_18':
        return 'Feynman I.9.18'
    elif df=='feynman_I_12_1':
        return 'Feynman I.12.1'
    elif df=='feynman_I_13_4':
        return 'Feynman I.13.4'
    elif df=='feynman_I_14_3':
        return 'Feynman I.14.3'
    elif df=='feynman_I_15_10':
        return 'Feynman I.15.10'

classes_x_dataset = {
    'cub': 200,
    'cub_incomplete': 200,
    'awa2': 50,
    'awa2_incomplete': 50,
    'cifar10': 10,
    'cifar100': 100,
}


#########################################
######### Data Extraction ###############
#########################################

def get_exp_from_path_cached(paths, cache_name, output_path):
    """
    Cached wrapper for get_exp_from_path function.
    
    Args:
        paths: List of paths to process
        cache_name: Name identifier for this experiment (e.g., 'sr_ablation', 'memory_ablation')
        output_path: Base output path where cache directory will be created
    
    Returns:
        Tuple of (performance_df, other_data) as returned by get_exp_from_path
    """
    # Create cache directory
    cache_dir = os.path.join(output_path, 'cached_results')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filename based on experiment name and parameters
    cache_file = os.path.join(cache_dir, f'{cache_name}_equations.csv')
    
    # Check if cache exists
    if os.path.exists(cache_file):
        print(f"Loading cached results for {cache_name} from {cache_file}")
        performance = pd.read_csv(cache_file)
        # Return tuple to match original function signature
        return performance
    else:
        print(f"No cache found for {cache_name}. Computing results...")
        # Call the original function
        performance = get_exp_from_path(paths)
        
        # Save to cache
        print(f"Saving results to cache: {cache_file}")
        performance.to_csv(cache_file, index=False)
        
        return performance

def get_exp_from_path(paths):
    from src.utils.complexity import compute_complexity
    from sympy import sympify, symbols
    
    # Collect all the experiments in the given paths
    exps_path = []
    lmr_paths = []
    for path in paths:
        if os.path.exists(path):
            experiment_dir = os.listdir(path)
            for exp in experiment_dir:
                exps_path += [os.path.join(path, exp, e) for e in os.listdir(os.path.join(path, exp)) if 'multirun' not in e]

    performance = pd.DataFrame()

    # Iterate over all the experiments and collect the performance metrics and the config
    for exp in tqdm(exps_path, desc="Processing experiments"):
        d = {}
        conf_file = os.path.join(exp, '.hydra/config.yaml')
        result_file = os.path.join(exp, 'logs/experiment_metrics/metrics.csv') 
        try: 
            if os.path.exists(conf_file) and os.path.exists(result_file):
                with open(conf_file, 'r') as file:
                    conf = yaml.safe_load(file)
                d['seed'] = conf['seed']
                d['dataset'] = conf['dataset']['metadata']['name']
                d['model'] = conf['model']['metadata']['name']
                d['memory_size'] = conf['memory_size']
                d['concept_percentage'] = conf['concept_percentage']
                # get also the task type (classification or regression)
                d['task_type'] = conf['dataset']['metadata']['task']
                d['path'] = exp

                with open(result_file, 'r') as file:
                    result = pd.read_csv(file)

                # Select the last row of the dataframe where we test the model
                # if 'test/y/acc' and 'test_concept_acc' are not in the dataframe, skip the experiment
                if 'test/y/mse' in result.columns:
                    d['task_mse'] = result['test/y/mse'].iloc[-1]
                    d['task_mae'] = result['test/y/mae'].iloc[-1]
                else:
                    d['task_acc'] = result['test/y/acc'].iloc[-1]

                if conf['model']['metadata']['name']=='blackbox':
                    d['concept'] = 0
                else:
                    if 'test/c/mse' in result.columns:
                        d['concept_mse'] = result['test/c/mse'].iloc[-1]
                        d['concept_mae'] = result['test/c/mae'].iloc[-1]
                    else:
                        d['concept_acc'] = result['test/c/acc'].iloc[-1]
                
                # Compute complexity for equations if available (skip for cem, blackbox, licem, dcr)
                if conf['model']['metadata']['name'] not in ['cem', 'blackbox', 'licem', 'dcr']:
                    predictions_file = os.path.join(exp, 'logs/experiment_metrics/test_predictions_per_sample.csv')
                    if os.path.exists(predictions_file):
                        try:
                            # with timeout(3):  # 3 second timeout
                            #     df_pred = pd.read_csv(predictions_file)

                            df_pred = pd.read_csv(predictions_file)

                            # Check if 'equation' column exists
                            if 'equation' in df_pred.columns:
                                # The variables of the equations are the concept names
                                vars = [x.replace('c_pred_','') for x in df_pred.columns if 'c_pred' in x]
                                
                                # Count unique equations and their occurrences
                                equation_counts = df_pred['equation'].value_counts().to_dict()

                                # Remove NaN or empty equations
                                equation_counts = {eq: cnt for eq, cnt in equation_counts.items() if pd.notna(eq) and eq != ''}

                                # Remove target from equations if present (e.g., "y: <equation>")
                                cleaned_equation_counts = {}
                                for eq, cnt in equation_counts.items():
                                    if ':' in eq:
                                        cleaned_eq = eq.split(':')[1].strip()
                                    else:
                                        cleaned_eq = eq
                                    # Sum counts if multiple equations clean to the same form
                                    if cleaned_eq in cleaned_equation_counts:
                                        cleaned_equation_counts[cleaned_eq] += cnt
                                    else:
                                        cleaned_equation_counts[cleaned_eq] = cnt

                                # Compute all complexity metrics
                                complexity_metrics = COMPLEXITY_METRICS_LIST
                                complexity_totals = {metric: 0 for metric in complexity_metrics}
                                
                                for eq, _ in tqdm(cleaned_equation_counts.items(), desc=f"Complexity, {d['model']}, {d['dataset']}", leave=False):
                                    sympy_eq = sympify(eq, locals={var: symbols(var) for var in vars})
                                    for metric in complexity_metrics:
                                        complexity_totals[metric] += compute_complexity(sympy_eq, metric=metric)
                                
                                # Store all complexity metrics
                                for metric in complexity_metrics:
                                    d[f'complexity_{metric}'] = complexity_totals[metric]
                        
                        except Exception as e:
                            print(f"Error computing complexity for {exp}: {e}")
                else:
                    # Store NaN for blackbox and cem models
                    complexity_metrics = COMPLEXITY_METRICS_LIST
                    for metric in complexity_metrics:
                        d[f'complexity_{metric}'] = np.nan

                performance = pd.concat([performance, pd.DataFrame([d])], ignore_index=True)
        except Exception as e:
            print(f"Error while processing {exp}: {e}")
            print(f"Skipping this experiment: {d}")
            continue

    return performance

def compute_pareto_knee(complexity, accuracy):
    """
    Compute the knee point of a Pareto front using the maximum distance to chord method.
    
    Args:
        complexity: Array-like of complexity values (to be minimized)
        accuracy: Array-like of accuracy values (to be maximized)
    
    Returns:
        knee_index: Index of the knee point in the input arrays
    """
    complexity = np.array(complexity)
    accuracy = np.array(accuracy)
    
    # Handle edge cases
    if len(complexity) <= 2:
        return 0  # Return first point if only 1-2 points
    
    # Step 1: Normalize the axes to [0, 1]
    complexity_min, complexity_max = complexity.min(), complexity.max()
    accuracy_min, accuracy_max = accuracy.min(), accuracy.max()
    
    # Avoid division by zero
    if complexity_max == complexity_min:
        complexity_norm = np.zeros_like(complexity)
    else:
        complexity_norm = (complexity - complexity_min) / (complexity_max - complexity_min)
    
    if accuracy_max == accuracy_min:
        accuracy_norm = np.zeros_like(accuracy)
    else:
        accuracy_norm = (accuracy - accuracy_min) / (accuracy_max - accuracy_min)
    
    # Step 2: Define the chord endpoints
    # A: point with minimum normalized complexity
    # B: point with maximum normalized accuracy
    a_idx = np.argmin(complexity_norm)
    b_idx = np.argmax(accuracy_norm)
    
    A = np.array([complexity_norm[a_idx], accuracy_norm[a_idx]])
    B = np.array([complexity_norm[b_idx], accuracy_norm[b_idx]])
    
    # Step 3: Compute distance to the chord for each point
    distances = []
    chord_vector = B - A
    chord_length = np.linalg.norm(chord_vector)
    
    # Avoid division by zero if A and B are the same point
    if chord_length == 0:
        return 0
    
    for i in range(len(complexity_norm)):
        P = np.array([complexity_norm[i], accuracy_norm[i]])
        AP = A - P
        
        # Compute perpendicular distance using cross product
        # In 2D, cross product gives scalar: (Bx - Ax)(Ay - Py) - (By - Ay)(Ax - Px)
        cross_product = chord_vector[0] * AP[1] - chord_vector[1] * AP[0]
        distance = abs(cross_product) / chord_length
        distances.append(distance)
    
    # Step 4: Select the knee point (maximum distance)
    knee_index = np.argmax(distances)
    
    return knee_index


def get_intervention_from_path(
        paths, 
        model_styles=None, 
        custom_order=None, 
        apply_filter=False, 
        fixed_memory=None, 
        selected_memory_size=1
):

    if apply_filter:
        # Eliminate Feynman datasets from custom_order
        custom_order = [d for d in custom_order if not d.startswith('feynman')]

        # Create the filtered_exps dict
        # If the dataset in fixed memory, filter the experiments to keep only those with memory_size equal to the fixed memory
        filtered_exps = []
        if fixed_memory is not None:
            for dataset in custom_order:
                if dataset not in fixed_memory.keys():
                    for model in model_styles.keys():
                        if model in MEMORY_MODELS_LIST:
                            filtered_exps.append({
                                'dataset': dataset,
                                'model': model,
                                'memory_size': selected_memory_size
                            })
                else:
                    for model in model_styles.keys():
                        if model in MEMORY_MODELS_LIST:
                            mem_size = fixed_memory[dataset]
                            filtered_exps.append({
                                'dataset': dataset,
                                'model': model,
                                'memory_size': mem_size
                        }) 
        else:          
            for dataset in custom_order:
                for model in model_styles.keys():
                    if model in MEMORY_MODELS_LIST:
                        filtered_exps.append({
                        'dataset': dataset,
                        'model': model,
                        'memory_size': selected_memory_size
                    })

    performance = pd.DataFrame()

    # Collect all the experiments in the given paths
    exps_path = []
    
    for path in paths:
        if os.path.exists(path):
            experiment_dir = os.listdir(path)
            for exp in experiment_dir:
                exps_path += [os.path.join(path, exp, e) for e in os.listdir(os.path.join(path, exp)) if 'multirun' not in e]

    for exp in exps_path:
        conf_file = os.path.join(exp, '.hydra/config.yaml')
        result_file = os.path.join(exp, 'logs/experiment_metrics/interventions.csv')        
        if os.path.exists(conf_file) and os.path.exists(result_file):
            with open(result_file, 'r') as file:
                d = pd.read_csv(result_file)[['noise','p_int','f1','accuracy','mae','mse']]
            
            with open(conf_file, 'r') as file:
                conf = yaml.safe_load(file)
            d['seed'] = conf['seed']
            d['dataset'] = conf['dataset']['metadata']['name']
            d['model'] = conf['model']['metadata']['name']
            d['memory_size'] = conf['memory_size']

            performance = pd.concat([performance, d], ignore_index=True)

    if apply_filter is False:
        return performance
    else:
        # For classification datasets with multiple memory sizes, select the memory size
        # that achieves the best performance when all concepts are intervened (p_int = 1.0)
        
        # Create the filtered_exps with best intervention performance selection
        filtered_exps_with_best = []
        
        for dataset in custom_order:
            # Check if it's a classification dataset (not in regression_datasets)
            is_classification = dataset not in regression_datasets
            
            for model in model_styles.keys():
                if model in MEMORY_MODELS_LIST:
                    # Check if dataset has fixed memory
                    if dataset in fixed_memory.keys():
                        # Use fixed memory size
                        mem_size = fixed_memory[dataset]
                        filtered_exps_with_best.append({
                            'dataset': dataset,
                            'model': model,
                            'memory_size': mem_size
                        })
                    elif is_classification:
                        # For classification datasets without fixed memory, select memory size
                        # with best performance at full intervention (p_int = 1.0)
                        model_dataset_data = performance[
                            (performance['dataset'] == dataset) & 
                            (performance['model'] == model) &
                            (performance['p_int'] == 1.0)
                        ]
                        
                        if len(model_dataset_data) > 0:
                            # Group by memory_size and compute mean accuracy at p_int=1.0
                            mean_perf_by_memory = model_dataset_data.groupby('memory_size')['accuracy'].mean()
                            best_memory_size = mean_perf_by_memory.idxmax()
                            
                            # Print the selected model, dataset, and memory size
                            print(f"[Best Intervention Selection] Dataset: {dataset}, Model: {model}, Memory Size: {best_memory_size}, Accuracy: {mean_perf_by_memory[best_memory_size]:.4f}")
                            
                            filtered_exps_with_best.append({
                                'dataset': dataset,
                                'model': model,
                                'memory_size': best_memory_size
                            })
                        else:
                            # Fallback if no data found
                            filtered_exps_with_best.append({
                                'dataset': dataset,
                                'model': model,
                                'memory_size': selected_memory_size
                            })
                    else:
                        # For regression datasets without fixed memory, use selected_memory_size
                        filtered_exps_with_best.append({
                            'dataset': dataset,
                            'model': model,
                            'memory_size': selected_memory_size
                        })
        
        # Keep only the experiments in filtered_exps_with_best
        criteria = pd.DataFrame(filtered_exps_with_best)
        filtered_performance = performance.merge(criteria, on=criteria.columns.tolist(), how="inner")

        # If the model belong to cbm_linear, licem, dcr, cem, blackbox, add them to performance
        memoryless_models = performance[performance['model'].isin(['cbm_linear', 'cbm_mlp', 'licem', 'dcr', 'cem', 'blackbox'])]
        filtered_performance = pd.concat([filtered_performance, memoryless_models], ignore_index=True)

        # lin symbolic cbm with memory size 1 is cbm_linear
        cbm = performance[(performance['model']=='linear_symbolic_cbm') & (performance['memory_size']==1)]
        cbm['model'] = 'cbm_linear'
        filtered_performance = pd.concat([filtered_performance, cbm], ignore_index=True)

        return filtered_performance

##########################################
######## Data Processing #################
##########################################

def compute_avg_and_uncertainty(performance, custom_order, complexity_metric='visitation_length'):
    """
    Compute average and uncertainty metrics for performance data.
    
    Args:
        performance: DataFrame with performance data
        custom_order: List of datasets in desired order
        complexity_metric: Complexity metric to use ('node_count', 'depth', 'visitation_length', 
                          'total_variables', 'total_operations', 'weighted_node_count'). Default: 'visitation_length'
    
    Returns:
        DataFrame with averaged metrics and standard errors
    """
    # Sort the points in the order of memory_size.
    performance = performance.sort_values(by=['dataset', 'memory_size'])

    # Set cmb_linear's memory_size to 1 and licem/dcr's memory_size to 500
    performance.loc[performance['model'].isin(['cmb_linear', 'cem', 'blackbox']), 'memory_size'] = 1
    performance.loc[performance['model'].isin(['licem', 'dcr']), 'memory_size'] = 500

    num_seeds = performance['seed'].nunique()

    # Multiply the accuracy values by 100 to convert them to percentages
    if 'task_acc' in performance.columns:
        performance['task_acc'] = performance['task_acc'] * 100

    # Determine which complexity column to use
    complexity_col = f'complexity_{complexity_metric}'
    if complexity_col not in performance.columns:
        # Fallback to 'complexity' if specific metric not found
        complexity_col = 'complexity'

    # Avg over the seeds for the performance metrics
    if 'task_acc' in performance.columns and 'task_mae' in performance.columns:
        # Handle both accuracy and MSE/MAE metrics
        agg_dict_acc = {
            'mean_task': ('task_acc', 'mean'),
            'std_task': ('task_acc', 'std'),
            'mean_concept': ('concept_acc', 'mean'),
            'std_concept': ('concept_acc', 'std'),
            'task_type': ('task_type', 'first')
        }
        if complexity_col in performance.columns:
            agg_dict_acc['mean_complexity'] = (complexity_col, 'mean')
            agg_dict_acc['std_complexity'] = (complexity_col, 'std')
        
        performance_acc = performance.dropna(subset=['task_acc']).groupby(['dataset', 'memory_size', 'model']).agg(
            **agg_dict_acc
        ).reset_index()
        performance_acc['metric_type'] = 'accuracy'
        
        agg_dict_mae = {
            'mean_task': ('task_mae', 'mean'),
            'std_task': ('task_mae', 'std'),
            'mean_concept': ('concept_mae', 'mean'),
            'std_concept': ('concept_mae', 'std'),
            'task_type': ('task_type', 'first')
        }
        if complexity_col in performance.columns:
            agg_dict_mae['mean_complexity'] = (complexity_col, 'mean')
            agg_dict_mae['std_complexity'] = (complexity_col, 'std')
        
        performance_mae = performance.dropna(subset=['task_mae']).groupby(['dataset', 'memory_size', 'model']).agg(
            **agg_dict_mae
        ).reset_index()
        performance_mae['metric_type'] = 'mae'
        
        performance = pd.concat([performance_acc, performance_mae], ignore_index=True)
    else:
        # Fallback to original logic
        task_col = 'task_acc' if 'task_acc' in performance.columns else 'task_mae'
        concept_col = 'concept_acc' if 'concept_acc' in performance.columns else 'concept_mae'
        agg_dict_fallback = {
            'mean_task': (task_col, 'mean'),
            'std_task': (task_col, 'std'),
            'mean_concept': (concept_col, 'mean'),
            'std_concept': (concept_col, 'std'),
            'task_type': ('task_type', 'first')
        }
        if complexity_col in performance.columns:
            agg_dict_fallback['mean_complexity'] = (complexity_col, 'mean')
            agg_dict_fallback['std_complexity'] = (complexity_col, 'std')
        
        performance = performance.groupby(['dataset', 'memory_size', 'model']).agg(
            **agg_dict_fallback
        ).reset_index()
        performance['metric_type'] = 'accuracy' if 'acc' in task_col else 'mae'

    # instead of the std compute the standard error at 95% confidence
    performance['se_task'] = 1.96 * performance['std_task'] / np.sqrt(num_seeds)
    performance['se_concept'] = 1.96 * performance['std_concept'] / np.sqrt(num_seeds)
    
    # Compute standard error for complexity if it exists
    if 'std_complexity' in performance.columns:
        performance['se_complexity'] = 1.96 * performance['std_complexity'] / np.sqrt(num_seeds)

    # Order the datasets according to custom_order and memory_size
    performance['dataset'] = pd.Categorical(performance['dataset'], categories=custom_order, ordered=True)
    performance = performance.sort_values(['dataset', 'memory_size'])

    return performance

###########################################
######### Plotting Interventions ##########
###########################################

def plot_intervention_results(
    df, 
    metric='accuracy', 
    unique_noises=[0.0], 
    classification_noise=None,
    regression_noise=None,
    title_font=None, 
    label_font=None, 
    tick_font=None, 
    legend_font=None,
    custom_order=None,
    model_styles=None,
    relative_accuracy=True,
    out_dir=None,
    ranges=None,
    regression_ranges=None,
):
    """
    Plot intervention results across datasets.
    
    Parameters:
    -----------
    unique_noises : list, optional
        Default noise level(s) to use if classification_noise and regression_noise are not specified.
    classification_noise : float, optional
        Noise level to use for classification datasets. If None, uses unique_noises.
    regression_noise : float, optional
        Noise level to use for regression datasets. If None, uses unique_noises.
    ranges : list, optional
        Y-axis range specification for classification datasets.
        Each element corresponds to a classification subplot by index.
        Formats:
        - [y_min, y_max]: Single continuous axis
        - [[y_low_min, y_low_max], [y_high_min, y_high_max]]: Broken/stacked axis
        If None or shorter than needed, uses matplotlib auto-scale for missing entries.
    regression_ranges : list, optional
        Y-axis range specification for regression datasets.
        Each element corresponds to a regression subplot by index.
        Formats:
        - [y_min, y_max]: Single continuous axis
        - [[y_low_min, y_low_max], [y_high_min, y_high_max]]: Broken/stacked axis
        If None or shorter than needed, uses matplotlib auto-scale for missing entries.
    """

    unique_datasets = [d for d in df['dataset'].unique()]

    # # sort the organized_datasets according to custom_order
    # organized_datasets = [d for d in custom_order if d in organized_datasets]

    # Separate datasets by task type
    classification_datasets = [d for d in unique_datasets if d not in regression_datasets]
    found_regression_datasets = [d for d in unique_datasets if d in regression_datasets]

    # Orgsanize dataset based on custom order
    classification_datasets = [d for d in custom_order if d in classification_datasets]
    found_regression_datasets = [d for d in custom_order if d in found_regression_datasets]
    
    # Determine if we have both types of datasets
    has_classification = len(classification_datasets) > 0
    has_regression = len(found_regression_datasets) > 0
    
    # If only one type exists, put all on a single row with legend below
    if has_classification and not has_regression:
        # Only classification datasets - put all on one row
        organized_datasets = classification_datasets
        n_cols = len(classification_datasets)
        n_rows = 1
    elif has_regression and not has_classification:
        # Only regression datasets - put all on one row
        organized_datasets = found_regression_datasets
        n_cols = len(found_regression_datasets)
        n_rows = 1
    else:
        # Both types exist - organize with regression first, then classification
        organized_datasets = found_regression_datasets + classification_datasets
        n_cols = max(len(classification_datasets), len(found_regression_datasets))
        n_rows = 2  # Force 2 rows: regression on first row, classification on second
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows), sharex=False, sharey=False)

    # Ensure axes is always a 2D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
    else:
        axes = np.array(axes)
    
    # Validate and set defaults for ranges parameters
    # ranges applies to classification datasets, regression_ranges applies to regression datasets
    n_classification = len(classification_datasets)
    n_regression = len(found_regression_datasets)
    
    if ranges is None:
        ranges = [None] * n_classification  # Default: let matplotlib auto-scale
    else:
        # Extend ranges with None for missing entries (auto-scale)
        if len(ranges) < n_classification:
            ranges = list(ranges) + [None] * (n_classification - len(ranges))
    
    if regression_ranges is None:
        regression_ranges = [None] * n_regression  # Default: let matplotlib auto-scale
    else:
        # Extend regression_ranges with None for missing entries (auto-scale)
        if len(regression_ranges) < n_regression:
            regression_ranges = list(regression_ranges) + [None] * (n_regression - len(regression_ranges))
    
    # Track which subplots need broken axes (to apply after tight_layout)
    broken_axes_specs = {}
    
    # Store plot data for broken axes reconstruction
    plot_data_for_broken_axes = {}
    
    for idx, dataset in enumerate(organized_datasets):
        # Determine row and column based on layout
        if has_classification and not has_regression:
            # Only classification - all on one row
            row = 0
            col = idx
        elif has_regression and not has_classification:
            # Only regression - all on one row
            row = 0
            col = idx
        else:
            # Both types - regression on row 0, classification on row 1
            if dataset in found_regression_datasets:
                row = 0
                col = found_regression_datasets.index(dataset)
            else:
                row = 1
                col = classification_datasets.index(dataset)
        
        # Access the axis consistently
        ax = axes[row][col]
        
        # Handle broken/stacked axes for both classification and regression datasets
        use_broken_axis = False
        
        # Determine if this is a classification or regression dataset and get the appropriate index
        is_classification = dataset in classification_datasets
        is_regression = dataset in found_regression_datasets
        
        if is_classification:
            class_idx = classification_datasets.index(dataset)
            range_spec = ranges[class_idx] if class_idx < len(ranges) else None
        elif is_regression:
            reg_idx = found_regression_datasets.index(dataset)
            range_spec = regression_ranges[reg_idx] if reg_idx < len(regression_ranges) else None
        else:
            range_spec = None
        
        # Check if this is a broken axis specification
        if (range_spec is not None and isinstance(range_spec, list) and len(range_spec) == 2 and 
            isinstance(range_spec[0], list) and isinstance(range_spec[1], list)):
            # Mark for broken axis creation after tight_layout
            use_broken_axis = True
            broken_axes_specs[(row, col)] = {
                'ax': ax,
                'y_low_range': range_spec[0],
                'y_high_range': range_spec[1],
                'dataset': dataset
            }
        elif range_spec is not None:
            # Regular single axis: set y-limits based on range_spec
            if isinstance(range_spec, list) and len(range_spec) == 2:
                ax.set_ylim(range_spec)
        
        # Determine which noise level to use based on dataset type
        if dataset in found_regression_datasets:
            if regression_noise is not None:
                dataset_noises = [regression_noise] if not isinstance(regression_noise, list) else regression_noise
            else:
                dataset_noises = unique_noises
        else:
            if classification_noise is not None:
                dataset_noises = [classification_noise] if not isinstance(classification_noise, list) else classification_noise
            else:
                dataset_noises = unique_noises
        
        for noise in dataset_noises:
            data = df[(df['noise'] == noise) & (df['dataset'] == dataset)]
            if dataset in found_regression_datasets:
                metric = 'mae'
            else:
                metric = 'accuracy'
            grouped_data = data.groupby(['p_int', 'model']).agg(
                mean_metric=(metric, 'mean'),
                std_metric=(metric, 'std')
            ).reset_index().fillna(0)
            
            # Calculate number of seeds for standard error
            num_seeds = data.groupby(['p_int', 'model']).size().reset_index(name='n_seeds')
            grouped_data = grouped_data.merge(num_seeds, on=['p_int', 'model'])
            grouped_data['se_metric'] = 1.96 * grouped_data['std_metric'] / np.sqrt(grouped_data['n_seeds'])
            
            # Convert accuracy to error rate for classification tasks (accuracy is in decimal form 0-1)
            if dataset not in found_regression_datasets:
                grouped_data['mean_metric'] = 100 * (1 - grouped_data['mean_metric'])
            
            if relative_accuracy:
                # Calculate relative accuracy for each model
                for model in grouped_data['model'].unique():
                    model_data = grouped_data[grouped_data['model'] == model]
                    baseline = model_data[model_data['p_int'] == 0]['mean_metric'].iloc[0] if len(model_data[model_data['p_int'] == 0]) > 0 else 0
                    grouped_data.loc[grouped_data['model'] == model, 'mean_metric'] -= baseline
            
            # Order models according to model_styles
            models_in_data = [m for m in model_styles.keys() if m in grouped_data['model'].unique()]
            for model in models_in_data:
                model_data = grouped_data[grouped_data['model'] == model]
                style = model_styles.get(model, {'marker': 'o', 'color': 'black', 'size': 10, 'name': model})
                
                # Plot the line and markers
                ax.plot(model_data['p_int'], model_data['mean_metric'],
                       marker=style['marker'], 
                       color=style['color'], 
                       markersize=style['size'], 
                       label=style['name'],
                       markeredgecolor='black',
                       markeredgewidth=0.1,
                       alpha=0.8,
                       linestyle='-')
                
                # Add shaded area for uncertainty
                ax.fill_between(model_data['p_int'],
                               model_data['mean_metric'] - model_data['se_metric'],
                               model_data['mean_metric'] + model_data['se_metric'],
                               color=style['color'],
                               alpha=0.2)
                
                # Store plot data for broken axes reconstruction
                if use_broken_axis:
                    if (row, col) not in plot_data_for_broken_axes:
                        plot_data_for_broken_axes[(row, col)] = []
                    plot_data_for_broken_axes[(row, col)].append({
                        'type': 'plot_with_fill',
                        'x': model_data['p_int'].values,
                        'y': model_data['mean_metric'].values,
                        'se': model_data['se_metric'].values,
                        'marker': style['marker'],
                        'color': style['color'],
                        'markersize': style['size'],
                        'label': style['name'],
                        'markeredgecolor': 'black',
                        'markeredgewidth': 0.1,
                        'alpha': 0.8,
                        'linestyle': '-'
                    })
        
        # Set x-axis ticks and labels
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
        
        # Show xlabel only for the last row
        if has_classification and not has_regression:
            # Only classification - all on one row, show xlabel on all
            ax.set_xlabel('$p_{int}$', fontsize=label_font['size'])
        elif has_regression and not has_classification:
            # Only regression - all on one row, show xlabel on all
            ax.set_xlabel('$p_{int}$', fontsize=label_font['size'])
        else:
            # Both types - show xlabel on row 1 (classification row)
            if row == 1:
                ax.set_xlabel('$p_{int}$', fontsize=label_font['size'])
        
        ax.set_title(f'{get_df_name(dataset)}', fontsize=label_font['size'])

        # Set ylabel only for the leftmost subplot in each row
        if col == 0:
            if has_classification and not has_regression:
                # Only classification datasets
                ylabel = '$\Delta$ (Error Rate)' if relative_accuracy else 'Error Rate'
                ax.set_ylabel(ylabel, fontsize=label_font['size'])
            elif has_regression and not has_classification:
                # Only regression datasets
                ylabel = '$\Delta$ MAE' if relative_accuracy else 'MAE'
                ax.set_ylabel(ylabel, fontsize=label_font['size'])
            else:
                # Both types - set ylabel based on dataset type
                if dataset in found_regression_datasets:
                    ylabel = '$\Delta$ MAE' if relative_accuracy else 'MAE'
                    ax.set_ylabel(ylabel, fontsize=label_font['size'])
                else:
                    ylabel = '$\Delta$ (Error Rate)' if relative_accuracy else 'Error Rate'
                    ax.set_ylabel(ylabel, fontsize=label_font['size'])
        
        ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
        ax.minorticks_off()
        ax.grid(True, alpha=0.3, zorder=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # Hide empty subplots and check if we can place legend in missing subplot
    legend_in_subplot = False
    legend_ax = None
    
    if has_classification and not has_regression:
        # Only classification on single row - legend always below
        legend_in_subplot = False
    elif has_regression and not has_classification:
        # Only regression on single row - legend always below
        legend_in_subplot = False
    else:
        # Both types exist - original logic
        total_datasets = len(classification_datasets) + len(found_regression_datasets)
        
        # Hide unused subplots in first row (regression)
        for col in range(len(found_regression_datasets), n_cols):
            axes[0][col].set_visible(False)
            # Use the rightmost empty subplot in first row for legend if available
            if col == n_cols - 1:
                legend_in_subplot = True
                legend_ax = axes[0][col]
        
        # Hide unused subplots in second row (classification) and check for legend placement
        for col in range(len(classification_datasets), n_cols):
            axes[1][col].set_visible(False)
            # Use the rightmost empty subplot in second row for legend if first row doesn't have one
            if legend_ax is None and col == n_cols - 1:
                legend_in_subplot = True
                legend_ax = axes[1][col]
    
    # Filter the style according to the models' names which are present in the df
    filtered_styles = {name: style for name, style in model_styles.items() if name in df['model'].values}

    # Create custom legend handles ordered by models_order
    custom_handles = [plt.Line2D([0], [0], marker=filtered_styles[model]['marker'], color='w', markerfacecolor=filtered_styles[model]['color'], markersize=filtered_styles[model]['size']+10, label=filtered_styles[model]['name'], markeredgewidth=0.5, markeredgecolor='black') for model in models_order if model in filtered_styles]

    # Place legend either in empty subplot or below plots
    if legend_in_subplot:
        legend_ax.set_visible(True)
        legend_ax.axis('off')  # Hide axes
        legend_ax.legend(handles=custom_handles, loc='center', ncol=1, fontsize=tick_font['size'], frameon=True)
    else:
        # Create a single legend below the plots
        fig.legend(
            handles=custom_handles,
            loc='lower center',
            ncol=(len(custom_handles) + 1) // 2,  # Split legend into two rows
            fontsize=tick_font['size'],
            frameon=True,
            bbox_to_anchor=(0.5, -0.3),
            columnspacing=1.0,
            handletextpad=0.5
        )

    plt.tight_layout()
    
    # Create broken axes after tight_layout to get correct positions
    for (row, col), spec in broken_axes_specs.items():
        original_ax = spec['ax']
        y_low_range = spec['y_low_range']
        y_high_range = spec['y_high_range']
        dataset = spec['dataset']
        
        # Get the final position after tight_layout
        pos = original_ax.get_position()
        
        # Create two stacked axes with precise dimensions
        # Bottom 60%, gap ~1.5%, top ~38.5% to sum to 100%
        height_ratio_bottom = 0.60
        height_ratio_top = 0.385
        gap_ratio = 0.015
        
        # Calculate absolute heights
        bottom_height = pos.height * height_ratio_bottom
        gap_height = pos.height * gap_ratio
        top_height = pos.height * height_ratio_top
        
        # Hide the original axis
        original_ax.set_visible(False)
        
        # Bottom axis
        ax_bottom = fig.add_axes([pos.x0, pos.y0, pos.width, bottom_height])
        ax_bottom.set_ylim(y_low_range)
        
        # Top axis
        ax_top = fig.add_axes([pos.x0, pos.y0 + bottom_height + gap_height, pos.width, top_height])
        ax_top.set_ylim(y_high_range)
        
        # Replay stored plot commands on both axes
        if (row, col) in plot_data_for_broken_axes:
            for plot_cmd in plot_data_for_broken_axes[(row, col)]:
                for target_ax in [ax_bottom, ax_top]:
                    if plot_cmd['type'] == 'plot_with_fill':
                        # Plot line with markers
                        target_ax.plot(
                            plot_cmd['x'],
                            plot_cmd['y'],
                            marker=plot_cmd['marker'],
                            color=plot_cmd['color'],
                            markersize=plot_cmd['markersize'],
                            label=plot_cmd['label'] if target_ax == ax_bottom else "",
                            markeredgecolor=plot_cmd['markeredgecolor'],
                            markeredgewidth=plot_cmd['markeredgewidth'],
                            alpha=plot_cmd['alpha'],
                            linestyle=plot_cmd['linestyle']
                        )
                        
                        # Add fill_between for uncertainty
                        target_ax.fill_between(
                            plot_cmd['x'],
                            plot_cmd['y'] - plot_cmd['se'],
                            plot_cmd['y'] + plot_cmd['se'],
                            color=plot_cmd['color'],
                            alpha=0.2
                        )
        
        # Configure x-axis: hide labels on top, show on bottom
        # IMPORTANT: Copy x-axis ticks to BOTH axes to ensure alignment
        tick_positions = original_ax.get_xticks()
        tick_labels = [t.get_text() for t in original_ax.get_xticklabels()]
        
        ax_bottom.set_xticks(tick_positions)
        ax_bottom.set_xticklabels(tick_labels)
        
        ax_top.set_xticks(tick_positions)
        ax_top.set_xticklabels([])  # Hide labels on top axis
        ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        # Add diagonal break markers
        d = 0.015
        kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        
        kwargs = dict(transform=ax_bottom.transAxes, color='k', clip_on=False, linewidth=1)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        
        # Copy styling
        for target_ax in [ax_bottom, ax_top]:
            target_ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
            target_ax.minorticks_off()
            target_ax.grid(True, alpha=0.3, zorder=0)
            target_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
        
        # Set title on top axis
        ax_top.set_title(f'{get_df_name(dataset)}', fontsize=label_font['size'])
        
        # Set xlabel on bottom axis
        ax_bottom.set_xlabel(original_ax.get_xlabel(), fontdict={'size': label_font['size']})
        
        # Set ylabel centered on the full subplot
        if col == 0:
            ylabel = original_ax.get_ylabel()
            ax_bottom.yaxis.label.set_visible(False)
            fig.text(pos.x0 - 0.039,
                    pos.y0 + pos.height / 2,
                    ylabel,
                    fontdict={'size': label_font['size']},
                    rotation=90,
                    va='center',
                    ha='center')
        
        # Update axes array reference
        axes[row][col] = ax_bottom
    
    str_store = str(unique_noises[0]).replace('.', '') + str(classification_noise).replace('.', '') + str(regression_noise).replace('.', '')
    suffix = 'relative_accuracy_difference' if relative_accuracy else 'absolute_accuracy'
    os.makedirs(f'{out_dir}/intervention/{suffix}', exist_ok=True)
    plt.savefig(f'{out_dir}/intervention/{suffix}/{str_store}.pdf')
    plt.show()

###########################################
######### Plotting Memory Ablation ########
###########################################

def plot_memory_ablation(performance, model_styles, title_font, label_font, tick_font, custom_order):

    performance = compute_avg_and_uncertainty(performance, custom_order)

    # Replace memory_size=500 with max_memory_size + 1 for proper tick positioning
    for dataset in performance['dataset'].unique():
        dataset_data = performance[performance['dataset'] == dataset]
        memory_sizes = dataset_data['memory_size'].unique()
        max_non_infinity = max([s for s in memory_sizes if s != 500]) if any(s != 500 for s in memory_sizes) else 0
        if 500 in memory_sizes:
            performance.loc[(performance['dataset'] == dataset) & (performance['memory_size'] == 500), 'memory_size'] = max_non_infinity + 1

    # Separate datasets by task type
    classification_datasets = performance[performance['task_type'] == 'classification']['dataset'].unique()
    found_regression_datasets = performance[performance['task_type'] == 'regression']['dataset'].unique()
    
    # Organize datasets with MAE first, then 1-accuracy
    organized_datasets = list(found_regression_datasets) + list(classification_datasets)
    
    n_datasets = len(organized_datasets)
    n_cols = max(len(classification_datasets), len(found_regression_datasets))  
    n_rows = 2  # Force 2 rows: MAE on first row, 1-accuracy on second

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows), sharey=False)
    
    # Handle case where we have only one subplot
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for idx, dataset in enumerate(organized_datasets):
        # Determine row based on task type
        if dataset in found_regression_datasets:
            row = 0
            col = list(found_regression_datasets).index(dataset)
        else:
            row = 1
            col = list(classification_datasets).index(dataset)
        
        if n_rows == 1:
            ax = axes[col] if n_cols > 1 else axes[0]
        else:
            ax = axes[row][col] if n_cols > 1 else axes[row][0] if col == 0 else axes[row][col]
            
        data = performance[performance['dataset'] == dataset]
        metric_type = data['metric_type'].iloc[0]
        
        # Get distinct memory sizes for this dataset
        distinct_memory_sizes = sorted(data['memory_size'].unique())
        
        for model in data['model'].unique():
            model_data = data[data['model'] == model]
            
            # Check if this model has unique memory size (only one data point)
            has_unique_memory_size = len(model_data) == 1
            
            # For accuracy metrics, plot 1-accuracy (error rate)
            if metric_type == 'accuracy':
                y_values = 100 - model_data['mean_task']
                y_errors = model_data['se_task']  # Error bars remain the same
            else:
                y_values = model_data['mean_task']
                y_errors = model_data['se_task']
            
            if has_unique_memory_size:
                # Show uncertainty as vertical error bars for single points
                ax.errorbar(
                    model_data['memory_size'], 
                    y_values,
                    yerr=y_errors,
                    label=model_styles[model]['name'], 
                    marker=model_styles[model]['marker'], 
                    color=model_styles[model]['color'], 
                    markersize=model_styles[model]['size'],
                    markeredgecolor='black',
                    markeredgewidth=0.1,
                    capsize=3,
                    capthick=1.5,
                    alpha=0.8,
                    linestyle='none'  # No line for single points
                )
            else:
                # Show uncertainty as shaded area for multiple points
                ax.plot(
                    model_data['memory_size'], 
                    y_values,
                    label=model_styles[model]['name'], 
                    marker=model_styles[model]['marker'], 
                    color=model_styles[model]['color'], 
                    markersize=model_styles[model]['size'],
                    markeredgecolor='black',
                    markeredgewidth=0.1,
                    alpha=0.8
                )
                ax.fill_between(
                    model_data['memory_size'],
                    y_values - y_errors,
                    y_values + y_errors,
                    color=model_styles[model]['color'],
                    alpha=0.2
                )
        
        ax.set_title(get_df_name(dataset), fontdict=title_font)

        if row==1:
            ax.set_xlabel('Memory Size', fontdict=label_font)
        else:
            ax.set_xlabel('')
        
        # Set y-label only for the leftmost subplot in each row
        if col == 0:
            ylabel = 'MAE' if metric_type == 'mae' else 'Error Rate'
            ax.set_ylabel(ylabel, fontdict=label_font)
        
        # Set x-axis to log scale
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        
        # Set x-axis ticks and labels with infinity symbol for adjusted memory_size
        ax.set_xticks(distinct_memory_sizes)
        x_labels = []
        for size in distinct_memory_sizes:
            # Check if this was originally 500 (now max+1) by checking if it's the max value
            if size == max(distinct_memory_sizes) and size > 2:
                x_labels.append('$\dots\infty$')
            else:
                x_labels.append(str(int(size)))
        ax.set_xticklabels(x_labels)
        
        ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
        ax.minorticks_off()
        ax.grid(True, alpha=0.3, zorder=0)

    # Hide empty subplots and find the rightmost empty subplot for legend
    legend_ax = None
    
    # Hide unused subplots in first row (MAE datasets)
    for col in range(len(found_regression_datasets), n_cols):
        axes[0][col].set_visible(False)
        # Use the rightmost empty subplot in first row for legend if available
        if col == n_cols - 1:
            legend_ax = axes[0][col]
    
    # Hide unused subplots in second row (1-accuracy datasets)
    for col in range(len(classification_datasets), n_cols):
        axes[1][col].set_visible(False)
        # Use the rightmost empty subplot in second row for legend if first row doesn't have one
        if legend_ax is None and col == n_cols - 1:
            legend_ax = axes[1][col]

    # Filter the style according to the models' names which are present in the performance df
    filtered_styles = {name: style for name, style in model_styles.items() if name in performance['model'].values}

    # Create custom legend handles ordered by models_order
    custom_handles = [plt.Line2D([0], [0], marker=filtered_styles[model]['marker'], color='w', markerfacecolor=filtered_styles[model]['color'], markersize=(filtered_styles[model]['size'])+10, label=filtered_styles[model]['name'], markeredgewidth=0.5, markeredgecolor='black') for model in models_order if model in filtered_styles]

    # Place legend in empty subplot if available, otherwise below plots
    if legend_ax is not None:
        legend_ax.set_visible(True)
        legend_ax.axis('off')  # Hide axes
        legend_ax.legend(handles=custom_handles, loc='center', ncol=1, fontsize=tick_font['size'], frameon=True)
    else:
        # Create a single legend below the plots with one model per column
        fig.legend(handles=custom_handles, loc='lower center', ncol=len(custom_handles), fontsize=tick_font['size'], frameon=True, bbox_to_anchor=(0.5, -0.15), columnspacing=1.0, handletextpad=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(result_figs, 'memory_ablation.pdf'))

    plt.show()


######################################
####### Concept Size Ablation ########
######################################

def plot_concept_size_ablation(
        performance, 
        model_styles, 
        title_font, 
        label_font, 
        tick_font,
        custom_order=None):
    """
    Plots the concept size ablation results.
    In this plot the concept accuracy is shown on the y axis while the
    concept size (percentage) is shown on the x axis.
    Each line represent a different model and each subplot the results on a different dataset
    """

    # Sort the points in the order of concept_percentage.
    performance = performance.sort_values(by=['dataset', 'concept_percentage'])

    num_seeds = performance['seed'].nunique()

    # Avg over the seeds for the performance metrics
    performance = performance.groupby(['dataset', 'concept_percentage', 'model']).agg(
        mean_task=('task_acc', 'mean'),
        std_task=('task_acc', 'std'),
        mean_concept=('concept_acc', 'mean'),
        std_concept=('concept_acc', 'std')
    ).reset_index()
    
    # instead of the std compute the standard error at 95% confidence
    performance['se_task'] = 1.96 * performance['std_task'] / np.sqrt(num_seeds)
    performance['se_concept'] = 1.96 * performance['std_concept'] / np.sqrt(num_seeds)

    # Get unique datasets and order them according to custom_order
    available_datasets = performance['dataset'].unique()
    if custom_order is not None:
        # Filter custom_order to only include datasets that exist in performance
        ordered_datasets = [d for d in custom_order if d in available_datasets]
        # Add any remaining datasets not in custom_order
        remaining_datasets = [d for d in available_datasets if d not in ordered_datasets]
        ordered_datasets.extend(remaining_datasets)
    else:
        ordered_datasets = list(available_datasets)
    
    n_datasets = len(ordered_datasets)
    
    # Calculate grid dimensions
    n_cols = min(n_datasets, 4)  # Max 4 columns
    n_rows = (n_datasets + n_cols - 1) // n_cols  # Ceiling division

    # Create subplots with larger figure size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 7*n_rows), squeeze=False)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()

    # Iterate over each dataset
    for idx, dataset in enumerate(ordered_datasets):
        ax = axes_flat[idx]
        dataset_data = performance[performance['dataset'] == dataset]
        
        # Iterate over each model and plot its performance
        for model, style in model_styles.items():
            if model in dataset_data['model'].values:
                data = dataset_data[dataset_data['model'] == model]
                ax.plot(data['concept_percentage'], data['mean_task'],
                        label=style['name'], marker=style['marker'], color=style['color'],
                        markersize=style['size'], markeredgecolor='black', markeredgewidth=0.1, alpha=0.8)
                ax.fill_between(data['concept_percentage'], 
                              data['mean_task'] - data['se_task'], 
                              data['mean_task'] + data['se_task'], 
                              color=style['color'], alpha=0.2)

        # Set the title using get_df_name
        ax.set_title(get_df_name(dataset), fontdict=title_font)
        ax.set_xlabel("Concept Percentage", **label_font)
        ax.set_ylabel("Accuracy", **label_font)

        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
        ax.tick_params(axis='both', which='minor', bottom=False, left=False)
        ax.minorticks_off()
        ax.grid(True)

    # Hide unused subplots
    for idx in range(n_datasets, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Create a single legend for all subplots
    # Collect all unique models across all datasets
    all_models_in_data = performance['model'].unique()
    handles, labels = [], []
    for model, style in model_styles.items():
        if model in all_models_in_data:
            handle = plt.Line2D([0], [0], marker=style['marker'], color=style['color'],
                               markersize=style['size']*0.6, markeredgecolor='black', 
                               markeredgewidth=0.1, alpha=0.8, label=style['name'])
            handles.append(handle)
            labels.append(style['name'])
    
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
               ncol=min(len(handles), 6), fontsize=tick_font['size']*0.8, frameon=True)

    # Save the figure
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at the bottom for legend
    plt.savefig(os.path.join(result_figs, 'concept_size_ablation.pdf'), bbox_inches='tight')



############################################
############## Pareto plots ################
############################################

def filter_pareto_models(df: pd.DataFrame, fixed_memory: dict = None, custom_order: list = None, delta_threshold: float = 0.1) -> pd.DataFrame:
    """
    Filters a dataframe of model results to keep only specific memory_size per model/dataset/seed:
    
    For classification datasets:
        - dcr, licem, cem, blackbox: take all experiments (all memory sizes)
        - Other memory-based models: take only memory_size = 2
    
    For regression datasets:
        - licem, cem, blackbox: take all experiments (all memory sizes)
        - Other memory-based models: use memory size from fixed_memory dictionary
    
    If `fixed_memory` is provided (dict mapping dataset -> memory_size), it's used for
    regression datasets with memory-based models.
    """

    df = compute_avg_and_uncertainty(df, custom_order)

    fixed_memory = fixed_memory or {}
    
    # Models that should take all experiments
    classification_all_models = ['cbm_linear', 'cbm_mlp', 'dcr', 'licem', 'cem', 'blackbox']
    regression_all_models = ['cbm_linear', 'cbm_mlp', 'licem', 'cem', 'blackbox']

    def filter_by_rules(group: pd.DataFrame) -> pd.DataFrame:
        dataset = group["dataset"].iloc[0]
        model = group["model"].iloc[0]
        task_type = group["task_type"].iloc[0]

        if task_type == "classification":
            # For classification datasets
            if model in classification_all_models:
                # Take all experiments for dcr, licem, cem, blackbox
                return group
            else:
                # For other memory-based models, take only memory_size = 2
                filtered_group = group[group["memory_size"] == 2]
                if not filtered_group.empty:
                    return filtered_group
                else:
                    # If memory_size = 2 doesn't exist, return the group as is
                    return group
        
        elif task_type == "regression":
            # For regression datasets
            if model in regression_all_models:
                # Take all experiments for licem, cem, blackbox
                return group
            else:
                # For other memory-based models, use fixed_memory
                if dataset in fixed_memory:
                    mem_size = fixed_memory[dataset]
                    filtered_group = group[group["memory_size"] == mem_size]
                    if not filtered_group.empty:
                        return filtered_group
                    else:
                        # If specified memory size doesn't exist, return the group as is
                        return group
                else:
                    # If no fixed_memory specified, return all
                    return group
        else:
            return group

    # Apply per dataset-model
    filtered = df.groupby(["dataset", "model"], group_keys=False).apply(filter_by_rules)

    # Return a dataframe containing:
    # - dataset
    # - model
    # - memory_size
    filtered = filtered[["dataset", "model", "memory_size"]]

    return filtered.reset_index(drop=True)

def plot_pareto_front(performance, model_styles, title_font, label_font, tick_font, custom_order, complexity_metric='visitation_length', ranges=None, regression_ranges=None):
    """
    Plot Pareto front for model complexity vs accuracy.
    
    Parameters:
    -----------
    complexity_metric : str
        The complexity metric to use for the x-axis. Options: 'node_count', 'depth', 
        'visitation_length', 'total_variables', 'total_operations', 'weighted_node_count'. Default: 'visitation_length'
    ranges : list, optional
        Y-axis range specification for classification datasets (bottom row).
        Each element corresponds to a classification subplot by index.
        Formats:
        - [y_min, y_max]: Single continuous axis
        - [[y_low_min, y_low_max], [y_high_min, y_high_max]]: Broken/stacked axis
        If None or shorter than needed, missing entries auto-scale.
    regression_ranges : list, optional
        Y-axis range specification for regression datasets (top row), same formats
        as `ranges`. If None or shorter than needed, missing entries auto-scale.
    """
    performance = compute_avg_and_uncertainty(performance, custom_order, complexity_metric=complexity_metric)

    # save the performance as csv
    performance.to_csv(os.path.join(table_path, 'memory_ablation_performance.csv'), index=False)

    # Filter out dcr and cmr for cub200, awa2, and cifar10 datasets
    # performance = performance[~((performance['dataset'].isin(['cub', 'awa2', 'cifar10'])) & (performance['model'].isin(['dcr', 'cmr'])))]

    # Separate datasets by task type
    classification_datasets = performance[performance['task_type'] == 'classification']['dataset'].unique()
    found_regression_datasets = performance[performance['task_type'] == 'regression']['dataset'].unique()

    # Organize datasets with MAE first, then 1-accuracy
    organized_datasets = list(found_regression_datasets) + list(classification_datasets)
    
    n_datasets = len(organized_datasets)
    n_cols = max(len(classification_datasets), len(found_regression_datasets))  
    n_rows = 2  # Force 2 rows: MAE on first row, 1-accuracy on second

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows), sharey=False)
    
    # Handle case where we have only one subplot
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # Validate and set defaults for ranges parameters
    n_classification = len(classification_datasets)
    n_regression = len(found_regression_datasets)
    # Classification ranges (bottom row)
    if ranges is None:
        ranges = [None] * n_classification
    else:
        if len(ranges) < n_classification:
            ranges = list(ranges) + [None] * (n_classification - len(ranges))
    # Regression ranges (top row)
    if regression_ranges is None:
        regression_ranges = [None] * n_regression
    else:
        if len(regression_ranges) < n_regression:
            regression_ranges = list(regression_ranges) + [None] * (n_regression - len(regression_ranges))
    
    # Track which model styles are actually plotted across all subplots
    all_plotted_styles = set()
    
    # Track which subplots need broken axes (to apply after tight_layout)
    broken_axes_specs = {}
    
    # Store plot data for broken axes reconstruction
    plot_data_for_broken_axes = {}
    
    # Store Pareto front data for each axis to draw dominated area after axis limits are finalized
    pareto_data_for_shading = {}

    # Defer uncertainty bands for cem/blackbox until after axis limits finalize
    cem_blackbox_bands = {}

    for idx, dataset in enumerate(organized_datasets):
        # Determine row based on task type
        if dataset in found_regression_datasets:
            row = 0
            col = list(found_regression_datasets).index(dataset)
        else:
            row = 1
            col = list(classification_datasets).index(dataset)
        
        if n_rows == 1:
            ax = axes[col] if n_cols > 1 else axes[0]
        else:
            ax = axes[row][col] if n_cols > 1 else axes[row][0] if col == 0 else axes[row][col]
        
        # Handle broken/stacked axes for classification (bottom) and regression (top) datasets
        use_broken_axis = False
        ax_bottom = None
        ax_top = None
        
        if row == 1:  # Classification dataset
            class_idx = list(classification_datasets).index(dataset)
            range_spec = ranges[class_idx]
            
            # Check if this is a broken axis specification (nested list with 2 elements)
            if (range_spec is not None and isinstance(range_spec, list) and len(range_spec) == 2 and 
                isinstance(range_spec[0], list) and isinstance(range_spec[1], list)):
                # Mark for broken axis creation after tight_layout
                use_broken_axis = True
                broken_axes_specs[(row, col)] = {
                    'ax': ax,
                    'y_low_range': range_spec[0],
                    'y_high_range': range_spec[1],
                    'dataset': dataset
                }
            elif range_spec is not None:
                # Regular single axis: set y-limits based on range_spec
                if isinstance(range_spec, list) and len(range_spec) == 2:
                    ax.set_ylim(range_spec)
            # If range_spec is None, let matplotlib auto-scale (do nothing)
        elif row == 0:  # Regression dataset
            reg_idx = list(found_regression_datasets).index(dataset)
            range_spec = regression_ranges[reg_idx]
            # Check if this is a broken axis specification (nested list with 2 elements)
            if (range_spec is not None and isinstance(range_spec, list) and len(range_spec) == 2 and 
                isinstance(range_spec[0], list) and isinstance(range_spec[1], list)):
                use_broken_axis = True
                broken_axes_specs[(row, col)] = {
                    'ax': ax,
                    'y_low_range': range_spec[0],
                    'y_high_range': range_spec[1],
                    'dataset': dataset
                }
            elif range_spec is not None:
                if isinstance(range_spec, list) and len(range_spec) == 2:
                    ax.set_ylim(range_spec)
            
        data = performance[performance['dataset'] == dataset]
        metric_type = data['metric_type'].iloc[0]
        
        # Debug: print models present for this dataset
        print(f"Dataset: {dataset}, Models: {data['model'].unique()}")
        
        # Compute complexity for each model
        data = data.copy()
        data['has_infinity'] = False  # Initialize the column
        
        # Note: mean_complexity is already set by compute_avg_and_uncertainty based on complexity_metric
        # No need to reselect the complexity column here
        
        # Mark dcr and licem as having infinity complexity
        data['has_infinity'] = data['model'].isin(['dcr', 'licem'])
        
        # # Mark memory_cbm as having infinity complexity for specific datasets
        # if dataset in ['awa2_incomplete', 'cub_incomplete', 'cifar10']:
        #     data.loc[data['model'] == 'memory_cbm', 'has_infinity'] = True
        
        # Find max finite complexity to set infinity value appropriately
        finite_complexities = data.loc[~data['has_infinity'], 'mean_complexity']
        finite_complexities = finite_complexities[finite_complexities.notna() & (finite_complexities > 0)]
        
        if len(finite_complexities) > 0:
            max_finite = finite_complexities.max()
            infinity_value = max_finite * 5  # Set infinity value higher than max
        else:
            # Fallback if no valid finite complexities exist
            infinity_value = 100000.0
        
        data.loc[data['has_infinity'], 'mean_complexity'] = infinity_value

        # Filter out rows with NaN or non-positive complexity values
        # BUT keep blackbox and cem models even if they have NaN complexity (they'll be plotted as horizontal lines)
        data = data[(data['mean_complexity'].notna() & (data['mean_complexity'] > 0)) | 
                    (data['model'].isin(['cem', 'blackbox']))]
        
        # Debug: check if data is empty after filtering
        if data.empty:
            print(f"WARNING: Dataset {dataset} has no valid complexity data after filtering!")
            continue
        
        # Get distinct complexities for this dataset (excluding cem and blackbox)
        pareto_data = data[~data['model'].isin(['cem', 'blackbox'])]
        distinct_complexities = sorted(pareto_data['mean_complexity'].unique())
        
        # Check if we have infinity models (dcr, licem, and memory_cbm for specific datasets) in the data
        infinity_models = ['dcr', 'licem']
        # if dataset in ['awa2_incomplete', 'cub_incomplete', 'cifar10']:
        #     infinity_models.append('memory_cbm')
        
        has_infinity_models = any(data['model'].isin(infinity_models))
        if has_infinity_models:
            infinity_complexity = data.loc[data['model'].isin(infinity_models), 'mean_complexity'].unique()
        else:
            infinity_complexity = np.array([])
        
        # Collect all points for Pareto front calculation
        all_points = []
        
        # Order models according to model_styles
        models_in_data = [m for m in model_styles.keys() if m in data['model'].unique()]
        for model in models_in_data:
            model_data = data[data['model'] == model]
            
            # For accuracy metrics, plot 1-accuracy (error rate)
            if metric_type == 'accuracy':
                y_values = 100 - model_data['mean_task']
                y_errors = model_data['se_task']  # Error bars remain the same
            else:
                y_values = model_data['mean_task']
                y_errors = model_data['se_task']
            
            if model in ['cem', 'blackbox']:
                # Track that this style was plotted
                all_plotted_styles.add(model)
                
            else:
                # Store points for Pareto front (excluding cem and blackbox)
                for i, (complexity, y_val, y_err, memory_size) in enumerate(zip(model_data['mean_complexity'], y_values, y_errors, model_data['memory_size'])):
                    all_points.append((complexity, y_val, y_err, model, memory_size))
                
                # Plot each point with appropriate style
                for i, (complexity, complexity_err, y_val, y_err, memory_size) in enumerate(zip(
                    model_data['mean_complexity'], 
                    model_data['se_complexity'],
                    y_values, 
                    y_errors, 
                    model_data['memory_size']
                )):
                    # Use cbm_linear style if linear_symbolic_cbm has memory_size = 1
                    if model == 'linear_symbolic_cbm' and memory_size == 1:
                        plot_style = model_styles.get('cbm_linear', model_styles[model])
                        plot_label = model_styles.get('cbm_linear', {})['name'] if 'cbm_linear' in model_styles else model_styles[model]['name']
                        style_key = 'cbm_linear'
                    else:
                        plot_style = model_styles[model].copy()
                        plot_label = model_styles[model]['name']
                        style_key = model
                    
                    # Override CMR style for cifar10 and cub datasets to grey hollow marker
                    marker_facecolor = plot_style['color']
                    marker_edge_width = 0.1
                    # if model == 'cmr' and dataset in ['cifar10', 'cub']:
                    #     plot_style['color'] = 'tab:grey'
                    #     marker_facecolor = 'none'
                    
                    # Override prior_symbolic_cbm style to white marker with thicker black border
                    if model == 'prior_symbolic_cbm':
                        marker_facecolor = 'white'
                        marker_edge_width = 1.5
                    
                    # Track that this style was plotted
                    all_plotted_styles.add(style_key)
                    
                    # Only add label for the first point of each unique style to avoid duplicate legends
                    label = plot_label if i == 0 else ""
                    
                    # Show uncertainty as error bars on both axes
                    ax.errorbar(
                        [complexity], 
                        [y_val],
                        xerr=[complexity_err],
                        yerr=[y_err],
                        label=label,
                        marker=plot_style['marker'], 
                        color=plot_style['color'], 
                        markersize=plot_style['size'],
                        markeredgecolor='black',
                        markeredgewidth=marker_edge_width,
                        markerfacecolor=marker_facecolor,
                        capsize=3,
                        capthick=1.5,
                        alpha=0.6,
                        linestyle='none'  # No line connecting dots
                    )
                    
                    # Store plot data for broken axes reconstruction
                    if use_broken_axis:
                        if (row, col) not in plot_data_for_broken_axes:
                            plot_data_for_broken_axes[(row, col)] = []
                        plot_data_for_broken_axes[(row, col)].append({
                            'type': 'errorbar',
                            'x': [complexity],
                            'y': [y_val],
                            'yerr': [y_err],
                            'label': label,
                            'marker': plot_style['marker'],
                            'color': plot_style['color'],
                            'markersize': plot_style['size'],
                            'markeredgecolor': 'black',
                            'markeredgewidth': 0.1,
                            'capsize': 3,
                            'capthick': 1.5,
                            'alpha': 0.8,
                            'linestyle': 'none'
                        })
        
        # Calculate and draw Pareto front (excluding cem, blackbox, and prior_symbolic_cbm)
        if all_points:
            # Sort points by complexity
            all_points.sort(key=lambda x: x[0])
            
            # Find Pareto optimal points
            pareto_points = []
            for point in all_points:
                complexity, y_val, y_err, model, memory_size = point
                is_pareto = True

                # Skip prior_symbolic_cbm from Pareto front
                if model == 'prior_symbolic_cbm':
                    continue
                
                # For each point, check if it's dominated by any other point
                for other_point in all_points:
                    other_complexity, other_y_val, other_y_err, other_model, other_memory_size = other_point
                    
                    # Skip prior_symbolic_cbm when checking dominance
                    if other_model == 'prior_symbolic_cbm':
                        continue
                    
                    # A point is dominated if another point has both:
                    # - Lower or equal complexity AND lower performance metric (better)
                    if (other_complexity <= complexity and 
                        other_y_val < y_val and 
                        (other_complexity < complexity or other_y_val < y_val)):
                        is_pareto = False
                        break
                
                if is_pareto:
                    pareto_points.append((complexity, y_val))
            
            # Sort Pareto points by complexity and draw connecting line
            if len(pareto_points) >= 1:
                pareto_points.sort(key=lambda x: x[0])
                pareto_x, pareto_y = zip(*pareto_points)

                # Draw the Pareto front line only if there are 2+ points
                if len(pareto_points) > 1:
                    ax.plot(pareto_x, pareto_y, 'gray', linestyle='--', alpha=0.7, linewidth=2, zorder=0)
                    
                    # Store for broken axes
                    if use_broken_axis and (row, col) not in plot_data_for_broken_axes:
                        plot_data_for_broken_axes[(row, col)] = []
                    if use_broken_axis:
                        plot_data_for_broken_axes[(row, col)].append({
                            'type': 'plot',
                            'x': pareto_x,
                            'y': pareto_y,
                            'color': 'gray',
                            'linestyle': '--',
                            'linewidth': 2,
                            'alpha': 0.7,
                            'label': '',
                            'zorder': 0
                        })

                # Store Pareto data for shading after axis limits are finalized
                pareto_data_for_shading[idx] = {
                    'ax': ax,
                    'pareto_x': pareto_x,
                    'pareto_y': pareto_y,
                    'use_broken_axis': use_broken_axis,
                    'row': row,
                    'col': col
                }
            
        # Set title on top axis if broken, otherwise on regular axis
        ax.set_title(get_df_name(dataset), fontdict=title_font)

        # Set xlabel on bottom axis (only for bottom row)
        if row == 1:
            ax.set_xlabel('Complexity', fontdict=label_font)
        
        # Set y-label only for the leftmost subplot in each row
        if col == 0:
            ylabel = 'MAE' if metric_type == 'mae' else 'Error Rate'
            ax.set_ylabel(ylabel, fontdict=label_font)
        
        # Set x-axis to log scale
        ax.set_xscale('log')
        
        # Set xlim to maximum complexity value (light padding)
        if distinct_complexities:
            max_complexity = max(distinct_complexities)
            if has_infinity_models and infinity_complexity.size > 0:
                max_complexity = max(max_complexity, infinity_complexity[0])
            min_complexity = min(distinct_complexities)
            ax.set_xlim(min_complexity / 1.2, max_complexity * 1.2)  # Add 20% padding on both sides
        
        # Plot horizontal lines for cem and blackbox after xlim is set
        for model in data['model'].unique():
            if model in ['cem', 'blackbox']:
                model_data = data[data['model'] == model]
                
                # For accuracy metrics, plot 1-accuracy (error rate)
                if metric_type == 'accuracy':
                    y_values = 100 - model_data['mean_task']
                    y_errors = model_data['se_task']
                else:
                    y_values = model_data['mean_task']
                    y_errors = model_data['se_task']
                
                y_val = y_values.iloc[0]
                y_err = y_errors.iloc[0]
                
                ax.axhline(y=y_val, color=model_styles[model]['color'], 
                            linestyle='-.', linewidth=2.5, alpha=0.8,
                            label=model_styles[model]['name'])
                
                # Defer uncertainty shading until xlim is finalized
                key = (row, col)
                if key not in cem_blackbox_bands:
                    cem_blackbox_bands[key] = []
                cem_blackbox_bands[key].append({
                    'ax': ax,
                    'y': y_val,
                    'yerr': y_err,
                    'color': model_styles[model]['color']
                })
                
                # Store for broken axes
                if use_broken_axis:
                    if (row, col) not in plot_data_for_broken_axes:
                        plot_data_for_broken_axes[(row, col)] = []
                    plot_data_for_broken_axes[(row, col)].append({
                        'type': 'axhline',
                        'y': y_val,
                        'yerr': y_err,
                        'color': model_styles[model]['color'],
                        'linestyle': '-.',
                        'linewidth': 2.5,
                        'alpha': 0.8,
                        'label': model_styles[model]['name']
                    })

        # Set x-axis ticks to 5 equally spaced values
        if distinct_complexities:
            # Separate finite and infinity complexities
            if has_infinity_models and infinity_complexity.size > 0:
                inf_value = infinity_complexity[0]
                finite_complexities = [c for c in distinct_complexities if c < inf_value]
            else:
                finite_complexities = distinct_complexities
            
            if finite_complexities:
                min_c = min(finite_complexities)
                max_c = max(finite_complexities)
                
                # Generate 5 equally spaced ticks in log space
                if has_infinity_models and infinity_complexity.size > 0:
                    # Generate 3 equally spaced finite ticks + infinity
                    log_min = np.log10(min_c)
                    log_max = np.log10(max_c)
                    log_spacing = (log_max - log_min) / 2  # 2 intervals for 3 ticks

                    # Generate 3 equally spaced ticks in log space
                    tick_values = [10 ** (log_min + i * log_spacing) for i in range(3)]
                    # Format labels with K notation for values >= 1000
                    tick_labels = []
                    for c in tick_values:
                        if c >= 1000:
                            tick_labels.append(f'{int(c/1000)}K')
                        else:
                            tick_labels.append(str(int(c)))

                    # Add infinity tick at the actual infinity value position
                    inf_tick_position = inf_value
                    tick_values.append(inf_tick_position)
                    tick_labels.append('$\infty$')

                    # Set xlim to show all ticks including infinity with margin on both sides
                    ax.set_xlim(min_c / 1.2, inf_tick_position * 1.2)
                else:
                    log_min = np.log10(min_c)
                    log_max = np.log10(max_c)
                    tick_values = np.logspace(log_min, log_max, 3)
                    # Format labels with K notation for values >= 1000
                    tick_labels = []
                    for c in tick_values:
                        if c >= 1000:
                            tick_labels.append(f'{int(c/1000)}K')
                        else:
                            tick_labels.append(str(int(c)))
                
                # Apply ticks to axis
                ax.set_xticks(tick_values)
                # Use FixedFormatter to ensure our custom labels are preserved
                from matplotlib.ticker import FixedFormatter
                ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
            elif has_infinity_models and infinity_complexity.size > 0:
                # Only infinity models exist (no finite complexities)
                # Just show the infinity value with infinity label
                inf_value = infinity_complexity[0]
                ax.set_xticks([inf_value])
                from matplotlib.ticker import FixedFormatter
                ax.xaxis.set_major_formatter(FixedFormatter(['$\infty$']))
                # Add margin on both sides around the infinity tick
                ax.set_xlim(inf_value * 0.8, inf_value * 1.2)
        
        # Apply tick params and grid
        ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
        ax.minorticks_off()
        ax.grid(True, alpha=0.3, zorder=0)

    # Draw dominated area (grey shading) after all axis limits are finalized
    for idx, pareto_info in pareto_data_for_shading.items():
        ax = pareto_info['ax']
        pareto_x = pareto_info['pareto_x']
        pareto_y = pareto_info['pareto_y']
        use_broken_axis = pareto_info['use_broken_axis']
        row = pareto_info['row']
        col = pareto_info['col']
        
        # Get final axis limits after all configurations
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Create extended Pareto front for shading
        # The dominated region extends from the Pareto front to the upper-right corner
        # Start from the leftmost Pareto point going up to ylim[1]
        # Then follow the Pareto front
        # Then extend right to xlim[1] and up to ylim[1]
        extended_x = [xlim[0], pareto_x[0]] + list(pareto_x) + [xlim[1], xlim[1], xlim[0]]
        extended_y = [ylim[1], ylim[1]]     + list(pareto_y) + [pareto_y[-1], ylim[1], ylim[1]]

        # Add shaded area above the Pareto front
        ax.fill(extended_x, extended_y, color='gray', alpha=0.1, zorder=0, 
                label='Dominated Region' if idx == 0 else "")
        
        # Store for broken axes reconstruction (include pareto points for recalculation)
        if use_broken_axis:
            if (row, col) not in plot_data_for_broken_axes:
                plot_data_for_broken_axes[(row, col)] = []
            plot_data_for_broken_axes[(row, col)].append({
                'type': 'fill',
                'x': extended_x,
                'y': extended_y,
                'pareto_x': pareto_x,
                'pareto_y': pareto_y,
                'color': 'gray',
                'alpha': 0.1,
                'zorder': 0,
                'label': 'Dominated Region' if idx == 0 else ""
            })

    # Now add uncertainty bands for cem/blackbox using finalized axis limits
    for (row, col), bands in cem_blackbox_bands.items():
        for band in bands:
            ax = band['ax']
            xlim = ax.get_xlim()
            ax.fill_between(
                xlim,
                band['y'] - band['yerr'],
                band['y'] + band['yerr'],
                color=band['color'],
                alpha=0.2
            )

    # Hide empty subplots and find the rightmost empty subplot for legend
    legend_ax = None
    
    # Hide unused subplots in first row (MAE datasets)
    for col in range(len(found_regression_datasets), n_cols):
        axes[0][col].set_visible(False)
        # Use the rightmost empty subplot in first row for legend if available
        if col == n_cols - 1:
            legend_ax = axes[0][col]
    
    # Hide unused subplots in second row (1-accuracy datasets)
    for col in range(len(classification_datasets), n_cols):
        axes[1][col].set_visible(False)
        # Use the rightmost empty subplot in second row for legend if first row doesn't have one
        if legend_ax is None and col == n_cols - 1:
            legend_ax = axes[1][col]

    # Create filtered styles based on what was actually plotted
    filtered_styles = {}
    for name in all_plotted_styles:
        if name in model_styles:
            filtered_styles[name] = model_styles[name]

    # Create custom legend handles ordered by models_order
    custom_handles = []
    for model in models_order:
        if model in filtered_styles:
            style = filtered_styles[model]
            if model in ['cem', 'blackbox']:
                # Use line handle for cem and blackbox
                custom_handles.append(plt.Line2D([0], [0], color=style['color'], linestyle=':', linewidth=2, label=style['name']))
            else:
                # Use marker handle for other models
                custom_handles.append(plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=(style['size'])+10, label=style['name'], markeredgewidth=0.5, markeredgecolor='black'))

    # Place legend in empty subplot if available, otherwise below plots
    if legend_ax is not None:
        legend_ax.set_visible(True)
        legend_ax.axis('off')  # Hide axes
        legend_ax.legend(handles=custom_handles, loc='center', ncol=1, fontsize=tick_font['size'], frameon=True)
    else:
        # Create a single legend below the plots with one model per column
        fig.legend(handles=custom_handles, loc='lower center', ncol=len(custom_handles), fontsize=tick_font['size'], frameon=True, bbox_to_anchor=(0.5, -0.15), columnspacing=1.0, handletextpad=0.5)

    plt.tight_layout()
    
    # Create broken axes after tight_layout to get correct positions
    for (row, col), spec in broken_axes_specs.items():
        original_ax = spec['ax']
        y_low_range = spec['y_low_range']
        y_high_range = spec['y_high_range']
        dataset = spec['dataset']
        
        # Get the final position after tight_layout
        pos = original_ax.get_position()
        
        # Create two stacked axes with precise dimensions
        # Bottom 60%, gap ~1.5%, top ~38.5% to sum to 100%
        height_ratio_bottom = 0.60
        height_ratio_top = 0.385
        gap_ratio = 0.015
        
        # Calculate absolute heights
        bottom_height = pos.height * height_ratio_bottom
        gap_height = pos.height * gap_ratio
        top_height = pos.height * height_ratio_top
        
        # Hide the original axis
        original_ax.set_visible(False)
        
        # Bottom axis
        ax_bottom = fig.add_axes([pos.x0, pos.y0, pos.width, bottom_height])
        ax_bottom.set_ylim(y_low_range)
        
        # Top axis
        ax_top = fig.add_axes([pos.x0, pos.y0 + bottom_height + gap_height, pos.width, top_height])
        ax_top.set_ylim(y_high_range)
        
        # Replay stored plot commands on both axes
        if (row, col) in plot_data_for_broken_axes:
            for plot_cmd in plot_data_for_broken_axes[(row, col)]:
                for target_ax in [ax_bottom, ax_top]:
                    if plot_cmd['type'] == 'errorbar':
                        target_ax.errorbar(
                            plot_cmd['x'],
                            plot_cmd['y'],
                            yerr=plot_cmd['yerr'],
                            label=plot_cmd['label'] if target_ax == ax_bottom else "",
                            marker=plot_cmd['marker'],
                            color=plot_cmd['color'],
                            markersize=plot_cmd['markersize'],
                            markeredgecolor=plot_cmd['markeredgecolor'],
                            markeredgewidth=plot_cmd['markeredgewidth'],
                            capsize=plot_cmd['capsize'],
                            capthick=plot_cmd['capthick'],
                            alpha=plot_cmd['alpha'],
                            linestyle=plot_cmd['linestyle']
                        )
                    elif plot_cmd['type'] == 'plot':
                        target_ax.plot(
                            plot_cmd['x'],
                            plot_cmd['y'],
                            color=plot_cmd['color'],
                            linestyle=plot_cmd['linestyle'],
                            linewidth=plot_cmd['linewidth'],
                            marker=plot_cmd.get('marker'),
                            markersize=plot_cmd.get('markersize'),
                            markeredgecolor=plot_cmd.get('markeredgecolor'),
                            markeredgewidth=plot_cmd.get('markeredgewidth'),
                            alpha=plot_cmd.get('alpha'),
                            label=plot_cmd['label'] if target_ax == ax_bottom else "",
                            zorder=plot_cmd.get('zorder', 0)
                        )
                    elif plot_cmd['type'] == 'axhline':
                        xlim = target_ax.get_xlim()
                        target_ax.axhline(
                            y=plot_cmd['y'],
                            color=plot_cmd['color'],
                            linestyle=plot_cmd['linestyle'],
                            linewidth=plot_cmd['linewidth'],
                            alpha=plot_cmd['alpha'],
                            label=plot_cmd['label'] if target_ax == ax_bottom else ""
                        )
                        target_ax.fill_between(
                            xlim,
                            plot_cmd['y'] - plot_cmd['yerr'],
                            plot_cmd['y'] + plot_cmd['yerr'],
                            color=plot_cmd['color'],
                            alpha=0.2
                        )
                    elif plot_cmd['type'] == 'fill':
                        # For fill type with Pareto front shading, recalculate to extend to this axis's limits
                        if 'pareto_x' in plot_cmd and 'pareto_y' in plot_cmd:
                            # Use stored Pareto points to create proper shading for this axis
                            ax_xlim = target_ax.get_xlim()
                            ax_ylim = target_ax.get_ylim()
                            pareto_x = plot_cmd['pareto_x']
                            pareto_y = plot_cmd['pareto_y']
                            ext_x = [ax_xlim[0], pareto_x[0]] + list(pareto_x) + [ax_xlim[1], ax_xlim[1], ax_xlim[0]]
                            ext_y = [ax_ylim[1], ax_ylim[1]]  + list(pareto_y) + [pareto_y[-1], ax_ylim[1], ax_ylim[1]]
                            target_ax.fill(
                                ext_x,
                                ext_y,
                                color=plot_cmd['color'],
                                alpha=plot_cmd['alpha'],
                                zorder=plot_cmd['zorder'],
                                label=plot_cmd['label'] if target_ax == ax_bottom else ""
                            )
                        else:
                            target_ax.fill(
                                plot_cmd['x'],
                                plot_cmd['y'],
                                color=plot_cmd['color'],
                                alpha=plot_cmd['alpha'],
                                zorder=plot_cmd['zorder'],
                                label=plot_cmd['label'] if target_ax == ax_bottom else ""
                            )
        
        # Configure x-axis: hide labels on top, show on bottom
        # IMPORTANT: Set xscale and xlim FIRST before setting ticks
        ax_bottom.set_xscale(original_ax.get_xscale())
        ax_bottom.set_xlim(original_ax.get_xlim())
        ax_top.set_xscale(original_ax.get_xscale())
        ax_top.set_xlim(original_ax.get_xlim())
        
        # Copy x-axis ticks to BOTH axes to ensure alignment
        tick_positions = original_ax.get_xticks()
        tick_labels = [t.get_text() for t in original_ax.get_xticklabels()]
        
        ax_bottom.set_xticks(tick_positions)
        ax_bottom.set_xticklabels(tick_labels)
        
        ax_top.set_xticks(tick_positions)
        ax_top.set_xticklabels([])  # Hide labels on top axis
        ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        # Add diagonal break markers
        d = 0.015
        kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        
        kwargs = dict(transform=ax_bottom.transAxes, color='k', clip_on=False, linewidth=1)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        
        # Copy styling
        for target_ax in [ax_bottom, ax_top]:
            target_ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
            target_ax.minorticks_off()
            target_ax.grid(True, alpha=0.3, zorder=0)
        
        # Set title on top axis
        ax_top.set_title(get_df_name(dataset), fontdict=title_font)
        
        # Set xlabel on bottom axis
        ax_bottom.set_xlabel(original_ax.get_xlabel(), fontdict=label_font)
        
        # Set ylabel centered on the full subplot (not just bottom axis)
        if col == 0:
            ylabel = original_ax.get_ylabel()
            # Set temporary ylabel to get proper x-position, then replace with fig.text
            ax_bottom.set_ylabel(ylabel, fontdict=label_font)
            # Get the label position
            ax_bottom.yaxis.label.set_visible(False)
            # Position ylabel at vertical center of full original subplot
            # Use labelpad to match the automatic positioning
            fig.text(pos.x0 - 0.025,  # Adjust x-position to match ylabel distance
                    pos.y0 + pos.height / 2,
                    ylabel, 
                    fontdict=label_font, 
                    rotation=90, 
                    va='center', 
                    ha='center')
        
        # Update axes array reference
        if n_cols > 1:
            axes[row][col] = ax_bottom
        else:
            axes[row][0] = ax_bottom
    
    # Save the figure with complexity metric in the filename
    save_path = os.path.join(result_figs, f'pareto_front_{complexity_metric}.pdf')
    plt.savefig(save_path)

#############################################
########## Latex Table Creation Utils #######
##############################################

def create_latex_tables_from_csv(csv_file_path, output_dir='tabs'):
    """
    Creates two LaTeX tables from a CSV file with performance metrics.
    
    Table 1: Task Performance (mean_task ± se_task)
    Table 2: Concept Performance (mean_concept ± se_concept)
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file containing the performance data
    output_dir : str
        Directory where the LaTeX tables will be saved
        
    Returns:
    --------
    tuple : (task_latex, concept_latex)
        Two strings containing the LaTeX table code
    """
    import pandas as pd
    import numpy as np
    import os
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get unique datasets and models, preserving order from custom_order and models_order
    all_datasets = df['dataset'].unique()
    datasets = [d for d in custom_order if d in all_datasets]
    
    all_models = df['model'].unique()
    models = [m for m in models_order if m in all_models]
    
    # Calculate se_task and se_concept if not already present
    # Note: Since we only have one row per model/dataset, se values should already be in the CSV
    # If std_task exists but se_task doesn't, we'd need num_seeds to calculate it
    
    # Create pivot tables for task and concept metrics
    # Task table: mean_task ± se_task
    task_table = []
    concept_table = []
    
    for model in models:
        # Get display name from model_styles
        model_display_name = model_styles.get(model, {}).get('name', model)
        task_row = [model_display_name]
        concept_row = [model_display_name]
        
        for dataset in datasets:
            # Filter data for this model and dataset
            data = df[(df['model'] == model) & (df['dataset'] == dataset)]
            
            if len(data) > 0:
                row = data.iloc[0]
                
                # Task performance
                mean_task = row['mean_task']
                se_task = row.get('se_task', np.nan)
                
                if pd.notna(mean_task):
                    if pd.notna(se_task):
                        task_cell = f"${mean_task:.2f}_{{\\tiny{{\\pm {se_task:.2f}}}}}$"
                    else:
                        task_cell = f"${mean_task:.2f}$"
                else:
                    task_cell = "-"
                
                task_row.append(task_cell)
                
                # Concept performance
                mean_concept = row.get('mean_concept', np.nan)
                se_concept = row.get('se_concept', np.nan)
                
                if pd.notna(mean_concept):
                    if pd.notna(se_concept):
                        concept_cell = f"${mean_concept:.2f}_{{\\tiny{{\\pm {se_concept:.2f}}}}}$"
                    else:
                        concept_cell = f"${mean_concept:.2f}$"
                else:
                    concept_cell = "-"
                
                concept_row.append(concept_cell)
            else:
                task_row.append("-")
                concept_row.append("-")
        
        task_table.append(task_row)
        concept_table.append(concept_row)
    
    # Create LaTeX table strings
    n_datasets = len(datasets)
    col_format = "l" + "c" * n_datasets
    
    # Format dataset names using get_df_name function
    dataset_names = [get_df_name(d) for d in datasets]
    
    # Task performance table
    task_latex = "\\begin{table}[h]\n"
    task_latex += "\\centering\n"
    task_latex += "\\caption{Predictive performance (MAE) of all models on regression tasks, reported as mean $\\pm$ 95\\% confidence interval.}\n"
    task_latex += "\\label{tab:task_performance}\n"
    task_latex += "\\resizebox{\\columnwidth}{!}{%\n"
    task_latex += f"\\begin{{tabular}}{{{col_format}}}\n"
    task_latex += "\\hline\n"
    # Be sure dataset_names do not contains nan, if so, replace with 'NaN'  
    dataset_names = [name if pd.notna(name) else 'NaN' for name in dataset_names]
    task_latex += "Model & " + " & ".join(dataset_names) + " \\\\\n"
    task_latex += "\\hline\n"
    
    for row in task_table:
        task_latex += " & ".join(row) + " \\\\\n"
    
    task_latex += "\\hline\n"
    task_latex += "\\end{tabular}\n"
    task_latex += "}\n"
    task_latex += "\\end{table}\n"
    
    # Concept performance table
    concept_latex = "\\begin{table}[h]\n"
    concept_latex += "\\centering\n"
    concept_latex += "\\caption{Concept Performance: $\\text{mean\\_concept} \\pm \\text{se\\_concept}$ (MAE for regression tasks)}\n"
    concept_latex += "\\label{tab:concept_performance}\n"
    concept_latex += "\\resizebox{\\columnwidth}{!}{%\n"
    concept_latex += f"\\begin{{tabular}}{{{col_format}}}\n"
    concept_latex += "\\hline\n"
    concept_latex += "Model & " + " & ".join(dataset_names) + " \\\\\n"
    concept_latex += "\\hline\n"
    
    for row in concept_table:
        concept_latex += " & ".join(row) + " \\\\\n"
    
    concept_latex += "\\hline\n"
    concept_latex += "\\end{tabular}\n"
    concept_latex += "}\n"
    concept_latex += "\\end{table}\n"
    
    # Save to files
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'task_performance_table.tex'), 'w') as f:
        f.write(task_latex)
    
    with open(os.path.join(output_dir, 'concept_performance_table.tex'), 'w') as f:
        f.write(concept_latex)



#######################################
##### Symbolic regression results #####
#######################################

def show_symbolic_regression_results(
    results_df,
    custom_order,
    table_path,):
    """
    Generate table with average and uncertainty for the Symbolic regression ablation.
    """

    table_dir = table_path + '/sr_ablation'
    os.makedirs(table_dir, exist_ok=True)

    performance = compute_avg_and_uncertainty(results_df, custom_order)

    # save the performance as csv
    performance.to_csv(f'{table_dir}/sr_ablation_performance.csv', index=False)

    # save the tex table
    create_latex_tables_from_csv(f'{table_dir}/sr_ablation_performance.csv', output_dir=table_dir)


def compute_ted_metrics(paths):
    """Compute Tree Edit Distance (TED) metrics for symbolic regression ablation experiments."""
    
    import dill
    import pickle
    try:
        from scipy.optimize import linear_sum_assignment
        use_scipy = True
    except Exception as e:
        print(f"Warning: Could not import scipy.optimize.linear_sum_assignment: {e}")
        print("Using greedy assignment algorithm instead")
        use_scipy = False
    from src.utils.ted import sympy_to_tree, ted_weighted, make_costs
    from sympy import sympify
    
    def greedy_assignment(cost_matrix):
        """Greedy assignment algorithm as fallback when scipy fails."""
        n_rows, n_cols = cost_matrix.shape
        row_ind = []
        col_ind = []
        available_cols = set(range(n_cols))
        
        # For each row, assign to the best available column
        for i in range(n_rows):
            if not available_cols:
                break
            best_col = min(available_cols, key=lambda j: cost_matrix[i, j])
            row_ind.append(i)
            col_ind.append(best_col)
            available_cols.remove(best_col)
        
        return np.array(row_ind), np.array(col_ind)
    
    # Collect all experiment paths
    exps_path = []
    for path in paths:
        experiment_dir = os.listdir(path)
        for exp in experiment_dir:
            exps_path += [os.path.join(path, exp, e) for e in os.listdir(os.path.join(path, exp)) if 'multirun' not in e]
    
    results = []
    
    # Group experiments by (dataset, seed) to find corresponding prior model
    exp_groups = {}
    for exp_path in exps_path:
        try:
            config_file = os.path.join(exp_path, '.hydra/config.yaml')
            if not os.path.exists(config_file):
                continue
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            dataset_name = config['dataset']['metadata']['name']
            model_name = config['model']['metadata']['name']
            seed = config['seed']
            memory_size = config.get('memory_size', None)
            
            key = (dataset_name, seed, memory_size)
            if key not in exp_groups:
                exp_groups[key] = {}
            exp_groups[key][model_name] = exp_path
        except Exception as e:
            continue
    
    # Process each experiment group
    for (dataset_name, seed, memory_size), models in tqdm(exp_groups.items(), desc="Processing experiment groups"):
        # Find the prior_symbolic_cbm model for this dataset/seed (ground truth)
        prior_model_path = models.get('prior_symbolic_cbm')
        
        if prior_model_path is None:
            print(f"Warning: No prior_symbolic_cbm found for dataset={dataset_name}, seed={seed}, memory_size={memory_size}")
            continue
        
        # Load true equations from prior model's predictions CSV
        prior_predictions_file = os.path.join(prior_model_path, 'logs/experiment_metrics/test_predictions_per_sample.csv')
        if not os.path.exists(prior_predictions_file):
            print(f"Warning: No predictions file found for prior model at {prior_model_path}")
            continue
        
        try:
            df_prior = pd.read_csv(prior_predictions_file)
            
            if 'equation' not in df_prior.columns:
                print(f"Warning: 'equation' column not found in prior predictions")
                continue
            
            # Get concept variable names
            prior_vars = [x.replace('c_pred_','') for x in df_prior.columns if 'c_pred' in x]
            
            # Get unique equations from prior model
            true_equation_strs = []
            for eq_str in df_prior['equation'].unique():
                if pd.notna(eq_str) and eq_str != '':
                    # Remove target prefix if present
                    if ':' in eq_str:
                        eq_str = eq_str.split(':')[1].strip()
                    true_equation_strs.append(eq_str)
            
            if not true_equation_strs:
                print(f"Warning: No equations found in prior model for dataset={dataset_name}, seed={seed}")
                continue
            
            # Convert true equations to trees
            weight_fn, rename_fn = make_costs()
            true_trees = []
            for eq_str in true_equation_strs:
                try:
                    equation = sympify(eq_str, locals={v: sympify(v) for v in prior_vars})
                    tree = sympy_to_tree(equation, canonicalize_commutative=True, enforce_mul_for_add_terms=True)
                    true_trees.append(tree)
                except Exception as e:
                    print(f"Error converting true equation '{eq_str}' to tree: {e}")
                    continue
            
            if not true_trees:
                print(f"Warning: Could not convert any true equations to trees for dataset={dataset_name}, seed={seed}")
                continue
            
        except Exception as e:
            print(f"Error loading prior equations from {prior_predictions_file}: {e}")
            continue
        
        # Now process models in MEMORY_MODELS_LIST for this dataset/seed
        for model_name, exp_path in models.items():
            # Only process models in the specified list
            if model_name not in MEMORY_MODELS_LIST:
                continue
            
            try:
                # Load learned equations from predictions CSV
                predictions_file = os.path.join(exp_path, 'logs/experiment_metrics/test_predictions_per_sample.csv')
                if not os.path.exists(predictions_file):
                    print(f"No predictions file found for {exp_path}")
                    continue
                
                df_pred = pd.read_csv(predictions_file)
                
                if 'equation' not in df_pred.columns:
                    print(f"Warning: 'equation' column not found in predictions for {model_name}")
                    continue
                
                # Get concept variable names
                vars = [x.replace('c_pred_','') for x in df_pred.columns if 'c_pred' in x]
                
                # Get unique learned equations
                learned_equation_strs = []
                for eq_str in df_pred['equation'].unique():
                    if pd.notna(eq_str) and eq_str != '':
                        # Remove target prefix if present
                        if ':' in eq_str:
                            eq_str = eq_str.split(':')[1].strip()
                        learned_equation_strs.append(eq_str)
                
                if not learned_equation_strs:
                    print(f"No learned equations found for {exp_path}")
                    continue
                
                # Convert learned equations to trees
                learned_trees = []
                for eq_str in learned_equation_strs:
                    try:
                        equation = sympify(eq_str, locals={v: sympify(v) for v in vars})
                        tree = sympy_to_tree(equation, canonicalize_commutative=True, enforce_mul_for_add_terms=True)
                        learned_trees.append(tree)
                    except Exception as e:
                        print(f"Error converting learned equation '{eq_str}' to tree: {e}")
                        continue
                
                if not learned_trees:
                    print(f"Could not convert any learned equations to trees for {exp_path}")
                    continue
                
                # Compute TED matrix: rows = learned equations, cols = true equations
                n_learned = len(learned_trees)
                n_true = len(true_trees)
                ted_matrix = np.zeros((n_learned, n_true))
                
                for i, learned_tree in enumerate(learned_trees):
                    for j, true_tree in enumerate(true_trees):
                        try:
                            ted = ted_weighted(learned_tree, true_tree, weight_fn, rename_fn)
                            ted_matrix[i, j] = ted
                        except Exception as e:
                            print(f"Error computing TED: {e}")
                            ted_matrix[i, j] = np.inf
                
                # Optimal assignment: minimize total TED
                # If dimensions don't match, pad with high cost
                if n_learned != n_true:
                    max_dim = max(n_learned, n_true)
                    padded_matrix = np.full((max_dim, max_dim), np.max(ted_matrix) * 10)
                    padded_matrix[:n_learned, :n_true] = ted_matrix
                    if use_scipy:
                        row_ind, col_ind = linear_sum_assignment(padded_matrix)
                    else:
                        row_ind, col_ind = greedy_assignment(padded_matrix)
                    # Filter out padded assignments
                    valid_mask = (row_ind < n_learned) & (col_ind < n_true)
                    row_ind = row_ind[valid_mask]
                    col_ind = col_ind[valid_mask]
                else:
                    if use_scipy:
                        row_ind, col_ind = linear_sum_assignment(ted_matrix)
                    else:
                        row_ind, col_ind = greedy_assignment(ted_matrix)
                
                # Compute average TED for the optimal assignment
                assigned_teds = [ted_matrix[i, j] for i, j in zip(row_ind, col_ind)]
                avg_ted = np.mean(assigned_teds) if assigned_teds else np.nan
                # compute standard error
                se_ted = 1.96 * np.std(assigned_teds) / np.sqrt(len(assigned_teds)) if assigned_teds else np.nan
                
                results.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'seed': seed,
                    'avg_ted': avg_ted,
                    'se_ted': se_ted,
                    'n_learned': n_learned,
                    'n_true': n_true,
                    'memory_size': memory_size
                })
                
            except Exception as e:
                print(f"Error processing experiment {exp_path}: {e}")
                continue
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Create equations dataframe for a randomly selected seed
    equations_data = []
    
    if len(results_df) > 0:
        # Get all available seeds
        available_seeds = results_df['seed'].unique()
        
        if len(available_seeds) > 0:
            # Randomly select one seed
            selected_seed = np.random.choice(available_seeds)
            print(f"\nSelected seed {selected_seed} for equation collection")
            
            # Collect equations for the selected seed
            for (dataset_name, seed, memory_size), models in exp_groups.items():
                if seed != selected_seed:
                    continue
                
                # Process models in MEMORY_MODELS_LIST
                for model_name, exp_path in models.items():
                    if model_name not in MEMORY_MODELS_LIST:
                        continue
                    
                    try:
                        # Load learned equations from predictions CSV
                        predictions_file = os.path.join(exp_path, 'logs/experiment_metrics/test_predictions_per_sample.csv')
                        if not os.path.exists(predictions_file):
                            continue
                        
                        df_pred = pd.read_csv(predictions_file)
                        
                        if 'equation' not in df_pred.columns:
                            continue
                        
                        # Get unique learned equations
                        for eq_str in df_pred['equation'].unique():
                            if pd.notna(eq_str) and eq_str != '':
                                # Remove target prefix if present
                                if ':' in eq_str:
                                    cleaned_eq = eq_str.split(':')[1].strip()
                                else:
                                    cleaned_eq = eq_str
                                
                                equations_data.append({
                                    'model': model_name,
                                    'dataset': dataset_name,
                                    'equation': cleaned_eq
                                })
                    
                    except Exception as e:
                        print(f"Error collecting equations from {exp_path}: {e}")
                        continue
    
    equations_df = pd.DataFrame(equations_data)
    
    return results_df, equations_df

def csv_to_table(
    path,
    df,
    dataset_col="dataset",
    model_col="model",
    mean_col="mean",
    std_col="std",
    custom_order=None,
    model_styles=None,
    float_fmt="{:.2f}",
    missing="--"
):
    """
    Store a string containing LaTeX tabular code.
    """

    # Filter only the datasetsin custom order and sort the df accordingly
    if custom_order is not None:
        df[dataset_col] = pd.Categorical(df[dataset_col], categories=custom_order, ordered=True)
        df = df.sort_values(by=[dataset_col, model_col])
    # Filter only the models to include
    models_to_include = list(model_styles.keys()) if model_styles is not None else None
    if models_to_include is not None:
        df = df[df[model_col].isin(models_to_include)]
    # Substitute the models' names according to model_styles[model]['name']
    if model_styles is not None:
        df[model_col] = df[model_col].apply(lambda x: model_styles[x]['name'] if x in model_styles else x)

    datasets = df[dataset_col].unique().dropna()
    datasets = [get_df_name(d) for d in datasets]
    models = sorted(df[model_col].unique())
    
    # substitue datasets' names to df
    df[dataset_col] = df[dataset_col].apply(get_df_name)

    col_format = "l" + "c" * len(datasets)

    lines = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_format + r"}")
    lines.append(r"\toprule")

    # Header (FIXED: Model & ...)
    header = "Model & " + " & ".join(datasets) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for model in models:
        row = [model.replace("_", r"\_")]

        for dataset in datasets:
            sub = df[
                (df[model_col] == model) &
                (df[dataset_col] == dataset)
            ]

            if len(sub) == 0:
                cell = missing
            else:
                mean = float_fmt.format(sub.iloc[0][mean_col])
                std = float_fmt.format(sub.iloc[0][std_col])
                cell = rf"${mean} \scriptscriptstyle{{\pm {std}}}$"

            row.append(cell)

        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")  # end resizebox

    lines.append(r"\end{table}")

    table = "\n".join(lines)
    with open(path, "w") as f:
        f.write(table)


def compute_pareto_frontier_count(df, performance_col, complexity_metrics, higher_is_better=True, dataset=None):
    """
    Compute how many times each model is on the Pareto frontier across different complexity metrics.
    
    A point is on the Pareto frontier for a given complexity metric if no other point
    dominates it (i.e., no other point is both better in performance AND less complex).
    
    For classification tasks, models with performance <= random classifier (1/num_classes)
    are excluded from the Pareto frontier computation.
    
    Args:
        df: DataFrame with model performance and complexity data
        performance_col: Name of the performance column (e.g., 'task_acc_mean', 'task_mae_mean')
        complexity_metrics: List of complexity metric names
        higher_is_better: True for accuracy (higher = better), False for MAE/MSE (lower = better)
        dataset: Dataset name (used to determine random classifier threshold for classification)
    
    Returns:
        dict: Mapping from row index to Pareto frontier count
    """
    pareto_counts = {idx: 0 for idx in df.index}
    
    # For classification datasets, determine the random classifier threshold
    random_threshold = None
    if higher_is_better and dataset is not None and dataset in NUMBER_OF_CLASSES_PER_DATASET:
        num_classes = NUMBER_OF_CLASSES_PER_DATASET[dataset]
        random_threshold = 1.0 / num_classes  # Random classifier accuracy
    
    for metric in complexity_metrics:
        complexity_col = f'complexity_{metric}_mean'
        
        if complexity_col not in df.columns or performance_col not in df.columns:
            continue
        
        # Get valid rows (non-NaN for both performance and complexity)
        valid_mask = df[performance_col].notna() & df[complexity_col].notna()
        
        # For classification, exclude models with performance <= random classifier
        if random_threshold is not None:
            valid_mask = valid_mask & (df[performance_col] > random_threshold)
        
        valid_indices = df[valid_mask].index.tolist()
        
        for idx in valid_indices:
            perf_i = df.loc[idx, performance_col]
            comp_i = df.loc[idx, complexity_col]
            
            # Check if any other point dominates this one
            is_dominated = False
            for other_idx in valid_indices:
                if other_idx == idx:
                    continue
                
                perf_other = df.loc[other_idx, performance_col]
                comp_other = df.loc[other_idx, complexity_col]
                
                # For domination:
                # - Other must be strictly better in at least one dimension
                # - Other must be at least as good in the other dimension
                if higher_is_better:
                    # Higher performance is better, lower complexity is better
                    # Other dominates if: (perf_other >= perf_i AND comp_other <= comp_i) 
                    #                     AND (perf_other > perf_i OR comp_other < comp_i)
                    at_least_as_good = (perf_other >= perf_i) and (comp_other <= comp_i)
                    strictly_better = (perf_other > perf_i) or (comp_other < comp_i)
                else:
                    # Lower performance is better (MAE), lower complexity is better
                    # Other dominates if: (perf_other <= perf_i AND comp_other <= comp_i)
                    #                     AND (perf_other < perf_i OR comp_other < comp_i)
                    at_least_as_good = (perf_other <= perf_i) and (comp_other <= comp_i)
                    strictly_better = (perf_other < perf_i) or (comp_other < comp_i)
                
                if at_least_as_good and strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_counts[idx] += 1
    
    return pareto_counts


def generate_performance_tables(
    performance,
    custom_order,
    model_styles,
    regression_datasets,
    output_path
):
    """
    Generate LaTeX tables for classification and regression datasets showing
    performance metrics and all complexity metrics.
    
    For models in MEMORY_MODELS_LIST, separate rows are created for each memory_size.
    
    Args:
        performance: DataFrame with performance data including complexity metrics
        custom_order: List of datasets in desired order
        model_styles: Dictionary with model styling information
        regression_datasets: List of regression dataset names
        output_path: Directory path to save the tables
    """
    # Filter and prepare performance data
    perf_filtered = performance[performance['dataset'].isin(custom_order)].copy()
    
    # Separate classification and regression datasets
    classification_datasets = [d for d in custom_order if d not in regression_datasets]
    
    # Complexity metrics to include
    complexity_metrics = COMPLEXITY_METRICS_LIST
    
    # Create table for classification datasets
    if len(classification_datasets) > 0:
        print("\nGenerating classification performance table...")
        cls_data = perf_filtered[perf_filtered['dataset'].isin(classification_datasets)].copy()

        if cls_data.empty:
            pass
        else:
            # For memory models, group by dataset, model, and memory_size
            # For other models, group by dataset and model only
            cls_data['grouping_key'] = cls_data.apply(
                lambda row: f"{row['model']}_{row['memory_size']}" if row['model'] in MEMORY_MODELS_LIST and 'memory_size' in cls_data.columns else row['model'],
                axis=1
            )
            
            # Aggregate by dataset and grouping_key
            group_cols = ['dataset', 'model', 'grouping_key']
            if 'memory_size' in cls_data.columns:
                group_cols.append('memory_size')
                
            cls_grouped = cls_data.groupby(group_cols).agg({
                'task_acc': ['mean', 'std'],
                **{f'complexity_{metric}': ['mean', 'std'] for metric in complexity_metrics}
            }).reset_index()
            
            # Flatten column names
            cls_grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                for col in cls_grouped.columns.values]
            
            # Create LaTeX table for classification
            _create_performance_latex_table(
                cls_grouped,
                classification_datasets,
                model_styles,
                complexity_metrics,
                task_type='classification',
                output_path=output_path
            )
    
    # Create table for regression datasets
    if len(regression_datasets) > 0:
        print("\nGenerating regression performance table...")
        reg_data = perf_filtered[perf_filtered['dataset'].isin(regression_datasets)].copy()
        
        if reg_data.empty:
            pass
        else:
            # For memory models, group by dataset, model, and memory_size
            # For other models, group by dataset and model only
            reg_data['grouping_key'] = reg_data.apply(
                lambda row: f"{row['model']}_{row['memory_size']}" if row['model'] in MEMORY_MODELS_LIST and 'memory_size' in reg_data.columns else row['model'],
                axis=1
            )
            
            # Aggregate by dataset and grouping_key
            group_cols = ['dataset', 'model', 'grouping_key']
            if 'memory_size' in reg_data.columns:
                group_cols.append('memory_size')
                
            reg_grouped = reg_data.groupby(group_cols).agg({
                'task_mae': ['mean', 'std'],
                'task_mse': ['mean', 'std'],
                **{f'complexity_{metric}': ['mean', 'std'] for metric in complexity_metrics}
            }).reset_index()
            
            # Flatten column names
            reg_grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                for col in reg_grouped.columns.values]
            
            # Create LaTeX table for regression
            _create_performance_latex_table(
                reg_grouped,
                [d for d in custom_order if d in regression_datasets],
                model_styles,
                complexity_metrics,
                task_type='regression',
                output_path=output_path
            )


def _create_performance_latex_table(
    df,
    datasets,
    model_styles,
    complexity_metrics,
    task_type='classification',
    output_path='results/tabs'
):
    """
    Helper function to create LaTeX table with performance and complexity metrics.
    
    Table structure:
    - Dataset column (with row spanning to avoid repetition)
    - Model column (includes memory size for memory models)
    - Performance metric column (Accuracy for classification, MAE/MSE for regression)
    - Complexity metric columns (one per metric)
    - Pareto Frontier column (count of how many complexity metrics the model is on the frontier)
    
    Args:
        df: DataFrame with aggregated performance data
        datasets: List of dataset names to include
        model_styles: Dictionary with model styling information
        complexity_metrics: List of complexity metric names
        task_type: 'classification' or 'regression'
        output_path: Directory path to save the table
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Filter datasets
    df = df[df['dataset'].isin(datasets)].copy()
    
    # Create display name for models (with memory size if applicable)
    def get_model_display_name(row):
        base_name = model_styles[row['model']]['name'] if row['model'] in model_styles else row['model']
        if row['model'] in MEMORY_MODELS_LIST and 'memory_size' in row and pd.notna(row['memory_size']):
            return f"{base_name} ({int(row['memory_size'])})"
        return base_name
    
    df['model_display_name'] = df.apply(get_model_display_name, axis=1)
    
    # Map dataset names
    df['dataset_name'] = df['dataset'].apply(get_df_name)
    
    # Create model order index for sorting (respect models_order)
    model_order_map = {model: idx for idx, model in enumerate(models_order)}
    df['model_order'] = df['model'].apply(lambda x: model_order_map.get(x, len(models_order)))
    
    # Sort by dataset, model order, and memory_size
    if 'memory_size' in df.columns:
        df = df.sort_values(['dataset', 'model_order', 'memory_size'])
    else:
        df = df.sort_values(['dataset', 'model_order'])
    
    # Get unique dataset names in order
    dataset_names = [get_df_name(d) for d in datasets if d in df['dataset'].unique()]
    
    # Complexity metric names mapping
    complexity_metric_names = {
        'node_count': 'Nodes',
        'depth': 'Depth',
        'visitation_length': 'Expr.-Comp.',
        'total_variables': 'Vars',
        'total_operations': 'Ops',
        'weighted_node_count': 'Weighted'
    }
    
    # Column format: Dataset + Model + Performance metric(s) + Complexity metrics + Pareto
    if task_type == 'classification':
        n_perf_cols = 1  # Just Accuracy
    else:  # regression
        n_perf_cols = 2  # MAE and MSE
    
    n_complexity_cols = len(complexity_metrics)
    total_metric_cols = n_perf_cols + n_complexity_cols + 1  # +1 for Pareto column
    col_format = "ll" + "c" * total_metric_cols  # Dataset + Model + metrics + Pareto
    
    # Determine performance column for Pareto computation
    if task_type == 'classification':
        perf_col = 'task_acc_mean'
        higher_is_better = True
    else:
        perf_col = 'task_mae_mean'
        higher_is_better = False
    
    # Define dataset groups
    if task_type == 'classification':
        dataset_groups = [
            (['awa2', 'awa2_incomplete'], 'awa2'),
            (['cub', 'cub_incomplete'], 'cub'),
            (['cifar10'], 'cifar10'),
        ]
    else:  # regression
        dataset_groups = [
            (['feynman_I_6_2', 'feynman_I_9_18', 'feynman_I_12_1', 'feynman_I_13_4', 'feynman_I_14_3', 'feynman_I_15_10'], 'feynman'),
            (['dsprites_simple', 'pendulum'], 'dsprites_pendulum'),
            (['mnist_arithmetic', 'mawps'], 'mnist_mawps'),
        ]
    
    # Build a table for each dataset group
    for group_datasets, group_name in dataset_groups:
        # Filter to datasets in this group that exist in our data
        group_datasets_present = [d for d in group_datasets if d in df['dataset'].unique()]
        
        if len(group_datasets_present) == 0:
            continue
        
        group_df = df[df['dataset'].isin(group_datasets_present)].copy()
        
        # Get dataset display names in order
        group_dataset_names = [get_df_name(d) for d in group_datasets if d in group_datasets_present]
        
        # Generate caption based on group
        if len(group_datasets_present) == 1:
            caption_datasets = get_df_name(group_datasets_present[0])
        else:
            caption_datasets = ", ".join([get_df_name(d) for d in group_datasets_present[:-1]]) + " and " + get_df_name(group_datasets_present[-1])
        
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Predictive performance and model complexity for all methods on the " + caption_datasets.replace("_", r"\_") + r" dataset" + ("s" if len(group_datasets_present) > 1 else "") + r".}")
        lines.append(r"\label{tab:" + task_type + "_" + group_name + r"}")
        lines.append(r"\resizebox{\columnwidth}{!}{%")
        lines.append(r"\begin{tabular}{" + col_format + r"}")
        lines.append(r"\toprule")
        
        # Header row
        header_parts = ["Dataset", "Model"]
        if task_type == 'classification':
            header_parts.append("Accuracy")
        else:  # regression
            header_parts.extend(["MAE", "MSE"])
        
        for metric in complexity_metrics:
            header_parts.append(complexity_metric_names.get(metric, metric))
        
        # Add Pareto column header
        header_parts.append("Pareto")
        
        lines.append(" & ".join(header_parts) + r" \\")
        lines.append(r"\midrule")
        
        # Data rows - grouped by dataset
        for dataset in group_dataset_names:
            dataset_key = [d for d in group_datasets_present if get_df_name(d) == dataset][0]
            dataset_df = group_df[group_df['dataset'] == dataset_key].copy()
            
            if len(dataset_df) == 0:
                continue
            
            # Compute Pareto frontier counts for this dataset
            pareto_counts = compute_pareto_frontier_count(
                dataset_df, 
                perf_col, 
                complexity_metrics, 
                higher_is_better,
                dataset=dataset_key  # Pass dataset for random classifier threshold
            )
            
            # Get all model rows for this dataset
            model_rows = []
            for idx, row_data in dataset_df.iterrows():
                model_display = row_data['model_display_name'].replace("_", r"\_")
                row_parts = [model_display]
                
                # Add performance metric(s)
                if task_type == 'classification':
                    # Accuracy
                    if 'task_acc_mean' in row_data and pd.notna(row_data['task_acc_mean']):
                        acc_mean = row_data['task_acc_mean'] * 100  # Convert to percentage
                        acc_std = row_data['task_acc_std'] * 100
                        row_parts.append(rf"${acc_mean:.1f}_{{\tiny{{\pm {acc_std:.1f}}}}}$")
                    else:
                        row_parts.append("--")
                else:  # regression
                    # MAE
                    if 'task_mae_mean' in row_data and pd.notna(row_data['task_mae_mean']):
                        mae_mean = row_data['task_mae_mean']
                        mae_std = row_data['task_mae_std']
                        row_parts.append(rf"${mae_mean:.3f}_{{\tiny{{\pm {mae_std:.3f}}}}}$")
                    else:
                        row_parts.append("--")
                    # MSE
                    if 'task_mse_mean' in row_data and pd.notna(row_data['task_mse_mean']):
                        mse_mean = row_data['task_mse_mean']
                        mse_std = row_data['task_mse_std']
                        row_parts.append(rf"${mse_mean:.3f}_{{\tiny{{\pm {mse_std:.3f}}}}}$")
                    else:
                        row_parts.append("--")
                
                # Add complexity metrics
                for metric in complexity_metrics:
                    mean_col = f'complexity_{metric}_mean'
                    std_col = f'complexity_{metric}_std'
                    if mean_col in row_data and pd.notna(row_data[mean_col]):
                        c_mean = row_data[mean_col]
                        c_std = row_data[std_col]
                        row_parts.append(rf"${c_mean:.1f}_{{\tiny{{\pm {c_std:.1f}}}}}$")
                    else:
                        row_parts.append("--")
                
                # Add Pareto frontier count
                pareto_count = pareto_counts.get(idx, 0)
                row_parts.append(f"${pareto_count}/{len(complexity_metrics)}$")
                
                model_rows.append(row_parts)
            
            # Write rows with dataset name only on first row
            if len(model_rows) > 0:
                dataset_display = dataset.replace("_", r"\_")
                # First row with dataset name
                first_row = [dataset_display] + model_rows[0]
                lines.append(" & ".join(first_row) + r" \\")
                
                # Remaining rows without dataset name
                for model_row in model_rows[1:]:
                    row_line = " & " + " & ".join(model_row) + r" \\"
                    lines.append(row_line)
                
                # Add separator between datasets within a group
                lines.append(r"\midrule")
        
        # Remove last midrule and add bottomrule
        if lines[-1] == r"\midrule":
            lines[-1] = r"\bottomrule"
        else:
            lines.append(r"\bottomrule")
        
        lines.append(r"\end{tabular}")
        lines.append(r"}")
        lines.append(r"\end{table}")
        
        # Save table for this dataset group
        table_content = "\n".join(lines)
        filename = os.path.join(output_path, f'{task_type}_{group_name}_performance_table.txt')
        with open(filename, 'w') as f:
            f.write(table_content)
        
        print(f"Table saved to: {filename}")


def generate_concept_metrics_table(
    performance,
    custom_order,
    model_styles,
    regression_datasets,
    output_path
):
    """
    Generate a LaTeX table showing concept-level metrics for all datasets.
    - For classification datasets: concept accuracy
    - For regression datasets: concept MAE
    
    Table structure:
    - Columns: dataset1, dataset2, ..., datasetk
    - Each column has a subcolumn showing the metric name (accuracy or MAE)
    - Rows: models
    - Each cell: mean ± std
    
    Args:
        performance: DataFrame with performance data including concept metrics
        custom_order: List of datasets in desired order
        model_styles: Dictionary with model styling information
        regression_datasets: List of regression dataset names
        output_path: Directory path to save the table
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Filter datasets that are in custom_order
    perf_filtered = performance[performance['dataset'].isin(custom_order)].copy()
    
    # Separate classification and regression datasets
    classification_datasets = [d for d in custom_order if d not in regression_datasets and d in perf_filtered['dataset'].unique()]
    regression_datasets_filtered = [d for d in custom_order if d in regression_datasets and d in perf_filtered['dataset'].unique()]
    
    # Combine in order
    all_datasets = classification_datasets + regression_datasets_filtered
    
    # Aggregate concept metrics by dataset and model
    agg_dict = {}
    
    # For classification datasets, use concept_acc
    for dataset in classification_datasets:
        dataset_data = perf_filtered[perf_filtered['dataset'] == dataset]
        if 'concept_acc' in dataset_data.columns:
            grouped = dataset_data.groupby('model')['concept_acc'].agg(['mean', 'std', 'count']).reset_index()
            # Compute 95% confidence interval: CI = 1.96 * std / sqrt(n)
            grouped['ci95'] = 1.96 * grouped['std'] / np.sqrt(grouped['count'])
            grouped['dataset'] = dataset
            grouped['metric_type'] = 'Accuracy'
            # Map model names
            if model_styles is not None:
                grouped['model_name'] = grouped['model'].apply(
                    lambda x: model_styles[x]['name'] if x in model_styles else x
                )
            else:
                grouped['model_name'] = grouped['model']
            grouped = grouped[['model', 'model_name', 'mean', 'ci95', 'dataset', 'metric_type']]
            agg_dict[dataset] = grouped
    
    # For regression datasets, use concept_mae
    for dataset in regression_datasets_filtered:
        dataset_data = perf_filtered[perf_filtered['dataset'] == dataset]
        if 'concept_mae' in dataset_data.columns:
            grouped = dataset_data.groupby('model')['concept_mae'].agg(['mean', 'std', 'count']).reset_index()
            # Compute 95% confidence interval: CI = 1.96 * std / sqrt(n)
            grouped['ci95'] = 1.96 * grouped['std'] / np.sqrt(grouped['count'])
            grouped['dataset'] = dataset
            grouped['metric_type'] = 'MAE'
            # Map model names
            if model_styles is not None:
                grouped['model_name'] = grouped['model'].apply(
                    lambda x: model_styles[x]['name'] if x in model_styles else x
                )
            else:
                grouped['model_name'] = grouped['model']
            grouped = grouped[['model', 'model_name', 'mean', 'ci95', 'dataset', 'metric_type']]
            agg_dict[dataset] = grouped
    
    if not agg_dict:
        print("No concept metrics found in the performance data.")
        return
    
    # Filter models in model_styles before aggregating
    if model_styles is not None:
        for dataset in agg_dict:
            agg_dict[dataset] = agg_dict[dataset][agg_dict[dataset]['model'].isin(model_styles.keys())]
    
    # Exclude blackbox model
    for dataset in agg_dict:
        agg_dict[dataset] = agg_dict[dataset][agg_dict[dataset]['model'] != 'blackbox']
    
    # Combine all datasets
    concept_df = pd.concat(agg_dict.values(), ignore_index=True)
    
    # Get unique models in the order specified by models_order
    all_models = concept_df['model'].unique()
    ordered_models = [m for m in models_order if m in all_models]
    # Get display names for ordered models
    models = [model_styles[m]['name'] if m in model_styles else m for m in ordered_models]
    
    # Build LaTeX table
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Concept accuracy across methods and datasets.}")
    lines.append(r"\label{tab:concept_metrics}")
    
    # Column format: Model + 1 column per dataset
    n_datasets = len(all_datasets)
    col_format = "l" + "c" * n_datasets
    
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_format + r"}")
    lines.append(r"\toprule")
    
    # Create multi-row header
    # First row: dataset names
    header_row1 = ["Model"]
    for dataset in all_datasets:
        dataset_name = get_df_name(dataset)
        header_row1.append(dataset_name)
    lines.append(" & ".join(header_row1) + r" \\")
    
    # Second row: metric type for each dataset
    header_row2 = [""]
    for dataset in all_datasets:
        metric_type = agg_dict[dataset].iloc[0]['metric_type'] if dataset in agg_dict else "N/A"
        header_row2.append(f"({metric_type})")
    lines.append(" & ".join(header_row2) + r" \\")
    lines.append(r"\midrule")
    
    # Data rows
    for model in models:
        row = [model.replace("_", r"\_")]
        
        for dataset in all_datasets:
            if dataset not in agg_dict:
                row.append("--")
                continue
            
            dataset_df = agg_dict[dataset]
            model_data = dataset_df[dataset_df['model_name'] == model]
            
            if len(model_data) == 0:
                row.append("--")
                continue
            
            metric_type = model_data.iloc[0]['metric_type']
            mean_val = model_data.iloc[0]['mean']
            ci95_val = model_data.iloc[0]['ci95']
            
            # Format based on metric type
            if metric_type == 'Accuracy':
                # Convert to percentage
                mean_val *= 100
                ci95_val *= 100
                mean_str = f"{mean_val:.2f}"
                ci95_str = f"{ci95_val:.2f}"
            else:  # MAE
                mean_str = f"{mean_val:.4f}"
                ci95_str = f"{ci95_val:.4f}"
            
            cell_value = rf"${mean_str} \scriptstyle{{\pm {ci95_str}}}$"
            
            row.append(cell_value)
        
        lines.append(" & ".join(row) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    
    table_content = "\n".join(lines)
    filename = os.path.join(output_path, 'concept_metrics_table.txt')
    with open(filename, 'w') as f:
        f.write(table_content)
    
    print(f"Concept metrics table saved to: {filename}")
    
    # Also save as CSV for reference
    csv_filename = os.path.join(output_path, 'concept_metrics_table.csv')
    concept_df.to_csv(csv_filename, index=False)
    print(f"Concept metrics CSV saved to: {csv_filename}")


def generate_ted_metrics_table(
    ted_results,
    custom_order,
    model_styles,
    output_path
):
    """
    Generate a LaTeX table showing Tree Edit Distance (TED) metrics for all datasets and models.
    
    Table structure:
    - Rows: Models
    - Columns: Datasets
    - Each cell: mean TED ± 95% CI
    
    Args:
        ted_results: DataFrame with TED metrics (dataset, model, seed, avg_ted, se_ted)
        custom_order: List of datasets in desired order
        model_styles: Dictionary with model styling information
        output_path: Directory path to save the table
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Filter datasets that are in custom_order
    ted_filtered = ted_results[ted_results['dataset'].isin(custom_order)].copy()
    
    if ted_filtered.empty:
        print("No TED metrics found for the specified datasets.")
        return
    
    # Get ordered datasets present in the data
    datasets = [d for d in custom_order if d in ted_filtered['dataset'].unique()]
    
    # Get ordered models present in the data
    all_models = ted_filtered['model'].unique()
    models = [m for m in models_order if m in all_models]
    
    # Aggregate TED metrics by dataset and model
    agg_df = ted_filtered.groupby(['dataset', 'model'])['avg_ted'].agg(['mean', 'std', 'count']).reset_index()
    agg_df['ci95'] = 1.96 * agg_df['std'] / np.sqrt(agg_df['count'])
    
    # Build LaTeX table
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{TED between learned and ground truth expressions, reported as mean $\pm$ 95\% confidence interval.}")
    lines.append(r"\label{tab:ted_metrics}")
    
    # Column format: Model + 1 column per dataset
    n_datasets = len(datasets)
    col_format = "l" + "c" * n_datasets
    
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_format + r"}")
    lines.append(r"\toprule")
    
    # Header row: dataset names
    header_row = ["Model"]
    for dataset in datasets:
        dataset_name = get_df_name(dataset)
        header_row.append(dataset_name)
    lines.append(" & ".join(header_row) + r" \\")
    lines.append(r"\midrule")
    
    # Data rows
    for model in models:
        model_display = model_styles[model]['name'] if model in model_styles else model
        row = [model_display.replace("_", r"\_")]
        
        for dataset in datasets:
            model_data = agg_df[(agg_df['dataset'] == dataset) & (agg_df['model'] == model)]
            
            if len(model_data) == 0:
                row.append("--")
                continue
            
            mean_val = model_data.iloc[0]['mean']
            ci95_val = model_data.iloc[0]['ci95']
            
            if pd.isna(mean_val):
                row.append("--")
            else:
                cell_value = rf"${mean_val:.2f}_{{\tiny{{\pm {ci95_val:.2f}}}}}$"
                row.append(cell_value)
        
        lines.append(" & ".join(row) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    
    table_content = "\n".join(lines)
    filename = os.path.join(output_path, 'ted_metrics_table.tex')
    with open(filename, 'w') as f:
        f.write(table_content)
    
    print(f"TED metrics table saved to: {filename}")


def generate_complexity_metrics_table(
    complexity_results,
    custom_order,
    model_styles,
    output_path,
    metric='node_count'
):
    """
    Generate a LaTeX table showing equation complexity metrics for all datasets and models.
    
    Table structure:
    - Rows: Models
    - Columns: Datasets
    - Each cell: mean complexity ± 95% CI
    
    Args:
        complexity_results: DataFrame with complexity metrics
        custom_order: List of datasets in desired order
        model_styles: Dictionary with model styling information
        output_path: Directory path to save the table
        metric: Which complexity metric to use (default: 'node_count')
    """
    os.makedirs(output_path, exist_ok=True)
    
    col_name = f'complexity_{metric}'
    
    # Filter datasets that are in custom_order
    comp_filtered = complexity_results[complexity_results['dataset'].isin(custom_order)].copy()
    
    if comp_filtered.empty or col_name not in comp_filtered.columns:
        print(f"No complexity metrics ({metric}) found for the specified datasets.")
        return
    
    # Get ordered datasets present in the data
    datasets = [d for d in custom_order if d in comp_filtered['dataset'].unique()]
    
    # Get ordered models present in the data - only include models that have complexity values
    comp_with_values = comp_filtered[comp_filtered[col_name].notna()]
    all_models = comp_with_values['model'].unique()
    models = [m for m in models_order if m in all_models]
    
    # Aggregate complexity metrics by dataset and model
    agg_df = comp_filtered.groupby(['dataset', 'model'])[col_name].agg(['mean', 'std', 'count']).reset_index()
    agg_df['ci95'] = 1.96 * agg_df['std'] / np.sqrt(agg_df['count'])
    
    # Metric display names
    metric_names = {
        'node_count': 'Node Count',
        'depth': 'Tree Depth',
        'visitation_length': 'Expression Complexity',
        'total_variables': 'Total Variables',
        'total_operations': 'Total Operations',
        'weighted_node_count': 'Weighted Node Count'
    }
    metric_display = metric_names.get(metric, metric)
    
    # Build LaTeX table
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Complexity (" + metric_display + r") of learned expressions, reported as mean $\pm$ 95\% confidence interval.}")
    lines.append(r"\label{tab:complexity_" + metric + r"}")
    
    # Column format: Model + 1 column per dataset
    n_datasets = len(datasets)
    col_format = "l" + "c" * n_datasets
    
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_format + r"}")
    lines.append(r"\toprule")
    
    # Header row: dataset names
    header_row = ["Model"]
    for dataset in datasets:
        dataset_name = get_df_name(dataset)
        header_row.append(dataset_name)
    lines.append(" & ".join(header_row) + r" \\")
    lines.append(r"\midrule")
    
    # Data rows
    for model in models:
        model_display = model_styles[model]['name'] if model in model_styles else model
        row = [model_display.replace("_", r"\_")]
        
        for dataset in datasets:
            model_data = agg_df[(agg_df['dataset'] == dataset) & (agg_df['model'] == model)]
            
            if len(model_data) == 0:
                row.append("--")
                continue
            
            mean_val = model_data.iloc[0]['mean']
            ci95_val = model_data.iloc[0]['ci95']
            
            if pd.isna(mean_val):
                row.append("--")
            else:
                cell_value = rf"${mean_val:.1f}_{{\tiny{{\pm {ci95_val:.1f}}}}}$"
                row.append(cell_value)
        
        lines.append(" & ".join(row) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    
    table_content = "\n".join(lines)
    filename = os.path.join(output_path, f'complexity_{metric}_table.tex')
    with open(filename, 'w') as f:
        f.write(table_content)
    
    print(f"Complexity ({metric}) table saved to: {filename}")


def global_verifiability_plot(paths, dataset='awa2_incomplete', memory_size=4, label_font=None, tick_font=None, legend_font=None):
    """
    Visualize the distribution of weights for specific concept-class pairs across 
    licem (continuous) and linear_symbolic_cbm (discrete) models.
    Generates 50 plots for randomly selected concept-class combinations.
    
    Args:
        paths: List of experiment paths to search
        dataset: Dataset name to filter experiments (default: 'awa2_incomplete')
        memory_size: Memory size for linear_symbolic_cbm model (default: 4)
        label_font: Dictionary with font properties for axis labels
        tick_font: Dictionary with font properties for tick labels
        legend_font: Dictionary with font properties for legend
    """
    # Set default font sizes if not provided
    if label_font is None:
        label_font = {'size': 14}
    if tick_font is None:
        tick_font = {'size': 12}
    if legend_font is None:
        legend_font = {'size': 12}
    import re
    from collections import defaultdict
    from scipy.ndimage import gaussian_filter1d
    
    # First pass: collect all data organized by concept, class, and model
    data_by_concept_class = defaultdict(lambda: {'licem': [], 'lin_sym': []})
    all_concepts = set()
    all_classes = set()
    
    # Track which models we've found
    found_licem = False
    found_lin_sym = False
    
    for path in paths:
        if not os.path.exists(path):
            continue
            
        experiment_dirs = os.listdir(path)
        for exp_dir in experiment_dirs:
            exp_base_path = os.path.join(path, exp_dir)
            if not os.path.isdir(exp_base_path):
                continue
                
            # List experiments in this directory
            experiments = [e for e in os.listdir(exp_base_path) if 'multirun' not in e]
            
            for exp in tqdm(experiments):
                exp_path = os.path.join(exp_base_path, exp)
                
                # Check if this is the specified dataset
                if dataset not in exp:
                    continue       

                if 'seed_1' not in exp:
                    continue         
                
                # Determine model type
                is_licem = 'licem' in exp
                is_lin_sym = 'lin_sym_cbm' in exp and f'memory_size_{memory_size}' in exp
                
                if not (is_licem or is_lin_sym):
                    continue
                
                # Skip if we already processed this model type
                if is_licem and found_licem:
                    continue
                if is_lin_sym and found_lin_sym:
                    continue
                
                # Load predictions file
                predictions_file = os.path.join(exp_path, 'logs/experiment_metrics/test_predictions_per_sample.csv')
                if not os.path.exists(predictions_file):
                    continue
                
                try:
                    df = pd.read_csv(predictions_file)
                    
                    if 'equation' not in df.columns or 'y_pred_task_name' not in df.columns:
                        continue
                    
                    # Get concept names
                    concept_cols = [col.replace('c_pred_', '') for col in df.columns if 'c_pred_' in col]
                    
                    if not concept_cols:
                        continue
                    
                    all_concepts.update(concept_cols)
                    
                    # Process each row (sample)
                    for _, row in df.iterrows():
                        eq_str = row['equation']
                        pred_class = row['y_pred_task_name']
                        
                        if pd.isna(eq_str) or eq_str == '' or pd.isna(pred_class):
                            continue
                        
                        all_classes.add(pred_class)
                        
                        # Extract weights for each concept
                        for concept in concept_cols:
                            # Pattern: (optional sign)(coefficient)*concept_name
                            pattern = rf'([+-]?\s*\d+\.?\d*)\s*\*\s*{concept}'
                            matches = re.findall(pattern, eq_str)
                            
                            if matches:
                                # Get the coefficient
                                coef = float(matches[0].replace(' ', ''))
                                
                                key = (concept, pred_class)
                                if is_licem:
                                    data_by_concept_class[key]['licem'].append(coef)
                                elif is_lin_sym:
                                    data_by_concept_class[key]['lin_sym'].append(coef)
                    
                    # Mark this model type as found
                    if is_licem:
                        found_licem = True
                        print(f"Found LICEM experiment: {exp}")
                    elif is_lin_sym:
                        found_lin_sym = True
                        print(f"Found Lin-Sym-CBM experiment: {exp}")
                    
                    # Stop processing if we have both models
                    if found_licem and found_lin_sym:
                        print("Found both model types, stopping search")
                        break
                
                except Exception as e:
                    print(f"Error processing {exp_path}: {e}")
                    continue
            
            # Break out of exp_dir loop if we found both
            if found_licem and found_lin_sym:
                break
        
        # Break out of path loop if we found both
        if found_licem and found_lin_sym:
            break
    
    if not data_by_concept_class:
        print("No data found for global verifiability plot")
        return
    
    print(f"Found {len(all_concepts)} concepts and {len(all_classes)} classes")
    print(f"Total concept-class pairs: {len(data_by_concept_class)}")
    
    # Filter pairs that have data for both models and lin_sym has at least 2 unique values
    valid_pairs = [(concept, cls) for (concept, cls), data in data_by_concept_class.items()
                   if len(data['licem']) > 0 and len(data['lin_sym']) > 0 
                   and len(set(data['lin_sym'])) >= 2]
    
    if not valid_pairs:
        print("No concept-class pairs with data from both models and at least 2 unique lin_sym values")
        return
    
    print(f"Valid concept-class pairs with both models and 2+ unique lin_sym values: {len(valid_pairs)}")
    
    # Randomly select 30 pairs
    np.random.seed(42)  # For reproducibility
    n_plots = min(30, len(valid_pairs))
    selected_pairs = np.random.choice(len(valid_pairs), size=n_plots, replace=False)
    selected_pairs = [valid_pairs[i] for i in selected_pairs]
    
    print(f"Generating {n_plots} plots...")
    
    # Create output directory
    output_dir = os.path.join(result_figs, 'global_verifiability')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate individual plots
    for concept, pred_class in selected_pairs:
        licem_weights = data_by_concept_class[(concept, pred_class)]['licem']
        lin_sym_weights = data_by_concept_class[(concept, pred_class)]['lin_sym']
        
        # Create the plot with dual y-axes (wider figure, more pronounced width)
        fig, ax1 = plt.subplots(figsize=(14, 4))
        ax2 = ax1.twinx()
        
        # Plot licem weights as continuous line on left axis using KDE
        if licem_weights:
            from scipy.stats import gaussian_kde
            
            licem_arr = np.array(licem_weights)
            
            # Compute data range with padding to show density tapering to zero
            data_min, data_max = licem_arr.min(), licem_arr.max()
            data_range = data_max - data_min
            padding = max(0.3 * data_range, 0.5)  # At least 30% padding or 0.5 units
            
            # Use Kernel Density Estimation for smooth density that tapers to zero
            try:
                kde = gaussian_kde(licem_arr, bw_method='silverman')
                
                # Extend evaluation range beyond data to show zero-tapering boundaries
                x_eval = np.linspace(data_min - padding, data_max + padding, 200)
                density = kde(x_eval)
                
                ax1.plot(x_eval, density, '-', color='tab:blue', linewidth=2.5, 
                        label=f'LICEM (n={len(licem_weights)})', alpha=0.8)
                # Optional: fill under the curve for better visualization
                ax1.fill_between(x_eval, density, alpha=0.2, color='tab:blue')
            except Exception as e:
                # Fallback to histogram if KDE fails (e.g., too few unique values)
                counts, bins = np.histogram(licem_weights, bins=50, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                smoothed_counts = gaussian_filter1d(counts, sigma=2)
                ax1.plot(bin_centers, smoothed_counts, '-', color='tab:blue', linewidth=2.5, 
                        label=f'LICEM (n={len(licem_weights)})', alpha=0.8)
        
        # Plot linear_symbolic_cbm weights as bars on right axis
        if lin_sym_weights:
            # Get unique values and their counts
            unique_vals, counts = np.unique(lin_sym_weights, return_counts=True)
            # Normalize counts to get probability
            total = len(lin_sym_weights)
            probs = counts / total
            
            # Use fixed bar width
            bar_width = 0.1
            
            # Plot as bars
            ax2.bar(unique_vals, probs, width=bar_width, color='tab:orange', 
                   alpha=0.7, edgecolor='black', linewidth=1.5,
                   label=f'Lin-Mem-CBM (n={len(lin_sym_weights)})')
            
            # Add markers on top of bars to make low-probability values visible
            # ax2.scatter(unique_vals, probs, color='black', s=100, zorder=5, marker='o')
        
        # Clean class name for display and filename
        class_display = pred_class.replace('+', ' ')
        class_filename = pred_class.replace('+', '_')
        
        # Use larger fonts specifically for this plot
        local_label_size = 44
        local_tick_size = 36
        
        ax1.set_xlabel('Weight', fontsize=local_label_size, fontweight='bold')
        ax1.set_ylabel('Density', fontsize=local_label_size, fontweight='bold', color='tab:blue')
        ax2.set_ylabel('Probability', fontsize=local_label_size, fontweight='bold', color='tab:orange')
        
        # Align both y-axes to start at zero
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        
        # Eliminate minor ticks
        ax1.minorticks_off()
        ax2.minorticks_off()
        
        ax1.grid(True, alpha=0.3, which='both')
        ax1.tick_params(axis='both', which='major', labelsize=local_tick_size, color='tab:blue', labelcolor='tab:blue')
        ax2.tick_params(axis='y', which='major', labelsize=local_tick_size, color='tab:orange', labelcolor='tab:orange')
        
        # Adjust layout to prevent right ylabel from being cut off
        fig.tight_layout(rect=[0, 0, 0.92, 1])
        
        # Save the figure
        save_filename = f'{concept}_{class_filename}.pdf'
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"Saved: {save_filename}")

#########################################
######### Adaptability Functions ########
#########################################

def get_adaptability_exp_from_path(base_paths, constraint_configs):
    """
    Gather results from adaptability experiments where multiple constraint sets 
    were tested from the same initial checkpoint.
    
    Args:
        base_paths: List of base paths containing adaptability experiments
        constraint_configs: List of constraint configuration names (e.g., ['simple', 'medium', 'complex'])
        
    Returns:
        DataFrame with performance metrics, including constraint configuration in model name
    """
    from src.utils.complexity import compute_complexity
    from sympy import sympify, symbols
    
    # Collect all the experiments in the given paths
    exps_path = []
    for path in base_paths:
        if os.path.exists(path):
            experiment_dir = os.listdir(path)
            for exp in experiment_dir:
                exp_full_path = os.path.join(path, exp)
                if os.path.isdir(exp_full_path) and 'multirun' not in exp:
                    exps_path.append(exp_full_path)

    performance = pd.DataFrame()

    # Each directory name_of_experiment/date_time/experiment contains:
    # - The original experiment (base level)
    # - Constrained alternatives in subdirectories constraint_0, constraint_1, etc.
    

    # Iterate over all the experiments
    for folder_exp in tqdm(exps_path, desc="Processing adaptability experiments"):
        for exp in os.listdir(folder_exp):
            if 'multirun' in exp:
                continue

            exp = os.path.join(folder_exp, exp)

            # Check if this is an adaptability experiment by looking for constraint subdirectories
            constraint_dirs = [d for d in os.listdir(os.path.join(exp, 'logs/experiment_metrics')) if d.startswith('constraint_')]
            
            if not constraint_dirs:
                continue
                
            # Load the main config to get base settings
            main_conf_file = os.path.join(exp, '.hydra/config.yaml')
            if not os.path.exists(main_conf_file):
                continue
                
            with open(main_conf_file, 'r') as file:
                main_conf = yaml.safe_load(file)
            
            # First, process the original experiment (without constraints)
            d = {}
            result_file = os.path.join(exp, 'logs/experiment_metrics/metrics.csv')
            
            try:
                if os.path.exists(result_file):
                    d['seed'] = main_conf['seed']
                    d['dataset'] = main_conf['dataset']['metadata']['name']
                    d['model'] = f"{main_conf['model']['metadata']['name']}_original"
                    d['base_model'] = main_conf['model']['metadata']['name']
                    d['constraint_config'] = 'original'
                    d['memory_size'] = main_conf['memory_size']
                    d['concept_percentage'] = main_conf.get('concept_percentage', 1.0)
                    d['task_type'] = main_conf['dataset']['metadata']['task']
                    d['path'] = exp

                    with open(result_file, 'r') as file:
                        result = pd.read_csv(file)

                    # Select the last row for test metrics
                    if 'test/y/mse' in result.columns:
                        d['task_mse'] = result['test/y/mse'].iloc[-1]
                        d['task_mae'] = result['test/y/mae'].iloc[-1]
                    else:
                        d['task_acc'] = result['test/y/acc'].iloc[-1]

                    if main_conf['model']['metadata']['name'] != 'blackbox':
                        if 'test/c/mse' in result.columns:
                            d['concept_mse'] = result['test/c/mse'].iloc[-1]
                            d['concept_mae'] = result['test/c/mae'].iloc[-1]
                        else:
                            d['concept_acc'] = result['test/c/acc'].iloc[-1]
                    
                    # Compute complexity for equations
                    predictions_file = os.path.join(exp, 'logs/experiment_metrics/test_predictions_per_sample.csv')
                    if os.path.exists(predictions_file):
                        try:
                            # with timeout(3):
                            #     df_pred = pd.read_csv(predictions_file)
                            df_pred = pd.read_csv(predictions_file)
                            
                            if 'equation' in df_pred.columns:
                                vars = [x.replace('c_pred_','') for x in df_pred.columns if 'c_pred' in x]
                                
                                equation_counts = df_pred['equation'].value_counts().to_dict()
                                equation_counts = {eq: cnt for eq, cnt in equation_counts.items() if pd.notna(eq) and eq != ''}

                                # Clean equations
                                cleaned_equation_counts = {}
                                for eq, cnt in equation_counts.items():
                                    if ':' in eq:
                                        cleaned_eq = eq.split(':')[1].strip()
                                    else:
                                        cleaned_eq = eq
                                    if cleaned_eq in cleaned_equation_counts:
                                        cleaned_equation_counts[cleaned_eq] += cnt
                                    else:
                                        cleaned_equation_counts[cleaned_eq] = cnt

                                # Compute all complexity metrics
                                complexity_metrics = COMPLEXITY_METRICS_LIST
                                complexity_totals = {metric: 0 for metric in complexity_metrics}
                                
                                for eq, _ in tqdm(cleaned_equation_counts.items(), 
                                                desc=f"Complexity, {d['model']}, {d['dataset']}", 
                                                leave=False):
                                    sympy_eq = sympify(eq, locals={var: symbols(var) for var in vars})
                                    for metric in complexity_metrics:
                                        complexity_totals[metric] += compute_complexity(sympy_eq, metric=metric)
                                
                                for metric in complexity_metrics:
                                    d[f'complexity_{metric}'] = complexity_totals[metric]
                        
                        except Exception as e:
                            print(f"Error computing complexity for {exp}: {e}")
                            complexity_metrics = COMPLEXITY_METRICS_LIST
                            for metric in complexity_metrics:
                                d[f'complexity_{metric}'] = np.nan

                    performance = pd.concat([performance, pd.DataFrame([d])], ignore_index=True)
                
            except Exception as e:
                print(f"Error while processing original experiment {exp}: {e}")
                
            # Now process each constraint directory
            for constraint_dir in constraint_dirs:
                constraint_name = constraint_dir.split('_')[1]
                constraint_path = os.path.join(exp, 'logs', 'experiment_metrics', constraint_dir)
                
                d = {}
                result_file = os.path.join(constraint_path, 'metrics.csv')
                
                try:
                    if not os.path.exists(result_file):
                        continue
                        
                    d['seed'] = main_conf['seed']
                    d['dataset'] = main_conf['dataset']['metadata']['name']
                    # Append constraint name to model name
                    d['model'] = f"{main_conf['model']['metadata']['name']}_{constraint_name}"
                    d['base_model'] = main_conf['model']['metadata']['name']
                    d['constraint_config'] = constraint_name
                    d['memory_size'] = main_conf['memory_size']
                    d['concept_percentage'] = main_conf.get('concept_percentage', 1.0)
                    d['task_type'] = main_conf['dataset']['metadata']['task']
                    d['path'] = constraint_path

                    with open(result_file, 'r') as file:
                        result = pd.read_csv(file)

                    # Select the last row for test metrics
                    if 'test/y/mse' in result.columns:
                        d['task_mse'] = result['test/y/mse'].iloc[-1]
                        d['task_mae'] = result['test/y/mae'].iloc[-1]
                    else:
                        d['task_acc'] = result['test/y/acc'].iloc[-1]

                    if main_conf['model']['metadata']['name'] != 'blackbox':
                        if 'test/c/mse' in result.columns:
                            d['concept_mse'] = result['test/c/mse'].iloc[-1]
                            d['concept_mae'] = result['test/c/mae'].iloc[-1]
                        else:
                            d['concept_acc'] = result['test/c/acc'].iloc[-1]
                    
                    # Compute complexity for equations
                    predictions_file = os.path.join(constraint_path, 'test_predictions_per_sample.csv')
                    if os.path.exists(predictions_file):
                        try:
                            with timeout(3):
                                df_pred = pd.read_csv(predictions_file)
                            
                            if 'equation' in df_pred.columns:
                                vars = [x.replace('c_pred_','') for x in df_pred.columns if 'c_pred' in x]
                                
                                equation_counts = df_pred['equation'].value_counts().to_dict()
                                equation_counts = {eq: cnt for eq, cnt in equation_counts.items() if pd.notna(eq) and eq != ''}

                                # Clean equations
                                cleaned_equation_counts = {}
                                for eq, cnt in equation_counts.items():
                                    if ':' in eq:
                                        cleaned_eq = eq.split(':')[1].strip()
                                    else:
                                        cleaned_eq = eq
                                    if cleaned_eq in cleaned_equation_counts:
                                        cleaned_equation_counts[cleaned_eq] += cnt
                                    else:
                                        cleaned_equation_counts[cleaned_eq] = cnt

                                # Compute all complexity metrics
                                complexity_metrics = COMPLEXITY_METRICS_LIST
                                complexity_totals = {metric: 0 for metric in complexity_metrics}
                                
                                for eq, _ in tqdm(cleaned_equation_counts.items(), 
                                                desc=f"Complexity, {d['model']}, {d['dataset']}", 
                                                leave=False):
                                    sympy_eq = sympify(eq, locals={var: symbols(var) for var in vars})
                                    for metric in complexity_metrics:
                                        complexity_totals[metric] += compute_complexity(sympy_eq, metric=metric)
                                
                                for metric in complexity_metrics:
                                    d[f'complexity_{metric}'] = complexity_totals[metric]
                        
                        except Exception as e:
                            print(f"Error computing complexity for {constraint_path}: {e}")
                            complexity_metrics = COMPLEXITY_METRICS_LIST
                            for metric in complexity_metrics:
                                d[f'complexity_{metric}'] = np.nan

                    performance = pd.concat([performance, pd.DataFrame([d])], ignore_index=True)
                    
                except Exception as e:
                    print(f"Error while processing {constraint_path}: {e}")
                    continue

    return performance


def generate_adaptability_table(performance, datasets, output_path, regression_datasets):
    """
    Generate a LaTeX table for adaptability experiments with datasets as columns,
    each split into MAE and Nodes sub-columns. Rows are models.
    Designed to fit in a single column of a two-column paper format.
    
    Args:
        performance: DataFrame with performance data
        datasets: List of dataset names (in order)
        output_path: Directory to save the table
        regression_datasets: List of regression dataset names
    """
    import pandas as pd
    
    os.makedirs(output_path, exist_ok=True)
    
    # Get models present in the data
    models_in_data = performance['model'].unique()
    
    # Sort models according to models_order (from show_results.py)
    # The model names in data are like "sr_symbolic_cbm_original", "sr_symbolic_cbm_simple"
    # We need to order by base model first, then by constraint config
    constraint_order = ['original', 'simple', 'medium', 'complex']
    
    def model_sort_key(model_name):
        # Extract base model and constraint config
        parts = model_name.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in constraint_order:
            base_model = parts[0]
            constraint = parts[1]
        else:
            base_model = model_name
            constraint = 'original'
        
        # Get base model order
        base_order = models_order.index(base_model) if base_model in models_order else 999
        constraint_order_idx = constraint_order.index(constraint) if constraint in constraint_order else 999
        
        return (base_order, constraint_order_idx)
    
    sorted_models = sorted(models_in_data, key=model_sort_key)
    
    # Filter datasets to only those present in data and in custom_order
    datasets_in_data = performance['dataset'].unique()
    ordered_datasets = [d for d in custom_order if d in datasets_in_data and d in datasets]
    
    # Group by model and dataset
    grouped = performance.groupby(['model', 'dataset']).agg({
        'task_mae': ['mean', 'std'],
        'complexity_node_count': ['mean', 'std'],
    })
    
    # Build LaTeX table - compact version for single column
    lines = []
    
    # Number of datasets
    n_datasets = len(ordered_datasets)
    
    # Column format: l for model name, then 2 columns per dataset (MAE | Nodes)
    col_format = "l" + "cc" * n_datasets
    
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Adaptability Experiment Results}")
    lines.append(r"\label{tab:adaptability}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")  # Reduce column padding
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_format + r"}")
    lines.append(r"\toprule")
    
    # First header row: dataset names spanning 2 columns each
    header1_parts = [""]
    for dataset in ordered_datasets:
        dataset_display = get_df_name(dataset)
        header1_parts.append(r"\multicolumn{2}{c}{" + dataset_display + "}")
    lines.append(" & ".join(header1_parts) + r" \\")
    
    # Add cmidrule for each dataset
    cmidrule_parts = []
    for i, _ in enumerate(ordered_datasets):
        start_col = 2 + i * 2
        end_col = start_col + 1
        cmidrule_parts.append(rf"\cmidrule(lr){{{start_col}-{end_col}}}")
    lines.append(" ".join(cmidrule_parts))
    
    # Second header row: MAE | Nodes for each dataset
    header2_parts = ["Model"]
    for _ in ordered_datasets:
        header2_parts.append("MAE")
        header2_parts.append("Complexity")
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")
    
    # Compact constraint abbreviations
    constraint_abbrev = {'original': 'Orig.', 'simple': 'S', 'medium': 'M', 'complex': 'C'}
    
    # Data rows
    for model in sorted_models:
        # Get display name for model
        # Extract base model and constraint
        parts = model.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in constraint_order:
            base_model = parts[0]
            constraint = parts[1]
        else:
            base_model = model
            constraint = ''
        
        # Get display name from model_styles
        if base_model in model_styles:
            base_display = model_styles[base_model]['name']
        else:
            base_display = base_model.replace('_', ' ').title()
        
        # Add constraint suffix with abbreviation
        if constraint and constraint in constraint_abbrev:
            model_display = f"{base_display} ({constraint_abbrev[constraint]})"
        else:
            model_display = base_display
        
        row_parts = [model_display]
        
        for dataset in ordered_datasets:
            try:
                if (model, dataset) in grouped.index:
                    model_dataset = grouped.loc[(model, dataset)]
                    
                    # MAE - compact format with std in scriptscriptstyle
                    mae_mean = model_dataset[('task_mae', 'mean')]
                    mae_std = model_dataset[('task_mae', 'std')]
                    if pd.notna(mae_mean):
                        if pd.notna(mae_std) and mae_std >= 0.01:
                            mae_cell = rf"${mae_mean:.2f}{{\scriptstyle\pm{mae_std:.2f}}}$"
                        else:
                            mae_cell = rf"${mae_mean:.2f}{{\scriptstyle<0.01}}$"
                    else:
                        mae_cell = "--"
                    
                    # Nodes (complexity_node_count) - compact format
                    # NA complexity means only one node, so replace with 1
                    nodes_mean = model_dataset[('complexity_node_count', 'mean')]
                    nodes_std = model_dataset[('complexity_node_count', 'std')]
                    if not pd.notna(nodes_mean):
                        nodes_mean = 1.0  # NA means only one node
                        nodes_std = 0.0
                    if pd.notna(nodes_std) and nodes_std >= 0.01:
                        nodes_cell = rf"${nodes_mean:.1f}{{\scriptstyle\pm{nodes_std:.1f}}}$"
                    else:
                        nodes_cell = rf"${nodes_mean:.1f}{{\scriptstyle<0.01}}$"
                else:
                    mae_cell = "--"
                    nodes_cell = "--"
            except Exception:
                mae_cell = "--"
                nodes_cell = "--"
            
            row_parts.append(mae_cell)
            row_parts.append(nodes_cell)
        
        lines.append(" & ".join(row_parts) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")  # end resizebox
    lines.append(r"\end{table}")
    
    # Join lines and save
    table_latex = "\n".join(lines)
    
    latex_path = os.path.join(output_path, 'adaptability_results.tex')
    with open(latex_path, 'w') as f:
        f.write(table_latex)
    print(f"LaTeX table saved to {latex_path}")
    
    # Also save CSV for reference
    table_data = []
    for model in sorted_models:
        parts = model.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in constraint_order:
            base_model = parts[0]
            constraint = parts[1]
        else:
            base_model = model
            constraint = ''
        
        if base_model in model_styles:
            base_display = model_styles[base_model]['name']
        else:
            base_display = base_model.replace('_', ' ').title()
        
        if constraint and constraint in constraint_abbrev:
            model_display = f"{base_display} ({constraint_abbrev[constraint]})"
        else:
            model_display = base_display
        
        row = {'Model': model_display}
        for dataset in ordered_datasets:
            try:
                if (model, dataset) in grouped.index:
                    model_dataset = grouped.loc[(model, dataset)]
                    mae_mean = model_dataset[('task_mae', 'mean')]
                    mae_std = model_dataset[('task_mae', 'std')]
                    nodes_mean = model_dataset[('complexity_node_count', 'mean')]
                    nodes_std = model_dataset[('complexity_node_count', 'std')]
                    row[f'{dataset}_mae'] = f"{mae_mean:.3f} ± {mae_std:.3f}" if pd.notna(mae_mean) else "N/A"
                    row[f'{dataset}_nodes'] = f"{nodes_mean:.1f} ± {nodes_std:.1f}" if pd.notna(nodes_mean) else "N/A"
                else:
                    row[f'{dataset}_mae'] = "N/A"
                    row[f'{dataset}_nodes'] = "N/A"
            except Exception:
                row[f'{dataset}_mae'] = "N/A"
                row[f'{dataset}_nodes'] = "N/A"
        table_data.append(row)
    
    result_df = pd.DataFrame(table_data)
    csv_path = os.path.join(output_path, 'adaptability_results.csv')
    result_df.to_csv(csv_path, index=False)
    print(f"CSV table saved to {csv_path}")
    
    return result_df


def extract_equation_examples(paths, output_path='results/tabs/equation_examples', 
                                n_examples=5, fixed_class=None, fixed_memory=None):
    """
    Extract example equations from test_predictions_per_sample.csv files.
    
    For each dataset and model, extracts n_examples of equations.
    For classification datasets, filters samples from a fixed class if provided.
    For datasets in fixed_memory with memory models, only uses experiments with matching memory_size.
    
    Args:
        paths: List of paths to experiment directories
        output_path: Path to save the output CSV files
        n_examples: Number of examples to extract per model per dataset
        fixed_class: Dict mapping dataset names to class indices for classification datasets.
                     If None, will use the first class (0) for classification datasets.
        fixed_memory: Dict mapping dataset names to their fixed memory size.
                      For memory models, only experiments with this memory_size will be used.
    
    Returns:
        None. Saves CSV files to output_path.
    """
    import glob
    
    os.makedirs(output_path, exist_ok=True)
    
    # Collect all test_predictions_per_sample.csv files
    all_results = {}
    
    for base_path in paths:
        # Find all experiment directories
        exp_dirs = glob.glob(os.path.join(base_path, '**/logs/experiment_metrics/test_predictions_per_sample.csv'), 
                             recursive=True)
        
        pbar = tqdm(exp_dirs, desc=f"Processing {base_path}")
        for predictions_file in pbar:
            # Update progress bar to show current file
            rel_path = os.path.relpath(predictions_file, base_path)
            pbar.set_description(f"Processing {base_path} | {rel_path}")
            
            if 'memory_cbm' in predictions_file or 'cem' in predictions_file or 'blackbox' in predictions_file:
                continue

            try:
                # Extract metadata from path
                parts = predictions_file.split('/')
                # Find the part with dataset and model info
                exp_folder = None
                for part in parts:
                    if 'dataset_' in part and 'model_' in part:
                        exp_folder = part
                        break
                
                if not exp_folder:
                    continue
                
                # Parse dataset, model, memory_size, and seed from folder name
                dataset = None
                model = None
                seed = None
                memory_size = None
                
                exp_parts = exp_folder.split('_')
                for i, item in enumerate(exp_parts):
                    if dataset is None and item in custom_order:
                        dataset = item
                    elif 'lin' in item or 'sym' in item or 'cbm' in item:
                        # Start collecting model name
                        if model is None:
                            model = item
                        else:
                            model += '_' + item
                    elif item == 'memory' and i + 1 < len(exp_parts) and exp_parts[i + 1] == 'size':
                        # Next item after 'size' is the memory size number
                        if i + 2 < len(exp_parts):
                            try:
                                memory_size = int(exp_parts[i + 2])
                            except:
                                pass
                    elif item == 'seed':
                        # Next item is the seed number
                        continue
                    elif model and dataset and seed is None:
                        try:
                            seed = int(item)
                        except:
                            pass
                
                if not dataset or not model:
                    continue
                
                # Skip memory_cbm, cem, and blackbox models BEFORE reading the file
                if model in ['memory_cbm', 'cem', 'blackbox']:
                    continue
                
                # Also check if any of these strings appear in the model name
                if any(skip_model in model for skip_model in ['memory_cbm', 'cem', 'blackbox']):
                    continue
                
                # For datasets with fixed_memory, filter by memory_size for memory models
                if fixed_memory and dataset in fixed_memory:
                    if model in MEMORY_MODELS_LIST:
                        required_memory_size = fixed_memory[dataset]
                        if memory_size != required_memory_size:
                            continue  # Skip this experiment if memory_size doesn't match
                
                # Read predictions file
                df_pred = pd.read_csv(predictions_file)
                
                if 'equation' not in df_pred.columns:
                    continue
                
                # Determine if this is a classification or regression dataset
                is_classification = dataset in NUMBER_OF_CLASSES_PER_DATASET or dataset not in regression_datasets
                
                # For classification, filter by class
                if is_classification and 'y_true' in df_pred.columns:
                    target_class = fixed_class.get(dataset, 0) if fixed_class else 0
                    df_pred = df_pred[df_pred['y_true'] == target_class]
                
                # Get unique equations
                if len(df_pred) == 0:
                    continue
                
                # For memory models, extract equations per memory expression (target)
                # Group by equation/target to get examples from different memory expressions
                if model in MEMORY_MODELS_LIST and 'equation' in df_pred.columns:
                    # Get unique equations (memory expressions)
                    unique_equations = df_pred['equation'].unique()
                    samples = []
                    for eq in unique_equations:
                        eq_samples = df_pred[df_pred['equation'] == eq]
                        # Sample up to n_examples per equation
                        sample_size = min(n_examples, len(eq_samples))
                        sampled = eq_samples.sample(n=sample_size, random_state=42)
                        samples.append(sampled)
                    samples = pd.concat(samples) if samples else pd.DataFrame()
                else:
                    # Sample up to n_examples total
                    sample_size = min(n_examples, len(df_pred))
                    samples = df_pred.sample(n=sample_size, random_state=42)
                
                if len(samples) == 0:
                    continue
                
                # Store results
                key = (dataset, model)
                if key not in all_results:
                    all_results[key] = []
                
                for idx, row in samples.iterrows():
                    equation = row['equation']
                    # Clean equation (remove target prefix if present)
                    if ':' in str(equation):
                        equation = equation.split(':', 1)[1].strip()
                    
                    result_entry = {
                        'dataset': dataset,
                        'model': model,
                        'seed': seed,
                        'memory_size': memory_size,
                        'sample_idx': row['sample_idx'],
                        'equation': equation,
                        'y_true': row.get('y_true', None),
                        'y_pred': row.get('y_pred', None),
                    }
                    
                    # Add class names if available
                    if 'y_true_task_name' in row:
                        result_entry['y_true_class'] = row['y_true_task_name']
                    if 'y_pred_task_name' in row:
                        result_entry['y_pred_class'] = row['y_pred_task_name']
                    
                    all_results[key].append(result_entry)
                    
            except Exception as e:
                print(f"Error processing {predictions_file}: {e}")
                continue
    
    # Save results per dataset
    datasets_processed = set()
    for (dataset, model), entries in all_results.items():
        datasets_processed.add(dataset)
    
    for dataset in sorted(datasets_processed):
        dataset_results = []
        
        # Collect all results for this dataset
        for (ds, model), entries in all_results.items():
            if ds == dataset:
                # For memory models, keep all entries (already limited per equation)
                # For non-memory models, limit to n_examples total
                if model in MEMORY_MODELS_LIST:
                    for i, entry in enumerate(entries):
                        entry['example_id'] = i + 1
                        dataset_results.append(entry)
                else:
                    for i, entry in enumerate(entries[:n_examples]):
                        entry['example_id'] = i + 1
                        dataset_results.append(entry)
        
        if dataset_results:
            df_results = pd.DataFrame(dataset_results)
            
            # Reorder columns
            base_cols = ['dataset', 'model', 'seed']
            if 'memory_size' in df_results.columns:
                base_cols.append('memory_size')
            base_cols.extend(['example_id', 'equation'])
            if 'y_true_class' in df_results.columns:
                base_cols.extend(['y_true_class', 'y_pred_class', 'y_true', 'y_pred'])
            else:
                base_cols.extend(['y_true', 'y_pred'])
            base_cols.append('sample_idx')
            
            df_results = df_results[base_cols]
            
            # Sort by model and example_id
            df_results = df_results.sort_values(['model', 'seed', 'example_id'])
            
            # Save to CSV
            csv_path = os.path.join(output_path, f'{dataset}_equation_examples.csv')
            df_results.to_csv(csv_path, index=False)
            print(f"Saved {len(df_results)} equation examples for {dataset} to {csv_path}")
    
    print(f"\nProcessed {len(datasets_processed)} datasets")
    print(f"Results saved to {output_path}")


def generate_intervention_delta_table(
    performance,
    custom_order,
    model_styles,
    regression_datasets,
    output_path,
    noise_level=0.0
):
    """
    Generate a table showing the delta in task performance when intervening on all concepts
    (p_int=1.0) versus not intervening (p_int=0.0).
    
    Args:
        performance: DataFrame with intervention data (from get_intervention_from_path)
        custom_order: List of datasets in desired order
        model_styles: Dictionary with model styling information
        regression_datasets: List of regression dataset names
        output_path: Directory path to save the table
        noise_level: Noise level to filter on (default 0.0)
    """
    import os
    import pandas as pd
    
    os.makedirs(output_path, exist_ok=True)
    
    # Filter for specified noise level
    perf_filtered = performance[performance['noise'] == noise_level].copy()
    
    # Filter for only datasets in custom_order
    perf_filtered = perf_filtered[perf_filtered['dataset'].isin(custom_order)]
    
    if perf_filtered.empty:
        print(f"No intervention data found for noise level {noise_level}")
        return None
    
    # Separate intervention data for p_int=0.0 and p_int=1.0
    no_intervention = perf_filtered[perf_filtered['p_int'] == 0.0].copy()
    full_intervention = perf_filtered[perf_filtered['p_int'] == 1.0].copy()
    
    if no_intervention.empty or full_intervention.empty:
        print("Missing data for p_int=0.0 or p_int=1.0")
        return None
    
    # Determine metric: accuracy for classification, MAE for regression
    results = []
    
    for dataset in custom_order:
        if dataset not in perf_filtered['dataset'].unique():
            continue
            
        is_regression = dataset in regression_datasets
        metric = 'mae' if is_regression else 'accuracy'
        
        for model in model_styles.keys():
            if model not in perf_filtered['model'].unique():
                continue
            
            # Get data for this dataset-model combination
            no_int_data = no_intervention[
                (no_intervention['dataset'] == dataset) & 
                (no_intervention['model'] == model)
            ][['seed', metric]]
            
            full_int_data = full_intervention[
                (full_intervention['dataset'] == dataset) & 
                (full_intervention['model'] == model)
            ][['seed', metric]]
            
            if len(no_int_data) == 0 or len(full_int_data) == 0:
                continue
            
            # Merge on seed to compute per-seed deltas
            merged = pd.merge(
                no_int_data, 
                full_int_data, 
                on='seed', 
                suffixes=('_no_int', '_full_int')
            )
            
            if len(merged) == 0:
                continue
            
            # Calculate delta for each seed
            # For accuracy: delta = full_int - no_int (positive means improvement)
            # For MAE: delta = no_int - full_int (positive means improvement, i.e., lower error)
            if is_regression:
                merged['delta'] = merged[f'{metric}_no_int'] - merged[f'{metric}_full_int']
            else:
                merged['delta'] = merged[f'{metric}_full_int'] - merged[f'{metric}_no_int']
            
            # Calculate mean and confidence interval on deltas
            delta = merged['delta'].mean()
            delta_std = merged['delta'].std()
            n_seeds = len(merged)
            
            # 95% confidence interval = 1.96 * SE
            delta_ci95 = 1.96 * delta_std / (n_seeds ** 0.5) if n_seeds > 0 else 0
            
            # Multiply by 100 if metric is accuracy (to show as percentage)
            if not is_regression:
                delta = delta * 100
                delta_ci95 = delta_ci95 * 100
            
            # Calculate mean performance for each intervention level (for reference)
            no_int_mean = merged[f'{metric}_no_int'].mean()
            full_int_mean = merged[f'{metric}_full_int'].mean()
            
            results.append({
                'dataset': dataset,
                'model': model,
                'delta_mean': delta,
                'delta_ci95': delta_ci95,
                'no_int_mean': no_int_mean,
                'full_int_mean': full_int_mean,
                'metric': metric,
                'n_seeds': n_seeds
            })
    
    if not results:
        print("No results to generate table")
        return None
    
    df_results = pd.DataFrame(results)
    
    # Save CSV
    csv_path = os.path.join(output_path, f'intervention_delta_noise_{noise_level}.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"Saved intervention delta results to {csv_path}")
    
    # Separate datasets into classification and regression
    classification_datasets = [d for d in custom_order if d not in regression_datasets and d in df_results['dataset'].unique()]
    regression_datasets_present = [d for d in custom_order if d in regression_datasets and d in df_results['dataset'].unique()]
    
    # Function to generate table for a specific dataset type
    def generate_table(datasets_list, table_type, metric_label):
        if not datasets_list:
            return None
            
        # Create pivot table for LaTeX
        pivot_data = []
        for model in model_styles.keys():
            model_data = {'model': model_styles[model]['name']}
            for dataset in datasets_list:
                row = df_results[(df_results['dataset'] == dataset) & (df_results['model'] == model)]
                if not row.empty:
                    delta_mean = row.iloc[0]['delta_mean']
                    delta_ci95 = row.iloc[0]['delta_ci95']
                    # Format uncertainty with \tiny and use <= 0.01 for very small values
                    if delta_ci95 < 0.01:
                        model_data[dataset] = f"{delta_mean:.2f} {{\\tiny $\\pm \\leq$ 0.01}}"
                    else:
                        model_data[dataset] = f"{delta_mean:.2f} {{\\tiny $\\pm$ {delta_ci95:.2f}}}"
                else:
                    model_data[dataset] = "-"
            # Only add row if model has at least one valid entry for these datasets
            if any(model_data.get(d, '-') != '-' for d in datasets_list):
                pivot_data.append(model_data)
        
        if not pivot_data:
            return None
            
        df_pivot = pd.DataFrame(pivot_data)
        
        # Generate LaTeX table
        latex_lines = []
        latex_lines.append(r"\begin{table}[t]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\scriptsize")
        latex_lines.append(r"\setlength{\tabcolsep}{4pt}")
        
        # Build column specification
        n_datasets = len(datasets_list)
        col_spec = "l" + "c" * n_datasets
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append(r"\toprule")
        
        # Header row with dataset names
        header = "\\textbf{Model}"
        for dataset in datasets_list:
            header += f" & \\textbf{{{get_df_name(dataset)}}}"
        header += " \\\\"
        latex_lines.append(header)
        latex_lines.append(r"\midrule")
        
        # Data rows
        for _, row in df_pivot.iterrows():
            line = row['model']
            for dataset in datasets_list:
                line += f" & {row.get(dataset, '-')}"
            line += " \\\\"
            latex_lines.append(line)
        
        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        
        # Caption based on table type
        if table_type == 'classification':
            caption = (r"\caption{Performance delta ($\Delta$) when intervening on all concepts (p\_int=1.0) vs. no intervention (p\_int=0.0) " +
                      f"for classification datasets. $\\Delta$ = accuracy(full) - accuracy(none). " +
                      f"Positive values indicate improvement. Values shown as mean $\\pm$ 95\\% CI. Noise: {noise_level}.}}")
            label = r"\label{tab:intervention_delta_classification}"
        else:
            caption = (r"\caption{Performance delta ($\Delta$) when intervening on all concepts (p\_int=1.0) vs. no intervention (p\_int=0.0) " +
                      f"for regression datasets. $\\Delta$ = MAE(none) - MAE(full). " +
                      f"Positive values indicate improvement. Values shown as mean $\\pm$ 95\\% CI. Noise: {noise_level}.}}")
            label = r"\label{tab:intervention_delta_regression}"
        
        latex_lines.append(caption)
        latex_lines.append(label)
        latex_lines.append(r"\end{table}")
        
        return '\n'.join(latex_lines)
    
    # Generate classification table
    if classification_datasets:
        classification_table = generate_table(classification_datasets, 'classification', 'Accuracy')
        if classification_table:
            tex_path = os.path.join(output_path, f'intervention_delta_classification_noise_{noise_level}.tex')
            with open(tex_path, 'w') as f:
                f.write(classification_table)
            print(f"Saved classification LaTeX table to {tex_path}")
    
    # Generate regression table
    if regression_datasets_present:
        regression_table = generate_table(regression_datasets_present, 'regression', 'MAE')
        if regression_table:
            tex_path = os.path.join(output_path, f'intervention_delta_regression_noise_{noise_level}.tex')
            with open(tex_path, 'w') as f:
                f.write(regression_table)
            print(f"Saved regression LaTeX table to {tex_path}")
    
    return df_results


########################################
#### Global Interpretability Table #####
########################################

def collect_global_interpretability_results(base_path):
    """
    Walk the output directory tree and collect all global_interpretability.csv files.
    """
    all_dfs = []
    if not os.path.exists(base_path):
        print(f"Warning: {base_path} does not exist.")
        return pd.DataFrame()

    for root, dirs, files in os.walk(base_path):
        for fname in files:
            if fname == 'global_interpretability.csv':
                fpath = os.path.join(root, fname)
                try:
                    df = pd.read_csv(fpath)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Warning: could not read {fpath}: {e}")

    if not all_dfs:
        print("No global_interpretability.csv files found.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def select_gi_concepts(df, dataset, task_idx=0, n_display=3):
    """
    Select top n_display concepts by training-range violation count.

    For each (dataset, seed, concept), check whether the test range
    exceeds the training range (test_min < train_min or test_max > train_max).
    Count violations across all seeds x noise levels. Pick the top n_display.
    """
    licem_df = df[
        (df['model'] == 'licem') &
        (df['dataset'] == dataset) &
        (df['concept_name'] != 'bias') &
        (df['task_idx'] == task_idx)
    ]

    if licem_df.empty:
        concepts = df[(df['dataset'] == dataset) & (df['concept_name'] != 'bias')][
            ['concept_idx', 'concept_name']
        ].drop_duplicates().sort_values('concept_idx')
        return concepts.head(n_display)

    concepts = licem_df[['concept_idx', 'concept_name']].drop_duplicates()
    seeds = licem_df['seed'].unique()
    test_noises = sorted(licem_df[licem_df['split'] == 'test']['noise'].unique())

    violation_counts = {}
    for _, row in concepts.iterrows():
        c_idx, c_name = row['concept_idx'], row['concept_name']
        count = 0
        for seed in seeds:
            train_sub = licem_df[
                (licem_df['concept_idx'] == c_idx) &
                (licem_df['seed'] == seed) &
                (licem_df['split'] == 'train')
            ]
            if train_sub.empty:
                continue
            train_min = train_sub['weight_min'].values[0]
            train_max = train_sub['weight_max'].values[0]

            for noise in test_noises:
                test_sub = licem_df[
                    (licem_df['concept_idx'] == c_idx) &
                    (licem_df['seed'] == seed) &
                    (licem_df['split'] == 'test') &
                    (np.isclose(licem_df['noise'], noise))
                ]
                if test_sub.empty:
                    continue
                test_min = test_sub['weight_min'].values[0]
                test_max = test_sub['weight_max'].values[0]
                if test_min < train_min or test_max > train_max:
                    count += 1

        violation_counts[(c_idx, c_name)] = count

    sorted_concepts = sorted(
        violation_counts.items(), key=lambda x: (-x[1], x[0][0])
    )
    top = sorted_concepts[:n_display]
    result = pd.DataFrame(
        [(idx, name) for (idx, name), _ in top],
        columns=['concept_idx', 'concept_name'],
    )
    return result


def build_global_interpretability_table(results_df, n_display_concepts=3, dataset_filter=None, label_suffix='', caption_suffix=''):
    """
    Build and return a LaTeX table string for global interpretability comparison.

    LICEM: weight ranges [min, max] per concept, per split (train, test+noise).
    Lin-M-CBE: fixed learned weight values per concept, per memory slot.

    Args:
        dataset_filter: optional list of dataset names to include.
        label_suffix: appended to the LaTeX label.
        caption_suffix: appended to the caption.
    """
    if results_df.empty:
        return "% No results to display."

    if dataset_filter is not None:
        results_df = results_df[results_df['dataset'].isin(dataset_filter)].copy()
        if results_df.empty:
            return "% No results for the given dataset filter."

    datasets = sorted(results_df['dataset'].unique())
    task_idx = 0

    dataset_concepts = {}
    for ds in datasets:
        sel = select_gi_concepts(results_df, ds, task_idx=task_idx, n_display=n_display_concepts)
        dataset_concepts[ds] = list(sel.itertuples(index=False, name=None))

    licem_df = results_df[results_df['model'] == 'licem'].copy()
    linsym_df = results_df[results_df['model'] == 'linear_symbolic_cbm'].copy()

    licem_rows = []
    if not licem_df.empty:
        licem_rows.append(('train', 0.0))
        test_noises = sorted(licem_df[licem_df['split'] == 'test']['noise'].unique())
        for noise in test_noises:
            licem_rows.append(('test', noise))

    linsym_slots = []
    if not linsym_df.empty:
        linsym_slots = sorted(linsym_df['memory_slot'].dropna().unique())

    col_spec = "ll|" + "|".join(
        "c" * len(dataset_concepts[ds]) for ds in datasets
    )

    def _format_range(row_min, row_max, decimals=2):
        return f"[{row_min:.{decimals}f}, {row_max:.{decimals}f}]"

    def _format_value(val, decimals=2):
        return f"{val:.{decimals}f}"

    def _is_classification(ds):
        return results_df[results_df['dataset'] == ds]['task_idx'].nunique() > 1

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    cap_extra = f" {caption_suffix}" if caption_suffix else ""
    lines.append(r"\caption{Global interpretability comparison. LICEM shows weight ranges $[\min, \max]$ "
                 r"across samples; Lin-M-CBE shows fixed learned weights per memory slot. "
                 r"Noise is applied to the encoder input. "
                 rf"Concepts selected by most training-range violations across seeds and noise levels.{cap_extra}}}")
    lab = f"tab:global_interpretability{label_suffix}"
    lines.append(rf"\label{{{lab}}}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: dataset names (with class name for classification)
    header1_parts = [r"\multicolumn{2}{c|}{}"]
    for ds in datasets:
        n_cols = len(dataset_concepts[ds])
        ds_display = get_df_name(ds) if get_df_name(ds) else ds
        if _is_classification(ds):
            task_name_row = results_df[
                (results_df['dataset'] == ds) & (results_df['task_idx'] == task_idx)
            ]['task_name'].iloc[0]
            task_clean = task_name_row.replace('+', ' ')
            ds_display = f"{ds_display} ({task_clean})"
        header1_parts.append(
            rf"\multicolumn{{{n_cols}}}{{c}}{{{ds_display}}}"
        )
    lines.append(" & ".join(header1_parts) + r" \\")

    # Header row 2: concept names
    header2_parts = ["Model", "Split"]
    for ds in datasets:
        for _, c_name in dataset_concepts[ds]:
            short = c_name.replace('value_', '').replace('_', r'\_')
            header2_parts.append(f"${short}$")
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    # LICEM rows
    if licem_rows:
        n_licem_rows = len(licem_rows)
        first = True
        for split, noise in licem_rows:
            row_parts = []
            if first:
                row_parts.append(rf"\multirow{{{n_licem_rows}}}{{*}}{{LICEM}}")
                first = False
            else:
                row_parts.append("")

            if split == 'train':
                row_parts.append("Train")
            else:
                row_parts.append(f"Test ($\\epsilon$={noise:.1f})")

            for ds in datasets:
                for c_idx, c_name in dataset_concepts[ds]:
                    mask = (
                        (licem_df['dataset'] == ds) &
                        (licem_df['concept_idx'] == c_idx) &
                        (licem_df['task_idx'] == task_idx)
                    )
                    if split == 'train':
                        mask = mask & (licem_df['split'] == 'train')
                    else:
                        mask = mask & (licem_df['split'] == 'test') & (np.isclose(licem_df['noise'], noise))

                    subset = licem_df[mask]
                    if subset.empty:
                        row_parts.append("--")
                    else:
                        env_min = subset['weight_min'].min()
                        env_max = subset['weight_max'].max()
                        row_parts.append(_format_range(env_min, env_max))

            lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\midrule")

    # Lin-M-CBE rows
    if linsym_slots is not None and len(linsym_slots) > 0:
        n_linsym_rows = len(linsym_slots)
        first = True
        for mem_idx in linsym_slots:
            row_parts = []
            if first:
                row_parts.append(rf"\multirow{{{n_linsym_rows}}}{{*}}{{Lin-M-CBE}}")
                first = False
            else:
                row_parts.append("")

            row_parts.append(f"Slot {int(mem_idx)}")

            for ds in datasets:
                for c_idx, c_name in dataset_concepts[ds]:
                    mask = (
                        (linsym_df['dataset'] == ds) &
                        (linsym_df['concept_idx'] == c_idx) &
                        (linsym_df['task_idx'] == task_idx) &
                        (linsym_df['memory_slot'] == mem_idx)
                    )
                    subset = linsym_df[mask]
                    if subset.empty:
                        row_parts.append("--")
                    else:
                        avg_val = subset['weight_value'].mean()
                        row_parts.append(_format_value(avg_val))

            lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def build_global_interpretability_violations_table(results_df, n_display_concepts=3, mode='count', dataset_filter=None, label_suffix='', caption_suffix=''):
    """
    Build a LaTeX table showing training ranges + per-sample violation counts or percentages.

    Structure:
    - LICEM Train row: shows training weight range [min, max] per concept.
    - LICEM Test rows: shows violation count/percentage per noise level.
    - Lin-M-CBE rows: shows 0 violations (fixed weights, by design).

    Args:
        results_df: DataFrame with n_violations and n_samples columns.
        n_display_concepts: Number of concepts to display per dataset.
        mode: 'count' for absolute violation counts, 'percentage' for percentages.
        dataset_filter: optional list of dataset names to include.
        label_suffix: appended to the LaTeX label.
        caption_suffix: appended to the caption.
    """
    if results_df.empty:
        return "% No results to display."

    if 'n_violations' not in results_df.columns:
        return "% No violation data available."

    if dataset_filter is not None:
        results_df = results_df[results_df['dataset'].isin(dataset_filter)].copy()
        if results_df.empty:
            return "% No results for the given dataset filter."

    licem_df = results_df[
        (results_df['model'] == 'licem') &
        (results_df['concept_name'] != 'bias')
    ].copy()

    linsym_df = results_df[
        (results_df['model'] == 'linear_symbolic_cbm') &
        (results_df['concept_name'] != 'bias')
    ].copy()

    if licem_df.empty:
        return "% No LICEM violation data available."

    datasets = sorted(licem_df['dataset'].unique())
    task_idx = 0

    dataset_concepts = {}
    for ds in datasets:
        sel = select_gi_concepts(results_df, ds, task_idx=task_idx, n_display=n_display_concepts)
        dataset_concepts[ds] = list(sel.itertuples(index=False, name=None))

    test_noises = sorted(licem_df[licem_df['split'] == 'test']['noise'].unique())

    # Lin-M-CBE memory slots
    linsym_slots = []
    if not linsym_df.empty:
        linsym_slots = sorted(linsym_df['memory_slot'].dropna().unique())

    col_spec = "ll|" + "|".join(
        "c" * len(dataset_concepts[ds]) for ds in datasets
    )

    def _is_classification(ds):
        return results_df[results_df['dataset'] == ds]['task_idx'].nunique() > 1

    def _format_range(v_min, v_max, decimals=2):
        return f"[{v_min:.{decimals}f}, {v_max:.{decimals}f}]"

    if mode == 'count':
        caption_metric = "number of samples"
        label_base = "violations_count"
        zero_val = "0"
    else:
        caption_metric = "percentage of samples"
        label_base = "violations_pct"
        zero_val = "0.0\\%"

    # Count rows: 1 train + len(test_noises) LICEM + len(linsym_slots) Lin-M-CBE
    n_licem_rows = 1 + len(test_noises)
    n_linsym_rows = max(len(linsym_slots), 1) if not linsym_df.empty else 0

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    cap_extra = f" {caption_suffix}" if caption_suffix else ""
    lines.append(rf"\caption{{{caption_metric.capitalize()} violating the training weight range per concept. "
                 r"The Train row shows the reference range $[\min, \max]$. "
                 r"Lin-M-CBE has zero violations by design (fixed weights). "
                 rf"Concepts selected by most training-range violations across seeds and noise levels.{cap_extra}}}")
    lab = f"tab:global_interpretability_{label_base}{label_suffix}"
    lines.append(rf"\label{{{lab}}}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: dataset names
    header1_parts = [r"\multicolumn{2}{c|}{}"]
    for ds in datasets:
        n_cols = len(dataset_concepts[ds])
        ds_display = get_df_name(ds) if get_df_name(ds) else ds
        if _is_classification(ds):
            task_name_row = results_df[
                (results_df['dataset'] == ds) & (results_df['task_idx'] == task_idx)
            ]['task_name'].iloc[0]
            task_clean = task_name_row.replace('+', ' ')
            ds_display = f"{ds_display} ({task_clean})"
        header1_parts.append(
            rf"\multicolumn{{{n_cols}}}{{c}}{{{ds_display}}}"
        )
    lines.append(" & ".join(header1_parts) + r" \\")

    # Header row 2: concept names
    header2_parts = ["Model", "Split"]
    for ds in datasets:
        for _, c_name in dataset_concepts[ds]:
            short = c_name.replace('value_', '').replace('_', r'\_')
            header2_parts.append(f"${short}$")
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    # --- LICEM rows ---
    first = True
    # Train row: show training range
    row_parts = []
    row_parts.append(rf"\multirow{{{n_licem_rows}}}{{*}}{{LICEM}}")
    row_parts.append("Train")
    for ds in datasets:
        for c_idx, c_name in dataset_concepts[ds]:
            mask = (
                (licem_df['dataset'] == ds) &
                (licem_df['concept_idx'] == c_idx) &
                (licem_df['task_idx'] == task_idx) &
                (licem_df['split'] == 'train')
            )
            subset = licem_df[mask]
            if subset.empty:
                row_parts.append("--")
            else:
                env_min = subset['weight_min'].min()
                env_max = subset['weight_max'].max()
                row_parts.append(_format_range(env_min, env_max))
    lines.append(" & ".join(row_parts) + r" \\")

    # Test rows: show violations
    for noise in test_noises:
        row_parts = ["", f"Test ($\\epsilon$={noise:.1f})"]
        for ds in datasets:
            for c_idx, c_name in dataset_concepts[ds]:
                mask = (
                    (licem_df['dataset'] == ds) &
                    (licem_df['concept_idx'] == c_idx) &
                    (licem_df['task_idx'] == task_idx) &
                    (licem_df['split'] == 'test') &
                    (np.isclose(licem_df['noise'], noise))
                )
                subset = licem_df[mask].dropna(subset=['n_violations'])
                if subset.empty:
                    row_parts.append("--")
                else:
                    total_violations = subset['n_violations'].sum()
                    total_samples = subset['n_samples'].sum()
                    if mode == 'count':
                        row_parts.append(f"{int(total_violations)}")
                    else:
                        pct = (total_violations / total_samples * 100) if total_samples > 0 else 0
                        row_parts.append(f"{pct:.1f}\\%")
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\midrule")

    # --- Lin-M-CBE rows: all zeros ---
    if n_linsym_rows > 0:
        if linsym_slots:
            first = True
            for mem_idx in linsym_slots:
                row_parts = []
                if first:
                    row_parts.append(rf"\multirow{{{n_linsym_rows}}}{{*}}{{Lin-M-CBE}}")
                    first = False
                else:
                    row_parts.append("")
                row_parts.append(f"Slot {int(mem_idx)}")
                for ds in datasets:
                    for _ in dataset_concepts[ds]:
                        row_parts.append(zero_val)
                lines.append(" & ".join(row_parts) + r" \\")
        else:
            row_parts = [r"Lin-M-CBE", "Fixed"]
            for ds in datasets:
                for _ in dataset_concepts[ds]:
                    row_parts.append(zero_val)
            lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


##########################################################
#### Expressions Intervention Experiment Functions #######
##########################################################

def collect_expressions_intervention_results(base_path):
    """
    Collect results from the expressions_intervention experiment.

    Walks subdirectories, loads MAE from metrics.csv and computes TED
    by comparing learned equations against the dataset ground-truth equations.

    Returns a DataFrame with columns:
        dataset, n_gt_equations, seed, memory_size, task_mae, avg_ted
    """
    from src.utils.ted import sympy_to_tree, ted_weighted, make_costs
    from sympy import sympify, Symbol
    try:
        from scipy.optimize import linear_sum_assignment
        use_scipy = True
    except ImportError:
        use_scipy = False

    if not os.path.exists(base_path):
        return pd.DataFrame()

    # Collect all experiment run directories
    exps_path = []
    for timestamp_dir in os.listdir(base_path):
        ts_full = os.path.join(base_path, timestamp_dir)
        if not os.path.isdir(ts_full):
            continue
        for run_dir in os.listdir(ts_full):
            run_full = os.path.join(ts_full, run_dir)
            if os.path.isdir(run_full) and 'multirun' not in run_dir:
                exps_path.append(run_full)

    results = []
    weight_fn, rename_fn = make_costs()

    for exp_path in tqdm(exps_path, desc="Processing expressions_intervention experiments"):
        config_file = os.path.join(exp_path, '.hydra/config.yaml')
        metrics_file = os.path.join(exp_path, 'logs/experiment_metrics/metrics.csv')
        predictions_file = os.path.join(exp_path, 'logs/experiment_metrics/test_predictions_per_sample.csv')

        if not os.path.exists(config_file) or not os.path.exists(metrics_file):
            continue

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            dataset_name = config['dataset']['metadata']['name']
            seed = config['seed']
            memory_size = config.get('memory_size', 1)
            n_gt = config.get('model', {}).get('params', {}).get('n_gt_equations', 0)

            # Load MAE from metrics.csv
            metrics_df = pd.read_csv(metrics_file)
            task_mae = None
            if 'test/y/mae' in metrics_df.columns:
                task_mae = metrics_df['test/y/mae'].iloc[-1]
            elif 'test/y/mse' in metrics_df.columns:
                task_mae = metrics_df['test/y/mse'].iloc[-1]

            # Compute TED if ground-truth equations and predictions are available
            avg_ted = np.nan
            gt_equation_strs = config.get('dataset', {}).get('equations', None)
            if gt_equation_strs and os.path.exists(predictions_file):
                try:
                    # Load predictions first to extract concept names
                    # (c_names is set at runtime and NOT saved in the Hydra config)
                    df_pred = pd.read_csv(predictions_file)
                    c_names = [x.replace('c_pred_', '') for x in df_pred.columns if 'c_pred' in x]

                    # Build true trees from dataset config equations
                    true_trees = []
                    for eq_str in gt_equation_strs:
                        eq_expr = sympify(eq_str)
                        # Rename c0, c1, ... to concept names
                        for i, c_name in enumerate(c_names):
                            eq_expr = eq_expr.subs(Symbol(f'c{i}'), Symbol(c_name))
                        tree = sympy_to_tree(eq_expr, canonicalize_commutative=True, enforce_mul_for_add_terms=True)
                        true_trees.append(tree)
                    if 'equation' in df_pred.columns:
                        pred_vars = [x.replace('c_pred_', '') for x in df_pred.columns if 'c_pred' in x]
                        learned_eq_strs = []
                        for eq_str in df_pred['equation'].unique():
                            if pd.notna(eq_str) and eq_str != '':
                                if ':' in eq_str:
                                    eq_str = eq_str.split(':')[1].strip()
                                learned_eq_strs.append(eq_str)

                        learned_trees = []
                        for eq_str in learned_eq_strs:
                            try:
                                equation = sympify(eq_str, locals={v: sympify(v) for v in pred_vars})
                                tree = sympy_to_tree(equation, canonicalize_commutative=True, enforce_mul_for_add_terms=True)
                                learned_trees.append(tree)
                            except Exception:
                                continue

                        if learned_trees and true_trees:
                            n_l, n_t = len(learned_trees), len(true_trees)
                            ted_matrix = np.zeros((n_l, n_t))
                            for i, lt in enumerate(learned_trees):
                                for j, tt in enumerate(true_trees):
                                    try:
                                        ted_matrix[i, j] = ted_weighted(lt, tt, weight_fn, rename_fn)
                                    except Exception:
                                        ted_matrix[i, j] = np.inf

                            if n_l != n_t:
                                max_dim = max(n_l, n_t)
                                padded = np.full((max_dim, max_dim), np.max(ted_matrix) * 10)
                                padded[:n_l, :n_t] = ted_matrix
                                if use_scipy:
                                    ri, ci = linear_sum_assignment(padded)
                                else:
                                    ri, ci = np.arange(min(n_l, n_t)), np.arange(min(n_l, n_t))
                                valid = (ri < n_l) & (ci < n_t)
                                ri, ci = ri[valid], ci[valid]
                            else:
                                if use_scipy:
                                    ri, ci = linear_sum_assignment(ted_matrix)
                                else:
                                    ri, ci = np.arange(n_l), np.arange(n_t)

                            assigned = [ted_matrix[i, j] for i, j in zip(ri, ci)]
                            avg_ted = np.mean(assigned) if assigned else np.nan

                except Exception as e:
                    pass  # TED stays NaN

            results.append({
                'dataset': dataset_name,
                'n_gt_equations': n_gt,
                'seed': seed,
                'memory_size': memory_size,
                'task_mae': task_mae,
                'avg_ted': avg_ted,
            })

        except Exception:
            continue

    return pd.DataFrame(results)


def build_expressions_intervention_table(results_df):
    """
    Build a LaTeX table for the expressions intervention experiment.

    Columns: one per dataset, each split into MAE and TED sub-columns.
    Rows: one per n_gt_equations value (aggregated across seeds as mean ± std).
    """
    if results_df.empty:
        return "% No expressions intervention results."

    datasets = sorted(results_df['dataset'].unique())
    n_gt_values = sorted(results_df['n_gt_equations'].unique())

    # Aggregate across seeds
    agg = results_df.groupby(['dataset', 'n_gt_equations']).agg(
        mae_mean=('task_mae', 'mean'),
        mae_std=('task_mae', 'std'),
        ted_mean=('avg_ted', 'mean'),
        ted_std=('avg_ted', 'std'),
        count=('seed', 'count'),
    ).reset_index()

    n_ds = len(datasets)
    # l | cc | cc | ...
    col_fmt = "l" + " cc" * n_ds

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_fmt + r"}")
    lines.append(r"\toprule")

    # First header row: dataset names spanning 2 columns each
    header1_parts = [""]
    for ds in datasets:
        ds_display = get_df_name(ds)
        header1_parts.append(r"\multicolumn{2}{c}{" + ds_display + r"}")
    lines.append(" & ".join(header1_parts) + r" \\")

    # Cmidrules for each dataset pair
    cmidrules = []
    for i, _ in enumerate(datasets):
        start = 2 + i * 2
        end = start + 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))

    # Second header row: MAE | TED for each dataset
    header2_parts = [r"\# GT Eq."]
    for _ in datasets:
        header2_parts.append("MAE")
        header2_parts.append("TED")
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    def fmt_cell(mean, std):
        if pd.isna(mean):
            return "--"
        if pd.isna(std) or std == 0:
            return rf"${mean:.3f}$"
        return rf"${mean:.3f}_{{\tiny{{\pm {std:.3f}}}}}$"

    # Data rows: one per n_gt_equations
    for n_gt in n_gt_values:
        row_parts = [str(n_gt)]
        for ds in datasets:
            row_data = agg[(agg['dataset'] == ds) & (agg['n_gt_equations'] == n_gt)]
            if len(row_data) == 0:
                row_parts.append("--")
                row_parts.append("--")
            else:
                r = row_data.iloc[0]
                row_parts.append(fmt_cell(r['mae_mean'], r['mae_std']))
                row_parts.append(fmt_cell(r['ted_mean'], r['ted_std']))
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\caption{Effect of providing ground-truth equations on task performance (MAE) "
                 r"and equation quality (TED). Each row corresponds to a different number of "
                 r"ground-truth equations injected; remaining equations are discovered via PySR.}")
    lines.append(r"\label{tab:expressions_intervention}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ---------------------------------------------------------------
# Explanation Robustness utilities
# ---------------------------------------------------------------

def collect_explanation_robustness_results(base_path):
    """Walk the output directory tree and collect all explanation_robustness.csv files."""
    all_dfs = []
    if not os.path.exists(base_path):
        print(f"Warning: {base_path} does not exist.")
        return pd.DataFrame()

    for root, dirs, files in os.walk(base_path):
        for fname in files:
            if fname == 'explanation_robustness.csv':
                fpath = os.path.join(root, fname)
                try:
                    df = pd.read_csv(fpath)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Warning: could not read {fpath}: {e}")

    if not all_dfs:
        print("No explanation_robustness.csv files found.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def build_explanation_robustness_tables(results_df, model_styles, agg='mean',
                                         custom_order=None, float_fmt="{:.4f}",
                                         metric_col='lipschitz_constant'):
    """
    Build a LaTeX table for explanation robustness.

    For each alpha level, produce one table where:
      - columns = datasets
      - rows = models
      - cell = mean±ci across seeds of the *per-alpha aggregate*

    Args:
        results_df: DataFrame with columns [sample_idx, lipschitz_constant,
                    lipschitz_constant_normalized, model, dataset, seed, alpha, K].
        model_styles: dict mapping model key -> {'name': ...}.
        agg: 'mean' or 'max' – how to aggregate L_{x_i} across samples
             within a single (model, dataset, seed, alpha) run.
        custom_order: optional list of datasets in desired column order.
        float_fmt: format string for floats.
        metric_col: column to aggregate ('lipschitz_constant' or
                    'lipschitz_constant_normalized').

    Returns:
        dict  alpha -> LaTeX table string.
    """
    if results_df.empty:
        return {}

    # Aggregate per (model, dataset, seed, alpha)
    grouped = (
        results_df
        .groupby(['model', 'dataset', 'seed', 'alpha'])[metric_col]
        .agg(agg)
        .reset_index()
        .rename(columns={metric_col: 'value'})
    )

    # Then mean, std and count across seeds for confidence interval
    summary = (
        grouped
        .groupby(['model', 'dataset', 'alpha'])['value']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    summary['std'] = summary['std'].fillna(0.0)
    # 95% confidence interval: 1.96 * std / sqrt(n)
    summary['ci'] = 1.96 * summary['std'] / np.sqrt(summary['count'])

    alphas = sorted(summary['alpha'].unique())
    datasets = sorted(summary['dataset'].unique())
    if custom_order is not None:
        datasets = [d for d in custom_order if d in datasets]

    models = sorted(summary['model'].unique())
    # Re-order models according to model_styles order if available
    if model_styles is not None:
        style_order = list(model_styles.keys())
        models = [m for m in style_order if m in models]

    tables = {}
    for alpha in alphas:
        sub = summary[summary['alpha'] == alpha]

        display_datasets = [get_df_name(d) for d in datasets]
        col_format = "l" + "c" * len(datasets)

        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\resizebox{\columnwidth}{!}{%")
        lines.append(r"\begin{tabular}{" + col_format + r"}")
        lines.append(r"\toprule")
        header = "Model & " + " & ".join(display_datasets) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")

        for model in models:
            name = model_styles[model]['name'] if model_styles and model in model_styles else model
            row = [name.replace("_", r"\_")]
            for dataset in datasets:
                cell_df = sub[(sub['model'] == model) & (sub['dataset'] == dataset)]
                if cell_df.empty:
                    row.append("--")
                else:
                    m = float_fmt.format(cell_df.iloc[0]['mean'])
                    ci = float_fmt.format(cell_df.iloc[0]['ci'])
                    row.append(rf"${m} \scriptscriptstyle{{\pm {ci}}}$")
            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")
        agg_label = r"$\overline{L}$" if agg == 'mean' else r"$L_{\max}$"
        norm_suffix = ", normalized" if metric_col == 'lipschitz_constant_normalized' else ""
        metric_tag = "norm_" if metric_col == 'lipschitz_constant_normalized' else ""
        lines.append(r"\caption{Explanation robustness (" + agg_label +
                     norm_suffix + r") at $\alpha=" + f"{alpha:.2f}" + r"$.}")
        lines.append(r"\label{tab:robustness_" + metric_tag + agg + f"_alpha_{alpha:.2f}" + r"}")
        lines.append(r"\end{table}")

        tables[alpha] = "\n".join(lines)

    return tables

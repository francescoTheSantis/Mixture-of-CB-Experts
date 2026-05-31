import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import pandas as pd
import scienceplots
import warnings
import yaml
from tqdm import tqdm
from show_results import result_figs, regression_datasets, MEMORY_MODELS_LIST, COMPLEXITY_METRICS_LIST, models_order, model_styles, custom_order, NUMBER_OF_CLASSES_PER_DATASET

# Plot style 🤙🏻. 
plt.style.use(['science', 'ieee', 'no-latex'])

########################################
########## Name conversion #############
########################################

def get_df_name(df):
    if df=='cub':
        return 'CUB200'
    elif df=='awa2':
        return 'AWA2'
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
                grouped_data['se_metric'] = 100 * grouped_data['se_metric']
            
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
    
    str_store = "cls_noise_" + str(classification_noise).replace('.', '') + "_reg_noise_" + str(regression_noise).replace('.', '')
    suffix = 'relative_accuracy_difference' if relative_accuracy else 'absolute_accuracy'
    os.makedirs(f'{out_dir}/intervention/{suffix}', exist_ok=True)
    plt.savefig(f'{out_dir}/intervention/{suffix}/{str_store}.pdf')
    plt.show()

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
                            'xerr': [complexity_err],
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
                            xerr=plot_cmd.get('xerr'),
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